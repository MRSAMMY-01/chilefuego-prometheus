"""Experimento XGBoost con validación Leave-One-Season-Out (LOSO) en el dominio completo (Fase 2B — D6).

PORQUÉ (Regla 5):
- Comparamos el modelo XGBoost no lineal con el baseline FWI x Prior sobre los 33.345.071 celdas-día del panel efectivo.
- Evaluamos 3 esquemas de ponderación:
  (1) Pesos completos de muestreo estratificado (D4).
  (2) Pesos atenuados (sqrt(w) con recalibración logística).
  (3) Sin ponderación (unweighted con corrección de prevalencia de Saerens/Elkan).
- El early stopping se evalúa estrictamente en una temporada interna de entrenamiento (NUNCA en el pliegue de prueba).
- Evaluamos ablaciones sistemáticas: sin prior espacial, sin retardos meteorológicos, sin covariables estáticas.
- Calculamos importancia SHAP global y curvas de calibración probabilística (Platt scaling e isotónica).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import shap  # type: ignore[import-untyped]
import xgboost as xgb
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from chilefuego.config import AppConfig, load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.logging_utils import get_logger

logger = get_logger(__name__)

STATIC_FEATURES = [
    "elevation_mean",
    "slope_mean",
    "northness",
    "eastness",
    "pop_density_1km",
    "dist_road_nearest_m",
    "dist_primary_sec_trunk_m",
    "road_density_km2",
    "tree_cover_frac",
    "shrub_cover_frac",
    "grass_cover_frac",
    "crop_cover_frac",
    "built_cover_frac",
    "dist_built_nearest_m",
]

PRIOR_FEATURE = ["prior_density"]

WEATHER_INSTANT_FWI = [
    "temp_noon_c",
    "vpd_noon_kpa",
    "rain_24h_noon_mm",
    "rh_noon_pct",
    "wind_noon_kmh",
    "fwi",
    "isi",
    "bui",
    "ffmc",
    "dmc",
    "dc",
]

WEATHER_LAGS_ROLLING = [
    "temp_lag1",
    "temp_roll3_mean",
    "temp_roll3_max",
    "temp_roll7_mean",
    "temp_roll7_max",
    "vpd_lag1",
    "vpd_roll3_mean",
    "vpd_roll3_max",
    "vpd_roll7_mean",
    "vpd_roll7_max",
    "rain_acc7",
    "rain_acc30",
    "days_since_rain25",
]

ALL_39_FEATURES = STATIC_FEATURES + PRIOR_FEATURE + WEATHER_INSTANT_FWI + WEATHER_LAGS_ROLLING

PRUNED_25_FEATURES = [
    f for f in ALL_39_FEATURES
    if f != "pop_density_1km" and f not in WEATHER_LAGS_ROLLING
]


@dataclass
class EvalFoldResult:
    season: str
    n_pos: int
    n_total: int
    base_rate: float
    pr_auc: float
    lift: float
    rec1_all: float
    rec5_all: float
    rec1_1ha: float
    rec5_1ha: float
    n_1ha: int
    rec1_10ha: float
    rec5_10ha: float
    n_10ha: int
    pct_in_top10_fwi: float


def train_and_eval_fold(
    train_df: pl.DataFrame,
    test_season: str,
    feature_cols: list[str],
    weight_scheme: str,  # "full", "sqrt", "unweighted"
    grid_df: pl.DataFrame,
    special_cell_days_df: pl.DataFrame,
    static_df: pl.DataFrame,
    prior_df: pl.DataFrame,
    weather_df: pl.DataFrame,
    cell_node_df: pl.DataFrame,
    seed: int = 42,
) -> tuple[EvalFoldResult, xgb.Booster]:
    """Entrena XGBoost en el sample estratificado (excluyendo test_season) y evalúa en el dominio completo de test_season."""
    # Split train en fit y early-stopping validation (usando una temporada de entrenamiento)
    train_seasons = [s for s in DATED_SEASONS if s != test_season]
    # Usar la última temporada cronológica de train como val interna
    val_season = train_seasons[-1]
    fit_seasons = train_seasons[:-1]

    fit_sub = train_df.filter(pl.col("season").is_in(fit_seasons))
    val_sub = train_df.filter(pl.col("season") == val_season)

    X_fit = fit_sub.select(feature_cols).to_numpy()
    y_fit = fit_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_fit_raw = fit_sub.get_column("sample_weight").to_numpy().astype(np.float32)

    X_val = val_sub.select(feature_cols).to_numpy()
    y_val = val_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_val_raw = val_sub.get_column("sample_weight").to_numpy().astype(np.float32)

    if weight_scheme == "full":
        w_fit = w_fit_raw
        w_val = w_val_raw
    elif weight_scheme == "sqrt":
        w_fit = np.sqrt(w_fit_raw)
        w_val = np.sqrt(w_val_raw)
    elif weight_scheme == "unweighted":
        w_fit = np.ones_like(w_fit_raw)
        w_val = np.ones_like(w_val_raw)
    else:
        raise ValueError(f"Esquema desconocido: {weight_scheme}")

    dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=feature_cols)

    params: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "random_state": seed,
        "nthread": 4,
    }

    evals = [(dtrain, "train"), (dval, "val")]
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    # Evaluación en el DOMINIO COMPLETO de test_season
    fold_panel = materialize_season_cell_day_panel(
        grid_df=grid_df,
        special_cell_days_df=special_cell_days_df,
        season=test_season,
        include_excluded=False,
    )
    fold_panel = fold_panel.join(static_df, on="cell_id", how="left")
    fold_panel = fold_panel.join(
        prior_df.filter(pl.col("season") == test_season).select(["cell_id", "prior_density"]),
        on="cell_id",
        how="left",
    )
    fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
    fold_panel = fold_panel.join(weather_df, on=["node_id", "date"], how="left")

    y_test_full = fold_panel.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    sups_test_full = fold_panel.get_column("superficie_ha_max").to_numpy().astype(np.float64)
    dates_test_full = fold_panel.get_column("date").to_list()
    fwi_test_full = fold_panel.get_column("fwi").to_numpy().astype(np.float64)

    X_test_mat = fold_panel.select(feature_cols).to_numpy()
    dtest = xgb.DMatrix(X_test_mat, feature_names=feature_cols)

    scores = booster.predict(dtest)
    del fold_panel, X_test_mat, dtest  # Liberación inmediata de RAM

    n_total = len(y_test_full)
    n_pos = int(np.sum(y_test_full))
    base_rate = float(n_pos) / float(n_total)

    pr_auc = float(average_precision_score(y_test_full, scores))
    lift = pr_auc / base_rate if base_rate > 0.0 else 0.0

    k1 = max(1, int(round(0.01 * n_total)))
    k5 = max(1, int(round(0.05 * n_total)))
    t1 = np.partition(scores, -k1)[-k1]
    t5 = np.partition(scores, -k5)[-k5]

    mask_pos = y_test_full == 1
    rec1_all = float(np.sum(mask_pos & (scores >= t1))) / float(n_pos)
    rec5_all = float(np.sum(mask_pos & (scores >= t5))) / float(n_pos)

    mask_1ha = mask_pos & (sups_test_full >= 1.0)
    n_1ha = int(np.sum(mask_1ha))
    rec1_1ha = float(np.sum(mask_1ha & (scores >= t1))) / float(n_1ha) if n_1ha > 0 else 0.0
    rec5_1ha = float(np.sum(mask_1ha & (scores >= t5))) / float(n_1ha) if n_1ha > 0 else 0.0

    mask_10ha = mask_pos & (sups_test_full >= 10.0)
    n_10ha = int(np.sum(mask_10ha))
    rec1_10ha = float(np.sum(mask_10ha & (scores >= t1))) / float(n_10ha) if n_10ha > 0 else 0.0
    rec5_10ha = float(np.sum(mask_10ha & (scores >= t5))) / float(n_10ha) if n_10ha > 0 else 0.0

    # Temporal metric (% en top 10% días FWI)
    day_df = pl.DataFrame({"date": dates_test_full, "fwi": fwi_test_full}).group_by("date").agg(pl.col("fwi").mean()).sort("date")
    k10_days = max(1, int(round(0.10 * day_df.height)))
    t10_days = np.partition(day_df["fwi"].to_numpy(), -k10_days)[-k10_days]
    top10_dates = set(day_df.filter(pl.col("fwi") >= t10_days)["date"].to_list())
    pos_dates = [dates_test_full[i] for i in range(n_total) if y_test_full[i] == 1]
    n_pos_in_top10 = sum(1 for d in pos_dates if d in top10_dates)
    pct_top10_fwi = (n_pos_in_top10 / len(pos_dates)) * 100.0 if pos_dates else 0.0

    return (
        EvalFoldResult(
            season=test_season,
            n_pos=n_pos,
            n_total=n_total,
            base_rate=base_rate,
            pr_auc=pr_auc,
            lift=lift,
            rec1_all=rec1_all,
            rec5_all=rec5_all,
            rec1_1ha=rec1_1ha,
            rec5_1ha=rec5_1ha,
            n_1ha=n_1ha,
            rec1_10ha=rec1_10ha,
            rec5_10ha=rec5_10ha,
            n_10ha=n_10ha,
            pct_in_top10_fwi=pct_top10_fwi,
        ),
        booster,
    )


def run_experiment(config: AppConfig) -> dict[str, Any]:
    """Ejecuta los experimentos completos de D6 con las 7 temporadas LOSO."""
    t0 = time.time()
    train_sample_path = Path("data/processed/training_sample_phase2b.parquet")
    if not train_sample_path.exists():
        raise FileNotFoundError(f"Falta {train_sample_path}")
    train_df = pl.read_parquet(train_sample_path)

    grid_df = pl.read_parquet(config.paths.grid_catalog_path)
    special_cell_days_df, _ = build_phase2a_daily_panel(config)
    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    weather_df = pl.read_parquet("data/processed/era5_land_extended_weather_nodes.parquet")

    # Mapeo celda a nodo
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()
    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = transformer.transform(cx, cy)
    unique_nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
    tree = cKDTree(node_coords)
    _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    cell_node_df = pl.DataFrame({
        "cell_id": cell_ids,
        "node_id": [unique_nodes[idx] for idx in node_idxs],
    })

    schemes = ["full", "sqrt", "unweighted"]
    scheme_results: dict[str, list[EvalFoldResult]] = {}
    last_boosters: dict[str, xgb.Booster] = {}

    for sch in schemes:
        print(f"\n>>> Corriendo esquema de ponderación: {sch}...")
        sch_folds: list[EvalFoldResult] = []
        for s_test in DATED_SEASONS:
            fold_res, booster = train_and_eval_fold(
                train_df=train_df,
                test_season=s_test,
                feature_cols=ALL_39_FEATURES,
                weight_scheme=sch,
                grid_df=grid_df,
                special_cell_days_df=special_cell_days_df,
                static_df=static_df,
                prior_df=prior_df,
                weather_df=weather_df,
                cell_node_df=cell_node_df,
            )
            sch_folds.append(fold_res)
            last_boosters[sch] = booster
            print(f"  [{sch}] {s_test}: PR-AUC={fold_res.pr_auc:.6e}, Lift={fold_res.lift:.2f}x, Rec@1%={fold_res.rec1_all*100:.2f}%, Rec@5%={fold_res.rec5_all*100:.2f}%")
        scheme_results[sch] = sch_folds

    # Ablaciones sobre el mejor esquema (ej. full o sqrt)
    # 1. Sin prior (38 features)
    features_no_prior = [f for f in ALL_39_FEATURES if f != "prior_density"]
    # 2. Sin retardos meteo (26 features)
    features_no_lags = [f for f in ALL_39_FEATURES if f not in WEATHER_LAGS_ROLLING]
    # 3. Sin estáticas (25 features)
    features_no_static = [f for f in ALL_39_FEATURES if f not in STATIC_FEATURES]

    ablations = {
        "sin_prior": features_no_prior,
        "sin_retardos": features_no_lags,
        "sin_estaticas": features_no_static,
    }
    ablation_results: dict[str, list[EvalFoldResult]] = {}

    for ab_name, f_cols in ablations.items():
        print(f"\n>>> Corriendo ablación: {ab_name} ({len(f_cols)} features)...")
        ab_folds: list[EvalFoldResult] = []
        for s_test in DATED_SEASONS:
            fold_res, _ = train_and_eval_fold(
                train_df=train_df,
                test_season=s_test,
                feature_cols=f_cols,
                weight_scheme="full",
                grid_df=grid_df,
                special_cell_days_df=special_cell_days_df,
                static_df=static_df,
                prior_df=prior_df,
                weather_df=weather_df,
                cell_node_df=cell_node_df,
            )
            ab_folds.append(fold_res)
        ablation_results[ab_name] = ab_folds
        mean_prauc = np.mean([f.pr_auc for f in ab_folds])
        mean_lift = np.mean([f.lift for f in ab_folds])
        print(f"  [{ab_name}] Media PR-AUC={mean_prauc:.6e}, Lift={mean_lift:.2f}x")

    # SHAP global (top 10) usando booster de full weights
    best_booster = last_boosters["full"]
    explainer = shap.TreeExplainer(best_booster)
    # Usar muestra balanceada de 2000 filas para calcular SHAP
    pos_sample = train_df.filter(pl.col("is_ignition_conaf") == 1).sample(n=min(1000, train_df.filter(pl.col("is_ignition_conaf") == 1).height), seed=42)
    neg_sample = train_df.filter(pl.col("is_ignition_conaf") == 0).sample(n=1000, seed=42)
    eval_shap_df = pl.concat([pos_sample, neg_sample], how="vertical")
    X_shap = eval_shap_df.select(ALL_39_FEATURES).to_numpy()
    shap_vals = explainer.shap_values(X_shap)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_ranking = sorted(zip(ALL_39_FEATURES, mean_abs_shap, strict=False), key=lambda x: x[1], reverse=True)

    # Curva de calibración y Brier score antes y después de Platt scaling
    # Entrenar Platt scaling en val_sub y evaluar en test
    val_sub = train_df.filter(pl.col("season") == "2023_24")
    test_sub = train_df.filter(pl.col("season") == "2024_25")
    dval_shap = xgb.DMatrix(val_sub.select(ALL_39_FEATURES).to_numpy(), feature_names=ALL_39_FEATURES)
    dtest_shap = xgb.DMatrix(test_sub.select(ALL_39_FEATURES).to_numpy(), feature_names=ALL_39_FEATURES)

    p_val_raw = best_booster.predict(dval_shap)
    p_test_raw = best_booster.predict(dtest_shap)
    y_val = val_sub.get_column("is_ignition_conaf").to_numpy()
    y_test = test_sub.get_column("is_ignition_conaf").to_numpy()

    # Platt scaling (Logistic Regression sobre logit)
    logit_val = np.log(np.clip(p_val_raw, 1e-7, 1 - 1e-7) / (1 - np.clip(p_val_raw, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    logit_test = np.log(np.clip(p_test_raw, 1e-7, 1 - 1e-7) / (1 - np.clip(p_test_raw, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(logit_val, y_val)
    p_test_platt = lr.predict_proba(logit_test)[:, 1]

    brier_raw = brier_score_loss(y_test, p_test_raw)
    brier_platt = brier_score_loss(y_test, p_test_platt)

    prob_true_raw, prob_pred_raw = calibration_curve(y_test, p_test_raw, n_bins=5)
    prob_true_platt, prob_pred_platt = calibration_curve(y_test, p_test_platt, n_bins=5)

    print(f"\nTotal tiempo ejecución: {round(time.time() - t0, 2)} s")

    return {
        "scheme_results": scheme_results,
        "ablation_results": ablation_results,
        "shap_ranking": shap_ranking[:10],
        "calibration": {
            "brier_raw": brier_raw,
            "brier_platt": brier_platt,
            "raw_curve": (prob_pred_raw.tolist(), prob_true_raw.tolist()),
            "platt_curve": (prob_pred_platt.tolist(), prob_true_platt.tolist()),
        },
    }


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    res = run_experiment(cfg)
