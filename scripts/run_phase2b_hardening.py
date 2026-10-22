"""Pipeline integral de robustecimiento y reconciliación (Fase 2B — E1 a E7).

PORQUÉ (Regla 5):
- E1: Reconcilia Baseline y XGBoost con la MISMA función de evaluación exacta.
- E2: Cuantifica incertidumbre con 5 semillas (42-46), intervalos bootstrap 95% y 6 ablaciones por grupo por temporada.
- E3: Evalúa ponderación por tamaño de incendio log10(superficie+1) para grandes incendios (>=10 ha).
- E4: Calibra en el dominio COMPLETO (Platt e Isotónica) con Brier score contra predictor constante y tabla de fiabilidad por 100k celdas-día.
- E5: Validación hacia adelante entrenando en 2017_18 a 2021_22 y probando en 2022_23, 2023_24 y 2024_25.
- E7: Define clases de riesgo fijas (Bajo, Moderado, Alto, Extremo) con percentiles históricos.
- Monitorea RSS real del proceso vía psutil en cada etapa.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import xgboost as xgb
from chilefuego.config import load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.models.xgboost_experiment import (
    ALL_39_FEATURES,
    PRIOR_FEATURE,
    WEATHER_LAGS_ROLLING,
)
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

# Grupos de covariables para E2
GROUP_DEM = ["elevation_mean", "slope_mean", "northness", "eastness"]
GROUP_CAMINOS = ["dist_road_nearest_m", "dist_primary_sec_trunk_m", "road_density_km2"]
GROUP_COBERTURA = [
    "tree_cover_frac",
    "shrub_cover_frac",
    "grass_cover_frac",
    "crop_cover_frac",
    "built_cover_frac",
    "dist_built_nearest_m",
]
GROUP_POBLACION = ["pop_density_1km"]
GROUP_PRIOR = PRIOR_FEATURE
GROUP_REZAGOS = WEATHER_LAGS_ROLLING

GROUP_ABLATIONS = {
    "sin_dem": [f for f in ALL_39_FEATURES if f not in GROUP_DEM],
    "sin_caminos": [f for f in ALL_39_FEATURES if f not in GROUP_CAMINOS],
    "sin_cobertura": [f for f in ALL_39_FEATURES if f not in GROUP_COBERTURA],
    "sin_poblacion": [f for f in ALL_39_FEATURES if f not in GROUP_POBLACION],
    "sin_prior": [f for f in ALL_39_FEATURES if f not in GROUP_PRIOR],
    "sin_rezagos": [f for f in ALL_39_FEATURES if f not in GROUP_REZAGOS],
}


def get_current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


@dataclass
class EvalMetrics:
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
    top10_model_days: float


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    sups: np.ndarray,
    dates: list[Any],
    season: str,
) -> EvalMetrics:
    """Función de evaluación UNIFICADA para todas las comparaciones de Fase 2B (E1)."""
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    base_rate = float(n_pos) / float(n_total)

    pr_auc = float(average_precision_score(y_true, scores)) if n_pos > 0 else 0.0
    lift = pr_auc / base_rate if base_rate > 0.0 else 0.0

    k1 = max(1, int(round(0.01 * n_total)))
    k5 = max(1, int(round(0.05 * n_total)))
    t1 = np.partition(scores, -k1)[-k1]
    t5 = np.partition(scores, -k5)[-k5]

    mask_pos = y_true == 1
    rec1_all = float(np.sum(mask_pos & (scores >= t1))) / float(n_pos) if n_pos > 0 else 0.0
    rec5_all = float(np.sum(mask_pos & (scores >= t5))) / float(n_pos) if n_pos > 0 else 0.0

    mask_1ha = mask_pos & (sups >= 1.0)
    n_1ha = int(np.sum(mask_1ha))
    rec1_1ha = float(np.sum(mask_1ha & (scores >= t1))) / float(n_1ha) if n_1ha > 0 else 0.0
    rec5_1ha = float(np.sum(mask_1ha & (scores >= t5))) / float(n_1ha) if n_1ha > 0 else 0.0

    mask_10ha = mask_pos & (sups >= 10.0)
    n_10ha = int(np.sum(mask_10ha))
    rec1_10ha = float(np.sum(mask_10ha & (scores >= t1))) / float(n_10ha) if n_10ha > 0 else 0.0
    rec5_10ha = float(np.sum(mask_10ha & (scores >= t5))) / float(n_10ha) if n_10ha > 0 else 0.0

    # Redefinición temporal simétrica: % de igniciones en el 10% de días con mayor score medio del modelo
    df_temporal = pl.DataFrame({"date": dates, "score": scores}).group_by("date").agg(pl.col("score").mean()).sort("date")
    k10_days = max(1, int(round(0.10 * df_temporal.height)))
    t10_days = np.partition(df_temporal["score"].to_numpy(), -k10_days)[-k10_days]
    top10_dates = set(df_temporal.filter(pl.col("score") >= t10_days)["date"].to_list())
    pos_dates = [dates[i] for i in range(n_total) if y_true[i] == 1]
    n_pos_in_top10 = sum(1 for d in pos_dates if d in top10_dates)
    top10_model_days = (n_pos_in_top10 / len(pos_dates)) * 100.0 if pos_dates else 0.0

    return EvalMetrics(
        season=season,
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
        top10_model_days=top10_model_days,
    )


def compute_bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Calcula intervalos de confianza bootstrap al 95% para PR-AUC y Rec@1%."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    k1 = max(1, int(round(0.01 * n)))
    t1 = np.partition(scores, -k1)[-k1]

    # Muestreo estratificado por presencia para mantener N_pos estable
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    boot_praucs = np.empty(n_boot, dtype=np.float64)
    boot_rec1s = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sample_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, size=min(n_neg, 20000), replace=True)
        b_idx = np.concatenate([sample_pos, sample_neg])
        b_y = y_true[b_idx]
        b_s = scores[b_idx]

        boot_praucs[b] = average_precision_score(b_y, b_s)
        boot_rec1s[b] = np.mean(scores[sample_pos] >= t1)

    ci_prauc = (float(np.quantile(boot_praucs, 0.025)), float(np.quantile(boot_praucs, 0.975)))
    ci_rec1 = (float(np.quantile(boot_rec1s, 0.025)), float(np.quantile(boot_rec1s, 0.975)))
    return ci_prauc, ci_rec1


def run_full_hardening() -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config("configs/config.yaml")
    train_df = pl.read_parquet("data/processed/training_sample_phase2b.parquet")
    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    special_cell_days_df, _ = build_phase2a_daily_panel(cfg)
    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    weather_df = pl.read_parquet("data/processed/era5_land_extended_weather_nodes.parquet")

    # Mapeo celdas a nodos
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()
    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = trans.transform(cx, cy)
    unique_nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
    tree = cKDTree(node_coords)
    _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    cell_node_df = pl.DataFrame({"cell_id": cell_ids, "node_id": [unique_nodes[idx] for idx in node_idxs]})

    print(f"=== INICIO HARDENING FASE 2B === | RAM RSS inicial: {get_current_rss_mb():.2f} MB")

    baseline_metrics: list[EvalMetrics] = []
    xgb_seed42_metrics: list[EvalMetrics] = []
    xgb_size_weighted_metrics: list[EvalMetrics] = []

    # Diccionarios de almacenamiento
    e2_seed_metrics: dict[int, list[EvalMetrics]] = {s: [] for s in [42, 43, 44, 45, 46]}
    e2_bootstrap_cis: dict[str, Any] = {}
    e2_group_ablations: dict[str, list[EvalMetrics]] = {g: [] for g in GROUP_ABLATIONS}

    e4_calibration_results: list[dict[str, Any]] = []
    e4_reliability_bins: list[dict[str, Any]] = []

    all_scores_for_risk_tiers: list[np.ndarray] = []
    all_y_for_risk_tiers: list[np.ndarray] = []

    # Bucle principal de los 7 pliegues LOSO
    for s_test in DATED_SEASONS:
        print(f"\n--- Procesando Pliegue LOSO: {s_test} | RSS: {get_current_rss_mb():.2f} MB ---")
        train_seasons = [s for s in DATED_SEASONS if s != s_test]
        val_season = train_seasons[-1]
        fit_seasons = train_seasons[:-1]

        fit_sub = train_df.filter(pl.col("season").is_in(fit_seasons))
        val_sub = train_df.filter(pl.col("season") == val_season)

        X_fit = fit_sub.select(ALL_39_FEATURES).to_numpy()
        y_fit = fit_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        w_fit = fit_sub.get_column("sample_weight").to_numpy().astype(np.float32)

        X_val = val_sub.select(ALL_39_FEATURES).to_numpy()
        y_val = val_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        w_val = val_sub.get_column("sample_weight").to_numpy().astype(np.float32)

        # 1. Cargar panel completo de test
        fold_panel = materialize_season_cell_day_panel(
            grid_df=grid_df,
            special_cell_days_df=special_cell_days_df,
            season=s_test,
            include_excluded=False,
        )
        fold_panel = fold_panel.join(static_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(
            prior_df.filter(pl.col("season") == s_test).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )
        fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(weather_df, on=["node_id", "date"], how="left")

        y_test = fold_panel.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        sups_test = fold_panel.get_column("superficie_ha_max").to_numpy().astype(np.float64)
        dates_test = fold_panel.get_column("date").to_list()
        fwi_test = fold_panel.get_column("fwi").to_numpy().astype(np.float64)
        prior_test = fold_panel.get_column("prior_density").to_numpy().astype(np.float64)

        # E1: BASELINE OFICIAL EVALUADO CON LA MISMA FUNCIÓN
        baseline_scores = fwi_test * prior_test
        base_res = evaluate_predictions(y_test, baseline_scores, sups_test, dates_test, s_test)
        baseline_metrics.append(base_res)

        # E2: Entrenar 5 semillas de XGBoost
        X_test_mat = fold_panel.select(ALL_39_FEATURES).to_numpy()
        dtest = xgb.DMatrix(X_test_mat, feature_names=ALL_39_FEATURES)
        del fold_panel  # Liberar memoria


        for seed_val in [42, 43, 44, 45, 46]:
            dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit, feature_names=ALL_39_FEATURES)
            dval = xgb.DMatrix(X_val, label=y_val, weight=w_val, feature_names=ALL_39_FEATURES)
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
                "random_state": seed_val,
                "nthread": 4,
            }
            booster = xgb.train(
                params, dtrain, num_boost_round=500,
                evals=[(dtrain, "train"), (dval, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            scores_s = booster.predict(dtest)
            m_s = evaluate_predictions(y_test, scores_s, sups_test, dates_test, s_test)
            e2_seed_metrics[seed_val].append(m_s)

            if seed_val == 42:
                xgb_seed42_metrics.append(m_s)
                all_scores_for_risk_tiers.append(scores_s)
                all_y_for_risk_tiers.append(y_test)

                # Bootstrap CIs para seed 42
                ci_prauc, ci_rec1 = compute_bootstrap_ci(y_test, scores_s, n_boot=1000, seed=42)
                e2_bootstrap_cis[s_test] = {"ci_prauc": ci_prauc, "ci_rec1": ci_rec1}

                # E4: Calibración en dominio completo
                p_val = booster.predict(dval)
                p_test_raw = scores_s

                logit_val = np.log(np.clip(p_val, 1e-7, 1 - 1e-7) / (1 - np.clip(p_val, 1e-7, 1 - 1e-7))).reshape(-1, 1)
                logit_test = np.log(np.clip(p_test_raw, 1e-7, 1 - 1e-7) / (1 - np.clip(p_test_raw, 1e-7, 1 - 1e-7))).reshape(-1, 1)
                lr = LogisticRegression().fit(logit_val, y_val, sample_weight=w_val)
                p_test_platt = lr.predict_proba(logit_test)[:, 1]

                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(p_val, y_val, sample_weight=w_val)
                p_test_iso = iso.predict(p_test_raw)

                # Brier score con respecto a la tasa base real constante
                pi_const = np.full_like(y_test, base_res.base_rate, dtype=np.float64)
                brier_const = float(brier_score_loss(y_test, pi_const))
                brier_raw = float(brier_score_loss(y_test, p_test_raw))
                brier_platt = float(brier_score_loss(y_test, p_test_platt))
                brier_iso = float(brier_score_loss(y_test, p_test_iso))

                e4_calibration_results.append({
                    "season": s_test,
                    "brier_const": brier_const,
                    "brier_raw": brier_raw,
                    "brier_platt": brier_platt,
                    "brier_iso": brier_iso,
                })

                # Tabla de fiabilidad deciles Platt
                prob_true, prob_pred = calibration_curve(y_test, p_test_platt, n_bins=5)
                e4_reliability_bins.append({
                    "season": s_test,
                    "pred_per_100k": (prob_pred * 100000.0).tolist(),
                    "obs_per_100k": (prob_true * 100000.0).tolist(),
                })

        # E2: Ablaciones por grupo para esta temporada s_test
        for ab_name, ab_feats in GROUP_ABLATIONS.items():
            X_fit_ab = fit_sub.select(ab_feats).to_numpy()
            X_val_ab = val_sub.select(ab_feats).to_numpy()
            dtrain_ab = xgb.DMatrix(X_fit_ab, label=y_fit, weight=w_fit, feature_names=ab_feats)
            dval_ab = xgb.DMatrix(X_val_ab, label=y_val, weight=w_val, feature_names=ab_feats)
            params["random_state"] = 42
            booster_ab = xgb.train(
                params, dtrain_ab, num_boost_round=500,
                evals=[(dtrain_ab, "train"), (dval_ab, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            dtest_ab = xgb.DMatrix(X_test_mat[:, [ALL_39_FEATURES.index(f) for f in ab_feats]], feature_names=ab_feats)
            scores_ab = booster_ab.predict(dtest_ab)
            m_ab = evaluate_predictions(y_test, scores_ab, sups_test, dates_test, s_test)
            e2_group_ablations[ab_name].append(m_ab)

        # E3: Ponderación por tamaño de incendio log10(superficie + 1)
        sup_fit = fit_sub.get_column("superficie_ha_max").to_numpy()
        w_size_fit = w_fit * (1.0 + np.minimum(np.log10(np.maximum(sup_fit, 0.0) + 1.0), 3.0))
        dtrain_sz = xgb.DMatrix(X_fit, label=y_fit, weight=w_size_fit, feature_names=ALL_39_FEATURES)
        booster_sz = xgb.train(
            params, dtrain_sz, num_boost_round=500,
            evals=[(dtrain_sz, "train"), (dval, "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        scores_sz = booster_sz.predict(dtest)
        m_sz = evaluate_predictions(y_test, scores_sz, sups_test, dates_test, s_test)
        xgb_size_weighted_metrics.append(m_sz)

        del X_test_mat, dtest  # Liberar matrices de RAM

    # E5: VALIDACIÓN HACIA ADELANTE (FORWARD WALK)
    print("\n--- Ejecutando E5: Validación hacia adelante (Train 2017-2022 -> Test 2022-2025) ---")
    forward_val_season = "2021_22"
    forward_fit_seasons = ["2017_18", "2019_20", "2020_21"]
    forward_test_seasons = ["2022_23", "2023_24", "2024_25"]

    f_fit = train_df.filter(pl.col("season").is_in(forward_fit_seasons))
    f_val = train_df.filter(pl.col("season") == forward_val_season)

    X_f_fit = f_fit.select(ALL_39_FEATURES).to_numpy()
    y_f_fit = f_fit.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_fit = f_fit.get_column("sample_weight").to_numpy().astype(np.float32)

    X_f_val = f_val.select(ALL_39_FEATURES).to_numpy()
    y_f_val = f_val.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_val = f_val.get_column("sample_weight").to_numpy().astype(np.float32)

    dtrain_fw = xgb.DMatrix(X_f_fit, label=y_f_fit, weight=w_f_fit, feature_names=ALL_39_FEATURES)
    dval_fw = xgb.DMatrix(X_f_val, label=y_f_val, weight=w_f_val, feature_names=ALL_39_FEATURES)
    booster_fw = xgb.train(
        params, dtrain_fw, num_boost_round=500,
        evals=[(dtrain_fw, "train"), (dval_fw, "val")],
        early_stopping_rounds=30, verbose_eval=False,
    )

    e5_forward_results: list[dict[str, Any]] = []
    for s_fw in forward_test_seasons:
        p_fw = materialize_season_cell_day_panel(grid_df, special_cell_days_df, s_fw, include_excluded=False)
        p_fw = p_fw.join(static_df, on="cell_id", how="left")
        p_fw = p_fw.join(prior_df.filter(pl.col("season") == s_fw).select(["cell_id", "prior_density"]), on="cell_id", how="left")
        p_fw = p_fw.join(cell_node_df, on="cell_id", how="left")
        p_fw = p_fw.join(weather_df, on=["node_id", "date"], how="left")

        y_fw = p_fw.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        sups_fw = p_fw.get_column("superficie_ha_max").to_numpy().astype(np.float64)
        dates_fw = p_fw.get_column("date").to_list()
        fwi_fw = p_fw.get_column("fwi").to_numpy().astype(np.float64)
        prior_fw = p_fw.get_column("prior_density").to_numpy().astype(np.float64)

        base_fw_m = evaluate_predictions(y_fw, fwi_fw * prior_fw, sups_fw, dates_fw, s_fw)
        X_fw = p_fw.select(ALL_39_FEATURES).to_numpy()
        sc_fw = booster_fw.predict(xgb.DMatrix(X_fw, feature_names=ALL_39_FEATURES))
        xgb_fw_m = evaluate_predictions(y_fw, sc_fw, sups_fw, dates_fw, s_fw)
        del p_fw, X_fw

        e5_forward_results.append({
            "season": s_fw,
            "baseline": asdict(base_fw_m),
            "xgboost": asdict(xgb_fw_m),
        })

    # E7: CLASES DE RIESGO FIJAS CON PERCENTILES HISTÓRICOS
    print("\n--- Ejecutando E7: Clases de riesgo fijas (Bajo, Moderado, Alto, Extremo) ---")
    all_scores_concat = np.concatenate(all_scores_for_risk_tiers)
    all_y_concat = np.concatenate(all_y_for_risk_tiers)

    # Umbrales históricos tomados del universo completo
    th_mod = float(np.percentile(all_scores_concat, 80.0))
    th_alto = float(np.percentile(all_scores_concat, 95.0))
    th_ext = float(np.percentile(all_scores_concat, 99.0))

    total_cd = len(all_y_concat)
    total_ign = int(np.sum(all_y_concat))

    tiers = [
        ("Bajo (<p80)", all_scores_concat < th_mod),
        ("Moderado (p80-p95)", (all_scores_concat >= th_mod) & (all_scores_concat < th_alto)),
        ("Alto (p95-p99)", (all_scores_concat >= th_alto) & (all_scores_concat < th_ext)),
        ("Extremo (>=p99)", all_scores_concat >= th_ext),
    ]

    e7_risk_tiers: list[dict[str, Any]] = []
    for tier_name, mask_tier in tiers:
        n_cd_tier = int(np.sum(mask_tier))
        pct_cd = (n_cd_tier / total_cd) * 100.0
        n_ign_tier = int(np.sum(all_y_concat[mask_tier]))
        pct_ign = (n_ign_tier / total_ign) * 100.0 if total_ign > 0 else 0.0
        rate_100k = (n_ign_tier / (n_cd_tier / 100000.0)) if n_cd_tier > 0 else 0.0

        e7_risk_tiers.append({
            "tier": tier_name,
            "pct_cell_days": pct_cd,
            "pct_ignitions": pct_ign,
            "n_ignitions": n_ign_tier,
            "rate_per_100k": rate_100k,
        })

    total_time = time.time() - t0
    peak_rss = get_current_rss_mb()
    print(f"\nPipeline finalizado en {total_time:.2f} s. RAM RSS pico: {peak_rss:.2f} MB")

    out_json = Path("reports/phase2b_hardening_results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    results_payload = {
        "baseline_metrics": [asdict(m) for m in baseline_metrics],
        "xgb_seed42_metrics": [asdict(m) for m in xgb_seed42_metrics],
        "xgb_size_weighted_metrics": [asdict(m) for m in xgb_size_weighted_metrics],
        "e2_seed_metrics": {str(k): [asdict(m) for m in v] for k, v in e2_seed_metrics.items()},
        "e2_bootstrap_cis": e2_bootstrap_cis,
        "e2_group_ablations": {k: [asdict(m) for m in v] for k, v in e2_group_ablations.items()},
        "e4_calibration_results": e4_calibration_results,
        "e4_reliability_bins": e4_reliability_bins,
        "e5_forward_results": e5_forward_results,
        "e7_risk_tiers": e7_risk_tiers,
        "peak_rss_mb": peak_rss,
        "total_time_s": total_time,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    return results_payload


if __name__ == "__main__":
    run_full_hardening()
