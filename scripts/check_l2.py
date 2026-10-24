"""Diagnóstico Check L2: Comparativa de Modelo A (Final) vs Modelo B (Forward Walk 2017-2022).

Evalúa el desempeño sobre entradas ERA5-Land en las temporadas prospectivas 2023_24 y 2024_25
bajo dos configuraciones temporales:
1. Año completo (julio a junio).
2. Ventana activa oficial (noviembre a abril).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.config import load_config
from chilefuego.inference.contract import PRUNED_25_FEATURES, InferencePipeline
from chilefuego.labels.panel import (
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

ACTIVE_MONTHS = [11, 12, 1, 2, 3, 4]
SEASONS = ["2023_24", "2024_25"]


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Calcula PR-AUC exacta y Rec@5% sin bootstrap para máxima velocidad."""
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return {"pr_auc": 0.0, "rec5_pct": 0.0, "n_pos": 0}

    pr_auc = float(average_precision_score(y_true, scores))

    k5 = max(1, int(round(0.05 * len(y_true))))
    t5 = float(np.partition(scores, -k5)[-k5])
    rec5 = float(np.sum((y_true == 1) & (scores >= t5)) / n_pos * 100.0)

    return {
        "pr_auc": pr_auc,
        "rec5_pct": rec5,
        "n_pos": n_pos,
        "n_total": len(y_true),
    }


def train_model_b_forward(model_path: Path) -> Path:
    """Entrena y persiste el Modelo B (Validación hacia adelante entrenado hasta 2021_22)."""
    train_df = pl.read_parquet("data/processed/training_sample_phase2b.parquet")
    fw_fit_seasons = ["2017_18", "2019_20", "2020_21"]
    fw_val_season = "2021_22"

    f_fit = train_df.filter(pl.col("season").is_in(fw_fit_seasons))
    f_val = train_df.filter(pl.col("season") == fw_val_season)

    y_f_fit = f_fit.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_fit = f_fit.get_column("sample_weight").to_numpy().astype(np.float32)
    y_f_val = f_val.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_val = f_val.get_column("sample_weight").to_numpy().astype(np.float32)

    X_p_ffit = f_fit.select(PRUNED_25_FEATURES).to_numpy()
    X_p_fval = f_val.select(PRUNED_25_FEATURES).to_numpy()

    dp_ffit = xgb.DMatrix(X_p_ffit, label=y_f_fit, weight=w_f_fit, feature_names=PRUNED_25_FEATURES)
    dp_fval = xgb.DMatrix(X_p_fval, label=y_f_val, weight=w_f_val, feature_names=PRUNED_25_FEATURES)

    params_base = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "random_state": 42,
        "nthread": 4,
    }

    booster_b = xgb.train(
        params_base,
        dp_ffit,
        num_boost_round=500,
        evals=[(dp_ffit, "train"), (dp_fval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster_b.save_model(str(model_path))
    return model_path


def main() -> None:
    t0 = time.time()
    cfg = load_config()

    print("=== INICIANDO DIAGNÓSTICO CHECK L2 ===", flush=True)

    # 1. Modelos
    model_a_path = Path("models/chilefuego_xgb_pruned_v1.json")
    model_b_path = Path("data/processed/chilefuego_xgb_forward_v1.json")

    if not model_b_path.exists():
        print("Entrenando Modelo B (Forward Walk hasta 2021_22)...", flush=True)
        train_model_b_forward(model_b_path)

    pipe_a = InferencePipeline(model_path=str(model_a_path))
    pipe_b = InferencePipeline(model_path=str(model_b_path))

    # 2. Insumos espaciales y meteorológicos
    print("Cargando catálogo de celdas, estáticas, prior y ERA5-Land...", flush=True)
    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    weather_df = pl.read_parquet("data/processed/era5_land_extended_weather_nodes.parquet")
    sp_df, _ = build_phase2a_daily_panel(cfg)

    u_nodes = sorted(weather_df.select(["node_id"]).unique().to_series().to_list())
    node_coords = np.array([[float(n.split("_")[1]), float(n.split("_")[0])] for n in u_nodes])

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    clon, clat = trans.transform(cx, cy)
    tree = cKDTree(node_coords)
    _, cell_node_idxs = tree.query(np.column_stack([clon, clat]), k=1)
    cell_node_df = pl.DataFrame(
        {"cell_id": cell_ids, "node_id": [u_nodes[idx] for idx in cell_node_idxs]}
    )

    # 3. Almacenamiento de evaluaciones
    # Estructura: results[scope][model] -> {"y": [], "sc": []}
    scopes = ["full_year", "nov_apr"]
    models = ["Model A (Final)", "Model B (Forward hasta 2021_22)"]

    eval_data: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        s: {m: {"y": [], "sc": []} for m in models} for s in scopes
    }

    per_season_data: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        season: {s: {} for s in scopes} for season in SEASONS
    }

    for season in SEASONS:
        print(f"\n--- Procesando temporada {season} ---", flush=True)
        panel = materialize_season_cell_day_panel(grid_df, sp_df, season, include_excluded=False)
        panel = panel.join(static_df, on="cell_id", how="left")
        panel = panel.join(
            prior_df.filter(pl.col("season") == season).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )
        panel = panel.join(cell_node_df, on="cell_id", how="left")
        panel = panel.join(weather_df, on=["node_id", "date"], how="left")

        # Seleccionar features requeridas
        feat_df = panel.select(PRUNED_25_FEATURES)
        y_all = panel.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        months_all = panel.get_column("date").dt.month().to_numpy()
        mask_nov_apr = np.isin(months_all, ACTIVE_MONTHS)

        print("Prediciendo con Modelo A (Final)...", flush=True)
        sc_a = pipe_a.predict(feat_df)["raw_score"]

        print("Prediciendo con Modelo B (Forward)...", flush=True)
        sc_b = pipe_b.predict(feat_df)["raw_score"]

        # Acumular full year
        eval_data["full_year"]["Model A (Final)"]["y"].append(y_all)
        eval_data["full_year"]["Model A (Final)"]["sc"].append(sc_a)
        eval_data["full_year"]["Model B (Forward hasta 2021_22)"]["y"].append(y_all)
        eval_data["full_year"]["Model B (Forward hasta 2021_22)"]["sc"].append(sc_b)

        # Acumular nov_apr
        y_na = y_all[mask_nov_apr]
        sc_a_na = sc_a[mask_nov_apr]
        sc_b_na = sc_b[mask_nov_apr]

        eval_data["nov_apr"]["Model A (Final)"]["y"].append(y_na)
        eval_data["nov_apr"]["Model A (Final)"]["sc"].append(sc_a_na)
        eval_data["nov_apr"]["Model B (Forward hasta 2021_22)"]["y"].append(y_na)
        eval_data["nov_apr"]["Model B (Forward hasta 2021_22)"]["sc"].append(sc_b_na)

        # Guardar desglose por temporada
        per_season_data[season]["full_year"]["Model A (Final)"] = compute_metrics(y_all, sc_a)
        per_season_data[season]["full_year"]["Model B (Forward hasta 2021_22)"] = compute_metrics(
            y_all, sc_b
        )
        per_season_data[season]["nov_apr"]["Model A (Final)"] = compute_metrics(y_na, sc_a_na)
        per_season_data[season]["nov_apr"]["Model B (Forward hasta 2021_22)"] = compute_metrics(
            y_na, sc_b_na
        )

    # 4. Consolidar métricas combinadas (2023_24 + 2024_25)
    combined_results: dict[str, dict[str, dict[str, float]]] = {m: {} for m in models}

    for m in models:
        for scope in scopes:
            y_cat = np.concatenate(eval_data[scope][m]["y"])
            sc_cat = np.concatenate(eval_data[scope][m]["sc"])
            combined_results[m][scope] = compute_metrics(y_cat, sc_cat)

    # 5. Imprimir tabla 2x2 markdown
    print("\n" + "=" * 80)
    print("TABLA 2x2: DIAGNÓSTICO L2 (TEMPORADAS 2023_24 Y 2024_25 COMBINADAS)")
    print("=" * 80)
    print()
    print("| Modelo | Año Completo (Jul–Jun) | Temporada Activa (Nov–Abr) |")
    print("| :--- | :--- | :--- |")

    for m in models:
        res_full = combined_results[m]["full_year"]
        res_na = combined_results[m]["nov_apr"]

        full_str = f"PR-AUC: {res_full['pr_auc']:.6f} <br> Rec@5%: {res_full['rec5_pct']:.2f}% <br> N: {int(res_full['n_pos'])} (de {int(res_full['n_total']):,})"
        na_str = f"PR-AUC: {res_na['pr_auc']:.6f} <br> Rec@5%: {res_na['rec5_pct']:.2f}% <br> N: {int(res_na['n_pos'])} (de {int(res_na['n_total']):,})"

        print(f"| **{m}** | {full_str} | {na_str} |")

    # 6. Imprimir desglose por temporada individual
    for season in SEASONS:
        print(f"\n### Desglose Individual: Temporada {season}")
        print("| Modelo | Año Completo (Jul–Jun) | Temporada Activa (Nov–Abr) |")
        print("| :--- | :--- | :--- |")
        for m in models:
            rf = per_season_data[season]["full_year"][m]
            rna = per_season_data[season]["nov_apr"][m]
            full_str = f"PR-AUC: {rf['pr_auc']:.6f} <br> Rec@5%: {rf['rec5_pct']:.2f}% <br> N: {int(rf['n_pos'])} ({int(rf['n_total']):,})"
            na_str = f"PR-AUC: {rna['pr_auc']:.6f} <br> Rec@5%: {rna['rec5_pct']:.2f}% <br> N: {int(rna['n_pos'])} ({int(rna['n_total']):,})"
            print(f"| **{m}** | {full_str} | {na_str} |")

    print(f"\nDiagnóstico completado en {time.time() - t0:.2f} segundos.")


if __name__ == "__main__":
    main()
