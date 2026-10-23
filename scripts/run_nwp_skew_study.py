"""Estudio de Desfase de Fuentes NWP y Emulación a un Paso (J2).

PORQUÉ (Regla 5):
- Evalúa el impacto de reemplazar el reanálisis ERA5-Land por pronósticos numéricos del tiempo (NWP)
  a horizonte 0 (nowcast del día D) y horizonte 1 (pronóstico emitido en D-1 para el día D).
- Emulación a un paso: el estado FWI de D-1 proviene de ERA5-Land (arranque en caliente de reanálisis),
  y las variables meteorológicas de D provienen de las emisiones NWP (H0 y H1).
- Calcula las 11 variables meteorológicas y de FWI para cada horizonte.
- Puntúa las celdas-día con el modelo podado de la validación hacia adelante (entrenado hasta 2021_22).
- Compara contra ERA5-Land en las mismas celdas-día evaluando:
  * PR-AUC exacta (average_precision_score).
  * Rec@5% (recall sobre el top-5% de celdas de mayor score).
  * Distribución porcentual de celdas por clase de riesgo relativo (Bajo, Moderado, Alto, Extremo).
  * Intervalos de confianza 95% mediante bootstrap pareado por bloques de días (B=1000).
- Persiste los resultados en 'results/j2_nwp_skew_study.json'.

CRÍTICO: Este script está preparado para ejecución manual y controlada.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from chilefuego.config import load_config
from chilefuego.features.fwi import compute_fwi_series_1d
from chilefuego.inference.contract import (
    PRUNED_25_FEATURES,
    InferencePipeline,
    build_inference_feature_matrix,
    compute_vpd,
)
from chilefuego.labels.panel import (
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_nwp_skew_study")

SEASONS_J2 = ["2023_24", "2024_25"]
ACTIVE_MONTHS = [11, 12, 1, 2, 3, 4]


def compute_fwi_warm_start_nwp(
    forecast_df: pl.DataFrame,
    fwi_reanalysis_path: Path,
    horizon_tag: str,  # 'h0' o 'h1'
) -> pl.DataFrame:
    """Calcula el FWI para cada nodo y fecha D usando las condiciones de D-1 de ERA5-Land."""
    logger.info(
        f"Calculando FWI con arranque en caliente (D-1 de reanálisis) para horizonte {horizon_tag.upper()}..."
    )
    reanalysis_df = pl.read_parquet(fwi_reanalysis_path)

    dates_unique = sorted(forecast_df.get_column("date").unique().to_list())
    min_date = min(dates_unique) - timedelta(days=1)

    reanalysis_map: dict[tuple[str, date], tuple[float, float, float]] = {}
    for r in reanalysis_df.filter(pl.col("date") >= min_date).iter_rows(named=True):
        reanalysis_map[(r["node_id"], r["date"])] = (
            float(r["ffmc"]),
            float(r["dmc"]),
            float(r["dc"]),
        )

    fwi_rows: list[dict[str, Any]] = []
    t_col = f"temp_noon_c_{horizon_tag}"
    rh_col = f"rh_noon_pct_{horizon_tag}"
    w_col = f"wind_noon_kmh_{horizon_tag}"
    p_col = f"rain_24h_noon_mm_{horizon_tag}"

    for curr_d in dates_unique:
        prev_d = curr_d - timedelta(days=1)
        sub_d = forecast_df.filter(pl.col("date") == curr_d)
        m_val = np.array([curr_d.month], dtype=np.int64)

        for row in sub_d.iter_rows(named=True):
            nid = row["node_id"]
            ffmc0, dmc0, dc0 = reanalysis_map.get((nid, prev_d), (85.0, 6.0, 15.0))

            fwi_res = compute_fwi_series_1d(
                temp_noon_c=np.array([row[t_col]]),
                rh_noon_pct=np.array([row[rh_col]]),
                wind_noon_kmh=np.array([row[w_col]]),
                rain_24h_noon_mm=np.array([row[p_col]]),
                months=m_val,
                ffmc0=ffmc0,
                dmc0=dmc0,
                dc0=dc0,
            )

            fwi_rows.append(
                {
                    "node_id": nid,
                    "date": curr_d,
                    f"ffmc_{horizon_tag}": float(fwi_res["ffmc"][0]),
                    f"dmc_{horizon_tag}": float(fwi_res["dmc"][0]),
                    f"dc_{horizon_tag}": float(fwi_res["dc"][0]),
                    f"isi_{horizon_tag}": float(fwi_res["isi"][0]),
                    f"bui_{horizon_tag}": float(fwi_res["bui"][0]),
                    f"fwi_{horizon_tag}": float(fwi_res["fwi"][0]),
                }
            )

    return pl.DataFrame(fwi_rows)


def paired_day_block_bootstrap(
    y_true: np.ndarray,
    scores_ref: np.ndarray,
    scores_test: np.ndarray,
    dates: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap pareado por bloques de días para evaluar significancia de delta PR-AUC y delta Rec@5%."""
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    k5 = max(1, int(round(0.05 * n_total)))

    delta_pr_list: list[float] = []
    delta_rec5_list: list[float] = []

    date_idx_map = {d: np.where(dates == d)[0] for d in unique_dates}

    for _ in range(n_boot):
        sample_dates = rng.choice(unique_dates, size=n_dates, replace=True)
        idx_sample = np.concatenate([date_idx_map[d] for d in sample_dates])

        y_b = y_true[idx_sample]
        pos_b = int(np.sum(y_b))
        if pos_b == 0:
            continue

        s_ref_b = scores_ref[idx_sample]
        s_test_b = scores_test[idx_sample]

        pr_ref_b = float(average_precision_score(y_b, s_ref_b))
        pr_test_b = float(average_precision_score(y_b, s_test_b))
        delta_pr_list.append(pr_test_b - pr_ref_b)

        kb = max(1, int(round(0.05 * len(y_b))))
        t_ref = np.partition(s_ref_b, -kb)[-kb]
        t_test = np.partition(s_test_b, -kb)[-kb]

        r_ref = float(np.sum((y_b == 1) & (s_ref_b >= t_ref))) / float(pos_b)
        r_test = float(np.sum((y_b == 1) & (s_test_b >= t_test))) / float(pos_b)
        delta_rec5_list.append((r_test - r_ref) * 100.0)

    # Estimadores puntuales en muestra completa
    pr_ref_pt = float(average_precision_score(y_true, scores_ref))
    pr_test_pt = float(average_precision_score(y_true, scores_test))
    delta_pr_pt = pr_test_pt - pr_ref_pt

    t_ref_pt = np.partition(scores_ref, -k5)[-k5]
    t_test_pt = np.partition(scores_test, -k5)[-k5]
    rec5_ref_pt = float(np.sum((y_true == 1) & (scores_ref >= t_ref_pt))) / float(n_pos) * 100.0
    rec5_test_pt = float(np.sum((y_true == 1) & (scores_test >= t_test_pt))) / float(n_pos) * 100.0
    delta_rec5_pt = rec5_test_pt - rec5_ref_pt

    pr_ci = [float(np.percentile(delta_pr_list, 2.5)), float(np.percentile(delta_pr_list, 97.5))]
    rec5_ci = [
        float(np.percentile(delta_rec5_list, 2.5)),
        float(np.percentile(delta_rec5_list, 97.5)),
    ]

    return {
        "pr_auc_ref": pr_ref_pt,
        "pr_auc_test": pr_test_pt,
        "delta_pr_auc": delta_pr_pt,
        "ci_delta_pr_auc": pr_ci,
        "sig_pr_auc": bool(pr_ci[0] > 0 or pr_ci[1] < 0),
        "rec5_ref_pct": rec5_ref_pt,
        "rec5_test_pct": rec5_test_pt,
        "delta_rec5_pp": delta_rec5_pt,
        "ci_delta_rec5_pp": rec5_ci,
        "sig_rec5": bool(rec5_ci[0] > 0 or rec5_ci[1] < 0),
    }


def run_skew_study(
    nwp_dir: Path = Path("data/raw/nwp_archive"),
    results_dir: Path = Path("results"),
    model_path: Path = Path("models/chilefuego_xgb_pruned_v1.json"),
    calibrator_path: Path = Path("models/platt_calibrator.json"),
    thresholds_path: Path = Path("models/risk_thresholds.json"),
) -> dict[str, Any]:
    """Ejecuta el estudio riguroso de desfase de fuentes NWP frente a ERA5-Land."""
    t0 = time.time()
    cfg = load_config()

    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    fwi_reanalysis_path = Path("data/processed/era5_land_daily_fwi_nodes.parquet")

    # Verificar existencia de archivos de pronósticos archivados
    for season in SEASONS_J2:
        s_file = nwp_dir / f"nwp_{season}.parquet"
        if not s_file.exists():
            raise FileNotFoundError(
                f"Archivo NWP para {season} no encontrado en {s_file}. "
                "Ejecuta primero: python scripts/download_nwp_archive.py --confirm"
            )

    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    # Mapeo a nodos
    era5_df = pl.read_parquet(weather_path)
    u_nodes = sorted(era5_df.select(["node_id"]).unique().to_series().to_list())
    node_coords = np.array([[float(n.split("_")[1]), float(n.split("_")[0])] for n in u_nodes])

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    clon, clat = trans.transform(cx, cy)
    tree = cKDTree(node_coords)
    _, cell_node_idxs = tree.query(np.column_stack([clon, clat]), k=1)
    cell_node_df = pl.DataFrame(
        {"cell_id": cell_ids, "node_id": [u_nodes[idx] for idx in cell_node_idxs]}
    )

    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    sp_df, _ = build_phase2a_daily_panel(cfg)

    pipeline = InferencePipeline(
        model_path=str(model_path),
        calibrator_path=str(calibrator_path),
        thresholds_path=str(thresholds_path),
    )

    eval_results: list[dict[str, Any]] = []

    for season in SEASONS_J2:
        logger.info(f"=== Procesando estudio NWP para temporada {season} ===")
        nwp_season_path = nwp_dir / f"nwp_{season}.parquet"
        nwp_df = pl.read_parquet(nwp_season_path)

        # FWI arranque en caliente para H0 y H1
        fwi_h0 = compute_fwi_warm_start_nwp(nwp_df, fwi_reanalysis_path, horizon_tag="h0")
        fwi_h1 = compute_fwi_warm_start_nwp(nwp_df, fwi_reanalysis_path, horizon_tag="h1")
        nwp_merged = nwp_df.join(fwi_h0, on=["node_id", "date"]).join(
            fwi_h1, on=["node_id", "date"]
        )

        # Panel de la temporada
        season_panel = materialize_season_cell_day_panel(
            grid_df, sp_df, season, include_excluded=False
        )
        # Filtrar solo meses activos nov-abr
        season_panel = season_panel.filter(pl.col("date").dt.month().is_in(ACTIVE_MONTHS))
        season_panel = season_panel.join(static_df, on="cell_id", how="left")
        season_panel = season_panel.join(
            prior_df.filter(pl.col("season") == season).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )
        season_panel = season_panel.join(cell_node_df, on="cell_id", how="left")

        # 1. Base ERA5-Land
        era5_sub = pl.read_parquet(weather_path)
        p_era5 = season_panel.join(era5_sub, on=["node_id", "date"], how="left")
        y_true = p_era5.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        dates_arr = p_era5.get_column("date").to_numpy()

        feat_era5 = p_era5.select(PRUNED_25_FEATURES)
        preds_era5 = pipeline.predict(feat_era5)

        # 2. Open-Meteo NWP H0
        p_nwp = season_panel.join(nwp_merged, on=["node_id", "date"], how="left")
        w_h0_dict: dict[str, np.ndarray] = {
            "temp_noon_c": p_nwp.get_column("temp_noon_c_h0").to_numpy().astype(np.float32),
            "rh_noon_pct": p_nwp.get_column("rh_noon_pct_h0").to_numpy().astype(np.float32),
            "wind_noon_kmh": p_nwp.get_column("wind_noon_kmh_h0").to_numpy().astype(np.float32),
            "rain_24h_noon_mm": p_nwp.get_column("rain_24h_noon_mm_h0")
            .to_numpy()
            .astype(np.float32),
            "fwi": p_nwp.get_column("fwi_h0").to_numpy().astype(np.float32),
            "isi": p_nwp.get_column("isi_h0").to_numpy().astype(np.float32),
            "bui": p_nwp.get_column("bui_h0").to_numpy().astype(np.float32),
            "ffmc": p_nwp.get_column("ffmc_h0").to_numpy().astype(np.float32),
            "dmc": p_nwp.get_column("dmc_h0").to_numpy().astype(np.float32),
            "dc": p_nwp.get_column("dc_h0").to_numpy().astype(np.float32),
        }
        w_h0_dict["vpd_noon_kpa"] = np.array(
            [
                compute_vpd(float(t), float(rh))
                for t, rh in zip(w_h0_dict["temp_noon_c"], w_h0_dict["rh_noon_pct"], strict=True)
            ],
            dtype=np.float32,
        )
        feat_h0 = build_inference_feature_matrix(
            static_df=p_nwp.select(static_df.columns),
            prior_series=p_nwp.get_column("prior_density").to_numpy().astype(np.float32),
            weather_dict_by_cell=w_h0_dict,
        )
        preds_h0 = pipeline.predict(feat_h0)

        # 3. Open-Meteo NWP H1
        w_h1_dict: dict[str, np.ndarray] = {
            "temp_noon_c": p_nwp.get_column("temp_noon_c_h1").to_numpy().astype(np.float32),
            "rh_noon_pct": p_nwp.get_column("rh_noon_pct_h1").to_numpy().astype(np.float32),
            "wind_noon_kmh": p_nwp.get_column("wind_noon_kmh_h1").to_numpy().astype(np.float32),
            "rain_24h_noon_mm": p_nwp.get_column("rain_24h_noon_mm_h1")
            .to_numpy()
            .astype(np.float32),
            "fwi": p_nwp.get_column("fwi_h1").to_numpy().astype(np.float32),
            "isi": p_nwp.get_column("isi_h1").to_numpy().astype(np.float32),
            "bui": p_nwp.get_column("bui_h1").to_numpy().astype(np.float32),
            "ffmc": p_nwp.get_column("ffmc_h1").to_numpy().astype(np.float32),
            "dmc": p_nwp.get_column("dmc_h1").to_numpy().astype(np.float32),
            "dc": p_nwp.get_column("dc_h1").to_numpy().astype(np.float32),
        }
        w_h1_dict["vpd_noon_kpa"] = np.array(
            [
                compute_vpd(float(t), float(rh))
                for t, rh in zip(w_h1_dict["temp_noon_c"], w_h1_dict["rh_noon_pct"], strict=True)
            ],
            dtype=np.float32,
        )
        feat_h1 = build_inference_feature_matrix(
            static_df=p_nwp.select(static_df.columns),
            prior_series=p_nwp.get_column("prior_density").to_numpy().astype(np.float32),
            weather_dict_by_cell=w_h1_dict,
        )
        preds_h1 = pipeline.predict(feat_h1)

        # Evaluación bootstrap pareado H0 vs ERA5 y H1 vs ERA5
        logger.info(f"Calculando bootstrap pareado por bloques de días para {season}...")
        boot_h0 = paired_day_block_bootstrap(
            y_true, preds_era5["raw_score"], preds_h0["raw_score"], dates_arr
        )
        boot_h1 = paired_day_block_bootstrap(
            y_true, preds_era5["raw_score"], preds_h1["raw_score"], dates_arr
        )

        def get_dist(tier_arr: np.ndarray) -> dict[str, float]:
            tot = len(tier_arr)
            return {
                "bajo": float(np.sum(tier_arr == "Bajo") / tot * 100.0),
                "moderado": float(np.sum(tier_arr == "Moderado") / tot * 100.0),
                "alto": float(np.sum(tier_arr == "Alto") / tot * 100.0),
                "extremo": float(np.sum(tier_arr == "Extremo") / tot * 100.0),
            }

        eval_results.append(
            {
                "season": season,
                "n_pos": int(np.sum(y_true)),
                "n_cell_days": len(y_true),
                "era5_land": {
                    "pr_auc": boot_h0["pr_auc_ref"],
                    "rec5_pct": boot_h0["rec5_ref_pct"],
                    "tier_distribution_pct": get_dist(preds_era5["risk_tier"]),
                },
                "nwp_h0": {
                    "pr_auc": boot_h0["pr_auc_test"],
                    "delta_pr_auc": boot_h0["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_h0["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_h0["sig_pr_auc"],
                    "rec5_pct": boot_h0["rec5_test_pct"],
                    "delta_rec5_pp": boot_h0["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_h0["ci_delta_rec5_pp"],
                    "sig_rec5": boot_h0["sig_rec5"],
                    "tier_distribution_pct": get_dist(preds_h0["risk_tier"]),
                },
                "nwp_h1": {
                    "pr_auc": boot_h1["pr_auc_test"],
                    "delta_pr_auc": boot_h1["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_h1["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_h1["sig_pr_auc"],
                    "rec5_pct": boot_h1["rec5_test_pct"],
                    "delta_rec5_pp": boot_h1["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_h1["ci_delta_rec5_pp"],
                    "sig_rec5": boot_h1["sig_rec5"],
                    "tier_distribution_pct": get_dist(preds_h1["risk_tier"]),
                },
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "j2_nwp_skew_study.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study": "J2 NWP Operational Latency and Forecast Skew Study",
                "executed_at": datetime.now(UTC).isoformat(),
                "model_version": "pruned_25_forward",
                "evaluations": eval_results,
                "elapsed_seconds": round(time.time() - t0, 2),
            },
            f,
            indent=2,
        )

    logger.info(f"Estudio J2 completado en {round(time.time() - t0, 2)}s. Resultados en {out_json}")
    return {"results_path": str(out_json), "evaluations": eval_results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutar estudio J2 de emulación NWP y desfase operativo."
    )
    parser.add_argument(
        "--nwp-dir",
        type=str,
        default="data/raw/nwp_archive",
        help="Directorio con archivos Parquet NWP",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directorio para resultados JSON"
    )
    args = parser.parse_args()

    res = run_skew_study(nwp_dir=Path(args.nwp_dir), results_dir=Path(args.results_dir))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
