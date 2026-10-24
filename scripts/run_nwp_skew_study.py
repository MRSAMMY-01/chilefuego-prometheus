"""Estudio de Desfase de Fuentes NWP y Persistencia Operativa (Fase 3A / J2 / K1-K6 / L3-L5).

PORQUÉ (Regla 5):
- Evalúa el impacto de reemplazar el reanálisis ERA5-Land por pronósticos numéricos del tiempo (NWP)
  a tres niveles operacionales:
  * H0: "desfase de fuente y resolución (GFS horizonte ~0 contra ERA5-Land)", sin atribuir error a 24h.
  * Persistencia 24 h: cota inferior metodológica (condiciones meteorológicas observadas de D-1 aplicadas a D).
  * H1 real: "pronóstico numérico a 24 h" usando las emisiones previas (*_previous_day1 a las 16:00 UTC y
    acumulado de 24 h de precipitación), evaluado en 2024_25 donde la cobertura de archivo es completa (0% nulos).
- Emulación a un paso: el estado FWI de D-1 proviene de ERA5-Land (arranque en caliente de reanálisis),
  y las variables meteorológicas de D provienen de NWP (H0, Persistencia, H1 real).
- Evalúa el Modelo B (validación hacia adelante entrenado hasta 2021_22 sin filtración de datos)
  y el Baseline oficial (FWI x Prior) bajo idénticas condiciones meteorológicas.
- Evalúa el mapeo de cuantiles monótono fuera de muestra (entrenado en una temporada, probado en la otra,
  en ambos sentidos) para corregir la inflación/deflación de clases de riesgo relativo.
- Criterios de aceptación K5:
  (a) Modelo con NWP supera a Baseline con NWP en Rec@5% con IC pareado excluyendo 0.
  (b) Reparto de clases dentro de +-30% relativo respecto a ERA5 en los mismos días.
  (c) Caída de PR-AUC con IC al 95%.
- Persiste los resultados completos en 'results/j2_nwp_skew_study.json'.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_nwp_skew_study")

SEASONS_J2 = ["2023_24", "2024_25"]
ACTIVE_MONTHS = [11, 12, 1, 2, 3, 4]

NWP_SEASONS_RANGES: dict[str, tuple[date, date]] = {
    "2023_24": (date(2023, 11, 1), date(2024, 4, 30)),
    "2024_25": (date(2024, 11, 1), date(2025, 4, 30)),
}


class MonotonicQuantileMapper:
    """Mapeo monótono de cuantiles (Q-Q mapping) a nivel de puntaje (NWP a ERA5).

    PORQUÉ (Regla K4):
    - Ajusta un mapeador empírico monótono para alinear la distribución de puntajes NWP
      con la distribución de referencia ERA5-Land.
    - Se ajusta estrictamente en una temporada y se evalúa en la otra (nunca en la evaluada).
    - Preserva el ordenamiento relativo de las celdas (monotonía estricta).
    """

    def __init__(self, n_quantiles: int = 2000) -> None:
        self.n_quantiles = n_quantiles
        self.q_grid = np.linspace(0.0, 100.0, n_quantiles)
        self.u_nwp: np.ndarray | None = None
        self.v_era5: np.ndarray | None = None

    def fit(self, scores_nwp: np.ndarray, scores_era5: np.ndarray) -> MonotonicQuantileMapper:
        self.u_nwp = np.percentile(scores_nwp, self.q_grid)
        self.v_era5 = np.percentile(scores_era5, self.q_grid)
        # Asegurar monotonía no decreciente
        self.u_nwp = np.maximum.accumulate(self.u_nwp)
        self.v_era5 = np.maximum.accumulate(self.v_era5)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.u_nwp is None or self.v_era5 is None:
            raise ValueError("El mapeador debe ajustarse antes de ser aplicado.")
        return np.interp(scores, self.u_nwp, self.v_era5)


def fast_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Calcula PR-AUC exacta de forma vectorizada y determinista."""
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores)
    y_ord = y_true[order]
    pos_idx = np.flatnonzero(y_ord)
    tp = np.arange(1, len(pos_idx) + 1, dtype=np.float64)
    prec = tp / (pos_idx.astype(np.float64) + 1.0)
    return float(np.sum(prec) / float(n_pos))


def compute_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Calcula PR-AUC y Rec@5% exactas."""
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return {"pr_auc": 0.0, "rec5_pct": 0.0, "n_pos": 0.0}

    pr = fast_average_precision(y_true, scores)
    k5 = max(1, int(round(0.05 * len(y_true))))
    t5 = float(np.partition(scores, -k5)[-k5])
    rec5 = float(np.sum((y_true == 1) & (scores >= t5)) / float(n_pos) * 100.0)
    return {"pr_auc": pr, "rec5_pct": rec5, "n_pos": float(n_pos)}


def get_tier_distribution(
    scores: np.ndarray, t80: float, t95: float, t99: float
) -> dict[str, float]:
    """Calcula la distribución porcentual de celdas por clase de riesgo relativo."""
    tot = float(len(scores))
    if tot == 0:
        return {"bajo": 0.0, "moderado": 0.0, "alto": 0.0, "extremo": 0.0}
    return {
        "bajo": float(np.sum(scores < t80) / tot * 100.0),
        "moderado": float(np.sum((scores >= t80) & (scores < t95)) / tot * 100.0),
        "alto": float(np.sum((scores >= t95) & (scores < t99)) / tot * 100.0),
        "extremo": float(np.sum(scores >= t99) / tot * 100.0),
    }


def load_or_extract_nwp_dataframe(
    season: str,
    nwp_dir: Path,
    expected_nodes: list[str],
) -> pl.DataFrame:
    """Extrae o consolida los pronósticos NWP para la temporada requerida.

    PORQUÉ (Adenda H4 / K2):
    - H0 representa el desfase de fuente y resolución (GFS horizonte ~0 contra ERA5-Land),
      tomando las condiciones de las 16:00 UTC (idx16).
    - Persistencia 24 h representa la cota inferior meteorológica (idx16 - 24).
    - H1 real representa el pronóstico numérico a 24 h (columnas *_previous_day1 a las 16:00 UTC
      y lluvia acumulada de 24 h), disponible para 2024_25.
    """
    nwp_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = nwp_dir / f"nwp_{season}.parquet"
    if parquet_path.exists():
        logger.info(f"Cargando dataset NWP pre-consolidado: {parquet_path}")
        return pl.read_parquet(parquet_path)

    logger.info(
        f"Consolidando pronósticos NWP para temporada {season} desde archivos JSON en disco..."
    )
    s_start, s_end = NWP_SEASONS_RANGES[season]

    search_dirs = [nwp_dir, Path("data/raw/open_meteo")]
    node_file_map: dict[str, tuple[Path, int]] = {}

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        s_season = sdir / season
        if s_season.exists():
            for f in s_season.glob("*.json"):
                nid = f.stem
                if nid in expected_nodes and nid not in node_file_map:
                    node_file_map[nid] = (f, 0)
        for f in sdir.glob(f"batch_{season}_*.json"):
            try:
                with open(f, encoding="utf-8") as jf:
                    c = json.load(jf)
                b_nodes = c.get("nodes", [])
                for b_idx, bn in enumerate(b_nodes):
                    if bn in expected_nodes and bn not in node_file_map:
                        node_file_map[bn] = (f, b_idx)
            except Exception:
                pass

    if not node_file_map:
        raise FileNotFoundError(
            f"No se encontraron pronósticos en disco para {season} en {nwp_dir} ni en data/raw/open_meteo."
        )

    logger.info(f"Nodos indexados para {season}: {len(node_file_map)}/{len(expected_nodes)}")

    extracted_records: list[dict[str, Any]] = []
    loaded_cache: dict[Path, Any] = {}

    for nid in expected_nodes:
        if nid not in node_file_map:
            continue
        fpath, b_idx = node_file_map[nid]
        if fpath not in loaded_cache:
            with open(fpath, encoding="utf-8") as jf:
                loaded_cache[fpath] = json.load(jf)
        content = loaded_cache[fpath]

        if "data" in content and isinstance(content["data"], list) and content["data"]:
            node_data = content["data"][b_idx]
        else:
            node_data = content

        hourly = node_data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            continue

        time_idx_map = {t_str: idx for idx, t_str in enumerate(times)}
        t_base = hourly.get("temperature_2m", [])
        rh_base = hourly.get("relative_humidity_2m", [])
        w_base = hourly.get("wind_speed_10m", [])
        p_base = hourly.get("precipitation", [])

        # Variables previous_day1 (pronóstico 24h)
        t_prev1 = hourly.get("temperature_2m_previous_day1", [])
        rh_prev1 = hourly.get("relative_humidity_2m_previous_day1", [])
        w_prev1 = hourly.get("wind_speed_10m_previous_day1", [])
        p_prev1 = hourly.get("precipitation_previous_day1", [])
        has_prev1 = bool(t_prev1 and len(t_prev1) > 0 and t_prev1[0] is not None)

        curr_d = s_start
        while curr_d <= s_end:
            t16_str = f"{curr_d.isoformat()}T16:00"
            if t16_str in time_idx_map:
                idx16 = time_idx_map[t16_str]

                # H0: nowcast / desfase de fuente y resolución
                t_h0 = float(t_base[idx16])
                rh_h0 = float(rh_base[idx16])
                w_h0 = float(w_base[idx16])
                rain24_h0 = float(sum(p_base[max(0, idx16 - 23) : idx16 + 1]))

                # Persistencia 24 h (cota inferior)
                idx_h1 = idx16 - 24
                if idx_h1 >= 0:
                    t_h1 = float(t_base[idx_h1])
                    rh_h1 = float(rh_base[idx_h1])
                    w_h1 = float(w_base[idx_h1])
                    rain24_h1 = float(sum(p_base[max(0, idx_h1 - 23) : idx_h1 + 1]))
                else:
                    t_h1 = t_h0
                    rh_h1 = rh_h0
                    w_h1 = w_h0
                    rain24_h1 = rain24_h0

                rec: dict[str, Any] = {
                    "node_id": nid,
                    "date": curr_d,
                    "temp_noon_c_h0": t_h0,
                    "rh_noon_pct_h0": rh_h0,
                    "wind_noon_kmh_h0": w_h0,
                    "rain_24h_noon_mm_h0": rain24_h0,
                    "temp_noon_c_h1": t_h1,
                    "rh_noon_pct_h1": rh_h1,
                    "wind_noon_kmh_h1": w_h1,
                    "rain_24h_noon_mm_h1": rain24_h1,
                }

                # H1 real: pronóstico numérico a 24 h
                if (
                    has_prev1
                    and idx16 < len(t_prev1)
                    and t_prev1[idx16] is not None
                    and rh_prev1[idx16] is not None
                ):
                    t_real = float(t_prev1[idx16])
                    rh_real = float(rh_prev1[idx16])
                    w_real = float(w_prev1[idx16])
                    p_slice = p_prev1[max(0, idx16 - 23) : idx16 + 1]
                    rain24_real = (
                        float(sum(p_slice))
                        if (p_slice and all(x is not None for x in p_slice))
                        else rain24_h0
                    )
                    rec["temp_noon_c_h1_real"] = t_real
                    rec["rh_noon_pct_h1_real"] = rh_real
                    rec["wind_noon_kmh_h1_real"] = w_real
                    rec["rain_24h_noon_mm_h1_real"] = rain24_real

                extracted_records.append(rec)
            curr_d += timedelta(days=1)

    df_consolidated = pl.DataFrame(extracted_records)
    if df_consolidated.height > 0:
        df_consolidated.write_parquet(parquet_path)
        logger.info(
            f"Archivo Parquet consolidado guardado en {parquet_path} ({df_consolidated.height} registros)."
        )
    return df_consolidated


def compute_fwi_warm_start_nwp(
    forecast_df: pl.DataFrame,
    fwi_reanalysis_path: Path,
    horizon_tag: str,  # 'h0', 'h1', 'h1_real'
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

    if t_col not in forecast_df.columns:
        return pl.DataFrame()

    for curr_d in dates_unique:
        prev_d = curr_d - timedelta(days=1)
        sub_d = forecast_df.filter(pl.col("date") == curr_d)
        m_val = np.array([curr_d.month], dtype=np.int64)

        for row in sub_d.iter_rows(named=True):
            nid = row["node_id"]
            if row[t_col] is None or row[rh_col] is None:
                continue

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


def run_joint_day_block_bootstrap(
    y_true: np.ndarray,
    dates_arr: np.ndarray,
    score_series: dict[str, np.ndarray],
    paired_comparisons: list[tuple[str, str, str, str]],  # (name, key_ref, key_test, metric_type)
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Ejecuta bootstrap pareado por bloques de días de manera conjunta y vectorizada."""
    rng = np.random.default_rng(seed)
    unique_dates = np.unique(dates_arr)
    n_dates = len(unique_dates)
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    k5 = max(1, int(round(0.05 * n_total)))

    date_idx_map = [np.flatnonzero(dates_arr == d) for d in unique_dates]

    # Almacenamiento de estimadores de muestra completa
    point_estimates: dict[str, dict[str, float]] = {}
    for k, s_arr in score_series.items():
        t5_pt = float(np.partition(s_arr, -k5)[-k5])
        rec5_pt = float(np.sum((y_true == 1) & (s_arr >= t5_pt)) / float(n_pos) * 100.0)
        pr_pt = fast_average_precision(y_true, s_arr)
        point_estimates[k] = {"pr_auc": pr_pt, "rec5_pct": rec5_pt}

    boot_deltas: dict[str, list[float]] = {}
    for comp in paired_comparisons:
        boot_deltas[comp[0]] = []
        if comp[3] == "both":
            boot_deltas[f"{comp[0]}_pr"] = []

    # Bucle conjunto de remuestreo
    for _ in range(n_boot):
        sample_dates = rng.choice(n_dates, size=n_dates, replace=True)
        idx_sample = np.concatenate([date_idx_map[d] for d in sample_dates])

        y_b = y_true[idx_sample]
        pos_b = np.flatnonzero(y_b)
        n_pos_b = len(pos_b)
        if n_pos_b == 0:
            continue

        kb = max(1, int(round(0.05 * len(y_b))))

        # Cache de métricas en la réplica
        replica_rec5: dict[str, float] = {}
        replica_pr: dict[str, float] = {}

        for comp_name, k_ref, k_test, mtype in paired_comparisons:
            needed_keys = {k_ref, k_test}
            for k in needed_keys:
                if k not in replica_rec5:
                    sb = score_series[k][idx_sample]
                    tb = float(np.partition(sb, -kb)[-kb])
                    replica_rec5[k] = float(np.sum(sb[pos_b] >= tb) / float(n_pos_b) * 100.0)
                if mtype == "both" and k not in replica_pr:
                    sb = score_series[k][idx_sample]
                    order = np.argsort(-sb)
                    pos_ord = np.flatnonzero(y_b[order])
                    pr_val = float(
                        np.sum(
                            np.arange(1, len(pos_ord) + 1, dtype=np.float64)
                            / (pos_ord.astype(np.float64) + 1.0)
                        )
                        / float(n_pos_b)
                    )
                    replica_pr[k] = pr_val

            if mtype == "rec5_only":
                delta_rec5 = replica_rec5[k_test] - replica_rec5[k_ref]
                boot_deltas[comp_name].append(delta_rec5)
            elif mtype == "both":
                delta_rec5 = replica_rec5[k_test] - replica_rec5[k_ref]
                delta_pr = replica_pr[k_test] - replica_pr[k_ref]
                boot_deltas[comp_name].append(delta_rec5)
                boot_deltas[f"{comp_name}_pr"].append(delta_pr)

    results: dict[str, dict[str, Any]] = {}
    for comp_name, k_ref, k_test, mtype in paired_comparisons:
        deltas = boot_deltas[comp_name]
        ci_rec5 = [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]
        pt_rec5 = point_estimates[k_test]["rec5_pct"] - point_estimates[k_ref]["rec5_pct"]

        res_dict: dict[str, Any] = {
            "ref_key": k_ref,
            "test_key": k_test,
            "pr_auc_ref": point_estimates[k_ref]["pr_auc"],
            "pr_auc_test": point_estimates[k_test]["pr_auc"],
            "delta_pr_auc": point_estimates[k_test]["pr_auc"] - point_estimates[k_ref]["pr_auc"],
            "ci_delta_pr_auc": [0.0, 0.0],
            "sig_pr_auc": False,
            "rec5_ref_pct": point_estimates[k_ref]["rec5_pct"],
            "rec5_test_pct": point_estimates[k_test]["rec5_pct"],
            "delta_rec5_pp": pt_rec5,
            "ci_delta_rec5_pp": ci_rec5,
            "sig_rec5": bool(ci_rec5[0] > 0 or ci_rec5[1] < 0),
        }

        if mtype == "both":
            pr_deltas = boot_deltas.get(f"{comp_name}_pr", [])
            ci_pr = (
                [float(np.percentile(pr_deltas, 2.5)), float(np.percentile(pr_deltas, 97.5))]
                if pr_deltas
                else [0.0, 0.0]
            )
            res_dict["ci_delta_pr_auc"] = ci_pr
            res_dict["sig_pr_auc"] = bool(ci_pr[0] > 0 or ci_pr[1] < 0)

        results[comp_name] = res_dict

    return results


def run_skew_study(
    nwp_dir: Path = Path("data/raw/nwp_archive"),
    results_dir: Path = Path("results"),
    model_path: Path = Path("models/chilefuego_xgb_forward_v1.json"),
    calibrator_path: Path = Path("models/platt_calibrator.json"),
    thresholds_path: Path = Path("models/risk_thresholds.json"),
    n_boot: int = 500,
) -> dict[str, Any]:
    """Ejecuta el estudio riguroso J2 / K1-K6 / L3-L5 de emulación y desfase operativo."""
    t0 = time.time()
    cfg = load_config()

    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    fwi_reanalysis_path = Path("data/processed/era5_land_daily_fwi_nodes.parquet")

    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    era5_df = pl.read_parquet(weather_path)
    u_nodes = sorted(era5_df.select(["node_id"]).unique().to_series().to_list())
    node_coords = np.array([[float(n.split("_")[1]), float(n.split("_")[0])] for n in u_nodes])

    # Verificar o consolidar archivos Parquet NWP
    for season in SEASONS_J2:
        _ = load_or_extract_nwp_dataframe(season=season, nwp_dir=nwp_dir, expected_nodes=u_nodes)

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

    logger.info(f"Cargando Modelo B (Validación Forward hasta 2021_22): {model_path}")
    pipeline = InferencePipeline(
        model_path=str(model_path),
        calibrator_path=str(calibrator_path),
        thresholds_path=str(thresholds_path),
    )

    season_cache: dict[str, dict[str, Any]] = {}
    eval_results: list[dict[str, Any]] = []

    for season in SEASONS_J2:
        logger.info(f"=== Procesando estudio NWP para temporada {season} ===")
        nwp_df = load_or_extract_nwp_dataframe(
            season=season, nwp_dir=nwp_dir, expected_nodes=u_nodes
        )

        fwi_h0 = compute_fwi_warm_start_nwp(nwp_df, fwi_reanalysis_path, horizon_tag="h0")
        fwi_h1 = compute_fwi_warm_start_nwp(nwp_df, fwi_reanalysis_path, horizon_tag="h1")
        nwp_merged = nwp_df.join(fwi_h0, on=["node_id", "date"]).join(
            fwi_h1, on=["node_id", "date"]
        )

        has_h1_real = "temp_noon_c_h1_real" in nwp_df.columns
        if has_h1_real:
            fwi_h1_real = compute_fwi_warm_start_nwp(
                nwp_df, fwi_reanalysis_path, horizon_tag="h1_real"
            )
            if fwi_h1_real.height > 0:
                nwp_merged = nwp_merged.join(fwi_h1_real, on=["node_id", "date"])

        # Panel de la temporada filtrado nov-abr
        season_panel = materialize_season_cell_day_panel(
            grid_df, sp_df, season, include_excluded=False
        )
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
        prior_arr = p_era5.get_column("prior_density").to_numpy().astype(np.float32)

        feat_era5 = p_era5.select(PRUNED_25_FEATURES)
        preds_era5 = pipeline.predict(feat_era5)["raw_score"]
        base_scores_era5 = p_era5.get_column("fwi").to_numpy().astype(np.float32) * prior_arr

        # Umbrales empíricos ERA5-Land (percentiles 80, 95, 99)
        m_t80 = float(np.percentile(preds_era5, 80))
        m_t95 = float(np.percentile(preds_era5, 95))
        m_t99 = float(np.percentile(preds_era5, 99))

        b_t80 = float(np.percentile(base_scores_era5, 80))
        b_t95 = float(np.percentile(base_scores_era5, 95))
        b_t99 = float(np.percentile(base_scores_era5, 99))

        # 2. NWP H0
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
            prior_series=prior_arr,
            weather_dict_by_cell=w_h0_dict,
        )
        preds_h0 = pipeline.predict(feat_h0)["raw_score"]
        base_scores_h0 = w_h0_dict["fwi"] * prior_arr

        # 3. NWP Persistencia 24h
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
            prior_series=prior_arr,
            weather_dict_by_cell=w_h1_dict,
        )
        preds_h1 = pipeline.predict(feat_h1)["raw_score"]
        base_scores_h1 = w_h1_dict["fwi"] * prior_arr

        # 4. NWP H1 real (2024_25)
        preds_h1_real: np.ndarray | None = None
        base_scores_h1_real: np.ndarray | None = None
        if "fwi_h1_real" in p_nwp.columns and "temp_noon_c_h1_real" in p_nwp.columns:
            w_h1_real_dict: dict[str, np.ndarray] = {
                "temp_noon_c": p_nwp.get_column("temp_noon_c_h1_real")
                .to_numpy()
                .astype(np.float32),
                "rh_noon_pct": p_nwp.get_column("rh_noon_pct_h1_real")
                .to_numpy()
                .astype(np.float32),
                "wind_noon_kmh": p_nwp.get_column("wind_noon_kmh_h1_real")
                .to_numpy()
                .astype(np.float32),
                "rain_24h_noon_mm": p_nwp.get_column("rain_24h_noon_mm_h1_real")
                .to_numpy()
                .astype(np.float32),
                "fwi": p_nwp.get_column("fwi_h1_real").to_numpy().astype(np.float32),
                "isi": p_nwp.get_column("isi_h1_real").to_numpy().astype(np.float32),
                "bui": p_nwp.get_column("bui_h1_real").to_numpy().astype(np.float32),
                "ffmc": p_nwp.get_column("ffmc_h1_real").to_numpy().astype(np.float32),
                "dmc": p_nwp.get_column("dmc_h1_real").to_numpy().astype(np.float32),
                "dc": p_nwp.get_column("dc_h1_real").to_numpy().astype(np.float32),
            }
            w_h1_real_dict["vpd_noon_kpa"] = np.array(
                [
                    compute_vpd(float(t), float(rh))
                    for t, rh in zip(
                        w_h1_real_dict["temp_noon_c"], w_h1_real_dict["rh_noon_pct"], strict=True
                    )
                ],
                dtype=np.float32,
            )
            feat_h1_real = build_inference_feature_matrix(
                static_df=p_nwp.select(static_df.columns),
                prior_series=prior_arr,
                weather_dict_by_cell=w_h1_real_dict,
            )
            preds_h1_real = pipeline.predict(feat_h1_real)["raw_score"]
            base_scores_h1_real = w_h1_real_dict["fwi"] * prior_arr

        # Colección de series para bootstrap conjunto
        series_dict: dict[str, np.ndarray] = {
            "model_era5": preds_era5,
            "model_h0": preds_h0,
            "model_h1_pers": preds_h1,
            "base_era5": base_scores_era5,
            "base_h0": base_scores_h0,
            "base_h1_pers": base_scores_h1,
        }
        if preds_h1_real is not None and base_scores_h1_real is not None:
            series_dict["model_h1_real"] = preds_h1_real
            series_dict["base_h1_real"] = base_scores_h1_real

        # Comparaciones pareadas requeridas
        comparisons: list[tuple[str, str, str, str]] = [
            # Modelo vs ERA5 (Ambas métricas para evaluar caída y significancia)
            ("model_h0_vs_era5", "model_era5", "model_h0", "both"),
            ("model_h1_pers_vs_era5", "model_era5", "model_h1_pers", "both"),
            # Baseline vs ERA5 (Rec@5% pareado)
            ("base_h0_vs_era5", "base_era5", "base_h0", "rec5_only"),
            ("base_h1_pers_vs_era5", "base_era5", "base_h1_pers", "rec5_only"),
            # Modelo vs Baseline con NWP (Criterio K5.a: Rec@5% pareado)
            ("model_vs_base_h0", "base_h0", "model_h0", "rec5_only"),
            ("model_vs_base_h1_pers", "base_h1_pers", "model_h1_pers", "rec5_only"),
        ]

        if "model_h1_real" in series_dict:
            comparisons.extend(
                [
                    ("model_h1_real_vs_era5", "model_era5", "model_h1_real", "both"),
                    ("base_h1_real_vs_era5", "base_era5", "base_h1_real", "rec5_only"),
                    ("model_vs_base_h1_real", "base_h1_real", "model_h1_real", "rec5_only"),
                ]
            )

        logger.info(
            f"Ejecutando bootstrap pareado conjunto (B={n_boot}) para temporada {season}..."
        )
        boot_res = run_joint_day_block_bootstrap(
            y_true=y_true,
            dates_arr=dates_arr,
            score_series=series_dict,
            paired_comparisons=comparisons,
            n_boot=n_boot,
        )

        # Cachear para Quantile Mapping cruzado
        season_cache[season] = {
            "y_true": y_true,
            "dates_arr": dates_arr,
            "model_era5": preds_era5,
            "model_h0": preds_h0,
            "model_h1_pers": preds_h1,
            "model_h1_real": preds_h1_real,
            "base_era5": base_scores_era5,
            "base_h0": base_scores_h0,
            "base_h1_pers": base_scores_h1,
            "base_h1_real": base_scores_h1_real,
            "m_thresholds": (m_t80, m_t95, m_t99),
            "b_thresholds": (b_t80, b_t95, b_t99),
        }

        # Estructurar resultado de la temporada
        season_eval: dict[str, Any] = {
            "season": season,
            "n_pos": int(np.sum(y_true)),
            "n_cell_days": len(y_true),
            "baseline_fwi_x_prior": {
                "era5_land": {
                    "pr_auc": boot_res["base_h0_vs_era5"]["pr_auc_ref"],
                    "rec5_pct": boot_res["base_h0_vs_era5"]["rec5_ref_pct"],
                    "tier_distribution_pct": get_tier_distribution(
                        base_scores_era5, b_t80, b_t95, b_t99
                    ),
                },
                "nwp_h0_source_skew": {
                    "description": "desfase de fuente y resolución (GFS horizonte ~0 contra ERA5-Land)",
                    "pr_auc": boot_res["base_h0_vs_era5"]["pr_auc_test"],
                    "delta_pr_auc": boot_res["base_h0_vs_era5"]["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_res["base_h0_vs_era5"]["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_res["base_h0_vs_era5"]["sig_pr_auc"],
                    "rec5_pct": boot_res["base_h0_vs_era5"]["rec5_test_pct"],
                    "delta_rec5_pp": boot_res["base_h0_vs_era5"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["base_h0_vs_era5"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["base_h0_vs_era5"]["sig_rec5"],
                    "tier_distribution_pct": get_tier_distribution(
                        base_scores_h0, b_t80, b_t95, b_t99
                    ),
                },
                "nwp_h1_persistence_24h": {
                    "description": "persistencia 24 h (cota inferior)",
                    "pr_auc": boot_res["base_h1_pers_vs_era5"]["pr_auc_test"],
                    "delta_pr_auc": boot_res["base_h1_pers_vs_era5"]["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_res["base_h1_pers_vs_era5"]["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_res["base_h1_pers_vs_era5"]["sig_pr_auc"],
                    "rec5_pct": boot_res["base_h1_pers_vs_era5"]["rec5_test_pct"],
                    "delta_rec5_pp": boot_res["base_h1_pers_vs_era5"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["base_h1_pers_vs_era5"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["base_h1_pers_vs_era5"]["sig_rec5"],
                    "tier_distribution_pct": get_tier_distribution(
                        base_scores_h1, b_t80, b_t95, b_t99
                    ),
                },
            },
            "model_xgb_pruned_25": {
                "era5_land": {
                    "pr_auc": boot_res["model_h0_vs_era5"]["pr_auc_ref"],
                    "rec5_pct": boot_res["model_h0_vs_era5"]["rec5_ref_pct"],
                    "tier_distribution_pct": get_tier_distribution(preds_era5, m_t80, m_t95, m_t99),
                },
                "nwp_h0_source_skew": {
                    "description": "desfase de fuente y resolución (GFS horizonte ~0 contra ERA5-Land)",
                    "pr_auc": boot_res["model_h0_vs_era5"]["pr_auc_test"],
                    "delta_pr_auc": boot_res["model_h0_vs_era5"]["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_res["model_h0_vs_era5"]["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_res["model_h0_vs_era5"]["sig_pr_auc"],
                    "rec5_pct": boot_res["model_h0_vs_era5"]["rec5_test_pct"],
                    "delta_rec5_pp": boot_res["model_h0_vs_era5"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["model_h0_vs_era5"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["model_h0_vs_era5"]["sig_rec5"],
                    "tier_distribution_pct": get_tier_distribution(preds_h0, m_t80, m_t95, m_t99),
                },
                "nwp_h1_persistence_24h": {
                    "description": "persistencia 24 h (cota inferior)",
                    "pr_auc": boot_res["model_h1_pers_vs_era5"]["pr_auc_test"],
                    "delta_pr_auc": boot_res["model_h1_pers_vs_era5"]["delta_pr_auc"],
                    "ci_delta_pr_auc": boot_res["model_h1_pers_vs_era5"]["ci_delta_pr_auc"],
                    "sig_pr_auc": boot_res["model_h1_pers_vs_era5"]["sig_pr_auc"],
                    "rec5_pct": boot_res["model_h1_pers_vs_era5"]["rec5_test_pct"],
                    "delta_rec5_pp": boot_res["model_h1_pers_vs_era5"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["model_h1_pers_vs_era5"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["model_h1_pers_vs_era5"]["sig_rec5"],
                    "tier_distribution_pct": get_tier_distribution(preds_h1, m_t80, m_t95, m_t99),
                },
            },
            "model_vs_baseline_nwp": {
                "nwp_h0": {
                    "description": "Modelo XGBoost vs Baseline FWI x Prior sobre NWP H0",
                    "delta_rec5_pp": boot_res["model_vs_base_h0"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["model_vs_base_h0"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["model_vs_base_h0"]["sig_rec5"],
                },
                "nwp_h1_persistence_24h": {
                    "description": "Modelo XGBoost vs Baseline FWI x Prior sobre Persistencia 24h",
                    "delta_rec5_pp": boot_res["model_vs_base_h1_pers"]["delta_rec5_pp"],
                    "ci_delta_rec5_pp": boot_res["model_vs_base_h1_pers"]["ci_delta_rec5_pp"],
                    "sig_rec5": boot_res["model_vs_base_h1_pers"]["sig_rec5"],
                },
            },
        }

        if preds_h1_real is not None:
            season_eval["baseline_fwi_x_prior"]["nwp_h1_real_forecast_24h"] = {
                "description": "pronóstico numérico a 24 h (GFS *_previous_day1)",
                "pr_auc": boot_res["base_h1_real_vs_era5"]["pr_auc_test"],
                "delta_pr_auc": boot_res["base_h1_real_vs_era5"]["delta_pr_auc"],
                "ci_delta_pr_auc": boot_res["base_h1_real_vs_era5"]["ci_delta_pr_auc"],
                "sig_pr_auc": boot_res["base_h1_real_vs_era5"]["sig_pr_auc"],
                "rec5_pct": boot_res["base_h1_real_vs_era5"]["rec5_test_pct"],
                "delta_rec5_pp": boot_res["base_h1_real_vs_era5"]["delta_rec5_pp"],
                "ci_delta_rec5_pp": boot_res["base_h1_real_vs_era5"]["ci_delta_rec5_pp"],
                "sig_rec5": boot_res["base_h1_real_vs_era5"]["sig_rec5"],
                "tier_distribution_pct": get_tier_distribution(
                    base_scores_h1_real, b_t80, b_t95, b_t99
                ),
            }
            season_eval["model_xgb_pruned_25"]["nwp_h1_real_forecast_24h"] = {
                "description": "pronóstico numérico a 24 h (GFS *_previous_day1)",
                "pr_auc": boot_res["model_h1_real_vs_era5"]["pr_auc_test"],
                "delta_pr_auc": boot_res["model_h1_real_vs_era5"]["delta_pr_auc"],
                "ci_delta_pr_auc": boot_res["model_h1_real_vs_era5"]["ci_delta_pr_auc"],
                "sig_pr_auc": boot_res["model_h1_real_vs_era5"]["sig_pr_auc"],
                "rec5_pct": boot_res["model_h1_real_vs_era5"]["rec5_test_pct"],
                "delta_rec5_pp": boot_res["model_h1_real_vs_era5"]["delta_rec5_pp"],
                "ci_delta_rec5_pp": boot_res["model_h1_real_vs_era5"]["ci_delta_rec5_pp"],
                "sig_rec5": boot_res["model_h1_real_vs_era5"]["sig_rec5"],
                "tier_distribution_pct": get_tier_distribution(preds_h1_real, m_t80, m_t95, m_t99),
            }
            season_eval["model_vs_baseline_nwp"]["nwp_h1_real_forecast_24h"] = {
                "description": "Modelo XGBoost vs Baseline FWI x Prior sobre NWP H1 Real",
                "delta_rec5_pp": boot_res["model_vs_base_h1_real"]["delta_rec5_pp"],
                "ci_delta_rec5_pp": boot_res["model_vs_base_h1_real"]["ci_delta_rec5_pp"],
                "sig_rec5": boot_res["model_vs_base_h1_real"]["sig_rec5"],
            }

        eval_results.append(season_eval)

    # =========================================================================
    # REGLA K4: Mapeo de Cuantiles Monótono Cruzado Fuera de Muestra
    # =========================================================================
    logger.info("=== Ejecutando Mapeo de Cuantiles Monótono Cruzado (K4) ===")
    qm_results: list[dict[str, Any]] = []

    # Configuración de direcciones
    directions = [
        ("2023_24", "2024_25"),  # Entrena en 23_24, prueba en 24_25
        ("2024_25", "2023_24"),  # Entrena en 24_25, prueba en 23_24
    ]

    for tr_season, te_season in directions:
        tr_data = season_cache[tr_season]
        te_data = season_cache[te_season]
        y_te = te_data["y_true"]
        m_t80, m_t95, m_t99 = te_data["m_thresholds"]
        b_t80, b_t95, b_t99 = te_data["b_thresholds"]

        # Horizontes a evaluar
        horizons_to_test = [
            ("nwp_h0", "model_h0", "model_era5", "base_h0", "base_era5"),
            ("nwp_h1_pers", "model_h1_pers", "model_era5", "base_h1_pers", "base_era5"),
        ]

        if te_data["model_h1_real"] is not None:
            # En 2024_25, H1 real usa el mapeador de GFS entrenado en 2023_24
            horizons_to_test.append(
                ("nwp_h1_real", "model_h1_real", "model_era5", "base_h1_real", "base_era5")
            )

        for h_tag, m_nwp_key, m_era_key, b_nwp_key, b_era_key in horizons_to_test:
            # Si entrenamos en 2024_25 y probamos en 2023_24 para H1 real:
            if h_tag == "nwp_h1_real" and tr_season == "2024_25":
                qm_results.append(
                    {
                        "direction": f"{tr_season} -> {te_season}",
                        "horizon": h_tag,
                        "status": "NO EJECUTADO",
                        "reason": "Temporada 2023_24 carece de pronóstico 24h completo previo a 2024-01-19",
                    }
                )
                continue

            # 1. Modelo XGBoost
            m_s_tr_nwp = tr_data["model_h0"] if h_tag == "nwp_h1_real" else tr_data[m_nwp_key]
            m_s_tr_era = tr_data[m_era_key]
            m_s_te_nwp = te_data[m_nwp_key]

            mapper_m = MonotonicQuantileMapper().fit(m_s_tr_nwp, m_s_tr_era)
            m_s_te_mapped = mapper_m.transform(m_s_te_nwp)

            m_before = compute_metrics(y_te, m_s_te_nwp)
            m_after = compute_metrics(y_te, m_s_te_mapped)
            m_dist_before = get_tier_distribution(m_s_te_nwp, m_t80, m_t95, m_t99)
            m_dist_after = get_tier_distribution(m_s_te_mapped, m_t80, m_t95, m_t99)

            # 2. Baseline FWI x Prior
            b_s_tr_nwp = tr_data["base_h0"] if h_tag == "nwp_h1_real" else tr_data[b_nwp_key]
            b_s_tr_era = tr_data[b_era_key]
            b_s_te_nwp = te_data[b_nwp_key]

            mapper_b = MonotonicQuantileMapper().fit(b_s_tr_nwp, b_s_tr_era)
            b_s_te_mapped = mapper_b.transform(b_s_te_nwp)

            b_before = compute_metrics(y_te, b_s_te_nwp)
            b_after = compute_metrics(y_te, b_s_te_mapped)
            b_dist_before = get_tier_distribution(b_s_te_nwp, b_t80, b_t95, b_t99)
            b_dist_after = get_tier_distribution(b_s_te_mapped, b_t80, b_t95, b_t99)

            # Desvíos relativos respecto al reparto canónico ERA5 (80, 15, 4, 1)
            def compute_rel_shifts(d: dict[str, float]) -> dict[str, float]:
                return {
                    "bajo": (d["bajo"] - 80.0) / 80.0 * 100.0,
                    "moderado": (d["moderado"] - 15.0) / 15.0 * 100.0,
                    "alto": (d["alto"] - 4.0) / 4.0 * 100.0,
                    "extremo": (d["extremo"] - 1.0) / 1.0 * 100.0,
                }

            qm_results.append(
                {
                    "direction": f"{tr_season} -> {te_season}",
                    "train_season": tr_season,
                    "test_season": te_season,
                    "horizon": h_tag,
                    "model_xgb_pruned_25": {
                        "pr_auc_before": m_before["pr_auc"],
                        "pr_auc_after": m_after["pr_auc"],
                        "rec5_pct_before": m_before["rec5_pct"],
                        "rec5_pct_after": m_after["rec5_pct"],
                        "tier_distribution_before": m_dist_before,
                        "tier_distribution_after": m_dist_after,
                        "relative_shifts_before_pct": compute_rel_shifts(m_dist_before),
                        "relative_shifts_after_pct": compute_rel_shifts(m_dist_after),
                        "within_30pct_before": all(
                            abs(v) <= 30.0 for v in compute_rel_shifts(m_dist_before).values()
                        ),
                        "within_30pct_after": all(
                            abs(v) <= 30.0 for v in compute_rel_shifts(m_dist_after).values()
                        ),
                    },
                    "baseline_fwi_x_prior": {
                        "pr_auc_before": b_before["pr_auc"],
                        "pr_auc_after": b_after["pr_auc"],
                        "rec5_pct_before": b_before["rec5_pct"],
                        "rec5_pct_after": b_after["rec5_pct"],
                        "tier_distribution_before": b_dist_before,
                        "tier_distribution_after": b_dist_after,
                        "relative_shifts_before_pct": compute_rel_shifts(b_dist_before),
                        "relative_shifts_after_pct": compute_rel_shifts(b_dist_after),
                        "within_30pct_before": all(
                            abs(v) <= 30.0 for v in compute_rel_shifts(b_dist_before).values()
                        ),
                        "within_30pct_after": all(
                            abs(v) <= 30.0 for v in compute_rel_shifts(b_dist_after).values()
                        ),
                    },
                }
            )

    # =========================================================================
    # REGLA K5: Evaluación de Criterios Oficiales de Aceptación
    # =========================================================================
    # (a) Modelo supera a baseline en Rec@5% con IC pareado excluyendo 0
    # (b) Reparto de clases dentro de +-30% relativo respecto a ERA5 en los mismos días
    # (c) Caída de PR-AUC con IC al 95%
    criteria_summary: dict[str, Any] = {}

    for ev in eval_results:
        s = ev["season"]
        # H0
        m_vs_b_h0 = ev["model_vs_baseline_nwp"]["nwp_h0"]
        crit_a_h0 = bool(m_vs_b_h0["ci_delta_rec5_pp"][0] > 0.0)

        # Buscar QM de esta temporada
        qm_h0 = next(
            (q for q in qm_results if q.get("test_season") == s and q.get("horizon") == "nwp_h0"),
            None,
        )
        crit_b_h0_before = qm_h0["model_xgb_pruned_25"]["within_30pct_before"] if qm_h0 else False
        crit_b_h0_after = qm_h0["model_xgb_pruned_25"]["within_30pct_after"] if qm_h0 else False

        m_drop_h0 = ev["model_xgb_pruned_25"]["nwp_h0_source_skew"]
        crit_c_h0_ci = m_drop_h0["ci_delta_pr_auc"]

        criteria_summary[f"{s}_h0"] = {
            "criterion_a_model_beats_baseline_rec5": {
                "delta_rec5_pp": m_vs_b_h0["delta_rec5_pp"],
                "ci_delta_rec5_pp": m_vs_b_h0["ci_delta_rec5_pp"],
                "verdict": "CUMPLE" if crit_a_h0 else "NO CUMPLE",
            },
            "criterion_b_class_distribution_30pct": {
                "before_qm": "CUMPLE" if crit_b_h0_before else "NO CUMPLE",
                "after_qm": "CUMPLE" if crit_b_h0_after else "NO CUMPLE",
            },
            "criterion_c_pr_auc_drop_95ci": {
                "delta_pr_auc": m_drop_h0["delta_pr_auc"],
                "ci_delta_pr_auc": crit_c_h0_ci,
                "significant_drop": m_drop_h0["sig_pr_auc"],
                "verdict": "CUMPLE"
                if not m_drop_h0["sig_pr_auc"] or crit_c_h0_ci[0] > -0.0005
                else "NO CUMPLE",
            },
        }

        if "nwp_h1_real_forecast_24h" in ev["model_vs_baseline_nwp"]:
            m_vs_b_real = ev["model_vs_baseline_nwp"]["nwp_h1_real_forecast_24h"]
            crit_a_real = bool(m_vs_b_real["ci_delta_rec5_pp"][0] > 0.0)

            qm_real = next(
                (
                    q
                    for q in qm_results
                    if q.get("test_season") == s and q.get("horizon") == "nwp_h1_real"
                ),
                None,
            )
            crit_b_real_before = (
                qm_real["model_xgb_pruned_25"]["within_30pct_before"] if qm_real else False
            )
            crit_b_real_after = (
                qm_real["model_xgb_pruned_25"]["within_30pct_after"] if qm_real else False
            )

            m_drop_real = ev["model_xgb_pruned_25"]["nwp_h1_real_forecast_24h"]
            crit_c_real_ci = m_drop_real["ci_delta_pr_auc"]

            criteria_summary[f"{s}_h1_real"] = {
                "criterion_a_model_beats_baseline_rec5": {
                    "delta_rec5_pp": m_vs_b_real["delta_rec5_pp"],
                    "ci_delta_rec5_pp": m_vs_b_real["ci_delta_rec5_pp"],
                    "verdict": "CUMPLE" if crit_a_real else "NO CUMPLE",
                },
                "criterion_b_class_distribution_30pct": {
                    "before_qm": "CUMPLE" if crit_b_real_before else "NO CUMPLE",
                    "after_qm": "CUMPLE" if crit_b_real_after else "NO CUMPLE",
                },
                "criterion_c_pr_auc_drop_95ci": {
                    "delta_pr_auc": m_drop_real["delta_pr_auc"],
                    "ci_delta_pr_auc": crit_c_real_ci,
                    "significant_drop": m_drop_real["sig_pr_auc"],
                    "verdict": "CUMPLE"
                    if not m_drop_real["sig_pr_auc"] or crit_c_real_ci[0] > -0.0005
                    else "NO CUMPLE",
                },
            }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "j2_nwp_skew_study.json"

    final_payload: dict[str, Any] = {
        "study": "J2 NWP Operational Latency and Forecast Skew Study",
        "executed_at": datetime.now(UTC).isoformat(),
        "model_version": "chilefuego_xgb_forward_v1 (Forward Walk 2017-2022)",
        "evaluations": eval_results,
        "quantile_mapping_cross_validation": qm_results,
        "criteria_evaluation": criteria_summary,
        "limitations_k6": {
            "seasons_evaluated": 2,
            "positive_cases": {"2023_24": 315, "2024_25": 411},
            "temporal_coverage": "Nov-Abr (ventana oficial activa de incendios, excluye mayo a octubre)",
            "nwp_source": "Open-Meteo GFS seamless archive",
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    logger.info(
        f"Estudio J2 completado con éxito en {round(time.time() - t0, 2)}s. Resultados persistidos en {out_json}"
    )
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutar estudio J2 / K1-K6 de emulación NWP y desfase operativo con Modelo B."
    )
    parser.add_argument(
        "--nwp-dir",
        type=str,
        default="data/raw/nwp_archive",
        help="Directorio con archivos Parquet y JSON de NWP.",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Directorio para resultados JSON."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/chilefuego_xgb_forward_v1.json",
        help="Ruta al modelo entrenado (default: Modelo B).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=250,
        help="Número de réplicas bootstrap por bloque de días (default: 250).",
    )
    args = parser.parse_args()

    _ = run_skew_study(
        nwp_dir=Path(args.nwp_dir),
        results_dir=Path(args.results_dir),
        model_path=Path(args.model_path),
        n_boot=args.n_boot,
    )
    print(f"Estudio completado. Payload JSON guardado en {args.results_dir}/j2_nwp_skew_study.json")


if __name__ == "__main__":
    main()
