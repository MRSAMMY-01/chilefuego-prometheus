"""Estudio de Desfase de Fuentes y Pronósticos Archivados Open-Meteo (H4).

PORQUÉ (Regla 5):
- Evalúa el impacto de reemplazar el reanálisis ERA5-Land por pronósticos numéricos del tiempo (NWP)
  a horizonte 0 (día D) y horizonte 1 (día D+1) sobre las temporadas fechadas 2023_24 y 2024_25.
- Descarga y almacena de forma reanudable y por lotes los pronósticos archivados de Open-Meteo
  (Previous Model Runs API) para los 399 nodos de Ñuble, respetando estrictamente los límites de la API.
- Ejecuta el avance diario del FWI utilizando el estado del día D-1 del reanálisis ERA5-Land
  (arranque en caliente diario sin acumulación de error entre temporadas).
- Compara PR-AUC, Rec@5% y distribución de clases de riesgo relativo contra ERA5-Land de referencia.

NOTA CRÍTICA: Este script está preparado para ejecución manual y controlada desde terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
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
logger = logging.getLogger("h4_openmeteo")

OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SEASONS_H4 = ["2023_24", "2024_25"]
DATE_START = date(2023, 7, 1)
DATE_END = date(2025, 6, 30)


def load_nuble_nodes(
    weather_path: Path,
) -> tuple[list[str], np.ndarray, dict[str, tuple[float, float]]]:
    """Carga los identificadores y coordenadas (lat, lon) de los 399 nodos de ERA5-Land en Ñuble."""
    weather_df = pl.read_parquet(weather_path)
    nodes = sorted(weather_df.select(["node_id"]).unique().to_series().to_list())
    coords = []
    coord_dict = {}
    for n in nodes:
        lat_str, lon_str = n.split("_")
        lat, lon = float(lat_str), float(lon_str)
        coords.append([lon, lat])
        coord_dict[n] = (lat, lon)
    return nodes, np.array(coords), coord_dict


def download_openmeteo_batch(
    lats: list[float],
    lons: list[float],
    start_date: str,
    end_date: str,
    max_retries: int = 10,
) -> list[dict[str, Any]] | None:
    """Descarga un lote de coordenadas desde Open-Meteo Previous Model Runs API con reintentos exponenciales."""
    lat_param = ",".join(f"{lat:.4f}" for lat in lats)
    lon_param = ",".join(f"{lon:.4f}" for lon in lons)
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "precipitation",
        "temperature_2m_previous_day1",
        "relative_humidity_2m_previous_day1",
        "wind_speed_10m_previous_day1",
        "precipitation_previous_day1",
    ]
    url = (
        f"{OPEN_METEO_PREVIOUS_RUNS_URL}?"
        f"latitude={lat_param}&longitude={lon_param}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(hourly_vars)}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-Study/1.0"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
                return [data]
        except urllib.error.HTTPError as http_err:
            if http_err.code == 429:
                wait_s = 30 * (2**attempt)
                logger.warning(
                    f"Rate limit 429 detectado. Pausando {wait_s}s antes de reintentar (intento {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_s)
            else:
                logger.error(f"Error HTTP {http_err.code} en descarga: {http_err.reason}")
                if attempt == max_retries - 1:
                    logger.error(
                        f"Agotados los {max_retries} intentos tras error HTTP {http_err.code}. Omitiendo lote."
                    )
                    return None
                time.sleep(10)
        except Exception as exc:
            wait_s = min(300, 15 * (2**attempt))
            logger.warning(
                f"Excepción en descarga ({exc}). Pausando {wait_s}s (intento {attempt + 1}/{max_retries})..."
            )
            time.sleep(wait_s)

    logger.error(
        f"Fallo al descargar lote de {len(lats)} coordenadas tras {max_retries} intentos. Retornando None."
    )
    return None


def fetch_all_nodes_forecasts(
    nodes: list[str],
    coord_dict: dict[str, tuple[float, float]],
    raw_dir: Path,
    batch_size: int = 15,
    sleep_between_batches: float = 2.0,
) -> Path:
    """Descarga de forma reanudable y por lotes los pronósticos archivados para los 399 nodos."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_combined_path = raw_dir / "open_meteo_forecast_nodes_2023_2025.parquet"
    if out_combined_path.exists():
        logger.info(f"Archivo consolidado ya existe en {out_combined_path}. Omitiendo descarga.")
        return out_combined_path

    n_nodes = len(nodes)
    n_batches = (n_nodes + batch_size - 1) // batch_size
    logger.info(
        f"Iniciando descarga de {n_nodes} nodos en {n_batches} lotes (tamaño lote = {batch_size})..."
    )

    batch_files: list[Path] = []
    for b_idx in range(n_batches):
        b_file = raw_dir / f"batch_{b_idx:03d}.json"
        batch_files.append(b_file)
        if b_file.exists():
            logger.info(f"Lote {b_idx + 1}/{n_batches} ya descargado ({b_file.name}). Saltando.")
            continue

        start_i = b_idx * batch_size
        end_i = min(start_i + batch_size, n_nodes)
        batch_nodes = nodes[start_i:end_i]
        b_lats = [coord_dict[n][0] for n in batch_nodes]
        b_lons = [coord_dict[n][1] for n in batch_nodes]

        logger.info(
            f"Descargando lote {b_idx + 1}/{n_batches} ({len(batch_nodes)} nodos: {batch_nodes[0]} .. {batch_nodes[-1]})..."
        )
        batch_data = download_openmeteo_batch(
            lats=b_lats,
            lons=b_lons,
            start_date=DATE_START.isoformat(),
            end_date=DATE_END.isoformat(),
        )

        if batch_data is not None:
            with open(b_file, "w", encoding="utf-8") as f:
                json.dump({"nodes": batch_nodes, "data": batch_data}, f)
        else:
            logger.error(
                f"Lote {b_idx + 1}/{n_batches} no se pudo descargar tras {10} reintentos. Omitiendo archivo temporal."
            )

        time.sleep(sleep_between_batches)

    # Consolidar lotes en un único DataFrame estructurado
    logger.info("Consolidando todos los lotes descargados...")
    extracted_records: list[dict[str, Any]] = []

    for b_file in batch_files:
        if not b_file.exists():
            logger.warning(f"Lote faltante {b_file.name} omitido en consolidación.")
            continue
        with open(b_file, encoding="utf-8") as f:
            content = json.load(f)
        b_nodes = content["nodes"]
        b_data_list = content.get("data")
        if not b_data_list:
            continue

        for nid, node_data in zip(b_nodes, b_data_list, strict=True):
            hourly = node_data["hourly"]
            times = hourly["time"]
            t_h0 = hourly["temperature_2m"]
            rh_h0 = hourly["relative_humidity_2m"]
            w_h0 = hourly["wind_speed_10m"]
            p_h0 = hourly["precipitation"]

            t_h1 = hourly.get("temperature_2m_previous_day1", t_h0)
            rh_h1 = hourly.get("relative_humidity_2m_previous_day1", rh_h0)
            w_h1 = hourly.get("wind_speed_10m_previous_day1", w_h0)
            p_h1 = hourly.get("precipitation_previous_day1", p_h0)

            # Mapear índice temporal de las 16:00 UTC para cada fecha
            time_idx_map = {t_str: idx for idx, t_str in enumerate(times)}

            curr_d = DATE_START
            while curr_d <= DATE_END:
                t16_str = f"{curr_d.isoformat()}T16:00"
                if t16_str in time_idx_map:
                    idx16 = time_idx_map[t16_str]
                    # Lluvia 24 h previa al mediodía (ventana de 24 horas previas)
                    start_rain_idx = max(0, idx16 - 23)
                    rain24_h0 = float(sum(p_h0[start_rain_idx : idx16 + 1]))
                    rain24_h1 = float(sum(p_h1[start_rain_idx : idx16 + 1]))

                    extracted_records.append(
                        {
                            "node_id": nid,
                            "date": curr_d,
                            "temp_noon_c_h0": float(t_h0[idx16]),
                            "rh_noon_pct_h0": float(rh_h0[idx16]),
                            "wind_noon_kmh_h0": float(w_h0[idx16]),
                            "rain_24h_noon_mm_h0": rain24_h0,
                            "temp_noon_c_h1": float(t_h1[idx16]),
                            "rh_noon_pct_h1": float(rh_h1[idx16]),
                            "wind_noon_kmh_h1": float(w_h1[idx16]),
                            "rain_24h_noon_mm_h1": rain24_h1,
                        }
                    )
                curr_d += timedelta(days=1)

    df_consolidated = pl.DataFrame(extracted_records)
    df_consolidated.write_parquet(out_combined_path)
    logger.info(
        f"Pronósticos consolidados guardados en {out_combined_path} ({df_consolidated.height} registros)."
    )
    return out_combined_path


def compute_fwi_warm_start(
    forecast_df: pl.DataFrame,
    fwi_reanalysis_path: Path,
    horizon_tag: str,  # 'h0' o 'h1'
) -> pl.DataFrame:
    """Calcula el FWI para cada día D usando las condiciones iniciales de D-1 del reanálisis ERA5-Land."""
    logger.info(
        f"Calculando FWI con arranque en caliente de reanálisis para horizonte {horizon_tag.upper()}..."
    )
    reanalysis_df = pl.read_parquet(fwi_reanalysis_path)

    # Filtrar solo nodos y fechas requeridas
    dates_unique = sorted(forecast_df.get_column("date").unique().to_list())

    # Mapa de estado previo (D-1) del reanálisis
    reanalysis_map = {}
    for r in reanalysis_df.filter(pl.col("date") >= DATE_START - timedelta(days=1)).iter_rows(
        named=True
    ):
        reanalysis_map[(r["node_id"], r["date"])] = (
            float(r["ffmc"]),
            float(r["dmc"]),
            float(r["dc"]),
        )

    fwi_rows = []
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
            # Warm start desde ERA5-Land D-1
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
                    f"fwi_{horizon_tag}": float(fwi_res["fwi"][0]),
                    f"isi_{horizon_tag}": float(fwi_res["isi"][0]),
                    f"bui_{horizon_tag}": float(fwi_res["bui"][0]),
                    f"ffmc_{horizon_tag}": float(fwi_res["ffmc"][0]),
                    f"dmc_{horizon_tag}": float(fwi_res["dmc"][0]),
                    f"dc_{horizon_tag}": float(fwi_res["dc"][0]),
                }
            )

    return pl.DataFrame(fwi_rows)


def run_h4_evaluation(
    batch_size: int = 15,
    sleep_between_batches: float = 2.0,
) -> dict[str, Any]:
    """Ejecuta el protocolo de comparación rigurosa entre ERA5-Land y Open-Meteo H0/H1."""
    t0 = time.time()
    cfg = load_config()

    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    fwi_reanalysis_path = Path("data/processed/era5_land_daily_fwi_nodes.parquet")
    raw_dir = Path("data/raw/open_meteo")
    results_dir = Path("results")

    # 1. Obtener y verificar pronósticos archivados
    nodes, node_coords, coord_dict = load_nuble_nodes(weather_path)
    forecast_parquet = fetch_all_nodes_forecasts(
        nodes,
        coord_dict,
        raw_dir,
        batch_size=batch_size,
        sleep_between_batches=sleep_between_batches,
    )
    forecast_df = pl.read_parquet(forecast_parquet)

    # 2. Calcular FWI para H0 y H1 con arranque en caliente
    fwi_h0_df = compute_fwi_warm_start(forecast_df, fwi_reanalysis_path, horizon_tag="h0")
    fwi_h1_df = compute_fwi_warm_start(forecast_df, fwi_reanalysis_path, horizon_tag="h1")

    merged_weather = forecast_df.join(fwi_h0_df, on=["node_id", "date"]).join(
        fwi_h1_df, on=["node_id", "date"]
    )

    # 3. Preparar infraestructura espacial
    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    clon, clat = trans.transform(cx, cy)
    tree = cKDTree(node_coords)
    _, cell_node_idxs = tree.query(np.column_stack([clon, clat]), k=1)
    cell_node_df = pl.DataFrame(
        {"cell_id": cell_ids, "node_id": [nodes[idx] for idx in cell_node_idxs]}
    )

    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    sp_df, _ = build_phase2a_daily_panel(cfg)

    pipeline = InferencePipeline()

    # Contenedores de resultados
    eval_results: list[dict[str, Any]] = []

    for season in SEASONS_H4:
        logger.info(f"Evaluando temporada {season}...")
        season_panel = materialize_season_cell_day_panel(
            grid_df, sp_df, season, include_excluded=False
        )
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
        n_pos = int(np.sum(y_true))
        n_total = len(y_true)

        feat_era5 = p_era5.select(PRUNED_25_FEATURES)
        preds_era5 = pipeline.predict(feat_era5)
        pr_era5 = float(average_precision_score(y_true, preds_era5["raw_score"]))

        k5 = max(1, int(round(0.05 * n_total)))
        t5_era5 = np.partition(preds_era5["raw_score"], -k5)[-k5]
        rec5_era5 = float(np.sum((y_true == 1) & (preds_era5["raw_score"] >= t5_era5))) / float(
            n_pos
        )

        # 2. Open-Meteo H0 (Nowcast / Horizonte 0 días)
        p_om = season_panel.join(merged_weather, on=["node_id", "date"], how="left")

        # Ensamblar covariables H0
        weather_h0_dict = {
            "temp_noon_c": p_om.get_column("temp_noon_c_h0").to_numpy().astype(np.float32),
            "rh_noon_pct": p_om.get_column("rh_noon_pct_h0").to_numpy().astype(np.float32),
            "wind_noon_kmh": p_om.get_column("wind_noon_kmh_h0").to_numpy().astype(np.float32),
            "rain_24h_noon_mm": p_om.get_column("rain_24h_noon_mm_h0")
            .to_numpy()
            .astype(np.float32),
            "fwi": p_om.get_column("fwi_h0").to_numpy().astype(np.float32),
            "isi": p_om.get_column("isi_h0").to_numpy().astype(np.float32),
            "bui": p_om.get_column("bui_h0").to_numpy().astype(np.float32),
            "ffmc": p_om.get_column("ffmc_h0").to_numpy().astype(np.float32),
            "dmc": p_om.get_column("dmc_h0").to_numpy().astype(np.float32),
            "dc": p_om.get_column("dc_h0").to_numpy().astype(np.float32),
        }
        weather_h0_dict["vpd_noon_kpa"] = np.array(
            [
                compute_vpd(float(t), float(rh))
                for t, rh in zip(
                    weather_h0_dict["temp_noon_c"], weather_h0_dict["rh_noon_pct"], strict=True
                )
            ],
            dtype=np.float32,
        )
        feat_h0 = build_inference_feature_matrix(
            static_df=p_om.select(static_df.columns),
            prior_series=p_om.get_column("prior_density").to_numpy().astype(np.float32),
            weather_dict_by_cell=weather_h0_dict,
        )
        preds_h0 = pipeline.predict(feat_h0)
        pr_h0 = float(average_precision_score(y_true, preds_h0["raw_score"]))
        t5_h0 = np.partition(preds_h0["raw_score"], -k5)[-k5]
        rec5_h0 = float(np.sum((y_true == 1) & (preds_h0["raw_score"] >= t5_h0))) / float(n_pos)

        # 3. Open-Meteo H1 (Horizonte 1 día / D+1)
        weather_h1_dict = {
            "temp_noon_c": p_om.get_column("temp_noon_c_h1").to_numpy().astype(np.float32),
            "rh_noon_pct": p_om.get_column("rh_noon_pct_h1").to_numpy().astype(np.float32),
            "wind_noon_kmh": p_om.get_column("wind_noon_kmh_h1").to_numpy().astype(np.float32),
            "rain_24h_noon_mm": p_om.get_column("rain_24h_noon_mm_h1")
            .to_numpy()
            .astype(np.float32),
            "fwi": p_om.get_column("fwi_h1").to_numpy().astype(np.float32),
            "isi": p_om.get_column("isi_h1").to_numpy().astype(np.float32),
            "bui": p_om.get_column("bui_h1").to_numpy().astype(np.float32),
            "ffmc": p_om.get_column("ffmc_h1").to_numpy().astype(np.float32),
            "dmc": p_om.get_column("dmc_h1").to_numpy().astype(np.float32),
            "dc": p_om.get_column("dc_h1").to_numpy().astype(np.float32),
        }
        weather_h1_dict["vpd_noon_kpa"] = np.array(
            [
                compute_vpd(float(t), float(rh))
                for t, rh in zip(
                    weather_h1_dict["temp_noon_c"], weather_h1_dict["rh_noon_pct"], strict=True
                )
            ],
            dtype=np.float32,
        )
        feat_h1 = build_inference_feature_matrix(
            static_df=p_om.select(static_df.columns),
            prior_series=p_om.get_column("prior_density").to_numpy().astype(np.float32),
            weather_dict_by_cell=weather_h1_dict,
        )
        preds_h1 = pipeline.predict(feat_h1)
        pr_h1 = float(average_precision_score(y_true, preds_h1["raw_score"]))
        t5_h1 = np.partition(preds_h1["raw_score"], -k5)[-k5]
        rec5_h1 = float(np.sum((y_true == 1) & (preds_h1["raw_score"] >= t5_h1))) / float(n_pos)

        def get_tier_dist(tier_arr: np.ndarray) -> dict[str, float]:
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
                "n_pos": n_pos,
                "n_total_cell_days": n_total,
                "era5_land": {
                    "pr_auc": pr_era5,
                    "rec5_pct": rec5_era5 * 100.0,
                    "tier_distribution_pct": get_tier_dist(preds_era5["risk_tier"]),
                },
                "open_meteo_h0": {
                    "pr_auc": pr_h0,
                    "rec5_pct": rec5_h0 * 100.0,
                    "delta_pr_vs_era5": pr_h0 - pr_era5,
                    "delta_rec5_vs_era5": (rec5_h0 - rec5_era5) * 100.0,
                    "tier_distribution_pct": get_tier_dist(preds_h0["risk_tier"]),
                },
                "open_meteo_h1": {
                    "pr_auc": pr_h1,
                    "rec5_pct": rec5_h1 * 100.0,
                    "delta_pr_vs_era5": pr_h1 - pr_era5,
                    "delta_rec5_vs_era5": (rec5_h1 - rec5_era5) * 100.0,
                    "tier_distribution_pct": get_tier_dist(preds_h1["risk_tier"]),
                },
            }
        )

    # Guardar resultados
    out_json = results_dir / "h4_nwp_leadtime_study.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "study": "H4 NWP Source Latency and Forecast Horizon Evaluation",
                "date_executed": datetime.utcnow().isoformat() + "Z",
                "api_endpoint": OPEN_METEO_PREVIOUS_RUNS_URL,
                "terms_of_use": "CC BY 4.0 (Non-commercial) / Commercial requires paid license or AGPLv3 self-hosting",
                "evaluations": eval_results,
                "elapsed_seconds": round(time.time() - t0, 2),
            },
            f,
            indent=2,
        )

    logger.info(f"Estudio H4 completado en {round(time.time() - t0, 2)}s. Resultados en {out_json}")
    return {"results_path": str(out_json), "evaluations": eval_results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ejecutar estudio H4 de desfase de fuentes Open-Meteo."
    )
    parser.add_argument("--batch-size", type=int, default=15, help="Número de nodos por lote API")
    parser.add_argument(
        "--sleep", type=float, default=2.0, help="Segundos de pausa entre llamadas API"
    )
    args = parser.parse_args()

    logger.info("=== INICIANDO PROTOCOLO H4 OPEN-METEO ===")
    res = run_h4_evaluation(
        batch_size=args.batch_size,
        sleep_between_batches=args.sleep,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
