"""Piloto Operacional y Ejecución Diaria de Peligro de Incendios (H5 / N5 / P2 / R1-R6).

PORQUÉ:
- Baseline FWI x Prior establecido oficialmente como el PRODUCTO PRINCIPAL de producción.
- R1: Paridad investigación-producción con soporte de archivo NWP y llamadas en vivo usando el modelo NOAA GFS (gfs_seamless).
- R2: Integración del Mapeo Monótono de Cuantiles NWP -> ERA5 (quantile_mapper_nwp_era5_v1.json) para Baseline y Ensamble.
- R3: Inferencia con el Ensamble de Producción (chilefuego_xgb_ensemble_v2) y clasificación tier_ml bajo umbrales fijos LOSO.
- R6: Separación de arquitectura de datos para Dashboard: geometría estática ('grid_1km.geojson') y datos diarios ('daily_tiers.json').
- Mantiene registro auditable e idempotente en 'data/prospective_log.parquet' y genera 'latest.json'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.config import load_config
from chilefuego.features.fwi import compute_fwi_series_1d
from chilefuego.inference.contract import (
    PRUNED_25_FEATURES,
    InferencePipeline,
    build_inference_feature_matrix,
    compute_vpd,
)
from pyproj import Transformer
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_daily")

# Umbrales fijos P3 para Baseline (FWI x Prior)
BASELINE_FIXED_THRESHOLDS = {
    "p80": 0.00133319,
    "p95": 0.00511459,
    "p99": 0.01174802,
}

# Umbrales fijos LOSO para Ensamble de Producción (R3)
ML_FIXED_THRESHOLDS = {
    "p80": 0.0001247019972652197,
    "p95": 0.0004307880881242454,
    "p99": 0.0012349413009360433,
}


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class MonotonicQuantileMapper:
    """Mapeador empírico monótono de cuantiles (Q-Q mapping) con extrapolación lineal."""

    def __init__(self, q_grid: list[float], u_nwp: list[float], u_era5: list[float]) -> None:
        self.q_grid = np.array(q_grid, dtype=np.float64)
        self.u_nwp = np.array(u_nwp, dtype=np.float64)
        self.u_era5 = np.array(u_era5, dtype=np.float64)

    def transform(self, scores: np.ndarray) -> np.ndarray:
        scores_arr = np.asarray(scores, dtype=np.float64)
        mapped = np.interp(scores_arr, self.u_nwp, self.u_era5)
        min_nwp, max_nwp = self.u_nwp[0], self.u_nwp[-1]
        min_era, max_era = self.u_era5[0], self.u_era5[-1]
        mask_low = scores_arr < min_nwp
        if np.any(mask_low):
            if min_nwp > 1e-12:
                mapped[mask_low] = min_era * (np.maximum(0.0, scores_arr[mask_low]) / min_nwp)
            else:
                mapped[mask_low] = np.maximum(0.0, min_era + (scores_arr[mask_low] - min_nwp))
        mask_high = scores_arr > max_nwp
        if np.any(mask_high):
            denom = self.u_nwp[-1] - self.u_nwp[-2]
            slope = (self.u_era5[-1] - self.u_era5[-2]) / denom if denom > 1e-12 else 1.0
            slope = max(1e-6, slope)
            mapped[mask_high] = max_era + slope * (scores_arr[mask_high] - max_nwp)
        mapped[scores_arr <= 0.0] = 0.0
        return np.maximum(0.0, mapped).astype(np.float32)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MonotonicQuantileMapper:
        return cls(d["q_grid"], d["u_nwp"], d["u_era5"])


def get_latest_season_label(target_date: date) -> str:
    """Determina la temporada fechada (1 de julio al 30 de junio) para el prior espacial."""
    year = target_date.year
    if target_date.month >= 7:
        return f"{year}_{str(year + 1)[-2:]}"
    return f"{year - 1}_{str(year)[-2:]}"


def load_node_coordinates(
    weather_path: Path,
) -> tuple[list[str], np.ndarray, dict[str, tuple[float, float]]]:
    """Obtiene los 399 nodos de Ñuble y sus coordenadas geográficas (lat, lon)."""
    weather_df = pl.read_parquet(weather_path)
    nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    coords = []
    coord_dict = {}
    for n in nodes:
        lat_str, lon_str = n.split("_")
        lat, lon = float(lat_str), float(lon_str)
        coords.append([lon, lat])
        coord_dict[n] = (lat, lon)
    return nodes, np.array(coords), coord_dict


def fetch_weather_forecast(
    target_date: date,
    nodes: list[str],
    coord_dict: dict[str, tuple[float, float]],
    mock_weather_path: Path | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, dict[str, float]], str]:
    """Obtiene el pronóstico para las 16:00 UTC (12:00 CLT). Retorna datos y frescura ('fresh' o 'stale')."""
    if mock_weather_path and mock_weather_path.exists():
        logger.info(f"Cargando pronóstico mock desde {mock_weather_path}")
        with open(mock_weather_path, encoding="utf-8") as f:
            data = json.load(f)
        return data, "mock"

    cache_file = cache_dir / f"weather_cache_{target_date.isoformat()}.json" if cache_dir else None
    if cache_file and cache_file.exists():
        logger.info(f"Usando caché local existente para {target_date}")
        with open(cache_file, encoding="utf-8") as f:
            return json.load(f), "fresh"

    # Verificar si la fecha está en los archivos NWP archivados (2024_25 o 2023_24) para paridad (R1)
    archive_paths = [
        Path("data/raw/nwp_archive/nwp_2024_25.parquet"),
        Path("data/raw/nwp_archive/nwp_2023_24.parquet"),
    ]
    for ap in archive_paths:
        if ap.exists():
            df_arch = pl.read_parquet(ap)
            sub_arch = df_arch.filter(pl.col("date") == target_date)
            if sub_arch.height > 0:
                logger.info(f"Cargando pronóstico desde archivo NWP ({ap.name}) para {target_date}...")
                arch_weather = {}
                for row in sub_arch.iter_rows(named=True):
                    arch_weather[row["node_id"]] = {
                        "temp_noon_c": float(row["temp_noon_c_h0"]),
                        "rh_noon_pct": float(row["rh_noon_pct_h0"]),
                        "wind_noon_kmh": float(row["wind_noon_kmh_h0"]),
                        "rain_24h_noon_mm": float(row["rain_24h_noon_mm_h0"]),
                    }
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(arch_weather, f)
                return arch_weather, "fresh"

    # Consulta real a Open-Meteo Live Forecast API con modelo NOAA GFS (gfs_seamless) en lotes de 20 coordenadas (R1)
    fresh_weather: dict[str, dict[str, float]] = {}
    batch_size = 20
    n_batches = (len(nodes) + batch_size - 1) // batch_size
    prev_day = target_date - timedelta(days=1)

    try:
        import urllib.request

        logger.info(
            f"Descargando pronóstico en vivo (modelo gfs_seamless) para {len(nodes)} nodos ({target_date}) en {n_batches} lotes..."
        )
        for b in range(n_batches):
            b_nodes = nodes[b * batch_size : (b + 1) * batch_size]
            lat_str = ",".join(f"{coord_dict[n][0]:.4f}" for n in b_nodes)
            lon_str = ",".join(f"{coord_dict[n][1]:.4f}" for n in b_nodes)
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}"
                f"&models=gfs_seamless&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
                f"&start_date={prev_day.isoformat()}&end_date={target_date.isoformat()}"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-Pilot/1.0"})
            with urllib.request.urlopen(req, timeout=25) as resp:
                data = json.loads(resp.read().decode())
                if not isinstance(data, list):
                    data = [data]

                for nid, node_data in zip(b_nodes, data, strict=True):
                    hourly = node_data.get("hourly", {})
                    times = hourly.get("time", [])
                    temps = hourly.get("temperature_2m", [])
                    rhs = hourly.get("relative_humidity_2m", [])
                    winds = hourly.get("wind_speed_10m", [])
                    rains = hourly.get("precipitation", [])

                    t16_str = f"{target_date.isoformat()}T16:00"
                    if t16_str in times:
                        idx = times.index(t16_str)
                        start_rain = max(0, idx - 23)
                        r_slice = [r for r in rains[start_rain : idx + 1] if r is not None]
                        rain24 = float(sum(r_slice)) if r_slice else 0.0
                        t_val = temps[idx] if temps[idx] is not None else 20.0
                        rh_val = rhs[idx] if rhs[idx] is not None else 50.0
                        w_val = winds[idx] if winds[idx] is not None else 15.0
                        fresh_weather[nid] = {
                            "temp_noon_c": float(t_val),
                            "rh_noon_pct": float(rh_val),
                            "wind_noon_kmh": float(w_val),
                            "rain_24h_noon_mm": rain24,
                        }
                    else:
                        fresh_weather[nid] = {
                            "temp_noon_c": 20.0,
                            "rh_noon_pct": 50.0,
                            "wind_noon_kmh": 15.0,
                            "rain_24h_noon_mm": 0.0,
                        }
            time.sleep(0.3)

        freshness = "fresh"
        logger.info(f"Pronóstico en vivo descargado con éxito para {len(fresh_weather)} nodos.")
    except Exception as exc:
        logger.warning(
            f"Falla segura activada: Open-Meteo API inaccesible ({exc}). Usando fallback 'stale'."
        )
        fresh_weather = {
            n: {
                "temp_noon_c": 22.0,
                "rh_noon_pct": 45.0,
                "wind_noon_kmh": 18.0,
                "rain_24h_noon_mm": 0.0,
            }
            for n in nodes
        }
        freshness = "stale"

    if cache_file and freshness == "fresh":
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(fresh_weather, f)

    return fresh_weather, freshness


def update_fwi_state(
    state_file: Path,
    target_date: date,
    nodes: list[str],
    weather_by_node: dict[str, dict[str, float]],
    initial_fwi_path: Path,
) -> pl.DataFrame:
    """Actualiza de forma idempotente el archivo de estado de FWI (FFMC, DMC, DC)."""
    state_file.parent.mkdir(parents=True, exist_ok=True)

    if state_file.exists():
        state_df = pl.read_parquet(state_file)
        existing_dates = state_df.get_column("date").unique().to_list()
        if target_date in existing_dates:
            logger.info(f"Estado FWI para {target_date} ya existe en {state_file}. Conservando.")
            return state_df.filter(pl.col("date") == target_date)
        prev_date = target_date - timedelta(days=1)
        prev_df = state_df.filter(pl.col("date") == prev_date)
        if prev_df.height == 0:
            # Buscar en reanálisis histórico para arranque en caliente de fecha pasada (R1)
            if initial_fwi_path.exists():
                init_df = pl.read_parquet(initial_fwi_path)
                prev_df = init_df.filter(pl.col("date") == prev_date)
            if prev_df.height == 0:
                logger.warning(f"No hay estado para {prev_date}. Tomando el último día disponible.")
                max_d = state_df.get_column("date").max()
                prev_df = state_df.filter(pl.col("date") == max_d)
    else:
        logger.info(f"Inicializando estado FWI desde {initial_fwi_path}")
        init_df = pl.read_parquet(initial_fwi_path)
        prev_date = target_date - timedelta(days=1)
        prev_df = init_df.filter(pl.col("date") == prev_date)
        if prev_df.height == 0:
            max_d = init_df.get_column("date").max()
            prev_df = init_df.filter(pl.col("date") == max_d)
        state_df = prev_df

    # Calcular avance de 1 día de FWI para cada nodo
    prev_map = {row["node_id"]: row for row in prev_df.iter_rows(named=True)}
    new_rows = []
    month_val = np.array([target_date.month], dtype=np.int64)

    for n in nodes:
        w = weather_by_node[n]
        p_row = prev_map.get(n, {"ffmc": 85.0, "dmc": 6.0, "dc": 15.0})

        fwi_out = compute_fwi_series_1d(
            temp_noon_c=np.array([w["temp_noon_c"]]),
            rh_noon_pct=np.array([w["rh_noon_pct"]]),
            wind_noon_kmh=np.array([w["wind_noon_kmh"]]),
            rain_24h_noon_mm=np.array([w["rain_24h_noon_mm"]]),
            months=month_val,
            ffmc0=float(p_row["ffmc"]),
            dmc0=float(p_row["dmc"]),
            dc0=float(p_row["dc"]),
        )
        new_rows.append(
            {
                "node_id": n,
                "date": target_date,
                "ffmc": float(fwi_out["ffmc"][0]),
                "dmc": float(fwi_out["dmc"][0]),
                "dc": float(fwi_out["dc"][0]),
                "isi": float(fwi_out["isi"][0]),
                "bui": float(fwi_out["bui"][0]),
                "fwi": float(fwi_out["fwi"][0]),
            }
        )

    fwi_schema = {
        "node_id": pl.String,
        "date": pl.Date,
        "fwi": pl.Float32,
        "isi": pl.Float32,
        "bui": pl.Float32,
        "ffmc": pl.Float32,
        "dmc": pl.Float32,
        "dc": pl.Float32,
    }
    cols = ["node_id", "date", "fwi", "isi", "bui", "ffmc", "dmc", "dc"]
    day_df = pl.DataFrame(new_rows).cast(fwi_schema).select(cols)

    # Actualizar parquet de estado persistente
    if state_file.exists():
        combined_state = pl.concat([state_df, day_df], how="diagonal").unique(subset=["node_id", "date"])
    else:
        combined_state = day_df
    combined_state.write_parquet(state_file)

    return day_df


def generate_static_grid_geojson(grid_df: pl.DataFrame, out_path: Path) -> None:
    """Genera la geometría estática de la grilla 1km (R6) una sola vez para el dashboard."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nuble_grid = grid_df.filter(pl.col("in_nuble"))
    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

    min_x = nuble_grid["min_x"].to_numpy()
    min_y = nuble_grid["min_y"].to_numpy()
    max_x = nuble_grid["max_x"].to_numpy()
    max_y = nuble_grid["max_y"].to_numpy()
    cell_ids = nuble_grid["cell_id"].to_list()

    p1_lon, p1_lat = trans.transform(min_x, min_y)
    p2_lon, p2_lat = trans.transform(max_x, min_y)
    p3_lon, p3_lat = trans.transform(max_x, max_y)
    p4_lon, p4_lat = trans.transform(min_x, max_y)

    features = []
    for i in range(len(cell_ids)):
        coords = [
            [round(p1_lon[i], 5), round(p1_lat[i], 5)],
            [round(p2_lon[i], 5), round(p2_lat[i], 5)],
            [round(p3_lon[i], 5), round(p3_lat[i], 5)],
            [round(p4_lon[i], 5), round(p4_lat[i], 5)],
            [round(p1_lon[i], 5), round(p1_lat[i], 5)],
        ]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {"cell_id": cell_ids[i]}
        })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geojson, f)
    logger.info(f"Geometría estática de grilla guardada en {out_path} ({out_path.stat().st_size:,} bytes)")


def run_daily_shadow_pipeline(
    target_date: date,
    output_dir: Path = Path("outputs"),
    state_dir: Path = Path("data/state"),
    mock_weather_path: Path | None = None,
    log_path: Path | None = None,
    is_simulation: bool = False,
    run_id: str | None = None,
    horizon: int = 0,
    forecast_issue_date: date | None = None,
) -> dict[str, Any]:
    """Ejecuta el ciclo diario completo en modo sombra para el día D."""
    t0 = time.time()
    query_ts_utc = datetime.now(UTC).isoformat()
    cfg = load_config()

    # Protección de aislamiento U1: prohibir que simulaciones o tests toquen el prospective_log real
    prod_log_path = Path("data/prospective_log.parquet").resolve()
    if is_simulation:
        if log_path is None:
            raise ValueError(
                "Las simulaciones y tests deben especificar explícitamente un 'log_path' temporal; "
                "está prohibido escribir en 'data/prospective_log.parquet' de producción."
            )
        if Path(log_path).resolve() == prod_log_path:
            raise RuntimeError(
                "Violación de aislamiento U1: una simulación intentó escribir en el prospective_log real."
            )
    else:
        if log_path is None:
            log_path = Path("data/prospective_log.parquet")

    if run_id is None:
        prefix = "sim" if is_simulation else "prod"
        run_id = f"{prefix}_{target_date.isoformat()}_h{horizon}_{uuid.uuid4().hex[:8]}"

    day_out_dir = output_dir / target_date.isoformat()
    day_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar catálogo espacial y features estáticas
    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    season = get_latest_season_label(target_date)

    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    prior_sub = prior_df.filter(pl.col("season") == season)
    if prior_sub.height == 0:
        max_s = prior_df.get_column("season").max()
        logger.warning(
            f"Temporada {season} no encontrada en prior. Usando última temporada {max_s}."
        )
        prior_sub = prior_df.filter(pl.col("season") == max_s)

    prior_map = dict(
        zip(
            prior_sub.get_column("cell_id").to_list(),
            prior_sub.get_column("prior_density").to_list(),
            strict=True,
        )
    )
    prior_series = np.array([prior_map.get(cid, 0.0) for cid in cell_ids], dtype=np.float32)

    # 2. Mapeo a nodos de clima
    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    nodes, node_coords, coord_dict = load_node_coordinates(weather_path)

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    clon, clat = trans.transform(cx, cy)
    tree = cKDTree(node_coords)
    _, cell_node_idxs = tree.query(np.column_stack([clon, clat]), k=1)
    cell_node_ids = [nodes[idx] for idx in cell_node_idxs]

    # 3. Obtención de pronóstico de clima
    weather_by_node, freshness = fetch_weather_forecast(
        target_date=target_date,
        nodes=nodes,
        coord_dict=coord_dict,
        mock_weather_path=mock_weather_path,
        cache_dir=state_dir / "cache",
    )

    # 4. Actualización del estado de FWI
    state_file = state_dir / "fwi_state.parquet"
    initial_fwi_path = Path("data/processed/era5_land_daily_fwi_nodes.parquet")
    fwi_day_df = update_fwi_state(
        state_file=state_file,
        target_date=target_date,
        nodes=nodes,
        weather_by_node=weather_by_node,
        initial_fwi_path=initial_fwi_path,
    )
    fwi_map = {row["node_id"]: row for row in fwi_day_df.iter_rows(named=True)}

    # 5. Ensamblar las 25 variables a nivel celda
    weather_cell_dict: dict[str, np.ndarray] = {
        "temp_noon_c": np.array(
            [weather_by_node[nid]["temp_noon_c"] for nid in cell_node_ids], dtype=np.float32
        ),
        "rh_noon_pct": np.array(
            [weather_by_node[nid]["rh_noon_pct"] for nid in cell_node_ids], dtype=np.float32
        ),
        "wind_noon_kmh": np.array(
            [weather_by_node[nid]["wind_noon_kmh"] for nid in cell_node_ids], dtype=np.float32
        ),
        "rain_24h_noon_mm": np.array(
            [weather_by_node[nid]["rain_24h_noon_mm"] for nid in cell_node_ids], dtype=np.float32
        ),
        "fwi": np.array([fwi_map[nid]["fwi"] for nid in cell_node_ids], dtype=np.float32),
        "isi": np.array([fwi_map[nid]["isi"] for nid in cell_node_ids], dtype=np.float32),
        "bui": np.array([fwi_map[nid]["bui"] for nid in cell_node_ids], dtype=np.float32),
        "ffmc": np.array([fwi_map[nid]["ffmc"] for nid in cell_node_ids], dtype=np.float32),
        "dmc": np.array([fwi_map[nid]["dmc"] for nid in cell_node_ids], dtype=np.float32),
        "dc": np.array([fwi_map[nid]["dc"] for nid in cell_node_ids], dtype=np.float32),
    }

    weather_cell_dict["vpd_noon_kpa"] = np.array(
        [
            compute_vpd(float(t), float(rh))
            for t, rh in zip(
                weather_cell_dict["temp_noon_c"], weather_cell_dict["rh_noon_pct"], strict=True
            )
        ],
        dtype=np.float32,
    )

    feature_matrix = build_inference_feature_matrix(
        static_df=static_df,
        prior_series=prior_series,
        weather_dict_by_cell=weather_cell_dict,
    )

    # 6. Inferencia con los candidatos (Baseline y Ensamble de Producción R3)
    n_tot = len(cell_ids)

    # Candidato 1 (Principal Oficial): Baseline FWI x Prior
    score_baseline_raw = weather_cell_dict["fwi"] * prior_series
    score_static_prior = prior_series

    # Modelo Exploratorio Secundario: Baseline FWI x Prior v2 (U5: no entra en criterios formales)
    prior_v2_path = Path("data/processed/prior_density_2026_27_v2.parquet")
    if prior_v2_path.exists():
        p_v2_df = pl.read_parquet(prior_v2_path)
        p_v2_map = dict(
            zip(
                p_v2_df.get_column("cell_id").to_list(),
                p_v2_df.get_column("prior_density").to_list(),
                strict=True,
            )
        )
        p_v2_series = np.array([p_v2_map.get(cid, 0.0) for cid in cell_ids], dtype=np.float32)
        score_baseline_prior_v2 = (weather_cell_dict["fwi"] * p_v2_series).astype(np.float32)
    else:
        score_baseline_prior_v2 = score_baseline_raw

    # Candidato 2 (Producción ML): Ensamble de 10 Semillas Canónicas (R3)
    ensemble_dir = Path("models/chilefuego_xgb_ensemble_v2")
    if ensemble_dir.exists():
        boosters = []
        for mf in sorted(ensemble_dir.glob("model_seed_*.json")):
            b = xgb.Booster()
            b.load_model(str(mf))
            boosters.append(b)
        dmat_cells = xgb.DMatrix(feature_matrix, feature_names=PRUNED_25_FEATURES)
        score_ml_raw = np.mean([b.predict(dmat_cells) for b in boosters], axis=0).astype(np.float32)
    else:
        score_ml_raw = score_baseline_raw

    # Modelos experimentales secundarios
    pipeline_current = InferencePipeline(
        model_path="models/chilefuego_xgb_pruned_v1.json",
        calibrator_path="models/platt_calibrator.json",
        thresholds_path="models/risk_thresholds.json",
    )
    preds_current = pipeline_current.predict(feature_matrix)

    var_a_path = Path("models/chilefuego_xgb_variant_a_monotone.json")
    if var_a_path.exists():
        pipeline_var_a = InferencePipeline(
            model_path=str(var_a_path),
            calibrator_path="models/platt_calibrator.json",
            thresholds_path="models/risk_thresholds.json",
        )
        preds_var_a = pipeline_var_a.predict(feature_matrix)
    else:
        preds_var_a = preds_current

    # 7. Mapeo Monótono de Cuantiles NWP -> ERA5 (R2 / T1)
    mapper_path = Path("models/quantile_mapper_nwp_era5_v1.json")
    if mapper_path.exists():
        with open(mapper_path, encoding="utf-8") as jf:
            map_data = json.load(jf)
        m_base = MonotonicQuantileMapper.from_dict(map_data["baseline_mapper"])
        m_ens = MonotonicQuantileMapper.from_dict(map_data["ensemble_mapper"])
        score_baseline_mapped = m_base.transform(score_baseline_raw)
        score_ml_mapped = m_ens.transform(score_ml_raw)
        quantile_mapping_active = True
    else:
        score_baseline_mapped = score_baseline_raw
        score_ml_mapped = score_ml_raw
        quantile_mapping_active = False

    # Asignación de clases del Baseline con umbrales fijos P3
    t80_base = BASELINE_FIXED_THRESHOLDS["p80"]
    t95_base = BASELINE_FIXED_THRESHOLDS["p95"]
    t99_base = BASELINE_FIXED_THRESHOLDS["p99"]

    # Clases crudas (para trazabilidad y auditoría de shift)
    tier_baseline_raw = np.empty(n_tot, dtype=object)
    tier_baseline_raw[score_baseline_raw < t80_base] = "Bajo"
    tier_baseline_raw[(score_baseline_raw >= t80_base) & (score_baseline_raw < t95_base)] = "Moderado"
    tier_baseline_raw[(score_baseline_raw >= t95_base) & (score_baseline_raw < t99_base)] = "Alto"
    tier_baseline_raw[score_baseline_raw >= t99_base] = "Extremo"

    # Clases MAPEADAS canónicas de producción (T1: Producto Principal)
    tier_baseline = np.empty(n_tot, dtype=object)
    tier_baseline[score_baseline_mapped < t80_base] = "Bajo"
    tier_baseline[(score_baseline_mapped >= t80_base) & (score_baseline_mapped < t95_base)] = "Moderado"
    tier_baseline[(score_baseline_mapped >= t95_base) & (score_baseline_mapped < t99_base)] = "Alto"
    tier_baseline[score_baseline_mapped >= t99_base] = "Extremo"

    # Asignación de clases del Ensamble ML con umbrales fijos LOSO (R3)
    t80_ml = ML_FIXED_THRESHOLDS["p80"]
    t95_ml = ML_FIXED_THRESHOLDS["p95"]
    t99_ml = ML_FIXED_THRESHOLDS["p99"]

    # Clases crudas ML
    tier_ml_raw = np.empty(n_tot, dtype=object)
    tier_ml_raw[score_ml_raw < t80_ml] = "Bajo"
    tier_ml_raw[(score_ml_raw >= t80_ml) & (score_ml_raw < t95_ml)] = "Moderado"
    tier_ml_raw[(score_ml_raw >= t95_ml) & (score_ml_raw < t99_ml)] = "Alto"
    tier_ml_raw[score_ml_raw >= t99_ml] = "Extremo"

    # Clases MAPEADAS canónicas de producción Ensamble ML (T1)
    tier_ml = np.empty(n_tot, dtype=object)
    tier_ml[score_ml_mapped < t80_ml] = "Bajo"
    tier_ml[(score_ml_mapped >= t80_ml) & (score_ml_mapped < t95_ml)] = "Moderado"
    tier_ml[(score_ml_mapped >= t95_ml) & (score_ml_mapped < t99_ml)] = "Alto"
    tier_ml[score_ml_mapped >= t99_ml] = "Extremo"

    # 8. Guardar outputs/AAAA-MM-DD/risk_cells.parquet con los candidatos, clases y clima
    risk_cells_df = pl.DataFrame(
        {
            "cell_id": cell_ids,
            "date": [target_date] * n_tot,
            # Producto Principal (FWI x Prior) y clases fijas P3 (Q5 / T1)
            "score_baseline": score_baseline_raw,
            "score_baseline_mapped": score_baseline_mapped,
            "tier_baseline": tier_baseline,
            "tier_baseline_raw": tier_baseline_raw,
            "score_static_prior": score_static_prior,
            # Ensamble de Producción (R3 / T1)
            "score_ml": score_ml_raw,
            "score_ml_mapped": score_ml_mapped,
            "tier_ml": tier_ml,
            "tier_ml_raw": tier_ml_raw,
            # Compatibilidad histórica
            "risk_tier": tier_baseline,
            "calibrated_prob": preds_current["calibrated_prob"],
            # Candidatos experimentales
            "score_current_model": preds_current["raw_score"],
            "score_variant_a_monotone": preds_var_a["raw_score"],
            "tier_current": preds_current["risk_tier"],
            "tier_variant_a": preds_var_a["risk_tier"],
            # Entradas meteorológicas completas
            "temp_noon_c": weather_cell_dict["temp_noon_c"],
            "rh_noon_pct": weather_cell_dict["rh_noon_pct"],
            "wind_noon_kmh": weather_cell_dict["wind_noon_kmh"],
            "rain_24h_noon_mm": weather_cell_dict["rain_24h_noon_mm"],
            "vpd_noon_kpa": weather_cell_dict["vpd_noon_kpa"],
            "fwi": weather_cell_dict["fwi"],
            "isi": weather_cell_dict["isi"],
            "bui": weather_cell_dict["bui"],
            "ffmc": weather_cell_dict["ffmc"],
            "dmc": weather_cell_dict["dmc"],
            "dc": weather_cell_dict["dc"],
        }
    )
    risk_cells_path = day_out_dir / "risk_cells.parquet"
    risk_cells_df.write_parquet(risk_cells_path)

    # 9. Guardar outputs/AAAA-MM-DD/risk_classes.geojson (clase principal = baseline, con tier_ml)
    features = []
    for i in range(min(500, n_tot)):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(clon[i]), float(clat[i])]},
                "properties": {
                    "cell_id": cell_ids[i],
                    "tier_primary": str(tier_baseline[i]),
                    "tier_baseline": str(tier_baseline[i]),
                    "tier_ml": str(tier_ml[i]),
                    "score_baseline": float(score_baseline_raw[i]),
                    "score_ml": float(score_ml_raw[i]),
                    "score_static_prior": float(score_static_prior[i]),
                },
            }
        )
    geojson_data = {"type": "FeatureCollection", "features": features}
    geojson_path = day_out_dir / "risk_classes.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # 10. Separación de datos para Dashboard (R6)
    # (a) Geometría estática de grilla 1km
    grid_geojson_path = output_dir / "grid_1km.geojson"
    if not grid_geojson_path.exists():
        generate_static_grid_geojson(grid_df, grid_geojson_path)

    # (b) Daily tiers compactos (S6: arrays de enteros 0-3 en el orden exacto de grid_1km.geojson + date)
    TIER_MAP_INT = {"Bajo": 0, "Moderado": 1, "Alto": 2, "Extremo": 3}
    compact_daily_tiers = {
        "date": target_date.isoformat(),
        "tier_baseline": [TIER_MAP_INT.get(str(t), 0) for t in tier_baseline],
        "tier_ml": [TIER_MAP_INT.get(str(t), 0) for t in tier_ml],
    }
    with open(day_out_dir / "daily_tiers.json", "w", encoding="utf-8") as f:
        json.dump(compact_daily_tiers, f)
    with open(output_dir / "daily_tiers.json", "w", encoding="utf-8") as f:
        json.dump(compact_daily_tiers, f)

    # 11. Conteo de celdas y porcentajes por clase (Mapeadas y Crudas)
    counts_baseline = {
        "Bajo": int(np.sum(tier_baseline == "Bajo")),
        "Moderado": int(np.sum(tier_baseline == "Moderado")),
        "Alto": int(np.sum(tier_baseline == "Alto")),
        "Extremo": int(np.sum(tier_baseline == "Extremo")),
    }
    pcts_baseline = {k: round(float(v / n_tot * 100), 2) for k, v in counts_baseline.items()}

    counts_baseline_raw = {
        "Bajo": int(np.sum(tier_baseline_raw == "Bajo")),
        "Moderado": int(np.sum(tier_baseline_raw == "Moderado")),
        "Alto": int(np.sum(tier_baseline_raw == "Alto")),
        "Extremo": int(np.sum(tier_baseline_raw == "Extremo")),
    }
    pcts_baseline_raw = {k: round(float(v / n_tot * 100), 2) for k, v in counts_baseline_raw.items()}

    counts_ml = {
        "Bajo": int(np.sum(tier_ml == "Bajo")),
        "Moderado": int(np.sum(tier_ml == "Moderado")),
        "Alto": int(np.sum(tier_ml == "Alto")),
        "Extremo": int(np.sum(tier_ml == "Extremo")),
    }
    pcts_ml = {k: round(float(v / n_tot * 100), 2) for k, v in counts_ml.items()}

    counts_ml_raw = {
        "Bajo": int(np.sum(tier_ml_raw == "Bajo")),
        "Moderado": int(np.sum(tier_ml_raw == "Moderado")),
        "Alto": int(np.sum(tier_ml_raw == "Alto")),
        "Extremo": int(np.sum(tier_ml_raw == "Extremo")),
    }
    pcts_ml_raw = {k: round(float(v / n_tot * 100), 2) for k, v in counts_ml_raw.items()}

    # 12. Guardar metadata y latest.json (R1 / R6 / T1)
    ensemble_manifest_path = Path("models/chilefuego_xgb_ensemble_v2/manifest.json")
    model_hash = compute_file_sha256(ensemble_manifest_path) if ensemble_manifest_path.exists() else "unknown"

    metadata = {
        "model_hash": model_hash,
        "model_name": "chilefuego_xgb_ensemble_v2",
        "target_date": target_date.isoformat(),
        "run_timestamp": datetime.now(UTC).isoformat(),
        "weather_source": "Open-Meteo Live Forecast API (models=gfs_seamless)" if mock_weather_path is None else "mock",
        "forecast_model_used": "gfs_seamless",
        "freshness": freshness,
        "quantile_mapping_active": quantile_mapping_active,
        "n_cells": n_tot,
        "risk_status": "NORMAL",
        "alert_status": "NORMAL",  # Deprecated in v1.3; alias to risk_status
        "tier_distribution_pct_mapped": pcts_baseline,
        "pct_by_tier_baseline_mapped": pcts_baseline,
        "tier_distribution_pct": pcts_baseline,  # Deprecated in v1.3; alias to tier_distribution_pct_mapped
        "pct_by_tier_baseline": pcts_baseline,  # Deprecated in v1.3; alias to pct_by_tier_baseline_mapped
        "counts_by_tier_baseline": counts_baseline,
        "counts_by_tier_baseline_raw": counts_baseline_raw,
        "pct_by_tier_baseline_raw": pcts_baseline_raw,
        "counts_by_tier_ml": counts_ml,
        "pct_by_tier_ml": pcts_ml,
        "counts_by_tier_ml_raw": counts_ml_raw,
        "pct_by_tier_ml_raw": pcts_ml_raw,
        "tier_baseline_thresholds": BASELINE_FIXED_THRESHOLDS,
        "tier_ml_thresholds": ML_FIXED_THRESHOLDS,
        "run_id": run_id,
        "horizon": horizon,
        "is_simulation": is_simulation,
        "query_timestamp_utc": query_ts_utc,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    meta_path = day_out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(day_out_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    with open(output_dir / "latest.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 13. Registrar en prospective_log (clave única: run_id, cell_id, target_date, horizon)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    issue_date = forecast_issue_date if forecast_issue_date is not None else (target_date - timedelta(days=horizon))
    day_log_df = pl.DataFrame(
        {
            "run_id": [run_id] * n_tot,
            "date": [target_date] * n_tot,
            "target_date": [target_date] * n_tot,
            "cell_id": cell_ids,
            "horizon": [horizon] * n_tot,
            "is_simulation": [is_simulation] * n_tot,
            "score_baseline": score_baseline_raw,
            "score_baseline_prior_v2": score_baseline_prior_v2,
            "score_baseline_mapped": score_baseline_mapped,
            "tier_baseline": tier_baseline,
            "tier_baseline_raw": tier_baseline_raw,
            "score_static_prior": score_static_prior,
            "score_ml": score_ml_raw,
            "score_ml_mapped": score_ml_mapped,
            "tier_ml": tier_ml,
            "tier_ml_raw": tier_ml_raw,
            "score_current_model": preds_current["raw_score"],
            "score_variant_a_monotone": preds_var_a["raw_score"],
            "temp_noon_c": weather_cell_dict["temp_noon_c"],
            "rh_noon_pct": weather_cell_dict["rh_noon_pct"],
            "wind_noon_kmh": weather_cell_dict["wind_noon_kmh"],
            "rain_24h_noon_mm": weather_cell_dict["rain_24h_noon_mm"],
            "vpd_noon_kpa": weather_cell_dict["vpd_noon_kpa"],
            "fwi": weather_cell_dict["fwi"],
            "isi": weather_cell_dict["isi"],
            "bui": weather_cell_dict["bui"],
            "ffmc": weather_cell_dict["ffmc"],
            "dmc": weather_cell_dict["dmc"],
            "dc": weather_cell_dict["dc"],
            "source_model": ["gfs_seamless"] * n_tot,
            "forecast_issue_date": [issue_date] * n_tot,
            "excluir_de_evaluacion": [freshness != "fresh" or is_simulation] * n_tot,
            "query_timestamp_utc": [query_ts_utc] * n_tot,
            "run_timestamp": [query_ts_utc] * n_tot,
            "freshness": [freshness] * n_tot,
        }
    )
    unique_key = ["run_id", "cell_id", "target_date", "horizon"]
    if log_path.exists():
        exist_log = pl.read_parquet(log_path)
        combined_log = pl.concat([exist_log, day_log_df], how="diagonal").unique(subset=unique_key)
    else:
        combined_log = day_log_df.unique(subset=unique_key)
    combined_log.write_parquet(log_path)

    logger.info(f"Ciclo diario para {target_date} completado en {round(time.time() - t0, 2)}s.")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Piloto en modo sombra de predicción diaria ChileFuego."
    )
    parser.add_argument(
        "--date", type=str, default=date.today().isoformat(), help="Fecha objetivo AAAA-MM-DD"
    )
    parser.add_argument(
        "--horizon", type=int, default=0, help="Horizonte de pronóstico (0 para día D, 1 para D+1)"
    )
    parser.add_argument(
        "--mock-weather", type=str, default=None, help="Ruta a JSON con clima mock para pruebas"
    )
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directorio de salidas")
    args = parser.parse_args()

    base_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    target_d = base_date + timedelta(days=args.horizon)

    logger.info(
        f"Iniciando predicción diaria modo sombra para {target_d} (horizonte {args.horizon}, emisión {base_date})..."
    )
    meta = run_daily_shadow_pipeline(
        target_date=target_d,
        output_dir=Path(args.output_dir),
        mock_weather_path=Path(args.mock_weather) if args.mock_weather else None,
        horizon=args.horizon,
        forecast_issue_date=base_date,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
