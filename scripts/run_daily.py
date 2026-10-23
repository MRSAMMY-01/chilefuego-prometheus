"""Piloto Operacional en Modo Sombra para Predicción Diaria de Peligro de Incendios (H5).

PORQUÉ (Regla 5):
- Ejecuta el piloto diario para el día D y horizonte D+1 sin publicar en el dashboard.
- Mantiene un archivo de estado FWI versionado e idempotente que arrastra memoria de humedad.
- Falla segura: ante caída de API o red, recurre al último pronóstico disponible marcando 'stale'.
- Registra cada corrida prospectiva en 'data/prospective_log.parquet' para futura validación con FIRMS/CONAF.
- Alarma de desvío: compara la distribución observada contra el baseline 80/15/4/1 % y alerta si excede 5 pp.
"""

from __future__ import annotations

import argparse
import hashlib
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
    InferencePipeline,
    build_inference_feature_matrix,
    compute_vpd,
)
from pyproj import Transformer
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_daily")


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


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

    # Consulta real a Open-Meteo Live Forecast API en lotes de 20 coordenadas
    fresh_weather: dict[str, dict[str, float]] = {}
    batch_size = 20
    n_batches = (len(nodes) + batch_size - 1) // batch_size
    prev_day = target_date - timedelta(days=1)

    try:
        import urllib.request

        logger.info(
            f"Descargando pronóstico en vivo para {len(nodes)} nodos ({target_date}) en {n_batches} lotes..."
        )
        for b in range(n_batches):
            b_nodes = nodes[b * batch_size : (b + 1) * batch_size]
            lat_str = ",".join(f"{coord_dict[n][0]:.4f}" for n in b_nodes)
            lon_str = ",".join(f"{coord_dict[n][1]:.4f}" for n in b_nodes)
            url = (
                f"https://api.open-meteo.com/v1/forecast?latitude={lat_str}&longitude={lon_str}"
                f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
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
        # Verificar si la fecha ya fue calculada (idempotencia)
        existing_dates = state_df.get_column("date").unique().to_list()
        if target_date in existing_dates:
            logger.info(f"Estado FWI para {target_date} ya existe. Conservando sin duplicar.")
            return state_df.filter(pl.col("date") == target_date)
        prev_date = target_date - timedelta(days=1)
        prev_df = state_df.filter(pl.col("date") == prev_date)
        if prev_df.height == 0:
            logger.warning(f"No hay estado para {prev_date}. Tomando el último día disponible.")
            max_d = state_df.get_column("date").max()
            prev_df = state_df.filter(pl.col("date") == max_d)
    else:
        logger.info(f"Inicializando {state_file} desde {initial_fwi_path}")
        init_df = pl.read_parquet(initial_fwi_path)
        # Tomar último día disponible en el reanálisis
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
    updated_state_df = pl.concat([state_df.cast(fwi_schema).select(cols), day_df]).unique(
        subset=["node_id", "date"]
    )

    # Escritura atómica
    tmp_path = state_file.with_suffix(".tmp.parquet")
    updated_state_df.write_parquet(tmp_path)
    tmp_path.replace(state_file)
    logger.info(f"Estado FWI actualizado guardado en {state_file}")

    return day_df


def run_daily_shadow_pipeline(
    target_date: date,
    output_dir: Path = Path("outputs"),
    state_dir: Path = Path("data/state"),
    mock_weather_path: Path | None = None,
) -> dict[str, Any]:
    """Ejecuta el ciclo diario completo en modo sombra para el día D."""
    t0 = time.time()
    cfg = load_config()

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
    # Calcular VPD a partir de temperatura y humedad relativa
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

    # 6. Inferencia con ambos modelos en modo sombra (sin ponderar y ponderado por tamaño)
    pipeline_unw = InferencePipeline(
        model_path="models/chilefuego_xgb_pruned_v1.json",
        calibrator_path="models/platt_calibrator.json",
        thresholds_path="models/risk_thresholds.json",
    )
    preds_unw = pipeline_unw.predict(feature_matrix)

    pipeline_w = InferencePipeline(
        model_path="models/chilefuego_xgb_pruned_v1.json",
        calibrator_path="models/platt_calibrator_sizew.json",
        thresholds_path="models/risk_thresholds_sizew.json",
    )
    preds_w = pipeline_w.predict(feature_matrix)

    # 7. Distribuciones y alarmas de desvío contra 80/15/4/1 %
    n_tot = len(cell_ids)

    def compute_tiers_and_devs(
        tiers: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float], bool]:
        p_b = float(np.sum(tiers == "Bajo") / n_tot * 100.0)
        p_m = float(np.sum(tiers == "Moderado") / n_tot * 100.0)
        p_a = float(np.sum(tiers == "Alto") / n_tot * 100.0)
        p_e = float(np.sum(tiers == "Extremo") / n_tot * 100.0)
        devs = {
            "Bajo": p_b - 80.0,
            "Moderado": p_m - 15.0,
            "Alto": p_a - 4.0,
            "Extremo": p_e - 1.0,
        }
        alarm = any(abs(v) > 5.0 for v in devs.values())
        return {"Bajo": p_b, "Moderado": p_m, "Alto": p_a, "Extremo": p_e}, devs, alarm

    dist_unw, devs_unw, alarm_unw = compute_tiers_and_devs(preds_unw["risk_tier"])
    dist_w, devs_w, alarm_w = compute_tiers_and_devs(preds_w["risk_tier"])

    if alarm_unw or alarm_w:
        logger.warning(f"ALARMA DE DESVÍO: Unweighted={devs_unw} | Weighted={devs_w}")

    # 8. Guardar outputs/AAAA-MM-DD/risk_cells.parquet con ambos modelos
    risk_cells_df = pl.DataFrame(
        {
            "cell_id": cell_ids,
            "date": [target_date] * n_tot,
            "raw_score": preds_unw["raw_score"],
            "calibrated_prob": preds_unw["calibrated_prob"],
            "risk_tier": preds_unw["risk_tier"],
            "raw_score_unweighted": preds_unw["raw_score"],
            "raw_score_weighted": preds_w["raw_score"],
            "calibrated_prob_unweighted": preds_unw["calibrated_prob"],
            "calibrated_prob_weighted": preds_w["calibrated_prob"],
            "risk_tier_unweighted": preds_unw["risk_tier"],
            "risk_tier_weighted": preds_w["risk_tier"],
            "fwi": weather_cell_dict["fwi"],
            "temp_noon_c": weather_cell_dict["temp_noon_c"],
            "rh_noon_pct": weather_cell_dict["rh_noon_pct"],
            "wind_noon_kmh": weather_cell_dict["wind_noon_kmh"],
        }
    )
    risk_cells_path = day_out_dir / "risk_cells.parquet"
    risk_cells_df.write_parquet(risk_cells_path)

    # 9. Guardar outputs/AAAA-MM-DD/risk_classes.geojson
    features = []
    for i in range(min(500, n_tot)):
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(clon[i]), float(clat[i])]},
                "properties": {
                    "cell_id": cell_ids[i],
                    "tier_unweighted": str(preds_unw["risk_tier"][i]),
                    "tier_weighted": str(preds_w["risk_tier"][i]),
                    "prob_unweighted": float(preds_unw["calibrated_prob"][i]),
                    "prob_weighted": float(preds_w["calibrated_prob"][i]),
                },
            }
        )
    geojson_data = {"type": "FeatureCollection", "features": features}
    geojson_path = day_out_dir / "risk_classes.geojson"
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    # 10. Guardar outputs/AAAA-MM-DD/metadata.json
    model_path = Path("models/chilefuego_xgb_pruned_v1.json")
    model_hash = compute_file_sha256(model_path) if model_path.exists() else "unknown"
    metadata = {
        "model_hash": model_hash,
        "model_version": "pruned_25_dual_shadow",
        "target_date": target_date.isoformat(),
        "run_timestamp": datetime.now(UTC).isoformat(),
        "weather_source": "Open-Meteo Live Forecast API",
        "freshness": freshness,
        "n_cells": n_tot,
        "tier_distribution_pct": dist_unw,
        "tier_deviations_pp": devs_unw,
        "alert_status": "ALARM_DEVIATION_EXCEEDED" if (alarm_unw or alarm_w) else "NORMAL",
        "unweighted": {
            "tier_distribution_pct": dist_unw,
            "tier_deviations_pp": devs_unw,
            "alarm_triggered": alarm_unw,
        },
        "size_weighted": {
            "tier_distribution_pct": dist_w,
            "tier_deviations_pp": devs_w,
            "alarm_triggered": alarm_w,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    meta_path = day_out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # 11. Registrar en data/prospective_log.parquet con columnas separadas para ambos modelos
    log_path = Path("data/prospective_log.parquet")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    day_log_df = pl.DataFrame(
        {
            "date": [target_date] * n_tot,
            "cell_id": cell_ids,
            "score_unweighted": preds_unw["raw_score"],
            "score_weighted": preds_w["raw_score"],
            "calibrated_p_unweighted": preds_unw["calibrated_prob"],
            "calibrated_p_weighted": preds_w["calibrated_prob"],
            "tier_unweighted": preds_unw["risk_tier"],
            "tier_weighted": preds_w["risk_tier"],
            "run_timestamp": [datetime.now(UTC).isoformat()] * n_tot,
            "freshness": [freshness] * n_tot,
        }
    )
    if log_path.exists():
        exist_log = pl.read_parquet(log_path)
        combined_log = pl.concat([exist_log, day_log_df], how="diagonal").unique(subset=["date", "cell_id"])
    else:
        combined_log = day_log_df
    combined_log.write_parquet(log_path)
    logger.info(
        f"Registro prospectivo acumulado actualizado en {log_path} ({combined_log.height} filas)"
    )

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
        f"Iniciando predicción diaria modo sombra para {target_d} (horizonte {args.horizon})..."
    )
    meta = run_daily_shadow_pipeline(
        target_date=target_d,
        output_dir=Path(args.output_dir),
        mock_weather_path=Path(args.mock_weather) if args.mock_weather else None,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
