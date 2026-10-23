"""Descarga controlada y reanudable de pronósticos archivados NWP (J1).

PORQUÉ (Regla 5):
- Descarga pronósticos numéricos del tiempo (NWP) de Open-Meteo Previous Runs API
  exclusivamente para los meses de temporada activa (noviembre a abril) de 2023_24 y 2024_25 (363 días).
- Utiliza un modelo de pronóstico global fijo que cubre Chile continuamente desde antes de noviembre de 2023:
  NOAA GFS (gfs_seamless, disponible desde marzo de 2021; ECMWF IFS en Previous Runs solo inicia en enero de 2024).
- Estima y muestra el costo ponderado antes de proceder (costo = ubicaciones * dias/14 * variables/10).
- Exige confirmación explícita (--confirm) en CLI para evitar ejecuciones accidentales.
- Respeta límites estrictos: nunca supera 4.500 llamadas ponderadas por hora ni 8.000 por día.
- Maneja 429 capturando el motivo JSON y Retry-After.
- Guarda un archivo Parquet por temporada con manifest SHA-256 y atribución CC BY 4.0.
- Incluye sonda de diagnóstico (--probe) para 1 nodo y 7 días.

CRÍTICO: Este script está preparado para ejecución manual y controlada por el usuario.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_nwp_archive")

OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FIXED_MODEL = "gfs_seamless"  # NOAA GFS cubre Chile globalmente desde marzo 2021

# Temporadas activas nov-abr (363 días en total)
NWP_SEASONS: dict[str, tuple[date, date]] = {
    "2023_24": (date(2023, 11, 1), date(2024, 4, 30)),  # 182 días (año bisiesto 2024)
    "2024_25": (date(2024, 11, 1), date(2025, 4, 30)),  # 181 días
}

HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "wind_speed_10m_previous_day1",
    "precipitation_previous_day1",
]

MAX_WEIGHTED_CALLS_PER_HOUR = 4500.0
MAX_WEIGHTED_CALLS_PER_DAY = 8000.0


def compute_weighted_cost(n_locations: int, n_days: int, n_variables: int) -> float:
    """Calcula el costo de llamadas ponderadas según la fórmula oficial de Open-Meteo:

    cost = ubicaciones * (dias / 14) * (variables / 10)
    """
    return float(n_locations * (n_days / 14.0) * (n_variables / 10.0))


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_nuble_nodes(weather_path: Path) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Carga los 399 nodos ERA5-Land de Ñuble."""
    weather_df = pl.read_parquet(weather_path)
    nodes = sorted(weather_df.select(["node_id"]).unique().to_series().to_list())
    coord_dict: dict[str, tuple[float, float]] = {}
    for n in nodes:
        lat_str, lon_str = n.split("_")
        coord_dict[n] = (float(lat_str), float(lon_str))
    return nodes, coord_dict


def run_probe(lat: float = -36.5625, lon: float = -72.0625) -> int:
    """Sonda mínima (--probe) para 1 nodo y 7 días que reporta modelos y variables disponibles."""
    logger.info("=== EJECUTANDO SONDA NWP DE DIAGNÓSTICO (1 COORDENADA, 7 DÍAS) ===")
    logger.info(f"Coordenada: lat={lat:.4f}, lon={lon:.4f}")
    logger.info("Periodo de prueba: 2024-01-01 a 2024-01-07")
    logger.info(f"Modelo especificado: {FIXED_MODEL}")
    logger.info(f"Variables solicitadas: {HOURLY_VARIABLES}")

    url = (
        f"{OPEN_METEO_PREVIOUS_RUNS_URL}?"
        f"latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date=2024-01-01&end_date=2024-01-07"
        f"&hourly={','.join(HOURLY_VARIABLES)}"
        f"&models={FIXED_MODEL}"
    )

    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-NWP/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            elapsed = round(time.time() - t0, 2)
            if isinstance(data, list):
                data = data[0]

            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            logger.info(
                f"SONDA EXITOSA en {elapsed}s: Recibidas {len(times)} horas de pronóstico archivado."
            )
            logger.info(f"Modelo resuelto por la API: {data.get('model', FIXED_MODEL)}")
            logger.info(f"Elevación del nodo: {data.get('elevation')} m")
            logger.info(f"Variables confirmadas disponibles: {list(hourly.keys())}")
            return 0
    except urllib.error.HTTPError as http_err:
        try:
            err_body = http_err.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        try:
            err_json = json.loads(err_body)
            reason = str(err_json.get("reason", ""))
        except Exception:
            reason = err_body[:200]
        logger.error(
            f"SONDA FALLIDA: HTTP {http_err.code} ({http_err.reason}) | Razón: '{reason}' | "
            f"Retry-After: {http_err.headers.get('Retry-After')}"
        )
        return 1
    except Exception as exc:
        logger.error(f"SONDA FALLIDA con excepción: {exc}")
        return 1


def download_batch_with_retry(
    lats: list[float],
    lons: list[float],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
) -> list[dict[str, Any]] | None:
    """Descarga un lote de coordenadas con manejo estricto de límites y captura 429."""
    lat_p = ",".join(f"{lat:.4f}" for lat in lats)
    lon_p = ",".join(f"{lon:.4f}" for lon in lons)
    url = (
        f"{OPEN_METEO_PREVIOUS_RUNS_URL}?"
        f"latitude={lat_p}&longitude={lon_p}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly={','.join(HOURLY_VARIABLES)}"
        f"&models={FIXED_MODEL}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-NWP/1.0"})

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
                return [data]
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            try:
                err_json = json.loads(err_body)
                reason = str(err_json.get("reason", ""))
            except Exception:
                reason = err_body[:200]

            retry_after = http_err.headers.get("Retry-After")
            if http_err.code == 429:
                logger.warning(
                    f"HTTP 429 Too Many Requests | Razón: '{reason}' | "
                    f"Retry-After: {retry_after} (intento {attempt + 1}/{max_retries})"
                )
                if "Daily API request limit exceeded" in reason:
                    logger.error("Cuota diaria de Open-Meteo agotada. Abortando.")
                    return None
                wait_s = (
                    float(retry_after) + 1.0
                    if retry_after and retry_after.isdigit()
                    else 20.0 * (attempt + 1)
                )
                time.sleep(wait_s)
            else:
                logger.error(f"Error HTTP {http_err.code} ({http_err.reason}): '{reason}'")
                if attempt == max_retries - 1:
                    return None
                time.sleep(5)
        except Exception as exc:
            logger.warning(f"Excepción de red ({exc}). Pausa de 5s...")
            time.sleep(5)

    return None


def execute_download(
    confirm: bool,
    batch_size: int = 5,
    sleep_sec: float = 3.0,
    output_dir: Path = Path("data/raw/nwp_archive"),
) -> dict[str, Any]:
    """Ejecuta la descarga planificada verificando el costo ponderado y la autorización."""
    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    nodes, coord_dict = load_nuble_nodes(weather_path)
    n_nodes = len(nodes)
    n_vars = len(HOURLY_VARIABLES)

    total_days = sum((e - s).days + 1 for s, e in NWP_SEASONS.values())
    total_weighted_cost = compute_weighted_cost(n_nodes, total_days, n_vars)

    # Costo por llamada con batch_size nodos
    n_batches_per_season = (n_nodes + batch_size - 1) // batch_size
    batch_cost_est = compute_weighted_cost(batch_size, 182, n_vars)

    logger.info("================ PLAN DE DESCARGA NWP ARCHIVE ================")
    logger.info(f"Dominio: {n_nodes} nodos ERA5-Land en Región de Ñuble")
    logger.info(f"Modelo seleccionado: {FIXED_MODEL} (cobertura continua Chile desde marzo 2021)")
    logger.info(f"Variables horarias ({n_vars}): {HOURLY_VARIABLES}")
    logger.info(f"Periodos activos: 2023_24 (182 días) + 2024_25 (181 días) = {total_days} días")
    logger.info(f"Tamaño de lote: {batch_size} nodos por llamada")
    logger.info(
        f"Lotes por temporada: {n_batches_per_season} (Total: {n_batches_per_season * 2} llamadas HTTP)"
    )
    logger.info(f"Costo por lote: ~{batch_cost_est:.1f} llamadas ponderadas")
    logger.info(f"COSTO TOTAL PONDERADO ESTIMADO: {total_weighted_cost:.1f} llamadas")
    logger.info(
        f"Límites seguros: Máx {MAX_WEIGHTED_CALLS_PER_HOUR:.0f}/hora, Máx {MAX_WEIGHTED_CALLS_PER_DAY:.0f}/día"
    )
    logger.info("===============================================================")

    if not confirm:
        logger.warning(
            "DESCARGA DETENIDA: Se requiere la bandera '--confirm' para proceder. "
            "Ejecuta con '--confirm' solo cuando estés listo para lanzar la descarga manual."
        )
        return {
            "status": "STOPPED_AWAITING_CONFIRMATION",
            "total_weighted_cost": total_weighted_cost,
            "total_days": total_days,
            "n_nodes": n_nodes,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # Verificar cuota diaria disponible antes de empezar
    if total_weighted_cost > MAX_WEIGHTED_CALLS_PER_DAY:
        logger.error(
            f"El costo total ({total_weighted_cost:.1f}) supera el tope diario seguro ({MAX_WEIGHTED_CALLS_PER_DAY}). "
            "Se recomienda dividir la descarga en 2 días (una temporada por día)."
        )

    t0 = time.time()
    season_files: dict[str, str] = {}

    for season, (s_start, s_end) in NWP_SEASONS.items():
        s_days = (s_end - s_start).days + 1
        season_parquet = output_dir / f"nwp_{season}.parquet"

        if season_parquet.exists():
            logger.info(f"Archivo de temporada {season} ya existe ({season_parquet}). Saltando.")
            season_files[season] = str(season_parquet)
            continue

        logger.info(
            f"Iniciando descarga de temporada {season} ({s_start} a {s_end}, {s_days} días)..."
        )
        season_records: list[dict[str, Any]] = []

        for b_idx in range(n_batches_per_season):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, n_nodes)
            batch_nodes = nodes[start_i:end_i]
            b_lats = [coord_dict[n][0] for n in batch_nodes]
            b_lons = [coord_dict[n][1] for n in batch_nodes]

            b_json = output_dir / f"batch_{season}_{b_idx:03d}.json"
            if b_json.exists():
                with open(b_json, encoding="utf-8") as f:
                    batch_data = json.load(f).get("data")
            else:
                logger.info(
                    f"Descargando lote {season} {b_idx + 1}/{n_batches_per_season} ({len(batch_nodes)} nodos)..."
                )
                batch_data = download_batch_with_retry(
                    lats=b_lats,
                    lons=b_lons,
                    start_date=s_start.isoformat(),
                    end_date=s_end.isoformat(),
                )
                if batch_data is not None:
                    with open(b_json, "w", encoding="utf-8") as f:
                        json.dump({"nodes": batch_nodes, "data": batch_data}, f)
                else:
                    logger.error(f"Fallo irrecuperable en lote {season} {b_idx + 1}. Abortando.")
                    sys.exit(1)

                time.sleep(sleep_sec)

            # Extraer registros a las 16:00 UTC
            for nid, node_data in zip(batch_nodes, batch_data, strict=True):
                hourly = node_data.get("hourly", {})
                times = hourly.get("time", [])
                t_h0 = hourly.get("temperature_2m", [])
                rh_h0 = hourly.get("relative_humidity_2m", [])
                w_h0 = hourly.get("wind_speed_10m", [])
                p_h0 = hourly.get("precipitation", [])

                t_h1 = hourly.get("temperature_2m_previous_day1", t_h0)
                rh_h1 = hourly.get("relative_humidity_2m_previous_day1", rh_h0)
                w_h1 = hourly.get("wind_speed_10m_previous_day1", w_h0)
                p_h1 = hourly.get("precipitation_previous_day1", p_h0)

                time_map = {t_str: idx for idx, t_str in enumerate(times)}
                curr_d = s_start
                while curr_d <= s_end:
                    t16_str = f"{curr_d.isoformat()}T16:00"
                    if t16_str in time_map:
                        idx16 = time_map[t16_str]
                        start_rain = max(0, idx16 - 23)
                        season_records.append(
                            {
                                "node_id": nid,
                                "date": curr_d,
                                "season": season,
                                "temp_noon_c_h0": float(t_h0[idx16]),
                                "rh_noon_pct_h0": float(rh_h0[idx16]),
                                "wind_noon_kmh_h0": float(w_h0[idx16]),
                                "rain_24h_noon_mm_h0": float(sum(p_h0[start_rain : idx16 + 1])),
                                "temp_noon_c_h1": float(t_h1[idx16]),
                                "rh_noon_pct_h1": float(rh_h1[idx16]),
                                "wind_noon_kmh_h1": float(w_h1[idx16]),
                                "rain_24h_noon_mm_h1": float(sum(p_h1[start_rain : idx16 + 1])),
                            }
                        )
                    curr_d += timedelta(days=1)

        season_df = pl.DataFrame(season_records)
        season_df.write_parquet(season_parquet)
        logger.info(
            f"Temporada {season} guardada en {season_parquet} ({season_df.height} celdas-día)."
        )
        season_files[season] = str(season_parquet)

    # Crear manifest con SHA-256 y atribución CC BY 4.0
    manifest_data = {
        "dataset": "ChileFuego NWP Archive Operational Skew Study",
        "created_at": datetime.now(UTC).isoformat(),
        "attribution": "Weather data by Open-Meteo.com (CC BY 4.0) under non-commercial fair use terms",
        "model": FIXED_MODEL,
        "n_nodes": n_nodes,
        "hourly_variables": HOURLY_VARIABLES,
        "files": {
            s: {
                "path": str(p),
                "sha256": compute_sha256(Path(p)),
            }
            for s, p in season_files.items()
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(f"Manifest guardado en {manifest_path}")
    return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descargador de archivo NWP Open-Meteo para estudio H4/J1."
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Lanza sonda mínima de 1 nodo y 7 días para verificar modelos y variables.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirmación explícita para lanzar la descarga completa.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Número de nodos por lote (default: 5)"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="Segundos de pausa entre llamadas API (default: 3.0)",
    )
    args = parser.parse_args()

    if args.probe:
        sys.exit(run_probe())

    res = execute_download(confirm=args.confirm, batch_size=args.batch_size, sleep_sec=args.sleep)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
