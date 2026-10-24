"""Descarga controlada y reanudable de pronósticos archivados NWP (J1 / Adenda).

PORQUÉ (Regla 5):
- Descarga pronósticos numéricos del tiempo (NWP) de Open-Meteo Previous Runs API
  exclusivamente para los meses de temporada activa (noviembre a abril) de 2024_25 y 2023_24 (363 días).
- Utiliza un modelo de pronóstico global fijo que cubre Chile continuamente desde antes de noviembre de 2023:
  NOAA GFS (gfs_seamless, disponible desde marzo de 2021; ECMWF IFS en Previous Runs solo inicia en enero de 2024).
- Adenda de Seguridad:
  1. Ante HTTP 429 ("Daily API request limit exceeded"): guarda progreso, imprime hora de reinicio (00:00 UTC en CLT) y hace EXIT inmediato.
  2. Caché por nodo y temporada en disco con validación de integridad; no vuelve a pedir nodos ya en disco.
  3. Alcance temporal estricto: nov-abr (363 días). Logs detallados de variables y modelo.
  4. Cálculo del costo ponderado (ubicaciones x días/14 x variables/10) exigiendo --confirm. Tope diario de 8.000 llamadas.
  5. Orden de descarga aleatorio con semilla fija (seed=42) para representatividad espacial en descargas parciales.
  6. Argumento --verify para auditar % de valores nulos por variable y rango de fechas por temporada.
  7. Interfaz WeatherProvider para desacoplar el cliente meteorológico de Open-Meteo.
  8. Procesa temporada 2024_25 PRIMERO, y luego 2023_24.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_nwp_archive")

OPEN_METEO_PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
FIXED_MODEL = "gfs_seamless"  # NOAA GFS cubre Chile globalmente desde marzo 2021

# Orden estricto: 2024_25 PRIMERO, luego 2023_24
NWP_SEASONS_ORDERED: list[tuple[str, date, date]] = [
    ("2024_25", date(2024, 11, 1), date(2025, 4, 30)),  # 181 días
    ("2023_24", date(2023, 11, 1), date(2024, 4, 30)),  # 182 días (año bisiesto 2024)
]

BASE_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
]

PREVIOUS_DAY1_VARIABLES = [
    "temperature_2m_previous_day1",
    "relative_humidity_2m_previous_day1",
    "wind_speed_10m_previous_day1",
    "precipitation_previous_day1",
]

# Configuración por defecto: 4 variables base continuas con 0% nulos.
ALL_REQUESTED_VARIABLES = BASE_VARIABLES
DAILY_BUDGET_CAP = 8000.0


# =====================================================================
# 7. INTERFAZ WEATHER PROVIDER (Abstracción de Fuentes Meteorológicas)
# =====================================================================


class WeatherProvider(ABC):
    """Interfaz abstracta proveedora de datos meteorológicos para desacoplar fuentes operacionales."""

    @abstractmethod
    def fetch_hourly(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: list[str],
        model: str,
    ) -> dict[str, Any] | None:
        """Obtiene series horarias para una coordenada y rango de fechas."""
        ...


class OpenMeteoNWPProvider(WeatherProvider):
    """Implementación de WeatherProvider sobre Open-Meteo Previous Runs API."""

    def __init__(self, base_url: str = OPEN_METEO_PREVIOUS_RUNS_URL):
        self.base_url = base_url

    def fetch_hourly(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        variables: list[str],
        model: str = FIXED_MODEL,
        max_retries: int = 2,
    ) -> dict[str, Any] | None:
        url = (
            f"{self.base_url}?"
            f"latitude={lat:.4f}&longitude={lon:.4f}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly={','.join(variables)}"
            f"&models={model}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-NWP/1.0"})

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode())
                    if isinstance(data, list):
                        return data[0] if data else None
                    return data
            except urllib.error.HTTPError as http_err:
                try:
                    err_body = http_err.read().decode("utf-8", errors="replace")
                except Exception:
                    err_body = ""
                reason = ""
                try:
                    err_json = json.loads(err_body)
                    reason = str(err_json.get("reason", ""))
                except Exception:
                    reason = err_body[:200]

                retry_after = http_err.headers.get("Retry-After")

                # REGLA 1: DETECCIÓN DE CUOTA DIARIA -> EXIT INMEDIATO
                if http_err.code == 429 and "Daily API request limit exceeded" in reason:
                    logger.critical(
                        "=================================================================="
                    )
                    logger.critical(
                        "EMERGENCIA CUOTA AGOTADA: HTTP 429 'Daily API request limit exceeded'"
                    )
                    logger.critical(
                        "El servidor de Open-Meteo ha bloqueado la IP por agotar el límite diario."
                    )
                    logger.critical("HORA DE REINICIO DE CUOTA: 00:00 UTC")
                    logger.critical(
                        "  -> En hora local de Chile continental estándar (UTC-4): 20:00 CLT"
                    )
                    logger.critical(
                        "  -> En hora local de Chile continental verano (UTC-3):  21:00 CLST"
                    )
                    logger.critical(
                        "Progreso guardado en disco. ABORTANDO EJECUCIÓN INMEDIATAMENTE."
                    )
                    logger.critical(
                        "=================================================================="
                    )
                    sys.exit(1)

                if http_err.code == 429:
                    wait_s = (
                        float(retry_after) + 1.0
                        if retry_after and retry_after.isdigit()
                        else 20.0 * (attempt + 1)
                    )
                    logger.warning(
                        f"HTTP 429 (Concurrencia/Rate limit) | Razón: '{reason}' | "
                        f"Pausando {wait_s}s antes de reintentar (intento {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_s)
                else:
                    logger.error(f"Error HTTP {http_err.code} ({http_err.reason}): '{reason}'")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(3)
            except Exception as exc:
                logger.warning(
                    f"Excepción de red ({exc}). Pausa de 3s (intento {attempt + 1}/{max_retries})..."
                )
                time.sleep(3)

        return None


# =====================================================================
# UTILIDADES Y CÁLCULO DE COSTOS
# =====================================================================


def compute_weighted_cost(n_locations: int, n_days: int, n_variables: int) -> float:
    """Calcula el costo ponderado según la fórmula oficial de Open-Meteo:

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


def build_cached_nodes_index(output_dir: Path, season: str) -> dict[str, Path]:
    """Indexa todos los nodos existentes en output_dir y en data/raw/open_meteo/."""
    found: dict[str, Path] = {}
    search_dirs = [output_dir, Path("data/raw/open_meteo")]
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        # 1. En output_dir/season/{node_id}.json
        s_season = sdir / season
        if s_season.exists():
            for f in s_season.glob("*.json"):
                if f.stat().st_size > 1000:
                    found[f.stem] = f
        # 2. En archivos batch_{season}_*.json (compatibilidad con lotes ya descargados)
        for f in sdir.glob(f"batch_{season}_*.json"):
            if f.stat().st_size > 1000:
                try:
                    with open(f, encoding="utf-8") as jf:
                        c = json.load(jf)
                    for n in c.get("nodes", []):
                        if n not in found:
                            found[n] = f
                except Exception:
                    pass
    return found


def get_cached_node_file(output_dir: Path, season: str, node_id: str) -> Path | None:
    """Busca si el nodo ya existe y es válido en output_dir o en data/raw/open_meteo/."""
    # 1. En output_dir/season/{node_id}.json
    primary = output_dir / season / f"{node_id}.json"
    if primary.exists() and primary.stat().st_size > 1000:
        return primary

    # 2. En data/raw/open_meteo/ (archivos batch_2023_24_XXX.json o similares)
    legacy_dir = Path("data/raw/open_meteo")
    if legacy_dir.exists():
        legacy_single = legacy_dir / f"{season}_{node_id}.json"
        if legacy_single.exists() and legacy_single.stat().st_size > 1000:
            return legacy_single
        # Buscar en archivos de lote
        for f in legacy_dir.glob(f"batch_{season}_*.json"):
            if f.stat().st_size > 1000:
                try:
                    with open(f, encoding="utf-8") as jf:
                        c = json.load(jf)
                    if node_id in c.get("nodes", []):
                        return f
                except Exception:
                    pass

    return None


def is_valid_node_json(path: Path, expected_min_hours: int = 3600) -> bool:
    """Verifica que el archivo JSON contenga variables horarias completas para las 4 variables base."""
    try:
        with open(path, encoding="utf-8") as f:
            content = json.load(f)
        # Soportar formato de lote {"nodes": [...], "data": [...]} o diccionario directo
        if "data" in content and isinstance(content["data"], list) and content["data"]:
            hourly = content["data"][0].get("hourly", {})
        else:
            hourly = content.get("hourly", {})

        times = hourly.get("time", [])
        if len(times) < expected_min_hours:
            return False

        # Verificar presencia y longitud de las 4 variables meteorológicas base
        for v in BASE_VARIABLES:
            vals = hourly.get(v, [])
            if len(vals) < expected_min_hours:
                return False
        return True
    except Exception:
        return False


# =====================================================================
# 6. AUDITORÍA DE ARCHIVOS (--verify)
# =====================================================================


def run_verification(output_dir: Path = Path("data/raw/nwp_archive")) -> None:
    """Inspecciona los archivos descargados y muestra el % de valores nulos por variable y rango de fechas."""
    logger.info("=== AUDITORÍA Y VERIFICACIÓN DE ARCHIVOS NWP DESCARGADOS (--verify) ===")
    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    nodes, _ = load_nuble_nodes(weather_path)
    n_total_nodes = len(nodes)

    search_dirs = [output_dir, Path("data/raw/open_meteo")]

    for season, s_start, s_end in NWP_SEASONS_ORDERED:
        logger.info(
            f"\n--- Auditoría Temporada {season} (Alcance Oficial: {s_start} a {s_end}) ---"
        )

        found_files: dict[str, Path] = {}
        for sdir in search_dirs:
            if not sdir.exists():
                continue
            # Buscar season/node.json
            s_season_dir = sdir / season
            if s_season_dir.exists():
                for f in s_season_dir.glob("*.json"):
                    nid = f.stem
                    if nid in nodes and nid not in found_files:
                        found_files[nid] = f
            # Buscar batch files que contengan el nodo
            for f in sdir.glob(f"batch_{season}_*.json"):
                try:
                    with open(f, encoding="utf-8") as jf:
                        c = json.load(jf)
                    b_nodes = c.get("nodes", [])
                    for bn in b_nodes:
                        if bn in nodes and bn not in found_files:
                            found_files[bn] = f
                except Exception:
                    pass

        n_found = len(found_files)
        pct_covered = (n_found / n_total_nodes) * 100.0
        logger.info(f"Nodos descargados: {n_found}/{n_total_nodes} ({pct_covered:.1f}%)")

        if n_found == 0:
            logger.warning(f"No hay archivos descargados para la temporada {season}.")
            continue

        # Analizar valores nulos por variable (audita tanto variables base como previous_day1)
        audit_vars = BASE_VARIABLES + PREVIOUS_DAY1_VARIABLES
        var_totals: dict[str, int] = {v: 0 for v in audit_vars}
        var_nulls: dict[str, int] = {v: 0 for v in audit_vars}
        min_date_found = "9999-99-99"
        max_date_found = "0000-00-00"

        for _, fpath in found_files.items():
            try:
                with open(fpath, encoding="utf-8") as jf:
                    c = json.load(jf)
                node_data = c["data"][0] if ("data" in c and c["data"]) else c
                hourly = node_data.get("hourly", {})
                times = hourly.get("time", [])
                if times:
                    min_date_found = min(min_date_found, times[0][:10])
                    max_date_found = max(max_date_found, times[-1][:10])

                for var in audit_vars:
                    if var in hourly:
                        vals = hourly.get(var, [])
                        var_totals[var] += len(vals)
                        var_nulls[var] += sum(1 for x in vals if x is None)
                    else:
                        var_totals[var] += len(times)
                        var_nulls[var] += len(times)
            except Exception:
                pass

        logger.info(f"Rango temporal observado en archivos: {min_date_found} a {max_date_found}")
        logger.info("Porcentaje de valores nulos por variable:")
        for var in audit_vars:
            tot = var_totals[var]
            nulls = var_nulls[var]
            pct_null = (nulls / tot * 100.0) if tot > 0 else 0.0
            status = "OK" if pct_null == 0.0 else f"ALERTA NULOS ({pct_null:.2f}%)"
            logger.info(f"  * {var:36s}: {pct_null:6.2f}% nulos ({nulls}/{tot}) [{status}]")

        missing_nodes = [n for n in nodes if n not in found_files]
        if missing_nodes:
            logger.info(
                f"Nodos faltantes en {season} ({len(missing_nodes)}): {missing_nodes[:5]} ... (usa --list-missing)"
            )


def list_missing_nodes(output_dir: Path = Path("data/raw/nwp_archive")) -> None:
    """Imprime el listado exacto de nodos pendientes por temporada."""
    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    nodes, _ = load_nuble_nodes(weather_path)

    for season, _, _ in NWP_SEASONS_ORDERED:
        season_dir = output_dir / season
        existing = set()
        if season_dir.exists():
            for f in season_dir.glob("*.json"):
                existing.add(f.stem)
        legacy_dir = Path("data/raw/open_meteo")
        if legacy_dir.exists():
            for f in legacy_dir.glob(f"batch_{season}_*.json"):
                try:
                    with open(f, encoding="utf-8") as jf:
                        c = json.load(jf)
                    for n in c.get("nodes", []):
                        existing.add(n)
                except Exception:
                    pass

        missing = [n for n in nodes if n not in existing]
        print(
            f"\n=== NODOS PENDIENTES EN TEMPORADA {season} ({len(missing)} faltantes de {len(nodes)}) ==="
        )
        for m in missing:
            print(m)


# =====================================================================
# EJECUCIÓN PRINCIPAL DE DESCARGA
# =====================================================================


def execute_download(
    confirm: bool,
    sleep_sec: float = 2.0,
    output_dir: Path = Path("data/raw/nwp_archive"),
    selected_season: str = "all",
    variables_mode: str = "base",
) -> dict[str, Any]:
    """Ejecuta la descarga con orden aleatorio, pausa controlada y salida inmediata ante 429."""
    weather_path = Path("data/processed/era5_land_extended_weather_nodes.parquet")
    nodes, coord_dict = load_nuble_nodes(weather_path)
    n_nodes = len(nodes)

    if variables_mode == "previous_day1":
        requested_vars = PREVIOUS_DAY1_VARIABLES
    elif variables_mode == "all":
        requested_vars = BASE_VARIABLES + PREVIOUS_DAY1_VARIABLES
    else:
        requested_vars = BASE_VARIABLES

    n_vars = len(requested_vars)

    if selected_season != "all":
        seasons_to_run = [s for s in NWP_SEASONS_ORDERED if s[0] == selected_season]
    else:
        seasons_to_run = NWP_SEASONS_ORDERED

    total_days = sum((e - s).days + 1 for _, s, e in seasons_to_run)
    total_weighted_cost = compute_weighted_cost(n_nodes, total_days, n_vars)

    logger.info("==================================================================")
    logger.info("PLAN DE DESCARGA NWP ARCHIVE (ADENDA FASE 3A / H4)")
    logger.info(f"Modelo seleccionado: {FIXED_MODEL} (cobertura continua Chile desde marzo 2021)")
    logger.info(f"Modo de variables: '{variables_mode}' ({n_vars} variables): {requested_vars}")
    logger.info("Alcance temporal estricto (SOLO noviembre a abril):")
    for s_name, s_start, s_end in seasons_to_run:
        s_days = (s_end - s_start).days + 1
        logger.info(f"  * Temporada {s_name}: {s_start} a {s_end} ({s_days} días)")
    logger.info(f"Dominio: {n_nodes} nodos ERA5-Land en Ñuble")
    logger.info(f"Cálculo de costo ponderado: {n_nodes} x ({total_days}/14) x ({n_vars}/10)")
    logger.info(f"COSTO TOTAL PONDERADO: ~{total_weighted_cost:.1f} llamadas")
    logger.info(f"Tope de seguridad diario: {DAILY_BUDGET_CAP:.0f} llamadas")
    logger.info("==================================================================")

    if not confirm:
        logger.warning(
            "DESCARGA DETENIDA: Se requiere la bandera '--confirm' para iniciar peticiones HTTP. "
            "Usa primero '--verify' para auditar datos locales, o '--confirm' para lanzar la descarga."
        )
        return {
            "status": "STOPPED_AWAITING_CONFIRMATION",
            "total_weighted_cost": total_weighted_cost,
            "total_days": total_days,
            "n_nodes": n_nodes,
            "variables_mode": variables_mode,
            "selected_season": selected_season,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    provider = OpenMeteoNWPProvider()

    # REGLA 5: Orden aleatorio con semilla fija 42
    rng = random.Random(42)
    shuffled_nodes = list(nodes)
    rng.shuffle(shuffled_nodes)
    logger.info(
        f"Orden de descarga barajado aleatoriamente (seed=42). Primeros 3 nodos: {shuffled_nodes[:3]}"
    )

    accumulated_cost = 0.0

    # REGLA 8: 2024_25 PRIMERO, luego 2023_24
    for season, s_start, s_end in seasons_to_run:
        season_dir = output_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        s_days = (s_end - s_start).days + 1
        cost_per_node = compute_weighted_cost(1, s_days, n_vars)

        logger.info(f"\n>>> INICIANDO PROCESAMIENTO TEMPORADA {season} ({s_start} a {s_end}) <<<")
        cached_nodes_map = build_cached_nodes_index(output_dir, season)

        for idx, nid in enumerate(shuffled_nodes, start=1):
            node_file = season_dir / f"{nid}.json"

            # REGLA 2: No volver a pedir por red si ya existe y es válido
            cached_file = cached_nodes_map.get(nid)
            if (
                variables_mode == "base"
                and cached_file
                and is_valid_node_json(cached_file, expected_min_hours=s_days * 20)
            ):
                logger.info(
                    f"[{season}] ({idx}/{n_nodes}) Nodo {nid} ya existe en caché ({cached_file.name}). Saltando."
                )
                continue

            # Verificar presupuesto diario
            if accumulated_cost + cost_per_node > DAILY_BUDGET_CAP:
                logger.warning(
                    f"Tope diario de llamadas ponderadas alcanzado ({accumulated_cost:.1f} >= {DAILY_BUDGET_CAP}). "
                    "Deteniendo descarga ordenada. Reanuda mañana tras el reinicio de las 00:00 UTC."
                )
                return {"status": "DAILY_BUDGET_REACHED", "accumulated_cost": accumulated_cost}

            lat, lon = coord_dict[nid]
            logger.info(
                f"[{season}] ({idx}/{n_nodes}) Descargando nodo {nid} (lat={lat:.4f}, lon={lon:.4f}) | "
                f"Costo estimado: {cost_per_node:.1f} llamadas..."
            )

            res = provider.fetch_hourly(
                lat=lat,
                lon=lon,
                start_date=s_start.isoformat(),
                end_date=s_end.isoformat(),
                variables=requested_vars,
                model=FIXED_MODEL,
            )

            if res is not None:
                # Si el archivo del nodo ya existe, fusionar variables para no pisar las existentes
                if node_file.exists():
                    try:
                        with open(node_file, encoding="utf-8") as jf:
                            existing_content = json.load(jf)
                        if "data" in existing_content and existing_content["data"]:
                            ex_hourly = existing_content["data"][0].get("hourly", {})
                            new_hourly = res.get("hourly", {})
                            for k, v in new_hourly.items():
                                if k != "time":
                                    ex_hourly[k] = v
                            res = existing_content["data"][0]
                    except Exception:
                        pass
                with open(node_file, "w", encoding="utf-8") as f:
                    json.dump({"node_id": nid, "data": [res]}, f)
                accumulated_cost += cost_per_node
                logger.info(
                    f"Nodo {nid} guardado en {node_file.name} (acumulado: {accumulated_cost:.1f} llamadas)."
                )
            else:
                logger.error(f"Fallo al descargar nodo {nid}. Continuando...")

            time.sleep(sleep_sec)

    logger.info("Descarga completada para todas las temporadas.")
    return {"status": "COMPLETED", "accumulated_cost": accumulated_cost}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descargador robusto de archivo NWP Open-Meteo (Adenda Fase 3A / H4)."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Audita los archivos descargados: % de nulos y rango de fechas.",
    )
    parser.add_argument(
        "--list-missing", action="store_true", help="Lista los nodos faltantes por temporada."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Autorización explícita para lanzar la descarga HTTP.",
    )
    parser.add_argument(
        "--season",
        type=str,
        default="all",
        choices=["all", "2024_25", "2023_24"],
        help="Temporada a descargar (default: all).",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default="base",
        choices=["base", "previous_day1", "all"],
        help="Conjunto de variables (default: base).",
    )
    parser.add_argument(
        "--sleep", type=float, default=2.0, help="Segundos de pausa entre nodos (default: 2.0)."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw/nwp_archive", help="Directorio de destino."
    )
    args = parser.parse_args()

    if args.verify:
        run_verification(output_dir=Path(args.output_dir))
        sys.exit(0)

    if args.list_missing:
        list_missing_nodes(output_dir=Path(args.output_dir))
        sys.exit(0)

    res = execute_download(
        confirm=args.confirm,
        sleep_sec=args.sleep,
        output_dir=Path(args.output_dir),
        selected_season=args.season,
        variables_mode=args.variables,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
