"""Ingesta de focos satelitales NASA FIRMS VIIRS (S-NPP, NOAA-20, NOAA-21) y archivo histórico.

Características (Fase 1.5 - T1 y Regla 4/5):
- PORQUÉ: Descartar columnas del CSV original (`type`, `daynight`, `bright_ti4`/`bright_ti5`,
  `scan`, `track`, `instrument`, `version`) impedía filtrar falsos positivos por `type==0` y
  auditar pasadas nocturnas (`daynight`). Además, depender del nombre del archivo CSV en vez
  del valor fila a fila de `satellite` era propenso a errores de etiquetado de sensor.
- CÓMO:
  1. `parse_firms_csv_bytes` preserva el 100% de las columnas originales del CSV y normaliza
     `bright_ti4` (desde `bright_ti4` o `brightness` en descargas de archivo) y `bright_ti5`
     (desde `bright_ti5` o `bright_t31`).
  2. La asignación de `satellite` se resuelve fila a fila leyendo la columna `satellite`:
     - `{'N', 'SNPP', 'SUOMI NPP', 'VIIRS_SNPP_SP', 'VIIRS_SNPP_NRT'}` -> `'VIIRS_SNPP_SP'`
     - `{'1', 'N20', 'J1', 'NOAA-20', 'JPSS-1', 'VIIRS_NOAA20_SP', ...}` -> `'VIIRS_NOAA20_SP'`
     - `{'2', 'N21', 'J2', 'NOAA-21', 'JPSS-2', 'VIIRS_NOAA21_SP', ...}` -> `'VIIRS_NOAA21_SP'`
     Si aparece cualquier código desconocido o nulo, lanza `ValueError` inmediato.
  3. `ingest_firms_archive_csv` deja intacto el CSV original y actualiza `manifest.json` con el
     hash SHA-256 de cada CSV procesado.
  4. `write_partitioned_firms_parquet` fusiona de manera idempotente las particiones de API y
     archivo histórico (`diagonal_relaxed`), priorizando las filas que contienen metadatos
     completos (`type`, `daynight`, `bright_ti4`) y eliminando duplicados sobre la clave natural
     `(latitude, longitude, acq_date, acq_time, satellite)`.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import httpx
import polars as pl

from chilefuego.config import AppConfig
from chilefuego.logging_utils import (
    atomic_write_parquet,
    compute_file_sha256,
    get_logger,
    log_event,
)
from chilefuego.schemas import validate_raw_firms

logger = get_logger(__name__)

SATELLITE_CODE_MAPPING: dict[str, str] = {
    "N": "VIIRS_SNPP_SP",
    "SNPP": "VIIRS_SNPP_SP",
    "SUOMI NPP": "VIIRS_SNPP_SP",
    "SUOMI-NPP": "VIIRS_SNPP_SP",
    "VIIRS_SNPP_SP": "VIIRS_SNPP_SP",
    "VIIRS_SNPP_NRT": "VIIRS_SNPP_SP",
    "1": "VIIRS_NOAA20_SP",
    "N20": "VIIRS_NOAA20_SP",
    "J1": "VIIRS_NOAA20_SP",
    "NOAA-20": "VIIRS_NOAA20_SP",
    "NOAA20": "VIIRS_NOAA20_SP",
    "JPSS-1": "VIIRS_NOAA20_SP",
    "VIIRS_NOAA20_SP": "VIIRS_NOAA20_SP",
    "VIIRS_NOAA20_NRT": "VIIRS_NOAA20_SP",
    "2": "VIIRS_NOAA21_SP",
    "N21": "VIIRS_NOAA21_SP",
    "J2": "VIIRS_NOAA21_SP",
    "NOAA-21": "VIIRS_NOAA21_SP",
    "NOAA21": "VIIRS_NOAA21_SP",
    "JPSS-2": "VIIRS_NOAA21_SP",
    "VIIRS_NOAA21_SP": "VIIRS_NOAA21_SP",
    "VIIRS_NOAA21_NRT": "VIIRS_NOAA21_SP",
}

STANDARD_FIRMS_COLUMNS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "bright_ti4",
    "scan",
    "track",
    "acq_date",
    "acq_time",
    "satellite",
    "instrument",
    "confidence",
    "version",
    "bright_ti5",
    "frp",
    "daynight",
    "type",
)


def get_firms_api_key(env_var_name: str = "FIRMS_MAP_KEY") -> str:
    """Obtiene la MAP_KEY desde variables de entorno o archivo .env local."""
    api_key = os.environ.get(env_var_name, "").strip()
    if not api_key:
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                clean = line.strip()
                if clean.startswith(f"{env_var_name}="):
                    api_key = clean.split("=", 1)[1].strip().strip("'\"")
                    break
    if not api_key or api_key == "tu_map_key_de_nasa_firms_aqui":
        raise RuntimeError(
            f"Variable de entorno {env_var_name} no configurada. "
            "Defínela en el entorno o en .env antes de ejecutar la descarga FIRMS."
        )
    return api_key


def generate_date_windows(
    start_date: date,
    end_date: date,
    window_days: int = 5,
) -> list[tuple[date, int]]:
    """Divide un rango [start_date, end_date] en ventanas de hasta `window_days` días (1..5)."""
    if start_date > end_date:
        raise ValueError("start_date debe ser <= end_date")
    if window_days < 1 or window_days > 5:
        raise ValueError("NASA FIRMS Area API requiere window_days entre 1 y 5")

    windows: list[tuple[date, int]] = []
    cursor = start_date
    while cursor <= end_date:
        remaining = (end_date - cursor).days + 1
        span = min(window_days, remaining)
        windows.append((cursor, span))
        cursor += timedelta(days=span)
    return windows


def map_and_validate_satellite_column(df: pl.DataFrame) -> pl.DataFrame:
    """Mapea la columna `satellite` fila a fila y falla ante cualquier código desconocido (T1)."""
    if "satellite" not in df.columns:
        raise ValueError("El CSV de FIRMS no contiene la columna obligatoria 'satellite'.")

    cleaned = df.with_columns(
        pl.col("satellite").cast(pl.String).str.strip_chars().str.to_uppercase().alias("_sat_raw")
    )
    unique_raw = cleaned.get_column("_sat_raw").unique().to_list()
    unknown = [
        code for code in unique_raw if code is None or str(code) not in SATELLITE_CODE_MAPPING
    ]
    if unknown:
        raise ValueError(
            f"Código(s) de satélite desconocido(s) en columna 'satellite': {unknown}. "
            f"Códigos soportados: {sorted(SATELLITE_CODE_MAPPING.keys())}"
        )

    return cleaned.with_columns(
        pl.col("_sat_raw").replace_strict(SATELLITE_CODE_MAPPING).alias("satellite")
    ).drop("_sat_raw")


def empty_raw_firms_df() -> pl.DataFrame:
    """Retorna un DataFrame vacío que cumple el esquema RawFirmsSchema completo (T1)."""
    return pl.DataFrame(
        schema={
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "bright_ti4": pl.Float64,
            "scan": pl.Float64,
            "track": pl.Float64,
            "acq_date": pl.String,
            "acq_time": pl.String,
            "satellite": pl.String,
            "instrument": pl.String,
            "confidence": pl.String,
            "version": pl.String,
            "bright_ti5": pl.Float64,
            "frp": pl.Float64,
            "daynight": pl.String,
            "type": pl.Int64,
        }
    )


def parse_firms_csv_bytes(csv_bytes: bytes, default_source: str | None = None) -> pl.DataFrame:
    """Parsea bytes CSV de NASA FIRMS conservando todas las columnas originales (T1).

    Nota: `default_source` se mantiene únicamente por compatibilidad de firma con llamadas
    antiguas de la API, pero la asignación de `satellite` depende estrictamente del valor
    de la columna `satellite` en cada fila y falla ante códigos desconocidos.
    """
    _ = default_source
    text = csv_bytes.decode("utf-8", errors="replace").strip()
    if not text:
        return validate_raw_firms(empty_raw_firms_df())

    lower_head = text[:200].lower()
    if (
        "invalid map_key" in lower_head
        or "exceeded" in lower_head
        or "invalid " in lower_head
        or "error" in lower_head
    ):
        raise ValueError(f"Respuesta de error desde NASA FIRMS: {text[:160]}")

    lines = text.splitlines()
    if len(lines) < 2:
        return validate_raw_firms(empty_raw_firms_df())

    raw_df = pl.read_csv(
        io.BytesIO(csv_bytes),
        infer_schema_length=2000,
        ignore_errors=False,
        schema_overrides={
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "bright_ti4": pl.Float64,
            "brightness": pl.Float64,
            "scan": pl.Float64,
            "track": pl.Float64,
            "acq_date": pl.String,
            "acq_time": pl.String,
            "satellite": pl.String,
            "instrument": pl.String,
            "confidence": pl.String,
            "version": pl.String,
            "bright_ti5": pl.Float64,
            "bright_t31": pl.Float64,
            "frp": pl.Float64,
            "daynight": pl.String,
            "type": pl.Int64,
        },
    )

    if raw_df.is_empty():
        return validate_raw_firms(empty_raw_firms_df())

    # Mapear y validar estrictamente la columna satellite fila a fila (sin usar nombre de archivo)
    mapped_df = map_and_validate_satellite_column(raw_df)

    # Normalizar bright_ti4 / bright_ti5 preservando también brightness / bright_t31 del CSV
    if "bright_ti4" not in mapped_df.columns:
        if "brightness" in mapped_df.columns:
            mapped_df = mapped_df.with_columns(
                pl.col("brightness").cast(pl.Float64).alias("bright_ti4")
            )
        else:
            mapped_df = mapped_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("bright_ti4"))

    if "bright_ti5" not in mapped_df.columns:
        if "bright_t31" in mapped_df.columns:
            mapped_df = mapped_df.with_columns(
                pl.col("bright_t31").cast(pl.Float64).alias("bright_ti5")
            )
        else:
            mapped_df = mapped_df.with_columns(pl.lit(None, dtype=pl.Float64).alias("bright_ti5"))

    optional_defaults: list[tuple[str, pl.DataType]] = [
        ("scan", pl.Float64()),
        ("track", pl.Float64()),
        ("instrument", pl.String()),
        ("version", pl.String()),
        ("daynight", pl.String()),
        ("type", pl.Int64()),
    ]
    for col_name, dtype in optional_defaults:
        if col_name not in mapped_df.columns:
            mapped_df = mapped_df.with_columns(pl.lit(None, dtype=dtype).alias(col_name))
        else:
            mapped_df = mapped_df.with_columns(pl.col(col_name).cast(dtype))

    if "frp" not in mapped_df.columns:
        mapped_df = mapped_df.with_columns(pl.lit(0.0, dtype=pl.Float64).alias("frp"))
    if "confidence" not in mapped_df.columns:
        mapped_df = mapped_df.with_columns(pl.lit("nominal", dtype=pl.String).alias("confidence"))

    # Conservar todas las columnas estándar en orden + cualquier columna extra del CSV original
    extra_cols = [c for c in mapped_df.columns if c not in STANDARD_FIRMS_COLUMNS]
    ordered_cols = list(STANDARD_FIRMS_COLUMNS) + extra_cols

    normalized = (
        mapped_df.select(ordered_cols)
        .drop_nulls(subset=["latitude", "longitude", "acq_date", "acq_time"])
        .with_columns(
            pl.col("acq_time").cast(pl.String).str.strip_chars().str.zfill(4),
            pl.col("confidence").cast(pl.String).str.strip_chars().str.to_lowercase(),
            pl.col("frp").cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0),
        )
    )
    return validate_raw_firms(normalized)


def attach_utc_and_chile_timestamps(
    df: pl.DataFrame,
    chile_tz: str = "America/Santiago",
) -> pl.DataFrame:
    """Convierte `acq_date` (YYYY-MM-DD) y `acq_time` (HHMM) de UTC a `America/Santiago`."""
    if df.is_empty():
        return df.with_columns(
            pl.Series("acq_ts_utc", [], dtype=pl.Datetime("us", "UTC")),
            pl.Series("local_date", [], dtype=pl.Date),
        )

    padded_time = pl.col("acq_time").cast(pl.String).str.strip_chars().str.zfill(4)
    iso_str = pl.concat_str([pl.col("acq_date"), pl.lit(" "), padded_time])

    return (
        df.with_columns(padded_time.alias("acq_time"))
        .with_columns(
            iso_str.str.to_datetime("%Y-%m-%d %H%M", time_unit="us", time_zone="UTC").alias(
                "acq_ts_utc"
            )
        )
        .with_columns(
            pl.col("acq_ts_utc").dt.convert_time_zone(chile_tz).dt.date().alias("local_date")
        )
    )


def fetch_firms_window_with_retry(
    client: httpx.Client,
    url: str,
    source: str,
    max_retries: int = 4,
    backoff_base_seconds: float = 1.5,
) -> pl.DataFrame:
    """Descarga una ventana temporal desde NASA FIRMS con reintentos y backoff exponencial."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.get(url)
            if response.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"HTTP transitorio {response.status_code}",
                    request=response.request,
                    response=response,
                )
            response.raise_for_status()
            return parse_firms_csv_bytes(response.content, default_source=source)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_s = backoff_base_seconds * (2 ** (attempt - 1))
            log_event(
                logger,
                "firms_retry",
                attempt=attempt,
                max_retries=max_retries,
                sleep_seconds=sleep_s,
                error=str(exc),
            )
            time.sleep(sleep_s)

    msg = f"Fallo definitivo descargando ventana FIRMS tras {max_retries} intentos"
    raise RuntimeError(msg) from last_error


def _deduplicate_merged_firms_partition(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplica registros sobre `(latitude, longitude, acq_date, acq_time, satellite)`.

    Si una misma detección existe tanto desde la descarga API previa (sin columna `type`) como
    desde el archivo histórico CSV (con `type`, `daynight`, `bright_ti4`), prioriza la fila con
    metadatos completos de archivo de forma determinista e idempotente.
    """
    if df.is_empty():
        return df

    has_type_expr = pl.col("type").is_not_null() if "type" in df.columns else pl.lit(False)
    has_ti4_expr = (
        pl.col("bright_ti4").is_not_null() if "bright_ti4" in df.columns else pl.lit(False)
    )

    return (
        df.with_columns(
            has_type_expr.alias("_has_type"),
            has_ti4_expr.alias("_has_ti4"),
        )
        .sort(
            by=[
                "acq_date",
                "acq_time",
                "latitude",
                "longitude",
                "satellite",
                "_has_type",
                "_has_ti4",
            ],
            descending=[False, False, False, False, False, True, True],
        )
        .unique(
            subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"],
            keep="first",
            maintain_order=True,
        )
        .drop(["_has_type", "_has_ti4"])
    )


def write_partitioned_firms_parquet(
    df: pl.DataFrame,
    raw_dir: str | Path,
    chile_tz: str = "America/Santiago",
) -> list[tuple[Path, str]]:
    """Guarda detecciones FIRMS particionadas por `year=YYYY/month=MM` con escritura atómica.

    Idempotente: fusiona esquemas con `diagonal_relaxed` y deduplica registros priorizando
    aquellos que contienen todas las columnas de archivo (`type`, `daynight`, `bright_ti4`).
    """
    if df.is_empty():
        return []

    enriched = attach_utc_and_chile_timestamps(df, chile_tz=chile_tz).with_columns(
        pl.col("local_date").dt.year().alias("year"),
        pl.col("local_date").dt.month().alias("month"),
    )

    base = Path(raw_dir) / "firms"
    written_partitions: list[tuple[Path, str]] = []

    for (year_val, month_val), part_df in enriched.group_by(["year", "month"], maintain_order=True):
        y_dir = f"year={int(year_val):04d}"
        m_dir = f"month={int(month_val):02d}"
        part_path = base / y_dir / m_dir / "data.parquet"
        clean_part = part_df.drop(["year", "month", "acq_ts_utc", "local_date"])

        if part_path.exists():
            existing = pl.read_parquet(part_path)
            combined = pl.concat([existing, clean_part], how="diagonal_relaxed")
            clean_part = _deduplicate_merged_firms_partition(combined)
        else:
            clean_part = _deduplicate_merged_firms_partition(clean_part)

        # Ordenar columnas dejando primero STANDARD_FIRMS_COLUMNS
        std_present = [c for c in STANDARD_FIRMS_COLUMNS if c in clean_part.columns]
        extra_present = [c for c in clean_part.columns if c not in STANDARD_FIRMS_COLUMNS]
        clean_part = clean_part.select(std_present + extra_present)

        sha256_hex = atomic_write_parquet(clean_part, part_path)
        written_partitions.append((part_path, sha256_hex))
        log_event(
            logger,
            "partition_written",
            path=str(part_path),
            rows=clean_part.height,
            sha256=sha256_hex,
        )

    return written_partitions


def update_firms_archive_manifest(
    manifest_path: str | Path,
    csv_path: Path,
    csv_sha256: str,
    df: pl.DataFrame,
) -> dict[str, object]:
    """Actualiza de forma idempotente `manifest.json` con el SHA-256 de cada CSV (T1)."""
    m_path = Path(manifest_path)
    m_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_data: dict[str, object] = {"files": {}}
    if m_path.exists():
        try:
            loaded = json.loads(m_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("files"), dict):
                manifest_data = loaded
        except Exception:
            manifest_data = {"files": {}}

    files_dict = manifest_data.get("files")
    if not isinstance(files_dict, dict):
        files_dict = {}
        manifest_data["files"] = files_dict

    entry: dict[str, object] = {
        "filename": csv_path.name,
        "sha256": csv_sha256,
        "size_bytes": csv_path.stat().st_size,
        "n_rows": df.height,
        "min_acq_date": str(df.get_column("acq_date").min()) if not df.is_empty() else None,
        "max_acq_date": str(df.get_column("acq_date").max()) if not df.is_empty() else None,
        "satellites": sorted(df.get_column("satellite").unique().to_list())
        if not df.is_empty()
        else [],
        "columns": list(df.columns),
        "ingested_at_utc": datetime.now(tz=UTC).isoformat(),
    }
    files_dict[csv_path.name] = entry
    m_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest_data


def ingest_firms_archive_csv(
    csv_path: str | Path,
    raw_dir: str | Path,
    default_source: str | None = None,
    chile_tz: str = "America/Santiago",
    manifest_path: str | Path | None = None,
) -> list[tuple[Path, str]]:
    """Ingiere un archivo CSV de descarga histórica FIRMS a Parquet sin modificar el CSV original.

    Calcula el SHA-256 del CSV, actualiza `manifest.json` y si el CSV con idéntico SHA-256 ya
    fue ingerido y todas las particiones existen, aún garantiza consistencia idempotente.
    """
    _ = default_source
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de descarga histórica FIRMS no encontrado: {path}")

    csv_sha256 = compute_file_sha256(path)
    df = parse_firms_csv_bytes(path.read_bytes())

    target_manifest = (
        Path(manifest_path) if manifest_path is not None else path.parent / "manifest.json"
    )
    update_firms_archive_manifest(
        manifest_path=target_manifest,
        csv_path=path,
        csv_sha256=csv_sha256,
        df=df,
    )

    return write_partitioned_firms_parquet(df, raw_dir=raw_dir, chile_tz=chile_tz)


def scan_raw_firms_lazy(raw_dir: str | Path) -> pl.LazyFrame:
    """Lee de forma perezosa (LazyFrame) todas las particiones Parquet bajo `raw_dir/firms`."""
    firms_root = Path(raw_dir) / "firms"
    parquet_files = sorted(firms_root.glob("year=*/month=*/*.parquet"))
    if not parquet_files:
        return empty_raw_firms_df().lazy()
    return pl.scan_parquet(
        [str(p) for p in parquet_files],
        missing_columns="insert",
        extra_columns="ignore",
    )


def download_firms_range_to_parquet(
    config: AppConfig,
    start_date: date,
    end_date: date,
    sources: list[str] | None = None,
    mode: Literal["history", "live"] = "history",
) -> list[tuple[Path, str]]:
    """Descarga un rango de fechas desde NASA FIRMS y lo persiste particionado en Parquet."""
    api_key = get_firms_api_key(config.firms.env_api_key_var)
    bbox = config.geospatial.bbox_wgs84
    bbox_str = f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"
    if sources is not None:
        active_sources = sources
    elif mode == "live":
        active_sources = config.firms.live_sources
    else:
        active_sources = config.firms.history_sources

    windows = generate_date_windows(start_date, end_date, config.firms.window_days)

    collected_frames: list[pl.DataFrame] = []
    with httpx.Client(timeout=config.firms.timeout_seconds) as client:
        for source in active_sources:
            for win_start, span in windows:
                url = (
                    f"{config.firms.base_url}/{api_key}/{source}/"
                    f"{bbox_str}/{span}/{win_start.isoformat()}"
                )
                df_win = fetch_firms_window_with_retry(
                    client=client,
                    url=url,
                    source=source,
                    max_retries=config.firms.max_retries,
                    backoff_base_seconds=config.firms.backoff_base_seconds,
                )
                if not df_win.is_empty():
                    collected_frames.append(df_win)

    if not collected_frames:
        return []

    merged = _deduplicate_merged_firms_partition(
        pl.concat(collected_frames, how="diagonal_relaxed")
    )
    return write_partitioned_firms_parquet(
        merged,
        raw_dir=config.paths.raw_dir,
        chile_tz=config.project.timezone,
    )
