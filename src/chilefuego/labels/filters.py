"""Máscaras de falsos positivos ANTES de etiquetar igniciones (Fase 1.3, C3 y Fase 1.5 - T4).

Incluye cuatro etapas puras y auditables:
1. Filtro por nivel de confianza VIIRS (`confidence` en `{'n', 'h', 'nominal', 'high'}`).
2. Filtro por tipo de anomalía estándar NASA FIRMS (`type == 0`, fuego de vegetación presunto),
   descartando `type == 1` (volcán activo), `type == 2` (otra fuente terrestre estática) y
   `type == 3` (offshore/marítimo) cuando la columna `type` está presente (T4).
3. Exclusión espacial de zonas volcánicas y de nieve/hielo permanente (`radius_m = 4000` sobre
   el cráter activo de Nevados de Chillán `-36.863, -71.377`).
4. Detección y filtrado de fuentes térmicas estacionarias (chimeneas industriales, celulosas
   o aserraderos con actividad persistente multimes o invernal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import Polygon

from chilefuego.config import ExclusionZoneConfig, FilteringConfig
from chilefuego.schemas import validate_filtered_firms

_TO_UTM19S = Transformer.from_crs("EPSG:4326", "EPSG:32719", always_xy=True)


@dataclass(frozen=True)
class FilterFunnelStats:
    """Métricas de trazabilidad del embudo de filtrado."""

    n_input_in_bbox: int
    n_after_confidence_mask: int
    n_after_type_mask: int
    n_after_volcanic_snow_mask: int
    n_after_stationary_mask: int


def evaluate_zone_exclusion_mask(
    df: pl.DataFrame,
    zone: ExclusionZoneConfig,
) -> np.ndarray:
    """Retorna máscara booleana `True` para cada fila que cae DENTRO de `zone` (a excluir)."""
    if df.is_empty():
        return np.zeros(0, dtype=bool)

    lons = df.get_column("longitude").to_numpy()
    lats = df.get_column("latitude").to_numpy()

    if zone.center_lon is not None and zone.center_lat is not None and zone.radius_m is not None:
        if "x_utm" in df.columns and "y_utm" in df.columns:
            xs = df.get_column("x_utm").to_numpy()
            ys = df.get_column("y_utm").to_numpy()
        else:
            x_proj, y_proj = _TO_UTM19S.transform(lons, lats)
            xs = np.asarray(x_proj, dtype=np.float64)
            ys = np.asarray(y_proj, dtype=np.float64)

        cx, cy = _TO_UTM19S.transform(zone.center_lon, zone.center_lat)
        dist_m = np.hypot(xs - float(cx), ys - float(cy))
        return np.asarray(dist_m <= float(zone.radius_m), dtype=bool)

    if zone.polygon_lon_lat is not None:
        poly = Polygon(zone.polygon_lon_lat)
        return np.asarray(contains_xy(poly.buffer(1e-9), lons, lats), dtype=bool)

    return np.zeros(df.height, dtype=bool)


def filter_by_confidence(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> pl.DataFrame:
    """Conserva únicamente detecciones cuyo nivel de confianza esté permitido en configuración."""
    if df.is_empty():
        return df

    allowed = {c.strip().lower() for c in filtering_cfg.allowed_confidences}
    return df.filter(
        pl.col("confidence").cast(pl.String).str.strip_chars().str.to_lowercase().is_in(allowed)
    )


def filter_by_type(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> pl.DataFrame:
    """Filtra por `type` de NASA FIRMS (conservando `type == 0` o nulo si proviene de NRT) (T4)."""
    if df.is_empty() or "type" not in df.columns:
        return df

    allowed_types = set(filtering_cfg.allowed_types)
    return df.filter(pl.col("type").is_null() | pl.col("type").cast(pl.Int64).is_in(allowed_types))


def filter_volcanic_and_snow_zones(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> pl.DataFrame:
    """Excluye focos ubicados dentro de las zonas volcánicas o de nieve eterna."""
    if df.is_empty() or not filtering_cfg.exclusion_zones:
        return df

    keep_mask = np.ones(df.height, dtype=bool)
    for zone in filtering_cfg.exclusion_zones:
        inside_zone = evaluate_zone_exclusion_mask(df, zone)
        keep_mask &= ~inside_zone

    return df.filter(pl.Series("_keep_volcanic", keep_mask))


def summarize_cell_stationarity(
    df: pl.DataFrame,
    winter_months: list[int] | None = None,
) -> pl.DataFrame:
    """Calcula métricas de persistencia temporal por `cell_id` para auditoría B2 y filtro C3."""
    w_months = winter_months if winter_months is not None else [6, 7, 8]
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "cell_id": pl.String,
                "mean_lat": pl.Float64,
                "mean_lon": pl.Float64,
                "n_detections": pl.Int64,
                "n_days": pl.Int64,
                "n_months": pl.Int64,
                "n_year_months": pl.Int64,
                "n_winter_days": pl.Int64,
                "only_nov_apr": pl.Boolean,
                "calendar_months": pl.List(pl.Int64),
                "active_years": pl.List(pl.Int64),
            }
        )

    enriched = df.with_columns(
        pl.col("local_date").dt.month().cast(pl.Int64).alias("_cal_month"),
        pl.col("local_date").dt.year().cast(pl.Int64).alias("_cal_year"),
        pl.col("local_date").dt.to_string("%Y-%m").alias("_ym"),
    )

    return (
        enriched.group_by("cell_id")
        .agg(
            pl.col("latitude").mean().round(5).alias("mean_lat"),
            pl.col("longitude").mean().round(5).alias("mean_lon"),
            pl.len().cast(pl.Int64).alias("n_detections"),
            pl.col("local_date").n_unique().cast(pl.Int64).alias("n_days"),
            pl.col("_cal_month").n_unique().cast(pl.Int64).alias("n_months"),
            pl.col("_ym").n_unique().cast(pl.Int64).alias("n_year_months"),
            pl.col("local_date")
            .filter(pl.col("_cal_month").is_in(w_months))
            .n_unique()
            .cast(pl.Int64)
            .alias("n_winter_days"),
            pl.col("_cal_month").is_in([11, 12, 1, 2, 3, 4]).all().alias("only_nov_apr"),
            pl.col("_cal_month").unique().sort().alias("calendar_months"),
            pl.col("_cal_year").unique().sort().alias("active_years"),
        )
        .sort(["n_days", "n_detections"], descending=[True, True])
    )


def flag_and_filter_stationary_sources(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> tuple[pl.DataFrame, set[str]]:
    """Filtra celdas con actividad térmica repetitiva no forestal (industrial/permanente)."""
    if df.is_empty():
        return df, set()

    stat_cfg = filtering_cfg.stationary_source
    summary = summarize_cell_stationarity(df, winter_months=stat_cfg.winter_months)

    cond_months = pl.col("n_months") >= stat_cfg.min_distinct_months
    if stat_cfg.flag_winter_activity:
        cond_persistence = cond_months | (pl.col("n_winter_days") >= stat_cfg.min_winter_days)
    else:
        cond_persistence = cond_months

    flagged = summary.filter((pl.col("n_days") >= stat_cfg.min_distinct_days) & cond_persistence)

    stationary_cells = set(flagged.get_column("cell_id").to_list())
    if not stationary_cells:
        return df, set()

    filtered = df.filter(~pl.col("cell_id").is_in(stationary_cells))
    return filtered, stationary_cells


def compute_type_and_manual_mask_overlap(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> pl.DataFrame:
    """Audita cuántos focos elimina `type` (1/2/3) frente a las máscaras manuales (T4)."""
    if df.is_empty() or "type" not in df.columns:
        return pl.DataFrame(
            schema={
                "type": pl.Int64,
                "n_total": pl.Int64,
                "n_in_volcano_4km": pl.Int64,
                "n_in_stationary": pl.Int64,
                "n_in_either_manual": pl.Int64,
                "n_only_caught_by_type": pl.Int64,
            }
        )

    volc_mask = np.zeros(df.height, dtype=bool)
    for zone in filtering_cfg.exclusion_zones:
        volc_mask |= evaluate_zone_exclusion_mask(df, zone)

    _, stat_cells = flag_and_filter_stationary_sources(df, filtering_cfg)
    stat_mask = df.get_column("cell_id").is_in(stat_cells).to_numpy()
    either_manual = volc_mask | stat_mask

    enriched = df.with_columns(
        pl.Series("_volc", volc_mask),
        pl.Series("_stat", stat_mask),
        pl.Series("_either", either_manual),
    )

    return (
        enriched.group_by("type")
        .agg(
            pl.len().cast(pl.Int64).alias("n_total"),
            pl.col("_volc").sum().cast(pl.Int64).alias("n_in_volcano_4km"),
            pl.col("_stat").sum().cast(pl.Int64).alias("n_in_stationary"),
            pl.col("_either").sum().cast(pl.Int64).alias("n_in_either_manual"),
            (~pl.col("_either")).sum().cast(pl.Int64).alias("n_only_caught_by_type"),
        )
        .sort("type")
    )


def apply_false_positive_masks(
    df: pl.DataFrame,
    filtering_cfg: FilteringConfig,
) -> tuple[pl.DataFrame, FilterFunnelStats, set[str]]:
    """Aplica secuencialmente las 4 máscaras de falsos positivos y valida el contrato Pandera."""
    n_in = df.height

    after_conf = filter_by_confidence(df, filtering_cfg)
    n_conf = after_conf.height

    after_type = filter_by_type(after_conf, filtering_cfg)
    n_type = after_type.height

    after_volc = filter_volcanic_and_snow_zones(after_type, filtering_cfg)
    n_volc = after_volc.height

    after_stat, stationary_cells = flag_and_filter_stationary_sources(after_volc, filtering_cfg)
    n_stat = after_stat.height

    stats = FilterFunnelStats(
        n_input_in_bbox=n_in,
        n_after_confidence_mask=n_conf,
        n_after_type_mask=n_type,
        n_after_volcanic_snow_mask=n_volc,
        n_after_stationary_mask=n_stat,
    )
    return validate_filtered_firms(after_stat), stats, stationary_cells
