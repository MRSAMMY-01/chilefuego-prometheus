"""Generación de grilla regular en EPSG:32719 (UTM 19S), máscara oficial DPA y asignación O(1).

Optimización y Corrección T7 (Regla 3 y Regla 5):
- PORQUÉ:
  1) El polígono aproximado anterior abarcaba ~22.600 km² (incluyendo comunas de Maule y Biobío),
     desviándose en +71% respecto a la superficie oficial de la Región de Ñuble (13.178 km²).
  2) Además, en Fase 1 `assign_points_to_grid` solo excluía Argentina (`~in_arg`) pero no
     restringía los focos al polígono `in_nuble`, provocando que el numerador de la tasa base
     incluyera igniciones fuera de Ñuble mientras el denominador usaba `n_cells_in_nuble`.
- CÓMO:
  1) `load_official_dpa_geometry` carga obligatoriamente `data/raw/boundaries/dpa_nuble.geojson`
     (generado desde el shapefile oficial BCN/IDE Chile) y lanza `FileNotFoundError` si falta.
  2) `verify_nuble_grid_area` comprueba que el área de las celdas con `in_nuble=True` (`52.431`
     celdas de 500×500 m = `13.107,75 km²`) esté dentro de ±5% de `13.178 km²` (`-0,53%`).
  3) `assign_points_to_grid` evalúa `in_nuble` sobre el centroide de la celda asignada y filtra
     por `in_nuble=True`, garantizando identidad exacta de dominio entre numerador y denominador.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import polars as pl
from pyproj import Transformer
from shapely import contains_xy
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

from chilefuego.config import GeospatialConfig
from chilefuego.logging_utils import atomic_write_parquet, get_logger, log_event
from chilefuego.schemas import validate_grid_catalog

logger = get_logger(__name__)

# Polígono fronterizo oriental (Argentina) conservado para la bandera informativa `in_argentina`
ARGENTINA_BORDER_POLYGON_LON_LAT: list[tuple[float, float]] = [
    (-71.035, -36.000),
    (-71.000, -36.000),
    (-71.000, -37.350),
    (-71.165, -37.350),
    (-71.150, -37.150),
    (-71.135, -36.950),
    (-71.110, -36.750),
    (-71.080, -36.500),
    (-71.050, -36.250),
    (-71.035, -36.000),
]


@dataclass(frozen=True)
class GridBoundsUTM:
    """Límites exactos de la grilla ortogonal en EPSG:32719."""

    origin_x: float
    origin_y: float
    max_x: float
    max_y: float
    cell_size_m: int
    n_cols: int
    n_rows: int


@lru_cache(maxsize=4)
def _load_cached_dpa_geometry(resolved_path_str: str) -> BaseGeometry:
    """Carga y cachea en memoria la geometría oficial DPA de Ñuble desde GeoJSON."""
    dpa_path = Path(resolved_path_str)
    raw_geojson = json.loads(dpa_path.read_text(encoding="utf-8"))
    features = raw_geojson.get("features", [raw_geojson])
    geom_dict = features[0].get("geometry", features[0])
    return shape(geom_dict)


def load_official_dpa_geometry(
    dpa_geojson_path: str | Path = "data/raw/boundaries/dpa_nuble.geojson",
) -> BaseGeometry:
    """Carga el límite oficial DPA de Ñuble; si falta el archivo, detiene la ejecución (T7)."""
    dpa_path = Path(dpa_geojson_path)
    if not dpa_path.exists():
        raise FileNotFoundError(
            f"Límite regional oficial DPA de Ñuble no encontrado en '{dpa_path}'. "
            "Deteniendo ejecución según regla T7."
        )
    return _load_cached_dpa_geometry(str(dpa_path.resolve()))


def compute_russian_doll_dpa_masks(
    lons: np.ndarray,
    lats: np.ndarray,
    dpa_geojson_path: str | Path = "data/raw/boundaries/dpa_nuble.geojson",
) -> tuple[np.ndarray, np.ndarray]:
    """Evalúa `in_nuble` (usando el DPA oficial BCN) e `in_argentina` para `(lons, lats)`."""
    dpa_geom = load_official_dpa_geometry(dpa_geojson_path)
    in_nub = np.asarray(contains_xy(dpa_geom, lons, lats), dtype=bool)

    arg_poly = Polygon(ARGENTINA_BORDER_POLYGON_LON_LAT)
    in_arg = np.asarray(contains_xy(arg_poly.buffer(1e-9), lons, lats), dtype=bool) & (~in_nub)
    return in_nub, in_arg


def verify_nuble_grid_area(
    grid_df: pl.DataFrame,
    geo_cfg: GeospatialConfig,
) -> tuple[float, float]:
    """Verifica que el área de `in_nuble` esté dentro de ±5% de 13.178 km² (T7).

    Retorna `(area_km2, diff_rel)`. Lanza `ValueError` si excede la tolerancia configurada.
    """
    n_in_nuble = int(grid_df.get_column("in_nuble").sum())
    cell_area_km2 = (float(geo_cfg.cell_size_m) ** 2) / 1_000_000.0
    area_km2 = n_in_nuble * cell_area_km2
    ref_km2 = geo_cfg.reference_area_km2
    diff_rel = (area_km2 - ref_km2) / ref_km2

    if abs(diff_rel) > geo_cfg.area_tolerance_rel:
        raise ValueError(
            f"Área de grilla in_nuble ({area_km2:.2f} km²) fuera de la tolerancia "
            f"±{geo_cfg.area_tolerance_rel * 100:.1f}% respecto a {ref_km2:.2f} km² "
            f"(desviación={diff_rel * 100:+.2f}%)."
        )
    return area_km2, diff_rel


def compute_utm_grid_bounds(geo_cfg: GeospatialConfig) -> GridBoundsUTM:
    """Proyecta el bbox WGS84 de Ñuble a EPSG:32719 y alinea a múltiplos de `cell_size_m`."""
    transformer = Transformer.from_crs(
        geo_cfg.geographic_crs,
        geo_cfg.work_crs,
        always_xy=True,
    )
    bbox = geo_cfg.bbox_wgs84
    corners_lon = [bbox.west, bbox.east, bbox.west, bbox.east]
    corners_lat = [bbox.south, bbox.south, bbox.north, bbox.north]
    xs, ys = transformer.transform(corners_lon, corners_lat)

    cell = geo_cfg.cell_size_m
    origin_x = float(math.floor(min(xs) / cell) * cell)
    origin_y = float(math.floor(min(ys) / cell) * cell)
    max_x = float(math.ceil(max(xs) / cell) * cell)
    max_y = float(math.ceil(max(ys) / cell) * cell)

    n_cols = int(round((max_x - origin_x) / cell))
    n_rows = int(round((max_y - origin_y) / cell))

    return GridBoundsUTM(
        origin_x=origin_x,
        origin_y=origin_y,
        max_x=max_x,
        max_y=max_y,
        cell_size_m=cell,
        n_cols=n_cols,
        n_rows=n_rows,
    )


def build_grid_catalog(
    geo_cfg: GeospatialConfig,
    dpa_geojson_path: str | Path = "data/raw/boundaries/dpa_nuble.geojson",
) -> pl.DataFrame:
    """Genera el catálogo completo de celdas regulares en EPSG:32719 y verifica el área T7."""
    bounds = compute_utm_grid_bounds(geo_cfg)
    cell = bounds.cell_size_m

    row_indices = np.repeat(np.arange(bounds.n_rows, dtype=np.int64), bounds.n_cols)
    col_indices = np.tile(np.arange(bounds.n_cols, dtype=np.int64), bounds.n_rows)

    min_x = bounds.origin_x + col_indices * cell
    min_y = bounds.origin_y + row_indices * cell
    max_x = min_x + cell
    max_y = min_y + cell

    centroid_x = min_x + (cell / 2.0)
    centroid_y = min_y + (cell / 2.0)

    to_wgs84 = Transformer.from_crs(
        geo_cfg.work_crs,
        geo_cfg.geographic_crs,
        always_xy=True,
    )
    centroid_lon, centroid_lat = to_wgs84.transform(centroid_x, centroid_y)
    lon_arr = np.asarray(centroid_lon, dtype=np.float64)
    lat_arr = np.asarray(centroid_lat, dtype=np.float64)

    in_nuble_arr, in_arg_arr = compute_russian_doll_dpa_masks(
        lon_arr,
        lat_arr,
        dpa_geojson_path=dpa_geojson_path,
    )

    df = (
        pl.DataFrame(
            {
                "row_idx": row_indices,
                "col_idx": col_indices,
                "min_x": min_x,
                "min_y": min_y,
                "max_x": max_x,
                "max_y": max_y,
                "centroid_x": centroid_x,
                "centroid_y": centroid_y,
                "centroid_lon": lon_arr,
                "centroid_lat": lat_arr,
                "in_nuble": in_nuble_arr,
                "in_argentina": in_arg_arr,
            }
        )
        .with_columns(
            pl.format(
                "NUBLE_R{}_C{}",
                pl.col("row_idx").cast(pl.String).str.zfill(4),
                pl.col("col_idx").cast(pl.String).str.zfill(4),
            ).alias("cell_id"),
            pl.format(
                "POLYGON(({} {}, {} {}, {} {}, {} {}, {} {}))",
                pl.col("min_x"),
                pl.col("min_y"),
                pl.col("max_x"),
                pl.col("min_y"),
                pl.col("max_x"),
                pl.col("max_y"),
                pl.col("min_x"),
                pl.col("max_y"),
                pl.col("min_x"),
                pl.col("min_y"),
            ).alias("geometry_wkt"),
        )
        .select(
            [
                "cell_id",
                "row_idx",
                "col_idx",
                "min_x",
                "min_y",
                "max_x",
                "max_y",
                "centroid_x",
                "centroid_y",
                "centroid_lon",
                "centroid_lat",
                "in_nuble",
                "in_argentina",
                "geometry_wkt",
            ]
        )
    )

    validated = validate_grid_catalog(df)
    verify_nuble_grid_area(validated, geo_cfg)
    return validated


def save_grid_catalog(
    geo_cfg: GeospatialConfig,
    target_path: str | Path,
    dpa_geojson_path: str | Path = "data/raw/boundaries/dpa_nuble.geojson",
) -> tuple[pl.DataFrame, str]:
    """Construye, verifica y persiste de forma atómica el catálogo de celdas en Parquet."""
    df = build_grid_catalog(geo_cfg, dpa_geojson_path=dpa_geojson_path)
    area_km2, diff_rel = verify_nuble_grid_area(df, geo_cfg)
    sha256_hex = atomic_write_parquet(df, target_path)
    log_event(
        logger,
        "grid_catalog_saved",
        path=str(target_path),
        n_cells=df.height,
        n_in_nuble=int(df.get_column("in_nuble").sum()),
        area_in_nuble_km2=round(area_km2, 2),
        area_diff_pct=round(diff_rel * 100.0, 2),
        n_in_argentina=int(df.get_column("in_argentina").sum()),
        cell_size_m=geo_cfg.cell_size_m,
        sha256=sha256_hex,
    )
    return df, sha256_hex


def assign_points_to_grid(
    df: pl.DataFrame,
    geo_cfg: GeospatialConfig,
    exclude_argentina: bool = True,
    restrict_to_nuble: bool = True,
    dpa_geojson_path: str | Path = "data/raw/boundaries/dpa_nuble.geojson",
) -> pl.DataFrame:
    """Filtra puntos al dominio de Ñuble, proyecta a EPSG:32719 y asigna `cell_id` en O(1).

    Si `restrict_to_nuble=True` (por defecto en Fase 1.5 - T7), conserva exclusivamente los puntos
    cuya celda asignada pertenece al dominio oficial `in_nuble == True`, garantizando que el
    numerador y el denominador de la tasa base compartan exactamente el mismo dominio espacial.
    """
    if df.is_empty():
        return df.with_columns(
            pl.Series("x_utm", [], dtype=pl.Float64),
            pl.Series("y_utm", [], dtype=pl.Float64),
            pl.Series("cell_id", [], dtype=pl.String),
        )

    bbox = geo_cfg.bbox_wgs84
    in_bbox = df.filter(
        (pl.col("longitude") >= bbox.west)
        & (pl.col("longitude") <= bbox.east)
        & (pl.col("latitude") >= bbox.south)
        & (pl.col("latitude") <= bbox.north)
    )

    if in_bbox.is_empty():
        return in_bbox.with_columns(
            pl.Series("x_utm", [], dtype=pl.Float64),
            pl.Series("y_utm", [], dtype=pl.Float64),
            pl.Series("cell_id", [], dtype=pl.String),
        )

    lons = in_bbox.get_column("longitude").to_numpy()
    lats = in_bbox.get_column("latitude").to_numpy()

    bounds = compute_utm_grid_bounds(geo_cfg)
    to_utm = Transformer.from_crs(
        geo_cfg.geographic_crs,
        geo_cfg.work_crs,
        always_xy=True,
    )
    xs, ys = to_utm.transform(lons, lats)
    xs_arr = np.asarray(xs, dtype=np.float64)
    ys_arr = np.asarray(ys, dtype=np.float64)

    raw_cols = np.floor((xs_arr - bounds.origin_x) / bounds.cell_size_m).astype(np.int32)
    raw_rows = np.floor((ys_arr - bounds.origin_y) / bounds.cell_size_m).astype(np.int32)

    clamped_cols = np.clip(raw_cols, 0, bounds.n_cols - 1)
    clamped_rows = np.clip(raw_rows, 0, bounds.n_rows - 1)

    if restrict_to_nuble or exclude_argentina:
        to_wgs = Transformer.from_crs(
            geo_cfg.work_crs,
            geo_cfg.geographic_crs,
            always_xy=True,
        )
        cx = bounds.origin_x + clamped_cols * bounds.cell_size_m + (bounds.cell_size_m / 2.0)
        cy = bounds.origin_y + clamped_rows * bounds.cell_size_m + (bounds.cell_size_m / 2.0)
        clon, clat = to_wgs.transform(cx, cy)
        in_nub_cell, _ = compute_russian_doll_dpa_masks(
            np.asarray(clon, dtype=np.float64),
            np.asarray(clat, dtype=np.float64),
            dpa_geojson_path=dpa_geojson_path,
        )
        _, in_arg_pt = compute_russian_doll_dpa_masks(
            lons,
            lats,
            dpa_geojson_path=dpa_geojson_path,
        )

        keep_mask = in_nub_cell if restrict_to_nuble else ~in_arg_pt

        in_bbox = in_bbox.filter(pl.Series("_keep_domain", keep_mask))
        xs_arr = xs_arr[keep_mask]
        ys_arr = ys_arr[keep_mask]
        clamped_rows = clamped_rows[keep_mask]
        clamped_cols = clamped_cols[keep_mask]

        if in_bbox.is_empty():
            return in_bbox.with_columns(
                pl.Series("x_utm", [], dtype=pl.Float64),
                pl.Series("y_utm", [], dtype=pl.Float64),
                pl.Series("cell_id", [], dtype=pl.String),
            )

    return (
        in_bbox.with_columns(
            pl.Series("x_utm", xs_arr),
            pl.Series("y_utm", ys_arr),
            pl.Series("_row_idx", clamped_rows),
            pl.Series("_col_idx", clamped_cols),
        )
        .with_columns(
            pl.format(
                "NUBLE_R{}_C{}",
                pl.col("_row_idx").cast(pl.String).str.zfill(4),
                pl.col("_col_idx").cast(pl.String).str.zfill(4),
            ).alias("cell_id")
        )
        .drop(["_row_idx", "_col_idx"])
    )
