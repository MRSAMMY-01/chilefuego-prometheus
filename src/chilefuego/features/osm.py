"""Extracción de características viales desde OpenStreetMap (Fase 2B — D2).

PORQUÉ (Regla 5):
- La accesibilidad humana es el principal determinante espacial de igniciones antrópicas accidentales e intencionales.
- Cero desbordamiento de RAM: Se filtra el PBF directamente por el bounding box de Ñuble a través del driver OSM de GDAL
  (pyogrio), extrayendo únicamente líneas con etiqueta 'highway'.
- Salidas en float32 por celda de 1 km:
  * dist_road_nearest_m: distancia al camino más cercano (cualquier vía).
  * dist_primary_sec_trunk_m: distancia a rutas principales (motorway, trunk, primary, secondary).
  * road_density_km2: densidad lineal de caminos (km de vía / km² de celda).
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import polars as pl
import pyogrio  # type: ignore[import-untyped]
from pyproj import Transformer
from shapely.strtree import STRtree

from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)

MAJOR_HIGHWAYS = {
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
}


def extract_osm_road_features(
    pbf_path: Path = Path("data/raw/osm/chile-latest.osm.pbf"),
    grid_path: Path = Path("data/processed/grid_catalog.parquet"),
    out_path: Path = Path("data/processed/osm_road_features_1km.parquet"),
    buffer_m: float = 10000.0,
) -> tuple[pl.DataFrame, float, float]:
    """Extrae características de distancia y densidad vial por celda de 1 km.

    Retorna: (df_features, elapsed_seconds, peak_memory_mb)
    """
    tracemalloc.start()
    t0 = time.time()

    # 1. Cargar celdas de Ñuble y calcular BBox dinámico + 10 km de buffer (E6)
    grid_df = pl.read_parquet(grid_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()
    n_cells = len(cell_ids)

    minx_utm = float(cx.min()) - 500.0 - buffer_m
    maxx_utm = float(cx.max()) + 500.0 + buffer_m
    miny_utm = float(cy.min()) - 500.0 - buffer_m
    maxy_utm = float(cy.max()) + 500.0 + buffer_m

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lons, lats = trans.transform(
        [minx_utm, minx_utm, maxx_utm, maxx_utm],
        [miny_utm, maxy_utm, miny_utm, maxy_utm],
    )
    bbox_wgs84 = (min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01)

    # 2. Leer líneas dentro de bbox de Ñuble + 10km sin cargar todo el país en memoria
    lines_gdf = pyogrio.read_dataframe(
        pbf_path,
        layer="lines",
        bbox=bbox_wgs84,
        columns=["highway"],
    )

    # Filtrar aquellas con etiqueta highway válida
    roads = lines_gdf[lines_gdf["highway"].notna()].copy()
    del lines_gdf

    # Reproyectar a EPSG:32719 (coordenadas en metros)
    roads_utm = roads.to_crs(epsg=32719)
    del roads

    # Separar red completa y red mayor (primarios, secundarios, troncales)
    major_mask = roads_utm["highway"].isin(MAJOR_HIGHWAYS)
    major_roads_utm = roads_utm[major_mask].copy()

    # 3. Crear geometrías de puntos de los centroides
    centroids_geom = gpd.points_from_xy(cx, cy, crs=32719)

    # 4. Distancia al camino más cercano (todos)
    all_tree = STRtree(roads_utm.geometry.values)
    nearest_idx_all = all_tree.nearest(centroids_geom)
    nearest_lines_all = roads_utm.geometry.values[nearest_idx_all]
    dist_all_m = centroids_geom.distance(nearest_lines_all).astype(np.float32)

    # 5. Distancia a caminos principales (primarios/secundarios/troncales)
    if len(major_roads_utm) > 0:
        major_tree = STRtree(major_roads_utm.geometry.values)
        nearest_idx_major = major_tree.nearest(centroids_geom)
        nearest_lines_major = major_roads_utm.geometry.values[nearest_idx_major]
        dist_major_m = centroids_geom.distance(nearest_lines_major).astype(np.float32)
    else:
        dist_major_m = np.full(n_cells, 99999.0, dtype=np.float32)

    # 6. Densidad de caminos (km / km²)
    # Cada celda es de 1000m x 1000m (1 km²). Creamos buffers de caja de 1 km alrededor del centroide.
    half = 500.0
    boxes = [
        gpd.GeoSeries.from_xy([x], [y], crs=32719).buffer(half, cap_style=3)[0]
        for x, y in zip(cx, cy, strict=False)
    ]
    grid_poly_gdf = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=boxes, crs=32719)

    # Spatial join para intersección de líneas
    intersections = gpd.overlay(roads_utm, grid_poly_gdf, how="intersection")
    intersections["length_km"] = intersections.geometry.length / 1000.0
    density_per_cell = intersections.groupby("cell_id")["length_km"].sum().to_dict()

    road_density = np.array([float(density_per_cell.get(cid, 0.0)) for cid in cell_ids], dtype=np.float32)

    res_df = pl.DataFrame({
        "cell_id": cell_ids,
        "dist_road_nearest_m": dist_all_m,
        "dist_primary_sec_trunk_m": dist_major_m,
        "road_density_km2": road_density,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.write_parquet(out_path)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = time.time() - t0
    peak_mb = peak_mem / (1024 * 1024)

    log_event(
        logger,
        "osm_features_extracted",
        rows=res_df.height,
        elapsed_s=round(elapsed, 2),
        peak_memory_mb=round(peak_mb, 2),
        path=str(out_path),
    )

    return res_df, elapsed, peak_mb


if __name__ == "__main__":
    df, dt, mem = extract_osm_road_features()
    print(f"OSM procesado exitosamente en {dt:.2f} s. Pico de memoria: {mem:.2f} MB.")
    print(df.describe())
