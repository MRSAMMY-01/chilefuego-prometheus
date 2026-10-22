"""Test de monotonía espacial en características de red vial OSM (F1).

PORQUÉ (Regla 5):
- Al ampliar el área de búsqueda espacial (bounding box + buffer de 10 km), el conjunto de vías
  es un superconjunto del área anterior.
- Por definición métrica euclidiana: dist_nueva <= dist_vieja para toda celda.
- Cualquier violación dist_nueva > dist_vieja indica un error de corte o pérdida de información geométrica.
"""

from __future__ import annotations

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
import polars as pl
import pyogrio  # type: ignore[import-untyped]
import shapely
from chilefuego.features.osm import MAJOR_HIGHWAYS
from pyproj import Transformer
from shapely.strtree import STRtree


def test_osm_distance_monotonicity() -> None:
    pbf_path = "data/raw/osm/chile-latest.osm.pbf"
    grid_path = "data/processed/grid_catalog.parquet"

    grid_df = pl.read_parquet(grid_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()
    centroids = gpd.points_from_xy(cx, cy, crs=32719)

    # 1. BBox antiguo sin buffer (original)
    old_bbox = (-73.0, -37.3, -70.9, -35.9)
    lines_old = pyogrio.read_dataframe(pbf_path, layer="lines", bbox=old_bbox, columns=["highway"])
    roads_old = lines_old[lines_old["highway"].notna()].to_crs(epsg=32719)
    major_old = roads_old[roads_old["highway"].isin(MAJOR_HIGHWAYS)]

    tree_old_all = STRtree(roads_old.geometry.values)
    idx_old_all = tree_old_all.nearest(centroids)
    dist_old_all = shapely.distance(centroids, roads_old.geometry.values[idx_old_all])

    tree_old_major = STRtree(major_old.geometry.values)
    idx_old_major = tree_old_major.nearest(centroids)
    dist_old_major = shapely.distance(centroids, major_old.geometry.values[idx_old_major])

    # 2. BBox ampliado con 10 km de buffer
    minx_utm = float(cx.min()) - 500.0 - 10000.0
    maxx_utm = float(cx.max()) + 500.0 + 10000.0
    miny_utm = float(cy.min()) - 500.0 - 10000.0
    maxy_utm = float(cy.max()) + 500.0 + 10000.0

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lons, lats = trans.transform(
        [minx_utm, minx_utm, maxx_utm, maxx_utm],
        [miny_utm, maxy_utm, miny_utm, maxy_utm],
    )
    bbox_new = (min(lons) - 0.01, min(lats) - 0.01, max(lons) + 0.01, max(lats) + 0.01)

    lines_new = pyogrio.read_dataframe(pbf_path, layer="lines", bbox=bbox_new, columns=["highway"])
    roads_new = lines_new[lines_new["highway"].notna()].to_crs(epsg=32719)
    major_new = roads_new[roads_new["highway"].isin(MAJOR_HIGHWAYS)]

    tree_new_all = STRtree(roads_new.geometry.values)
    idx_new_all = tree_new_all.nearest(centroids)
    dist_new_all = shapely.distance(centroids, roads_new.geometry.values[idx_new_all])

    tree_new_major = STRtree(major_new.geometry.values)
    idx_new_major = tree_new_major.nearest(centroids)
    dist_new_major = shapely.distance(centroids, major_new.geometry.values[idx_new_major])

    # 3. Comprobar que dist_nueva <= dist_vieja en TODAS las celdas (tolerando epsilon 1 mm)
    diff_all = dist_new_all - dist_old_all
    diff_major = dist_new_major - dist_old_major

    assert np.all(diff_all <= 1e-3), f"Violación de monotonía en red total: máx diff = {diff_all.max()}"
    assert np.all(diff_major <= 1e-3), f"Violación de monotonía en vías principales: máx diff = {diff_major.max()}"
