"""Tests de geometría sintética para el extractor de red vial OSM (E6c).

PORQUÉ (Regla 5):
- Verificar la corrección matemática del cálculo de distancias ortogonales euclidianas a segmentos de línea.
- Con geometrías controladas de coordenadas conocidas, confirmamos que el árbol espacial STRtree
  y las distancias devueltas coinciden exactamente con la distancia perpendicular teórica.
"""

from __future__ import annotations

import geopandas as gpd  # type: ignore[import-untyped]
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


def test_strtree_synthetic_line_distance() -> None:
    # Línea horizontal a lo largo de Y = 1000 de X = 0 a X = 2000
    line = LineString([(0.0, 1000.0), (2000.0, 1000.0)])
    lines_series = gpd.GeoSeries([line], crs="EPSG:32719")
    tree = STRtree(lines_series.geometry.values)

    # Tres puntos sintéticos:
    # 1. (500, 500) -> distancia perpendicular = 500 m
    # 2. (1000, 1200) -> distancia perpendicular = 200 m
    # 3. (1000, 1000) -> sobre la línea -> distancia = 0 m
    pts = [Point(500.0, 500.0), Point(1000.0, 1200.0), Point(1000.0, 1000.0)]
    pts_series = gpd.GeoSeries(pts, crs="EPSG:32719")

    nearest_idx = tree.nearest(pts_series.geometry.values)
    nearest_geoms = lines_series.geometry.values[nearest_idx]
    import shapely
    dists = shapely.distance(pts_series.geometry.values, nearest_geoms)

    assert np.isclose(dists[0], 500.0, atol=1e-3)
    assert np.isclose(dists[1], 200.0, atol=1e-3)
    assert np.isclose(dists[2], 0.0, atol=1e-3)
