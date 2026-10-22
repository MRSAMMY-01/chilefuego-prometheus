"""Extracción de densidad poblacional por celda de 1 km desde WorldPop 2020 (Fase 2B — C2).

Dataset: WorldPop Global 2020 (Constrained, 1 km resolution).
Proveedor: WorldPop / University of Southampton.
Licencia: Creative Commons Attribution 4.0 International (CC-BY 4.0).
Año: 2020.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import rasterio
from pyproj import Transformer


def extract_population_density_1km(
    grid_df: pl.DataFrame,
    worldpop_tif: Path = Path("data/raw/worldpop/chl_pop_2020_1km.tif"),
) -> pl.DataFrame:
    """Muestrea la densidad poblacional (hab/km²) de WorldPop en el centroide de cada celda de 1 km."""
    if not worldpop_tif.exists():
        raise FileNotFoundError(f"Archivo WorldPop no encontrado en {worldpop_tif}")

    nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    # Transformar centroides de EPSG:32719 a WGS 84 (EPSG:4326)
    to_wgs84 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lons, lats = to_wgs84.transform(cx, cy)

    pop_vals = np.zeros(len(cell_ids), dtype=np.float32)

    with rasterio.open(worldpop_tif) as src:
        nodata = src.nodata
        inv_transform = ~src.transform
        pop_arr = src.read(1)
        h, w = pop_arr.shape

        for i, (lon, lat) in enumerate(zip(lons, lats, strict=False)):
            col, row = inv_transform * (lon, lat)
            r = int(round(row))
            c = int(round(col))
            if 0 <= r < h and 0 <= c < w:
                val = pop_arr[r, c]
                if nodata is not None and val == nodata or np.isnan(val) or val < 0.0:
                    pop_vals[i] = 0.0
                else:
                    pop_vals[i] = float(val)
            else:
                pop_vals[i] = 0.0

    return pl.DataFrame({
        "cell_id": cell_ids,
        "pop_density_1km": pop_vals,
    })
