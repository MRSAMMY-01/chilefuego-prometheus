"""Extracción de coberturas de suelo desde ESA WorldCover 10m por ventanas (Fase 2B — D3).

PORQUÉ (Regla 5):
- La vegetación y el tejido urbano determinan la disponibilidad de combustible y la exposición al fuego:
  * tree_cover_frac (clase 10): combustible forestal denso (plantaciones y bosque nativo).
  * shrub_cover_frac (clase 20): matorrales con rápida desecación estival.
  * grass_cover_frac (clase 30): pastizales finos de rápida ignición y propagación.
  * crop_cover_frac (clase 40): cultivos agrícolas (quemas y rastrojos).
  * built_cover_frac (clase 50): edificaciones e infraestructura (interfaz urbano-forestal WUI).
  * dist_built_nearest_m: proximidad física en metros a la celda edificada más cercana (sin depender de fuentes externas INE).
- Cero desbordamiento de RAM: Lectura estrictamente por ventanas espaciales (block windows de 1000m x 1000m ~100x100 px),
  sin cargar ninguna de las teselas de 36.000 x 36.000 píxeles completas en memoria.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path

import numpy as np
import polars as pl
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy.spatial import cKDTree

from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)


def extract_worldcover_features(
    worldcover_dir: Path = Path("data/raw/worldcover"),
    grid_path: Path = Path("data/processed/grid_catalog.parquet"),
    out_path: Path = Path("data/processed/worldcover_features_1km.parquet"),
) -> tuple[pl.DataFrame, float, float]:
    """Extrae fracciones de cobertura y distancia a edificado por celda de 1 km."""
    tracemalloc.start()
    t0 = time.time()

    grid_df = pl.read_parquet(grid_path)
    nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()
    n_cells = len(cell_ids)

    tif_72 = worldcover_dir / "ESA_WorldCover_10m_2021_v200_S39W072_Map.tif"
    tif_75 = worldcover_dir / "ESA_WorldCover_10m_2021_v200_S39W075_Map.tif"

    if not tif_72.exists() or not tif_75.exists():
        raise FileNotFoundError(f"Faltan teselas WorldCover en {worldcover_dir}")

    to_wgs84 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)

    tree_frac = np.zeros(n_cells, dtype=np.float32)
    shrub_frac = np.zeros(n_cells, dtype=np.float32)
    grass_frac = np.zeros(n_cells, dtype=np.float32)
    crop_frac = np.zeros(n_cells, dtype=np.float32)
    built_frac = np.zeros(n_cells, dtype=np.float32)

    half = 500.0  # celda de 1 km

    # Abrir ambos datasets en modo lectura por ventana
    with rasterio.open(tif_72) as src_72, rasterio.open(tif_75) as src_75:
        for i, (x_m, y_m) in enumerate(zip(cx, cy, strict=False)):
            corners_x = [x_m - half, x_m + half, x_m + half, x_m - half]
            corners_y = [y_m - half, y_m - half, y_m + half, y_m + half]
            lons, lats = to_wgs84.transform(corners_x, corners_y)
            w_lon, e_lon = min(lons), max(lons)
            s_lat, n_lat = min(lats), max(lats)
            center_lon = (w_lon + e_lon) / 2.0

            src = src_75 if center_lon < -72.0 else src_72
            win = from_bounds(w_lon, s_lat, e_lon, n_lat, transform=src.transform)

            try:
                data = src.read(1, window=win)
            except Exception:
                data = np.array([], dtype=np.uint8)

            if data.size > 0:
                n_total = float(data.size)
                tree_frac[i] = float(np.sum(data == 10)) / n_total
                shrub_frac[i] = float(np.sum(data == 20)) / n_total
                grass_frac[i] = float(np.sum(data == 30)) / n_total
                crop_frac[i] = float(np.sum(data == 40)) / n_total
                built_frac[i] = float(np.sum(data == 50)) / n_total

    # Distancia a la celda edificada más cercana (en metros, EPSG:32719)
    # Identificar celdas con presencia de edificado (built_cover_frac > 0)
    built_mask = built_frac > 0.0
    if np.any(built_mask):
        built_xy = np.column_stack([cx[built_mask], cy[built_mask]])
        built_tree = cKDTree(built_xy)
        all_xy = np.column_stack([cx, cy])
        dist_built_m, _ = built_tree.query(all_xy, k=1)
        dist_built_m = dist_built_m.astype(np.float32)
    else:
        dist_built_m = np.full(n_cells, 99999.0, dtype=np.float32)

    res_df = pl.DataFrame({
        "cell_id": cell_ids,
        "tree_cover_frac": tree_frac,
        "shrub_cover_frac": shrub_frac,
        "grass_cover_frac": grass_frac,
        "crop_cover_frac": crop_frac,
        "built_cover_frac": built_frac,
        "dist_built_nearest_m": dist_built_m,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.write_parquet(out_path)

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = time.time() - t0
    peak_mb = peak_mem / (1024 * 1024)

    log_event(
        logger,
        "worldcover_features_extracted",
        rows=res_df.height,
        elapsed_s=round(elapsed, 2),
        peak_memory_mb=round(peak_mb, 2),
        path=str(out_path),
    )

    return res_df, elapsed, peak_mb


if __name__ == "__main__":
    df, dt, mem = extract_worldcover_features()
    print(f"WorldCover procesado exitosamente en {dt:.2f} s. Pico de memoria: {mem:.2f} MB.")
    print("Null count:", df.null_count())
    for col_name in df.columns:
        if col_name == "cell_id":
            continue
        arr = df[col_name].to_numpy()
        v_min = float(np.min(arr))
        v_max = float(np.max(arr))
        v_mean = float(np.mean(arr))
        print(f"{col_name}: min={v_min:.4f}, max={v_max:.4f}, mean={v_mean:.4f}")
