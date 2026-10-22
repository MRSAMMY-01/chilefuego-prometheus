"""Extracción de características topográficas estáticas (DEM, pendiente, aspect) por celda de 1 km (Fase 2B — C2).

PORQUÉ:
- La topografía ejerce un control fundamental sobre la probabilidad y comportamiento del fuego:
  * Elevación: gradiente altitudinal de temperatura, humedad y tipos de vegetación.
  * Pendiente (media y máxima): acelera la propagación ladera arriba por transferencia convectiva y radiativa.
  * Aspect (northness y eastness): en el hemisferio sur, las laderas norte (northness > 0) reciben máxima
    insolación diurna, mientras las laderas sur son húmedas y sombrías.
- Se utiliza Copernicus DEM GLO-30 (30 m) mosaicateado y reproyectado a EPSG:32719 UTM Huso 19S.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.ndimage import sobel

from chilefuego.logging_utils import get_logger

logger = get_logger(__name__)


def compute_slope_and_aspect(
    dem_arr: np.ndarray,
    res_m: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calcula pendiente (grados), northness y eastness a partir de una matriz de elevación regular en metros."""
    # Gradientes espaciales usando filtros de Sobel
    # Sobel en axis 0 es dZ/dY (filas, norte-sur)
    # Sobel en axis 1 es dZ/dX (columnas, oeste-este)
    # Factor de escala para Sobel 3x3 normalizado: 8 * res_m
    scale = 8.0 * float(res_m)
    dz_dy = -sobel(dem_arr, axis=0, mode="reflect") / scale  # Negativo porque el eje Y de imagen va de norte a sur
    dz_dx = sobel(dem_arr, axis=1, mode="reflect") / scale

    grad_mag = np.hypot(dz_dx, dz_dy)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Evitar división por cero en zonas planas
    eps = 1e-6
    mask_flat = grad_mag < eps

    # Aspect theta: en sentido horario desde el norte (0 = N, pi/2 = E, pi = S, 3pi/2 = W)
    # northness = cos(aspect) = dz_dy / grad_mag
    # eastness = sin(aspect) = -dz_dx / grad_mag  (o dz_dx / grad_mag según convención hacia donde cae el agua vs hacia donde mira la ladera)
    # Convención estándar geomorfológica:
    # Una ladera que mira al Norte tiene cota creciente hacia el Sur (dz_dy < 0), pero aquí dz_dy es dZ hacia el Norte.
    # Si sube hacia el Norte, mira al Sur. Si baja hacia el Norte (dz_dy < 0), mira al Norte.
    # Por tanto, vector normal que apunta hacia la orientación de la ladera:
    # aspect_x = -dz_dx, aspect_y = -dz_dy
    northness = np.where(mask_flat, 0.0, -dz_dy / np.maximum(grad_mag, eps))
    eastness = np.where(mask_flat, 0.0, -dz_dx / np.maximum(grad_mag, eps))

    northness = np.clip(northness, -1.0, 1.0)
    eastness = np.clip(eastness, -1.0, 1.0)

    return slope_deg, northness, eastness, grad_mag


def extract_topography_features_1km(
    grid_df: pl.DataFrame,
    dem_dir: Path = Path("data/raw/dem"),
) -> pl.DataFrame:
    """Extrae elevación media, pendiente media/máxima, northness y eastness por celda de 1 km."""
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cell_ids = nuble_cells.get_column("cell_id").to_list()
    cx = nuble_cells.get_column("centroid_x").to_numpy()
    cy = nuble_cells.get_column("centroid_y").to_numpy()

    tif_files = sorted(dem_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No se encontraron tiles DEM en {dem_dir}")

    # 1. Abrir y mosaicatizar en memoria o leer por celda
    srcs = [rasterio.open(f) for f in tif_files]
    mosaic, mosaic_transform = merge(srcs)
    mosaic_meta = srcs[0].meta.copy()
    for s in srcs:
        s.close()

    mosaic_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": mosaic_transform,
        "crs": srcs[0].crs,
    })

    # 2. Reproyectar mosaico a EPSG:32719 a resolución 30m
    dst_crs = "EPSG:32719"
    transform_dst, width_dst, height_dst = calculate_default_transform(
        mosaic_meta["crs"], dst_crs, mosaic.shape[2], mosaic.shape[1],
        *rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], mosaic_transform),
        resolution=(30.0, 30.0)
    )

    dem_utm = np.empty((height_dst, width_dst), dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=dem_utm,
        src_transform=mosaic_transform,
        src_crs=mosaic_meta["crs"],
        dst_transform=transform_dst,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    # 3. Calcular derivadas de terreno a 30m
    slope_deg, northness, eastness, _ = compute_slope_and_aspect(dem_utm, res_m=30.0)

    # 4. Asignar a celdas de 1 km (muestreo de ventana +/- 500 m alrededor del centroide)
    inv_transform = ~transform_dst
    half_w = 16  # 16 celdas de 30m = ~480m (celda de 1 km)

    elev_mean = np.zeros(len(cell_ids), dtype=np.float32)
    slope_m = np.zeros(len(cell_ids), dtype=np.float32)
    slope_mx = np.zeros(len(cell_ids), dtype=np.float32)
    north_m = np.zeros(len(cell_ids), dtype=np.float32)
    east_m = np.zeros(len(cell_ids), dtype=np.float32)

    for i, (x, y) in enumerate(zip(cx, cy, strict=False)):
        col_c, row_c = inv_transform * (x, y)
        r0 = max(0, int(round(row_c)) - half_w)
        r1 = min(height_dst, int(round(row_c)) + half_w + 1)
        c0 = max(0, int(round(col_c)) - half_w)
        c1 = min(width_dst, int(round(col_c)) + half_w + 1)

        sub_elev = dem_utm[r0:r1, c0:c1]
        sub_slope = slope_deg[r0:r1, c0:c1]
        sub_north = northness[r0:r1, c0:c1]
        sub_east = eastness[r0:r1, c0:c1]

        if sub_elev.size > 0:
            elev_mean[i] = float(np.mean(sub_elev))
            slope_m[i] = float(np.mean(sub_slope))
            slope_mx[i] = float(np.max(sub_slope))
            north_m[i] = float(np.mean(sub_north))
            east_m[i] = float(np.mean(sub_east))

    return pl.DataFrame({
        "cell_id": cell_ids,
        "elevation_mean": elev_mean,
        "slope_mean": slope_m,
        "slope_max": slope_mx,
        "northness": north_m,
        "eastness": east_m,
    })
