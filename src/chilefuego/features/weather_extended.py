"""Generación de características meteorológicas extendidas por nodo ERA5-Land (Fase 2B — C3).

PORQUÉ (Regla 5):
- El FWI sintetiza la sequedad atmosférica general, pero la física del fuego responde directamente a:
  * VPD (Vapour Pressure Deficit): demanda evaporativa atmosférica inmediata sobre los combustibles finos.
  * Inercia térmica y desecación reciente: valores de D-1, y medias/máximos móviles de 3 y 7 días de T y VPD.
  * Estrés hídrico acumulado: lluvias acumuladas a 7 y 30 días.
  * Días consecutivos sin lluvia significativa: racha de días con precipitación < 2.5 mm.
- Cero fuga hacia el futuro: cada ventana móvil incluye estrictamente el día D y días anteriores (t <= D).
- Eficiencia en memoria (16 GB RAM): se computa por nodo (~399 nodos x 3.013 días = 1.2M filas en float32, ~60 MB),
  y se proyecta a celdas bajo demanda o en lotes por temporada.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from chilefuego.features.weather import (
    deaccumulate_era5_land_tp_m_to_mm,
    relative_humidity_from_t2m_d2m_k,
    wind_speed_kmh_from_uv_ms,
)
from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)


def compute_vpd_kpa(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Calcula el déficit de presión de vapor (VPD en kPa) mediante la formulación estándar de Tetens.

    e_s(T) = 0.61078 * exp(17.27 * T / (T + 237.3)) [kPa]
    VPD = e_s * (1 - RH / 100) [kPa]
    """
    t = np.asarray(temp_c, dtype=np.float64)
    rh = np.clip(np.asarray(rh_pct, dtype=np.float64), 0.0, 100.0)

    # Presión de vapor de saturación en kPa
    es_kpa = 0.61078 * np.exp((17.27 * t) / (t + 237.3))
    vpd = es_kpa * (1.0 - (rh / 100.0))
    return np.maximum(0.0, vpd)


def compute_days_since_rain_threshold(
    rain_mm: np.ndarray,
    threshold_mm: float = 2.5,
) -> np.ndarray:
    """Calcula la racha de días consecutivos sin lluvia >= threshold_mm hasta el día D."""
    r = np.asarray(rain_mm, dtype=np.float64)
    n = r.shape[0]
    out = np.zeros(n, dtype=np.int32)
    streak = 0
    for i in range(n):
        if r[i] >= threshold_mm:
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return out


def generate_extended_weather_nodes(
    era5_dir: Path = Path("data/raw/era5_land"),
    fwi_nodes_path: Path = Path("data/processed/era5_land_daily_fwi_nodes.parquet"),
    out_path: Path = Path("data/processed/era5_land_extended_weather_nodes.parquet"),
) -> pl.DataFrame:
    """Procesa los 99 NetCDF para generar la tabla diaria extendida de meteorología por nodo."""
    t0 = time.time()
    nc_files = sorted(p for p in era5_dir.glob("*.nc") if not p.name.startswith("_"))
    if not nc_files:
        raise FileNotFoundError(f"No se encontraron NetCDF en {era5_dir}")

    # 1. Coordenadas y máscara terrestre
    ds0 = xr.open_dataset(nc_files[0])
    valid_mask = ~np.isnan(ds0["tp"].values[0])
    land_coords = np.argwhere(valid_mask)
    r_idx = land_coords[:, 0]
    c_idx = land_coords[:, 1]
    node_lats = ds0["latitude"].values[r_idx]
    node_lons = ds0["longitude"].values[c_idx]
    n_nodes = len(r_idx)
    node_ids = [f"{lat:.2f}_{lon:.2f}" for lat, lon in zip(node_lats, node_lons, strict=False)]
    ds0.close()

    # 2. Extracción de valores al mediodía (16:00 UTC = 12:00 CLT) y lluvia de 24h
    daily_t: list[np.ndarray] = []
    daily_rh: list[np.ndarray] = []
    daily_w: list[np.ndarray] = []
    daily_r24: list[np.ndarray] = []
    daily_dates: list[pd.Timestamp] = []

    prev_tp_last = None

    for f in nc_files:
        ds = xr.open_dataset(f)
        time_vals = pd.to_datetime(ds["valid_time"].values)
        utc_hours = time_vals.hour.values

        tp = ds["tp"].values[:, r_idx, c_idx]
        t2m = ds["t2m"].values[:, r_idx, c_idx]
        d2m = ds["d2m"].values[:, r_idx, c_idx]
        u10 = ds["u10"].values[:, r_idx, c_idx]
        v10 = ds["v10"].values[:, r_idx, c_idx]
        ds.close()

        rain_h = deaccumulate_era5_land_tp_m_to_mm(tp, utc_hours)
        if prev_tp_last is not None:
            rain_h[0] = np.maximum(0.0, (tp[0] - prev_tp_last) * 1000.0)
        prev_tp_last = tp[-1]

        temp_c = t2m - 273.15
        rh_pct = relative_humidity_from_t2m_d2m_k(t2m, d2m)
        wind_kmh = wind_speed_kmh_from_uv_ms(u10, v10)

        noon_mask = utc_hours == 16
        noon_indices = np.where(noon_mask)[0]

        for idx in noon_indices:
            daily_t.append(temp_c[idx])
            daily_rh.append(rh_pct[idx])
            daily_w.append(wind_kmh[idx])
            daily_dates.append(time_vals[idx].floor("D"))

            # Lluvia 24h previa (idx-23 a idx inclusive)
            if idx >= 23:
                r24 = np.sum(rain_h[idx - 23 : idx + 1], axis=0)
            else:
                r24 = np.sum(rain_h[: idx + 1], axis=0)
            daily_r24.append(r24)

    t_noon_mat = np.array(daily_t, dtype=np.float32)       # (n_days, n_nodes)
    rh_noon_mat = np.array(daily_rh, dtype=np.float32)     # (n_days, n_nodes)
    wind_noon_mat = np.array(daily_w, dtype=np.float32)    # (n_days, n_nodes)
    rain_noon_mat = np.array(daily_r24, dtype=np.float32)  # (n_days, n_nodes)
    dates_list = [d.date() for d in daily_dates]
    n_days = len(dates_list)

    # 3. Calcular VPD al mediodía
    vpd_noon_mat = compute_vpd_kpa(t_noon_mat, rh_noon_mat).astype(np.float32)

    # 4. Cargar FWI existente para asociar códigos
    fwi_df = pl.read_parquet(fwi_nodes_path)

    # 5. Calcular variables móviles y lags por nodo (vectorizado por columna de nodo)
    node_parts: list[pl.DataFrame] = []

    for i in range(n_nodes):
        nid = node_ids[i]
        t_col = t_noon_mat[:, i]
        vpd_col = vpd_noon_mat[:, i]
        r_col = rain_noon_mat[:, i]
        rh_col = rh_noon_mat[:, i]
        w_col = wind_noon_mat[:, i]

        # Lag 1 (D-1)
        t_lag1 = np.empty_like(t_col)
        t_lag1[0] = t_col[0]
        t_lag1[1:] = t_col[:-1]

        vpd_lag1 = np.empty_like(vpd_col)
        vpd_lag1[0] = vpd_col[0]
        vpd_lag1[1:] = vpd_col[:-1]

        # Medias y máximos móviles (usando Series de polars rolling)
        s_t = pl.Series(t_col)
        s_vpd = pl.Series(vpd_col)
        s_r = pl.Series(r_col)

        t_r3_mean = s_t.rolling_mean(window_size=3, min_samples=1).to_numpy().astype(np.float32)
        t_r3_max = s_t.rolling_max(window_size=3, min_samples=1).to_numpy().astype(np.float32)
        t_r7_mean = s_t.rolling_mean(window_size=7, min_samples=1).to_numpy().astype(np.float32)
        t_r7_max = s_t.rolling_max(window_size=7, min_samples=1).to_numpy().astype(np.float32)

        vpd_r3_mean = s_vpd.rolling_mean(window_size=3, min_samples=1).to_numpy().astype(np.float32)
        vpd_r3_max = s_vpd.rolling_max(window_size=3, min_samples=1).to_numpy().astype(np.float32)
        vpd_r7_mean = s_vpd.rolling_mean(window_size=7, min_samples=1).to_numpy().astype(np.float32)
        vpd_r7_max = s_vpd.rolling_max(window_size=7, min_samples=1).to_numpy().astype(np.float32)

        r_acc7 = s_r.rolling_sum(window_size=7, min_samples=1).to_numpy().astype(np.float32)
        r_acc30 = s_r.rolling_sum(window_size=30, min_samples=1).to_numpy().astype(np.float32)

        dsr25 = compute_days_since_rain_threshold(r_col, threshold_mm=2.5)

        node_df = pl.DataFrame({
            "node_id": [nid] * n_days,
            "date": dates_list,
            "temp_noon_c": t_col,
            "temp_lag1": t_lag1,
            "temp_roll3_mean": t_r3_mean,
            "temp_roll3_max": t_r3_max,
            "temp_roll7_mean": t_r7_mean,
            "temp_roll7_max": t_r7_max,
            "vpd_noon_kpa": vpd_col,
            "vpd_lag1": vpd_lag1,
            "vpd_roll3_mean": vpd_r3_mean,
            "vpd_roll3_max": vpd_r3_max,
            "vpd_roll7_mean": vpd_r7_mean,
            "vpd_roll7_max": vpd_r7_max,
            "rain_24h_noon_mm": r_col,
            "rain_acc7": r_acc7,
            "rain_acc30": r_acc30,
            "days_since_rain25": dsr25,
            "rh_noon_pct": rh_col,
            "wind_noon_kmh": w_col,
        })
        node_parts.append(node_df)

    extended_df = pl.concat(node_parts, how="vertical")

    # Unir los 6 índices FWI desde fwi_nodes_df
    fwi_sub = fwi_df.select(["node_id", "date", "fwi", "isi", "bui", "ffmc", "dmc", "dc"])
    merged = extended_df.join(fwi_sub, on=["node_id", "date"], how="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(out_path)
    log_event(
        logger,
        "extended_weather_saved",
        path=str(out_path),
        rows=merged.height,
        cols=len(merged.columns),
        elapsed_s=round(time.time() - t0, 2),
    )
    return merged
