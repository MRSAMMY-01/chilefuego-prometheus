"""Desacumulación horaria de precipitación ERA5-Land y agregación meteorológica diaria (Fase 2A — F2.4 / F2.5).

PORQUÉ (Regla 5):
- En el dataset oficial `reanalysis-era5-land` de ECMWF/Copernicus (verificado en ECMWF Confluence:
  *ERA5-Land: data documentation — Accumulated fields*), la precipitación total (`total_precipitation`,
  nombre corto NetCDF `tp`, unidad `m` de agua equivalente) **NO** viene como tasa horaria instantánea,
  sino **acumulada desde las `00:00 UTC` del inicio de cada ciclo diario de pronóstico**:
  * Paso `1` -> validez `01:00 UTC` del día $D$: contiene la lluvia entre `00:00` y `01:00 UTC` del día $D$.
  * Pasos `2..23` -> validez `02:00..23:00 UTC` del día $D$: contienen la lluvia acumulada desde `00:00 UTC`
    hasta esa hora del día $D$.
  * Paso `24` -> validez `00:00 UTC` del día $D+1$: contiene la lluvia acumulada de las **24 horas completas**
    del día $D$ (`00:00 UTC` de $D$ a `00:00 UTC` de $D+1$).
- Si se restaran pasos consecutivos sin distinguir el reinicio en `01:00 UTC` (`tp(01:00 UTC) - tp(00:00 UTC)`),
  se producirían picos negativos espurios en cada cambio de día UTC o se duplicaría la lluvia de 24 h.
- Para la Fase 2A (F2.5) se requiere **UNA sola regla uniforme para todos los registros** (sin tratamiento
  especial por hora de ignición):
  1) Estado meteorológico instantáneo al **mediodía local estándar (`12:00 UTC-4` = `16:00 UTC`)** para el
     sistema FWI canadiense (`temp_noon_c`, `rh_noon_pct`, `wind_noon_kmh`, y lluvia acumulada en las 24 h
     previas al mediodía `rain_24h_noon_mm` entre `16:00 UTC` de $D-1$ y `16:00 UTC` de $D$).
  2) Agregados diarios por día local estándar (`UTC-4`, `04:00 UTC` de $D$ a `03:00 UTC` de $D+1$):
     `tmax_c`, `rh_min_pct`, `wind_max_kmh`, `rain_24h_mm`, `swvl1_mean` y `days_since_rain`.

CÓMO:
- `deaccumulate_era5_land_tp_m_to_mm()` recibe un arreglo horario de `tp` (en metros) y sus horas de
  validez UTC (`0..23`) y calcula exactamente:
  * Si `utc_hour == 1` (o primer registro de la serie): `rain_mm[t] = max(0.0, tp[t] * 1000.0)`.
  * Si `utc_hour != 1`: `rain_mm[t] = max(0.0, (tp[t] - tp[t - 1]) * 1000.0)`.
- `relative_humidity_from_t2m_d2m_k()` aplica la fórmula de Magnus-Tetens (Alduchov & Eskridge 1996 / WMO)
  sobre `t2m` y `d2m` en Kelvin y acota a `[0.0, 100.0]`.
- `wind_speed_kmh_from_uv_ms()` convierte `(u10, v10)` en $\text{m s}^{-1}$ a rapidez en $\text{km h}^{-1}$
  ($3.6 \sqrt{u_{10}^2 + v_{10}^2}$).
- `compute_daily_weather_table()` transforma una tabla horaria `(era5_point_id, valid_time_utc, t2m, d2m,
  u10, v10, tp, swvl1)` en la tabla diaria uniforme por `(era5_point_id, date)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree


def deaccumulate_era5_land_tp_m_to_mm(
    tp_m: np.ndarray,
    utc_hours: np.ndarray,
) -> np.ndarray:
    """Desacumula la precipitación horaria `tp` (en metros) de ERA5-Land a milímetros horarios (`mm`).

    Parámetros:
    - `tp_m`: arreglo 1D (por punto espacial ordenado cronológicamente) o 2D `(n_times, n_points)`
      con la variable `tp` (`total_precipitation`) en metros de agua equivalente.
    - `utc_hours`: arreglo 1D `(n_times,)` con la hora de validez UTC (`0..23`) de cada paso.

    Regla oficial ERA5-Land:
    - En `utc_hour == 1` comienza un nuevo ciclo de acumulación desde `00:00 UTC`: el incremento
      de la hora `00:00 -> 01:00 UTC` es directamente `tp[t]`.
    - En `utc_hour in {2..23, 0}` el incremento de la hora `(t-1) -> t` es `tp[t] - tp[t-1]`.
    - Se aplica `np.maximum(0.0, ...)` para suprimir ruido numérico de punto flotante (`~ -1e-9 m`).
    """
    tp_arr = np.asarray(tp_m, dtype=np.float64)
    hours_arr = np.asarray(utc_hours, dtype=np.int64)
    if tp_arr.shape[0] != hours_arr.shape[0]:
        raise ValueError(
            f"Longitud de tp_m ({tp_arr.shape[0]}) no coincide con utc_hours ({hours_arr.shape[0]})."
        )
    if tp_arr.size == 0:
        return np.empty_like(tp_arr, dtype=np.float64)

    out_m = np.empty_like(tp_arr, dtype=np.float64)
    out_m[0] = tp_arr[0]
    if tp_arr.shape[0] > 1:
        diffs = tp_arr[1:] - tp_arr[:-1]
        is_reset = hours_arr[1:] == 1
        if tp_arr.ndim == 1:
            out_m[1:] = np.where(is_reset, tp_arr[1:], diffs)
        else:
            out_m[1:] = np.where(is_reset[:, None], tp_arr[1:], diffs)

    return np.maximum(0.0, out_m * 1000.0)


def relative_humidity_from_t2m_d2m_k(
    t2m_k: np.ndarray,
    d2m_k: np.ndarray,
) -> np.ndarray:
    """Calcula humedad relativa (%) a 2 m desde `t2m` y `d2m` (en Kelvin) vía Magnus-Tetens."""
    t_c = np.asarray(t2m_k, dtype=np.float64) - 273.15
    td_c = np.asarray(d2m_k, dtype=np.float64) - 273.15
    a = 17.625
    b = 243.04
    gamma_td = (a * td_c) / (b + td_c)
    gamma_t = (a * t_c) / (b + t_c)
    rh = 100.0 * np.exp(gamma_td - gamma_t)
    return np.clip(rh, 0.0, 100.0)


def wind_speed_kmh_from_uv_ms(
    u10_ms: np.ndarray,
    v10_ms: np.ndarray,
) -> np.ndarray:
    """Convierte componentes de viento a 10 m (`u10`, `v10` en m/s) a rapidez en km/h."""
    return 3.6 * np.hypot(
        np.asarray(u10_ms, dtype=np.float64),
        np.asarray(v10_ms, dtype=np.float64),
    )


def compute_days_since_rain(
    rain_24h_mm: np.ndarray,
    wet_threshold_mm: float = 1.0,
) -> np.ndarray:
    """Calcula para cada día $D$ el número de días consecutivos desde la última lluvia `>= wet_threshold_mm`.

    Si en el día $D$ llueve `rain_24h_mm[D] >= wet_threshold_mm`, `days_since_rain[D] = 0`.
    De lo contrario, incrementa en `+1` respecto al día $D-1$.
    """
    r = np.asarray(rain_24h_mm, dtype=np.float64)
    out = np.zeros(r.shape[0], dtype=np.int32)
    streak = 0
    for i in range(r.shape[0]):
        if r[i] >= wet_threshold_mm:
            streak = 0
        else:
            streak += 1
        out[i] = streak
    return out


def compute_daily_weather_table(
    hourly_df: pl.DataFrame,
    wet_threshold_mm: float = 1.0,
) -> pl.DataFrame:
    """Agrega una tabla horaria ERA5-Land a resolución diaria bajo UNA sola regla uniforme (F2.5).

    Columnas de entrada requeridas en `hourly_df`:
    - `point_id` (`String` o `Int64`)
    - `valid_time_utc` (`Datetime`, en UTC)
    - `t2m` (K), `d2m` (K), `u10` (m/s), `v10` (m/s), `tp` (m acumulados ERA5-Land), `swvl1` (m3/m3)

    Convención horaria única (sin excepción por hora de ignición):
    - Hora local estándar de Chile Continental: `UTC-4` fijo todo el año (`std_time = valid_time_utc - 4h`),
      tal que el mediodía local estándar `12:00 UTC-4` corresponde exactamente a `16:00 UTC`.
    - Lluvia de 24 h para el FWI al mediodía (`rain_24h_noon_mm`): suma de las 24 horas desacumuladas
      en `(12:00 UTC-4 de D-1, 12:00 UTC-4 de D]`, que se obtiene asignando `fwi_date = (std_time + 11h).dt.date()`.
    - Agregados diarios del día calendario local estándar `date = std_time.dt.date()`:
      `tmax_c`, `rh_min_pct`, `wind_max_kmh`, `rain_24h_mm`, `swvl1_mean`, `days_since_rain`.
    """
    if hourly_df.is_empty():
        return pl.DataFrame(
            schema={
                "point_id": pl.String,
                "date": pl.Date,
                "temp_noon_c": pl.Float64,
                "rh_noon_pct": pl.Float64,
                "wind_noon_kmh": pl.Float64,
                "rain_24h_noon_mm": pl.Float64,
                "tmax_c": pl.Float64,
                "rh_min_pct": pl.Float64,
                "wind_max_kmh": pl.Float64,
                "rain_24h_mm": pl.Float64,
                "swvl1_noon": pl.Float64,
                "days_since_rain": pl.Int32,
            }
        )

    sorted_df = hourly_df.with_columns(pl.col("point_id").cast(pl.String)).sort(
        ["point_id", "valid_time_utc"]
    )

    # Desacumular tp por cada point_id respetando el ciclo 01:00..00:00 UTC
    parts: list[pl.DataFrame] = []
    for _pid, sub in sorted_df.group_by("point_id", maintain_order=True):
        tp_vals = sub.get_column("tp").to_numpy()
        utc_hrs = sub.get_column("valid_time_utc").dt.hour().to_numpy()
        rain_hr_mm = deaccumulate_era5_land_tp_m_to_mm(tp_vals, utc_hrs)
        t2m_k = sub.get_column("t2m").to_numpy()
        d2m_k = sub.get_column("d2m").to_numpy()
        u10 = sub.get_column("u10").to_numpy()
        v10 = sub.get_column("v10").to_numpy()

        temp_c = t2m_k - 273.15
        rh_pct = relative_humidity_from_t2m_d2m_k(t2m_k, d2m_k)
        wind_kmh = wind_speed_kmh_from_uv_ms(u10, v10)

        parts.append(
            sub.with_columns(
                pl.Series("temp_c", temp_c, dtype=pl.Float64),
                pl.Series("rh_pct", rh_pct, dtype=pl.Float64),
                pl.Series("wind_kmh", wind_kmh, dtype=pl.Float64),
                pl.Series("rain_hr_mm", rain_hr_mm, dtype=pl.Float64),
                (pl.col("valid_time_utc") - pl.duration(hours=4)).alias("_std_ts"),
            )
        )

    enriched = pl.concat(parts, how="vertical").with_columns(
        pl.col("_std_ts").dt.date().alias("date"),
        pl.col("_std_ts").dt.hour().alias("_std_hour"),
        # Ventana (12:00 UTC-4 de D-1, 12:00 UTC-4 de D]: restar 1h para incluir 12:00 en D y desplazar +12h
        (pl.col("_std_ts") + pl.duration(hours=11)).dt.date().alias("_noon_window_date"),
    )

    # Valores al mediodía local estándar (12:00 UTC-4 = 16:00 UTC)
    noon_df = (
        enriched.filter(pl.col("_std_hour") == 12)
        .group_by(["point_id", "date"])
        .agg(
            pl.col("temp_c").first().alias("temp_noon_c"),
            pl.col("rh_pct").first().alias("rh_noon_pct"),
            pl.col("wind_kmh").first().alias("wind_noon_kmh"),
            pl.col("swvl1").first().cast(pl.Float64).alias("swvl1_noon"),
        )
    )

    # Lluvia de 24 h acumulada hasta el mediodía local estándar (12:00 UTC-4)
    noon_rain_df = (
        enriched.group_by(["point_id", "_noon_window_date"])
        .agg(pl.col("rain_hr_mm").sum().cast(pl.Float64).alias("rain_24h_noon_mm"))
        .rename({"_noon_window_date": "date"})
    )

    # Agregados diarios del día calendario local estándar (00:00..23:00 UTC-4)
    daily_agg_df = enriched.group_by(["point_id", "date"]).agg(
        pl.col("temp_c").max().cast(pl.Float64).alias("tmax_c"),
        pl.col("rh_pct").min().cast(pl.Float64).alias("rh_min_pct"),
        pl.col("wind_kmh").max().cast(pl.Float64).alias("wind_max_kmh"),
        pl.col("rain_hr_mm").sum().cast(pl.Float64).alias("rain_24h_mm"),
    )

    merged = (
        daily_agg_df.join(noon_df, on=["point_id", "date"], how="inner")
        .join(noon_rain_df, on=["point_id", "date"], how="left")
        .with_columns(pl.col("rain_24h_noon_mm").fill_null(pl.col("rain_24h_mm")))
        .sort(["point_id", "date"])
    )

    out_parts: list[pl.DataFrame] = []
    for _pid, sub_d in merged.group_by("point_id", maintain_order=True):
        dsr = compute_days_since_rain(
            sub_d.get_column("rain_24h_mm").to_numpy(),
            wet_threshold_mm=wet_threshold_mm,
        )
        out_parts.append(sub_d.with_columns(pl.Series("days_since_rain", dsr, dtype=pl.Int32)))

    return (
        pl.concat(out_parts, how="vertical")
        .select(
            [
                "point_id",
                "date",
                "temp_noon_c",
                "rh_noon_pct",
                "wind_noon_kmh",
                "rain_24h_noon_mm",
                "tmax_c",
                "rh_min_pct",
                "wind_max_kmh",
                "rain_24h_mm",
                "swvl1_noon",
                "days_since_rain",
            ]
        )
        .sort(["point_id", "date"])
    )


def map_grid_cells_to_era5_points(
    grid_df: pl.DataFrame,
    era5_points_df: pl.DataFrame,
) -> pl.DataFrame:
    """Asocia cada celda de 1 km (`in_nuble`) a su nodo ERA5-Land más cercano en coordenadas geográficas."""
    from pyproj import Transformer

    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    to_wgs84 = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = to_wgs84.transform(
        nuble_cells.get_column("centroid_x").to_numpy(),
        nuble_cells.get_column("centroid_y").to_numpy(),
    )

    pts_lon = era5_points_df.get_column("longitude").to_numpy()
    pts_lat = era5_points_df.get_column("latitude").to_numpy()
    pts_ids = era5_points_df.get_column("point_id").cast(pl.String).to_list()

    tree = cKDTree(np.column_stack([pts_lon, pts_lat]))
    _dists, indices = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    mapped_pids = [pts_ids[int(i)] for i in indices]

    return pl.DataFrame(
        {
            "cell_id": nuble_cells.get_column("cell_id").to_list(),
            "point_id": mapped_pids,
        }
    )


def load_era5_netcdfs_to_hourly_df(era5_dir: Path) -> pl.DataFrame:
    """Carga los archivos NetCDF de `data/raw/era5_land/` en un DataFrame horario Polars."""
    import xarray as xr

    nc_files = sorted(p for p in era5_dir.glob("*.nc") if not p.name.startswith("_"))
    if not nc_files:
        raise FileNotFoundError(f"No se encontraron archivos NetCDF mensuales en '{era5_dir}'.")

    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    df_pd = ds.to_dataframe().reset_index()
    ds.close()

    time_col = "valid_time" if "valid_time" in df_pd.columns else "time"
    pl_df = pl.from_pandas(df_pd).rename({time_col: "valid_time_utc"})
    pl_df = pl_df.with_columns(
        pl.format(
            "{}_{}",
            pl.col("latitude").round(2),
            pl.col("longitude").round(2),
        ).alias("point_id")
    )
    return pl_df
