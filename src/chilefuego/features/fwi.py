"""Cálculo del sistema canadiense Fire Weather Index (`FFMC`, `DMC`, `DC`, `ISI`, `BUI`, `FWI`) (Fase 2A — F2.5).

PORQUÉ (Regla 5):
- El peligro meteorológico de ignición depende tanto del estado instantáneo de los combustibles finos
  (`FFMC`, `ISI`) como de la sequedad acumulada a mediano y largo plazo (`DMC`, `DC`, `BUI`).
- Siguiendo la instrucción F2.5:
  1) Se usa una implementación madura (`xclim.indices.fire._cffwis` basada en Van Wagner & Pickett 1985 /
     Van Wagner 1987 / Wang et al. 2015, con ajuste de fotoperiodo del hemisferio sur para `lat = -36.6°`).
  2) Se evalúa **UNA sola regla uniforme para todos los registros** (sin tratamiento especial por hora):
     las entradas de los 6 códigos FWI se toman al **mediodía local estándar (`12:00 UTC-4` = `16:00 UTC`)**,
     con arranque de 3 meses (`2017-04-01` a `2017-06-30`, condiciones iniciales estándar `FFMC0 = 85.0`,
     `DMC0 = 6.0`, `DC0 = 15.0`) antes del inicio de la primera temporada fechada (`2017-07-01`).
  3) El vector de features diarias del día $D$ combina:
     - Agregados diarios (`tmax_c`, `rh_min_pct`, `wind_max_kmh`, `rain_24h_mm`, `swvl1_noon`,
       `days_since_rain`).
     - Los 6 códigos FWI de $D$ (`ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`).

CÓMO:
- `compute_fwi_series_1d()` ejecuta la recurrencia diaria exacta de `xclim.indices.fire._cffwis`
  (`_fine_fuel_moisture_code`, `_duff_moisture_code`, `_drought_code`, `initial_spread_index`,
  `build_up_index`, `fire_weather_index`) sobre arreglos 1D por punto espacial.
- `attach_fwi_to_daily_weather()` aplica `compute_fwi_series_1d()` a cada `point_id` de la tabla
  meteorológica diaria ordenada por `date`.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from xclim.indices.fire import _cffwis as xclim_cffwis

DEFAULT_FFMC0: float = 85.0
DEFAULT_DMC0: float = 6.0
DEFAULT_DC0: float = 15.0
NUBLE_DEFAULT_LAT: float = -36.6


def compute_fwi_series_1d(
    temp_noon_c: np.ndarray,
    rh_noon_pct: np.ndarray,
    wind_noon_kmh: np.ndarray,
    rain_24h_noon_mm: np.ndarray,
    months: np.ndarray,
    latitude_deg: float = NUBLE_DEFAULT_LAT,
    ffmc0: float = DEFAULT_FFMC0,
    dmc0: float = DEFAULT_DMC0,
    dc0: float = DEFAULT_DC0,
) -> dict[str, np.ndarray]:
    """Calcula las 6 componentes del Canadian FWI System día a día usando `xclim.indices.fire._cffwis`.

    Validado contra los valores de referencia de Van Wagner & Pickett (1985) / Van Wagner (1987):
    Para `T = 17.0 °C`, `RH = 42.0 %`, `Wind = 25.0 km/h`, `Rain = 0.0 mm`, `mes = 4`, `lat = 46.0°N`
    y estado inicial `(85.0, 6.0, 15.0)`:
    `FFMC = 87.6930`, `DMC = 8.5451`, `DC = 19.0140`, `ISI = 10.8537`, `BUI = 8.4904`, `FWI = 10.0964`.
    """
    t_arr = np.asarray(temp_noon_c, dtype=np.float64)
    rh_arr = np.clip(np.asarray(rh_noon_pct, dtype=np.float64), 0.0, 100.0)
    w_arr = np.maximum(0.0, np.asarray(wind_noon_kmh, dtype=np.float64))
    p_arr = np.maximum(0.0, np.asarray(rain_24h_noon_mm, dtype=np.float64))
    m_arr = np.asarray(months, dtype=np.int64)

    n = t_arr.shape[0]
    if not (rh_arr.shape[0] == w_arr.shape[0] == p_arr.shape[0] == m_arr.shape[0] == n):
        raise ValueError(
            "Todas las series de entrada a compute_fwi_series_1d deben tener igual longitud."
        )

    ffmc_out = np.empty(n, dtype=np.float64)
    dmc_out = np.empty(n, dtype=np.float64)
    dc_out = np.empty(n, dtype=np.float64)
    isi_out = np.empty(n, dtype=np.float64)
    bui_out = np.empty(n, dtype=np.float64)
    fwi_out = np.empty(n, dtype=np.float64)

    lat_1d = np.array([float(latitude_deg)], dtype=np.float64)
    ffmc_prev = np.array([float(ffmc0)], dtype=np.float64)
    dmc_prev = np.array([float(dmc0)], dtype=np.float64)
    dc_prev = np.array([float(dc0)], dtype=np.float64)

    for i in range(n):
        t_step = np.array([t_arr[i]], dtype=np.float64)
        p_step = np.array([p_arr[i]], dtype=np.float64)
        w_step = np.array([w_arr[i]], dtype=np.float64)
        rh_step = np.array([rh_arr[i]], dtype=np.float64)
        m_step = np.array([int(m_arr[i])], dtype=np.int64)

        ffmc_cur = np.asarray(
            xclim_cffwis._fine_fuel_moisture_code(t_step, p_step, w_step, rh_step, ffmc_prev),
            dtype=np.float64,
        )
        dmc_cur = np.asarray(
            xclim_cffwis._duff_moisture_code(t_step, p_step, rh_step, m_step, lat_1d, dmc_prev),
            dtype=np.float64,
        )
        dc_cur = np.asarray(
            xclim_cffwis._drought_code(t_step, p_step, m_step, lat_1d, dc_prev),
            dtype=np.float64,
        )
        isi_cur = np.asarray(
            xclim_cffwis.initial_spread_index(w_step, ffmc_cur),
            dtype=np.float64,
        )
        bui_cur = np.asarray(
            xclim_cffwis.build_up_index(dmc_cur, dc_cur),
            dtype=np.float64,
        )
        fwi_cur = np.asarray(
            xclim_cffwis.fire_weather_index(isi_cur, bui_cur),
            dtype=np.float64,
        )

        ffmc_out[i] = ffmc_cur[0]
        dmc_out[i] = dmc_cur[0]
        dc_out[i] = dc_cur[0]
        isi_out[i] = isi_cur[0]
        bui_out[i] = bui_cur[0]
        fwi_out[i] = fwi_cur[0]

        ffmc_prev = ffmc_cur
        dmc_prev = dmc_cur
        dc_prev = dc_cur

    return {
        "ffmc": ffmc_out,
        "dmc": dmc_out,
        "dc": dc_out,
        "isi": isi_out,
        "bui": bui_out,
        "fwi": fwi_out,
    }


def attach_fwi_to_daily_weather(
    daily_weather_df: pl.DataFrame,
    default_latitude_deg: float = NUBLE_DEFAULT_LAT,
    ffmc0: float = DEFAULT_FFMC0,
    dmc0: float = DEFAULT_DMC0,
    dc0: float = DEFAULT_DC0,
) -> pl.DataFrame:
    """Agrega las 6 columnas del sistema FWI (`ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`) por `point_id`."""
    if daily_weather_df.is_empty():
        return daily_weather_df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("ffmc"),
            pl.lit(None, dtype=pl.Float64).alias("dmc"),
            pl.lit(None, dtype=pl.Float64).alias("dc"),
            pl.lit(None, dtype=pl.Float64).alias("isi"),
            pl.lit(None, dtype=pl.Float64).alias("bui"),
            pl.lit(None, dtype=pl.Float64).alias("fwi"),
        )

    sorted_df = daily_weather_df.sort(["point_id", "date"])
    parts: list[pl.DataFrame] = []

    for _pid, sub in sorted_df.group_by("point_id", maintain_order=True):
        months = sub.get_column("date").dt.month().to_numpy()
        lat_val = default_latitude_deg
        if "latitude" in sub.columns:
            lat_val = float(sub.get_column("latitude")[0])

        res = compute_fwi_series_1d(
            temp_noon_c=sub.get_column("temp_noon_c").to_numpy(),
            rh_noon_pct=sub.get_column("rh_noon_pct").to_numpy(),
            wind_noon_kmh=sub.get_column("wind_noon_kmh").to_numpy(),
            rain_24h_noon_mm=sub.get_column("rain_24h_noon_mm").to_numpy(),
            months=months,
            latitude_deg=lat_val,
            ffmc0=ffmc0,
            dmc0=dmc0,
            dc0=dc0,
        )
        parts.append(
            sub.with_columns(
                pl.Series("ffmc", res["ffmc"], dtype=pl.Float64),
                pl.Series("dmc", res["dmc"], dtype=pl.Float64),
                pl.Series("dc", res["dc"], dtype=pl.Float64),
                pl.Series("isi", res["isi"], dtype=pl.Float64),
                pl.Series("bui", res["bui"], dtype=pl.Float64),
                pl.Series("fwi", res["fwi"], dtype=pl.Float64),
            )
        )

    return pl.concat(parts, how="vertical").sort(["point_id", "date"])
