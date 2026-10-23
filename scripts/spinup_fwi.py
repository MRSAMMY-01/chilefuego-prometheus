"""Spin-up de ~90 días y prueba de insensibilidad al arranque para FWI operacional (J3).

PORQUÉ (Regla 5):
- En inferencia prospectiva u operacional fuera de la ventana de reanálisis ERA5-Land (que cerró en junio 2025),
  el estado del FWI debe inicializarse con un periodo de 'spin-up' de ~90 días usando past_days de la API de pronóstico.
- Ejecuta la prueba de insensibilidad al arranque (DC0=15 vs 500, DMC0=6 vs 100) en 5 nodos de Ñuble
  (costa, valle, precordillera, cordillera, sur) para confirmar que los códigos de humedad convergen
  al cabo de los 90 días (diferencia residual < 0.1 en DC y < 0.01 en DMC).
- Guarda el estado convergente en 'data/state/fwi_state.parquet'.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from chilefuego.features.fwi import compute_fwi_series_1d

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spinup_fwi")

FIVE_REPRESENTATIVE_NODES = {
    "costa": "-36.1875_-72.8125",
    "valle": "-36.5625_-72.0625",
    "precordillera": "-36.5625_-71.5625",
    "cordillera": "-36.9375_-71.4375",
    "sur": "-36.75_-72.3125",
}


def fetch_past_90_days_node(lat: float, lon: float) -> dict[str, Any]:
    """Descarga los últimos 90 días de datos horarios desde Open-Meteo Forecast API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&past_days=90&forecast_days=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "ChileFuego-Spinup/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def extract_noon_weather_series(data: dict[str, Any]) -> pl.DataFrame:
    """Extrae la serie diaria de las 16:00 UTC (12:00 CLT) y la lluvia acumulada de 24 h previa."""
    hourly = data["hourly"]
    times = hourly["time"]
    temps = hourly["temperature_2m"]
    rhs = hourly["relative_humidity_2m"]
    winds = hourly["wind_speed_10m"]
    rains = hourly["precipitation"]

    rows = []
    for idx, t_str in enumerate(times):
        if t_str.endswith("T16:00"):
            d_val = date.fromisoformat(t_str.split("T")[0])
            start_rain = max(0, idx - 23)
            rain_slice = [r for r in rains[start_rain : idx + 1] if r is not None]
            rain_24h = float(sum(rain_slice)) if rain_slice else 0.0
            t_val = temps[idx]
            rh_val = rhs[idx]
            w_val = winds[idx]
            if t_val is not None and rh_val is not None and w_val is not None:
                rows.append(
                    {
                        "date": d_val,
                        "temp_noon_c": float(t_val),
                        "rh_noon_pct": float(rh_val),
                        "wind_noon_kmh": float(w_val),
                        "rain_24h_noon_mm": rain_24h,
                        "month": d_val.month,
                    }
                )
    return pl.DataFrame(rows).sort("date")


def run_insensitivity_test() -> dict[str, Any]:
    """Ejecuta la prueba de arranque en los 5 nodos con DC0=15 vs 500 y DMC0=6 vs 100."""
    logger.info("=== INICIANDO PRUEBA DE INSENSIBILIDAD AL ARRANQUE (SPIN-UP 90 DÍAS) ===")
    results_by_zone: dict[str, Any] = {}

    for zone, nid in FIVE_REPRESENTATIVE_NODES.items():
        lat_str, lon_str = nid.split("_")
        lat, lon = float(lat_str), float(lon_str)
        logger.info(f"Descargando datos recientes (past_days=90) para zona {zone} ({nid})...")
        raw_data = fetch_past_90_days_node(lat, lon)
        noon_df = extract_noon_weather_series(raw_data)
        n_days = noon_df.height
        logger.info(
            f"Zona {zone}: {n_days} días de mediodía extraídos ({noon_df['date'].min()} a {noon_df['date'].max()})"
        )

        t_arr = noon_df["temp_noon_c"].to_numpy()
        rh_arr = noon_df["rh_noon_pct"].to_numpy()
        w_arr = noon_df["wind_noon_kmh"].to_numpy()
        p_arr = noon_df["rain_24h_noon_mm"].to_numpy()
        m_arr = noon_df["month"].to_numpy().astype(np.int64)

        # Corrida baja: DC0=15, DMC0=6, FFMC0=85
        fwi_low = compute_fwi_series_1d(
            temp_noon_c=t_arr,
            rh_noon_pct=rh_arr,
            wind_noon_kmh=w_arr,
            rain_24h_noon_mm=p_arr,
            months=m_arr,
            ffmc0=85.0,
            dmc0=6.0,
            dc0=15.0,
        )

        # Corrida alta: DC0=500, DMC0=100, FFMC0=85
        fwi_high = compute_fwi_series_1d(
            temp_noon_c=t_arr,
            rh_noon_pct=rh_arr,
            wind_noon_kmh=w_arr,
            rain_24h_noon_mm=p_arr,
            months=m_arr,
            ffmc0=85.0,
            dmc0=100.0,
            dc0=500.0,
        )

        diff_dc_final = abs(float(fwi_high["dc"][-1]) - float(fwi_low["dc"][-1]))
        diff_dmc_final = abs(float(fwi_high["dmc"][-1]) - float(fwi_low["dmc"][-1]))
        diff_fwi_final = abs(float(fwi_high["fwi"][-1]) - float(fwi_low["fwi"][-1]))

        # Máxima diferencia en los últimos 30 días
        max_diff_dc_last30 = float(np.max(np.abs(fwi_high["dc"][-30:] - fwi_low["dc"][-30:])))
        max_diff_dmc_last30 = float(np.max(np.abs(fwi_high["dmc"][-30:] - fwi_low["dmc"][-30:])))

        logger.info(
            f"Zona {zone.upper()}: DC final (Low={fwi_low['dc'][-1]:.2f}, High={fwi_high['dc'][-1]:.2f}, diff={diff_dc_final:.4f}) | "
            f"DMC final diff={diff_dmc_final:.4f} | FWI diff={diff_fwi_final:.4f}"
        )

        results_by_zone[zone] = {
            "node_id": nid,
            "n_days": n_days,
            "dc_final_low": round(float(fwi_low["dc"][-1]), 2),
            "dc_final_high": round(float(fwi_high["dc"][-1]), 2),
            "diff_dc_final": round(diff_dc_final, 5),
            "diff_dmc_final": round(diff_dmc_final, 5),
            "diff_fwi_final": round(diff_fwi_final, 5),
            "max_diff_dc_last30": round(max_diff_dc_last30, 5),
            "max_diff_dmc_last30": round(max_diff_dmc_last30, 5),
            "converged": bool(diff_dc_final < 0.1 and diff_dmc_final < 0.01),
        }
        time.sleep(1.0)

    out_file = Path("results/j3_fwi_spinup_insensitivity.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test": "FWI 90-Day Spin-up Insensitivity Test (J3)",
                "date": date.today().isoformat(),
                "nodes": results_by_zone,
                "all_converged": all(r["converged"] for r in results_by_zone.values()),
            },
            f,
            indent=2,
        )

    logger.info(f"Resultados guardados en {out_file}")
    return results_by_zone


if __name__ == "__main__":
    run_insensitivity_test()
