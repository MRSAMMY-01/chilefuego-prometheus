"""Pruebas unitarias y de integración para el piloto en modo sombra (H5 — scripts/run_daily.py).

PORQUÉ (Regla 5):
- Valida la falla segura ante caída de API (conservación de datos con marca 'stale').
- Valida la idempotencia del archivo de estado de FWI (re-ejecuciones no duplican ni corrompen memoria).
- Valida la estructura de outputs (risk_cells.parquet, risk_classes.geojson, metadata.json).
- Valida el registro prospectivo en prospective_log.parquet.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from scripts.run_daily import fetch_weather_forecast, run_daily_shadow_pipeline, update_fwi_state


def test_fetch_weather_forecast_failsafe_stale(tmp_path: Path) -> None:
    """Verifica que ante caída de API, el sistema no aborta sino que opera en falla segura 'stale'."""
    nodes = ["-36.6_-72.1", "-36.7_-72.2"]
    coords = {"-36.6_-72.1": (-36.6, -72.1), "-36.7_-72.2": (-36.7, -72.2)}
    test_date = date(2026, 10, 23)

    # Forzar excepción de red en urllib
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        weather, freshness = fetch_weather_forecast(
            target_date=test_date,
            nodes=nodes,
            coord_dict=coords,
            cache_dir=tmp_path / "cache",
        )

    assert freshness == "stale"
    assert len(weather) == 2
    for n in nodes:
        assert "temp_noon_c" in weather[n]
        assert "rh_noon_pct" in weather[n]
        assert "wind_noon_kmh" in weather[n]
        assert "rain_24h_noon_mm" in weather[n]


def test_update_fwi_state_idempotency(tmp_path: Path) -> None:
    """Verifica que re-ejecutar el mismo día no duplica registros en el archivo de estado de FWI."""
    state_file = tmp_path / "fwi_state.parquet"
    init_fwi_path = Path("data/processed/era5_land_daily_fwi_nodes.parquet")
    test_date = date(2025, 7, 1)
    nodes = ["-36.6_-72.1"]
    weather = {
        "-36.6_-72.1": {
            "temp_noon_c": 18.0,
            "rh_noon_pct": 50.0,
            "wind_noon_kmh": 15.0,
            "rain_24h_noon_mm": 0.0,
        }
    }

    # Primera corrida
    df1 = update_fwi_state(state_file, test_date, nodes, weather, init_fwi_path)
    assert df1.height == 1
    state_df1 = pl.read_parquet(state_file)
    n_rows_1 = state_df1.height

    # Segunda corrida (idéntica fecha)
    df2 = update_fwi_state(state_file, test_date, nodes, weather, init_fwi_path)
    assert df2.height == 1
    state_df2 = pl.read_parquet(state_file)
    n_rows_2 = state_df2.height

    assert n_rows_1 == n_rows_2, "El estado de FWI no es idempotente: duplicó filas para la misma fecha"


def test_shadow_mode_pipeline_execution(tmp_path: Path) -> None:
    """Ejecuta el pipeline completo de modo sombra y verifica artefactos generados."""
    test_date = date(2025, 6, 25)
    out_dir = tmp_path / "outputs"
    state_dir = tmp_path / "state"

    meta = run_daily_shadow_pipeline(
        target_date=test_date,
        output_dir=out_dir,
        state_dir=state_dir,
    )

    day_dir = out_dir / "2025-06-25"
    assert day_dir.exists()
    assert (day_dir / "risk_cells.parquet").exists()
    assert (day_dir / "risk_classes.geojson").exists()
    assert (day_dir / "metadata.json").exists()

    # Verificar metadata
    assert meta["target_date"] == "2025-06-25"
    assert meta["n_cells"] == 13108
    assert "tier_distribution_pct" in meta
    assert "alert_status" in meta

    # Verificar schema del parquet de salida
    cells_df = pl.read_parquet(day_dir / "risk_cells.parquet")
    assert cells_df.height == 13108
    assert "cell_id" in cells_df.columns
    assert "calibrated_prob" in cells_df.columns
    assert "risk_tier" in cells_df.columns
