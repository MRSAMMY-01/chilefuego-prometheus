"""Test de contrato y validación del pipeline de inferencia operativa (Fase 2D — G4).

PORQUÉ (Regla 5):
- G4 exige documentar y probar contractualmente cómo se calculan las 25 variables para un día nuevo,
  incluyendo el estado inicial del FWI, el prior con toda la historia previa y el mapeo de nodos a celdas.
- Valida la consistencia de tipos (float32), orden exacto de columnas y generación de clases de riesgo.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from chilefuego.inference.contract import (
    InferencePipeline,
    build_inference_feature_matrix,
    compute_vpd,
    get_operational_contract_documentation,
)
from chilefuego.models.xgboost_experiment import PRUNED_25_FEATURES


def test_vpd_calculation_known_values() -> None:
    # A 25°C, es ≈ 3.167 kPa. Con 50% RH, ea ≈ 1.583 kPa -> VPD ≈ 1.584 kPa
    vpd = compute_vpd(25.0, 50.0)
    assert 1.50 < vpd < 1.65, f"VPD fuera de rango esperado: {vpd}"

    # A 100% RH, VPD debe ser 0.0
    assert compute_vpd(30.0, 100.0) == pytest.approx(0.0, abs=1e-5)


def test_build_inference_feature_matrix_contract() -> None:
    n_test_cells = 10
    static_cols = {
        "elevation_mean": np.random.uniform(100, 2000, n_test_cells).astype(np.float32),
        "slope_mean": np.random.uniform(0, 30, n_test_cells).astype(np.float32),
        "northness": np.random.uniform(-1, 1, n_test_cells).astype(np.float32),
        "eastness": np.random.uniform(-1, 1, n_test_cells).astype(np.float32),
        "dist_road_nearest_m": np.random.uniform(10, 5000, n_test_cells).astype(np.float32),
        "dist_primary_sec_trunk_m": np.random.uniform(50, 20000, n_test_cells).astype(np.float32),
        "road_density_km2": np.random.uniform(0, 5, n_test_cells).astype(np.float32),
        "tree_cover_frac": np.random.uniform(0, 1, n_test_cells).astype(np.float32),
        "shrub_cover_frac": np.random.uniform(0, 1, n_test_cells).astype(np.float32),
        "grass_cover_frac": np.random.uniform(0, 1, n_test_cells).astype(np.float32),
        "crop_cover_frac": np.random.uniform(0, 1, n_test_cells).astype(np.float32),
        "built_cover_frac": np.random.uniform(0, 1, n_test_cells).astype(np.float32),
        "dist_built_nearest_m": np.random.uniform(10, 5000, n_test_cells).astype(np.float32),
    }
    static_df = pl.DataFrame(static_cols)
    prior_arr = np.random.uniform(0, 0.05, n_test_cells).astype(np.float32)

    weather_dict = {
        "temp_noon_c": np.random.uniform(15, 35, n_test_cells).astype(np.float32),
        "vpd_noon_kpa": np.random.uniform(0.5, 4.0, n_test_cells).astype(np.float32),
        "rain_24h_noon_mm": np.random.uniform(0, 5, n_test_cells).astype(np.float32),
        "rh_noon_pct": np.random.uniform(15, 80, n_test_cells).astype(np.float32),
        "wind_noon_kmh": np.random.uniform(5, 40, n_test_cells).astype(np.float32),
        "fwi": np.random.uniform(5, 50, n_test_cells).astype(np.float32),
        "isi": np.random.uniform(2, 20, n_test_cells).astype(np.float32),
        "bui": np.random.uniform(10, 80, n_test_cells).astype(np.float32),
        "ffmc": np.random.uniform(80, 95, n_test_cells).astype(np.float32),
        "dmc": np.random.uniform(10, 60, n_test_cells).astype(np.float32),
        "dc": np.random.uniform(50, 500, n_test_cells).astype(np.float32),
    }

    assembled_df = build_inference_feature_matrix(
        static_df=static_df,
        prior_series=prior_arr,
        weather_dict_by_cell=weather_dict,
    )

    assert assembled_df.columns == PRUNED_25_FEATURES
    assert assembled_df.height == n_test_cells
    for col, dtype in zip(assembled_df.columns, assembled_df.dtypes, strict=True):
        assert dtype == pl.Float32, f"Columna {col} tiene tipo {dtype}, se requería Float32"


def test_inference_pipeline_predictions() -> None:
    pipeline = InferencePipeline()

    sample_df = pl.read_parquet("data/processed/training_sample_phase2b.parquet").slice(0, 20)
    feat_df = sample_df.select(PRUNED_25_FEATURES)

    results = pipeline.predict(feat_df)

    assert "raw_score" in results
    assert "calibrated_prob" in results
    assert "risk_tier" in results

    assert len(results["raw_score"]) == 20
    assert len(results["calibrated_prob"]) == 20
    assert len(results["risk_tier"]) == 20

    # Probabilidades calibradas deben estar en [0, 1]
    assert np.all(results["calibrated_prob"] >= 0.0)
    assert np.all(results["calibrated_prob"] <= 1.0)

    # Clases deben ser del conjunto permitido
    valid_classes = {"Bajo", "Moderado", "Alto", "Extremo"}
    assert set(results["risk_tier"]).issubset(valid_classes)


def test_operational_contract_documentation() -> None:
    doc = get_operational_contract_documentation()
    assert doc["contract_version"] == "1.0.0"
    assert "era5_land_reanalysis" in doc["sources_and_latency"]
    assert "nwp_forecast_operational" in doc["sources_and_latency"]
    assert doc["spatial_grid"]["n_cells_nuble"] == 13108
