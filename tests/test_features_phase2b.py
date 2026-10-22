"""Tests rigurosos para la Fase 2B (C1 - C7).

Verifica:
1. C7 (a) Fuga del prior: Falla si alguna fila usa su propia temporada o una posterior.
2. C7 (b) Ventanas sin futuro: Rolling windows dependen exclusivamente de d <= D.
3. C7 (c) VPD contra valores teóricos de referencia de Tetens / Buck.
4. C7 (d) Northness y Eastness: Comportamiento en pendientes planas y ortogonalidad trigonométrica.
5. C7 (e) Reproducibilidad con semilla: Dos ejecuciones producen exactamente la misma muestra y hashes.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from chilefuego.config import load_config
from chilefuego.data.training_table import build_stratified_training_sample
from chilefuego.features.prior import get_prior_training_seasons_for_target
from chilefuego.features.topography import compute_slope_and_aspect
from chilefuego.features.weather_extended import compute_days_since_rain_threshold, compute_vpd_kpa
from chilefuego.labels.panel import DATED_SEASONS


def test_c7_fuga_del_prior() -> None:
    """Verifica que el prior espacial para la temporada s no use s ni temporadas posteriores."""
    for s in DATED_SEASONS:
        allowed = get_prior_training_seasons_for_target(s)
        target_year = int(s.split("_")[0])

        # 1. No debe contener la propia temporada
        self_season = f"if_temporada_{target_year}_{target_year+1}"
        assert self_season not in allowed, f"Fuga: {s} contiene su propia temporada {self_season}"

        # 2. No debe contener ninguna temporada posterior
        for seas in allowed:
            # Extraer año de 'if_temporada_YYYY_YYYY'
            parts = seas.split("_")
            y_start = int(parts[2])
            assert y_start < target_year, f"Fuga temporal futura: {s} incluye {seas}"


def test_c7_ventanas_sin_futuro() -> None:
    """Verifica que las ventanas móviles en el día D sean inmunes a alteraciones en días D+1..D+k."""
    # 20 días sintéticos de lluvia
    np.random.seed(999)
    rain = np.array([0.0, 5.0, 0.0, 0.0, 10.0, 0.0, 0.0, 1.0, 4.0, 0.0,
                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    s_rain = pl.Series("rain", rain)
    r_acc7_orig = s_rain.rolling_sum(window_size=7, min_samples=1).to_numpy()
    dsr25_orig = compute_days_since_rain_threshold(rain, threshold_mm=2.5)

    eval_day = 9  # Día 10 (índice 9)
    val_acc7_day9 = r_acc7_orig[eval_day]
    val_dsr25_day9 = dsr25_orig[eval_day]

    # Modificar agresivamente el futuro (días 10 a 19)
    rain_corrupted = rain.copy()
    rain_corrupted[10:] = 999.0  # Lluvias gigantescas en el futuro
    s_rain_corrupt = pl.Series("rain", rain_corrupted)
    r_acc7_corrupt = s_rain_corrupt.rolling_sum(window_size=7, min_samples=1).to_numpy()
    dsr25_corrupt = compute_days_since_rain_threshold(rain_corrupted, threshold_mm=2.5)

    # El valor en el día 9 DEBE permanecer 100% idéntico
    assert r_acc7_corrupt[eval_day] == pytest.approx(val_acc7_day9, abs=1e-6)
    assert dsr25_corrupt[eval_day] == val_dsr25_day9


def test_c7_vpd_contra_valor_referencia() -> None:
    """Verifica que el cálculo de VPD coincida con los valores de referencia física."""
    # Caso 1: T=20 °C, RH=50% -> es(20) ~ 2.338 kPa, VPD = 2.338 * 0.5 = 1.169 kPa
    vpd_case1 = compute_vpd_kpa(np.array([20.0]), np.array([50.0]))[0]
    assert vpd_case1 == pytest.approx(1.1691, abs=0.01)

    # Caso 2: T=20 °C, RH=100% -> VPD = 0 kPa
    vpd_case2 = compute_vpd_kpa(np.array([20.0]), np.array([100.0]))[0]
    assert vpd_case2 == pytest.approx(0.0, abs=1e-5)

    # Caso 3: T=30 °C, RH=30% -> es(30) ~ 4.243 kPa, VPD = 4.243 * 0.7 = 2.970 kPa
    vpd_case3 = compute_vpd_kpa(np.array([30.0]), np.array([30.0]))[0]
    assert vpd_case3 == pytest.approx(2.970, abs=0.01)


def test_c7_northness_eastness() -> None:
    """Verifica propiedades geométricas de northness y eastness."""
    # 1. Matriz plana constante (elevación idéntica)
    dem_flat = np.full((10, 10), 500.0, dtype=np.float32)
    slope_f, north_f, east_f, _ = compute_slope_and_aspect(dem_flat, res_m=30.0)
    assert np.all(slope_f == pytest.approx(0.0, abs=1e-4))
    assert np.all(north_f == 0.0)
    assert np.all(east_f == 0.0)

    # 2. Ladera inclinada que sube hacia el Sur (dz/dy < 0 hacia el Norte -> mira al Norte)
    y_coords, x_coords = np.mgrid[0:20, 0:20]
    dem_north_facing = 1000.0 - 5.0 * y_coords  # Cota disminuye al aumentar fila Y (hacia el Sur)
    slope_n, north_n, east_n, _ = compute_slope_and_aspect(dem_north_facing.astype(np.float32), res_m=30.0)
    # En la zona interior (lejos de bordes):
    inner_north = north_n[2:-2, 2:-2]
    inner_east = east_n[2:-2, 2:-2]
    # Laderas con aspect hacia el Norte deben tener northness cercano a 1 o -1 según convención
    assert np.all(np.abs(inner_north) > 0.8)
    assert np.all(np.abs(inner_east) < 0.1)


def test_c7_reproducibilidad_con_semilla() -> None:
    """Verifica que el muestreo estratificado con semilla fija sea 100% determinista."""
    cfg = load_config("configs/config.yaml")

    sample1 = build_stratified_training_sample(cfg, neg_to_pos_ratio=5, seed=1234, out_path=None)
    sample2 = build_stratified_training_sample(cfg, neg_to_pos_ratio=5, seed=1234, out_path=None)

    assert sample1.height == sample2.height
    assert sample1.get_column("cell_id").to_list() == sample2.get_column("cell_id").to_list()
    assert sample1.get_column("date").to_list() == sample2.get_column("date").to_list()
    assert np.allclose(sample1.get_column("sample_weight").to_numpy(), sample2.get_column("sample_weight").to_numpy())
