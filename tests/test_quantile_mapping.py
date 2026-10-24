r"""Tests de verificación matemática y estabilidad para MonotonicQuantileMapper (Fase 2 / K4).

PORQUÉ (Regla 5):
- El Mapeo Monótono de Cuantiles (Q-Q mapping) corrige la inflación o deflación de celdas
  en niveles de riesgo extremo causada por sesgos distributivos entre NWP y reanálisis.
- Debe preservar estrictamente el ordenamiento de riesgo: si el pronóstico indica que la celda A
  es más peligrosa que la celda B ($s_A > s_B$), el mapeo jamás puede invertir el orden ($T(s_A) \ge T(s_B)$).
- Debe garantizar rank preservation cuantitativo (correlación de Spearman >= 0.9999).
- En los extremos (valores fuera del soporte empírico de entrenamiento), no debe colapsar en un
  valor constante plano (flat clipping), ya que ello crearía empates artificiales e invalidaría
  la priorización operativa de los eventos de riesgo más extremos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.run_nwp_skew_study import MonotonicQuantileMapper


def test_monotonicity_10000_samples() -> None:
    """Verifica que el mapeador sea estrictamente no decreciente en 10.000 muestras sintéticas."""
    rng = np.random.default_rng(42)

    # Distribución fuente (NWP con sesgo positivo) y referencia (ERA5)
    source_train = rng.gamma(shape=2.0, scale=0.002, size=15_000)
    target_train = rng.gamma(shape=1.8, scale=0.0015, size=15_000)

    mapper = MonotonicQuantileMapper(n_quantiles=2000).fit(source_train, target_train)

    # 10.000 puntajes ordenados de prueba
    scores_test = np.sort(rng.uniform(0.0001, 0.02, size=10_000))
    mapped_test = mapper.transform(scores_test)

    diffs = np.diff(mapped_test)
    assert np.all(diffs >= 0.0), f"Violación de monotonía: existen {np.sum(diffs < 0)} caídas en puntajes ordenados"


def test_rank_preservation_spearman() -> None:
    """Verifica que el mapeador preserve el orden de rangos con correlación de Spearman >= 0.9999."""
    rng = np.random.default_rng(123)

    source_train = rng.normal(loc=0.05, scale=0.015, size=20_000)
    target_train = rng.normal(loc=0.03, scale=0.008, size=20_000)

    mapper = MonotonicQuantileMapper(n_quantiles=2500).fit(source_train, target_train)

    # Puntajes de prueba no ordenados
    test_scores = rng.normal(loc=0.05, scale=0.015, size=5_000)
    mapped_scores = mapper.transform(test_scores)

    rho, _ = spearmanr(test_scores, mapped_scores)
    assert rho >= 0.9999, f"Pérdida inaceptable de orden de rangos: Spearman rho = {rho:.6f} < 0.9999"


def test_distribution_alignment_percentiles() -> None:
    """Verifica que los puntajes mapeados reproduzcan los percentiles de referencia dentro de tolerancia."""
    rng = np.random.default_rng(999)

    source_train = rng.lognormal(mean=-5.0, sigma=0.6, size=50_000)
    target_train = rng.lognormal(mean=-5.5, sigma=0.4, size=50_000)

    mapper = MonotonicQuantileMapper(n_quantiles=3000).fit(source_train, target_train)

    # Evaluación fuera de muestra de la misma distribución fuente
    source_eval = rng.lognormal(mean=-5.0, sigma=0.6, size=50_000)
    mapped_eval = mapper.transform(source_eval)

    quantiles_to_check = [80.0, 95.0, 99.0]
    expected_pcts = np.percentile(target_train, quantiles_to_check)
    actual_pcts = np.percentile(mapped_eval, quantiles_to_check)

    for q, exp, act in zip(quantiles_to_check, expected_pcts, actual_pcts, strict=True):
        rel_err = abs(act - exp) / exp
        assert rel_err < 0.02, (
            f"Error excesivo en percentil p{int(q)}: esperado {exp:.6f}, obtenido {act:.6f} (error rel: {rel_err:.2%})"
        )


def test_boundary_extrapolation_strictly_monotonic() -> None:
    """Verifica que fuera del rango empírico de entrenamiento los puntajes no colapsen en empates planos.

    PORQUÉ:
    - Si un valor excede el cuantil máximo observado en entrenamiento, no debe colapsar a una constante,
      pues impediría ordenar las celdas de riesgo más crítico del evento.
    - La extrapolación debe ser estrictamente monótona creciente: x1 < x2 => T(x1) < T(x2).
    """
    source_train = np.linspace(0.1, 0.9, 1000)
    target_train = np.linspace(0.2, 0.95, 1000)

    mapper = MonotonicQuantileMapper(n_quantiles=500).fit(source_train, target_train)

    u_min = float(mapper.u_nwp[0])
    u_max = float(mapper.u_nwp[-1])

    # Puntos estrictamente superiores a u_max
    high_x1 = u_max + 0.05
    high_x2 = u_max + 0.10
    high_x3 = u_max + 0.20

    # Puntos estrictamente inferiores a u_min
    low_x1 = u_min - 0.08
    low_x2 = u_min - 0.04
    low_x3 = u_min - 0.01

    eval_points = np.array([low_x1, low_x2, low_x3, u_min, u_max, high_x1, high_x2, high_x3])
    mapped_points = mapper.transform(eval_points)

    # Validar que los puntos superiores no sean idénticos (sin flat clipping)
    assert mapped_points[5] < mapped_points[6] < mapped_points[7], (
        f"Colapso de extrapolación superior: {mapped_points[5:]} colapsaron a un valor constante"
    )

    # Validar que los puntos inferiores no sean idénticos (sin flat clipping)
    assert mapped_points[0] < mapped_points[1] < mapped_points[2], (
        f"Colapso de extrapolación inferior: {mapped_points[:3]} colapsaron a un valor constante"
    )

    # Validar ordenamiento estricto a través de toda la frontera
    assert np.all(np.diff(mapped_points) > 0.0), (
        f"La transformación no es estrictamente monótona en las fronteras: diffs = {np.diff(mapped_points)}"
    )
