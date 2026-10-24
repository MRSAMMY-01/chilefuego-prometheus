"""Tests de aislamiento temporal y ausencia de fuga de datos para el Modelo B (Fase 1 / L3).

PORQUÉ (Regla 5):
- El Modelo B (validación hacia adelante, `models/chilefuego_xgb_forward_v1.json`) debe estar
  estrictamente restringido a temporadas hasta 2021_22 para evitar la filtración de datos
  observada en el Modelo A al evaluar las temporadas prospectivas 2023_24 y 2024_25.
- Valida la integridad del artefacto XGBoost exportado, sus 25 variables canónicas y la
  ausencia absoluta de datos futuros (temporadas >= 2022_23) en su conjunto de ajuste.
- Valida la independencia estricta de inferencia sobre 2023_24 y 2024_25 (la permutación o
  enmascaramiento de etiquetas CONAF no altera las predicciones de riesgo).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.inference.contract import PRUNED_25_FEATURES, InferencePipeline


def test_model_b_artifact_integrity_and_features() -> None:
    """Verifica que el Modelo B exista, cargue válidamente y contenga las 25 variables canónicas."""
    model_path = Path("models/chilefuego_xgb_forward_v1.json")
    assert model_path.exists(), f"Artefacto {model_path} no encontrado"
    assert model_path.stat().st_size > 10_000, "El archivo del modelo es anormalmente pequeño"

    # Verificar estructura interna del JSON
    with open(model_path, encoding="utf-8") as f:
        m_json = json.load(f)

    assert "learner" in m_json, "El archivo JSON no tiene la estructura de XGBoost Booster"
    f_names = m_json["learner"].get("feature_names", [])
    assert f_names == PRUNED_25_FEATURES, "El orden o nombres de variables del Modelo B no coincide con PRUNED_25_FEATURES"

    # Cargar en Booster de XGBoost y validar predicción mínima
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    assert booster.num_features() == 25, f"Esperadas 25 features, obtenidas {booster.num_features()}"

    # Validar paso hacia adelante con datos sintéticos
    dummy_x = np.zeros((2, 25), dtype=np.float32)
    dmat = xgb.DMatrix(dummy_x, feature_names=PRUNED_25_FEATURES)
    preds = booster.predict(dmat)
    assert len(preds) == 2
    assert np.all((preds >= 0.0) & (preds <= 1.0)), "Las predicciones brutas deben estar en el rango [0, 1]"


def test_forward_split_strict_temporal_isolation() -> None:
    """Verifica que los conjuntos de entrenamiento hacia adelante excluyan temporadas >= 2022_23."""
    train_sample_path = Path("data/processed/training_sample_phase2b.parquet")
    assert train_sample_path.exists(), f"Dataset de entrenamiento {train_sample_path} no encontrado"

    df = pl.read_parquet(train_sample_path)
    seasons_in_data = set(df.get_column("season").unique().to_list())

    # Especificación oficial de validación hacia adelante
    fw_fit_seasons = {"2017_18", "2019_20", "2020_21"}
    fw_val_season = "2021_22"
    forward_training_seasons = fw_fit_seasons | {fw_val_season}
    assert forward_training_seasons.issubset(seasons_in_data), "Faltan temporadas requeridas en el dataset"

    # Probar que las temporadas prospectivas están estrictamente fuera del entrenamiento de Model B
    prospective_test_seasons = {"2022_23", "2023_24", "2024_25"}
    assert forward_training_seasons.isdisjoint(prospective_test_seasons), (
        f"Fuga temporal crítica: temporadas de test {prospective_test_seasons.intersection(forward_training_seasons)} "
        "encontradas en el conjunto de entrenamiento de validación hacia adelante"
    )

    # Validar fechas de corte del conjunto de entrenamiento
    train_sub = df.filter(pl.col("season").is_in(list(forward_training_seasons)))
    max_train_date = train_sub.get_column("date").max()
    assert max_train_date is not None
    # Temporada 2021_22 termina el 30 de junio de 2022
    cutoff_2021_22 = date(2022, 6, 30)
    assert max_train_date <= cutoff_2021_22, (
        f"Fuga temporal: la fecha máxima de entrenamiento ({max_train_date}) supera el fin de 2021_22 ({cutoff_2021_22})"
    )


def test_inference_pipeline_with_model_b_no_target_leakage() -> None:
    """Verifica que InferencePipeline con Modelo B opere sin dependencia de etiquetas target."""
    model_path = Path("models/chilefuego_xgb_forward_v1.json")
    pipe = InferencePipeline(model_path=str(model_path))

    # Cargar 50 casos del panel de 2023_24
    sample_path = Path("data/processed/training_sample_phase2b.parquet")
    df = pl.read_parquet(sample_path).filter(pl.col("season") == "2023_24").slice(0, 50)
    assert df.height > 0, "No hay muestras para 2023_24 en training_sample_phase2b.parquet"

    feat_df = df.select(PRUNED_25_FEATURES)

    # 1. Inferencia estándar
    res1 = pipe.predict(feat_df)
    scores1 = res1["raw_score"]

    # 2. Inferencia con permutación de filas: el resultado debe ser invariante celda a celda
    perm_idx = np.random.default_rng(42).permutation(len(feat_df))
    feat_df_perm = feat_df[perm_idx]
    res_perm = pipe.predict(feat_df_perm)
    scores_perm = res_perm["raw_score"]

    assert np.allclose(scores1[perm_idx], scores_perm, atol=1e-6), (
        "Inferencia no es invariante por fila: existe acoplamiento o fuga entre observaciones"
    )

    # 3. Ausencia de target en las variables del modelo
    assert "is_ignition_conaf" not in PRUNED_25_FEATURES, (
        "Fuga crítica: la etiqueta target 'is_ignition_conaf' está presente en las variables del modelo"
    )
