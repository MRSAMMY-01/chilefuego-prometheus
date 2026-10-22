"""Tests de propiedades físicas de las fracciones de cobertura WorldCover (E6c).

PORQUÉ (Regla 5):
- Cada celda de 1 km² está compuesta por píxeles mutuamente excluyentes (árboles, matorral, pastizal, cultivo, edificado, etc.).
- Las fracciones deben estar acotadas en [0, 1] y su suma no puede exceder 1.0 (tolerando épsilon numérico float32).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def test_worldcover_fractions_validity() -> None:
    wc_path = Path("data/processed/worldcover_features_1km.parquet")
    assert wc_path.exists(), f"No existe {wc_path}"

    df = pl.read_parquet(wc_path)
    assert df.height == 13108

    fraction_cols = [
        "tree_cover_frac",
        "shrub_cover_frac",
        "grass_cover_frac",
        "crop_cover_frac",
        "built_cover_frac",
    ]

    for col in fraction_cols:
        arr = df[col].to_numpy()
        assert np.all(arr >= 0.0), f"{col} contiene valores negativos"
        assert np.all(arr <= 1.0 + 1e-6), f"{col} excede 1.0"

    sum_fractions = np.zeros(df.height, dtype=np.float64)
    for col in fraction_cols:
        sum_fractions += df[col].to_numpy()

    # La suma de las 5 clases analizadas no puede exceder 1.0 (puede ser menor si hay agua, suelo desnudo o nieve)
    assert np.all(sum_fractions <= 1.0001), "La suma de fracciones excede 1.0"
    assert np.all(sum_fractions >= 0.0), "La suma de fracciones es negativa"
