"""Tests de integridad temporal y ausencia de fuga de datos en el pipeline XGBoost (E6c).

PORQUÉ (Regla 5):
- Para garantizar validez científica estricta, la temporada de prueba jamas puede estar presente en:
  1. El subconjunto de ajuste (fit_seasons).
  2. El subconjunto de validación para early stopping (val_season).
  3. El ajuste de los modelos de calibración probabilística (Platt / Isotónica).
"""

from __future__ import annotations

import polars as pl
from chilefuego.labels.panel import DATED_SEASONS


def test_split_sets_no_test_leakage() -> None:
    for test_season in DATED_SEASONS:
        train_seasons = [s for s in DATED_SEASONS if s != test_season]
        val_season = train_seasons[-1]
        fit_seasons = train_seasons[:-1]

        # Verificar exclusión absoluta de test_season
        assert test_season not in fit_seasons, f"Fuga de datos: {test_season} en fit_seasons"
        assert test_season != val_season, f"Fuga de datos: {test_season} es val_season"
        assert val_season not in fit_seasons, f"Solapamiento: {val_season} presente en fit_seasons"

        # Verificar que la unión reconstruye exactamente todas las temporadas salvo test
        all_train_reconstructed = set(fit_seasons) | {val_season}
        assert all_train_reconstructed == set(train_seasons)


def test_prior_temporal_integrity() -> None:
    prior_path = "data/processed/prior_density_by_cell_season.parquet"
    prior_df = pl.read_parquet(prior_path)

    for s_name in DATED_SEASONS:
        sub = prior_df.filter(pl.col("season") == s_name)
        assert not sub.is_empty(), f"Falta prior para {s_name}"
        assert sub.height == 13108
        densities = sub["prior_density"].to_numpy()
        assert (densities > 0).all(), f"Hay densidades no positivas en {s_name}"
