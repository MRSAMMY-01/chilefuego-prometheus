"""Test de regresión para el corte de temporada forestal CONAF (F6).

PORQUÉ (Regla 5):
- La temporada forestal/hidrológica en Chile se define estrictamente del 1 de julio (año Y)
  al 30 de junio (año Y+1).
- Un evento ocurrido el 30 de junio a las 23:59 pertenece a la temporada que cierra (Y-1_Y),
  mientras que un evento el 1 de julio a las 00:00 pertenece a la nueva temporada (Y_Y+1).
- En versiones preliminares, 28 eventos de junio de 2017 fueron erróneamente atribuidos a 2017_18
  (dando 444 en vez de 416), y correspondientemente en 2024_25 se reportaron 409 en vez de 437.
- Este test fija contractualmente la función canónica `season_from_date_expr` y los conteos de corte.
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from chilefuego.config import load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    season_from_date_expr,
)


@pytest.mark.parametrize(
    ("test_date", "expected_season"),
    [
        (date(2017, 6, 30), "2016_17"),
        (date(2017, 7, 1), "2017_18"),
        (date(2023, 6, 30), "2022_23"),
        (date(2023, 7, 1), "2023_24"),
        (date(2024, 6, 30), "2023_24"),
        (date(2024, 7, 1), "2024_25"),
        (date(2025, 6, 30), "2024_25"),
        (date(2025, 7, 1), "2025_26"),
    ],
)
def test_season_boundary_cutoff(test_date: date, expected_season: str) -> None:
    """Verifica que el 30-jun pertenezca a la temporada saliente y 01-jul a la entrante."""
    df = pl.DataFrame({"date": [test_date]}).with_columns(season_from_date_expr("date").alias("season"))
    actual_season = df.get_column("season")[0]
    assert actual_season == expected_season, f"Error en fecha {test_date}: esperada {expected_season}, obtenida {actual_season}"


def test_conaf_season_positive_counts() -> None:
    """Verifica los conteos exactos de positivos bajo el corte canónico del 1 de julio."""
    cfg = load_config("configs/config.yaml")
    special_cell_days_df, summary = build_phase2a_daily_panel(cfg)

    # 3.237 positivos en total
    assert summary.total_positive_cell_days == 3237

    # Conteos exactos por pliegue fechado
    season_counts = dict(
        special_cell_days_df.filter(pl.col("is_ignition_conaf") == 1)
        .group_by("season")
        .len()
        .iter_rows()
    )

    expected_counts = {
        "2017_18": 416,  # 416 (no 444)
        "2019_20": 471,
        "2020_21": 469,
        "2021_22": 516,
        "2022_23": 584,
        "2023_24": 344,
        "2024_25": 437,  # 437 (no 409)
    }

    for s in DATED_SEASONS:
        assert season_counts.get(s, 0) == expected_counts[s], (
            f"Conteo incorrecto en temporada {s}: esperados {expected_counts[s]}, obtenidos {season_counts.get(s, 0)}"
        )
