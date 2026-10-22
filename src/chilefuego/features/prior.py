"""Cálculo del prior espacial creciente por (celda, temporada) (Fase 2B — C4).

PORQUÉ (Regla 5):
- La densidad espacial histórica de igniciones antrópicas cambia gradualmente a medida que se acumula
  evidencia en temporadas anteriores.
- CERO FUGA TEMPORAL: Para cualquier fila de la temporada s, el prior debe calcularse estrictamente
  con temporadas anteriores:
  * Temporadas base 2010–2016 (7 temporadas).
  * Temporada 2018_19 únicamente si s >= 2019_20.
  * Temporadas fechadas estrictamente anteriores a s (< s).
  * JAMÁS se utilizan igniciones de la propia temporada s ni de temporadas futuras (> s).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from chilefuego.labels.panel import DATED_SEASONS
from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)


def get_prior_training_seasons_for_target(target_season: str) -> list[str]:
    """Retorna la lista de temporadas CONAF estrictamente anteriores a target_season."""
    target_start_year = int(target_season.split("_")[0])

    base_seasons = [f"if_temporada_{y}_{y+1}" for y in range(2010, 2017)]
    if target_start_year >= 2019:
        base_seasons.append("if_temporada_2018_2019")

    for ds in DATED_SEASONS:
        ds_year = int(ds.split("_")[0])
        if ds_year < target_start_year:
            base_seasons.append(f"if_temporada_{ds_year}_{ds_year+1}")

    return sorted(base_seasons)


def compute_prior_density_table(
    grid_df: pl.DataFrame,
    ignitions_conaf_df: pl.DataFrame,
    bandwidth_m: float = 3000.0,
    out_path: Path | None = Path("data/processed/prior_density_by_cell_season.parquet"),
) -> pl.DataFrame:
    """Calcula `prior_density` para cada par (celda, temporada fechada) con ventana creciente."""
    nuble_grid = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    cell_ids = nuble_grid.get_column("cell_id").to_list()
    n_cells = len(cell_ids)

    grid_xy = np.column_stack(
        [
            nuble_grid.get_column("centroid_x").to_numpy(),
            nuble_grid.get_column("centroid_y").to_numpy(),
        ]
    )
    tree = cKDTree(grid_xy)
    sigma = float(bandwidth_m)
    cutoff = 3.0 * sigma

    qf = ignitions_conaf_df.get_column("quality_flags")
    valid_conaf = ignitions_conaf_df.with_columns(
        qf.struct.field("crs_ok").alias("_crs_ok"),
        qf.struct.field("dup_removed").alias("_dup_removed"),
    ).filter(pl.col("_crs_ok") & (~pl.col("_dup_removed")))

    season_dfs: list[pl.DataFrame] = []

    for s_name in DATED_SEASONS:
        allowed_seasons = get_prior_training_seasons_for_target(s_name)
        s_conaf = valid_conaf.filter(pl.col("season").is_in(allowed_seasons))

        density = np.zeros(n_cells, dtype=np.float64)

        if not s_conaf.is_empty():
            fire_xy = np.column_stack(
                [
                    s_conaf.get_column("x_utm").to_numpy(),
                    s_conaf.get_column("y_utm").to_numpy(),
                ]
            )
            nbrs_list = tree.query_ball_point(fire_xy, r=cutoff)
            for f_idx, nbr_indices in enumerate(nbrs_list):
                if nbr_indices:
                    nbr_arr = np.asarray(nbr_indices, dtype=np.int64)
                    dists = np.hypot(
                        grid_xy[nbr_arr, 0] - fire_xy[f_idx, 0],
                        grid_xy[nbr_arr, 1] - fire_xy[f_idx, 1],
                    )
                    density[nbr_arr] += np.exp(-0.5 * (dists / sigma) ** 2)

        eps = 0.01 * float(np.mean(density)) if np.mean(density) > 0 else 1e-6
        density_reg = density + eps
        density_norm = (density_reg / float(np.sum(density_reg))).astype(np.float32)

        s_df = pl.DataFrame({
            "cell_id": cell_ids,
            "season": [s_name] * n_cells,
            "prior_density": density_norm,
        })
        season_dfs.append(s_df)

    full_prior_df = pl.concat(season_dfs, how="vertical")
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full_prior_df.write_parquet(out_path)
        log_event(
            logger,
            "prior_density_table_saved",
            path=str(out_path),
            rows=full_prior_df.height,
            cells=n_cells,
            seasons=len(DATED_SEASONS),
        )

    return full_prior_df
