"""Baseline sin Machine Learning y validación Leave-One-Season-Out (7 pliegues) (Fase 2A — B4).

PORQUÉ (Regla 5):
- Antes de modelos ML (Fase 2B), se evalúa un benchmark honesto sin ML con tres modelos:
  (a) Ranking por FWI solo (peligro meteorológico diario instantáneo).
  (b) Ranking por prior espacial con ventana creciente: para la temporada s, utiliza 2010–2016,
      las ubicaciones de 2018_19 (solo si s >= 2019_20), y las temporadas fechadas anteriores a s.
  (c) Ranking conjunto interactivo FWI × prior.
- Validación Leave-One-Season-Out (LOSO) en 7 pliegues fechados (2017_18 y 2019_20 a 2024_25).
- Métrica temporal: % de igniciones ocurridas en el 10% de días con FWI regional más alto (esperado por azar: 10%).
- Evaluación estratificada por tamaño de incendio: todas las superficies, >= 1 ha y >= 10 ha (superficie_ha
  usada EXCLUSIVAMENTE para evaluar, nunca como predictor).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

from chilefuego.config import AppConfig, load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class MetricBundle:
    pr_auc: float
    lift_over_base_rate: float
    recall_at_1pct: float
    recall_at_5pct: float
    n_positives: int
    n_total: int


@dataclass(frozen=True)
class SeasonEvaluationRow:
    season: str
    model_name: str
    all_fires: MetricBundle
    ge_1ha: MetricBundle
    ge_10ha: MetricBundle
    pct_ignitions_in_top10pct_fwi_days: float = 0.0


def build_increasing_window_prior(
    grid_df: pl.DataFrame,
    ignitions_conaf_df: pl.DataFrame,
    target_season: str,
    bandwidth_m: float = 3000.0,
) -> pl.DataFrame:
    """Calcula el prior espacial con ventana creciente según B4.

    Para la temporada target_season:
    - Incluye temporadas históricas 2010–2016 (7 temporadas base).
    - Incluye las ubicaciones de 2018_19 solo si target_season >= 2019_20.
    - Incluye temporadas fechadas estrictamente anteriores a target_season (sin fuga temporal del futuro).
    """
    nuble_grid = grid_df.filter(pl.col("in_nuble")).select(["cell_id", "centroid_x", "centroid_y"])
    target_start_year = int(target_season.split("_")[0])

    base_seasons = [f"if_temporada_{y}_{y+1}" for y in range(2010, 2017)]
    if target_start_year >= 2019:
        base_seasons.append("if_temporada_2018_2019")

    for ds in DATED_SEASONS:
        ds_year = int(ds.split("_")[0])
        if ds_year < target_start_year:
            base_seasons.append(f"if_temporada_{ds_year}_{ds_year+1}")

    qf = ignitions_conaf_df.get_column("quality_flags")
    filtered_conaf = ignitions_conaf_df.with_columns(
        qf.struct.field("crs_ok").alias("_crs_ok"),
        qf.struct.field("dup_removed").alias("_dup_removed"),
    ).filter(
        pl.col("_crs_ok") & (~pl.col("_dup_removed")) & pl.col("season").is_in(base_seasons)
    )

    n_cells = nuble_grid.height
    if filtered_conaf.is_empty():
        return nuble_grid.with_columns(
            pl.lit(1.0 / float(n_cells)).alias("spatial_prior")
        ).select(["cell_id", "spatial_prior"])

    grid_xy = np.column_stack(
        [
            nuble_grid.get_column("centroid_x").to_numpy(),
            nuble_grid.get_column("centroid_y").to_numpy(),
        ]
    )
    fire_xy = np.column_stack(
        [
            filtered_conaf.get_column("x_utm").to_numpy(),
            filtered_conaf.get_column("y_utm").to_numpy(),
        ]
    )

    tree = cKDTree(grid_xy)
    sigma = float(bandwidth_m)
    cutoff = 3.0 * sigma
    density = np.zeros(n_cells, dtype=np.float64)

    neighbors_list = tree.query_ball_point(fire_xy, r=cutoff)
    for f_idx, nbr_indices in enumerate(neighbors_list):
        if nbr_indices:
            nbr_arr = np.asarray(nbr_indices, dtype=np.int64)
            dists = np.hypot(
                grid_xy[nbr_arr, 0] - fire_xy[f_idx, 0],
                grid_xy[nbr_arr, 1] - fire_xy[f_idx, 1],
            )
            density[nbr_arr] += np.exp(-0.5 * (dists / sigma) ** 2)

    eps = 0.01 * float(np.mean(density)) if np.mean(density) > 0 else 1e-6
    density_reg = density + eps
    density_norm = density_reg / float(np.sum(density_reg))

    return nuble_grid.with_columns(
        pl.Series("spatial_prior", density_norm, dtype=pl.Float64)
    ).select(["cell_id", "spatial_prior"])


def evaluate_ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    superficies_ha: np.ndarray | None = None,
) -> tuple[MetricBundle, MetricBundle, MetricBundle]:
    """Calcula PR-AUC, Lift, Recall@1% y Recall@5% para todas las igniciones, >= 1 ha y >= 10 ha."""
    y = np.asarray(y_true, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    n = y.shape[0]

    def _calc_bundle(mask_pos: np.ndarray) -> MetricBundle:
        n_pos = int(np.sum(mask_pos))
        if n_pos == 0 or n == 0:
            return MetricBundle(0.0, 0.0, 0.0, 0.0, 0, n)

        base_rate = float(n_pos) / float(n)

        try:
            pr_auc = float(average_precision_score(mask_pos.astype(int), s))
        except Exception:
            pr_auc = 0.0

        lift = pr_auc / base_rate if base_rate > 0.0 else 0.0

        k1 = max(1, int(round(0.01 * n)))
        k5 = max(1, int(round(0.05 * n)))
        top_k1_thresh = np.partition(s, -k1)[-k1]
        top_k5_thresh = np.partition(s, -k5)[-k5]

        rec1 = float(np.sum(mask_pos & (s >= top_k1_thresh))) / float(n_pos)
        rec5 = float(np.sum(mask_pos & (s >= top_k5_thresh))) / float(n_pos)

        return MetricBundle(
            pr_auc=pr_auc,
            lift_over_base_rate=lift,
            recall_at_1pct=rec1,
            recall_at_5pct=rec5,
            n_positives=n_pos,
            n_total=n,
        )

    all_bundle = _calc_bundle(y == 1)

    if superficies_ha is not None:
        sup = np.asarray(superficies_ha, dtype=np.float64)
        ge_1ha_bundle = _calc_bundle((y == 1) & (sup >= 1.0))
        ge_10ha_bundle = _calc_bundle((y == 1) & (sup >= 10.0))
    else:
        ge_1ha_bundle = MetricBundle(0.0, 0.0, 0.0, 0.0, 0, n)
        ge_10ha_bundle = MetricBundle(0.0, 0.0, 0.0, 0.0, 0, n)

    return all_bundle, ge_1ha_bundle, ge_10ha_bundle


def run_phase2a_loso_benchmark(
    config: AppConfig,
    fwi_nodes_df: pl.DataFrame | None = None,
) -> tuple[list[SeasonEvaluationRow], dict[str, Any]]:
    """Ejecuta los 7 pliegues LOSO comparando:

    (a) FWI solo
    (b) Prior con ventana creciente solo
    (c) FWI x Prior
    """
    grid_df = pl.read_parquet(config.paths.grid_catalog_path)
    ignitions_conaf_df = pl.read_parquet(config.paths.ignitions_conaf_path)

    # Si fwi_nodes_df no se pasa, intentar cargarlo desde disco si existe
    if fwi_nodes_df is None:
        default_fwi_path = config.paths.processed_dir / "era5_land_daily_fwi_nodes.parquet"
        if default_fwi_path.exists():
            fwi_nodes_df = pl.read_parquet(default_fwi_path)

    # Mapeo de celdas 1 km a nodo ERA5-Land más cercano
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()

    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = transformer.transform(cx, cy)

    if fwi_nodes_df is not None:
        unique_nodes = fwi_nodes_df.select(["node_id"]).unique().to_series().to_list()
        node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
        tree = cKDTree(node_coords)
        _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
        cell_node_df = pl.DataFrame({
            "cell_id": cell_ids,
            "node_id": [unique_nodes[idx] for idx in node_idxs],
        })
    else:
        cell_node_df = pl.DataFrame()

    special_cell_days_df, summary = build_phase2a_daily_panel(config)
    rows: list[SeasonEvaluationRow] = []

    for s_name in DATED_SEASONS:
        fold_panel = materialize_season_cell_day_panel(
            grid_df=grid_df,
            special_cell_days_df=special_cell_days_df,
            season=s_name,
            include_excluded=False,
        )

        # Prior con ventana creciente
        prior_df = build_increasing_window_prior(grid_df, ignitions_conaf_df, target_season=s_name)
        fold_panel = fold_panel.join(prior_df, on="cell_id", how="left")

        y_true = fold_panel.get_column("is_ignition_conaf").to_numpy()
        sups = fold_panel.get_column("superficie_ha_max").to_numpy()
        prior_scores = fold_panel.get_column("spatial_prior").to_numpy()

        # Modelo (b): Prior creciente
        b_all, b_1ha, b_10ha = evaluate_ranking_metrics(y_true, prior_scores, sups)

        pct_in_top10 = 0.0
        # Modelos (a) FWI y (c) FWI x Prior si hay datos FWI
        if fwi_nodes_df is not None and not cell_node_df.is_empty():
            fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
            fold_panel = fold_panel.join(
                fwi_nodes_df.select(["node_id", "date", "fwi"]),
                on=["node_id", "date"],
                how="left",
            ).with_columns(pl.col("fwi").fill_null(0.0))

            fwi_scores = fold_panel.get_column("fwi").to_numpy()
            prod_scores = fwi_scores * prior_scores

            # Métrica temporal: % de igniciones en el 10% de días con FWI regional más alto
            daily_reg_fwi = fold_panel.group_by("date").agg(pl.col("fwi").mean().alias("reg_fwi")).sort("date")
            k10_days = max(1, int(round(0.10 * daily_reg_fwi.height)))
            top10_thresh = np.partition(daily_reg_fwi["reg_fwi"].to_numpy(), -k10_days)[-k10_days]
            top10_dates = set(daily_reg_fwi.filter(pl.col("reg_fwi") >= top10_thresh)["date"].to_list())
            pos_dates = fold_panel.filter(pl.col("is_ignition_conaf") == 1)["date"].to_list()
            n_pos_in_top10 = sum(1 for d in pos_dates if d in top10_dates)
            pct_in_top10 = (n_pos_in_top10 / len(pos_dates)) * 100.0 if pos_dates else 0.0

            a_all, a_1ha, a_10ha = evaluate_ranking_metrics(y_true, fwi_scores, sups)
            rows.append(
                SeasonEvaluationRow(
                    season=s_name,
                    model_name="a_fwi_ranking",
                    all_fires=a_all,
                    ge_1ha=a_1ha,
                    ge_10ha=a_10ha,
                    pct_ignitions_in_top10pct_fwi_days=pct_in_top10,
                )
            )

            rows.append(
                SeasonEvaluationRow(
                    season=s_name,
                    model_name="b_prior_spatial_increasing",
                    all_fires=b_all,
                    ge_1ha=b_1ha,
                    ge_10ha=b_10ha,
                    pct_ignitions_in_top10pct_fwi_days=pct_in_top10,
                )
            )

            c_all, c_1ha, c_10ha = evaluate_ranking_metrics(y_true, prod_scores, sups)
            rows.append(
                SeasonEvaluationRow(
                    season=s_name,
                    model_name="c_fwi_x_prior",
                    all_fires=c_all,
                    ge_1ha=c_1ha,
                    ge_10ha=c_10ha,
                    pct_ignitions_in_top10pct_fwi_days=pct_in_top10,
                )
            )
        else:
            rows.append(
                SeasonEvaluationRow(
                    season=s_name,
                    model_name="b_prior_spatial_increasing",
                    all_fires=b_all,
                    ge_1ha=b_1ha,
                    ge_10ha=b_10ha,
                    pct_ignitions_in_top10pct_fwi_days=0.0,
                )
            )

    meta = {
        "n_dated_seasons": len(DATED_SEASONS),
        "total_effective_cell_days": summary.effective_training_cell_days,
        "total_positive_cell_days": summary.total_positive_cell_days,
        "fwi_evaluated": fwi_nodes_df is not None,
    }
    return rows, meta


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config("configs/config.yaml")
    results, meta = run_phase2a_loso_benchmark(cfg)
    print("=== RESULTADOS BASELINE SIN ML (LEAVE-ONE-SEASON-OUT, 7 PLIEGUES) ===")
    print(
        f"Temporadas: {DATED_SEASONS} | Celdas-día efectivas: {meta['total_effective_cell_days']:,}"
    )
    print(f"Igniciones positivas CONAF: {meta['total_positive_cell_days']:,}")

    for model_key in ["a_fwi_ranking", "b_prior_spatial_increasing", "c_fwi_x_prior"]:
        m_rows = [r for r in results if r.model_name == model_key]
        if not m_rows:
            continue
        print(f"\n--- MODELO: {model_key} ---")
        for r in m_rows:
            print(
                f"  [{r.season}] PR-AUC: {r.all_fires.pr_auc:.6e} (Lift: {r.all_fires.lift_over_base_rate:.2f}x) | "
                f"Rec@1%: {r.all_fires.recall_at_1pct * 100:.2f}%, Rec@5%: {r.all_fires.recall_at_5pct * 100:.2f}% | "
                f">=1ha Rec@1%: {r.ge_1ha.recall_at_1pct * 100:.2f}% (N={r.ge_1ha.n_positives}) | "
                f">=10ha Rec@1%: {r.ge_10ha.recall_at_1pct * 100:.2f}% (N={r.ge_10ha.n_positives}) | "
                f"Top10% días FWI: {r.pct_ignitions_in_top10pct_fwi_days:.2f}%"
            )
        mean_pr = np.mean([r.all_fires.pr_auc for r in m_rows])
        mean_lift = np.mean([r.all_fires.lift_over_base_rate for r in m_rows])
        mean_rec1 = np.mean([r.all_fires.recall_at_1pct for r in m_rows])
        mean_rec5 = np.mean([r.all_fires.recall_at_5pct for r in m_rows])
        mean_rec1_1ha = np.mean([r.ge_1ha.recall_at_1pct for r in m_rows])
        mean_rec1_10ha = np.mean([r.ge_10ha.recall_at_1pct for r in m_rows])
        print(
            f"  PROMEDIO 7 PLIEGUES: PR-AUC={mean_pr:.6e}, Lift={mean_lift:.2f}x, "
            f"Rec@1%={mean_rec1 * 100:.2f}%, Rec@5%={mean_rec5 * 100:.2f}%, "
            f">=1ha Rec@1%={mean_rec1_1ha * 100:.2f}%, >=10ha Rec@1%={mean_rec1_10ha * 100:.2f}%"
        )
