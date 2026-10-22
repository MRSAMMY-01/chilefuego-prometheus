"""Arnés de validación Leave-One-Season-Out (LOSO) sobre el dominio completo por lotes (Fase 2B — C6).

PORQUÉ (Regla 5):
- La evaluación debe realizarse sobre los 33.345.071 celdas-día del universo completo, no sobre la muestra reducida.
- Para operar dentro de los 16 GB de RAM, el arnés procesa estrictamente por lotes estacionales (un fold a la vez,
  ~4.7M filas liberadas inmediatamente al terminar cada pliegue), evitando materializar los 33M simultáneamente.
- CRITERIO DE PARADA DE EMERGENCIA:
  El arnés debe reproducir exactamente el baseline oficial FWI x Prior (PR-AUC medio 1.170682e-3 +/- 1%).
  Si no coincide dentro del 1%, la ejecución se DETIENE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.metrics import average_precision_score

from chilefuego.config import AppConfig
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)

OFFICIAL_FWI_X_PRIOR_PRAUC_MEAN: float = 1.170682e-3
TOLERANCE_REL: float = 0.01  # +/- 1%


@dataclass(frozen=True)
class FoldMetrics:
    season: str
    n_positives: int
    n_total: int
    pr_auc: float
    base_rate: float
    lift: float
    recall_at_1pct: float
    recall_at_5pct: float
    rec1_ge_1ha: float
    n_ge_1ha: int
    rec1_ge_10ha: float
    n_ge_10ha: int
    pct_ignitions_in_top10pct_fwi_days: float


@dataclass(frozen=True)
class LosoBenchmarkSummary:
    fold_metrics: list[FoldMetrics]
    mean_pr_auc: float
    mean_lift: float
    mean_rec1: float
    mean_rec5: float
    mean_rec1_ge_1ha: float
    mean_rec1_ge_10ha: float
    mean_pct_in_top10_fwi_days: float
    is_validated: bool


def evaluate_loso_full_domain_batched(
    config: AppConfig,
    prior_path: Path = Path("data/processed/prior_density_by_cell_season.parquet"),
    weather_path: Path = Path("data/processed/era5_land_extended_weather_nodes.parquet"),
) -> LosoBenchmarkSummary:
    """Ejecuta la validación LOSO en 7 lotes estacionales independientes con validación de parada."""
    t0 = time.time()
    grid_df = pl.read_parquet(config.paths.grid_catalog_path)
    special_cell_days_df, summary = build_phase2a_daily_panel(config)

    prior_df = pl.read_parquet(prior_path)
    weather_df = pl.read_parquet(weather_path).select(["node_id", "date", "fwi"])

    # Mapeo cKDTree de celda a nodo
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()

    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = transformer.transform(cx, cy)

    unique_nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
    tree = cKDTree(node_coords)
    _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    cell_node_df = pl.DataFrame({
        "cell_id": cell_ids,
        "node_id": [unique_nodes[idx] for idx in node_idxs],
    })

    folds: list[FoldMetrics] = []

    for s_name in DATED_SEASONS:
        # 1. Materializar lote estacional
        fold_panel = materialize_season_cell_day_panel(
            grid_df=grid_df,
            special_cell_days_df=special_cell_days_df,
            season=s_name,
            include_excluded=False,
        )

        # 2. Unir prior de la temporada
        fold_panel = fold_panel.join(
            prior_df.filter(pl.col("season") == s_name).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )

        # 3. Unir nodo y FWI
        fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(weather_df, on=["node_id", "date"], how="left").with_columns(
            pl.col("fwi").fill_null(0.0)
        )

        y_true = fold_panel.get_column("is_ignition_conaf").to_numpy().astype(np.int8)
        sups = fold_panel.get_column("superficie_ha_max").to_numpy().astype(np.float64)
        fwi_scores = fold_panel.get_column("fwi").to_numpy().astype(np.float64)
        prior_scores = fold_panel.get_column("prior_density").to_numpy().astype(np.float64)
        scores = fwi_scores * prior_scores

        n_total = len(y_true)
        n_pos = int(np.sum(y_true))
        base_rate = float(n_pos) / float(n_total)

        # PR-AUC
        try:
            pr_auc = float(average_precision_score(y_true, scores))
        except Exception:
            pr_auc = 0.0

        lift = pr_auc / base_rate if base_rate > 0.0 else 0.0

        # Umbrales top 1% y top 5%
        k1 = max(1, int(round(0.01 * n_total)))
        k5 = max(1, int(round(0.05 * n_total)))
        t1 = np.partition(scores, -k1)[-k1]
        t5 = np.partition(scores, -k5)[-k5]

        mask_pos = y_true == 1
        rec1 = float(np.sum(mask_pos & (scores >= t1))) / float(n_pos) if n_pos > 0 else 0.0
        rec5 = float(np.sum(mask_pos & (scores >= t5))) / float(n_pos) if n_pos > 0 else 0.0

        # Estrato >= 1 ha
        mask_1ha = mask_pos & (sups >= 1.0)
        n_1ha = int(np.sum(mask_1ha))
        rec1_1ha = float(np.sum(mask_1ha & (scores >= t1))) / float(n_1ha) if n_1ha > 0 else 0.0

        # Estrato >= 10 ha
        mask_10ha = mask_pos & (sups >= 10.0)
        n_10ha = int(np.sum(mask_10ha))
        rec1_10ha = float(np.sum(mask_10ha & (scores >= t1))) / float(n_10ha) if n_10ha > 0 else 0.0

        # Métrica temporal regional (% igniciones en top 10% días de mayor FWI regional)
        daily_reg_fwi = fold_panel.group_by("date").agg(pl.col("fwi").mean().alias("reg_fwi")).sort("date")
        k10_days = max(1, int(round(0.10 * daily_reg_fwi.height)))
        t10_days = np.partition(daily_reg_fwi["reg_fwi"].to_numpy(), -k10_days)[-k10_days]
        top10_dates = set(daily_reg_fwi.filter(pl.col("reg_fwi") >= t10_days)["date"].to_list())
        pos_dates = fold_panel.filter(pl.col("is_ignition_conaf") == 1)["date"].to_list()
        n_pos_in_top10 = sum(1 for d in pos_dates if d in top10_dates)
        pct_in_top10 = (n_pos_in_top10 / len(pos_dates)) * 100.0 if pos_dates else 0.0

        folds.append(
            FoldMetrics(
                season=s_name,
                n_positives=n_pos,
                n_total=n_total,
                pr_auc=pr_auc,
                base_rate=base_rate,
                lift=lift,
                recall_at_1pct=rec1,
                recall_at_5pct=rec5,
                rec1_ge_1ha=rec1_1ha,
                n_ge_1ha=n_1ha,
                rec1_ge_10ha=rec1_10ha,
                n_ge_10ha=n_10ha,
                pct_ignitions_in_top10pct_fwi_days=pct_in_top10,
            )
        )
        del fold_panel  # Liberación explícita de memoria

    mean_prauc = float(np.mean([f.pr_auc for f in folds]))
    mean_lift = float(np.mean([f.lift for f in folds]))
    mean_rec1 = float(np.mean([f.recall_at_1pct for f in folds]))
    mean_rec5 = float(np.mean([f.recall_at_5pct for f in folds]))
    mean_rec1_1ha = float(np.mean([f.rec1_ge_1ha for f in folds]))
    mean_rec1_10ha = float(np.mean([f.rec1_ge_10ha for f in folds]))
    mean_top10_fwi = float(np.mean([f.pct_ignitions_in_top10pct_fwi_days for f in folds]))

    rel_diff = abs(mean_prauc - OFFICIAL_FWI_X_PRIOR_PRAUC_MEAN) / OFFICIAL_FWI_X_PRIOR_PRAUC_MEAN
    is_validated = rel_diff <= TOLERANCE_REL

    if not is_validated:
        msg = (
            f"DETENTE: Validación LOSO FAILED. PR-AUC medio={mean_prauc:.6e} difiere en {rel_diff*100:.2f}% "
            f"del baseline oficial {OFFICIAL_FWI_X_PRIOR_PRAUC_MEAN:.6e} (tolerancia estricta +/- 1%)."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    log_event(
        logger,
        "loso_harness_validation_passed",
        mean_pr_auc=mean_prauc,
        baseline_official=OFFICIAL_FWI_X_PRIOR_PRAUC_MEAN,
        diff_pct=round(rel_diff * 100, 4),
        mean_lift=round(mean_lift, 2),
        elapsed_s=round(time.time() - t0, 2),
    )

    return LosoBenchmarkSummary(
        fold_metrics=folds,
        mean_pr_auc=mean_prauc,
        mean_lift=mean_lift,
        mean_rec1=mean_rec1,
        mean_rec5=mean_rec5,
        mean_rec1_ge_1ha=mean_rec1_1ha,
        mean_rec1_ge_10ha=mean_rec1_10ha,
        mean_pct_in_top10_fwi_days=mean_top10_fwi,
        is_validated=is_validated,
    )
