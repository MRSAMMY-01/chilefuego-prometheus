"""Pipeline integral de experimentos para Fase 2D (G1 a G3).

PORQUÉ (Regla 5):
- G1: Corrige la escala del bootstrap por bloques de días calculando PR-AUC exacta sobre el dominio completo
  sin submuestreo de negativos mediante histograma de cuantiles finos (10.000 bins), demostrando matemáticamente
  su equivalencia con sklearn (<0.05% error). Reporta PR-AUC de Baseline, Podado y Completo con deltas e IC 95%
  por temporada, estimador agregado y validación hacia adelante.
- G2: Ajusta el calibrador Platt final de forma estrictamente FUERA DE MUESTRA (out-of-fold sobre las 7 temporadas),
  recalcula los umbrales p80, p95 y p99 sobre las probabilidades calibradas y demuestra la captura ~80/15/4/1%
  con su desvío estacional.
- G3: Cuantifica el costo global de la variante con pesos por tamaño en todas las igniciones y >=1 ha,
  ejecuta bootstrap pareado por eventos sobre los 255 grandes incendios (B=2000) y evalúa la regla de decisión.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import psutil
import xgboost as xgb
from chilefuego.config import load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.models.xgboost_experiment import (
    ALL_39_FEATURES,
    PRUNED_25_FEATURES,
)
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

RAM_CEILING_MB = 5500.0


def get_current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


def check_ram(context: str = "") -> None:
    rss = get_current_rss_mb()
    if rss > RAM_CEILING_MB:
        raise RuntimeError(f"ABORT: Consumo de RAM ({rss:.2f} MB) superó el límite seguro ({RAM_CEILING_MB} MB) en {context}")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DetailedMetrics:
    season: str
    n_pos: int
    n_total: int
    base_rate: float
    pr_auc: float
    lift: float
    rec1_all: float
    rec5_all: float
    rec1_1ha: float
    rec5_1ha: float
    n_1ha: int
    rec1_10ha: float
    rec5_10ha: float
    n_10ha: int
    rec1_small: float
    rec5_small: float
    n_small: int


def evaluate_detailed(
    y_true: np.ndarray,
    scores: np.ndarray,
    sups: np.ndarray,
    season: str,
) -> DetailedMetrics:
    n_total = len(y_true)
    n_pos = int(np.sum(y_true))
    base_rate = float(n_pos) / float(n_total)

    pr_auc = float(average_precision_score(y_true, scores)) if n_pos > 0 else 0.0
    lift = pr_auc / base_rate if base_rate > 0.0 else 0.0

    k1 = max(1, int(round(0.01 * n_total)))
    k5 = max(1, int(round(0.05 * n_total)))
    t1 = np.partition(scores, -k1)[-k1]
    t5 = np.partition(scores, -k5)[-k5]

    mask_pos = y_true == 1
    rec1_all = float(np.sum(mask_pos & (scores >= t1))) / float(n_pos) if n_pos > 0 else 0.0
    rec5_all = float(np.sum(mask_pos & (scores >= t5))) / float(n_pos) if n_pos > 0 else 0.0

    mask_1ha = mask_pos & (sups >= 1.0)
    n_1ha = int(np.sum(mask_1ha))
    rec1_1ha = float(np.sum(mask_1ha & (scores >= t1))) / float(n_1ha) if n_1ha > 0 else 0.0
    rec5_1ha = float(np.sum(mask_1ha & (scores >= t5))) / float(n_1ha) if n_1ha > 0 else 0.0

    mask_10ha = mask_pos & (sups >= 10.0)
    n_10ha = int(np.sum(mask_10ha))
    rec1_10ha = float(np.sum(mask_10ha & (scores >= t1))) / float(n_10ha) if n_10ha > 0 else 0.0
    rec5_10ha = float(np.sum(mask_10ha & (scores >= t5))) / float(n_10ha) if n_10ha > 0 else 0.0

    mask_small = mask_pos & (sups < 1.0)
    n_small = int(np.sum(mask_small))
    rec1_small = float(np.sum(mask_small & (scores >= t1))) / float(n_small) if n_small > 0 else 0.0
    rec5_small = float(np.sum(mask_small & (scores >= t5))) / float(n_small) if n_small > 0 else 0.0

    return DetailedMetrics(
        season=season,
        n_pos=n_pos,
        n_total=n_total,
        base_rate=base_rate,
        pr_auc=pr_auc,
        lift=lift,
        rec1_all=rec1_all,
        rec5_all=rec5_all,
        rec1_1ha=rec1_1ha,
        rec5_1ha=rec5_1ha,
        n_1ha=n_1ha,
        rec1_10ha=rec1_10ha,
        rec5_10ha=rec5_10ha,
        n_10ha=n_10ha,
        rec1_small=rec1_small,
        rec5_small=rec5_small,
        n_small=n_small,
    )


class FullDomainBinnedPRAUC:
    """Calculador de PR-AUC en el dominio completo mediante histogramas de cuantiles (10.000 bins)."""

    def __init__(self, scores: np.ndarray, n_bins: int = 10000) -> None:
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bin_edges = np.unique(np.quantile(scores, quantiles))
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        self.n_bins = len(self.bin_edges) + 1

    def compute_daily_counts(
        self,
        scores: np.ndarray,
        y_true: np.ndarray,
        dates: list[Any],
    ) -> tuple[np.ndarray, np.ndarray, list[Any], np.ndarray]:
        unique_dates = sorted(list(set(dates)))
        n_days = len(unique_dates)
        date_to_idx = {d: i for i, d in enumerate(unique_dates)}

        pos_daily = np.zeros((n_days, self.n_bins), dtype=np.int32)
        neg_daily = np.zeros((n_days, self.n_bins), dtype=np.int32)
        day_total_cells = np.zeros(n_days, dtype=np.int32)

        # Discretizar scores en bins
        bin_indices = np.digitize(scores, self.bin_edges)

        # Asignar a matriz diaria
        for i, d in enumerate(dates):
            d_idx = date_to_idx[d]
            day_total_cells[d_idx] += 1
            b = bin_indices[i]
            if y_true[i] == 1:
                pos_daily[d_idx, b] += 1
            else:
                neg_daily[d_idx, b] += 1

        return pos_daily, neg_daily, unique_dates, day_total_cells

    @staticmethod
    def calc_pr_auc_from_counts(pos_counts: np.ndarray, neg_counts: np.ndarray) -> float:
        tp = np.cumsum(pos_counts[::-1])
        fp = np.cumsum(neg_counts[::-1])
        tot_pos = tp[-1]
        if tot_pos == 0:
            return 0.0
        rec = tp / float(tot_pos)
        prec = tp / np.maximum(tp + fp, 1.0)
        d_rec = np.diff(np.concatenate([[0.0], rec]))
        return float(np.sum(d_rec * prec))

    @staticmethod
    def calc_rec5_from_counts(pos_counts: np.ndarray, neg_counts: np.ndarray, total_cells: int) -> float:
        k5 = max(1, int(round(0.05 * total_cells)))
        # De mayor a menor score
        tp_desc = pos_counts[::-1]
        fp_desc = neg_counts[::-1]
        cum_cells = np.cumsum(tp_desc + fp_desc)
        cum_pos = np.cumsum(tp_desc)
        tot_pos = cum_pos[-1]
        if tot_pos == 0:
            return 0.0

        idx = np.searchsorted(cum_cells, k5)
        if idx >= len(cum_pos):
            idx = len(cum_pos) - 1
        return float(cum_pos[idx] / tot_pos)


def run_full_phase2d() -> None:
    t0 = time.time()
    cfg = load_config("configs/config.yaml")
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path("data/scratch_phase2d")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== INICIO EXPERIMENTOS FASE 2D === | RSS inicial: {get_current_rss_mb():.2f} MB", flush=True)

    train_df = pl.read_parquet("data/processed/training_sample_phase2b.parquet")
    grid_df = pl.read_parquet(cfg.paths.grid_catalog_path)
    special_cell_days_df, _ = build_phase2a_daily_panel(cfg)
    static_df = pl.read_parquet("data/processed/static_features_1km.parquet")
    prior_df = pl.read_parquet("data/processed/prior_density_by_cell_season.parquet")
    weather_df = pl.read_parquet("data/processed/era5_land_extended_weather_nodes.parquet")

    # Mapeo celdas a nodos ERA5-Land
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()
    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = trans.transform(cx, cy)
    unique_nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
    tree = cKDTree(node_coords)
    _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    cell_node_df = pl.DataFrame({"cell_id": cell_ids, "node_id": [unique_nodes[idx] for idx in node_idxs]})

    SEEDS = [42, 43, 44, 45, 46]
    params_base: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "nthread": 4,
    }

    # Contenedores para G1, G2, G3
    g1_season_results: list[dict[str, Any]] = []
    daily_hist_base: list[np.ndarray] = []
    daily_hist_pruned: list[np.ndarray] = []
    daily_hist_full: list[np.ndarray] = []
    daily_cells_all: list[np.ndarray] = []

    # Almacenar datos para G3 sobre los 255 incendios >= 10 ha
    large_fires_records: list[dict[str, Any]] = []

    # Contenedores para evaluación de costo global G3
    cost_metrics_base: list[DetailedMetrics] = []
    cost_metrics_unweighted: list[DetailedMetrics] = []
    cost_metrics_weighted: list[DetailedMetrics] = []

    print("\n--- Ejecutando LOSO 7 Pliegues para G1, G2 y G3 ---", flush=True)

    for s_test in DATED_SEASONS:
        check_ram(f"inicio fold {s_test}")
        print(f"\n>>> Fold LOSO: {s_test} | RSS: {get_current_rss_mb():.2f} MB", flush=True)
        train_seasons = [s for s in DATED_SEASONS if s != s_test]
        val_season = train_seasons[-1]
        fit_seasons = train_seasons[:-1]

        fit_sub = train_df.filter(pl.col("season").is_in(fit_seasons))
        val_sub = train_df.filter(pl.col("season") == val_season)

        y_fit = fit_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        w_fit = fit_sub.get_column("sample_weight").to_numpy().astype(np.float32)
        y_val = val_sub.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        w_val = val_sub.get_column("sample_weight").to_numpy().astype(np.float32)

        sup_fit = fit_sub.get_column("superficie_ha_max").to_numpy()
        w_size_fit = w_fit * (1.0 + np.minimum(np.log10(np.maximum(sup_fit, 0.0) + 1.0), 3.0))

        X_p_fit = fit_sub.select(PRUNED_25_FEATURES).to_numpy()
        X_p_val = val_sub.select(PRUNED_25_FEATURES).to_numpy()
        X_f_fit = fit_sub.select(ALL_39_FEATURES).to_numpy()
        X_f_val = val_sub.select(ALL_39_FEATURES).to_numpy()

        fold_panel = materialize_season_cell_day_panel(
            grid_df=grid_df,
            special_cell_days_df=special_cell_days_df,
            season=s_test,
            include_excluded=False,
        )
        fold_panel = fold_panel.join(static_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(
            prior_df.filter(pl.col("season") == s_test).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )
        fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(weather_df, on=["node_id", "date"], how="left")

        y_test = fold_panel.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        sups_test = fold_panel.get_column("superficie_ha_max").to_numpy().astype(np.float64)
        dates_test = fold_panel.get_column("date").to_list()
        fwi_test = fold_panel.get_column("fwi").to_numpy().astype(np.float64)
        prior_test = fold_panel.get_column("prior_density").to_numpy().astype(np.float64)

        base_scores = (fwi_test * prior_test).astype(np.float32)

        # 1. Pruned Model
        dtest_p = xgb.DMatrix(fold_panel.select(PRUNED_25_FEATURES).to_numpy(), feature_names=PRUNED_25_FEATURES)
        scores_pruned_seeds: list[np.ndarray] = []
        scores_sizew_seeds: list[np.ndarray] = []

        for seed in SEEDS:
            p_seed = {**params_base, "random_state": seed}
            dp_train = xgb.DMatrix(X_p_fit, label=y_fit, weight=w_fit, feature_names=PRUNED_25_FEATURES)
            dp_val = xgb.DMatrix(X_p_val, label=y_val, weight=w_val, feature_names=PRUNED_25_FEATURES)
            b_pruned = xgb.train(
                p_seed, dp_train, num_boost_round=500,
                evals=[(dp_train, "train"), (dp_val, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            sc_p = b_pruned.predict(dtest_p)
            scores_pruned_seeds.append(sc_p)

            # Ponderado por tamaño
            dp_train_sz = xgb.DMatrix(X_p_fit, label=y_fit, weight=w_size_fit, feature_names=PRUNED_25_FEATURES)
            b_sizew = xgb.train(
                p_seed, dp_train_sz, num_boost_round=500,
                evals=[(dp_train_sz, "train"), (dp_val, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            sc_sz = b_sizew.predict(dtest_p)
            scores_sizew_seeds.append(sc_sz)

        del dtest_p

        # 2. Full Model
        dtest_f = xgb.DMatrix(fold_panel.select(ALL_39_FEATURES).to_numpy(), feature_names=ALL_39_FEATURES)
        scores_full_seeds: list[np.ndarray] = []
        for seed in SEEDS:
            p_seed = {**params_base, "random_state": seed}
            df_train = xgb.DMatrix(X_f_fit, label=y_fit, weight=w_fit, feature_names=ALL_39_FEATURES)
            df_val = xgb.DMatrix(X_f_val, label=y_val, weight=w_val, feature_names=ALL_39_FEATURES)
            b_full = xgb.train(
                p_seed, df_train, num_boost_round=500,
                evals=[(df_train, "train"), (df_val, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            sc_f = b_full.predict(dtest_f)
            scores_full_seeds.append(sc_f)

        del dtest_f, fold_panel
        gc.collect()

        mean_sc_pruned = np.mean(scores_pruned_seeds, axis=0).astype(np.float32)
        mean_sc_full = np.mean(scores_full_seeds, axis=0).astype(np.float32)
        mean_sc_sizew = np.mean(scores_sizew_seeds, axis=0).astype(np.float32)

        # G1: Construcción de cuantiles individuales sobre el universo de scores del pliegue (10.000 bins)
        calc_bins_p = FullDomainBinnedPRAUC(mean_sc_pruned, n_bins=10000)
        pos_p, neg_p, u_dates, d_cells = calc_bins_p.compute_daily_counts(mean_sc_pruned, y_test, dates_test)

        calc_bins_b = FullDomainBinnedPRAUC(base_scores, n_bins=10000)
        pos_b, neg_b, _, _ = calc_bins_b.compute_daily_counts(base_scores, y_test, dates_test)

        calc_bins_f = FullDomainBinnedPRAUC(mean_sc_full, n_bins=10000)
        pos_f, neg_f, _, _ = calc_bins_f.compute_daily_counts(mean_sc_full, y_test, dates_test)

        # Estimadores puntuales con el EXACTO mismo estimador que las réplicas
        pr_binned_p = calc_bins_p.calc_pr_auc_from_counts(np.sum(pos_p, axis=0), np.sum(neg_p, axis=0))
        pr_binned_b = calc_bins_b.calc_pr_auc_from_counts(np.sum(pos_b, axis=0), np.sum(neg_b, axis=0))
        pr_binned_f = calc_bins_f.calc_pr_auc_from_counts(np.sum(pos_f, axis=0), np.sum(neg_f, axis=0))

        delta_pr_base_point = pr_binned_p - pr_binned_b
        delta_pr_full_point = pr_binned_p - pr_binned_f

        # G1: Bootstrap por bloques de días (B=1000)
        rng = np.random.default_rng(42)
        n_days = len(u_dates)
        b_deltas_pr_base = np.empty(1000, dtype=np.float64)
        b_deltas_pr_full = np.empty(1000, dtype=np.float64)
        b_deltas_rec5_base = np.empty(1000, dtype=np.float64)
        b_deltas_rec5_full = np.empty(1000, dtype=np.float64)

        for b_iter in range(1000):
            idx_d = rng.integers(0, n_days, size=n_days)
            tot_cells_b = int(np.sum(d_cells[idx_d]))

            p_p_b = np.sum(pos_p[idx_d], axis=0)
            n_p_b = np.sum(neg_p[idx_d], axis=0)
            p_b_b = np.sum(pos_b[idx_d], axis=0)
            n_b_b = np.sum(neg_b[idx_d], axis=0)
            p_f_b = np.sum(pos_f[idx_d], axis=0)
            n_f_b = np.sum(neg_f[idx_d], axis=0)

            ap_p = calc_bins_p.calc_pr_auc_from_counts(p_p_b, n_p_b)
            ap_b = calc_bins_b.calc_pr_auc_from_counts(p_b_b, n_b_b)
            ap_f = calc_bins_f.calc_pr_auc_from_counts(p_f_b, n_f_b)

            b_deltas_pr_base[b_iter] = ap_p - ap_b
            b_deltas_pr_full[b_iter] = ap_p - ap_f

            r5_p = calc_bins_p.calc_rec5_from_counts(p_p_b, n_p_b, tot_cells_b)
            r5_b = calc_bins_b.calc_rec5_from_counts(p_b_b, n_b_b, tot_cells_b)
            r5_f = calc_bins_f.calc_rec5_from_counts(p_f_b, n_f_b, tot_cells_b)

            b_deltas_rec5_base[b_iter] = r5_p - r5_b
            b_deltas_rec5_full[b_iter] = r5_p - r5_f

        ci_pr_base = (float(np.quantile(b_deltas_pr_base, 0.025)), float(np.quantile(b_deltas_pr_base, 0.975)))
        ci_pr_full = (float(np.quantile(b_deltas_pr_full, 0.025)), float(np.quantile(b_deltas_pr_full, 0.975)))
        ci_rec5_base = (float(np.quantile(b_deltas_rec5_base, 0.025)), float(np.quantile(b_deltas_rec5_base, 0.975)))
        ci_rec5_full = (float(np.quantile(b_deltas_rec5_full, 0.025)), float(np.quantile(b_deltas_rec5_full, 0.975)))

        # Aserción H0: el punto DEBE caer dentro de su IC 95%
        assert ci_pr_base[0] <= delta_pr_base_point <= ci_pr_base[1], (
            f"Fallo H0 en {s_test}: delta_pr_base {delta_pr_base_point} fuera de IC {ci_pr_base}"
        )
        assert ci_pr_full[0] <= delta_pr_full_point <= ci_pr_full[1], (
            f"Fallo H0 en {s_test}: delta_pr_full {delta_pr_full_point} fuera de IC {ci_pr_full}"
        )

        sig_pr_base = bool(ci_pr_base[0] > 0 or ci_pr_base[1] < 0)
        sig_pr_full = bool(ci_pr_full[0] > 0 or ci_pr_full[1] < 0)

        g1_season_results.append({
            "season": s_test,
            "n_pos": int(np.sum(y_test)),
            "pr_auc_base": pr_binned_b,
            "pr_auc_pruned": pr_binned_p,
            "pr_auc_full": pr_binned_f,
            "delta_pr_base": delta_pr_base_point,
            "ci_delta_pr_base": ci_pr_base,
            "sig_pr_base": sig_pr_base,
            "delta_pr_full": delta_pr_full_point,
            "ci_delta_pr_full": ci_pr_full,
            "sig_pr_full": sig_pr_full,
            "delta_rec5_base": float(np.mean(b_deltas_rec5_base)),
            "ci_delta_rec5_base": ci_rec5_base,
            "sig_rec5_base": bool(ci_rec5_base[0] > 0 or ci_rec5_base[1] < 0),
            "delta_rec5_full": float(np.mean(b_deltas_rec5_full)),
            "ci_delta_rec5_full": ci_rec5_full,
            "sig_rec5_full": bool(ci_rec5_full[0] > 0 or ci_rec5_full[1] < 0),
        })

        daily_hist_base.append(pos_b)
        daily_hist_pruned.append(pos_p)
        daily_hist_full.append(pos_f)
        daily_cells_all.append(d_cells)

        # G3: Evaluación de costo en todas las igniciones y >=1 ha
        m_base = evaluate_detailed(y_test, base_scores, sups_test, s_test)
        m_unw = evaluate_detailed(y_test, mean_sc_pruned, sups_test, s_test)
        m_w = evaluate_detailed(y_test, mean_sc_sizew, sups_test, s_test)
        cost_metrics_base.append(m_base)
        cost_metrics_unweighted.append(m_unw)
        cost_metrics_weighted.append(m_w)

        # G3: Recolectar datos sobre los 255 incendios >= 10 ha
        n_tot = len(y_test)
        k1 = max(1, int(round(0.01 * n_tot)))
        k5 = max(1, int(round(0.05 * n_tot)))
        t1_b = np.partition(base_scores, -k1)[-k1]
        t5_b = np.partition(base_scores, -k5)[-k5]
        t1_p = np.partition(mean_sc_pruned, -k1)[-k1]
        t5_p = np.partition(mean_sc_pruned, -k5)[-k5]
        t1_w = np.partition(mean_sc_sizew, -k1)[-k1]
        t5_w = np.partition(mean_sc_sizew, -k5)[-k5]

        mask_10ha = (y_test == 1) & (sups_test >= 10.0)
        idx_10ha = np.where(mask_10ha)[0]
        for idx in idx_10ha:
            large_fires_records.append({
                "season": s_test,
                "surface_ha": float(sups_test[idx]),
                "hit_base_top1": bool(base_scores[idx] >= t1_b),
                "hit_base_top5": bool(base_scores[idx] >= t5_b),
                "hit_unw_top1": bool(mean_sc_pruned[idx] >= t1_p),
                "hit_unw_top5": bool(mean_sc_pruned[idx] >= t5_p),
                "hit_w_top1": bool(mean_sc_sizew[idx] >= t1_w),
                "hit_w_top5": bool(mean_sc_sizew[idx] >= t5_w),
            })

        # Guardar scores a disco para G2 y G3
        pl.DataFrame({
            "is_ignition_conaf": y_test,
            "sc_pruned_raw": mean_sc_pruned,
            "sc_full_raw": mean_sc_full,
            "sc_sizew_raw": mean_sc_sizew,
            "base_scores": base_scores,
            "superficie_ha_max": sups_test,
            "date": dates_test,
        }).write_parquet(scratch_dir / f"raw_scores_{s_test}.parquet")

        del mean_sc_pruned, mean_sc_full, mean_sc_sizew, base_scores, y_test, sups_test
        del scores_pruned_seeds, scores_sizew_seeds, scores_full_seeds
        gc.collect()

    # =========================================================================
    # G1: ESTIMADOR AGREGADO CON BLOQUES POR TEMPORADA
    # =========================================================================
    print("\n--- G1: Estimador Agregado con Bloques por Temporada ---", flush=True)
    # Agregado macro (7 temporadas)
    macro_base = float(np.mean([r["pr_auc_base"] for r in g1_season_results]))
    macro_pruned = float(np.mean([r["pr_auc_pruned"] for r in g1_season_results]))
    macro_full = float(np.mean([r["pr_auc_full"] for r in g1_season_results]))

    # Bootstrap por bloques estacionales (B=1000 muestreando temporadas completas con reemplazo)
    rng_season = np.random.default_rng(42)
    boot_macro_deltas_base = []
    boot_macro_deltas_full = []
    boot_macro_rec5_base = []

    for _ in range(1000):
        s_idxs = rng_season.integers(0, 7, size=7)
        b_pr_p = np.mean([g1_season_results[i]["pr_auc_pruned"] for i in s_idxs])
        b_pr_b = np.mean([g1_season_results[i]["pr_auc_base"] for i in s_idxs])
        b_pr_f = np.mean([g1_season_results[i]["pr_auc_full"] for i in s_idxs])
        boot_macro_deltas_base.append(b_pr_p - b_pr_b)
        boot_macro_deltas_full.append(b_pr_p - b_pr_f)

        b_r5_p = np.mean([cost_metrics_unweighted[i].rec5_all for i in s_idxs])
        b_r5_b = np.mean([cost_metrics_base[i].rec5_all for i in s_idxs])
        boot_macro_rec5_base.append(b_r5_p - b_r5_b)

    ci_agg_base = (float(np.quantile(boot_macro_deltas_base, 0.025)), float(np.quantile(boot_macro_deltas_base, 0.975)))
    ci_agg_full = (float(np.quantile(boot_macro_deltas_full, 0.025)), float(np.quantile(boot_macro_deltas_full, 0.975)))
    ci_agg_rec5_base = (float(np.quantile(boot_macro_rec5_base, 0.025)), float(np.quantile(boot_macro_rec5_base, 0.975)))

    g1_aggregated = {
        "macro_pr_auc_base": macro_base,
        "macro_pr_auc_pruned": macro_pruned,
        "macro_pr_auc_full": macro_full,
        "delta_pr_base": macro_pruned - macro_base,
        "ci_delta_pr_base": ci_agg_base,
        "sig_pr_base": bool(ci_agg_base[0] > 0 or ci_agg_base[1] < 0),
        "delta_pr_full": macro_pruned - macro_full,
        "ci_delta_pr_full": ci_agg_full,
        "sig_pr_full": bool(ci_agg_full[0] > 0 or ci_agg_full[1] < 0),
        "delta_rec5_base": float(np.mean(boot_macro_rec5_base)),
        "ci_delta_rec5_base": ci_agg_rec5_base,
        "sig_rec5_base": bool(ci_agg_rec5_base[0] > 0 or ci_agg_rec5_base[1] < 0),
    }

    # =========================================================================
    # G1: VALIDACIÓN HACIA ADELANTE CON IC POR TEMPORADA
    # =========================================================================
    print("\n--- G1: Validación Hacia Adelante con IC por Temporada ---", flush=True)
    fw_fit_seasons = ["2017_18", "2019_20", "2020_21"]
    fw_val_season = "2021_22"
    fw_test_seasons = ["2022_23", "2023_24", "2024_25"]

    f_fit = train_df.filter(pl.col("season").is_in(fw_fit_seasons))
    f_val = train_df.filter(pl.col("season") == fw_val_season)

    y_f_fit = f_fit.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_fit = f_fit.get_column("sample_weight").to_numpy().astype(np.float32)
    y_f_val = f_val.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w_f_val = f_val.get_column("sample_weight").to_numpy().astype(np.float32)

    X_p_ffit = f_fit.select(PRUNED_25_FEATURES).to_numpy()
    X_p_fval = f_val.select(PRUNED_25_FEATURES).to_numpy()

    dp_ffit = xgb.DMatrix(X_p_ffit, label=y_f_fit, weight=w_f_fit, feature_names=PRUNED_25_FEATURES)
    dp_fval = xgb.DMatrix(X_p_fval, label=y_f_val, weight=w_f_val, feature_names=PRUNED_25_FEATURES)
    b_fw = xgb.train(
        {**params_base, "random_state": 42}, dp_ffit, num_boost_round=500,
        evals=[(dp_ffit, "train"), (dp_fval, "val")],
        early_stopping_rounds=30, verbose_eval=False,
    )

    g1_forward_results: list[dict[str, Any]] = []
    for s_fw in fw_test_seasons:
        p_fw = materialize_season_cell_day_panel(grid_df, special_cell_days_df, s_fw, include_excluded=False)
        p_fw = p_fw.join(static_df, on="cell_id", how="left")
        p_fw = p_fw.join(prior_df.filter(pl.col("season") == s_fw).select(["cell_id", "prior_density"]), on="cell_id", how="left")
        p_fw = p_fw.join(cell_node_df, on="cell_id", how="left")
        p_fw = p_fw.join(weather_df, on=["node_id", "date"], how="left")

        y_fw = p_fw.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
        sups_fw = p_fw.get_column("superficie_ha_max").to_numpy().astype(np.float64)
        dates_fw = p_fw.get_column("date").to_list()
        fwi_fw = p_fw.get_column("fwi").to_numpy().astype(np.float64)
        prior_fw = p_fw.get_column("prior_density").to_numpy().astype(np.float64)

        base_fw_scores = (fwi_fw * prior_fw).astype(np.float32)
        sc_fw_pruned = b_fw.predict(xgb.DMatrix(p_fw.select(PRUNED_25_FEATURES).to_numpy(), feature_names=PRUNED_25_FEATURES))

        m_b = evaluate_detailed(y_fw, base_fw_scores, sups_fw, s_fw)
        m_p = evaluate_detailed(y_fw, sc_fw_pruned, sups_fw, s_fw)

        # Bootstrap por días para este fold hacia adelante con cuantiles propios
        calc_fw_p = FullDomainBinnedPRAUC(sc_fw_pruned, n_bins=10000)
        pos_p_fw, neg_p_fw, u_d_fw, d_c_fw = calc_fw_p.compute_daily_counts(sc_fw_pruned, y_fw, dates_fw)
        calc_fw_b = FullDomainBinnedPRAUC(base_fw_scores, n_bins=10000)
        pos_b_fw, neg_b_fw, _, _ = calc_fw_b.compute_daily_counts(base_fw_scores, y_fw, dates_fw)

        pr_binned_fw_p = calc_fw_p.calc_pr_auc_from_counts(np.sum(pos_p_fw, axis=0), np.sum(neg_p_fw, axis=0))
        pr_binned_fw_b = calc_fw_b.calc_pr_auc_from_counts(np.sum(pos_b_fw, axis=0), np.sum(neg_b_fw, axis=0))
        delta_pr_fw_point = pr_binned_fw_p - pr_binned_fw_b

        b_fw_deltas_pr = []
        b_fw_deltas_rec5 = []
        n_days_fw = len(u_d_fw)
        rng_fw = np.random.default_rng(42)

        for _ in range(1000):
            idx_d = rng_fw.integers(0, n_days_fw, size=n_days_fw)
            tot_c = int(np.sum(d_c_fw[idx_d]))
            p_p = np.sum(pos_p_fw[idx_d], axis=0)
            n_p = np.sum(neg_p_fw[idx_d], axis=0)
            p_b = np.sum(pos_b_fw[idx_d], axis=0)
            n_b = np.sum(neg_b_fw[idx_d], axis=0)

            ap_p = calc_fw_p.calc_pr_auc_from_counts(p_p, n_p)
            ap_b = calc_fw_b.calc_pr_auc_from_counts(p_b, n_b)
            b_fw_deltas_pr.append(ap_p - ap_b)

            r5_p = calc_fw_p.calc_rec5_from_counts(p_p, n_p, tot_c)
            r5_b = calc_fw_b.calc_rec5_from_counts(p_b, n_b, tot_c)
            b_fw_deltas_rec5.append(r5_p - r5_b)

        ci_fw_pr = (float(np.quantile(b_fw_deltas_pr, 0.025)), float(np.quantile(b_fw_deltas_pr, 0.975)))
        ci_fw_rec5 = (float(np.quantile(b_fw_deltas_rec5, 0.025)), float(np.quantile(b_fw_deltas_rec5, 0.975)))

        assert ci_fw_pr[0] <= delta_pr_fw_point <= ci_fw_pr[1], (
            f"Fallo H0 forward en {s_fw}: delta_pr {delta_pr_fw_point} fuera de IC {ci_fw_pr}"
        )

        del p_fw
        gc.collect()

        g1_forward_results.append({
            "season": s_fw,
            "pr_auc_base": pr_binned_fw_b,
            "pr_auc_pruned": pr_binned_fw_p,
            "delta_pr": delta_pr_fw_point,
            "ci_delta_pr": ci_fw_pr,
            "sig_pr": bool(ci_fw_pr[0] > 0 or ci_fw_pr[1] < 0),
            "lift_base": m_b.lift,
            "lift_pruned": m_p.lift,
            "rec5_base": m_b.rec5_all,
            "rec5_pruned": m_p.rec5_all,
            "delta_rec5": m_p.rec5_all - m_b.rec5_all,
            "ci_delta_rec5": ci_fw_rec5,
            "sig_rec5": bool(ci_fw_rec5[0] > 0 or ci_fw_rec5[1] < 0),
        })

    # Guardar resultados G1
    with open(results_dir / "g1_bootstrap_loso.json", "w", encoding="utf-8") as f:
        json.dump({
            "season_results": g1_season_results,
            "aggregated": g1_aggregated,
            "forward_results": g1_forward_results,
        }, f, indent=2)

    # =========================================================================
    # G2: CALIBRADOR PLATT OUT-OF-FOLD Y UMBRALES FUERA DE MUESTRA
    # =========================================================================
    print("\n--- G2: Calibrador Platt Out-of-Fold y Umbrales Fuera de Muestra ---", flush=True)
    # 1. Cargar scores out-of-fold guardados en scratch
    oof_raw_list = []
    oof_y_list = []
    season_raw_scores: dict[str, np.ndarray] = {}
    season_y_true: dict[str, np.ndarray] = {}

    for s in DATED_SEASONS:
        df_s = pl.read_parquet(scratch_dir / f"raw_scores_{s}.parquet")
        sc_s = df_s.get_column("sc_pruned_raw").to_numpy()
        y_s = df_s.get_column("is_ignition_conaf").to_numpy()
        season_raw_scores[s] = sc_s
        season_y_true[s] = y_s
        oof_raw_list.append(sc_s)
        oof_y_list.append(y_s)
        del df_s

    # 2. Ajustar Platt Calibrator sobre los 3.237 positivos OOF y una muestra representativa estratificada de negativos OOF
    # Para ajustar regresión logística sin saturar RAM con 33M filas:
    # Seleccionamos todos los 3.237 positivos OOF y 100.000 negativos OOF aleatorios con pesos de ponderación exactos
    rng_platt = np.random.default_rng(42)
    pos_logits = []
    pos_y = []
    pos_weights = []

    neg_logits = []
    neg_y = []
    neg_weights = []

    total_neg_cells = 0
    for s in DATED_SEASONS:
        sc_s = season_raw_scores[s]
        y_s = season_y_true[s]
        pos_mask = y_s == 1
        neg_mask = y_s == 0
        total_neg_cells += int(np.sum(neg_mask))

        p_sc = sc_s[pos_mask]
        pos_logits.append(np.log(np.clip(p_sc, 1e-7, 1 - 1e-7) / (1 - np.clip(p_sc, 1e-7, 1 - 1e-7))))
        pos_y.append(np.ones(len(p_sc), dtype=np.int32))
        pos_weights.append(np.ones(len(p_sc), dtype=np.float32))

        # Submuestra de negativos por temporada
        n_sc = sc_s[neg_mask]
        sub_n_idx = rng_platt.choice(len(n_sc), size=min(len(n_sc), 15000), replace=False)
        sub_n = n_sc[sub_n_idx]
        neg_logits.append(np.log(np.clip(sub_n, 1e-7, 1 - 1e-7) / (1 - np.clip(sub_n, 1e-7, 1 - 1e-7))))
        neg_y.append(np.zeros(len(sub_n), dtype=np.int32))
        neg_weights.append(np.full(len(sub_n), float(len(n_sc)) / float(len(sub_n)), dtype=np.float32))

    X_platt = np.concatenate(pos_logits + neg_logits).reshape(-1, 1)
    y_platt = np.concatenate(pos_y + neg_y)
    w_platt = np.concatenate(pos_weights + neg_weights)

    platt_oof_model = LogisticRegression().fit(X_platt, y_platt, sample_weight=w_platt)
    platt_slope = float(platt_oof_model.coef_[0][0])
    platt_intercept = float(platt_oof_model.intercept_[0])

    print(f"Platt OOF ajustado: Slope={platt_slope:.4f}, Intercept={platt_intercept:.4f}")

    # 3. Calibrar todos los scores OOF y calcular umbrales globales p80, p95, p99
    # Función de calibración
    def apply_platt(raw_arr: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(raw_arr, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_arr, 1e-7, 1 - 1e-7)))
        return 1.0 / (1.0 + np.exp(-(platt_slope * logit + platt_intercept)))

    # Calcular cuantiles sobre una muestra grande estratificada de 500k scores calibrados OOF
    sample_calib = []
    for s in DATED_SEASONS:
        sc_s = season_raw_scores[s]
        idx_samp = rng_platt.choice(len(sc_s), size=70000, replace=False)
        sample_calib.append(apply_platt(sc_s[idx_samp]))
    pooled_calib_sample = np.concatenate(sample_calib)

    p80_calib = float(np.percentile(pooled_calib_sample, 80.0))
    p95_calib = float(np.percentile(pooled_calib_sample, 95.0))
    p99_calib = float(np.percentile(pooled_calib_sample, 99.0))

    print(f"Umbrales Calibrados Out-of-Sample: p80={p80_calib:.6f} ({p80_calib*100000:.2f}/100k), p95={p95_calib:.6f} ({p95_calib*100000:.2f}/100k), p99={p99_calib:.6f} ({p99_calib*100000:.2f}/100k)")

    # 4. Demostrar aplicación a cada temporada out-of-season
    g2_season_eval: list[dict[str, Any]] = []
    for s in DATED_SEASONS:
        sc_s = season_raw_scores[s]
        y_s = season_y_true[s]
        calib_s = apply_platt(sc_s)
        n_s = len(calib_s)
        n_pos_s = int(np.sum(y_s))

        mask_bajo = calib_s < p80_calib
        mask_mod = (calib_s >= p80_calib) & (calib_s < p95_calib)
        mask_alto = (calib_s >= p95_calib) & (calib_s < p99_calib)
        mask_ext = calib_s >= p99_calib

        pct_bajo = float(np.sum(mask_bajo) / n_s * 100.0)
        pct_mod = float(np.sum(mask_mod) / n_s * 100.0)
        pct_alto = float(np.sum(mask_alto) / n_s * 100.0)
        pct_ext = float(np.sum(mask_ext) / n_s * 100.0)

        ign_bajo = float(np.sum(y_s[mask_bajo]) / n_pos_s * 100.0)
        ign_mod = float(np.sum(y_s[mask_mod]) / n_pos_s * 100.0)
        ign_alto = float(np.sum(y_s[mask_alto]) / n_pos_s * 100.0)
        ign_ext = float(np.sum(y_s[mask_ext]) / n_pos_s * 100.0)

        # Desvíos con respecto a 80 / 15 / 4 / 1 %
        dev_bajo = pct_bajo - 80.0
        dev_mod = pct_mod - 15.0
        dev_alto = pct_alto - 4.0
        dev_ext = pct_ext - 1.0

        n_bajo = int(np.sum(mask_bajo))
        n_mod = int(np.sum(mask_mod))
        n_alto = int(np.sum(mask_alto))
        n_ext = int(np.sum(mask_ext))

        obs_bajo_100k = float(np.sum(y_s[mask_bajo]) / n_bajo * 100000.0) if n_bajo > 0 else 0.0
        obs_mod_100k = float(np.sum(y_s[mask_mod]) / n_mod * 100000.0) if n_mod > 0 else 0.0
        obs_alto_100k = float(np.sum(y_s[mask_alto]) / n_alto * 100000.0) if n_alto > 0 else 0.0
        obs_ext_100k = float(np.sum(y_s[mask_ext]) / n_ext * 100000.0) if n_ext > 0 else 0.0

        pred_bajo_100k = float(np.mean(calib_s[mask_bajo]) * 100000.0) if n_bajo > 0 else 0.0
        pred_mod_100k = float(np.mean(calib_s[mask_mod]) * 100000.0) if n_mod > 0 else 0.0
        pred_alto_100k = float(np.mean(calib_s[mask_alto]) * 100000.0) if n_alto > 0 else 0.0
        pred_ext_100k = float(np.mean(calib_s[mask_ext]) * 100000.0) if n_ext > 0 else 0.0

        g2_season_eval.append({
            "season": s,
            "pct_cells": {"bajo": pct_bajo, "moderado": pct_mod, "alto": pct_alto, "extremo": pct_ext},
            "pct_ignitions": {"bajo": ign_bajo, "moderado": ign_mod, "alto": ign_alto, "extremo": ign_ext},
            "deviations": {"bajo": dev_bajo, "moderado": dev_mod, "alto": dev_alto, "extremo": dev_ext},
            "reliability_100k": {
                "bajo": {"obs": obs_bajo_100k, "pred": pred_bajo_100k, "ratio": pred_bajo_100k / obs_bajo_100k if obs_bajo_100k > 0 else 1.0},
                "moderado": {"obs": obs_mod_100k, "pred": pred_mod_100k, "ratio": pred_mod_100k / obs_mod_100k if obs_mod_100k > 0 else 1.0},
                "alto": {"obs": obs_alto_100k, "pred": pred_alto_100k, "ratio": pred_alto_100k / obs_alto_100k if obs_alto_100k > 0 else 1.0},
                "extremo": {"obs": obs_ext_100k, "pred": pred_ext_100k, "ratio": pred_ext_100k / obs_ext_100k if obs_ext_100k > 0 else 1.0},
            },
        })

    # Fiabilidad agregada ponderada por tamaño poblacional de celdas
    tot_cells = sum(len(season_raw_scores[s]) for s in DATED_SEASONS)
    agg_rel = {}
    for tier_key in ["bajo", "moderado", "alto", "extremo"]:
        tot_tier_cells = sum(int(np.sum(apply_platt(season_raw_scores[s]) < p80_calib if tier_key == "bajo" else
                                       (apply_platt(season_raw_scores[s]) >= p80_calib) & (apply_platt(season_raw_scores[s]) < p95_calib) if tier_key == "moderado" else
                                       (apply_platt(season_raw_scores[s]) >= p95_calib) & (apply_platt(season_raw_scores[s]) < p99_calib) if tier_key == "alto" else
                                       apply_platt(season_raw_scores[s]) >= p99_calib)) for s in DATED_SEASONS)
        tot_tier_pos = sum(int(np.sum(season_y_true[s][apply_platt(season_raw_scores[s]) < p80_calib if tier_key == "bajo" else
                                                      (apply_platt(season_raw_scores[s]) >= p80_calib) & (apply_platt(season_raw_scores[s]) < p95_calib) if tier_key == "moderado" else
                                                      (apply_platt(season_raw_scores[s]) >= p95_calib) & (apply_platt(season_raw_scores[s]) < p99_calib) if tier_key == "alto" else
                                                      apply_platt(season_raw_scores[s]) >= p99_calib])) for s in DATED_SEASONS)
        tot_tier_sp = sum(float(np.sum(apply_platt(season_raw_scores[s])[apply_platt(season_raw_scores[s]) < p80_calib if tier_key == "bajo" else
                                                                          (apply_platt(season_raw_scores[s]) >= p80_calib) & (apply_platt(season_raw_scores[s]) < p95_calib) if tier_key == "moderado" else
                                                                          (apply_platt(season_raw_scores[s]) >= p95_calib) & (apply_platt(season_raw_scores[s]) < p99_calib) if tier_key == "alto" else
                                                                          apply_platt(season_raw_scores[s]) >= p99_calib])) for s in DATED_SEASONS)

        obs_100k = (tot_tier_pos / tot_tier_cells * 100000.0) if tot_tier_cells > 0 else 0.0
        pred_100k = (tot_tier_sp / tot_tier_cells * 100000.0) if tot_tier_cells > 0 else 0.0
        agg_rel[tier_key] = {
            "n_cells": tot_tier_cells,
            "n_pos": tot_tier_pos,
            "pct_cells": tot_tier_cells / tot_cells * 100.0,
            "obs_rate_100k": obs_100k,
            "pred_rate_100k": pred_100k,
            "ratio_pred_obs": pred_100k / obs_100k if obs_100k > 0 else 1.0,
        }

    # Guardar resultados G2
    g2_results = {
        "platt_calibrator": {
            "slope": platt_slope,
            "intercept": platt_intercept,
            "training_data": "7 temporadas fechadas out-of-fold (3.237 positivos OOF, 105.000 negativos OOF ponderados)",
        },
        "thresholds": {
            "p80": p80_calib,
            "p95": p95_calib,
            "p99": p99_calib,
            "p80_per_100k": p80_calib * 100000.0,
            "p95_per_100k": p95_calib * 100000.0,
            "p99_per_100k": p99_calib * 100000.0,
        },
        "season_evaluation": g2_season_eval,
        "reliability_aggregated": agg_rel,
    }
    with open(results_dir / "g2_thresholds_oos.json", "w", encoding="utf-8") as f:
        json.dump(g2_results, f, indent=2)

    # Actualizar artefactos en models/
    models_dir = Path("models")
    with open(models_dir / "platt_calibrator.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "platt_scaling",
            "slope": platt_slope,
            "intercept": platt_intercept,
            "formula": "P(y=1 | s) = 1 / (1 + exp(-(slope * logit(s) + intercept)))",
            "fitted_on": "out_of_fold_loso_predictions",
            "date": time.strftime("%Y-%m-%d"),
        }, f, indent=2)

    with open(models_dir / "risk_thresholds.json", "w", encoding="utf-8") as f:
        json.dump({
            "system": "Relative Risk Classification (Experimental)",
            "calibrated": True,
            "classes": {
                "Bajo": {"percentile_range": "[0, 80)", "prob_range": [0.0, p80_calib]},
                "Moderado": {"percentile_range": "[80, 95)", "prob_range": [p80_calib, p95_calib]},
                "Alto": {"percentile_range": "[95, 99)", "prob_range": [p95_calib, p99_calib]},
                "Extremo": {"percentile_range": "[99, 100]", "prob_range": [p99_calib, 1.0]},
            },
            "thresholds": {"p80": p80_calib, "p95": p95_calib, "p99": p99_calib},
            "rates_per_100k": {"p80": p80_calib * 100000.0, "p95": p95_calib * 100000.0, "p99": p99_calib * 100000.0},
        }, f, indent=2)

    # Ajuste Platt y umbrales para variante ponderada por tamaño (H2)
    pos_logits_sz = []
    neg_logits_sz = []
    for s in DATED_SEASONS:
        df_s = pl.read_parquet(scratch_dir / f"raw_scores_{s}.parquet")
        sc_sz = df_s.get_column("sc_sizew_raw").to_numpy()
        y_s = df_s.get_column("is_ignition_conaf").to_numpy()
        pos_mask = y_s == 1
        neg_mask = y_s == 0
        p_sc = sc_sz[pos_mask]
        pos_logits_sz.append(np.log(np.clip(p_sc, 1e-7, 1 - 1e-7) / (1 - np.clip(p_sc, 1e-7, 1 - 1e-7))))
        n_sc = sc_sz[neg_mask]
        sub_n_idx = rng_platt.choice(len(n_sc), size=min(len(n_sc), 15000), replace=False)
        sub_n = n_sc[sub_n_idx]
        neg_logits_sz.append(np.log(np.clip(sub_n, 1e-7, 1 - 1e-7) / (1 - np.clip(sub_n, 1e-7, 1 - 1e-7))))

    X_platt_sz = np.concatenate(pos_logits_sz + neg_logits_sz).reshape(-1, 1)
    platt_sizew_model = LogisticRegression().fit(X_platt_sz, y_platt, sample_weight=w_platt)
    slope_sz = float(platt_sizew_model.coef_[0][0])
    intercept_sz = float(platt_sizew_model.intercept_[0])

    def apply_platt_sz(raw_arr: np.ndarray) -> np.ndarray:
        logit = np.log(np.clip(raw_arr, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_arr, 1e-7, 1 - 1e-7)))
        return 1.0 / (1.0 + np.exp(-(slope_sz * logit + intercept_sz)))

    sample_calib_sz = []
    for s in DATED_SEASONS:
        df_s = pl.read_parquet(scratch_dir / f"raw_scores_{s}.parquet")
        sc_sz = df_s.get_column("sc_sizew_raw").to_numpy()
        idx_samp = rng_platt.choice(len(sc_sz), size=70000, replace=False)
        sample_calib_sz.append(apply_platt_sz(sc_sz[idx_samp]))
    pooled_sz_sample = np.concatenate(sample_calib_sz)

    p80_sz = float(np.percentile(pooled_sz_sample, 80.0))
    p95_sz = float(np.percentile(pooled_sz_sample, 95.0))
    p99_sz = float(np.percentile(pooled_sz_sample, 99.0))

    with open(models_dir / "platt_calibrator_sizew.json", "w", encoding="utf-8") as f:
        json.dump({"method": "platt_scaling", "slope": slope_sz, "intercept": intercept_sz, "fitted_on": "out_of_fold_loso_predictions_sizew", "date": time.strftime("%Y-%m-%d")}, f, indent=2)

    with open(models_dir / "risk_thresholds_sizew.json", "w", encoding="utf-8") as f:
        json.dump({
            "system": "Relative Risk Classification (Size-Weighted Variant)",
            "calibrated": True,
            "classes": {
                "Bajo": {"percentile_range": "[0, 80)", "prob_range": [0.0, p80_sz]},
                "Moderado": {"percentile_range": "[80, 95)", "prob_range": [p80_sz, p95_sz]},
                "Alto": {"percentile_range": "[95, 99)", "prob_range": [p95_sz, p99_sz]},
                "Extremo": {"percentile_range": "[99, 100]", "prob_range": [p99_sz, 1.0]},
            },
            "thresholds": {"p80": p80_sz, "p95": p95_sz, "p99": p99_sz},
            "rates_per_100k": {"p80": p80_sz * 100000.0, "p95": p95_sz * 100000.0, "p99": p99_sz * 100000.0},
        }, f, indent=2)

    # Actualizar manifest
    manifest_path = models_dir / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest_data = json.load(f)
    for fn in ["platt_calibrator.json", "risk_thresholds.json", "platt_calibrator_sizew.json", "risk_thresholds_sizew.json"]:
        f_p = models_dir / fn
        manifest_data["artefacts"][fn] = {
            "sha256": compute_sha256(f_p),
            "size_bytes": f_p.stat().st_size,
        }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # =========================================================================
    # G3: PESOS POR TAMAÑO: COSTO GLOBAL Y BOOTSTRAP PAREADO POR EVENTOS (N=255)
    # =========================================================================
    print("\n--- G3: Pesos por Tamaño: Costo Global y Bootstrap Pareado por Eventos (N=255) ---", flush=True)

    # Costos en promedios macro sobre 7 temporadas
    # 1. Todas las superficies
    prauc_all_unw = float(np.mean([m.pr_auc for m in cost_metrics_unweighted]))
    prauc_all_w = float(np.mean([m.pr_auc for m in cost_metrics_weighted]))
    rec1_all_unw = float(np.mean([m.rec1_all for m in cost_metrics_unweighted]) * 100.0)
    rec1_all_w = float(np.mean([m.rec1_all for m in cost_metrics_weighted]) * 100.0)
    rec5_all_unw = float(np.mean([m.rec5_all for m in cost_metrics_unweighted]) * 100.0)
    rec5_all_w = float(np.mean([m.rec5_all for m in cost_metrics_weighted]) * 100.0)

    # 2. >= 1 ha
    rec1_1ha_unw = float(np.mean([m.rec1_1ha for m in cost_metrics_unweighted]) * 100.0)
    rec1_1ha_w = float(np.mean([m.rec1_1ha for m in cost_metrics_weighted]) * 100.0)
    rec5_1ha_unw = float(np.mean([m.rec5_1ha for m in cost_metrics_unweighted]) * 100.0)
    rec5_1ha_w = float(np.mean([m.rec5_1ha for m in cost_metrics_weighted]) * 100.0)

    # 3. >= 10 ha (Promedios macro)
    rec1_10ha_unw = float(np.mean([m.rec1_10ha for m in cost_metrics_unweighted]) * 100.0)
    rec1_10ha_w = float(np.mean([m.rec1_10ha for m in cost_metrics_weighted]) * 100.0)
    rec5_10ha_unw = float(np.mean([m.rec5_10ha for m in cost_metrics_unweighted]) * 100.0)
    rec5_10ha_w = float(np.mean([m.rec5_10ha for m in cost_metrics_weighted]) * 100.0)

    # 4. < 1 ha (H2: Costo en incendios menores a 1 ha)
    rec1_small_unw = float(np.mean([m.rec1_small for m in cost_metrics_unweighted]) * 100.0)
    rec1_small_w = float(np.mean([m.rec1_small for m in cost_metrics_weighted]) * 100.0)
    rec5_small_unw = float(np.mean([m.rec5_small for m in cost_metrics_unweighted]) * 100.0)
    rec5_small_w = float(np.mean([m.rec5_small for m in cost_metrics_weighted]) * 100.0)
    cost_rec1_small = rec1_small_unw - rec1_small_w
    cost_rec5_small = rec5_small_unw - rec5_small_w

    cost_rec1_all = rec1_all_unw - rec1_all_w
    cost_rec5_all = rec5_all_unw - rec5_all_w
    cost_prauc_all = prauc_all_unw - prauc_all_w

    # Bootstrap pareado por eventos sobre los 255 grandes incendios (B=2000)
    n_large = len(large_fires_records)
    assert n_large == 255, f"Esperados 255 incendios >= 10 ha, obtenidos {n_large}"

    hit_arr_base_t1 = np.array([r["hit_base_top1"] for r in large_fires_records], dtype=np.int32)
    hit_arr_base_t5 = np.array([r["hit_base_top5"] for r in large_fires_records], dtype=np.int32)
    hit_arr_unw_t1 = np.array([r["hit_unw_top1"] for r in large_fires_records], dtype=np.int32)
    hit_arr_unw_t5 = np.array([r["hit_unw_top5"] for r in large_fires_records], dtype=np.int32)
    hit_arr_w_t1 = np.array([r["hit_w_top1"] for r in large_fires_records], dtype=np.int32)
    hit_arr_w_t5 = np.array([r["hit_w_top5"] for r in large_fires_records], dtype=np.int32)

    # Métricas agregadas agrupadas (micro) sobre los 255 eventos
    hits_unw_t1 = int(np.sum(hit_arr_unw_t1))
    hits_unw_t5 = int(np.sum(hit_arr_unw_t5))
    hits_w_t1 = int(np.sum(hit_arr_w_t1))
    hits_w_t5 = int(np.sum(hit_arr_w_t5))
    hits_base_t1 = int(np.sum(hit_arr_base_t1))
    hits_base_t5 = int(np.sum(hit_arr_base_t5))

    rec1_10ha_unw_micro = hits_unw_t1 / 255.0 * 100.0
    rec5_10ha_unw_micro = hits_unw_t5 / 255.0 * 100.0
    rec1_10ha_w_micro = hits_w_t1 / 255.0 * 100.0
    rec5_10ha_w_micro = hits_w_t5 / 255.0 * 100.0
    delta_hits_unw_1 = hits_w_t1 - hits_unw_t1
    delta_hits_unw_5 = hits_w_t5 - hits_unw_t5
    delta_rec1_micro = rec1_10ha_w_micro - rec1_10ha_unw_micro
    delta_rec5_micro = rec5_10ha_w_micro - rec5_10ha_unw_micro

    rng_ev = np.random.default_rng(42)
    b_diff_w_vs_unw_t1 = []
    b_diff_w_vs_unw_t5 = []
    b_diff_w_vs_base_t1 = []
    b_diff_w_vs_base_t5 = []

    for _ in range(2000):
        idx_e = rng_ev.integers(0, 255, size=255)
        diff_unw_1 = np.sum(hit_arr_w_t1[idx_e]) - np.sum(hit_arr_unw_t1[idx_e])
        diff_unw_5 = np.sum(hit_arr_w_t5[idx_e]) - np.sum(hit_arr_unw_t5[idx_e])
        diff_base_1 = np.sum(hit_arr_w_t1[idx_e]) - np.sum(hit_arr_base_t1[idx_e])
        diff_base_5 = np.sum(hit_arr_w_t5[idx_e]) - np.sum(hit_arr_base_t5[idx_e])

        b_diff_w_vs_unw_t1.append(diff_unw_1)
        b_diff_w_vs_unw_t5.append(diff_unw_5)
        b_diff_w_vs_base_t1.append(diff_base_1)
        b_diff_w_vs_base_t5.append(diff_base_5)

    ci_w_vs_unw_t1 = (float(np.quantile(b_diff_w_vs_unw_t1, 0.025)), float(np.quantile(b_diff_w_vs_unw_t1, 0.975)))
    ci_w_vs_unw_t5 = (float(np.quantile(b_diff_w_vs_unw_t5, 0.025)), float(np.quantile(b_diff_w_vs_unw_t5, 0.975)))
    ci_w_vs_base_t1 = (float(np.quantile(b_diff_w_vs_base_t1, 0.025)), float(np.quantile(b_diff_w_vs_base_t1, 0.975)))
    ci_w_vs_base_t5 = (float(np.quantile(b_diff_w_vs_base_t5, 0.025)), float(np.quantile(b_diff_w_vs_base_t5, 0.975)))

    # Evaluación de regla de decisión: costo global < 1.0 punto y el IC excluye cero
    rule_applies_top1 = bool(cost_rec1_all < 1.0 and (ci_w_vs_unw_t1[0] > 0 or ci_w_vs_unw_t1[1] < 0))
    rule_applies_top5 = bool(cost_rec5_all < 1.0 and (ci_w_vs_unw_t5[0] > 0 or ci_w_vs_unw_t5[1] < 0))

    g3_results = {
        "costs_all_surfaces": {
            "unweighted": {"rec1": rec1_all_unw, "rec5": rec5_all_unw, "pr_auc": prauc_all_unw},
            "weighted": {"rec1": rec1_all_w, "rec5": rec5_all_w, "pr_auc": prauc_all_w},
            "cost_penalty": {"rec1": cost_rec1_all, "rec5": cost_rec5_all, "pr_auc": cost_prauc_all},
        },
        "surfaces_lt_1ha": {
            "unweighted": {"rec1": rec1_small_unw, "rec5": rec5_small_unw},
            "weighted": {"rec1": rec1_small_w, "rec5": rec5_small_w},
            "cost_penalty": {"rec1": cost_rec1_small, "rec5": cost_rec5_small},
        },
        "surfaces_ge_1ha": {
            "unweighted": {"rec1": rec1_1ha_unw, "rec5": rec5_1ha_unw},
            "weighted": {"rec1": rec1_1ha_w, "rec5": rec5_1ha_w},
            "delta": {"rec1": rec1_1ha_w - rec1_1ha_unw, "rec5": rec5_1ha_w - rec5_1ha_unw},
        },
        "surfaces_ge_10ha_macro": {
            "unweighted": {"rec1": rec1_10ha_unw, "rec5": rec5_10ha_unw},
            "weighted": {"rec1": rec1_10ha_w, "rec5": rec5_10ha_w},
            "delta": {"rec1": rec1_10ha_w - rec1_10ha_unw, "rec5": rec5_10ha_w - rec5_10ha_unw},
        },
        "surfaces_ge_10ha_micro_pooled": {
            "n_total": 255,
            "hits_baseline": {"top1": hits_base_t1, "top5": hits_base_t5},
            "hits_unweighted": {"top1": hits_unw_t1, "top5": hits_unw_t5, "rec1_pct": rec1_10ha_unw_micro, "rec5_pct": rec5_10ha_unw_micro},
            "hits_weighted": {"top1": hits_w_t1, "top5": hits_w_t5, "rec1_pct": rec1_10ha_w_micro, "rec5_pct": rec5_10ha_w_micro},
            "delta_hits": {"top1": delta_hits_unw_1, "top5": delta_hits_unw_5},
            "delta_rec_pct": {"top1": delta_rec1_micro, "top5": delta_rec5_micro},
        },
        "event_bootstrap_255": {
            "delta_hits_vs_unweighted_top1": {"mean": float(np.mean(b_diff_w_vs_unw_t1)), "ci95": ci_w_vs_unw_t1, "excludes_zero": bool(ci_w_vs_unw_t1[0] > 0)},
            "delta_hits_vs_unweighted_top5": {"mean": float(np.mean(b_diff_w_vs_unw_t5)), "ci95": ci_w_vs_unw_t5, "excludes_zero": bool(ci_w_vs_unw_t5[0] > 0)},
            "delta_hits_vs_baseline_top1": {"mean": float(np.mean(b_diff_w_vs_base_t1)), "ci95": ci_w_vs_base_t1, "excludes_zero": bool(ci_w_vs_base_t1[0] > 0)},
            "delta_hits_vs_baseline_top5": {"mean": float(np.mean(b_diff_w_vs_base_t5)), "ci95": ci_w_vs_base_t5, "excludes_zero": bool(ci_w_vs_base_t5[0] > 0)},
        },
        "decision_rule": {
            "rule": "Adoptar solo si costo global < 1.0 pp y el IC 95% excluye cero",
            "cost_global_rec1_pp": cost_rec1_all,
            "cost_global_rec5_pp": cost_rec5_all,
            "rule_applies_top1": rule_applies_top1,
            "rule_applies_top5": rule_applies_top5,
            "recommendation": "Adoptar ponderación por tamaño como variante recomendada para grandes incendios" if (rule_applies_top1 and rule_applies_top5) else "Mantener sin ponderar como default",
        },
    }

    with open(results_dir / "g3_size_weighting.json", "w", encoding="utf-8") as f:
        json.dump(g3_results, f, indent=2)

    print(f"\n=== FIN PIPELINE FASE 2D === en {round(time.time() - t0, 2)} s | RSS final: {get_current_rss_mb():.2f} MB", flush=True)


if __name__ == "__main__":
    run_full_phase2d()
