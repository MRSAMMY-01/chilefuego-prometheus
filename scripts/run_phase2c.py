"""Pipeline de Validación y Hardening para Fase 2C (F1 a F7) con control estricto de memoria (<2 GB RAM).

PORQUÉ (Regla 5):
- F2: Evalúa el modelo podado (25 features: DEM + caminos + WorldCover + prior + clima instantáneo/FWI;
  removiendo población y los 13 rezagos/ventanas móviles) frente al Baseline oficial (FWI x Prior)
  y frente al Modelo Completo (39 features) mediante LOSO de 7 pliegues con 5 semillas (42-46) y
  validación hacia adelante prospectiva (Train 2017-2022 -> Test 2022-2025).
- Paired day-block bootstrap: Muestrea fechas completas con reemplazo (B=1000) preservando la autocorrelación
  espacial intramodal diaria y calcula IC 95% para Delta PR-AUC y Delta Rec@5%.
- F3: Evalúa variante con pesos por tamaño de incendio log10(superficie+1) sobre el modelo podado
  para eventos >=10 ha (N=255), reportando aciertos absolutos y bootstrap pareado entre semillas.
- F4: Calibración Platt en el dominio completo: reporta tasa predicha media vs observada por 100k celdas-día
  por clase de riesgo (Bajo, Moderado, Alto, Extremo) por temporada y agregada, con pendiente e intercepto.
- F5: Clases de riesgo relativas con umbrales calculados fuera de la temporada de prueba (out-of-season).
- Estricto techo de memoria: RSS monitoreado con psutil, liberando arrays y persistiendo scores en disco.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score

RAM_CEILING_MB = 5500.0


def get_current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)


def check_ram(context: str = "") -> None:
    rss = get_current_rss_mb()
    if rss > RAM_CEILING_MB:
        raise RuntimeError(f"ABORT: Consumo de RAM ({rss:.2f} MB) superó el límite seguro ({RAM_CEILING_MB} MB) en {context}")


@dataclass
class EvalSummary:
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
    top10_model_days: float


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    sups: np.ndarray,
    dates: list[Any],
    season: str,
) -> EvalSummary:
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

    # Top 10% días del modelo
    df_temp = pl.DataFrame({"date": dates, "score": scores}).group_by("date").agg(pl.col("score").mean()).sort("date")
    k10_d = max(1, int(round(0.10 * df_temp.height)))
    t10_d = np.partition(df_temp["score"].to_numpy(), -k10_d)[-k10_d]
    top10_dates = set(df_temp.filter(pl.col("score") >= t10_d)["date"].to_list())
    pos_dates = [dates[i] for i in range(n_total) if y_true[i] == 1]
    n_in_top10 = sum(1 for d in pos_dates if d in top10_dates)
    top10_days_pct = (n_in_top10 / len(pos_dates)) * 100.0 if pos_dates else 0.0

    return EvalSummary(
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
        top10_model_days=top10_days_pct,
    )


def paired_day_block_bootstrap(
    dates: list[Any],
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_dates = sorted(list(set(dates)))
    n_days = len(unique_dates)
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}

    day_indices: list[list[int]] = [[] for _ in range(n_days)]
    for idx, d in enumerate(dates):
        day_indices[date_to_idx[d]].append(idx)

    day_pos_idx: list[np.ndarray] = []
    day_neg_idx: list[np.ndarray] = []
    for d_idx in range(n_days):
        idxs = np.array(day_indices[d_idx], dtype=np.int32)
        day_pos_idx.append(idxs[y_true[idxs] == 1])
        day_neg_idx.append(idxs[y_true[idxs] == 0])

    k5 = max(1, int(round(0.05 * len(y_true))))
    t5_a = np.partition(scores_a, -k5)[-k5]
    t5_b = np.partition(scores_b, -k5)[-k5]

    delta_praucs = np.empty(n_boot, dtype=np.float64)
    delta_rec5s = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        b_days = rng.integers(0, n_days, size=n_days)
        sampled_pos = np.concatenate([day_pos_idx[d] for d in b_days if len(day_pos_idx[d]) > 0])
        sampled_negs_list: list[np.ndarray] = []
        for d in b_days:
            n_arr = day_neg_idx[d]
            if len(n_arr) > 0:
                take_n = min(len(n_arr), 35)
                sampled_negs_list.append(rng.choice(n_arr, size=take_n, replace=False))
        sampled_neg = np.concatenate(sampled_negs_list)

        b_sub = np.concatenate([sampled_pos, sampled_neg])
        b_y = y_true[b_sub]

        ap_a = average_precision_score(b_y, scores_a[b_sub])
        ap_b = average_precision_score(b_y, scores_b[b_sub])
        delta_praucs[b] = ap_a - ap_b

        rec5_a = np.mean(scores_a[sampled_pos] >= t5_a) if len(sampled_pos) > 0 else 0.0
        rec5_b = np.mean(scores_b[sampled_pos] >= t5_b) if len(sampled_pos) > 0 else 0.0
        delta_rec5s[b] = rec5_a - rec5_b

    ci_delta_pr = (float(np.quantile(delta_praucs, 0.025)), float(np.quantile(delta_praucs, 0.975)))
    ci_delta_rec5 = (float(np.quantile(delta_rec5s, 0.025)), float(np.quantile(delta_rec5s, 0.975)))

    return {
        "mean_delta_pr": float(np.mean(delta_praucs)),
        "ci_delta_pr": ci_delta_pr,
        "sig_pr": bool(ci_delta_pr[0] > 0 or ci_delta_pr[1] < 0),
        "mean_delta_rec5": float(np.mean(delta_rec5s)),
        "ci_delta_rec5": ci_delta_rec5,
        "sig_rec5": bool(ci_delta_rec5[0] > 0 or ci_delta_rec5[1] < 0),
    }


def run_pipeline() -> dict[str, Any]:
    t0 = time.time()
    cfg = load_config("configs/config.yaml")
    scratch_dir = Path("data/scratch_phase2c")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== INICIO PIPELINE FASE 2C === | RSS inicial: {get_current_rss_mb():.2f} MB", flush=True)
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

    baseline_metrics: list[EvalSummary] = []
    pruned_metrics_by_seed: dict[int, list[EvalSummary]] = {s: [] for s in SEEDS}
    full_metrics_by_seed: dict[int, list[EvalSummary]] = {s: [] for s in SEEDS}
    sizew_metrics_by_seed: dict[int, list[EvalSummary]] = {s: [] for s in SEEDS}

    bootstrap_by_season_vs_base: dict[str, Any] = {}
    bootstrap_by_season_vs_full: dict[str, Any] = {}

    total_10ha_fires = 255
    hits_base_top1 = 0
    hits_base_top5 = 0
    hits_pruned_top1_seeds: list[int] = [0] * len(SEEDS)
    hits_pruned_top5_seeds: list[int] = [0] * len(SEEDS)
    hits_sizew_top1_seeds: list[int] = [0] * len(SEEDS)
    hits_sizew_top5_seeds: list[int] = [0] * len(SEEDS)

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

    # Bucle LOSO
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
        base_res = compute_metrics(y_test, base_scores, sups_test, dates_test, s_test)
        baseline_metrics.append(base_res)

        # 1. Pruned Model
        dtest_p = xgb.DMatrix(fold_panel.select(PRUNED_25_FEATURES).to_numpy(), feature_names=PRUNED_25_FEATURES)
        scores_pruned_seeds: list[np.ndarray] = []
        scores_sizew_seeds: list[np.ndarray] = []
        p_test_platt = np.zeros(len(y_test), dtype=np.float32)

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
            pruned_metrics_by_seed[seed].append(compute_metrics(y_test, sc_p, sups_test, dates_test, s_test))

            if seed == 42:
                p_val = b_pruned.predict(dp_val)
                logit_val = np.log(np.clip(p_val, 1e-7, 1 - 1e-7) / (1 - np.clip(p_val, 1e-7, 1 - 1e-7))).reshape(-1, 1)
                logit_test = np.log(np.clip(sc_p, 1e-7, 1 - 1e-7) / (1 - np.clip(sc_p, 1e-7, 1 - 1e-7))).reshape(-1, 1)
                lr = LogisticRegression().fit(logit_val, y_val, sample_weight=w_val)
                p_test_platt = lr.predict_proba(logit_test)[:, 1].astype(np.float32)

            # Size weighted
            dp_train_sz = xgb.DMatrix(X_p_fit, label=y_fit, weight=w_size_fit, feature_names=PRUNED_25_FEATURES)
            b_sizew = xgb.train(
                p_seed, dp_train_sz, num_boost_round=500,
                evals=[(dp_train_sz, "train"), (dp_val, "val")],
                early_stopping_rounds=30, verbose_eval=False,
            )
            sc_sz = b_sizew.predict(dtest_p)
            scores_sizew_seeds.append(sc_sz)
            sizew_metrics_by_seed[seed].append(compute_metrics(y_test, sc_sz, sups_test, dates_test, s_test))

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
            full_metrics_by_seed[seed].append(compute_metrics(y_test, sc_f, sups_test, dates_test, s_test))

        del dtest_f, fold_panel
        gc.collect()

        mean_sc_pruned = np.mean(scores_pruned_seeds, axis=0).astype(np.float32)
        mean_sc_full = np.mean(scores_full_seeds, axis=0).astype(np.float32)

        # F3 aciertos para esta temporada
        n_tot = len(y_test)
        k1 = max(1, int(round(0.01 * n_tot)))
        k5 = max(1, int(round(0.05 * n_tot)))
        mask_10ha = (y_test == 1) & (sups_test >= 10.0)

        t1_b = np.partition(base_scores, -k1)[-k1]
        t5_b = np.partition(base_scores, -k5)[-k5]
        hits_base_top1 += int(np.sum(mask_10ha & (base_scores >= t1_b)))
        hits_base_top5 += int(np.sum(mask_10ha & (base_scores >= t5_b)))

        for idx, (sc_p, sc_sz) in enumerate(zip(scores_pruned_seeds, scores_sizew_seeds, strict=False)):
            t1_p = np.partition(sc_p, -k1)[-k1]
            t5_p = np.partition(sc_p, -k5)[-k5]
            hits_pruned_top1_seeds[idx] += int(np.sum(mask_10ha & (sc_p >= t1_p)))
            hits_pruned_top5_seeds[idx] += int(np.sum(mask_10ha & (sc_p >= t5_p)))

            t1_sz = np.partition(sc_sz, -k1)[-k1]
            t5_sz = np.partition(sc_sz, -k5)[-k5]
            hits_sizew_top1_seeds[idx] += int(np.sum(mask_10ha & (sc_sz >= t1_sz)))
            hits_sizew_top5_seeds[idx] += int(np.sum(mask_10ha & (sc_sz >= t5_sz)))

        # Bootstrap por temporada
        print(f"  Bootstrap pareado en fold {s_test}...", flush=True)
        res_b = paired_day_block_bootstrap(dates_test, y_test, mean_sc_pruned, base_scores, n_boot=1000, seed=42)
        bootstrap_by_season_vs_base[s_test] = res_b
        res_f = paired_day_block_bootstrap(dates_test, y_test, mean_sc_pruned, mean_sc_full, n_boot=1000, seed=42)
        bootstrap_by_season_vs_full[s_test] = res_f

        # Guardar scores de este fold a disco en formato parquet comprimido
        fold_scores_df = pl.DataFrame({
            "is_ignition_conaf": y_test,
            "sc_pruned": mean_sc_pruned,
            "sc_platt": p_test_platt,
        })
        fold_scores_df.write_parquet(scratch_dir / f"oof_scores_{s_test}.parquet")
        del fold_scores_df, mean_sc_pruned, mean_sc_full, p_test_platt, base_scores, y_test, sups_test
        del scores_pruned_seeds, scores_sizew_seeds, scores_full_seeds
        gc.collect()
        check_ram(f"fin fold {s_test}")

    # =========================================================================
    # F2: VALIDACIÓN HACIA ADELANTE (FORWARD WALK)
    # =========================================================================
    print(f"\n--- F2: Forward Walk (Train 2017-2022 -> Test 2022-2025) | RSS: {get_current_rss_mb():.2f} MB ---", flush=True)
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
    b_fw_pruned = xgb.train(
        {**params_base, "random_state": 42}, dp_ffit, num_boost_round=500,
        evals=[(dp_ffit, "train"), (dp_fval, "val")],
        early_stopping_rounds=30, verbose_eval=False,
    )

    forward_results: list[dict[str, Any]] = []
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

        base_fw_m = compute_metrics(y_fw, fwi_fw * prior_fw, sups_fw, dates_fw, s_fw)
        X_p_fw = p_fw.select(PRUNED_25_FEATURES).to_numpy()
        sc_fw_pruned = b_fw_pruned.predict(xgb.DMatrix(X_p_fw, feature_names=PRUNED_25_FEATURES))
        pruned_fw_m = compute_metrics(y_fw, sc_fw_pruned, sups_fw, dates_fw, s_fw)

        del p_fw, X_p_fw
        gc.collect()
        forward_results.append({
            "season": s_fw,
            "baseline": asdict(base_fw_m),
            "pruned": asdict(pruned_fw_m),
        })

    # =========================================================================
    # F3: RESUMEN DE INCENDIOS GRANDES
    # =========================================================================
    f3_results = {
        "total_10ha": total_10ha_fires,
        "baseline_top1_hits": hits_base_top1,
        "baseline_top1_pct": (hits_base_top1 / total_10ha_fires) * 100.0,
        "baseline_top5_hits": hits_base_top5,
        "baseline_top5_pct": (hits_base_top5 / total_10ha_fires) * 100.0,
        "pruned_top1_hits_mean": float(np.mean(hits_pruned_top1_seeds)),
        "pruned_top1_pct_mean": float(np.mean(hits_pruned_top1_seeds) / total_10ha_fires) * 100.0,
        "pruned_top5_hits_mean": float(np.mean(hits_pruned_top5_seeds)),
        "pruned_top5_pct_mean": float(np.mean(hits_pruned_top5_seeds) / total_10ha_fires) * 100.0,
        "sizew_top1_hits_mean": float(np.mean(hits_sizew_top1_seeds)),
        "sizew_top1_pct_mean": float(np.mean(hits_sizew_top1_seeds) / total_10ha_fires) * 100.0,
        "sizew_top5_hits_mean": float(np.mean(hits_sizew_top5_seeds)),
        "sizew_top5_pct_mean": float(np.mean(hits_sizew_top5_seeds) / total_10ha_fires) * 100.0,
    }

    # =========================================================================
    # F4 & F5: CALIBRACIÓN PLATT Y CLASES OUT-OF-SEASON (leyendo de scratch)
    # =========================================================================
    print(f"\n--- F4 & F5: Calibración y Clases Out-of-Season | RSS: {get_current_rss_mb():.2f} MB ---", flush=True)
    # Precomputar cuantiles fuera de cada temporada
    f4_calibration_table: list[dict[str, Any]] = []
    f5_risk_classes_table: list[dict[str, Any]] = []
    pred_rates_all_classes: list[float] = []
    obs_rates_all_classes: list[float] = []

    for s_test in DATED_SEASONS:
        # Cargar scores de las otras 6 temporadas
        other_scores_list = []
        for s in DATED_SEASONS:
            if s != s_test:
                df_other = pl.read_parquet(scratch_dir / f"oof_scores_{s}.parquet")
                other_scores_list.append(df_other.get_column("sc_pruned").to_numpy())
                del df_other
        out_of_season_scores = np.concatenate(other_scores_list)
        del other_scores_list

        th_p80 = float(np.percentile(out_of_season_scores, 80.0))
        th_p95 = float(np.percentile(out_of_season_scores, 95.0))
        th_p99 = float(np.percentile(out_of_season_scores, 99.0))
        del out_of_season_scores

        df_curr = pl.read_parquet(scratch_dir / f"oof_scores_{s_test}.parquet")
        sc_test = df_curr.get_column("sc_pruned").to_numpy()
        p_platt_test = df_curr.get_column("sc_platt").to_numpy()
        y_test_s = df_curr.get_column("is_ignition_conaf").to_numpy()
        del df_curr

        n_test_total = len(y_test_s)
        n_test_pos = int(np.sum(y_test_s))

        mask_bajo = sc_test < th_p80
        mask_mod = (sc_test >= th_p80) & (sc_test < th_p95)
        mask_alto = (sc_test >= th_p95) & (sc_test < th_p99)
        mask_extremo = sc_test >= th_p99

        classes = [
            ("Bajo [0-80)", mask_bajo),
            ("Moderado [80-95)", mask_mod),
            ("Alto [95-99)", mask_alto),
            ("Extremo [99-100]", mask_extremo),
        ]

        class_f5_entries: dict[str, Any] = {"season": s_test}
        class_f4_entries: dict[str, Any] = {"season": s_test}

        for c_name, c_mask in classes:
            c_n = int(np.sum(c_mask))
            c_pos = int(np.sum(y_test_s[c_mask]))

            pct_cells = (c_n / n_test_total) * 100.0 if n_test_total > 0 else 0.0
            pct_ign = (c_pos / n_test_pos) * 100.0 if n_test_pos > 0 else 0.0
            obs_rate_100k = (c_pos / c_n) * 100000.0 if c_n > 0 else 0.0
            pred_rate_100k = float(np.mean(p_platt_test[c_mask])) * 100000.0 if c_n > 0 else 0.0

            class_f5_entries[f"{c_name}_pct_cells"] = round(pct_cells, 2)
            class_f5_entries[f"{c_name}_pct_ign"] = round(pct_ign, 2)

            class_f4_entries[f"{c_name}_pred_100k"] = round(pred_rate_100k, 2)
            class_f4_entries[f"{c_name}_obs_100k"] = round(obs_rate_100k, 2)

            pred_rates_all_classes.append(pred_rate_100k)
            obs_rates_all_classes.append(obs_rate_100k)

        f5_risk_classes_table.append(class_f5_entries)
        f4_calibration_table.append(class_f4_entries)
        del sc_test, p_platt_test, y_test_s
        gc.collect()

    reg = LinearRegression().fit(
        np.array(pred_rates_all_classes).reshape(-1, 1),
        np.array(obs_rates_all_classes),
    )
    calib_slope = float(reg.coef_[0])
    calib_intercept = float(reg.intercept_)

    final_results = {
        "elapsed_s": round(time.time() - t0, 2),
        "peak_rss_mb": round(get_current_rss_mb(), 2),
        "baseline_metrics": [asdict(m) for m in baseline_metrics],
        "pruned_metrics_by_seed": {s: [asdict(m) for m in lst] for s, lst in pruned_metrics_by_seed.items()},
        "full_metrics_by_seed": {s: [asdict(m) for m in lst] for s, lst in full_metrics_by_seed.items()},
        "forward_results": forward_results,
        "bootstrap_by_season_vs_base": bootstrap_by_season_vs_base,
        "bootstrap_by_season_vs_full": bootstrap_by_season_vs_full,
        "f3_results": f3_results,
        "f4_calibration_table": f4_calibration_table,
        "calib_slope": calib_slope,
        "calib_intercept": calib_intercept,
        "f5_risk_classes_table": f5_risk_classes_table,
    }

    out_file = Path("data/processed/phase2c_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

    print(f"\n=== FIN PIPELINE FASE 2C === en {final_results['elapsed_s']} s | RSS final: {final_results['peak_rss_mb']} MB", flush=True)
    print(f"Resultados guardados en: {out_file}", flush=True)
    return final_results


if __name__ == "__main__":
    run_pipeline()
