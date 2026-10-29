"""Script congelado de evaluación prospectiva pre-registrada (Temporada 2026_27).

PORQUÉ (Regla R5):
- Implementa el protocolo formal de evaluación prospectiva pre-registrado antes del inicio de la temporada.
- Compara el candidato 'chilefuego_xgb_ensemble_v2' contra el 'Baseline FWI x Prior' bajo inferencia diaria.
- Criterio Principal: Rec@5% en Nov–Abr con IC 95% pareado (bootstrap por bloques de días, B>=1000)
  que excluya estrictamente el cero a favor del modelo ML.
- Criterio Secundario: Rec@20% (generado a partir del backtest de +7.59 pp, a confirmar prospectivamente).
- Fuentes de datos:
  1. Lectura temprana durante la temporada: Clusters FIRMS (type 0 - vegetation fire, standard quality),
     marcada explícitamente como proxy satelital preliminar.
  2. Lectura contractual final: Ocurrencias terrestres de incendios validadas de CONAF (julio 2027) bajo filtros V1-V8.
- Regla de Bloqueo: Sin modificaciones tras el 1 de diciembre de 2026.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import numpy as np
import polars as pl
from chilefuego.models.training import assert_no_temporal_leakage

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preregistration_eval")

ACTIVE_MONTHS = {11, 12, 1, 2, 3, 4}
N_BOOT = 1000


class FastBinnedEvaluator:
    """Evaluador binned vectorizado para bootstrap pareado por bloques de días."""

    def __init__(self, scores: np.ndarray, n_bins: int = 2000) -> None:
        quantiles = np.linspace(0, 1, n_bins + 1)
        self.bin_edges = np.unique(np.quantile(scores, quantiles))
        self.bin_edges[0] = -np.inf
        self.bin_edges[-1] = np.inf
        self.n_bins = len(self.bin_edges) - 1

    def build_daily_matrix(
        self, scores: np.ndarray, y_true: np.ndarray, date_indices: np.ndarray, n_days: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bin_idx = np.clip(np.digitize(scores, self.bin_edges) - 1, 0, self.n_bins - 1)
        pos_daily = np.zeros((n_days, self.n_bins), dtype=np.float32)
        neg_daily = np.zeros((n_days, self.n_bins), dtype=np.float32)
        day_cells = np.bincount(date_indices, minlength=n_days).astype(np.float32)

        pos_mask = y_true == 1
        neg_mask = ~pos_mask
        np.add.at(pos_daily, (date_indices[pos_mask], bin_idx[pos_mask]), 1.0)
        np.add.at(neg_daily, (date_indices[neg_mask], bin_idx[neg_mask]), 1.0)
        return pos_daily, neg_daily, day_cells

    @staticmethod
    def calc_rec_k_batch(P: np.ndarray, N: np.ndarray, total_cells_arr: np.ndarray, frac: float) -> np.ndarray:
        B, n_bins = P.shape
        tp = np.cumsum(P[:, ::-1], axis=1)
        fp = np.cumsum(N[:, ::-1], axis=1)
        cum_cells = tp + fp
        tot_pos = tp[:, -1]
        k_arr = np.maximum(1, np.round(frac * total_cells_arr).astype(np.int64))
        rec_out = np.zeros(B, dtype=np.float64)

        for b in range(B):
            if tot_pos[b] == 0:
                rec_out[b] = 0.0
                continue
            idx = np.searchsorted(cum_cells[b], k_arr[b])
            if idx >= n_bins:
                idx = n_bins - 1
            rec_out[b] = tp[b, idx] / tot_pos[b]
        return rec_out


def run_paired_bootstrap(
    y_true: np.ndarray,
    dates: list[Any],
    s_test: np.ndarray,
    s_ref: np.ndarray,
    fracs: tuple[float, ...] = (0.05, 0.20),
    n_boot: int = N_BOOT,
    seed: int = 42,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique_dates, date_indices = np.unique(dates, return_inverse=True)
    n_dates = len(unique_dates)

    ev_test = FastBinnedEvaluator(s_test)
    ev_ref = FastBinnedEvaluator(s_ref)

    p_test, n_test, d_cells = ev_test.build_daily_matrix(s_test, y_true, date_indices, n_dates)
    p_ref, n_ref, _ = ev_ref.build_daily_matrix(s_ref, y_true, date_indices, n_dates)

    counts_mat = np.zeros((n_boot, n_dates), dtype=np.float32)
    sample_indices = rng.choice(n_dates, size=(n_boot, n_dates), replace=True)
    for b in range(n_boot):
        np.add.at(counts_mat[b], sample_indices[b], 1.0)

    total_cells = counts_mat @ d_cells
    P_test = counts_mat @ p_test
    N_test = counts_mat @ n_test
    P_ref = counts_mat @ p_ref
    N_ref = counts_mat @ n_ref

    res: dict[str, Any] = {}
    for frac in fracs:
        pct_name = int(round(frac * 100))
        r_test = ev_test.calc_rec_k_batch(P_test, N_test, total_cells, frac) * 100.0
        r_ref = ev_ref.calc_rec_k_batch(P_ref, N_ref, total_cells, frac) * 100.0
        d_r = r_test - r_ref
        ci_low = float(np.percentile(d_r, 2.5))
        ci_high = float(np.percentile(d_r, 97.5))
        delta_val = float(np.median(d_r))

        res[f"rec{pct_name}_candidate"] = float(np.mean(r_test))
        res[f"rec{pct_name}_baseline"] = float(np.mean(r_ref))
        res[f"delta_rec{pct_name}_pp"] = delta_val
        res[f"ci_delta_rec{pct_name}_pp"] = [ci_low, ci_high]
        res["significant_superiority"] = bool(ci_low > 0.0)

    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación pre-registrada de temporada prospectiva.")
    parser.add_argument("--log-path", type=str, default="data/prospective_log.parquet", help="Log diario prospectivo")
    parser.add_argument("--labels-path", type=str, required=True, help="Ocurrencias CONAF o clusters FIRMS")
    parser.add_argument("--is-proxy-firms", action="store_true", help="Indica si la evaluación usa proxy FIRMS")
    parser.add_argument("--out-json", type=str, default="results/preregistration_evaluation.json", help="Salida JSON")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        logger.error(f"Log prospectivo no encontrado: {log_path}")
        sys.exit(1)

    df_log = pl.read_parquet(log_path)
    # Filtrar temporada activa nov-abr
    df_eval = df_log.filter(pl.col("date").dt.month().is_in(ACTIVE_MONTHS))

    # Aserción obligatoria de no fuga temporal (S1)
    ensemble_manifest_path = Path("models/chilefuego_xgb_ensemble_v2/manifest.json")
    if ensemble_manifest_path.exists():
        with open(ensemble_manifest_path, encoding="utf-8") as emf:
            manifest_data = json.load(emf)
        train_seasons = manifest_data.get("train_seasons", [])
        if "season" in df_eval.columns:
            eval_seasons = df_eval.get_column("season").unique().to_list()
            assert_no_temporal_leakage(train_seasons, eval_seasons)

    labels_df = pl.read_parquet(args.labels_path)
    merged = df_eval.join(labels_df, on=["cell_id", "date"], how="left")
    merged = merged.with_columns(pl.col("is_positive").fill_null(0))

    y_true = merged.get_column("is_positive").to_numpy().astype(np.int32)
    dates = merged.get_column("date").to_list()
    s_cand = merged.get_column("score_ml_mapped").to_numpy().astype(np.float32)
    s_base = merged.get_column("score_baseline_mapped").to_numpy().astype(np.float32)

    logger.info(f"Evaluando {len(y_true):,} celdas-día con {int(np.sum(y_true))} igniciones...")
    results = run_paired_bootstrap(y_true, dates, s_cand, s_base, fracs=(0.05, 0.20), n_boot=N_BOOT)
    results["is_proxy_firms"] = args.is_proxy_firms
    results["n_pos"] = int(np.sum(y_true))
    results["n_cells"] = len(y_true)
    results["decision_rule"] = {
        "primary_criterion_met": results["rec5_candidate"] > results["rec5_baseline"] and results["ci_delta_rec5_pp"][0] > 0.0,
        "secondary_criterion_met": results["rec20_candidate"] > results["rec20_baseline"] and results["ci_delta_rec20_pp"][0] > 0.0,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Evaluación pre-registrada guardada en {args.out_json}")


if __name__ == "__main__":
    main()
