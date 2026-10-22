"""Script para formatear e imprimir los resultados de E1 a E7 con precisión."""

from __future__ import annotations

import json

import numpy as np


def main() -> None:
    with open("reports/phase2b_hardening_results.json", encoding="utf-8") as f:
        d = json.load(f)

    b_list = d["baseline_metrics"]
    x_list = d["xgb_seed42_metrics"]

    print("=== E1: TABLA RECONCILIADA (MISMA FUNCION DE EVALUACION) ===")
    print("Temporada | N | Base PR-AUC | Base Lift | Base Rec@1% | Base Rec@5% | Base Rec1>=1ha | Base Rec1>=10ha | Base Top10%D || XGB PR-AUC | XGB Lift | XGB Rec@1% | XGB Rec@5% | XGB Rec1>=1ha | XGB Rec1>=10ha | XGB Top10%D")
    for b, x in zip(b_list, x_list, strict=False):
        print(
            f"{b['season']} | {b['n_pos']} | {b['pr_auc']:.6e} | {b['lift']:5.2f}x | {b['rec1_all']*100:5.2f}% | {b['rec5_all']*100:5.2f}% | {b['rec1_1ha']*100:5.2f}% | {b['rec1_10ha']*100:5.2f}% | {b['top10_model_days']:5.2f}% || "
            f"{x['pr_auc']:.6e} | {x['lift']:5.2f}x | {x['rec1_all']*100:5.2f}% | {x['rec5_all']*100:5.2f}% | {x['rec1_1ha']*100:5.2f}% | {x['rec1_10ha']*100:5.2f}% | {x['top10_model_days']:5.2f}%"
        )

    b_prauc = np.mean([m['pr_auc'] for m in b_list])
    b_lift = np.mean([m['lift'] for m in b_list])
    b_rec1 = np.mean([m['rec1_all'] for m in b_list]) * 100
    b_rec5 = np.mean([m['rec5_all'] for m in b_list]) * 100
    b_rec1_1ha = np.mean([m['rec1_1ha'] for m in b_list]) * 100
    b_rec1_10ha = np.mean([m['rec1_10ha'] for m in b_list]) * 100
    b_top10 = np.mean([m['top10_model_days'] for m in b_list])

    x_prauc = np.mean([m['pr_auc'] for m in x_list])
    x_lift = np.mean([m['lift'] for m in x_list])
    x_rec1 = np.mean([m['rec1_all'] for m in x_list]) * 100
    x_rec5 = np.mean([m['rec5_all'] for m in x_list]) * 100
    x_rec1_1ha = np.mean([m['rec1_1ha'] for m in x_list]) * 100
    x_rec1_10ha = np.mean([m['rec1_10ha'] for m in x_list]) * 100
    x_top10 = np.mean([m['top10_model_days'] for m in x_list])

    print("--- PROMEDIOS MACRO (7 TEMPORADAS) ---")
    print(f"Base Media: {b_prauc:.6e} | {b_lift:5.2f}x | {b_rec1:5.2f}% | {b_rec5:5.2f}% | {b_rec1_1ha:5.2f}% | {b_rec1_10ha:5.2f}% | {b_top10:5.2f}%")
    print(f"XGB  Media: {x_prauc:.6e} | {x_lift:5.2f}x | {x_rec1:5.2f}% | {x_rec5:5.2f}% | {x_rec1_1ha:5.2f}% | {x_rec1_10ha:5.2f}% | {x_top10:5.2f}%")

    # E2 Incertidumbre
    print("\n=== E2: INCERTIDUMBRE (5 SEMILLAS) Y BOOTSTRAP CI 95% ===")
    seeds = ["42", "43", "44", "45", "46"]
    for i, s_name in enumerate([b['season'] for b in b_list]):
        s_praucs = [d["e2_seed_metrics"][sd][i]["pr_auc"] for sd in seeds]
        [d["e2_seed_metrics"][sd][i]["lift"] for sd in seeds]
        s_rec1s = [d["e2_seed_metrics"][sd][i]["rec1_all"] * 100 for sd in seeds]
        ci_p = d["e2_bootstrap_cis"][s_name]["ci_prauc"]
        ci_r = d["e2_bootstrap_cis"][s_name]["ci_rec1"]
        print(f"{s_name}: PR-AUC = {np.mean(s_praucs):.6e} +/- {np.std(s_praucs):.6e} (CI95: [{ci_p[0]:.6e}, {ci_p[1]:.6e}]) | Rec@1% = {np.mean(s_rec1s):.2f}% +/- {np.std(s_rec1s):.2f}% (CI95: [{ci_r[0]*100:.2f}%, {ci_r[1]*100:.2f}%])")

    print("\n=== E2: ABLACIONES POR GRUPO POR TEMPORADA (PR-AUC) ===")
    ab_names = list(d["e2_group_ablations"].keys())
    header = "Temporada | " + " | ".join(ab_names)
    print(header)
    for i, s_name in enumerate([b['season'] for b in b_list]):
        vals = [f"{d['e2_group_ablations'][ab][i]['pr_auc']:.4e}" for ab in ab_names]
        print(f"{s_name} | " + " | ".join(vals))
    print("PROMEDIO  | " + " | ".join([f"{np.mean([d['e2_group_ablations'][ab][i]['pr_auc'] for i in range(7)]):.4e}" for ab in ab_names]))

    # E3 Incendios grandes
    print("\n=== E3: EXPERIMENTO PONDERACION POR TAMAÑO DE INCENDIO (LOG10) ===")
    sz_list = d["xgb_size_weighted_metrics"]
    for i, s_name in enumerate([b['season'] for b in b_list]):
        x_std = x_list[i]
        x_sz = sz_list[i]
        print(f"{s_name} | Total Rec1: {x_std['rec1_all']*100:.2f}% -> {x_sz['rec1_all']*100:.2f}% | >=1ha: {x_std['rec1_1ha']*100:.2f}% -> {x_sz['rec1_1ha']*100:.2f}% | >=10ha: {x_std['rec1_10ha']*100:.2f}% -> {x_sz['rec1_10ha']*100:.2f}%")
    print(f"MEDIA MACRO | Total Rec1: {np.mean([m['rec1_all'] for m in x_list])*100:.2f}% -> {np.mean([m['rec1_all'] for m in sz_list])*100:.2f}% | >=1ha: {np.mean([m['rec1_1ha'] for m in x_list])*100:.2f}% -> {np.mean([m['rec1_1ha'] for m in sz_list])*100:.2f}% | >=10ha: {np.mean([m['rec1_10ha'] for m in x_list])*100:.2f}% -> {np.mean([m['rec1_10ha'] for m in sz_list])*100:.2f}%")
    print(f"MEDIA MACRO | Total Rec5: {np.mean([m['rec5_all'] for m in x_list])*100:.2f}% -> {np.mean([m['rec5_all'] for m in sz_list])*100:.2f}% | >=1ha: {np.mean([m['rec5_1ha'] for m in x_list])*100:.2f}% -> {np.mean([m['rec5_1ha'] for m in sz_list])*100:.2f}% | >=10ha: {np.mean([m['rec5_10ha'] for m in x_list])*100:.2f}% -> {np.mean([m['rec5_10ha'] for m in sz_list])*100:.2f}%")

    # E4 Calibración
    print("\n=== E4: CALIBRACION EN DOMINIO COMPLETO ===")
    cal_list = d["e4_calibration_results"]
    for c in cal_list:
        print(f"{c['season']} | Brier Const: {c['brier_const']:.6e} | Brier Raw: {c['brier_raw']:.6e} | Brier Platt: {c['brier_platt']:.6e} | Brier Iso: {c['brier_iso']:.6e}")
    print(f"MEDIAS | Brier Const: {np.mean([c['brier_const'] for c in cal_list]):.6e} | Brier Raw: {np.mean([c['brier_raw'] for c in cal_list]):.6e} | Brier Platt: {np.mean([c['brier_platt'] for c in cal_list]):.6e} | Brier Iso: {np.mean([c['brier_iso'] for c in cal_list]):.6e}")

    # Tabla de fiabilidad tramos por 100k
    print("\n--- Fiabilidad Platt (Igniciones por 100k celdas-dia, Temporada 2024_25) ---")
    bins_24_25 = d["e4_reliability_bins"][-1]
    for p, o in zip(bins_24_25["pred_per_100k"], bins_24_25["obs_per_100k"], strict=False):
        print(f"Predicho: {p:6.1f} / 100k | Observado: {o:6.1f} / 100k")

    # E5 Forward walk
    print("\n=== E5: VALIDACION HACIA ADELANTE (TRAIN 2017-2022 -> TEST 2022-2025) ===")
    for fw in d["e5_forward_results"]:
        s = fw["season"]
        b = fw["baseline"]
        x = fw["xgboost"]
        print(f"{s} | Base PR-AUC: {b['pr_auc']:.6e} (Lift {b['lift']:.2f}x, Rec1 {b['rec1_all']*100:.2f}%) || XGB PR-AUC: {x['pr_auc']:.6e} (Lift {x['lift']:.2f}x, Rec1 {x['rec1_all']*100:.2f}%)")

    # E7 Clases de riesgo
    print("\n=== E7: CLASES DE RIESGO FIJAS ===")
    for t in d["e7_risk_tiers"]:
        print(f"{t['tier']:20s} | % Celdas-dia: {t['pct_cell_days']:6.2f}% | % Igniciones: {t['pct_ignitions']:6.2f}% (N={t['n_ignitions']}) | Tasa por 100k: {t['rate_per_100k']:6.2f}")

if __name__ == "__main__":
    main()
