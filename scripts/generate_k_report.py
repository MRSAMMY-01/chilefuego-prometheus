"""Generador de Reporte Fase 3A / J2 / K1-K6 / L3-L5 a partir de results/j2_nwp_skew_study.json.

PORQUÉ (Reglas Asesor):
- Cumple la regla estricta: 'tablas generadas por script desde results/*.json'.
- Cero invención: formatea exactamente los datos numéricos calculados por el estudio.
- Reporte conciso (máximo 100 líneas) con estructura PORQUÉ antes del CÓMO.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    json_path = Path("results/j2_nwp_skew_study.json")
    if not json_path.exists():
        print(f"ERROR: Archivo {json_path} no encontrado.")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    evals = {e["season"]: e for e in data.get("evaluations", [])}
    qm = data.get("quantile_mapping_cross_validation", [])
    crit = data.get("criteria_evaluation", {})

    print("# REPORTE EJECUTIVO: IMPACTO NWP, PERSISTENCIA Y MAPEO DE CUANTILES (L3-L5 / K1-K6)\n")

    print("## 1. PORQUÉ: JUSTIFICACIÓN METODOLÓGICA Y DESACOPLE OPERATIVO")
    print(
        "- Reemplaza reanálisis ERA5-Land por pronósticos numéricos del tiempo (NWP GFS de Open-Meteo)."
    )
    print(
        "- Corrige la filtración de datos evaluando Modelo B (validación hacia adelante entrenado hasta 2021_22)."
    )
    print(
        "- Emulación a un paso: estado FWI de D-1 desde ERA5-Land; clima de D desde pronóstico o persistencia."
    )
    print(
        "- Mapeo monótono de cuantiles (Q-Q) ajustado en una temporada y evaluado en la otra (nunca en la misma).\n"
    )

    print("## 2. TABLA 1: DESEMPEÑO MODELO B VS BASELINE POR TEMPORADA Y HORIZONTE")
    print(
        "| Temporada | Enfoque | Insumo / Horizonte | PR-AUC | Delta PR-AUC [IC 95%] | Rec@5% (%) | Delta Rec@5% pp [IC 95%] |"
    )
    print("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |")

    for s in ["2023_24", "2024_25"]:
        if s not in evals:
            continue
        ev = evals[s]

        # Modelo B
        m_era = ev["model_xgb_pruned_25"]["era5_land"]
        m_h0 = ev["model_xgb_pruned_25"]["nwp_h0_source_skew"]
        m_h1 = ev["model_xgb_pruned_25"]["nwp_h1_persistence_24h"]

        print(
            f"| {s} | Modelo B | ERA5-Land (Reanálisis) | {m_era['pr_auc']:.6f} | [Ref] | {m_era['rec5_pct']:.2f}% | [Ref] |"
        )
        ci_pr_h0 = f"[{m_h0['ci_delta_pr_auc'][0]:+.6f}, {m_h0['ci_delta_pr_auc'][1]:+.6f}]"
        ci_r5_h0 = f"[{m_h0['ci_delta_rec5_pp'][0]:+.2f}, {m_h0['ci_delta_rec5_pp'][1]:+.2f}]"
        print(
            f"| {s} | Modelo B | NWP H0 (Nowcast ~0h) | {m_h0['pr_auc']:.6f} | {m_h0['delta_pr_auc']:+.6f} {ci_pr_h0} | {m_h0['rec5_pct']:.2f}% | {m_h0['delta_rec5_pp']:+.2f} {ci_r5_h0} |"
        )

        ci_pr_h1 = f"[{m_h1['ci_delta_pr_auc'][0]:+.6f}, {m_h1['ci_delta_pr_auc'][1]:+.6f}]"
        ci_r5_h1 = f"[{m_h1['ci_delta_rec5_pp'][0]:+.2f}, {m_h1['ci_delta_rec5_pp'][1]:+.2f}]"
        print(
            f"| {s} | Modelo B | Persistencia 24h (D-1) | {m_h1['pr_auc']:.6f} | {m_h1['delta_pr_auc']:+.6f} {ci_pr_h1} | {m_h1['rec5_pct']:.2f}% | {m_h1['delta_rec5_pp']:+.2f} {ci_r5_h1} |"
        )

        if "nwp_h1_real_forecast_24h" in ev["model_xgb_pruned_25"]:
            m_real = ev["model_xgb_pruned_25"]["nwp_h1_real_forecast_24h"]
            ci_pr_r = f"[{m_real['ci_delta_pr_auc'][0]:+.6f}, {m_real['ci_delta_pr_auc'][1]:+.6f}]"
            ci_r5_r = (
                f"[{m_real['ci_delta_rec5_pp'][0]:+.2f}, {m_real['ci_delta_rec5_pp'][1]:+.2f}]"
            )
            print(
                f"| {s} | Modelo B | H1 Real (Pronóstico 24h) | {m_real['pr_auc']:.6f} | {m_real['delta_pr_auc']:+.6f} {ci_pr_r} | {m_real['rec5_pct']:.2f}% | {m_real['delta_rec5_pp']:+.2f} {ci_r5_r} |"
            )
        elif s == "2023_24":
            print(
                f"| {s} | Modelo B | H1 Real (Pronóstico 24h) | NO EJECUTADO | (falta cobertura pre-2024-01-19) | NO EJECUTADO | (falta cobertura pre-2024-01-19) |"
            )

        # Baseline
        b_era = ev["baseline_fwi_x_prior"]["era5_land"]
        b_h0 = ev["baseline_fwi_x_prior"]["nwp_h0_source_skew"]
        b_h1 = ev["baseline_fwi_x_prior"]["nwp_h1_persistence_24h"]
        print(
            f"| {s} | Baseline | ERA5-Land (Reanálisis) | {b_era['pr_auc']:.6f} | [Ref] | {b_era['rec5_pct']:.2f}% | [Ref] |"
        )
        ci_b_pr0 = f"[{b_h0['ci_delta_pr_auc'][0]:+.6f}, {b_h0['ci_delta_pr_auc'][1]:+.6f}]"
        ci_b_r50 = f"[{b_h0['ci_delta_rec5_pp'][0]:+.2f}, {b_h0['ci_delta_rec5_pp'][1]:+.2f}]"
        print(
            f"| {s} | Baseline | NWP H0 (Nowcast ~0h) | {b_h0['pr_auc']:.6f} | {b_h0['delta_pr_auc']:+.6f} {ci_b_pr0} | {b_h0['rec5_pct']:.2f}% | {b_h0['delta_rec5_pp']:+.2f} {ci_b_r50} |"
        )

        ci_b_pr1 = f"[{b_h1['ci_delta_pr_auc'][0]:+.6f}, {b_h1['ci_delta_pr_auc'][1]:+.6f}]"
        ci_b_r51 = f"[{b_h1['ci_delta_rec5_pp'][0]:+.2f}, {b_h1['ci_delta_rec5_pp'][1]:+.2f}]"
        print(
            f"| {s} | Baseline | Persistencia 24h (D-1) | {b_h1['pr_auc']:.6f} | {b_h1['delta_pr_auc']:+.6f} {ci_b_pr1} | {b_h1['rec5_pct']:.2f}% | {b_h1['delta_rec5_pp']:+.2f} {ci_b_r51} |"
        )

        if "nwp_h1_real_forecast_24h" in ev["baseline_fwi_x_prior"]:
            b_real = ev["baseline_fwi_x_prior"]["nwp_h1_real_forecast_24h"]
            ci_b_prr = f"[{b_real['ci_delta_pr_auc'][0]:+.6f}, {b_real['ci_delta_pr_auc'][1]:+.6f}]"
            ci_b_r5r = (
                f"[{b_real['ci_delta_rec5_pp'][0]:+.2f}, {b_real['ci_delta_rec5_pp'][1]:+.2f}]"
            )
            print(
                f"| {s} | Baseline | H1 Real (Pronóstico 24h) | {b_real['pr_auc']:.6f} | {b_real['delta_pr_auc']:+.6f} {ci_b_prr} | {b_real['rec5_pct']:.2f}% | {b_real['delta_rec5_pp']:+.2f} {ci_b_r5r} |"
            )

    print("\n## 3. TABLA 2: TEST PAREADO DE SUPERIORIDAD OPERATIVA (CRITERIO K5.a)")
    print(
        "| Temporada | Horizonte NWP | Rec@5% Baseline | Rec@5% Modelo B | Delta Rec@5% pp [IC 95%] | Criterio K5(a) |"
    )
    print("| :--- | :--- | :---: | :---: | :---: | :---: |")

    for s in ["2023_24", "2024_25"]:
        if s not in evals:
            continue
        ev = evals[s]
        mvb = ev.get("model_vs_baseline_nwp", {})

        # H0
        b_r0 = ev["baseline_fwi_x_prior"]["nwp_h0_source_skew"]["rec5_pct"]
        m_r0 = ev["model_xgb_pruned_25"]["nwp_h0_source_skew"]["rec5_pct"]
        d_h0 = mvb["nwp_h0"]
        ci0 = f"[{d_h0['ci_delta_rec5_pp'][0]:+.2f}, {d_h0['ci_delta_rec5_pp'][1]:+.2f}]"
        verd0 = "CUMPLE (Excluye 0)" if d_h0["ci_delta_rec5_pp"][0] > 0 else "NO CUMPLE"
        print(
            f"| {s} | NWP H0 (Nowcast) | {b_r0:.2f}% | {m_r0:.2f}% | {d_h0['delta_rec5_pp']:+.2f} {ci0} | {verd0} |"
        )

        # Persistencia
        b_r1 = ev["baseline_fwi_x_prior"]["nwp_h1_persistence_24h"]["rec5_pct"]
        m_r1 = ev["model_xgb_pruned_25"]["nwp_h1_persistence_24h"]["rec5_pct"]
        d_h1 = mvb["nwp_h1_persistence_24h"]
        ci1 = f"[{d_h1['ci_delta_rec5_pp'][0]:+.2f}, {d_h1['ci_delta_rec5_pp'][1]:+.2f}]"
        verd1 = "CUMPLE (Excluye 0)" if d_h1["ci_delta_rec5_pp"][0] > 0 else "NO CUMPLE"
        print(
            f"| {s} | Persistencia 24h | {b_r1:.2f}% | {m_r1:.2f}% | {d_h1['delta_rec5_pp']:+.2f} {ci1} | {verd1} |"
        )

        # H1 Real
        if "nwp_h1_real_forecast_24h" in mvb:
            b_rr = ev["baseline_fwi_x_prior"]["nwp_h1_real_forecast_24h"]["rec5_pct"]
            m_rr = ev["model_xgb_pruned_25"]["nwp_h1_real_forecast_24h"]["rec5_pct"]
            d_hr = mvb["nwp_h1_real_forecast_24h"]
            cir = f"[{d_hr['ci_delta_rec5_pp'][0]:+.2f}, {d_hr['ci_delta_rec5_pp'][1]:+.2f}]"
            verdr = "CUMPLE (Excluye 0)" if d_hr["ci_delta_rec5_pp"][0] > 0 else "NO CUMPLE"
            print(
                f"| {s} | H1 Real (Pronóstico 24h) | {b_rr:.2f}% | {m_rr:.2f}% | {d_hr['delta_rec5_pp']:+.2f} {cir} | {verdr} |"
            )
        elif s == "2023_24":
            print(
                f"| {s} | H1 Real (Pronóstico 24h) | NO EJECUTADO | NO EJECUTADO | NO EJECUTADO | NO EJECUTADO |"
            )

    print("\n## 4. TABLA 3: EFECTIVIDAD DEL MAPEO MONÓTONO DE CUANTILES (K4 Y CRITERIO K5.b)")
    print(
        "| Dirección (Ajuste -> Test) | Horizonte | % Extremo Antes | Desvío Rel. Antes | % Extremo Después | Desvío Rel. Después | Criterio K5(b) |"
    )
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for item in qm:
        if item.get("status") == "NO EJECUTADO":
            print(
                f"| {item['direction']} | {item['horizon']} | NO EJECUTADO | NO EJECUTADO | NO EJECUTADO | NO EJECUTADO | NO EJECUTADO |"
            )
            continue
        m_mod = item["model_xgb_pruned_25"]
        p_bef = m_mod["tier_distribution_before"]["extremo"]
        s_bef = m_mod["relative_shifts_before_pct"]["extremo"]
        p_aft = m_mod["tier_distribution_after"]["extremo"]
        s_aft = m_mod["relative_shifts_after_pct"]["extremo"]
        verd_b = "CUMPLE (<=+-30%)" if m_mod["within_30pct_after"] else "NO CUMPLE"
        print(
            f"| {item['direction']} | {item['horizon']} | {p_bef:.2f}% | {s_bef:+.1f}% | {p_aft:.2f}% | {s_aft:+.1f}% | {verd_b} |"
        )

    print("\n## 5. TABLA 4: SÍNTESIS DE VEREDICTO DE CRITERIOS OFICIALES K5")
    print(
        "| Condición Evaluada | (a) Superioridad vs Base | (b) Reparto tras QM | (c) Caída PR-AUC 95% IC | Veredicto Global |"
    )
    print("| :--- | :---: | :---: | :---: | :---: |")
    for k, v in crit.items():
        v_a = v["criterion_a_model_beats_baseline_rec5"]["verdict"]
        v_b = v["criterion_b_class_distribution_30pct"]["after_qm"]
        v_c = v["criterion_c_pr_auc_drop_95ci"]["verdict"]
        global_v = (
            "ACEPTADO" if (v_a == "CUMPLE" and v_b == "CUMPLE" and v_c == "CUMPLE") else "RECHAZADO"
        )
        print(f"| {k} | {v_a} | {v_b} | {v_c} | **{global_v}** |")

    print("\n## 6. LIMITACIONES METODOLÓGICAS FORMALES (K6)")
    print(
        "- Ventana activa estricta: evaluado en 2 temporadas (2023_24 con N=315 y 2024_25 con N=411 igniciones CONAF)."
    )
    print(
        "- Alcance temporal restringido a noviembre-abril; se excluye mayo a octubre (inactividad invernal)."
    )
    print(
        "- H1 real disponible exclusivamente en 2024_25 debido a falta de archivo GFS previous_day1 previo a enero 2024."
    )


if __name__ == "__main__":
    main()
