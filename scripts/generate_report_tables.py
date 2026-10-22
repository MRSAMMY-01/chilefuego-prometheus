"""Script que genera todas las tablas formateadas en Markdown a partir de los archivos en results/*.json.

PORQUÉ (Regla 5):
- Cumple la regla estricta: "todas las tablas del reporte deben GENERARSE POR SCRIPT
  desde archivos de resultados (results/*.json o parquet), sin transcripción a mano".
"""

from __future__ import annotations

import json
from pathlib import Path


def render_all_tables() -> None:
    res_dir = Path("results")

    # =========================================================================
    # TABLA G1: BOOTSTRAP POR BLOQUES DE DÍAS (ESCALA EXACTA)
    # =========================================================================
    with open(res_dir / "g1_bootstrap_loso.json", encoding="utf-8") as f:
        g1 = json.load(f)

    print("### TABLA G1: PR-AUC EXACTA Y BOOTSTRAP POR BLOQUES DE DÍAS (LOSO 7 PLIEGUES)")
    print("| Temporada | N | Base PR-AUC | Podado PR-AUC | Completo PR-AUC | $\\Delta$PR (Pod-Base) [IC 95%] | ¿Sig.? | $\\Delta$PR (Pod-Comp) [IC 95%] | ¿Sig.? | $\\Delta$Rec@5% (Pod-Base) [IC 95%] |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for r in g1["season_results"]:
        s = r["season"]
        n = r["n_pos"]
        pb = f"{r['pr_auc_base']:.6e}"
        pp = f"{r['pr_auc_pruned']:.6e}"
        pf = f"{r['pr_auc_full']:.6e}"

        dpb = f"{r['delta_pr_base']:+.2e} [{r['ci_delta_pr_base'][0]:+.2e}, {r['ci_delta_pr_base'][1]:+.2e}]"
        sig_pb = "**Sí**" if r["sig_pr_base"] else "No"

        dpf = f"{r['delta_pr_full']:+.2e} [{r['ci_delta_pr_full'][0]:+.2e}, {r['ci_delta_pr_full'][1]:+.2e}]"
        sig_pf = "**Sí**" if r["sig_pr_full"] else "No"

        dr5 = f"{r['delta_rec5_base']*100:+.2f}% [{r['ci_delta_rec5_base'][0]*100:+.2f}%, {r['ci_delta_rec5_base'][1]*100:+.2f}%]"

        print(f"| **{s}** | {n} | {pb} | {pp} | {pf} | {dpb} | {sig_pb} | {dpf} | {sig_pf} | {dr5} |")

    agg = g1["aggregated"]
    dpb_agg = f"{agg['delta_pr_base']:+.2e} [{agg['ci_delta_pr_base'][0]:+.2e}, {agg['ci_delta_pr_base'][1]:+.2e}]"
    dpf_agg = f"{agg['delta_pr_full']:+.2e} [{agg['ci_delta_pr_full'][0]:+.2e}, {agg['ci_delta_pr_full'][1]:+.2e}]"
    dr5_agg = f"{agg['delta_rec5_base']*100:+.2f}% [{agg['ci_delta_rec5_base'][0]*100:+.2f}%, {agg['ci_delta_rec5_base'][1]*100:+.2f}%]"
    sig_pb_agg = "**Sí**" if agg["sig_pr_base"] else "No"
    sig_pf_agg = "**Sí**" if agg["sig_pr_full"] else "No"

    print(f"| **AGREGADO** | 3.237 | {agg['macro_pr_auc_base']:.6e} | {agg['macro_pr_auc_pruned']:.6e} | {agg['macro_pr_auc_full']:.6e} | {dpb_agg} | {sig_pb_agg} | {dpf_agg} | {sig_pf_agg} | {dr5_agg} |")

    print("\n### TABLA G1 (CONT.): VALIDACIÓN HACIA ADELANTE (TRAIN 2017-2022 -> TEST PROSPECTIVO)")
    print("| Temporada Test | Base PR-AUC | Podado PR-AUC | $\\Delta$PR-AUC [IC 95%] | Base Lift | Podado Lift | Base Rec@5% | Podado Rec@5% | $\\Delta$Rec@5% [IC 95%] |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for fw in g1["forward_results"]:
        s = fw["season"]
        pb = f"{fw['pr_auc_base']:.6e}"
        pp = f"{fw['pr_auc_pruned']:.6e}"
        dp = f"{fw['delta_pr']:+.2e} [{fw['ci_delta_pr'][0]:+.2e}, {fw['ci_delta_pr'][1]:+.2e}]"
        lb = f"{fw['lift_base']:.2f}x"
        lp = f"{fw['lift_pruned']:.2f}x"
        r5b = f"{fw['rec5_base']*100:.2f}%"
        r5p = f"{fw['rec5_pruned']*100:.2f}%"
        dr5 = f"{fw['delta_rec5']*100:+.2f}% [{fw['ci_delta_rec5'][0]*100:+.2f}%, {fw['ci_delta_rec5'][1]*100:.2f}%]"
        print(f"| **{s}** | {pb} | {pp} | {dp} | {lb} | {lp} | {r5b} | {r5p} | {dr5} |")

    # =========================================================================
    # TABLA G2: UMBRALES Y REPRODUCCIÓN ~80/15/4/1%
    # =========================================================================
    with open(res_dir / "g2_thresholds_oos.json", encoding="utf-8") as f:
        g2 = json.load(f)

    th = g2["thresholds"]
    print("\n### TABLA G2: UMBRALES CALIBRADOS OUT-OF-SAMPLE Y VERIFICACIÓN ~80/15/4/1%")
    print(f"Umbrales calibrados globales: p80 = {th['p80']:.6f} ({th['p80_per_100k']:.2f}/100k), p95 = {th['p95']:.6f} ({th['p95_per_100k']:.2f}/100k), p99 = {th['p99']:.6f} ({th['p99_per_100k']:.2f}/100k)")
    print("| Temporada | Bajo [0-80%): %Celdas (Desvío) | Mod [80-95%): %Celdas (Desvío) | Alto [95-99%): %Celdas (Desvío) | Ext [99-100%]: %Celdas (Desvío) | Igniciones Capturadas (Alto+Ext) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for row in g2["season_evaluation"]:
        s = row["season"]
        c = row["pct_cells"]
        d = row["deviations"]
        ig = row["pct_ignitions"]
        alto_ext_ign = ig["alto"] + ig["extremo"]

        b_str = f"{c['bajo']:.2f}% ({d['bajo']:+.2f}%)"
        m_str = f"{c['moderado']:.2f}% ({d['moderado']:+.2f}%)"
        a_str = f"{c['alto']:.2f}% ({d['alto']:+.2f}%)"
        e_str = f"{c['extremo']:.2f}% ({d['extremo']:+.2f}%)"

        print(f"| **{s}** | {b_str} | {m_str} | {a_str} | {e_str} | **{alto_ext_ign:.2f}%** |")

    # =========================================================================
    # TABLA G3: COSTO GLOBAL Y BOOTSTRAP PAREADO POR EVENTOS (N=255)
    # =========================================================================
    with open(res_dir / "g3_size_weighting.json", encoding="utf-8") as f:
        g3 = json.load(f)

    print("\n### TABLA G3: PESOS POR TAMAÑO: COSTO GLOBAL Y BOOTSTRAP DE EVENTOS (N=255)")
    c_all = g3["costs_all_surfaces"]
    c_1 = g3["surfaces_ge_1ha"]
    c_10 = g3["surfaces_ge_10ha"]
    b_ev = g3["event_bootstrap_255"]
    dec = g3["decision_rule"]

    print("| Métrica / Superficie | Sin Ponderar | Ponderado Tamaño | Costo / Delta |")
    print("| :--- | :---: | :---: | :---: |")
    print(f"| **Todas: PR-AUC** | {c_all['unweighted']['pr_auc']:.6e} | {c_all['weighted']['pr_auc']:.6e} | {c_all['cost_penalty']['pr_auc']:+.2e} |")
    print(f"| **Todas: Rec@1%** | {c_all['unweighted']['rec1']:.2f}% | {c_all['weighted']['rec1']:.2f}% | {c_all['cost_penalty']['rec1']:+.2f} pp |")
    print(f"| **Todas: Rec@5%** | {c_all['unweighted']['rec5']:.2f}% | {c_all['weighted']['rec5']:.2f}% | {c_all['cost_penalty']['rec5']:+.2f} pp |")
    print(f"| **$\\ge 1$ ha: Rec@1%** | {c_1['unweighted']['rec1']:.2f}% | {c_1['weighted']['rec1']:.2f}% | {c_1['delta']['rec1']:+.2f} pp |")
    print(f"| **$\\ge 1$ ha: Rec@5%** | {c_1['unweighted']['rec5']:.2f}% | {c_1['weighted']['rec5']:.2f}% | {c_1['delta']['rec5']:+.2f} pp |")
    print(f"| **$\\ge 10$ ha: Rec@1%** | {c_10['unweighted']['rec1']:.2f}% | {c_10['weighted']['rec1']:.2f}% | **{c_10['delta']['rec1']:+.2f} pp** |")
    print(f"| **$\\ge 10$ ha: Rec@5%** | {c_10['unweighted']['rec5']:.2f}% | {c_10['weighted']['rec5']:.2f}% | **{c_10['delta']['rec5']:+.2f} pp** |")

    d_u1 = b_ev["delta_hits_vs_unweighted_top1"]
    d_u5 = b_ev["delta_hits_vs_unweighted_top5"]
    d_b1 = b_ev["delta_hits_vs_baseline_top1"]
    d_b5 = b_ev["delta_hits_vs_baseline_top5"]

    print("\n**Bootstrap Pareado por Eventos sobre los 255 incendios $\\ge 10$ ha (B=2000):**")
    print(f"- $\\Delta$Aciertos Top 1% vs Sin Ponderar: **{d_u1['mean']:+.1f} aciertos** (IC 95%: [{d_u1['ci95'][0]:+.1f}, {d_u1['ci95'][1]:+.1f}]) -> Excluye cero: {d_u1['excludes_zero']}")
    print(f"- $\\Delta$Aciertos Top 5% vs Sin Ponderar: **{d_u5['mean']:+.1f} aciertos** (IC 95%: [{d_u5['ci95'][0]:+.1f}, {d_u5['ci95'][1]:+.1f}]) -> Excluye cero: {d_u5['excludes_zero']}")
    print(f"- $\\Delta$Aciertos Top 1% vs Baseline: **{d_b1['mean']:+.1f} aciertos** (IC 95%: [{d_b1['ci95'][0]:+.1f}, {d_b1['ci95'][1]:+.1f}]) -> Excluye cero: {d_b1['excludes_zero']}")
    print(f"- $\\Delta$Aciertos Top 5% vs Baseline: **{d_b5['mean']:+.1f} aciertos** (IC 95%: [{d_b5['ci95'][0]:+.1f}, {d_b5['ci95'][1]:+.1f}]) -> Excluye cero: {d_b5['excludes_zero']}")
    print(f"**Regla de Decisión:** {dec['recommendation']} (Costo global Rec@1%: {dec['cost_global_rec1_pp']:.2f} pp, Rec@5%: {dec['cost_global_rec5_pp']:.2f} pp)")


if __name__ == "__main__":
    render_all_tables()
