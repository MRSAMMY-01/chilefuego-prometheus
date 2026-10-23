"""Generador automatizado de tablas para el reporte técnico Fase 3A (J1-J5).

PORQUÉ (Regla 5):
- Todas las tablas del reporte deben generarse por script a partir de archivos de resultados
  persistidos en results/*.json, eliminando la transcripción manual.
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_report_tables() -> None:
    # 1. Tabla J3: Spin-up e Insensibilidad al Arranque FWI (90 días)
    spinup_path = Path("results/j3_fwi_spinup_insensitivity.json")
    if spinup_path.exists():
        with open(spinup_path, encoding="utf-8") as f:
            spin_data = json.load(f)
        print(
            "### Tabla J3.1: Prueba de Insensibilidad al Arranque FWI (Spin-up 90 días, Open-Meteo)"
        )
        print(
            "| Zona Ñuble | Nodo ERA5-Land | Días | DC Final (Low=15) | DC Final (High=500) | Diff DC | Diff DMC | Convergencia |"
        )
        print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for zone, r in spin_data.get("nodes", {}).items():
            print(
                f"| {zone.capitalize()} | `{r['node_id']}` | {r['n_days']} | {r['dc_final_low']:.2f} | {r['dc_final_high']:.2f} | {r['diff_dc_final']:.4f} | {r['diff_dmc_final']:.4f} | {'SÍ (Exacta)' if r['converged'] else 'NO'} |"
            )
        print()

    # 2. Tabla J4(b): Calibradores y Umbrales (Sin Ponderar vs Ponderado)
    unw_cal_path = Path("models/platt_calibrator.json")
    w_cal_path = Path("models/platt_calibrator_sizew.json")
    unw_th_path = Path("models/risk_thresholds.json")
    w_th_path = Path("models/risk_thresholds_sizew.json")

    if unw_cal_path.exists() and w_cal_path.exists():
        with open(unw_cal_path, encoding="utf-8") as f:
            unw_c = json.load(f)
        with open(w_cal_path, encoding="utf-8") as f:
            w_c = json.load(f)
        with open(unw_th_path, encoding="utf-8") as f:
            unw_t = json.load(f)
        with open(w_th_path, encoding="utf-8") as f:
            w_t = json.load(f)

        print(
            "### Tabla J4.1: Comparativa de Calibrador Platt y Umbrales OOF (Sin Ponderar vs Ponderado por Tamaño)"
        )
        print(
            "| Componente / Parámetro | Modelo Sin Ponderar | Variante Ponderada (>=10 ha) | Delta (Ponderado - Base) |"
        )
        print("| :--- | :---: | :---: | :---: |")
        print(
            f"| Platt: Pendiente (Slope) | {unw_c['slope']:.4f} | {w_c['slope']:.4f} | {w_c['slope'] - unw_c['slope']:+.4f} |"
        )
        print(
            f"| Platt: Intercepto | {unw_c['intercept']:.4f} | {w_c['intercept']:.4f} | {w_c['intercept'] - unw_c['intercept']:+.4f} |"
        )
        print(
            f"| Umbral p80 (por 100k) | {unw_t['rates_per_100k']['p80']:.2f} | {w_t['rates_per_100k']['p80']:.2f} | {w_t['rates_per_100k']['p80'] - unw_t['rates_per_100k']['p80']:+.2f} |"
        )
        print(
            f"| Umbral p95 (por 100k) | {unw_t['rates_per_100k']['p95']:.2f} | {w_t['rates_per_100k']['p95']:.2f} | {w_t['rates_per_100k']['p95'] - unw_t['rates_per_100k']['p95']:+.2f} |"
        )
        print(
            f"| Umbral p99 (por 100k) | {unw_t['rates_per_100k']['p99']:.2f} | {w_t['rates_per_100k']['p99']:.2f} | {w_t['rates_per_100k']['p99'] - unw_t['rates_per_100k']['p99']:+.2f} |"
        )
        print()

    # 3. Tabla J4(c): Validación Cruzada LOSO y Validación Hacia Adelante (PR-AUC Exacta e IC Bootstrap)
    g1_path = Path("results/g1_bootstrap_loso.json")
    if g1_path.exists():
        with open(g1_path, encoding="utf-8") as f:
            g1_data = json.load(f)
        print(
            "### Tabla J4.2: PR-AUC Exacta, Deltas e Intervalos Bootstrap (LOSO y Validación Hacia Adelante)"
        )
        print(
            "| Temporada / Partición | PR-AUC Base | PR-AUC Podado | Delta PR-AUC | IC 95% Bootstrap | Significativo |"
        )
        print("| :--- | :---: | :---: | :---: | :---: | :---: |")
        for s in g1_data.get("season_results", []):
            ci = s["ci_delta_pr_base"]
            print(
                f"| LOSO {s['season']} | {s['pr_auc_base']:.6f} | {s['pr_auc_pruned']:.6f} | {s['delta_pr_base']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] | {'SÍ' if s['sig_pr_base'] else 'No'} |"
            )
        agg = g1_data.get("aggregated", {})
        ci_agg = agg.get("ci_delta_pr_base", [0, 0])
        print(
            f"| **LOSO Macro-Agregado** | **{agg.get('macro_pr_auc_base', 0):.6f}** | **{agg.get('macro_pr_auc_pruned', 0):.6f}** | **{agg.get('delta_pr_base', 0):+.6f}** | **[{ci_agg[0]:+.6f}, {ci_agg[1]:+.6f}]** | **{'SÍ' if agg.get('sig_pr_base') else 'No'}** |"
        )

        fw_deltas = []
        fw_lows = []
        fw_highs = []
        for f_item in g1_data.get("forward_results", []):
            ci_f = f_item["ci_delta_pr"]
            fw_deltas.append(f_item["delta_pr"])
            fw_lows.append(ci_f[0])
            fw_highs.append(ci_f[1])
            print(
                f"| Forward Walk {f_item['season']} | {f_item['pr_auc_base']:.6f} | {f_item['pr_auc_pruned']:.6f} | {f_item['delta_pr']:+.6f} | [{ci_f[0]:+.6f}, {ci_f[1]:+.6f}] | {'SÍ' if f_item['sig_pr'] else 'No'} |"
            )

        mean_fw_d = sum(fw_deltas) / len(fw_deltas)
        mean_fw_ci = [sum(fw_lows) / len(fw_lows), sum(fw_highs) / len(fw_highs)]
        print(
            f"| **Forward Walk Agregado** | — | — | **{mean_fw_d:+.6f}** | **[{mean_fw_ci[0]:+.6f}, {mean_fw_ci[1]:+.6f}]** | **{'SÍ' if mean_fw_ci[0] > 0 else 'No'}** |"
        )
        print()


if __name__ == "__main__":
    generate_report_tables()
