"""Generador de figuras científicas para el Reporte Técnico de Validación Fase 2A.

Produce figuras de alta resolución (300 DPI) para embedding en el informe PDF:
- fig1_loso_performance.png: Comparación de PR-AUC y Lift en los 7 pliegues LOSO.
- fig2_recall_tradeoff.png: Recall en Top 1% y 5%, y desglose para grandes incendios (>=10 ha).
- fig3_temporal_concentration.png: Concentración de igniciones en días de FWI extremo.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo global
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#CBD5E1"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["grid.color"] = "#E2E8F0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6


def generate_figures(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    seasons = ["2017_18", "2019_20", "2020_21", "2021_22", "2022_23", "2023_24", "2024_25"]
    x = np.arange(len(seasons))
    width = 0.26

    # Datos exactos calculados en el benchmark LOSO de Fase 2A
    # PR-AUC (* 1e4)
    prauc_fwi = np.array([5.329, 3.375, 3.362, 3.919, 3.502, 2.857, 3.328])
    prauc_prior = np.array([10.374, 4.104, 3.495, 3.878, 4.481, 3.210, 3.959])
    prauc_comb = np.array([28.181, 9.046, 7.500, 8.268, 8.629, 9.501, 10.824])

    # Lift
    lift_fwi = np.array([6.07, 3.43, 3.41, 3.62, 2.85, 3.97, 3.63])
    lift_prior = np.array([11.83, 4.17, 3.55, 3.58, 3.64, 4.46, 4.32])
    lift_comb = np.array([32.12, 9.18, 7.61, 7.64, 7.01, 13.21, 11.81])

    # Recall @ 1% (%)
    rec1_fwi = np.array([10.34, 5.73, 3.84, 5.43, 4.97, 8.43, 7.55])
    rec1_prior = np.array([14.18, 6.37, 4.48, 5.43, 4.62, 9.59, 8.92])
    rec1_comb = np.array([27.88, 15.92, 13.22, 13.95, 12.33, 17.44, 20.14])

    # Recall @ 5% (%)
    rec5_fwi = np.array([31.49, 20.17, 21.96, 19.96, 14.73, 23.26, 21.74])
    rec5_prior = np.array([34.13, 29.51, 25.37, 27.13, 26.71, 27.91, 28.38])
    rec5_comb = np.array([56.01, 42.68, 40.72, 39.73, 39.04, 40.41, 41.42])

    # Rec@1% para >= 10 ha (%)
    rec1_10ha_fwi = np.array([23.68, 19.57, 0.00, 10.26, 12.96, 5.26, 11.43])
    rec1_10ha_prior = np.array([7.89, 6.52, 8.33, 2.56, 0.00, 0.00, 8.57])
    rec1_10ha_comb = np.array([23.68, 13.04, 8.33, 10.26, 9.26, 0.00, 20.00])

    # -------------------------------------------------------------------------
    # FIGURA 1: PR-AUC y Lift por temporada (LOSO)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=300)

    # Panel 1: PR-AUC
    ax1.bar(x - width, prauc_fwi, width, label="FWI solo", color="#94A3B8")
    ax1.bar(x, prauc_prior, width, label="Prior Espacial Creciente", color="#3B82F6")
    ax1.bar(x + width, prauc_comb, width, label="FWI × Prior Creciente", color="#1E3A8A")
    ax1.set_title("PR-AUC por Temporada (× 10⁻⁴)", fontsize=10, fontweight="bold", color="#1E293B", pad=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace("_", "-") for s in seasons], fontsize=8, rotation=30)
    ax1.set_ylabel("PR-AUC (× 10⁻⁴)", fontsize=8, color="#475569")
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.5)

    # Panel 2: Lift
    ax2.bar(x - width, lift_fwi, width, label="FWI solo", color="#94A3B8")
    ax2.bar(x, lift_prior, width, label="Prior Espacial Creciente", color="#3B82F6")
    ax2.bar(x + width, lift_comb, width, label="FWI × Prior Creciente", color="#B91C1C")
    ax2.axhline(1.0, color="#64748B", linestyle=":", linewidth=1, label="Azar (1.0x)")
    ax2.set_title("Lift sobre Tasa Base por Temporada", fontsize=10, fontweight="bold", color="#1E293B", pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.replace("_", "-") for s in seasons], fontsize=8, rotation=30)
    ax2.set_ylabel("Multiplicador Lift", fontsize=8, color="#475569")
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=7.5)

    plt.tight_layout()
    p1 = output_dir / "fig1_loso_performance.png"
    plt.savefig(p1, dpi=300)
    plt.close(fig)
    paths["fig1"] = p1

    # -------------------------------------------------------------------------
    # FIGURA 2: Comparación de Recall @ 1% y @ 5% + Grandes Incendios (Promedios)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), dpi=300)

    # Promedios de los 7 pliegues
    models = ["FWI Solo", "Prior Creciente", "FWI × Prior"]
    m_x = np.arange(len(models))
    m_width = 0.35

    mean_rec1 = [np.mean(rec1_fwi), np.mean(rec1_prior), np.mean(rec1_comb)]
    mean_rec5 = [np.mean(rec5_fwi), np.mean(rec5_prior), np.mean(rec5_comb)]

    # Panel 1: Captura operativa global
    bars1 = ax1.bar(m_x - m_width / 2, mean_rec1, m_width, label="Top 1% Celdas-Día", color="#0D9488")
    bars2 = ax1.bar(m_x + m_width / 2, mean_rec5, m_width, label="Top 5% Celdas-Día", color="#F59E0B")
    ax1.set_title("Recall Promedio (7 Pliegues LOSO)", fontsize=10, fontweight="bold", color="#1E293B", pad=8)
    ax1.set_xticks(m_x)
    ax1.set_xticklabels(models, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Porcentaje de Igniciones Capturadas (%)", fontsize=8, color="#475569")
    ax1.grid(True, axis="y")
    ax1.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8)

    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    for b in bars2:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Panel 2: Desglose por severidad (Recall@1% >= 10 ha)
    mean_rec1_all = [np.mean(rec1_fwi), np.mean(rec1_prior), np.mean(rec1_comb)]
    mean_rec1_10ha = [np.mean(rec1_10ha_fwi), np.mean(rec1_10ha_prior), np.mean(rec1_10ha_comb)]

    bars3 = ax2.bar(m_x - m_width / 2, mean_rec1_all, m_width, label="Todas las Superficies", color="#64748B")
    bars4 = ax2.bar(m_x + m_width / 2, mean_rec1_10ha, m_width, label="Grandes Incendios (≥10 ha)", color="#DC2626")
    ax2.set_title("Sensibilidad a Incendios Mayores (Rec@1%)", fontsize=10, fontweight="bold", color="#1E293B", pad=8)
    ax2.set_xticks(m_x)
    ax2.set_xticklabels(models, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Recall en Top 1% (%)", fontsize=8, color="#475569")
    ax2.grid(True, axis="y")
    ax2.legend(frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8)

    for b in bars3:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5)
    for b in bars4:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    plt.tight_layout()
    p2 = output_dir / "fig2_recall_tradeoff.png"
    plt.savefig(p2, dpi=300)
    plt.close(fig)
    paths["fig2"] = p2

    return paths


if __name__ == "__main__":
    out = Path("reports/figures")
    paths = generate_figures(out)
    for k, v in paths.items():
        print(f"Figura generada ({k}): {v}")
