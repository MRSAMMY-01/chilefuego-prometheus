import json

with open("data/processed/phase2c_results.json", encoding="utf-8") as f:
    d = json.load(f)

print("=== F2: BOOTSTRAP PAREADO POR BLOQUES DE DIAS ===")
for s in d["bootstrap_by_season_vs_base"]:
    b_base = d["bootstrap_by_season_vs_base"][s]
    b_full = d["bootstrap_by_season_vs_full"][s]
    print(f"{s}:")
    print(f"  vs Base: delta_PR = {b_base['mean_delta_pr']:+.6e} CI95=[{b_base['ci_delta_pr'][0]:.6e}, {b_base['ci_delta_pr'][1]:.6e}] sig={b_base['sig_pr']} | delta_Rec5 = {b_base['mean_delta_rec5']*100:+.2f}% CI95=[{b_base['ci_delta_rec5'][0]*100:.2f}%, {b_base['ci_delta_rec5'][1]*100:.2f}%] sig={b_base['sig_rec5']}")
    print(f"  vs Full: delta_PR = {b_full['mean_delta_pr']:+.6e} CI95=[{b_full['ci_delta_pr'][0]:.6e}, {b_full['ci_delta_pr'][1]:.6e}] sig={b_full['sig_pr']} | delta_Rec5 = {b_full['mean_delta_rec5']*100:+.2f}% CI95=[{b_full['ci_delta_rec5'][0]*100:.2f}%, {b_full['ci_delta_rec5'][1]*100:.2f}%] sig={b_full['sig_rec5']}")

print("\n=== F2: FORWARD WALK (PROSPECTIVA) ===")
for fw in d["forward_results"]:
    s = fw["season"]
    b = fw["baseline"]
    p = fw["pruned"]
    print(f"{s}: Base PR={b['pr_auc']:.6e} Lift={b['lift']:.2f}x Rec5={b['rec5_all']*100:.2f}% || Pruned PR={p['pr_auc']:.6e} Lift={p['lift']:.2f}x Rec5={p['rec5_all']*100:.2f}%")

print("\n=== F3: RESULTADOS INCENDIOS GRANDES (>=10 ha, N=255) ===")
f3 = d["f3_results"]
print(f"Base:   Top1 = {f3['baseline_top1_hits']} ({f3['baseline_top1_pct']:.2f}%) | Top5 = {f3['baseline_top5_hits']} ({f3['baseline_top5_pct']:.2f}%)")
print(f"Pruned: Top1 = {f3['pruned_top1_hits_mean']:.1f} ({f3['pruned_top1_pct_mean']:.2f}%) | Top5 = {f3['pruned_top5_hits_mean']:.1f} ({f3['pruned_top5_pct_mean']:.2f}%)")
print(f"Size-W: Top1 = {f3['sizew_top1_hits_mean']:.1f} ({f3['sizew_top1_pct_mean']:.2f}%) | Top5 = {f3['sizew_top5_hits_mean']:.1f} ({f3['sizew_top5_pct_mean']:.2f}%)")

print("\n=== F4: CALIBRACION (PLATT) ===")
print(f"Slope: {d['calib_slope']:.4f} | Intercept: {d['calib_intercept']:.4f}")
print("Temporada | Clase | Tasa Pred (por 100k) | Tasa Obs (por 100k)")
for row in d["f4_calibration_table"]:
    s = row["season"]
    for c in ["Bajo [0-80)", "Moderado [80-95)", "Alto [95-99)", "Extremo [99-100]"]:
        print(f"{s} | {c:18} | Pred: {row[f'{c}_pred_100k']:8.2f} | Obs: {row[f'{c}_obs_100k']:8.2f}")

print("\n=== F5: CLASES OUT-OF-SEASON ===")
print("Temporada | Clase | % Celdas-Día | % Igniciones")
for row in d["f5_risk_classes_table"]:
    s = row["season"]
    for c in ["Bajo [0-80)", "Moderado [80-95)", "Alto [95-99)", "Extremo [99-100]"]:
        print(f"{s} | {c:18} | Celdas: {row[f'{c}_pct_cells']:5.2f}% | Ign: {row[f'{c}_pct_ign']:5.2f}%")
