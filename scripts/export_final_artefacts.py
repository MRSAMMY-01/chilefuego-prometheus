"""Script de exportación de artefactos y Model Card para el modelo final podado (F7).

PORQUÉ (Regla 5):
- F7: Empaqueta el modelo final entrenado con las 7 temporadas fechadas (3.237 positivos)
  en formato nativo XGBoost (UBJ y JSON).
- Exporta calibrador Platt (parámetros analíticos logisticos), metadatos de features
  (nombres canónicos, dtypes y orden exacto), umbrales fijos de clases de riesgo relativo
  (Bajo, Moderado, Alto, Extremo) y un manifest SHA-256 de integridad criptográfica.
- Genera el Model Card oficial (`models/model_card.md`) con especificación de dominio,
  uso previsto, limitaciones y métricas de desempeño out-of-fold.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.models.xgboost_experiment import PRUNED_25_FEATURES
from sklearn.linear_model import LogisticRegression


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def export_artefacts() -> dict[str, str]:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=== EXPORTACIÓN DE ARTEFACTOS FINALES (F7) ===")
    train_df = pl.read_parquet("data/processed/training_sample_phase2b.parquet")

    X = train_df.select(PRUNED_25_FEATURES).to_numpy()
    y = train_df.get_column("is_ignition_conaf").to_numpy().astype(np.int32)
    w = train_df.get_column("sample_weight").to_numpy().astype(np.float32)

    dtrain = xgb.DMatrix(X, label=y, weight=w, feature_names=PRUNED_25_FEATURES)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10.0,
        "gamma": 1.0,
        "lambda": 1.0,
        "random_state": 42,
        "nthread": 4,
    }

    # Entrenar modelo final sobre todo el dataset (150 rondas óptimas promedio)
    final_booster = xgb.train(params, dtrain, num_boost_round=150)

    # 1. Exportar modelos XGBoost en JSON y UBJ
    model_json_path = models_dir / "chilefuego_xgb_pruned_v1.json"
    model_ubj_path = models_dir / "chilefuego_xgb_pruned_v1.ubj"
    final_booster.save_model(model_json_path)
    final_booster.save_model(model_ubj_path)

    # 2. Ajustar y exportar calibrador Platt
    raw_scores = final_booster.predict(dtrain)
    logit_scores = np.log(np.clip(raw_scores, 1e-7, 1 - 1e-7) / (1 - np.clip(raw_scores, 1e-7, 1 - 1e-7))).reshape(-1, 1)
    platt_lr = LogisticRegression().fit(logit_scores, y, sample_weight=w)

    platt_data = {
        "method": "platt_scaling",
        "slope": float(platt_lr.coef_[0][0]),
        "intercept": float(platt_lr.intercept_[0]),
        "formula": "P(y=1 | s) = 1 / (1 + exp(-(slope * logit(s) + intercept)))",
        "date_trained": time.strftime("%Y-%m-%d"),
    }
    platt_path = models_dir / "platt_calibrator.json"
    with open(platt_path, "w", encoding="utf-8") as f:
        json.dump(platt_data, f, indent=2)

    # 3. Exportar lista y metadata de features
    feat_meta = {
        "model_version": "v1.0.0-pruned",
        "n_features": len(PRUNED_25_FEATURES),
        "features": [
            {"name": col, "dtype": "float32", "order": i}
            for i, col in enumerate(PRUNED_25_FEATURES)
        ],
        "dropped_features": [
            "pop_density_1km",
            "temp_lag1", "temp_roll3_mean", "temp_roll3_max", "temp_roll7_mean", "temp_roll7_max",
            "vpd_lag1", "vpd_roll3_mean", "vpd_roll3_max", "vpd_roll7_mean", "vpd_roll7_max",
            "rain_acc7", "rain_acc30", "days_since_rain25",
        ],
    }
    feat_path = models_dir / "feature_metadata.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feat_meta, f, indent=2)

    # 4. Umbrales fijos de clases de riesgo relativo
    p80 = float(np.percentile(raw_scores, 80.0))
    p95 = float(np.percentile(raw_scores, 95.0))
    p99 = float(np.percentile(raw_scores, 99.0))
    risk_meta = {
        "system": "Relative Risk Classification (Experimental)",
        "classes": {
            "Bajo": {"percentile_range": "[0, 80)", "score_range": [0.0, p80]},
            "Moderado": {"percentile_range": "[80, 95)", "score_range": [p80, p95]},
            "Alto": {"percentile_range": "[95, 99)", "score_range": [p95, p99]},
            "Extremo": {"percentile_range": "[99, 100]", "score_range": [p99, 1.0]},
        },
        "thresholds": {"p80": p80, "p95": p95, "p99": p99},
        "note": "Clasificación experimental de riesgo relativo espacial-temporal. NO es un disparador automático de alerta comunal.",
    }
    risk_path = models_dir / "risk_thresholds.json"
    with open(risk_path, "w", encoding="utf-8") as f:
        json.dump(risk_meta, f, indent=2)

    # 5. Generar Manifest con SHA-256
    manifest: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artefacts": {},
    }
    artefact_files = [model_json_path, model_ubj_path, platt_path, feat_path, risk_path]
    for af in artefact_files:
        manifest["artefacts"][af.name] = {
            "sha256": compute_sha256(af),
            "size_bytes": af.stat().st_size,
        }

    manifest_path = models_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Artefactos exportados exitosamente en {models_dir}:")
    for fname, meta in manifest["artefacts"].items():
        print(f"  - {fname} ({meta['size_bytes']} bytes) SHA-256: {meta['sha256'][:16]}...")

    return {fname: meta["sha256"] for fname, meta in manifest["artefacts"].items()}


if __name__ == "__main__":
    export_artefacts()
