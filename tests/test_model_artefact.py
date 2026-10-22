"""Test de verificación de artefactos del modelo podado final (F7).

PORQUÉ (Regla 5):
- F7 exige contractualmente que el modelo exportado cargue limpiamente en un proceso
  independiente y reproduzca exactamente sus predicciones con diferencia < 1e-6 sobre 100 casos.
- Valida la integridad criptográfica de todos los artefactos exportados contra `models/manifest.json`.
- Valida que el orden y tipos de las 25 variables coincidan exactamente con la especificación de diseño.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.models.xgboost_experiment import PRUNED_25_FEATURES


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_manifest_sha256_integrity() -> None:
    models_dir = Path("models")
    manifest_path = models_dir / "manifest.json"
    assert manifest_path.exists(), f"No se encuentra {manifest_path}"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    for fname, meta in manifest["artefacts"].items():
        fpath = models_dir / fname
        assert fpath.exists(), f"Archivo declarado en manifest no existe: {fpath}"
        actual_sha = compute_sha256(fpath)
        assert actual_sha == meta["sha256"], f"SHA-256 no coincide para {fname}: esperado {meta['sha256']}, obtenido {actual_sha}"
        assert fpath.stat().st_size == meta["size_bytes"], f"Tamaño no coincide para {fname}"


def test_feature_metadata_spec() -> None:
    feat_path = Path("models/feature_metadata.json")
    assert feat_path.exists()
    with open(feat_path, encoding="utf-8") as f:
        data = json.load(f)

    assert data["n_features"] == 25
    assert len(data["features"]) == 25
    feature_names = [f["name"] for f in data["features"]]
    assert feature_names == PRUNED_25_FEATURES
    assert "pop_density_1km" in data["dropped_features"]


def test_clean_process_inference_reproducibility() -> None:
    """Carga el modelo en un subproceso independiente y verifica diferencia < 1e-6 en 100 casos."""
    sample_path = Path("data/processed/training_sample_phase2b.parquet")
    assert sample_path.exists()

    df = pl.read_parquet(sample_path).slice(0, 100)
    X_sample = df.select(PRUNED_25_FEATURES).to_numpy().astype(np.float32)

    # Predicción en el proceso actual
    booster_main = xgb.Booster()
    booster_main.load_model("models/chilefuego_xgb_pruned_v1.json")
    dmat_main = xgb.DMatrix(X_sample, feature_names=PRUNED_25_FEATURES)
    preds_main = booster_main.predict(dmat_main)

    # Guardar array temporal para contrastar en subproceso
    tmp_npy = Path("models/test_sample_100.npy")
    np.save(tmp_npy, X_sample)

    script_clean = """
import numpy as np
import xgboost as xgb
from chilefuego.models.xgboost_experiment import PRUNED_25_FEATURES

X = np.load('models/test_sample_100.npy')
booster = xgb.Booster()
booster.load_model('models/chilefuego_xgb_pruned_v1.json')
dmat = xgb.DMatrix(X, feature_names=PRUNED_25_FEATURES)
preds = booster.predict(dmat)
np.save('models/clean_preds_100.npy', preds)
"""
    result = subprocess.run(
        [sys.executable, "-c", script_clean],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Error en subproceso limpio: {result.stderr}"

    preds_clean = np.load("models/clean_preds_100.npy")
    max_diff = float(np.max(np.abs(preds_main - preds_clean)))

    # Limpiar archivos temporales
    tmp_npy.unlink(missing_ok=True)
    Path("models/clean_preds_100.npy").unlink(missing_ok=True)

    assert max_diff < 1e-6, f"Diferencia máxima entre procesos ({max_diff:.2e}) superó el umbral 1e-6"
