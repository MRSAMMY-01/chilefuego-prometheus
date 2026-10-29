"""Pruebas de Aislamiento Estricto para prospective_log.parquet (U1).

PORQUÉ:
- Garantiza que ninguna simulación de falla (como T4), backtest o prueba unitaria pueda
  escribir registros en data/prospective_log.parquet de producción.
- Exige que toda simulación use una ruta temporal y tenga is_simulation=True.
- Valida la clave única (run_id, cell_id, target_date, horizon) y la ausencia de 2026-11-01 en el log real.
- Este test FALLA si cualquier simulación o test toca el log real.
"""

from __future__ import annotations

import hashlib
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from scripts.run_daily import run_daily_shadow_pipeline


def _get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_simulation_fails_if_attempting_to_touch_real_log() -> None:
    """Verifica que el pipeline RECHAZA explícitamente escribir en el log real durante una simulación."""
    real_path = Path("data/prospective_log.parquet")
    test_date = date(2026, 11, 1)

    with pytest.raises(RuntimeError, match="Violación de aislamiento U1"):
        run_daily_shadow_pipeline(
            target_date=test_date,
            log_path=real_path,
            is_simulation=True,
        )


def test_simulation_fails_if_log_path_is_none() -> None:
    """Verifica que no se permite ejecutar simulaciones sin especificar una ruta temporal."""
    test_date = date(2026, 11, 1)

    with pytest.raises(ValueError, match="Las simulaciones y tests deben especificar explícitamente"):
        run_daily_shadow_pipeline(
            target_date=test_date,
            log_path=None,
            is_simulation=True,
        )


def test_real_log_integrity_and_unique_key() -> None:
    """Valida la integridad del prospective_log real: sin 2026-11-01, sin is_simulation=True, clave única."""
    real_path = Path("data/prospective_log.parquet")
    assert real_path.exists(), "No se encontró prospective_log.parquet real"

    df = pl.read_parquet(real_path)

    # 1. No debe existir la fecha de simulación 2026-11-01
    sim_rows = df.filter(pl.col("date") == date(2026, 11, 1))
    assert sim_rows.height == 0, f"El prospective_log real contiene {sim_rows.height} filas de la simulación 2026-11-01"

    # 2. Todas las filas reales deben tener is_simulation == False
    assert "is_simulation" in df.columns, "Falta la columna is_simulation en prospective_log.parquet"
    assert not any(df.get_column("is_simulation").to_list()), "Existen filas marcadas con is_simulation=True en el log real"

    # 3. Clave única (run_id, cell_id, target_date, horizon)
    unique_key = ["run_id", "cell_id", "target_date", "horizon"]
    for col in unique_key:
        assert col in df.columns, f"Falta la columna {col} de la clave única en prospective_log.parquet"

    n_unique = df.unique(subset=unique_key).height
    assert n_unique == df.height, f"Violación de clave única: {df.height} filas vs {n_unique} únicas"


def test_simulation_in_temp_path_does_not_modify_real_log(tmp_path: Path) -> None:
    """Verifica que una simulación en ruta temporal funciona y deja el log real 100% inalterado."""
    real_path = Path("data/prospective_log.parquet")
    hash_before = _get_file_hash(real_path)
    rows_before = pl.read_parquet(real_path).height

    sim_date = date(2025, 6, 25)
    temp_log = tmp_path / "temp_sim_log.parquet"
    out_dir = tmp_path / "outputs"
    state_dir = tmp_path / "state"

    _ = run_daily_shadow_pipeline(
        target_date=sim_date,
        output_dir=out_dir,
        state_dir=state_dir,
        log_path=temp_log,
        is_simulation=True,
    )

    assert temp_log.exists()
    temp_df = pl.read_parquet(temp_log)
    assert temp_df.height == 13108
    assert all(temp_df.get_column("is_simulation").to_list())
    assert all(temp_df.get_column("excluir_de_evaluacion").to_list())

    # Comprobar que el log real NO fue tocado en absoluto
    hash_after = _get_file_hash(real_path)
    rows_after = pl.read_parquet(real_path).height
    assert hash_before == hash_after, "FALLA CRÍTICA: El log real fue modificado durante la simulación"
    assert rows_before == rows_after, "FALLA CRÍTICA: Cambió el conteo de filas del log real"
