"""Utilidades de logging estructurado, hashing reproducible y escritura atómica (Reglas 4 y 6)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


class StructuredJsonFormatter(logging.Formatter):
    """Formateador JSON de una línea para trazabilidad reproducible."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "msg": record.getMessage(),
        }
        extra_data = getattr(record, "structured_data", None)
        if isinstance(extra_data, dict):
            payload.update(extra_data)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """Retorna un logger con formato estructurado JSON."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, **kwargs: Any) -> None:
    """Registra un evento estructurado con metadatos clave-valor."""
    logger.info(message, extra={"structured_data": kwargs})


def compute_file_sha256(file_path: str | Path) -> str:
    """Calcula el hash SHA-256 de un archivo en disco por bloques sin cargarlo entero en RAM."""
    path = Path(file_path)
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_write_parquet(df: pl.DataFrame, target_path: str | Path) -> str:
    """Escribe un DataFrame de Polars a Parquet de forma atómica y retorna su SHA-256.

    Previene archivos corruptos ante caídas o interrupciones de red/proceso:
    escribe primero en un archivo temporal en el mismo directorio y aplica os.replace().
    """
    dest = Path(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(dest.parent),
        prefix=f".tmp_{dest.stem}_",
        suffix=".parquet",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.write_parquet(tmp_path, compression="zstd")
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return compute_file_sha256(dest)
