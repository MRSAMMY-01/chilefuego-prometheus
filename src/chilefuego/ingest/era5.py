"""Descarga mensual de ERA5-Land vía `cdsapi` con manifest SHA-256 (Fase 2A — F2.4).

Verificado en documentación oficial (consultada 2026-10-18):
- CDS API (https://cds.climate.copernicus.eu/how-to-api): `$HOME/.cdsapirc` con
  `url: https://cds.climate.copernicus.eu/api` y `key: <PERSONAL-ACCESS-TOKEN>`; cliente
  `pip install "cdsapi>=0.7.7"`. Aquí las credenciales se leen de variables de entorno o `.env`
  (`CDSAPI_URL`, `CDSAPI_KEY`) y JAMÁS se imprimen ni se registran.
- ERA5-Land data documentation (ECMWF Confluence): nombres CDS `2m_temperature`,
  `2m_dewpoint_temperature`, `10m_u_component_of_wind`, `10m_v_component_of_wind`,
  `total_precipitation` (m), `volumetric_soil_water_layer_1` (m3 m-3). Las acumulaciones de ERA5-Land
  van desde 00 UTC hasta el paso del pronóstico; el valor con validez 00 UTC es la acumulación de las
  24 h del día anterior (la desacumulación está en `chilefuego.features.weather`).

Los archivos se guardan intactos tal como los entrega el CDS (`.zip` si el servidor empaqueta y los
NetCDF extraídos sin modificar), y cada uno queda registrado con su SHA-256 en `manifest.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from chilefuego.logging_utils import get_logger, log_event

logger = get_logger(__name__)

DATASET = "reanalysis-era5-land"
ERA5_LAND_VARIABLES: tuple[str, ...] = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "volumetric_soil_water_layer_1",
)
HOURS: tuple[str, ...] = tuple(f"{h:02d}:00" for h in range(24))


@dataclass(frozen=True)
class CdsCredentials:
    url: str
    key: str

    def __repr__(self) -> str:  # nunca exponer el token
        return "CdsCredentials(url=<set>, key=<redacted>)"


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def load_cds_credentials(dotenv_path: Path = Path(".env")) -> CdsCredentials:
    """Lee `CDSAPI_URL`/`CDSAPI_KEY` de entorno o `.env`. Falla sin revelar valores."""
    env = _read_dotenv(dotenv_path)
    url = os.environ.get("CDSAPI_URL") or env.get("CDSAPI_URL", "")
    key = os.environ.get("CDSAPI_KEY") or env.get("CDSAPI_KEY", "")
    missing = [n for n, v in (("CDSAPI_URL", url), ("CDSAPI_KEY", key)) if not v]
    if missing:
        raise RuntimeError(
            f"Faltan credenciales CDS: {missing}. Pasos: (1) crear cuenta ECMWF/CDS, (2) copiar el "
            "Personal Access Token desde el perfil, (3) aceptar la licencia del dataset "
            "'reanalysis-era5-land' en su página del CDS, (4) definir CDSAPI_URL y CDSAPI_KEY."
        )
    return CdsCredentials(url=url, key=key)


def era5_area(
    west: float, south: float, east: float, north: float, pad_deg: float = 0.2
) -> list[float]:
    """Área CDS en orden [North, West, South, East] con margen `pad_deg`."""
    return [
        round(north + pad_deg, 2),
        round(west - pad_deg, 2),
        round(south - pad_deg, 2),
        round(east + pad_deg, 2),
    ]


def month_range(start: date, end: date) -> list[tuple[int, int]]:
    """Meses (año, mes) entre `start` y `end`, ambos inclusive."""
    if start > end:
        raise ValueError("start no puede ser posterior a end")
    out: list[tuple[int, int]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def build_month_request(year: int, month: int, area: list[float]) -> dict[str, Any]:
    """Request mensual (todas las horas/días; el CDS ignora días inexistentes del mes)."""
    return {
        "variable": list(ERA5_LAND_VARIABLES),
        "year": f"{year:04d}",
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": list(HOURS),
        "data_format": "netcdf",
        "download_format": "zip",
        "area": area,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data
    return {"dataset": DATASET, "files": {}}


def _save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def register_downloaded_month(
    archive_path: Path, out_dir: Path, manifest: dict[str, Any], request: dict[str, Any]
) -> list[Path]:
    """Extrae (sin modificar) los NetCDF del archivo y registra SHA-256 de todo en el manifest."""
    entries: dict[str, Any] = manifest.setdefault("files", {})
    produced: list[Path] = []
    stem = archive_path.stem
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                target = out_dir / f"{stem}__{Path(member).name}"
                with zf.open(member) as src, target.open("wb") as dst:
                    dst.write(src.read())
                produced.append(target)
    else:  # el servidor entregó un NetCDF suelto
        target = archive_path.with_suffix(".nc")
        archive_path.replace(target)
        archive_path = target
        produced.append(target)
    for p in [archive_path, *produced]:
        entries[p.name] = {
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
            "role": "archive" if p == archive_path and p.suffix == ".zip" else "netcdf",
            "request": {k: v for k, v in request.items() if k != "day"},
            "registered_at_utc": datetime.now(UTC).isoformat(),
        }
    return produced


def download_era5_land(
    start: date,
    end: date,
    area: list[float],
    out_dir: Path,
    manifest_path: Path,
    dotenv_path: Path = Path(".env"),
) -> None:
    """Descarga idempotente mes a mes: omite meses ya registrados en el manifest."""
    import cdsapi  # import diferido: solo necesario al descargar

    creds = load_cds_credentials(dotenv_path)
    client = cdsapi.Client(url=creds.url, key=creds.key, quiet=True, progress=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(manifest_path)

    for year, month in month_range(start, end):
        stem = f"era5land_{year:04d}_{month:02d}"
        archive = out_dir / f"{stem}.zip"
        entries = manifest.get("files", {})

        # Reanudable y estricto: salta el mes si los archivos extraídos existen y coinciden con su SHA-256
        month_files = [p for p in out_dir.glob(f"{stem}*") if p.is_file() and p.name in entries]
        netcdf_files = [p for p in month_files if p.suffix == ".nc"]
        if netcdf_files:
            hashes_match = all(
                sha256_file(p) == entries[p.name].get("sha256") for p in netcdf_files
            )
            if hashes_match:
                log_event(logger, "era5_month_skip_verified", year=year, month=month, n_files=len(netcdf_files))
                continue

        request = build_month_request(year, month, area)
        log_event(logger, "era5_month_request", year=year, month=month)
        client.retrieve(DATASET, request, str(archive))
        register_downloaded_month(archive, out_dir, manifest, request)
        _save_manifest(manifest_path, manifest)
        log_event(logger, "era5_month_done", year=year, month=month)


def main() -> None:
    from chilefuego.config import load_config

    parser = argparse.ArgumentParser(description="Descarga ERA5-Land mensual (F2.4)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--start", default="2017-04-01")
    parser.add_argument("--end", default="2025-06-30")
    args = parser.parse_args()
    cfg = load_config(args.config)
    bb = cfg.geospatial.bbox_wgs84
    area = era5_area(bb.west, bb.south, bb.east, bb.north, pad_deg=0.2)
    out_dir = cfg.paths.raw_dir / "era5_land"
    download_era5_land(
        date.fromisoformat(args.start),
        date.fromisoformat(args.end),
        area,
        out_dir,
        out_dir / "manifest.json",
    )


if __name__ == "__main__":
    main()
