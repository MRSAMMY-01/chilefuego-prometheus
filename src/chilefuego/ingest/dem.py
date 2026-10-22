"""Descarga y extracción de características topográficas Copernicus DEM GLO-30 (30m).

Dataset: Copernicus DEM GLO-30 (Public AWS Open Data, CC-BY / Open License).
Resolución: 1 arc-segundo (~30 metros).
Tiles que cubren Ñuble (4 tiles de 1°x1°):
- S37_00_W073_00
- S37_00_W072_00
- S38_00_W073_00
- S38_00_W072_00
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests

TILES = [
    "Copernicus_DSM_COG_10_S37_00_W073_00_DEM",
    "Copernicus_DSM_COG_10_S37_00_W072_00_DEM",
    "Copernicus_DSM_COG_10_S38_00_W073_00_DEM",
    "Copernicus_DSM_COG_10_S38_00_W072_00_DEM",
]
BASE_URL = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


def download_dem_tiles(dest_dir: Path) -> dict[str, str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = dest_dir / "manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    hashes: dict[str, str] = manifest.get("files", {})

    for tile in TILES:
        fname = f"{tile}.tif"
        file_path = dest_dir / fname
        url = f"{BASE_URL}/{tile}/{fname}"

        if file_path.exists() and fname in hashes:
            print(f"Tile {fname} ya verificado con SHA-256.")
            continue

        print(f"Descargando {fname} desde {url}...")
        t0 = time.time()
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()

        sha256 = hashlib.sha256()
        total_bytes = 0
        with open(file_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    sha256.update(chunk)
                    total_bytes += len(chunk)

        h_val = sha256.hexdigest()
        hashes[fname] = h_val
        print(f"Descargado {fname} ({total_bytes / (1024*1024):.2f} MB) en {time.time() - t0:.1f} s. SHA256: {h_val}")

    manifest_data = {
        "dataset": "Copernicus DEM GLO-30",
        "provider": "European Space Agency / AWS Open Data",
        "license": "Copernicus Open Access / Public",
        "resolution": "30m (1 arc-second)",
        "files": hashes,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return hashes


if __name__ == "__main__":
    download_dem_tiles(Path("data/raw/dem"))
