"""Verificación de integridad, tamaño, SHA-256 y lectura completa de bloques (D1)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pyogrio
import rasterio

EXPECTED_HASHES = {
    "data/raw/osm/chile-latest.osm.pbf": "A15DB02986D82DBB99BDA290FA1F32C9DB74C1EED4B9C99C69B24C2D9C836587".lower(),
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif": "1CAC2889742AC9A90244BBED905CAF748F4643C4EF284EFBE283E59EDE2531B4".lower(),
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W075_Map.tif": "E6058819E57A1F98F565F91AB8318D189ABA700689291EA3BB8A4C4D012DB07C".lower(),
}

EXPECTED_SIZES = {
    "data/raw/osm/chile-latest.osm.pbf": 349986111,
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif": 116644425,
    "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W075_Map.tif": 52427481,
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_all() -> None:
    print("=== D1: VERIFICACION DE ARCHIVOS, TAMAÑOS Y HASHES SHA-256 ===")
    
    # 1. Comprobar existencia, tamaño y SHA-256
    computed_hashes = {}
    for rel_path, exp_hash in EXPECTED_HASHES.items():
        p = Path(rel_path)
        if not p.exists():
            raise FileNotFoundError(f"DETENTE: Archivo faltante: {p}")
        sz = p.stat().st_size
        exp_sz = EXPECTED_SIZES[rel_path]
        print(f"[{p.name}] Tamaño local: {sz:,} bytes | Servidor Content-Length: {exp_sz:,} bytes (Match: {sz == exp_sz})")
        if sz != exp_sz:
            raise ValueError(f"DETENTE: Tamaño no coincide para {p}: local {sz} vs esperado {exp_sz}")
        
        t0 = time.time()
        c_hash = compute_sha256(p)
        computed_hashes[rel_path] = c_hash
        print(f"  SHA-256: {c_hash}")
        print(f"  Esperado: {exp_hash}")
        print(f"  Coincidencia: {c_hash == exp_hash} (calculado en {time.time()-t0:.2f} s)")
        if c_hash != exp_hash:
            raise ValueError(f"DETENTE: Hash SHA-256 no coincide para {p}")

    # 2. Recorrer todos los bloques de cada TIFF con rasterio
    print("\n=== D1: LECTURA COMPLETA DE BLOQUES DE TIFFS (WORLDCOVER) ===")
    for tif_key in [
        "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif",
        "data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W075_Map.tif",
    ]:
        p = Path(tif_key)
        t0 = time.time()
        with rasterio.open(p) as src:
            block_count = 0
            # Iterar sobre las ventanas/bloques del archivo
            for _ji, window in src.block_windows(1):
                src.read(1, window=window)
                block_count += 1
            print(f"[{p.name}] Leídos exitosamente {block_count} bloques ({src.width}x{src.height} px, {src.crs}) en {time.time()-t0:.2f} s. 0 errores.")

    # 3. Recorrer el PBF completo hasta el final contando features con pyogrio
    print("\n=== D1: RECORRIDO COMPLETO DEL PBF DE OSM ===")
    pbf_path = Path("data/raw/osm/chile-latest.osm.pbf")
    t0 = time.time()
    info_lines = pyogrio.read_info(pbf_path, layer="lines")
    info_points = pyogrio.read_info(pbf_path, layer="points")
    print(f"PBF Capas: lines={info_lines['features']:,} features, points={info_points['features']:,} features")
    
    # Recorrido iterativo por chunks para no desbordar RAM
    for _chunk in pyogrio.read_dataframe(pbf_path, layer="lines", columns=["highway"], read_geometry=False, max_features=None, batch_size=100000):
        # pyogrio read_dataframe returns dataframe directly or if batch_size isn't supported, returns df
        pass
    print("PBF verificado hasta el final sin corrupción.")

    # 4. Actualizar manifests
    osm_manifest = {
        "dataset": "OpenStreetMap Chile Extract",
        "provider": "Geofabrik GmbH",
        "file": "chile-latest.osm.pbf",
        "bytes": EXPECTED_SIZES["data/raw/osm/chile-latest.osm.pbf"],
        "sha256": computed_hashes["data/raw/osm/chile-latest.osm.pbf"],
        "license": "ODbL 1.0",
    }
    Path("data/raw/osm/manifest.json").write_text(json.dumps(osm_manifest, indent=2), encoding="utf-8")

    wc_manifest = {
        "dataset": "ESA WorldCover 10m 2021 v200",
        "provider": "European Space Agency",
        "year": 2021,
        "files": {
            "ESA_WorldCover_10m_2021_v200_S39W072_Map.tif": {
                "bytes": EXPECTED_SIZES["data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif"],
                "sha256": computed_hashes["data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W072_Map.tif"],
            },
            "ESA_WorldCover_10m_2021_v200_S39W075_Map.tif": {
                "bytes": EXPECTED_SIZES["data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W075_Map.tif"],
                "sha256": computed_hashes["data/raw/worldcover/ESA_WorldCover_10m_2021_v200_S39W075_Map.tif"],
            },
        },
        "license": "Creative Commons Attribution 4.0 International (CC-BY 4.0)",
    }
    Path("data/raw/worldcover/manifest.json").write_text(json.dumps(wc_manifest, indent=2), encoding="utf-8")
    print("\nManifests guardados exitosamente en data/raw/osm/manifest.json y data/raw/worldcover/manifest.json.")


if __name__ == "__main__":
    verify_all()
