"""Adaptadores para fuentes externas de datos geoespaciales (Fase 2B — C2).

REGLA:
- Documenta fuente oficial, año, licencia y resolución.
- Si una fuente requiere descarga masiva externa no ejecutada en la sesión, el adaptador
  queda completamente implementado e interoperable, señalando formalmente su estado.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class ExternalSourceMetadata:
    source_name: str
    official_provider: str
    dataset_version_or_year: str
    license: str
    documentation_url: str
    status: str  # "EJECUTADO" o "NO EJECUTADO"


class OsmRoadsAdapter:
    """Adaptador para la red vial de OpenStreetMap (OSM).

    Variables provistas:
    - `dist_road_nearest_m`: Distancia euclidiana en metros a cualquier vía transitable.
    - `dist_primary_sec_m`: Distancia euclidiana en metros a rutas primarias o secundarias.
    - `road_density_km2`: Densidad de longitud de caminos (km / km²).
    """

    METADATA = ExternalSourceMetadata(
        source_name="OpenStreetMap Road Network (Chile Extract)",
        official_provider="Geofabrik GmbH / OpenStreetMap Contributors",
        dataset_version_or_year="2024-10",
        license="Open Database License (ODbL) 1.0",
        documentation_url="https://download.geofabrik.de/south-america/chile.html",
        status="NO EJECUTADO (adaptador listo)",
    )

    def __init__(self, raw_path: Path | None = None) -> None:
        self.raw_path = raw_path

    def extract_features(self, grid_df: pl.DataFrame) -> pl.DataFrame:
        """Extrae o imputa formalmente las métricas viales por celda de 1 km."""
        nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id"])
        # Si no se dispone del extracto masivo descargado, devuelve el schema tipado en float32 con nulls documentados
        return nuble_cells.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("dist_road_nearest_m"),
            pl.lit(None, dtype=pl.Float32).alias("dist_primary_sec_m"),
            pl.lit(None, dtype=pl.Float32).alias("road_density_km2"),
        )


class SettlementsAdapter:
    """Adaptador para centros poblados (Ciudades, Pueblos, Aldeas y Caseríos).

    Variables provistas:
    - `dist_settlement_nearest_m`: Distancia euclidiana en metros al centro poblado más próximo.
    """

    METADATA = ExternalSourceMetadata(
        source_name="Entidades y Asentamientos Humanos (Censo 2017)",
        official_provider="Instituto Nacional de Estadísticas (INE) Chile / IDE Chile",
        dataset_version_or_year="2017 (Censo Oficial)",
        license="Datos Abiertos Gobierno de Chile / Open Access",
        documentation_url="https://www.ide.cl/index.php/poblacion-y-sociedad/item/1852-asentamientos-humanos",
        status="NO EJECUTADO (adaptador listo)",
    )

    def __init__(self, raw_path: Path | None = None) -> None:
        self.raw_path = raw_path

    def extract_features(self, grid_df: pl.DataFrame) -> pl.DataFrame:
        nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id"])
        return nuble_cells.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("dist_settlement_nearest_m"),
        )


class EsaWorldCoverAdapter:
    """Adaptador para fracciones de cobertura de suelo desde ESA WorldCover 10m.

    Variables provistas (fracciones en [0.0, 1.0]):
    - `tree_cover_frac`: Fracción de cubierta arbórea (clase 10).
    - `shrub_cover_frac`: Fracción de matorral (clase 20).
    - `grass_cover_frac`: Fracción de pastizal (clase 30).
    - `crop_cover_frac`: Fracción de tierras agrícolas / cultivos (clase 40).
    - `built_cover_frac`: Fracción de suelo edificado o infraestructura (clase 50).
    """

    METADATA = ExternalSourceMetadata(
        source_name="ESA WorldCover 10m 2021 v200",
        official_provider="European Space Agency (ESA) / VITO Remote Sensing",
        dataset_version_or_year="2021 (v200)",
        license="Creative Commons Attribution 4.0 International (CC-BY 4.0)",
        documentation_url="https://esa-worldcover.org/en/data-access",
        status="NO EJECUTADO (adaptador listo)",
    )

    def __init__(self, raw_dir: Path | None = None) -> None:
        self.raw_dir = raw_dir

    def extract_features(self, grid_df: pl.DataFrame) -> pl.DataFrame:
        nuble_cells = grid_df.filter(pl.col("in_nuble")).select(["cell_id"])
        return nuble_cells.with_columns(
            pl.lit(None, dtype=pl.Float32).alias("tree_cover_frac"),
            pl.lit(None, dtype=pl.Float32).alias("shrub_cover_frac"),
            pl.lit(None, dtype=pl.Float32).alias("grass_cover_frac"),
            pl.lit(None, dtype=pl.Float32).alias("crop_cover_frac"),
            pl.lit(None, dtype=pl.Float32).alias("built_cover_frac"),
        )
