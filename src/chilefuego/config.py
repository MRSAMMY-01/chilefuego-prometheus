"""Validación estricta de configuración con Pydantic (Regla 5 y Fase 1.5)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class BBoxWGS84(BaseModel):
    """Bounding Box geográfico en EPSG:4326 (grados decimales)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    west: float = Field(..., ge=-180.0, le=180.0)
    south: float = Field(..., ge=-90.0, le=90.0)
    east: float = Field(..., ge=-180.0, le=180.0)
    north: float = Field(..., ge=-90.0, le=90.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> BBoxWGS84:
        if self.west >= self.east:
            raise ValueError(f"west ({self.west}) debe ser menor que east ({self.east})")
        if self.south >= self.north:
            raise ValueError(f"south ({self.south}) debe ser menor que north ({self.north})")
        return self


class ProjectConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    version: str
    random_seed: int = Field(..., ge=0)
    timezone: str = Field(default="America/Santiago")


class GeospatialConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    work_crs: Literal["EPSG:32719"] = "EPSG:32719"
    geographic_crs: Literal["EPSG:4326"] = "EPSG:4326"
    cell_size_m: int = Field(default=500, gt=0, le=10000)
    reference_area_km2: float = Field(default=13178.0, gt=0.0)
    area_tolerance_rel: float = Field(default=0.05, gt=0.0, le=0.5)
    bbox_wgs84: BBoxWGS84


class TemporalConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start_year: int = Field(..., ge=2012, le=2030)
    end_year: int = Field(..., ge=2012, le=2030)
    start_date: date = Field(default=date(2012, 1, 20))
    end_date: date = Field(default=date(2025, 12, 31))

    @model_validator(mode="after")
    def validate_years(self) -> TemporalConfig:
        if self.start_year > self.end_year:
            raise ValueError("start_year no puede ser mayor que end_year")
        if self.start_date > self.end_date:
            raise ValueError("start_date no puede ser mayor que end_date")
        return self


class PathsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    dpa_geojson_path: Path = Field(default=Path("data/raw/boundaries/dpa_nuble.geojson"))
    conaf_geojson_path: Path = Field(default=Path("data/raw/conaf/conaf_nuble_historico.geojson"))
    manifest_path: Path = Field(default=Path("data/raw/firms_archive/manifest.json"))
    grid_catalog_path: Path
    labels_path: Path
    ignitions_conaf_path: Path = Field(default=Path("data/processed/ignitions_conaf.parquet"))
    excluded_cell_days_path: Path = Field(default=Path("data/processed/excluded_cell_days.parquet"))
    report_path: Path


class FirmsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_url: str
    env_api_key_var: str = "FIRMS_MAP_KEY"
    primary_sources: list[str] = Field(
        default_factory=lambda: ["VIIRS_SNPP_SP"],
        min_length=1,
    )
    sensitivity_sources: list[str] = Field(
        default_factory=lambda: ["VIIRS_NOAA20_SP"],
    )
    history_sources: list[str] = Field(
        default_factory=lambda: ["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP"],
        min_length=1,
    )
    live_sources: list[str] = Field(
        default_factory=lambda: ["VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT"],
        min_length=1,
    )
    window_days: int = Field(default=5, ge=1, le=5)
    max_retries: int = Field(default=4, ge=1, le=10)
    backoff_base_seconds: float = Field(default=1.5, gt=0.0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class ExclusionZoneConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    center_lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    center_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    radius_m: float | None = Field(default=None, gt=0.0)
    polygon_lon_lat: list[tuple[float, float]] | None = Field(default=None, min_length=4)

    @model_validator(mode="after")
    def validate_geometry_definition(self) -> ExclusionZoneConfig:
        has_circle = (
            self.center_lon is not None
            and self.center_lat is not None
            and self.radius_m is not None
        )
        has_poly = self.polygon_lon_lat is not None and len(self.polygon_lon_lat) >= 4
        if not has_circle and not has_poly:
            raise ValueError(
                f"Zona de exclusión '{self.name}' debe definir (center_lon, center_lat, radius_m) "
                "o polygon_lon_lat."
            )
        return self


class StationarySourceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    min_distinct_days: int = Field(default=12, ge=2)
    min_distinct_months: int = Field(default=9, ge=1, le=12)
    flag_winter_activity: bool = Field(default=True)
    winter_months: list[int] = Field(default_factory=lambda: [6, 7, 8], min_length=1)
    min_winter_days: int = Field(default=1, ge=1)


class FilteringConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_confidences: list[str] = Field(..., min_length=1)
    allowed_types: list[int] = Field(default_factory=lambda: [0], min_length=1)
    exclusion_zones: list[ExclusionZoneConfig] = Field(default_factory=list)
    stationary_source: StationarySourceConfig


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spatial_eps_m: float = Field(..., gt=0.0)
    temporal_eps_hours: float = Field(..., gt=0.0)
    min_samples: int = Field(default=1, ge=1)
    max_cluster_duration_hours: float | None = Field(default=240.0, gt=0.0)
    max_cluster_radius_m: float | None = Field(default=15000.0, gt=0.0)
    event_class_rule: Literal["rule_a", "rule_b", "rule_c"] = "rule_b"
    exclusion_buffer_m: float = Field(default=2500.0, ge=0.0)
    exclusion_post_hours: float = Field(default=72.0, ge=0.0)


class AppConfig(BaseModel):
    """Esquema raíz validado del archivo configs/config.yaml."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project: ProjectConfig
    geospatial: GeospatialConfig
    temporal: TemporalConfig
    paths: PathsConfig
    firms: FirmsConfig
    filtering: FilteringConfig
    clustering: ClusteringConfig


def load_config(config_path: str | Path = "configs/config.yaml") -> AppConfig:
    """Carga y valida el archivo YAML de configuración."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw_dict = yaml.safe_load(fh)
    return AppConfig.model_validate(raw_dict)
