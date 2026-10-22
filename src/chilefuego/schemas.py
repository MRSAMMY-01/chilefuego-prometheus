"""Contratos de datos con Pandera en cada frontera entre etapas (Regla 7 y Fase 1.5)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

GRID_CATALOG_SCHEMA = pa.DataFrameSchema(
    columns={
        "cell_id": pa.Column(pl.String, unique=True, checks=pa.Check.str_startswith("NUBLE_")),
        "row_idx": pa.Column(pl.Int64, checks=pa.Check.ge(0)),
        "col_idx": pa.Column(pl.Int64, checks=pa.Check.ge(0)),
        "min_x": pa.Column(pl.Float64),
        "min_y": pa.Column(pl.Float64),
        "max_x": pa.Column(pl.Float64),
        "max_y": pa.Column(pl.Float64),
        "centroid_x": pa.Column(pl.Float64),
        "centroid_y": pa.Column(pl.Float64),
        "centroid_lon": pa.Column(pl.Float64, checks=pa.Check.in_range(-75.0, -70.0)),
        "centroid_lat": pa.Column(pl.Float64, checks=pa.Check.in_range(-39.0, -35.0)),
        "in_nuble": pa.Column(pl.Boolean),
        "in_argentina": pa.Column(pl.Boolean),
        "geometry_wkt": pa.Column(pl.String, checks=pa.Check.str_startswith("POLYGON")),
    }
)

RAW_FIRMS_SCHEMA = pa.DataFrameSchema(
    columns={
        "latitude": pa.Column(pl.Float64, checks=pa.Check.in_range(-90.0, 90.0)),
        "longitude": pa.Column(pl.Float64, checks=pa.Check.in_range(-180.0, 180.0)),
        "bright_ti4": pa.Column(pl.Float64, nullable=True),
        "scan": pa.Column(pl.Float64, nullable=True),
        "track": pa.Column(pl.Float64, nullable=True),
        "acq_date": pa.Column(pl.String, checks=pa.Check.str_matches(r"^\d{4}-\d{2}-\d{2}$")),
        "acq_time": pa.Column(pl.String, checks=pa.Check.str_matches(r"^\d{4}$")),
        "satellite": pa.Column(
            pl.String,
            checks=pa.Check.isin(["VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "VIIRS_NOAA21_SP"]),
        ),
        "instrument": pa.Column(pl.String, nullable=True),
        "confidence": pa.Column(pl.String),
        "version": pa.Column(pl.String, nullable=True),
        "bright_ti5": pa.Column(pl.Float64, nullable=True),
        "frp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "daynight": pa.Column(pl.String, nullable=True),
        "type": pa.Column(pl.Int64, nullable=True),
    },
    strict=False,
)

FILTERED_FIRMS_SCHEMA = pa.DataFrameSchema(
    columns={
        "cell_id": pa.Column(pl.String, checks=pa.Check.str_startswith("NUBLE_")),
        "x_utm": pa.Column(pl.Float64),
        "y_utm": pa.Column(pl.Float64),
        "latitude": pa.Column(pl.Float64, checks=pa.Check.in_range(-37.5, -35.9)),
        "longitude": pa.Column(pl.Float64, checks=pa.Check.in_range(-73.0, -70.9)),
        "acq_ts_utc": pa.Column(pl.Datetime("us", "UTC")),
        "local_date": pa.Column(pl.Date),
        "confidence": pa.Column(pl.String),
        "frp": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "satellite": pa.Column(pl.String),
    },
    strict=False,
)

IGNITION_LABELS_SCHEMA = pa.DataFrameSchema(
    columns={
        "cell_id": pa.Column(pl.String, checks=pa.Check.str_startswith("NUBLE_")),
        "date": pa.Column(pl.Date),
        "is_ignition": pa.Column(pl.Int64, checks=pa.Check.isin([0, 1])),
        "cluster_id": pa.Column(pl.Int64, checks=pa.Check.ge(0)),
        "n_detections": pa.Column(pl.Int64, checks=pa.Check.ge(1)),
        "n_pixels": pa.Column(pl.Int64, checks=pa.Check.ge(1)),
        "n_passes": pa.Column(pl.Int64, checks=pa.Check.ge(1)),
        "frp_max": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "frp_sum": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "duration_hours": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "extent_m": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
        "event_class": pa.Column(
            pl.String,
            checks=pa.Check.isin(["wildfire_candidate", "small_or_agri"]),
        ),
        "first_detection_ts": pa.Column(pl.Datetime("us", "UTC")),
        "detected_night": pa.Column(pl.Boolean),
        "window_start_utc": pa.Column(pl.Datetime("us", "UTC")),
    }
)

EXCLUDED_CELL_DAYS_SCHEMA = pa.DataFrameSchema(
    columns={
        "cell_id": pa.Column(pl.String, checks=pa.Check.str_startswith("NUBLE_")),
        "date": pa.Column(pl.Date),
        "cluster_id": pa.Column(pl.Int64, checks=pa.Check.ge(0)),
        "reason": pa.Column(
            pl.String,
            checks=pa.Check.isin(["active_cluster_footprint", "small_or_agri_burn"]),
        ),
    }
)

QUALITY_FLAGS_DTYPE = pl.Struct(
    {
        "crs_ok": pl.Boolean,
        "date_ok": pl.Boolean,
        "time_ok": pl.Boolean,
        "dup_removed": pl.Boolean,
        "matched_firms": pl.Boolean,
    }
)

IGNITIONS_CONAF_SCHEMA = pa.DataFrameSchema(
    columns={
        "cell_id": pa.Column(pl.String, checks=pa.Check.str_startswith("NUBLE_")),
        "date_local": pa.Column(pl.Date, nullable=True),
        "start_ts_local": pa.Column(pl.Datetime("us", "America/Santiago"), nullable=True),
        "season": pa.Column(pl.String, checks=pa.Check.str_startswith("if_temporada_")),
        "causa": pa.Column(pl.String),
        "superficie_ha": pa.Column(pl.Float64, checks=pa.Check.ge(0.0)),
    },
    strict=False,
)


def validate_grid_catalog(df: pl.DataFrame) -> pl.DataFrame:
    return GRID_CATALOG_SCHEMA.validate(df)


def validate_raw_firms(df: pl.DataFrame) -> pl.DataFrame:
    return RAW_FIRMS_SCHEMA.validate(df)


def validate_filtered_firms(df: pl.DataFrame) -> pl.DataFrame:
    return FILTERED_FIRMS_SCHEMA.validate(df)


def validate_ignition_labels(df: pl.DataFrame) -> pl.DataFrame:
    return IGNITION_LABELS_SCHEMA.validate(df)


def validate_excluded_cell_days(df: pl.DataFrame) -> pl.DataFrame:
    return EXCLUDED_CELL_DAYS_SCHEMA.validate(df)


def validate_ignitions_conaf(df: pl.DataFrame) -> pl.DataFrame:
    validated = IGNITIONS_CONAF_SCHEMA.validate(df)
    if "quality_flags" not in validated.columns:
        raise ValueError("Columna 'quality_flags' ausente en ignitions_conaf.parquet")
    qf_dtype = validated.schema["quality_flags"]
    if qf_dtype != QUALITY_FLAGS_DTYPE:
        raise ValueError(
            f"Tipo de 'quality_flags' inválido: esperado {QUALITY_FLAGS_DTYPE}, recibido {qf_dtype}"
        )
    return validated
