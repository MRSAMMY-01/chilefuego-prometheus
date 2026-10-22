"""Contrato de Inferencia Operativa y Ensamblado de Covariables (Fase 2D — G4).

PORQUÉ (Regla 5):
- Fija contractualmente cómo se construyen las 25 variables para una nueva fecha objetivo D.
- Define la memoria continua de los códigos de humedad del FWI (FFMC, DMC, DC) desde D-1.
- Documenta el mapeo geoespacial por vecino más cercano de nodos meteorológicos a celdas de 1 km.
- Establece la especificación técnica para la transición entre reanálisis ERA5-Land (latencia ~5 días)
  y pronósticos numéricos del tiempo operativos (NWP, e.g. Open-Meteo, ECMWF IFS, GFS, WRF).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from chilefuego.models.xgboost_experiment import PRUNED_25_FEATURES


@dataclass
class FWIState:
    """Estado de memoria de los códigos de humedad del suelo para el inicio del día D."""
    ffmc: float = 85.0
    dmc: float = 6.0
    dc: float = 15.0


def compute_vpd(temperature_c: float | np.ndarray, rh_pct: float | np.ndarray) -> float | np.ndarray:
    """Calcula el Déficit de Presión de Vapor (VPD, kPa) a partir de temperatura y humedad relativa."""
    es = 0.61078 * np.exp((17.27 * temperature_c) / (temperature_c + 237.3))
    ea = es * (np.clip(rh_pct, 0.0, 100.0) / 100.0)
    return np.maximum(es - ea, 0.0)


def build_inference_feature_matrix(
    static_df: pl.DataFrame,
    prior_series: pl.Series | np.ndarray,
    weather_dict_by_cell: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Ensambla contractualmente la matriz de 25 variables podadas en orden canónico estricto."""
    n_cells = static_df.height

    # Validar que static_df contenga las 13 estáticas requeridas
    required_static = [
        "elevation_mean", "slope_mean", "northness", "eastness",
        "dist_road_nearest_m", "dist_primary_sec_trunk_m", "road_density_km2",
        "tree_cover_frac", "shrub_cover_frac", "grass_cover_frac",
        "crop_cover_frac", "built_cover_frac", "dist_built_nearest_m",
    ]
    missing = [c for c in required_static if c not in static_df.columns]
    if missing:
        raise ValueError(f"Faltan covariables estáticas en static_df: {missing}")

    assembled: dict[str, Any] = {}

    # 1. Topografía y DEM (4)
    for c in ["elevation_mean", "slope_mean", "northness", "eastness"]:
        assembled[c] = static_df.get_column(c).cast(pl.Float32)

    # 2. Infraestructura Vial OSM (3)
    for c in ["dist_road_nearest_m", "dist_primary_sec_trunk_m", "road_density_km2"]:
        assembled[c] = static_df.get_column(c).cast(pl.Float32)

    # 3. Cobertura de Suelo WorldCover (6)
    for c in ["tree_cover_frac", "shrub_cover_frac", "grass_cover_frac", "crop_cover_frac", "built_cover_frac", "dist_built_nearest_m"]:
        assembled[c] = static_df.get_column(c).cast(pl.Float32)

    # 4. Prior espacial histórico (1)
    if isinstance(prior_series, np.ndarray):
        assembled["prior_density"] = pl.Series("prior_density", prior_series, dtype=pl.Float32)
    else:
        assembled["prior_density"] = prior_series.cast(pl.Float32)

    # 5. Meteorología instantánea y FWI de mediodía (11)
    required_weather = [
        "temp_noon_c", "vpd_noon_kpa", "rain_24h_noon_mm", "rh_noon_pct",
        "wind_noon_kmh", "fwi", "isi", "bui", "ffmc", "dmc", "dc",
    ]
    for c in required_weather:
        if c not in weather_dict_by_cell:
            raise ValueError(f"Falta variable meteorológica requerida '{c}' en weather_dict_by_cell")
        w_arr = np.asarray(weather_dict_by_cell[c], dtype=np.float32)
        if len(w_arr) != n_cells:
            raise ValueError(f"Largo de '{c}' ({len(w_arr)}) no coincide con n_cells ({n_cells})")
        assembled[c] = pl.Series(c, w_arr, dtype=pl.Float32)

    df_out = pl.DataFrame(assembled)

    # Validar orden exacto de columnas
    if df_out.columns != PRUNED_25_FEATURES:
        raise ValueError("El orden de columnas ensamblado difiere del orden canónico de PRUNED_25_FEATURES")

    return df_out


class InferencePipeline:
    """Pipeline de inferencia en producción para ChileFuego."""

    def __init__(
        self,
        model_path: str | Path = "models/chilefuego_xgb_pruned_v1.json",
        calibrator_path: str | Path = "models/platt_calibrator.json",
        thresholds_path: str | Path = "models/risk_thresholds.json",
    ) -> None:
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))

        with open(calibrator_path, encoding="utf-8") as f:
            c_data = json.load(f)
            self.platt_slope = float(c_data["slope"])
            self.platt_intercept = float(c_data["intercept"])

        with open(thresholds_path, encoding="utf-8") as f:
            t_data = json.load(f)
            self.p80 = float(t_data["thresholds"]["p80"])
            self.p95 = float(t_data["thresholds"]["p95"])
            self.p99 = float(t_data["thresholds"]["p99"])

    def predict(self, feature_df: pl.DataFrame) -> dict[str, np.ndarray]:
        X = feature_df.select(PRUNED_25_FEATURES).to_numpy().astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=PRUNED_25_FEATURES)

        raw_scores = self.booster.predict(dmat)

        # Calibración Platt analítica
        logit = np.log(np.clip(raw_scores, 1e-7, 1.0 - 1e-7) / (1.0 - np.clip(raw_scores, 1e-7, 1.0 - 1e-7)))
        calibrated_probs = 1.0 / (1.0 + np.exp(-(self.platt_slope * logit + self.platt_intercept)))

        # Clasificación en niveles de riesgo relativo
        tiers = np.full(len(calibrated_probs), "Bajo", dtype=object)
        tiers[(calibrated_probs >= self.p80) & (calibrated_probs < self.p95)] = "Moderado"
        tiers[(calibrated_probs >= self.p95) & (calibrated_probs < self.p99)] = "Alto"
        tiers[calibrated_probs >= self.p99] = "Extremo"

        return {
            "raw_score": raw_scores,
            "calibrated_prob": calibrated_probs,
            "risk_tier": tiers,
        }


def get_operational_contract_documentation() -> dict[str, Any]:
    """Retorna la especificación técnica de desacople entre ERA5-Land y pronósticos NWP."""
    return {
        "contract_version": "1.0.0",
        "inputs_noon_standard": {
            "time_utc": "15:00 UTC",
            "time_local_clt": "12:00 CLT (11:00 CLST en horario de invierno)",
            "observation_window_rain24h": "D-1 12:00 CLT a D 12:00 CLT",
        },
        "spatial_grid": {
            "projection": "EPSG:32719 (WGS 84 / UTM zone 19S)",
            "resolution_m": 1000,
            "n_cells_nuble": 13108,
            "n_era5_nodes": 399,
            "mapping_method": "KDTree Euclidean Nearest Neighbor (Centroid to ERA5 grid node)",
        },
        "sources_and_latency": {
            "era5_land_reanalysis": {
                "role": "Entrenamiento histórico, benchmarking y reanálisis diagnóstico",
                "latency_days": "5 días (ERA5T) a 60-90 días (ERA5 consolidado)",
                "suitability_for_realtime": "Inadecuado para despacho operativo el mismo día D",
            },
            "nwp_forecast_operational": {
                "role": "Inferencia operativa en día D y proyección prospectiva (D+1, D+2)",
                "candidate_models": ["ECMWF IFS (0.1°)", "GFS (0.25°)", "WRF Dirección Meteorológica de Chile (3 km)", "Open-Meteo"],
                "required_corrections": [
                    "Alineación horaria estricta de la lluvia acumulada a las 12:00 CLT (no a 00:00 UTC)",
                    "Corrección de sesgo local en humedad relativa mínima y rachas de viento andino (Puelche)",
                    "Conservación de memoria continua de códigos FWI (DC tiene memoria de ~52 días; encadenar NWP sobre el estado previo sin resetear)",
                ],
            },
        },
    }
