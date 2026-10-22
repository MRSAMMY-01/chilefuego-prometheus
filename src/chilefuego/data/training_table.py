"""Construcción de la tabla de entrenamiento estratificada con pesos de muestreo (Fase 2B — D4/D5).

PORQUÉ (Regla 5):
- En el panel completo existen 33.345.071 celdas-día efectivas y 3.237 positivos (desbalance 1 : 10.301).
- MUESTREO ESTRATIFICADO PROBABILÍSTICO (D4):
  * Todos los positivos se conservan (N = 3.237, peso = 1.0).
  * Los negativos se dividen en cada (temporada, mes) en dos estratos según umbral de riesgo compuesto
    (fwi * prior_density):
    - Estrato de Alto Riesgo ("hard negatives"): percentil >= 80 de riesgo en ese mes.
    - Estrato Estándar ("regular negatives"): percentil < 80.
  * Se muestrea ALEATORIAMENTE dentro de cada estrato (sin top-N determinístico).
  * Ponderación exacta por estrato: peso = tamaño_poblacional_estrato / tamaño_muestra_estrato.
  * La suma total de pesos reconstruye exactamente los 33.345.071 celdas-día del panel efectivo.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from chilefuego.config import AppConfig, load_config
from chilefuego.labels.panel import (
    DATED_SEASONS,
    build_phase2a_daily_panel,
    materialize_season_cell_day_panel,
)
from chilefuego.logging_utils import get_logger, log_event
from pyproj import Transformer
from scipy.spatial import cKDTree

logger = get_logger(__name__)


def build_stratified_training_sample(
    config: AppConfig,
    neg_to_pos_ratio: int = 20,
    hard_negative_pct: float = 0.80,
    seed: int = 42,
    out_path: Path | None = Path("data/processed/training_sample_phase2b.parquet"),
) -> pl.DataFrame:
    """Construye la tabla de entrenamiento uniendo positivos y negativos estratificados con covariables."""
    t0 = time.time()
    grid_df = pl.read_parquet(config.paths.grid_catalog_path)
    special_cell_days_df, summary = build_phase2a_daily_panel(config)

    # Cargar features estáticas (14 variables sin slope_max)
    static_path = config.paths.processed_dir / "static_features_1km.parquet"
    if not static_path.exists():
        raise FileNotFoundError(f"No se encuentra {static_path}.")
    static_df = pl.read_parquet(static_path)

    # Cargar prior por temporada
    prior_path = config.paths.processed_dir / "prior_density_by_cell_season.parquet"
    if not prior_path.exists():
        raise FileNotFoundError(f"No se encuentra {prior_path}.")
    prior_df = pl.read_parquet(prior_path)

    # Cargar meteorología extendida por nodo
    weather_path = config.paths.processed_dir / "era5_land_extended_weather_nodes.parquet"
    if not weather_path.exists():
        raise FileNotFoundError(f"No se encuentra {weather_path}.")
    weather_df = pl.read_parquet(weather_path)

    # Mapeo celda 1 km a nodo ERA5-Land
    nuble_cells = grid_df.filter(pl.col("in_nuble"))
    cx = nuble_cells["centroid_x"].to_numpy()
    cy = nuble_cells["centroid_y"].to_numpy()
    cell_ids = nuble_cells["cell_id"].to_list()

    transformer = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    cell_lons, cell_lats = transformer.transform(cx, cy)

    unique_nodes = weather_df.select(["node_id"]).unique().to_series().to_list()
    node_coords = np.array([[float(p.split("_")[1]), float(p.split("_")[0])] for p in unique_nodes])
    tree = cKDTree(node_coords)
    _, node_idxs = tree.query(np.column_stack([cell_lons, cell_lats]), k=1)
    cell_node_df = pl.DataFrame({
        "cell_id": cell_ids,
        "node_id": [unique_nodes[idx] for idx in node_idxs],
    })

    sampled_dfs: list[pl.DataFrame] = []

    for s_name in DATED_SEASONS:
        fold_panel = materialize_season_cell_day_panel(
            grid_df=grid_df,
            special_cell_days_df=special_cell_days_df,
            season=s_name,
            include_excluded=False,
        )

        fold_panel = fold_panel.join(cell_node_df, on="cell_id", how="left")
        fold_panel = fold_panel.join(
            prior_df.filter(pl.col("season") == s_name).select(["cell_id", "prior_density"]),
            on="cell_id",
            how="left",
        )
        fold_panel = fold_panel.join(
            weather_df.select(["node_id", "date", "fwi"]),
            on=["node_id", "date"],
            how="left",
        ).with_columns(pl.col("fwi").fill_null(0.0))

        # Positivos y negativos de la temporada
        pos_df = fold_panel.filter(pl.col("is_ignition_conaf") == 1)
        neg_df = fold_panel.filter(pl.col("is_ignition_conaf") == 0)

        n_pos = pos_df.height
        target_negs_season = n_pos * neg_to_pos_ratio

        # Agregar mes y score de riesgo compuesto
        neg_df = neg_df.with_columns(
            pl.col("date").dt.month().alias("month"),
            (pl.col("fwi") * pl.col("prior_density")).alias("_risk_score"),
        )

        # Muestreo estratificado por (temporada, mes, estrato_de_riesgo)
        neg_parts: list[pl.DataFrame] = []
        months_list = sorted(neg_df.get_column("month").unique().to_list())
        pop_neg_season = neg_df.height

        for m_val in months_list:
            m_sub = neg_df.filter(pl.col("month") == m_val)
            m_pop = m_sub.height
            if m_pop == 0:
                continue

            quota_month = max(1, int(round(target_negs_season * (m_pop / pop_neg_season))))
            quota_month = min(quota_month, m_pop)

            # Umbral del estrato de alto riesgo (percentil 80 dentro del mes)
            risk_arr = m_sub.get_column("_risk_score").to_numpy()
            tau = float(np.quantile(risk_arr, hard_negative_pct))

            hard_candidates = m_sub.filter(pl.col("_risk_score") >= tau)
            regular_candidates = m_sub.filter(pl.col("_risk_score") < tau)

            # Cuota de 50% hard negatives y 50% regulares
            q_hard = min(quota_month // 2, hard_candidates.height)
            q_reg = min(quota_month - q_hard, regular_candidates.height)

            # MUESTREO ALEATORIO DENTRO DE CADA ESTRATO DE RIESGO (D4)
            stratum_seed_h = int((seed * 1000 + int(m_val) * 37 + 1) % (2**31 - 1))
            rng_h = np.random.default_rng(stratum_seed_h)
            if q_hard > 0:
                idx_h = rng_h.choice(hard_candidates.height, size=q_hard, replace=False)
                sample_h = hard_candidates[idx_h].with_columns(
                    pl.lit(True).alias("is_hard_negative"),
                    pl.lit(float(hard_candidates.height) / float(q_hard), dtype=pl.Float32).alias("sample_weight"),
                )
                neg_parts.append(sample_h)

            stratum_seed_r = int((seed * 1000 + int(m_val) * 37 + 2) % (2**31 - 1))
            rng_r = np.random.default_rng(stratum_seed_r)
            if q_reg > 0:
                idx_r = rng_r.choice(regular_candidates.height, size=q_reg, replace=False)
                sample_r = regular_candidates[idx_r].with_columns(
                    pl.lit(False).alias("is_hard_negative"),
                    pl.lit(float(regular_candidates.height) / float(q_reg), dtype=pl.Float32).alias("sample_weight"),
                )
                neg_parts.append(sample_r)

        sampled_negs = pl.concat(neg_parts, how="vertical")

        # Positivos con peso 1.0
        pos_df = pos_df.with_columns(
            pl.lit(1.0, dtype=pl.Float32).alias("sample_weight"),
            pl.lit(False).alias("is_hard_negative"),
            pl.col("date").dt.month().alias("month"),
            pl.lit(0.0, dtype=pl.Float64).alias("_risk_score"),
        )

        fold_sample = pl.concat(
            [
                pos_df.select(["cell_id", "date", "season", "is_ignition_conaf", "superficie_ha_max", "sample_weight", "is_hard_negative"]),
                sampled_negs.select(["cell_id", "date", "season", "is_ignition_conaf", "superficie_ha_max", "sample_weight", "is_hard_negative"]),
            ],
            how="vertical",
        )
        sampled_dfs.append(fold_sample)

    combined_sample = pl.concat(sampled_dfs, how="vertical")

    # Unir todas las covariables:
    # 1. Estáticas consolidadas (DEM, Pop, OSM, WorldCover)
    combined_sample = combined_sample.join(static_df, on="cell_id", how="left")

    # 2. Prior espacial por temporada
    combined_sample = combined_sample.join(prior_df, on=["cell_id", "season"], how="left")

    # 3. Meteorología extendida vía nodo
    combined_sample = combined_sample.join(cell_node_df, on="cell_id", how="left")
    combined_sample = combined_sample.join(weather_df, on=["node_id", "date"], how="left")

    # Formatear columnas y tipos float32, ordenamiento determinista
    float64_cols = [c for c, dtype in zip(combined_sample.columns, combined_sample.dtypes, strict=False) if dtype == pl.Float64]
    combined_sample = combined_sample.with_columns(
        [pl.col(c).cast(pl.Float32) for c in float64_cols]
    ).sort(["season", "date", "cell_id"])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_sample.write_parquet(out_path)
        log_event(
            logger,
            "training_sample_saved",
            path=str(out_path),
            rows=combined_sample.height,
            positives=int(combined_sample.filter(pl.col("is_ignition_conaf") == 1).height),
            negatives=int(combined_sample.filter(pl.col("is_ignition_conaf") == 0).height),
            cols=len(combined_sample.columns),
            size_mb=round(out_path.stat().st_size / (1024 * 1024), 2),
            elapsed_s=round(time.time() - t0, 2),
        )

    return combined_sample


if __name__ == "__main__":
    cfg = load_config("configs/config.yaml")
    df = build_stratified_training_sample(cfg, neg_to_pos_ratio=20, seed=42)
    weights = df.get_column("sample_weight").to_numpy()
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights**2))
    n_eff = (sum_w**2) / sum_w2

    print("=== D4: METRICAS DEL MUESTREO ESTRATIFICADO ===")
    print(f"Filas totales : {df.height:,} (Positivos: {df.filter(pl.col('is_ignition_conaf') == 1).height:,}, Negativos: {df.filter(pl.col('is_ignition_conaf') == 0).height:,})")
    print(f"Hard negatives: {df.filter(pl.col('is_hard_negative')).height:,}")
    print(f"Pesos min     : {float(np.min(weights)):.2f}")
    print(f"Pesos mediana : {float(np.median(weights)):.2f}")
    print(f"Pesos max     : {float(np.max(weights)):.2f}")
    print(f"Suma de pesos : {sum_w:,.1f} (Poblacion real: 33,345,071)")
    print(f"Tamaño efectivo (N_eff): {n_eff:,.1f}")
    print(f"Columnas ({len(df.columns)}): {df.columns}")
