"""Clustering espacio-temporal DBSCAN, clasificación `event_class`, ventana T6 y huella T8.

Cambios de Fase 1.5 (Regla 3 y Regla 5):
- T2 (Eliminación de `deduplicate_inter_satellite_passes`):
  - PORQUÉ: La función anterior no distinguía satélites y fusionaba el 67% de los focos
    (37.622 -> 12.278) suprimiendo píxeles vecinos legítimos de una misma barrida orbital.
    Dado que la etiqueta de ignición toma únicamente la primera detección del cluster,
    adelgazar píxeles antes del clustering distorsionaba `n_pixels`, `frp_sum` y `extent_m`.
  - CÓMO: Se elimina `deduplicate_inter_satellite_passes` y se conserva `n_pixels` íntegro.
- T5 (`event_class` por cluster):
  - PORQUÉ: En otoño (abr-may) ocurren miles de quemas agrícolas controladas en Ñuble que no
    deben etiquetarse como incendios forestales (`is_ignition=1`) ni como no-igniciones (`0`).
  - CÓMO: Se calculan por cluster `n_pixels`, `n_passes`, `frp_max`, `frp_sum`,
    `duration_hours` y `extent_m`, y se evalúan reglas físicas candidatas (`rule_a`, `rule_b`,
    `rule_c`) para asignar `event_class` (`'wildfire_candidate'` vs `'small_or_agri'`).
- T6 (Ventana meteorológica antecedente):
  - PORQUÉ: Una primera detección nocturna (00-04h local) reflejaría fuga del futuro si se
    cruzara con el clima vespertino del mismo día calendario.
  - CÓMO: Se agregan `detected_night` (booleano `00-04h` local en `America/Santiago`) y
    `window_start_utc = first_detection_ts - 24h`.
- T8 (Exclusión de huella activa para negativos):
  - PORQUÉ: Las celdas vecinas a un incendio en curso (extensión + 2,5 km, desde la primera
    detección hasta la última + 72 h) no son negativos independientes.
  - CÓMO: `build_excluded_cell_days` consulta mediante un `cKDTree` 2D todas las celdas de
    Ñuble dentro de `buffer_m = 2500 m` de cada cluster y expande su rango diario hasta
    `last_detection_ts + 72 h` usando `pl.date_ranges` vectorizado.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from chilefuego.config import ClusteringConfig
from chilefuego.schemas import validate_excluded_cell_days, validate_ignition_labels


def _partition_chained_clusters(
    xs_m: np.ndarray,
    ys_m: np.ndarray,
    ts_hours: np.ndarray,
    base_labels: np.ndarray,
    max_duration_hours: float | None,
    max_radius_m: float | None,
) -> np.ndarray:
    """Subdivide cronológicamente clusters que exceden el diámetro o duración máxima."""
    if max_duration_hours is None and max_radius_m is None:
        return base_labels

    new_labels = np.full_like(base_labels, fill_value=-1)
    next_cluster_id = 0

    for cid in np.unique(base_labels):
        if cid < 0:
            continue
        idx = np.flatnonzero(base_labels == cid)
        if idx.size == 1:
            new_labels[idx[0]] = next_cluster_id
            next_cluster_id += 1
            continue

        c_xs = xs_m[idx]
        c_ys = ys_m[idx]
        c_ts = ts_hours[idx]

        dt_from_origin = c_ts - c_ts[0]
        dr_from_origin = np.hypot(c_xs - c_xs[0], c_ys - c_ys[0])
        exceeds_t = max_duration_hours is not None and np.any(dt_from_origin > max_duration_hours)
        exceeds_r = max_radius_m is not None and np.any(dr_from_origin > max_radius_m)

        if not exceeds_t and not exceeds_r:
            new_labels[idx] = next_cluster_id
            next_cluster_id += 1
            continue

        anchors_x: list[float] = []
        anchors_y: list[float] = []
        anchors_t: list[float] = []
        anchor_cids: list[int] = []

        for local_i, global_i in enumerate(idx):
            px = float(c_xs[local_i])
            py = float(c_ys[local_i])
            pt = float(c_ts[local_i])

            assigned_cid = -1
            for a_idx in range(len(anchor_cids)):
                dt_a = pt - anchors_t[a_idx]
                if max_duration_hours is not None and dt_a > max_duration_hours:
                    continue
                dr_a = float(np.hypot(px - anchors_x[a_idx], py - anchors_y[a_idx]))
                if max_radius_m is not None and dr_a > max_radius_m:
                    continue
                assigned_cid = anchor_cids[a_idx]
                break

            if assigned_cid == -1:
                assigned_cid = next_cluster_id
                next_cluster_id += 1
                anchors_x.append(px)
                anchors_y.append(py)
                anchors_t.append(pt)
                anchor_cids.append(assigned_cid)

            new_labels[global_i] = assigned_cid

    return new_labels


def cluster_spatiotemporal_detections(
    df: pl.DataFrame,
    clustering_cfg: ClusteringConfig,
    apply_chaining_bounds: bool = True,
) -> pl.DataFrame:
    """Agrupa detecciones térmicas usando DBSCAN con métrica cilíndrica espacio-temporal exacta."""
    if df.is_empty():
        return df.with_columns(pl.Series("cluster_id", [], dtype=pl.Int64))

    if df.height == 1:
        return df.with_columns(pl.lit(0, dtype=pl.Int64).alias("cluster_id"))

    sorted_df = df.sort(["acq_ts_utc", "cell_id", "satellite", "latitude", "longitude"])
    n_pts = sorted_df.height

    xs_m = sorted_df.get_column("x_utm").to_numpy()
    ys_m = sorted_df.get_column("y_utm").to_numpy()
    xs = xs_m / clustering_cfg.spatial_eps_m
    ys = ys_m / clustering_cfg.spatial_eps_m

    ts_us = sorted_df.get_column("acq_ts_utc").dt.epoch("us").to_numpy()
    ts_hours = (ts_us - ts_us.min()) / (3600.0 * 1_000_000.0)
    ts_scaled = ts_hours / clustering_cfg.temporal_eps_hours

    coords_3d = np.column_stack([xs, ys, ts_scaled])
    tree = cKDTree(coords_3d)
    pairs = tree.query_pairs(r=1.0, p=np.inf, output_type="ndarray")

    if pairs.size > 0:
        dx = xs[pairs[:, 0]] - xs[pairs[:, 1]]
        dy = ys[pairs[:, 0]] - ys[pairs[:, 1]]
        dt = np.abs(ts_scaled[pairs[:, 0]] - ts_scaled[pairs[:, 1]])
        spatial_d = np.hypot(dx, dy)
        valid_mask = (spatial_d <= 1.0) & (dt <= 1.0)
        valid_pairs = pairs[valid_mask]
        valid_dists = np.maximum(spatial_d[valid_mask], dt[valid_mask])
    else:
        valid_pairs = np.empty((0, 2), dtype=np.int64)
        valid_dists = np.empty(0, dtype=np.float64)

    if clustering_cfg.min_samples == 1:
        if valid_pairs.size == 0:
            raw_labels = np.arange(n_pts, dtype=np.int64)
        else:
            row_ind = np.concatenate([valid_pairs[:, 0], valid_pairs[:, 1]])
            col_ind = np.concatenate([valid_pairs[:, 1], valid_pairs[:, 0]])
            data = np.ones(row_ind.size, dtype=np.int8)
            adj = coo_matrix((data, (row_ind, col_ind)), shape=(n_pts, n_pts)).tocsr()
            _n_comp, comp_labels = connected_components(adj, directed=False)
            _, first_idx = np.unique(comp_labels, return_index=True)
            order = np.argsort(first_idx)
            remap = np.empty_like(order)
            remap[order] = np.arange(order.size)
            raw_labels = remap[comp_labels].astype(np.int64)
    else:
        row_ind = np.concatenate([valid_pairs[:, 0], valid_pairs[:, 1], np.arange(n_pts)])
        col_ind = np.concatenate([valid_pairs[:, 1], valid_pairs[:, 0], np.arange(n_pts)])
        dist_vals = np.concatenate([valid_dists, valid_dists, np.zeros(n_pts, dtype=np.float64)])
        sparse_dist = coo_matrix((dist_vals, (row_ind, col_ind)), shape=(n_pts, n_pts)).tocsr()
        dbscan = DBSCAN(
            eps=1.0,
            min_samples=clustering_cfg.min_samples,
            metric="precomputed",
        )
        raw_labels = dbscan.fit_predict(sparse_dist).astype(np.int64)

    if apply_chaining_bounds:
        cluster_labels = _partition_chained_clusters(
            xs_m=xs_m,
            ys_m=ys_m,
            ts_hours=ts_hours,
            base_labels=raw_labels,
            max_duration_hours=clustering_cfg.max_cluster_duration_hours,
            max_radius_m=clustering_cfg.max_cluster_radius_m,
        )
    else:
        cluster_labels = raw_labels

    return sorted_df.with_columns(pl.Series("cluster_id", cluster_labels))


def classify_cluster_event_expr(
    rule_name: Literal["rule_a", "rule_b", "rule_c"] = "rule_b",
) -> pl.Expr:
    """Retorna la expresión Polars para clasificar `event_class` según la regla física (T5).

    - `rule_a` (Sensible / Inclusiva):
      `n_passes >= 2` O `n_pixels >= 3` O `frp_max >= 20.0`
    - `rule_b` (Intermedia / Energética-Temporal):
      `(n_passes >= 2 Y duration_hours >= 6.0)` O `frp_max >= 30.0` O `frp_sum >= 60.0`
    - `rule_c` (Estricta / Incendio Sostenido):
      `(n_passes >= 3 Y duration_hours >= 12.0)` O `frp_max >= 50.0` O `frp_sum >= 100.0`
    """
    if rule_name == "rule_a":
        cond = (pl.col("n_passes") >= 2) | (pl.col("n_pixels") >= 3) | (pl.col("frp_max") >= 20.0)
    elif rule_name == "rule_c":
        cond = (
            ((pl.col("n_passes") >= 3) & (pl.col("duration_hours") >= 12.0))
            | (pl.col("frp_max") >= 50.0)
            | (pl.col("frp_sum") >= 100.0)
            | (pl.col("n_pixels") >= 6)
        )
    else:
        cond = (
            ((pl.col("n_passes") >= 2) & (pl.col("duration_hours") >= 6.0))
            | (pl.col("frp_max") >= 30.0)
            | (pl.col("frp_sum") >= 60.0)
        )

    return pl.when(cond).then(pl.lit("wildfire_candidate")).otherwise(pl.lit("small_or_agri"))


def summarize_clusters(
    clustered_df: pl.DataFrame,
    rule_name: Literal["rule_a", "rule_b", "rule_c"] = "rule_b",
    chile_tz: str = "America/Santiago",
) -> pl.DataFrame:
    """Calcula métricas físicas por cluster (`T5`) y variables de primera detección (`T6`)."""
    valid_clusters_df = (
        clustered_df.filter(pl.col("cluster_id") >= 0)
        if "cluster_id" in clustered_df.columns and not clustered_df.is_empty()
        else pl.DataFrame()
    )

    if valid_clusters_df.is_empty():
        return pl.DataFrame(
            schema={
                "cluster_id": pl.Int64,
                "cell_id": pl.String,
                "date": pl.Date,
                "n_detections": pl.Int64,
                "n_pixels": pl.Int64,
                "n_passes": pl.Int64,
                "frp_max": pl.Float64,
                "frp_sum": pl.Float64,
                "duration_hours": pl.Float64,
                "extent_m": pl.Float64,
                "event_class": pl.String,
                "first_detection_ts": pl.Datetime("us", "UTC"),
                "detected_night": pl.Boolean,
                "window_start_utc": pl.Datetime("us", "UTC"),
            }
        )

    ordered = valid_clusters_df.sort(
        by=["cluster_id", "acq_ts_utc", "frp", "cell_id"],
        descending=[False, False, True, False],
    )

    summary = (
        ordered.group_by("cluster_id", maintain_order=True)
        .agg(
            pl.col("cell_id").first().alias("cell_id"),
            pl.col("local_date").first().alias("date"),
            pl.len().cast(pl.Int64).alias("n_detections"),
            pl.len().cast(pl.Int64).alias("n_pixels"),
            pl.col("acq_ts_utc").n_unique().cast(pl.Int64).alias("n_passes"),
            pl.col("frp").max().cast(pl.Float64).alias("frp_max"),
            pl.col("frp").sum().cast(pl.Float64).alias("frp_sum"),
            ((pl.col("acq_ts_utc").max() - pl.col("acq_ts_utc").min()).dt.total_seconds() / 3600.0)
            .cast(pl.Float64)
            .alias("duration_hours"),
            (
                (
                    (pl.col("x_utm").max() - pl.col("x_utm").min()) ** 2
                    + (pl.col("y_utm").max() - pl.col("y_utm").min()) ** 2
                ).sqrt()
            )
            .cast(pl.Float64)
            .alias("extent_m"),
            pl.col("acq_ts_utc").first().alias("first_detection_ts"),
        )
        .with_columns(
            classify_cluster_event_expr(rule_name).alias("event_class"),
            pl.col("first_detection_ts")
            .dt.convert_time_zone(chile_tz)
            .dt.hour()
            .is_between(0, 4)
            .alias("detected_night"),
            (pl.col("first_detection_ts") - pl.duration(hours=24)).alias("window_start_utc"),
        )
    )
    return summary


def extract_ignitions_from_clusters(
    clustered_df: pl.DataFrame,
    rule_name: Literal["rule_a", "rule_b", "rule_c"] = "rule_b",
    chile_tz: str = "America/Santiago",
) -> pl.DataFrame:
    """Extrae la primera detección de cada cluster válido con `event_class` y ventana T6."""
    summary = summarize_clusters(clustered_df, rule_name=rule_name, chile_tz=chile_tz)
    if summary.is_empty():
        empty_labels = pl.DataFrame(
            schema={
                "cell_id": pl.String,
                "date": pl.Date,
                "is_ignition": pl.Int64,
                "cluster_id": pl.Int64,
                "n_detections": pl.Int64,
                "n_pixels": pl.Int64,
                "n_passes": pl.Int64,
                "frp_max": pl.Float64,
                "frp_sum": pl.Float64,
                "duration_hours": pl.Float64,
                "extent_m": pl.Float64,
                "event_class": pl.String,
                "first_detection_ts": pl.Datetime("us", "UTC"),
                "detected_night": pl.Boolean,
                "window_start_utc": pl.Datetime("us", "UTC"),
            }
        )
        return validate_ignition_labels(empty_labels)

    # Si dos clusters iniciaran en la misma celda y fecha local, prioriza wildfire_candidate
    # y luego el menor timestamp de detección.
    ordered_ign = summary.with_columns(
        (pl.col("event_class") == "wildfire_candidate").alias("_is_wf")
    ).sort(
        by=["cell_id", "date", "_is_wf", "first_detection_ts"],
        descending=[False, False, True, False],
    )

    dedup_cell_day = (
        ordered_ign.group_by(["cell_id", "date"], maintain_order=True)
        .agg(
            pl.lit(1, dtype=pl.Int64).alias("is_ignition"),
            pl.col("cluster_id").first().alias("cluster_id"),
            pl.col("n_detections").sum().cast(pl.Int64).alias("n_detections"),
            pl.col("n_pixels").sum().cast(pl.Int64).alias("n_pixels"),
            pl.col("n_passes").max().cast(pl.Int64).alias("n_passes"),
            pl.col("frp_max").max().cast(pl.Float64).alias("frp_max"),
            pl.col("frp_sum").sum().cast(pl.Float64).alias("frp_sum"),
            pl.col("duration_hours").max().cast(pl.Float64).alias("duration_hours"),
            pl.col("extent_m").max().cast(pl.Float64).alias("extent_m"),
            pl.col("event_class").first().alias("event_class"),
            pl.col("first_detection_ts").first().alias("first_detection_ts"),
            pl.col("detected_night").first().alias("detected_night"),
            pl.col("window_start_utc").first().alias("window_start_utc"),
        )
        .select(
            [
                "cell_id",
                "date",
                "is_ignition",
                "cluster_id",
                "n_detections",
                "n_pixels",
                "n_passes",
                "frp_max",
                "frp_sum",
                "duration_hours",
                "extent_m",
                "event_class",
                "first_detection_ts",
                "detected_night",
                "window_start_utc",
            ]
        )
        .sort(["date", "cell_id"])
    )

    return validate_ignition_labels(dedup_cell_day)


def build_excluded_cell_days(
    clustered_df: pl.DataFrame,
    grid_df: pl.DataFrame,
    buffer_m: float = 2500.0,
    post_hours: float = 72.0,
    chile_tz: str = "America/Santiago",
) -> pl.DataFrame:
    """Genera el conjunto de `celdas × día` dentro de la huella de cada cluster (T8).

    Para cada cluster válido (`cluster_id >= 0`):
    - Extensión espacial: todas las celdas dentro de Ñuble (`in_nuble == True`) cuyo centroide
      esté a `<= buffer_m` (2,5 km) de cualquier detección del cluster.
    - Extensión temporal: todos los días calendario locales desde la primera detección del
      cluster hasta `(última detección + post_hours)` (72 h).
    """
    valid_df = (
        clustered_df.filter(pl.col("cluster_id") >= 0)
        if "cluster_id" in clustered_df.columns and not clustered_df.is_empty()
        else pl.DataFrame()
    )
    nuble_grid = grid_df.filter(pl.col("in_nuble"))

    if valid_df.is_empty() or nuble_grid.is_empty():
        empty_excl = pl.DataFrame(
            schema={
                "cell_id": pl.String,
                "date": pl.Date,
                "cluster_id": pl.Int64,
                "reason": pl.String,
            }
        )
        return validate_excluded_cell_days(empty_excl)

    grid_xy = np.column_stack(
        [
            nuble_grid.get_column("centroid_x").to_numpy(),
            nuble_grid.get_column("centroid_y").to_numpy(),
        ]
    )
    grid_cell_ids = nuble_grid.get_column("cell_id").to_list()
    grid_tree = cKDTree(grid_xy)

    # Deduplicar por (cluster_id, cell_id) para minimizar consultas espaciales al KD-Tree
    unique_cluster_cells = valid_df.select(["cluster_id", "cell_id", "x_utm", "y_utm"]).unique(
        subset=["cluster_id", "cell_id"]
    )
    uc_xy = np.column_stack(
        [
            unique_cluster_cells.get_column("x_utm").to_numpy(),
            unique_cluster_cells.get_column("y_utm").to_numpy(),
        ]
    )
    uc_cids = unique_cluster_cells.get_column("cluster_id").to_list()
    neighbors_per_pt = grid_tree.query_ball_point(uc_xy, r=float(buffer_m))

    cluster_to_cells: dict[int, set[str]] = {}
    for cid_val, nbr_indices in zip(uc_cids, neighbors_per_pt, strict=False):
        cid_int = int(cid_val)
        cell_names = {grid_cell_ids[i] for i in nbr_indices}
        if cid_int not in cluster_to_cells:
            cluster_to_cells[cid_int] = cell_names
        else:
            cluster_to_cells[cid_int].update(cell_names)

    # Asegurar que la propia celda de cada detección esté incluida aunque buffer_m sea pequeño
    for cid_val, cname in zip(
        unique_cluster_cells.get_column("cluster_id").to_list(),
        unique_cluster_cells.get_column("cell_id").to_list(),
        strict=False,
    ):
        cluster_to_cells.setdefault(int(cid_val), set()).add(str(cname))

    cluster_cells_df = pl.DataFrame(
        {
            "cluster_id": list(cluster_to_cells.keys()),
            "cell_id": [sorted(cells) for cells in cluster_to_cells.values()],
        },
        schema={"cluster_id": pl.Int64, "cell_id": pl.List(pl.String)},
    ).explode("cell_id")

    post_minutes = int(round(post_hours * 60.0))
    cluster_dates_df = (
        valid_df.with_columns(
            (pl.col("acq_ts_utc") + pl.duration(minutes=post_minutes))
            .dt.convert_time_zone(chile_tz)
            .dt.date()
            .alias("_end_local_date")
        )
        .group_by("cluster_id")
        .agg(
            pl.col("local_date").min().alias("_start_date"),
            pl.col("_end_local_date").max().alias("_end_date"),
        )
        .with_columns(pl.date_ranges("_start_date", "_end_date", interval="1d").alias("date"))
        .select(["cluster_id", "date"])
        .explode("date")
    )

    excluded = (
        cluster_cells_df.join(cluster_dates_df, on="cluster_id", how="inner")
        .group_by(["cell_id", "date"])
        .agg(pl.col("cluster_id").min().cast(pl.Int64).alias("cluster_id"))
        .with_columns(pl.lit("active_cluster_footprint", dtype=pl.String).alias("reason"))
        .select(["cell_id", "date", "cluster_id", "reason"])
        .sort(["date", "cell_id"])
    )

    return validate_excluded_cell_days(excluded)
