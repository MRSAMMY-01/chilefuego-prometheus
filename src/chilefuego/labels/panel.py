"""Construcción del panel diario CONAF de Fase 2A (F2.2) sobre la rejilla de 1 km (`in_nuble`).

PORQUÉ (Regla 5):
- En la Fase 1.9 se demostró que los 3.579 incendios de las 8 temporadas CONAF sin fecha
  (`2010_2011`..`2016_2017` y `2018_2019`) no pueden asignarse a ningún día calendario sin
  inventar etiquetas ni contaminar los negativos de esas temporadas. Por ello, el dominio temporal
  de entrenamiento supervisado de la Fase 2A se restringe exclusivamente a las **7 temporadas con
  fecha válida**: `2017_18` (`2017-07-01` a `2018-06-30`) y `2019_20` a `2024_25` (`2019-07-01` a
  `2025-06-30`), excluyendo por completo `2018_19` (`2018-07-01` a `2019-06-30`) y todo lo anterior
  a `2017-07-01`.
- El target principal es `is_ignition_conaf` (`date_ok == True` y `dup_removed == False`), colapsado
  a nivel `(cell_id, date)` sobre las `13.108` celdas de 1 km dentro de Ñuble (`in_nuble == True`).
- Para evitar falsos negativos en el conjunto de entrenamiento (`is_ignition_conaf == 0`), se
  construye la columna `exclusion_reason` con tres motivos explícitos:
  1) `undated_season`: cualquier celda-día perteneciente a la temporada sin fecha `2018_19`
     (`2018-07-01` a `2019-06-30`) dentro del intervalo `2017-07-01..2025-06-30` (las cuales se
     excluyen por completo del panel de las 7 temporadas fechadas).
  2) `conaf_active_footprint`: celdas-día en la vecindad espacial (`<= 2,5 km`) de un incendio
     CONAF activo desde `date_local` hasta `extincion_date_local` (acotada a `[date_local,
     date_local + 30 días]`, o `date_local + 3 días` cuando `extincion_date_local` es nula o
     anterior a `date_local`), excluyendo la propia celda-día de ignición (`is_ignition_conaf == 1`).
     La columna `fecha_extincion` se utiliza **EXCLUSIVAMENTE** para esta máscara y jamás como
     predictor.
  3) `firms_unmatched_summer_rule_b`: celdas-día en la huella activa (`<= 2,5 km`, `+72 h`) de
     clusters satelitales S-NPP `rule_b` (`event_class == 'wildfire_candidate'`) ocurridos en verano
     (`diciembre–marzo`) de las 7 temporadas fechadas que **no** tienen una ignición CONAF a
     `<= 3 km` y `±2 días`.
- Las detecciones `small_or_agri` **NO se excluyen** cuando el target es `is_ignition_conaf` (F2.2).

CÓMO:
- `season_from_date_expr()` asigna cada día calendario a su temporada hidrológica/forestal chilena
  (`julio` del año $Y$ a `junio` del año $Y+1$, p.ej. `2017-07-01..2018-06-30` -> `2017_18`).
- `build_conaf_active_footprint_days()` usa `cKDTree` en `EPSG:32719` (`r = 2500 m`) sobre las
  igniciones CONAF válidas (`date_ok & ~dup_removed`) para expandir los días activos hasta
  `extincion_date_local` (o `+3 días` si no hay fecha de extinción válida).
- `build_firms_unmatched_summer_rule_b_days()` identifica los clusters S-NPP `wildfire_candidate`
  (`rule_b`) en verano (`mes in {12, 1, 2, 3}`) de temporadas fechadas sin match CONAF a `<= 3 km`
  y `±2 días` y extrae sus celdas-día de `excluded_cell_days.parquet`.
- `build_phase2a_daily_panel()` ensambla el panel de las 7 temporadas fechadas, colapsa
  `is_ignition_conaf` por `(cell_id, date)`, asigna `exclusion_reason` y calcula las tablas de
  auditoría F2.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

from chilefuego.config import AppConfig, load_config
from chilefuego.logging_utils import atomic_write_parquet, get_logger, log_event

logger = get_logger(__name__)

DATED_SEASONS: tuple[str, ...] = (
    "2017_18",
    "2019_20",
    "2020_21",
    "2021_22",
    "2022_23",
    "2023_24",
    "2024_25",
)

UNDATED_GAP_SEASON = "2018_19"

CONAF_SEASON_NORMALIZATION: dict[str, str] = {
    "if_temporada_2010_2011": "2010_11",
    "if_temporada_2011_2012": "2011_12",
    "if_temporada_2012_2013": "2012_13",
    "if_temporada_2013_2014": "2013_14",
    "if_temporada_2014_2015": "2014_15",
    "if_temporada_2015_2016": "2015_16",
    "if_temporada_2016_2017": "2016_17",
    "if_temporada_2017_2018": "2017_18",
    "if_temporada_2018_2019": "2018_19",
    "if_temporada_2019_2020": "2019_20",
    "if_temporada_2020_2021": "2020_21",
    "if_temporada_2021_2022": "2021_22",
    "if_temporada_2022_2023": "2022_23",
    "if_temporada_2023_2024": "2023_24",
    "if_temporada_2024_2025": "2024_25",
}


@dataclass(frozen=True)
class SeasonPositiveRow:
    """Resumen por temporada de igniciones CONAF y celdas-día positivas colapsadas (F2.2)."""

    season: str
    n_days: int
    n_total_cell_days: int
    n_conaf_records_valid: int
    n_positive_cell_days: int
    n_positive_ge_1ha: int
    n_positive_ge_10ha: int
    n_excluded_conaf_footprint: int
    n_excluded_firms_unmatched_summer: int
    n_effective_cell_days: int
    base_rate_effective: float


@dataclass(frozen=True)
class Phase2APanelSummary:
    """Métricas globales auditables del panel diario de Fase 2A (F2.2)."""

    n_cells_in_nuble: int
    n_dated_seasons: int
    n_dated_days: int
    n_undated_2018_19_days: int
    excluded_undated_season_cell_days: int
    total_dated_cell_days: int
    excluded_conaf_active_footprint: int
    excluded_firms_unmatched_summer_rule_b: int
    excluded_small_or_agri: int
    total_excluded_in_dated_panel: int
    effective_training_cell_days: int
    total_conaf_valid_records_in_domain: int
    total_positive_cell_days: int
    total_positive_ge_1ha: int
    total_positive_ge_10ha: int
    base_rate_raw_dated: float
    base_rate_effective: float
    season_rows: list[SeasonPositiveRow]
    panel_sha256: str


def season_from_date_expr(date_col: str = "date") -> pl.Expr:
    """Asigna cada fecha calendario a su temporada forestal chilena (`julio Y` a `junio Y+1`)."""
    yr = pl.col(date_col).dt.year()
    mo = pl.col(date_col).dt.month()
    start_yr = pl.when(mo >= 7).then(yr).otherwise(yr - 1)
    end_yr_short = (start_yr + 1) % 100
    return pl.format(
        "{}_{}",
        start_yr,
        end_yr_short.cast(pl.String).str.pad_start(2, "0"),
    )


def build_dated_calendar_df(
    start_date: date = date(2017, 7, 1),
    end_date: date = date(2025, 6, 30),
    include_undated_2018_19: bool = False,
) -> pl.DataFrame:
    """Construye el calendario diario entre `2017-07-01` y `2025-06-30` con su `season`."""
    cal = pl.DataFrame(
        {"date": pl.date_range(start_date, end_date, interval="1d", eager=True)}
    ).with_columns(season_from_date_expr("date").alias("season"))
    if not include_undated_2018_19:
        cal = cal.filter(pl.col("season").is_in(DATED_SEASONS))
    return cal


def collapse_conaf_ignitions_by_cell_day(
    ignitions_conaf_df: pl.DataFrame,
    valid_cell_ids: set[str] | None = None,
    start_date: date = date(2017, 7, 1),
    end_date: date = date(2025, 6, 30),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filtra igniciones CONAF (`date_ok & ~dup_removed`) en las 7 temporadas fechadas y colapsa por `(cell_id, date)`.

    Retorna:
      1) `valid_records_df`: registros individuales CONAF válidos dentro del dominio de las 7 temporadas.
      2) `collapsed_df`: tabla colapsada a nivel `(cell_id, date)` con `is_ignition_conaf = 1`,
         `n_conaf_fires`, `superficie_ha_max`, `superficie_ha_sum` y `season`.
    """
    qf = ignitions_conaf_df.get_column("quality_flags")
    valid = (
        ignitions_conaf_df.with_columns(
            qf.struct.field("crs_ok").alias("_crs_ok"),
            qf.struct.field("date_ok").alias("_date_ok"),
            qf.struct.field("dup_removed").alias("_dup_removed"),
        )
        .filter(
            pl.col("_crs_ok")
            & pl.col("_date_ok")
            & (~pl.col("_dup_removed"))
            & pl.col("date_local").is_not_null()
            & (pl.col("date_local") >= start_date)
            & (pl.col("date_local") <= end_date)
        )
        .with_columns(
            pl.col("date_local").alias("date"),
            season_from_date_expr("date_local").alias("calendar_season"),
        )
        .filter(pl.col("calendar_season").is_in(DATED_SEASONS))
    )

    if valid_cell_ids is not None:
        valid = valid.filter(pl.col("cell_id").is_in(valid_cell_ids))

    collapsed = (
        valid.group_by(["cell_id", "date"])
        .agg(
            pl.lit(1, dtype=pl.Int8).alias("is_ignition_conaf"),
            pl.len().cast(pl.Int32).alias("n_conaf_fires"),
            pl.col("superficie_ha").max().cast(pl.Float64).alias("superficie_ha_max"),
            pl.col("superficie_ha").sum().cast(pl.Float64).alias("superficie_ha_sum"),
            pl.col("calendar_season").first().alias("season"),
        )
        .sort(["date", "cell_id"])
    )
    return valid, collapsed


def build_conaf_active_footprint_days(
    valid_conaf_df: pl.DataFrame,
    grid_df: pl.DataFrame,
    buffer_m: float = 2500.0,
    default_post_days: int = 3,
    max_duration_days: int = 30,
) -> pl.DataFrame:
    """Construye la máscara `conaf_active_footprint` (F2.2) usando `extincion_date_local` solo aquí.

    Para cada ignición CONAF válida (`date_ok & ~dup_removed`):
    - Vecindad espacial: todas las celdas `in_nuble` cuyo centroide dista `<= buffer_m` (2,5 km)
      de `(x_utm, y_utm)` (incluyendo siempre la propia `cell_id`).
    - Ventana temporal activa: desde `date_local` hasta `end_date_active`, donde:
      * Si `extincion_date_local` es válida y `>= date_local`:
        `min(extincion_date_local, date_local + max_duration_days)`.
      * Si `extincion_date_local` es nula o `< date_local`:
        `date_local + default_post_days` (+72 h).
    """
    nuble_grid = grid_df.filter(pl.col("in_nuble"))
    if valid_conaf_df.is_empty() or nuble_grid.is_empty():
        return pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date})

    grid_xy = np.column_stack(
        [
            nuble_grid.get_column("centroid_x").to_numpy(),
            nuble_grid.get_column("centroid_y").to_numpy(),
        ]
    )
    grid_cell_ids = nuble_grid.get_column("cell_id").to_list()
    grid_tree = cKDTree(grid_xy)

    fire_xy = np.column_stack(
        [
            valid_conaf_df.get_column("x_utm").to_numpy(),
            valid_conaf_df.get_column("y_utm").to_numpy(),
        ]
    )
    neighbors_list = grid_tree.query_ball_point(fire_xy, r=float(buffer_m))

    fire_cells = valid_conaf_df.get_column("cell_id").to_list()
    start_dates = valid_conaf_df.get_column("date_local").to_list()
    ext_dates = (
        valid_conaf_df.get_column("extincion_date_local").to_list()
        if "extincion_date_local" in valid_conaf_df.columns
        else [None] * valid_conaf_df.height
    )

    fire_ids: list[int] = []
    cell_lists: list[list[str]] = []
    d_starts: list[date] = []
    d_ends: list[date] = []

    for idx, (c_self, d_ini, d_ext, nbrs) in enumerate(
        zip(fire_cells, start_dates, ext_dates, neighbors_list, strict=False)
    ):
        if d_ini is None:
            continue
        c_set = {grid_cell_ids[i] for i in nbrs}
        c_set.add(str(c_self))

        if d_ext is not None and d_ext >= d_ini:
            d_end = min(d_ext, d_ini + timedelta(days=max_duration_days))
        else:
            d_end = d_ini + timedelta(days=default_post_days)

        fire_ids.append(idx)
        cell_lists.append(sorted(c_set))
        d_starts.append(d_ini)
        d_ends.append(d_end)

    if not fire_ids:
        return pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date})

    events_df = pl.DataFrame(
        {
            "_fid": fire_ids,
            "cell_id": cell_lists,
            "_d_start": d_starts,
            "_d_end": d_ends,
        },
        schema={
            "_fid": pl.Int64,
            "cell_id": pl.List(pl.String),
            "_d_start": pl.Date,
            "_d_end": pl.Date,
        },
    ).with_columns(pl.date_ranges("_d_start", "_d_end", interval="1d").alias("date"))

    footprint_df = (
        events_df.select(["_fid", "cell_id"])
        .explode("cell_id")
        .join(events_df.select(["_fid", "date"]).explode("date"), on="_fid", how="inner")
        .select(["cell_id", "date"])
        .unique()
        .with_columns(season_from_date_expr("date").alias("_season"))
        .filter(pl.col("_season").is_in(DATED_SEASONS))
        .select(["cell_id", "date"])
        .sort(["date", "cell_id"])
    )
    return footprint_df


def build_firms_unmatched_summer_rule_b_days(
    labels_firms_df: pl.DataFrame,
    excluded_firms_df: pl.DataFrame,
    valid_conaf_df: pl.DataFrame,
    grid_df: pl.DataFrame,
    match_radius_m: float = 3000.0,
    match_days_tol: int = 2,
) -> pl.DataFrame:
    """Identifica las celdas-día de la huella `firms_unmatched_summer_rule_b` en temporadas fechadas.

    Definición (F2.2):
    - Clusters FIRMS S-NPP con `event_class == 'wildfire_candidate'` (`rule_b`).
    - Ocurridos en meses de verano (`diciembre, enero, febrero, marzo`) dentro de las 7 temporadas
      fechadas (`2017_18` y `2019_20`..`2024_25`).
    - Sin ninguna ignición CONAF a `<= 3 km` y `±2 días`.
    - Se retorna la huella de celdas-día (`excluded_cell_days.parquet`) de dichos clusters dentro
      de las 7 temporadas fechadas.
    """
    if labels_firms_df.is_empty() or excluded_firms_df.is_empty():
        return pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date})

    summer_wf = (
        labels_firms_df.with_columns(
            season_from_date_expr("date").alias("season"),
            pl.col("date").dt.month().alias("_month"),
        )
        .filter(
            (pl.col("event_class") == "wildfire_candidate")
            & pl.col("season").is_in(DATED_SEASONS)
            & pl.col("_month").is_in([12, 1, 2, 3])
        )
        .join(
            grid_df.select(["cell_id", "centroid_x", "centroid_y"]),
            on="cell_id",
            how="left",
        )
    )

    if summer_wf.is_empty():
        return pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date})

    if valid_conaf_df.is_empty():
        unmatched_cids = summer_wf.get_column("cluster_id").to_list()
    else:
        c_x = valid_conaf_df.get_column("x_utm").to_numpy()
        c_y = valid_conaf_df.get_column("y_utm").to_numpy()
        c_d = np.array(
            [d.toordinal() for d in valid_conaf_df.get_column("date_local").to_list()],
            dtype=np.int64,
        )

        f_x = summer_wf.get_column("centroid_x").to_numpy()
        f_y = summer_wf.get_column("centroid_y").to_numpy()
        f_d = np.array(
            [d.toordinal() for d in summer_wf.get_column("date").to_list()],
            dtype=np.int64,
        )
        f_cids = summer_wf.get_column("cluster_id").to_list()

        unmatched_cids = []
        for idx, cid in enumerate(f_cids):
            t_mask = np.abs(c_d - f_d[idx]) <= match_days_tol
            if not np.any(t_mask):
                unmatched_cids.append(int(cid))
                continue
            dists = np.hypot(c_x[t_mask] - f_x[idx], c_y[t_mask] - f_y[idx])
            if float(np.min(dists)) > match_radius_m:
                unmatched_cids.append(int(cid))

    if not unmatched_cids:
        return pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date})

    unmatched_footprint = (
        excluded_firms_df.filter(pl.col("cluster_id").is_in(unmatched_cids))
        .select(["cell_id", "date"])
        .unique()
        .with_columns(season_from_date_expr("date").alias("_season"))
        .filter(pl.col("_season").is_in(DATED_SEASONS))
        .select(["cell_id", "date"])
        .sort(["date", "cell_id"])
    )
    return unmatched_footprint


def build_phase2a_daily_panel(
    config: AppConfig,
    persist_sparse_path: Path | None = None,
) -> tuple[pl.DataFrame, Phase2APanelSummary]:
    """Construye y audita el panel diario de Fase 2A (F2.2) sobre las 7 temporadas con fecha.

    Para mantener eficiencia en memoria y disco sin perder ninguna celda-día, calcula el resumen
    exacto sobre el producto cartesiano completo (`13.108 celdas × 2.557 días = 33.517.156`
    celdas-día en las 7 temporadas fechadas, más las `4.784.420` celdas-día excluidas por
    `undated_season` en `2018_19`) y retorna el `events_and_exclusions_df` compacto con todas las
    celdas-día que tienen `is_ignition_conaf == 1` o `exclusion_reason IS NOT NULL`, el cual permite
    materializar cualquier temporada o pliegue LOSO en milisegundos.
    """
    grid_df = pl.read_parquet(config.paths.grid_catalog_path)
    nuble_cells_df = grid_df.filter(pl.col("in_nuble")).select(
        ["cell_id", "centroid_x", "centroid_y"]
    )
    n_cells_in_nuble = nuble_cells_df.height
    valid_cell_ids = set(nuble_cells_df.get_column("cell_id").to_list())

    ignitions_conaf_df = pl.read_parquet(config.paths.ignitions_conaf_path)
    labels_firms_df = pl.read_parquet(config.paths.labels_path)
    excluded_firms_df = pl.read_parquet(config.paths.excluded_cell_days_path)

    # Calendario de las 7 temporadas fechadas y conteo de 2018_19 (undated_season)
    full_cal_2017_2025 = build_dated_calendar_df(
        start_date=date(2017, 7, 1),
        end_date=date(2025, 6, 30),
        include_undated_2018_19=True,
    )
    dated_cal_df = full_cal_2017_2025.filter(pl.col("season").is_in(DATED_SEASONS))
    undated_2018_19_cal_df = full_cal_2017_2025.filter(pl.col("season") == UNDATED_GAP_SEASON)

    n_dated_days = dated_cal_df.height
    n_undated_2018_19_days = undated_2018_19_cal_df.height
    excluded_undated_season_cell_days = n_cells_in_nuble * n_undated_2018_19_days
    total_dated_cell_days = n_cells_in_nuble * n_dated_days

    # 1. Positivos CONAF colapsados por (cell_id, date)
    valid_conaf_df, pos_collapsed_df = collapse_conaf_ignitions_by_cell_day(
        ignitions_conaf_df=ignitions_conaf_df,
        valid_cell_ids=valid_cell_ids,
        start_date=date(2017, 7, 1),
        end_date=date(2025, 6, 30),
    )

    # 2. Huella activa CONAF (conaf_active_footprint)
    conaf_fp_df = build_conaf_active_footprint_days(
        valid_conaf_df=valid_conaf_df,
        grid_df=grid_df,
        buffer_m=config.clustering.exclusion_buffer_m,
        default_post_days=int(round(config.clustering.exclusion_post_hours / 24.0)),
        max_duration_days=30,
    )

    # 3. Huella FIRMS rule_b de verano sin match CONAF (firms_unmatched_summer_rule_b)
    firms_unmatched_fp_df = build_firms_unmatched_summer_rule_b_days(
        labels_firms_df=labels_firms_df,
        excluded_firms_df=excluded_firms_df,
        valid_conaf_df=valid_conaf_df,
        grid_df=grid_df,
        match_radius_m=3000.0,
        match_days_tol=2,
    )

    # Combinar positivos y máscaras de exclusión (un positivo nunca se excluye)
    pos_keys = pos_collapsed_df.select(
        [
            "cell_id",
            "date",
            "season",
            "is_ignition_conaf",
            "n_conaf_fires",
            "superficie_ha_max",
            "superficie_ha_sum",
        ]
    )

    conaf_fp_tagged = conaf_fp_df.with_columns(pl.lit(True).alias("_in_conaf_fp"))
    firms_fp_tagged = firms_unmatched_fp_df.with_columns(pl.lit(True).alias("_in_firms_fp"))

    special_cell_days = (
        pos_keys.join(conaf_fp_tagged, on=["cell_id", "date"], how="full", coalesce=True)
        .join(firms_fp_tagged, on=["cell_id", "date"], how="full", coalesce=True)
        .with_columns(
            pl.col("is_ignition_conaf").fill_null(0).cast(pl.Int8),
            pl.col("n_conaf_fires").fill_null(0).cast(pl.Int32),
            pl.col("superficie_ha_max").fill_null(0.0).cast(pl.Float64),
            pl.col("superficie_ha_sum").fill_null(0.0).cast(pl.Float64),
            pl.col("_in_conaf_fp").fill_null(False),
            pl.col("_in_firms_fp").fill_null(False),
            season_from_date_expr("date").alias("season"),
        )
        .filter(pl.col("season").is_in(DATED_SEASONS))
        .with_columns(
            pl.when(pl.col("is_ignition_conaf") == 1)
            .then(pl.lit(None, dtype=pl.String))
            .when(pl.col("_in_conaf_fp"))
            .then(pl.lit("conaf_active_footprint"))
            .when(pl.col("_in_firms_fp"))
            .then(pl.lit("firms_unmatched_summer_rule_b"))
            .otherwise(pl.lit(None, dtype=pl.String))
            .alias("exclusion_reason")
        )
        .drop(["_in_conaf_fp", "_in_firms_fp"])
        .sort(["date", "cell_id"])
    )

    out_path = (
        persist_sparse_path
        if persist_sparse_path is not None
        else config.paths.processed_dir / "conaf_panel_special_cell_days.parquet"
    )
    panel_sha256 = atomic_write_parquet(special_cell_days, out_path)

    # Calcular resumen por temporada (7 temporadas fechadas)
    days_per_season = dict(dated_cal_df.group_by("season").len().iter_rows())
    valid_records_per_season = dict(valid_conaf_df.group_by("calendar_season").len().iter_rows())

    season_rows: list[SeasonPositiveRow] = []
    for s_name in DATED_SEASONS:
        s_days = int(days_per_season.get(s_name, 0))
        s_total_cd = n_cells_in_nuble * s_days
        s_sub = special_cell_days.filter(pl.col("season") == s_name)
        s_pos = s_sub.filter(pl.col("is_ignition_conaf") == 1)
        n_pos = s_pos.height
        n_pos_1ha = int((s_pos.get_column("superficie_ha_max") >= 1.0).sum()) if n_pos > 0 else 0
        n_pos_10ha = int((s_pos.get_column("superficie_ha_max") >= 10.0).sum()) if n_pos > 0 else 0

        n_excl_conaf = int((s_sub.get_column("exclusion_reason") == "conaf_active_footprint").sum())
        n_excl_firms = int(
            (s_sub.get_column("exclusion_reason") == "firms_unmatched_summer_rule_b").sum()
        )
        n_eff = s_total_cd - n_excl_conaf - n_excl_firms
        rate_eff = n_pos / float(n_eff) if n_eff > 0 else 0.0

        season_rows.append(
            SeasonPositiveRow(
                season=s_name,
                n_days=s_days,
                n_total_cell_days=s_total_cd,
                n_conaf_records_valid=int(valid_records_per_season.get(s_name, 0)),
                n_positive_cell_days=n_pos,
                n_positive_ge_1ha=n_pos_1ha,
                n_positive_ge_10ha=n_pos_10ha,
                n_excluded_conaf_footprint=n_excl_conaf,
                n_excluded_firms_unmatched_summer=n_excl_firms,
                n_effective_cell_days=n_eff,
                base_rate_effective=rate_eff,
            )
        )

    tot_excl_conaf = sum(r.n_excluded_conaf_footprint for r in season_rows)
    tot_excl_firms = sum(r.n_excluded_firms_unmatched_summer for r in season_rows)
    tot_excl_dated = tot_excl_conaf + tot_excl_firms
    eff_cd = total_dated_cell_days - tot_excl_dated
    tot_pos = sum(r.n_positive_cell_days for r in season_rows)
    tot_pos_1ha = sum(r.n_positive_ge_1ha for r in season_rows)
    tot_pos_10ha = sum(r.n_positive_ge_10ha for r in season_rows)

    summary = Phase2APanelSummary(
        n_cells_in_nuble=n_cells_in_nuble,
        n_dated_seasons=len(DATED_SEASONS),
        n_dated_days=n_dated_days,
        n_undated_2018_19_days=n_undated_2018_19_days,
        excluded_undated_season_cell_days=excluded_undated_season_cell_days,
        total_dated_cell_days=total_dated_cell_days,
        excluded_conaf_active_footprint=tot_excl_conaf,
        excluded_firms_unmatched_summer_rule_b=tot_excl_firms,
        excluded_small_or_agri=0,
        total_excluded_in_dated_panel=tot_excl_dated,
        effective_training_cell_days=eff_cd,
        total_conaf_valid_records_in_domain=valid_conaf_df.height,
        total_positive_cell_days=tot_pos,
        total_positive_ge_1ha=tot_pos_1ha,
        total_positive_ge_10ha=tot_pos_10ha,
        base_rate_raw_dated=tot_pos / float(total_dated_cell_days),
        base_rate_effective=tot_pos / float(eff_cd),
        season_rows=season_rows,
        panel_sha256=panel_sha256,
    )

    log_event(
        logger,
        "phase2a_panel_built",
        n_cells_in_nuble=n_cells_in_nuble,
        n_dated_days=n_dated_days,
        total_dated_cell_days=total_dated_cell_days,
        excluded_undated_season=excluded_undated_season_cell_days,
        excluded_conaf_active_footprint=tot_excl_conaf,
        excluded_firms_unmatched_summer_rule_b=tot_excl_firms,
        excluded_small_or_agri=0,
        effective_training_cell_days=eff_cd,
        total_positive_cell_days=tot_pos,
        base_rate_effective=summary.base_rate_effective,
        sha256=panel_sha256,
    )
    return special_cell_days, summary


def materialize_season_cell_day_panel(
    grid_df: pl.DataFrame,
    special_cell_days_df: pl.DataFrame,
    season: str,
    include_excluded: bool = False,
) -> pl.DataFrame:
    """Materializa el panel completo `(cell_id, date)` para una temporada fechada dada."""
    if season not in DATED_SEASONS:
        raise ValueError(
            f"La temporada '{season}' no pertenece al dominio fechado {DATED_SEASONS} "
            "(exclusion_reason = 'undated_season')."
        )
    cal_df = build_dated_calendar_df().filter(pl.col("season") == season)
    cells_df = grid_df.filter(pl.col("in_nuble")).select("cell_id")
    sub_special = special_cell_days_df.filter(pl.col("season") == season).drop("season")

    panel = (
        cells_df.join(cal_df, how="cross")
        .join(sub_special, on=["cell_id", "date"], how="left")
        .with_columns(
            pl.col("is_ignition_conaf").fill_null(0).cast(pl.Int8),
            pl.col("n_conaf_fires").fill_null(0).cast(pl.Int32),
            pl.col("superficie_ha_max").fill_null(0.0).cast(pl.Float64),
            pl.col("superficie_ha_sum").fill_null(0.0).cast(pl.Float64),
        )
    )
    if not include_excluded:
        panel = panel.filter(pl.col("exclusion_reason").is_null())
    return panel


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config("configs/config.yaml")
    _, s = build_phase2a_daily_panel(cfg)
    print(f"n_cells_in_nuble = {s.n_cells_in_nuble:,}")
    print(
        f"n_dated_days = {s.n_dated_days:,} | total_dated_cell_days = {s.total_dated_cell_days:,}"
    )
    print(
        f"exclusion_reason counts: undated_season (2018_19) = {s.excluded_undated_season_cell_days:,}, "
        f"conaf_active_footprint = {s.excluded_conaf_active_footprint:,}, "
        f"firms_unmatched_summer_rule_b = {s.excluded_firms_unmatched_summer_rule_b:,}, "
        f"small_or_agri = {s.excluded_small_or_agri:,}"
    )
    print(
        f"effective_training_cell_days = {s.effective_training_cell_days:,} | "
        f"total_conaf_valid_records = {s.total_conaf_valid_records_in_domain:,} | "
        f"total_positive_cell_days = {s.total_positive_cell_days:,} "
        f"(>=1ha: {s.total_positive_ge_1ha:,}, >=10ha: {s.total_positive_ge_10ha:,})"
    )
    print(
        f"base_rate_raw_dated = {s.base_rate_raw_dated:.8e} "
        f"(1 per {int(round(1.0 / s.base_rate_raw_dated)):,}) | "
        f"base_rate_effective = {s.base_rate_effective:.8e} "
        f"(1 per {int(round(1.0 / s.base_rate_effective)):,})"
    )
    for r in s.season_rows:
        print(
            f"  {r.season}: days={r.n_days}, records={r.n_conaf_records_valid}, "
            f"pos_cell_days={r.n_positive_cell_days} (>=1ha:{r.n_positive_ge_1ha}, >=10ha:{r.n_positive_ge_10ha}), "
            f"excl_conaf={r.n_excluded_conaf_footprint:,}, excl_firms={r.n_excluded_firms_unmatched_summer:,}, "
            f"eff_cd={r.n_effective_cell_days:,}, rate={r.base_rate_effective:.6e}"
        )
