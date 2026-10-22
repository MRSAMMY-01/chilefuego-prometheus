"""Pipeline idempotente de Fase 1 y Fase 1.5: Grilla, Filtrado, DBSCAN, Etiquetado y Huella T8.

Cambios de Fase 1.5 (Regla 1 a Regla 5):
- T1: Ingesta de archivos en `data/raw/firms_archive/` sin depender del nombre del archivo,
  conservando todas las columnas originales y generando `manifest.json` con SHA-256.
- T2: Eliminación de `deduplicate_inter_satellite_passes`; se conserva `n_pixels` por cluster.
- T3: Serie principal restringida a `config.firms.primary_sources` (`VIIRS_SNPP_SP`) en el
  rango `2012-01-20` a `2025-12-31`; `VIIRS_NOAA20_SP` se evalúa como serie de sensibilidad.
- T4: Aplicación de máscara `type == 0` y trazabilidad de `n_after_type`.
- T5: Clasificación `event_class` (`wildfire_candidate` vs `small_or_agri`) guardada en
  `labels.parquet`; las quemas (`small_or_agri`) se tratan como 'ignorar' (nunca como ceros).
- T6: Adición de `detected_night` (`00-04h` local) y `window_start_utc = first_detection_ts - 24h`.
- T7: Dominio espacial idéntico (`in_nuble == True` sobre DPA oficial de Ñuble, área verificada
  dentro de ±5% de 13.178 km²) en numerador y denominador de la tasa base.
- T8: Generación de `excluded_cell_days.parquet` con la huella activa (`extensión + 2,5 km`,
  desde `first_detection_ts` hasta `last_detection_ts + 72 h`).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import polars as pl

from chilefuego.config import AppConfig, load_config
from chilefuego.ingest.firms import (
    attach_utc_and_chile_timestamps,
    ingest_firms_archive_csv,
    scan_raw_firms_lazy,
)
from chilefuego.labels.clustering import (
    build_excluded_cell_days,
    cluster_spatiotemporal_detections,
    extract_ignitions_from_clusters,
)
from chilefuego.labels.filters import (
    filter_by_confidence,
    filter_by_type,
    filter_volcanic_and_snow_zones,
    flag_and_filter_stationary_sources,
)
from chilefuego.labels.grid import (
    assign_points_to_grid,
    save_grid_catalog,
    verify_nuble_grid_area,
)
from chilefuego.logging_utils import atomic_write_parquet, get_logger, log_event
from chilefuego.schemas import validate_filtered_firms, validate_ignition_labels

logger = get_logger(__name__)


@dataclass(frozen=True)
class YearlyFunnelRow:
    """Fila anual del embudo de procesamiento de focos -> igniciones (Fase 1.5)."""

    year: int
    n_raw: int
    n_after_conf: int
    n_after_type: int
    n_after_excl: int
    n_filtered: int
    n_clusters: int
    n_ignitions: int
    n_wildfire_candidates: int
    n_small_or_agri: int


@dataclass(frozen=True)
class Phase1PipelineResult:
    """Resultado auditable de la ejecución completa de Fase 1.5."""

    n_cells: int
    n_cells_in_nuble: int
    area_in_nuble_km2: float
    area_diff_pct: float
    n_days: int
    total_cell_days: int
    total_excluded_cell_days: int
    total_raw_detections: int
    total_filtered_detections: int
    total_clusters: int
    total_ignitions: int
    total_wildfire_candidates: int
    total_small_or_agri: int
    base_ignition_rate_all: float
    base_ignition_rate_wildfire: float
    grid_sha256: str
    labels_sha256: str
    excluded_sha256: str
    yearly_funnel: list[YearlyFunnelRow]
    report_markdown: str


def run_labeling_from_raw_df(
    raw_df: pl.DataFrame,
    config: AppConfig,
    sources_override: list[str] | None = None,
    return_clustered: bool = False,
) -> (
    tuple[pl.DataFrame, list[YearlyFunnelRow]]
    | tuple[pl.DataFrame, list[YearlyFunnelRow], pl.DataFrame]
):
    """Transforma un DataFrame crudo FIRMS en `labels` validados y métricas anuales (Fase 1.5)."""
    years = list(range(config.temporal.start_year, config.temporal.end_year + 1))
    empty_clustered = pl.DataFrame(schema={"cluster_id": pl.Int64})

    if raw_df.is_empty():
        empty_labels = extract_ignitions_from_clusters(
            empty_clustered,
            rule_name=config.clustering.event_class_rule,
            chile_tz=config.project.timezone,
        )
        zero_rows = [
            YearlyFunnelRow(
                year=y,
                n_raw=0,
                n_after_conf=0,
                n_after_type=0,
                n_after_excl=0,
                n_filtered=0,
                n_clusters=0,
                n_ignitions=0,
                n_wildfire_candidates=0,
                n_small_or_agri=0,
            )
            for y in years
        ]
        if return_clustered:
            return empty_labels, zero_rows, empty_clustered
        return empty_labels, zero_rows

    active_sources = (
        set(sources_override) if sources_override is not None else set(config.firms.primary_sources)
    )
    source_filtered = raw_df.filter(pl.col("satellite").is_in(active_sources))

    if source_filtered.is_empty():
        empty_labels = extract_ignitions_from_clusters(
            empty_clustered,
            rule_name=config.clustering.event_class_rule,
            chile_tz=config.project.timezone,
        )
        zero_rows = [
            YearlyFunnelRow(
                year=y,
                n_raw=0,
                n_after_conf=0,
                n_after_type=0,
                n_after_excl=0,
                n_filtered=0,
                n_clusters=0,
                n_ignitions=0,
                n_wildfire_candidates=0,
                n_small_or_agri=0,
            )
            for y in years
        ]
        if return_clustered:
            return empty_labels, zero_rows, empty_clustered
        return empty_labels, zero_rows

    # 1. Conversión horaria UTC -> America/Santiago y filtro temporal [start_date, end_date]
    with_ts = attach_utc_and_chile_timestamps(source_filtered, chile_tz=config.project.timezone)
    with_ts = with_ts.filter(
        (pl.col("local_date") >= config.temporal.start_date)
        & (pl.col("local_date") <= config.temporal.end_date)
    ).with_columns(pl.col("local_date").dt.year().alias("_local_year"))

    # 2. Indexación espacial a la grilla EPSG:32719 y recorte estricto a in_nuble=True (T7)
    in_grid = assign_points_to_grid(
        with_ts,
        config.geospatial,
        exclude_argentina=True,
        restrict_to_nuble=True,
        dpa_geojson_path=config.paths.dpa_geojson_path,
    )
    raw_counts_by_year = dict(in_grid.group_by("_local_year").len().iter_rows())

    # 3. Cascada de máscaras de falsos positivos (confianza -> type==0 -> volcán -> estacionaria)
    after_conf = filter_by_confidence(in_grid, config.filtering)
    conf_counts_by_year = dict(after_conf.group_by("_local_year").len().iter_rows())

    after_type = filter_by_type(after_conf, config.filtering)
    type_counts_by_year = dict(after_type.group_by("_local_year").len().iter_rows())

    after_volc = filter_volcanic_and_snow_zones(after_type, config.filtering)
    excl_counts_by_year = dict(after_volc.group_by("_local_year").len().iter_rows())

    after_stat, _stationary_cells = flag_and_filter_stationary_sources(after_volc, config.filtering)

    clean_stat = after_stat.drop("_local_year", strict=False)
    filtered_df = validate_filtered_firms(clean_stat).with_columns(
        pl.col("local_date").dt.year().alias("_local_year")
    )
    filtered_counts_by_year = dict(filtered_df.group_by("_local_year").len().iter_rows())

    # 4. Clustering espacio-temporal con DBSCAN (sin deduplicate_inter_satellite_passes, T2)
    clustered_df = cluster_spatiotemporal_detections(
        filtered_df.drop("_local_year"),
        config.clustering,
    )
    ignitions_df = extract_ignitions_from_clusters(
        clustered_df,
        rule_name=config.clustering.event_class_rule,
        chile_tz=config.project.timezone,
    )

    # Estadísticas anuales de clusters, igniciones y event_class
    ign_with_year = ignitions_df.with_columns(pl.col("date").dt.year().alias("_year"))
    ign_counts_by_year = dict(ign_with_year.group_by("_year").len().iter_rows())
    wf_counts_by_year = dict(
        ign_with_year.filter(pl.col("event_class") == "wildfire_candidate")
        .group_by("_year")
        .len()
        .iter_rows()
    )
    agri_counts_by_year = dict(
        ign_with_year.filter(pl.col("event_class") == "small_or_agri")
        .group_by("_year")
        .len()
        .iter_rows()
    )

    clusters_by_year: dict[int, int] = {}
    if not clustered_df.is_empty():
        first_per_cluster = (
            clustered_df.filter(pl.col("cluster_id") >= 0)
            .sort("acq_ts_utc")
            .group_by("cluster_id")
            .agg(pl.col("local_date").first().dt.year().alias("_year"))
        )
        clusters_by_year = dict(first_per_cluster.group_by("_year").len().iter_rows())

    active_years = sorted(set(raw_counts_by_year.keys()) | set(years))
    funnel_rows: list[YearlyFunnelRow] = []
    for y in active_years:
        funnel_rows.append(
            YearlyFunnelRow(
                year=int(y),
                n_raw=int(raw_counts_by_year.get(y, 0)),
                n_after_conf=int(conf_counts_by_year.get(y, 0)),
                n_after_type=int(type_counts_by_year.get(y, 0)),
                n_after_excl=int(excl_counts_by_year.get(y, 0)),
                n_filtered=int(filtered_counts_by_year.get(y, 0)),
                n_clusters=int(clusters_by_year.get(y, 0)),
                n_ignitions=int(ign_counts_by_year.get(y, 0)),
                n_wildfire_candidates=int(wf_counts_by_year.get(y, 0)),
                n_small_or_agri=int(agri_counts_by_year.get(y, 0)),
            )
        )

    validated_labels = validate_ignition_labels(ignitions_df)
    if return_clustered:
        return validated_labels, funnel_rows, clustered_df
    return validated_labels, funnel_rows


def build_lazy_cell_day_panel(
    grid_lf: pl.LazyFrame,
    labels_lf: pl.LazyFrame,
    start_date: date,
    end_date: date,
    excluded_cell_days_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Construye de forma PEREZOSA (LazyFrame) el universo `celda × día` (Fase 1.5 - T5 y T8).

    Reglas críticas:
    1) Las quemas (`event_class == 'small_or_agri'`) NO son ceros: se excluyen del conjunto de
       entrenamiento (`anti join`) para que nunca se etiqueten como `is_ignition = 0`.
    2) Si se provee `excluded_cell_days_lf` (T8: huella del cluster + 2,5 km y + 72 h), las
       celdas-día en la huella que no sean ignición positiva (`wildfire_candidate`)
       también se excluyen de los negativos.
    """
    dates_lf = pl.DataFrame(
        {"date": pl.date_range(start_date, end_date, interval="1d", eager=True)}
    ).lazy()

    base_cells_lf = (
        grid_lf.filter(pl.col("in_nuble")).select("cell_id")
        if "in_nuble" in grid_lf.collect_schema().names()
        else grid_lf.select("cell_id")
    )
    full_panel_lf = base_cells_lf.join(dates_lf, how="cross")

    has_event_class = "event_class" in labels_lf.collect_schema().names()
    if has_event_class:
        agri_lf = labels_lf.filter(pl.col("event_class") == "small_or_agri").select(
            ["cell_id", "date"]
        )
        positive_labels_lf = labels_lf.filter(pl.col("event_class") == "wildfire_candidate")
        full_panel_lf = full_panel_lf.join(agri_lf, on=["cell_id", "date"], how="anti")
    else:
        positive_labels_lf = labels_lf

    panel_with_pos = full_panel_lf.join(
        positive_labels_lf.select(
            [
                "cell_id",
                "date",
                "is_ignition",
                "cluster_id",
                "n_detections",
                "first_detection_ts",
            ]
        ),
        on=["cell_id", "date"],
        how="left",
    ).with_columns(
        pl.col("is_ignition").fill_null(0).cast(pl.Int8),
        pl.col("n_detections").fill_null(0).cast(pl.Int32),
    )

    if excluded_cell_days_lf is not None:
        # Excluir de los negativos (is_ignition == 0) toda celda-día dentro de la huella T8
        excl_keys_lf = (
            excluded_cell_days_lf.select(["cell_id", "date"])
            .unique()
            .with_columns(pl.lit(True).alias("_in_footprint"))
        )
        panel_with_pos = (
            panel_with_pos.join(excl_keys_lf, on=["cell_id", "date"], how="left")
            .filter((pl.col("is_ignition") == 1) | pl.col("_in_footprint").is_null())
            .drop("_in_footprint")
        )

    return panel_with_pos


def format_markdown_report(result: Phase1PipelineResult, config: AppConfig) -> str:
    """Genera el reporte Markdown de Fase 1.5."""
    bbox_dict = config.geospatial.bbox_wgs84.model_dump()
    cell_m = config.geospatial.cell_size_m
    eps_m = config.clustering.spatial_eps_m
    eps_h = config.clustering.temporal_eps_hours
    inv_rate_all = int(round(1.0 / max(result.base_ignition_rate_all, 1e-12)))
    inv_rate_wf = int(round(1.0 / max(result.base_ignition_rate_wildfire, 1e-12)))

    header_cols = (
        "| Año | Crudos (`in_nuble`) | Post-Confianza | Post-`type==0` "
        "| Post-Volcán 4km | Post-Estacionarias | Clusters DBSCAN "
        "| Igniciones Totales | `wildfire_candidate` | `small_or_agri` (Ignorar) |"
    )
    area_line = (
        f"- **Área `in_nuble` Verificada (T7):** `{result.area_in_nuble_km2:,.2f} km²` "
        f"(`{result.area_diff_pct:+.2f}%` vs `13.178,00 km²`)"
    )
    src_line = (
        f"- **Serie Principal (T3):** `{config.firms.primary_sources}` "
        f"(`{config.temporal.start_date}` a `{config.temporal.end_date}`, `{result.n_days:,}` días)"
    )
    univ_line = (
        f"- **Universo Total (`Celda × Día` en Ñuble):** `{result.total_cell_days:,}` observaciones"
    )
    excl_line = (
        f"- **Celdas-Día Excluidas por Huella Activa (T8):** `{result.total_excluded_cell_days:,}`"
    )
    dbscan_line = (
        f"- **Convención DBSCAN (T9):** `spatial_eps_m = {eps_m} m`, "
        f"`temporal_eps_h = {eps_h} h` (`min_samples = 1`)"
    )

    lines = [
        "# Reporte de Etiquetado de Igniciones — ChileFuego (Fase 1.5)",
        "",
        "## 1. Dominio Geoespacial Oficial (DPA BCN Ñuble) y Espacio-Temporal",
        f"- **Región de Estudio:** Región de Ñuble, Chile (`BBox WGS84`: `{bbox_dict}`)",
        f"- **CRS Métrico de Trabajo:** `{config.geospatial.work_crs}` (UTM Zona 19S)",
        f"- **Resolución de Celda:** `{cell_m} m × {cell_m} m`",
        f"- **Total de Celdas en BBox (`N_cells`):** `{result.n_cells:,}`",
        f"- **Celdas dentro de Ñuble (`in_nuble=True`):** `{result.n_cells_in_nuble:,}`",
        area_line,
        src_line,
        univ_line,
        excl_line,
        dbscan_line,
        "",
        "## 2. Embudo Anual (Serie Principal S-NPP 2012–2025 en `in_nuble`)",
        "",
        header_cols,
        "| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in result.yearly_funnel:
        row_str = (
            f"| **{row.year}** | {row.n_raw:,} | {row.n_after_conf:,} | {row.n_after_type:,} "
            f"| {row.n_after_excl:,} | {row.n_filtered:,} | {row.n_clusters:,} "
            f"| **{row.n_ignitions:,}** | **{row.n_wildfire_candidates:,}** "
            f"| {row.n_small_or_agri:,} |"
        )
        lines.append(row_str)

    tot_str = (
        f"| **TOTAL** | **{result.total_raw_detections:,}** "
        f"| **{sum(r.n_after_conf for r in result.yearly_funnel):,}** "
        f"| **{sum(r.n_after_type for r in result.yearly_funnel):,}** "
        f"| **{sum(r.n_after_excl for r in result.yearly_funnel):,}** "
        f"| **{result.total_filtered_detections:,}** "
        f"| **{result.total_clusters:,}** "
        f"| **{result.total_ignitions:,}** "
        f"| **{result.total_wildfire_candidates:,}** "
        f"| **{result.total_small_or_agri:,}** |"
    )

    rate_all_str = (
        f"- **Igniciones Totales (`labels.parquet`):** `{result.total_ignitions:,}` "
        f"-> Tasa base: `{result.base_ignition_rate_all:.8e}` "
        f"(`1 cada {inv_rate_all:,}` celdas-día)"
    )
    rate_wf_str = (
        f"- **Candidatos a Incendio Forestal (`wildfire_candidate`):** "
        f"`{result.total_wildfire_candidates:,}` -> Tasa base limpia: "
        f"`{result.base_ignition_rate_wildfire:.8e}` (`1 cada {inv_rate_wf:,}` celdas-día)"
    )

    lines.extend(
        [
            tot_str,
            "",
            "## 3. Tasa Base de Ignición sobre el Mismo Dominio (`in_nuble` × `5.095` días)",
            "",
            rate_all_str,
            rate_wf_str,
            "",
            "## 4. Hashes de Reproducibilidad (SHA-256)",
            f"- `grid_catalog.parquet`: `{result.grid_sha256}`",
            f"- `labels.parquet`: `{result.labels_sha256}`",
            f"- `excluded_cell_days.parquet`: `{result.excluded_sha256}`",
            "",
        ]
    )
    return "\n".join(lines)


def execute_phase1_pipeline(
    config_path: str | Path = "configs/config.yaml",
) -> Phase1PipelineResult:
    """Ejecuta de extremo a extremo la Fase 1.5 sobre datos reales en `data/raw/firms/`."""
    config = load_config(config_path)

    # 1. Generar catálogo de celdas regulares en EPSG:32719 y verificar área T7 (±5%)
    grid_df, grid_sha256 = save_grid_catalog(
        config.geospatial,
        config.paths.grid_catalog_path,
        dpa_geojson_path=config.paths.dpa_geojson_path,
    )
    area_km2, diff_rel = verify_nuble_grid_area(grid_df, config.geospatial)

    # 2. Ingerir volcados CSV en data/raw/firms_archive/ fila a fila por columna satellite (T1)
    archive_dir = config.paths.raw_dir / "firms_archive"
    if archive_dir.exists():
        for csv_file in sorted(archive_dir.glob("*.csv")):
            ingest_firms_archive_csv(
                csv_path=csv_file,
                raw_dir=config.paths.raw_dir,
                chile_tz=config.project.timezone,
                manifest_path=config.paths.manifest_path,
            )

    # 3. Leer particiones Parquet reales en data/raw/firms (PROHIBIDO generar datos sintéticos)
    raw_lf = scan_raw_firms_lazy(config.paths.raw_dir)
    raw_df = raw_lf.collect()
    if raw_df.is_empty():
        raise FileNotFoundError(
            f"No se encontraron particiones Parquet en '{config.paths.raw_dir / 'firms'}'. "
            "Ejecute primero la descarga desde NASA FIRMS API o deposite el archivo histórico "
            "CSV en 'data/raw/firms_archive/'."
        )

    # 4. Ejecutar filtrado (con type==0), DBSCAN, clasificación event_class y extracción T6
    res_tuple = run_labeling_from_raw_df(raw_df, config, return_clustered=True)
    assert len(res_tuple) == 3
    labels_df, yearly_funnel, clustered_df = res_tuple
    labels_sha256 = atomic_write_parquet(labels_df, config.paths.labels_path)

    # 5. Generar excluded_cell_days.parquet (T8: huella del cluster + 2,5 km y + 72 h)
    excluded_df = build_excluded_cell_days(
        clustered_df=clustered_df,
        grid_df=grid_df,
        buffer_m=config.clustering.exclusion_buffer_m,
        post_hours=config.clustering.exclusion_post_hours,
        chile_tz=config.project.timezone,
    )
    excluded_sha256 = atomic_write_parquet(excluded_df, config.paths.excluded_cell_days_path)

    # 5b. Si existe conaf_nuble_historico.geojson, generar ignitions_conaf.parquet (V8)
    if config.paths.conaf_geojson_path.exists():
        from chilefuego.ingest.occurrence import build_ignitions_conaf

        build_ignitions_conaf(config, raw_firms_df=raw_df)

    # 6. Calcular universo celda × día estrictamente sobre el mismo dominio espacial y temporal (T7)
    active_rows = [r for r in yearly_funnel if r.n_raw > 0]
    if active_rows:
        min_active_year = min(r.year for r in active_rows)
        max_active_year = max(r.year for r in active_rows)
        start_dt = max(config.temporal.start_date, date(min_active_year, 1, 1))
        end_dt = min(config.temporal.end_date, date(max_active_year, 12, 31))
    else:
        start_dt = config.temporal.start_date
        end_dt = config.temporal.end_date

    n_days = (end_dt - start_dt).days + 1
    n_cells = grid_df.height
    n_cells_in_nuble = int(grid_df.get_column("in_nuble").sum())
    total_cell_days = n_cells_in_nuble * n_days

    total_raw = sum(r.n_raw for r in yearly_funnel)
    total_filtered = sum(r.n_filtered for r in yearly_funnel)
    total_clusters = sum(r.n_clusters for r in yearly_funnel)
    total_ignitions = labels_df.height
    total_wf = int((labels_df.get_column("event_class") == "wildfire_candidate").sum())
    total_agri = int((labels_df.get_column("event_class") == "small_or_agri").sum())

    # Denominador efectivo tras descontar quemas agrícolas ('ignorar') y huella activa T8
    denom_effective = max(total_cell_days - total_agri, 1)
    base_rate_all = total_ignitions / float(total_cell_days) if total_cell_days > 0 else 0.0
    base_rate_wf = total_wf / float(denom_effective) if denom_effective > 0 else 0.0

    prelim_result = Phase1PipelineResult(
        n_cells=n_cells,
        n_cells_in_nuble=n_cells_in_nuble,
        area_in_nuble_km2=area_km2,
        area_diff_pct=diff_rel * 100.0,
        n_days=n_days,
        total_cell_days=total_cell_days,
        total_excluded_cell_days=excluded_df.height,
        total_raw_detections=total_raw,
        total_filtered_detections=total_filtered,
        total_clusters=total_clusters,
        total_ignitions=total_ignitions,
        total_wildfire_candidates=total_wf,
        total_small_or_agri=total_agri,
        base_ignition_rate_all=base_rate_all,
        base_ignition_rate_wildfire=base_rate_wf,
        grid_sha256=grid_sha256,
        labels_sha256=labels_sha256,
        excluded_sha256=excluded_sha256,
        yearly_funnel=yearly_funnel,
        report_markdown="",
    )

    report_md = format_markdown_report(prelim_result, config)
    report_path = Path(config.paths.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_md, encoding="utf-8")

    log_event(
        logger,
        "phase1_5_completed",
        n_cells=n_cells,
        n_cells_in_nuble=n_cells_in_nuble,
        area_in_nuble_km2=round(area_km2, 2),
        n_days=n_days,
        total_cell_days=total_cell_days,
        total_excluded_cell_days=excluded_df.height,
        total_ignitions=total_ignitions,
        total_wildfire_candidates=total_wf,
        total_small_or_agri=total_agri,
        base_ignition_rate_all=base_rate_all,
        base_ignition_rate_wildfire=base_rate_wf,
        labels_sha256=labels_sha256,
        excluded_sha256=excluded_sha256,
    )

    return Phase1PipelineResult(**{**prelim_result.__dict__, "report_markdown": report_md})


def main() -> None:
    import sys

    parser = argparse.ArgumentParser(
        description="ChileFuego Fase 1.5 — Grilla, Filtrado, Etiquetado y Huella de Exclusión"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Ruta al archivo config.yaml",
    )
    args = parser.parse_args()
    result = execute_phase1_pipeline(args.config)
    sys.stdout.buffer.write((result.report_markdown + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
