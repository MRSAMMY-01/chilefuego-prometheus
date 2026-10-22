"""Suite de pruebas para Fase 0, Fase 1 y Fase 1.5 (T1 a T10)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pandera.errors
import polars as pl
import pytest
import yaml
from chilefuego.config import AppConfig, load_config
from chilefuego.ingest.firms import (
    attach_utc_and_chile_timestamps,
    download_firms_range_to_parquet,
    empty_raw_firms_df,
    fetch_firms_window_with_retry,
    generate_date_windows,
    get_firms_api_key,
    ingest_firms_archive_csv,
    parse_firms_csv_bytes,
    scan_raw_firms_lazy,
    write_partitioned_firms_parquet,
)
from chilefuego.ingest.occurrence import (
    ConafCsvOccurrenceSource,
    ConafStubOccurrenceSource,
    OccurrenceSource,
)
from chilefuego.labels.clustering import (
    build_excluded_cell_days,
    cluster_spatiotemporal_detections,
)
from chilefuego.labels.filters import (
    apply_false_positive_masks,
    compute_type_and_manual_mask_overlap,
)
from chilefuego.labels.grid import (
    assign_points_to_grid,
    compute_utm_grid_bounds,
    load_official_dpa_geometry,
    save_grid_catalog,
    verify_nuble_grid_area,
)
from chilefuego.labels.pipeline import (
    build_lazy_cell_day_panel,
    execute_phase1_pipeline,
    run_labeling_from_raw_df,
)
from chilefuego.logging_utils import compute_file_sha256
from chilefuego.schemas import (
    validate_excluded_cell_days,
    validate_filtered_firms,
    validate_grid_catalog,
    validate_ignition_labels,
    validate_raw_firms,
)


@pytest.fixture()
def app_config() -> AppConfig:
    """Carga la configuración oficial validada con Pydantic."""
    return load_config("configs/config.yaml")


# ==============================================================================
# T1: Ingesta conservando TODAS las columnas, mapeo estricto de satellite, manifest.json y fusión
# ==============================================================================
def test_t1_archive_parser_preserves_all_columns_and_maps_satellite_strictly(
    tmp_path: Path,
) -> None:
    """Verifica T1: conserva columnas, mapea N/1/2/SNPP y falla ante código desconocido."""
    csv_bytes = (
        b"latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
        b"instrument,confidence,version,bright_ti5,frp,daynight,type\n"
        b"-36.6200,-72.1500,335.4,0.41,0.38,2023-02-02,430,N,VIIRS,h,2.0NRT,295.1,42.5,N,0\n"
        b"-36.6210,-72.1510,340.2,0.42,0.38,2023-02-02,520,1,VIIRS,n,2.0SP,298.0,51.0,N,0\n"
        b"-36.6220,-72.1520,328.0,0.39,0.36,2023-02-02,1810,2,VIIRS,n,2.0SP,291.2,19.8,D,0\n"
    )
    df = parse_firms_csv_bytes(csv_bytes)
    assert df.height == 3
    for col in (
        "type",
        "daynight",
        "bright_ti4",
        "bright_ti5",
        "scan",
        "track",
        "instrument",
        "version",
    ):
        assert col in df.columns

    assert df.get_column("satellite").to_list() == [
        "VIIRS_SNPP_SP",
        "VIIRS_NOAA20_SP",
        "VIIRS_NOAA21_SP",
    ]
    assert df.get_column("acq_time").to_list() == ["0430", "0520", "1810"]

    # Falla ante código de satélite desconocido
    bad_sat_csv = (
        b"latitude,longitude,acq_date,acq_time,confidence,frp,satellite\n"
        b"-36.62,-72.15,2023-02-02,0430,h,25.0,TERRA_UNKNOWN\n"
    )
    with pytest.raises(ValueError, match="desconocido"):
        parse_firms_csv_bytes(bad_sat_csv)

    # Verifica manifest.json con SHA-256, preservación del CSV original y fusión idempotente con API
    archive_dir = tmp_path / "firms_archive"
    archive_dir.mkdir(parents=True)
    # Nombre deliberadamente engañoso para probar que NO depende del nombre del archivo
    csv_file = archive_dir / "nombre_arbitrario_sin_satelite.csv"
    csv_file.write_bytes(csv_bytes)
    orig_sha = compute_file_sha256(csv_file)

    # Primero simulamos partición API previa (sin columna type)
    api_partial_df = df.head(1).with_columns(pl.lit(None, dtype=pl.Int64).alias("type"))
    write_partitioned_firms_parquet(api_partial_df, raw_dir=tmp_path)

    # Ahora ingerimos el archivo CSV completo dos veces seguidas
    manifest_file = archive_dir / "manifest.json"
    ingest_firms_archive_csv(csv_file, raw_dir=tmp_path, manifest_path=manifest_file)
    ingest_firms_archive_csv(csv_file, raw_dir=tmp_path, manifest_path=manifest_file)

    # CSV original intacto
    assert compute_file_sha256(csv_file) == orig_sha
    assert manifest_file.exists()
    manifest_json = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest_json["files"][csv_file.name]["sha256"] == orig_sha
    assert manifest_json["files"][csv_file.name]["n_rows"] == 3

    # Sin duplicados y enriquecido con type==0 desde el archivo
    merged = scan_raw_firms_lazy(tmp_path).collect()
    assert merged.height == 3
    assert merged.get_column("type").null_count() == 0


# ==============================================================================
# T2: Dos detecciones del mismo satélite a 400 m en el mismo instante sobreviven ambas
# ==============================================================================
def test_t2_two_same_satellite_detections_at_400m_same_instant_both_survive(
    app_config: AppConfig,
) -> None:
    """Verifica T2: dos píxeles del mismo satélite a ~400 m en el mismo instante sobreviven."""
    # En latitud -36.62, 0.0036 grados de latitud equivalen a ~400 metros en dirección norte-sur
    df = pl.DataFrame(
        {
            "latitude": [-36.6200, -36.6164],
            "longitude": [-72.1500, -72.1500],
            "acq_date": ["2023-02-03", "2023-02-03"],
            "acq_time": ["1815", "1815"],
            "confidence": ["h", "n"],
            "frp": [45.0, 38.0],
            "satellite": ["VIIRS_SNPP_SP", "VIIRS_SNPP_SP"],
            "type": [0, 0],
        }
    )

    res = run_labeling_from_raw_df(df, app_config, return_clustered=True)
    assert len(res) == 3
    labels_df, _funnel, clustered_df = res

    # Ambas detecciones sobreviven al filtrado y al clustering
    assert clustered_df.height == 2
    dist_m = float(
        (
            (clustered_df.item(0, "x_utm") - clustered_df.item(1, "x_utm")) ** 2
            + (clustered_df.item(0, "y_utm") - clustered_df.item(1, "y_utm")) ** 2
        )
        ** 0.5
    )
    assert 350.0 <= dist_m <= 450.0

    # Generan 1 cluster de ignición pero conservando n_pixels == 2 y frp_sum == 83.0
    assert labels_df.height == 1
    assert labels_df.item(0, "n_pixels") == 2
    assert labels_df.item(0, "n_detections") == 2
    assert labels_df.item(0, "frp_sum") == pytest.approx(83.0)


# ==============================================================================
# CRITERIO DE ACEPTACIÓN 1 y T4: Foco volcánico y máscara type==0 vs manuales
# ==============================================================================
def test_volcanic_crater_focus_and_t4_type_mask(
    app_config: AppConfig,
) -> None:
    """Demuestra que anomalías volcánicas (por radio 4 km o por type!=0) generan 0 igniciones."""
    volcanic_df = pl.DataFrame(
        {
            "latitude": [-36.863, -36.850, -36.620],
            "longitude": [-71.377, -71.385, -72.150],
            "acq_date": ["2024-02-10", "2024-07-15", "2024-01-20"],
            "acq_time": ["1430", "0410", "1600"],
            "confidence": ["high", "high", "nominal"],
            "frp": [85.0, 62.4, 41.0],
            "satellite": ["VIIRS_SNPP_SP", "VIIRS_SNPP_SP", "VIIRS_SNPP_SP"],
            "type": [1, 0, 2],
        }
    )

    with_ts = attach_utc_and_chile_timestamps(volcanic_df)
    in_grid = assign_points_to_grid(with_ts, app_config.geospatial)
    overlap_df = compute_type_and_manual_mask_overlap(in_grid, app_config.filtering)
    assert overlap_df.height >= 2

    labels_df, funnel = run_labeling_from_raw_df(volcanic_df, app_config)  # type: ignore[misc]
    assert labels_df.is_empty()
    assert sum(r.n_ignitions for r in funnel) == 0


# ==============================================================================
# CRITERIO DE ACEPTACIÓN 2, T5 y T6: Cluster, event_class, detected_night y window_start_utc
# ==============================================================================
def test_cluster_of_30_detections_t5_event_class_and_t6_night_window(
    app_config: AppConfig,
) -> None:
    """Verifica cluster de 30 detecciones, clasificación wildfire_candidate y ventana -24h."""
    base_lon = -72.1500
    base_lat = -36.6200
    # 04:30 UTC en enero es 01:30 hora local de Chile (verano UTC-3) -> detected_night == True
    start_dt = datetime(2024, 1, 10, 4, 30, tzinfo=UTC)

    rows: list[dict[str, object]] = []
    for i in range(30):
        det_dt = start_dt + timedelta(hours=i * 4)
        rows.append(
            {
                "latitude": base_lat + (i * 0.0004),
                "longitude": base_lon + (i * 0.0004),
                "acq_date": det_dt.strftime("%Y-%m-%d"),
                "acq_time": det_dt.strftime("%H%M"),
                "confidence": "high" if i % 2 == 0 else "nominal",
                "frp": 35.0 + i * 1.5,
                "satellite": "VIIRS_SNPP_SP",
                "type": 0,
            }
        )

    raw_cluster_df = pl.DataFrame(rows)
    labels_df, _funnel = run_labeling_from_raw_df(raw_cluster_df, app_config)  # type: ignore[misc]

    assert labels_df.height == 1
    row = labels_df.row(0, named=True)
    assert row["is_ignition"] == 1
    assert row["n_pixels"] == 30
    assert row["n_passes"] == 30
    assert row["event_class"] == "wildfire_candidate"
    assert row["detected_night"] is True
    assert row["first_detection_ts"] == start_dt
    assert row["window_start_utc"] == start_dt - timedelta(hours=24)


# ==============================================================================
# DIAGNÓSTICO B4: acq_time ("142", "0142", "2359", "0000") y transiciones DST Chile 2019-2025
# ==============================================================================
def test_acq_time_padding_and_utc_conversion_b4() -> None:
    """Verifica padding `.str.zfill(4)` para '142', '0142', '2359', '0000' (Sección B4)."""
    df = pl.DataFrame(
        {
            "latitude": [-36.60, -36.60, -36.60, -36.60],
            "longitude": [-72.10, -72.10, -72.10, -72.10],
            "acq_date": ["2023-02-03", "2023-02-03", "2023-02-03", "2023-02-03"],
            "acq_time": ["142", "0142", "2359", "0000"],
            "confidence": ["h", "h", "n", "n"],
            "frp": [10.0, 10.0, 15.0, 12.0],
            "satellite": ["VIIRS_SNPP_SP"] * 4,
        }
    )
    res = attach_utc_and_chile_timestamps(df, chile_tz="America/Santiago")
    padded = res.get_column("acq_time").to_list()
    assert padded == ["0142", "0142", "2359", "0000"]

    utc_ts = res.get_column("acq_ts_utc").to_list()
    assert utc_ts[0] == datetime(2023, 2, 3, 1, 42, tzinfo=UTC)
    assert utc_ts[2] == datetime(2023, 2, 3, 23, 59, tzinfo=UTC)

    local_dates = res.get_column("local_date").to_list()
    assert local_dates == [
        date(2023, 2, 2),
        date(2023, 2, 2),
        date(2023, 2, 3),
        date(2023, 2, 2),
    ]


def test_chile_dst_transition_dates_2019_to_2025_b4() -> None:
    """Compara la conversión Polars contra zoneinfo('America/Santiago') en cambios DST 2019-2025."""
    santiago_tz = ZoneInfo("America/Santiago")
    dst_test_cases = [
        ("2019-04-07", "0330"),
        ("2019-09-08", "0330"),
        ("2020-04-05", "0330"),
        ("2020-09-06", "0330"),
        ("2021-04-04", "0330"),
        ("2021-09-05", "0330"),
        ("2022-05-01", "0330"),
        ("2022-09-11", "0330"),
        ("2023-04-02", "0330"),
        ("2023-09-03", "0330"),
        ("2024-04-07", "0330"),
        ("2024-09-08", "0330"),
        ("2025-04-06", "0330"),
        ("2025-09-07", "0330"),
    ]
    df = pl.DataFrame(
        {
            "latitude": [-36.60] * len(dst_test_cases),
            "longitude": [-72.10] * len(dst_test_cases),
            "acq_date": [d for d, _ in dst_test_cases],
            "acq_time": [t for _, t in dst_test_cases],
            "confidence": ["h"] * len(dst_test_cases),
            "frp": [20.0] * len(dst_test_cases),
            "satellite": ["VIIRS_SNPP_SP"] * len(dst_test_cases),
        }
    )
    converted = attach_utc_and_chile_timestamps(df, chile_tz="America/Santiago")
    polars_dates = converted.get_column("local_date").to_list()

    expected_dates: list[date] = []
    for d_str, t_str in dst_test_cases:
        dt_utc = datetime.strptime(f"{d_str} {t_str.zfill(4)}", "%Y-%m-%d %H%M").replace(tzinfo=UTC)
        expected_dates.append(dt_utc.astimezone(santiago_tz).date())

    assert polars_dates == expected_dates


# ==============================================================================
# T7 y T8: Límite oficial DPA Ñuble, chequeo de área ±5% y huella excluded_cell_days
# ==============================================================================
def test_t7_dpa_boundary_area_and_t8_excluded_footprint(app_config: AppConfig) -> None:
    """Verifica que falta de DPA detiene ejecución, área in_nuble en ±5% y huella T8."""
    with pytest.raises(FileNotFoundError, match="DPA"):
        load_official_dpa_geometry("data/raw/boundaries/no_existe_dpa.geojson")

    # Con una celda gruesa de 2000 m, verificar área dentro de ±5% de 13.178 km²
    coarse_geo = app_config.geospatial.model_copy(update={"cell_size_m": 2000})
    from chilefuego.labels.grid import build_grid_catalog

    grid_df = build_grid_catalog(coarse_geo)
    area_km2, diff_rel = verify_nuble_grid_area(grid_df, coarse_geo)
    assert abs(diff_rel) <= 0.05
    assert 12519.1 <= area_km2 <= 13836.9

    # Verificar que assign_points_to_grid con restrict_to_nuble=True descarta puntos fuera de Ñuble
    bbox = app_config.geospatial.bbox_wgs84
    pts_df = pl.DataFrame(
        {
            "latitude": [bbox.south, -36.60, -36.50, -33.45],
            "longitude": [bbox.west, -72.10, -71.01, -70.66],
            "acq_date": ["2024-01-10"] * 4,
            "acq_time": ["1200", "1205", "1210", "1215"],
            "confidence": ["high"] * 4,
            "frp": [15.0, 18.0, 25.0, 50.0],
            "satellite": ["VIIRS_SNPP_SP"] * 4,
        }
    )
    only_nuble = assign_points_to_grid(pts_df, coarse_geo, restrict_to_nuble=True)
    assert only_nuble.height == 1

    # Probar T8: build_excluded_cell_days sobre un foco en (-36.60, -72.10)
    with_ts = attach_utc_and_chile_timestamps(only_nuble)
    clustered = cluster_spatiotemporal_detections(with_ts, app_config.clustering)
    excl_df = build_excluded_cell_days(
        clustered_df=clustered,
        grid_df=grid_df,
        buffer_m=2500.0,
        post_hours=72.0,
    )
    assert validate_excluded_cell_days(excl_df).height > 0
    # Debería cubrir al menos 4 días calendario (día de detección + 3 días = 72 h)
    assert excl_df.get_column("date").n_unique() >= 4


def test_t5_small_or_agri_ignored_and_t8_footprint_excluded_in_panel(
    app_config: AppConfig,
) -> None:
    """Verifica que quemas (small_or_agri) y huella activa NO sean etiquetadas como ceros."""
    small_grid = pl.DataFrame(
        {
            "cell_id": ["NUBLE_R0001_C0001", "NUBLE_R0001_C0002", "NUBLE_R0001_C0003"],
            "in_nuble": [True, True, True],
        }
    ).lazy()

    ts = datetime(2024, 4, 15, 18, 0, tzinfo=UTC)
    labels_df = pl.DataFrame(
        {
            "cell_id": ["NUBLE_R0001_C0001", "NUBLE_R0001_C0002"],
            "date": [date(2024, 4, 15), date(2024, 4, 15)],
            "is_ignition": [1, 1],
            "cluster_id": [0, 1],
            "n_detections": [1, 8],
            "n_pixels": [1, 8],
            "n_passes": [1, 3],
            "frp_max": [5.0, 65.0],
            "frp_sum": [5.0, 210.0],
            "duration_hours": [0.0, 18.0],
            "extent_m": [0.0, 1200.0],
            "event_class": ["small_or_agri", "wildfire_candidate"],
            "first_detection_ts": [ts, ts],
            "detected_night": [False, False],
            "window_start_utc": [ts - timedelta(hours=24), ts - timedelta(hours=24)],
        }
    )
    excl_df = pl.DataFrame(
        {
            "cell_id": ["NUBLE_R0001_C0003"],
            "date": [date(2024, 4, 15)],
            "cluster_id": [1],
            "reason": ["active_cluster_footprint"],
        }
    )

    panel = build_lazy_cell_day_panel(
        grid_lf=small_grid,
        labels_lf=labels_df.lazy(),
        start_date=date(2024, 4, 15),
        end_date=date(2024, 4, 15),
        excluded_cell_days_lf=excl_df.lazy(),
    ).collect()

    # C0001 (small_or_agri) y C0003 (huella activa) se excluyen; solo queda C0002 (=1)
    assert panel.height == 1
    assert panel.item(0, "cell_id") == "NUBLE_R0001_C0002"
    assert panel.item(0, "is_ignition") == 1


def test_low_confidence_and_stationary_industrial_sources_filtered_c3(
    app_config: AppConfig,
) -> None:
    """Focos de baja confianza ('l') y fuentes estacionarias (invierno o >=9 meses) se excluyen."""
    rows: list[dict[str, object]] = [
        {
            "latitude": -36.60,
            "longitude": -72.10,
            "acq_date": "2024-01-15",
            "acq_time": "1530",
            "confidence": "l",
            "frp": 8.0,
            "satellite": "VIIRS_SNPP_SP",
            "type": 0,
        }
    ]
    for month in (1, 3, 5, 7, 9):
        for day in (4, 12, 20):
            rows.append(
                {
                    "latitude": -36.6520,
                    "longitude": -72.4210,
                    "acq_date": date(2024, month, day).isoformat(),
                    "acq_time": "0300",
                    "confidence": "nominal",
                    "frp": 25.0,
                    "satellite": "VIIRS_SNPP_SP",
                    "type": 0,
                }
            )

    raw_df = pl.DataFrame(rows)
    with_ts = attach_utc_and_chile_timestamps(raw_df)
    in_grid = assign_points_to_grid(with_ts, app_config.geospatial)
    filtered_df, stats, stationary_cells = apply_false_positive_masks(in_grid, app_config.filtering)

    assert stats.n_input_in_bbox == 16
    assert stats.n_after_confidence_mask == 15
    assert stats.n_after_type_mask == 15
    assert stats.n_after_stationary_mask == 0
    assert len(stationary_cells) == 1
    assert filtered_df.is_empty()


def test_pandera_validates_empty_dataframes_strictly_c5() -> None:
    """Verifica que validate_* ejecuta Pandera aún cuando df.is_empty() es True (Corrección C5)."""
    valid_empty = empty_raw_firms_df()
    assert valid_empty.is_empty()
    assert validate_raw_firms(valid_empty).is_empty()

    bad_empty = pl.DataFrame(schema={"latitude": pl.String, "wrong_col": pl.Int64})
    assert bad_empty.is_empty()
    with pytest.raises(pandera.errors.SchemaError):
        validate_raw_firms(bad_empty)

    with pytest.raises(pandera.errors.SchemaError):
        validate_filtered_firms(bad_empty)

    with pytest.raises(pandera.errors.SchemaError):
        validate_ignition_labels(bad_empty)


def test_firms_retry_and_atomic_parquet_partitioning(
    tmp_path: Path, app_config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simula fallos HTTP 503 seguidos de éxito, CSV bordes y particionado atómico en Parquet."""
    attempts = 0
    csv_payload = (
        b"latitude,longitude,acq_date,acq_time,confidence,frp,satellite\n"
        b"-36.70,-72.10,2024-02-01,540,high,44.2,N\n"
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, content=csv_payload)

    transport = httpx.MockTransport(mock_handler)
    with httpx.Client(transport=transport) as client:
        df = fetch_firms_window_with_retry(
            client=client,
            url="https://firms.modaps.eosdis.nasa.gov/api/area/csv/KEY/VIIRS_SNPP_SP/bbox/1/2024-02-01",
            source="VIIRS_SNPP_SP",
            max_retries=4,
            backoff_base_seconds=0.01,
        )

    assert attempts == 3
    assert df.height == 1
    assert df.item(0, "acq_time") == "0540"
    assert df.item(0, "satellite") == "VIIRS_SNPP_SP"

    assert parse_firms_csv_bytes(b"").is_empty()
    assert parse_firms_csv_bytes(b"latitude,longitude,acq_date,satellite\n").is_empty()
    with pytest.raises(ValueError, match="Respuesta de error"):
        parse_firms_csv_bytes(b"Invalid MAP_KEY provided")

    written = write_partitioned_firms_parquet(df, raw_dir=tmp_path)
    assert len(written) == 1
    part_path, sha = written[0]
    assert part_path.exists()
    assert len(sha) == 64

    write_partitioned_firms_parquet(df, raw_dir=tmp_path)
    scanned = scan_raw_firms_lazy(tmp_path).collect()
    assert scanned.height == 1

    custom_cfg = app_config.model_copy(
        update={
            "paths": app_config.paths.model_copy(update={"raw_dir": tmp_path / "dl_raw"}),
        }
    )
    real_client_cls = httpx.Client

    def ok_handler(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=csv_payload)

    monkeypatch.setenv("FIRMS_MAP_KEY", "test_valid_key_placeholder")
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **_kwargs: real_client_cls(transport=httpx.MockTransport(ok_handler)),
    )
    parts = download_firms_range_to_parquet(
        custom_cfg,
        start_date=date(2024, 2, 1),
        end_date=date(2024, 2, 2),
        mode="history",
    )
    assert len(parts) == 1


def test_occurrence_sources_and_grid_catalog_c4_e(tmp_path: Path, app_config: AppConfig) -> None:
    """Verifica ConafStubOccurrenceSource, ConafCsvOccurrenceSource y banderas C4 en grilla."""
    stub: OccurrenceSource = ConafStubOccurrenceSource()
    occ_df = stub.fetch_occurrences(date(2024, 1, 1), date(2024, 1, 31)).collect()
    assert occ_df.is_empty()

    conaf_csv = tmp_path / "conaf_occurrences.csv"
    conaf_csv.write_text(
        "fecha,lat,lon,causa,superficie\n"
        "2023-02-02,-36.62,-72.25,Intencional,1450.5\n"
        "2022-01-10,-36.50,-72.10,Accidental,12.0\n",
        encoding="utf-8",
    )
    csv_source = ConafCsvOccurrenceSource(csv_path=conaf_csv)
    occ_2023 = csv_source.fetch_occurrences(date(2023, 1, 1), date(2023, 12, 31)).collect()
    assert occ_2023.height == 1
    assert occ_2023.item(0, "superficie") == 1450.5

    windows = generate_date_windows(date(2024, 1, 1), date(2024, 1, 12), window_days=5)
    assert windows == [
        (date(2024, 1, 1), 5),
        (date(2024, 1, 6), 5),
        (date(2024, 1, 11), 2),
    ]

    small_geo = app_config.geospatial.model_copy(update={"cell_size_m": 2000})
    small_bounds = compute_utm_grid_bounds(small_geo)
    grid_path = tmp_path / "grid.parquet"
    grid_df, sha = save_grid_catalog(small_geo, grid_path)
    assert grid_df.height == small_bounds.n_cols * small_bounds.n_rows
    assert int(grid_df.get_column("in_nuble").sum()) > 0
    assert validate_grid_catalog(grid_df).height == grid_df.height
    assert compute_file_sha256(grid_path) == sha

    key = get_firms_api_key("FIRMS_MAP_KEY")
    assert len(key) > 5


def test_chaining_partition_b6(app_config: AppConfig) -> None:
    """Verifica partición anti-encadenamiento cuando supera max_cluster_duration_hours (B6)."""
    base_dt = datetime(2023, 2, 1, 16, 0, tzinfo=UTC)
    rows = []
    for step in range(36):
        dt = base_dt + timedelta(hours=step * 10)
        rows.append(
            {
                "cell_id": "NUBLE_R0050_C0050",
                "x_utm": 220000.0 + step * 200.0,
                "y_utm": 5940000.0,
                "latitude": -36.65,
                "longitude": -72.15,
                "acq_ts_utc": dt,
                "local_date": dt.date(),
                "confidence": "h",
                "frp": 40.0,
                "satellite": "VIIRS_SNPP_SP",
            }
        )
    df = pl.DataFrame(rows).with_columns(
        pl.col("acq_ts_utc").dt.replace_time_zone("UTC").cast(pl.Datetime("us", "UTC"))
    )

    unbounded = cluster_spatiotemporal_detections(
        df, app_config.clustering, apply_chaining_bounds=False
    )
    bounded = cluster_spatiotemporal_detections(
        df, app_config.clustering, apply_chaining_bounds=True
    )
    assert unbounded.get_column("cluster_id").n_unique() == 1
    assert bounded.get_column("cluster_id").n_unique() == 2


def test_end_to_end_phase1_pipeline_idempotent(tmp_path: Path, app_config: AppConfig) -> None:
    """Verifica que el pipeline falla sin datos y es idempotente al ingerir un CSV real."""
    raw_dir = tmp_path / "raw"
    custom_cfg = app_config.model_copy(
        update={
            "geospatial": app_config.geospatial.model_copy(update={"cell_size_m": 2000}),
            "paths": app_config.paths.model_copy(
                update={
                    "raw_dir": raw_dir,
                    "interim_dir": tmp_path / "interim",
                    "processed_dir": tmp_path / "processed",
                    "dpa_geojson_path": Path("data/raw/boundaries/dpa_nuble.geojson"),
                    "conaf_geojson_path": raw_dir / "conaf_test.geojson",
                    "manifest_path": raw_dir / "firms_archive" / "manifest.json",
                    "grid_catalog_path": tmp_path / "processed" / "grid_catalog.parquet",
                    "labels_path": tmp_path / "processed" / "labels.parquet",
                    "ignitions_conaf_path": tmp_path / "processed" / "ignitions_conaf.parquet",
                    "excluded_cell_days_path": tmp_path
                    / "processed"
                    / "excluded_cell_days.parquet",
                    "report_path": tmp_path / "processed" / "report.md",
                }
            ),
        }
    )
    cfg_file = tmp_path / "config.yaml"
    cfg_dict = custom_cfg.model_dump(mode="json")
    cfg_file.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        execute_phase1_pipeline(cfg_file)

    archive_dir = raw_dir / "firms_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "viirs_sample.csv").write_text(
        "latitude,longitude,acq_date,acq_time,confidence,frp,satellite,type,daynight\n"
        "-36.62,-72.20,2023-02-02,0445,h,95.4,N,0,N\n"
        "-36.621,-72.201,2023-02-02,1835,n,82.1,1,0,D\n",
        encoding="utf-8",
    )

    res1 = execute_phase1_pipeline(cfg_file)
    res2 = execute_phase1_pipeline(cfg_file)

    assert res1.labels_sha256 == res2.labels_sha256
    assert res1.excluded_sha256 == res2.excluded_sha256
    assert res1.total_ignitions == 1
    assert res1.total_wildfire_candidates == 1
    assert (raw_dir / "firms_archive" / "manifest.json").exists()
    assert (tmp_path / "processed" / "excluded_cell_days.parquet").exists()


def test_v2_v8_conaf_parsing_dedup_firms_match_and_parquet(
    tmp_path: Path, app_config: AppConfig
) -> None:
    """Verifica V2-V8: parser español/Date, resolución de duplicados <=500m y cruce FIRMS <=3km."""
    from chilefuego.ingest.occurrence import (
        build_ignitions_conaf,
        match_conaf_with_firms,
        parse_conaf_fecha_inicio,
        resolve_conaf_duplicates,
    )
    from chilefuego.schemas import validate_ignitions_conaf

    raw_conaf = pl.DataFrame(
        {
            "cell_id": ["NUBLE_R0010_C0010"] * 5,
            "x_utm": [220000.0, 220150.0, 225000.0, 230000.0, 235000.0],
            "y_utm": [5945000.0, 5945100.0, 5945000.0, 5945000.0, 5945000.0],
            "season": [
                "if_temporada_2022_2023",
                "if_temporada_2022_2023",
                "if_temporada_2023_2024",
                "if_temporada_2014_2015",
                "if_temporada_2024_2025",
            ],
            "fecha_inicio": [
                "14-ene-2023 15:30",
                "14-ene-2023 16:10",
                "2024-02-05 00:00:00",
                None,
                "10-abr-2025 00:00",
            ],
            "causa": ["Intencional", "Intencional", "Accidental", "Desconocida", "Quema"],
            "superficie_ha": [120.0, 15.0, 4.5, 1.2, 0.5],
            "ambito": ["Conaf", "Empresa", "Conaf", "Conaf", "Conaf"],
            "nombre_incendio": ["SANTA ANA", "SANTA ANA 2", "QUILMO", "EL ROBLE", "RULO"],
            "numero_reg": [1.0, 2.0, 3.0, 4.0, 5.0],
            "crs_ok": [True, True, True, True, True],
        }
    )

    dated = parse_conaf_fecha_inicio(raw_conaf)
    assert dated.get_column("date_ok").to_list() == [True, True, True, False, True]
    # 2023_2024 Date 00:00:00 y 00:00 tienen time_ok=False
    assert dated.get_column("time_ok").to_list() == [True, True, False, False, False]
    assert dated.get_column("start_ts_local").to_list()[2] is None

    deduped = resolve_conaf_duplicates(dated, max_dist_m=500.0)
    # Filas 0 y 1 están a ~180 m el mismo día 2023-01-14: fila 0 (Conaf, 120 ha) se conserva,
    # fila 1 (Empresa, 15 ha) se marca dup_removed=True
    assert deduped.get_column("dup_removed").to_list() == [False, True, False, False, False]

    firms_df = pl.DataFrame(
        {
            "x_utm": [220300.0],
            "y_utm": [5945200.0],
            "local_date": [date(2023, 1, 15)],
        }
    )
    matched = match_conaf_with_firms(deduped, firms_df, radius_m=3000.0)
    assert matched.get_column("matched_firms").to_list() == [True, True, False, False, False]

    # Verificar generación end-to-end sobre un GeoJSON sintético mínimo en tmp_path
    geojson_path = tmp_path / "conaf_mini.geojson"
    geojson_path.write_text(
        '{"type":"FeatureCollection","crs":{"type":"name","properties":{"name":"EPSG:4326"}},'
        '"features":[{"type":"Feature","properties":{"temporada_origen":"if_temporada_2022_2023",'
        '"fecha_inicio":"2-feb-2023 14:25","causa_general":"Intencional","causa_especifica":"Otra",'
        '"superficie":55.0,"ambito":"Conaf","nombre_incendio":"TEST","numero_reg":1},'
        '"geometry":{"type":"Point","coordinates":[-72.10,-36.60]}}]}',
        encoding="utf-8",
    )
    out_parquet = tmp_path / "ignitions_conaf.parquet"
    cfg_conaf = app_config.model_copy(
        update={
            "paths": app_config.paths.model_copy(
                update={
                    "conaf_geojson_path": geojson_path,
                    "ignitions_conaf_path": out_parquet,
                }
            )
        }
    )
    conaf_df, sha = build_ignitions_conaf(cfg_conaf)
    assert out_parquet.exists()
    assert len(sha) == 64
    validated = validate_ignitions_conaf(conaf_df)
    assert validated.height == 1
    qf = validated.get_column("quality_flags").to_list()[0]
    assert qf["crs_ok"] is True
    assert qf["date_ok"] is True
    assert qf["time_ok"] is True
    assert qf["dup_removed"] is False
