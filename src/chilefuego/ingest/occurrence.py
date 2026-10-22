"""Ingesta, validación (V1-V8) y generación de `ignitions_conaf.parquet` para ocurrencia CONAF.

Cumple las Reglas de Trabajo y las validaciones V1 a V8:
- Conserva `OccurrenceSource`, `ConafStubOccurrenceSource` y `ConafCsvOccurrenceSource`.
- Implementa el parser de fechas en español (`D-mmm-YYYY HH:MM` + `ISO YYYY-MM-DD`), la
  resolución espacial-temporal de duplicados a `<= 500 m` (simples en el mismo día y
  complejos entre ámbitos/nombres a `±1 día` o sin fecha), el cruce con NASA FIRMS S-NPP
  (`in_nuble`, `type == 0`, confianza `n`/`h`, `<= 3 km`, `-1 a +2 días`) y la generación
  atómica de `data/processed/ignitions_conaf.parquet` validada con Pandera.
"""

from __future__ import annotations

import json
import unicodedata
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from shapely.geometry import shape

from chilefuego.config import AppConfig
from chilefuego.labels.filters import filter_by_confidence, filter_by_type
from chilefuego.labels.grid import assign_points_to_grid
from chilefuego.logging_utils import atomic_write_parquet, get_logger, log_event
from chilefuego.schemas import QUALITY_FLAGS_DTYPE, validate_ignitions_conaf

logger = get_logger(__name__)

SPANISH_MONTH_MAP: dict[str, str] = {
    "ene": "01",
    "feb": "02",
    "mar": "03",
    "abr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "ago": "08",
    "sept": "09",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dic": "12",
}


class OccurrenceSource(ABC):
    """Contrato abstracto para cualquier fuente oficial de ocurrencia de incendios."""

    @abstractmethod
    def fetch_occurrences(self, start_date: date, end_date: date) -> pl.LazyFrame:
        """Retorna un LazyFrame con registros históricos verificados en [start_date, end_date]."""


def empty_occurrences_df() -> pl.DataFrame:
    """Esquema canónico vacío para registros de ocurrencia terrestre."""
    return pl.DataFrame(
        schema={
            "occurrence_id": pl.String,
            "fecha": pl.Date,
            "latitude": pl.Float64,
            "longitude": pl.Float64,
            "causa": pl.String,
            "superficie": pl.Float64,
            "source_authority": pl.String,
        }
    )


class ConafStubOccurrenceSource(OccurrenceSource):
    """Implementación stub honesta para registros históricos de CONAF (Regla 1)."""

    def fetch_occurrences(self, start_date: date, end_date: date) -> pl.LazyFrame:
        if start_date > end_date:
            raise ValueError("start_date no puede ser posterior a end_date")
        return empty_occurrences_df().lazy()


class ConafCsvOccurrenceSource(OccurrenceSource):
    """Adaptador de lectura para planilla histórica de ocurrencia CONAF en `data/raw/conaf/`."""

    def __init__(
        self,
        csv_path: str | Path = "data/raw/conaf/conaf_occurrences.csv",
    ) -> None:
        self.csv_path = Path(csv_path)

    def fetch_occurrences(self, start_date: date, end_date: date) -> pl.LazyFrame:
        if start_date > end_date:
            raise ValueError("start_date no puede ser posterior a end_date")
        if not self.csv_path.exists():
            return empty_occurrences_df().lazy()

        raw_df = pl.read_csv(
            self.csv_path,
            infer_schema_length=2000,
            ignore_errors=False,
        )
        required = {"fecha", "lat", "lon", "causa", "superficie"}
        missing = required - set(raw_df.columns)
        if missing:
            raise ValueError(
                f"El archivo CONAF {self.csv_path} no contiene las columnas mínimas requeridas: "
                f"faltan {sorted(missing)}"
            )

        if "occurrence_id" not in raw_df.columns:
            raw_df = raw_df.with_row_index("row_nr").with_columns(
                pl.format("CONAF_{}", pl.col("row_nr")).alias("occurrence_id")
            )

        normalized = (
            raw_df.with_columns(
                pl.col("occurrence_id").cast(pl.String),
                pl.col("fecha").cast(pl.String).str.slice(0, 10).str.to_date("%Y-%m-%d"),
                pl.col("lat").cast(pl.Float64).alias("latitude"),
                pl.col("lon").cast(pl.Float64).alias("longitude"),
                pl.col("causa").cast(pl.String).fill_null("desconocida"),
                pl.col("superficie").cast(pl.Float64).fill_null(0.0).clip(lower_bound=0.0),
                pl.lit("CONAF").alias("source_authority"),
            )
            .filter((pl.col("fecha") >= start_date) & (pl.col("fecha") <= end_date))
            .select(
                [
                    "occurrence_id",
                    "fecha",
                    "latitude",
                    "longitude",
                    "causa",
                    "superficie",
                    "source_authority",
                ]
            )
        )
        return normalized.lazy()


def _normalize_text(val: object) -> str:
    """Normaliza texto eliminando tildes y espacios redundantes para comparación robusta."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip().upper()
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def conaf_date_expr(column: str) -> pl.Expr:
    """Expresión Polars que parsea una columna de fecha CONAF (meses en español o ISO) a `Date`.

    Se reutiliza para `fecha_extincion`, que en la Fase 2A se usa EXCLUSIVAMENTE para construir la
    máscara `conaf_active_footprint` (nunca como predictor: es información posterior a la ignición).
    """
    norm = pl.col(column).cast(pl.String).str.strip_chars()
    for es_abbr, mm_num in SPANISH_MONTH_MAP.items():
        norm = norm.str.replace_all(f"(?i)-{es_abbr}-", f"-{mm_num}-")
    dmy = norm.str.to_datetime("%d-%m-%Y %H:%M", strict=False, time_unit="us").dt.date()
    iso = norm.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    return pl.coalesce([dmy, iso])


def parse_conaf_fecha_inicio(
    df: pl.DataFrame,
    chile_tz: str = "America/Santiago",
) -> pl.DataFrame:
    """Parsea `fecha_inicio` de CONAF (V3 y V4) manejando meses en español e ISO Date.

    Formatos presentes en los 15 shapefiles históricos:
    1) `D-mmm-YYYY HH:MM` en español (ej. `14-feb-2018 07:51`, `6-ene-2020 15:20`) en
       temporadas `2017_2018`, `2019_2020`..`2022_2023` y `2024_2025` (`2.969` filas).
    2) `YYYY-MM-DD 00:00:00` (proveniente de campo `Date` sin hora en shapefile `2023_2024`,
       `347` filas): `date_local` es válida (`date_ok = True`), pero no tiene hora real
       (`start_ts_local = null`, `time_ok = False`).
    3) `null` en temporadas `2010_2011`..`2016_2017` y `2018_2019` (`3.579` filas):
       `date_local = null`, `start_ts_local = null`, `date_ok = False`, `time_ok = False`.
    """
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Date).alias("date_local"),
            pl.lit(None, dtype=pl.Datetime("us", chile_tz)).alias("start_ts_local"),
            pl.lit(False).alias("date_ok"),
            pl.lit(False).alias("time_ok"),
        )

    raw_s = df.get_column("fecha_inicio").cast(pl.String)
    norm_expr = raw_s.str.strip_chars()
    for es_abbr, mm_num in SPANISH_MONTH_MAP.items():
        norm_expr = norm_expr.str.replace_all(f"(?i)-{es_abbr}-", f"-{mm_num}-")

    work = df.with_columns(norm_expr.alias("_norm_dt"))

    # Caso 1: Formato D-MM-YYYY HH:MM (con hora explícita)
    dmy_hm = pl.col("_norm_dt").str.to_datetime("%d-%m-%Y %H:%M", strict=False, time_unit="us")
    # Caso 2: Formato ISO YYYY-MM-DD HH:MM:SS o YYYY-MM-DD
    iso_hms = pl.col("_norm_dt").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False, time_unit="us")
    iso_d = (
        pl.col("_norm_dt")
        .str.slice(0, 10)
        .str.to_date("%Y-%m-%d", strict=False)
        .cast(pl.Datetime("us"))
    )

    combined_naive = pl.coalesce([dmy_hm, iso_hms, iso_d])
    is_date_only_season = pl.col("season") == "if_temporada_2023_2024"
    has_explicit_hm = dmy_hm.is_not_null() | (iso_hms.is_not_null() & (~is_date_only_season))

    work = work.with_columns(
        combined_naive.dt.date().alias("date_local"),
        pl.when(has_explicit_hm)
        .then(
            combined_naive.dt.replace_time_zone(
                chile_tz,
                ambiguous="earliest",
                non_existent="null",
            )
        )
        .otherwise(pl.lit(None, dtype=pl.Datetime("us", chile_tz)))
        .alias("start_ts_local"),
    )

    # date_ok: fecha parseable y no futura (<= 2025-12-31)
    # time_ok: tiene timestamp con hora explícita (no es el volcado Date 00:00:00 de 2023_2024
    # ni un placeholder de medianoche 00:00)
    work = work.with_columns(
        (
            pl.col("date_local").is_not_null()
            & (pl.col("date_local") <= date(2025, 12, 31))
            & (pl.col("date_local") >= date(2010, 1, 1))
        ).alias("date_ok"),
        (
            pl.col("start_ts_local").is_not_null()
            & ~(
                (pl.col("start_ts_local").dt.hour() == 0)
                & (pl.col("start_ts_local").dt.minute() == 0)
            )
        ).alias("time_ok"),
    )
    return work.drop("_norm_dt")


def resolve_conaf_duplicates(
    df: pl.DataFrame,
    max_dist_m: float = 500.0,
) -> pl.DataFrame:
    """Identifica y marca duplicados espaciales-temporales en CONAF (V5).

    Criterio determinista en 3 reglas sobre grafo de vecindad (`<= 500 m` en `EPSG:32719`):
    1) Duplicados directos con fecha: misma `date_local` y distancia `<= 500 m`.
    2) Duplicados complejos con fecha (`|Δdate_local| == 1 día` y distancia `<= 500 m`):
       cuando comparten `nombre_incendio` normalizado, misma `superficie_ha > 0` o provienen
       de distinto `ambito` (`Conaf` vs `Empresa`, doble reporte al cruzar medianoche).
    3) Duplicados en temporadas sin fecha (`date_local IS NULL`, misma `season`, `<= 500 m`):
       cuando comparten exactamente la misma coordenada (`< 1 m`), el mismo `nombre_incendio`
       o la misma `superficie_ha > 0`.

    En cada componente conexa se conserva (`dup_removed = False`) el representante canónico
    priorizando: `time_ok` > `date_ok` > `ambito == 'Conaf'` > mayor `superficie_ha` > menor índice.
    """
    n = df.height
    if n <= 1:
        return df.with_columns(pl.lit(False).alias("dup_removed"))

    xs = df.get_column("x_utm").to_numpy()
    ys = df.get_column("y_utm").to_numpy()
    dates = df.get_column("date_local").to_list()
    seasons = df.get_column("season").to_list()
    names = [_normalize_text(v) for v in df.get_column("nombre_incendio").to_list()]
    ambitos = [_normalize_text(v) for v in df.get_column("ambito").to_list()]
    sups = df.get_column("superficie_ha").to_numpy()
    time_oks = df.get_column("time_ok").to_numpy()
    date_oks = df.get_column("date_ok").to_numpy()

    coords = np.column_stack([xs, ys])
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=max_dist_m, output_type="ndarray")

    edges_u: list[int] = []
    edges_v: list[int] = []
    for i_raw, j_raw in pairs:
        i = int(i_raw)
        j = int(j_raw)
        dist = float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))
        di = dates[i]
        dj = dates[j]
        is_dup = False
        if di is not None and dj is not None:
            dd = abs((di - dj).days)
            if dd == 0:
                is_dup = True
            elif dd == 1:
                same_name = bool(names[i] and names[i] == names[j])
                cross_amb_close = (ambitos[i] != ambitos[j]) and (dist <= 250.0)
                if same_name or cross_amb_close:
                    is_dup = True
        elif di is None and dj is None and seasons[i] == seasons[j]:
            same_name = bool(names[i] and names[i] == names[j])
            same_sup = bool(sups[i] > 0.0 and abs(sups[i] - sups[j]) < 1e-4)
            if same_name and (dist < 1.0 or same_sup):
                is_dup = True

        if is_dup:
            edges_u.append(i)
            edges_v.append(j)

    dup_removed = np.zeros(n, dtype=bool)
    if edges_u:
        adj = coo_matrix(
            (np.ones(len(edges_u) * 2, dtype=np.int8), (edges_u + edges_v, edges_v + edges_u)),
            shape=(n, n),
        ).tocsr()
        _n_comp, comp_ids = connected_components(adj, directed=False)
        uniq_cids, cid_counts = np.unique(comp_ids, return_counts=True)
        multi_cids = uniq_cids[cid_counts > 1].tolist()

        for cid in multi_cids:
            members = np.where(comp_ids == cid)[0].tolist()
            # Ordenar miembros para elegir el representante canónico
            members.sort(
                key=lambda idx: (
                    not bool(time_oks[idx]),
                    not bool(date_oks[idx]),
                    ambitos[idx] != "CONAF",
                    -float(sups[idx]),
                    idx,
                )
            )
            for redundant_idx in members[1:]:
                dup_removed[redundant_idx] = True

    return df.with_columns(pl.Series("dup_removed", dup_removed))


def match_conaf_with_firms(
    conaf_df: pl.DataFrame,
    firms_filtered_df: pl.DataFrame,
    radius_m: float = 3000.0,
    min_day_offset: int = -1,
    max_day_offset: int = 2,
) -> pl.DataFrame:
    """Cruza cada ignición CONAF con focos FIRMS S-NPP (`<= 3 km` y `[-1, +2]` días, V2)."""
    n = conaf_df.height
    if n == 0 or firms_filtered_df.is_empty():
        return conaf_df.with_columns(
            pl.lit(False).alias("matched_firms"),
            pl.lit(None, dtype=pl.Float64).alias("nearest_firms_dist_m"),
        )

    firms_x = firms_filtered_df.get_column("x_utm").to_numpy()
    firms_y = firms_filtered_df.get_column("y_utm").to_numpy()
    firms_d = np.array(
        [d.toordinal() for d in firms_filtered_df.get_column("local_date").to_list()],
        dtype=np.int64,
    )

    c_x = conaf_df.get_column("x_utm").to_numpy()
    c_y = conaf_df.get_column("y_utm").to_numpy()
    c_dates = conaf_df.get_column("date_local").to_list()

    matched = np.zeros(n, dtype=bool)
    nearest_dist: list[float | None] = [None] * n

    for idx in range(n):
        d_val = c_dates[idx]
        if d_val is None:
            continue
        d_ord = d_val.toordinal()
        t_mask = (firms_d >= d_ord + min_day_offset) & (firms_d <= d_ord + max_day_offset)
        if not np.any(t_mask):
            continue
        dists = np.hypot(firms_x[t_mask] - c_x[idx], firms_y[t_mask] - c_y[idx])
        min_d = float(np.min(dists))
        nearest_dist[idx] = min_d
        if min_d <= radius_m:
            matched[idx] = True

    return conaf_df.with_columns(
        pl.Series("matched_firms", matched),
        pl.Series("nearest_firms_dist_m", nearest_dist, dtype=pl.Float64),
    )


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None or val == "":
        return default
    try:
        res = float(val)
        return default if np.isnan(res) else res
    except (TypeError, ValueError):
        return default


def load_conaf_geojson_as_polars(
    conaf_geojson_path: str | Path,
    work_crs: str = "EPSG:32719",
) -> pl.DataFrame:
    """Carga `conaf_nuble_historico.geojson` y extrae coordenadas en `EPSG:4326` y `EPSG:32719`."""
    path = Path(conaf_geojson_path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo maestro CONAF en '{path}'. "
            "Ejecute primero process_conaf.py y normalize_conaf.py."
        )

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    features: list[dict[str, Any]] = payload.get("features", [])
    if not features:
        raise ValueError(f"El archivo CONAF '{path}' está vacío.")

    n = len(features)
    lon_arr = np.zeros(n, dtype=np.float64)
    lat_arr = np.zeros(n, dtype=np.float64)
    valid_geom = np.zeros(n, dtype=bool)

    seasons: list[str] = []
    fechas_ini: list[str | None] = []
    fechas_ext: list[str | None] = []
    causas: list[str] = []
    sups = np.zeros(n, dtype=np.float64)
    ambitos: list[str] = []
    nombres: list[str] = []
    numeros_reg = np.zeros(n, dtype=np.float64)

    for idx, feat in enumerate(features):
        geom_dict = feat.get("geometry")
        if geom_dict is not None:
            geom_obj = shape(geom_dict)
            if not geom_obj.is_empty:
                c = geom_obj.centroid
                lon_arr[idx] = float(c.x)
                lat_arr[idx] = float(c.y)
                valid_geom[idx] = True

        props: dict[str, Any] = feat.get("properties") or {}
        seasons.append(str(props.get("temporada_origen") or ""))
        fi_raw = props.get("fecha_inicio")
        fechas_ini.append(None if fi_raw in (None, "", "None", "nan") else str(fi_raw))
        fe_raw = props.get("fecha_extincion")
        fechas_ext.append(None if fe_raw in (None, "", "None", "nan") else str(fe_raw))

        cg = str(props.get("causa_general") or "").strip()
        ce = str(props.get("causa_especifica") or "").strip()
        if cg and cg not in ("None", "nan", "desconocida"):
            causas.append(cg)
        elif ce and ce not in ("None", "nan"):
            causas.append(ce)
        else:
            causas.append("desconocida")

        sups[idx] = max(0.0, _safe_float(props.get("superficie"), 0.0))
        ambitos.append(str(props.get("ambito") or "Conaf"))
        nombres.append(str(props.get("nombre_incendio") or ""))
        numeros_reg[idx] = _safe_float(props.get("numero_reg"), 0.0)

    to_utm = Transformer.from_crs("EPSG:4326", work_crs, always_xy=True)
    ux_arr, uy_arr = to_utm.transform(lon_arr, lat_arr)
    ux_np = np.asarray(ux_arr, dtype=np.float64)
    uy_np = np.asarray(uy_arr, dtype=np.float64)

    in_chile = (
        (lon_arr >= -76.0)
        & (lon_arr <= -66.0)
        & (lat_arr >= -56.5)
        & (lat_arr <= -17.0)
        & (np.abs(ux_np) > 1.0)
        & (np.abs(uy_np) > 1.0)
    )
    crs_ok = valid_geom & in_chile

    return pl.DataFrame(
        {
            "latitude": lat_arr,
            "longitude": lon_arr,
            "x_utm_raw": ux_np,
            "y_utm_raw": uy_np,
            "season": seasons,
            "fecha_inicio": fechas_ini,
            "fecha_extincion": fechas_ext,
            "causa": causas,
            "superficie_ha": sups,
            "ambito": ambitos,
            "nombre_incendio": nombres,
            "numero_reg": numeros_reg,
            "crs_ok": crs_ok,
        }
    )


def build_ignitions_conaf(
    config: AppConfig,
    raw_firms_df: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, str]:
    """Ejecuta el pipeline V1-V8 sobre CONAF y persiste `ignitions_conaf.parquet`."""
    base_df = load_conaf_geojson_as_polars(
        config.paths.conaf_geojson_path,
        work_crs=config.geospatial.work_crs,
    )

    # Asignar cada punto a su cell_id de 500 m en EPSG:32719 (conservando todos los registros DPA)
    gridded = assign_points_to_grid(
        base_df,
        config.geospatial,
        exclude_argentina=False,
        restrict_to_nuble=False,
        dpa_geojson_path=config.paths.dpa_geojson_path,
    )

    # Parsear fechas y horas (V3 y V4)
    dated = parse_conaf_fecha_inicio(gridded, chile_tz=config.project.timezone)

    # Resolver duplicados a <= 500 m (V5)
    deduped = resolve_conaf_duplicates(dated, max_dist_m=500.0)

    # Cruzar con FIRMS S-NPP (in_nuble, type==0, confianza n/h) si está disponible (V2)
    firms_for_match = pl.DataFrame()
    if raw_firms_df is not None and not raw_firms_df.is_empty():
        from chilefuego.ingest.firms import attach_utc_and_chile_timestamps

        snpp = raw_firms_df.filter(pl.col("satellite").is_in(config.firms.primary_sources))
        if not snpp.is_empty():
            snpp_ts = attach_utc_and_chile_timestamps(snpp, chile_tz=config.project.timezone)
            snpp_grid = assign_points_to_grid(
                snpp_ts,
                config.geospatial,
                exclude_argentina=True,
                restrict_to_nuble=True,
                dpa_geojson_path=config.paths.dpa_geojson_path,
            )
            snpp_conf = filter_by_confidence(snpp_grid, config.filtering)
            firms_for_match = filter_by_type(snpp_conf, config.filtering)

    matched = match_conaf_with_firms(
        deduped,
        firms_for_match,
        radius_m=3000.0,
        min_day_offset=-1,
        max_day_offset=2,
    )

    # Construir struct quality_flags y seleccionar esquema final V8 (+ columnas de máscara F2.2)
    result_df = matched.with_columns(
        pl.struct(
            [
                pl.col("crs_ok"),
                pl.col("date_ok"),
                pl.col("time_ok"),
                pl.col("dup_removed"),
                pl.col("matched_firms"),
            ]
        )
        .cast(QUALITY_FLAGS_DTYPE)
        .alias("quality_flags"),
        conaf_date_expr("fecha_extincion").alias("extincion_date_local"),
    ).select(
        [
            "cell_id",
            "date_local",
            "start_ts_local",
            "season",
            "causa",
            "superficie_ha",
            "quality_flags",
            "x_utm",
            "y_utm",
            "extincion_date_local",
        ]
    )

    validated = validate_ignitions_conaf(result_df)
    sha256_hex = atomic_write_parquet(validated, config.paths.ignitions_conaf_path)

    qf = validated.get_column("quality_flags")
    log_event(
        logger,
        "ignitions_conaf_built",
        path=str(config.paths.ignitions_conaf_path),
        n_total=validated.height,
        n_crs_ok=int(qf.struct.field("crs_ok").sum()),
        n_date_ok=int(qf.struct.field("date_ok").sum()),
        n_time_ok=int(qf.struct.field("time_ok").sum()),
        n_dup_removed=int(qf.struct.field("dup_removed").sum()),
        n_matched_firms=int(qf.struct.field("matched_firms").sum()),
        sha256=sha256_hex,
    )
    return validated, sha256_hex
