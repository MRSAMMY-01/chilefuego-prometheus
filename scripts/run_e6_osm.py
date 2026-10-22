"""Regeneración de características OSM con BBox bufferizado (+10 km) y extracción de Top 20 (E6a)."""

from __future__ import annotations

import polars as pl
from pyproj import Transformer

from chilefuego.features.osm import extract_osm_road_features

if __name__ == "__main__":
    df, dt, mem = extract_osm_road_features()
    print(f"OSM features regeneradas en {dt:.2f} s. RAM pico: {mem:.2f} MB")

    grid_df = pl.read_parquet("data/processed/grid_catalog.parquet")
    df = df.join(grid_df.select(["cell_id", "centroid_x", "centroid_y"]), on="cell_id", how="left")

    trans = Transformer.from_crs("EPSG:32719", "EPSG:4326", always_xy=True)
    lons, lats = trans.transform(df["centroid_x"].to_numpy(), df["centroid_y"].to_numpy())
    df = df.with_columns(pl.Series("lon", lons), pl.Series("lat", lats))

    top20 = df.sort("dist_primary_sec_trunk_m", descending=True).head(20)
    print("Top 20 celdas con mayor dist_primary_sec_trunk_m:")
    for i, row in enumerate(top20.iter_rows(named=True), 1):
        cid = row["cell_id"]
        d = row["dist_primary_sec_trunk_m"]
        la = row["lat"]
        lo = row["lon"]
        print(f"{i:2d}. cell_id: {cid} | dist: {d:8.1f} m | lat: {la:9.5f} | lon: {lo:9.5f}")
