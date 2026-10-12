"""LanceDB tables: pieces (per-tile clusters), regions (cross-tile partition), tiles (naive).

Regions and tiles share one schema, so search and segmentation treat them the
same way:

    region_id, dino_vector, [skyclip_vector], x_min, y_min, x_max, y_max,
    area, num_members, member_ids, mask_geojson

Geometries (mask_geojson, bounds) are in the raster CRS. Each table's schema
metadata (key "unboxed") records its kind, the raster CRS and pixel size, so
later steps need neither the source raster nor dataset constants.
"""

import json
from datetime import timedelta
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa
from rasterio.crs import CRS
from shapely.geometry import mapping, shape

from .raster import write_geojson

DINO_COLUMN = "dino_vector"
SKYCLIP_COLUMN = "skyclip_vector"
META_KEY = b"unboxed"


def table_meta(table) -> dict:
    meta = (table.schema.metadata or {}).get(META_KEY)
    if meta is None:
        raise ValueError(f"Table '{table.name}' has no unboxed metadata")
    return json.loads(meta)


def make_meta(kind: str, crs: CRS, pixel_size: float, **extra) -> dict:
    return {"kind": kind, "crs": crs.to_wkt(), "pixel_size": float(pixel_size), **extra}


def parse_geometry(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    geom = shape(json.loads(value) if isinstance(value, str) else value)
    return None if geom.is_empty else geom


# ──────── Pieces (written incrementally at ingestion) ────────


def pieces_schema(ndims: int, meta: dict) -> pa.Schema:
    return pa.schema(
        [
            pa.field("piece_id", pa.int64()),
            pa.field(DINO_COLUMN, pa.list_(pa.float32(), ndims)),
            pa.field("x_min", pa.float64()),
            pa.field("y_min", pa.float64()),
            pa.field("x_max", pa.float64()),
            pa.field("y_max", pa.float64()),
            pa.field("mask_geojson", pa.string()),
        ],
        metadata={META_KEY: json.dumps(meta).encode()},
    )


class PiecesWriter:
    """Append pieces to a new table (an existing table of the same name is replaced)."""

    def __init__(self, db_path: str, table_name: str, meta: dict):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self.name = table_name
        self.meta = meta
        self.table = None
        self.count = 0

    def add(self, vectors: list[np.ndarray], geoms: list) -> None:
        if not vectors:
            return
        schema = pieces_schema(len(vectors[0]), self.meta)
        batch = pa.table(
            {
                "piece_id": np.arange(
                    self.count, self.count + len(vectors), dtype=np.int64
                ),
                DINO_COLUMN: [np.asarray(v, np.float32) for v in vectors],
                "x_min": [g.bounds[0] for g in geoms],
                "y_min": [g.bounds[1] for g in geoms],
                "x_max": [g.bounds[2] for g in geoms],
                "y_max": [g.bounds[3] for g in geoms],
                "mask_geojson": [json.dumps(mapping(g)) for g in geoms],
            },
            schema=schema,
        )
        if self.table is None:
            self.table = self.db.create_table(self.name, data=batch, mode="overwrite")
        else:
            self.table.add(batch)
        self.count += len(vectors)


# ──────── Regions / tiles ────────


def write_region_table(db, table_name: str, rows: list[dict], meta: dict):
    """rows: dicts with geometry (shapely), dino_vector, member_ids. Returns the table."""
    ndims = len(rows[0]["dino_vector"])
    schema = pa.schema(
        [
            pa.field("region_id", pa.int64()),
            pa.field(DINO_COLUMN, pa.list_(pa.float32(), ndims)),
            pa.field("x_min", pa.float64()),
            pa.field("y_min", pa.float64()),
            pa.field("x_max", pa.float64()),
            pa.field("y_max", pa.float64()),
            pa.field("area", pa.float64()),
            pa.field("num_members", pa.int64()),
            pa.field("member_ids", pa.list_(pa.int64())),
            pa.field("mask_geojson", pa.string()),
        ],
        metadata={META_KEY: json.dumps(meta).encode()},
    )
    geoms = [r["geometry"] for r in rows]
    data = pa.table(
        {
            "region_id": np.arange(len(rows), dtype=np.int64),
            DINO_COLUMN: [np.asarray(r["dino_vector"], np.float32) for r in rows],
            "x_min": [g.bounds[0] for g in geoms],
            "y_min": [g.bounds[1] for g in geoms],
            "x_max": [g.bounds[2] for g in geoms],
            "y_max": [g.bounds[3] for g in geoms],
            "area": [g.area for g in geoms],
            "num_members": [len(r["member_ids"]) for r in rows],
            "member_ids": [list(map(int, r["member_ids"])) for r in rows],
            "mask_geojson": [json.dumps(mapping(g)) for g in geoms],
        },
        schema=schema,
    )
    return db.create_table(table_name, data=data, mode="overwrite")


def add_vector_column(db, table_name: str, column: str, vectors: list[np.ndarray]):
    """Rewrite a table with an extra (or replaced) vector column, keeping its metadata."""
    table = db.open_table(table_name)
    data = table.to_arrow()
    ndims = len(vectors[0])
    arr = pa.array(
        [np.asarray(v, np.float32) for v in vectors], type=pa.list_(pa.float32(), ndims)
    )
    if column in data.column_names:
        data = data.set_column(
            data.column_names.index(column), pa.field(column, arr.type), arr
        )
    else:
        data = data.append_column(pa.field(column, arr.type), arr)
    data = data.replace_schema_metadata(table.schema.metadata)
    table = db.create_table(table_name, data=data, mode="overwrite")
    # Drop the superseded version's files (Lance keeps old versions by default)
    table.optimize(cleanup_older_than=timedelta(0), delete_unverified=True)
    return table


def export_geojson(
    table, output_path: str, columns: tuple[str, ...] = ("region_id", "num_members")
) -> None:
    """Write a region or tile table's geometries to GeoJSON (EPSG:4326)."""
    meta = table_meta(table)
    df: pd.DataFrame = table.to_pandas()
    records = [
        (parse_geometry(g), {c: int(row[c]) for c in columns if c in df.columns})
        for g, (_, row) in zip(df["mask_geojson"], df.iterrows())
    ]
    write_geojson(Path(output_path), records, CRS.from_wkt(meta["crs"]))
