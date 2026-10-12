"""Naive tiling baseline: one embedding per 448 px tile.

Each tile on the ingestion grid becomes one searchable row whose DINOv3 vector
is the mean of its debiased patch features (mostly-nodata patches excluded)
and whose geometry is the tile footprint. Same schema as the region table.
"""

import argparse
import logging
from pathlib import Path

import lancedb
import rasterio
from shapely.geometry import box
from tqdm import tqdm

from . import insid3, raster
from .tables import export_geojson, make_meta, write_region_table

logger = logging.getLogger(__name__)
TILE_SIZE = insid3.IMAGE_SIZE


def build_tile_rows(
    image_path: str,
    perimeter_path: str | None,
    grid_offset=(0, 0),
    batch_size: int = 64,
):
    rows = []
    with rasterio.open(image_path) as src:
        perimeter = (
            raster.load_perimeter(perimeter_path, src.crs) if perimeter_path else None
        )
        windows = raster.tile_windows(src.height, src.width, TILE_SIZE, grid_offset)
        windows = raster.windows_in_perimeter(windows, perimeter, src.transform)
        batch: list[tuple] = []

        def flush():
            vectors = insid3.tile_embeddings(
                [raster.to_model_image(rgb) for rgb, _, _ in batch],
                [v for _, v, _ in batch],
            )
            for (_, _, window), vector in zip(batch, vectors):
                rows.append(
                    {
                        "geometry": box(
                            *rasterio.windows.bounds(window, src.transform)
                        ),
                        "dino_vector": vector,
                        "member_ids": [],
                    }
                )
            batch.clear()

        for window in tqdm(windows, desc="Embedding tiles"):
            tile = raster.read_tile(src, window, TILE_SIZE)
            if tile is None:
                continue
            batch.append((*tile, window))
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
        meta = make_meta(
            "tiles",
            src.crs,
            abs(src.res[0]),
            tile_size=TILE_SIZE,
            patch_size=insid3.PATCH_SIZE,
            grid_offset=list(grid_offset),
        )
    return rows, meta


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a naive one-embedding-per-tile index (baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", required=True, help="RGB(A) Cloud Optimised GeoTIFF, projected CRS"
    )
    p.add_argument(
        "--dino-weights", required=True, help="DINOv3 ViT-L/16 checkpoint (.pth)"
    )
    p.add_argument("--db", default="lancedb", help="LanceDB directory")
    p.add_argument("--table", required=True, help="Output tile table")
    p.add_argument(
        "--perimeter",
        default=None,
        help="GeoJSON (EPSG:4326): only tiles intersecting it",
    )
    p.add_argument(
        "--grid-offset-px", type=int, nargs=2, default=[0, 0], metavar=("X", "Y")
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--geojson",
        default=None,
        help="Also write the tiles to this GeoJSON (EPSG:4326)",
    )
    p.add_argument(
        "--no-cog-check", action="store_true", help="Skip strict COG validation"
    )
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    raster.validate_cog(args.image, strict=not args.no_cog_check)
    insid3.init(args.dino_weights)
    rows, meta = build_tile_rows(
        args.image, args.perimeter, tuple(args.grid_offset_px), args.batch_size
    )
    if not rows:
        raise SystemExit(
            "No tiles indexed: none inside the perimeter, or all mostly nodata"
        )
    Path(args.db).mkdir(parents=True, exist_ok=True)
    table = write_region_table(lancedb.connect(args.db), args.table, rows, meta)
    logger.info(f"Wrote {len(rows):,} tiles to '{args.table}'")
    if args.geojson:
        export_geojson(table, args.geojson)


if __name__ == "__main__":
    cli()
