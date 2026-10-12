"""Add SkyCLIP image embeddings to a region or tile table (combined index).

Each row is embedded from a square crop centred on its geometry: the geometry's
extent at native resolution, at least MIN_CROP_PX (SkyCLIP's input size), so
small regions keep surrounding context. SkyCLIP's preprocessing resizes the
crop to 224 px. The vectors are stored in the same table as `skyclip_vector`,
next to `dino_vector`, so text and example queries search the same rows.
"""

import argparse
import logging
import math

import lancedb
import rasterio
from tqdm import tqdm

from . import raster, skyclip
from .tables import SKYCLIP_COLUMN, add_vector_column, parse_geometry

logger = logging.getLogger(__name__)
MIN_CROP_PX = 224


def crop_window(src, geom, min_crop_px: int = MIN_CROP_PX):
    size = max(int(math.ceil(raster.extent_px(src, geom.bounds))), min_crop_px)
    return raster.window_around_bounds(
        src, geom.bounds, min(size, src.width, src.height)
    )


def skyclip_vectors(image_path: str, geoms: list, batch_size: int = 128) -> list:
    vectors, batch = [], []
    with rasterio.open(image_path) as src:
        for geom in tqdm(geoms, desc="SkyCLIP embedding"):
            batch.append(
                raster.to_model_image(raster.read_rgb(src, crop_window(src, geom)))
            )
            if len(batch) >= batch_size:
                vectors += skyclip.image_embeddings(batch)
                batch = []
        if batch:
            vectors += skyclip.image_embeddings(batch)
    return vectors


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add SkyCLIP embeddings (skyclip_vector) to a region or tile table",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="The COG the table was built from")
    p.add_argument(
        "--skyclip-weights",
        required=True,
        help="SkyCLIP ViT-L/14 checkpoint (OpenCLIP format)",
    )
    p.add_argument("--db", default="lancedb", help="LanceDB directory")
    p.add_argument("--table", required=True, help="Region or tile table to update")
    p.add_argument("--batch-size", type=int, default=128)
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    skyclip.init(args.skyclip_weights)
    db = lancedb.connect(args.db)
    geoms = [
        parse_geometry(g) for g in db.open_table(args.table).to_pandas()["mask_geojson"]
    ]
    vectors = skyclip_vectors(args.image, geoms, args.batch_size)
    add_vector_column(db, args.table, SKYCLIP_COLUMN, vectors)
    logger.info(f"Added {len(vectors):,} SkyCLIP vectors to '{args.table}'")


if __name__ == "__main__":
    cli()
