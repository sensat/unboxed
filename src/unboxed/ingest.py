"""Ingestion: COG -> per-tile pieces table.

Each TILE_SIZE (448 px) tile is read at native resolution and passed through
DINOv3. Its patches are clustered (INSID3 agglomerative clustering) and every
cluster is split into spatially connected pieces; each piece is stored with its
mean debiased DINOv3 feature and its polygon (raster CRS). The pieces table is
the input to `unboxed-partition`.
"""

import argparse
import logging

import rasterio
from tqdm import tqdm

from . import insid3, raster
from .tables import PiecesWriter, make_meta

logger = logging.getLogger(__name__)
TILE_SIZE = insid3.IMAGE_SIZE


def ingest(
    image_path: str,
    db_path: str,
    table_name: str,
    perimeter_path: str | None = None,
    grid_offset: tuple[int, int] = (0, 0),
    batch_size: int = 64,
    split_disjoint: bool = True,
) -> int:
    with rasterio.open(image_path) as src:
        perimeter = (
            raster.load_perimeter(perimeter_path, src.crs) if perimeter_path else None
        )
        windows = raster.tile_windows(src.height, src.width, TILE_SIZE, grid_offset)
        windows = raster.windows_in_perimeter(windows, perimeter, src.transform)
        logger.info(f"{len(windows):,} tiles to process")
        writer = PiecesWriter(
            db_path,
            table_name,
            make_meta(
                "pieces",
                src.crs,
                abs(src.res[0]),
                tile_size=TILE_SIZE,
                patch_size=insid3.PATCH_SIZE,
                grid_offset=list(grid_offset),
            ),
        )
        batch: list[tuple] = []

        def flush():
            images = [raster.to_model_image(rgb) for rgb, _, _ in batch]
            pieces_per_tile = insid3.cluster_pieces(
                images, [v for _, v, _ in batch], split_disjoint
            )
            vectors, geoms = [], []
            for (_, _, window), pieces in zip(batch, pieces_per_tile):
                bounds = rasterio.windows.bounds(window, src.transform)
                for piece in pieces:
                    tf = raster.tile_pixel_transform(
                        bounds, src.transform, piece.mask.shape
                    )
                    geom = raster.mask_to_geometry(piece.mask, tf)
                    if geom is not None:
                        vectors.append(piece.vector)
                        geoms.append(geom)
            writer.add(vectors, geoms)
            batch.clear()

        for window in tqdm(windows, desc="Ingesting tiles"):
            tile = raster.read_tile(src, window, TILE_SIZE)
            if tile is None:
                continue
            batch.append((*tile, window))
            if len(batch) >= batch_size:
                flush()
        if batch:
            flush()
    if writer.count == 0:
        raise SystemExit(
            "No pieces produced: no tiles inside the perimeter, or all tiles mostly nodata"
        )
    logger.info(f"Wrote {writer.count:,} pieces to '{table_name}'")
    return writer.count


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cluster each 448 px tile of a COG into pieces (DINOv3 / INSID3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--image", required=True, help="RGB(A) Cloud Optimised GeoTIFF, projected CRS"
    )
    p.add_argument(
        "--dino-weights", required=True, help="DINOv3 ViT-L/16 checkpoint (.pth)"
    )
    p.add_argument("--db", default="lancedb", help="LanceDB directory")
    p.add_argument("--table", required=True, help="Output pieces table")
    p.add_argument(
        "--perimeter",
        default=None,
        help="GeoJSON (EPSG:4326): only tiles intersecting it",
    )
    p.add_argument(
        "--grid-offset-px",
        type=int,
        nargs=2,
        default=[0, 0],
        metavar=("X", "Y"),
        help="Shift the tile grid origin by X, Y pixels",
    )
    p.add_argument("--batch-size", type=int, default=64, help="Tiles per DINOv3 batch")
    p.add_argument(
        "--keep-disjoint-clusters",
        action="store_true",
        help="Keep each cluster as one piece instead of splitting it into connected pieces",
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
    ingest(
        args.image,
        args.db,
        args.table,
        args.perimeter,
        tuple(args.grid_offset_px),
        args.batch_size,
        not args.keep_disjoint_clusters,
    )


if __name__ == "__main__":
    cli()
