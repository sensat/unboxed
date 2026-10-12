"""Search a region (or tile) table by text or by example and segment the results.

    unboxed-query --image ortho.tif --table regions --text "car park" ...
    unboxed-query --image ortho.tif --table regions --references pools.geojson ...

Text queries search the SkyCLIP vectors; the top --pseudo-refs results (single
regions) then serve as the segmentation head's reference shots. Example
queries build a DINOv3 prototype from the reference polygons and search the
DINOv3 vectors; the same polygons are the head's reference shots.

Writes GeoJSON (EPSG:4326) to --output-dir: results (ranked hits), areas
(what was segmented, after query merge), crops and masks.
"""

import argparse
import json
import logging
from pathlib import Path

import lancedb
import rasterio

from . import insid3, raster, skyclip
from .pipeline import SegmentConfig, min_patches_for, segment_hits
from .references import build_query
from .search import DEFAULT_MERGE_MAX_EXTENT_PX, DEFAULT_MERGE_QUANTILE, Index
from .segment import make_head
from .tables import DINO_COLUMN, SKYCLIP_COLUMN

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", required=True, help="The COG the table was built from")
    p.add_argument("--db", default="lancedb")
    p.add_argument("--table", required=True, help="Region or tile table")
    p.add_argument("--dino-weights", required=True)
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument(
        "--text",
        help="Text prompt (needs --skyclip-weights and skyclip_vector in the table)",
    )
    q.add_argument("--references", help="GeoJSON (EPSG:4326) of example objects")
    p.add_argument("--skyclip-weights", default=None)
    p.add_argument(
        "--label",
        default=None,
        help="Name for the query (default: the text, or 'object')",
    )
    p.add_argument("--top-k", type=int, default=200, help="Results to segment")
    p.add_argument(
        "--pseudo-refs",
        type=int,
        default=3,
        help="Text queries: top results used as reference shots",
    )
    p.add_argument(
        "--no-query-merge",
        action="store_true",
        help="Segment hits as retrieved, without query merge",
    )
    p.add_argument("--merge-quantile", type=float, default=DEFAULT_MERGE_QUANTILE)
    p.add_argument(
        "--merge-max-extent-px", type=float, default=DEFAULT_MERGE_MAX_EXTENT_PX
    )
    p.add_argument(
        "--perimeter",
        default=None,
        help="Optional GeoJSON (EPSG:4326) to clip masks to",
    )
    p.add_argument("--output-dir", required=True)
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = insid3.init(args.dino_weights)
    index = Index(lancedb.connect(args.db).open_table(args.table))
    cfg = SegmentConfig(
        top_k=args.top_k,
        query_merge=not args.no_query_merge,
        merge_quantile=args.merge_quantile,
        merge_max_extent_px=args.merge_max_extent_px,
    )

    with rasterio.open(args.image) as src:
        clip = (
            raster.load_perimeter(args.perimeter, src.crs) if args.perimeter else None
        )
        if args.text:
            if not args.skyclip_weights:
                raise SystemExit("--text needs --skyclip-weights")
            skyclip.init(args.skyclip_weights)
            vector, column = skyclip.text_embedding(args.text), SKYCLIP_COLUMN
            hits, geoms = index.search(
                vector, column, args.top_k, min_patches_for(index)
            )
            query = build_query(src, args.label or args.text, geoms[: args.pseudo_refs])
        else:
            refs = raster.load_geometries(args.references, src.crs)
            query = build_query(src, args.label or "object", refs)
            vector, column = query.prototype, DINO_COLUMN
            hits, geoms = index.search(
                vector, column, args.top_k, min_patches_for(index)
            )

        result = segment_hits(
            src,
            index,
            hits,
            geoms,
            vector,
            column,
            query,
            make_head(index.kind, model),
            cfg,
            clip=clip,
        )
        raster.write_geojson(
            out / "results.geojson",
            [
                (g, {"rank": r + 1, "region_id": int(rid), "score": float(1 - d)})
                for r, (g, rid, d) in enumerate(
                    zip(geoms, hits["region_id"], hits["_distance"])
                )
            ],
            src.crs,
        )
        raster.write_geojson(
            out / "areas.geojson",
            [
                (
                    a.geometry,
                    {
                        "rank": a.rank,
                        "region_id": a.region_id,
                        "num_regions": len(a.member_ids),
                    },
                )
                for a in result.areas
            ],
            src.crs,
        )
        raster.write_geojson(
            out / "crops.geojson",
            [
                (raster.window_box(src, c.window), {"side_px": c.side()})
                for c in result.crops
            ],
            src.crs,
        )
        raster.write_geojson(
            out / "masks.geojson",
            [(m, {"area": m.area}) for m in result.instances],
            src.crs,
        )
    summary = {
        "query": args.text or args.references,
        "table": args.table,
        "kind": index.kind,
        "hits": len(hits),
        "areas": len(result.areas),
        "crops": len(result.crops),
        "masks": len(result.instances),
        "crops_skipped_nodata": result.crops_skipped,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(summary)


if __name__ == "__main__":
    cli()
