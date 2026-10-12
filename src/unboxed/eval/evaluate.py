"""Evaluate example-based search + segmentation for one labelled class.

  1. Split the class's labelled polygons (inside the perimeter) into a few
     reference shots and a held-out evaluation set.
  2. Build the query from the reference shots and search the table's DINOv3
     vectors (regions or naive tiles).
  3. Score the ranking (head-independent): each ranked row is judged by the
     crop the head would see for it; an object is found at the first rank whose
     crop contains >= coverage_threshold of it. Object AP and recall@k.
  4. Segment the top-k results (with query merge for region tables, unless
     disabled) and score the masks against the held-out objects (instance
     F1 at --iou-threshold, mean best IoU per object, area IoU). Masks touching a
     reference object are ignored.

Writes metrics.json and GeoJSON layers (EPSG:4326) to --output-dir.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import lancedb
import numpy as np
import rasterio
from shapely.ops import unary_union

from .. import insid3, raster
from ..pipeline import SegmentConfig, min_patches_for, segment_hits
from ..references import build_query, split_references
from ..search import DEFAULT_MERGE_MAX_EXTENT_PX, DEFAULT_MERGE_QUANTILE, Index
from ..segment import downsample_limit, make_crops, make_head
from ..tables import DINO_COLUMN
from .metrics import area_metrics, instance_metrics, ranked_retrieval_metrics

logger = logging.getLogger(__name__)
KS = [20, 50, 100, 200]


def score_segmentation(instances: list, gts: list, iou_threshold: float) -> dict:
    inst = instance_metrics(instances, gts, iou_threshold)
    return {
        "instances": inst,
        "mean_best_iou": float(np.mean(inst["gt_best_iou"])) if gts else None,
        "area": area_metrics(unary_union(instances) if instances else None, gts),
        "num_masks": len(instances),
    }


def ranking_metrics(
    src,
    index: Index,
    hit_geoms: list,
    query,
    gts: list,
    ks: list[int],
    coverage_threshold: float,
    adaptive_downsample: bool = True,
) -> dict:
    """Ranking quality of the atomic hits, each judged by the crop it would be shown in."""
    size_factor = (
        downsample_limit(src, query)
        if adaptive_downsample and index.kind == "regions"
        else 1.0
    )
    _, footprints = make_crops(index.kind, src, hit_geoms, size_factor)
    return ranked_retrieval_metrics(hit_geoms, footprints, gts, ks, coverage_threshold)


def write_layers(out_dir: Path, src, hits, hit_geoms, result, gts_records) -> None:
    raster.write_geojson(
        out_dir / "results.geojson",
        [
            (g, {"rank": r + 1, "region_id": int(rid), "score": float(1 - d)})
            for r, (g, rid, d) in enumerate(
                zip(hit_geoms, hits["region_id"], hits["_distance"])
            )
        ],
        src.crs,
    )
    raster.write_geojson(
        out_dir / "areas.geojson",
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
        out_dir / "crops.geojson",
        [
            (raster.window_box(src, c.window), {"side_px": c.side()})
            for c in result.crops
        ],
        src.crs,
    )
    raster.write_geojson(
        out_dir / "masks.geojson",
        [(m, {"area": m.area}) for m in result.instances],
        src.crs,
    )
    raster.write_geojson(out_dir / "labels.geojson", gts_records, src.crs)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate example-based search + segmentation for one class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", help="The COG the table was built from")
    p.add_argument(
        "--labels", required=True, help="GeoJSON (EPSG:4326) of labelled polygons"
    )
    p.add_argument(
        "--label-property",
        default="label",
        help="Feature property holding the class name",
    )
    p.add_argument("--label", help="Class to evaluate")
    p.add_argument(
        "--list-labels", action="store_true", help="Print label counts and exit"
    )
    p.add_argument(
        "--perimeter", default=None, help="GeoJSON (EPSG:4326): evaluate inside it only"
    )
    p.add_argument("--db", default="lancedb")
    p.add_argument("--table", help="Region or tile table")
    p.add_argument("--dino-weights", help="DINOv3 ViT-L/16 checkpoint (.pth)")
    p.add_argument("--num-references", type=int, default=3)
    p.add_argument("--reference-seed", type=int, default=0)
    p.add_argument(
        "--reference-indices", type=int, nargs="*", help="Explicit reference indices"
    )
    p.add_argument("--top-k", type=int, default=200, help="Results to segment")
    p.add_argument(
        "--ks", type=int, nargs="+", default=KS, help="Cut-offs for ranking metrics"
    )
    p.add_argument("--no-query-merge", action="store_true")
    p.add_argument("--merge-quantile", type=float, default=DEFAULT_MERGE_QUANTILE)
    p.add_argument(
        "--merge-max-extent-px", type=float, default=DEFAULT_MERGE_MAX_EXTENT_PX
    )
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--coverage-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", required=False)
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    if args.list_labels:
        for label, n in raster.count_labels(
            args.labels, args.label_property
        ).most_common():
            print(f"{n:>7}  {label}")
        return
    for name in ("image", "label", "table", "dino_weights", "output_dir"):
        if not getattr(args, name):
            raise SystemExit(f"--{name.replace('_', '-')} is required")
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
        perimeter = (
            raster.load_perimeter(args.perimeter, src.crs) if args.perimeter else None
        )
        gts = raster.load_labels(
            args.labels, args.label, perimeter, src.crs, args.label_property
        )
        ref_idx, eval_idx = split_references(
            len(gts), args.num_references, args.reference_seed, args.reference_indices
        )
        refs, eval_gts = [gts[i] for i in ref_idx], [gts[i] for i in eval_idx]
        logger.info(
            f"'{args.label}': {len(gts)} objects, references {ref_idx}, {len(eval_gts)} held out"
        )

        query = build_query(src, args.label, refs)
        hits, hit_geoms = index.search(
            query.prototype,
            DINO_COLUMN,
            max(args.top_k, *args.ks),
            min_patches_for(index),
        )
        ranked = ranking_metrics(
            src, index, hit_geoms, query, eval_gts, args.ks, args.coverage_threshold
        )
        start = time.perf_counter()
        result = segment_hits(
            src,
            index,
            hits,
            hit_geoms,
            query.prototype,
            DINO_COLUMN,
            query,
            make_head(index.kind, model),
            cfg,
            clip=perimeter,
            exclude=refs,
        )
        seconds = time.perf_counter() - start
        write_layers(
            out,
            src,
            hits,
            hit_geoms,
            result,
            [
                (g, {"index": i, "role": "reference" if i in ref_idx else "eval"})
                for i, g in enumerate(gts)
            ],
        )

    metrics = {
        "label": args.label,
        "table": args.table,
        "kind": index.kind,
        "num_references": len(refs),
        "reference_indices": ref_idx,
        "num_eval_objects": len(eval_gts),
        "ranked": ranked,
        "segmentation": score_segmentation(
            result.instances, eval_gts, args.iou_threshold
        ),
        "num_areas": len(result.areas),
        "num_crops": len(result.crops),
        "patches_processed": int(
            sum((c.side() // insid3.PATCH_SIZE) ** 2 for c in result.crops)
        ),
        "crops_skipped_nodata": result.crops_skipped,
        "size_factor": result.size_factor,
        "segment_seconds": seconds,
        "config": vars(args),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    seg = metrics["segmentation"]
    print(
        f"{args.label}: object AP {ranked['object_ap']:.3f} | recall@50 "
        f"{ranked['at_k'].get(50, {}).get('object_recall', float('nan')):.3f} | F1 {seg['instances']['f1']:.3f} "
        f"| mean IoU {seg['mean_best_iou']:.3f} | {len(result.crops)} crops"
    )


if __name__ == "__main__":
    cli()
