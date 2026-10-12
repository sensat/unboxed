"""Evaluate text search (SkyCLIP) + segmentation (DINOv3 / INSID3) per class.

For each class and each prompt from skyclip.generate_prompts:
  1. Embed the prompt with SkyCLIP and search the table's skyclip_vector.
  2. Score the ranking against the class's labelled polygons (object AP,
     recall@k; head-independent, as in unboxed-eval).
  3. Segment the top-k results with the INSID3 head (with query merge for region
     tables, unless disabled). Reference shots for the head are either
     labelled objects (--reference-seeds; held out from scoring, one run per
     seed) or the top --pseudo-refs retrieved regions (no labels used).

Per-run metrics.json files go to --output-dir/<class>_p<prompt>[_seed<s>]/ and
a summary over prompts (and seeds) to --output-dir/summary.json.
"""

import argparse
import json
import logging
from pathlib import Path

import lancedb
import numpy as np
import rasterio

from .. import insid3, raster, skyclip
from ..pipeline import SegmentConfig, min_patches_for, segment_hits
from ..references import build_query, split_references
from ..search import DEFAULT_MERGE_MAX_EXTENT_PX, DEFAULT_MERGE_QUANTILE, Index
from ..segment import make_head
from ..tables import SKYCLIP_COLUMN
from .evaluate import KS, ranking_metrics, score_segmentation, write_layers

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate text search + segmentation per class",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--image", required=True, help="The COG the table was built from")
    p.add_argument(
        "--labels", required=True, help="GeoJSON (EPSG:4326) of labelled polygons"
    )
    p.add_argument("--label-property", default="label")
    p.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names (also used as prompt subjects)",
    )
    p.add_argument(
        "--perimeter", default=None, help="GeoJSON (EPSG:4326): evaluate inside it only"
    )
    p.add_argument("--db", default="lancedb")
    p.add_argument(
        "--table", required=True, help="Region or tile table with skyclip_vector"
    )
    p.add_argument("--dino-weights", required=True)
    p.add_argument("--skyclip-weights", required=True)
    p.add_argument(
        "--reference-seeds",
        type=int,
        nargs="*",
        default=[],
        help="Segment with labelled references (one run per seed); default: pseudo-references",
    )
    p.add_argument(
        "--num-references",
        type=int,
        nargs="*",
        default=[],
        help="Labelled references per class (same order as --classes; default 3 each)",
    )
    p.add_argument(
        "--pseudo-refs",
        type=int,
        default=3,
        help="Top results used as references without labels",
    )
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--ks", type=int, nargs="+", default=KS)
    p.add_argument("--no-query-merge", action="store_true")
    p.add_argument("--merge-quantile", type=float, default=DEFAULT_MERGE_QUANTILE)
    p.add_argument(
        "--merge-max-extent-px", type=float, default=DEFAULT_MERGE_MAX_EXTENT_PX
    )
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--coverage-threshold", type=float, default=0.5)
    p.add_argument("--output-dir", required=True)
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    out_root = Path(args.output_dir)
    model = insid3.init(args.dino_weights)
    skyclip.init(args.skyclip_weights)
    index = Index(lancedb.connect(args.db).open_table(args.table))
    head = make_head(index.kind, model)
    cfg = SegmentConfig(
        top_k=args.top_k,
        query_merge=not args.no_query_merge,
        merge_quantile=args.merge_quantile,
        merge_max_extent_px=args.merge_max_extent_px,
    )
    if args.num_references and len(args.num_references) != len(args.classes):
        raise SystemExit("--num-references needs one value per class in --classes")
    n_refs = dict(zip(args.classes, args.num_references or [3] * len(args.classes)))
    summary = {}

    with rasterio.open(args.image) as src:
        perimeter = (
            raster.load_perimeter(args.perimeter, src.crs) if args.perimeter else None
        )
        for label in args.classes:
            all_gts = raster.load_labels(
                args.labels, label, perimeter, src.crs, args.label_property
            )
            if not all_gts:
                logger.warning(
                    f"No labelled '{label}' objects (inside the perimeter); skipping"
                )
                continue
            if args.reference_seeds:
                setups = []
                for seed in args.reference_seeds:
                    ref_idx, held = split_references(len(all_gts), n_refs[label], seed)
                    refs = [all_gts[i] for i in ref_idx]
                    setups.append(
                        (
                            f"_seed{seed}",
                            refs,
                            [all_gts[i] for i in held],
                            build_query(src, label, refs),
                        )
                    )
            else:
                setups = [("", [], all_gts, None)]
            runs = []
            for suffix, refs, gts, labelled_query in setups:
                for p_idx, prompt in enumerate(skyclip.generate_prompts(label)):
                    out = out_root / f"{label.replace(' ', '_')}_p{p_idx}{suffix}"
                    out.mkdir(parents=True, exist_ok=True)
                    vector = skyclip.text_embedding(prompt)
                    hits, geoms = index.search(
                        vector,
                        SKYCLIP_COLUMN,
                        max(args.top_k, *args.ks),
                        min_patches_for(index),
                    )
                    query = labelled_query or build_query(
                        src, label, geoms[: args.pseudo_refs]
                    )
                    ranked = ranking_metrics(
                        src, index, geoms, query, gts, args.ks, args.coverage_threshold
                    )
                    result = segment_hits(
                        src,
                        index,
                        hits,
                        geoms,
                        vector,
                        SKYCLIP_COLUMN,
                        query,
                        head,
                        cfg,
                        clip=perimeter,
                        exclude=refs,
                        desc=f"{label} | {prompt}",
                    )
                    write_layers(
                        out,
                        src,
                        hits,
                        geoms,
                        result,
                        [(g, {"index": i}) for i, g in enumerate(gts)],
                    )
                    m = {
                        "label": label,
                        "prompt": prompt,
                        "num_eval_objects": len(gts),
                        "references": "labelled"
                        if refs
                        else f"top {args.pseudo_refs} results",
                        "ranked": ranked,
                        "segmentation": score_segmentation(
                            result.instances, gts, args.iou_threshold
                        ),
                        "num_crops": len(result.crops),
                        "num_areas": len(result.areas),
                    }
                    (out / "metrics.json").write_text(
                        json.dumps(m, indent=2, default=str)
                    )
                    runs.append(m)
                    logger.info(
                        f"{label} | '{prompt}'{suffix}: object AP {ranked['object_ap']:.2f}, "
                        f"mean IoU {m['segmentation']['mean_best_iou']:.2f}"
                    )
            summary[label] = {
                "objects": runs[0]["num_eval_objects"],
                "runs": len(runs),
                "object_ap": float(np.mean([r["ranked"]["object_ap"] for r in runs])),
                "recall@50": float(
                    np.mean(
                        [
                            r["ranked"]["at_k"].get(50, {}).get("object_recall", np.nan)
                            for r in runs
                        ]
                    )
                ),
                "f1": float(
                    np.mean([r["segmentation"]["instances"]["f1"] for r in runs])
                ),
                "mean_iou": float(
                    np.mean([r["segmentation"]["mean_best_iou"] for r in runs])
                ),
            }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"{'class':20s} {'objects':>8s} {'AP':>6s} {'R@50':>6s} {'F1':>6s} {'mIoU':>6s}"
    )
    for label, s in summary.items():
        print(
            f"{label:20s} {s['objects']:8d} {s['object_ap']:6.2f} {s['recall@50']:6.2f} {s['f1']:6.2f} {s['mean_iou']:6.2f}"
        )


if __name__ == "__main__":
    cli()
