"""Score region (or tile / piece) tables as partitions against labelled objects.

Query-independent: how well regions line up with objects, i.e. what a head
working one region at a time could at best achieve. Per class, over labelled
objects inside the perimeter:
  completeness    largest share of an object inside a single region (1 = never
                  split); also the share of objects with completeness >= 0.9
  achievable IoU  IoU of an object with the union of regions lying mostly
                  (> 50% of their area) inside it
  UE              undersegmentation error: sum over overlapping regions of
                  min(inside, outside) area, divided by object area
Plus region size statistics.
"""

import argparse
import json
from pathlib import Path

import lancedb
import numpy as np
from rasterio.crs import CRS
from shapely import STRtree
from shapely.ops import unary_union

from .. import raster
from ..tables import parse_geometry, table_meta


def object_metrics(objects: list, regions: list, tree: STRtree) -> dict:
    completeness, achievable, ue = [], [], []
    for obj in objects:
        hits = [regions[i] for i in tree.query(obj, predicate="intersects")]
        inter = [obj.intersection(r).area for r in hits]
        completeness.append(max(inter, default=0.0) / obj.area)
        ue.append(sum(min(a, r.area - a) for a, r in zip(inter, hits)) / obj.area)
        inside = [r for a, r in zip(inter, hits) if a > 0.5 * r.area]
        if inside:
            sel = unary_union(inside)
            union = obj.union(sel).area
            achievable.append(obj.intersection(sel).area / union if union else 0.0)
        else:
            achievable.append(0.0)
    completeness = np.array(completeness)
    return {
        "num_objects": len(objects),
        "completeness_mean": float(completeness.mean()),
        "complete_share": float(np.mean(completeness >= 0.9)),
        "achievable_iou_mean": float(np.mean(achievable)),
        "undersegmentation_error": float(np.mean(ue)),
    }


def region_stats(regions: list, pixel_size: float) -> dict:
    extents = (
        np.array(
            [max(r.bounds[2] - r.bounds[0], r.bounds[3] - r.bounds[1]) for r in regions]
        )
        / pixel_size
    )
    return {
        "num_regions": len(regions),
        "median_area": float(np.median([r.area for r in regions])),
        "median_extent_px": float(np.median(extents)),
        "max_extent_px": float(extents.max()),
    }


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Score tables as partitions against labelled objects",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("tables", nargs="+", help="Region, tile or piece tables to score")
    p.add_argument("--db", default="lancedb")
    p.add_argument(
        "--labels", required=True, help="GeoJSON (EPSG:4326) of labelled polygons"
    )
    p.add_argument("--label-property", default="label")
    p.add_argument(
        "--perimeter", default=None, help="GeoJSON (EPSG:4326): objects inside it only"
    )
    p.add_argument("--output", default=None, help="Write results as JSON")
    return p


def cli() -> None:
    args = create_parser().parse_args()
    db = lancedb.connect(args.db)
    results = {}
    for name in args.tables:
        table = db.open_table(name)
        meta = table_meta(table)
        crs = CRS.from_wkt(meta["crs"])
        perimeter = (
            raster.load_perimeter(args.perimeter, crs) if args.perimeter else None
        )
        labels = {
            label: raster.load_labels(
                args.labels, label, perimeter, crs, args.label_property
            )
            for label in raster.count_labels(args.labels, args.label_property)
        }
        regions = [
            g
            for g in (
                parse_geometry(v)
                for v in table.to_arrow().column("mask_geojson").to_pylist()
            )
            if g
        ]
        tree = STRtree(regions)
        results[name] = {
            "regions": region_stats(regions, meta["pixel_size"]),
            "classes": {
                label: object_metrics(objs, regions, tree)
                for label, objs in labels.items()
                if objs
            },
        }

    width = max(len(t) for t in args.tables)
    for table, r in results.items():
        s = r["regions"]
        print(
            f"{table:{width}s}: {s['num_regions']:,} regions, median extent {s['median_extent_px']:.0f} px, "
            f"max {s['max_extent_px']:.0f} px"
        )
    for c in sorted({c for r in results.values() for c in r["classes"]}):
        print(f"\n{c}: completeness mean / >=0.9 | achievable IoU | UE")
        for table, r in results.items():
            m = r["classes"].get(c)
            if m:
                print(
                    f"  {table:{width}s} (n={m['num_objects']}) {m['completeness_mean']:.2f} / "
                    f"{m['complete_share']:.2f} | {m['achievable_iou_mean']:.2f} | {m['undersegmentation_error']:.2f}"
                )
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    cli()
