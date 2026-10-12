"""Retrieval and segmentation metrics computed on polygons (raster CRS)."""

import numpy as np
from shapely import STRtree
from shapely.ops import unary_union


def _iou(a, b) -> float:
    inter = a.intersection(b).area
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def coverage(gt, covering) -> float:
    """Fraction of gt's area covered by `covering`."""
    if covering is None or covering.is_empty or gt.area == 0:
        return 0.0
    return gt.intersection(covering).area / gt.area


def instance_metrics(preds: list, gts: list, iou_threshold: float) -> dict:
    """Greedy one-to-one matching of predicted and labelled polygons by IoU."""
    pairs = []
    if preds and gts:
        tree = STRtree(preds)
        for gi, gt in enumerate(gts):
            for pi in tree.query(gt, predicate="intersects"):
                pairs.append((_iou(preds[pi], gt), int(pi), gi))
    pairs.sort(reverse=True)

    best_iou = [0.0] * len(gts)
    for iou, _, gi in pairs:
        best_iou[gi] = max(best_iou[gi], iou)

    used_preds, used_gts, matched = set(), set(), []
    for iou, pi, gi in pairs:
        if iou < iou_threshold:
            break
        if pi in used_preds or gi in used_gts:
            continue
        used_preds.add(pi)
        used_gts.add(gi)
        matched.append(iou)

    tp = len(matched)
    fp = len(preds) - tp
    fn = len(gts) - tp
    precision = tp / (tp + fp) if preds else 0.0
    recall = tp / (tp + fn) if gts else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": sum(matched) / tp if tp else 0.0,
        "gt_best_iou": best_iou,
    }


def area_metrics(pred_union, gts: list) -> dict:
    """Pixel-style metrics on the union of predictions vs the union of labels."""
    gt_union = unary_union(gts) if gts else None
    pred_area = pred_union.area if pred_union is not None else 0.0
    gt_area = gt_union.area if gt_union is not None else 0.0
    inter = (
        pred_union.intersection(gt_union).area
        if pred_union is not None and gt_union is not None
        else 0.0
    )
    union = pred_area + gt_area - inter
    return {
        "iou": inter / union if union else 0.0,
        "precision": inter / pred_area if pred_area else 0.0,
        "recall": inter / gt_area if gt_area else 0.0,
        "pred_area_m2": pred_area,
        "gt_area_m2": gt_area,
    }


def ranked_retrieval_metrics(
    region_geoms: list,
    crop_geoms: list,
    gts: list,
    ks: list[int],
    coverage_threshold: float = 0.5,
    region_threshold: float = 0.5,
) -> dict:
    """Head-independent ranking metrics for a ranked list of retrieved items.

    Item r is region_geoms[r] (the indexed region) shown as crop_geoms[r] (the
    area the head would see). An object is *found* at the first rank whose
    single crop contains >= coverage_threshold of it.

    Returns, per k:
      object_recall      share of objects found within the top k
      crop_precision     share of top-k crops that contain a found object
      region_precision   share of top-k regions that are on target: >=
                         region_threshold of the region lies on objects, or the
                         region covers >= region_threshold of an object
      objects_per_crop   unique objects found in the top k, divided by k
    plus object AP (sum of crop_precision@r * delta object_recall@r) and the
    median rank at which objects are first found (None if most never are).
    """
    tree = STRtree(gts) if gts else None
    first_hit: list[int | None] = [None] * len(gts)
    crop_rel, region_rel = [], []
    for rank, (region, crop) in enumerate(zip(region_geoms, crop_geoms)):
        hit = False
        for gi in tree.query(crop, predicate="intersects") if tree else []:
            if coverage(gts[gi], crop) >= coverage_threshold:
                hit = True
                if first_hit[gi] is None:
                    first_hit[gi] = rank
        crop_rel.append(hit)

        near = list(tree.query(region, predicate="intersects")) if tree else []
        on_target = (
            min(
                1.0, sum(region.intersection(gts[gi]).area for gi in near) / region.area
            )
            if region.area
            else 0.0
        )
        contains = any(coverage(gts[gi], region) >= region_threshold for gi in near)
        region_rel.append(on_target >= region_threshold or contains)

    n_obj = max(len(gts), 1)
    found_by = np.full(len(region_geoms), 0)
    for r in first_hit:
        if r is not None:
            found_by[r] += 1
    found_cum = np.cumsum(found_by)
    crop_cum = np.cumsum(crop_rel)
    region_cum = np.cumsum(region_rel)

    ap = 0.0
    prev = 0.0
    for r in range(len(region_geoms)):
        recall = found_cum[r] / n_obj
        ap += (crop_cum[r] / (r + 1)) * (recall - prev)
        prev = recall

    ranks = sorted(r + 1 for r in first_hit if r is not None)
    median_rank = ranks[len(gts) // 2] if len(ranks) > len(gts) // 2 else None

    per_k = {}
    for k in ks:
        k_eff = min(k, len(region_geoms))
        if k_eff == 0:
            continue
        per_k[k] = {
            "object_recall": float(found_cum[k_eff - 1] / n_obj),
            "crop_precision": float(crop_cum[k_eff - 1] / k_eff),
            "region_precision": float(region_cum[k_eff - 1] / k_eff),
            "objects_per_crop": float(found_cum[k_eff - 1] / k_eff),
        }
    return {
        "object_ap": float(ap),
        "median_first_hit_rank": median_rank,
        "num_ranked": len(region_geoms),
        "at_k": per_k,
    }
