"""Query pipeline: search -> (query merge) -> crops -> segmentation -> instances."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .references import Query
from .search import (
    DEFAULT_MERGE_MAX_EXTENT_PX,
    DEFAULT_MERGE_QUANTILE,
    DEFAULT_MIN_REGION_PATCHES,
    Area,
    Index,
    plain_areas,
    query_merge,
)
from .segment import Crop, downsample_limit, make_crops, run_head, to_instances


@dataclass
class SegmentConfig:
    top_k: int = 200
    query_merge: bool = True
    merge_quantile: float = DEFAULT_MERGE_QUANTILE
    merge_max_extent_px: float = DEFAULT_MERGE_MAX_EXTENT_PX
    adaptive_downsample: bool = True


@dataclass
class SegmentResult:
    areas: list[Area]
    crops: list[Crop]
    instances: list
    crops_skipped: int
    size_factor: float


def min_patches_for(
    index: Index, min_region_patches: float = DEFAULT_MIN_REGION_PATCHES
) -> float:
    """Tiny regions are excluded from search; tiles are never filtered."""
    return min_region_patches if index.kind == "regions" else 0.0


def segment_hits(
    src,
    index: Index,
    hits: pd.DataFrame,
    hit_geoms: list,
    vector: np.ndarray,
    column: str,
    query: Query,
    head,
    cfg: SegmentConfig,
    clip=None,
    exclude: list = (),
    desc: str = "segmenting",
) -> SegmentResult:
    """Segment the top cfg.top_k hits (merged by query if enabled, region tables only)."""
    top, top_geoms = hits.iloc[: cfg.top_k], hit_geoms[: cfg.top_k]
    if cfg.query_merge and index.kind == "regions":
        areas = query_merge(
            index, vector, top, column, cfg.merge_quantile, cfg.merge_max_extent_px
        )
    else:
        areas = plain_areas(top, top_geoms)
    size_factor = (
        downsample_limit(src, query)
        if cfg.adaptive_downsample and index.kind == "regions"
        else 1.0
    )
    crops, _ = make_crops(index.kind, src, [a.geometry for a in areas], size_factor)
    predictions, skipped = run_head(src, head, query, crops, desc)
    instances = to_instances(predictions, clip=clip, exclude=list(exclude))
    return SegmentResult(areas, crops, instances, skipped, size_factor)
