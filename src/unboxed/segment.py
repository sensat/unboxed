"""Segmentation of retrieved areas: crops -> INSID3 head -> polygon instances.

Region tables use grouped native-resolution crops: walking down the ranking,
an area already mostly visible in earlier crops is skipped; otherwise nearby
later-ranked areas are grouped with it while the group fits the largest crop
size (times the query's downsample limit). Each group is read at the smallest
crop size that holds it, at native resolution; only groups larger than the
largest size are downsampled, by at most the query's limit, which keeps the
reference objects at >= min_object_patches DINOv3 patches.

Tile tables use one 448 px crop per tile (the tile itself).
"""

from dataclasses import dataclass

import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
from tqdm import tqdm

from . import insid3, raster
from .heads import INSID3Head, NativeScaleINSID3Head, head_model
from .references import REFERENCE_CROP_PX, Query

CROP_SIZES = (448, 672, 896)
SKIP_VISIBLE = 0.9
GROUP_NEAR_PX = 16
MIN_OBJECT_PATCHES = 8
MAX_DOWNSAMPLE = 2.0


@dataclass
class Crop:
    window: object
    out_size: int | None = None  # side the window is resampled to (None = native)

    def side(self) -> int:
        return self.out_size or int(self.window.width)


def downsample_limit(
    src,
    query: Query,
    min_object_patches: float = MIN_OBJECT_PATCHES,
    max_downsample: float = MAX_DOWNSAMPLE,
) -> float:
    """How far crops may be downsampled: the median reference must keep >= min_object_patches patches."""
    extents = [raster.extent_px(src, r.geometry.bounds) for r in query.references]
    return float(
        np.clip(
            np.median(extents) / (min_object_patches * insid3.PATCH_SIZE),
            1.0,
            max_downsample,
        )
    )


def group_crops(
    src,
    geoms: list,
    size_factor: float = 1.0,
    sizes=CROP_SIZES,
    skip_visible: float = SKIP_VISIBLE,
    near_px: float = GROUP_NEAR_PX,
) -> tuple[list[Crop], list]:
    """Grouped native-resolution crops for ranked geometries; also each one's crop footprint."""
    sizes = sorted(sizes)
    max_size, res = sizes[-1], abs(src.res[0])
    n = len(geoms)
    assigned = [False] * n
    footprints: list = [None] * n
    crops: list[Crop] = []
    shown: list = []
    for i in range(n):
        if assigned[i]:
            continue
        geom = geoms[i]
        if skip_visible > 0 and shown and geom.area > 0:
            overlaps = [fp.intersection(geom).area for fp in shown]
            best = int(np.argmax(overlaps))
            if overlaps[best] >= skip_visible * geom.area or (
                unary_union(shown).intersection(geom).area >= skip_visible * geom.area
            ):
                assigned[i] = True
                footprints[i] = shown[best]
                continue
        members, b = [i], list(geom.bounds)
        for j in range(i + 1, n):
            if assigned[j] or box(*b).distance(geoms[j]) > near_px * res:
                continue
            gb = geoms[j].bounds
            nb = [
                min(b[0], gb[0]),
                min(b[1], gb[1]),
                max(b[2], gb[2]),
                max(b[3], gb[3]),
            ]
            if raster.extent_px(src, nb) <= max_size * size_factor + 1e-3:
                members.append(j)
                b = nb
        ext = raster.extent_px(src, b)
        size = next((s for s in sizes if ext <= s + 1e-3), None)
        if size is not None:
            crop = Crop(raster.window_around_bounds(src, b, size))
        else:
            side = min(int(np.ceil(ext)), src.width, src.height)
            crop = Crop(raster.window_around_bounds(src, b, side), out_size=max_size)
        footprint = raster.window_box(src, crop.window)
        crops.append(crop)
        shown.append(footprint)
        for m in members:
            assigned[m] = True
            footprints[m] = footprint
    return crops, footprints


def tile_crops(
    src, geoms: list, crop_px: int = insid3.IMAGE_SIZE
) -> tuple[list[Crop], list]:
    """One crop_px crop per ranked geometry (duplicates removed); also each one's footprint."""
    crops: dict[tuple, Crop] = {}
    footprints = []
    for geom in geoms:
        window = raster.window_covering(src, geom.bounds, crop_px)
        key = (int(window.col_off), int(window.row_off), int(window.width))
        crops.setdefault(key, Crop(window, out_size=crop_px))
        footprints.append(raster.window_box(src, window))
    return list(crops.values()), footprints


def make_crops(
    kind: str, src, geoms: list, size_factor: float = 1.0
) -> tuple[list[Crop], list]:
    return (
        group_crops(src, geoms, size_factor)
        if kind == "regions"
        else tile_crops(src, geoms)
    )


def make_head(kind: str, model):
    if kind == "regions":
        return NativeScaleINSID3Head(model, ref_size=REFERENCE_CROP_PX)
    return INSID3Head(head_model(model, insid3.IMAGE_SIZE))


def run_head(
    src, head, query: Query, crops: list[Crop], desc: str = "segmenting"
) -> tuple[list, int]:
    """Predicted polygons (raster CRS) for each crop; also the number of crops skipped as nodata."""
    predictions, skipped = [], 0
    for crop in tqdm(crops, desc=desc):
        if (
            raster.nodata_ratio(src, crop.window, crop.out_size)
            > raster.NODATA_THRESHOLD
        ):
            skipped += 1
            continue
        rgb = raster.read_rgb(src, crop.window, crop.out_size)
        mask = head.predict(raster.to_model_image(rgb), query)
        geom = raster.mask_to_geometry(
            mask, raster.output_transform(src, crop.window, crop.out_size)
        )
        if geom is not None and not geom.is_empty:
            predictions.append(geom)
    return predictions, skipped


def to_instances(
    predictions: list, clip=None, exclude: list = (), min_area: float = 0.0
) -> list:
    """Union overlapping predictions, clip, split into connected polygons, drop excluded ones."""
    if not predictions:
        return []
    union = unary_union(predictions)
    if clip is not None:
        union = union.intersection(clip)
    return [
        p
        for p in raster.explode(union)
        if p.area >= min_area and not any(p.intersects(e) for e in exclude)
    ]
