"""Unit tests that need no model weights or imagery."""

import numpy as np
import pandas as pd
import pytest
from rasterio.crs import CRS
from shapely.geometry import box

from unboxed import raster
from unboxed.eval.metrics import instance_metrics, ranked_retrieval_metrics
from unboxed.partition import partition_pieces
from unboxed.search import Index, query_merge
from unboxed.tables import DINO_COLUMN, make_meta, table_meta, write_region_table

PIXEL = 0.1  # metres per pixel
PATCH = 16


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def test_tile_windows_cover_raster_without_overlap():
    windows = raster.tile_windows(1000, 900, 448)
    starts = {(int(w.col_off), int(w.row_off)) for w in windows}
    assert len(windows) == len(starts) == 3 * 3
    # Edge tiles are shifted inwards to stay inside the raster
    assert max(int(w.col_off) + 448 for w in windows) == 900
    assert max(int(w.row_off) + 448 for w in windows) == 1000


def test_partition_joins_similar_neighbours_across_a_tile_seam():
    # Two pieces of one object either side of a tile seam, plus a dissimilar neighbour
    side = 10 * PATCH * PIXEL
    geoms = [
        box(0, 0, side, side),
        box(side, 0, 2 * side, side),
        box(2 * side, 0, 3 * side, side),
    ]
    vectors = np.stack([unit([1, 0.01, 0]), unit([1, 0, 0.01]), unit([0, 1, 0])])
    rows = partition_pieces(
        [10, 11, 12], geoms, vectors, PIXEL, PATCH, region_cost=12.25, max_extent_px=448
    )
    members = sorted(sorted(r["member_ids"]) for r in rows)
    assert members == [[10, 11], [12]]


def test_partition_respects_size_cap():
    side = 10 * PATCH * PIXEL  # 160 px each
    geoms = [box(i * side, 0, (i + 1) * side, side) for i in range(4)]
    vectors = np.stack([unit([1, 0, 0])] * 4)
    rows = partition_pieces(
        list(range(4)),
        geoms,
        vectors,
        PIXEL,
        PATCH,
        region_cost=12.25,
        max_extent_px=448,
    )
    # Identical pieces would all merge, but at most two 160 px pieces fit in 448 px
    assert all(len(r["member_ids"]) <= 2 for r in rows)
    extents = [
        (r["geometry"].bounds[2] - r["geometry"].bounds[0]) / PIXEL for r in rows
    ]
    assert max(extents) <= 448 + 1e-6


def test_instance_metrics_match_and_threshold():
    gts = [box(0, 0, 10, 10), box(20, 0, 30, 10)]
    preds = [box(0, 0, 10, 9), box(20, 0, 26, 10), box(50, 50, 60, 60)]
    m = instance_metrics(preds, gts, 0.5)
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0
    assert m["gt_best_iou"] == pytest.approx([0.9, 0.6])


def test_ranked_metrics_object_found_at_first_covering_crop():
    gts = [box(0, 0, 10, 10)]
    regions = [box(100, 100, 101, 101), box(0, 0, 5, 10)]
    crops = [box(100, 100, 150, 150), box(-5, -5, 20, 20)]
    r = ranked_retrieval_metrics(regions, crops, gts, [1, 2])
    assert r["at_k"][1]["object_recall"] == 0.0
    assert r["at_k"][2]["object_recall"] == 1.0
    assert r["object_ap"] == pytest.approx(0.5)  # found at rank 2, precision 1/2


@pytest.fixture
def region_table(tmp_path):
    import lancedb

    side = 10 * PATCH * PIXEL
    # A row of five regions: 0-1-2 belong to one large object, 3-4 are background
    rows = [
        {
            "geometry": box(i * side, 0, (i + 1) * side, side),
            "dino_vector": v,
            "member_ids": [i],
        }
        for i, v in enumerate(
            [
                unit([1, 0, 0]),
                unit([0.9, 0.3, 0]),
                unit([1, 0.1, 0]),
                unit([0, 0, 1]),
                unit([0, 1, 1]),
            ]
        )
    ]
    meta = make_meta(
        "regions", CRS.from_epsg(3857), PIXEL, tile_size=448, patch_size=PATCH
    )
    db = lancedb.connect(str(tmp_path))
    return write_region_table(db, "regions", rows, meta)


def test_region_table_metadata_round_trip(region_table):
    meta = table_meta(region_table)
    assert meta["kind"] == "regions" and meta["pixel_size"] == PIXEL
    assert CRS.from_wkt(meta["crs"]) == CRS.from_epsg(3857)


def test_query_merge_grows_hits_into_hot_neighbours(region_table):
    index = Index(region_table)
    query = unit([1, 0.05, 0])
    hits, _ = index.search(query, DINO_COLUMN, limit=1)
    # Top 60% of the five regions are "hot": regions 0, 1 and 2
    areas = query_merge(
        index, query, hits, DINO_COLUMN, quantile=0.4, max_extent_px=2000
    )
    assert len(areas) == 1
    assert sorted(areas[0].member_ids) == [0, 1, 2]


def test_query_merge_respects_extent_cap(region_table):
    index = Index(region_table)
    query = unit([1, 0.05, 0])
    hits, _ = index.search(query, DINO_COLUMN, limit=1)
    areas = query_merge(
        index, query, hits, DINO_COLUMN, quantile=0.4, max_extent_px=330
    )
    assert (
        len(areas[0].member_ids) == 2
    )  # two 160 px regions fit in 330 px, three do not


def test_plain_hits_are_not_merged(region_table):
    index = Index(region_table)
    hits, geoms = index.search(unit([1, 0.05, 0]), DINO_COLUMN, limit=3)
    assert isinstance(hits, pd.DataFrame) and len(geoms) == 3
