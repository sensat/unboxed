"""Cross-tile partition: per-tile pieces -> size-capped regions.

Pieces (from `unboxed-ingest`) are merged across tile boundaries into connected
regions that each fit one segmentation crop (bbox <= max_extent_px native
pixels per side), as internally coherent as possible:

    minimise   sum_R SSE(R) + region_cost * |R|
    SSE(R) = sum_{i in R} w_i * ||v_i - mu_R||^2,   mu_R = sum w_i v_i / sum w_i

v_i are unit piece embeddings and w_i piece sizes in DINOv3 patches. Merging
two regions changes the objective by (Ward cost - region_cost), so region_cost
is the single knob: higher gives fewer, larger regions.

  1. Greedy Ward agglomeration on the piece adjacency graph, rejecting merges
     whose bbox would exceed the size limit, until the cheapest merge costs
     more than region_cost.
  2. Boundary refinement: move single boundary pieces to a neighbouring region
     when that lowers the objective, keeps both regions connected and keeps the
     target within the size limit.

A region's DINOv3 vector is the size-weighted mean of its pieces' vectors.
"""

import argparse
import heapq
import logging
from collections import deque

import lancedb
import numpy as np
from rasterio.crs import CRS
from shapely import STRtree
from shapely.ops import unary_union

from .tables import (
    DINO_COLUMN,
    export_geojson,
    make_meta,
    parse_geometry,
    table_meta,
    write_region_table,
)

logger = logging.getLogger(__name__)

DEFAULT_REGION_COST = 12.25
DEFAULT_MAX_EXTENT_PX = 448


class Partition:
    """Mutable partition of pieces into regions with O(1) objective deltas."""

    def __init__(self, vectors, weights, bounds, neighbours, max_extent, region_cost):
        self.v = vectors  # (N, C) unit piece embeddings
        self.w = weights  # (N,) piece sizes in patches
        self.bounds = bounds  # (N, 4) piece bboxes
        self.nbrs = neighbours  # list[set[int]] piece adjacency
        self.max_extent = max_extent
        self.region_cost = region_cost
        n = len(weights)
        self.region_of = np.arange(n)
        self.members: dict[int, set[int]] = {i: {i} for i in range(n)}
        self.sum: dict[int, np.ndarray] = {i: weights[i] * vectors[i] for i in range(n)}
        self.weight: dict[int, float] = {i: float(weights[i]) for i in range(n)}
        self.bbox: dict[int, np.ndarray] = {i: bounds[i].copy() for i in range(n)}

    @staticmethod
    def sse(total_weight: float, weighted_sum: np.ndarray) -> float:
        # sum w ||v - mu||^2 = W - ||S||^2 / W  for unit vectors v
        if total_weight <= 0:
            return 0.0
        return total_weight - float(weighted_sum @ weighted_sum) / total_weight

    def ward_cost(self, a: int, b: int) -> float:
        wa, wb = self.weight[a], self.weight[b]
        diff = self.sum[a] / wa - self.sum[b] / wb
        return wa * wb / (wa + wb) * float(diff @ diff)

    def objective(self) -> float:
        return sum(
            self.sse(self.weight[r], self.sum[r]) for r in self.members
        ) + self.region_cost * len(self.members)

    def fits(self, box: np.ndarray) -> bool:
        return (
            box[2] - box[0] <= self.max_extent + 1e-6
            and box[3] - box[1] <= self.max_extent + 1e-6
        )

    @staticmethod
    def union_box(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.array(
            [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        )

    def region_neighbours(self, r: int) -> set[int]:
        return {self.region_of[n] for m in self.members[r] for n in self.nbrs[m]} - {r}

    # ──────── Stage 1: constrained Ward agglomeration ────────

    def agglomerate(self) -> int:
        heap: list[tuple[float, int, int]] = []
        for a in self.members:
            for b in self.region_neighbours(a):
                if a < b and self.fits(self.union_box(self.bbox[a], self.bbox[b])):
                    heapq.heappush(heap, (self.ward_cost(a, b), a, b))
        next_id = len(self.region_of)
        merges = 0
        while heap:
            cost, a, b = heapq.heappop(heap)
            if cost > self.region_cost:
                break
            if a not in self.members or b not in self.members:
                continue
            new = next_id
            next_id += 1
            self.members[new] = self.members.pop(a) | self.members.pop(b)
            self.sum[new] = self.sum.pop(a) + self.sum.pop(b)
            self.weight[new] = self.weight.pop(a) + self.weight.pop(b)
            self.bbox[new] = self.union_box(self.bbox.pop(a), self.bbox.pop(b))
            for m in self.members[new]:
                self.region_of[m] = new
            # Regions only grow, so a pair that is too big now stays too big
            for other in self.region_neighbours(new):
                if self.fits(self.union_box(self.bbox[new], self.bbox[other])):
                    heapq.heappush(heap, (self.ward_cost(new, other), other, new))
            merges += 1
        return merges

    # ──────── Stage 2: boundary refinement ────────

    def _connected_without(self, r: int, piece: int) -> bool:
        rest = self.members[r] - {piece}
        if len(rest) <= 1:
            return True
        start = next(iter(rest))
        seen = {start}
        queue = deque([start])
        while queue:
            m = queue.popleft()
            for n in self.nbrs[m]:
                if n in rest and n not in seen:
                    seen.add(n)
                    queue.append(n)
        return len(seen) == len(rest)

    def _bbox_of(self, pieces) -> np.ndarray:
        b = self.bounds[list(pieces)]
        return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])

    def _move_delta(self, piece: int, src: int, dst: int) -> float:
        wp, vp = self.w[piece], self.w[piece] * self.v[piece]
        before = self.sse(self.weight[src], self.sum[src]) + self.sse(
            self.weight[dst], self.sum[dst]
        )
        after = self.sse(self.weight[src] - wp, self.sum[src] - vp) + self.sse(
            self.weight[dst] + wp, self.sum[dst] + vp
        )
        delta = after - before
        if len(self.members[src]) == 1:
            delta -= self.region_cost  # source region disappears
        return delta

    def refine(self, max_passes: int) -> int:
        moves = 0
        for pass_idx in range(max_passes):
            pass_moves = 0
            for piece in range(len(self.region_of)):
                src = self.region_of[piece]
                candidates = {self.region_of[n] for n in self.nbrs[piece]} - {src}
                best, best_delta = None, -1e-9
                for dst in candidates:
                    delta = self._move_delta(piece, src, dst)
                    if delta < best_delta and self.fits(
                        self.union_box(self.bbox[dst], self.bounds[piece])
                    ):
                        best, best_delta = dst, delta
                if best is None or not self._connected_without(src, piece):
                    continue
                wp, vp = self.w[piece], self.w[piece] * self.v[piece]
                self.members[src].discard(piece)
                self.members[best].add(piece)
                self.region_of[piece] = best
                self.sum[best] = self.sum[best] + vp
                self.weight[best] += wp
                self.bbox[best] = self.union_box(self.bbox[best], self.bounds[piece])
                if self.members[src]:
                    self.sum[src] = self.sum[src] - vp
                    self.weight[src] -= wp
                    self.bbox[src] = self._bbox_of(self.members[src])
                else:
                    for d in (self.members, self.sum, self.weight, self.bbox):
                        d.pop(src)
                pass_moves += 1
            moves += pass_moves
            logger.info(f"Refinement pass {pass_idx + 1}: {pass_moves} moves")
            if pass_moves == 0:
                break
        return moves


def partition_pieces(
    piece_ids: list[int],
    geoms: list,
    vectors: np.ndarray,
    pixel_size: float,
    patch_size_px: int,
    region_cost: float = DEFAULT_REGION_COST,
    max_extent_px: float = DEFAULT_MAX_EXTENT_PX,
    refine_passes: int = 10,
) -> list[dict]:
    """Partition pieces into regions; returns rows for tables.write_region_table."""
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    patch_area = (patch_size_px * pixel_size) ** 2
    weights = np.array([g.area / patch_area for g in geoms])
    bounds = np.array([g.bounds for g in geoms])

    tree = STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    neighbours: list[set[int]] = [set() for _ in geoms]
    for a, b in zip(left, right):
        if a != b:
            neighbours[a].add(int(b))

    part = Partition(
        vectors, weights, bounds, neighbours, max_extent_px * pixel_size, region_cost
    )
    logger.info(f"{len(geoms):,} pieces, objective {part.objective():.1f}")
    merges = part.agglomerate()
    logger.info(
        f"Agglomeration: {merges:,} merges -> {len(part.members):,} regions, objective {part.objective():.1f}"
    )
    moves = part.refine(refine_passes)
    logger.info(
        f"Refinement: {moves:,} moves -> {len(part.members):,} regions, objective {part.objective():.1f}"
    )

    rows = []
    for region_id, members in part.members.items():
        rows.append(
            {
                "geometry": unary_union([geoms[m] for m in members]),
                "dino_vector": part.sum[region_id]
                / np.linalg.norm(part.sum[region_id]),
                "member_ids": sorted(piece_ids[m] for m in members),
            }
        )
    return rows


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge per-tile pieces into size-capped cross-tile regions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="lancedb", help="LanceDB directory")
    p.add_argument(
        "--pieces-table", required=True, help="Input pieces table (unboxed-ingest)"
    )
    p.add_argument("--table", required=True, help="Output region table")
    p.add_argument(
        "--region-cost",
        type=float,
        default=DEFAULT_REGION_COST,
        help="Objective cost per region; higher gives fewer, larger regions",
    )
    p.add_argument(
        "--max-extent-px",
        type=float,
        default=DEFAULT_MAX_EXTENT_PX,
        help="Maximum region bbox side in native pixels",
    )
    p.add_argument("--refine-passes", type=int, default=10)
    p.add_argument(
        "--geojson",
        default=None,
        help="Also write the regions to this GeoJSON (EPSG:4326)",
    )
    return p


def cli() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = create_parser().parse_args()
    db = lancedb.connect(args.db)
    pieces = db.open_table(args.pieces_table)
    meta = table_meta(pieces)
    df = pieces.to_pandas()
    geoms = [parse_geometry(g) for g in df["mask_geojson"]]
    keep = [i for i, g in enumerate(geoms) if g is not None]
    if not keep:
        raise SystemExit(f"Pieces table '{args.pieces_table}' has no pieces")
    rows = partition_pieces(
        [int(df["piece_id"].iloc[i]) for i in keep],
        [geoms[i] for i in keep],
        np.stack([np.asarray(df[DINO_COLUMN].iloc[i], np.float64) for i in keep]),
        meta["pixel_size"],
        meta["patch_size"],
        args.region_cost,
        args.max_extent_px,
        args.refine_passes,
    )
    out_meta = make_meta(
        "regions",
        CRS.from_wkt(meta["crs"]),
        meta["pixel_size"],
        tile_size=meta["tile_size"],
        patch_size=meta["patch_size"],
        pieces_table=args.pieces_table,
        region_cost=args.region_cost,
        max_extent_px=args.max_extent_px,
    )
    table = write_region_table(db, args.table, rows, out_meta)
    logger.info(f"Wrote {len(rows):,} regions to '{args.table}'")
    if args.geojson:
        export_geojson(table, args.geojson)


if __name__ == "__main__":
    cli()
