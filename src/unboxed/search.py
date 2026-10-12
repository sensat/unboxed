"""Search a region or tile table, and expand hits by query merge for segmentation.

Retrieval always ranks the atomic rows (cross-tile regions or naive tiles).
Query merge happens afterwards, as part of segmentation: each retrieved hit that
scores in the top (1 - quantile) of the whole index for this query grows into
the connected set of equally "hot" neighbouring regions (best-scoring first,
while the merged bbox stays within max_extent_px). Regions need not resemble
each other, only the query: tarmac and car regions both score well for "car
park", so a car park is segmented as one area.
"""

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.ops import unary_union

from .tables import DINO_COLUMN, SKYCLIP_COLUMN, parse_geometry, table_meta

DEFAULT_MIN_REGION_PATCHES = 5
DEFAULT_MERGE_QUANTILE = 0.95
DEFAULT_MERGE_MAX_EXTENT_PX = 1792


class Index:
    """A region or tile table with its metadata."""

    def __init__(self, table):
        self.table = table
        self.meta = table_meta(table)
        self.kind = self.meta["kind"]  # "regions" or "tiles"
        self.pixel_size = self.meta["pixel_size"]
        self._adjacency = None

    @property
    def patch_area(self) -> float:
        """Ground area of one DINOv3 patch (CRS units squared)."""
        return (self.meta["patch_size"] * self.pixel_size) ** 2

    def search(
        self,
        vector: np.ndarray,
        column: str = DINO_COLUMN,
        limit: int = 200,
        min_patches: float = 0.0,
    ) -> tuple[pd.DataFrame, list]:
        """Top `limit` rows by cosine similarity on `column`; rows below min_patches are excluded."""
        if column not in self.table.schema.names:
            raise ValueError(
                f"Table has no '{column}' column"
                + (" (run unboxed-skyclip)" if column == SKYCLIP_COLUMN else "")
            )
        q = self.table.search(
            np.asarray(vector, np.float32), vector_column_name=column
        ).distance_type("cosine")
        min_area = min_patches * self.patch_area
        if min_area > 0:
            q = q.where(f"area >= {min_area}", prefilter=True)
        df = q.limit(limit).to_pandas()
        return df, [parse_geometry(g) for g in df["mask_geojson"]]

    def adjacency(self, tolerance: float = 0.05):
        """All rows by region_id, their geometries and touching neighbours (cached)."""
        if self._adjacency is None:
            df = self.table.to_arrow().select(["region_id", "mask_geojson"]).to_pandas()
            df = df.sort_values("region_id").reset_index(drop=True)
            geoms = [parse_geometry(g) for g in df["mask_geojson"]]
            tree = STRtree(geoms)
            left, right = tree.query(
                [g.buffer(tolerance) for g in geoms], predicate="intersects"
            )
            nbrs: list[set[int]] = [set() for _ in geoms]
            for a, b in zip(left, right):
                if a != b:
                    nbrs[a].add(int(b))
                    nbrs[b].add(int(a))
            pos = {int(r): i for i, r in enumerate(df["region_id"])}
            self._adjacency = (df["region_id"].to_numpy(), geoms, nbrs, pos)
        return self._adjacency


@dataclass
class Area:
    """One area to segment: a retrieved hit, possibly merged with hot neighbours."""

    rank: int  # rank of the seed hit (1-based)
    region_id: int
    geometry: object
    member_ids: list[int] = field(default_factory=list)


def query_merge(
    index: Index,
    vector: np.ndarray,
    hits: pd.DataFrame,
    column: str = DINO_COLUMN,
    quantile: float = DEFAULT_MERGE_QUANTILE,
    max_extent_px: float = DEFAULT_MERGE_MAX_EXTENT_PX,
) -> list[Area]:
    """Expand ranked hits into merged areas (see module docstring).

    Hits already absorbed by a better-ranked hit's area are not repeated, so the
    result has at most len(hits) areas, in rank order.
    """
    ids, geoms, nbrs, pos = index.adjacency()
    scores = (
        index.table.search(np.asarray(vector, np.float32), vector_column_name=column)
        .distance_type("cosine")
        .select(["region_id"])
        .limit(len(ids))
        .to_pandas()
    )
    sim = np.full(len(ids), -np.inf)
    sim[[pos[int(r)] for r in scores["region_id"]]] = (
        1.0 - scores["_distance"].to_numpy()
    )
    hot = sim >= np.quantile(sim[np.isfinite(sim)], quantile)
    bounds = np.array([g.bounds for g in geoms])
    max_extent = max_extent_px * index.pixel_size
    used = np.zeros(len(ids), bool)

    areas = []
    for rank, rid in enumerate(hits["region_id"], start=1):
        seed = pos[int(rid)]
        if used[seed]:
            continue
        members, used[seed] = [seed], True
        if hot[seed]:
            b = bounds[seed].copy()
            heap = [(-sim[m], m) for m in nbrs[seed] if hot[m] and not used[m]]
            heapq.heapify(heap)
            while heap:
                _, m = heapq.heappop(heap)
                if used[m]:
                    continue
                nb = np.array(
                    [
                        min(b[0], bounds[m, 0]),
                        min(b[1], bounds[m, 1]),
                        max(b[2], bounds[m, 2]),
                        max(b[3], bounds[m, 3]),
                    ]
                )
                if max(nb[2] - nb[0], nb[3] - nb[1]) > max_extent:
                    continue
                used[m] = True
                members.append(m)
                b = nb
                for n in nbrs[m]:
                    if hot[n] and not used[n]:
                        heapq.heappush(heap, (-sim[n], n))
        geom = (
            geoms[seed]
            if len(members) == 1
            else unary_union([geoms[m] for m in members])
        )
        areas.append(Area(rank, int(rid), geom, [int(ids[m]) for m in members]))
    return areas


def plain_areas(hits: pd.DataFrame, geoms: list) -> list[Area]:
    """Hits as areas without merging."""
    return [
        Area(r, int(rid), g, [int(rid)])
        for r, (rid, g) in enumerate(zip(hits["region_id"], geoms), start=1)
    ]
