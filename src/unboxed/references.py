"""Few-shot class queries built from reference objects.

A query is a set of reference shots (a crop centred on each reference object
and its mask) plus their DINOv3 prototype, which lives in the same space as the
index's dino_vector. References can be labelled objects (example-based search,
evaluation) or the top retrieved regions of a text query (pseudo-references).
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from . import insid3, raster

REFERENCE_CROP_PX = insid3.IMAGE_SIZE


@dataclass
class Reference:
    geometry: object
    image: Image.Image
    mask: np.ndarray


@dataclass
class Query:
    label: str
    references: list[Reference]
    prototype: np.ndarray


def build_query(
    src, label: str, reference_geoms: list, crop_px: int = REFERENCE_CROP_PX
) -> Query:
    """Crop each reference at tile scale (native resolution) and average their prototypes."""
    references = []
    for geom in reference_geoms:
        window = raster.window_around_bounds(src, geom.bounds, crop_px)
        rgb = raster.read_rgb(src, window)
        mask = raster.rasterize_geoms(
            [geom], src.window_transform(window), rgb.shape[:2]
        )
        references.append(Reference(geom, raster.to_model_image(rgb), mask))
    prototype = insid3.reference_prototype(
        [r.image for r in references], [r.mask for r in references]
    )
    return Query(label, references, prototype)


def split_references(
    num_objects: int,
    num_references: int,
    seed: int,
    reference_indices: list[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Pick reference objects at random (or explicitly); the rest are held out."""
    if reference_indices:
        refs = sorted(set(reference_indices))
    else:
        if num_references >= num_objects:
            raise ValueError(
                f"Need more than {num_references} labelled objects, got {num_objects}"
            )
        rng = np.random.default_rng(seed)
        refs = sorted(
            rng.choice(num_objects, size=num_references, replace=False).tolist()
        )
    return refs, [i for i in range(num_objects) if i not in refs]
