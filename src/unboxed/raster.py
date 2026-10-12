"""Raster (COG) and GeoJSON helpers shared by ingestion, indexing and evaluation.

Conventions:
- Input imagery is an RGB(A) Cloud Optimised GeoTIFF in a projected CRS with
  metre units. All geometries handled in memory and stored in tables are in the
  raster's CRS.
- GeoJSON inputs (perimeter, labels) and outputs are in EPSG:4326.
"""

import base64
import io
import json
from collections import Counter
from itertools import product
from pathlib import Path

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.features import shapes as raster_shapes
from rasterio.warp import transform_geom
from rasterio.windows import Window
from scipy import ndimage
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

GEOJSON_CRS = "EPSG:4326"
RGB_BANDS = [1, 2, 3]
# Nodata areas smaller than this are treated as real pixels: with only a nodata
# value, genuinely pure-white or pure-black pixels are indistinguishable from fill
MIN_NODATA_REGION_PX = 256
# Tiles (and crops) with more than this fraction of nodata are skipped
NODATA_THRESHOLD = 0.75


# ──────── Input validation ────────


def validate_cog(path: str, strict: bool = True, min_size: int = 448) -> None:
    """Check the raster is an 8-bit RGB(A) COG in a projected, metre-based CRS."""
    from rio_cogeo.cogeo import cog_validate

    with rasterio.open(path) as src:
        if src.count not in (3, 4):
            raise ValueError(f"Expected an RGB or RGBA raster, got {src.count} bands")
        if src.crs is None:
            raise ValueError("Raster has no CRS (not georeferenced)")
        if not src.crs.is_projected:
            raise ValueError(
                f"Raster CRS {src.crs} is not projected; reproject to a metric CRS"
            )
        units = (src.crs.linear_units or "").lower()
        if units not in ("metre", "meter", "metres", "meters", "m"):
            raise ValueError(
                f"Raster CRS units are '{src.crs.linear_units}'; expected metres"
            )
        if abs(abs(src.res[0]) - abs(src.res[1])) > 1e-6 * abs(src.res[0]):
            raise ValueError(f"Expected square pixels, got resolution {src.res}")
        if any(dt != "uint8" for dt in src.dtypes[:3]):
            raise ValueError(f"Expected 8-bit RGB, got band types {src.dtypes}")
        if src.width < min_size or src.height < min_size:
            raise ValueError(
                f"Raster is {src.width} x {src.height} px; needs at least {min_size} x {min_size}"
            )
    if strict:
        is_valid, errors, _ = cog_validate(path, strict=True)
        if not is_valid:
            raise ValueError(f"Not a valid Cloud Optimised GeoTIFF: {errors}")


# ──────── GeoJSON inputs ────────


def load_perimeter(geojson_path: str, raster_crs: CRS):
    """Union of all (multi)polygons in a GeoJSON file, reprojected to the raster CRS."""
    with open(geojson_path) as f:
        data = json.load(f)
    if data["type"] == "FeatureCollection":
        geometries = [feat["geometry"] for feat in data["features"]]
    elif data["type"] == "Feature":
        geometries = [data["geometry"]]
    else:
        geometries = [data]
    return unary_union(
        [
            make_valid(shape(transform_geom(GEOJSON_CRS, raster_crs, g)))
            for g in geometries
        ]
    )


def count_labels(labels_path: str, label_property: str = "label") -> Counter:
    with open(labels_path) as f:
        return Counter(
            feat["properties"].get(label_property) for feat in json.load(f)["features"]
        )


def load_labels(
    labels_path: str, label: str, perimeter, raster_crs, label_property: str = "label"
) -> list:
    """Labelled polygons of one class (by `label_property`) intersecting the perimeter."""
    with open(labels_path) as f:
        features = json.load(f)["features"]
    geoms = []
    for feat in features:
        if feat["properties"].get(label_property) != label:
            continue
        geom = shape(transform_geom(GEOJSON_CRS, raster_crs, feat["geometry"]))
        if not geom.is_valid:
            geom = make_valid(geom)
        if not geom.is_empty and (perimeter is None or geom.intersects(perimeter)):
            geoms.append(geom)
    return geoms


def load_geometries(geojson_path: str, raster_crs) -> list:
    """All geometries in a GeoJSON file (e.g. reference objects), in the raster CRS."""
    with open(geojson_path) as f:
        data = json.load(f)
    feats = data["features"] if data["type"] == "FeatureCollection" else [data]
    return [
        make_valid(shape(transform_geom(GEOJSON_CRS, raster_crs, f.get("geometry", f))))
        for f in feats
    ]


def write_geojson(path: Path, records: list[tuple], raster_crs) -> None:
    """Write (geometry, properties) records as a FeatureCollection in EPSG:4326."""
    features = [
        {
            "type": "Feature",
            "geometry": transform_geom(raster_crs, GEOJSON_CRS, mapping(g)),
            "properties": props,
        }
        for g, props in records
        if g is not None and not g.is_empty
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)


# ──────── Tile grid ────────


def tile_windows(
    height: int, width: int, size: int, offset: tuple[int, int] = (0, 0)
) -> list[Window]:
    """Non-overlapping size x size grid; offset (x, y) shifts the origin in pixels.

    Edge tiles are shifted inwards to stay inside the raster.
    """
    ys = sorted({min(y, height - size) for y in range(offset[1], height, size)})
    xs = sorted({min(x, width - size) for x in range(offset[0], width, size)})
    return [Window(x, y, size, size) for y, x in product(ys, xs)]


def windows_in_perimeter(windows: list[Window], perimeter, transform) -> list[Window]:
    if perimeter is None:
        return windows
    return [
        w
        for w in windows
        if perimeter.intersects(box(*rasterio.windows.bounds(w, transform)))
    ]


# ──────── Reading ────────


def read_valid_mask(src, window: Window, out_shape: tuple[int, int]) -> np.ndarray:
    """(H, W) bool: True where the pixel holds real imagery (not nodata fill)."""
    valid = src.dataset_mask(window=window, out_shape=out_shape) > 0
    if valid.all():
        return valid
    components, n = ndimage.label(~valid)
    sizes = np.bincount(components.ravel(), minlength=n + 1)
    small = sizes < MIN_NODATA_REGION_PX
    small[0] = False
    valid[small[components]] = True
    return valid


def nodata_ratio(src, window: Window, out_size: int | None = None) -> float:
    shape_ = (
        (out_size, out_size) if out_size else (int(window.height), int(window.width))
    )
    return 1.0 - float(read_valid_mask(src, window, shape_).mean())


def read_rgb(
    src, window: Window, out_size: int | None = None, resampling=Resampling.average
) -> np.ndarray:
    """(H, W, 3) uint8, resampled to out_size x out_size if given."""
    if out_size is None or (out_size == window.width and out_size == window.height):
        data = src.read(RGB_BANDS, window=window)
    else:
        data = src.read(
            RGB_BANDS,
            window=window,
            out_shape=(3, out_size, out_size),
            resampling=resampling,
        )
    return np.moveaxis(data, 0, -1)


def read_tile(src, window: Window, size: int) -> tuple[np.ndarray, np.ndarray] | None:
    """A tile at size x size with its validity mask; None if mostly nodata."""
    valid = read_valid_mask(src, window, (size, size))
    if 1.0 - float(valid.mean()) > NODATA_THRESHOLD:
        return None
    return read_rgb(src, window, size, Resampling.bilinear), valid


def to_model_image(rgb: np.ndarray) -> Image.Image:
    """JPEG round trip, as at ingestion, so query-time features match the index."""
    _, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    )
    return Image.open(io.BytesIO(buf.tobytes())).convert("RGB")


def to_jpeg_b64(rgb: np.ndarray) -> str:
    _, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
    )
    return base64.b64encode(buf).decode("utf-8")


# ──────── Windows around geometries ────────


def _clamp_start(start: float, size: int, limit: int) -> int:
    return int(max(0, min(round(start), limit - size)))


def window_around_bounds(src, bounds: tuple, size: int) -> Window:
    """A size x size window centred on bounds, shifted to stay inside the raster."""
    col_min, row_min = ~src.transform * (bounds[0], bounds[3])
    col_max, row_max = ~src.transform * (bounds[2], bounds[1])
    return Window(
        _clamp_start((col_min + col_max) / 2 - size / 2, size, src.width),
        _clamp_start((row_min + row_max) / 2 - size / 2, size, src.height),
        size,
        size,
    )


def extent_px(src, bounds: tuple) -> float:
    """Longest side of bounds in native pixels."""
    return max(bounds[2] - bounds[0], bounds[3] - bounds[1]) / abs(src.res[0])


def window_covering(src, bounds: tuple, min_size: int) -> Window:
    """Square window covering bounds: min_size if it fits, else the bounds' own extent."""
    ext = extent_px(src, bounds)
    size = min_size if ext <= min_size + 1e-3 else int(np.ceil(ext))
    return window_around_bounds(src, bounds, min(size, src.width, src.height))


def output_transform(src, window: Window, out_size: int | None = None) -> Affine:
    """Affine from pixels of the (possibly resampled) crop image to the CRS."""
    transform = src.window_transform(window)
    if out_size is None:
        return transform
    return transform * Affine.scale(window.width / out_size, window.height / out_size)


def window_box(src, window: Window):
    return box(*rasterio.windows.bounds(window, src.transform))


# ──────── Masks and polygons ────────


def tile_pixel_transform(
    tile_bounds: tuple, transform: Affine, image_shape: tuple[int, int]
) -> Affine:
    """Affine from tile-image pixels to the CRS (the image may be resampled from its window)."""
    left, bottom, right, top = tile_bounds
    col_off, row_off = (~transform) * (left, top)
    window_w = round((right - left) / transform.a)
    window_h = round((bottom - top) / transform.e)
    h, w = image_shape
    return (
        transform
        * Affine.translation(col_off, row_off)
        * Affine.scale(window_w / w, window_h / h)
    )


def rasterize_geoms(geoms: list, transform, out_shape: tuple[int, int]) -> np.ndarray:
    if not geoms:
        return np.zeros(out_shape, dtype=bool)
    return rasterize(
        [(g, 1) for g in geoms], out_shape=out_shape, transform=transform, fill=0
    ).astype(bool)


def mask_to_geometry(mask: np.ndarray, transform):
    """Polygonise a boolean mask; None when empty."""
    m = np.asarray(mask, np.uint8)
    if not m.any():
        return None
    geoms = [
        shape(g)
        for g, v in raster_shapes(m, mask=m.astype(bool), transform=transform)
        if int(v) == 1
    ]
    return unary_union(geoms) if geoms else None


def explode(geom) -> list:
    """Polygon parts of a (multi)geometry."""
    if geom is None or geom.is_empty:
        return []
    if hasattr(geom, "geoms"):
        return [g for part in geom.geoms for g in explode(part)]
    return [geom] if geom.geom_type == "Polygon" else []
