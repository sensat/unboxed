"""DINOv3 / INSID3 model wrapper: loading, per-tile clustering, prototypes.

Images reach DINOv3 at IMAGE_SIZE (448 px = one ingestion tile at native
resolution), giving a 28 x 28 grid of 16 px patches. Patch features are L2
normalised and positionally debiased (INSID3) before clustering or averaging.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .models.insid3 import INSID3
from .utils.clustering import agglomerative_clustering, split_disconnected_clusters
from .utils.data import build_transform, downsample_mask, load_image, load_mask

logger = logging.getLogger(__name__)

IMAGE_SIZE = 448
PATCH_SIZE = 16
# Patches with less than this fraction of real (non-nodata) pixels are excluded
MIN_PATCH_VALID_FRACTION = 0.5
# Agglomerative clustering distance threshold for per-tile clusters
CLUSTER_TAU = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model: INSID3 | None = None


def init(weights_path: str, device: str = DEVICE) -> INSID3:
    """Load DINOv3 ViT-L/16 (torch hub) with the given weights, wrapped in INSID3."""
    global _model
    if _model is not None:
        return _model
    # Use the cached hub repo when present; loading by GitHub name contacts GitHub every time
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov3_main"
    if local_repo.is_dir():
        encoder = torch.hub.load(
            str(local_repo), "dinov3_vitl16", source="local", weights=weights_path
        )
    else:
        encoder = torch.hub.load(
            "facebookresearch/dinov3", "dinov3_vitl16", weights=weights_path
        )
    model = INSID3(
        encoder=encoder,
        image_size=IMAGE_SIZE,
        svd_components=500,
        tau=CLUSTER_TAU,
        merge_threshold=0.2,
        mask_refiner="bilinear",
        resize_to_orig_size=True,
        device=device,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    _model = model
    return model


def get_model() -> INSID3:
    if _model is None:
        raise RuntimeError(
            "Model not initialised: call unboxed.insid3.init(weights_path)"
        )
    return _model


@torch.no_grad()
def encode(images: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised and debiased patch features, each (1, T, C, h, w)."""
    model = get_model()
    tensors = [
        load_image(im, build_transform(IMAGE_SIZE), model.device)[0][0] for im in images
    ]
    fmaps = F.normalize(
        model._extract_features(torch.stack(tensors).unsqueeze(0)), p=2, dim=2
    )
    return fmaps, model._debias_features(fmaps)


def patch_validity(valid: np.ndarray | None, h: int, w: int, device) -> torch.Tensor:
    """(h*w,) bool: patches with at least MIN_PATCH_VALID_FRACTION real pixels."""
    if valid is None:
        return torch.ones(h * w, dtype=torch.bool, device=device)
    frac = F.interpolate(
        torch.from_numpy(np.asarray(valid, bool)).float()[None, None],
        size=(h, w),
        mode="area",
    )
    return (frac[0, 0].view(-1) >= MIN_PATCH_VALID_FRACTION).to(device)


@torch.no_grad()
def reference_prototype(
    images: list[Image.Image], masks: list[np.ndarray]
) -> np.ndarray:
    """Class prototype: mean debiased feature under each reference mask, averaged over shots."""
    model = get_model()
    _, debiased = encode(images)
    _, _, _, h, w = debiased.shape
    protos = []
    for s, mask in enumerate(masks):
        mask_t = load_mask(
            Image.fromarray(np.asarray(mask, np.uint8) * 255), IMAGE_SIZE, model.device
        )
        fg = debiased[0, s, :, downsample_mask(mask_t[None], h, w)]
        if fg.shape[1] > 0:
            protos.append(fg.mean(dim=1))
    if not protos:
        raise ValueError("No reference mask covers any patch")
    return (
        F.normalize(torch.stack(protos).mean(dim=0), p=2, dim=0).float().cpu().numpy()
    )


@dataclass
class Piece:
    vector: np.ndarray  # (C,) unit-norm mean debiased feature of the piece's patches
    bbox: tuple[
        float, float, float, float
    ]  # pixel bounds in the tile image, max exclusive
    mask: np.ndarray  # (H, W) bool at the tile image's size


@torch.no_grad()
def cluster_pieces(
    images: list[Image.Image],
    valid_masks: list[np.ndarray] | None = None,
    split_disjoint: bool = True,
) -> list[list[Piece]]:
    """Cluster each tile's patches and return its pieces.

    Valid patches are clustered (agglomerative, cosine, tau=CLUSTER_TAU); with
    split_disjoint each cluster is split into spatially connected pieces, each
    embedded from its own patches. Masks exclude nodata pixels.
    """
    fmaps, debiased = encode(images)
    _, _, C, h, w = fmaps.shape
    out = []
    for t, image in enumerate(images):
        feats = fmaps[0, t].reshape(C, -1).permute(1, 0)
        feats_deb = debiased[0, t].reshape(C, -1).permute(1, 0)
        valid = None if valid_masks is None else np.asarray(valid_masks[t], bool)
        patch_valid = patch_validity(valid, h, w, feats.device)

        labels = torch.full((h * w,), -1, dtype=torch.long, device=feats.device)
        n_valid = int(patch_valid.sum())
        if n_valid > 1:
            labels[patch_valid] = agglomerative_clustering(
                feats[patch_valid], tau=CLUSTER_TAU
            )
        elif n_valid == 1:
            labels[patch_valid] = 0
        labels = labels.reshape(h, w)
        if split_disjoint:
            labels = split_disconnected_clusters(labels)

        size = (image.height, image.width)
        labels_img = (
            F.interpolate(labels.float()[None, None], size=size, mode="nearest")[0, 0]
            .long()
            .cpu()
            .numpy()
        )
        flat = labels.view(-1)
        pieces = []
        for k in range(int(flat.max()) + 1):
            idx = flat == k
            mask = labels_img == k
            if valid is not None:
                mask &= valid
            if not idx.any() or not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            vector = (
                F.normalize(feats_deb[idx].mean(dim=0), p=2, dim=0)
                .float()
                .cpu()
                .numpy()
            )
            pieces.append(
                Piece(
                    vector,
                    (
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max() + 1),
                        float(ys.max() + 1),
                    ),
                    mask,
                )
            )
        out.append(pieces)
    return out


@torch.no_grad()
def tile_embeddings(
    images: list[Image.Image], valid_masks: list[np.ndarray] | None = None
) -> list[np.ndarray]:
    """One unit-norm vector per tile: the mean debiased feature of its valid patches."""
    _, debiased = encode(images)
    _, _, C, h, w = debiased.shape
    vectors = []
    for t in range(len(images)):
        feats = debiased[0, t].reshape(C, -1).permute(1, 0)
        valid = patch_validity(
            None if valid_masks is None else valid_masks[t], h, w, feats.device
        )
        mean = feats[valid].mean(0) if valid.any() else feats.mean(0)
        vectors.append(F.normalize(mean, p=2, dim=0).float().cpu().numpy())
    return vectors
