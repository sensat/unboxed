"""INSID3 segmentation heads: an image crop + class query -> binary mask."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .insid3.models.insid3 import INSID3
from .insid3.utils.clustering import (
    agglomerative_clustering,
    compute_cluster_prototypes,
)
from .insid3.utils.data import downsample_mask, load_image, load_mask
from .insid3.utils.refinement import upsample_mask
from .references import Query


def head_model(model: INSID3, image_size: int) -> INSID3:
    """An INSID3 sharing model's encoder but running at image_size."""
    if image_size == model.image_size:
        return model
    return INSID3(
        encoder=model.encoder,
        image_size=image_size,
        svd_components=model.svd_components,
        tau=model.tau,
        merge_threshold=model.merge_threshold,
        mask_refiner=model.mask_refiner,
        resize_to_orig_size=model.resize_to_orig_size,
        device=model.device,
    ).eval()


class INSID3Head:
    """INSID3 at a fixed input size (crops are resized to it). Used for naive tiles."""

    def __init__(self, model: INSID3):
        self.model = model

    @torch.no_grad()
    def predict(self, image: Image.Image, query: Query) -> np.ndarray:
        for ref in query.references:
            self.model.set_reference(
                ref.image, Image.fromarray(ref.mask.astype(np.uint8) * 255)
            )
        self.model.set_target(image)
        return self.model.segment().cpu().numpy()


class NativeScaleINSID3Head:
    """INSID3 run at each crop's own size, with no resampling.

    Crops of different sizes (e.g. 448 / 672 / 896 px) each reach DINOv3 at
    native resolution, so objects keep the same scale as in the reference shots.
    References are encoded once per query at their own size and cached; each
    input size gets its own positional-debiasing basis (an INSID3 sharing the
    encoder). Otherwise follows INSID3's own prediction.
    """

    def __init__(self, model: INSID3, ref_size: int):
        self.base = model
        self.ref_size = ref_size
        self._models: dict[int, INSID3] = {}
        self._ref_query = None
        self._refs = None

    def _model(self, size: int) -> INSID3:
        if size not in self._models:
            self._models[size] = head_model(self.base, size)
        return self._models[size]

    @torch.no_grad()
    def _encode(self, model: INSID3, images: list) -> tuple[torch.Tensor, torch.Tensor]:
        tensors = torch.stack(
            [load_image(im, model._transform, model.device)[0][0] for im in images]
        ).unsqueeze(0)
        fmaps = F.normalize(model._extract_features(tensors), p=2, dim=2)
        return fmaps, model._debias_features(fmaps)

    @torch.no_grad()
    def _references(self, query: Query):
        if self._ref_query is not query:
            model = self._model(self.ref_size)
            _, ref_deb = self._encode(model, [r.image for r in query.references])
            masks = torch.stack(
                [
                    load_mask(
                        Image.fromarray(r.mask.astype(np.uint8) * 255),
                        self.ref_size,
                        model.device,
                    )
                    for r in query.references
                ]
            )  # (S, 1, H, W)
            _, num_refs, _, hr, wr = ref_deb.shape
            protos = []
            for s in range(num_refs):
                fg = ref_deb[0, s, :, downsample_mask(masks[s : s + 1], hr, wr)]
                if fg.shape[1] > 0:
                    protos.append(fg.mean(dim=1))
            prototype = F.normalize(
                torch.stack(protos).mean(dim=0), p=2, dim=0
            ).unsqueeze(1)
            self._refs = (ref_deb, masks, prototype)
            self._ref_query = query
        return self._refs

    @torch.no_grad()
    def predict(self, image: Image.Image, query: Query) -> np.ndarray:
        if image.width != image.height:
            raise ValueError("Native-scale head expects square crops")
        model = self._model(image.width)
        ref_deb, ref_masks, prototype = self._references(query)
        fmaps, debiased = self._encode(model, [image])
        feat_tgt, feat_tgt_deb = fmaps[:, 0], debiased[:, 0]
        _, C, h, w = feat_tgt.shape
        sim_maps = [
            torch.einsum("bchw,bcxy->bhwxy", ref_deb[:, s], feat_tgt_deb)
            for s in range(ref_deb.shape[1])
        ]
        candidate = model._locate_candidates(
            sim_maps, ref_masks, feat_tgt_deb, prototype, h, w
        )
        if candidate.sum() == 0:
            pred = candidate
        else:
            labels = agglomerative_clustering(
                feat_tgt[0].reshape(C, -1).permute(1, 0), model.tau
            ).reshape(h, w)
            K = int(labels.max().item()) + 1
            cluster_protos = compute_cluster_prototypes(
                feat_tgt_deb[0].reshape(C, -1).permute(1, 0), labels.view(-1), K
            )
            pred = model._seed_and_aggregate(
                candidate,
                labels,
                cluster_protos,
                K,
                prototype,
                feat_tgt,
                feat_tgt_deb,
                h,
                w,
            )
        return upsample_mask(pred, image.height, image.width).cpu().numpy()
