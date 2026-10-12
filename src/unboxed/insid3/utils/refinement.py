"""Mask refinement: upsampling"""

import torch
import torch.nn.functional as F


def upsample_mask(mask: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Upsample a feature-resolution binary mask to (H, W)."""
    return (
        F.interpolate(
            mask[None, None].float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        > 0.5
    )
