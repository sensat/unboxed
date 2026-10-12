"""SkyCLIP (OpenCLIP ViT-L/14) image and text embeddings."""

import numpy as np
import torch
from PIL import Image

MODEL_NAME = "ViT-L-14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_preprocess = None
_tokenizer = None


def init(weights_path: str, model_name: str = MODEL_NAME):
    """Load a SkyCLIP checkpoint (OpenCLIP format)."""
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return _model
    from open_clip import create_model_and_transforms, get_tokenizer

    model, _, preprocess = create_model_and_transforms(
        model_name, pretrained=weights_path, weights_only=False
    )
    _model, _preprocess, _tokenizer = (
        model.eval().to(DEVICE),
        preprocess,
        get_tokenizer(model_name),
    )
    return _model


def _check():
    if _model is None:
        raise RuntimeError(
            "SkyCLIP not initialised: call unboxed.skyclip.init(weights_path)"
        )


@torch.no_grad()
def image_embeddings(images: list[Image.Image]) -> list[np.ndarray]:
    """Unit-norm image embeddings (the model's own preprocessing resizes to 224 px)."""
    _check()
    batch = torch.stack([_preprocess(im) for im in images]).to(DEVICE)
    feats = _model.encode_image(batch)
    feats /= feats.norm(dim=-1, keepdim=True)
    return list(feats.float().cpu().numpy())


@torch.no_grad()
def text_embedding(text: str) -> np.ndarray:
    _check()
    tokens = _tokenizer([text])
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    feats = _model.encode_text(tokens.to(DEVICE))
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats[0].float().cpu().numpy()


def generate_prompts(label: str) -> list[str]:
    """Prompt variants used for text queries of one class."""
    return [
        f"{label}",
        f"an image containing a {label}",
        f"a photo containing a {label}",
        f"an aerial image containing a {label}",
        f"an aerial image of a {label}",
        f"an image of a {label}",
        f"a photo of a {label}",
    ]
