from .clip import CLIPEncoder
from .dinov2 import DINOv2Encoder
from .inception import InceptionV3

MODELS = {
    "inception" : InceptionV3,
    "clip": CLIPEncoder,
    "dinov2": DINOv2Encoder,
}


def load_encoder(model_name, device, resize_inside=True):
    """Load feature extractor"""
    assert model_name in MODELS, f"Unknown model name: {model_name}"
    model_cls = MODELS[model_name]

    encoder = model_cls(resize_inside=resize_inside)
    encoder.name = model_name
    return encoder.eval().to(device)
