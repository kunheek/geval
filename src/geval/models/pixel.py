import torch

from ..resize import pil_resize
from .encoder import Encoder


class PixelEncoder(Encoder):
    def setup(self):
        self.model = torch.nn.Identity()

    def transform(self, image):
        image = pil_resize(image, (32, 32))
        return image
