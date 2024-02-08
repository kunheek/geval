from abc import ABC, abstractmethod

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Encoder(ABC, nn.Module):
    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        self.setup(*args, **kwargs)
        self.name = 'encoder'

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, x):
        """Converts a PIL Image to an input for the model"""

    @property
    @abstractmethod
    def input_size(self):
        pass

    def transform_tensor(self, image, normalize=True):
        input_size = self.input_size
        H = image.shape[-2]
        W = image.shape[-1]
        if H != input_size[0] or W != input_size[1]:
            image = F.interpolate(
                image, size=input_size, mode='bicubic', align_corners=False,
            )
        if normalize:
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            image = TF.functional.normalize(image, imagenet_mean, imagenet_std)
        return image
