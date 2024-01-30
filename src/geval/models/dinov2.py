# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torchvision.transforms as TF
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

from .encoder import Encoder

from ..resizer import pil_resize

VALID_ARCHITECTURES = [
                        'vits14',
                        'vitb14',
                        'vitl14',
                        'vitg14',
                    ]

class DINOv2Encoder(Encoder):
    def setup(self, arch=None, clean_resize:bool=False):
        if arch is None: 
            arch = 'vitl14'

        self.arch = arch

        arch_str = f'dinov2_{self.arch}'

        if self.arch not in VALID_ARCHITECTURES:
            sys.exit(f"arch={self.arch} is not a valid architecture. Choose from {VALID_ARCHITECTURES}")

        self.model = torch.hub.load('facebookresearch/dinov2', arch_str)
        self.clean_resize = clean_resize

    def transform(self, image):

        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])

        if self.clean_resize:
            image = pil_resize(image, (224, 224))
        else:
            image = TF.Compose([
                TF.Resize((224, 224), TF.InterpolationMode.BICUBIC),
                TF.ToTensor(),
            ])(image)

        return TF.Normalize(imagenet_mean, imagenet_std)(image)

    def transform_tensor(self, image, normalize=True):
        H = image.shape[-2]
        W = image.shape[-1]
        if H != 224 or W != 224:
            image = F.interpolate(
                image, size=(224, 224), mode='bicubic', align_corners=False,
            )
        if normalize:
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            image = TF.functional.normalize(image, imagenet_mean, imagenet_std)
        return image
