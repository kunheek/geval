# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import numpy as np
import torch
import torchvision.transforms.functional as TF

import sys

from .encoder import Encoder

from ..resize import pil_resize

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
        if self.clean_resize:
            image = pil_resize(image, self.input_size)
        else:
            image = TF.resize(image, self.input_size, TF.InterpolationMode.BICUBIC)
            image = TF.to_tensor(image)

        return TF.normalize(image, mean=self.mean, std=self.std)

    @property
    def input_size(self):
        return (224, 224)

    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def std(self):
        return (0.229, 0.224, 0.225)
