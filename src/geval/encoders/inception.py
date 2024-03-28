import os
import shutil
import urllib

import torch

# Inception weights ported to Pytorch from
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
FID_WEIGHTS_URL = "https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth"  # noqa: E501
INCEPTION_URL = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt"  # noqa: E501
# INCEPTION_URL = "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl"  # noqa: E501


def download_inception(fpath="./"):
    inception_path = os.path.join(fpath, "inception-2015-12-05.pt")
    if not os.path.exists(inception_path):
        # download the file
        with urllib.request.urlopen(INCEPTION_URL) as response, open(inception_path, 'wb') as f:
            shutil.copyfileobj(response, f)
    return inception_path


class InceptionV3(torch.nn.Module):
    def __init__(self, resize_inside=False):
        super().__init__()
        inception_path = download_inception()
        self.base = torch.jit.load(inception_path).eval()
        self.name = "inception"
        self.eval().requires_grad_(False)

        # NOTE: these values are required for data loaders
        self.resize_inside = resize_inside
        self.input_size = (299, 299)
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.require_normalization = False

    def forward(self, x):
        assert x.min() >= 0, "input should be in the range [0, 255]"
        if self.resize_inside:
            features = self.base(x, return_features=True).view((x.shape[0], 2048))
        else:
            assert x.shape[1:] == (3, 299, 299)
            x = x - 128
            x = x / 128
            features = self.base.layers.forward(x,).view((x.shape[0], 2048))
        return features
