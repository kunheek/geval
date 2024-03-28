import open_clip
import torch
from torchvision.transforms.functional import normalize

ARCH_WEIGHT_DEFAULTS = {
    'ViT-B-32': 'laion2b_s34b_b79k',
    'ViT-B-16': 'laion2b_s34b_b88k',
    'ViT-L-14': 'datacomp_xl_s13b_b90k',
    'ViT-bigG-14': 'laion2b_s39b_b160k',
}

INPUT_SIZE = {
    'ViT-B-32': (256, 256),
    'ViT-B-16': (224, 224),
    'ViT-L-14': (224, 224),
    'ViT-bigG-14': (224, 224),
}


class CLIPEncoder(torch.nn.Module):
    def __init__(self, arch=None, resize_inside=True):
        super().__init__()
        if arch is None:
            arch = "ViT-L-14"
        pretrained_weights = ARCH_WEIGHT_DEFAULTS[arch]
        self.model = open_clip.create_model(arch, pretrained_weights)

        # NOTE
        self.resize_inside = resize_inside
        self.input_size = INPUT_SIZE[arch]
        self.mean = (0.48145466, 0.4578275, 0.40821073)
        self.std = (0.26862954, 0.26130258, 0.27577711)
        self.require_normalization = True

    def forward(self, x):
        if self.resize_inside:
            x = torch.nn.functional.interpolate(
                x, size=self.input_size, mode='bicubic', antialias=True,
            )
        assert x.shape[2:] == self.input_size
        x = x.float() / 255
        x = normalize(x, mean=self.mean, std=self.std)

        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.model.visual.class_embedding.to(x.dtype) +
                torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        blocks = self.model.visual.transformer.resblocks
        for r in blocks:
            x = r(x, attn_mask=None)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if self.model.visual.global_average_pool:
            x = x.mean(dim=1)
        else:
            x = x[:, 0]

        x = self.model.visual.ln_post(x)

        return x
