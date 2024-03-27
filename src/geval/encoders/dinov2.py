import torch

VALID_ARCHITECTURES = (
    'vits14',
    'vitb14',
    'vitl14',
    'vitg14',
)

class DINOv2Encoder(torch.nn.Module):
    def __init__(self, arch="vitl14", resize_inside=False):
        assert arch in VALID_ARCHITECTURES, f"Invalid architecture: {arch}"
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_{arch}",
            trust_repo=True, verbose=False, skip_validation=True,
        )
        self.model.eval().requires_grad_(False)

        #
        self.resize_inside = resize_inside
        self.input_size = (224, 224)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.require_normalization = True

    def forward(self, x):
        if self.resize_inside:
            x = torch.nn.functional.interpolate(
                x, size=self.input_size, mode='bicubic', antialias=True,
            )

        assert x.shape[2:] == self.input_size
        return self.model(x)
