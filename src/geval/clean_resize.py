import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor


def clean_resize(output_size):
    s1, s2 = output_size

    def _resize_single_channel(x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    def resize_fn(x):
        x = np.asarray(x.convert('RGB')).astype(np.float32)
        x = [_resize_single_channel(x[:, :, idx]) for idx in range(3)]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return to_tensor(x) / 255

    return resize_fn


class CleanResize:
    def __init__(self, output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def _resize_single_channel(self, x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(self.output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(self.output_size[1], self.output_size[0], 1)

    def __call__(self, x):
        x = np.asarray(x.convert('RGB')).astype(np.float32)
        x = [self._resize_single_channel(x[:, :, idx]) for idx in range(3)]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return to_tensor(x) / 255

    def __repr__(self) -> str:
        detail = f"(size={self.size})"
        return f"{self.__class__.__name__}{detail}"
