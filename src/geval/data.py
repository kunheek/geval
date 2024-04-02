import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from .clean_resize import CleanResize

IMAGE_EXTENSIONS = {'.bmp', '.jpg', '.jpeg', '.pgm', '.png', '.ppm',
                    '.tif', '.tiff', '.webp'}

DATA_ROOT = os.environ.get("DATA_ROOT", "./data")


def is_image_file(filename):
    ext = os.path.splitext(filename.lower())[-1]
    return ext in IMAGE_EXTENSIONS


def get_image_files(data_dir, max_dataset_size=None):
    assert os.path.isdir(data_dir), f"{data_dir} is not a valid directory."
    assert isinstance(max_dataset_size, (int, type(None)))

    paths = glob.glob(os.path.join(data_dir, "**"), recursive=True)
    paths = sorted(filter(is_image_file, paths))
    if not paths:
        raise RuntimeError(
            f"Found 0 images in: {data_dir}\n"
            "Supported image extensions are: " + ",".join(Image.EXTENSION)
        )
    return paths[:max_dataset_size]


class ToUint8Tensor:
    def __call__(self, img):
        img = np.asarray(img)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((2, 0, 1)).copy()  # HWC -> CHW
        return torch.as_tensor(img, dtype=torch.uint8)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.open_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0

    def open_image(self, i):
        raise NotImplementedError


class NpzDataset(ImageDataset):
    def __init__(self, path, transform=None):
        assert path.endswith(".npz")
        images = np.load(path)['images']
        super().__init__(images, transform)

    def open_image(self, i):
        return Image.fromarray(self.images[i])


class FolderDataset(ImageDataset):
    def __init__(self, path, transform=None):
        assert os.path.isdir(path)
        images = get_image_files(path)
        super().__init__(images, transform)

    def open_image(self, i):
        return Image.open(self.images[i]).convert('RGB')


def get_dataloader(path, model, image_size=None, batch_size=128, num_workers=4):
    transform = []
    if image_size is not None and image_size != model.input_size[0]:
        transform.append(transform.Resize(image_size, interpolation=Image.BICUBIC))
    if not model.resize_inside:
        transform.append(CleanResize(model.input_size[0]))
        transform.append(transforms.CenterCrop(image_size))
    transform.append(ToUint8Tensor())
    transform = transforms.Compose(transform)
    print("transform:\n", transform)


    if path.lower().startswith("cifar100"):
        if ":" in path:
            train = (path.split(":")[-1].lower() == "train")
        else:
            train = True
        dataset = datasets.CIFAR100(
            DATA_ROOT, train=train, transform=transform, download=True,
        )
    elif path.lower().startswith("cifar10"):
        if ":" in path:
            train = (path.split(":")[-1].lower() == "train")
        else:
            train = True
        dataset = datasets.CIFAR10(
            DATA_ROOT, train=train, transform=transform, download=True,
        )
    elif path.lower().startswith("celeba"):
        if ":" in path:
            split = path.split(":")[-1].lower()
        else:
            split = "test"
        dataset = datasets.CelebA(
            DATA_ROOT, split=split, transform=transform, download=True,
        )

    elif path.endswith(".npz"):
        dataset = NpzDataset(path, transform)
    elif os.path.isdir(path):
        dataset = FolderDataset(path, transform)
    else:
        raise ValueError(f"Invalid path: {path}")

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
