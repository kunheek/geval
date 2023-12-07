import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

Image.init()


def is_image_file(filename):
    ext = os.path.splitext(filename.lower())[-1]
    return ext in Image.EXTENSION


def open_image_as_np(path):
    """
    Open an image as a numpy array.
    """
    img = Image.open(path).convert("RGB")
    return np.asarray(img)


def clean_resize(np_image, output_size, resample=Image.BICUBIC):
    """
    Resize the PIL image to the specified size using the specified filter.
    """
    assert isinstance(np_image, np.ndarray)
    s1, s2 = output_size
    def resize_single_channel(x_np):
        img = Image.fromarray(x_np.astype(np.float32), mode='F')
        img = img.resize(output_size, resample=resample)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    np_image = [resize_single_channel(np_image[:, :, idx]) for idx in range(3)]
    np_image = np.concatenate(np_image, axis=2).astype(np.float32)
    return np_image


def get_image_files(root_dir, max_dataset_size=None):
    assert os.path.isdir(root_dir), f"{root_dir} is not a valid directory."
    assert isinstance(max_dataset_size, (int, type(None)))

    paths = glob.glob(os.path.join(root_dir, "**"), recursive=True)
    paths = sorted(filter(is_image_file, paths))
    if not paths:
        raise RuntimeError(
            f"Found 0 images in: {root_dir}\n"
            "Supported image extensions are: " + ",".join(Image.EXTENSION)
        )
    return paths[:None]


class ImageDataset(Dataset):
    def __init__(self, target, transform=None):
        if os.path.isfile(target) and os.path.splitext(target)[1] == ".npz":
            with open(target, "rb") as f:
                self.images = np.load(f)["arr_0"]
        elif os.path.isdir(target):
            self.images = get_image_files(target)
        else:
            raise ValueError(f"Invalid target: {target}")
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if os.path.isfile(self.images[index]):
            image = open_image_as_np(self.images[index])
        if image.dtype == "uint8":
            image = image.astype(np.float32)
        assert image.dtype == "float32"

        if self.transform is not None:
            image = self.transform(image)
            assert isinstance(image, torch.Tensor)
            assert image.dtype == torch.float32
        return image


class AugmentationDataset(ImageDataset):
    def __init__(self, root, image_size, augmentations, load_size=256):
        super().__init__(root, image_size)
        self.load_size = load_size
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((image_size,)*2, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3),
        ])
        self.augmentations = augmentations

    def __getitem__(self, index):
        path = self.images[index]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.load_size,)*2, Image.BICUBIC)

        if torch.rand(1) < 0.5:  # Do augmentation
            image = self.augmentations(image)
            target = torch.ones(1)
        else:
            target = torch.zeros(1)
        image = self.transform(image)

        return image, target
