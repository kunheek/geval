from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from cleanfid import fid
from torchvision import models
from tqdm import tqdm

from evaluations.utils import ImageDataset, clean_resize


def vgg16_extractor(device, use_dataparallel=True):
    weights = models.vgg.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    head = nn.Sequential(*list(model.classifier.children())[:4])
    model.classifier = head
    model.eval().requires_grad_(False)
    if use_dataparallel:
        model = nn.DataParallel(model)
    return model.to(device)


def dino_extractor(device, dino_model='dino_vitb16', use_dataparallel=True):
    model = torch.hub.load('facebookresearch/dino:main', dino_model)
    model.eval().requires_grad_(False)
    if use_dataparallel:
        model = nn.DataParallel(model)
    return model.to(device)


def read_activations(
        target, network="inception_v3", device="cuda",
        batch_size=128, num_workers=4,
):
    if network == "inception_v3":
        extractor = fid.build_feature_extractor("clean", device)
        image_size = (299, 299)
        # Inception-v3 expects values in [0, 1].
        # normalization will be during the forward pass.
        def normalize(x):
            return x
    elif network == "vgg16":
        extractor = vgg16_extractor(device)
        image_size = (224, 224)
        imagenet_normalize = partial(
            transforms.functional.normalize,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        def normalize(x):
            x = x / 255.0
            x = imagenet_normalize(x)
            return x
    elif network.startswith("dino_"):
        extractor = dino_extractor(device, dino_model=network)
        image_size = (224, 224)
        imagenet_normalize = partial(
            transforms.functional.normalize,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        def normalize(x):
            x = x / 255.0
            x = imagenet_normalize(x)
            return x
    else:
        raise ValueError(f"Invalid network: {network}")

    transform_fn = transforms.Compose([
        partial(clean_resize, output_size=image_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageDataset(target, transform=transform_fn)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        pin_memory=True,
    )
    features = []
    for batch in tqdm(loader, desc=f"extracting {network} features"):
        batch = batch.to(device, non_blocking=True)
        feature = extractor(batch)
        features.append(feature)
    features = torch.cat(features, dim=0)
    return features


def read_softmax_scores(
        target, classifier=None, device="cuda",
        batch_size=128, num_workers=4,
):
    if classifier is None:
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        classifier = models.inception_v3(weights=weights)
        image_size = (299, 299)    
        imagenet_normalize = partial(
            transforms.functional.normalize,
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
        )
        def normalize(x):
            return imagenet_normalize(x / 255.0)
    else:
        raise NotImplementedError
    classifier.to(device).eval().requires_grad_(False)

    transform_fn = transforms.Compose([
        partial(clean_resize, output_size=image_size),
        transforms.ToTensor(),
        normalize,
    ])
    dataset = ImageDataset(target, transform=transform_fn)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False,
        pin_memory=True,
    )
    softmax_outs = []
    for batch in tqdm(loader, desc=f"computing softmax scores"):
        batch = batch.to(device, non_blocking=True)
        pred = classifier(batch)
        softmax_out = F.softmax(pred, dim=1)
        softmax_outs.append(softmax_out)
    preds = torch.cat(softmax_outs, dim=0).cpu().numpy()
    return preds
