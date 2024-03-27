import os
import pathlib

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

from .data import get_dataloader
from .download import download
from .encoders import load_encoder


@torch.no_grad()
def encode_features(model, dataloader, device):
    """Extracts features from all images in DataLoader given model.

    Params:
    -- model       : Instance of Encoder such as inception or CLIP or dinov2
    -- DataLoader  : DataLoader containing image files, or torchvision.dataset

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    feats = []
    for batch in tqdm(dataloader):
        if isinstance(batch, list):
            # batch is likely list[array(images), array(labels)]
            batch = batch[0]

        if not torch.is_tensor(batch):
            # assume batch is then e.g. AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
            batch = batch['pixel_values']
            batch = batch[:,0]

        # Convert grayscale to RGB
        if batch.ndim == 3:
            batch.unsqueeze_(1)
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = batch.to(device)

        pred = model(batch)

        if not torch.is_tensor(pred): # Some encoders output tuples or lists
            pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu().numpy()

        feats.append(pred)

    feats = np.concatenate(feats, axis=0)
    return feats


def encode_feats_from_path(
        path,
        image_size=None,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        cache_dir=".cache/geval",
):
    if path.endswith("/"):
        path = path[:-1]
    basename = os.path.basename(path)
    cachename = f"{basename}_{image_size}_{model_name}"
    if clean_resize:
        cachename += "_clean"
    cache_path = os.path.join(cache_dir, f"{cachename}.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached representations from {cache_path}")
        return np.load(cache_path)["reps"]

    model = load_encoder(model_name, device, resize_inside=(not clean_resize))
    dataloader = get_dataloader(path, model, batch_size)
    feats = encode_features(model, dataloader, device)

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, feats=feats, model_name=model_name)

    return feats


def encode_feats_from_batch(
        batch,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        data_format="NCHW",
):
    assert data_format in {"NCHW", "NHWC"}
    assert batch.dim() in (3, 4), batch.shape
    if batch.dim() == 3:
        batch = batch.unsqueeze(0)

    if data_format == "NHWC":
        batch = batch.permute(0, 2, 3, 1).contiguous()

    if batch.dtype == torch.uint8:
        batch = batch.float() / 255.0

    # Convert grayscale to RGB
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)

    assert batch.min() >= 0 and batch.max() <= 1, f"{batch.min()}, {batch.max()}"

    model = load_encoder(model_name, device, resize_inside=(not clean_resize))

    dataset = torch.utils.data.TensorDataset(batch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=False,
    )

    mean = model.mean
    std = model.std
    normalize = transforms.Normalize(mean, std)

    feats = []
    # for idx in range(0, batch.shape[0], batch_size):
    for batch in dataloader:
        batch = batch[0].to(device)
        if not model.resize_inside:
            batch = F.interpolate(
                batch, size=model.input_size,
                mode='bicubic', antialias=True,
            )
        batch = normalize(batch)

        pred = model(batch)

        if not torch.is_tensor(pred):  # Some encoders output tuples or lists
            pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        pred = pred.cpu()
        feats.append(pred)

    feats = torch.cat(feats, dim=0)
    return feats


def save_outputs(output_dir, feats, model, checkpoint, DataLoader):
    """Save representations and other info to disk at file_path"""
    out_path = get_path(output_dir, model, checkpoint, DataLoader)

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(out_path, model=model, feats=feats)


def load_feats_from_path(saved_dir, model, checkpoint, DataLoader):
    """Load representations and other info to disk at file_path"""
    save_path = get_path(saved_dir, model, checkpoint, DataLoader)
    feats = None
    print('Loading from:', save_path)
    if os.path.exists(f'{save_path}.npz'):
        saved_file = np.load(f'{save_path}.npz')
        feats = saved_file['feats']
    return feats


def get_path(output_dir, model, checkpoint, DataLoader):
    train_str = 'train' if DataLoader.train_set else 'test'

    ckpt_str = '' if checkpoint is None else f'_ckpt-{os.path.splitext(os.path.basename(checkpoint))[0]}'

    hparams_str = f'reps_{DataLoader.dataset_name}_{model}{ckpt_str}_nimage-{len(DataLoader.data_set)}_{train_str}'
    return os.path.join(output_dir, hparams_str)
