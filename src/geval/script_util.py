import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .dataloaders import get_dataloader
from .models import load_encoder
from .representations import get_representations


@torch.no_grad()
def compute_reps_from_path(
        path,
        image_size,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        depth=0,
        cache_dir="~/.cache/geval",
):
    basename = os.path.basename(path)
    cachename = f"{basename}_{image_size}_{model_name}_{depth}"
    if clean_resize:
        cachename += "_clean"
    cache_path = os.path.join(cache_dir, f"{cachename}.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached representations from {cache_path}")
        return np.load(cache_path)["reps"]

    model = load_encoder(model_name, device, ckpt=None, arch=None,
                         clean_resize=clean_resize,
                         sinception=(model_name == 'sinception'),
                         depth=depth)
    model.eval()

    num_workers = 4
    transform = transforms.Compose([
        transforms.Resize(image_size, transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: model.transform(x)),
    ])
    DL = get_dataloader(path, -1, batch_size, num_workers, seed=0,
                        sample_w_replacement=False,
                        transform=transform)

    reps = get_representations(model, DL, device, normalized=False)

    # Save to cache.
    hparams = vars(DL).copy()  # Remove keys that can't be pickled
    hparams.pop("transform")
    hparams.pop("data_loader")
    hparams.pop("data_set")

    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, reps=reps, model_name=model_name, hparams=hparams)

    return reps


@torch.no_grad()
def compute_reps_from_batch(
        batch,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        depth=0,
        normalized=False,
):
    assert batch.dim() in (3, 4), batch.shape
    assert batch.min() >= 0 and batch.max() <= 1, f"{batch.min()}, {batch.max()}"

    # Convert grayscale to RGB
    if batch.ndim == 3:
        batch.unsqueeze_(1)
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)

    model = load_encoder(model_name, device, ckpt=None, arch=None,
                         clean_resize=clean_resize,
                         sinception=(model_name == 'sinception'),
                         depth=depth)
    model.eval()

    for idx in range(0, batch.shape[0], batch_size):
        minibatch = batch[idx:idx+batch_size].to(device)

        minibatch = model.transform_tensor(minibatch, normalize=True)
        pred = model(minibatch)

        if not torch.is_tensor(pred):  # Some encoders output tuples or lists
            pred = pred[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.dim() > 2:
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2)

        if normalized:
            pred = F.normalize(pred, dim=-1)
        pred = pred.cpu().numpy()

        if idx == 0:
            # initialize output array with full dataset size
            dims = pred.shape[-1]
            pred_arr = np.empty((minibatch.shape[0], dims))

        pred_arr[idx:idx+pred.shape[0]] = pred

    return pred_arr
