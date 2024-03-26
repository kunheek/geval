import numpy as np
import torch

from .download import download
from .features import encode_feats_from_path, encode_feats_from_batch


def compute_reps_from_path(
        path,
        image_size=None,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        cache_dir=".cache/geval",
):
    return encode_feats_from_path(
        path,
        image_size,
        model_name,
        batch_size,
        device,
        clean_resize,
        cache_dir,
    )


def compute_reps_from_batch(
        batch,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        depth=0,
        data_format="NCHW",
):
    return encode_feats_from_batch(
        batch,
        model_name,
        batch_size,
        device,
        clean_resize,
        depth,
        data_format,
    )


def get_precomputed_reps(
        dataset,
        image_size,
        model_name='dinov2',
        clean_resize=False,
        depth=0,
        cache_dir=".cache/geval",
):
    npzpath = download(
        dataset,
        image_size,
        model_name,
        clean_resize,
        depth,
        cache_dir,
    )
    try:
        reps = np.load(npzpath)["feats"]
    except:
        reps = np.load(npzpath)["reps"]
    return reps
