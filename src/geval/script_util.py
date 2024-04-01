import warnings

import numpy as np
import torch

from .download import CACHE_DIR, download
from .features import extract_feats_from_batch, extract_feats_from_path


def compute_reps_from_path(
        path,
        image_size=None,
        model_name='dinov2',
        batch_size=256,
        device=torch.device("cpu"),
        clean_resize=False,
        cache_dir=".cache/geval",
):
    warnings.warn(
        "DEPRECATED: use `geval.features.compute_reps_from_path` instead.",
        DeprecationWarning,
    )
    return extract_feats_from_path(
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
        data_format="NCHW",
):
    warnings.warn(
        "DEPRECATED: use `geval.features.compute_reps_from_batch` instead.",
        DeprecationWarning,
    )
    return extract_feats_from_batch(
        batch,
        model_name,
        batch_size,
        device,
        clean_resize,
        data_format,
    )


def get_precomputed_feats(
        dataset,
        image_size,
        model_name='dinov2',
        clean_resize=False,
        cache_dir=CACHE_DIR,
):
    npzpath = download(
        dataset,
        image_size,
        model_name,
        clean_resize,
        cache_dir=cache_dir,
    )
    try:
        reps = np.load(npzpath)["feats"]
    except:
        reps = np.load(npzpath)["reps"]
    return reps


def get_precomputed_reps(
        dataset,
        image_size,
        model_name='dinov2',
        clean_resize=False,
        cache_dir=CACHE_DIR,
):
    warnings.warn(
        "DEPRECATED: use `get_precomputed_feats` instead.",
        DeprecationWarning,
    )
    return get_precomputed_feats(
        dataset,
        image_size,
        model_name,
        clean_resize,
        cache_dir,
    )
