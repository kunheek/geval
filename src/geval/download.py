import os

from huggingface_hub import hf_hub_download


STATS = (
    "cifar10-train_32_inception_depth0",
    "cifar10-train_32_dinov2_depth0",
    "celeba-test_64_inception_depth0",
    "celeba-test_64_dinov2_depth0",
)


def download(
        dataset,
        image_size,
        model_name="dinov2",
        clean_resize=False,
        depth=0,
        cache_dir=".cache/geval",
):
    repo_id = "kunheekim/geval"
    filename = "_".join([
        dataset,
        str(image_size),
        model_name,
        f"depth{int(depth)}",
    ])
    if clean_resize:
        filename += "_clean"
    if filename not in STATS:
        msg = f"Unknown stats: {filename}. Available stats: {STATS}"
        raise ValueError(msg)
    filename = filename + ".npz"

    hf_hub_download(repo_id, filename, local_dir=cache_dir)
    return os.path.join(cache_dir, filename)
