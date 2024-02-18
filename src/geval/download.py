import os

from huggingface_hub import hf_hub_download


def get_filename(dataset, image_size, model_name, depth, clean_resize):
    filename = "_".join([
        dataset,
        str(image_size),
        model_name,
        f"depth{int(depth)}",
    ])
    if clean_resize:
        filename += "_clean"
    return filename


def download(
        dataset,
        image_size,
        model_name="dinov2",
        clean_resize=False,
        depth=0,
        cache_dir=".cache/geval",
):
    repo_id = "kunheekim/geval"
    filename = get_filename(dataset, image_size, model_name, depth, clean_resize)
    filename = filename + ".npz"

    try:
        hf_hub_download(repo_id, filename, local_dir=cache_dir)
        return os.path.join(cache_dir, filename)
    except Exception as e:
        print(e)
        print("Failed to download from Hugging Face Hub.")
        return False
