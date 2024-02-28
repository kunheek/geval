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
    filename = get_filename(dataset, image_size, model_name, depth, clean_resize)
    filename = filename + ".npz"
    res = False

    try:
        repo_id = "kunheekim/geval"
        hf_hub_download(repo_id, filename, local_dir=cache_dir)
        res = os.path.join(cache_dir, filename)
    except:
        print("Not found in kunheekim/geval")
    try:
        repo_id = "nahyeonkaty/geval"
        hf_hub_download(repo_id, filename, local_dir=cache_dir)
        res = os.path.join(cache_dir, filename)
    except:
        print("Not found in nahyeonkaty/geval")
    return res
