import argparse

import torch
from geval.script_util import compute_reps_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute representations from a dataset")
    parser.add_argument("path", type=str, help="Path to the dataset")
    parser.add_argument("--image-size", type=int, help="Size of the images")
    parser.add_argument("--model-name", type=str, help="Name of the model")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--depth", type=int, default=0, help="Depth")
    parser.add_argument("--clean-resize", action="store_true", help="Clean resize")
    return parser.parse_args()


def main():
    args = parse_args()

    # Use deterministic algorithms for reproducibility.
    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    reps = compute_reps_from_path(
        args.path,
        args.image_size,
        args.model_name,
        args.batch_size,
        torch.device(args.device),
        args.clean_resize,
        args.depth,
        cache_dir="assets/stats",
    )


if __name__ == "__main__":
    main()
