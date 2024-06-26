#!/usr/bin/env python
import csv
import os
import sys
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import torch

from geval import download, metrics

from .data import get_dataloader
from .encoders import MODELS, load_encoder
from .features import encode_features

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument('path', type=str, nargs='+',
                    help='Paths to the images, the first one is the real dataset, followed by generated')

parser.add_argument('--metrics', type=str, nargs='+', default=('fid',),
                    help="metrics to compute")

parser.add_argument('--model', type=str, default="inception", choices=MODELS.keys(),
                    help='Model to use for generating feature representations.')

parser.add_argument('--image-size', type=int,
                    help='Model to use for generating feature representations.')

parser.add_argument('--train-dataset', type=str, default='imagenet',
                    help='Dataset that model was trained on. Sets proper normalization for MAE.')

parser.add_argument('-bs', '--batch-size', type=int, default=128,
                    help='Batch size to use')

parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of processes to use for data loading. '
                         'Defaults to `min(8, num_cpus)`')

parser.add_argument('--device', type=str, default=None,
                    help='Device to use. Like cuda, cuda:0 or cpu')

parser.add_argument('--nearest_k', type=int, default=5,
                    help='Number of neighbours for precision, recall, density, and coverage')

parser.add_argument('--reduced_n', type=int, default=10000,
                    help='Number of samples used for train, baseline, test, and generated sets for FLS')

parser.add_argument('--output_dir', type=str, default='experiments/',
                    help='Directory to save outputs in')

parser.add_argument('--filename', type=str, default="metrics.csv",
                    help='Filename to save scores to')

parser.add_argument('--clean-resize', action='store_true',
                    help='Use clean resizing (from pillow)')


def get_device_and_num_workers(device, num_workers):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 4)
    else:
        num_workers = num_workers

    return device, num_workers


def compute_metrics(ref_feats, gen_feats, args):
    metrics_list = list(map(lambda m: m.lower(), args.metrics))

    scores = {"reference": args.path[0], "generated": args.path[1], "model": args.model}
    for m in metrics_list:
        scores[m] = None
    if 'fid' in metrics_list:
        print("Computing FID \n", file=sys.stderr)
        feat1, feat2 = ref_feats["inception"], gen_feats["inception"]
        scores['fid'] = metrics.compute_FD_with_feats(feat1, feat2)

    if "fd_dinov2" in metrics_list:
        print("Computing FD_dinov2 \n", file=sys.stderr)
        feat1, feat2 = ref_feats["dinov2"], gen_feats["dinov2"]
        scores['fd_dinov2'] = metrics.compute_FD_with_feats(feat1, feat2)

    if "clip_mmd" in metrics_list:
        print("Computing Clip-MMD \n", file=sys.stderr)
        mmd_values = metrics.compute_mmd(*feats) * 1e3
        scores['clip_mmd'] = mmd_values.mean()
        scores['clip_mmd_var'] = mmd_values.std()

    feats = ref_feats[args.model], gen_feats[args.model]
    if 'fd-infinity' in metrics_list:
        print("Computing fd-infinity \n", file=sys.stderr)
        scores['fd_infinity_value'] = metrics.compute_FD_infinity(*feats)

    if 'kd' in metrics_list:
        print("Computing KD \n", file=sys.stderr)
        mmd_values = metrics.compute_mmd(*feats) * 1e3
        scores['kd_value'] = mmd_values.mean()
        scores['kd_variance'] = mmd_values.std()

    if 'prdc' in metrics_list:
        print("Computing precision, recall, density, and coverage \n", file=sys.stderr)
        reduced_n = min(args.reduced_n, feats[0].shape[0], feats[1].shape[0])
        inds0 = np.random.choice(feats[0].shape[0], reduced_n, replace=False)

        inds1 = np.arange(feats[1].shape[0])
        if 'realism' not in metrics_list:
            # Realism is returned for each sample, so do not shuffle if this metric is desired.
            # Else filenames and realism scores will not align
            inds1 = np.random.choice(inds1, min(inds1.shape[0], reduced_n), replace=False)

        prdc_dict = metrics.compute_prdc(
            feats[0][inds0],
            feats[1][inds1],
            nearest_k=args.nearest_k,
            realism=True if 'realism' in metrics_list else False)
        scores = dict(scores, **prdc_dict)

    for key, value in scores.items():
        if key in ("reference", "generated", "model", "realism"):
            continue
        print(f'{key}: {value:.5f}\n')

    return scores


def main():
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device, num_workers = get_device_and_num_workers(args.device, args.num_workers)

    print('Loading Model', file=sys.stderr)
    models = dict(inception=None, clip=None, dinov2=None)
    if "fid" in args.metrics or args.model is None:
        models["inception"] = load_encoder("inception", device, resize_inside=(not args.clean_resize))
    if "fd_dinov2" in args.metrics or args.model == "dinov2":
        models["dinov2"] = load_encoder("dinov2", device, resize_inside=(not args.clean_resize))
    if "clip_mmd" in args.metrics or args.model == "clip":
        models["clip"] = load_encoder("clip", device, resize_inside=(not args.clean_resize))

    # Compute features for reference dataset.
    print(f"Computing features for reference dataset: {args.path[0]}\n", file=sys.stderr)
    ref_feats = {}
    for name, model in models.items():
        if model is None:
            continue
        feat_path = download.download(
            args.path[0], args.image_size,
            model_name=name, clean_resize=args.clean_resize,
        )
        if not feat_path:
            dataloader = get_dataloader(args.path[0], model, args.image_size, args.batch_size)
            ref_feats[name] = encode_features(model, dataloader, device)
            del dataloader
        else:
            ref_feats[name] = np.load(feat_path)["feats"]

    # Compute features for generated datasets.
    for gen_path in args.path[1:]:
        print(f"Computing features for generated samples: {gen_path}\n", file=sys.stderr)
        gen_feats = {}
        for name, model in models.items():
            if model is None:
                continue
            dataloader = get_dataloader(gen_path, model, args.image_size, args.batch_size)
            gen_feats[name] = encode_features(model, dataloader, device)

        scores = compute_metrics(ref_feats, gen_feats, args)
        exists = os.path.exists(args.filename)
        with open(args.filename, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=scores.keys())

            if not exists:
                writer.writeheader()

            writer.writerow(scores)


if __name__ == "__main__":
    main()
