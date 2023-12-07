#!/usr/bin/env python
import argparse
import csv
import os

import torch

from evaluations.features import read_activations, read_softmax_scores
from evaluations.frechet_distance import (compute_fid_statistics,
                                          frechet_distance)
from evaluations.inception_score import compute_inception_score
from evaluations.kernel_distance import kernel_distance
from evaluations.precision_recall import (compute_manifold,
                                          compute_precision_recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="path to reference batch npz file")
    parser.add_argument("--sample", help="path to sample batch npz file")
    parser.add_argument("--csv_file", default="metrics.csv", help="csv file to write evaluation results")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for computing activations")
    parser.add_argument("--network", type=str, default="inception_v3")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    # torch.use_deterministic_algorithms(True)  # Use deterministic algorithms for reproducibility.
    # Disable TF32 on Ampere GPUs for better reproducibility.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    ref_name = os.path.splitext(args.reference)[0]
    ref_acts_pt = f"{ref_name}_{args.network}_acts.pt"
    if os.path.isfile(ref_acts_pt):
        print("Reading reference batch activations...", end="", flush=True)
        ref_acts = torch.load(ref_acts_pt, map_location="cuda")
    else:
        print("Computing reference batch activations...", end="", flush=True)
        ref_acts = read_activations(args.reference, args.network, batch_size=args.batch_size)
        torch.save(ref_acts.cpu(), ref_acts_pt)
    print("Done!")

    # NOTE: to verify that the activations are the same as the legacy code
    # import cleanfid
    # from evaluations.features import get_npz_features
    # model = cleanfid.fid.build_feature_extractor("clean", torch.device("cuda"), use_dataparallel=True)
    # cfid_ref_acts = get_npz_features(args.reference, model)
    # print(ref_acts.shape, cfid_ref_acts.shape)
    # diff = np.abs(cfid_ref_acts - ref_acts)
    # print(diff.max(), diff.sum())
    # exit()

    print("computing/reading reference batch statistics...")
    # ref_stats, ref_stats_spatial = evaluator.read_statistics(args.reference, ref_acts)
    ref_stats = compute_fid_statistics(ref_acts)

    print("computing sample batch activations...")
    sample_acts = read_activations(args.sample, args.network, batch_size=args.batch_size)
    print("computing/reading sample batch statistics...")
    # sample_stats, sample_stats_spatial = evaluator.read_statistics(args.sample, sample_acts)
    sample_stats = compute_fid_statistics(sample_acts)

    print("Computing evaluations...")
    # inception_score = compute_inception_score(sample_acts[0])
    softmax_scores = read_softmax_scores(args.sample, batch_size=args.batch_size)
    inception_score, is_std = compute_inception_score(softmax_scores)
    print(f"{'Inception Score':>15} : {inception_score:.3f} {chr(177)} {is_std:.3f}")

    fid = frechet_distance(ref_stats, sample_stats)
    print(f"{'(clean) FID':>15} : {fid:.2f}")
    # sfid = frechet_distance(ref_stats_spatial)
    # print(f"sFID: {sfid:.3f}")

    kid = kernel_distance(ref_acts, sample_acts) * 1e3
    print(f"{'(clean) KID':>15} : {kid:.2f} (x 10^-3)")

    ref_manifold = compute_manifold(ref_acts)
    sample_manifold = compute_manifold(sample_acts)
    prec, recall = compute_precision_recall(ref_manifold, sample_manifold)
    print(f"{'Precision':>15} : {prec:.3f}")
    print(f"{'Recall':>15} : {recall:.3f}")

    # If file not exists, create it and write header
    if not os.path.isfile(args.csv_file):
        with open(args.csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "reference", "dataset", "experiment", "npz",
                "inception_score", "fid", "kid", "precision", "recall",
                "framework",
            ])

    with open(args.csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        cmd_parse = args.sample.split('/')
        refname = os.path.basename(args.reference)
        datasetname, expname, npzname = cmd_parse[-4], cmd_parse[-3], cmd_parse[-1]

        is_str = f"{inception_score:.2f}+/-{is_std:.3f}"
        writer.writerow([
            refname, datasetname, expname, npzname,
            is_str, f"{fid:.2f}", f"{kid:.2f}", f"{prec:.2f}", f"{recall:.2f}",
            "pytorch",
        ])


if __name__ == "__main__":
    main()
