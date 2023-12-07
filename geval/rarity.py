#!/usr/bin/env python
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

from evaluations.features import read_activations
from evaluations.precision_recall import compute_pairwise_distances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="path to reference batch npz file")
    parser.add_argument("--sample", help="path to sample batch npz file")
    parser.add_argument("--csv_file", default="evaluations/metrics.csv", help="csv file to write evaluation results")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size for computing activations")
    parser.add_argument("--network", type=str, default="dino_vits16")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
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
    print("computing sample batch activations...")
    sample_acts = read_activations(args.sample, args.network, batch_size=args.batch_size)

    scores, score_ids = rarity(ref_acts.cuda(), sample_acts.cuda(), k=args.k)
    with open(args.sample, "rb") as f:
        sample_images = np.load(f)["arr_0"]
    for i, id in enumerate(score_ids):
        score = int(scores[id])
        image = Image.fromarray(sample_images[int(id)])
        image.save(f"tmp/rarity_{score:03d}_{i:04d}.png")

    num_out_balls = len(sample_acts) - len(score_ids)
    print(f"Number of out of balls: {num_out_balls}")
    print(f"Mean rarity score: {np.mean(scores):.2f}")
    scores = scores[score_ids]  # scores in decreasing order, removing out of ball samples
    plt.hist(scores, bins=100)
    plt.savefig("rarity.png")
    print(scores)


def is_in_ball(ref_acts, sample_acts, k=3):
    """ Compute the differences between radii of kNN balls and distances
        for judging whether they are in each kNN ball or not.

		args:
			k (int): real ball's size is distance between reference real sample
						and k th nearest real sample.
			samples (np.array, num_samples * embed_dim): embedded generation samples
		return:
			dist_radi (np.array, num_reals * num_samples): each element means
						(radii of a ball - (distance between center of the ball and a sample)).
						if it is larger than 0, it is out of the ball.
			r (np.array, num_reals * 1): radii of each kNN real balls
			out_ball_ids (np.array, num_out_ball_samples): indices of samples outside of balls.
	"""
    ref2sample_distances = compute_pairwise_distances(ref_acts, sample_acts)
    ref2sample_distances = ref2sample_distances.cpu().numpy()

    ref2ref_distances = compute_pairwise_distances(ref_acts, ref_acts).cpu()
    ref2ref_sorted, _ = torch.sort(ref2ref_distances)
    r = ref2ref_sorted[:, k].cpu().numpy()

    dist_radi = (r[:,None].repeat(sample_acts.shape[0], axis = 1) - ref2sample_distances)
    out_ball_ids = np.where((dist_radi > 0).any(axis = 0) == False)[0]

    return dist_radi, r, out_ball_ids


def rarity(ref_acts, sample_acts, k=3):
    """ The larger the real ball's size, the rare the real sample would be.    
    	args:
    		k (int): real ball's size is distance between reference real sample
    					and k th nearest real sample.
    		samples (np.array, N * embed_dim): embedded generation samples
    	return:
    		scores (np.array, num_samples): scores of each samples which are not sorted.
    		scores_ids (np.array, num_samples_in_valid_ball): for samples in valid real balls,
    				sorted indices in decreasing order.
    """

    in_ball_dist, r, out_ball_ids = is_in_ball(ref_acts, sample_acts=sample_acts, k=k)

    num_out_ball = len(out_ball_ids)
    valid_real_balls = (in_ball_dist>0)

    scores = np.zeros(sample_acts.shape[0])

    for i in range(sample_acts.shape[0]):
        if i not in out_ball_ids:
            scores[i] = r[valid_real_balls[:,i]].min()

    scores_ids = (-scores).argsort()[:sample_acts.shape[0] - num_out_ball]

    return scores, scores_ids


if __name__ == "__main__":
	main()
