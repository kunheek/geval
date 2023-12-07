import numpy as np


def compute_inception_score(preds: np.ndarray, num_splits: int = 10):
    scores = []
    split_size = preds.shape[0] // num_splits
    for i in range(0, preds.shape[0], split_size):
        part = preds[i : i+split_size]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))
