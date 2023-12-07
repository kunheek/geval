from collections import namedtuple

import numpy as np
import torch
from tqdm import trange

Manifold = namedtuple('Manifold', ['features', 'radii', 'distances'])


def _pairwise_distances(U, V):
    '''
    args:
        U: torch.Tensor of shape N x dim
        V: torch.Tensor of shape N x dim
    returns:
        N x N symmetric torch.Tensor
    '''
    # Squared norms of each row in U and V.
    norm_u = torch.sum(U.square(), dim=1, keepdim=True)
    norm_v = torch.sum(V.square(), dim=1, keepdim=True)

    # norm_u as a column and norm_v as a row vectors.
    norm_v = norm_v.T

    # Pairwise squared Euclidean distances.
    UV = U @ V.T
    diff_square = norm_v - 2*UV + norm_u

    diff_square = torch.maximum(diff_square, torch.zeros_like(diff_square))

    distances = torch.sqrt(diff_square)
    return distances


def compute_pairwise_distances(U, V, batch_size=1024) -> np.ndarray:
    '''
    args:
        U: torch.Tensor of shape N x dim
        V: torch.Tensor of shape N x dim
    returns:
        N x N symmetric torch.Tensor
    '''
    assert U.device == V.device
    num_U = U.shape[0]
    num_V = V.shape[0]
    D = torch.zeros(num_U, num_V, dtype=U.dtype, device=U.device)

    for i in range(0, num_U, batch_size):
        for j in range(0, num_V, batch_size):
            D[i:i+batch_size, j:j+batch_size] = _pairwise_distances(U[i:i+batch_size], V[j:j+batch_size])
    return D


def compute_pairwise_distances_np(U, V) -> np.ndarray:
    '''
    args:
        U: np.ndarray of shape N x dim
        V: np.ndarray of shape N x dim
    returns:
        N x N symmetric np.ndarray
    '''
    if isinstance(U, torch.Tensor):
        U = U.cpu().numpy()
    if isinstance(V, torch.Tensor):
        V = V.cpu().numpy()

    num_U = U.shape[0]
    num_V = V.shape[0]
    U = U.astype(np.float64)  # to prevent underflow

    # Squared norms of each row in U and V.
    norm_u = np.sum(U**2, axis=1, keepdims=True)
    norm_v = np.sum(V**2, axis=1, keepdims=True)

    # norm_u as a column and norm_v as a row vectors.
    u_square = np.repeat(norm_u, num_V, axis=1)
    v_square = np.repeat(norm_v.T, num_U, axis=0)

    # Pairwise squared Euclidean distances.
    UV = np.dot(U, V.T)
    diff_square = u_square - 2*UV + v_square

    # check negative distance
    min_diff_square = diff_square.min()
    if min_diff_square < 0:
        idx = diff_square < 0
        diff_square[idx] = 0
        print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
              min_diff_square)

    distances = np.sqrt(diff_square)
    return distances


def distances2radii(distances, k=3):
    num_features = distances.shape[0]
    radii = np.zeros(num_features)
    for i in range(num_features):
        radii[i] = get_kth_value(distances[i], k=k)
    return radii


def get_kth_value(np_array, k):
    kprime = k + 1  # kth NN should be (k+1)th because closest one is itself
    idx = np.argpartition(np_array, kprime)
    k_smallests = np_array[idx[:kprime]]
    kth_value = k_smallests.max()
    return kth_value


def compute_metric(ref_manifold, subject_feats, desc=''):
    num_subjects = subject_feats.shape[0]
    count = 0
    dist = compute_pairwise_distances(ref_manifold.features, subject_feats)
    for i in trange(num_subjects, desc=desc):
        count += (dist[:, i].cpu().numpy() < ref_manifold.radii).any()
    return count / num_subjects


def is_in_ball(center, radius, subject):
    return distance(center, subject) < radius


def distance(feat1, feat2):
    return np.linalg.norm(feat1 - feat2)


def realism(manifold_real, feat_subject, eps=1e-6):
    feats_real = manifold_real.features
    radii_real = manifold_real.radii
    diff = feats_real - feat_subject
    dists = np.linalg.norm(diff, axis=1)
    ratios = radii_real / (dists + eps)
    max_realism = float(ratios.max())
    return max_realism


def compute_manifold(acts, k=3):
    distances = compute_pairwise_distances(acts, acts).cpu().numpy()
    radii = distances2radii(distances, k=k)
    return Manifold(acts, radii, distances)


def compute_precision_recall(ref_manifold, sample_manifold):
    precision = compute_metric(ref_manifold, sample_manifold.features, 'computing precision...')
    recall = compute_metric(sample_manifold, ref_manifold.features, 'computing recall...')
    return precision, recall
