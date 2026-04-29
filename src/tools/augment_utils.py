from __future__ import annotations

import os
import json
import h5py
import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple, Callable


def kmeans_cluster(features: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    N, D = features.shape
    K_eff = max(1, min(K, N))

    try:
        import faiss  # type: ignore
        km = faiss.Kmeans(D, K_eff, niter=25, seed=seed, verbose=False)
        km.train(features.astype(np.float32))
        _, I = km.index.search(features.astype(np.float32), 1)
        return I.reshape(-1).astype(np.int64)
    except Exception:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K_eff, n_init=10, random_state=seed)
        return km.fit_predict(features).astype(np.int64)


def ensure_cluster_ids(slide: Dict[str, np.ndarray], K: int, seed: int = 0) -> np.ndarray:
    if "cluster_ids" in slide:
        return slide["cluster_ids"]
    slide["cluster_ids"] = kmeans_cluster(slide["features"], K=K, seed=seed)
    return slide["cluster_ids"]


def mask_cluster(features: np.ndarray,
                 cluster_ids: np.ndarray,
                 target_cluster: int) -> np.ndarray:
    keep = cluster_ids != target_cluster
    if not keep.any():
        return features
    return features[keep]


def compute_per_sample_prior(
    delta: np.ndarray,             # [K], Δ^{(i)}_c
    cluster_sizes: np.ndarray,     # [K], number of patches in each cluster
    N: int,                        # total number of patches in the slide
    mu: float = 0.7,               # interpolation weight
    tau: float = 1.0,              # softmax temperature
    epsilon: float = 0.02,         # min cluster size ratio to be sampled
) -> np.ndarray:

    K = delta.shape[0]
    valid = cluster_sizes >= max(1, int(epsilon * N))

    if not valid.any():
        return np.full(K, 1.0 / K, dtype=np.float32)

    scaled = delta / max(tau, 1e-8)
    scaled = scaled - scaled[valid].max()  # numerical stability on valid entries
    exp = np.exp(scaled)
    exp[~valid] = 0.0
    Z = exp.sum()
    sharp = exp / Z if Z > 0 else np.full(K, 1.0 / max(1, valid.sum())) * valid

    uniform = np.where(valid, 1.0 / valid.sum(), 0.0)

    p = mu * sharp + (1.0 - mu) * uniform
    s = p.sum()
    if s <= 0:
        return uniform.astype(np.float32)
    return (p / s).astype(np.float32)

def sample_cluster_to_mask(prior: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(prior), p=prior))


def save_prior_cache(path: str,
                     priors: Dict[str, np.ndarray],
                     cluster_ids_by_item: Optional[Dict[str, np.ndarray]] = None,
                     meta: Optional[dict] = None) -> None:
    payload = {f"prior::{k}": v for k, v in priors.items()}
    if cluster_ids_by_item is not None:
        for k, v in cluster_ids_by_item.items():
            payload[f"cluster_ids::{k}"] = v
    if meta is not None:
        payload["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(path, **payload)


def load_prior_cache(path: str) -> Tuple[Dict[str, np.ndarray],
                                         Dict[str, np.ndarray],
                                         dict]:
    data = np.load(path, allow_pickle=False)
    priors, cluster_ids_by_item = {}, {}
    meta = {}
    for k in data.files:
        if k == "__meta__":
            meta = json.loads(bytes(data[k].tobytes()).decode("utf-8"))
        elif k.startswith("prior::"):
            priors[k[len("prior::"):]] = data[k]
        elif k.startswith("cluster_ids::"):
            cluster_ids_by_item[k[len("cluster_ids::"):]] = data[k]
    return priors, cluster_ids_by_item, meta