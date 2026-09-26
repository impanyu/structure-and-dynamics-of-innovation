"""Cluster corpus idea embeddings into a reproducible topic pool.

`cluster_topics` is the pure partitioning step (KMeans over unit-normalized
embedding vectors); `scripts/gen_topics.py` wraps it with I/O and LLM naming
to produce the on-disk topic pool used by paper 2 (spec: shared-board
architecture, configs/p2_forum/topics-k128.yaml).
"""
import numpy as np


def cluster_topics(vecs: np.ndarray, k: int, seed: int = 0) -> list[list[int]]:
    """Partition rows of `vecs` into k clusters; returns row indices per
    cluster, ordered largest first. Deterministic for a given seed."""
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vecs)
    groups = [np.flatnonzero(labels == c).tolist() for c in range(k)]
    return sorted(groups, key=len, reverse=True)
