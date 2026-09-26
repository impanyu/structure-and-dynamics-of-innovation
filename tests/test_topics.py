import numpy as np

from innovation.core.ideas.topics import cluster_topics


def test_cluster_topics_partitions_every_row_exactly_once():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(60, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    clusters = cluster_topics(vecs, k=5, seed=0)

    assert len(clusters) == 5
    flat = sorted(i for c in clusters for i in c)
    assert flat == list(range(60))
    assert all(len(c) > 0 for c in clusters)


def test_cluster_topics_is_deterministic():
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(40, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    assert cluster_topics(vecs, k=4, seed=7) == cluster_topics(vecs, k=4, seed=7)
