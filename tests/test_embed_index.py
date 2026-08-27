import numpy as np
import pytest

from innovation.ideas.embed import (FakeEmbedder, load_embeddings,
                                    save_embeddings)
from innovation.network.index import VectorIndex


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder()
    v1 = e.encode(["hello", "world"])
    v2 = e.encode(["hello", "world"])
    assert v1.shape == (2, e.dim)
    assert v1.dtype == np.float32
    np.testing.assert_allclose(v1, v2)
    np.testing.assert_allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(v1[0], v1[1])


def test_vector_index_returns_nearest_first():
    e = FakeEmbedder()
    vecs = e.encode(["alpha", "beta", "gamma"])
    idx = VectorIndex(e.dim)
    idx.add(["a", "b", "g"], vecs)
    hits = idx.search(vecs[0], k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 2


def test_embeddings_roundtrip(tmp_path):
    e = FakeEmbedder()
    vecs = e.encode(["x", "y"])
    save_embeddings(["x", "y"], vecs, tmp_path)
    ids, loaded = load_embeddings(tmp_path)
    assert ids == ["x", "y"]
    np.testing.assert_allclose(loaded, vecs)


def test_empty_index_returns_no_hits():
    idx = VectorIndex(8)
    assert idx.search(np.zeros(8, dtype=np.float32), k=3) == []
