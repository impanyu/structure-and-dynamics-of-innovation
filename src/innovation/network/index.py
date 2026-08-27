"""Brute-force cosine index behind a FAISS-swappable interface (plan: Global Constraints)."""
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vecs = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        assert vecs.shape == (len(ids), self.dim)
        self.ids.extend(ids)
        self.vecs = np.vstack([self.vecs, vecs.astype(np.float32)])

    def search(self, vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = self.vecs @ vec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]
