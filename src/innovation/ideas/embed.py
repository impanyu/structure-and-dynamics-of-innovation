"""Embeddings: real sentence-transformers model + a deterministic test fake."""
import hashlib
import json
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class FakeEmbedder:
    """Deterministic vectors from sha256(text); unit tests never download models."""

    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(digest[: self.dim * 4], dtype=np.uint32).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def save_embeddings(ids: list[str], vecs: np.ndarray, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "embedding_ids.json").write_text(json.dumps(ids))


def load_embeddings(out_dir):
    out = Path(out_dir)
    ids = json.loads((out / "embedding_ids.json").read_text())
    return ids, np.load(out / "embeddings.npy")
