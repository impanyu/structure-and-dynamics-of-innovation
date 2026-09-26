import numpy as np
import pytest

from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.workspace import Workspace


class FakeEmbedder:
    dim = 4

    def encode(self, texts):
        return np.array([[len(t) % 3, 1.0, 0.0, 0.0] for t in texts],
                        dtype=np.float32)


def _make_workspace() -> Workspace:
    corpus = IdeaGraph()
    corpus.add_idea("p1", "paper one", [], source="corpus", year=2020)
    corpus.add_idea("p2", "paper two", ["p1"], source="corpus", year=2021)
    corpus.freeze()
    ci = VectorIndex(4)
    ci.add(["p1", "p2"], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    return Workspace(corpus=corpus, corpus_index=ci,
                     board_index=VectorIndex(4), embedder=FakeEmbedder(),
                     run_id="t")


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


@pytest.fixture
def make_workspace():
    """Factory, not a single instance: several tests need two independent
    workspaces (e.g. to replay an event log into a fresh one)."""
    return _make_workspace
