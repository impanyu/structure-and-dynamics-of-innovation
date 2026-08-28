import numpy as np
import pytest

from innovation.eval.metrics import (aggregate_run, bridging, diversity,
                                     novelty, past_dup_flag, precision)
from innovation.eval.search_verify import Verdict


def verdicts():
    return [Verdict("g0", True, {"paper_id": "P1"}),
            Verdict("g1", True, {"paper_id": "P1"}),   # same paper twice
            Verdict("g2", False, None),
            Verdict("g3", True, {"paper_id": "P2"})]


def test_precision_excludes_dup_flagged_from_numerator():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    # hits g0,g1 count; g3 is a hit but dup-flagged -> excluded; 2/4
    assert precision(verdicts(), dup) == pytest.approx(0.5)


def test_embedding_metrics():
    corpus = np.array([[1, 0], [0, 1]], dtype=np.float32)
    v_new = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=np.float32)
    assert novelty(v_new, corpus) == pytest.approx(1 - np.cos(np.pi / 4), abs=1e-5)
    assert past_dup_flag(np.array([1, 0], dtype=np.float32), corpus, ceiling=0.95)
    assert not past_dup_flag(v_new, corpus, ceiling=0.95)
    assert diversity(np.array([[1, 0], [0, 1]], dtype=np.float32)) == pytest.approx(1.0)
    assert bridging(["a", "b", "c"], {"a": 0, "b": 0, "c": 2}) == 2


def test_aggregate_run():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    agg = aggregate_run(verdicts(), dup)
    assert agg == {"precision": 0.5, "n_ideas": 4,
                   "n_hits": 3, "n_dup_flagged": 1}


def test_empty_edge_cases():
    assert precision([], {}) == 0.0
    assert diversity(np.array([[1.0, 0.0]], dtype=np.float32)) == 0.0
