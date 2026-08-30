import numpy as np
import pytest

from innovation.eval.metrics import (aggregate_run, bridging, diversity,
                                     idea_levels, novelty, past_dup_flag)
from innovation.eval.search_verify import Verdict


def make_verdict(idea_id, t1=0, t3=None):
    v = Verdict(idea_id)
    v.best["tier1"] = {"level": t1, "paper": {"paper_id": "P"} if t1 else None}
    v.best["tier2"] = dict(v.best["tier1"])
    t3 = t1 if t3 is None else t3
    v.best["tier3"] = {"level": t3, "paper": {"paper_id": "P"} if t3 else None}
    return v


def verdicts():
    return [make_verdict("g0", t1=5),
            make_verdict("g1", t1=3),
            make_verdict("g2", t1=0),
            make_verdict("g3", t1=4)]


def test_idea_levels_zeroes_dup_flagged():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    assert idea_levels(verdicts(), dup, "tier1") == [5, 3, 0, 0]


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
    t1 = agg["tier1"]
    assert t1["mean_level"] == pytest.approx(2.0)   # (5+3+0+0)/4
    assert t1["acc_ge3"] == pytest.approx(0.5)
    assert t1["acc_ge4"] == pytest.approx(0.25)
    assert t1["acc_eq5"] == pytest.approx(0.25)
    assert agg["n_ideas"] == 4 and agg["n_dup_flagged"] == 1
    assert agg["tier3"]["mean_level"] >= t1["mean_level"] - 1e-9


def test_empty_edge_cases():
    assert aggregate_run([], {})["tier1"]["mean_level"] == 0.0
    assert diversity(np.array([[1.0, 0.0]], dtype=np.float32)) == 0.0
