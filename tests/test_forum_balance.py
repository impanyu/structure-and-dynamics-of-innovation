import json
import math

import numpy as np
import pytest

from innovation.core.llm import FakeLLM
from innovation.p2_forum.balance import (LLMRewriter, band, choose_merge_target,
                                         colabel_rate, parse_merge, parse_split,
                                         rebalance_round, restore_count, run_balance,
                                         split_count, write_outputs)
from innovation.p2_forum.capacity import Estimate
from innovation.p2_forum.topics import load_topics


def test_band_is_tau_over_and_times_sqrt3_so_max_over_min_is_3():
    tau, lo, hi = band([10, 20, 30, 40], n=4)
    assert tau == 25
    assert (lo, hi) == pytest.approx((25 / math.sqrt(3), 25 * math.sqrt(3)))
    assert hi / lo == pytest.approx(3)


@pytest.mark.parametrize("c, p", [(50, 2), (100, 4), (130, 5), (44, 2)])
def test_split_count(c, p):
    assert split_count(c, tau=25) == p


def W(name, cap, members, definition=None):
    return {"name": name, "definition": definition or name.lower(), "sources": ["AAAI"],
            "capacity": cap, "members": frozenset(members)}


class TableSim:
    """Cosine from a fixed table keyed by unordered name pairs (default 0)."""

    def __init__(self, table=None):
        self.table = {frozenset(k): v for k, v in (table or {}).items()}

    def __call__(self, a, b):
        return self.table.get(frozenset((a["name"], b["name"])), 0.0)


def test_colabel_rate_conditions_on_the_first_topic():
    labels = {"p1": [0, 1], "p2": [0], "p3": [1, 2], "p4": None}
    assert colabel_rate(labels, frozenset({0}), frozenset({1})) == 0.5
    assert colabel_rate(labels, frozenset({1}), frozenset({0})) == 0.5
    assert colabel_rate(labels, frozenset({5}), frozenset({0})) == 0.0


def test_merge_target_prefers_colabeling_over_cosine():
    s, a, b = W("S", 1, {0}), W("A", 5, {1}), W("B", 5, {2})
    labels = {"p1": [0, 1], "p2": [0]}
    sim = TableSim({("S", "B"): 0.99, ("S", "A"): 0.1})
    assert choose_merge_target(s, [s, a, b], labels, sim, cap=100) is a


def test_merge_target_breaks_colabel_ties_by_cosine():
    s, a, b = W("S", 1, {0}), W("A", 5, {1}), W("B", 5, {2})
    sim = TableSim({("S", "B"): 0.9, ("S", "A"): 0.2})
    assert choose_merge_target(s, [s, a, b], {}, sim, cap=100) is b


def test_merge_target_respects_the_capacity_cap():
    s, a, b = W("S", 3, {0}), W("A", 10, {1}), W("B", 5, {2})
    labels = {"p1": [0, 1]}
    assert choose_merge_target(s, [s, a, b], labels, TableSim(), cap=9) is b
    assert choose_merge_target(s, [s, a, b], labels, TableSim(), cap=4) is None


class FakeRewriter:
    def __init__(self, refuse=()):
        self.refuse, self.splits, self.merges = set(refuse), [], []

    def split(self, t, p, titles, taken):
        self.splits.append((t["name"], p))
        return [{"name": f"{t['name']}/{k}", "definition": f"part {k}"} for k in range(p)]

    def merge(self, s, t, taken):
        self.merges.append((s["name"], t["name"]))
        if s["name"] in self.refuse:
            return None
        return {"name": f"{s['name']}+{t['name']}", "definition": "merged"}


def _est(caps, labels=None):
    return Estimate(records=[{"name": f"T{i}", "capacity": c} for i, c in enumerate(caps)],
                    labels=labels or {}, titles=[[f"title {i}"] for i in range(len(caps))])


def _topics(k):
    return [{"name": f"T{i}", "definition": f"d{i}", "sources": ["AAAI"]} for i in range(k)]


def test_round_splits_large_merges_small_and_keeps_n():
    # n = 4, tau = 25, band [14.4, 43.3]: T0 (60) splits in 2, T3 (5) merges.
    caps = [60, 20, 15, 5]
    labels = {"p": [3, 2]}                     # T3 co-labels with T2
    rw = FakeRewriter()
    topics, flags, info = rebalance_round(_topics(4), _est(caps, labels), rw, TableSim(), n=4)
    assert rw.splits == [("T0", 2)]
    assert rw.merges == [("T3", "T2")]
    assert [t["name"] for t in topics] == ["T0/0", "T0/1", "T1", "T3+T2"]
    assert flags == {}
    assert topics[0]["sources"] == ["AAAI"]


def test_no_sensible_merge_keeps_the_topic_and_flags_it():
    caps = [30, 30, 30, 5]
    rw = FakeRewriter(refuse={"T3"})
    topics, flags, _ = rebalance_round(_topics(4), _est(caps), rw, TableSim(), n=4)
    assert [t["name"] for t in topics] == ["T0", "T1", "T2", "T3"]
    assert "no sensible merge" in flags["T3"]


def test_restore_merges_smallest_when_too_many_and_skips_refusals():
    work = [W("A", 10, {0}), W("B", 1, {1}), W("C", 2, {2}), W("D", 20, {3})]
    rw = FakeRewriter(refuse={"B"})
    out = restore_count(work, 3, {}, TableSim({("C", "A"): 0.5}), rw,
                        cap=100, titles_of=lambda w: [])
    assert [w["name"] for w in out] == ["C+A", "B", "D"]
    assert out[0]["capacity"] == 12


def test_restore_splits_largest_when_too_few():
    work = [W("A", 10, {0}), W("B", 40, {1})]
    out = restore_count(work, 3, {}, TableSim(), FakeRewriter(), cap=100,
                        titles_of=lambda w: [])
    assert [w["name"] for w in out] == ["A", "B/0", "B/1"]
    assert out[1]["capacity"] == 20


def test_restore_raises_when_every_merge_is_refused():
    work = [W("A", 10, {0}), W("B", 1, {1})]
    with pytest.raises(RuntimeError, match="refused"):
        restore_count(work, 1, {}, TableSim(), FakeRewriter(refuse={"A", "B"}),
                      cap=100, titles_of=lambda w: [])


def test_run_stops_when_in_band():
    calls = []

    def estimate(topics):
        calls.append(len(topics))
        return _est([25] * len(topics))
    out = run_balance(_topics(4), estimate, FakeRewriter(), TableSim(), n=4, log=lambda *_: None)
    assert out["rounds"] == 0 and calls == [4] and out["exceptions"] == []
    assert out["max_min_ratio"] == 1


def test_run_terminates_after_max_rounds_and_lists_exceptions():
    calls = []

    def estimate(topics):                      # never balances: T3 always tiny
        calls.append(1)
        return _est([30, 30, 30, 1])
    out = run_balance(_topics(4), estimate, FakeRewriter(refuse={"T3"}), TableSim(),
                      n=4, max_rounds=3, log=lambda *_: None)
    assert out["rounds"] == 3 and len(calls) == 4
    assert [e["name"] for e in out["exceptions"]] == ["T3"]
    assert out["exceptions"][0]["side"] == "below"
    assert "no sensible merge" in out["exceptions"][0]["reason"]
    assert len(out["history"]) == 4


def test_parse_split_and_merge():
    reply = json.dumps([{"name": "X", "definition": "x"}, {"name": "Y", "definition": "y"}])
    assert [t["name"] for t in parse_split(reply, 2, set())] == ["X", "Y"]
    with pytest.raises(ValueError, match="already used"):
        parse_split(reply, 2, {"x"})
    with pytest.raises(ValueError, match="expected 3"):
        parse_split(reply, 3, set())
    assert parse_merge('{"no_merge": true}', set()) is None
    assert parse_merge('ok {"name": "M", "definition": "m"}', set()) == {"name": "M", "definition": "m"}


def test_llm_rewriter_retries_with_feedback():
    llm = FakeLLM(["garbage", '{"no_merge": true}'])
    rw = LLMRewriter(llm, "m")
    assert rw.merge({"name": "A", "definition": "a"}, {"name": "B", "definition": "b"}, set()) is None
    assert "rejected" in llm.calls[1]["user"]


def test_outputs_load_as_128_topics(tmp_path):
    topics = [{"name": f"Topic {i}", "definition": f"def {i}", "sources": ["AAAI"]}
              for i in range(128)]
    est = Estimate(records=[{"name": t["name"], "capacity": 10.0, "n_v": 1, "n_c": 2,
                             "n_vc": 0, "p_v": 0.5, "p_c": 0.5, "query": "a | b"}
                            for t in topics], labels={}, titles=[[]] * 128)
    result = {"topics": topics, "estimate": est, "tau": 10.0, "band": [5.8, 17.3],
              "rounds": 1, "history": [], "exceptions": [], "max_min_ratio": 1.0}
    y, j = tmp_path / "t.yaml", tmp_path / "c.json"
    write_outputs(result, y, j, header="# DRAFT\n")
    assert y.read_text().startswith("# DRAFT")
    assert len(load_topics(y, expected=128)) == 128
    payload = json.loads(j.read_text())
    assert {"tau", "band", "rounds", "max_min_ratio", "exceptions", "topics"} <= set(payload)
    assert set(payload["topics"][0]) >= {"name", "capacity", "n_v", "n_c", "n_vc",
                                         "p_v", "p_c", "query"}


def test_similarity_uses_normalized_embeddings():
    from innovation.p2_forum.balance import Similarity

    def embed(texts):
        return np.array([[2.0, 0.0] if "a" in t else [1.0, 1.0] for t in texts])
    sim = Similarity(embed)
    assert sim({"name": "a", "definition": ""}, {"name": "b", "definition": ""}) == pytest.approx(
        1 / math.sqrt(2))


class FailingRewriter(FakeRewriter):
    def __init__(self, fail_split=(), fail_merge=(), **kw):
        super().__init__(**kw)
        self.fail_split, self.fail_merge = set(fail_split), set(fail_merge)

    def split(self, t, p, titles, taken):
        if t["name"] in self.fail_split:
            raise RuntimeError("no valid LLM answer in 3 attempts")
        return super().split(t, p, titles, taken)

    def merge(self, s, t, taken):
        if s["name"] in self.fail_merge:
            raise RuntimeError("no valid LLM answer in 3 attempts")
        return super().merge(s, t, taken)


def test_failed_split_and_merge_keep_the_topics_and_flag_them():
    caps = [60, 20, 15, 5]
    rw = FailingRewriter(fail_split={"T0"}, fail_merge={"T3"})
    topics, flags, info = rebalance_round(_topics(4), _est(caps), rw, TableSim(), n=4)
    assert [t["name"] for t in topics] == ["T0", "T1", "T2", "T3"]
    assert flags["T0"].startswith("split failed") and flags["T3"].startswith("merge failed")
    assert info["splits"] == [] and info["merges"] == []


def test_a_topic_whose_estimate_failed_is_frozen_and_listed():
    est = _est([30, 30, 30, 0])
    est.records[3]["error"] = "no valid query (LLM and name fallback)"
    rw = FakeRewriter()
    topics, flags, _ = rebalance_round(_topics(4), est, rw, TableSim(), n=4)
    assert rw.merges == [] and [t["name"] for t in topics] == ["T0", "T1", "T2", "T3"]
    assert flags["T3"].startswith("no valid query")

    def estimate(ts):
        e = _est([25, 25, 25, 25])
        e.records[3]["error"] = "no valid query (LLM and name fallback)"
        return e
    out = run_balance(_topics(4), estimate, rw, TableSim(), n=4, log=lambda *_: None)
    assert [(x["name"], x["side"]) for x in out["exceptions"]] == [("T3", "in band")]
    assert out["exceptions"][0]["reason"].startswith("no valid query")


def test_frozen_topics_are_never_merge_targets():
    s, a = W("S", 1, {0}), W("A", 5, {1})
    a["frozen"] = True
    assert choose_merge_target(s, [s, a], {"p": [0, 1]}, TableSim(), cap=None) is None


def test_restore_moves_past_a_failed_split():
    work = [W("A", 10, {0}), W("B", 40, {1})]
    out = restore_count(work, 3, {}, TableSim(), FailingRewriter(fail_split={"B"}),
                        cap=100, titles_of=lambda w: [])
    assert [w["name"] for w in out] == ["A/0", "A/1", "B"]


def test_round_is_reverted_when_n_cannot_be_restored():
    # T0 splits (4 -> 6 topics) and every merge is refused: n cannot be restored.
    caps = [80, 20, 15, 5]

    class RefuseAll(FakeRewriter):
        def merge(self, s, t, taken):
            return None
    rw = RefuseAll()
    topics, flags, info = rebalance_round(_topics(4), _est(caps), rw, TableSim(), n=4)
    assert [t["name"] for t in topics] == ["T0", "T1", "T2", "T3"]
    assert "reverted" in info and flags["T0"].startswith("round reverted")
