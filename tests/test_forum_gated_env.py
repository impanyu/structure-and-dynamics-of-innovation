# tests/test_forum_gated_env.py
import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import Action
from innovation.p2_forum.gated_env import GatedForumEnvironment
from innovation.p2_forum.literature import Paper
from innovation.p2_forum.tagger import UnlabeledError
from innovation.p2_forum.workspace import Workspace
from conftest import FakeEmbedder

NAMES = [f"T{i}" for i in range(4)]
P = {pid: Paper(pid, f"title {pid}", f"abstract {pid}", 2023, "NeurIPS", "2023-01-01", 0, "venue")
     for pid in ["p0", "p1", "p2", "p3"]}
LAB = {"p0": [0], "p1": [1], "p2": [0, 2], "p3": [3]}


class FakeLit:
    def __init__(self):
        self.refs = {"p0": ["p1", "p2"]}
        self.cits = {"p0": ["p3"]}
        self.last_scope_dropped = 0

    def search(self, q):
        return list(P.values())

    def get(self, pid):
        return P.get(pid)

    def references(self, pid):
        return [P[x] for x in self.refs.get(pid, [])]

    def citations(self, pid):
        return [P[x] for x in self.cits.get(pid, [])]

    def labels(self, pids):
        return {p: LAB.get(p) for p in pids}

    def has(self, pid):
        return pid in P

    def remember_ids(self, ids):
        pass


class FakeTagger:
    """Labels by keyword: text containing 'Tn' gets [n]; 'junk' fails."""
    def label(self, text):
        if "junk" in text:
            raise UnlabeledError("x")
        got = [i for i in range(4) if f"T{i}" in text]
        return got or [3]


def make(tmp_path, topics=None):
    empty = IdeaGraph()
    empty.freeze()
    ws = Workspace(corpus=empty, corpus_index=VectorIndex(4), board_index=VectorIndex(4),
                   embedder=FakeEmbedder(), run_id="t", external_papers=FakeLit())
    return GatedForumEnvironment(
        run_id="t", workspace=ws, event_log=EventLog(tmp_path / "e.jsonl"),
        rng=np.random.default_rng(0), literature=FakeLit(), tagger=FakeTagger(),
        agent_topics=topics or {"a": {0}, "b": {1}}, topic_names=NAMES)


def test_query_outside_topics_is_refused(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "about T1", "k": 5}))
    assert out["gate"] == "query" and out["topics"] == ["T1"]


def test_search_returns_only_readable_papers_with_topic_names(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "about T0", "k": 5}))
    assert [h["node_id"] for h in out["hits"]] == ["p0", "p2"]
    assert out["hits"][1]["topics"] == ["T0", "T2"]


def test_browse_unreadable_errors_and_neighbors_are_filtered(tmp_path):
    env = make(tmp_path)
    assert env.execute("a", 0, Action("browse", {"node_id": "p1"}))["gate"] == "result"
    v = env.execute("a", 1, Action("browse", {"node_id": "p0"}))
    assert [c["node_id"] for c in v["cites"]] == ["p2"]
    assert v["cited_by"] == []


def test_sample_frontier_stays_readable(tmp_path):
    env = make(tmp_path)
    for s in range(10):
        out = env.execute("a", s, Action("sample_frontier", {}))
        assert out["node_id"] in {"p0", "p2"}


def test_off_topic_post_is_not_published(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "idea on T1", "cited_ids": []}))
    assert out["gate"] == "post" and out["topics"] == [1]
    assert env.ws.board_post_ids() == []


def test_unlabeled_post_is_not_published(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "junk", "cited_ids": []}))
    assert out["gate"] == "unlabeled" and env.ws.board_post_ids() == []


def test_post_drops_unreadable_cites_and_is_readable_by_author(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": ["p0", "p1"]}))
    assert out["topics"] == [0] and out["dropped_cites"] == ["p1"]
    assert env.readable("a", out["node_id"]) and not env.readable("b", out["node_id"])


def test_board_reads_are_gated(tmp_path):
    env = make(tmp_path)
    pa = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": []}))["node_id"]
    pb = env.execute("b", 1, Action("generate", {"text": "idea T1", "cited_ids": []}))["node_id"]
    hits = env.execute("a", 2, Action("search_board", {"query": "T0", "k": 5}))["hits"]
    assert [h["node_id"] for h in hits] == [pa]
    assert env.execute("a", 3, Action("browse_board", {"node_id": pb}))["gate"] == "result"
    assert env.execute("a", 4, Action("sample_board", {}))["node_id"] == pa


def test_links_need_both_ends_readable(tmp_path):
    env = make(tmp_path)
    pa = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": []}))["node_id"]
    assert env.execute("a", 1, Action("add_links", {"src_id": pa, "dst_ids": ["p1"]}))["gate"] == "link"
    ok = env.execute("a", 2, Action("add_links", {"src_id": pa, "dst_ids": ["p2"]}))
    assert "gate" not in ok


def test_restore_reuses_logged_labels_without_tagging(tmp_path):
    env = make(tmp_path)
    env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": ["p0"]}))
    events = env.event_log.read_all()
    fresh = make(tmp_path / "x")
    fresh.tagger = None                      # any tagger call would crash
    fresh.restore(events)
    pid = events[0]["result"]["node_id"]
    assert fresh.post_labels[pid] == [0] and fresh.readable("a", pid)


def test_all_topics_means_no_gating(tmp_path):
    env = make(tmp_path, topics={"a": {0, 1, 2, 3}})
    out = env.execute("a", 0, Action("search", {"query": "about T1", "k": 9}))
    assert len(out["hits"]) == 4


class ColdLit(FakeLit):
    """has() knows only ids seen via search in this process or remembered."""
    def __init__(self):
        super().__init__()
        self.seen: set[str] = set()

    def search(self, q):
        self.seen.update(P)
        return super().search(q)

    def remember_ids(self, ids):
        self.seen.update(ids)

    def has(self, pid):
        return pid in self.seen


def make_cold(tmp_path):
    lit = ColdLit()
    empty = IdeaGraph()
    empty.freeze()
    ws = Workspace(corpus=empty, corpus_index=VectorIndex(4), board_index=VectorIndex(4),
                   embedder=FakeEmbedder(), run_id="t", external_papers=lit)
    return GatedForumEnvironment(
        run_id="t", workspace=ws, event_log=EventLog(tmp_path / "e.jsonl"),
        rng=np.random.default_rng(0), literature=lit, tagger=FakeTagger(),
        agent_topics={"a": {0}, "b": {1}}, topic_names=NAMES)


def test_restore_with_cold_literature_cache_keeps_cites_and_links(tmp_path):
    env = make_cold(tmp_path)
    env.execute("a", 0, Action("search", {"query": "about T0", "k": 5}))
    pid = env.execute("a", 1, Action("generate", {"text": "idea T0", "cited_ids": ["p0"]}))["node_id"]
    assert env.execute("a", 2, Action("add_links", {"src_id": pid, "dst_ids": ["p2"]}))["added"]
    events = env.event_log.read_all()
    fresh = make_cold(tmp_path / "x")
    fresh.tagger = None
    fresh.restore(events)
    assert fresh.ws.board.has_node("p0") and "p2" in fresh.ws.board_neighbors(pid)[0]
    fresh.tagger = FakeTagger()
    out = fresh.execute("a", 3, Action("generate", {"text": "idea T0", "cited_ids": ["p0"]}))
    assert "dropped_cites" not in out


def test_closed_corpus_search_is_refused(tmp_path):
    from innovation.p2_forum.env import Navigation
    env = make(tmp_path)
    env.nav = Navigation(corpus_search=False)
    out = env.execute("a", 0, Action("search", {"query": "about T0", "k": 5}))
    assert "closed" in out["error"] and "hits" not in out


def test_browse_out_of_scope_paper_reports_scope_gate(tmp_path):
    env = make(tmp_path)
    assert env.execute("a", 0, Action("browse", {"node_id": "zzz"}))["gate"] == "scope"


def test_search_and_browse_log_filtered_counts(tmp_path):
    env = make(tmp_path)
    env.lit.last_scope_dropped = 3
    s = env.execute("a", 0, Action("search", {"query": "about T0", "k": 1}))
    assert len(s["hits"]) == 1 and s["filtered"] == {"scope": 3, "topic": 2}
    b = env.execute("a", 1, Action("browse", {"node_id": "p0"}))
    assert b["filtered"] == {"scope": 6, "topic": 2}      # p1 (cites) and p3 (cited_by)
    f = env.execute("a", 2, Action("sample_frontier", {}))
    assert f["filtered"] == {"scope": 3, "topic": 2}


def test_search_board_logs_filtered_topic(tmp_path):
    env = make(tmp_path)
    env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": []}))
    env.execute("b", 1, Action("generate", {"text": "idea T1", "cited_ids": []}))
    out = env.execute("a", 2, Action("search_board", {"query": "T0", "k": 5}))
    assert out["filtered"] == {"scope": 0, "topic": 1}


class ManyRefsLit(FakeLit):
    """p0 cites twelve readable papers r0..r11 and one unreadable one."""
    def __init__(self):
        super().__init__()
        self.extra = {f"r{i}": Paper(f"r{i}", f"ref {i}", "long abstract " * 50, 2020 + i % 3,
                                     "ICML", "2021-01-01", 0, "venue") for i in range(12)}
        self.refs = {"p0": [f"r{i}" for i in range(12)] + ["p1"]}

    def get(self, pid):
        return self.extra.get(pid) or P.get(pid)

    def references(self, pid):
        return [self.get(x) for x in self.refs.get(pid, [])]

    def labels(self, pids):
        return {p: [0] if p in self.extra else LAB.get(p) for p in pids}


def test_browse_lists_every_readable_reference_compactly(tmp_path):
    env = make(tmp_path)
    env.lit = ManyRefsLit()
    v = env.execute("a", 0, Action("browse", {"node_id": "p0"}))
    assert [c["node_id"] for c in v["cites"]] == [f"r{i}" for i in range(12)]
    assert all(set(c) == {"node_id", "title", "year", "venue"} for c in v["cites"])
    assert v["cites"][3] == {"node_id": "r3", "title": "ref 3", "year": 2020, "venue": "ICML"}
    assert v["filtered"] == {"scope": 0, "topic": 2}       # p1 (cites) and p3 (cited_by)
    assert "text" in v and "topics" in v                   # the paper itself stays full
