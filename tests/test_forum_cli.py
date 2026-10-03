"""Spec §8's invariant test: the corpus is read-only.

This assertion is the empirical backing for the paper's central architectural
claim, so the run under it has to be a real one — all nine actions, including
the only paths that touch corpus ids at all (`add_links` / `remove_links` aimed
at a corpus destination, and attempted from a corpus source) — and the
comparison has to cover node payloads, not just the node and edge sets.
"""
import json

import numpy as np
import pytest

from innovation.core.events import EventLog, load_events
from innovation.core.network.graph import FrozenGraphError
from innovation.core.network.index import FrozenIndexError
from innovation.p2_forum.agent import VALID_ACTIONS
from innovation.p2_forum.env import Action, ForumEnvironment
from innovation.p2_forum.runner import ForumRunConfig, run_forum

# Ids are deterministic: run_id "inv", so the two posts are gen:inv:0 and
# gen:inv:1, and the corpus fixture holds p1 and p2 (p2 cites p1).
SCRIPT = [
    ("generate", {"text": "first idea", "cited_ids": ["p1"]}),
    ("search", {"query": "paper", "k": 5}),
    ("browse", {"node_id": "p1"}),
    ("sample_frontier", {}),
    ("generate", {"text": "second idea", "cited_ids": ["gen:inv:0", "p2"]}),
    ("search_board", {"query": "idea", "k": 5}),
    ("browse_board", {"node_id": "gen:inv:0"}),
    ("sample_board", {}),
    # the only actions that name a corpus id as a link destination
    ("add_links", {"src_id": "gen:inv:1", "dst_ids": ["p1"]}),
    ("add_links", {"src_id": "gen:inv:0", "dst_ids": ["p2"]}),
    ("remove_links", {"src_id": "gen:inv:0", "dst_ids": ["p2"]}),
    # and the two that try to make the corpus itself the source (must be refused)
    ("add_links", {"src_id": "p1", "dst_ids": ["p2"]}),
    ("remove_links", {"src_id": "p2", "dst_ids": ["p1"]}),
    # a post->post link, so the board really does gain structure
    ("add_links", {"src_id": "gen:inv:0", "dst_ids": ["gen:inv:1"]}),
]


class CyclingLLM:
    """Drives every one of the nine actions in turn, then repeats."""

    def __init__(self):
        self.n = 0

    def complete(self, *, model, system, user, max_tokens):
        name, args = SCRIPT[self.n % len(SCRIPT)]
        self.n += 1
        return json.dumps({"action": name, "args": args})


def _snapshot(corpus, index):
    """Node set, edge set, every node's payload, and the index, together."""
    return {
        "nodes": sorted(corpus.node_ids()),
        "edges": sorted((s, d) for s in corpus.node_ids()
                        for d in corpus.citations_out(s)),
        "payload": {n: (corpus.node(n).text, corpus.node(n).year,
                        corpus.node(n).source,
                        json.dumps(corpus.node(n).meta, sort_keys=True))
                    for n in sorted(corpus.node_ids())},
        "index_ids": list(index.ids),
        "index_vecs": index.vecs.tobytes(),
    }


def test_a_full_run_leaves_the_corpus_bit_identical(tmp_path, make_workspace,
                                                    fake_embedder):
    ws = make_workspace()
    corpus, index = ws.corpus, ws.corpus_index
    before = _snapshot(corpus, index)

    cfg = ForumRunConfig(run_id="inv", seed=0, total_steps=len(SCRIPT),
                         agents=[{"agent_id": "a0", "k_topics": 1},
                                 {"agent_id": "a1", "k_topics": 1}],
                         topic_pool=["alpha", "beta"])
    out = run_forum(cfg, corpus=corpus, corpus_index=index,
                    embedder=fake_embedder, llm=CyclingLLM(), model="m",
                    out_dir=tmp_path)

    assert _snapshot(corpus, index) == before
    assert len(out["generated"]) == 2

    events = load_events(tmp_path / "inv" / "events.jsonl")
    # the run really did drive all nine old-mode actions ("related" is online-only)
    assert {e["action"] for e in events} == VALID_ACTIONS - {"related"}
    by_action = {}
    for e in events:
        by_action.setdefault(e["action"], []).append(e["result"])
    # the corpus-destination links landed on the board
    assert by_action["add_links"][0] == {"added": ["p1"], "skipped": []}
    # the edge removed is the one add_links had just made, so it is typed
    # agent_link rather than a generate-time citation
    assert by_action["remove_links"][0]["removed"] == [
        {"dst_id": "p2", "etype": "agent_link"}]
    # and the corpus-source attempts were refused, which is the path that would
    # have mutated the corpus if the write rule were not centralized
    assert "board node" in by_action["add_links"][2]["error"]
    assert "board node" in by_action["remove_links"][1]["error"]


def test_the_corpus_index_never_gains_a_board_vector(tmp_path, make_workspace):
    ws = make_workspace()
    n_before = len(ws.corpus_index.ids)
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "e.jsonl"),
                           rng=np.random.default_rng(0))

    env.execute("a", 0, Action("generate", {"text": "x", "cited_ids": []}))

    assert len(ws.corpus_index.ids) == n_before
    assert len(ws.board_index.ids) == 1


def test_frozen_corpus_rejects_writes_even_directly(make_workspace):
    ws = make_workspace()
    with pytest.raises(FrozenGraphError):
        ws.corpus.add_links("p2", ["p1"])
    with pytest.raises(FrozenGraphError):
        ws.corpus.remove_links("p2", ["p1"])
    with pytest.raises(FrozenGraphError):
        ws.corpus.add_idea("p3", "smuggled", [], source="corpus")


# --- I6: read-only is a property of the type, not a convention (spec §3.1) ---

def test_a_corpus_nodes_payload_cannot_be_rewritten(make_workspace):
    ws = make_workspace()

    ws.node("p1").text = "REWRITTEN"
    ws.node("p1").meta["venue"] = "forged"
    ws.corpus.node("p1").year = 1999

    assert ws.corpus.node("p1").text == "paper one"
    assert ws.corpus.node("p1").meta == {}
    assert ws.corpus.node("p1").year == 2020


def test_the_board_still_hands_out_live_nodes(make_workspace):
    """Only the frozen store detaches its payload; the board is mutable state
    and paper 1 (which never freezes) must keep the behaviour it had."""
    ws = make_workspace()
    nid = ws.post_idea("a post", [], {"agent_id": "a0"})

    assert ws.board.node(nid) is ws.board.node(nid)


def test_the_corpus_index_vectors_cannot_be_rewritten_in_place(make_workspace):
    ws = make_workspace()

    assert ws.corpus_index.frozen
    with pytest.raises(ValueError):
        ws.corpus_index.vecs[0][0] = 99.0
    assert ws.corpus_index.vecs[0][0] == 1.0


def test_the_corpus_index_refuses_new_vectors(make_workspace):
    ws = make_workspace()

    with pytest.raises(FrozenIndexError):
        ws.corpus_index.add(["gen:x:0"], np.zeros((1, 4), dtype=np.float32))
    assert ws.corpus_index.ids == ["p1", "p2"]


def test_a_workspace_refuses_a_frozen_board_index(make_workspace):
    from innovation.core.network.index import VectorIndex
    from innovation.p2_forum.workspace import Workspace

    ws = make_workspace()
    frozen_board = VectorIndex(4)
    frozen_board.freeze()

    with pytest.raises(ValueError, match="board index must stay writable"):
        Workspace(corpus=ws.corpus, corpus_index=ws.corpus_index,
                  board_index=frozen_board, embedder=ws.embedder, run_id="t")


def test_online_eval_reference_collects_shown_papers(tmp_path, monkeypatch):
    from conftest import FakeEmbedder
    from innovation import cli
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    ev = [{"action": "search", "result": {"hits": [{"node_id": "p0", "title": "A", "text": "x"}]}},
          {"action": "browse", "result": {"node_id": "p1", "title": "B", "text": "y",
                                          "cites": [{"node_id": "p2", "title": "C"}], "cited_by": []}},
          {"action": "generate", "result": {"node_id": "gen:r:0", "topics": [0]}}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    cfg = {"literature": "online", "out_dir": str(tmp_path / "runs"), "run": {"run_id": "r"},
           "embedding_model": "fake"}
    monkeypatch.setattr(cli, "Embedder", lambda name: FakeEmbedder())
    titles, vecs = cli._eval_reference(cfg)
    assert titles == {"a", "b", "c"} and vecs.shape[0] == 3


def test_shown_papers_includes_related_hits():
    from innovation import cli
    ev = [{"action": "related", "result": {"node_id": "p0", "related": [
        {"node_id": "p5", "title": "R", "text": "t"}], "filtered": {"scope": 0, "topic": 0}}}]
    assert cli._shown_papers(ev) == {"p5": ("R", "t")}
