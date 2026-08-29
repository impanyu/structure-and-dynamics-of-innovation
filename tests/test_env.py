import numpy as np
import pandas as pd

from innovation.experiments.env import Action, Environment
from innovation.experiments.events import EventLog
from innovation.ideas.embed import FakeEmbedder
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def make_env(tmp_path, budget=5):
    ideas = pd.DataFrame([
        {"paper_id": "W1", "idea_text": "deep nets", "year": 2019, "venue": "V"},
        {"paper_id": "W2", "idea_text": "graph nets", "year": 2020, "venue": "V"},
    ])
    edges = pd.DataFrame([{"src": "W2", "dst": "W1"}])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    index.add(ids, emb.encode([graph.node(n).text for n in ids]))
    log = EventLog(tmp_path / "events.jsonl")
    return Environment(run_id="r1", graph=graph, index=index, embedder=emb,
                       event_log=log, rng=np.random.default_rng(0),
                       generation_budget=budget)


def test_search_browse_sample_and_logging(tmp_path):
    env = make_env(tmp_path)
    res = env.execute("a0", 0, Action("search", {"query": "deep nets", "k": 1}))
    assert res["hits"][0]["node_id"] == "W1"
    res = env.execute("a0", 1, Action("browse", {"node_id": "W2"}))
    assert res["cites"][0]["node_id"] == "W1"
    res = env.execute("a0", 2, Action("sample_frontier", {}))
    assert res["node_id"] in {"W1", "W2"}
    events = env.event_log.read_all()
    assert [e["action"] for e in events] == ["search", "browse", "sample_frontier"]
    assert events[0]["agent_id"] == "a0" and events[1]["step"] == 1


def test_generate_adds_node_indexes_it_and_decrements_budget(tmp_path):
    env = make_env(tmp_path, budget=1)
    res = env.execute("a0", 0, Action("generate",
                                      {"text": "combine deep and graph nets",
                                       "cited_ids": ["W1", "W2"]}))
    assert res["node_id"] == "gen:r1:0"
    assert env.graph.node("gen:r1:0").meta["agent_id"] == "a0"
    # The new idea is immediately searchable (stigmergy channel, spec §3.4).
    hits = env.execute("a0", 1, Action("search", {"query": "combine deep and graph nets", "k": 1}))
    assert hits["hits"][0]["node_id"] == "gen:r1:0"
    # Budget exhausted -> error, no node added.
    res2 = env.execute("a0", 2, Action("generate", {"text": "x", "cited_ids": ["W1"]}))
    assert "error" in res2
    assert env.generated_ids() == ["gen:r1:0"]


def test_generate_rejects_unknown_citation_and_unknown_action(tmp_path):
    env = make_env(tmp_path)
    res = env.execute("a0", 0, Action("generate", {"text": "x", "cited_ids": ["nope"]}))
    assert "error" in res
    res = env.execute("a0", 1, Action("fly_to_moon", {}))
    assert "error" in res


def test_restore_replays_generate_events_without_relogging(tmp_path):
    env = make_env(tmp_path)
    env.execute("a0", 0, Action("generate", {"text": "new", "cited_ids": ["W1"]}))
    events = env.event_log.read_all()

    env2 = make_env(tmp_path / "fresh")
    env2.restore(events)
    assert env2.graph.has_node("gen:r1:0")
    assert env2.event_log.read_all() == []          # restore does not log
    res = env2.execute("a0", 1, Action("generate", {"text": "next", "cited_ids": ["W1"]}))
    assert res["node_id"] == "gen:r1:1"             # counter continues


def test_add_links_action_logs_and_restores(tmp_path):
    env = make_env(tmp_path)
    res = env.execute("a0", 0, Action("add_links", {"src_id": "W1", "dst_ids": ["W2"]}))
    assert res == {"added": ["W2"], "skipped": []}
    res2 = env.execute("a0", 1, Action("add_links", {"src_id": "nope", "dst_ids": ["W1"]}))
    assert "error" in res2
    events = env.event_log.read_all()
    assert [e["action"] for e in events] == ["add_links", "add_links"]
    # restore replays only the successful add_links, without logging
    env2 = make_env(tmp_path / "fresh")
    env2.restore(events)
    assert "W2" in env2.graph.citations_out("W1")
    assert env2.event_log.read_all() == []


def test_remove_links_action_logs_and_restores(tmp_path):
    env = make_env(tmp_path)
    # W2->W1 exists in the fixture
    res = env.execute("a0", 0, Action("remove_links", {"src_id": "W2", "dst_ids": ["W1"]}))
    assert res["removed"] == [{"dst_id": "W1", "etype": "citation"}]
    assert "W1" not in env.graph.citations_out("W2")
    res2 = env.execute("a0", 1, Action("remove_links", {"src_id": "nope", "dst_ids": ["W1"]}))
    assert "error" in res2
    events = env.event_log.read_all()
    env2 = make_env(tmp_path / "fresh")
    env2.restore(events)
    assert "W1" not in env2.graph.citations_out("W2")
    assert env2.event_log.read_all() == []


def make_scoped_env(tmp_path, scopes, communities):
    ideas = pd.DataFrame([
        {"paper_id": "A1", "idea_text": "alpha one", "year": 2019, "venue": "V"},
        {"paper_id": "A2", "idea_text": "alpha two", "year": 2020, "venue": "V"},
        {"paper_id": "B1", "idea_text": "beta one", "year": 2020, "venue": "V"},
    ])
    edges = pd.DataFrame([{"src": "A2", "dst": "A1"}])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    index.add(ids, emb.encode([graph.node(n).text for n in ids]))
    log = EventLog(tmp_path / "events.jsonl")
    return Environment(run_id="r1", graph=graph, index=index, embedder=emb,
                       event_log=log, rng=np.random.default_rng(0),
                       generation_budget=5, scopes=scopes,
                       communities=communities)


COMM = {"A1": 0, "A2": 0, "B1": 1}


def test_scoped_read_filters_search_browse_and_jump(tmp_path):
    from innovation.experiments.env import AgentScope
    scopes = {"a0": AgentScope(read={0}, write={0})}
    env = make_scoped_env(tmp_path, scopes, COMM)
    hits = env.execute("a0", 0, Action("search", {"query": "one", "k": 3}))
    assert {h["node_id"] for h in hits["hits"]} <= {"A1", "A2"}
    assert "error" in env.execute("a0", 1, Action("browse", {"node_id": "B1"}))
    for step in range(2, 8):  # jump stays inside the readable region
        res = env.execute("a0", step, Action("sample_frontier", {}))
        assert res["node_id"] in {"A1", "A2"}
    # unrestricted agent still sees everything
    assert "error" not in env.execute("other", 9, Action("browse", {"node_id": "B1"}))


def test_no_jump_scope_blocks_sample_frontier(tmp_path):
    from innovation.experiments.env import AgentScope
    scopes = {"a0": AgentScope(allow_jump=False)}
    env = make_scoped_env(tmp_path, scopes, COMM)
    assert "error" in env.execute("a0", 0, Action("sample_frontier", {}))


def test_scoped_write_constrains_generate_and_links(tmp_path):
    from innovation.experiments.env import AgentScope
    scopes = {"a0": AgentScope(read=None, write={0})}  # read all, write only C0
    env = make_scoped_env(tmp_path, scopes, COMM)
    res = env.execute("a0", 0, Action("generate", {"text": "x", "cited_ids": ["A1", "B1"]}))
    assert "error" in res  # B1 is outside the write scope
    res = env.execute("a0", 1, Action("generate", {"text": "x", "cited_ids": ["A1", "A2"]}))
    assert res["node_id"] == "gen:r1:0"
    # new node inherits majority community of its citations -> writable/readable
    res = env.execute("a0", 2, Action("add_links", {"src_id": "gen:r1:0", "dst_ids": ["A2"]}))
    assert "error" not in res
    assert "error" in env.execute("a0", 3, Action("add_links", {"src_id": "B1", "dst_ids": ["A1"]}))
    assert "error" in env.execute("a0", 4, Action("remove_links", {"src_id": "B1", "dst_ids": ["A1"]}))


def test_semantic_radius_scope_read_and_write(tmp_path):
    from innovation.experiments.env import AgentScope
    emb = FakeEmbedder()
    # Anchor = exact text of A1's idea; tiny radius admits only that node.
    anchors = emb.encode(["alpha one"])
    scopes = {"a0": AgentScope(read_anchors=anchors, read_radius=1e-6,
                               write_anchors=anchors, write_radius=1e-6)}
    env = make_scoped_env(tmp_path, scopes, communities=None)
    hits = env.execute("a0", 0, Action("search", {"query": "alpha one", "k": 3}))
    assert {h["node_id"] for h in hits["hits"]} == {"A1"}
    assert "error" in env.execute("a0", 1, Action("browse", {"node_id": "B1"}))
    assert "error" not in env.execute("a0", 2, Action("browse", {"node_id": "A1"}))
    res = env.execute("a0", 3, Action("sample_frontier", {}))
    assert res["node_id"] == "A1"  # only readable node
    # citations follow the READ scope: an out-of-region cite is DROPPED (not
    # fatal); the idea itself must still sit inside the write region.
    res = env.execute("a0", 4, Action("generate", {"text": "alpha one",
                                                   "cited_ids": ["A2", "A1"]}))
    assert res["node_id"] == "gen:r1:0"
    assert res["dropped_cites"] == ["A2"]
    assert env.graph.citations_out("gen:r1:0") == ["A1"]
    res = env.execute("a0", 5, Action("generate", {"text": "alpha one",
                                                   "cited_ids": ["A1"]}))
    assert res["node_id"] == "gen:r1:1"
    # the generated node's own embedding places it inside the region
    assert "error" not in env.execute("a0", 6, Action("browse", {"node_id": "gen:r1:0"}))


def test_broad_reader_cites_anywhere(tmp_path):
    """Read-unrestricted, write-scoped agents may cite ANY paper."""
    from innovation.experiments.env import AgentScope
    emb = FakeEmbedder()
    anchors = emb.encode(["alpha one"])
    scopes = {"a0": AgentScope(write_anchors=anchors, write_radius=1e-6)}
    env = make_scoped_env(tmp_path, scopes, communities=None)
    res = env.execute("a0", 0, Action("generate", {"text": "alpha one",
                                                   "cited_ids": ["B1", "A2"]}))
    assert "node_id" in res and "dropped_cites" not in res
    assert set(env.graph.citations_out(res["node_id"])) == {"B1", "A2"}


def test_semantic_write_scope_binds_generated_text(tmp_path):
    from innovation.experiments.env import AgentScope
    emb = FakeEmbedder()
    anchors = emb.encode(["alpha one"])
    scopes = {"a0": AgentScope(write_anchors=anchors, write_radius=1e-6)}
    env = make_scoped_env(tmp_path, scopes, communities=None)
    # citations are inside the region proxy is irrelevant here: A1 is writable
    # (its text IS the anchor), but the new idea's own text is off-topic.
    res = env.execute("a0", 0, Action("generate", {"text": "totally different",
                                                   "cited_ids": ["A1"]}))
    assert res == {"error": "the idea itself is outside your writable topic region"}
    res = env.execute("a0", 1, Action("generate", {"text": "alpha one",
                                                   "cited_ids": ["A1"]}))
    assert "node_id" in res


def test_per_anchor_radii_membership(tmp_path):
    """AgentScope radii may be per-anchor arrays (equal-mass regions)."""
    import numpy as np
    from innovation.experiments.env import AgentScope
    emb = FakeEmbedder()
    anchors = emb.encode(["alpha one", "totally different"])
    # anchor 0 tight (only exact match), anchor 1 radius 0 (nothing)
    scopes = {"a0": AgentScope(read_anchors=anchors,
                               read_radius=np.array([1e-6, 0.0]))}
    env = make_scoped_env(tmp_path, scopes, communities=None)
    assert "error" not in env.execute("a0", 0, Action("browse", {"node_id": "A1"}))
    assert "error" in env.execute("a0", 1, Action("browse", {"node_id": "B1"}))
