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
