import json

import pandas as pd

from innovation.experiments.events import load_events
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import FakeEmbedder
from innovation.llm import FakeLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def fixtures():
    ideas = pd.DataFrame(
        [{"paper_id": f"W{i}", "idea_text": f"idea {i}", "year": 2020, "venue": "V"}
         for i in range(6)])
    edges = pd.DataFrame([{"src": f"W{i}", "dst": "W0"} for i in range(1, 6)])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    index.add(ids, emb.encode([graph.node(n).text for n in ids]))
    return graph, index, emb


def test_round_robin_two_llm_agents_share_budget(tmp_path):
    graph, index, emb = fixtures()
    gen = json.dumps({"action": "generate",
                      "args": {"text": "a new idea", "cited_ids": ["W0"]}})
    llm = FakeLLM(default=gen)  # every agent generates every turn
    cfg = RunConfig(run_id="r1", seed=0, total_steps=6, generation_budget=3,
                    agents=[{"agent_id": "a0", "policy": "llm"},
                            {"agent_id": "a1", "policy": "llm"}])
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=llm, model="m", out_dir=tmp_path)
    assert out["steps"] == 6
    assert len(out["generated"]) == 3  # global budget enforced across agents
    events = load_events(tmp_path / "r1" / "events.jsonl")
    assert [e["agent_id"] for e in events[:4]] == ["a0", "a1", "a0", "a1"]  # round-robin


def test_mixed_policies_and_determinism(tmp_path):
    graph, index, emb = fixtures()
    cfg = RunConfig(run_id="r2", seed=42, total_steps=4, generation_budget=4,
                    agents=[{"agent_id": "pa", "policy": "pa", "m": 2},
                            {"agent_id": "nn", "policy": "nonav", "k": 2}])
    llm = FakeLLM(default="A generated idea paragraph.")
    out1 = run_simulation(cfg, graph=fixtures()[0], index=fixtures()[1],
                          embedder=emb, llm=llm, model="m", out_dir=tmp_path / "x")
    out2 = run_simulation(cfg, graph=fixtures()[0], index=fixtures()[1],
                          embedder=emb, llm=llm, model="m", out_dir=tmp_path / "y")
    e1 = load_events(tmp_path / "x" / "r2" / "events.jsonl")
    e2 = load_events(tmp_path / "y" / "r2" / "events.jsonl")
    assert [ev["args"] for ev in e1] == [ev["args"] for ev in e2]  # same seed, same trace
    assert len(out1["generated"]) == len(out2["generated"]) == 4
