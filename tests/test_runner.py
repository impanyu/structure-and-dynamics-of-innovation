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


def test_build_policy_rejects_unknown_kind():
    import numpy as np
    import pytest
    from innovation.experiments.runner import build_policy
    with pytest.raises(ValueError):
        build_policy({"policy": "quantum"}, llm=None, model="m",
                     graph=None, rng=np.random.default_rng(0))


def test_agent_scopes_flow_from_config(tmp_path):
    """Agent specs with read/write/jump constraints reach the Environment."""
    import json
    from innovation.llm import FakeLLM

    graph, index, emb = fixtures()
    gen_out = json.dumps({"action": "generate",
                          "args": {"text": "t", "cited_ids": ["W5"]}})
    llm = FakeLLM(default=gen_out)
    cfg = RunConfig(run_id="rs", seed=0, total_steps=2, generation_budget=2,
                    agents=[{"agent_id": "a0", "policy": "llm",
                             "write_communities": [-999],  # nothing writable
                             "allow_jump": False}])
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=llm, model="m", out_dir=tmp_path)
    assert out["generated"] == []  # every generate blocked by write scope
    events = load_events(tmp_path / "rs" / "events.jsonl")
    assert all("error" in e["result"] for e in events if e["action"] == "generate")


def test_topic_scope_flows_from_config(tmp_path):
    """read_topics in an agent spec becomes semantic anchors in the env."""
    import json
    import pytest
    from innovation.experiments.runner import build_scope
    from innovation.ideas.embed import FakeEmbedder
    from innovation.llm import FakeLLM

    emb = FakeEmbedder()
    scope = build_scope({"read_topics": ["idea 0"], "read_radius": 1e-6}, emb)
    assert scope.read_anchors.shape == (1, emb.dim)
    assert scope.read is None
    with pytest.raises(ValueError):
        build_scope({"read_communities": [0], "read_topics": ["x"]}, emb)

    # End-to-end: an agent whose readable region is only "idea 0" can only
    # sample-jump to W0 (fixtures name node texts "idea 0".."idea 5").
    graph, index, _ = fixtures()
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    cfg = RunConfig(run_id="rt", seed=0, total_steps=4, generation_budget=1,
                    agents=[{"agent_id": "a0", "policy": "llm",
                             "read_topics": ["idea 0"], "read_radius": 1e-6}])
    run_simulation(cfg, graph=graph, index=index, embedder=emb,
                   llm=llm, model="m", out_dir=tmp_path)
    events = load_events(tmp_path / "rt" / "events.jsonl")
    jumps = [e for e in events if e["action"] == "sample_frontier"]
    assert jumps and all(e["result"]["node_id"] == "W0" for e in jumps)


def test_random_topic_assignment_is_seeded_and_distinct(tmp_path):
    """read/write_topics: "random" resolves to one pool topic per specialist,
    distinct within a run, deterministic per seed, recorded in run_meta."""
    import json
    from innovation.llm import FakeLLM

    graph, index, emb = fixtures()
    pool = [f"topic {i}" for i in range(50)]
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    agents = [{"agent_id": f"s{i}", "policy": "llm",
               "read_topics": "random", "write_topics": "random",
               "read_radius": 0.3, "write_radius": 0.3} for i in range(5)]
    cfg = RunConfig(run_id="rta", seed=3, total_steps=2, generation_budget=1,
                    agents=agents, topic_pool=pool)
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=llm, model="m", out_dir=tmp_path / "a")
    assigned = out["topic_assignments"]
    assert set(assigned) == {f"s{i}" for i in range(5)}
    topics = list(assigned.values())
    assert len(set(topics)) == 5 and all(t in pool for t in topics)
    meta = json.loads((tmp_path / "a" / "rta" / "run_meta.json").read_text())
    assert meta["topic_assignments"] == assigned
    # same seed -> same assignment; different seed -> (almost surely) different
    out2 = run_simulation(cfg, graph=fixtures()[0], index=fixtures()[1],
                          embedder=emb, llm=llm, model="m", out_dir=tmp_path / "b")
    assert out2["topic_assignments"] == assigned


def test_equal_mass_scope_covers_exactly_k_corpus_papers():
    """read_mass: k derives a per-anchor radius = distance to the k-th nearest
    corpus paper, so every specialist region holds exactly k corpus papers."""
    import numpy as np
    from innovation.experiments.runner import build_scope
    from innovation.ideas.embed import FakeEmbedder

    emb = FakeEmbedder()
    corpus_vecs = emb.encode([f"paper {i}" for i in range(200)])
    scope = build_scope({"read_topics": ["some topic"], "read_mass": 20,
                         "write_topics": ["some topic"], "write_mass": 20},
                        emb, corpus_vecs=corpus_vecs)
    sims = corpus_vecs @ scope.read_anchors.T  # (200, 1)
    inside = (sims >= 1 - np.asarray(scope.read_radius)).any(axis=1)
    assert inside.sum() == 20
    assert (np.asarray(scope.write_radius) == np.asarray(scope.read_radius)).all()


def test_obs_carries_live_budget_status(tmp_path):
    import json
    from innovation.llm import FakeLLM

    graph, index, emb = fixtures()
    gen = json.dumps({"action": "generate", "args": {"text": "t", "cited_ids": ["W0"]}})
    llm = FakeLLM(default=gen)
    cfg = RunConfig(run_id="rb", seed=0, total_steps=3, generation_budget=5,
                    agents=[{"agent_id": "a0", "policy": "llm"}])
    run_simulation(cfg, graph=graph, index=index, embedder=emb,
                   llm=llm, model="m", out_dir=tmp_path)
    # prompt at step 2 must reflect 2 ideas already generated
    assert "team ideas 2/5" in llm.calls[2]["user"]


def test_resume_continues_steps_memory_and_gen_counter(tmp_path):
    """resume_simulation continues a finished run: steps and gen ids carry on,
    agent memories are rebuilt from the event log, topic assignments reused."""
    import json
    from innovation.experiments.events import load_events
    from innovation.experiments.runner import resume_simulation
    from innovation.llm import FakeLLM

    graph, index, emb = fixtures()
    gen = json.dumps({"action": "generate", "args": {"text": "t", "cited_ids": ["W0"]}})
    search = json.dumps({"action": "search", "args": {"query": "idea", "k": 2}})
    llm = FakeLLM(default=search)
    pool = [f"topic {i}" for i in range(10)]
    agents = [{"agent_id": "a0", "policy": "llm"},
              {"agent_id": "a1", "policy": "llm",
               "write_topics": "random", "write_mass": 3}]
    cfg = RunConfig(run_id="rr", seed=1, total_steps=2,
                    agents=agents, topic_pool=pool)
    out1 = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                          llm=llm, model="m", out_dir=tmp_path)
    assigned = out1["topic_assignments"]["a1"]

    # resume on a FRESH world (as a new process would): extend to 6 steps
    graph2, index2, emb2 = fixtures()
    llm2 = FakeLLM(default=gen)
    cfg2 = RunConfig(run_id="rr", seed=1, total_steps=6,
                     agents=agents, topic_pool=pool)
    out2 = resume_simulation(cfg2, graph=graph2, index=index2, embedder=emb2,
                             llm=llm2, model="m", out_dir=tmp_path)
    events = load_events(tmp_path / "rr" / "events.jsonl")
    assert [e["step"] for e in events] == [0, 1, 2, 3, 4, 5]
    assert out2["topic_assignments"]["a1"] == assigned  # reused, not re-sampled
    # unscoped a0 generates at steps 2 and 4; counter starts at gen:rr:0
    gen_ids = [e["result"]["node_id"] for e in events
               if e["action"] == "generate" and "node_id" in e["result"]]
    assert gen_ids[:2] == ["gen:rr:0", "gen:rr:1"]
    # memory rebuilt: a0's first resumed prompt references its past search
    assert "search ->" in llm2.calls[0]["user"]
