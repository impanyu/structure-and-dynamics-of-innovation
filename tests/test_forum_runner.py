import json

import numpy as np

from innovation.p2_forum.runner import ForumRunConfig, draw_topics, run_forum


def test_draw_topics_gives_each_agent_k_distinct_topics():
    pool = [f"topic {i}" for i in range(20)]
    agents = [{"agent_id": "a0", "k_topics": 3},
              {"agent_id": "a1", "k_topics": 5}]

    out = draw_topics(agents, pool, np.random.default_rng(0))

    assert len(out["a0"]) == 3 and len(set(out["a0"])) == 3
    assert len(out["a1"]) == 5 and len(set(out["a1"])) == 5
    assert set(out["a0"]) <= set(pool)


def test_draw_topics_is_seeded_and_agents_may_overlap():
    pool = [f"topic {i}" for i in range(8)]
    agents = [{"agent_id": f"a{i}", "k_topics": 4} for i in range(5)]

    first = draw_topics([dict(a) for a in agents], pool, np.random.default_rng(3))
    again = draw_topics([dict(a) for a in agents], pool, np.random.default_rng(3))

    assert first == again
    # 5 agents x 4 topics from a pool of 8 cannot be disjoint
    assert len({t for ts in first.values() for t in ts}) <= 8


def test_draw_topics_rejects_k_larger_than_the_pool():
    import pytest
    with pytest.raises(ValueError, match="k_topics"):
        draw_topics([{"agent_id": "a", "k_topics": 9}],
                    [f"t{i}" for i in range(8)], np.random.default_rng(0))


class ScriptedLLM:
    def complete(self, *, model, system, user, max_tokens):
        return '{"action": "sample_frontier", "args": {}}'


def test_run_forum_writes_meta_before_driving_and_records_topics(
        tmp_path, make_workspace, fake_embedder):
    ws = make_workspace()
    cfg = ForumRunConfig(run_id="r", seed=0, total_steps=4,
                         agents=[{"agent_id": "a0", "k_topics": 1},
                                 {"agent_id": "a1", "k_topics": 1}],
                         topic_pool=["alpha", "beta", "gamma"])

    out = run_forum(cfg, corpus=ws.corpus, corpus_index=ws.corpus_index,
                    embedder=fake_embedder, llm=ScriptedLLM(), model="m",
                    out_dir=tmp_path)

    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["arch"] == "p2_forum"
    assert set(meta["topic_assignments"]) == {"a0", "a1"}
    assert out["steps"] == 4
    assert (tmp_path / "r" / "events.jsonl").exists()


def test_per_agent_navigation_flags_are_refused(make_workspace):
    """They used to be collapsed with all(...), so one agent opting out of
    search silently disabled search for the whole team (spec §4.3 makes these
    environmental). The config must not be able to express that."""
    import pytest

    with pytest.raises(ValueError, match="navigation ablations are environment-wide"):
        ForumRunConfig(run_id="r", seed=0, total_steps=2,
                       agents=[{"agent_id": "a0", "k_topics": 1},
                               {"agent_id": "a1", "k_topics": 1,
                                "allow_search": False}],
                       topic_pool=["alpha"])


def test_run_forum_applies_and_records_the_navigation_ablation(
        tmp_path, make_workspace, fake_embedder):
    from innovation.p2_forum.env import Navigation

    ws = make_workspace()
    cfg = ForumRunConfig(run_id="abl", seed=0, total_steps=2,
                         agents=[{"agent_id": "a0", "k_topics": 1}],
                         topic_pool=["alpha"],
                         navigation=Navigation.from_config(
                             {"board": {"jump": False}}))

    run_forum(cfg, corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=ScriptedLLM(), model="m",
              out_dir=tmp_path)

    meta = json.loads((tmp_path / "abl" / "run_meta.json").read_text())
    assert meta["navigation"]["board_jump"] is False
    assert meta["navigation"]["corpus_jump"] is True
    events = [json.loads(line) for line
              in (tmp_path / "abl" / "events.jsonl").read_text().splitlines()]
    # the scripted agent only jumps the corpus, which is still open
    assert all("error" not in e["result"] for e in events)


# --- nested topic draws -------------------------------------------------------

POOL = [f"topic {i}" for i in range(128)]
KS = [1, 16, 32, 48, 64, 80, 96, 112, 128]


def _nested(k, seed, n_agents=10):
    agents = [{"agent_id": f"a{i}", "k_topics": k} for i in range(n_agents)]
    return draw_topics(agents, POOL, np.random.default_rng(seed),
                       scheme="nested", seed=seed)


def test_nested_topics_are_strict_prefixes_across_k():
    """The point of the scheme: moving from k to a larger k only ADDS topics."""
    for seed in (0, 1, 2):
        draws = {k: _nested(k, seed) for k in KS}
        for small, big in zip(KS, KS[1:]):
            for a in draws[small]:
                assert draws[big][a][:small] == draws[small][a]


def test_nested_topics_at_the_full_pool_are_a_permutation_of_it():
    full = _nested(128, 0)
    assert all(sorted(t) == sorted(POOL) for t in full.values())


def test_nested_topics_depend_on_the_seed_and_differ_between_agents():
    s0, s1 = _nested(16, 0), _nested(16, 1)
    assert s0 != s1
    assert s0["a0"] != s0["a1"]


def test_nested_topics_do_not_consume_the_environment_rng():
    """The env's random jumps draw from the same rng; the nested scheme must
    leave it untouched so topic breadth does not perturb the jump stream."""
    rng = np.random.default_rng(7)
    before = rng.bit_generator.state
    draw_topics([{"agent_id": "a0", "k_topics": 64}], POOL, rng,
                scheme="nested", seed=7)
    assert rng.bit_generator.state == before


def test_nested_scheme_requires_the_seed():
    import pytest
    with pytest.raises(ValueError, match="seed"):
        draw_topics([{"agent_id": "a0", "k_topics": 2}], POOL,
                    np.random.default_rng(0), scheme="nested")


def test_unknown_topic_draw_is_refused():
    import pytest
    with pytest.raises(ValueError, match="topic_draw"):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, topic_draw="shuffled")


def test_run_forum_records_the_topic_draw_scheme(
        tmp_path, make_workspace, fake_embedder):
    ws = make_workspace()
    cfg = ForumRunConfig(run_id="n", seed=3, total_steps=2,
                         agents=[{"agent_id": "a0", "k_topics": 2}],
                         topic_pool=["alpha", "beta", "gamma"],
                         topic_draw="nested")

    run_forum(cfg, corpus=ws.corpus, corpus_index=ws.corpus_index,
              embedder=fake_embedder, llm=ScriptedLLM(), model="m",
              out_dir=tmp_path)

    meta = json.loads((tmp_path / "n" / "run_meta.json").read_text())
    assert meta["topic_draw"] == "nested"
    expected = draw_topics([{"agent_id": "a0", "k_topics": 2}],
                           ["alpha", "beta", "gamma"], np.random.default_rng(3),
                           scheme="nested", seed=3)
    assert meta["topic_assignments"] == expected


def test_display_order_is_a_seeded_shuffle_independent_of_nesting():
    from innovation.p2_forum.runner import display_order
    t = [f"T{i}" for i in range(16)]
    a = display_order(t, seed=0, agent_index=0)
    assert sorted(a) == sorted(t) and a != t
    assert a == display_order(t, seed=0, agent_index=0)
    assert a != display_order(t, seed=0, agent_index=1)


def test_mode_combinations_are_validated():
    import pytest
    from innovation.p2_forum.runner import ForumRunConfig
    with pytest.raises(ValueError):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, literature="online", gating="none")
    with pytest.raises(ValueError):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, literature="corpus", gating="topics")


def _online_cfg(total_steps=6):
    from test_forum_gated_env import NAMES
    return ForumRunConfig(run_id="r", seed=0, total_steps=total_steps,
                          literature="online", gating="topics", topic_draw="nested",
                          topic_pool=NAMES,
                          topic_definitions={n: f"definition of {n}" for n in NAMES},
                          agents=[{"agent_id": "a", "k_topics": 1},
                                  {"agent_id": "b", "k_topics": 1}])


def test_online_run_end_to_end_with_fakes(tmp_path):
    """Two agents, fake literature + tagger + LLM: the run writes run_meta with
    topic ids and display orders, and every logged search hit is readable."""
    from innovation.core.events import load_events
    from innovation.core.llm import FakeLLM
    from test_forum_gated_env import FakeLit, FakeTagger, NAMES
    from conftest import FakeEmbedder
    cfg = _online_cfg()
    llm = FakeLLM(default='{"action": "search", "args": {"query": "T0 T1 T2 T3"}}')
    run_forum(cfg, embedder=FakeEmbedder(), llm=llm, model="m", out_dir=tmp_path,
              literature=FakeLit(), tagger=FakeTagger())
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["literature"] == "online" and meta["gating"] == "topics"
    assert set(meta["topic_ids"]) == {"a", "b"}
    for aid, names in meta["topic_assignments"].items():
        assert meta["topic_ids"][aid] == sorted(NAMES.index(t) for t in names)
        assert sorted(meta["display_orders"][aid]) == sorted(names)

    events = load_events(tmp_path / "r" / "events.jsonl")
    assert len(events) == 6
    hits = 0
    for e in events:
        mine = {NAMES[i] for i in meta["topic_ids"][e["agent_id"]]}
        for h in e["result"].get("hits", []):
            assert set(h["topics"]) & mine
            hits += 1
    assert hits > 0
    # the gated prompt and the topic definitions reach the agent
    assert "You work ONLY within your topics" in llm.calls[0]["system"]
    assert "definition of" in llm.calls[0]["system"]


def test_online_resume_reuses_the_recorded_topic_ids(tmp_path):
    from innovation.core.events import load_events
    from innovation.core.llm import FakeLLM
    from innovation.p2_forum.runner import resume_forum
    from test_forum_gated_env import FakeLit, FakeTagger
    from conftest import FakeEmbedder
    post = json.dumps({"action": "generate",
                       "args": {"text": "T0 T1 T2 T3 idea", "cited_ids": ["p0", "p1"]}})
    kw = dict(embedder=FakeEmbedder(), model="m", out_dir=tmp_path,
              literature=FakeLit(), tagger=FakeTagger())
    run_forum(_online_cfg(4), llm=FakeLLM(default=post), **kw)
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    out = resume_forum(_online_cfg(8), llm=FakeLLM(default=post), **kw)
    assert out["resumed_from_step"] == 4
    events = load_events(tmp_path / "r" / "events.jsonl")
    ids = [e["result"]["node_id"] for e in events if "node_id" in e["result"]]
    assert len(ids) == 8 and len(set(ids)) == 8
    after = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert after["topic_ids"] == meta["topic_ids"]
    assert after["display_orders"] == meta["display_orders"]
