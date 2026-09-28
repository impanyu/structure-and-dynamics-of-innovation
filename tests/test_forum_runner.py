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
