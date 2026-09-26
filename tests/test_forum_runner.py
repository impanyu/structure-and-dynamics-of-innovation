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
