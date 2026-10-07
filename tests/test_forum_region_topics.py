"""Each region's topic list (R5, 2026-10-07): k-means clusters of its papers,
named by one LLM call each, recorded in run_meta and listed in the prompt."""
import json

import numpy as np
import pytest

from innovation.core.llm import CachedLLM, FakeLLM
from innovation.p2_forum.agent import (REGION_SYSTEM, REGION_SYSTEM_NO_TOPICS,
                                       ForumAgentPolicy)
from innovation.p2_forum.region import build_region
from innovation.p2_forum.region_topics import (cluster_region, n_topics, name_topic,
                                               region_topics)
from innovation.p2_forum.runner import resume_forum, run_forum
from test_forum_region_runner import SEARCH, cfg, world
from test_forum_region_env import AngleEmbedder

NAMED = json.dumps({"name": "Sparse attention for long documents",
                    "description": "Making transformer attention cheaper on long inputs."})


class AgentAndNamer(FakeLLM):
    """Agent turns get the agent fake; model "namer" gets NAMED. Counts namer calls."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.namer_calls = []

    def complete(self, *, model, system, user, max_tokens=1024):
        if model == "namer":
            self.namer_calls.append(user)
            return NAMED
        return super().complete(model=model, system=system, user=user,
                                max_tokens=max_tokens)


@pytest.mark.parametrize("n,k", [(185, 3), (1854, 12), (3708, 25), (18542, 30),
                                 (1, 3), (300, 3), (4500, 30), (10 ** 6, 30)])
def test_n_topics_rule(n, k):
    assert n_topics(n) == k


def groups():
    """Three tight groups in 4-d: 5 papers near e0, 3 near e1, 2 near e2."""
    rng = np.random.default_rng(1)
    ids, vecs = [], []
    for g, size in enumerate([5, 3, 2]):
        for j in range(size):
            v = np.eye(4)[g] + 0.05 * rng.standard_normal(4)
            ids.append(f"g{g}p{j}")
            vecs.append(v)
    return ids, np.array(vecs, dtype=np.float32)


def test_clusters_are_sorted_by_size_deterministic_and_reps_are_central():
    ids, vecs = groups()
    region = build_region("g0p0", ids, vecs, 1.0)
    a = cluster_region(region, ids, vecs, seed=0)
    b = cluster_region(region, ids, vecs, seed=0)
    assert a == b
    assert [len(c.members) for c in a] == [5, 3, 2]
    assert [{m[:2] for m in c.members} for c in a] == [{"g0"}, {"g1"}, {"g2"}]
    big = a[0]
    unit = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    rows = [ids.index(m) for m in big.members]
    centroid = unit[rows].mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    expect = sorted(big.members, key=lambda m: (-float(unit[ids.index(m)] @ centroid), m))
    assert list(big.representatives) == expect             # all 5 (< 8), closest first


def test_representatives_are_capped_at_eight_and_a_tiny_region_has_few_clusters():
    rng = np.random.default_rng(2)
    ids = [f"p{i:02d}" for i in range(40)]
    vecs = rng.standard_normal((40, 4)).astype(np.float32)
    region = build_region("p00", ids, vecs, 1.0)
    cl = cluster_region(region, ids, vecs, seed=0)
    assert len(cl) == 3 and sum(len(c.members) for c in cl) == 40
    assert all(len(c.representatives) == min(8, len(c.members)) for c in cl)
    tiny = build_region("p00", ids, vecs, 2 / 40)             # 2 members
    assert len(cluster_region(tiny, ids, vecs, seed=0)) <= 2


def test_name_topic_parses_leniently():
    llm = FakeLLM(responses=["Here you go: " + NAMED + " hope it helps"])
    reps = [("Title A", "x" * 500), ("Title B", "abstract b")]
    assert name_topic(llm, "namer", reps) == json.loads(NAMED)
    user = llm.calls[0]["user"]
    assert "Title A" in user and "x" * 300 in user and "x" * 301 not in user
    assert '"name"' in user and "venue" in user


def test_name_topic_retries_once_with_a_different_prompt_then_raises():
    llm = FakeLLM(responses=["not json", NAMED])
    assert name_topic(llm, "namer", [("T", "A")])["name"] == "Sparse attention for long documents"
    assert len(llm.calls) == 2 and llm.calls[0]["user"] != llm.calls[1]["user"]
    bad = FakeLLM(responses=["nope", '{"name": ""}'], default=NAMED)
    with pytest.raises(ValueError):
        name_topic(bad, "namer", [("T", "A")])
    assert len(bad.calls) == 2


def test_region_topics_names_each_cluster():
    ids, vecs = groups()
    region = build_region("g0p0", ids, vecs, 1.0)
    llm = AgentAndNamer()
    out = region_topics(llm, "namer", region, ids, vecs, 0,
                        lambda pid: (f"Title {pid}", f"Abstract {pid}"))
    assert [t["n_papers"] for t in out] == [5, 3, 2]
    assert all(t["name"] == "Sparse attention for long documents" for t in out)
    assert len(llm.namer_calls) == 3 and "Title g0p" in llm.namer_calls[0]


def run(tmp_path, c, llm, fn=run_forum, **kw):
    corpus, index = world()
    return fn(c, corpus=corpus, corpus_index=index, embedder=AngleEmbedder(),
              llm=llm, model="m", out_dir=tmp_path, **kw)


def test_run_meta_records_topics_and_the_prompt_lists_them(tmp_path):
    llm = AgentAndNamer(default=SEARCH)
    run(tmp_path, cfg(), llm, topic_namer="namer")
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    for rec in meta["regions"].values():                   # 4 members -> 3 topics
        assert [t["name"] for t in rec["topics"]] == ["Sparse attention for long documents"] * 3
        assert sum(t["n_papers"] for t in rec["topics"]) == 4
    system = llm.calls[0]["system"]
    assert ("- Sparse attention for long documents — Making transformer attention "
            "cheaper on long inputs.") in system
    assert "Plan your work inside these topics" in system


def test_resume_reuses_recorded_topics_without_calling_the_namer(tmp_path):
    run(tmp_path, cfg(4), AgentAndNamer(default=SEARCH), topic_namer="namer")
    meta_path = tmp_path / "r" / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["regions"]["a"]["topics"] = [{"name": "Recorded topic", "description": "D.",
                                       "n_papers": 4}]
    meta_path.write_text(json.dumps(meta))
    llm = AgentAndNamer(default=SEARCH)
    run(tmp_path, cfg(8), llm, fn=resume_forum, topic_namer="namer")
    assert llm.namer_calls == []
    a_prompts = [c["system"] for c in llm.calls if "[agent r:a]" in c["user"]]
    assert a_prompts and all("- Recorded topic — D." in s for s in a_prompts)
    after = json.loads(meta_path.read_text())
    assert after["regions"]["a"]["topics"] == meta["regions"]["a"]["topics"]


def test_old_run_meta_without_topics_resumes_with_the_fallback_prompt(tmp_path):
    run(tmp_path, cfg(4), AgentAndNamer(default=SEARCH), topic_namer="namer")
    meta_path = tmp_path / "r" / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    for rec in meta["regions"].values():
        del rec["topics"]
    meta_path.write_text(json.dumps(meta))
    llm = AgentAndNamer(default=SEARCH)
    out = run(tmp_path, cfg(8), llm, fn=resume_forum)
    assert out["resumed_from_step"] == 4 and llm.namer_calls == []
    assert all(c["system"] == REGION_SYSTEM_NO_TOPICS for c in llm.calls)


def test_identical_regions_share_cached_topic_names(tmp_path):
    inner = AgentAndNamer(default=SEARCH)
    llm = CachedLLM(inner, tmp_path / "cache")
    run(tmp_path, cfg(2, coverage=1.0), llm, topic_namer="namer")
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["regions"]["a"]["topics"] == meta["regions"]["b"]["topics"]
    assert len(inner.namer_calls) == len(meta["regions"]["a"]["topics"])   # b hit the cache


def test_prompt_lists_topics_and_the_fallback_has_no_empty_bullets():
    with_topics = ForumAgentPolicy(llm=FakeLLM(), model="m",
                                   topics=["Topic one — First.", "Topic two — Second."],
                                   system_template=REGION_SYSTEM)
    assert "- Topic one — First.\n- Topic two — Second.\n" in with_topics.system
    assert "{topics}" not in with_topics.system
    bare = ForumAgentPolicy(llm=FakeLLM(), model="m", topics=[],
                            system_template=REGION_SYSTEM_NO_TOPICS)
    assert "\n- " not in bare.system and "following topics" not in bare.system
    assert "within your own research area" in bare.system
