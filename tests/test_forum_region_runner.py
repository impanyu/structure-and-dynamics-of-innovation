"""The region-gated mode through the runner (spec, REVISION 2026-10-06):
run_meta records each agent's region, resume rebuilds the regions from it."""
import json

import numpy as np
import pytest

from innovation.core.events import load_events
from innovation.core.llm import FakeLLM
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum import runner
from innovation.p2_forum.region import build_region, draw_seeds
from innovation.p2_forum.runner import ForumRunConfig, resume_forum, run_forum
from test_forum_region_env import (ANGLES, CITES, INSERT_ORDER, VENUES,
                                   AngleEmbedder, unit)

SEARCH = json.dumps({"action": "search", "args": {"query": "near @95", "k": 8}})


def world():
    corpus = IdeaGraph()
    for pid in INSERT_ORDER:
        corpus.add_idea(pid, f"Title {pid}\n\nAbstract of {pid}.", CITES.get(pid, []),
                        source="corpus", year=2020, meta={"venue": VENUES[pid]})
    corpus.freeze()
    ids = list(ANGLES)
    index = VectorIndex(4)
    index.add(ids, np.stack([unit(ANGLES[i]) for i in ids]))
    return corpus, index


def cfg(total_steps=6, coverage=0.5, **kw):
    return ForumRunConfig(run_id="r", seed=0, total_steps=total_steps,
                          literature="corpus", gating="region",
                          agents=[{"agent_id": "a", "coverage": coverage},
                                  {"agent_id": "b", "coverage": coverage}], **kw)


def run(tmp_path, c, llm, fn=run_forum):
    corpus, index = world()
    return fn(c, corpus=corpus, corpus_index=index, embedder=AngleEmbedder(),
              llm=llm, model="m", out_dir=tmp_path)


def test_region_run_end_to_end_records_each_agents_region(tmp_path):
    llm = FakeLLM(default=SEARCH)
    run(tmp_path, cfg(), llm)
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["literature"] == "corpus" and meta["gating"] == "region"
    assert meta["corpus_size"] == 8
    ids = list(ANGLES)
    vecs = np.stack([unit(ANGLES[i]) for i in ids])
    seeds = draw_seeds(2, ids, 0)
    assert set(meta["regions"]) == {"a", "b"}
    for aid, seed in zip(["a", "b"], seeds):
        rec = meta["regions"][aid]
        region = build_region(seed, ids, vecs, 0.5)
        assert rec == {"seed_id": seed, "seed_title": f"Title {seed}",
                       "coverage": 0.5, "achieved_coverage": 0.5,
                       "radius": region.radius, "n_members": 4,
                       "topics": []}                  # no topic namer given
    assert meta["mean_achieved_coverage"] == 0.5
    assert "topic_assignments" not in meta

    events = load_events(tmp_path / "r" / "events.jsonl")
    assert len(events) == 6
    hits = 0
    for e in events:
        seed = meta["regions"][e["agent_id"]]["seed_id"]
        members = build_region(seed, ids, vecs, 0.5).members
        for h in e["result"]["hits"]:
            assert h["node_id"] in members
            hits += 1
    assert hits == 6 * 4                      # every search returns the whole region
    assert "within your own research area" in llm.calls[0]["system"]
    assert '"action": "related"' in llm.calls[0]["user"]
    assert "Latest result (full):\n" in llm.calls[1]["user"]


def test_region_resume_rebuilds_regions_from_run_meta_without_redrawing(tmp_path, monkeypatch):
    run(tmp_path, cfg(4), FakeLLM(default=SEARCH))
    meta_path = tmp_path / "r" / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    # Rewrite a's recorded seed: a resume that re-drew would ignore this.
    meta["regions"]["a"]["seed_id"] = "c7"
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(runner, "draw_seeds",
                        lambda *a, **k: pytest.fail("resume must not re-draw seeds"))

    out = run(tmp_path, cfg(8), FakeLLM(default=SEARCH), fn=resume_forum)

    assert out["resumed_from_step"] == 4
    after = json.loads(meta_path.read_text())
    assert after["regions"] == meta["regions"] and after["resumed_from"] == [4]
    ids = list(ANGLES)
    vecs = np.stack([unit(ANGLES[i]) for i in ids])
    a_members = build_region("c7", ids, vecs, 0.5).members
    resumed = [e for e in load_events(tmp_path / "r" / "events.jsonl")
               if e["step"] >= 4 and e["agent_id"] == "a"]
    assert resumed and all({h["node_id"] for h in e["result"]["hits"]} == a_members
                           for e in resumed)


def test_region_resume_refuses_a_changed_coverage_or_mode(tmp_path):
    run(tmp_path, cfg(4), FakeLLM(default=SEARCH))
    with pytest.raises(SystemExit, match="coverage"):
        run(tmp_path, cfg(8, coverage=0.25), FakeLLM(default=SEARCH), fn=resume_forum)
    plain = ForumRunConfig(run_id="r", seed=0, total_steps=8,
                           agents=[{"agent_id": "a", "k_topics": 1},
                                   {"agent_id": "b", "k_topics": 1}],
                           topic_pool=["t0", "t1"])
    with pytest.raises(SystemExit, match="gating"):
        run(tmp_path, plain, FakeLLM(default=SEARCH), fn=resume_forum)


def test_region_resume_refuses_a_changed_corpus(tmp_path):
    run(tmp_path, cfg(4), FakeLLM(default=SEARCH))
    meta_path = tmp_path / "r" / "run_meta.json"
    meta = json.loads(meta_path.read_text())
    meta["regions"]["a"]["n_members"] = 5
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="members"):
        run(tmp_path, cfg(8), FakeLLM(default=SEARCH), fn=resume_forum)


@pytest.mark.parametrize("kw,agents", [
    (dict(literature="online", gating="region"), [{"agent_id": "a", "coverage": 0.1}]),
    (dict(gating="region"), [{"agent_id": "a"}]),
    (dict(gating="region"), [{"agent_id": "a", "coverage": 0}]),
    (dict(gating="region"), [{"agent_id": "a", "coverage": 1.5}]),
    (dict(gating="region"), [{"agent_id": "a", "coverage": "10%"}]),
    (dict(), [{"agent_id": "a", "k_topics": 1, "coverage": 0.1}]),
])
def test_region_mode_combinations_are_validated(kw, agents):
    with pytest.raises(ValueError):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, agents=agents, **kw)


def test_region_mode_accepts_full_coverage():
    c = cfg(coverage=1.0)
    assert c.region and not c.online


def test_region_resume_rebuilds_the_same_compact_history(tmp_path):
    """The resumed prompts equal those of an uninterrupted run: resume builds
    each history entry with the policy's own (compact, 4000-char) form."""
    gen = lambda d: json.dumps({"action": "generate",
                                "args": {"text": f"idea @{d} " + "w" * 400, "cited_ids": []}})
    browse = json.dumps({"action": "browse", "args": {"node_id": "c1"}})
    script = [gen(5), gen(25), SEARCH, browse, SEARCH, SEARCH, browse, SEARCH]
    full = FakeLLM(list(script), default=SEARCH)
    run(tmp_path / "full", cfg(8), full)
    first = FakeLLM(script[:4], default=SEARCH)
    run(tmp_path / "split", cfg(4), first)
    second = FakeLLM(script[4:], default=SEARCH)
    run(tmp_path / "split", cfg(8), second, fn=resume_forum)
    assert [c["user"] for c in second.calls] == [c["user"] for c in full.calls[4:]]
    history = full.calls[4]["user"].split("Recent history (oldest first):\n")[1]
    history = history.split("\n\nLatest result (full):")[0]
    assert '"notice"' not in history and '"store":"corpus"' in history
    assert '"text"' not in history.split("search -> ")[1]          # snippets dropped
