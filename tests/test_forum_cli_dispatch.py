"""cli.py's `arch` dispatch and `_load_forum_world` (spec §7).

`arch` has exactly two values: p2_forum, and p1_dial (the default when the key
is absent). Everything here runs offline against a synthetic corpus.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

import innovation.cli as cli
from innovation.core.config import load_config
from innovation.core.events import load_events
from innovation.core.ideas.embed import FakeEmbedder, save_embeddings
from innovation.core.ideas.summarize import save_ideas
from innovation.core.data.corpus import save_corpus
from innovation.core.llm import FakeLLM

POST = json.dumps({"action": "generate",
                   "args": {"text": "an idea", "cited_ids": ["W0"]}})
JUMP = json.dumps({"action": "sample_frontier", "args": {}})


@pytest.fixture
def forum_cfg(tmp_path, monkeypatch):
    """A minimal arch=p2_forum config over a 4-paper synthetic corpus."""
    papers = pd.DataFrame([
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"A{i}",
         "year": 2020, "venue": "V"} for i in range(4)])
    edges = pd.DataFrame([{"src": "W1", "dst": "W0"}, {"src": "W2", "dst": "W1"}])
    ideas = papers.rename(columns={"abstract": "idea_text"})[
        ["paper_id", "idea_text", "year", "venue"]]
    data_dir = tmp_path / "data"
    save_corpus(papers, edges, data_dir)
    save_ideas(ideas, data_dir)
    emb = FakeEmbedder()
    save_embeddings(list(ideas["paper_id"]),
                    emb.encode(list(ideas["idea_text"])), data_dir)
    topics = tmp_path / "topics.yaml"
    topics.write_text("topics:\n- {topic: alpha}\n- {topic: beta}\n")

    monkeypatch.setattr(cli, "Embedder", lambda name: FakeEmbedder())
    return {"data_dir": str(data_dir), "out_dir": str(tmp_path / "runs"),
            "embedding_model": "fake", "arch": "p2_forum",
            "topics_file": str(topics),
            "models": {"agent": "m"},
            "run": {"run_id": "f1", "seed": 0, "total_steps": 4,
                    "agents": [{"agent_id": "a0", "k_topics": 1},
                               {"agent_id": "a1", "k_topics": 1}]}}


def _use_llm(monkeypatch, llm):
    monkeypatch.setattr(cli, "_llm", lambda c: llm)


def test_arch_p2_forum_dispatches_to_the_forum_runner(forum_cfg, monkeypatch):
    _use_llm(monkeypatch, FakeLLM(default=POST))

    cli.cmd_run(forum_cfg)

    run_dir = Path(forum_cfg["out_dir"]) / "f1"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["arch"] == "p2_forum"
    assert set(meta["topic_assignments"]) == {"a0", "a1"}
    events = load_events(run_dir / "events.jsonl")
    # board ids, not paper 1's shared-graph ids: the forum runner ran
    assert [e["result"]["node_id"] for e in events] == [f"gen:f1:{i}"
                                                        for i in range(4)]


def test_a_missing_arch_key_still_means_paper_one(forum_cfg, monkeypatch):
    seen = {}

    def spy(cfg, **kw):
        seen["run_id"] = cfg.run_id
        return {}

    monkeypatch.setattr(cli, "run_simulation", spy)
    _use_llm(monkeypatch, FakeLLM(default=JUMP))
    del forum_cfg["arch"]
    forum_cfg["run"]["agents"] = [{"agent_id": "a0", "policy": "pa", "m": 1}]

    cli.cmd_run(forum_cfg)

    assert seen["run_id"] == "f1"


def test_cmd_run_refuses_to_overwrite_an_existing_forum_log(forum_cfg,
                                                           monkeypatch):
    _use_llm(monkeypatch, FakeLLM(default=POST))
    cli.cmd_run(forum_cfg)

    with pytest.raises(SystemExit, match="already has events"):
        cli.cmd_run(forum_cfg)


def test_cmd_run_resume_extends_a_forum_log_without_repeating_ids(forum_cfg,
                                                                 monkeypatch):
    """C1: --resume used to be ignored for p2_forum, which appended a second
    gen:<run_id>:0 and left the log unreplayable."""
    _use_llm(monkeypatch, FakeLLM(default=POST))
    cli.cmd_run(forum_cfg)

    forum_cfg["run"]["total_steps"] = 7
    cli.cmd_run(forum_cfg, resume=True)

    run_dir = Path(forum_cfg["out_dir"]) / "f1"
    events = load_events(run_dir / "events.jsonl")
    ids = [e["result"]["node_id"] for e in events]
    assert [e["step"] for e in events] == list(range(7))
    assert ids == [f"gen:f1:{i}" for i in range(7)]
    assert len(set(ids)) == len(ids)
    assert json.loads((run_dir / "run_meta.json").read_text())["resumed_from"] == [4]


def test_load_forum_world_freezes_the_corpus(forum_cfg):
    corpus, index, _ = cli._load_forum_world(forum_cfg)

    assert corpus.frozen
    assert corpus.num_nodes == 4 and corpus.num_edges == 2
    assert sorted(index.ids) == ["W0", "W1", "W2", "W3"]


def test_load_forum_world_honours_the_no_edges_ablation(forum_cfg):
    forum_cfg["run"]["init_edges"] = "none"

    corpus, _, _ = cli._load_forum_world(forum_cfg)

    assert corpus.num_nodes == 4 and corpus.num_edges == 0


def test_all_p2_forum_experiment_configs_load():
    files = sorted(Path("configs/p2_forum/experiments").glob("*.yaml"))
    pool = load_config("configs/p2_forum/base.yaml")["topics_file"]
    n_topics = len(cli._topic_pool({"topics_file": pool}))
    assert len(files) == 12 and n_topics == 128

    for f in files:
        cfg = load_config(f)
        run = cfg["run"]
        assert cfg["arch"] == "p2_forum"
        assert cfg["out_dir"] == "runs/p2_forum"
        assert Path(cfg["topics_file"]).exists()
        assert run["run_id"] == f.stem.replace("k", "forum-k")
        assert run["total_steps"] == 1000
        assert len(run["agents"]) == 10
        ks = {a["k_topics"] for a in run["agents"]}
        assert len(ks) == 1 and ks.pop() <= n_topics
        # the headline metric must stay comparable with paper 1
        assert cfg["eval"]["realized_min_date"] == "2025-06-01"
        assert cfg["data_dir"] == "data/stage1"


def test_the_k_sweep_covers_the_spec_grid():
    ks = []
    for f in sorted(Path("configs/p2_forum/experiments").glob("*.yaml")):
        run = load_config(f)["run"]
        ks.append(run["agents"][0]["k_topics"])
    assert sorted(ks) == [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]


def test_a_completed_forum_run_writes_structural_metrics(forum_cfg, monkeypatch):
    """Spec §8: a run produces headline AND structural metrics. The headline
    half needs the judge; the structural half is a replay and lands here."""
    _use_llm(monkeypatch, FakeLLM(default=POST))

    cli.cmd_run(forum_cfg)

    path = Path(forum_cfg["out_dir"]) / "f1" / "board_metrics.json"
    doc = json.loads(path.read_text())
    assert doc["run_id"] == "f1" and doc["n_agents"] == 2
    # 4 events, 2 agents -> 2 rounds
    assert doc["n_rounds"] == 2 and [r["round"] for r in doc["rounds"]] == [1, 2]
    assert [r["n_events"] for r in doc["rounds"]] == [2, 4]
    assert [r["n_posts"] for r in doc["rounds"]] == [2, 4]
    assert doc["final"]["n_posts"] == 4
    # every §5 quantity is present
    for key in ("post_post_share", "n_components", "longest_chain",
                "post_in_degree_distribution",
                "first_cross_agent_citation_step"):
        assert key in doc["final"]


def test_board_metrics_can_be_rederived_without_rerunning(forum_cfg, monkeypatch):
    _use_llm(monkeypatch, FakeLLM(default=POST))
    cli.cmd_run(forum_cfg)
    path = Path(forum_cfg["out_dir"]) / "f1" / "board_metrics.json"
    first = path.read_text()
    path.unlink()

    cli.cmd_board_metrics(forum_cfg)

    assert path.read_text() == first


def test_board_metrics_refuses_a_paper_one_run(forum_cfg):
    forum_cfg["arch"] = "p1_dial"
    with pytest.raises(SystemExit, match="p2_forum"):
        cli.cmd_board_metrics(forum_cfg)


def test_resume_refreshes_the_structural_metrics(forum_cfg, monkeypatch):
    _use_llm(monkeypatch, FakeLLM(default=POST))
    cli.cmd_run(forum_cfg)
    forum_cfg["run"]["total_steps"] = 8

    cli.cmd_run(forum_cfg, resume=True)

    doc = json.loads((Path(forum_cfg["out_dir"]) / "f1"
                      / "board_metrics.json").read_text())
    assert doc["n_rounds"] == 4 and doc["final"]["n_posts"] == 8


def test_the_navigation_section_reaches_the_environment(forum_cfg, monkeypatch):
    """Spec §4.1: the ablations are per store, read from the top level."""
    _use_llm(monkeypatch, FakeLLM(default=JUMP))
    forum_cfg["navigation"] = {"corpus": {"jump": False}}

    cli.cmd_run(forum_cfg)

    run_dir = Path(forum_cfg["out_dir"]) / "f1"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["navigation"] == {"corpus_search": True, "corpus_edges": True,
                                  "corpus_jump": False, "board_search": True,
                                  "board_edges": True, "board_jump": True}
    events = load_events(run_dir / "events.jsonl")
    assert all(e["result"]["error"].startswith("random jumps into the literature")
               for e in events)


def test_a_typo_in_the_navigation_section_fails_the_run(forum_cfg, monkeypatch):
    _use_llm(monkeypatch, FakeLLM(default=JUMP))
    forum_cfg["navigation"] = {"corpus": {"jmup": False}}

    with pytest.raises(ValueError, match="unknown navigation.corpus channel"):
        cli.cmd_run(forum_cfg)


def test_the_nested_sweep_configs_load_and_nest_on_the_real_pool():
    import numpy as np

    from innovation.p2_forum.runner import draw_topics

    files = sorted(Path("configs/p2_forum/experiments/nested").glob("*.yaml"))
    assert len(files) == 27
    runs = [load_config(f)["run"] for f in files]
    cfg = load_config(files[0])
    assert cfg["arch"] == "p2_forum"                     # base.yaml reached
    assert cfg["eval"]["realized_min_date"] == "2025-06-01"  # paper 1 reached
    assert all(r["topic_draw"] == "nested" and r["total_steps"] == 400 for r in runs)
    assert len({r["run_id"] for r in runs}) == 27
    grid = sorted({(r["agents"][0]["k_topics"], r["seed"]) for r in runs})
    assert grid == [(k, s) for k in [1, 16, 32, 48, 64, 80, 96, 112, 128] for s in (0, 1, 2)]

    pool = cli._topic_pool(cfg)
    by = {(r["agents"][0]["k_topics"], r["seed"]): r for r in runs}
    for s in (0, 1, 2):
        prev = None
        for k in [1, 16, 32, 48, 64, 80, 96, 112, 128]:
            r = by[(k, s)]
            t = draw_topics(r["agents"], pool, np.random.default_rng(s),
                            scheme="nested", seed=s)
            if prev is not None:
                assert all(set(prev[a]) < set(t[a]) for a in t)
            prev = t


def test_online_configs_load_and_are_nested():
    from innovation.core.config import load_config
    files = sorted(Path("configs/p2_forum/experiments/online").glob("*.yaml"))
    assert len(files) == 9
    cfg = load_config(files[0])
    assert cfg["literature"] == "online" and cfg["gating"] == "topics"
    assert cfg["run"]["topic_draw"] == "nested" and cfg["run"]["total_steps"] == 400
    assert cfg["models"]["tagger"] == "claude-sonnet-5:medium"
    assert cfg["models"]["judge"] == "claude-opus-5-5"
    assert cfg["online"]["max_pub_date"] == "2024-09-30"
    assert cfg["models"]["agent"] == "openai:gpt-5:medium"
    runs = {load_config(f)["run"]["run_id"]: load_config(f)["run"] for f in files}
    assert set(runs) == {f"forum-online-k{k}-s0"
                         for k in [1, 16, 32, 48, 64, 80, 96, 112, 128]}
    assert all(r["seed"] == 0 and len(r["agents"]) == 10 for r in runs.values())


def test_online_mode_dispatches_with_the_online_world(forum_cfg, monkeypatch):
    """cmd_run in online mode builds the gated run from _load_online_world and
    replays the log into board_metrics.json without any corpus."""
    from conftest import FakeEmbedder as Emb4
    from test_forum_gated_env import FakeLit, FakeTagger, NAMES
    from innovation.p2_forum.topics import Topic

    topics = [Topic(id=i, name=n, definition=f"def {n}") for i, n in enumerate(NAMES)]
    monkeypatch.setattr(cli, "_load_online_world",
                        lambda c: (FakeLit(), FakeTagger(), Emb4(), topics))
    monkeypatch.setattr(cli, "_load_forum_world",
                        lambda c: pytest.fail("online mode must not load the corpus"))
    post = json.dumps({"action": "generate",
                       "args": {"text": "T0 T1 T2 T3 idea", "cited_ids": ["p0", "p3"]}})
    llm = FakeLLM(default=post)
    _use_llm(monkeypatch, llm)
    forum_cfg.update(literature="online", gating="topics")
    forum_cfg["models"]["agent"] = "openai:gpt-5:medium"
    forum_cfg["run"]["topic_draw"] = "nested"

    cli.cmd_run(forum_cfg)

    run_dir = Path(forum_cfg["out_dir"]) / "f1"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["literature"] == "online"
    assert {c["model"] for c in llm.calls} == {"openai:gpt-5:medium"}
    events = load_events(run_dir / "events.jsonl")
    posted = [e for e in events if "node_id" in e["result"]]
    assert len(posted) == 4
    final = json.loads((run_dir / "board_metrics.json").read_text())["final"]
    assert final["n_posts"] == 4
    kept = sum(len(e["args"]["cited_ids"]) - len(e["result"].get("dropped_cites", []))
               for e in posted)
    assert final["n_post_corpus_edges"] == kept


def _online_cfg(**online_extra):
    from innovation.core.config import load_config
    cfg = load_config("configs/p2_forum/base-online.yaml")
    cfg["online"].update(online_extra)
    return cfg


def test_online_scope_uses_the_tier1_rule(monkeypatch, tmp_path):
    """Reading scope and cmd_evaluate share one helper: same aliases, same floor."""
    from innovation import cli
    cfg = _online_cfg(cache_dir=str(tmp_path))
    monkeypatch.setattr(cli, "Embedder", lambda name: object())
    monkeypatch.setattr(cli, "RoutedLLM", lambda: object())
    monkeypatch.setattr(cli, "CachedLLM", lambda llm, d: object())
    import innovation.p2_forum.topics as topics_mod
    monkeypatch.setattr(topics_mod, "load_topics", lambda path: [])
    lit, _, _, _ = cli._load_online_world(cfg)
    aliases = cli._tier1_aliases(cfg)
    assert len(aliases) > 50 and "neurips" in aliases and "cvpr" in aliases
    assert lit.scope.venue_aliases == aliases
    assert lit.scope.min_citations == cfg["eval"]["recognized_min_citations"] == 50
    assert lit.scope.max_date == "2024-09-30"
    assert cfg["eval"]["exclude_workshops"] is True
    assert "venues" not in cfg["online"]
    assert "min_citations_any_venue" not in cfg["online"]


@pytest.mark.parametrize("key,val", [("venues", [{"name": "X", "aliases": ["x"]}]),
                                     ("min_citations_any_venue", 10)])
def test_online_stale_scope_keys_are_refused(key, val):
    from innovation import cli
    with pytest.raises(SystemExit, match=f"online.{key}"):
        cli._load_online_world(_online_cfg(**{key: val}))


def _region(forum_cfg, coverage):
    forum_cfg.update(gating="region", literature="corpus")
    del forum_cfg["topics_file"]          # region mode draws no topics
    for a in forum_cfg["run"]["agents"]:
        del a["k_topics"]
        a["coverage"] = coverage
    return forum_cfg


def test_region_mode_dispatches_with_the_corpus_world(forum_cfg, monkeypatch):
    """cmd_run in region mode builds the balls from the frozen corpus and its
    embeddings, runs, and replays the log into board_metrics.json."""
    cfg = _region(forum_cfg, 0.5)
    llm = FakeLLM(default=POST)
    _use_llm(monkeypatch, llm)

    cli.cmd_run(cfg)

    run_dir = Path(cfg["out_dir"]) / "f1"
    meta = json.loads((run_dir / "run_meta.json").read_text())
    assert meta["gating"] == "region" and meta["corpus_size"] == 4
    assert {r["n_members"] for r in meta["regions"].values()} == {2}
    assert {r["achieved_coverage"] for r in meta["regions"].values()} == {0.5}
    assert "within your own research area" in llm.calls[0]["system"]
    events = load_events(run_dir / "events.jsonl")
    posted = [e for e in events if "node_id" in e["result"]]
    final = json.loads((run_dir / "board_metrics.json").read_text())["final"]
    assert final["n_posts"] == len(posted)


def test_region_runs_evaluate_on_the_corpus_path(forum_cfg, monkeypatch):
    """cmd_evaluate for a region run: contamination guard = corpus titles,
    duplicates checked against the corpus embeddings, models.judge,
    eval.exclude_workshops."""
    from innovation.core.eval.search_verify import Verdict
    cfg = _region(forum_cfg, 1.0)             # full coverage: every post is published
    _use_llm(monkeypatch, FakeLLM(default=POST))
    cli.cmd_run(cfg)
    cfg.update(cutoff_date="2024-09-30", mailto="x@y",
               recognized_venues=[{"name": "ICML", "aliases": ["icml"]}])
    cfg["models"]["judge"] = "claude-opus-5-5"
    cfg["eval"] = {"n_queries": 1, "top_k": 1, "dup_ceiling": 0.95,
                   "exclude_workshops": True}
    seen = []

    def fake_verify(llm, **kw):
        seen.append(kw)
        return Verdict(idea_id=kw["idea_id"])

    monkeypatch.setattr(cli, "verify_idea", fake_verify)
    monkeypatch.setattr(cli, "_eval_reference",
                        lambda c: pytest.fail("region runs use the corpus path"))
    cli.cmd_evaluate(cfg)

    assert len(seen) == 4
    assert all(kw["model"] == "claude-opus-5-5" and kw["exclude_workshops"] is True
               and kw["corpus_titles"] == {"t0", "t1", "t2", "t3"} for kw in seen)
    verdicts = json.loads((Path(cfg["out_dir"]) / "f1" / "verdicts.json").read_text())
    assert len(verdicts) == 4 and all("dup_flag" in v for v in verdicts)


def test_region_configs_load():
    from innovation.core.config import load_config
    files = sorted(Path("configs/p2_forum/experiments/region").glob("*.yaml"))
    pcts = [1, 5, 10, 20, 30, 40, 50, 100]
    assert sorted(f.stem for f in files) == sorted(f"c{p}-s0" for p in pcts)
    for p in pcts:
        cfg = load_config(f"configs/p2_forum/experiments/region/c{p}-s0.yaml")
        r = cfg["run"]
        assert r["run_id"] == f"forum-region-c{p}-s0" and r["seed"] == 0
        assert r["total_steps"] == 400
        assert [a["agent_id"] for a in r["agents"]] == [f"a{i}" for i in range(10)]
        assert all(a["coverage"] == p / 100 and set(a) == {"agent_id", "coverage"}
                   for a in r["agents"])
        assert cfg["arch"] == "p2_forum" and cfg["data_dir"] == "data/p2_corpus"
        assert cfg["literature"] == "corpus" and cfg["gating"] == "region"
        assert cfg["models"]["agent"] == "openai:gpt-5:medium"
        assert cfg["models"]["judge"] == "claude-opus-5-5"
        assert cfg["eval"]["exclude_workshops"] is True
        assert cfg["eval"]["recognized_min_citations"] == 50   # inherited
        assert cfg["eval"]["realized_min_date"] == "2025-06-01"
        assert len(cfg["recognized_venues"]) > 50
        assert cfg["embedding_model"] == "BAAI/bge-small-en-v1.5"
        assert cfg["corpus"]["idea_text"] == "abstract"
        assert cfg["out_dir"] == "runs/p2_forum"
        # Building the run config validates the mode and every coverage.
        from innovation.p2_forum.runner import ForumRunConfig
        ForumRunConfig(run_id=r["run_id"], seed=r["seed"], total_steps=r["total_steps"],
                       agents=r["agents"], literature=cfg["literature"],
                       gating=cfg["gating"])
