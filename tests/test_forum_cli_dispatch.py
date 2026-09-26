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
    assert len(files) == 7 and n_topics == 128

    for f in files:
        cfg = load_config(f)
        run = cfg["run"]
        assert cfg["arch"] == "p2_forum"
        assert cfg["out_dir"] == "runs/p2_forum"
        assert Path(cfg["topics_file"]).exists()
        assert run["run_id"] == f.stem.replace("k", "forum-k")
        assert run["total_steps"] == 800
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
    assert sorted(ks) == [1, 2, 4, 8, 16, 32, 64]


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
