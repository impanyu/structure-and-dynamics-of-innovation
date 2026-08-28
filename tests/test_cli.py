"""Test CLI commands and config loader."""
import json
from pathlib import Path

import pandas as pd
import pytest

from innovation.config import load_config
from innovation.cli import cmd_run, cmd_evaluate
from innovation.data.corpus import save_corpus
from innovation.ideas.embed import FakeEmbedder, save_embeddings
from innovation.ideas.summarize import save_ideas
from innovation.llm import FakeLLM
from innovation.eval.search_verify import Verdict


def test_load_config_reads_stage1_yaml():
    """Load stage1.yaml and verify key fields."""
    cfg = load_config("configs/stage1.yaml")
    assert isinstance(cfg, dict)
    assert cfg["cutoff_date"] == "2025-01-01"
    assert cfg["models"]["summarizer"] == "claude-haiku-4-5-20251001"

    # Recognition list: evaluation-only concept, entries carry name + aliases
    recognized = cfg["recognized_venues"]
    assert isinstance(recognized, list) and len(recognized) >= 8
    for v in recognized:
        assert isinstance(v["name"], str)
        assert isinstance(v["aliases"], list) and v["aliases"]
    assert cfg["eval"]["recognized_min_citations"] == 10


def test_cmd_run_and_evaluate_wiring(tmp_path, monkeypatch):
    """Test cmd_run and cmd_evaluate end-to-end with offline fakes."""
    # 1) Build synthetic corpus
    papers = pd.DataFrame([
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"A{i}",
         "year": 2020 + (i % 3), "venue": "V"}
        for i in range(6)
    ])
    # Chain edges: W1->W0, W2->W1, ..., W5->W4
    edges = pd.DataFrame([
        {"src": f"W{i}", "dst": f"W{i-1}"}
        for i in range(1, 6)
    ])

    data_dir = tmp_path / "data"
    save_corpus(papers, edges, data_dir)

    # 2) Build ideas and embeddings
    ideas = pd.DataFrame([
        {"paper_id": f"W{i}", "idea_text": f"Idea {i}", "year": 2020 + (i % 3), "venue": "V"}
        for i in range(6)
    ])
    save_ideas(ideas, data_dir)

    emb = FakeEmbedder()
    vecs = emb.encode([f"Idea {i}" for i in range(6)])
    save_embeddings([f"W{i}" for i in range(6)], vecs, data_dir)

    # 3) Build config dict
    cfg = {
        "data_dir": str(data_dir),
        "out_dir": str(tmp_path / "runs"),
        "mailto": "t@t",
        "cutoff_date": "2025-01-01",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "eval": {"n_queries": 1, "top_k": 3, "dup_ceiling": 0.95, "arxiv_min_citations": 10},
        "models": {"summarizer": "m", "agent": "m", "judge": "m"},
        "recognized_venues": [{"name": "V", "aliases": ["v"]}],
        "run": {
            "run_id": "t1",
            "seed": 0,
            "total_steps": 4,
            "generation_budget": 2,
            "agents": [{"agent_id": "a0", "policy": "pa", "m": 2}]
        }
    }

    # 4) Monkeypatch functions
    def fake_embedder_factory(name):
        return FakeEmbedder()

    def fake_llm_factory(cfg):
        return FakeLLM(default="q")

    def fake_find_source_id(name, **kwargs):
        return "S1"

    def fake_verify_idea(llm, **kwargs):
        return Verdict(idea_id=kwargs["idea_id"], hit=False, paper=None)

    population_calls = []

    def fake_openalex_population_count(filter_str, **kwargs):
        population_calls.append(filter_str)
        # First call is the venue population, second is the arXiv population
        # (spec §3.6: population = venue + arXiv-above-citation-threshold).
        return 10 if len(population_calls) == 1 else 5

    monkeypatch.setattr("innovation.cli.Embedder", fake_embedder_factory)
    monkeypatch.setattr("innovation.cli._llm", fake_llm_factory)
    monkeypatch.setattr("innovation.cli.find_source_id", fake_find_source_id)
    monkeypatch.setattr("innovation.cli.verify_idea", fake_verify_idea)
    monkeypatch.setattr("innovation.cli.openalex_population_count", fake_openalex_population_count)

    # 5) Run cmd_run
    cmd_run(cfg)

    # 6) Assert events.jsonl exists
    events_file = Path(cfg["out_dir"]) / "t1" / "events.jsonl"
    assert events_file.exists(), f"events.jsonl not found at {events_file}"

    # 7) Run cmd_evaluate
    cmd_evaluate(cfg)

    # 8) Assert metrics.json exists and has required fields
    metrics_file = Path(cfg["out_dir"]) / "t1" / "metrics.json"
    assert metrics_file.exists(), f"metrics.json not found at {metrics_file}"

    metrics = json.loads(metrics_file.read_text())
    required_keys = {"precision", "recall", "n_ideas", "n_hits", "n_dup_flagged"}
    assert set(metrics.keys()) >= required_keys, \
        f"metrics missing keys. Expected {required_keys}, got {set(metrics.keys())}"

    # Population = venue (10) + arXiv (5) = 15; n_hits is 0 in this test (all
    # fake verdicts are misses), so recall stays 0.0 regardless of population.
    assert len(population_calls) == 2
    assert metrics["n_hits"] == 0
    assert metrics["recall"] == 0.0

    # 9) Assert verdicts.json exists with one entry per generated idea.
    verdicts_file = Path(cfg["out_dir"]) / "t1" / "verdicts.json"
    assert verdicts_file.exists(), f"verdicts.json not found at {verdicts_file}"
    verdict_records = json.loads(verdicts_file.read_text())
    assert len(verdict_records) == metrics["n_ideas"]
    required_verdict_keys = {"idea_id", "hit", "paper", "excluded_pre_cutoff",
                             "unknown_date", "dup_flag"}
    for rec in verdict_records:
        assert set(rec.keys()) >= required_verdict_keys


def test_cmd_run_refuses_to_overwrite_existing_run(tmp_path, monkeypatch):
    """Second cmd_run with the same run_id must raise SystemExit, not clobber events."""
    papers = pd.DataFrame([
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"A{i}",
         "year": 2020 + (i % 3), "venue": "V"}
        for i in range(6)
    ])
    edges = pd.DataFrame([
        {"src": f"W{i}", "dst": f"W{i-1}"}
        for i in range(1, 6)
    ])

    data_dir = tmp_path / "data"
    save_corpus(papers, edges, data_dir)

    ideas = pd.DataFrame([
        {"paper_id": f"W{i}", "idea_text": f"Idea {i}", "year": 2020 + (i % 3), "venue": "V"}
        for i in range(6)
    ])
    save_ideas(ideas, data_dir)

    emb = FakeEmbedder()
    vecs = emb.encode([f"Idea {i}" for i in range(6)])
    save_embeddings([f"W{i}" for i in range(6)], vecs, data_dir)

    cfg = {
        "data_dir": str(data_dir),
        "out_dir": str(tmp_path / "runs"),
        "mailto": "t@t",
        "cutoff_date": "2025-01-01",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "eval": {"n_queries": 1, "top_k": 3, "dup_ceiling": 0.95, "arxiv_min_citations": 10},
        "models": {"summarizer": "m", "agent": "m", "judge": "m"},
        "recognized_venues": [{"name": "V", "aliases": ["v"]}],
        "run": {
            "run_id": "t2",
            "seed": 0,
            "total_steps": 4,
            "generation_budget": 2,
            "agents": [{"agent_id": "a0", "policy": "pa", "m": 2}]
        }
    }

    def fake_embedder_factory(name):
        return FakeEmbedder()

    def fake_llm_factory(cfg):
        return FakeLLM(default="q")

    monkeypatch.setattr("innovation.cli.Embedder", fake_embedder_factory)
    monkeypatch.setattr("innovation.cli._llm", fake_llm_factory)

    cmd_run(cfg)
    with pytest.raises(SystemExit):
        cmd_run(cfg)


def test_cmd_fetch_and_summarize_wiring(tmp_path, monkeypatch):
    """Drive cmd_fetch and cmd_summarize offline with fakes."""
    import innovation.cli as cli

    cfg = {"data_dir": str(tmp_path / "data"), "mailto": "t@t",
           "recognized_venues": [{"name": "VenueA", "aliases": ["venuea"]}],
           "year_from": 2020, "cutoff_year": 2024,
           "models": {"summarizer": "m"}, "embedding_model": "fake"}

    works = [{"id": f"https://openalex.org/W{i}", "title": f"T{i}",
              "publication_year": 2020,
              "abstract_inverted_index": {"hello": [0], str(i): [1]},
              "referenced_works": ["https://openalex.org/W0"] if i else []}
             for i in range(3)]
    monkeypatch.setattr(cli, "find_source_id", lambda name, **kw: "S1")
    monkeypatch.setattr(cli, "fetch_source_works", lambda sid, y0, y1, **kw: works)
    cli.cmd_fetch(cfg)
    assert (Path(cfg["data_dir"]) / "papers.parquet").exists()
    assert (Path(cfg["data_dir"]) / "edges.parquet").exists()

    monkeypatch.setattr(cli, "_llm", lambda c: FakeLLM(default="An idea."))
    monkeypatch.setattr(cli, "Embedder", lambda name: FakeEmbedder())
    cli.cmd_summarize(cfg)
    assert (Path(cfg["data_dir"]) / "ideas.parquet").exists()
    assert (Path(cfg["data_dir"]) / "embeddings.npy").exists()


def test_cmd_fetch_field_mode(tmp_path, monkeypatch):
    """corpus.mode=field routes through fetch_field_works (citation floor only)."""
    import innovation.cli as cli

    cfg = {"data_dir": str(tmp_path / "data"), "mailto": "t@t",
           "year_from": 2016, "cutoff_year": 2024,
           "corpus": {"mode": "field", "field_query": "federated learning",
                      "min_citations": 5}}

    works = [{"id": "https://openalex.org/W1", "title": "T1",
              "publication_year": 2020,
              "abstract_inverted_index": {"fl": [0]}, "referenced_works": []}]
    calls = []

    def fake_field(query, y0, y1, **kw):
        calls.append({"query": query, "min_citations": kw.get("min_citations", 0)})
        return works

    monkeypatch.setattr(cli, "fetch_field_works", fake_field)
    cli.cmd_fetch(cfg)
    # The recognized-venue list must NOT participate in downloading.
    assert calls == [{"query": "federated learning", "min_citations": 5}]
    papers = pd.read_parquet(Path(cfg["data_dir"]) / "papers.parquet")
    assert papers.iloc[0]["venue"] == "field:federated learning"


def test_load_config_extends_and_deep_merges(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "a: 1\nrun:\n  seed: 0\n  total_steps: 100\n  agents: [{agent_id: x}]\n")
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nrun:\n  seed: 7\n  agents: [{agent_id: y}]\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["a"] == 1                       # inherited
    assert cfg["run"]["seed"] == 7             # overridden
    assert cfg["run"]["total_steps"] == 100    # deep-merged
    assert cfg["run"]["agents"] == [{"agent_id": "y"}]  # lists replace
    assert "extends" not in cfg


def test_cmd_run_no_edges_and_seed_override(tmp_path, monkeypatch):
    import innovation.cli as cli
    from innovation.experiments.events import load_events

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
    save_embeddings(list(ideas["paper_id"]), emb.encode(list(ideas["idea_text"])), data_dir)

    cfg = {"data_dir": str(data_dir), "out_dir": str(tmp_path / "runs"),
           "embedding_model": "fake",
           "models": {"agent": "m"},
           "run": {"run_id": "ne", "seed": 0, "total_steps": 2,
                   "generation_budget": 1, "init_edges": "none",
                   "agents": [{"agent_id": "a0", "policy": "pa", "m": 1}]}}
    monkeypatch.setattr(cli, "Embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr(cli, "_llm", lambda c: FakeLLM())
    cli.cmd_run(cfg, seed=9, run_id="ne-s9")
    events = load_events(Path(cfg["out_dir"]) / "ne-s9" / "events.jsonl")
    assert events and all(e["run_id"] == "ne-s9" for e in events)
    # with init_edges: none the PA policy still works (degrees all 0 -> uniform)
    gen = [e for e in events if e["action"] == "generate"]
    assert gen and "node_id" in gen[0]["result"]


def test_all_experiment_configs_load_and_scopes_build():
    from innovation.experiments.runner import build_scope
    from innovation.ideas.embed import FakeEmbedder

    emb = FakeEmbedder()
    exp_dir = Path("configs/experiments")
    files = sorted(exp_dir.glob("*.yaml"))
    assert len(files) >= 10
    for f in files:
        cfg = load_config(f)
        run = cfg["run"]
        assert run["run_id"] == f.stem
        assert run["total_steps"] == 400 and run["generation_budget"] == 40
        for spec in run["agents"]:
            build_scope(spec, emb)  # must not raise
        if f.stem.startswith("core-"):
            assert len(run["agents"]) == 10
