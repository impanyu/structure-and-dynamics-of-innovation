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
    assert cfg["cutoff_date"] == "2024-09-30"
    assert cfg["models"]["summarizer"] == "openai:gpt-5-mini:minimal"

    # Recognition list: evaluation-only concept, entries carry name + aliases
    recognized = cfg["recognized_venues"]
    assert isinstance(recognized, list) and len(recognized) >= 8
    for v in recognized:
        assert isinstance(v["name"], str)
        assert isinstance(v["aliases"], list) and v["aliases"]
    assert cfg["eval"]["recognized_min_citations"] == 51


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
        "eval": {"n_queries": 1, "top_k": 3, "dup_ceiling": 0.95},
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

    def fake_verify_idea(llm, **kwargs):
        return Verdict(idea_id=kwargs["idea_id"], hit=False, paper=None)

    monkeypatch.setattr("innovation.cli.Embedder", fake_embedder_factory)
    monkeypatch.setattr("innovation.cli._llm", fake_llm_factory)
    monkeypatch.setattr("innovation.cli.verify_idea", fake_verify_idea)

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
    required_keys = {"precision", "n_ideas", "n_hits", "n_dup_flagged"}
    assert set(metrics.keys()) >= required_keys, \
        f"metrics missing keys. Expected {required_keys}, got {set(metrics.keys())}"

    assert metrics["n_hits"] == 0

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
        "eval": {"n_queries": 1, "top_k": 3, "dup_ceiling": 0.95},
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
    corpus_vecs = emb.encode([f"paper {i}" for i in range(1000)])
    exp_dir = Path("configs/experiments")
    files = sorted(exp_dir.glob("*.yaml"))
    assert len(files) >= 10
    for f in files:
        cfg = load_config(f)
        run = cfg["run"]
        assert run["run_id"] == f.stem
        assert run["total_steps"] == 400 and "generation_budget" not in run
        for spec in run["agents"]:
            resolved = dict(spec)
            for key in ("read_topics", "write_topics"):
                if resolved.get(key) == "random":
                    resolved[key] = ["placeholder topic"]
            build_scope(resolved, emb, corpus_vecs=corpus_vecs)  # must not raise
        if f.stem.startswith("core-"):
            assert len(run["agents"]) == 10


def test_cmd_fetch_s2_venues_mode(tmp_path, monkeypatch):
    """corpus.mode=s2_venues builds the corpus from Semantic Scholar."""
    import innovation.cli as cli

    cfg = {"data_dir": str(tmp_path / "data"), "mailto": "t@t",
           "year_from": 2016, "cutoff_year": 2025, "cutoff_date": "2026-01-31",
           "corpus": {"mode": "s2_venues", "venues": ["NeurIPS", "ICLR"],
                      "min_citations": 50}}
    raws = {"NeurIPS": [{"paperId": "p1", "title": "A", "abstract": "aa",
                         "year": 2020, "venue": "NeurIPS", "citationCount": 60}],
            "ICLR": [{"paperId": "p2", "title": "B", "abstract": "bb",
                      "year": 2021, "venue": "ICLR", "citationCount": 70}]}

    monkeypatch.setattr(cli, "s2_bulk_venue_search",
                        lambda venue, yr, **kw: raws[venue])
    monkeypatch.setattr(cli, "s2_fetch_references",
                        lambda ids, **kw: {"p2": ["p1"]})
    monkeypatch.setattr(cli, "s2_fetch_citations", lambda ids, **kw: {})
    monkeypatch.setattr(cli, "augment_edges",
                        lambda papers, edges, **kw: edges)
    cli.cmd_fetch(cfg)
    papers = pd.read_parquet(Path(cfg["data_dir"]) / "papers.parquet")
    edges = pd.read_parquet(Path(cfg["data_dir"]) / "edges.parquet")
    assert sorted(papers["paper_id"]) == ["p1", "p2"]
    assert [(r.src, r.dst) for r in edges.itertuples()] == [("p2", "p1")]


def test_load_env_sets_missing_vars_only(tmp_path, monkeypatch):
    from innovation.config import load_env
    import os

    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment line\n"
        "OPENAI_API_KEY=sk-test-123\n"
        "ALREADY_SET=from-file\n"
        "\n"
        "QUOTED=\"q-value\"\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ALREADY_SET", "from-shell")
    load_env(env_file)
    assert os.environ["OPENAI_API_KEY"] == "sk-test-123"
    assert os.environ["ALREADY_SET"] == "from-shell"  # shell wins over file
    assert os.environ["QUOTED"] == "q-value"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
