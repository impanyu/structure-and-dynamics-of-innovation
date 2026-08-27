"""Test CLI commands and config loader."""
import json
from pathlib import Path
from unittest.mock import patch

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

    # Venues should be names (list of str, not source IDs starting with S + digits)
    venues = cfg["venues"]
    assert isinstance(venues, list)
    for v in venues:
        assert isinstance(v, str)
        # Should be human-readable names, not S123456 style IDs
        assert not (v.startswith("S") and len(v) > 1 and v[1:].split()[0].isdigit())


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
        "venues": ["V"],
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

    def fake_openalex_population_count(filter_dict, **kwargs):
        return 10

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
