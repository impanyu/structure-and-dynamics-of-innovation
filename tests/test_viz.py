"""Tests for trajectory extraction and run plotting (offline)."""
import json

import numpy as np

from innovation.analysis.viz import (extract_trajectories, place_new_points,
                                     plot_run)
from innovation.ideas.embed import FakeEmbedder, save_embeddings


def test_extract_trajectories_reads_and_writes():
    events = [
        {"agent_id": "a0", "step": 0, "action": "sample_frontier",
         "args": {}, "result": {"node_id": "W1", "text": "t"}},
        {"agent_id": "a0", "step": 2, "action": "browse",
         "args": {"node_id": "W2"}, "result": {"node_id": "W2", "text": "t"}},
        {"agent_id": "a0", "step": 4, "action": "generate",
         "args": {"text": "x", "cited_ids": ["W1"]}, "result": {"node_id": "gen:r:0"}},
        {"agent_id": "a0", "step": 6, "action": "browse",
         "args": {"node_id": "nope"}, "result": {"error": "boom"}},
        {"agent_id": "a1", "step": 1, "action": "search",
         "args": {"query": "q"}, "result": {"hits": []}},
    ]
    t = extract_trajectories(events)
    assert [v["node_id"] for v in t["a0"]] == ["W1", "W2", "gen:r:0"]
    assert [v["kind"] for v in t["a0"]] == ["read", "read", "write"]
    assert "a1" not in t  # search-only agents have no path points


def test_place_new_points_lands_near_neighbors():
    emb = FakeEmbedder()
    corpus = emb.encode([f"p{i}" for i in range(20)])
    coords = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
    new = place_new_points(corpus[:1], corpus, coords, k=1)
    np.testing.assert_allclose(new[0], coords[0], atol=1e-4)  # identical vec -> same spot


def test_plot_run_writes_png_and_json(tmp_path):
    emb = FakeEmbedder()
    ids = [f"W{i}" for i in range(30)]
    vecs = emb.encode([f"paper {i}" for i in range(30)])
    save_embeddings(ids, vecs, tmp_path / "data")
    # tiny precomputed projection to skip TSNE
    np.save(tmp_path / "data" / "proj_tsne.npy",
            np.random.default_rng(0).normal(size=(30, 2)).astype(np.float32))
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    events = [
        {"seq": 0, "run_id": "r1", "agent_id": "a0", "step": 0,
         "action": "browse", "args": {"node_id": "W3"},
         "result": {"node_id": "W3", "text": "t"}},
        {"seq": 1, "run_id": "r1", "agent_id": "a0", "step": 1,
         "action": "generate", "args": {"text": "new idea", "cited_ids": ["W3"]},
         "result": {"node_id": "gen:r1:0"}},
    ]
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    png, tj = plot_run(run_dir, tmp_path / "data", emb)
    assert png.exists() and png.stat().st_size > 10000
    assert json.loads(tj.read_text())["a0"][1]["kind"] == "write"


def test_plot_run_tints_topic_regions(tmp_path):
    emb = FakeEmbedder()
    ids = [f"W{i}" for i in range(30)]
    vecs = emb.encode([f"paper {i}" for i in range(30)])
    save_embeddings(ids, vecs, tmp_path / "data")
    np.save(tmp_path / "data" / "proj_tsne.npy",
            np.random.default_rng(0).normal(size=(30, 2)).astype(np.float32))
    run_dir = tmp_path / "runs" / "r2"
    run_dir.mkdir(parents=True)
    (run_dir / "events.jsonl").write_text(json.dumps(
        {"seq": 0, "run_id": "r2", "agent_id": "s0", "step": 0,
         "action": "browse", "args": {"node_id": "W3"},
         "result": {"node_id": "W3", "text": "t"}}))
    (run_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": "r2", "seed": 0,
         "topic_assignments": {"s0": "paper 3", "s1": "paper 7"}}))
    png, _ = plot_run(run_dir, tmp_path / "data", emb, topic_mass=5)
    assert png.exists() and png.stat().st_size > 10000
