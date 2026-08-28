"""End-to-end: synthetic corpus -> ideas -> graph -> simulation -> evaluation.
Everything offline: FakeLLM, FakeEmbedder, fake http_get."""
import json

import pandas as pd

from innovation.eval.metrics import aggregate_run, past_dup_flag
from innovation.eval.search_verify import verify_idea
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import FakeEmbedder
from innovation.ideas.summarize import summarize_corpus
from innovation.llm import CachedLLM, FakeLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_full_pipeline_smoke(tmp_path):
    # 1) Synthetic papers -> ideas
    papers = pd.DataFrame(
        [{"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"About topic {i}",
          "year": 2018 + i % 5, "venue": "NeurIPS"} for i in range(10)])
    sum_llm = CachedLLM(FakeLLM(default="A distilled idea."), tmp_path / "llm_cache")
    ideas = summarize_corpus(sum_llm, papers, model="haiku")
    ideas["idea_text"] = [f"Idea {i} about topic {i}" for i in range(10)]  # unique texts

    # 2) Graph + index
    edges = pd.DataFrame([{"src": f"W{i}", "dst": f"W{i - 1}"} for i in range(1, 10)])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    corpus_vecs = emb.encode([graph.node(n).text for n in ids])
    index.add(ids, corpus_vecs)

    # 3) Simulate: 2 LLM agents, stigmergy, shared budget
    gen = json.dumps({"action": "generate",
                      "args": {"text": "a fresh combination idea",
                               "cited_ids": ["W1", "W5"]}})
    search = json.dumps({"action": "search", "args": {"query": "topic", "k": 2}})
    agent_llm = FakeLLM(responses=[search, search], default=gen)
    cfg = RunConfig(run_id="smoke", seed=7, total_steps=8, generation_budget=3,
                    agents=[{"agent_id": "a0", "policy": "llm"},
                            {"agent_id": "a1", "policy": "llm"}])
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=agent_llm, model="sonnet", out_dir=tmp_path / "runs")
    assert len(out["generated"]) == 3

    # 4) Evaluate: one post-cutoff realization exists in the fake index
    payload = {"data": [{"paperId": "future", "title": "Future paper",
                         "abstract": "does it", "publicationDate": "2025-09-09"}]}

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {"count": 50}})

    eval_llm = FakeLLM(responses=["some query", "YES"] * 10)
    verdicts = [verify_idea(eval_llm, model="sonnet", idea_id=nid,
                            idea_text=graph.node(nid).text,
                            cutoff_date="2025-01-01", mailto="a@b.c",
                            cache_dir=tmp_path / "search", http_get=fake_get,
                            n_queries=1, top_k=3)
                for nid in out["generated"]]
    dup_flags = {nid: past_dup_flag(emb.encode([graph.node(nid).text])[0],
                                    corpus_vecs)
                 for nid in out["generated"]}
    agg = aggregate_run(verdicts, dup_flags)
    assert agg["n_ideas"] == 3
    assert 0.0 <= agg["precision"] <= 1.0
    assert agg["n_hits"] >= 1
