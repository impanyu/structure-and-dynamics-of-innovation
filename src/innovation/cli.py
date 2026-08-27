"""CLI: fetch -> summarize -> run -> evaluate, all driven by one YAML config."""
import argparse
import dataclasses
import datetime
import json
from pathlib import Path

import numpy as np

from innovation.config import load_config
from innovation.data.corpus import build_corpus, load_corpus, save_corpus
from innovation.data.openalex import (fetch_field_works, fetch_source_works,
                                      find_source_id)
from innovation.eval.metrics import (aggregate_run, arxiv_population_filter,
                                     past_dup_flag, openalex_population_count,
                                     venue_population_filter)
from innovation.eval.search_verify import verify_idea
from innovation.experiments.events import load_events
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import Embedder, load_embeddings, save_embeddings
from innovation.ideas.summarize import load_ideas, save_ideas, summarize_corpus
from innovation.llm import AnthropicLLM, CachedLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def _llm(cfg):
    return CachedLLM(AnthropicLLM(), Path(cfg["data_dir"]) / "llm_cache")


def cmd_fetch(cfg):
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    corpus_cfg = cfg.get("corpus", {})
    if corpus_cfg.get("mode") == "field":
        # Small-field initial graph (spec §2): one coherent subfield, so the
        # innovation dynamics are observable on a dense, bounded network.
        query = corpus_cfg["field_query"]
        works = fetch_field_works(
            query, cfg["year_from"], cfg["cutoff_year"],
            mailto=cfg["mailto"], cache_dir=cache,
            min_citations=corpus_cfg.get("min_citations", 0))
        works_by_venue = {f"field:{query}": works}
    else:
        works_by_venue = {}
        for name in cfg["venues"]:
            sid = find_source_id(name, mailto=cfg["mailto"], cache_dir=cache)
            print(f"{name} -> {sid}")
            works_by_venue[name] = fetch_source_works(
                sid, cfg["year_from"], cfg["cutoff_year"],
                mailto=cfg["mailto"], cache_dir=cache)
    papers, edges = build_corpus(works_by_venue)
    save_corpus(papers, edges, cfg["data_dir"])
    print(f"papers={len(papers)} edges={len(edges)}")


def cmd_summarize(cfg):
    papers, _ = load_corpus(cfg["data_dir"])
    ideas = summarize_corpus(_llm(cfg), papers, model=cfg["models"]["summarizer"])
    save_ideas(ideas, cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    vecs = emb.encode(list(ideas["idea_text"]))
    save_embeddings(list(ideas["paper_id"]), vecs, cfg["data_dir"])
    print(f"ideas={len(ideas)} dim={emb.dim}")


def _load_world(cfg):
    _, edges = load_corpus(cfg["data_dir"])
    ideas = load_ideas(cfg["data_dir"])
    graph = IdeaGraph.from_tables(ideas, edges)
    ids, vecs = load_embeddings(cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    index = VectorIndex(emb.dim)
    index.add(ids, vecs)
    return graph, index, emb, dict(zip(ids, vecs))


def cmd_run(cfg):
    r = cfg["run"]
    events_path = Path(cfg["out_dir"]) / r["run_id"] / "events.jsonl"
    if events_path.exists():
        raise SystemExit(
            f"run '{r['run_id']}' already has events at {events_path}; "
            "delete it or choose a new run_id — resuming is not yet supported")
    graph, index, emb, _ = _load_world(cfg)
    run_cfg = RunConfig(run_id=r["run_id"], seed=r["seed"],
                        total_steps=r["total_steps"],
                        generation_budget=r["generation_budget"],
                        agents=r["agents"])
    out = run_simulation(run_cfg, graph=graph, index=index, embedder=emb,
                         llm=_llm(cfg), model=cfg["models"]["agent"],
                         out_dir=cfg["out_dir"])
    print(json.dumps(out, indent=2))


def cmd_evaluate(cfg):
    graph, index, emb, vec_by_id = _load_world(cfg)
    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    events = load_events(run_dir / "events.jsonl")
    generated = [(e["result"]["node_id"], e["args"]["text"])
                 for e in events
                 if e["action"] == "generate" and "node_id" in e.get("result", {})]
    corpus_vecs = np.stack(list(vec_by_id.values()))
    llm = _llm(cfg)
    verdicts, dup_flags = [], {}
    for nid, text in generated:
        verdicts.append(verify_idea(
            llm, model=cfg["models"]["judge"], idea_id=nid, idea_text=text,
            cutoff_date=cfg["cutoff_date"], mailto=cfg["mailto"],
            cache_dir=run_dir / "search_cache",
            n_queries=cfg["eval"]["n_queries"], top_k=cfg["eval"]["top_k"]))
        dup_flags[nid] = past_dup_flag(emb.encode([text])[0], corpus_vecs,
                                       ceiling=cfg["eval"]["dup_ceiling"])
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    source_ids = [find_source_id(v, mailto=cfg["mailto"], cache_dir=cache)
                  for v in cfg["venues"]]
    # Hits require pub_date strictly > cutoff_date, but from_publication_date
    # is inclusive, so start the population window one day after the cutoff.
    from_date = (datetime.date.fromisoformat(cfg["cutoff_date"])
                + datetime.timedelta(days=1)).isoformat()
    arxiv_sid = find_source_id(cfg["eval"].get("arxiv_source_name", "arXiv"),
                               mailto=cfg["mailto"], cache_dir=cache)
    # Population = post-cutoff venue papers + post-cutoff arXiv papers above a
    # citation threshold (spec §3.6). This approximates "arXiv-only": overlap
    # between the arXiv set and the venue list is not subtracted.
    population = (
        openalex_population_count(
            venue_population_filter(source_ids, from_date), mailto=cfg["mailto"])
        + openalex_population_count(
            arxiv_population_filter(arxiv_sid, from_date,
                                    cfg["eval"]["arxiv_min_citations"]),
            mailto=cfg["mailto"]))
    agg = aggregate_run(verdicts, dup_flags, population)
    (run_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    verdict_records = [
        {**dataclasses.asdict(v), "dup_flag": dup_flags[v.idea_id]}
        for v in verdicts]
    (run_dir / "verdicts.json").write_text(json.dumps(verdict_records, indent=2))
    print(json.dumps(agg, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="innovation")
    parser.add_argument("command",
                        choices=["fetch", "summarize", "run", "evaluate"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "summarize": cmd_summarize,
     "run": cmd_run, "evaluate": cmd_evaluate}[args.command](cfg)


if __name__ == "__main__":
    main()
