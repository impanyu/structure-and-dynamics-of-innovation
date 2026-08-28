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
from innovation.data.edge_augment import augment_edges
from innovation.data.s2 import (build_s2_corpus, s2_bulk_venue_search,
                                s2_fetch_references)
from innovation.eval.metrics import (aggregate_run, arxiv_population_filter,
                                     past_dup_flag, openalex_population_count,
                                     venue_population_filter)
from innovation.eval.search_verify import verify_idea
from innovation.experiments.events import load_events
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import Embedder, load_embeddings, save_embeddings
from innovation.ideas.summarize import load_ideas, save_ideas, summarize_corpus
from innovation.llm import CachedLLM, RoutedLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def _llm(cfg):
    return CachedLLM(RoutedLLM(), Path(cfg["data_dir"]) / "llm_cache")


def cmd_fetch(cfg):
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    corpus_cfg = cfg.get("corpus", {})
    if corpus_cfg.get("mode") == "s2_venues":
        # Top-venue corpus via Semantic Scholar (OpenAlex venue coverage is
        # fragmented). Broad AI/ML coverage; citation floor bounds the size.
        raw = []
        for venue in corpus_cfg["venues"]:
            batch = s2_bulk_venue_search(
                venue, f"{cfg['year_from']}-{cfg['cutoff_year']}",
                min_citations=corpus_cfg.get("min_citations", 0),
                cache_dir=cache)
            print(f"{venue}: {len(batch)} papers")
            raw.extend(batch)
        # Corpus admits only papers published BEFORE the agent model's
        # training-cutoff month (first day of cutoff_date's month).
        before_date = cfg["cutoff_date"][:8] + "01"
        admitted, _ = build_s2_corpus(raw, {}, before_date=before_date)
        ids = sorted(admitted["paper_id"])
        refs = s2_fetch_references(ids, cache_dir=cache)
        papers, edges = build_s2_corpus(raw, refs, before_date=before_date)
        print(f"papers={len(papers)} s2_edges={len(edges)}")
        # S2 reference coverage is incomplete for these venues; union in
        # OpenAlex references matched by MAG/DOI/arXiv id.
        edges = augment_edges(papers, edges, cache_dir=cache,
                              mailto=cfg["mailto"], delay=2.5)
        save_corpus(papers, edges, cfg["data_dir"])
        print(f"papers={len(papers)} edges={len(edges)} (after OpenAlex augmentation)")
        return
    if corpus_cfg.get("mode") == "field":
        # Small-field initial graph (spec §2): field query + year range +
        # citation floor. The recognized-venue list is an EVALUATION concept
        # (hit recognition + recall denominator), not a download filter.
        query = corpus_cfg["field_query"]
        works = fetch_field_works(
            query, cfg["year_from"], cfg["cutoff_year"],
            mailto=cfg["mailto"], cache_dir=cache,
            min_citations=corpus_cfg.get("min_citations", 0))
        works_by_venue = {f"field:{query}": works}
    else:
        works_by_venue = {}
        for venue in cfg["recognized_venues"]:
            name = venue["name"]
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
    if cfg.get("run", {}).get("init_edges", "citations") == "none":
        # Ablation (supplementary experiments): independent nodes, no edges —
        # isolates the value of the citation structure itself.
        edges = edges.iloc[0:0]
    graph = IdeaGraph.from_tables(ideas, edges)
    ids, vecs = load_embeddings(cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    index = VectorIndex(emb.dim)
    index.add(ids, vecs)
    return graph, index, emb, dict(zip(ids, vecs))


def cmd_run(cfg, seed=None, run_id=None):
    r = cfg["run"]
    if seed is not None:
        r["seed"] = seed
    if run_id is not None:
        r["run_id"] = run_id
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
    # Recognition rule (evaluation only): a realizing paper counts iff its
    # venue matches the recognized-venue alias list OR its citations clear
    # eval.recognized_min_citations.
    recognized = cfg.get("recognized_venues") or []
    aliases = [a.lower() for v in recognized for a in v.get("aliases", [])]
    # Contamination guard: papers already in the initial graph can never be
    # anticipation hits (the agent may simply have read them).
    corpus_papers, _ = load_corpus(cfg["data_dir"])
    corpus_titles = {t.strip().lower() for t in corpus_papers["title"] if t}
    verdicts, dup_flags = [], {}
    for nid, text in generated:
        verdicts.append(verify_idea(
            llm, model=cfg["models"]["judge"], idea_id=nid, idea_text=text,
            cutoff_date=cfg["cutoff_date"], mailto=cfg["mailto"],
            cache_dir=run_dir / "search_cache",
            n_queries=cfg["eval"]["n_queries"], top_k=cfg["eval"]["top_k"],
            recognized_aliases=aliases or None,
            recognized_min_citations=cfg["eval"].get("recognized_min_citations", 10),
            corpus_titles=corpus_titles))
        dup_flags[nid] = past_dup_flag(emb.encode([text])[0], corpus_vecs,
                                       ceiling=cfg["eval"]["dup_ceiling"])
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    source_ids = [find_source_id(v["name"], mailto=cfg["mailto"], cache_dir=cache)
                  for v in recognized]
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
    parser.add_argument("--seed", type=int, default=None,
                        help="override run.seed (for multi-seed sweeps)")
    parser.add_argument("--run-id", default=None,
                        help="override run.run_id (for multi-seed sweeps)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "run":
        cmd_run(cfg, seed=args.seed, run_id=args.run_id)
        return
    if args.command == "evaluate" and args.run_id is not None:
        cfg["run"]["run_id"] = args.run_id
    {"fetch": cmd_fetch, "summarize": cmd_summarize,
     "evaluate": cmd_evaluate}[args.command](cfg)


if __name__ == "__main__":
    main()
