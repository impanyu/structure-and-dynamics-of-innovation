"""CLI: fetch -> summarize -> run -> evaluate, all driven by one YAML config."""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from innovation.core.config import load_config, load_env
from innovation.core.data.corpus import build_corpus, load_corpus, save_corpus
from innovation.core.data.openalex import (fetch_field_works, fetch_source_works,
                                      find_source_id)
from innovation.core.data.edge_augment import augment_edges
from innovation.core.data.s2 import (build_s2_corpus, s2_bulk_venue_search,
                                s2_fetch_citations, s2_fetch_references)
from innovation.core.eval.metrics import aggregate_run, past_dup_flag
from innovation.core.eval.search_verify import verify_idea
from innovation.core.events import load_events
from innovation.p1_dial.runner import (RunConfig, resume_simulation,
                                            run_simulation)
from innovation.core.ideas.embed import Embedder, load_embeddings, save_embeddings
from innovation.core.ideas.summarize import load_ideas, save_ideas, summarize_corpus
from innovation.core.llm import CachedLLM, RoutedLLM
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex


def _llm(cfg):
    return CachedLLM(RoutedLLM(), Path(cfg["data_dir"]) / "llm_cache")


def _topic_pool(cfg) -> list[str]:
    pool = yaml.safe_load(Path(cfg["topics_file"]).read_text())
    return [t["topic"] for t in pool["topics"]]


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
        # Reference lists alone are incomplete (many corpus papers have no
        # parsed bibliography anywhere). Recover A->B edges from B's CITATIONS
        # list too, then union in OpenAlex references (MAG/DOI/arXiv match).
        in_corpus = set(papers["paper_id"])
        cits = s2_fetch_citations(sorted(in_corpus), cache_dir=cache)
        rev_rows = [{"src": citer, "dst": cited}
                    for cited, citers in cits.items()
                    for citer in citers if citer in in_corpus]
        edges = pd.concat([edges, pd.DataFrame(rev_rows, columns=["src", "dst"])],
                          ignore_index=True).drop_duplicates()
        print(f"papers={len(papers)} edges={len(edges)} (after citations arm)")
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
    ideas = summarize_corpus(_llm(cfg), papers, model=cfg["models"]["summarizer"],
                             workers=int(cfg.get("summarize_workers", 12)))
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


def _load_forum_world(cfg):
    """Paper 2's world: the corpus, frozen, plus its index. The board is
    created per run by the runner."""
    _, edges = load_corpus(cfg["data_dir"])
    ideas = load_ideas(cfg["data_dir"])
    if cfg.get("run", {}).get("init_edges", "citations") == "none":
        edges = edges.iloc[0:0]
    corpus = IdeaGraph.from_tables(ideas, edges)
    corpus.freeze()
    ids, vecs = load_embeddings(cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    index = VectorIndex(emb.dim)
    index.add(ids, vecs)
    return corpus, index, emb


def _write_board_metrics(cfg, corpus, index, emb) -> Path:
    """The board's structure as a function of round (spec §5), replayed from the
    event log into <run_dir>/board_metrics.json. Round = one action per agent."""
    from innovation.p2_forum.board_metrics import board_trajectory

    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    events = load_events(run_dir / "events.jsonl")
    n_agents = len(cfg["run"]["agents"])
    series = board_trajectory(events, corpus=corpus, corpus_index=index,
                              embedder=emb, run_id=cfg["run"]["run_id"],
                              n_agents=n_agents)
    out_path = run_dir / "board_metrics.json"
    out_path.write_text(json.dumps(
        {"run_id": cfg["run"]["run_id"], "n_agents": n_agents,
         "n_events": len(events), "n_rounds": len(series),
         "final": series[-1] if series else None,
         "rounds": series}, indent=1))
    return out_path


def cmd_board_metrics(cfg):
    """Re-derive the structural metrics of an existing paper-2 run."""
    if cfg.get("arch", "p1_dial") != "p2_forum":
        raise SystemExit("board-metrics applies to arch: p2_forum runs only")
    corpus, index, emb = _load_forum_world(cfg)
    path = _write_board_metrics(cfg, corpus, index, emb)
    print(f"wrote {path}")
    print(json.dumps(json.loads(path.read_text())["final"], indent=2))


def cmd_run(cfg, seed=None, run_id=None, resume=False):
    r = cfg["run"]
    if seed is not None:
        r["seed"] = seed
    if run_id is not None:
        r["run_id"] = run_id
    events_path = Path(cfg["out_dir"]) / r["run_id"] / "events.jsonl"
    if events_path.exists() and not resume:
        raise SystemExit(
            f"run '{r['run_id']}' already has events at {events_path}; "
            "pass --resume (with a raised total_steps) to extend it, or use a new run_id")
    if cfg.get("arch", "p1_dial") == "p2_forum":
        from innovation.p2_forum.runner import (ForumRunConfig, resume_forum,
                                                run_forum)
        corpus, index, emb = _load_forum_world(cfg)
        run_cfg = ForumRunConfig(
            run_id=r["run_id"], seed=r["seed"], total_steps=r["total_steps"],
            agents=r["agents"], topic_pool=_topic_pool(cfg),
            generation_budget=r.get("generation_budget"))
        # resume replays the log before continuing; running fresh over an
        # existing log would re-issue gen:<run_id>:<n> ids and make the log
        # unreplayable (the primary research artifact).
        forum = resume_forum if resume else run_forum
        out = forum(run_cfg, corpus=corpus, corpus_index=index, embedder=emb,
                    llm=_llm(cfg), model=cfg["models"]["agent"],
                    out_dir=cfg["out_dir"])
        # spec §8: a completed run produces headline AND structural metrics.
        # The headline ones need the judge (cmd_evaluate); the structural ones
        # are a pure replay of the log we just wrote, so write them here.
        print(f"wrote {_write_board_metrics(cfg, corpus, index, emb)}")
        print(json.dumps(out, indent=2))
        return
    graph, index, emb, _ = _load_world(cfg)
    topic_pool = _topic_pool(cfg) if cfg.get("topics_file") else None
    run_cfg = RunConfig(run_id=r["run_id"], seed=r["seed"],
                        total_steps=r["total_steps"],
                        generation_budget=r.get("generation_budget"),
                        agents=r["agents"], topic_pool=topic_pool)
    simulate = resume_simulation if resume else run_simulation
    out = simulate(run_cfg, graph=graph, index=index, embedder=emb,
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
    tier2_aliases = [a.lower() for a in (cfg.get("ccf_b_aliases") or [])]
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
            tier2_aliases=tier2_aliases or None,
            recognized_min_citations=cfg["eval"].get("recognized_min_citations", 50),
            tier2_min_citations=cfg["eval"].get("tier2_min_citations", 10),
            corpus_titles=corpus_titles))
        dup_flags[nid] = past_dup_flag(emb.encode([text])[0], corpus_vecs,
                                       ceiling=cfg["eval"]["dup_ceiling"])
    agg = aggregate_run(verdicts, dup_flags,
                        realized_min_date=cfg["eval"].get("realized_min_date"))
    (run_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    verdict_records = [
        {**dataclasses.asdict(v), "dup_flag": dup_flags[v.idea_id]}
        for v in verdicts]
    (run_dir / "verdicts.json").write_text(json.dumps(verdict_records, indent=2))
    print(json.dumps(agg, indent=2))


def cmd_visualize(cfg):
    from innovation.core.analysis.viz import plot_run

    emb = Embedder(cfg["embedding_model"])
    png, tj = plot_run(Path(cfg["out_dir"]) / cfg["run"]["run_id"],
                       cfg["data_dir"], emb)
    print(f"wrote {png}\nwrote {tj}")


def main():
    parser = argparse.ArgumentParser(prog="innovation")
    parser.add_argument("command",
                        choices=["fetch", "summarize", "run", "evaluate",
                                 "visualize", "board-metrics"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="override run.seed (for multi-seed sweeps)")
    parser.add_argument("--run-id", default=None,
                        help="override run.run_id (for multi-seed sweeps)")
    parser.add_argument("--steps", type=int, default=None,
                        help="override run.total_steps (e.g. to extend before --resume)")
    parser.add_argument("--resume", action="store_true",
                        help="continue an existing run up to total_steps")
    args = parser.parse_args()
    load_env()  # API keys from ./.env (shell env takes precedence)
    cfg = load_config(args.config)
    if args.command == "run":
        if args.steps is not None:
            cfg["run"]["total_steps"] = args.steps
        cmd_run(cfg, seed=args.seed, run_id=args.run_id, resume=args.resume)
        return
    if (args.command in ("evaluate", "visualize", "board-metrics")
            and args.run_id is not None):
        cfg["run"]["run_id"] = args.run_id
    {"fetch": cmd_fetch, "summarize": cmd_summarize,
     "evaluate": cmd_evaluate, "visualize": cmd_visualize,
     "board-metrics": cmd_board_metrics}[args.command](cfg)


if __name__ == "__main__":
    main()
