"""CLI: fetch -> summarize -> run -> evaluate, all driven by one YAML config."""
import argparse
import dataclasses
import json
import os
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
        # Abstract-less papers are dropped by build_s2_corpus; count them.
        dated, _ = build_s2_corpus(
            [{**r, "abstract": r.get("abstract") or "x"} for r in raw], {},
            before_date=before_date)
        print(f"raw={len(raw)} date_admitted={len(dated)} "
              f"dropped_no_abstract={len(dated) - len(admitted)}")
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


def _abstract_ideas(papers: pd.DataFrame) -> pd.DataFrame:
    """idea_text = title + blank line + original abstract; no LLM. Papers
    without an abstract are dropped (the caller prunes their edges)."""
    papers = papers[papers["abstract"].fillna("").str.strip() != ""]
    text = (papers["title"].fillna("").str.strip() + "\n\n"
            + papers["abstract"].str.strip())
    return pd.DataFrame({"paper_id": papers["paper_id"].values,
                         "idea_text": text.values,
                         "year": papers["year"].values,
                         "venue": papers["venue"].values})


def cmd_summarize(cfg):
    papers, edges = load_corpus(cfg["data_dir"])
    if cfg.get("corpus", {}).get("idea_text") == "abstract":
        ideas = _abstract_ideas(papers)
        dropped = len(papers) - len(ideas)
        if dropped:
            keep = set(ideas["paper_id"])
            edges = edges[edges["src"].isin(keep) & edges["dst"].isin(keep)]
            save_corpus(papers[papers["paper_id"].isin(keep)], edges,
                        cfg["data_dir"])
        print(f"dropped_no_abstract={dropped}")
        save_ideas(ideas, cfg["data_dir"])
        emb = Embedder(cfg["embedding_model"])
        vecs = emb.encode(list(ideas["idea_text"]))
        save_embeddings(list(ideas["paper_id"]), vecs, cfg["data_dir"])
        print(f"ideas={len(ideas)} dim={emb.dim}")
        return
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


def _tier1_aliases(cfg) -> tuple[str, ...]:
    """The lowercased venue aliases of the tier-1 recognition rule
    (cfg["recognized_venues"]). One source of truth: the evaluation's tier-1
    venue test and the online reading scope both use it."""
    return tuple(a.lower() for v in (cfg.get("recognized_venues") or [])
                 for a in v.get("aliases", []))


def _load_online_world(cfg):
    """Paper 2's online world: the frozen topic list, its tagger, the online
    literature behind the scope rule, and the embedder (for the board)."""
    from innovation.p2_forum.literature import OnlineLiterature, Scope
    from innovation.p2_forum.s2_online import S2Online
    from innovation.p2_forum.tagger import TopicTagger
    from innovation.p2_forum.topics import load_topics

    on = cfg["online"]
    stale = [k for k in ("venues", "min_citations_any_venue") if k in on]
    if stale:
        raise SystemExit(
            f"online.{stale[0]} is no longer a config key: the reading scope uses "
            "the evaluation's tier-1 rule (top-level recognized_venues and "
            "eval.recognized_min_citations). Remove "
            + ", ".join(f"online.{k}" for k in stale) + " from the config.")
    topics = load_topics(cfg["topics_file"])
    llm = CachedLLM(RoutedLLM(), Path(on["cache_dir"]) / "llm")
    tagger = TopicTagger(llm=llm, model=cfg["models"]["tagger"], topics=topics,
                         refusal_log=Path(on["cache_dir"]) / "tagger_refusals.jsonl")
    scope = Scope(venue_aliases=_tier1_aliases(cfg),
                  min_citations=cfg["eval"].get("recognized_min_citations", 50),
                  max_date=on["max_pub_date"])
    openalex = None
    if os.environ.get("OPENALEX_API_KEY", "").strip():
        from innovation.p2_forum.openalex_online import OpenAlexOnline
        openalex = OpenAlexOnline(on["cache_dir"])
    else:
        print("WARN OPENALEX_API_KEY is not set: agent search has no OpenAlex fallback")
    lit = OnlineLiterature(client=S2Online(on["cache_dir"]), scope=scope, tagger=tagger,
                           cache_dir=on["cache_dir"], search_pool=on.get("search_pool", 50),
                           openalex=openalex)
    return lit, tagger, Embedder(cfg["embedding_model"]), topics


class _AnyPaper:
    """Online replay needs no literature: every id the log cites was accepted
    when it was written, so any non-board id is taken as an existing paper."""

    @staticmethod
    def has(node_id: str) -> bool:
        return not str(node_id).startswith("gen:")


def _write_board_metrics(cfg, corpus, index, emb) -> Path:
    """The board's structure as a function of round (spec §5), replayed from the
    event log into <run_dir>/board_metrics.json. Round = one action per agent."""
    from innovation.p2_forum.board_metrics import board_trajectory

    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    events = load_events(run_dir / "events.jsonl")
    n_agents = len(cfg["run"]["agents"])
    online = cfg.get("literature") == "online"
    series = board_trajectory(events, corpus=None if online else corpus,
                              corpus_index=None if online else index,
                              embedder=emb, run_id=cfg["run"]["run_id"],
                              n_agents=n_agents,
                              external_papers=_AnyPaper() if online else None)
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
    if cfg.get("literature") == "online":
        corpus, index, emb = None, None, Embedder(cfg["embedding_model"])
    else:
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
        from innovation.p2_forum.env import Navigation
        from innovation.p2_forum.runner import (ForumRunConfig, resume_forum,
                                                run_forum)
        common = dict(
            run_id=r["run_id"], seed=r["seed"], total_steps=r["total_steps"],
            agents=r["agents"],
            generation_budget=r.get("generation_budget"),
            navigation=Navigation.from_config(cfg.get("navigation")),
            topic_draw=r.get("topic_draw", "independent"),
            # ForumRunConfig refuses a mismatched pair (e.g. gating: topics
            # over the corpus), so a half-edited config cannot run silently.
            gating=cfg.get("gating", "none"))
        # resume replays the log before continuing; running fresh over an
        # existing log would re-issue gen:<run_id>:<n> ids and make the log
        # unreplayable (the primary research artifact).
        forum = resume_forum if resume else run_forum
        if cfg.get("literature") == "online":
            lit, tagger, emb, topics = _load_online_world(cfg)
            corpus = index = None
            run_cfg = ForumRunConfig(
                **common, literature="online",
                topic_pool=[t.name for t in topics],
                topic_definitions={t.name: t.definition for t in topics})
            out = forum(run_cfg, corpus=None, corpus_index=None, embedder=emb,
                        llm=_llm(cfg), model=cfg["models"]["agent"],
                        out_dir=cfg["out_dir"], literature=lit, tagger=tagger)
        else:
            # Corpus mode, soft or region-gated. Region mode builds each
            # agent's ball from the corpus index's ids and embedding matrix
            # and draws no topics.
            corpus, index, emb = _load_forum_world(cfg)
            region = cfg.get("gating") == "region"
            # a typo such as `literature: onlne` is refused here, not run as corpus
            run_cfg = ForumRunConfig(**common,
                                     literature=cfg.get("literature", "corpus"),
                                     topic_pool=[] if region else _topic_pool(cfg))
            extra = {}
            if region:
                # The model that names each region's topics for the prompt
                # (region_topics.py). A fresh region run without it would
                # silently describe no area, so refuse; resume reads the
                # topics from run_meta and never calls it.
                namer = cfg.get("models", {}).get("topic_namer")
                if namer is None and not resume:
                    raise SystemExit("gating: region needs models.topic_namer "
                                     "(see configs/p2_forum/base-region.yaml)")
                extra["topic_namer"] = namer
            out = forum(run_cfg, corpus=corpus, corpus_index=index, embedder=emb,
                        llm=_llm(cfg), model=cfg["models"]["agent"],
                        out_dir=cfg["out_dir"], **extra)
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


def _shown_papers(events):
    """Papers any agent was shown in an online run, as {node_id: (title, text)}.

    Reads hits, browse views, sample_frontier views and their cites/cited_by
    items; board (gen:) ids are skipped.
    """
    shown = {}

    def take(item):
        if not isinstance(item, dict):
            return
        nid, title = item.get("node_id"), item.get("title")
        if not nid or not title or str(nid).startswith("gen:"):
            return
        shown.setdefault(nid, (title, item.get("text") or ""))

    for e in events:
        res = e.get("result")
        if not isinstance(res, dict) or "error" in res:
            continue
        if e.get("action") == "search":
            for h in res.get("hits", []):
                take(h)
        elif e.get("action") == "related":
            for h in res.get("related", []):
                take(h)
        elif e.get("action") in ("browse", "sample_frontier"):
            take(res)
            for k in ("cites", "cited_by"):
                for it in res.get(k, []) or []:
                    take(it)
    return shown


def _eval_reference(cfg):
    """(corpus_titles, corpus_vecs) used for the contamination guard and the
    near-duplicate check. Online: the papers the run's agents were shown."""
    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    shown = _shown_papers(load_events(run_dir / "events.jsonl"))
    titles = {t.strip().lower() for t, _ in shown.values()}
    texts = [f"{t}\n\n{x}" for t, x in shown.values()]
    emb = Embedder(cfg["embedding_model"])
    if not texts:
        return titles, np.zeros((0, 0))
    return titles, np.asarray(emb.encode(texts))


def cmd_evaluate(cfg):
    online = cfg.get("literature") == "online"
    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    if online:
        emb = Embedder(cfg["embedding_model"])
        corpus_titles, corpus_vecs = _eval_reference(cfg)
    else:
        graph, index, emb, vec_by_id = _load_world(cfg)
    events = load_events(run_dir / "events.jsonl")
    generated = [(e["result"]["node_id"], e["args"]["text"])
                 for e in events
                 if e["action"] == "generate" and "node_id" in e.get("result", {})]
    if not online:
        corpus_vecs = np.stack(list(vec_by_id.values()))
    llm = _llm(cfg)
    # Recognition rule (evaluation only): a realizing paper counts iff its
    # venue matches the recognized-venue alias list OR its citations clear
    # eval.recognized_min_citations.
    aliases = list(_tier1_aliases(cfg))
    tier2_aliases = [a.lower() for a in (cfg.get("ccf_b_aliases") or [])]
    # Contamination guard: papers already in the initial graph can never be
    # anticipation hits (the agent may simply have read them).
    if not online:
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
            exclude_workshops=cfg["eval"].get("exclude_workshops", False),
            corpus_titles=corpus_titles))
        dup_flags[nid] = (len(corpus_vecs) > 0 and past_dup_flag(
            emb.encode([text])[0], corpus_vecs, ceiling=cfg["eval"]["dup_ceiling"]))
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
