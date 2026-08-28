"""Search-verified realization: the sole evaluation channel (spec §3.6).
Semantic Scholar + OpenAlex only; hits count anticipation only."""
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from innovation.llm import LLM

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_BASE = "https://api.openalex.org/works"

QUERY_SYSTEM = "You turn research ideas into literature search queries."
QUERY_TEMPLATE = """Give {n} short keyword search queries (4-8 words each) that would find \
papers implementing this research idea. One query per line, no numbering, nothing else.

Idea: {idea}"""

JUDGE_SYSTEM = "You judge whether a paper realizes a proposed research idea."
JUDGE_TEMPLATE = """Research idea:
{idea}

Candidate paper:
Title: {title}
Abstract: {abstract}

Does this paper genuinely realize the core of the idea (same problem AND same key \
approach, not merely the same topic)? Answer YES or NO on the first line, then one \
sentence of justification."""


def extract_queries(llm: LLM, *, model: str, idea_text: str, n: int = 3) -> list[str]:
    reply = llm.complete(model=model, system=QUERY_SYSTEM,
                         user=QUERY_TEMPLATE.format(n=n, idea=idea_text),
                         max_tokens=200)
    return [line.strip() for line in reply.splitlines() if line.strip()][:n]


def _cached_get(url: str, params: dict, cache_dir: Path, http_get) -> dict:
    """Disk cache with fetched_at timestamp — live indexes drift (spec §3.6)."""
    key = hashlib.sha256(json.dumps({"url": url, "params": params},
                                    sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]
    r = http_get(url, params=params)
    r.raise_for_status()
    payload = r.json()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "url": url, "params": params, "response": payload}))
    return payload


def s2_search(query: str, *, cache_dir, http_get=None) -> list[dict]:
    http_get = http_get or requests.get
    payload = _cached_get(S2_BASE, {"query": query, "limit": 10,
                                    "fields": "title,abstract,publicationDate,"
                                              "venue,citationCount"},
                          Path(cache_dir), http_get)
    return [{"paper_id": p.get("paperId", ""), "title": p.get("title") or "",
             "abstract": p.get("abstract") or "",
             "pub_date": p.get("publicationDate") or "",
             "venue": p.get("venue") or "",
             "citations": p.get("citationCount") or 0, "source_api": "s2"}
            for p in payload.get("data", [])]


def openalex_search(query: str, *, mailto: str, cache_dir, http_get=None) -> list[dict]:
    from innovation.data.openalex import reconstruct_abstract

    http_get = http_get or requests.get
    payload = _cached_get(OPENALEX_BASE,
                          {"search": query, "per-page": 10, "mailto": mailto},
                          Path(cache_dir), http_get)

    def venue_of(w):
        loc = w.get("primary_location") or {}
        return ((loc.get("source") or {}).get("display_name")) or ""

    return [{"paper_id": (w.get("id") or "").rsplit("/", 1)[-1],
             "title": w.get("title") or "",
             "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
             "pub_date": w.get("publication_date") or "",
             "venue": venue_of(w),
             "citations": w.get("cited_by_count") or 0, "source_api": "openalex"}
            for w in payload.get("results", [])]


def judge_realization(llm: LLM, *, model: str, idea_text: str, candidate: dict) -> bool:
    reply = llm.complete(model=model, system=JUDGE_SYSTEM,
                         user=JUDGE_TEMPLATE.format(idea=idea_text,
                                                    title=candidate["title"],
                                                    abstract=candidate["abstract"]),
                         max_tokens=150)
    return reply.strip().upper().startswith("YES")


@dataclass
class Verdict:
    idea_id: str
    hit: bool
    paper: dict | None
    excluded_pre_cutoff: list[dict] = field(default_factory=list)
    unknown_date: list[dict] = field(default_factory=list)
    excluded_unrecognized: list[dict] = field(default_factory=list)
    excluded_in_corpus: list[dict] = field(default_factory=list)


def is_recognized(cand: dict, recognized_aliases: list[str] | None,
                  min_citations: int) -> bool:
    """A realizing paper counts as recognized innovation iff its venue matches
    the recognized top-venue alias list OR its citation count clears the
    impact floor. With no alias list configured, everything is recognized."""
    if recognized_aliases is None:
        return True
    venue = (cand.get("venue") or "").lower()
    if venue and any(alias in venue for alias in recognized_aliases):
        return True
    return (cand.get("citations") or 0) >= min_citations


def verify_idea(llm: LLM, *, model: str, idea_id: str, idea_text: str,
                cutoff_date: str, mailto: str, cache_dir, http_get=None,
                n_queries: int = 3, top_k: int = 5,
                recognized_aliases: list[str] | None = None,
                recognized_min_citations: int = 10,
                corpus_titles: set[str] | None = None) -> Verdict:
    queries = extract_queries(llm, model=model, idea_text=idea_text, n=n_queries)
    candidates, seen_titles = [], set()
    for q in queries:
        for cand in (s2_search(q, cache_dir=cache_dir, http_get=http_get)[:top_k]
                     + openalex_search(q, mailto=mailto, cache_dir=cache_dir,
                                       http_get=http_get)[:top_k]):
            title_key = cand["title"].strip().lower()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                candidates.append(cand)

    hit_paper, excluded, unknown, unrecognized, in_corpus = None, [], [], [], []
    for cand in candidates:
        if not judge_realization(llm, model=model, idea_text=idea_text, candidate=cand):
            continue
        if (corpus_titles is not None
                and cand["title"].strip().lower() in corpus_titles):
            # Contamination guard: the paper is IN the initial graph, so the
            # agent could simply have read it — never an anticipation hit.
            in_corpus.append(cand)
            continue
        if cand["pub_date"] and cand["pub_date"] > cutoff_date:
            if not is_recognized(cand, recognized_aliases, recognized_min_citations):
                unrecognized.append(cand)  # realized, but not by a recognized paper
            elif hit_paper is None:
                hit_paper = cand  # first recognized post-cutoff realization
        elif cand["pub_date"]:
            excluded.append(cand)  # logged, NEVER scored (spec §3.6)
        else:
            unknown.append(cand)  # unknown date realizations logged separately
    return Verdict(idea_id=idea_id, hit=hit_paper is not None,
                   paper=hit_paper, excluded_pre_cutoff=excluded,
                   unknown_date=unknown, excluded_unrecognized=unrecognized,
                   excluded_in_corpus=in_corpus)
