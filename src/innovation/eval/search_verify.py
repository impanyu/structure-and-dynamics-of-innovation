"""Search-verified realization: the sole evaluation channel (spec §3.6).
Semantic Scholar + OpenAlex only; hits count anticipation only."""
import functools
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from innovation.llm import LLM

S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_BASE = "https://api.openalex.org/works"

QUERY_SYSTEM = "You turn research ideas into literature search queries."
QUERY_TEMPLATE = """Give {n} short keyword search queries (4-8 words each) for finding papers \
related to this research idea: the FIRST focused on the PROBLEM it addresses, the \
SECOND on its core METHOD, the THIRD combining both. One query per line, no \
numbering, nothing else.

Idea: {idea}"""

JUDGE_SYSTEM = "You grade how closely a candidate paper realizes a proposed research idea."
JUDGE_TEMPLATE = """Research idea:
{idea}

Candidate paper:
Title: {title}
Abstract: {abstract}

Grade how closely the paper realizes the idea on this scale:
0 = different problem entirely
1 = same broad area/task, but a different goal
2 = same specific problem/goal, unrelated approach
3 = same problem AND the approaches belong to the same method family
4 = same problem AND the same core mechanism, differing in secondary components
5 = essentially the same idea: problem, core mechanism, and key design choices match

Reply with JSON only: {{"level": <0-5>, "evidence": "<one short sentence>"}}"""


def extract_queries(llm: LLM, *, model: str, idea_text: str, n: int = 3) -> list[str]:
    reply = llm.complete(model=model, system=QUERY_SYSTEM,
                         user=QUERY_TEMPLATE.format(n=n, idea=idea_text),
                         max_tokens=200)
    return [line.strip() for line in reply.splitlines() if line.strip()][:n]


RETRYABLE = {429, 500, 502, 503, 504}


def _cached_get(url: str, params: dict, cache_dir: Path, http_get,
                headers: dict | None = None, delay: float = 1.1,
                attempts: int = 10, max_sleep: float = 300.0) -> dict:
    """Disk cache with fetched_at timestamp — live indexes drift (spec §3.6).
    Retries rate-limit/server errors with capped exponential backoff and
    paces live calls (S2 keys allow 1 request/second). max_sleep caps EVERY
    per-attempt sleep, including a server-sent Retry-After: a quota-exhausted
    API advertising hours-long waits must fail into the caller's degradation
    path, not stall the evaluation."""
    key = hashlib.sha256(json.dumps({"url": url, "params": params},
                                    sort_keys=True).encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]
    for attempt in range(attempts):
        r = http_get(url, params=params, **({"headers": headers} if headers else {}))
        if getattr(r, "status_code", 200) in RETRYABLE:
            retry_after = 0.0
            try:
                retry_after = float(getattr(r, "headers", {}).get("Retry-After", 0))
            except (TypeError, ValueError):
                pass
            time.sleep(min(max_sleep,
                           max(retry_after, max(delay, 1.0) * (2 ** attempt))))
            continue
        break
    r.raise_for_status()
    payload = r.json()
    if delay:
        time.sleep(delay)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "url": url, "params": params, "response": payload}))
    return payload


# Circuit breaker: after 5 consecutive s2_search failures the channel is
# almost certainly quota-exhausted for a while — skip it for 10 minutes
# instead of burning ~30s of retries per query (OpenAlex still supplies
# candidates; failed S2 queries are uncached, so a later re-run fills gaps).
_S2_BREAKER = {"fails": 0, "until": 0.0}


def s2_search(query: str, *, cache_dir, http_get=None) -> list[dict]:
    from innovation.data.s2 import s2_headers

    if time.time() < _S2_BREAKER["until"]:
        return []
    # 30s timeout: a blackholed connection must fail into the retry/
    # degradation path, not hang the evaluation for hours. Fail fast
    # (4 attempts, sleeps capped at 10s) since verify_idea degrades S2
    # failures per-query to OpenAlex-only.
    http_get = http_get or functools.partial(requests.get, timeout=30)
    try:
        payload = _cached_get(S2_BASE, {"query": query, "limit": 10,
                                        "fields": "title,abstract,publicationDate,"
                                                  "venue,citationCount"},
                              Path(cache_dir), http_get, headers=s2_headers(),
                              attempts=4, max_sleep=10.0)
    except requests.RequestException:
        _S2_BREAKER["fails"] += 1
        if _S2_BREAKER["fails"] >= 5:
            _S2_BREAKER.update(fails=0, until=time.time() + 600)
            print("WARN s2_search circuit OPEN for 10min (5 consecutive failures)")
        raise
    _S2_BREAKER["fails"] = 0
    return [{"paper_id": p.get("paperId", ""), "title": p.get("title") or "",
             "abstract": p.get("abstract") or "",
             "pub_date": p.get("publicationDate") or "",
             "venue": p.get("venue") or "",
             "citations": p.get("citationCount") or 0, "source_api": "s2"}
            for p in payload.get("data", [])]


# Same circuit-breaker rationale as S2: a budget-exhausted OpenAlex 429
# lasts hours; skip the channel for 10 minutes after 5 consecutive failures.
_OA_BREAKER = {"fails": 0, "until": 0.0}


def openalex_search(query: str, *, mailto: str, cache_dir, http_get=None) -> list[dict]:
    from innovation.data.openalex import reconstruct_abstract

    if time.time() < _OA_BREAKER["until"]:
        return []
    # 30s timeout + fail fast (3 attempts, sleeps capped at 5s): the caller
    # degrades per-query to S2-only rather than stalling on a dead channel.
    http_get = http_get or functools.partial(requests.get, timeout=30)
    try:
        payload = _cached_get(OPENALEX_BASE,
                              {"search": query, "per-page": 10, "mailto": mailto},
                              Path(cache_dir), http_get, attempts=3, delay=1.0,
                              max_sleep=5.0)
    except requests.RequestException:
        _OA_BREAKER["fails"] += 1
        if _OA_BREAKER["fails"] >= 5:
            _OA_BREAKER.update(fails=0, until=time.time() + 600)
            print("WARN openalex_search circuit OPEN for 10min (5 consecutive failures)")
        raise
    _OA_BREAKER["fails"] = 0

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


def judge_level(llm: LLM, *, model: str, idea_text: str, candidate: dict) -> tuple[int, str]:
    """(realization level 0-5, one-line evidence). Unparsable replies -> (0, ...)."""
    reply = llm.complete(model=model, system=JUDGE_SYSTEM,
                         user=JUDGE_TEMPLATE.format(idea=idea_text,
                                                    title=candidate["title"],
                                                    abstract=candidate["abstract"]),
                         max_tokens=200)
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return 0, "unparsable judge reply"
    try:
        obj = json.loads(reply[start:end + 1])
        level = int(obj.get("level", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0, "unparsable judge reply"
    return max(0, min(5, level)), str(obj.get("evidence", ""))[:300]


EMPTY_BEST = {"level": 0, "paper": None}


@dataclass
class Verdict:
    """Three recognition tiers, CUMULATIVE scoring: best["tier1"] considers
    only tier-1 realizations, best["tier2"] tier-1+2, best["tier3"] any
    published realization. candidates buckets hold every level>=2
    post-cutoff, non-corpus candidate in its OWN tier, with level+evidence.
    Time/corpus gates precede tiering (their buckets unchanged)."""
    idea_id: str
    best: dict = field(default_factory=lambda: {
        "tier1": dict(EMPTY_BEST), "tier2": dict(EMPTY_BEST),
        "tier3": dict(EMPTY_BEST)})
    candidates: dict = field(default_factory=lambda: {
        "tier1": [], "tier2": [], "tier3": []})
    excluded_pre_cutoff: list[dict] = field(default_factory=list)
    unknown_date: list[dict] = field(default_factory=list)
    excluded_in_corpus: list[dict] = field(default_factory=list)


def tier_of(cand: dict, tier1_aliases: list[str] | None,
            tier2_aliases: list[str] | None, min_citations: int,
            tier2_min_citations: int = 10) -> str:
    """Recognition tier of a realizing paper (user rule 2026-08-30):
    tier1 = CCF-A venue OR citations >= 50;
    tier2 = CCF-A/B venue OR citations >= 10;
    tier3 = any other published paper. Unmatched venues fall DOWNWARD
    (conservative). With no alias lists configured, everything is tier1."""
    if tier1_aliases is None:
        return "tier1"
    venue = (cand.get("venue") or "").lower()
    cites = cand.get("citations") or 0
    if (venue and any(a in venue for a in tier1_aliases)) or cites >= min_citations:
        return "tier1"
    if (venue and tier2_aliases and any(a in venue for a in tier2_aliases))             or cites >= tier2_min_citations:
        return "tier2"
    return "tier3"


def verify_idea(llm: LLM, *, model: str, idea_id: str, idea_text: str,
                cutoff_date: str, mailto: str, cache_dir, http_get=None,
                n_queries: int = 3, top_k: int = 5,
                recognized_aliases: list[str] | None = None,
                tier2_aliases: list[str] | None = None,
                recognized_min_citations: int = 50,
                tier2_min_citations: int = 10,
                corpus_titles: set[str] | None = None) -> Verdict:
    queries = extract_queries(llm, model=model, idea_text=idea_text, n=n_queries)
    candidates, seen_titles = [], set()
    for q in queries:
        # Both channels degrade independently: a sustained outage on one
        # (S2 429/5xx storms, OpenAlex daily budget) must not kill a long
        # evaluation — the other channel still supplies candidates, and the
        # disk cache lets a later re-run fill the gaps.
        pool = []
        try:
            pool = s2_search(q, cache_dir=cache_dir, http_get=http_get)[:top_k]
        except requests.RequestException as exc:
            print(f"WARN s2_search degraded ({exc}); OpenAlex-only for: {q[:60]}")
        try:
            pool += openalex_search(q, mailto=mailto, cache_dir=cache_dir,
                                    http_get=http_get)[:top_k]
        except requests.RequestException as exc:
            print(f"WARN openalex_search degraded ({exc}); S2-only for: {q[:60]}")
        for cand in pool:
            title_key = cand["title"].strip().lower()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                candidates.append(cand)

    v = Verdict(idea_id=idea_id)
    for cand in candidates:
        level, evidence = judge_level(llm, model=model, idea_text=idea_text,
                                      candidate=cand)
        if level < 2:
            continue  # below "same specific problem" — not informative
        entry = {**cand, "level": level, "evidence": evidence}
        if (corpus_titles is not None
                and cand["title"].strip().lower() in corpus_titles):
            # Contamination guard: the paper is IN the initial graph.
            v.excluded_in_corpus.append(entry)
            continue
        if not cand["pub_date"]:
            v.unknown_date.append(entry)
            continue
        if cand["pub_date"] <= cutoff_date:
            v.excluded_pre_cutoff.append(entry)  # NEVER scored (spec §3.6)
            continue
        tier = tier_of(cand, recognized_aliases, tier2_aliases,
                       recognized_min_citations, tier2_min_citations)
        entry["tier"] = tier
        v.candidates[tier].append(entry)
        # cumulative bests: a tier-1 paper scores at every tier, tier-2 at
        # tier2+tier3, tier-3 only at tier3
        reach = {"tier1": ("tier1", "tier2", "tier3"),
                 "tier2": ("tier2", "tier3"),
                 "tier3": ("tier3",)}[tier]
        for t in reach:
            if level > v.best[t]["level"]:
                v.best[t] = {"level": level, "paper": entry}
    return v
