"""Semantic Scholar venue corpus: top-venue papers + reference edges.

OpenAlex's CS-conference coverage is fragmented (NeurIPS ~4k vs the real ~23k),
so the top-venue initial corpus comes from S2 bulk search instead (spec §5
fallback). Every response page is disk-cached; a `delay` throttles live calls
(S2 unauthenticated rate limits are tight; 429s are retried with backoff).
"""
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
S2_BATCH = "https://api.semanticscholar.org/graph/v1/paper/batch"
PAPER_FIELDS = "paperId,title,abstract,year,venue,citationCount"


RETRYABLE = {429, 500, 502, 503, 504}


def _cached_call(cache_file: Path, do_call, delay: float, attempts: int = 12):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    for attempt in range(attempts):
        resp = do_call()
        if resp.status_code in RETRYABLE:
            # Unauthenticated S2 shares a heavily-loaded pool; back off hard
            # (capped at 90s) and keep trying — every page is cached, so a
            # crashed run resumes where it left off.
            time.sleep(min(90.0, max(delay, 2.0) * (2 ** attempt)))
            continue
        resp.raise_for_status()
        payload = resp.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))
        if delay:
            time.sleep(delay)
        return payload
    resp.raise_for_status()
    raise RuntimeError("S2 rate limit: retries exhausted")


def s2_bulk_venue_search(venue: str, year_range: str, *, min_citations: int,
                         cache_dir, http_get=None, delay: float = 1.1) -> list[dict]:
    """All papers of a venue in year_range with citations >= min_citations."""
    http_get = http_get or requests.get
    papers, token, page_i = [], None, 0
    key = f"{venue.replace(' ', '_')}_{year_range}_c{min_citations}"
    while True:
        cache_file = Path(cache_dir) / f"s2bulk_{key}_p{page_i}.json"
        params = {"query": "", "venue": venue, "year": year_range,
                  "minCitationCount": min_citations, "fields": PAPER_FIELDS}
        if token:
            params["token"] = token
        payload = _cached_call(cache_file,
                               lambda: http_get(S2_BULK, params=params), delay)
        papers.extend(payload.get("data") or [])
        token = payload.get("token")
        page_i += 1
        if not token:
            return papers


def s2_fetch_references(paper_ids: list[str], *, cache_dir, http_post=None,
                        delay: float = 4.0, batch_size: int = 500) -> dict:
    """paper_id -> list of referenced S2 paper ids, via the batch endpoint."""
    http_post = http_post or requests.post
    refs: dict[str, list[str]] = {}
    for i in range(0, len(paper_ids), batch_size):
        batch = paper_ids[i:i + batch_size]
        key = hashlib.sha256(json.dumps(batch).encode()).hexdigest()[:24]
        cache_file = Path(cache_dir) / f"s2refs_{key}.json"
        payload = _cached_call(
            cache_file,
            lambda: http_post(S2_BATCH, params={"fields": "references.paperId"},
                              json={"ids": batch}), delay)
        for entry in payload or []:
            if not entry:
                continue
            refs[entry["paperId"]] = [
                r["paperId"] for r in entry.get("references") or []
                if r and r.get("paperId")]
    return refs


def build_s2_corpus(raw_papers: list[dict], refs: dict):
    """Normalize S2 records into the canonical papers/edges tables."""
    rows, seen = [], set()
    for p in raw_papers:
        pid = p.get("paperId")
        if not pid or pid in seen or not (p.get("abstract") or "").strip():
            continue
        seen.add(pid)
        rows.append({"paper_id": pid, "title": p.get("title") or "",
                     "abstract": p["abstract"].strip(),
                     "year": int(p.get("year") or 0),
                     "venue": p.get("venue") or ""})
    papers = pd.DataFrame(rows, columns=["paper_id", "title", "abstract",
                                         "year", "venue"])
    in_corpus = set(papers["paper_id"])
    edge_rows = [{"src": pid, "dst": ref}
                 for pid, ref_list in refs.items() if pid in in_corpus
                 for ref in ref_list if ref in in_corpus]
    edges = pd.DataFrame(edge_rows, columns=["src", "dst"])
    return papers, edges
