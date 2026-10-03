"""OpenAlex as the agents' last-resort search (Task 12e). One search costs
10 credits of a ~10,000/day budget, so it runs only after every Semantic
Scholar stage admitted nothing. Hits are mapped to Semantic Scholar ids and
S2 stays the single source of truth for scope and metadata."""
import functools
import hashlib
import json
import os
from pathlib import Path

import requests

from innovation.core.data.s2 import _cached_call

WORKS = "https://api.openalex.org/works"
SELECT = "id,doi,title,ids,publication_date"


class OpenAlexOnline:
    def __init__(self, cache_dir, http_get=None, delay: float = 0.1):
        self.cache_dir = Path(cache_dir) / "openalex"
        self.get = http_get or functools.partial(requests.get, timeout=30)
        self.delay = delay

    def search(self, query: str, *, max_date: str, limit: int = 25) -> list[dict]:
        params = {"search": query, "filter": f"to_publication_date:{max_date}",
                  "per_page": limit, "select": SELECT}
        # the key authenticates the call; it is not part of what was asked
        key = hashlib.sha256(json.dumps([WORKS, params], sort_keys=True).encode()).hexdigest()
        api_key = os.environ.get("OPENALEX_API_KEY", "").strip()
        sent = {**params, "api_key": api_key} if api_key else params
        out = _cached_call(self.cache_dir / f"{key}.json",
                           lambda: self.get(WORKS, params=sent), self.delay)
        return [w for w in ((out or {}).get("results") or []) if isinstance(w, dict)]


def s2_lookup_id(work: dict) -> str | None:
    """"DOI:<doi>" when the work has a DOI, else "ARXIV:<id>" from its ids."""
    doi = work.get("doi") or (work.get("ids") or {}).get("doi")
    if doi:
        return "DOI:" + doi.split("doi.org/", 1)[-1]
    arxiv = (work.get("ids") or {}).get("arxiv")
    if arxiv:
        return "ARXIV:" + arxiv.rstrip("/").rsplit("/", 1)[-1]
    return None


def map_to_s2(works: list[dict], s2, *, max_date: str) -> tuple[list[dict], int]:
    """(S2 records in OpenAlex order, number of hits that could not be mapped).
    Works with a DOI/arXiv id go through one S2 batch call; the rest fall back
    to an S2 title match."""
    ids = [s2_lookup_id(w) for w in works]
    wanted = [i for i in dict.fromkeys(ids) if i]
    found = dict(zip(wanted, s2.batch(wanted))) if wanted else {}
    out, seen, unmapped = [], set(), 0
    for w, i in zip(works, ids):
        rec = found.get(i) if i else None
        if rec is None and w.get("title"):
            hit = s2.match(w["title"], max_date=max_date)
            rec = hit[0] if hit else None
        if rec is None:
            unmapped += 1
        elif rec["paperId"] not in seen:
            seen.add(rec["paperId"])
            out.append(rec)
    return out, unmapped
