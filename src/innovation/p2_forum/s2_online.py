"""Live Semantic Scholar lookups behind a disk cache shared by all runs
(spec §5). Every response is cached by (url, params), so a cached item is
never refetched; run reproducibility comes from the event log, not from
this cache."""
import functools
import hashlib
import json
from pathlib import Path

import requests

from innovation.core.data.s2 import _cached_call, s2_headers

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = ("paperId,title,abstract,year,venue,publicationVenue,"
          "publicationDate,citationCount")


class S2Online:
    def __init__(self, cache_dir, http_get=None, delay: float = 1.1):
        self.cache_dir = Path(cache_dir) / "s2"
        self.get = http_get or functools.partial(requests.get, timeout=30)
        self.delay = delay

    def _call(self, url: str, params: dict) -> dict | None:
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        try:
            return _cached_call(self.cache_dir / f"{key}.json",
                                lambda: self.get(url, params=params, headers=s2_headers(),
                                                 timeout=30),
                                self.delay)
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                return None
            raise

    @staticmethod
    def _clean(items) -> list[dict]:
        return [p for p in items if p and p.get("paperId")]

    def search(self, query: str, *, limit: int, max_date: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/search",
                         {"query": query, "limit": limit, "fields": FIELDS,
                          "publicationDateOrYear": f":{max_date}"})
        return self._clean((out or {}).get("data", []))

    def paper(self, pid: str) -> dict | None:
        return self._call(f"{BASE}/paper/{pid}", {"fields": FIELDS})

    def references(self, pid: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/{pid}/references", {"fields": FIELDS, "limit": 1000})
        return self._clean(d.get("citedPaper") for d in (out or {}).get("data", []))

    def citations(self, pid: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/{pid}/citations", {"fields": FIELDS, "limit": 1000})
        return self._clean(d.get("citingPaper") for d in (out or {}).get("data", []))
