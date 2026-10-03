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
from innovation.core.fsutil import atomic_write_text

BASE = "https://api.semanticscholar.org/graph/v1"
REC_BASE = "https://api.semanticscholar.org/recommendations/v1"
FIELDS = ("paperId,title,abstract,year,venue,publicationVenue,"
          "publicationDate,citationCount")


class S2Online:
    def __init__(self, cache_dir, http_get=None, delay: float = 1.1, http_post=None):
        self.cache_dir = Path(cache_dir) / "s2"
        self.get = http_get or functools.partial(requests.get, timeout=30)
        self.delay = delay
        self.post = http_post

    def _call(self, url: str, params: dict, *, cache_404=None) -> dict | None:
        """A cached GET. A 404 returns None, or, when cache_404 is given, caches
        and returns that payload so the same miss is not asked again."""
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        try:
            return _cached_call(cache_file,
                                lambda: self.get(url, params=params, headers=s2_headers(),
                                                 timeout=30),
                                self.delay)
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) != 404:
                raise
            if cache_404 is None:
                return None
            atomic_write_text(cache_file, json.dumps(cache_404))
            return cache_404

    @staticmethod
    def _data(out) -> list[dict]:
        """The `data` list of a response; S2 sometimes sends {"data": null} or null entries."""
        return [d for d in ((out or {}).get("data") or []) if isinstance(d, dict)]

    @staticmethod
    def _clean(items) -> list[dict]:
        return [p for p in items if isinstance(p, dict) and p.get("paperId")]

    def search(self, query: str, *, limit: int, max_date: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/search",
                         {"query": query, "limit": limit, "fields": FIELDS,
                          "publicationDateOrYear": f":{max_date}"})
        return self._clean(self._data(out))

    def match(self, title: str, *, max_date: str) -> list[dict]:
        """The single best title match (a list of at most one); [] when S2 finds
        none (404, cached as empty). max_date is not sent: the scope rule applies the cutoff."""
        out = self._call(f"{BASE}/paper/search/match", {"query": title, "fields": FIELDS},
                         cache_404={"data": []})
        return self._clean(self._data(out))

    def batch(self, ids: list[str]) -> list[dict | None]:
        """Resolve external ids (DOI:..., ARXIV:...) in one POST; one entry per id,
        None where S2 does not know it. Cached by the sorted id list."""
        ids = list(ids)
        if not ids:
            return []
        key = hashlib.sha256(json.dumps(["batch", sorted(ids)]).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        post = self.post or functools.partial(requests.post, timeout=30)
        out = _cached_call(cache_file,
                           lambda: post(f"{BASE}/paper/batch", params={"fields": FIELDS},
                                        json={"ids": ids}, headers=s2_headers()),
                           self.delay)
        return [p if isinstance(p, dict) and p.get("paperId") else None for p in (out or [])]

    def paper(self, pid: str) -> dict | None:
        return self._call(f"{BASE}/paper/{pid}", {"fields": FIELDS})

    def references(self, pid: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/{pid}/references", {"fields": FIELDS, "limit": 1000})
        return self._clean(d.get("citedPaper") for d in self._data(out))

    def citations(self, pid: str) -> list[dict]:
        out = self._call(f"{BASE}/paper/{pid}/citations", {"fields": FIELDS, "limit": 1000})
        return self._clean(d.get("citingPaper") for d in self._data(out))

    def recommend(self, pid: str) -> list[dict]:
        """Embedding-based recommendations for a paper (Scholar's "Related
        articles"). The all-cs pool, not the default last-60-days one; 500 is
        the maximum, over-fetched because the scope rule drops many."""
        out = self._call(f"{REC_BASE}/papers/forpaper/{pid}",
                         {"fields": FIELDS, "limit": 500, "from": "all-cs"},
                         cache_404={"recommendedPapers": []})
        return self._clean((out or {}).get("recommendedPapers") or [])
