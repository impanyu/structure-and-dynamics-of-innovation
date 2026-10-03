"""Topic capacity (Task 2c): the estimated number of papers in the readable
scope that genuinely belong to a topic.

Readable scope = published on/before MAX_DATE AND (venue in the tier-1 list
OR citationCount >= MIN_CITATIONS). Per topic, the tagger model writes a
Semantic Scholar bulk-search query; S2 returns the match totals
  n_v  (venue filter), n_c (minCitationCount), n_vc (both),
so the scope union is n_v + n_c - n_vc. Keyword matches are not topic
members, so each pool is corrected by a labeling precision measured on a
sample of its records with the current topic list:
  capacity = n_v * p_v + (n_c - n_vc) * p_c.
Every S2 response is cached (trimmed to what is used) under cache_dir."""
import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import requests

from innovation.core.data.s2 import _cached_call, s2_headers
from innovation.p2_forum.topics import Topic

S2_BULK = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
MAX_DATE = "2024-09-30"
MIN_CITATIONS = 50
SAMPLE_FIELDS = "paperId,title,abstract,venue,publicationVenue"


def venue_chunks(names: list[str]) -> list[list[str]]:
    """S2's venue filter is a comma-separated list, so a name that itself
    contains a comma is queried alone; the rest go in one chunk. A paper has
    one venue, so per-chunk totals are disjoint and sum."""
    plain = [v for v in names if "," not in v]
    return ([plain] if plain else []) + [[v] for v in names if "," in v]


def union_count(n_v: int, n_c: int, n_vc: int) -> int:
    return n_v + n_c - n_vc


def capacity(n_v: int, n_c: int, n_vc: int, p_v: float, p_c: float) -> float:
    """Venue pool counted at its precision plus the citation-only pool
    (cited >= MIN_CITATIONS, venue not in the list) at its own."""
    return n_v * p_v + (n_c - n_vc) * p_c


def precision(sample: list[str], labels: dict[str, list[int] | None], topic_id: int):
    """Share of the labeled sample papers whose labels include topic_id.
    Unlabeled papers (tagger failure) are left out. Returns (p, n_labeled);
    p = 0.0 for an empty sample."""
    labeled = [pid for pid in sample if labels.get(pid) is not None]
    if not labeled:
        return 0.0, 0
    return sum(topic_id in labels[pid] for pid in labeled) / len(labeled), len(labeled)


# --- the per-topic query --------------------------------------------------

QUERY_SYSTEM = "You write Semantic Scholar bulk-search queries."
_BAD_CHARS = re.compile(r'[+()*~\[\]{}]')


def query_prompt(topic: dict) -> str:
    return (
        f"Research topic: {topic['name']} — {topic['definition']}\n\n"
        "Write a Semantic Scholar bulk-search boolean query that finds papers of "
        "this topic in titles and abstracts: 2 to 5 core phrases that papers of "
        "the topic actually use, joined by ' | '. Put every multi-word or "
        "hyphenated phrase in double quotes. No other operators.\n"
        'Example: "graph neural network" | "message passing" | "node classification"\n'
        "Reply with ONLY the query on one line.")


def parse_query(reply: str) -> str:
    line = next((ln.strip().strip("`").strip() for ln in reply.splitlines()
                 if ln.strip().strip("`").strip()), "")
    terms = [t.strip() for t in line.split("|")]
    if not 2 <= len(terms) <= 5 or not all(terms):
        raise ValueError(f"need 2-5 non-empty phrases joined by |, got {line!r}")
    out = []
    for t in terms:
        inner = t[1:-1] if len(t) >= 2 and t[0] == t[-1] == '"' else t
        if '"' in inner or _BAD_CHARS.search(inner) or not inner.strip():
            raise ValueError(f"phrase {t!r} has operators or stray quotes")
        inner = inner.strip()
        out.append(f'"{inner}"' if re.search(r"[\s-]", inner) else inner)
    return " | ".join(out)


# --- S2 counts ------------------------------------------------------------

class _Payload:
    """A requests-like response around an already-trimmed JSON payload, so
    _cached_call caches only what is used (bulk pages are up to 1000 records)."""
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        return None


def _venue_of(rec: dict) -> set[str]:
    names = {rec.get("venue") or ""}
    names.add(((rec.get("publicationVenue") or {}).get("name")) or "")
    return {n for n in names if n}


@dataclass
class Counts:
    n_v: int
    n_c: int
    n_vc: int
    venue_sample: list[dict] = field(default_factory=list)
    citation_sample: list[dict] = field(default_factory=list)


class BulkCounter:
    """Bulk-search totals within the readable scope, cached per request."""

    def __init__(self, cache_dir, venues: list[str], *, http_get=None,
                 delay: float = 1.1, max_date: str = MAX_DATE,
                 min_citations: int = MIN_CITATIONS, sample_n: int = 20):
        self.cache_dir = Path(cache_dir)
        self.venues, self.chunks = list(venues), venue_chunks(list(venues))
        self.venue_set = set(venues)
        self.get = http_get or functools.partial(requests.get, timeout=60)
        self.delay, self.max_date = delay, max_date
        self.min_citations, self.sample_n = min_citations, sample_n
        self.live_calls = 0

    def _fetch(self, query: str, *, chunk=None, cited=False, sample=None) -> dict:
        """{"total", "page", "matched", "data"}: data = up to sample_n records of
        the first page satisfying `sample` ("venue" = any, "cited_only" = venue
        not in the list); matched counts first-page records whose venue is the
        single requested name (used for comma names, see count())."""
        params = {"query": query, "publicationDateOrYear": f":{self.max_date}",
                  "fields": SAMPLE_FIELDS if sample else "paperId,venue"}
        if chunk:
            params["venue"] = ",".join(chunk)
        if cited:
            params["minCitationCount"] = str(self.min_citations)
        spec = [params, sample, self.sample_n]
        key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        target = chunk[0] if chunk and len(chunk) == 1 else None

        def trim(raw: dict) -> dict:
            page = [r for r in (raw.get("data") or []) if isinstance(r, dict)]
            keep = []
            for r in page:
                if len(keep) >= self.sample_n or not sample:
                    break
                if not (r.get("title") or "").strip():
                    continue
                if sample == "cited_only" and _venue_of(r) & self.venue_set:
                    continue
                keep.append({k: r.get(k) for k in ("paperId", "title", "abstract", "venue")})
            return {"total": int(raw.get("total") or 0), "page": len(page),
                    "matched": sum(target in _venue_of(r) for r in page) if target else None,
                    "data": keep}

        def do_call():
            self.live_calls += 1
            resp = self.get(S2_BULK, params=params, headers=s2_headers())
            return _Payload(trim(resp.json())) if resp.status_code == 200 else resp

        return _cached_call(self.cache_dir / f"{key}.json", do_call, self.delay)

    @staticmethod
    def _effective(out: dict, chunk: list[str]) -> int:
        """S2 splits the venue filter on commas even for a name queried alone,
        so a comma name can match unrelated venues (e.g. 'Languages') or none.
        Its total is scaled by the share of first-page records that really
        carry the name (exact when total <= one page)."""
        if "," not in chunk[0] or len(chunk) > 1:
            return out["total"]
        if not out["page"]:
            return 0
        return round(out["total"] * out["matched"] / out["page"])

    def count(self, query: str) -> Counts:
        n_v, n_vc, venue_sample = 0, 0, []
        c = self._fetch(query, cited=True, sample="cited_only")
        for i, chunk in enumerate(self.chunks):
            v = self._fetch(query, chunk=chunk, sample="venue" if i == 0 else None)
            nv = self._effective(v, chunk)
            n_v += nv
            if i == 0:
                venue_sample = v["data"]
            if nv and c["total"]:
                n_vc += self._effective(self._fetch(query, chunk=chunk, cited=True), chunk)
        return Counts(n_v=n_v, n_c=c["total"], n_vc=n_vc,
                      venue_sample=venue_sample, citation_sample=c["data"])


# --- the estimator --------------------------------------------------------

class NoTextAsEmpty:
    """Wraps the tagger's LLM (outside its cache): a reply without text (e.g.
    stop_reason=refusal) becomes "", so the tagger retries and finally leaves
    that sample paper unlabeled instead of aborting the whole estimate.
    Nothing is cached for such a reply."""

    def __init__(self, inner):
        self.inner = inner

    def complete(self, **kw) -> str:
        try:
            return self.inner.complete(**kw)
        except ValueError as e:
            if not str(e).startswith("no text in reply"):
                raise
            return ""


def _text(rec: dict) -> str:
    title, abstract = (rec.get("title") or "").strip(), (rec.get("abstract") or "").strip()
    return f"{title}\n\n{abstract}" if abstract else title


@dataclass
class Estimate:
    records: list[dict]                       # per topic, in list order
    labels: dict[str, list[int] | None]       # sample paper id -> labels
    titles: list[list[str]]                   # per topic: sample titles labeled with it

    @property
    def capacities(self) -> list[float]:
        return [r["capacity"] for r in self.records]


class CapacityEstimator:
    """estimate(topics) -> Estimate. `llm` writes the queries (cached by the
    caller's CachedLLM); `tagger_factory(list[Topic])` returns a tagger with
    label_many (the current list labels the samples: labels compete)."""

    def __init__(self, *, counter: BulkCounter, llm, model: str, tagger_factory,
                 workers: int = 16, log=print):
        self.counter, self.llm, self.model = counter, llm, model
        self.tagger_factory, self.workers, self.log = tagger_factory, workers, log

    def query(self, topic: dict) -> str:
        prompt, feedback = query_prompt(topic), ""
        for _ in range(3):
            reply = self.llm.complete(model=self.model, system=QUERY_SYSTEM,
                                      user=prompt + feedback, max_tokens=200)
            try:
                return parse_query(reply)
            except ValueError as e:
                feedback = f"\n\nYour previous answer was rejected: {e}. Fix it."
        raise RuntimeError(f"no valid query for topic {topic['name']!r} in 3 attempts")

    def __call__(self, topics: list[dict]) -> Estimate:
        queries, counts = [], []
        for i, t in enumerate(topics):
            q = self.query(t)
            queries.append(q)
            counts.append(self.counter.count(q))
            if (i + 1) % 16 == 0:
                self.log(f"  counted {i + 1}/{len(topics)} topics "
                         f"(live S2 calls so far: {self.counter.live_calls})")
        papers: dict[str, dict] = {}
        for c in counts:
            for rec in c.venue_sample + c.citation_sample:
                papers.setdefault(rec["paperId"], rec)
        pids = list(papers)
        tagger = self.tagger_factory([Topic(id=i, name=t["name"], definition=t["definition"],
                                            sources=list(t.get("sources", [])))
                                      for i, t in enumerate(topics)])
        self.log(f"  labeling {len(pids)} sample papers")
        got = tagger.label_many([_text(papers[p]) for p in pids], workers=self.workers)
        labels = dict(zip(pids, got))
        records, titles = [], []
        for i, (t, q, c) in enumerate(zip(topics, queries, counts)):
            sv = [r["paperId"] for r in c.venue_sample]
            sc = [r["paperId"] for r in c.citation_sample]
            p_v, k_v = precision(sv, labels, i)
            p_c, k_c = precision(sc, labels, i)
            records.append({"name": t["name"], "capacity": capacity(c.n_v, c.n_c, c.n_vc, p_v, p_c),
                            "n_v": c.n_v, "n_c": c.n_c, "n_vc": c.n_vc,
                            "p_v": p_v, "p_c": p_c, "sample_v": k_v, "sample_c": k_c,
                            "query": q})
            own = [papers[p]["title"] for p in sv + sc if i in (labels.get(p) or [])]
            titles.append(own or [papers[p]["title"] for p in sv + sc])
        return Estimate(records=records, labels=labels, titles=titles)
