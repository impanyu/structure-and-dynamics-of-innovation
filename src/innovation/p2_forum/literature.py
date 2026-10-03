"""The online literature store (spec §5): Semantic Scholar behind the scope
rule, plus a persistent label store. Scope = (one of the seven venues OR
>= min_citations) AND published on/before max_date. Topic gating is NOT done
here; the environment applies it per agent."""
import fcntl
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    abstract: str
    year: int | None
    venue: str
    pub_date: str
    citations: int
    branch: str              # "venue" or "citations": which scope arm admitted it

    def text(self) -> str:
        return f"{self.title}\n\n{self.abstract}" if self.abstract else self.title


@dataclass(frozen=True)
class Scope:
    venue_aliases: tuple[str, ...]
    min_citations: int
    max_date: str            # inclusive, YYYY-MM-DD

    def _venue_text(self, r: dict) -> str:
        pv = r.get("publicationVenue") or {}
        parts = [r.get("venue") or "", pv.get("name") or "", *(pv.get("alternate_names") or [])]
        return " | ".join(parts).lower()

    def _dated_ok(self, r: dict) -> bool:
        d = r.get("publicationDate")
        if d:
            return d <= self.max_date
        y = r.get("year")
        # Undated: only a year strictly before the cutoff year proves it.
        return bool(y) and int(y) < int(self.max_date[:4])

    def admit(self, r: dict) -> str | None:
        if not self._dated_ok(r):
            return None
        v = self._venue_text(r)
        if v and "workshop" not in v and any(a in v for a in self.venue_aliases):
            return "venue"
        if (r.get("citationCount") or 0) >= self.min_citations:
            return "citations"
        return None


STOPWORDS = frozenset(
    "a an the of to for in on with and or how what which using use via by from "
    "into is are be that this we our".split())
PUNCT = ".,;:!?\"'()[]{}"


def drop_stopwords(query: str, max_words: int = 8) -> str:
    """A search engine's fallback rewrite: drop English stopwords and keep the
    first max_words remaining words (surrounding punctuation stripped)."""
    words = [w.strip(PUNCT) for w in query.split()]
    kept = [w for w in words if w and w.lower() not in STOPWORDS]
    return " ".join(kept[:max_words])


def label_store_path(cache_dir, tagger) -> Path:
    """Labels depend on the topic list, so the store is keyed by a hash of the
    tagger's system prompt (which embeds the list). Taggers without one
    (fakes) share labels.jsonl."""
    system = getattr(tagger, "system", None)
    if not isinstance(system, str):
        return Path(cache_dir) / "labels.jsonl"
    return Path(cache_dir) / f"labels-{hashlib.sha256(system.encode()).hexdigest()[:12]}.jsonl"


class OnlineLiterature:
    def __init__(self, *, client, scope: Scope, tagger, cache_dir, search_pool: int = 50):
        self.client, self.scope, self.tagger = client, scope, tagger
        self.search_pool = search_pool
        self.last_scope_dropped = 0   # raw records the latest search/references/citations dropped
        self.last_query_used: str | None = None   # the rewrite the latest search ran, if any
        self._papers: dict[str, Paper] = {}
        self._known_ids: set[str] = set()
        self._lock = threading.Lock()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._store = label_store_path(cache_dir, tagger)
        self._labels: dict[str, list[int] | None] = {}
        self._failed: set[str] = set()
        if self._store.exists():
            self._merge(self._store.read_text())

    def _merge(self, text: str) -> set[str]:
        """Fold store lines into memory: skip malformed lines, first write wins
        (a None kept in memory only is replaced by a stored label)."""
        seen = set()
        for line in text.splitlines():
            try:
                rec = json.loads(line)
                pid, lab = rec["paper_id"], rec["labels"]
            except (ValueError, KeyError, TypeError):
                continue
            if not pid or not isinstance(lab, list) or pid in seen:
                continue
            seen.add(pid)
            if self._labels.get(pid) is None:
                self._labels[pid] = lab
        return seen

    def _admit(self, raws) -> list[Paper]:
        out = []
        dropped = 0
        for r in raws:
            branch = self.scope.admit(r) if r.get("paperId") else None
            if branch is None:
                dropped += 1
                continue
            p = Paper(paper_id=r["paperId"], title=r.get("title") or "",
                      abstract=r.get("abstract") or "", year=r.get("year"),
                      venue=r.get("venue") or (r.get("publicationVenue") or {}).get("name") or "",
                      pub_date=r.get("publicationDate") or "",
                      citations=r.get("citationCount") or 0, branch=branch)
            self._papers[p.paper_id] = p
            out.append(p)
        self.last_scope_dropped = dropped
        return out

    def search(self, query: str) -> list[Paper]:
        """Relevance search, forgiving like a search engine: when it admits no
        in-scope paper, try the query as a paper title, then once more with
        the stopwords dropped (recorded in last_query_used). last_scope_dropped
        sums the drops of every stage that ran."""
        self.last_query_used = None
        dropped = 0

        def stage(raws):
            nonlocal dropped
            got = self._admit(raws)
            dropped += self.last_scope_dropped
            return got

        run = lambda q: self.client.search(q, limit=self.search_pool, max_date=self.scope.max_date)
        got = stage(run(query))
        if not got:
            got = stage(self.client.match(query, max_date=self.scope.max_date))
        if not got:
            short = drop_stopwords(query)
            if short and short != query:
                self.last_query_used = short
                got = stage(run(short))
        self.last_scope_dropped = dropped
        return got

    def get(self, pid: str) -> Paper | None:
        if pid in self._papers:
            return self._papers[pid]
        r = self.client.paper(pid)
        got = self._admit([r]) if r else []
        return got[0] if got else None

    def references(self, pid: str) -> list[Paper]:
        return self._admit(self.client.references(pid))

    def citations(self, pid: str) -> list[Paper]:
        return self._admit(self.client.citations(pid))

    def related(self, pid: str) -> list[Paper]:
        return self._admit(self.client.recommend(pid))

    def remember_ids(self, ids) -> None:
        """Mark ids as known in-scope papers (seen in an earlier process,
        e.g. from a replayed event log) without fetching them."""
        self._known_ids.update(ids)

    def has(self, pid: str) -> bool:
        return pid in self._papers or pid in self._known_ids

    def known(self) -> list[Paper]:
        return list(self._papers.values())

    def _locked(self, fn):
        """Run fn(file, ids_in_file) under a short exclusive flock, after
        re-reading and merging the store."""
        with self._store.open("a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                return fn(f, self._merge(f.read()))
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _missing(self, wanted: list[str]) -> list[str]:
        return [p for p in wanted if self._labels.get(p) is None and p not in self._failed]

    def labels(self, pids: list[str]) -> dict[str, list[int] | None]:
        with self._lock:                 # one in-process tagger at a time
            wanted = [p for p in dict.fromkeys(pids) if p in self._papers]
            if self._missing(wanted):
                todo = self._locked(lambda f, ids: self._missing(wanted))
                if todo:                 # tag outside the file lock
                    got = self.tagger.label_many([self._papers[p].text() for p in todo])
                    fresh = {}
                    for pid, lab in zip(todo, got):
                        if lab is None:
                            self._failed.add(pid)   # this process only; others retry
                            self._labels[pid] = None
                        else:
                            fresh[pid] = lab

                    def append(f, ids):
                        for pid, lab in fresh.items():
                            if pid not in ids:      # another run wrote it first: keep theirs
                                f.write(json.dumps({"paper_id": pid, "labels": lab}) + "\n")
                                self._labels[pid] = lab
                        f.flush()
                    self._locked(append)
            return {p: self._labels.get(p) for p in pids}
