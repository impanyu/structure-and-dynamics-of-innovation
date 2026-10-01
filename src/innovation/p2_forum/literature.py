"""The online literature store (spec §5): Semantic Scholar behind the scope
rule, plus a persistent label store. Scope = (one of the seven venues OR
>= min_citations) AND published on/before max_date. Topic gating is NOT done
here; the environment applies it per agent."""
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


class OnlineLiterature:
    def __init__(self, *, client, scope: Scope, tagger, cache_dir, search_pool: int = 50):
        self.client, self.scope, self.tagger = client, scope, tagger
        self.search_pool = search_pool
        self._papers: dict[str, Paper] = {}
        self._lock = threading.Lock()
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self._store = Path(cache_dir) / "labels.jsonl"
        self._labels: dict[str, list[int] | None] = {}
        if self._store.exists():
            for line in self._store.read_text().splitlines():
                rec = json.loads(line)
                self._labels[rec["paper_id"]] = rec["labels"]

    def _admit(self, raws) -> list[Paper]:
        out = []
        for r in raws:
            branch = self.scope.admit(r)
            if branch is None:
                continue
            p = Paper(paper_id=r["paperId"], title=r.get("title") or "",
                      abstract=r.get("abstract") or "", year=r.get("year"),
                      venue=r.get("venue") or (r.get("publicationVenue") or {}).get("name") or "",
                      pub_date=r.get("publicationDate") or "",
                      citations=r.get("citationCount") or 0, branch=branch)
            self._papers[p.paper_id] = p
            out.append(p)
        return out

    def search(self, query: str) -> list[Paper]:
        return self._admit(self.client.search(query, limit=self.search_pool,
                                              max_date=self.scope.max_date))

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

    def has(self, pid: str) -> bool:
        return pid in self._papers

    def known(self) -> list[Paper]:
        return list(self._papers.values())

    def labels(self, pids: list[str]) -> dict[str, list[int] | None]:
        todo = [p for p in dict.fromkeys(pids) if p not in self._labels and p in self._papers]
        if todo:
            got = self.tagger.label_many([self._papers[p].text() for p in todo])
            with self._lock:
                with self._store.open("a") as f:
                    for pid, lab in zip(todo, got):
                        self._labels[pid] = lab
                        f.write(json.dumps({"paper_id": pid, "labels": lab}) + "\n")
        return {p: self._labels.get(p) for p in pids}
