# Stage 1 Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full experimental apparatus for the "idea networks and agent dynamics" study at Stage 1 scale (NeurIPS + ICLR, ~20k papers): data → idea summaries → idea network → agent simulation → search-verified evaluation.

**Architecture:** Python monorepo (`src/innovation/`), six modules: `data` (OpenAlex fetch), `ideas` (LLM summaries + embeddings), `network` (IdeaGraph over networkx + vector index), `agents` (Policy interface: LLM agent + two baselines), `experiments` (environment, event log, runner), `eval` (search-verified realization + metrics). Storage is file-based: parquet tables + JSONL event logs; graph state is always reconstructible as initial tables + event replay. Every LLM call and HTTP response is disk-cached.

**Tech Stack:** Python ≥3.12, uv, pytest, networkx, pandas+pyarrow, numpy, sentence-transformers, anthropic SDK, requests, PyYAML.

**Spec:** `docs/superpowers/specs/2026-08-27-idea-network-agent-dynamics-design.md`

## Global Constraints

- Python ≥ 3.12, managed with `uv`; run everything via `uv run …`.
- TDD: every task writes its failing test first; commit per task.
- All LLM calls disk-cached (key = sha256 of model+system+user). All external HTTP responses disk-cached with `fetched_at` timestamps.
- Literature search uses Semantic Scholar + OpenAlex APIs only — **never Google Scholar** (spec §3.6).
- Evaluation hits count **anticipation only**: realizing paper published after the model cutoff (config `cutoff_date: "2025-01-01"`); pre-cutoff matches logged with `excluded_pre_cutoff`, never scored (spec §3.6).
- Single vs multi-agent comparisons hold **total generation budget** equal (spec §3.4).
- Unit tests use fakes (FakeLLM, FakeEmbedder, fake `http_get`); nothing in `uv run pytest` touches the network or downloads models. Integration tests that do are marked `@pytest.mark.slow`.
- Stage 1 backend note (documented deviation): the spec's FAISS role is filled by a numpy brute-force `VectorIndex` behind a swappable interface — at 20k×384 a dot product is milliseconds; FAISS (like igraph for graphs) is the Stage 2 swap. The spec's philosophy (interface isolates backend) is unchanged.
- OpenAlex venue/source IDs are never hardcoded: configs carry venue *names*; IDs are resolved at runtime via the `/sources?search=` endpoint and cached.

---

### Task 1: Project scaffold + cached LLM client

**Files:**
- Create: `pyproject.toml`, `src/innovation/__init__.py`, `src/innovation/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `LLM` protocol with `complete(*, model: str, system: str, user: str, max_tokens: int = 1024) -> str`; classes `FakeLLM(responses=None, default="ok")` (records `.calls`), `CachedLLM(inner, cache_dir)`, `AnthropicLLM()`. All later LLM-using tasks accept any `LLM`.

- [ ] **Step 1: Write `pyproject.toml` and package skeleton**

```toml
[project]
name = "innovation"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "networkx>=3.3",
    "pandas>=2.2",
    "pyarrow>=16",
    "numpy>=1.26",
    "requests>=2.32",
    "pyyaml>=6",
    "anthropic>=0.40",
    "sentence-transformers>=3",
]

[dependency-groups]
dev = ["pytest>=8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/innovation"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow: touches network or downloads models"]
```

Create empty `src/innovation/__init__.py`. Run `uv sync`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_llm.py
from innovation.llm import CachedLLM, FakeLLM


def test_fake_llm_returns_canned_responses_and_records_calls():
    llm = FakeLLM(responses=["first", "second"])
    assert llm.complete(model="m", system="s", user="u1") == "first"
    assert llm.complete(model="m", system="s", user="u2") == "second"
    assert llm.complete(model="m", system="s", user="u3") == "ok"  # default
    assert [c["user"] for c in llm.calls] == ["u1", "u2", "u3"]


def test_cached_llm_hits_disk_cache(tmp_path):
    inner = FakeLLM(responses=["expensive"])
    llm = CachedLLM(inner, cache_dir=tmp_path)
    assert llm.complete(model="m", system="s", user="u") == "expensive"
    # Second identical call must come from cache, not the inner client.
    assert llm.complete(model="m", system="s", user="u") == "expensive"
    assert len(inner.calls) == 1
    # A different prompt is a cache miss.
    llm.complete(model="m", system="s", user="other")
    assert len(inner.calls) == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.llm'`

- [ ] **Step 4: Implement `src/innovation/llm.py`**

```python
"""LLM clients: a Protocol, a test fake, a disk cache, and the real Anthropic client."""
import hashlib
import json
from pathlib import Path
from typing import Protocol


class LLM(Protocol):
    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str: ...


class FakeLLM:
    """Returns canned responses in order, then `default`. Records every call."""

    def __init__(self, responses=None, default: str = "ok"):
        self.responses = list(responses or [])
        self.default = default
        self.calls: list[dict] = []

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        self.calls.append({"model": model, "system": system, "user": user})
        return self.responses.pop(0) if self.responses else self.default


class CachedLLM:
    """Disk cache keyed by sha256(model+system+user). Reproducibility + cost control (spec §3.2)."""

    def __init__(self, inner: LLM, cache_dir: Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, model: str, system: str, user: str) -> Path:
        payload = json.dumps({"model": model, "system": system, "user": user}, sort_keys=True)
        return self.cache_dir / (hashlib.sha256(payload.encode()).hexdigest() + ".json")

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        path = self._path(model, system, user)
        if path.exists():
            return json.loads(path.read_text())["response"]
        response = self.inner.complete(model=model, system=system, user=user, max_tokens=max_tokens)
        path.write_text(json.dumps(
            {"model": model, "system": system, "user": user, "response": response}))
        return response


class AnthropicLLM:
    """Real client. Needs ANTHROPIC_API_KEY in the environment."""

    def __init__(self):
        import anthropic

        self.client = anthropic.Anthropic()

    def complete(self, *, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        msg = self.client.messages.create(
            model=model, system=system, max_tokens=max_tokens,
            messages=[{"role": "user", "content": user}])
        return msg.content[0].text
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_llm.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/innovation tests/test_llm.py
git commit -m "feat: project scaffold + disk-cached LLM client"
```

---

### Task 2: OpenAlex client

**Files:**
- Create: `src/innovation/data/__init__.py`, `src/innovation/data/openalex.py`
- Test: `tests/test_openalex.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `reconstruct_abstract(inv: dict | None) -> str`; `find_source_id(name: str, *, mailto: str, cache_dir: Path, http_get=None) -> str`; `fetch_source_works(source_id: str, year_from: int, year_to: int, *, mailto: str, cache_dir: Path, http_get=None) -> list[dict]` (raw OpenAlex work dicts). `http_get` has the `requests.get(url, params=...)` signature and returns an object with `.json()` and `.raise_for_status()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openalex.py
import json

from innovation.data.openalex import (fetch_source_works, find_source_id,
                                      reconstruct_abstract)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def make_fake_get(pages):
    """pages: list of payloads returned in call order. Records params."""
    calls = []

    def fake_get(url, params=None):
        calls.append({"url": url, "params": params})
        return FakeResponse(pages[len(calls) - 1])

    fake_get.calls = calls
    return fake_get


def test_reconstruct_abstract_orders_words_by_position():
    inv = {"networks": [2], "idea": [1], "grow": [3], "the": [0]}
    assert reconstruct_abstract(inv) == "the idea networks grow"
    assert reconstruct_abstract(None) == ""


def test_fetch_source_works_paginates_and_caches(tmp_path):
    page1 = {"results": [{"id": "W1"}, {"id": "W2"}],
             "meta": {"next_cursor": "abc"}}
    page2 = {"results": [{"id": "W3"}], "meta": {"next_cursor": None}}
    fake_get = make_fake_get([page1, page2])
    works = fetch_source_works("S123", 2013, 2024, mailto="a@b.c",
                               cache_dir=tmp_path, http_get=fake_get)
    assert [w["id"] for w in works] == ["W1", "W2", "W3"]
    assert len(fake_get.calls) == 2
    assert "S123" in fake_get.calls[0]["params"]["filter"]
    # Second call: everything comes from the disk cache, zero HTTP.
    fake_get2 = make_fake_get([])
    works2 = fetch_source_works("S123", 2013, 2024, mailto="a@b.c",
                                cache_dir=tmp_path, http_get=fake_get2)
    assert works2 == works
    assert len(fake_get2.calls) == 0


def test_find_source_id_returns_top_search_hit(tmp_path):
    payload = {"results": [{"id": "https://openalex.org/S999",
                            "display_name": "NeurIPS"}]}
    fake_get = make_fake_get([payload])
    sid = find_source_id("NeurIPS", mailto="a@b.c", cache_dir=tmp_path,
                         http_get=fake_get)
    assert sid == "S999"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_openalex.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.data'`

- [ ] **Step 3: Implement `src/innovation/data/openalex.py`** (and empty `__init__.py`)

```python
"""Thin OpenAlex API client with disk caching. https://docs.openalex.org"""
import json
from pathlib import Path

import requests

OPENALEX_BASE = "https://api.openalex.org"


def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex ships abstracts as {word: [positions]}; invert back to text."""
    if not inv:
        return ""
    positions = [(p, w) for w, ps in inv.items() for p in ps]
    return " ".join(w for _, w in sorted(positions))


def _cached_json(cache_file: Path, fetch):
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    payload = fetch()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload))
    return payload


def find_source_id(name: str, *, mailto: str, cache_dir: Path, http_get=None) -> str:
    """Resolve a venue name to its OpenAlex source id (short form, e.g. 'S999')."""
    http_get = http_get or requests.get
    cache_file = Path(cache_dir) / f"source_{name.replace(' ', '_')}.json"

    def fetch():
        r = http_get(f"{OPENALEX_BASE}/sources",
                     params={"search": name, "mailto": mailto})
        r.raise_for_status()
        return r.json()

    payload = _cached_json(cache_file, fetch)
    full_id = payload["results"][0]["id"]  # top hit; audited via CLI in Task 14
    return full_id.rsplit("/", 1)[-1]


def fetch_source_works(source_id: str, year_from: int, year_to: int, *,
                       mailto: str, cache_dir: Path, http_get=None) -> list[dict]:
    """All works of a source in [year_from, year_to], cursor-paginated, page-cached."""
    http_get = http_get or requests.get
    works: list[dict] = []
    cursor, page_i = "*", 0
    while cursor:
        cache_file = (Path(cache_dir)
                      / f"works_{source_id}_{year_from}_{year_to}_p{page_i}.json")

        def fetch(cursor=cursor):
            r = http_get(f"{OPENALEX_BASE}/works", params={
                "filter": (f"primary_location.source.id:{source_id},"
                           f"publication_year:{year_from}-{year_to}"),
                "per-page": 200, "cursor": cursor, "mailto": mailto})
            r.raise_for_status()
            return r.json()

        page = _cached_json(cache_file, fetch)
        works.extend(page["results"])
        cursor = page["meta"].get("next_cursor")
        page_i += 1
    return works
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_openalex.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/data tests/test_openalex.py
git commit -m "feat: OpenAlex client with pagination, caching, source lookup"
```

---

### Task 3: Corpus builder

**Files:**
- Create: `src/innovation/data/corpus.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: raw OpenAlex work dicts (Task 2), `reconstruct_abstract`.
- Produces: `build_corpus(works_by_venue: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]` returning `papers` (columns: `paper_id, title, abstract, year, venue`) and `edges` (columns: `src, dst`, src cites dst, both endpoints in-corpus); `save_corpus(papers, edges, out_dir)` / `load_corpus(out_dir)` writing `papers.parquet` + `edges.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_corpus.py
import pandas as pd

from innovation.data.corpus import build_corpus, load_corpus, save_corpus


def make_work(wid, year, refs=(), abstract=True):
    return {
        "id": f"https://openalex.org/{wid}",
        "title": f"Paper {wid}",
        "publication_year": year,
        "abstract_inverted_index": {"hello": [0], "world": [1]} if abstract else None,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
    }


def test_build_corpus_filters_and_keeps_within_corpus_edges():
    works = {
        "NeurIPS": [make_work("W1", 2020),
                    make_work("W2", 2021, refs=["W1", "W999"]),  # W999 external
                    make_work("W3", 2022, abstract=False)],      # dropped: no abstract
        "ICLR": [make_work("W4", 2021, refs=["W1", "W3"])],      # W3 dropped upstream
    }
    papers, edges = build_corpus(works)
    assert set(papers["paper_id"]) == {"W1", "W2", "W4"}
    assert papers.set_index("paper_id").loc["W2", "venue"] == "NeurIPS"
    assert papers.set_index("paper_id").loc["W1", "abstract"] == "hello world"
    got = {(r.src, r.dst) for r in edges.itertuples()}
    assert got == {("W2", "W1"), ("W4", "W1")}


def test_build_corpus_dedupes_papers_across_venues():
    works = {"A": [make_work("W1", 2020)], "B": [make_work("W1", 2020)]}
    papers, _ = build_corpus(works)
    assert len(papers) == 1


def test_save_and_load_roundtrip(tmp_path):
    works = {"A": [make_work("W1", 2020), make_work("W2", 2021, refs=["W1"])]}
    papers, edges = build_corpus(works)
    save_corpus(papers, edges, tmp_path)
    p2, e2 = load_corpus(tmp_path)
    pd.testing.assert_frame_equal(papers.reset_index(drop=True), p2)
    pd.testing.assert_frame_equal(edges.reset_index(drop=True), e2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ImportError` (no `innovation.data.corpus`)

- [ ] **Step 3: Implement `src/innovation/data/corpus.py`**

```python
"""Turn raw OpenAlex works into the canonical papers/edges tables (spec §3.1)."""
from pathlib import Path

import pandas as pd

from innovation.data.openalex import reconstruct_abstract


def _short_id(openalex_url: str) -> str:
    return openalex_url.rsplit("/", 1)[-1]


def build_corpus(works_by_venue: dict[str, list[dict]]):
    rows, refs_by_paper = [], {}
    seen: set[str] = set()
    for venue, works in works_by_venue.items():
        for w in works:
            pid = _short_id(w["id"])
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            if not abstract or pid in seen:
                continue
            seen.add(pid)
            rows.append({"paper_id": pid, "title": w.get("title") or "",
                         "abstract": abstract, "year": w["publication_year"],
                         "venue": venue})
            refs_by_paper[pid] = [_short_id(r) for r in w.get("referenced_works", [])]
    papers = pd.DataFrame(rows, columns=["paper_id", "title", "abstract", "year", "venue"])
    in_corpus = set(papers["paper_id"])
    edge_rows = [{"src": pid, "dst": ref}
                 for pid, refs in refs_by_paper.items()
                 for ref in refs if ref in in_corpus]
    edges = pd.DataFrame(edge_rows, columns=["src", "dst"])
    return papers, edges


def save_corpus(papers: pd.DataFrame, edges: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(out / "papers.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)


def load_corpus(out_dir):
    out = Path(out_dir)
    return (pd.read_parquet(out / "papers.parquet"),
            pd.read_parquet(out / "edges.parquet"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/data/corpus.py tests/test_corpus.py
git commit -m "feat: corpus builder with within-corpus citation edges"
```

---

### Task 4: Idea summarization

**Files:**
- Create: `src/innovation/ideas/__init__.py`, `src/innovation/ideas/summarize.py`
- Test: `tests/test_summarize.py`

**Interfaces:**
- Consumes: `LLM` (Task 1), `papers` DataFrame (Task 3).
- Produces: constants `SUMMARY_SYSTEM: str`, `SUMMARY_TEMPLATE: str`; `summarize_paper(llm, *, model: str, title: str, abstract: str) -> str`; `summarize_corpus(llm, papers: pd.DataFrame, *, model: str) -> pd.DataFrame` with columns `paper_id, idea_text, year, venue`; `save_ideas(ideas, out_dir)` / `load_ideas(out_dir)` → `ideas.parquet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_summarize.py
from innovation.ideas.summarize import (load_ideas, save_ideas,
                                        summarize_corpus, summarize_paper)
from innovation.llm import FakeLLM
import pandas as pd


def papers_df():
    return pd.DataFrame([
        {"paper_id": "W1", "title": "T1", "abstract": "A1", "year": 2020, "venue": "NeurIPS"},
        {"paper_id": "W2", "title": "T2", "abstract": "A2", "year": 2021, "venue": "ICLR"},
    ])


def test_summarize_paper_fills_template():
    llm = FakeLLM(responses=["An idea."])
    out = summarize_paper(llm, model="m", title="Attention", abstract="We propose...")
    assert out == "An idea."
    assert "Attention" in llm.calls[0]["user"]
    assert "We propose..." in llm.calls[0]["user"]


def test_summarize_corpus_and_roundtrip(tmp_path):
    llm = FakeLLM(responses=["Idea one.", "Idea two."])
    ideas = summarize_corpus(llm, papers_df(), model="m")
    assert list(ideas.columns) == ["paper_id", "idea_text", "year", "venue"]
    assert list(ideas["idea_text"]) == ["Idea one.", "Idea two."]
    save_ideas(ideas, tmp_path)
    pd.testing.assert_frame_equal(load_ideas(tmp_path), ideas)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/ideas/summarize.py`** (and empty `__init__.py`)

```python
"""Paper -> one-paragraph idea (spec §3.2). Fixed template; caching lives in CachedLLM."""
from pathlib import Path

import pandas as pd

from innovation.llm import LLM

SUMMARY_SYSTEM = (
    "You summarize research papers into a single self-contained idea paragraph.")

SUMMARY_TEMPLATE = """Summarize the following paper as ONE paragraph of 3-4 sentences \
covering: (1) the problem it addresses, (2) the key insight, (3) the method. \
Write only the paragraph, no preamble.

Title: {title}

Abstract: {abstract}"""


def summarize_paper(llm: LLM, *, model: str, title: str, abstract: str) -> str:
    user = SUMMARY_TEMPLATE.format(title=title, abstract=abstract)
    return llm.complete(model=model, system=SUMMARY_SYSTEM, user=user, max_tokens=400).strip()


def summarize_corpus(llm: LLM, papers: pd.DataFrame, *, model: str) -> pd.DataFrame:
    rows = []
    for p in papers.itertuples():
        idea = summarize_paper(llm, model=model, title=p.title, abstract=p.abstract)
        rows.append({"paper_id": p.paper_id, "idea_text": idea,
                     "year": p.year, "venue": p.venue})
    return pd.DataFrame(rows, columns=["paper_id", "idea_text", "year", "venue"])


def save_ideas(ideas: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ideas.to_parquet(out / "ideas.parquet", index=False)


def load_ideas(out_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / "ideas.parquet")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_summarize.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/ideas tests/test_summarize.py
git commit -m "feat: idea summarization with fixed template"
```

---

### Task 5: Embedder + vector index

**Files:**
- Create: `src/innovation/ideas/embed.py`, `src/innovation/network/__init__.py`, `src/innovation/network/index.py`
- Test: `tests/test_embed_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Embedder(model_name="BAAI/bge-small-en-v1.5")` with `.dim: int` and `.encode(texts: list[str]) -> np.ndarray` (float32, L2-normalized rows); `FakeEmbedder()` (same interface, `dim=8`, deterministic hash-based vectors, no downloads); `VectorIndex(dim)` with `.add(ids: list[str], vecs: np.ndarray)` and `.search(vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]` (cosine, descending); `save_embeddings(ids, vecs, out_dir)` / `load_embeddings(out_dir) -> tuple[list[str], np.ndarray]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_embed_index.py
import numpy as np
import pytest

from innovation.ideas.embed import (FakeEmbedder, load_embeddings,
                                    save_embeddings)
from innovation.network.index import VectorIndex


def test_fake_embedder_is_deterministic_and_normalized():
    e = FakeEmbedder()
    v1 = e.encode(["hello", "world"])
    v2 = e.encode(["hello", "world"])
    assert v1.shape == (2, e.dim)
    assert v1.dtype == np.float32
    np.testing.assert_allclose(v1, v2)
    np.testing.assert_allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(v1[0], v1[1])


def test_vector_index_returns_nearest_first():
    e = FakeEmbedder()
    vecs = e.encode(["alpha", "beta", "gamma"])
    idx = VectorIndex(e.dim)
    idx.add(["a", "b", "g"], vecs)
    hits = idx.search(vecs[0], k=2)
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 2


def test_embeddings_roundtrip(tmp_path):
    e = FakeEmbedder()
    vecs = e.encode(["x", "y"])
    save_embeddings(["x", "y"], vecs, tmp_path)
    ids, loaded = load_embeddings(tmp_path)
    assert ids == ["x", "y"]
    np.testing.assert_allclose(loaded, vecs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embed_index.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/ideas/embed.py`**

```python
"""Embeddings: real sentence-transformers model + a deterministic test fake."""
import hashlib
import json
from pathlib import Path

import numpy as np


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


class FakeEmbedder:
    """Deterministic vectors from sha256(text); unit tests never download models."""

    dim = 8

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            digest = hashlib.sha256(t.encode()).digest()
            v = np.frombuffer(digest[: self.dim * 4], dtype=np.uint32).astype(np.float32)
            out[i] = v / np.linalg.norm(v)
        return out


def save_embeddings(ids: list[str], vecs: np.ndarray, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    (out / "embedding_ids.json").write_text(json.dumps(ids))


def load_embeddings(out_dir):
    out = Path(out_dir)
    ids = json.loads((out / "embedding_ids.json").read_text())
    return ids, np.load(out / "embeddings.npy")
```

- [ ] **Step 4: Implement `src/innovation/network/index.py`** (and empty `__init__.py`)

```python
"""Brute-force cosine index behind a FAISS-swappable interface (plan: Global Constraints)."""
import numpy as np


class VectorIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.vecs = np.zeros((0, dim), dtype=np.float32)

    def add(self, ids: list[str], vecs: np.ndarray) -> None:
        assert vecs.shape == (len(ids), self.dim)
        self.ids.extend(ids)
        self.vecs = np.vstack([self.vecs, vecs.astype(np.float32)])

    def search(self, vec: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        scores = self.vecs @ vec.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in top]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_embed_index.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add src/innovation/ideas/embed.py src/innovation/network tests/test_embed_index.py
git commit -m "feat: embedder (real + fake) and cosine vector index"
```

---

### Task 6: IdeaGraph

**Files:**
- Create: `src/innovation/network/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `ideas` DataFrame (Task 4), `edges` DataFrame (Task 3).
- Produces: `IdeaNode` dataclass (`node_id, text, year, source, meta`); `IdeaGraph` with `from_tables(ideas: pd.DataFrame, edges: pd.DataFrame) -> IdeaGraph` (classmethod; corpus nodes get `source="corpus"`), `node(node_id) -> IdeaNode`, `has_node(node_id) -> bool`, `add_idea(node_id, text, cited_ids, *, source="generated", year=None, meta=None)`, `citations_out(node_id) -> list[str]`, `citations_in(node_id) -> list[str]`, `node_ids(source=None) -> list[str]`, `in_degree(node_id) -> int`, `network_at(year) -> IdeaGraph` (corpus nodes with `year <= year` only), `communities() -> dict[str, int]` (Louvain, seeded), `num_nodes`/`num_edges` properties.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_graph.py
import pandas as pd
import pytest

from innovation.network.graph import IdeaGraph


def small_graph():
    ideas = pd.DataFrame([
        {"paper_id": "W1", "idea_text": "i1", "year": 2019, "venue": "V"},
        {"paper_id": "W2", "idea_text": "i2", "year": 2020, "venue": "V"},
        {"paper_id": "W3", "idea_text": "i3", "year": 2021, "venue": "V"},
    ])
    edges = pd.DataFrame([{"src": "W2", "dst": "W1"}, {"src": "W3", "dst": "W2"}])
    return IdeaGraph.from_tables(ideas, edges)


def test_from_tables_builds_nodes_and_citations():
    g = small_graph()
    assert g.num_nodes == 3 and g.num_edges == 2
    assert g.node("W1").text == "i1"
    assert g.node("W1").source == "corpus"
    assert g.citations_out("W2") == ["W1"]
    assert g.citations_in("W2") == ["W3"]
    assert g.in_degree("W1") == 1


def test_add_idea_appends_generated_node_with_provenance():
    g = small_graph()
    g.add_idea("gen:r1:0", "new idea", ["W1", "W3"],
               meta={"run_id": "r1", "agent_id": "a0", "step": 4})
    assert g.num_nodes == 4
    assert set(g.citations_out("gen:r1:0")) == {"W1", "W3"}
    assert g.node("gen:r1:0").source == "generated"
    assert g.node("gen:r1:0").meta["agent_id"] == "a0"
    assert g.node_ids(source="generated") == ["gen:r1:0"]
    with pytest.raises(KeyError):
        g.add_idea("gen:r1:1", "bad", ["W_missing"])


def test_network_at_slices_by_year():
    g = small_graph()
    g2020 = g.network_at(2020)
    assert set(g2020.node_ids()) == {"W1", "W2"}
    assert g2020.num_edges == 1


def test_communities_cover_all_nodes():
    g = small_graph()
    comm = g.communities()
    assert set(comm) == {"W1", "W2", "W3"}
    assert all(isinstance(c, int) for c in comm.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/network/graph.py`**

```python
"""IdeaGraph: networkx behind our own interface so the backend can swap (spec §3.3)."""
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd


@dataclass
class IdeaNode:
    node_id: str
    text: str
    year: int | None
    source: str  # "corpus" | "generated"
    meta: dict = field(default_factory=dict)


class IdeaGraph:
    def __init__(self):
        self._g = nx.DiGraph()  # edge src -> dst means "src cites dst"

    @classmethod
    def from_tables(cls, ideas: pd.DataFrame, edges: pd.DataFrame) -> "IdeaGraph":
        g = cls()
        for r in ideas.itertuples():
            g._g.add_node(r.paper_id, data=IdeaNode(
                node_id=r.paper_id, text=r.idea_text, year=int(r.year),
                source="corpus", meta={"venue": r.venue}))
        for r in edges.itertuples():
            g._g.add_edge(r.src, r.dst)
        return g

    # --- reads ---
    def node(self, node_id: str) -> IdeaNode:
        return self._g.nodes[node_id]["data"]

    def has_node(self, node_id: str) -> bool:
        return self._g.has_node(node_id)

    def node_ids(self, source: str | None = None) -> list[str]:
        if source is None:
            return list(self._g.nodes)
        return [n for n in self._g.nodes if self._g.nodes[n]["data"].source == source]

    def citations_out(self, node_id: str) -> list[str]:
        return list(self._g.successors(node_id))

    def citations_in(self, node_id: str) -> list[str]:
        return list(self._g.predecessors(node_id))

    def in_degree(self, node_id: str) -> int:
        return self._g.in_degree(node_id)

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    # --- writes ---
    def add_idea(self, node_id: str, text: str, cited_ids: list[str], *,
                 source: str = "generated", year: int | None = None,
                 meta: dict | None = None) -> None:
        missing = [c for c in cited_ids if not self._g.has_node(c)]
        if missing:
            raise KeyError(f"cited ids not in graph: {missing}")
        self._g.add_node(node_id, data=IdeaNode(
            node_id=node_id, text=text, year=year, source=source, meta=meta or {}))
        for c in cited_ids:
            self._g.add_edge(node_id, c)

    # --- analysis ---
    def network_at(self, year: int) -> "IdeaGraph":
        keep = [n for n in self._g.nodes
                if self._g.nodes[n]["data"].source == "corpus"
                and self._g.nodes[n]["data"].year is not None
                and self._g.nodes[n]["data"].year <= year]
        sub = IdeaGraph()
        sub._g = self._g.subgraph(keep).copy()
        return sub

    def communities(self, seed: int = 0) -> dict[str, int]:
        parts = nx.community.louvain_communities(self._g.to_undirected(), seed=seed)
        return {n: i for i, part in enumerate(parts) for n in part}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_graph.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/network/graph.py tests/test_graph.py
git commit -m "feat: IdeaGraph with provenance, time slices, communities"
```

---

### Task 7: Event log

**Files:**
- Create: `src/innovation/experiments/__init__.py`, `src/innovation/experiments/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `EventLog(path: Path)` with `.append(event: dict) -> dict` (adds monotonically increasing `"seq"`, writes one JSON line, returns the enriched event) and `.read_all() -> list[dict]`; module function `load_events(path) -> list[dict]`. Event schema used by all later tasks: `{"seq": int, "run_id": str, "agent_id": str, "step": int, "action": str, "args": dict, "result": dict}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events.py
from innovation.experiments.events import EventLog, load_events


def test_append_assigns_seq_and_persists(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    e1 = log.append({"run_id": "r", "agent_id": "a0", "step": 0,
                     "action": "search", "args": {"query": "q"}, "result": {}})
    e2 = log.append({"run_id": "r", "agent_id": "a1", "step": 1,
                     "action": "generate", "args": {}, "result": {"node_id": "gen:r:0"}})
    assert (e1["seq"], e2["seq"]) == (0, 1)
    assert [e["action"] for e in load_events(path)] == ["search", "generate"]


def test_event_log_resumes_seq_from_existing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLog(path).append({"run_id": "r", "agent_id": "a", "step": 0,
                           "action": "search", "args": {}, "result": {}})
    log2 = EventLog(path)  # reopen, e.g. after a crash
    e = log2.append({"run_id": "r", "agent_id": "a", "step": 1,
                     "action": "search", "args": {}, "result": {}})
    assert e["seq"] == 1
    assert len(log2.read_all()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/experiments/events.py`** (and empty `__init__.py`)

```python
"""Append-only JSONL event log; the action trace is primary research data (spec §3.5)."""
import json
from pathlib import Path


def load_events(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


class EventLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = len(load_events(self.path))

    def append(self, event: dict) -> dict:
        enriched = {"seq": self._seq, **event}
        with self.path.open("a") as f:
            f.write(json.dumps(enriched) + "\n")
        self._seq += 1
        return enriched

    def read_all(self) -> list[dict]:
        return load_events(self.path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_events.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/experiments tests/test_events.py
git commit -m "feat: append-only JSONL event log with resumable seq"
```

---

### Task 8: Environment (actions on the living network)

**Files:**
- Create: `src/innovation/experiments/env.py`
- Test: `tests/test_env.py`

**Interfaces:**
- Consumes: `IdeaGraph` (Task 6), `VectorIndex` (Task 5), an embedder (Task 5), `EventLog` (Task 7).
- Produces: `Action` dataclass (`name: str, args: dict`); `Environment(run_id, graph, index, embedder, event_log, rng, generation_budget: int)` with `execute(agent_id: str, step: int, action: Action) -> dict` dispatching to the four spec actions — `search(query, k=5)`, `browse(node_id)`, `sample_frontier()`, `generate(text, cited_ids)` — logging every executed action as one event; `restore(events: list[dict])` re-applies generate events without logging (replay/resume, spec §3.3); `generated_ids() -> list[str]`. Result shapes: search → `{"hits": [{"node_id", "text", "score"}]}`; browse → `{"node_id", "text", "cites": [preview], "cited_by": [preview]}` (preview = `{"node_id", "text"}` truncated to 200 chars); sample_frontier → `{"node_id", "text"}`; generate → `{"node_id": "gen:<run_id>:<n>"}` or `{"error": ...}` when budget is exhausted / a cited id is unknown / an unknown action name arrives.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_env.py
import numpy as np
import pandas as pd

from innovation.experiments.env import Action, Environment
from innovation.experiments.events import EventLog
from innovation.ideas.embed import FakeEmbedder
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def make_env(tmp_path, budget=5):
    ideas = pd.DataFrame([
        {"paper_id": "W1", "idea_text": "deep nets", "year": 2019, "venue": "V"},
        {"paper_id": "W2", "idea_text": "graph nets", "year": 2020, "venue": "V"},
    ])
    edges = pd.DataFrame([{"src": "W2", "dst": "W1"}])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    index.add(ids, emb.encode([graph.node(n).text for n in ids]))
    log = EventLog(tmp_path / "events.jsonl")
    return Environment(run_id="r1", graph=graph, index=index, embedder=emb,
                       event_log=log, rng=np.random.default_rng(0),
                       generation_budget=budget)


def test_search_browse_sample_and_logging(tmp_path):
    env = make_env(tmp_path)
    res = env.execute("a0", 0, Action("search", {"query": "deep nets", "k": 1}))
    assert res["hits"][0]["node_id"] == "W1"
    res = env.execute("a0", 1, Action("browse", {"node_id": "W2"}))
    assert res["cites"][0]["node_id"] == "W1"
    res = env.execute("a0", 2, Action("sample_frontier", {}))
    assert res["node_id"] in {"W1", "W2"}
    events = env.event_log.read_all()
    assert [e["action"] for e in events] == ["search", "browse", "sample_frontier"]
    assert events[0]["agent_id"] == "a0" and events[1]["step"] == 1


def test_generate_adds_node_indexes_it_and_decrements_budget(tmp_path):
    env = make_env(tmp_path, budget=1)
    res = env.execute("a0", 0, Action("generate",
                                      {"text": "combine deep and graph nets",
                                       "cited_ids": ["W1", "W2"]}))
    assert res["node_id"] == "gen:r1:0"
    assert env.graph.node("gen:r1:0").meta["agent_id"] == "a0"
    # The new idea is immediately searchable (stigmergy channel, spec §3.4).
    hits = env.execute("a0", 1, Action("search", {"query": "combine deep and graph nets", "k": 1}))
    assert hits["hits"][0]["node_id"] == "gen:r1:0"
    # Budget exhausted -> error, no node added.
    res2 = env.execute("a0", 2, Action("generate", {"text": "x", "cited_ids": ["W1"]}))
    assert "error" in res2
    assert env.generated_ids() == ["gen:r1:0"]


def test_generate_rejects_unknown_citation_and_unknown_action(tmp_path):
    env = make_env(tmp_path)
    res = env.execute("a0", 0, Action("generate", {"text": "x", "cited_ids": ["nope"]}))
    assert "error" in res
    res = env.execute("a0", 1, Action("fly_to_moon", {}))
    assert "error" in res


def test_restore_replays_generate_events_without_relogging(tmp_path):
    env = make_env(tmp_path)
    env.execute("a0", 0, Action("generate", {"text": "new", "cited_ids": ["W1"]}))
    events = env.event_log.read_all()

    env2 = make_env(tmp_path / "fresh")
    env2.restore(events)
    assert env2.graph.has_node("gen:r1:0")
    assert env2.event_log.read_all() == []          # restore does not log
    res = env2.execute("a0", 1, Action("generate", {"text": "next", "cited_ids": ["W1"]}))
    assert res["node_id"] == "gen:r1:1"             # counter continues
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_env.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/experiments/env.py`**

```python
"""The simulation environment: executes agent actions on the living idea network (spec §3.4)."""
from dataclasses import dataclass, field


def _preview(graph, node_id: str) -> dict:
    return {"node_id": node_id, "text": graph.node(node_id).text[:200]}


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


class Environment:
    def __init__(self, *, run_id, graph, index, embedder, event_log, rng,
                 generation_budget: int):
        self.run_id = run_id
        self.graph = graph
        self.index = index
        self.embedder = embedder
        self.event_log = event_log
        self.rng = rng
        self.generation_budget = generation_budget
        self._gen_counter = 0

    # --- public entry point: execute + log ---
    def execute(self, agent_id: str, step: int, action: Action) -> dict:
        handler = getattr(self, f"_do_{action.name}", None)
        if handler is None:
            result = {"error": f"unknown action: {action.name}"}
        else:
            try:
                result = handler(agent_id=agent_id, step=step, **action.args)
            except (KeyError, TypeError) as exc:
                result = {"error": str(exc)}
        self.event_log.append({"run_id": self.run_id, "agent_id": agent_id,
                               "step": step, "action": action.name,
                               "args": action.args, "result": result})
        return result

    # --- actions (spec §3.4) ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        vec = self.embedder.encode([query])[0]
        hits = [{"node_id": nid, "text": self.graph.node(nid).text[:300],
                 "score": score} for nid, score in self.index.search(vec, k=k)]
        return {"hits": hits}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        node = self.graph.node(node_id)  # KeyError -> error result
        return {"node_id": node_id, "text": node.text,
                "cites": [_preview(self.graph, n) for n in self.graph.citations_out(node_id)[:10]],
                "cited_by": [_preview(self.graph, n) for n in self.graph.citations_in(node_id)[:10]]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        node_id = str(self.rng.choice(self.graph.node_ids()))
        return _preview(self.graph, node_id) | {"text": self.graph.node(node_id).text}

    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        node_id = self._apply_generate(text, cited_ids,
                                       meta={"run_id": self.run_id,
                                             "agent_id": agent_id, "step": step})
        self.generation_budget -= 1
        return {"node_id": node_id}

    # --- shared by generate and restore ---
    def _apply_generate(self, text: str, cited_ids: list[str], meta: dict) -> str:
        node_id = f"gen:{self.run_id}:{self._gen_counter}"
        self.graph.add_idea(node_id, text, cited_ids, meta=meta)  # KeyError propagates
        self.index.add([node_id], self.embedder.encode([text]))
        self._gen_counter += 1
        return node_id

    def restore(self, events: list[dict]) -> None:
        """Replay generate events to rebuild state (spec §3.3); no logging."""
        for e in events:
            if e["action"] == "generate" and "node_id" in e.get("result", {}):
                self._apply_generate(e["args"]["text"], e["args"]["cited_ids"],
                                     meta={"run_id": e["run_id"],
                                           "agent_id": e["agent_id"],
                                           "step": e["step"]})
                self.generation_budget -= 1

    def generated_ids(self) -> list[str]:
        return self.graph.node_ids(source="generated")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_env.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/experiments/env.py tests/test_env.py
git commit -m "feat: environment with search/browse/sample/generate + replay"
```

---

### Task 9: Policy interface + baseline policies

**Files:**
- Create: `src/innovation/agents/__init__.py`, `src/innovation/agents/policy.py`, `src/innovation/agents/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `Action` (Task 8), `IdeaGraph` (Task 6), `LLM` (Task 1).
- Produces: `Policy` ABC with `act(obs: dict) -> Action`; `obs` is `{"step": int, "last_result": dict}` (empty dict on the first step). `NoNavLLMPolicy(llm, model, graph, rng, k=3)` — spec §3.4 baseline 2: each `act` samples k random **corpus** ideas, asks the LLM for a new idea citing them. `PreferentialAttachmentPolicy(graph, rng, m=3)` — spec §3.4 baseline 3: no LLM; picks m distinct nodes with probability ∝ in_degree+1, emits a template-text generate. Constant `NONAV_TEMPLATE` used by `NoNavLLMPolicy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
import pandas as pd

from innovation.agents.baselines import (NoNavLLMPolicy,
                                         PreferentialAttachmentPolicy)
from innovation.llm import FakeLLM
from innovation.network.graph import IdeaGraph


def graph_with_hub():
    ideas = pd.DataFrame(
        [{"paper_id": f"W{i}", "idea_text": f"idea {i}", "year": 2020, "venue": "V"}
         for i in range(5)])
    # W0 is a hub: cited by everyone else.
    edges = pd.DataFrame([{"src": f"W{i}", "dst": "W0"} for i in range(1, 5)])
    return IdeaGraph.from_tables(ideas, edges)


def test_nonav_policy_generates_citing_sampled_corpus_ideas():
    g = graph_with_hub()
    llm = FakeLLM(responses=["A brand new idea."])
    pol = NoNavLLMPolicy(llm=llm, model="m", graph=g,
                         rng=np.random.default_rng(0), k=3)
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "generate"
    assert action.args["text"] == "A brand new idea."
    assert len(action.args["cited_ids"]) == 3
    assert all(cid.startswith("W") for cid in action.args["cited_ids"])
    # The sampled idea texts were shown to the LLM.
    assert "idea" in llm.calls[0]["user"]


def test_pa_policy_prefers_high_in_degree_nodes():
    g = graph_with_hub()
    pol = PreferentialAttachmentPolicy(graph=g, rng=np.random.default_rng(0), m=2)
    hub_hits = 0
    for _ in range(50):
        action = pol.act({"step": 0, "last_result": {}})
        assert action.name == "generate"
        assert len(set(action.args["cited_ids"])) == 2
        if "W0" in action.args["cited_ids"]:
            hub_hits += 1
    assert hub_hits > 40  # in_degree(W0)=4 vs 0 elsewhere -> nearly always picked
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `policy.py` and `baselines.py`** (and empty `__init__.py`)

```python
# src/innovation/agents/policy.py
"""Policy ABC: the LLM agent is one policy among several (spec §3.4)."""
from abc import ABC, abstractmethod

from innovation.experiments.env import Action


class Policy(ABC):
    @abstractmethod
    def act(self, obs: dict) -> Action:
        """obs = {"step": int, "last_result": dict}; returns the next Action."""
```

```python
# src/innovation/agents/baselines.py
"""Baseline policies: isolate the contributions of navigation and of the LLM (spec §3.4)."""
import numpy as np

from innovation.experiments.env import Action
from innovation.agents.policy import Policy
from innovation.llm import LLM

NONAV_TEMPLATE = """Here are {k} ideas from the research literature:

{ideas}

Propose ONE new research idea that builds on them. Write a single 3-4 sentence \
paragraph covering the problem, the key insight, and the method. Write only the paragraph."""


class NoNavLLMPolicy(Policy):
    """Baseline 2: LLM without navigation — random sample of corpus ideas, generate."""

    def __init__(self, *, llm: LLM, model: str, graph, rng: np.random.Generator, k: int = 3):
        self.llm, self.model, self.graph, self.rng, self.k = llm, model, graph, rng, k

    def act(self, obs: dict) -> Action:
        corpus = self.graph.node_ids(source="corpus")
        cited = [str(c) for c in self.rng.choice(corpus, size=self.k, replace=False)]
        ideas = "\n\n".join(f"- {self.graph.node(c).text}" for c in cited)
        text = self.llm.complete(
            model=self.model, system="You propose new research ideas.",
            user=NONAV_TEMPLATE.format(k=self.k, ideas=ideas), max_tokens=400).strip()
        return Action("generate", {"text": text, "cited_ids": cited})


class PreferentialAttachmentPolicy(Policy):
    """Baseline 3: structural null model — no LLM, cites ∝ in-degree, template text."""

    def __init__(self, *, graph, rng: np.random.Generator, m: int = 3):
        self.graph, self.rng, self.m = graph, rng, m

    def act(self, obs: dict) -> Action:
        nodes = self.graph.node_ids()
        weights = np.array([self.graph.in_degree(n) + 1 for n in nodes], dtype=float)
        weights /= weights.sum()
        cited = [str(c) for c in self.rng.choice(nodes, size=self.m,
                                                 replace=False, p=weights)]
        text = "A recombination of the ideas in: " + ", ".join(cited) + "."
        return Action("generate", {"text": text, "cited_ids": cited})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/agents tests/test_baselines.py
git commit -m "feat: Policy ABC + no-navigation LLM and preferential-attachment baselines"
```

---

### Task 10: LLM agent policy

**Files:**
- Create: `src/innovation/agents/llm_agent.py`
- Test: `tests/test_llm_agent.py`

**Interfaces:**
- Consumes: `Policy`, `Action`, `LLM`.
- Produces: `LLMAgentPolicy(llm, model, memory_size=6, persona="")` — the full navigating agent (spec §3.4 policy 1). Constants `AGENT_SYSTEM`, `ACTIONS_DOC`. Behavior: keeps a rolling window of the last `memory_size` (action, result) pairs; renders them plus the last result into the user prompt; parses the model's JSON reply into an `Action`; any unparsable/invalid reply falls back to `Action("sample_frontier", {})`. `persona` (non-empty for the heterogeneity ablation) is appended to the system prompt.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm_agent.py
import json

from innovation.agents.llm_agent import LLMAgentPolicy
from innovation.llm import FakeLLM


def test_parses_json_action_from_reply():
    reply = 'Thinking... {"action": "search", "args": {"query": "sparse attention", "k": 3}}'
    pol = LLMAgentPolicy(llm=FakeLLM(responses=[reply]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "search"
    assert action.args == {"query": "sparse attention", "k": 3}


def test_falls_back_to_sample_frontier_on_garbage():
    pol = LLMAgentPolicy(llm=FakeLLM(responses=["I refuse to answer with JSON"]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "sample_frontier"
    pol2 = LLMAgentPolicy(llm=FakeLLM(
        responses=[json.dumps({"action": "hack_the_planet", "args": {}})]), model="m")
    assert pol2.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_memory_window_appears_in_prompt_and_is_bounded():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m", memory_size=2)
    pol.act({"step": 0, "last_result": {}})
    pol.act({"step": 1, "last_result": {"node_id": "W7", "text": "old idea"}})
    pol.act({"step": 2, "last_result": {"node_id": "W8", "text": "newer idea"}})
    prompt = llm.calls[-1]["user"]
    assert "W8" in prompt
    # memory_size=2: the step-0 empty result is beyond the window by call 3
    assert len(pol.memory) == 2


def test_persona_is_added_to_system_prompt():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m", persona="You are a risk-taking theorist.")
    pol.act({"step": 0, "last_result": {}})
    assert "risk-taking theorist" in llm.calls[0]["system"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/agents/llm_agent.py`**

```python
"""The navigating LLM agent: JSON tool-calling loop over the action space (spec §3.4)."""
import json

from innovation.agents.policy import Policy
from innovation.experiments.env import Action
from innovation.llm import LLM

VALID_ACTIONS = {"search", "browse", "sample_frontier", "generate"}

AGENT_SYSTEM = """You are a research agent exploring a network of research ideas \
distilled from published papers. Ideas cite the ideas they build on. Your goal is to \
find promising unexplored directions and, when you see one, contribute a genuinely \
new idea to the network. Prefer exploring (search, browse) until you understand a \
neighborhood well enough that your new idea is specific and well-grounded."""

ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- semantic search over all ideas
{"action": "browse", "args": {"node_id": "<id>"}} -- read an idea and its citation neighbors
{"action": "sample_frontier", "args": {}} -- jump to a random idea
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- add your new idea, citing the existing ideas it builds on"""


class LLMAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, memory_size: int = 6, persona: str = ""):
        self.llm = llm
        self.model = model
        self.memory_size = memory_size
        self.system = AGENT_SYSTEM + ("\n\n" + persona if persona else "")
        self.memory: list[str] = []  # rendered "(action -> result)" lines
        self._last_action: str = "(none)"

    def act(self, obs: dict) -> Action:
        result_snippet = json.dumps(obs.get("last_result", {}))[:1500]
        self.memory.append(f"step {obs['step'] - 1}: {self._last_action} -> {result_snippet}")
        self.memory = self.memory[-self.memory_size:]
        user = (ACTIONS_DOC + "\n\nRecent history:\n" + "\n".join(self.memory)
                + "\n\nChoose your next action (JSON only):")
        reply = self.llm.complete(model=self.model, system=self.system,
                                  user=user, max_tokens=600)
        action = self._parse(reply)
        self._last_action = action.name
        return action

    @staticmethod
    def _parse(reply: str) -> Action:
        start, end = reply.find("{"), reply.rfind("}")
        if start == -1 or end <= start:
            return Action("sample_frontier", {})
        try:
            obj = json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            return Action("sample_frontier", {})
        name = obj.get("action")
        if name not in VALID_ACTIONS or not isinstance(obj.get("args", {}), dict):
            return Action("sample_frontier", {})
        return Action(name, obj.get("args", {}))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_agent.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/agents/llm_agent.py tests/test_llm_agent.py
git commit -m "feat: navigating LLM agent policy with JSON action loop"
```

---

### Task 11: Simulation runner

**Files:**
- Create: `src/innovation/experiments/runner.py`
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: `Environment`, `EventLog`, policies (Tasks 8–10), `IdeaGraph`, `VectorIndex`, embedder.
- Produces: `RunConfig` dataclass (`run_id: str, seed: int, total_steps: int, generation_budget: int, agents: list[dict]`) — each agent dict is `{"agent_id": str, "policy": str, ...policy kwargs}` with `policy` ∈ `{"llm", "nonav", "pa"}`; `build_policy(spec: dict, *, llm, model, graph, rng) -> Policy`; `run_simulation(cfg: RunConfig, *, graph, index, embedder, llm, model, out_dir) -> dict` — round-robin scheduling, per-agent last-result threading, writes `out_dir/<run_id>/events.jsonl`, returns `{"run_id", "steps", "generated": [node_ids]}`. Multi-agent stigmergy and the equal-total-generation-budget control both live here (budget is global in `Environment`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import json

import pandas as pd

from innovation.experiments.events import load_events
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import FakeEmbedder
from innovation.llm import FakeLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def fixtures():
    ideas = pd.DataFrame(
        [{"paper_id": f"W{i}", "idea_text": f"idea {i}", "year": 2020, "venue": "V"}
         for i in range(6)])
    edges = pd.DataFrame([{"src": f"W{i}", "dst": "W0"} for i in range(1, 6)])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    index.add(ids, emb.encode([graph.node(n).text for n in ids]))
    return graph, index, emb


def test_round_robin_two_llm_agents_share_budget(tmp_path):
    graph, index, emb = fixtures()
    gen = json.dumps({"action": "generate",
                      "args": {"text": "a new idea", "cited_ids": ["W0"]}})
    llm = FakeLLM(default=gen)  # every agent generates every turn
    cfg = RunConfig(run_id="r1", seed=0, total_steps=6, generation_budget=3,
                    agents=[{"agent_id": "a0", "policy": "llm"},
                            {"agent_id": "a1", "policy": "llm"}])
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=llm, model="m", out_dir=tmp_path)
    assert out["steps"] == 6
    assert len(out["generated"]) == 3  # global budget enforced across agents
    events = load_events(tmp_path / "r1" / "events.jsonl")
    assert [e["agent_id"] for e in events[:4]] == ["a0", "a1", "a0", "a1"]  # round-robin


def test_mixed_policies_and_determinism(tmp_path):
    graph, index, emb = fixtures()
    cfg = RunConfig(run_id="r2", seed=42, total_steps=4, generation_budget=4,
                    agents=[{"agent_id": "pa", "policy": "pa", "m": 2},
                            {"agent_id": "nn", "policy": "nonav", "k": 2}])
    llm = FakeLLM(default="A generated idea paragraph.")
    out1 = run_simulation(cfg, graph=fixtures()[0], index=fixtures()[1],
                          embedder=emb, llm=llm, model="m", out_dir=tmp_path / "x")
    out2 = run_simulation(cfg, graph=fixtures()[0], index=fixtures()[1],
                          embedder=emb, llm=llm, model="m", out_dir=tmp_path / "y")
    e1 = load_events(tmp_path / "x" / "r2" / "events.jsonl")
    e2 = load_events(tmp_path / "y" / "r2" / "events.jsonl")
    assert [ev["args"] for ev in e1] == [ev["args"] for ev in e2]  # same seed, same trace
    assert len(out1["generated"]) == len(out2["generated"]) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/experiments/runner.py`**

```python
"""Round-robin simulation runner (spec §3.4-3.5): stigmergy via the shared network."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from innovation.agents.baselines import (NoNavLLMPolicy,
                                         PreferentialAttachmentPolicy)
from innovation.agents.llm_agent import LLMAgentPolicy
from innovation.experiments.env import Environment
from innovation.experiments.events import EventLog


@dataclass
class RunConfig:
    run_id: str
    seed: int
    total_steps: int
    generation_budget: int  # GLOBAL: the fairness control of spec §3.4
    agents: list[dict] = field(default_factory=list)


def build_policy(spec: dict, *, llm, model, graph, rng):
    kind = spec["policy"]
    if kind == "llm":
        return LLMAgentPolicy(llm=llm, model=model,
                              memory_size=spec.get("memory_size", 6),
                              persona=spec.get("persona", ""))
    if kind == "nonav":
        return NoNavLLMPolicy(llm=llm, model=model, graph=graph, rng=rng,
                              k=spec.get("k", 3))
    if kind == "pa":
        return PreferentialAttachmentPolicy(graph=graph, rng=rng, m=spec.get("m", 3))
    raise ValueError(f"unknown policy kind: {kind}")


def run_simulation(cfg: RunConfig, *, graph, index, embedder, llm, model,
                   out_dir) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    env = Environment(run_id=cfg.run_id, graph=graph, index=index,
                      embedder=embedder, event_log=EventLog(run_dir / "events.jsonl"),
                      rng=rng, generation_budget=cfg.generation_budget)
    policies = {a["agent_id"]: build_policy(a, llm=llm, model=model,
                                            graph=graph, rng=rng)
                for a in cfg.agents}
    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}

    for step in range(cfg.total_steps):
        agent_id = order[step % len(order)]
        obs = {"step": step, "last_result": last_result[agent_id]}
        action = policies[agent_id].act(obs)
        last_result[agent_id] = env.execute(agent_id, step, action)

    return {"run_id": cfg.run_id, "steps": cfg.total_steps,
            "generated": env.generated_ids()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runner.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/experiments/runner.py tests/test_runner.py
git commit -m "feat: round-robin simulation runner with global generation budget"
```

---

### Task 12: Search-verified realization (evaluation core)

**Files:**
- Create: `src/innovation/eval/__init__.py`, `src/innovation/eval/search_verify.py`
- Test: `tests/test_search_verify.py`

**Interfaces:**
- Consumes: `LLM` (Task 1), cached HTTP pattern (Task 2 style).
- Produces: `extract_queries(llm, *, model, idea_text, n=3) -> list[str]`; `s2_search(query, *, cache_dir, http_get=None) -> list[dict]` and `openalex_search(query, *, mailto, cache_dir, http_get=None) -> list[dict]`, both returning candidates normalized to `{"paper_id", "title", "abstract", "pub_date" (ISO str or ""), "source_api"}`, responses cached with `fetched_at`; `judge_realization(llm, *, model, idea_text, candidate) -> bool` (LLM judge, YES/NO); `Verdict` dataclass (`idea_id, hit: bool, paper: dict | None, excluded_pre_cutoff: list[dict]`); `verify_idea(llm, *, model, idea_id, idea_text, cutoff_date: str, mailto, cache_dir, http_get=None, n_queries=3, top_k=5) -> Verdict` implementing the spec §3.6 pipeline with **anticipation-only hits**.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_search_verify.py
import json

from innovation.eval.search_verify import (Verdict, extract_queries,
                                           judge_realization, s2_search,
                                           verify_idea)
from innovation.llm import FakeLLM


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def s2_payload(papers):
    return {"data": [{"paperId": p["id"], "title": p["title"],
                      "abstract": p.get("abstract", ""),
                      "publicationDate": p.get("date")} for p in papers]}


def test_extract_queries_splits_lines():
    llm = FakeLLM(responses=["sparse attention transformers\nefficient long context\n"])
    qs = extract_queries(llm, model="m", idea_text="An idea about sparse attention.")
    assert qs == ["sparse attention transformers", "efficient long context"]


def test_s2_search_normalizes_and_caches(tmp_path):
    payload = s2_payload([{"id": "p1", "title": "T", "date": "2025-06-01"}])
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return FakeResponse(payload)

    hits = s2_search("q", cache_dir=tmp_path, http_get=fake_get)
    assert hits[0] == {"paper_id": "p1", "title": "T", "abstract": "",
                       "pub_date": "2025-06-01", "source_api": "s2"}
    s2_search("q", cache_dir=tmp_path, http_get=fake_get)
    assert len(calls) == 1  # cached
    cache_files = list(tmp_path.glob("*.json"))
    assert "fetched_at" in json.loads(cache_files[0].read_text())


def test_judge_realization_yes_no():
    assert judge_realization(FakeLLM(responses=["YES"]), model="m",
                             idea_text="i", candidate={"title": "t", "abstract": "a"})
    assert not judge_realization(FakeLLM(responses=["NO, unrelated"]), model="m",
                                 idea_text="i", candidate={"title": "t", "abstract": "a"})


def test_verify_idea_anticipation_only(tmp_path):
    # Two candidates realize the idea: one pre-cutoff (excluded), one post-cutoff (hit).
    payload = s2_payload([{"id": "old", "title": "Old paper", "date": "2024-01-01"},
                          {"id": "new", "title": "New paper", "date": "2025-08-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["one query"] + ["YES", "YES"])
    v = verify_idea(llm, model="m", idea_id="gen:r:0", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1, top_k=5)
    assert isinstance(v, Verdict)
    assert v.hit and v.paper["paper_id"] == "new"
    assert [p["paper_id"] for p in v.excluded_pre_cutoff] == ["old"]


def test_verify_idea_no_hit_when_only_pre_cutoff(tmp_path):
    payload = s2_payload([{"id": "old", "title": "Old", "date": "2020-01-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", "YES"])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert not v.hit and v.paper is None
    assert len(v.excluded_pre_cutoff) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_search_verify.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/eval/search_verify.py`** (and empty `__init__.py`)

```python
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
                                    "fields": "title,abstract,publicationDate"},
                          Path(cache_dir), http_get)
    return [{"paper_id": p.get("paperId", ""), "title": p.get("title") or "",
             "abstract": p.get("abstract") or "",
             "pub_date": p.get("publicationDate") or "", "source_api": "s2"}
            for p in payload.get("data", [])]


def openalex_search(query: str, *, mailto: str, cache_dir, http_get=None) -> list[dict]:
    from innovation.data.openalex import reconstruct_abstract

    http_get = http_get or requests.get
    payload = _cached_get(OPENALEX_BASE,
                          {"search": query, "per-page": 10, "mailto": mailto},
                          Path(cache_dir), http_get)
    return [{"paper_id": w.get("id", "").rsplit("/", 1)[-1],
             "title": w.get("title") or "",
             "abstract": reconstruct_abstract(w.get("abstract_inverted_index")),
             "pub_date": w.get("publication_date") or "", "source_api": "openalex"}
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


def verify_idea(llm: LLM, *, model: str, idea_id: str, idea_text: str,
                cutoff_date: str, mailto: str, cache_dir, http_get=None,
                n_queries: int = 3, top_k: int = 5) -> Verdict:
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

    hit_paper, excluded = None, []
    for cand in candidates:
        if not judge_realization(llm, model=model, idea_text=idea_text, candidate=cand):
            continue
        if cand["pub_date"] and cand["pub_date"] > cutoff_date:
            if hit_paper is None:
                hit_paper = cand  # first post-cutoff realization = the hit
        else:
            excluded.append(cand)  # logged, NEVER scored (spec §3.6)
    return Verdict(idea_id=idea_id, hit=hit_paper is not None,
                   paper=hit_paper, excluded_pre_cutoff=excluded)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_search_verify.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/eval tests/test_search_verify.py
git commit -m "feat: search-verified realization with anticipation-only hits"
```

---

### Task 13: Metrics

**Files:**
- Create: `src/innovation/eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `Verdict` (Task 12), embeddings (Task 5), `IdeaGraph` communities (Task 6).
- Produces: `precision(verdicts, dup_flags) -> float` (dup-flagged ideas excluded from the numerator per spec §3.6); `recall(verdicts, population: int) -> float` (distinct hit paper_ids / population); `openalex_population_count(filter_str: str, *, mailto, http_get=None) -> int` (single `/works?per-page=1` count query — the recall denominator); `venue_population_filter(source_ids: list[str], from_date: str) -> str` and `arxiv_population_filter(arxiv_source_id: str, from_date: str, min_citations: int) -> str`; `novelty(vec, corpus_vecs) -> float` (1 − max cosine); `past_dup_flag(vec, corpus_vecs, ceiling=0.95) -> bool`; `bridging(cited_ids, communities) -> int` (distinct communities cited); `diversity(vecs) -> float` (mean pairwise 1 − cosine); `aggregate_run(verdicts, dup_flags, population) -> dict` with keys `precision, recall, n_ideas, n_hits, n_dup_flagged`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
import pytest

from innovation.eval.metrics import (aggregate_run, arxiv_population_filter,
                                     bridging, diversity, novelty,
                                     openalex_population_count, past_dup_flag,
                                     precision, recall,
                                     venue_population_filter)
from innovation.eval.search_verify import Verdict


def verdicts():
    return [Verdict("g0", True, {"paper_id": "P1"}),
            Verdict("g1", True, {"paper_id": "P1"}),   # same paper twice
            Verdict("g2", False, None),
            Verdict("g3", True, {"paper_id": "P2"})]


def test_precision_excludes_dup_flagged_from_numerator():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    # hits g0,g1 count; g3 is a hit but dup-flagged -> excluded; 2/4
    assert precision(verdicts(), dup) == pytest.approx(0.5)


def test_recall_counts_distinct_papers():
    assert recall(verdicts(), population=10) == pytest.approx(0.2)  # {P1,P2}/10


def test_population_filters_and_count():
    f = venue_population_filter(["S1", "S2"], "2025-01-01")
    assert f == "primary_location.source.id:S1|S2,from_publication_date:2025-01-01"
    f2 = arxiv_population_filter("S99", "2025-01-01", 10)
    assert f2 == ("primary_location.source.id:S99,"
                  "from_publication_date:2025-01-01,cited_by_count:>10")

    class FakeResponse:
        def json(self):
            return {"meta": {"count": 1234}, "results": []}

        def raise_for_status(self):
            pass

    n = openalex_population_count(f, mailto="a@b.c",
                                  http_get=lambda url, params=None: FakeResponse())
    assert n == 1234


def test_embedding_metrics():
    corpus = np.array([[1, 0], [0, 1]], dtype=np.float32)
    v_new = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=np.float32)
    assert novelty(v_new, corpus) == pytest.approx(1 - np.cos(np.pi / 4), abs=1e-5)
    assert past_dup_flag(np.array([1, 0], dtype=np.float32), corpus, ceiling=0.95)
    assert not past_dup_flag(v_new, corpus, ceiling=0.95)
    assert diversity(np.array([[1, 0], [0, 1]], dtype=np.float32)) == pytest.approx(1.0)
    assert bridging(["a", "b", "c"], {"a": 0, "b": 0, "c": 2}) == 2


def test_aggregate_run():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    agg = aggregate_run(verdicts(), dup, population=10)
    assert agg == {"precision": 0.5, "recall": 0.2, "n_ideas": 4,
                   "n_hits": 3, "n_dup_flagged": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/innovation/eval/metrics.py`**

```python
"""Headline metrics (precision/recall, spec §3.6) + process observables (novelty,
bridging, diversity)."""
import numpy as np
import requests

OPENALEX_WORKS = "https://api.openalex.org/works"


# --- headline ---
def precision(verdicts, dup_flags: dict[str, bool]) -> float:
    if not verdicts:
        return 0.0
    good = sum(1 for v in verdicts if v.hit and not dup_flags.get(v.idea_id, False))
    return good / len(verdicts)


def recall(verdicts, population: int) -> float:
    papers = {v.paper["paper_id"] for v in verdicts if v.hit}
    return len(papers) / population if population else 0.0


def venue_population_filter(source_ids: list[str], from_date: str) -> str:
    return (f"primary_location.source.id:{'|'.join(source_ids)},"
            f"from_publication_date:{from_date}")


def arxiv_population_filter(arxiv_source_id: str, from_date: str,
                            min_citations: int) -> str:
    return (f"primary_location.source.id:{arxiv_source_id},"
            f"from_publication_date:{from_date},cited_by_count:>{min_citations}")


def openalex_population_count(filter_str: str, *, mailto: str, http_get=None) -> int:
    """Recall denominator via a metadata count — no downloads (spec §3.6)."""
    http_get = http_get or requests.get
    r = http_get(OPENALEX_WORKS,
                 params={"filter": filter_str, "per-page": 1, "mailto": mailto})
    r.raise_for_status()
    return r.json()["meta"]["count"]


# --- process observables ---
def novelty(vec: np.ndarray, corpus_vecs: np.ndarray) -> float:
    return float(1.0 - np.max(corpus_vecs @ vec))


def past_dup_flag(vec: np.ndarray, corpus_vecs: np.ndarray,
                  ceiling: float = 0.95) -> bool:
    """Anti-plagiarism-of-the-past (spec §3.6): near-duplicate of the <=T corpus."""
    return bool(np.max(corpus_vecs @ vec) >= ceiling)


def bridging(cited_ids: list[str], communities: dict[str, int]) -> int:
    return len({communities[c] for c in cited_ids if c in communities})


def diversity(vecs: np.ndarray) -> float:
    if len(vecs) < 2:
        return 0.0
    sims = vecs @ vecs.T
    n = len(vecs)
    off_diag = sims[np.triu_indices(n, k=1)]
    return float(np.mean(1.0 - off_diag))


def aggregate_run(verdicts, dup_flags: dict[str, bool], population: int) -> dict:
    return {"precision": precision(verdicts, dup_flags),
            "recall": recall(verdicts, population),
            "n_ideas": len(verdicts),
            "n_hits": sum(1 for v in verdicts if v.hit),
            "n_dup_flagged": sum(1 for f in dup_flags.values() if f)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/eval/metrics.py tests/test_metrics.py
git commit -m "feat: precision/recall with OpenAlex population count + process observables"
```

---

### Task 14: Config, CLI, and end-to-end smoke test

**Files:**
- Create: `src/innovation/config.py`, `src/innovation/cli.py`, `configs/stage1.yaml`
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `load_config(path) -> dict` (YAML); CLI `uv run python -m innovation.cli <subcommand> --config configs/stage1.yaml` with subcommands `fetch` (resolve venue names → source ids, fetch works, build + save corpus), `summarize` (papers → ideas.parquet + embeddings), `run` (build graph+index from tables, run one `RunConfig` from the config, write events), `evaluate` (verify each generated idea from a run's events, compute metrics, write `metrics.json`). The smoke test wires the whole pipeline in-process with fakes and is the Stage 1 definition of done.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_smoke.py
"""End-to-end: synthetic corpus -> ideas -> graph -> simulation -> evaluation.
Everything offline: FakeLLM, FakeEmbedder, fake http_get."""
import json

import pandas as pd

from innovation.eval.metrics import aggregate_run, past_dup_flag
from innovation.eval.search_verify import verify_idea
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import FakeEmbedder
from innovation.ideas.summarize import summarize_corpus
from innovation.llm import CachedLLM, FakeLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_full_pipeline_smoke(tmp_path):
    # 1) Synthetic papers -> ideas
    papers = pd.DataFrame(
        [{"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"About topic {i}",
          "year": 2018 + i % 5, "venue": "NeurIPS"} for i in range(10)])
    sum_llm = CachedLLM(FakeLLM(default="A distilled idea."), tmp_path / "llm_cache")
    ideas = summarize_corpus(sum_llm, papers, model="haiku")
    ideas["idea_text"] = [f"Idea {i} about topic {i}" for i in range(10)]  # unique texts

    # 2) Graph + index
    edges = pd.DataFrame([{"src": f"W{i}", "dst": f"W{i - 1}"} for i in range(1, 10)])
    graph = IdeaGraph.from_tables(ideas, edges)
    emb = FakeEmbedder()
    index = VectorIndex(emb.dim)
    ids = graph.node_ids()
    corpus_vecs = emb.encode([graph.node(n).text for n in ids])
    index.add(ids, corpus_vecs)

    # 3) Simulate: 2 LLM agents, stigmergy, shared budget
    gen = json.dumps({"action": "generate",
                      "args": {"text": "a fresh combination idea",
                               "cited_ids": ["W1", "W5"]}})
    search = json.dumps({"action": "search", "args": {"query": "topic", "k": 2}})
    agent_llm = FakeLLM(responses=[search, search], default=gen)
    cfg = RunConfig(run_id="smoke", seed=7, total_steps=8, generation_budget=3,
                    agents=[{"agent_id": "a0", "policy": "llm"},
                            {"agent_id": "a1", "policy": "llm"}])
    out = run_simulation(cfg, graph=graph, index=index, embedder=emb,
                         llm=agent_llm, model="sonnet", out_dir=tmp_path / "runs")
    assert len(out["generated"]) == 3

    # 4) Evaluate: one post-cutoff realization exists in the fake index
    payload = {"data": [{"paperId": "future", "title": "Future paper",
                         "abstract": "does it", "publicationDate": "2025-09-09"}]}

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {"count": 50}})

    eval_llm = FakeLLM(responses=["some query", "YES"] * 10)
    verdicts = [verify_idea(eval_llm, model="sonnet", idea_id=nid,
                            idea_text=graph.node(nid).text,
                            cutoff_date="2025-01-01", mailto="a@b.c",
                            cache_dir=tmp_path / "search", http_get=fake_get,
                            n_queries=1, top_k=3)
                for nid in out["generated"]]
    dup_flags = {nid: past_dup_flag(emb.encode([graph.node(nid).text])[0],
                                    corpus_vecs)
                 for nid in out["generated"]}
    agg = aggregate_run(verdicts, dup_flags, population=50)
    assert agg["n_ideas"] == 3
    assert 0.0 <= agg["precision"] <= 1.0
    assert agg["n_hits"] >= 1
```

- [ ] **Step 2: Run smoke test to verify it fails**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: FAIL (imports fine, but `verify_idea` caching between identical queries may return same verdicts — the test should still pass logically; if it fails on any interface mismatch, that is exactly what this task exists to catch and fix). If it passes immediately, continue — the smoke test is still the regression net.

- [ ] **Step 3: Implement `src/innovation/config.py`**

```python
"""YAML experiment configuration (spec §3.5: configs are checked into the repo)."""
from pathlib import Path

import yaml


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())
```

- [ ] **Step 4: Write `configs/stage1.yaml`**

```yaml
# Stage 1 (spec §2): NeurIPS + ICLR slice, cutoff aligned to model training cutoff.
mailto: impanyu@gmail.com
data_dir: data/stage1
venues: ["Neural Information Processing Systems", "International Conference on Learning Representations"]
year_from: 2013
cutoff_year: 2024        # last full year in the graph
cutoff_date: "2025-01-01"  # anticipation boundary = agent model training cutoff
models:
  summarizer: claude-haiku-4-5-20251001
  agent: claude-sonnet-5
  judge: claude-sonnet-5
embedding_model: BAAI/bge-small-en-v1.5
eval:
  n_queries: 3
  top_k: 5
  dup_ceiling: 0.95
  arxiv_min_citations: 10
run:
  run_id: stage1-pilot
  seed: 0
  total_steps: 200
  generation_budget: 20
  agents:
    - {agent_id: a0, policy: llm}
    - {agent_id: a1, policy: llm}
    - {agent_id: a2, policy: llm}
    - {agent_id: a3, policy: llm}
    - {agent_id: a4, policy: llm}
out_dir: runs
```

- [ ] **Step 5: Implement `src/innovation/cli.py`**

```python
"""CLI: fetch -> summarize -> run -> evaluate, all driven by one YAML config."""
import argparse
import json
from pathlib import Path

import numpy as np

from innovation.config import load_config
from innovation.data.corpus import build_corpus, load_corpus, save_corpus
from innovation.data.openalex import fetch_source_works, find_source_id
from innovation.eval.metrics import (aggregate_run, past_dup_flag,
                                     openalex_population_count,
                                     venue_population_filter)
from innovation.eval.search_verify import verify_idea
from innovation.experiments.events import load_events
from innovation.experiments.runner import RunConfig, run_simulation
from innovation.ideas.embed import Embedder, load_embeddings, save_embeddings
from innovation.ideas.summarize import load_ideas, save_ideas, summarize_corpus
from innovation.llm import AnthropicLLM, CachedLLM
from innovation.network.graph import IdeaGraph
from innovation.network.index import VectorIndex


def _llm(cfg):
    return CachedLLM(AnthropicLLM(), Path(cfg["data_dir"]) / "llm_cache")


def cmd_fetch(cfg):
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    works_by_venue = {}
    for name in cfg["venues"]:
        sid = find_source_id(name, mailto=cfg["mailto"], cache_dir=cache)
        print(f"{name} -> {sid}")
        works_by_venue[name] = fetch_source_works(
            sid, cfg["year_from"], cfg["cutoff_year"],
            mailto=cfg["mailto"], cache_dir=cache)
    papers, edges = build_corpus(works_by_venue)
    save_corpus(papers, edges, cfg["data_dir"])
    print(f"papers={len(papers)} edges={len(edges)}")


def cmd_summarize(cfg):
    papers, _ = load_corpus(cfg["data_dir"])
    ideas = summarize_corpus(_llm(cfg), papers, model=cfg["models"]["summarizer"])
    save_ideas(ideas, cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    vecs = emb.encode(list(ideas["idea_text"]))
    save_embeddings(list(ideas["paper_id"]), vecs, cfg["data_dir"])
    print(f"ideas={len(ideas)} dim={emb.dim}")


def _load_world(cfg):
    _, edges = load_corpus(cfg["data_dir"])
    ideas = load_ideas(cfg["data_dir"])
    graph = IdeaGraph.from_tables(ideas, edges)
    ids, vecs = load_embeddings(cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    index = VectorIndex(emb.dim)
    index.add(ids, vecs)
    return graph, index, emb, dict(zip(ids, vecs))


def cmd_run(cfg):
    graph, index, emb, _ = _load_world(cfg)
    r = cfg["run"]
    run_cfg = RunConfig(run_id=r["run_id"], seed=r["seed"],
                        total_steps=r["total_steps"],
                        generation_budget=r["generation_budget"],
                        agents=r["agents"])
    out = run_simulation(run_cfg, graph=graph, index=index, embedder=emb,
                         llm=_llm(cfg), model=cfg["models"]["agent"],
                         out_dir=cfg["out_dir"])
    print(json.dumps(out, indent=2))


def cmd_evaluate(cfg):
    graph, index, emb, vec_by_id = _load_world(cfg)
    run_dir = Path(cfg["out_dir"]) / cfg["run"]["run_id"]
    events = load_events(run_dir / "events.jsonl")
    generated = [(e["result"]["node_id"], e["args"]["text"])
                 for e in events
                 if e["action"] == "generate" and "node_id" in e.get("result", {})]
    corpus_vecs = np.stack(list(vec_by_id.values()))
    llm = _llm(cfg)
    verdicts, dup_flags = [], {}
    for nid, text in generated:
        verdicts.append(verify_idea(
            llm, model=cfg["models"]["judge"], idea_id=nid, idea_text=text,
            cutoff_date=cfg["cutoff_date"], mailto=cfg["mailto"],
            cache_dir=run_dir / "search_cache",
            n_queries=cfg["eval"]["n_queries"], top_k=cfg["eval"]["top_k"]))
        dup_flags[nid] = past_dup_flag(emb.encode([text])[0], corpus_vecs,
                                       ceiling=cfg["eval"]["dup_ceiling"])
    cache = Path(cfg["data_dir"]) / "openalex_cache"
    source_ids = [find_source_id(v, mailto=cfg["mailto"], cache_dir=cache)
                  for v in cfg["venues"]]
    population = openalex_population_count(
        venue_population_filter(source_ids, cfg["cutoff_date"]),
        mailto=cfg["mailto"])
    agg = aggregate_run(verdicts, dup_flags, population)
    (run_dir / "metrics.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))


def main():
    parser = argparse.ArgumentParser(prog="innovation")
    parser.add_argument("command",
                        choices=["fetch", "summarize", "run", "evaluate"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    {"fetch": cmd_fetch, "summarize": cmd_summarize,
     "run": cmd_run, "evaluate": cmd_evaluate}[args.command](cfg)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the whole test suite**

Run: `uv run pytest -v`
Expected: all tests pass, including the smoke test.

- [ ] **Step 7: Commit**

```bash
git add src/innovation/config.py src/innovation/cli.py configs/stage1.yaml tests/test_smoke.py
git commit -m "feat: config, CLI pipeline commands, end-to-end smoke test"
```

---

## After the plan (not tasks — checkpoints for the human)

1. **Real-data pilot** (needs network + API key, run manually):
   `uv run python -m innovation.cli fetch --config configs/stage1.yaml`, inspect paper/edge counts and venue resolution (the `find_source_id` top hit must be audited by eye — OpenAlex search can return the wrong source).
2. **Summary audit gate** (spec §5): sample ~30 idea summaries, human-check before any experiment.
3. **Pilot run + evaluate**, then review `metrics.json` and the event trace together.

## Self-Review (completed)

- **Spec coverage:** data layer → T2–3; idea layer → T4–5; network → T5–6; actions/policies/stigmergy/budget fairness → T8–11; personas → `persona` kwarg (T10) reachable from agent spec dicts (T11); search-only eval, anticipation-only hits, cached timestamped responses, population-count recall, dup ceiling → T12–13; configs/YAML/CLI → T14. Structure-placement conditions (dense/boundary/cross community, spec §3.5) are *experiment configs* built on `communities()` (T6) — they are Stage 1 experiment design work on top of this apparatus, listed as a checkpoint deliverable, not a code task.
- **Placeholder scan:** none found; all steps carry real code.
- **Type consistency:** `Action(name, args)` used identically in T8–11; `Verdict` fields consistent between T12 and T13; embedding arrays float32 L2-normalized throughout; `obs = {"step", "last_result"}` consistent across T9–11.
