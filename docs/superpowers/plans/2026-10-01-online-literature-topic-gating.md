# Online Literature + Topic Gating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace paper 2's frozen corpus with online Semantic Scholar search over the seven CCF-A AI venues, label every paper, post and query with 1-5 topics from a frozen 128-topic list via an independent LLM, and gate every agent action by the agent's topics.

**Architecture:** New, separately testable units under `src/innovation/p2_forum/`:
- `topics.py`: the topic list and its builder helpers.
- `tagger.py`: the labeling LLM.
- `s2_online.py`: a cached Semantic Scholar client.
- `literature.py`: scope filter, paper records and label store.
- `gated_env.py`: a `ForumEnvironment` subclass that serves the literature online and applies the topic gate.

The existing frozen-corpus mode (`literature: corpus`, `gating: none`) keeps working unchanged for the 27 soft runs. The runner and CLI pick the mode from config.

**Tech Stack:** Python 3.14, `uv`, pytest, requests, numpy, PyYAML, the repo's `CachedLLM`/`RoutedLLM` (Anthropic + OpenAI), `Embedder` (BAAI/bge-small-en-v1.5).

**Spec:** `docs/superpowers/specs/2026-09-30-online-literature-topic-gating-design.md`

## Global Constraints

- Venues: **AAAI, NeurIPS, ACL, CVPR, ICCV, ICML, ICLR** (CCF-A AI, 2026 list).
- Scope rule: (venue ∈ the seven **OR** citationCount ≥ **50**) **AND** publication date ≤ **2024-09-30**. No lower year bound. Workshop venues never satisfy the venue branch.
- Topic list: exactly **128** topics, each with name, one-sentence definition, provenance; frozen after user review as `configs/p2_forum/topics-v2.yaml`; ids are list positions 0-127.
- Labels: **1-5** topic ids per item, ranked; tagger = **`claude-sonnet-5`**; up to **3** attempts, then the item is unlabeled (unreadable; a post is not published).
- Gate: `readable(agent, item) := labels(item) ∩ topics(agent) ≠ ∅`. No own-post exception. All nine actions gated as in spec §5-§6. Refusals carry `"gate": <reason>`.
- Agent model `openai:gpt-5:medium`; judge `claude-opus-5-5`; embeddings `BAAI/bge-small-en-v1.5`.
- Posts stay 3-4 sentence idea paragraphs.
- Paper 2 is standalone: no paper-1 comparison code, figures or judge runs.
- No test touches the network or a real LLM. Use fakes.
- Old mode must stay reproducible. Existing tests must keep passing.
- Commit attribution line: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. After each task: `git -C ~/git_repo/structure-and-dynamics-of-innovation merge --ff-only claude/multi-agent-research-ideas-f5a440 && git push origin claude/multi-agent-research-ideas-f5a440 main`.
- Run tests with `uv run pytest -q` and read the summary line, not `| tail` of a piped run that can hide failures.

## File map

| File | Responsibility |
|---|---|
| `configs/p2_forum/venue_areas.yaml` (new) | Official submission areas per venue, with source URLs |
| `src/innovation/p2_forum/topics.py` (new) | `Topic`, `load_topics`, consolidation prompt + parser |
| `scripts/p2_forum/build_topic_list.py` (new) | One-shot builder → `topics-v2.draft.yaml` |
| `src/innovation/p2_forum/tagger.py` (new) | `TopicTagger`, `parse_labels`, `UnlabeledError` |
| `src/innovation/p2_forum/s2_online.py` (new) | `S2Online` cached client: search / paper / references / citations |
| `src/innovation/p2_forum/literature.py` (new) | `Paper`, `Scope`, `OnlineLiterature` (scope filter, label store) |
| `src/innovation/p2_forum/workspace.py` (modify) | Optional `external_papers` lookup for board stubs |
| `src/innovation/p2_forum/gated_env.py` (new) | `GatedForumEnvironment` |
| `src/innovation/p2_forum/agent.py` (modify) | Gated prompt and actions doc; injectable templates |
| `src/innovation/p2_forum/runner.py` (modify) | Mode wiring, display-order shuffle, topic ids |
| `src/innovation/p2_forum/board_metrics.py` (modify) | Replay online runs (stubs need no corpus) |
| `src/innovation/cli.py` (modify) | `_load_online_world`, online run path, online evaluation reference |
| `configs/p2_forum/base-online.yaml` (new), `scripts/gen_forum_configs.py` (modify) | Online sweep configs |
| `scripts/p2_forum/gate_stats.py`, `scripts/p2_forum/tag_quality.py` (new) | Spec §10 measurements |

---

### Task 1: Venue submission areas (data collection)

**Files:**
- Create: `configs/p2_forum/venue_areas.yaml`

**Interfaces:**
- Produces: YAML `{venues: [{venue: str, edition: str, url: str, areas: [str, ...]}]}`, consumed by Task 2.

- [ ] **Step 1: Collect each venue's official area list.**

Fetch each call for papers (WebFetch) and copy the area/topic/keyword list verbatim. Use the latest edition on or before 2024. ICCV meets in odd years, so use 2023.

| venue | edition | where |
|---|---|---|
| NeurIPS | 2024 | https://neurips.cc/Conferences/2024/CallForPapers (topic list) |
| ICML | 2024 | https://icml.cc/Conferences/2024/CallForPapers (topics) |
| ICLR | 2024 | https://iclr.cc/Conferences/2024/CallForPapers (subject areas) |
| AAAI | AAAI-25 | AAAI-25 call for papers / keywords page on aaai.org (fine-grained keyword list) |
| ACL | 2024 | https://2024.aclweb.org/calls/main_conference_papers (submission topics/tracks) |
| CVPR | 2024 | https://cvpr.thecvf.com/Conferences/2024/CallForPapers (topics) |
| ICCV | 2023 | https://iccv2023.thecvf.com/main.track.call.for.papers-362500-2-20-16.php (topics) |

If a URL has moved, search for "<venue> <year> call for papers" and record the URL actually used. Do not paraphrase or invent areas. If a venue's list cannot be retrieved, stop and report it rather than filling it from memory.

- [ ] **Step 2: Write the file.**

```yaml
# Official submission areas of the seven CCF-A AI venues, copied verbatim
# from each call for papers. Input to scripts/p2_forum/build_topic_list.py.
venues:
  - venue: NeurIPS
    edition: "2024"
    url: https://neurips.cc/Conferences/2024/CallForPapers
    areas:
      - "<verbatim area 1>"
      # ... every area, one per line
```

- [ ] **Step 3: Sanity check and commit.**

Run: `uv run python -c "import yaml;d=yaml.safe_load(open('configs/p2_forum/venue_areas.yaml'));print({v['venue']:len(v['areas']) for v in d['venues']})"`
Expected: 7 venues, each with a non-zero count (AAAI the largest).

```bash
git add configs/p2_forum/venue_areas.yaml
git commit -m "paper 2: official submission areas of the seven CCF-A AI venues"
```

---

### Task 2: Topic list module and one-shot builder

**Files:**
- Create: `src/innovation/p2_forum/topics.py`
- Create: `scripts/p2_forum/build_topic_list.py`
- Test: `tests/test_forum_topics_v2.py`

**Interfaces:**
- Consumes: `configs/p2_forum/venue_areas.yaml` (Task 1).
- Produces:
  - `Topic(id: int, name: str, definition: str, sources: list[str])`
  - `load_topics(path, expected: int = 128) -> list[Topic]`
  - `consolidation_prompt(venues: list[dict], n: int) -> str`
  - `parse_topic_list(reply: str, n: int, venue_names: set[str]) -> list[dict]`
  - file `configs/p2_forum/topics-v2.yaml`, schema `{topics: [{name, definition, sources}]}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forum_topics_v2.py
import json

import pytest
import yaml

from innovation.p2_forum.topics import (Topic, consolidation_prompt,
                                        load_topics, parse_topic_list)

VENUES = [{"venue": "NeurIPS", "areas": ["Deep learning", "Theory"]},
          {"venue": "ACL", "areas": ["Machine translation"]}]


def _topics(n):
    return [{"name": f"T{i}", "definition": f"d{i}", "sources": ["NeurIPS"]}
            for i in range(n)]


def test_prompt_lists_every_area_with_its_venue():
    p = consolidation_prompt(VENUES, n=128)
    assert "NeurIPS: Deep learning" in p and "ACL: Machine translation" in p
    assert "exactly 128" in p


def test_parse_accepts_a_valid_list_inside_fences():
    reply = "```json\n" + json.dumps(_topics(4)) + "\n```"
    out = parse_topic_list(reply, n=4, venue_names={"NeurIPS", "ACL"})
    assert [t["name"] for t in out] == ["T0", "T1", "T2", "T3"]


@pytest.mark.parametrize("bad, msg", [
    (_topics(3), "expected 4"),
    (_topics(3) + [{"name": "T0", "definition": "x", "sources": ["ACL"]}], "duplicate"),
    (_topics(3) + [{"name": "T9", "definition": "", "sources": ["ACL"]}], "definition"),
    (_topics(3) + [{"name": "T9", "definition": "x", "sources": ["KDD"]}], "unknown venue"),
])
def test_parse_rejects_bad_lists(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_topic_list(json.dumps(bad), n=4, venue_names={"NeurIPS", "ACL"})


def test_load_topics_assigns_ids_by_position(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump({"topics": _topics(3)}))
    ts = load_topics(f, expected=3)
    assert ts[2] == Topic(id=2, name="T2", definition="d2", sources=["NeurIPS"])


def test_load_topics_refuses_wrong_count(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump({"topics": _topics(3)}))
    with pytest.raises(ValueError, match="128"):
        load_topics(f)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_topics_v2.py -q`
Expected: FAIL with `ModuleNotFoundError: innovation.p2_forum.topics`

- [ ] **Step 3: Implement `topics.py`**

```python
"""The frozen paper-2 topic list (spec §3): built once from the seven
venues' official submission areas, reviewed by the user, then never
regenerated. Topic ids are list positions."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    definition: str
    sources: list[str] = field(default_factory=list)


def load_topics(path, expected: int = 128) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text())["topics"]
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} topics, found {len(raw)}")
    names = [t["name"] for t in raw]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate topic names")
    return [Topic(id=i, name=t["name"], definition=t["definition"],
                  sources=list(t.get("sources", []))) for i, t in enumerate(raw)]


def consolidation_prompt(venues: list[dict], n: int) -> str:
    lines = [f"{v['venue']}: {a}" for v in venues for a in v["areas"]]
    return (
        "Below are the official submission areas of top AI venues, one per line "
        "as '<venue>: <area>'.\n\n" + "\n".join(lines) + "\n\n"
        f"Merge and deduplicate them into exactly {n} research topics that together "
        "cover all of them. Topics must not overlap in meaning, should be of similar "
        "granularity (split broad areas along the finer keyword lists, merge "
        "near-synonyms), and must avoid generic modifiers such as 'robust', "
        "'efficient', 'scalable' or 'advances in' unless they name the topic itself.\n"
        "Reply with ONLY a JSON list of objects: "
        '{"name": "<3-8 word topic name>", "definition": "<one sentence: what is in, '
        'what is out>", "sources": ["<venue>", ...]} where sources lists every venue '
        "whose areas this topic absorbs.")


def parse_topic_list(reply: str, n: int, venue_names: set[str]) -> list[dict]:
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON list in reply")
    items = json.loads(reply[start:end + 1])
    if len(items) != n:
        raise ValueError(f"expected {n} topics, got {len(items)}")
    seen = set()
    for t in items:
        name = str(t.get("name", "")).strip()
        if not name or name.lower() in seen:
            raise ValueError(f"duplicate or empty topic name: {name!r}")
        seen.add(name.lower())
        if not str(t.get("definition", "")).strip():
            raise ValueError(f"topic {name!r} has no definition")
        unknown = set(t.get("sources", [])) - venue_names
        if unknown or not t.get("sources"):
            raise ValueError(f"topic {name!r} cites unknown venue(s) {sorted(unknown)}")
    return [{"name": t["name"].strip(), "definition": t["definition"].strip(),
             "sources": list(t["sources"])} for t in items]
```

- [ ] **Step 4: Run tests and confirm they pass**

Run: `uv run pytest tests/test_forum_topics_v2.py -q`
Expected: 8 passed

- [ ] **Step 5: Write the builder script**

```python
# scripts/p2_forum/build_topic_list.py
"""One-shot: venue_areas.yaml -> configs/p2_forum/topics-v2.draft.yaml.

The user reviews the draft; renaming it to topics-v2.yaml freezes it (spec §3).
Run: uv run python scripts/p2_forum/build_topic_list.py
"""
from pathlib import Path

import yaml

from innovation.core.config import load_env
from innovation.core.llm import CachedLLM, RoutedLLM
from innovation.p2_forum.topics import consolidation_prompt, parse_topic_list

N, MODEL = 128, "claude-sonnet-5"
load_env()
venues = yaml.safe_load(open("configs/p2_forum/venue_areas.yaml"))["venues"]
names = {v["venue"] for v in venues}
llm = CachedLLM(RoutedLLM(), Path("data/online_cache/llm"))
prompt, feedback = consolidation_prompt(venues, N), ""
for attempt in range(3):
    reply = llm.complete(model=MODEL, system="You design research taxonomies.",
                         user=prompt + feedback, max_tokens=32000)
    try:
        topics = parse_topic_list(reply, N, names)
        break
    except ValueError as e:
        feedback = f"\n\nYour previous answer was rejected: {e}. Fix it."
else:
    raise SystemExit("could not obtain a valid topic list in 3 attempts")
out = Path("configs/p2_forum/topics-v2.draft.yaml")
out.write_text("# DRAFT topic list built from configs/p2_forum/venue_areas.yaml by\n"
               "# scripts/p2_forum/build_topic_list.py. Review, edit, then rename to\n"
               "# topics-v2.yaml to freeze it (ids = list positions).\n"
               + yaml.safe_dump({"topics": topics}, allow_unicode=True, sort_keys=False, width=100))
print(f"wrote {out} ({len(topics)} topics)")
```

- [ ] **Step 6: Run the builder (live LLM; a few dollars at most)**

Run: `uv run python scripts/p2_forum/build_topic_list.py`
Expected: `wrote configs/p2_forum/topics-v2.draft.yaml (128 topics)`

- [ ] **Step 7: Commit, then STOP for user review**

```bash
git add src/innovation/p2_forum/topics.py scripts/p2_forum/build_topic_list.py tests/test_forum_topics_v2.py configs/p2_forum/topics-v2.draft.yaml
git commit -m "paper 2: topic-list module and builder; draft 128-topic list from venue areas"
```

Send the user the draft list. Show each topic's name, definition and sources, and point out the ones that look too broad or too narrow. Apply their edits. Then freeze it with `git mv configs/p2_forum/topics-v2.draft.yaml configs/p2_forum/topics-v2.yaml`, remove the DRAFT header lines, and commit `paper 2: freeze topics-v2 (user-reviewed)`. **Tasks 8-12 need the frozen file. Tasks 3-7 use fakes and can proceed in parallel.**

---

### Task 3: The tagger

**Files:**
- Create: `src/innovation/p2_forum/tagger.py`
- Test: `tests/test_forum_tagger.py`

**Interfaces:**
- Consumes: `Topic` (Task 2); any `LLM` (`complete(model=, system=, user=, max_tokens=)`).
- Produces:
  - `parse_labels(reply: str, n_topics: int, max_labels: int = 5) -> list[int] | None`
  - `class UnlabeledError(Exception)`
  - `class TopicTagger(llm, model: str, topics: list[Topic], max_labels: int = 5, attempts: int = 3)`
    - `.system: str` (the fixed prefix)
    - `.label(text: str) -> list[int]` (raises `UnlabeledError`)
    - `.label_many(texts: list[str], workers: int = 8) -> list[list[int] | None]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forum_tagger.py
import pytest

from innovation.core.llm import FakeLLM
from innovation.p2_forum.tagger import TopicTagger, UnlabeledError, parse_labels
from innovation.p2_forum.topics import Topic

TOPICS = [Topic(i, f"T{i}", f"def {i}") for i in range(10)]


@pytest.mark.parametrize("reply, want", [
    ("[3, 1]", [3, 1]),
    ("labels: [0]", [0]),
    ("[2, 2, 5]", [2, 5]),           # duplicates collapse, order kept
    ("[]", None),                    # at least one
    ("[1,2,3,4,5,6]", None),         # at most five
    ("[10]", None),                  # out of range
    ("no list", None),
])
def test_parse_labels(reply, want):
    assert parse_labels(reply, n_topics=10) == want


def test_system_prompt_lists_every_topic_with_id_and_definition():
    t = TopicTagger(llm=FakeLLM(), model="m", topics=TOPICS)
    assert "7: T7 — def 7" in t.system


def test_label_retries_then_succeeds():
    llm = FakeLLM(responses=["garbage", "[4]"])
    assert TopicTagger(llm=llm, model="m", topics=TOPICS).label("x") == [4]
    assert len(llm.calls) == 2


def test_label_gives_up_after_three_attempts():
    llm = FakeLLM(default="garbage")
    with pytest.raises(UnlabeledError):
        TopicTagger(llm=llm, model="m", topics=TOPICS).label("x")
    assert len(llm.calls) == 3


def test_retry_prompts_differ_so_a_cache_cannot_replay_the_bad_reply():
    llm = FakeLLM(responses=["garbage", "[4]"])
    TopicTagger(llm=llm, model="m", topics=TOPICS).label("x")
    assert llm.calls[0]["user"] != llm.calls[1]["user"]


def test_label_many_returns_none_for_failures():
    llm = FakeLLM(responses=["[1]"], default="garbage")
    out = TopicTagger(llm=llm, model="m", topics=TOPICS).label_many(["a", "b"], workers=1)
    assert out == [[1], None]
```

Check `FakeLLM` before relying on it. If `FakeLLM.calls` entries are not dicts with a `"user"` key, adapt the assertions to its actual record format; do not change `FakeLLM`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_tagger.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""The independent labeling LLM (spec §4). One call per item; the frozen list is
a fixed system-prompt prefix so a provider-side prompt cache can reuse it.
Agents never label their own work: this runs inside the environment."""
import json
import re
from concurrent.futures import ThreadPoolExecutor

from innovation.p2_forum.topics import Topic


class UnlabeledError(Exception):
    pass


def parse_labels(reply: str, n_topics: int, max_labels: int = 5) -> list[int] | None:
    m = re.search(r"\[[^\[\]]*\]", reply)
    if not m:
        return None
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out: list[int] = []
    for x in raw:
        if not isinstance(x, int) or not 0 <= x < n_topics:
            return None
        if x not in out:
            out.append(x)
    return out if 1 <= len(out) <= max_labels else None


class TopicTagger:
    def __init__(self, *, llm, model: str, topics: list[Topic],
                 max_labels: int = 5, attempts: int = 3):
        self.llm, self.model, self.topics = llm, model, topics
        self.max_labels, self.attempts = max_labels, attempts
        listing = "\n".join(f"{t.id}: {t.name} — {t.definition}" for t in topics)
        self.system = (
            "You label AI research texts with topics from a fixed list.\n\n"
            f"TOPICS (id: name — definition):\n{listing}\n\n"
            f"Reply with ONLY a JSON list of 1 to {max_labels} topic ids, most "
            "relevant first. Include a topic only if the text genuinely belongs "
            f"to it; never pad the list to {max_labels}.")

    def label(self, text: str) -> list[int]:
        for attempt in range(self.attempts):
            # The attempt number is part of the prompt, so a disk cache keyed on
            # the prompt never replays a reply that already failed to parse.
            user = f"TEXT:\n{text}" + (f"\n\n(retry {attempt})" if attempt else "")
            reply = self.llm.complete(model=self.model, system=self.system,
                                      user=user, max_tokens=100)
            labels = parse_labels(reply, len(self.topics), self.max_labels)
            if labels is not None:
                return labels
        raise UnlabeledError(f"no valid labels after {self.attempts} attempts")

    def label_many(self, texts: list[str], workers: int = 8) -> list[list[int] | None]:
        def one(t):
            try:
                return self.label(t)
            except UnlabeledError:
                return None
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, texts))
```

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_forum_tagger.py -q`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/tagger.py tests/test_forum_tagger.py
git commit -m "paper 2: independent topic tagger (1-5 labels, retries, batch)"
```

---

### Task 4: Cached Semantic Scholar client

**Files:**
- Create: `src/innovation/p2_forum/s2_online.py`
- Test: `tests/test_forum_s2_online.py`

**Interfaces:**
- Consumes: `innovation.core.data.s2._cached_call(cache_file, do_call, delay)` and `s2_headers()` (existing).
- Produces: `class S2Online(cache_dir, http_get=None, delay: float = 1.1)` with:
  - `.search(query: str, *, limit: int, max_date: str) -> list[dict]`
  - `.paper(pid: str) -> dict | None`
  - `.references(pid: str) -> list[dict]`
  - `.citations(pid: str) -> list[dict]`
  - Raw S2 paper dicts have keys `paperId, title, abstract, year, venue, publicationVenue, publicationDate, citationCount`.
  - Module constant `FIELDS`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forum_s2_online.py
from innovation.p2_forum.s2_online import S2Online


class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.headers = {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def fake_get(routes):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {})))
        for key, resp in routes.items():
            if key in url:
                return resp
        raise AssertionError(url)
    get.calls = calls
    return get


P = {"paperId": "a", "title": "A", "abstract": "x", "year": 2023,
     "venue": "NeurIPS", "publicationVenue": None,
     "publicationDate": "2023-12-01", "citationCount": 3}


def test_search_passes_date_cap_and_caches(tmp_path):
    get = fake_get({"/paper/search": Resp(200, {"data": [P]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.search("q", limit=50, max_date="2024-09-30") == [P]
    assert c.search("q", limit=50, max_date="2024-09-30") == [P]
    assert len(get.calls) == 1
    params = get.calls[0][1]
    assert params["publicationDateOrYear"] == ":2024-09-30"
    assert params["limit"] == 50


def test_references_and_citations_unwrap(tmp_path):
    get = fake_get({"/references": Resp(200, {"data": [{"citedPaper": P}]}),
                    "/citations": Resp(200, {"data": [{"citingPaper": P}]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.references("z") == [P] and c.citations("z") == [P]


def test_missing_paper_is_none(tmp_path):
    c = S2Online(tmp_path, http_get=fake_get({"/paper/": Resp(404, {})}), delay=0)
    assert c.paper("nope") is None


def test_null_entries_are_dropped(tmp_path):
    get = fake_get({"/references": Resp(200, {"data": [{"citedPaper": {"paperId": None}}, {"citedPaper": P}]})})
    assert S2Online(tmp_path, http_get=get, delay=0).references("z") == [P]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_s2_online.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Read `_cached_call` in `src/innovation/core/data/s2.py` first. It retries the `RETRYABLE` statuses and raises `HTTPError` through `raise_for_status()` on others.

```python
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
```

If `_cached_call` does not pass a `timeout` through or has a different signature, adapt the lambda to the real signature and keep the behavior.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_forum_s2_online.py -q`
Expected: 4 passed

- [ ] **Step 5: Live check (one call) and commit**

Run: `uv run python -c "from innovation.core.config import load_env;load_env();from innovation.p2_forum.s2_online import S2Online;r=S2Online('data/online_cache').search('graph neural network expressivity',limit=5,max_date='2024-09-30');print([(p['title'][:50],p['publicationDate'],p['venue']) for p in r])"`
Expected: 5 papers, all dated ≤ 2024-09-30. If the date filter is not honored, report it. Task 5's scope filter re-checks dates anyway.

```bash
git add src/innovation/p2_forum/s2_online.py tests/test_forum_s2_online.py
git commit -m "paper 2: cached Semantic Scholar client (search, paper, references, citations)"
```

---

### Task 5: Scope filter, paper records, online literature with label store

**Files:**
- Create: `src/innovation/p2_forum/literature.py`
- Test: `tests/test_forum_literature.py`

**Interfaces:**
- Consumes: `S2Online` (Task 4) or any object with the same four methods; `TopicTagger.label_many` (Task 3).
- Produces:
  - `Paper(paper_id, title, abstract, year, venue, pub_date, citations, branch)` with `.text() -> str`
  - `Scope(venue_aliases: tuple[str, ...], min_citations: int, max_date: str)` with `.admit(raw: dict) -> str | None` (`"venue"`, `"citations"` or `None`)
  - `OnlineLiterature(client, scope, tagger, cache_dir, search_pool: int = 50)` with:
    - `.search(query) -> list[Paper]`
    - `.get(pid) -> Paper | None`
    - `.references(pid) -> list[Paper]`
    - `.citations(pid) -> list[Paper]`
    - `.labels(pids: list[str]) -> dict[str, list[int] | None]`
    - `.has(pid) -> bool`
    - `.known() -> list[Paper]`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forum_literature.py
from innovation.p2_forum.literature import OnlineLiterature, Scope

ALIASES = ("neurips", "neural information processing", "computer vision and pattern recognition")
SCOPE = Scope(venue_aliases=ALIASES, min_citations=50, max_date="2024-09-30")


def raw(pid, venue="NeurIPS", date="2023-12-01", year=2023, cites=0, abstract="abs"):
    return {"paperId": pid, "title": f"T{pid}", "abstract": abstract, "year": year,
            "venue": venue, "publicationVenue": None, "publicationDate": date,
            "citationCount": cites}


def test_scope_branches():
    assert SCOPE.admit(raw("a")) == "venue"
    assert SCOPE.admit(raw("b", venue="Nature", cites=80)) == "citations"
    assert SCOPE.admit(raw("c", venue="Nature", cites=49)) is None
    assert SCOPE.admit(raw("d", date="2024-10-01")) is None
    assert SCOPE.admit(raw("e", date="2024-09-30")) == "venue"


def test_scope_workshops_and_undated_boundary_year():
    assert SCOPE.admit(raw("w", venue="CVPR Workshops")) is None
    assert SCOPE.admit(raw("w2", venue="2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)")) is None
    assert SCOPE.admit(raw("u", date=None, year=2024)) is None   # cannot prove <= cutoff
    assert SCOPE.admit(raw("v", date=None, year=2023)) == "venue"


def test_scope_reads_publication_venue_names():
    r = raw("p", venue="")
    r["publicationVenue"] = {"name": "Neural Information Processing Systems",
                             "alternate_names": ["NeurIPS"]}
    assert SCOPE.admit(r) == "venue"


class FakeClient:
    def __init__(self, papers, refs=None, cits=None):
        self.papers, self.refs, self.cits = papers, refs or {}, cits or {}
        self.search_calls = 0

    def search(self, q, *, limit, max_date):
        self.search_calls += 1
        return list(self.papers.values())[:limit]

    def paper(self, pid):
        return self.papers.get(pid)

    def references(self, pid):
        return self.refs.get(pid, [])

    def citations(self, pid):
        return self.cits.get(pid, [])


class FakeTagger:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def label_many(self, texts, workers=8):
        self.calls += len(texts)
        return [self.table.get(t.split("\n")[0]) for t in texts]


def lit(tmp_path, papers, **kw):
    tagger = FakeTagger({f"T{k}": [i] for i, k in enumerate(papers)})
    return OnlineLiterature(client=FakeClient(papers, **kw), scope=SCOPE,
                            tagger=tagger, cache_dir=tmp_path), tagger


def test_search_drops_out_of_scope_and_remembers_branch(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a"), "b": raw("b", venue="Nature", cites=99),
                          "c": raw("c", date="2025-01-01")})
    got = L.search("q")
    assert [(p.paper_id, p.branch) for p in got] == [("a", "venue"), ("b", "citations")]
    assert L.has("a") and not L.has("c")


def test_labels_are_computed_once_and_persisted(tmp_path):
    papers = {"a": raw("a"), "b": raw("b")}
    L, tagger = lit(tmp_path, papers)
    L.search("q")
    assert L.labels(["a", "b"]) == {"a": [0], "b": [1]}
    L.labels(["a", "b"])
    assert tagger.calls == 2
    L2, tagger2 = lit(tmp_path, papers)        # a fresh process reads the store
    L2.search("q")
    assert L2.labels(["a"]) == {"a": [0]} and tagger2.calls == 0


def test_get_and_neighbors_are_scope_filtered(tmp_path):
    papers = {"a": raw("a"), "x": raw("x", venue="Nature", cites=1)}
    L, _ = lit(tmp_path, papers, refs={"a": [raw("r", date="2022-01-01"), raw("late", date="2025-02-02")]})
    assert L.get("x") is None
    assert [p.paper_id for p in L.references("a")] == ["r"]


def test_text_uses_title_when_abstract_missing(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a", abstract=None)})
    assert L.get("a").text() == "Ta"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_literature.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
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
```

`Path(cache_dir)` may not exist yet. Create it with `Path(cache_dir).mkdir(parents=True, exist_ok=True)` in `__init__`, before touching `_store`.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_forum_literature.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/literature.py tests/test_forum_literature.py
git commit -m "paper 2: scope rule (venue OR >=50 cites, <=2024-09-30) and online literature with label store"
```

---

### Task 6: Workspace accepts external papers for board stubs

**Files:**
- Modify: `src/innovation/p2_forum/workspace.py` (`__init__`, `has_node`, `_ensure_stub`)
- Test: `tests/test_workspace.py` (append)

**Interfaces:**
- Produces: `Workspace(..., external_papers=None)`. `external_papers` is any object with `.has(pid) -> bool`. When it is given, corpus-store ids resolve through it instead of the frozen corpus graph. Default `None` keeps current behavior.

- [ ] **Step 1: Write the failing test** (append to `tests/test_workspace.py`)

```python
def test_external_papers_back_corpus_ids_for_stubs():
    import numpy as np
    from innovation.core.network.graph import IdeaGraph
    from innovation.core.network.index import VectorIndex
    from innovation.p2_forum.workspace import Workspace
    from tests.conftest import FakeEmbedder

    class Known:
        def has(self, pid):
            return pid == "s2:abc"

    empty = IdeaGraph()
    empty.freeze()
    ws = Workspace(corpus=empty, corpus_index=VectorIndex(4), board_index=VectorIndex(4),
                   embedder=FakeEmbedder(), run_id="t", external_papers=Known())
    nid = ws.post_idea("idea", ["s2:abc"], meta={})
    assert ws.has_node("s2:abc")
    assert ws.board.has_node("s2:abc")          # stub created
    assert ws.board.citations_out(nid) == ["s2:abc"]
    import pytest
    with pytest.raises(KeyError):
        ws.post_idea("idea2", ["unknown"], meta={})
```

If `tests/conftest.py` cannot be imported as `tests.conftest`, use the `fake_embedder` fixture as a test argument instead.

- [ ] **Step 2: Run it and confirm it fails**

Run: `uv run pytest tests/test_workspace.py -q -k external`
Expected: FAIL with `TypeError: unexpected keyword argument 'external_papers'`

- [ ] **Step 3: Implement**

In `Workspace.__init__`, add the parameter `external_papers=None` and store it as `self.external = external_papers`. Then:

```python
    def has_node(self, node_id: str) -> bool:
        if self.store_of(node_id) == "board":
            return self.board.has_node(node_id)
        if self.external is not None:
            return self.external.has(node_id)
        return self.corpus.has_node(node_id)
```

In `_ensure_stub`, replace `if not self.corpus.has_node(node_id):` with `if not self.has_node(node_id):`.

- [ ] **Step 4: Run the whole suite**

Run: `uv run pytest -q`
Expected: all previous tests plus the new one pass.

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/workspace.py tests/test_workspace.py
git commit -m "workspace: optional external paper lookup for board stubs (online literature)"
```

---

### Task 7: The gated environment

**Files:**
- Create: `src/innovation/p2_forum/gated_env.py`
- Test: `tests/test_forum_gated_env.py`

**Interfaces:**
- Consumes: `ForumEnvironment` (`env.py`), `Workspace(external_papers=)` (Task 6), `OnlineLiterature` interface (Task 5: `search/get/references/citations/labels/has`), `TopicTagger.label` + `UnlabeledError` (Task 3).
- Produces: `GatedForumEnvironment(*, run_id, workspace, event_log, rng, literature, tagger, agent_topics: dict[str, set[int]], topic_names: list[str], generation_budget=None)`.
  - Public: `.readable(agent_id, node_id) -> bool`, `.post_labels: dict[str, list[int]]`, `.restore(events)`.
  - Result schemas:
    - search hits are `{node_id, store:"corpus", title, text (≤300 chars of abstract), year, venue, topics:[names]}`;
    - refusals are `{"error": str, "gate": "query"|"result"|"post"|"cite"|"link"|"unlabeled", ...}`;
    - a successful generate returns `{"node_id", "topics": [ids]}` plus optional `"dropped_cites"`;
    - a refused generate returns `{"error", "gate": "post", "topics": [ids]}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forum_gated_env.py
import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import Action
from innovation.p2_forum.gated_env import GatedForumEnvironment
from innovation.p2_forum.literature import Paper
from innovation.p2_forum.tagger import UnlabeledError
from innovation.p2_forum.workspace import Workspace
from tests.conftest import FakeEmbedder

NAMES = [f"T{i}" for i in range(4)]
P = {pid: Paper(pid, f"title {pid}", f"abstract {pid}", 2023, "NeurIPS", "2023-01-01", 0, "venue")
     for pid in ["p0", "p1", "p2", "p3"]}
LAB = {"p0": [0], "p1": [1], "p2": [0, 2], "p3": [3]}


class FakeLit:
    def __init__(self):
        self.refs = {"p0": ["p1", "p2"]}
        self.cits = {"p0": ["p3"]}

    def search(self, q):
        return list(P.values())

    def get(self, pid):
        return P.get(pid)

    def references(self, pid):
        return [P[x] for x in self.refs.get(pid, [])]

    def citations(self, pid):
        return [P[x] for x in self.cits.get(pid, [])]

    def labels(self, pids):
        return {p: LAB.get(p) for p in pids}

    def has(self, pid):
        return pid in P


class FakeTagger:
    """Labels by keyword: text containing 'Tn' gets [n]; 'junk' fails."""
    def label(self, text):
        if "junk" in text:
            raise UnlabeledError("x")
        got = [i for i in range(4) if f"T{i}" in text]
        return got or [3]


def make(tmp_path, topics=None):
    empty = IdeaGraph()
    empty.freeze()
    ws = Workspace(corpus=empty, corpus_index=VectorIndex(4), board_index=VectorIndex(4),
                   embedder=FakeEmbedder(), run_id="t", external_papers=FakeLit())
    return GatedForumEnvironment(
        run_id="t", workspace=ws, event_log=EventLog(tmp_path / "e.jsonl"),
        rng=np.random.default_rng(0), literature=FakeLit(), tagger=FakeTagger(),
        agent_topics=topics or {"a": {0}, "b": {1}}, topic_names=NAMES)


def test_query_outside_topics_is_refused(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "about T1", "k": 5}))
    assert out["gate"] == "query" and out["topics"] == ["T1"]


def test_search_returns_only_readable_papers_with_topic_names(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "about T0", "k": 5}))
    assert [h["node_id"] for h in out["hits"]] == ["p0", "p2"]
    assert out["hits"][1]["topics"] == ["T0", "T2"]


def test_browse_unreadable_errors_and_neighbors_are_filtered(tmp_path):
    env = make(tmp_path)
    assert env.execute("a", 0, Action("browse", {"node_id": "p1"}))["gate"] == "result"
    v = env.execute("a", 1, Action("browse", {"node_id": "p0"}))
    assert [c["node_id"] for c in v["cites"]] == ["p2"]
    assert v["cited_by"] == []


def test_sample_frontier_stays_readable(tmp_path):
    env = make(tmp_path)
    for s in range(10):
        out = env.execute("a", s, Action("sample_frontier", {}))
        assert out["node_id"] in {"p0", "p2"}


def test_off_topic_post_is_not_published(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "idea on T1", "cited_ids": []}))
    assert out["gate"] == "post" and out["topics"] == [1]
    assert env.ws.board_post_ids() == []


def test_unlabeled_post_is_not_published(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "junk", "cited_ids": []}))
    assert out["gate"] == "unlabeled" and env.ws.board_post_ids() == []


def test_post_drops_unreadable_cites_and_is_readable_by_author(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": ["p0", "p1"]}))
    assert out["topics"] == [0] and out["dropped_cites"] == ["p1"]
    assert env.readable("a", out["node_id"]) and not env.readable("b", out["node_id"])


def test_board_reads_are_gated(tmp_path):
    env = make(tmp_path)
    pa = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": []}))["node_id"]
    pb = env.execute("b", 1, Action("generate", {"text": "idea T1", "cited_ids": []}))["node_id"]
    hits = env.execute("a", 2, Action("search_board", {"query": "T0", "k": 5}))["hits"]
    assert [h["node_id"] for h in hits] == [pa]
    assert env.execute("a", 3, Action("browse_board", {"node_id": pb}))["gate"] == "result"
    assert env.execute("a", 4, Action("sample_board", {}))["node_id"] == pa


def test_links_need_both_ends_readable(tmp_path):
    env = make(tmp_path)
    pa = env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": []}))["node_id"]
    assert env.execute("a", 1, Action("add_links", {"src_id": pa, "dst_ids": ["p1"]}))["gate"] == "link"
    ok = env.execute("a", 2, Action("add_links", {"src_id": pa, "dst_ids": ["p2"]}))
    assert "gate" not in ok


def test_restore_reuses_logged_labels_without_tagging(tmp_path):
    env = make(tmp_path)
    env.execute("a", 0, Action("generate", {"text": "idea T0", "cited_ids": ["p0"]}))
    events = env.event_log.read_all()
    fresh = make(tmp_path / "x")
    fresh.tagger = None                      # any tagger call would crash
    fresh.restore(events)
    pid = events[0]["result"]["node_id"]
    assert fresh.post_labels[pid] == [0] and fresh.readable("a", pid)


def test_all_topics_means_no_gating(tmp_path):
    env = make(tmp_path, topics={"a": {0, 1, 2, 3}})
    out = env.execute("a", 0, Action("search", {"query": "about T1", "k": 9}))
    assert len(out["hits"]) == 4
```

(`(tmp_path / "x")` must exist: call `(tmp_path / "x").mkdir()` before `make`, or let `EventLog` create parents. Check `EventLog.__init__`.)

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_gated_env.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

```python
"""Paper 2's environment with online literature and hard topic gating
(spec §5-§6). An item is readable by an agent iff its labels intersect the
agent's topics. Filtering what is RETURNED is the primary mechanism; the
checks on citations and links are a safety net. Every refusal carries a
"gate" field so gate activity can be counted from the event log."""
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.tagger import UnlabeledError


class GatedForumEnvironment(ForumEnvironment):
    def __init__(self, *, literature, tagger, agent_topics: dict[str, set[int]],
                 topic_names: list[str], **kw):
        super().__init__(**kw)
        self.lit, self.tagger = literature, tagger
        self.agent_topics = {a: set(t) for a, t in agent_topics.items()}
        self.topic_names = topic_names
        self.post_labels: dict[str, list[int]] = {}

    # --- labels and the gate ---
    def _labels(self, node_id: str) -> list[int] | None:
        if self.ws.store_of(node_id) == "board":
            return self.post_labels.get(node_id)
        return self.lit.labels([node_id]).get(node_id)

    def _ok(self, agent_id: str, labels) -> bool:
        return bool(labels) and bool(set(labels) & self.agent_topics[agent_id])

    def readable(self, agent_id: str, node_id: str) -> bool:
        return self._ok(agent_id, self._labels(node_id))

    def _names(self, labels) -> list[str]:
        return [self.topic_names[i] for i in labels or []]

    def _gated_papers(self, agent_id, papers):
        labs = self.lit.labels([p.paper_id for p in papers])
        return [(p, labs[p.paper_id]) for p in papers if self._ok(agent_id, labs[p.paper_id])]

    def _paper_hit(self, p, labels) -> dict:
        return {"node_id": p.paper_id, "store": "corpus", "title": p.title,
                "text": p.abstract[:300], "year": p.year, "venue": p.venue,
                "topics": self._names(labels)}

    def _query_gate(self, agent_id, query):
        try:
            labels = self.tagger.label(query)
        except UnlabeledError:
            return {"error": "this query could not be labeled", "gate": "unlabeled"}
        if not self._ok(agent_id, labels):
            return {"error": "this query is outside your topics", "gate": "query",
                    "topics": self._names(labels)}
        return None

    # --- literature ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        refused = self._query_gate(agent_id, query)
        if refused:
            return refused
        kept = self._gated_papers(agent_id, self.lit.search(query))[:k]
        return {"hits": [self._paper_hit(p, l) for p, l in kept]}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        p = self.lit.get(node_id)
        if p is None:
            return {"error": f"{node_id} is not an available paper", "gate": "result"}
        labels = self.lit.labels([node_id])[node_id]
        if not self._ok(agent_id, labels):
            return {"error": f"{node_id} is outside your topics", "gate": "result"}
        view = {**self._paper_hit(p, labels), "text": p.abstract}
        if not self.nav.corpus_edges:
            return {**view, "cites": [], "cited_by": []}
        cites = self._gated_papers(agent_id, self.lit.references(node_id))[:10]
        cited_by = self._gated_papers(agent_id, self.lit.citations(node_id))[:10]
        return {**view, "cites": [self._paper_hit(q, l) for q, l in cites],
                "cited_by": [self._paper_hit(q, l) for q, l in cited_by]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.nav.corpus_jump:
            return {"error": "random jumps into the literature are closed"}
        topic = self.topic_names[int(self.rng.choice(sorted(self.agent_topics[agent_id])))]
        kept = self._gated_papers(agent_id, self.lit.search(topic))
        if not kept:
            return {"error": "no paper found for a random jump"}
        p, l = kept[int(self.rng.integers(len(kept)))]
        return {**self._paper_hit(p, l), "text": p.abstract}

    # --- board ---
    def _post_view(self, nid) -> dict:
        if self.ws.store_of(nid) == "board":
            return {"node_id": nid, "store": "board", "text": self.ws.node(nid).text[:200],
                    "topics": self._names(self.post_labels.get(nid))}
        p = self.lit.get(nid)
        return {"node_id": nid, "store": "corpus", "title": p.title if p else "",
                "topics": self._names(self._labels(nid))}

    def _readable_posts(self, agent_id) -> list[str]:
        return [n for n in self.ws.board_post_ids() if self.readable(agent_id, n)]

    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.board_search:
            return {"error": "semantic search over the board is closed"}
        refused = self._query_gate(agent_id, query)
        if refused:
            return refused
        vec = self.ws.embedder.encode([query])[0]
        ranked = self.ws.board_search(vec, k=len(self.ws.board_post_ids()) or 1)
        kept = [(n, s) for n, s in ranked if self.readable(agent_id, n)][:k]
        return {"hits": [{**self._post_view(n), "text": self.ws.node(n).text[:300],
                          "score": s} for n, s in kept]}

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your topics", "gate": "result"}
        out_ids, in_ids = (self.ws.board_neighbors(node_id) if self.nav.board_edges
                           else ([], []))
        keep = lambda ids: [self._post_view(n) for n in ids if self.readable(agent_id, n)][:10]
        return {**self._post_view(node_id), "text": self.ws.node(node_id).text,
                "cites": keep(out_ids), "cited_by": keep(in_ids)}

    def _do_sample_board(self, *, agent_id, step) -> dict:
        if not self.nav.board_jump:
            return {"error": "random jumps into the board are closed"}
        posts = self._readable_posts(agent_id)
        if not posts:
            return {"error": "no readable posts on the board"}
        nid = posts[int(self.rng.integers(len(posts)))]
        return {**self._post_view(nid), "text": self.ws.node(nid).text}

    # --- writes ---
    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        try:
            labels = self.tagger.label(text)
        except UnlabeledError:
            return {"error": "this idea could not be labeled; it was not published",
                    "gate": "unlabeled"}
        if not self._ok(agent_id, labels):
            return {"error": "this idea is outside your topics; it was not published",
                    "gate": "post", "topics": labels}
        kept = [c for c in cited_ids if self.ws.has_node(c) and self.readable(agent_id, c)]
        dropped = [c for c in cited_ids if c not in kept]
        node_id = self.ws.post_idea(text, kept, meta=self._meta(agent_id, step))
        self.post_labels[node_id] = labels
        if self.generation_budget is not None:
            self.generation_budget -= 1
        out = {"node_id": node_id, "topics": labels}
        if dropped:
            out["dropped_cites"] = dropped
        return out

    def _links_gate(self, agent_id, src_id, dst_ids):
        blocked = [n for n in [src_id, *dst_ids] if not self.readable(agent_id, n)]
        if blocked:
            return {"error": f"outside your topics: {blocked}", "gate": "link"}
        return None

    def _do_add_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_add_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    def _do_remove_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_remove_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    # --- replay ---
    def restore(self, events: list[dict]) -> None:
        """Replay the board; post labels come from the log, so a resumed run
        makes no tagger calls for replayed steps."""
        for e in events:
            r = e.get("result", {})
            if e["action"] == "generate" and "node_id" in r:
                self.post_labels[r["node_id"]] = r["topics"]
                kept = [c for c in e["args"]["cited_ids"] if c not in r.get("dropped_cites", [])]
                self.ws.post_idea(e["args"]["text"], kept,
                                  meta={"run_id": e["run_id"], "agent_id": e["agent_id"],
                                        "step": e["step"]}, node_id=r["node_id"])
                if self.generation_budget is not None:
                    self.generation_budget -= 1
            else:
                super().restore([e])
```

Notes for the implementer:
- `rng.choice` over a sorted list keeps sampling deterministic per seed.
- `cited_ids` in the events are the agent's raw request. Replay must post only the kept ones, which the code above does by subtracting `dropped_cites`.
- `ForumEnvironment.execute` catches `KeyError/TypeError/ValueError`. A `has_node` on an unknown S2 id triggers `lit.has`, which is safe.

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_forum_gated_env.py -q`
Expected: 11 passed. Then run `uv run pytest -q`: everything passes.

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/gated_env.py tests/test_forum_gated_env.py
git commit -m "paper 2: gated environment over online literature (query/result/post/cite/link gates, log-only replay)"
```

---

### Task 8: Agent prompt for the gated, online mode

**Files:**
- Modify: `src/innovation/p2_forum/agent.py`
- Test: `tests/test_forum_agent.py` (append)

**Interfaces:**
- Produces:
  - constants `GATED_SYSTEM` and `GATED_ACTIONS_DOC`;
  - `ForumAgentPolicy(..., system_template: str = FORUM_SYSTEM, actions_doc: str = ACTIONS_DOC)`, where `topics` items are preformatted lines (name, or `"name — definition"`).

  Old-mode defaults are unchanged.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_gated_prompt_states_the_hard_rule_and_uses_given_order():
    from innovation.core.llm import FakeLLM
    from innovation.p2_forum.agent import GATED_ACTIONS_DOC, GATED_SYSTEM, ForumAgentPolicy
    pol = ForumAgentPolicy(llm=FakeLLM(default='{"action":"search","args":{"query":"x"}}'),
                           model="m", topics=["B — def b", "A — def a"],
                           system_template=GATED_SYSTEM, actions_doc=GATED_ACTIONS_DOC)
    assert pol.system.index("B — def b") < pol.system.index("A — def a")
    assert "will not be published" in pol.system
    pol.act({"step": 0, "last_result": {}})
    assert GATED_ACTIONS_DOC.splitlines()[1] in pol.llm.calls[0]["user"]
```

- [ ] **Step 2: Run, expect FAIL** (`ImportError: GATED_SYSTEM`)

Run: `uv run pytest tests/test_forum_agent.py -q -k gated`

- [ ] **Step 3: Implement**

Add the two constants and the two keyword arguments. In `__init__`, use `self.system = system_template.format(topics=bullets)` and `self.actions_doc = actions_doc`. In `act`, use `self.actions_doc` instead of `ACTIONS_DOC`.

```python
GATED_SYSTEM = """You are a research agent. Two things are in front of you.

The LITERATURE is published research you can search online: papers from top AI \
venues, or highly cited papers, up to September 2024. Each search or read shows \
a paper's title, abstract, topics and its references and citations.

The BOARD is a shared space where you and other agents publish new ideas. Anyone \
may adjust the reference links on any post. It starts empty.

Your goal is to find promising unexplored directions and publish genuinely new \
ideas to the board. Ground them: cite the papers they build on, and cite other \
agents' posts when your idea builds on theirs.

You work ONLY within your topics. Searches outside them are refused, papers and \
posts outside them are hidden from you, and an idea outside them will not be \
published.

Your topics:
{topics}"""

GATED_ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- search the literature online
{"action": "browse", "args": {"node_id": "<paper id>"}} -- read a paper's abstract, references and citations
{"action": "sample_frontier", "args": {}} -- jump to a random paper in one of your topics
{"action": "search_board", "args": {"query": "<text>", "k": 5}} -- semantic search over the board
{"action": "browse_board", "args": {"node_id": "<post id>"}} -- read a post and its reference neighbors
{"action": "sample_board", "args": {}} -- jump to a random post
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- publish your new idea to the board, citing what it builds on (papers or posts)
{"action": "add_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- add reference links from a post to what it builds on
{"action": "remove_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- remove reference links from a post that do not actually support it"""
```

Update the module docstring: specialization is soft in `corpus` mode and hard in `online` mode (`gated_env.py`).

- [ ] **Step 4: Run tests, expect pass**

Run: `uv run pytest tests/test_forum_agent.py -q`
Expected: all pass, the old ones included.

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/agent.py tests/test_forum_agent.py
git commit -m "agent: gated online prompt; injectable system template and actions doc"
```

---

### Task 9: Runner, CLI and config wiring

**Files:**
- Modify: `src/innovation/p2_forum/runner.py`, `src/innovation/cli.py`, `src/innovation/p2_forum/board_metrics.py`
- Create: `configs/p2_forum/base-online.yaml`
- Modify: `scripts/gen_forum_configs.py` (append an online block)
- Test: `tests/test_forum_runner.py`, `tests/test_forum_cli_dispatch.py` (append)

**Interfaces:**
- Consumes: everything above; the frozen `configs/p2_forum/topics-v2.yaml` (Task 2).
- Produces:
  - `ForumRunConfig.literature: str = "corpus"` (`"corpus"|"online"`) and `ForumRunConfig.gating: str = "none"` (`"none"|"topics"`). `"online"` requires `"topics"`, and `"topics"` requires `"online"`; validate this in `__post_init__`.
  - `ForumRunConfig.topic_definitions: dict[str, str] = {}`
  - `run_forum` / `resume_forum` gain `literature=None, tagger=None` kwargs; `corpus` and `corpus_index` become optional (`None` in online mode).
  - `display_order(topics: list[str], seed: int, agent_index: int) -> list[str]` (salt `0xD15B1A`).
  - CLI `_load_online_world(cfg) -> (literature, tagger, embedder)`.
  - Configs `configs/p2_forum/experiments/online/k{k}-s0.yaml` (one seed, seed 0), run_id `forum-online-k{k}-s0`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_forum_runner.py`:

```python
def test_display_order_is_a_seeded_shuffle_independent_of_nesting():
    from innovation.p2_forum.runner import display_order
    t = [f"T{i}" for i in range(16)]
    a = display_order(t, seed=0, agent_index=0)
    assert sorted(a) == sorted(t) and a != t
    assert a == display_order(t, seed=0, agent_index=0)
    assert a != display_order(t, seed=0, agent_index=1)


def test_mode_combinations_are_validated():
    import pytest
    from innovation.p2_forum.runner import ForumRunConfig
    with pytest.raises(ValueError):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, literature="online", gating="none")
    with pytest.raises(ValueError):
        ForumRunConfig(run_id="r", seed=0, total_steps=1, literature="corpus", gating="topics")


def test_online_run_end_to_end_with_fakes(tmp_path):
    """Two agents, fake literature + tagger + LLM: the run writes run_meta with
    topic ids and display orders, and every logged search hit is readable."""
    import json
    from innovation.core.llm import FakeLLM
    from innovation.p2_forum.runner import ForumRunConfig, run_forum
    from tests.test_forum_gated_env import FakeLit, FakeTagger, NAMES
    from tests.conftest import FakeEmbedder
    cfg = ForumRunConfig(run_id="r", seed=0, total_steps=6, literature="online",
                         gating="topics", topic_draw="nested", topic_pool=NAMES,
                         agents=[{"agent_id": "a", "k_topics": 1},
                                 {"agent_id": "b", "k_topics": 1}])
    llm = FakeLLM(default='{"action": "search", "args": {"query": "T0 T1 T2 T3"}}')
    run_forum(cfg, embedder=FakeEmbedder(), llm=llm, model="m", out_dir=tmp_path,
              literature=FakeLit(), tagger=FakeTagger())
    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["literature"] == "online" and meta["gating"] == "topics"
    assert set(meta["topic_ids"]) == {"a", "b"}
```

Append to `tests/test_forum_cli_dispatch.py`:

```python
def test_online_configs_load_and_are_nested():
    from innovation.core.config import load_config
    files = sorted(Path("configs/p2_forum/experiments/online").glob("*.yaml"))
    assert len(files) == 9
    cfg = load_config(files[0])
    assert cfg["literature"] == "online" and cfg["gating"] == "topics"
    assert cfg["run"]["topic_draw"] == "nested" and cfg["run"]["total_steps"] == 400
    assert cfg["models"]["tagger"] == "claude-sonnet-5"
    assert cfg["models"]["judge"] == "claude-opus-5-5"
    assert cfg["online"]["max_pub_date"] == "2024-09-30"
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_forum_runner.py tests/test_forum_cli_dispatch.py -q`
Expected: the new tests FAIL.

- [ ] **Step 3: Implement the runner changes**

In `runner.py`:

```python
LITERATURES = ("corpus", "online")
GATINGS = ("none", "topics")
_DISPLAY_SALT = 0xD15B1A


def display_order(topics: list[str], seed: int, agent_index: int) -> list[str]:
    """Prompt display order, shuffled independently of the nested draw so the
    first-listed topics are not the small-k topics (spec §7)."""
    own = np.random.default_rng(np.random.SeedSequence([int(seed), agent_index, _DISPLAY_SALT]))
    return [topics[int(j)] for j in own.permutation(len(topics))]
```

- Add the fields `literature`, `gating` and `topic_definitions` to `ForumRunConfig`. In `__post_init__`, validate membership and that `(literature == "online") == (gating == "topics")`.
- `_build_env`: if `cfg.literature == "online"`:
  - build `Workspace(corpus=<empty frozen IdeaGraph>, corpus_index=VectorIndex(embedder.dim), board_index=VectorIndex(embedder.dim), embedder=embedder, run_id=cfg.run_id, external_papers=literature)`;
  - return `GatedForumEnvironment(run_id=..., workspace=ws, event_log=..., rng=rng, generation_budget=cfg.generation_budget, navigation=cfg.navigation, literature=literature, tagger=tagger, agent_topics=topic_ids, topic_names=cfg.topic_pool)`.

  Otherwise keep the current path.
- `topic_ids = {aid: {cfg.topic_pool.index(t) for t in names} for aid, names in assignments.items()}`
- `_build_policies`:
  - online: for agent index `i`, `lines = [f"{t} — {cfg.topic_definitions.get(t, '')}" for t in display_order(names, cfg.seed, i)]`, with `system_template=GATED_SYSTEM, actions_doc=GATED_ACTIONS_DOC`;
  - corpus mode: unchanged.
- `run_meta.json` additionally records `"literature"`, `"gating"`, `"topic_ids"` (sorted lists) and `"display_orders"`.
- `resume_forum`: same wiring; `env.restore(events)` dispatches to the gated restore automatically. Reuse the recorded `topic_ids`.

- [ ] **Step 4: Implement the CLI changes**

In `cli.py`:

```python
def _load_online_world(cfg):
    from innovation.p2_forum.literature import OnlineLiterature, Scope
    from innovation.p2_forum.s2_online import S2Online
    from innovation.p2_forum.tagger import TopicTagger
    from innovation.p2_forum.topics import load_topics

    on = cfg["online"]
    topics = load_topics(cfg["topics_file"])
    llm = CachedLLM(RoutedLLM(), Path(on["cache_dir"]) / "llm")
    tagger = TopicTagger(llm=llm, model=cfg["models"]["tagger"], topics=topics)
    aliases = tuple(a.lower() for v in on["venues"] for a in v["aliases"])
    scope = Scope(venue_aliases=aliases, min_citations=on["min_citations_any_venue"],
                  max_date=on["max_pub_date"])
    lit = OnlineLiterature(client=S2Online(on["cache_dir"]), scope=scope, tagger=tagger,
                           cache_dir=on["cache_dir"], search_pool=on.get("search_pool", 50))
    return lit, tagger, Embedder(cfg["embedding_model"]), topics
```

In `cmd_run`'s p2 branch:
- when `cfg.get("literature") == "online"`, call `_load_online_world`;
- build `ForumRunConfig(..., literature="online", gating="topics", topic_pool=[t.name for t in topics], topic_definitions={t.name: t.definition for t in topics})`;
- call `forum(run_cfg, corpus=None, corpus_index=None, embedder=emb, llm=_llm(cfg), model=cfg["models"]["agent"], out_dir=cfg["out_dir"], literature=lit, tagger=tagger)`.

`_write_board_metrics` in online mode passes `corpus=None, corpus_index=None, embedder=emb, external_papers=<object whose has() returns True for any non-'gen:' id>`.

- [ ] **Step 5: Implement the board-metrics change**

In `board_metrics.board_trajectory`:
- add the kwarg `external_papers=None`;
- when it is given, build the Workspace from an empty frozen `IdeaGraph()` plus `VectorIndex(embedder.dim)` with `external_papers=external_papers`;
- the replay goes through the base `ForumEnvironment.restore`, which posts the raw `cited_ids`; for online runs those include ids the gate dropped. Make the base restore subtract `result.get("dropped_cites", [])` before posting (no-op for old runs, which never log it), via a small helper `_kept_cites(e)` that `GatedForumEnvironment.restore` also uses.

Add a test to `tests/test_board_metrics.py`: an online-style event list (a post citing an S2 id and a dropped id) replays without a corpus and yields one post→paper edge.

- [ ] **Step 6: Write `configs/p2_forum/base-online.yaml`**

```yaml
# Paper 2, online literature + hard topic gating
# (docs/superpowers/specs/2026-09-30-online-literature-topic-gating-design.md).
extends: base.yaml
literature: online
gating: topics
topics_file: configs/p2_forum/topics-v2.yaml
online:
  cache_dir: data/online_cache
  max_pub_date: "2024-09-30"
  min_citations_any_venue: 50
  search_pool: 50
  venues:
    - {name: AAAI, aliases: ["aaai conference on artificial intelligence", "aaai"]}
    - {name: NeurIPS, aliases: ["neurips", "neural information processing"]}
    - {name: ACL, aliases: ["annual meeting of the association for computational linguistics"]}
    - {name: CVPR, aliases: ["computer vision and pattern recognition", "cvpr"]}
    - {name: ICCV, aliases: ["international conference on computer vision", "iccv"]}
    - {name: ICML, aliases: ["international conference on machine learning", "icml"]}
    - {name: ICLR, aliases: ["international conference on learning representations", "iclr"]}
models:
  agent: openai:gpt-5:medium
  tagger: claude-sonnet-5
  judge: claude-opus-5-5
```

Check that the bare `"aaai"` alias does not admit non-main-track venues seen in practice (for example "AAAI Spring Symposium"). The live check in Task 12 lists the distinct venue strings admitted through the venue branch; tighten the aliases there if needed.

- [ ] **Step 7: Generate the online configs**

Append to `scripts/gen_forum_configs.py` a block mirroring the nested one:
- `ONLINE_KS = NESTED_KS`, `ONLINE_SEEDS = [0]` (one seed per k, user decision 2026-10-01), `ONLINE_ROUNDS = 40`;
- write to `configs/p2_forum/experiments/online/k{k}-s{s}.yaml`;
- content: `extends: ../../base-online.yaml`, `run: {run_id: forum-online-k{k}-s{s}, seed: s, total_steps: 400, topic_draw: nested, agents: [...]}`.

Run: `uv run python scripts/gen_forum_configs.py`. Commit the 9 new files.

- [ ] **Step 8: Run the whole suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/innovation/p2_forum/runner.py src/innovation/cli.py src/innovation/p2_forum/board_metrics.py configs/p2_forum/base-online.yaml configs/p2_forum/experiments/online scripts/gen_forum_configs.py tests/
git commit -m "paper 2: online+gated mode wired through runner, CLI, board metrics; 9 online sweep configs (one seed)"
```

---

### Task 10: Evaluation for online runs

**Files:**
- Modify: `src/innovation/cli.py` (`cmd_evaluate`)
- Test: `tests/test_forum_cli.py` (append)

**Interfaces:**
- Produces: `_eval_reference(cfg) -> (corpus_titles: set[str], corpus_vecs: np.ndarray)`.
  - Online mode: the titles and embeddings of every paper in the online cache that **any agent of this run was shown** (collected from the run's event log: hit and browse node ids that are not `gen:` ids, plus `cites`/`cited_by` items), embedded from `Paper.text()`.
  - Corpus mode: the current behavior.
  - The judge model comes from `cfg["models"]["judge"]` (already the case; base-online sets `claude-opus-5-5`).

- [ ] **Step 1: Write the failing test**

```python
def test_online_eval_reference_collects_shown_papers(tmp_path, monkeypatch):
    import json
    from innovation import cli
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    ev = [{"action": "search", "result": {"hits": [{"node_id": "p0", "title": "A", "text": "x"}]}},
          {"action": "browse", "result": {"node_id": "p1", "title": "B", "text": "y",
                                          "cites": [{"node_id": "p2", "title": "C"}], "cited_by": []}},
          {"action": "generate", "result": {"node_id": "gen:r:0", "topics": [0]}}]
    (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    cfg = {"literature": "online", "out_dir": str(tmp_path / "runs"), "run": {"run_id": "r"},
           "embedding_model": "fake"}
    monkeypatch.setattr(cli, "Embedder", lambda name: __import__("tests.conftest", fromlist=["FakeEmbedder"]).FakeEmbedder())
    titles, vecs = cli._eval_reference(cfg)
    assert titles == {"a", "b", "c"} and vecs.shape[0] == 3
```

- [ ] **Step 2: Run, expect FAIL** (`AttributeError: _eval_reference`)

- [ ] **Step 3: Implement**

Walk the events.
- For `search` results, take `hits`.
- For `browse` results, take the view itself plus `cites` and `cited_by`.
- Skip `sample_frontier` results only if they lack a `title`; take them otherwise.
- Skip ids starting with `gen:`.
- Build `text = title + "\n\n" + text` from what was logged (the agent saw exactly this).
- Return the lowercased stripped titles, and `Embedder(cfg["embedding_model"]).encode(texts)`.

In `cmd_evaluate`:
- if `cfg.get("literature") == "online"`, use `_eval_reference(cfg)` and do not call `_load_world`;
- keep `verify_idea(...)` and `aggregate_run(...)` unchanged;
- `past_dup_flag` gets the online `corpus_vecs`.

- [ ] **Step 4: Run tests, expect pass; then commit**

```bash
git add src/innovation/cli.py tests/test_forum_cli.py
git commit -m "evaluate: online runs use the papers agents were shown as the contamination/dup reference"
```

---

### Task 11: Gate and tagging measurements

**Files:**
- Create: `scripts/p2_forum/gate_stats.py`, `scripts/p2_forum/tag_quality.py`
- Test: `tests/test_forum_gate_stats.py`

**Interfaces:**
- Produces:
  - `gate_counts(events) -> dict`. Keys: `query`, `result`, `post`, `unlabeled`, `link`, `dropped_cites`, `posts`, `searches`, `rejection_rate` (= post refusals / (post refusals + posts)).
  - `tag_quality.py` prints the labels-per-item histogram over `data/online_cache/labels.jsonl`, the share of items labeled 5, and re-label agreement on a random 200-paper sample (fresh calls through an uncached `RoutedLLM`, retry suffix `(stability)`): mean Jaccard and primary-label agreement.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forum_gate_stats.py
import importlib.util
from pathlib import Path


def load():
    spec = importlib.util.spec_from_file_location("gs", Path("scripts/p2_forum/gate_stats.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gate_counts():
    ev = [{"action": "search", "result": {"gate": "query"}},
          {"action": "search", "result": {"hits": []}},
          {"action": "generate", "result": {"gate": "post"}},
          {"action": "generate", "result": {"node_id": "gen:r:0", "dropped_cites": ["x", "y"]}},
          {"action": "add_links", "result": {"gate": "link"}}]
    c = load().gate_counts(ev)
    assert c["query"] == 1 and c["post"] == 1 and c["link"] == 1
    assert c["dropped_cites"] == 2 and c["posts"] == 1 and c["searches"] == 2
    assert c["rejection_rate"] == 0.5
```

- [ ] **Step 2: Run, expect FAIL; Step 3: implement**

```python
# scripts/p2_forum/gate_stats.py
"""Gate activity per online run (spec §10). Run: uv run python scripts/p2_forum/gate_stats.py"""
import collections
import json
from pathlib import Path


def gate_counts(events) -> dict:
    c = collections.Counter()
    for e in events:
        r = e.get("result", {})
        if "gate" in r:
            c[r["gate"]] += 1
        if e["action"] in ("search", "search_board"):
            c["searches"] += 1
        if e["action"] == "generate" and "node_id" in r:
            c["posts"] += 1
            c["dropped_cites"] += len(r.get("dropped_cites", []))
    out = {k: c.get(k, 0) for k in ("query", "result", "post", "unlabeled", "link",
                                    "dropped_cites", "posts", "searches")}
    tried = out["post"] + out["posts"]
    out["rejection_rate"] = out["post"] / tried if tried else 0.0
    return out


if __name__ == "__main__":
    rows = []
    for d in sorted(Path("runs/p2_forum").glob("forum-online-k*-s*")):
        ev = [json.loads(l) for l in open(d / "events.jsonl")]
        rows.append({"run": d.name, **gate_counts(ev)})
    json.dump(rows, open("runs/p2_forum/gate_stats.json", "w"), indent=1)
    for r in rows:
        print(r)
```

Write `tag_quality.py` as specified in the interface. It is a plain script with no test, because its only logic is counting and two set metrics.

- [ ] **Step 4: Run tests, expect pass; commit**

```bash
git add scripts/p2_forum/gate_stats.py scripts/p2_forum/tag_quality.py tests/test_forum_gate_stats.py
git commit -m "paper 2: gate-activity and tagging-quality measurements"
```

---

### Task 12: Live smoke run, then the sweep (operational; needs user approval for cost)

**Files:** none are created; runs land in `runs/p2_forum/`.

- [ ] **Step 1: Smoke run (20 steps, one config)**

Run: `uv run python -m innovation.cli run --config configs/p2_forum/experiments/online/k16-s0.yaml --steps 20 --run-id forum-online-smoke`
Expected: the run completes and writes `events.jsonl`, `run_meta.json` and `board_metrics.json`.

- [ ] **Step 2: Inspect the smoke run and report to the user**
  - `uv run python scripts/p2_forum/gate_stats.py` (it globs `forum-online-k*`; temporarily point it at `forum-online-smoke` or pass the path);
  - the distinct venue strings admitted via the venue branch. Tighten the aliases if any are not main-track;
  - wall-clock per step, and the number of tagger calls (`ls data/online_cache/llm | wc -l` before and after);
  - a sample of three search results and one browse, to confirm abstracts, topics and filtered references look right.

  Report the projected cost and time for 9 × 400 steps, then **wait for the user's go-ahead.**

- [ ] **Step 3: Launch the sweep (after approval)**

```bash
python3 scripts/detach.py scripts/p2_forum/run_queue.sh 4 $(for k in 1 16 32 48 64 80 96 112 128; do echo configs/p2_forum/experiments/online/k$k-s0.yaml; done)
```

Parallelism is 4, not 9, because of the Semantic Scholar rate limit. Raise it only if the smoke run shows headroom. Monitor `runs/p2_forum/logs/forum-online-*.log` for `Traceback|EXIT=[1-9]`, and resume failed runs with `-- --resume --steps 400`.

- [ ] **Step 4: Evaluate (after the user approves the judge cost)**

```bash
CMD=evaluate python3 scripts/detach.py scripts/p2_forum/run_queue.sh 3 configs/p2_forum/experiments/online/*.yaml
```

---

## Self-review notes

- **Spec coverage:**

  | spec section | task |
  |---|---|
  | §2 scope | Tasks 5, 9 |
  | §3 topic list | Tasks 1-2 |
  | §4 tagger | Task 3; stubs in Task 7 via `lit.labels` |
  | §5 online tools and cache | Tasks 4, 5, 7 |
  | §6 board gates | Task 7 |
  | §7 nested draw and display shuffle | Task 9 |
  | §8 config | Task 9 |
  | §9 experiment and evaluation | Tasks 10, 12 |
  | §10 measurements | Task 11 |
  | §11 rate limits, venue matching, missing abstracts | Tasks 4, 5, 12 |
  | §12 tests | Tasks 3-11 |

- The spec's "board stubs take the labels of the paper they mirror" holds because `_labels()` routes non-`gen:` ids to `lit.labels`.
- Resume makes zero tagger calls for replayed posts (Task 7 test). Replayed literature reads make no network calls at all, because the board replay never touches the literature.
