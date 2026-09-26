# Shared Board Architecture — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the simulation state into a frozen corpus and a separate shared board that every agent reads and writes, so agent-to-agent structure can emerge and be measured.

**Architecture:** Two `IdeaGraph` instances and two `VectorIndex` instances behind a single `Workspace` facade. The corpus is frozen after load; the board starts empty and takes every agent write. Agents get six navigation actions (three per store) and three write actions that target the board only. Specialization is soft: `k` topics in the system prompt, with no environment-side filtering of any kind. The repo splits into `core` (shared), `p1_dial` (paper 1, unchanged behaviour), and `p2_forum` (this work).

**Tech Stack:** Python 3.12+, networkx, numpy, pandas, sentence-transformers, scikit-learn, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-09-26-forum-architecture-design.md`

## Global Constraints

- Package names use underscores: `innovation.core`, `innovation.p1_dial`, `innovation.p2_forum`. Directory names under `configs/`, `runs/`, `paper/` use the same underscored names.
- Store names in code and configs are exactly `"corpus"` and `"board"`.
- Stub nodes for cross-store edge targets carry `IdeaNode.source == "corpus_ref"`.
- Board node ids keep paper 1's format: `gen:<run_id>:<n>`.
- Action names are exactly: `search`, `browse`, `sample_frontier`, `search_board`, `browse_board`, `sample_board`, `generate`, `add_links`, `remove_links`.
- Config key `arch` selects the track; its values are exactly `p1_dial` and `p2_forum`. Absent `arch` means `p1_dial` (paper 1's configs are not edited).
- Agent config key for the topic count is `k_topics` (an integer).
- The topic pool is 128 topics at `data/stage1/topics_k128.json`, surfaced as `configs/p2_forum/topics-k128.yaml`.
- `data/stage1/` is NOT renamed. Both tracks read it.
- The environment filters nothing. No scope check may appear in `p2_forum`.
- Edge writes require the source node to be in the board. Corpus-internal edges can never be added or removed. There is no ownership check — any agent may edit any board post's links.
- Headline metric and evaluation pipeline are paper 1's, unchanged: `acc@≥2`, `realized_min_date = "2025-06-01"`.
- Run `uv run pytest` after every task. It must be green before committing.

---

### Task 1: Repository split into core / p1_dial

Mechanical migration. Paper 1's behaviour must not change; its test suite is the proof.

**Files:**
- Create: `src/innovation/core/__init__.py`, `src/innovation/p1_dial/__init__.py`, `src/innovation/p1_dial/agents/__init__.py`
- Move: see the table in Step 2
- Modify: `scripts/gen_topic_configs.py`, `tests/test_cli.py:19`, `tests/test_cli.py:288`
- Test: the existing suite under `tests/`

**Interfaces:**
- Consumes: nothing.
- Produces: import paths `innovation.core.network.graph`, `innovation.core.network.index`, `innovation.core.ideas.embed`, `innovation.core.eval.metrics`, `innovation.core.eval.search_verify`, `innovation.core.events`, `innovation.core.policy`, `innovation.core.llm`, `innovation.core.config`, `innovation.p1_dial.env`, `innovation.p1_dial.runner`, `innovation.p1_dial.agents.llm_agent`, `innovation.p1_dial.agents.baselines`. Every later task imports from these paths.

- [ ] **Step 1: Confirm the starting state is green**

Run: `uv run pytest -q`
Expected: all tests pass. If not, stop and report — do not migrate a red suite.

- [ ] **Step 2: Move the modules with `git mv`**

| From | To |
|---|---|
| `src/innovation/config.py` | `src/innovation/core/config.py` |
| `src/innovation/llm.py` | `src/innovation/core/llm.py` |
| `src/innovation/experiments/events.py` | `src/innovation/core/events.py` |
| `src/innovation/agents/policy.py` | `src/innovation/core/policy.py` |
| `src/innovation/data/` | `src/innovation/core/data/` |
| `src/innovation/ideas/` | `src/innovation/core/ideas/` |
| `src/innovation/network/` | `src/innovation/core/network/` |
| `src/innovation/eval/` | `src/innovation/core/eval/` |
| `src/innovation/analysis/` | `src/innovation/core/analysis/` |
| `src/innovation/experiments/env.py` | `src/innovation/p1_dial/env.py` |
| `src/innovation/experiments/runner.py` | `src/innovation/p1_dial/runner.py` |
| `src/innovation/agents/llm_agent.py` | `src/innovation/p1_dial/agents/llm_agent.py` |
| `src/innovation/agents/baselines.py` | `src/innovation/p1_dial/agents/baselines.py` |

```bash
cd src/innovation
mkdir -p core p1_dial/agents
git mv config.py llm.py core/
git mv experiments/events.py core/events.py
git mv agents/policy.py core/policy.py
git mv data ideas network eval analysis core/
git mv experiments/env.py p1_dial/env.py
git mv experiments/runner.py p1_dial/runner.py
git mv agents/llm_agent.py agents/baselines.py p1_dial/agents/
git rm experiments/__init__.py agents/__init__.py
rmdir experiments agents
touch core/__init__.py p1_dial/__init__.py p1_dial/agents/__init__.py
git add core/__init__.py p1_dial/__init__.py p1_dial/agents/__init__.py
```

- [ ] **Step 3: Rewrite every import**

Apply these substitutions across `src/`, `tests/`, and `scripts/`:

```bash
cd /path/to/repo
FILES=$(git ls-files 'src/*.py' 'tests/*.py' 'scripts/*.py')
sed -i '' \
  -e 's/innovation\.experiments\.events/innovation.core.events/g' \
  -e 's/innovation\.experiments\.env/innovation.p1_dial.env/g' \
  -e 's/innovation\.experiments\.runner/innovation.p1_dial.runner/g' \
  -e 's/innovation\.agents\.policy/innovation.core.policy/g' \
  -e 's/innovation\.agents\.llm_agent/innovation.p1_dial.agents.llm_agent/g' \
  -e 's/innovation\.agents\.baselines/innovation.p1_dial.agents.baselines/g' \
  -e 's/innovation\.config/innovation.core.config/g' \
  -e 's/innovation\.llm/innovation.core.llm/g' \
  -e 's/innovation\.data/innovation.core.data/g' \
  -e 's/innovation\.ideas/innovation.core.ideas/g' \
  -e 's/innovation\.network/innovation.core.network/g' \
  -e 's/innovation\.eval/innovation.core.eval/g' \
  -e 's/innovation\.analysis/innovation.core.analysis/g' \
  $FILES
```

(On GNU sed drop the `''` after `-i`.)

- [ ] **Step 4: Move configs, runs and paper directories**

```bash
mkdir -p configs/p1_dial runs/p1_dial paper/p1_dial
git mv configs/stage1.yaml configs/topics-k50.yaml configs/experiments configs/p1_dial/
for d in runs/*/; do git mv "$d" runs/p1_dial/; done
git mv runs/*.png runs/*.json runs/p1_dial/
git mv paper/* paper/p1_dial/
```

- [ ] **Step 5: Fix the three hardcoded paths**

In `scripts/gen_topic_configs.py`, replace every `configs/` literal with `configs/p1_dial/` and add a module docstring line: `Paper 1 only; paper 2's pool is generated by scripts/gen_topics.py.`

In `tests/test_cli.py:19`, `load_config("configs/stage1.yaml")` becomes `load_config("configs/p1_dial/stage1.yaml")`.

In `tests/test_cli.py:288`, `Path("configs/experiments")` becomes `Path("configs/p1_dial/experiments")`.

Also update `configs/p1_dial/stage1.yaml`: `out_dir: runs` becomes `out_dir: runs/p1_dial`, and `topics_file: configs/topics-k50.yaml` becomes `topics_file: configs/p1_dial/topics-k50.yaml`.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS, with the same number of tests as Step 1.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: split the package into core / p1_dial ahead of paper 2

Behaviour-preserving move. Shared machinery (graph, index, embeddings, LLM
client, evaluation, event log, Policy ABC) goes to innovation.core; paper 1's
environment, runner and agents go to innovation.p1_dial. Configs, runs and
paper sources move under p1_dial/ so paper 2 gets a sibling rather than a
subdirectory.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: 128-topic pool

No clustering code exists in the repo — `topics_k50.json` was produced ad hoc. This task writes the generator so the pool is reproducible.

**Files:**
- Create: `scripts/gen_topics.py`, `data/stage1/topics_k128.json`, `configs/p2_forum/topics-k128.yaml`
- Modify: `pyproject.toml` (declare `scikit-learn`)
- Test: `tests/test_gen_topics.py`

**Interfaces:**
- Consumes: `innovation.core.ideas.embed.load_embeddings`, `innovation.core.ideas.summarize.load_ideas`, `innovation.core.llm.CachedLLM`.
- Produces: `cluster_topics(vecs, k, seed) -> list[list[int]]` returning row indices per cluster; a JSON file that is a list of `{"topic": str, "size": int}`, matching `topics_k50.json`'s shape exactly so `cli.py`'s existing reader works unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gen_topics.py
import numpy as np

from scripts.gen_topics import cluster_topics


def test_cluster_topics_partitions_every_row_exactly_once():
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(60, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    clusters = cluster_topics(vecs, k=5, seed=0)

    assert len(clusters) == 5
    flat = sorted(i for c in clusters for i in c)
    assert flat == list(range(60))
    assert all(len(c) > 0 for c in clusters)


def test_cluster_topics_is_deterministic():
    rng = np.random.default_rng(1)
    vecs = rng.normal(size=(40, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    assert cluster_topics(vecs, k=4, seed=7) == cluster_topics(vecs, k=4, seed=7)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_gen_topics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.gen_topics'`

- [ ] **Step 3: Declare scikit-learn**

In `pyproject.toml`, add `"scikit-learn>=1.5",` to `dependencies`. Then run `uv sync`.

- [ ] **Step 4: Write the generator**

```python
# scripts/gen_topics.py
"""Generate a topic pool by clustering corpus idea embeddings and naming each
cluster with an LLM. Paper 2 uses K=128 (configs/p2_forum/topics-k128.yaml).

Run: uv run python scripts/gen_topics.py --k 128
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def cluster_topics(vecs: np.ndarray, k: int, seed: int = 0) -> list[list[int]]:
    """Partition rows of `vecs` into k clusters; returns row indices per
    cluster, ordered largest first. Deterministic for a given seed."""
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(vecs)
    groups = [np.flatnonzero(labels == c).tolist() for c in range(k)]
    return sorted(groups, key=len, reverse=True)


NAME_SYSTEM = ("You name research topics. Given several paper-idea paragraphs "
               "from one cluster, reply with a single noun phrase of 4-8 words "
               "naming what they have in common. Reply with the phrase only.")


def name_cluster(llm, model: str, texts: list[str]) -> str:
    sample = "\n\n".join(t[:400] for t in texts[:8])
    reply = llm.complete(model=model, system=NAME_SYSTEM,
                         user=sample, max_tokens=60)
    return reply.strip().strip('."')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--data-dir", default="data/stage1")
    ap.add_argument("--config", default="configs/p2_forum/topics-k128.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from innovation.core.config import load_config, load_env
    from innovation.core.ideas.embed import load_embeddings
    from innovation.core.ideas.summarize import load_ideas
    from innovation.core.llm import CachedLLM, RoutedLLM

    load_env()
    cfg = load_config("configs/p1_dial/stage1.yaml")
    ids, vecs = load_embeddings(args.data_dir)
    ideas = load_ideas(args.data_dir)
    text_by_id = dict(zip(ideas["paper_id"], ideas["idea_text"]))

    clusters = cluster_topics(vecs, args.k, args.seed)
    llm = CachedLLM(RoutedLLM(), Path(args.data_dir) / "llm_cache")
    model = cfg["models"]["summarizer"]

    pool = []
    for rows in clusters:
        texts = [text_by_id[ids[i]] for i in rows]
        pool.append({"topic": name_cluster(llm, model, texts), "size": len(rows)})

    out = Path(args.data_dir) / f"topics_k{args.k}.json"
    out.write_text(json.dumps(pool, indent=1))
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"topics": pool}, open(args.config, "w"), allow_unicode=True)
    print(f"{len(pool)} topics -> {out} and {args.config}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `uv run pytest tests/test_gen_topics.py -q`
Expected: PASS

- [ ] **Step 6: Generate the real pool**

Run: `uv run python scripts/gen_topics.py --k 128`
Expected: `128 topics -> data/stage1/topics_k128.json and configs/p2_forum/topics-k128.yaml`

Sanity-check by eye: `python3 -c "import json;p=json.load(open('data/stage1/topics_k128.json'));print(len(p));print([t['topic'] for t in p[:5]])"`. Names should read like `topics_k50.json`'s — short noun phrases, no duplicates. If many names repeat, raise `--k` sampling breadth by editing `texts[:8]` to `texts[:12]` and regenerate.

- [ ] **Step 7: Commit**

```bash
git add scripts/gen_topics.py tests/test_gen_topics.py pyproject.toml uv.lock \
        data/stage1/topics_k128.json configs/p2_forum/topics-k128.yaml
git commit -m "feat: reproducible topic-pool generator; 128-topic pool for paper 2

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Freeze the corpus graph

**Files:**
- Modify: `src/innovation/core/network/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: `IdeaGraph` from Task 1's path.
- Produces: `FrozenGraphError` (a `RuntimeError` subclass) and `IdeaGraph.freeze() -> None`. After `freeze()`, `add_idea`, `add_links` and `remove_links` all raise `FrozenGraphError`. `IdeaGraph.frozen` is a bool property.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_graph.py
import pytest

from innovation.core.network.graph import FrozenGraphError, IdeaGraph


def _two_node_graph() -> IdeaGraph:
    g = IdeaGraph()
    g.add_idea("a", "idea a", [], source="corpus", year=2020)
    g.add_idea("b", "idea b", ["a"], source="corpus", year=2021)
    return g


def test_freeze_blocks_every_mutation():
    g = _two_node_graph()
    g.freeze()

    assert g.frozen is True
    with pytest.raises(FrozenGraphError):
        g.add_idea("c", "idea c", ["a"])
    with pytest.raises(FrozenGraphError):
        g.add_links("b", ["a"])
    with pytest.raises(FrozenGraphError):
        g.remove_links("b", ["a"])


def test_freeze_leaves_reads_working():
    g = _two_node_graph()
    g.freeze()

    assert g.num_nodes == 2
    assert g.citations_out("b") == ["a"]
    assert g.node("a").text == "idea a"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_graph.py -q`
Expected: FAIL with `ImportError: cannot import name 'FrozenGraphError'`

- [ ] **Step 3: Implement**

In `src/innovation/core/network/graph.py`, add above the `IdeaGraph` class:

```python
class FrozenGraphError(RuntimeError):
    """Raised on any attempt to mutate a frozen graph (the read-only corpus)."""
```

In `IdeaGraph.__init__`, add `self._frozen = False` after the `self._g = ...` line. Then add:

```python
    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Make this graph permanently read-only. There is no unfreeze: the
        corpus is historical fact, and paper 2's claim rests on it."""
        self._frozen = True

    def _check_mutable(self) -> None:
        if self._frozen:
            raise FrozenGraphError("this graph is frozen (read-only corpus)")
```

Add `self._check_mutable()` as the first statement of `add_idea`, `add_links` and `remove_links`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS — paper 1 never calls `freeze()`, so nothing regresses.

- [ ] **Step 6: Commit**

```bash
git add src/innovation/core/network/graph.py tests/test_graph.py
git commit -m "feat: IdeaGraph.freeze() makes read-only a property of the type

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Workspace — two stores behind one facade

**Files:**
- Create: `src/innovation/p2_forum/__init__.py`, `src/innovation/p2_forum/workspace.py`
- Test: `tests/test_workspace.py`

**Interfaces:**
- Consumes: `IdeaGraph`, `FrozenGraphError`, `IdeaNode` from `innovation.core.network.graph`; `VectorIndex` from `innovation.core.network.index`.
- Produces: `Workspace(corpus, corpus_index, board_index, embedder, run_id)` with `store_of(node_id) -> str`, `has_node(node_id) -> bool`, `node(node_id) -> IdeaNode`, `corpus_search(vec, k) -> list[tuple[str, float]]`, `board_search(vec, k) -> list[tuple[str, float]]`, `corpus_neighbors(node_id) -> tuple[list[str], list[str]]`, `board_neighbors(node_id) -> tuple[list[str], list[str]]`, `corpus_sample(rng) -> str`, `board_sample(rng) -> str | None`, `post_idea(text, cited_ids, meta, node_id=None) -> str`, `add_links(src_id, dst_ids, meta) -> dict`, `remove_links(src_id, dst_ids) -> dict`, `board_post_ids() -> list[str]`, and the attribute `board` (an `IdeaGraph`). `add_links` returns `{"added": [...], "skipped": [...]}` and `remove_links` returns `{"removed": [{"dst_id": ..., "etype": ...}], "skipped": [...]}` — the same shapes paper 1's `IdeaGraph` returns, so the event log format is unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_workspace.py
import numpy as np
import pytest

from innovation.core.network.graph import FrozenGraphError, IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.workspace import Workspace


class FakeEmbedder:
    dim = 4

    def encode(self, texts):
        return np.array([[len(t) % 3, 1.0, 0.0, 0.0] for t in texts],
                        dtype=np.float32)


def make_workspace() -> Workspace:
    corpus = IdeaGraph()
    corpus.add_idea("p1", "paper one", [], source="corpus", year=2020)
    corpus.add_idea("p2", "paper two", ["p1"], source="corpus", year=2021)
    corpus.freeze()
    ci = VectorIndex(4)
    ci.add(["p1", "p2"], np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    return Workspace(corpus=corpus, corpus_index=ci,
                     board_index=VectorIndex(4), embedder=FakeEmbedder(),
                     run_id="t")


def test_store_of_routes_by_id_prefix():
    ws = make_workspace()
    nid = ws.post_idea("a new idea", ["p1"], meta={})

    assert ws.store_of("p1") == "corpus"
    assert ws.store_of(nid) == "board"


def test_post_idea_never_touches_the_corpus():
    ws = make_workspace()
    before_nodes, before_edges = ws.corpus.num_nodes, ws.corpus.num_edges

    ws.post_idea("grounded idea", ["p1", "p2"], meta={})

    assert (ws.corpus.num_nodes, ws.corpus.num_edges) == (before_nodes, before_edges)


def test_cross_store_citation_becomes_a_stub_node_on_the_board():
    ws = make_workspace()
    nid = ws.post_idea("grounded idea", ["p1"], meta={})

    assert ws.board.citations_out(nid) == ["p1"]
    assert ws.board.node("p1").source == "corpus_ref"
    # stubs are not posts
    assert ws.board_post_ids() == [nid]


def test_board_post_is_searchable_and_corpus_is_not_in_the_board_index():
    ws = make_workspace()
    nid = ws.post_idea("searchable idea", [], meta={})

    hits = ws.board_search(FakeEmbedder().encode(["searchable idea"])[0], k=5)
    assert [h[0] for h in hits] == [nid]


def test_link_source_must_be_on_the_board():
    ws = make_workspace()
    nid = ws.post_idea("an idea", [], meta={})

    assert ws.add_links(nid, ["p1"], meta={})["added"] == ["p1"]
    with pytest.raises(ValueError, match="source must be a board node"):
        ws.add_links("p1", ["p2"], meta={})
    with pytest.raises(ValueError, match="source must be a board node"):
        ws.remove_links("p2", ["p1"], )


def test_any_agent_may_edit_any_post_wiki_semantics():
    ws = make_workspace()
    a = ws.post_idea("post by agent A", [], meta={"agent_id": "a"})
    b = ws.post_idea("post by agent B", [], meta={"agent_id": "b"})

    assert ws.add_links(b, [a], meta={"agent_id": "a"})["added"] == [a]
    removed = ws.remove_links(b, [a])["removed"]
    assert [r["dst_id"] for r in removed] == [a]


def test_unknown_link_target_raises():
    ws = make_workspace()
    nid = ws.post_idea("an idea", [], meta={})

    with pytest.raises(KeyError):
        ws.add_links(nid, ["nope"], meta={})


def test_corpus_stays_frozen_through_the_facade():
    ws = make_workspace()
    with pytest.raises(FrozenGraphError):
        ws.corpus.add_idea("x", "x", [])
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_workspace.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.p2_forum'`

- [ ] **Step 3: Implement the Workspace**

```python
# src/innovation/p2_forum/workspace.py
"""Two stores behind one facade (spec §3).

The corpus is frozen external background; the board is the only thing that
evolves. Every write rule lives here, in one place: an edge's source must be a
board node, so corpus-internal structure can never be added to or removed.
"""
import numpy as np

from innovation.core.network.graph import IdeaGraph, IdeaNode
from innovation.core.network.index import VectorIndex

BOARD_PREFIX = "gen:"
CORPUS_REF = "corpus_ref"


class Workspace:
    def __init__(self, *, corpus: IdeaGraph, corpus_index: VectorIndex,
                 board_index: VectorIndex, embedder, run_id: str):
        if not corpus.frozen:
            raise ValueError("the corpus must be frozen before use")
        self.corpus = corpus
        self.board = IdeaGraph()
        self.corpus_index = corpus_index
        self.board_index = board_index
        self.embedder = embedder
        self.run_id = run_id
        self._counter = 0

    # --- routing ---
    def store_of(self, node_id: str) -> str:
        return "board" if str(node_id).startswith(BOARD_PREFIX) else "corpus"

    def has_node(self, node_id: str) -> bool:
        if self.store_of(node_id) == "board":
            return self.board.has_node(node_id)
        return self.corpus.has_node(node_id)

    def node(self, node_id: str) -> IdeaNode:
        if self.store_of(node_id) == "board":
            return self.board.node(node_id)
        return self.corpus.node(node_id)

    def board_post_ids(self) -> list[str]:
        """Board nodes that are real posts — stubs excluded."""
        return [n for n in self.board.node_ids()
                if self.board.node(n).source != CORPUS_REF]

    # --- reads ---
    def corpus_search(self, vec, k: int = 5) -> list[tuple[str, float]]:
        return self.corpus_index.search(vec, k=k)

    def board_search(self, vec, k: int = 5) -> list[tuple[str, float]]:
        return self.board_index.search(vec, k=k)

    def corpus_neighbors(self, node_id: str) -> tuple[list[str], list[str]]:
        return (self.corpus.citations_out(node_id),
                self.corpus.citations_in(node_id))

    def board_neighbors(self, node_id: str) -> tuple[list[str], list[str]]:
        return (self.board.citations_out(node_id),
                self.board.citations_in(node_id))

    def corpus_sample(self, rng) -> str:
        return str(rng.choice(self.corpus.node_ids()))

    def board_sample(self, rng) -> str | None:
        posts = self.board_post_ids()
        return str(rng.choice(posts)) if posts else None

    # --- writes (all validation lives here) ---
    def _require_board_source(self, src_id: str) -> None:
        if self.store_of(src_id) != "board":
            raise ValueError(
                f"edge source must be a board node; {src_id} is in the corpus")
        if not self.board.has_node(src_id):
            raise KeyError(f"unknown board node: {src_id}")

    def _ensure_stub(self, node_id: str) -> None:
        """Corpus ids referenced from the board get a text-free stub so the
        edge is an ordinary networkx edge and board algorithms need no
        special cases."""
        if self.board.has_node(node_id):
            return
        if not self.corpus.has_node(node_id):
            raise KeyError(f"unknown node: {node_id}")
        self.board.add_idea(node_id, "", [], source=CORPUS_REF, year=None)

    def post_idea(self, text: str, cited_ids: list[str], meta: dict,
                  node_id: str | None = None) -> str:
        missing = [c for c in cited_ids if not self.has_node(c)]
        if missing:
            raise KeyError(f"cited ids not found: {missing}")
        for c in cited_ids:
            self._ensure_stub(c)
        if node_id is None:
            node_id = f"{BOARD_PREFIX}{self.run_id}:{self._counter}"
        self.board.add_idea(node_id, text, cited_ids, source="generated",
                            meta=meta)
        self.board_index.add([node_id],
                             np.asarray(self.embedder.encode([text])))
        self._counter += 1
        return node_id

    def add_links(self, src_id: str, dst_ids: list[str], meta: dict) -> dict:
        self._require_board_source(src_id)
        missing = [d for d in dst_ids if not self.has_node(d)]
        if missing:
            raise KeyError(f"link targets not found: {missing}")
        for d in dst_ids:
            self._ensure_stub(d)
        return self.board.add_links(src_id, dst_ids, meta=meta)

    def remove_links(self, src_id: str, dst_ids: list[str]) -> dict:
        self._require_board_source(src_id)
        present = [d for d in dst_ids if self.board.has_node(d)]
        missing = [d for d in dst_ids if d not in present]
        out = self.board.remove_links(src_id, present) if present else {
            "removed": [], "skipped": []}
        out["skipped"] = list(out.get("skipped", [])) + missing
        return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_workspace.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/ tests/test_workspace.py
git commit -m "feat: Workspace — frozen corpus plus a shared board, one write rule

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The p2_forum environment

**Files:**
- Create: `src/innovation/p2_forum/env.py`
- Test: `tests/test_forum_env.py`

**Interfaces:**
- Consumes: `Workspace` from Task 4; `Action` — redefined here, see below.
- Produces: `Action(name, args)` (a dataclass identical in shape to paper 1's), `ForumEnvironment(run_id, workspace, event_log, rng, allow_jump=True, allow_search=True, generation_budget=None)` with `execute(agent_id, step, action) -> dict`, `restore(events) -> None`, `generated_ids() -> list[str]`. Handler methods follow paper 1's `_do_<action>` convention.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forum_env.py
import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.p2_forum.env import Action, ForumEnvironment
from tests.test_workspace import make_workspace


def make_env(tmp_path, **kw) -> ForumEnvironment:
    return ForumEnvironment(run_id="t", workspace=make_workspace(),
                            event_log=EventLog(tmp_path / "events.jsonl"),
                            rng=np.random.default_rng(0), **kw)


def test_search_and_search_board_hit_different_stores(tmp_path):
    env = make_env(tmp_path)
    env.execute("a", 0, Action("generate", {"text": "board idea", "cited_ids": []}))

    corpus_hits = env.execute("a", 1, Action("search", {"query": "paper", "k": 5}))
    board_hits = env.execute("a", 2, Action("search_board", {"query": "board idea", "k": 5}))

    assert all(h["node_id"].startswith("p") for h in corpus_hits["hits"])
    assert all(h["node_id"].startswith("gen:") for h in board_hits["hits"])


def test_nothing_is_filtered_search_returns_every_hit(tmp_path):
    env = make_env(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "anything", "k": 5}))
    assert len(out["hits"]) == 2  # the whole two-paper corpus


def test_generate_posts_to_the_board_and_leaves_the_corpus_alone(tmp_path):
    env = make_env(tmp_path)
    before = (env.ws.corpus.num_nodes, env.ws.corpus.num_edges)

    out = env.execute("a", 0, Action("generate", {"text": "new", "cited_ids": ["p1"]}))

    assert out["node_id"].startswith("gen:")
    assert (env.ws.corpus.num_nodes, env.ws.corpus.num_edges) == before


def test_add_links_from_a_corpus_node_is_rejected(tmp_path):
    env = make_env(tmp_path)
    out = env.execute("a", 0, Action("add_links", {"src_id": "p1", "dst_ids": ["p2"]}))
    assert "error" in out
    assert "board node" in out["error"]


def test_browse_board_on_an_unknown_id_returns_an_error_not_a_crash(tmp_path):
    env = make_env(tmp_path)
    out = env.execute("a", 0, Action("browse_board", {"node_id": "gen:t:99"}))
    assert "error" in out


def test_sample_board_is_an_error_while_the_board_is_empty(tmp_path):
    env = make_env(tmp_path)
    out = env.execute("a", 0, Action("sample_board", {}))
    assert "error" in out


def test_allow_jump_false_blocks_both_jump_actions(tmp_path):
    env = make_env(tmp_path, allow_jump=False)
    assert "error" in env.execute("a", 0, Action("sample_frontier", {}))
    assert "error" in env.execute("a", 1, Action("sample_board", {}))


def test_allow_search_false_blocks_both_search_actions(tmp_path):
    env = make_env(tmp_path, allow_search=False)
    assert "error" in env.execute("a", 0, Action("search", {"query": "x"}))
    assert "error" in env.execute("a", 1, Action("search_board", {"query": "x"}))


def test_restore_rebuilds_the_board_from_the_event_log(tmp_path):
    env = make_env(tmp_path)
    nid = env.execute("a", 0, Action("generate", {"text": "one", "cited_ids": ["p1"]}))["node_id"]
    env.execute("a", 1, Action("generate", {"text": "two", "cited_ids": [nid]}))
    events = env.event_log.read_all()

    fresh = make_env(tmp_path / "other")
    fresh.restore(events)

    assert fresh.generated_ids() == env.generated_ids()
    assert fresh.ws.board.citations_out(env.generated_ids()[1]) == [nid]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_forum_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.p2_forum.env'`

- [ ] **Step 3: Implement the environment**

```python
# src/innovation/p2_forum/env.py
"""Paper 2's environment: six navigation actions over two stores, three writes
that target the board (spec §4).

The environment filters nothing. Every agent may read every node in either
store; what an agent cares about is decided by the topics in its prompt, and a
result it does not want is a result it ignores.
"""
from dataclasses import dataclass, field


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


class ForumEnvironment:
    def __init__(self, *, run_id, workspace, event_log, rng,
                 allow_jump: bool = True, allow_search: bool = True,
                 generation_budget: int | None = None):
        self.run_id = run_id
        self.ws = workspace
        self.event_log = event_log
        self.rng = rng
        self.allow_jump = allow_jump
        self.allow_search = allow_search
        self.generation_budget = generation_budget

    # --- entry point ---
    def execute(self, agent_id: str, step: int, action: Action) -> dict:
        handler = getattr(self, f"_do_{action.name}", None)
        if handler is None:
            result = {"error": f"unknown action: {action.name}"}
        else:
            try:
                result = handler(agent_id=agent_id, step=step, **action.args)
            except (KeyError, TypeError, ValueError) as exc:
                result = {"error": str(exc)}
        self.event_log.append({"run_id": self.run_id, "agent_id": agent_id,
                               "step": step, "action": action.name,
                               "args": action.args, "result": result})
        return result

    def _meta(self, agent_id, step) -> dict:
        return {"run_id": self.run_id, "agent_id": agent_id, "step": step}

    def _hits(self, raw, store: str) -> dict:
        return {"hits": [{"node_id": nid, "store": store,
                          "text": self.ws.node(nid).text[:300], "score": score}
                         for nid, score in raw]}

    def _view(self, node_id: str, store: str, out_ids, in_ids) -> dict:
        def preview(nid):
            return {"node_id": nid, "store": self.ws.store_of(nid),
                    "text": self.ws.node(nid).text[:200]}
        return {"node_id": node_id, "store": store,
                "text": self.ws.node(node_id).text,
                "cites": [preview(n) for n in out_ids[:10]],
                "cited_by": [preview(n) for n in in_ids[:10]]}

    # --- corpus navigation ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.allow_search:
            return {"error": "semantic search is not allowed for this agent"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.corpus_search(vec, k=k), "corpus")

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "corpus" or not self.ws.corpus.has_node(node_id):
            return {"error": f"{node_id} is not a corpus node"}
        out_ids, in_ids = self.ws.corpus_neighbors(node_id)
        return self._view(node_id, "corpus", out_ids, in_ids)

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.allow_jump:
            return {"error": "random jump is not allowed for this agent"}
        nid = self.ws.corpus_sample(self.rng)
        return {"node_id": nid, "store": "corpus", "text": self.ws.node(nid).text}

    # --- board navigation ---
    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.allow_search:
            return {"error": "semantic search is not allowed for this agent"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.board_search(vec, k=k), "board")

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        out_ids, in_ids = self.ws.board_neighbors(node_id)
        return self._view(node_id, "board", out_ids, in_ids)

    def _do_sample_board(self, *, agent_id, step) -> dict:
        if not self.allow_jump:
            return {"error": "random jump is not allowed for this agent"}
        nid = self.ws.board_sample(self.rng)
        if nid is None:
            return {"error": "the board is empty"}
        return {"node_id": nid, "store": "board", "text": self.ws.node(nid).text}

    # --- writes ---
    def _do_generate(self, *, agent_id, step, text: str,
                     cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        node_id = self.ws.post_idea(text, cited_ids, meta=self._meta(agent_id, step))
        if self.generation_budget is not None:
            self.generation_budget -= 1
        return {"node_id": node_id}

    def _do_add_links(self, *, agent_id, step, src_id: str,
                      dst_ids: list[str]) -> dict:
        return self.ws.add_links(src_id, dst_ids, meta=self._meta(agent_id, step))

    def _do_remove_links(self, *, agent_id, step, src_id: str,
                         dst_ids: list[str]) -> dict:
        return self.ws.remove_links(src_id, dst_ids)

    # --- replay ---
    def restore(self, events: list[dict]) -> None:
        """Rebuild the board from the event log; no logging. Events replay in
        order, so a post citing an earlier post resolves."""
        for e in events:
            result = e.get("result", {})
            if e["action"] == "generate" and "node_id" in result:
                self.ws.post_idea(e["args"]["text"], e["args"]["cited_ids"],
                                  meta={"run_id": e["run_id"],
                                        "agent_id": e["agent_id"],
                                        "step": e["step"]},
                                  node_id=result["node_id"])
                if self.generation_budget is not None:
                    self.generation_budget -= 1
            elif e["action"] == "add_links" and "added" in result:
                self.ws.add_links(e["args"]["src_id"], result["added"],
                                  meta={"run_id": e["run_id"],
                                        "agent_id": e["agent_id"],
                                        "step": e["step"]})
            elif e["action"] == "remove_links" and "removed" in result:
                self.ws.remove_links(e["args"]["src_id"],
                                     [r["dst_id"] for r in result["removed"]])

    def generated_ids(self) -> list[str]:
        return self.ws.board_post_ids()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forum_env.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/env.py tests/test_forum_env.py
git commit -m "feat: p2_forum environment — six navigation actions, no filtering

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: The k-topic agent policy

**Files:**
- Create: `src/innovation/p2_forum/agent.py`
- Test: `tests/test_forum_agent.py`

**Interfaces:**
- Consumes: `Action` from `innovation.p2_forum.env`; `Policy` from `innovation.core.policy`; `LLM` from `innovation.core.llm`.
- Produces: `ForumAgentPolicy(llm, model, topics, memory_size=20, identity="", total_steps=0)` with `act(obs) -> Action`; module constants `VALID_ACTIONS` (the nine names) and `FORUM_SYSTEM`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forum_agent.py
import json

from innovation.p2_forum.agent import VALID_ACTIONS, ForumAgentPolicy


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, model, system, user, max_tokens):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_every_action_name_is_available_to_the_agent():
    assert VALID_ACTIONS == {"search", "browse", "sample_frontier",
                             "search_board", "browse_board", "sample_board",
                             "generate", "add_links", "remove_links"}


def test_the_agents_topics_appear_in_its_system_prompt():
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'])
    pol = ForumAgentPolicy(llm=llm, model="m",
                           topics=["federated learning privacy",
                                   "sparse attention kernels"])

    pol.act({"step": 0, "last_result": {}})

    system, _ = llm.prompts[0]
    assert "federated learning privacy" in system
    assert "sparse attention kernels" in system


def test_board_actions_are_documented_in_the_user_prompt():
    llm = ScriptedLLM(['{"action": "search_board", "args": {"query": "x"}}'])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    action = pol.act({"step": 0, "last_result": {}})

    _, user = llm.prompts[0]
    assert "search_board" in user and "browse_board" in user
    assert action.name == "search_board"


def test_malformed_reply_falls_back_to_a_corpus_jump():
    llm = ScriptedLLM(["not json at all"])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    assert pol.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_unknown_action_name_falls_back_to_a_corpus_jump():
    llm = ScriptedLLM([json.dumps({"action": "delete_corpus", "args": {}})])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    assert pol.act({"step": 0, "last_result": {}}).name == "sample_frontier"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_forum_agent.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.p2_forum.agent'`

- [ ] **Step 3: Implement the policy**

```python
# src/innovation/p2_forum/agent.py
"""Paper 2's agent: the same JSON tool-calling loop as paper 1, over two
stores instead of one.

Specialization is soft (spec §4.3). The agent's topics appear only here, in
its system prompt. Nothing in the environment enforces them: the agent sees
every result and decides for itself what is worth following.
"""
import json
from collections import deque

from innovation.core.llm import LLM
from innovation.core.policy import Policy
from innovation.p2_forum.env import Action

VALID_ACTIONS = {"search", "browse", "sample_frontier",
                 "search_board", "browse_board", "sample_board",
                 "generate", "add_links", "remove_links"}

FORUM_SYSTEM = """You are a research agent. Two things are in front of you.

The LITERATURE is a fixed network of research ideas distilled from published \
papers, each citing the ideas it builds on. You can read it but never change it.

The BOARD is a shared space where you and other agents publish new ideas. \
Everything anyone posts is visible to everyone, and anyone may adjust the \
reference links on any post. The board starts empty and grows only from what \
the agents put there.

Your goal is to find promising unexplored directions and publish genuinely new \
ideas to the board. Ground them: cite the literature they build on, and cite \
other agents' posts when your idea builds on theirs. Prefer exploring until you \
understand a neighborhood well enough that your idea is specific.

You are interested in these topics:
{topics}

You may read anything in either the literature or the board. Most of what you \
find will be outside your interests — skim it and move on."""

ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- semantic search over the literature
{"action": "browse", "args": {"node_id": "<id>"}} -- read a paper-idea and its citation neighbors
{"action": "sample_frontier", "args": {}} -- jump to a random paper-idea
{"action": "search_board", "args": {"query": "<text>", "k": 5}} -- semantic search over the board
{"action": "browse_board", "args": {"node_id": "<id>"}} -- read a post and its reference neighbors
{"action": "sample_board", "args": {}} -- jump to a random post
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- publish your new idea to the board, citing what it builds on (papers or posts)
{"action": "add_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- add reference links from a post to what it builds on
{"action": "remove_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- remove reference links from a post that do not actually support it"""


class ForumAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, topics: list[str],
                 memory_size: int = 20, identity: str = "",
                 total_steps: int = 0):
        self.llm = llm
        self.model = model
        self.topics = list(topics)
        # identity (run:agent) is embedded in every prompt so identical memory
        # states of DIFFERENT agents/runs never share a disk-cache entry.
        self.identity = identity
        self.total_steps = total_steps
        bullets = "\n".join(f"- {t}" for t in self.topics)
        self.system = FORUM_SYSTEM.format(topics=bullets)
        self.memory: deque[tuple[str, str]] = deque(maxlen=memory_size)
        self._last_action: str = "(none)"

    def act(self, obs: dict) -> Action:
        result_snippet = json.dumps(obs.get("last_result", {}))[:1500]
        self.memory.append((self._last_action, result_snippet))
        history = "\n".join(f"{a} -> {r}" for a, r in self.memory)
        header = f"[agent {self.identity}]\n\n" if self.identity else ""
        user = (header + ACTIONS_DOC + "\n\nRecent history (oldest first):\n"
                + history + "\n\nChoose your next action (JSON only):")
        reply = self.llm.complete(model=self.model, system=self.system,
                                  user=user, max_tokens=2000)
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forum_agent.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/agent.py tests/test_forum_agent.py
git commit -m "feat: p2_forum agent — k topics in the prompt, nine actions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The p2_forum runner

**Files:**
- Create: `src/innovation/p2_forum/runner.py`
- Test: `tests/test_forum_runner.py`

**Interfaces:**
- Consumes: `Workspace`, `ForumEnvironment`, `ForumAgentPolicy`, `EventLog`, `load_events`.
- Produces: `ForumRunConfig(run_id, seed, total_steps, agents, topic_pool, generation_budget=None)`; `draw_topics(agents, pool, rng) -> dict[str, list[str]]`; `run_forum(cfg, *, corpus, corpus_index, embedder, llm, model, out_dir) -> dict`. `run_forum` writes `run_meta.json` containing `run_id`, `seed`, `arch: "p2_forum"` and `topic_assignments` (agent id → list of topic strings) BEFORE driving, and returns `{"run_id", "steps", "generated", "topic_assignments"}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forum_runner.py
import json

import numpy as np

from innovation.p2_forum.runner import ForumRunConfig, draw_topics, run_forum
from tests.test_workspace import FakeEmbedder, make_workspace


def test_draw_topics_gives_each_agent_k_distinct_topics():
    pool = [f"topic {i}" for i in range(20)]
    agents = [{"agent_id": "a0", "k_topics": 3},
              {"agent_id": "a1", "k_topics": 5}]

    out = draw_topics(agents, pool, np.random.default_rng(0))

    assert len(out["a0"]) == 3 and len(set(out["a0"])) == 3
    assert len(out["a1"]) == 5 and len(set(out["a1"])) == 5
    assert set(out["a0"]) <= set(pool)


def test_draw_topics_is_seeded_and_agents_may_overlap():
    pool = [f"topic {i}" for i in range(8)]
    agents = [{"agent_id": f"a{i}", "k_topics": 4} for i in range(5)]

    first = draw_topics([dict(a) for a in agents], pool, np.random.default_rng(3))
    again = draw_topics([dict(a) for a in agents], pool, np.random.default_rng(3))

    assert first == again
    # 5 agents x 4 topics from a pool of 8 cannot be disjoint
    assert len({t for ts in first.values() for t in ts}) <= 8


def test_draw_topics_rejects_k_larger_than_the_pool():
    import pytest
    with pytest.raises(ValueError, match="k_topics"):
        draw_topics([{"agent_id": "a", "k_topics": 9}],
                    [f"t{i}" for i in range(8)], np.random.default_rng(0))


class ScriptedLLM:
    def complete(self, *, model, system, user, max_tokens):
        return '{"action": "sample_frontier", "args": {}}'


def test_run_forum_writes_meta_before_driving_and_records_topics(tmp_path):
    ws = make_workspace()
    cfg = ForumRunConfig(run_id="r", seed=0, total_steps=4,
                         agents=[{"agent_id": "a0", "k_topics": 1},
                                 {"agent_id": "a1", "k_topics": 1}],
                         topic_pool=["alpha", "beta", "gamma"])

    out = run_forum(cfg, corpus=ws.corpus, corpus_index=ws.corpus_index,
                    embedder=FakeEmbedder(), llm=ScriptedLLM(), model="m",
                    out_dir=tmp_path)

    meta = json.loads((tmp_path / "r" / "run_meta.json").read_text())
    assert meta["arch"] == "p2_forum"
    assert set(meta["topic_assignments"]) == {"a0", "a1"}
    assert out["steps"] == 4
    assert (tmp_path / "r" / "events.jsonl").exists()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_forum_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.p2_forum.runner'`

- [ ] **Step 3: Implement the runner**

```python
# src/innovation/p2_forum/runner.py
"""Round-robin runner for paper 2 (spec §6).

Each agent draws k topics uniformly at random from the pool. Draws are seeded
and recorded in run_meta.json. Agents may share topics — with N agents and a
128-topic pool that is unavoidable above a certain N, and it is the substrate
collaboration needs, not a defect.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.agent import ForumAgentPolicy
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.workspace import Workspace


@dataclass
class ForumRunConfig:
    run_id: str
    seed: int
    total_steps: int
    agents: list[dict] = field(default_factory=list)
    topic_pool: list[str] = field(default_factory=list)
    generation_budget: int | None = None


def draw_topics(agents: list[dict], pool: list[str], rng) -> dict[str, list[str]]:
    """k distinct topics per agent, drawn independently. Different agents may
    draw the same topic; that overlap is what makes collaboration possible."""
    if not pool:
        raise ValueError("no topic_pool given")
    out: dict[str, list[str]] = {}
    for a in agents:
        k = int(a.get("k_topics", 1))
        if k > len(pool):
            raise ValueError(
                f"k_topics={k} exceeds the {len(pool)}-topic pool")
        picks = rng.choice(len(pool), size=k, replace=False)
        out[a["agent_id"]] = [pool[int(i)] for i in picks]
    return out


def run_forum(cfg: ForumRunConfig, *, corpus, corpus_index, embedder, llm,
              model, out_dir) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assignments = draw_topics(cfg.agents, cfg.topic_pool, rng)
    # run_meta is written BEFORE driving: an interrupted run must stay
    # reproducible from its recorded draws rather than re-sampling them.
    (run_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": cfg.run_id, "seed": cfg.seed, "arch": "p2_forum",
         "topic_assignments": assignments}, indent=1))

    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=cfg.run_id)
    env = ForumEnvironment(run_id=cfg.run_id, workspace=ws,
                           event_log=EventLog(run_dir / "events.jsonl"),
                           rng=rng,
                           generation_budget=cfg.generation_budget,
                           allow_jump=all(a.get("allow_jump", True)
                                          for a in cfg.agents),
                           allow_search=all(a.get("allow_search", True)
                                            for a in cfg.agents))

    policies = {
        a["agent_id"]: ForumAgentPolicy(
            llm=llm, model=model, topics=assignments[a["agent_id"]],
            memory_size=a.get("memory_size", 20),
            identity=f"{cfg.run_id}:{a['agent_id']}",
            total_steps=cfg.total_steps)
        for a in cfg.agents}

    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}
    for step in range(cfg.total_steps):
        agent_id = order[step % len(order)]
        action = policies[agent_id].act(
            {"step": step, "last_result": last_result[agent_id]})
        last_result[agent_id] = env.execute(agent_id, step, action)

    return {"run_id": cfg.run_id, "steps": cfg.total_steps,
            "generated": env.generated_ids(),
            "topic_assignments": assignments}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_forum_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/runner.py tests/test_forum_runner.py
git commit -m "feat: p2_forum runner with seeded k-topic draws

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Board trajectory metrics

**Files:**
- Create: `src/innovation/p2_forum/board_metrics.py`
- Test: `tests/test_board_metrics.py`

**Interfaces:**
- Consumes: `ForumEnvironment.restore`, `Workspace`.
- Produces: `board_at(events, round_index, *, corpus, corpus_index, embedder, run_id) -> Workspace` and `board_structure(ws, events) -> dict` returning keys `n_posts`, `n_post_post_edges`, `n_post_corpus_edges`, `post_post_share`, `n_components`, `longest_chain`, `first_cross_agent_round`, `cross_agent_edges`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_board_metrics.py
import numpy as np

from innovation.core.events import EventLog
from innovation.p2_forum.board_metrics import board_at, board_structure
from innovation.p2_forum.env import Action, ForumEnvironment
from tests.test_workspace import FakeEmbedder, make_workspace


def build_events(tmp_path):
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "e.jsonl"),
                           rng=np.random.default_rng(0))
    a = env.execute("a0", 0, Action("generate", {"text": "one", "cited_ids": ["p1"]}))["node_id"]
    env.execute("a1", 1, Action("generate", {"text": "two", "cited_ids": [a]}))
    return env.event_log.read_all()


def test_board_structure_separates_post_post_from_post_corpus_edges(tmp_path):
    events = build_events(tmp_path)
    ws = make_workspace()
    rebuilt = board_at(events, len(events), corpus=ws.corpus,
                       corpus_index=ws.corpus_index, embedder=FakeEmbedder(),
                       run_id="t")

    s = board_structure(rebuilt, events)

    assert s["n_posts"] == 2
    assert s["n_post_corpus_edges"] == 1   # post -> p1
    assert s["n_post_post_edges"] == 1     # post -> post
    assert s["post_post_share"] == 0.5


def test_first_cross_agent_round_is_the_step_of_the_first_a_to_b_citation(tmp_path):
    events = build_events(tmp_path)
    ws = make_workspace()
    rebuilt = board_at(events, len(events), corpus=ws.corpus,
                       corpus_index=ws.corpus_index, embedder=FakeEmbedder(),
                       run_id="t")

    s = board_structure(rebuilt, events)

    assert s["first_cross_agent_round"] == 1
    assert s["cross_agent_edges"] == 1


def test_board_at_truncates_to_a_prefix_of_the_log(tmp_path):
    events = build_events(tmp_path)
    ws = make_workspace()
    early = board_at(events, 1, corpus=ws.corpus,
                     corpus_index=ws.corpus_index, embedder=FakeEmbedder(),
                     run_id="t")

    assert len(early.board_post_ids()) == 1


def test_an_isolated_board_has_one_component_per_post(tmp_path):
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "iso.jsonl"),
                           rng=np.random.default_rng(0))
    env.execute("a0", 0, Action("generate", {"text": "x", "cited_ids": []}))
    env.execute("a1", 1, Action("generate", {"text": "y", "cited_ids": []}))
    events = env.event_log.read_all()

    fresh = make_workspace()
    rebuilt = board_at(events, len(events), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index,
                       embedder=FakeEmbedder(), run_id="t")
    s = board_structure(rebuilt, events)

    assert s["n_components"] == 2
    assert s["n_post_post_edges"] == 0
    assert s["first_cross_agent_round"] is None
    assert s["longest_chain"] == 0
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_board_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'innovation.p2_forum.board_metrics'`

- [ ] **Step 3: Implement the metrics**

```python
# src/innovation/p2_forum/board_metrics.py
"""The board's structure as a function of round (spec §5).

Paper 1 could report one scalar — one cross-agent citation in 10,300 actions.
Here every structural quantity is a function of time, so the analysis entry
point is replay of the event log, not the final state.
"""
import networkx as nx
import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.workspace import CORPUS_REF, Workspace


class _NullLog(EventLog):
    """Replay must not re-log; board_at is read-only analysis."""

    def __init__(self):
        pass

    def append(self, event: dict) -> dict:
        return event

    def read_all(self) -> list[dict]:
        return []


def board_at(events: list[dict], round_index: int, *, corpus, corpus_index,
             embedder, run_id: str) -> Workspace:
    """Rebuild the board as it stood after the first `round_index` events."""
    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=run_id)
    env = ForumEnvironment(run_id=run_id, workspace=ws, event_log=_NullLog(),
                           rng=np.random.default_rng(0))
    env.restore(events[:round_index])
    return ws


def board_structure(ws: Workspace, events: list[dict]) -> dict:
    """Structural summary of one board state. `events` supplies authorship,
    which the graph itself does not carry."""
    posts = set(ws.board_post_ids())
    author = {e["result"]["node_id"]: e["agent_id"] for e in events
              if e["action"] == "generate" and "node_id" in e.get("result", {})}

    post_post, post_corpus, cross = 0, 0, 0
    first_cross = None
    for src in posts:
        for dst in ws.board.citations_out(src):
            if ws.board.node(dst).source == CORPUS_REF:
                post_corpus += 1
                continue
            post_post += 1
            if author.get(src) is not None and author.get(src) != author.get(dst):
                cross += 1

    for e in events:
        if e["action"] != "generate" or "node_id" not in e.get("result", {}):
            continue
        src = e["result"]["node_id"]
        for dst in e["args"].get("cited_ids", []):
            if dst in author and author[dst] != e["agent_id"]:
                first_cross = e["step"] if first_cross is None else first_cross
                break
        if first_cross is not None:
            break

    sub = nx.DiGraph()
    sub.add_nodes_from(posts)
    sub.add_edges_from((s, d) for s in posts
                       for d in ws.board.citations_out(s) if d in posts)
    total = post_post + post_corpus

    return {"n_posts": len(posts),
            "n_post_post_edges": post_post,
            "n_post_corpus_edges": post_corpus,
            "post_post_share": (post_post / total) if total else 0.0,
            "n_components": nx.number_weakly_connected_components(sub),
            "longest_chain": nx.dag_longest_path_length(sub)
            if nx.is_directed_acyclic_graph(sub) else -1,
            "first_cross_agent_round": first_cross,
            "cross_agent_edges": cross}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_board_metrics.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/innovation/p2_forum/board_metrics.py tests/test_board_metrics.py
git commit -m "feat: board trajectory metrics — post-post share, components, cross-agent edges

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: CLI dispatch, configs, and the corpus-immutability invariant

**Files:**
- Modify: `src/innovation/cli.py`
- Create: `configs/p2_forum/base.yaml`, `scripts/gen_forum_configs.py`, `configs/p2_forum/experiments/k{1,2,4,8,16,32,64}.yaml`
- Test: `tests/test_forum_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 2–8.
- Produces: `cmd_run` dispatching on `cfg.get("arch", "p1_dial")`; `_load_forum_world(cfg) -> (corpus, corpus_index, embedder)` returning a FROZEN corpus.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_forum_cli.py
import json

import numpy as np

from innovation.core.network.graph import FrozenGraphError
from innovation.p2_forum.env import Action, ForumEnvironment
from innovation.p2_forum.runner import ForumRunConfig, run_forum
from tests.test_workspace import FakeEmbedder, make_workspace


class ScriptedLLM:
    """Posts once, then jumps forever."""

    def __init__(self):
        self.n = 0

    def complete(self, *, model, system, user, max_tokens):
        self.n += 1
        if self.n == 1:
            return json.dumps({"action": "generate",
                               "args": {"text": "an idea", "cited_ids": ["p1"]}})
        return json.dumps({"action": "sample_frontier", "args": {}})


def test_a_full_run_leaves_the_corpus_bit_identical(tmp_path):
    ws = make_workspace()
    corpus, index = ws.corpus, ws.corpus_index
    before = (sorted(corpus.node_ids()),
              sorted((s, d) for s in corpus.node_ids()
                     for d in corpus.citations_out(s)),
              index.vecs.tobytes(),
              list(index.ids))

    cfg = ForumRunConfig(run_id="inv", seed=0, total_steps=6,
                         agents=[{"agent_id": "a0", "k_topics": 1},
                                 {"agent_id": "a1", "k_topics": 1}],
                         topic_pool=["alpha", "beta"])
    out = run_forum(cfg, corpus=corpus, corpus_index=index,
                    embedder=FakeEmbedder(), llm=ScriptedLLM(), model="m",
                    out_dir=tmp_path)

    after = (sorted(corpus.node_ids()),
             sorted((s, d) for s in corpus.node_ids()
                    for d in corpus.citations_out(s)),
             index.vecs.tobytes(),
             list(index.ids))
    assert before == after
    assert len(out["generated"]) >= 1


def test_the_corpus_index_never_gains_a_board_vector(tmp_path):
    ws = make_workspace()
    n_before = len(ws.corpus_index.ids)
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=__import__(
                               "innovation.core.events", fromlist=["EventLog"]
                           ).EventLog(tmp_path / "e.jsonl"),
                           rng=np.random.default_rng(0))

    env.execute("a", 0, Action("generate", {"text": "x", "cited_ids": []}))

    assert len(ws.corpus_index.ids) == n_before
    assert len(ws.board_index.ids) == 1


def test_frozen_corpus_rejects_writes_even_directly():
    ws = make_workspace()
    try:
        ws.corpus.add_links("p2", ["p1"])
    except FrozenGraphError:
        return
    raise AssertionError("expected FrozenGraphError")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_forum_cli.py -q`
Expected: FAIL — `test_a_full_run_leaves_the_corpus_bit_identical` errors because `run_forum` is reached but the imports resolve; if any fail for a different reason, fix the cause before continuing.

- [ ] **Step 3: Add the base config**

```yaml
# configs/p2_forum/base.yaml
# Paper 2: frozen corpus + shared board. Inherits the corpus, recognition
# lists, models and evaluation settings from paper 1 so the headline metric is
# unchanged; overrides only what the architecture changes.
extends: ../p1_dial/stage1.yaml
arch: p2_forum
out_dir: runs/p2_forum
topics_file: configs/p2_forum/topics-k128.yaml
run:
  run_id: forum-pilot
  seed: 0
  total_steps: 800
  agents:
    - {agent_id: a0, k_topics: 1}
```

- [ ] **Step 4: Add the CLI dispatch**

In `src/innovation/cli.py`, add after `_load_world`:

```python
def _load_forum_world(cfg):
    """Paper 2's world: the corpus, frozen, plus its index. The board is
    created per run by the runner."""
    from innovation.core.network.graph import IdeaGraph

    _, edges = load_corpus(cfg["data_dir"])
    ideas = load_ideas(cfg["data_dir"])
    if cfg.get("run", {}).get("init_edges", "citations") == "none":
        edges = edges.iloc[0:0]
    corpus = IdeaGraph.from_tables(ideas, edges)
    corpus.freeze()
    ids, vecs = load_embeddings(cfg["data_dir"])
    emb = Embedder(cfg["embedding_model"])
    index = VectorIndex(emb.dim)
    index.add(ids, vecs)
    return corpus, index, emb
```

Then, at the top of `cmd_run`, immediately after `r = cfg["run"]` and the `seed`/`run_id` overrides and the `events_path` guard, insert:

```python
    if cfg.get("arch", "p1_dial") == "p2_forum":
        from innovation.p2_forum.runner import ForumRunConfig, run_forum
        import yaml
        corpus, index, emb = _load_forum_world(cfg)
        pool_data = yaml.safe_load(Path(cfg["topics_file"]).read_text())
        run_cfg = ForumRunConfig(
            run_id=r["run_id"], seed=r["seed"], total_steps=r["total_steps"],
            agents=r["agents"], topic_pool=[t["topic"] for t in pool_data["topics"]],
            generation_budget=r.get("generation_budget"))
        out = run_forum(run_cfg, corpus=corpus, corpus_index=index, embedder=emb,
                        llm=_llm(cfg), model=cfg["models"]["agent"],
                        out_dir=cfg["out_dir"])
        print(json.dumps(out, indent=2))
        return
```

- [ ] **Step 5: Write the config generator**

```python
# scripts/gen_forum_configs.py
"""Regenerate paper 2's k-sweep configs (spec §6, stage 1).

k in {1,2,4,8,16,32,64} at N=10, 80 rounds = 800 steps — the same per-agent
depth as paper 1's core runs.
Run: uv run python scripts/gen_forum_configs.py
"""
from pathlib import Path

import yaml

KS = [1, 2, 4, 8, 16, 32, 64]
N = 10
ROUNDS = 80

HEADER = ("# GENERATED by scripts/gen_forum_configs.py.\n"
          "# Each agent draws k distinct topics from configs/p2_forum/topics-k128.yaml\n"
          "# (seeded, recorded in run_meta.json). Agents may share topics.\n"
          "# Edit the generator, not this file.\n")

out_dir = Path("configs/p2_forum/experiments")
out_dir.mkdir(parents=True, exist_ok=True)
for k in KS:
    cfg = {"extends": "../base.yaml",
           "run": {"run_id": f"forum-k{k}", "seed": 0,
                   "total_steps": ROUNDS * N,
                   "agents": [{"agent_id": f"a{i}", "k_topics": k}
                              for i in range(N)]}}
    path = out_dir / f"k{k}.yaml"
    with path.open("w") as f:
        f.write(HEADER + f"# k={k}: {N} agents, each interested in {k} topics\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False, width=100)
print(f"{len(KS)} configs regenerated in {out_dir}")
```

- [ ] **Step 6: Generate the configs and run the tests**

```bash
uv run python scripts/gen_forum_configs.py
uv run pytest tests/test_forum_cli.py -q
uv run pytest -q
```
Expected: 7 configs written; both test runs PASS.

- [ ] **Step 7: Smoke-run the smallest condition**

Run: `uv run python -m innovation.cli run --config configs/p2_forum/experiments/k1.yaml --run-id forum-smoke --steps 6`

If `--steps` is not an existing CLI flag, instead copy `k1.yaml` to `configs/p2_forum/experiments/smoke.yaml`, set `total_steps: 6` and `run_id: forum-smoke`, and run that config.

Expected: JSON output with `"steps": 6`, and `runs/p2_forum/forum-smoke/{events.jsonl,run_meta.json}` present. Inspect `run_meta.json` — `arch` is `p2_forum` and each agent has exactly one topic.

Then delete the smoke run: `rm -rf runs/p2_forum/forum-smoke`.

- [ ] **Step 8: Commit**

```bash
git add src/innovation/cli.py configs/p2_forum scripts/gen_forum_configs.py \
        tests/test_forum_cli.py
git commit -m "feat: arch dispatch, paper 2 configs, corpus-immutability invariant

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** §2 architectural change → Tasks 3–5. §3.1 two stores → Task 4. §3.2 stub nodes → Task 4 (`_ensure_stub`, tested). §3.3 two indices → Task 4, asserted in Task 9. §3.4 Workspace API → Task 4; `board_at` → Task 8. §4.1 six navigation actions → Task 5. §4.2 write rules → Task 4 (`_require_board_source`) and Task 5. §4.3 k topics in the prompt, no scope machinery → Tasks 6–7. §4.4 K=128 pool → Task 2. §5 headline metric unchanged → Task 9's `extends: ../p1_dial/stage1.yaml`; structural metrics → Task 8. §6 k sweep → Task 9's generator. §7 repo layout → Task 1. §8 acceptance: `pytest` green (every task), invariant test (Task 9), end-to-end run (Task 9 Step 7).

**Gap found and closed.** The spec's §8 acceptance asks for a `k=1, N=10` run producing *both* headline and structural metrics. Structural metrics exist (Task 8) but nothing wires `board_structure` into `cmd_evaluate`. This is deliberate: evaluation of real runs is a separate, API-cost-bearing activity that belongs with the first real experiment batch, not with the architecture. Flagged in Open Items below rather than silently dropped.

**Placeholder scan.** No TBD/TODO; every code step carries runnable code; no "similar to Task N" references.

**Type consistency.** `Workspace.post_idea(text, cited_ids, meta, node_id=None)` is called with that signature in `env.py` (`_do_generate`, `restore`) and `board_metrics.py`. `add_links(src_id, dst_ids, meta)` and `remove_links(src_id, dst_ids)` match between `workspace.py`, `env.py` and the tests. `CORPUS_REF` is defined in `workspace.py` and imported by `board_metrics.py`. `Action` is defined in `p2_forum/env.py` and imported by `p2_forum/agent.py` — paper 1's `Action` is a separate class in `p1_dial/env.py`, and the two never meet.

## Open Items

- Wiring `board_structure` into `cmd_evaluate` (a per-round metrics table alongside the anticipation verdicts) — belongs with the first real k-sweep batch.
- Stage-2 `N` sweep — deferred by design until the k sweep is read (spec §9).
