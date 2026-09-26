# Paper 2 — The Shared Board: Agent-to-Agent Innovation on a Static Literature

**Date:** 2026-09-26
**Status:** Approved design, pre-implementation
**Deliverable:** A second research paper, sharing this repository with paper 1
(`docs/superpowers/specs/2026-08-27-idea-network-agent-dynamics-design.md`).

## 1. Research Question

Paper 1 ended on a specific frontier, quoted from its Conclusion:

> The immediate frontier is switching on the silent agent-to-agent channel —
> making contributions discoverable to teammates — and measuring whether the
> scaling curve bends.

That channel was silent for a structural reason. In paper 1 the agents' output was
written into the *same* graph as the corpus, where it was indistinguishable from
background literature and effectively invisible: across 10,300 actions, exactly one
agent cited another agent's generated idea.

Paper 2 changes the architecture so the channel can exist, and asks whether it
carries anything:

**When agent-generated ideas live on a shared board that every agent can read and
write, does collective structure emerge — and does team size stop being a pure
throughput knob?**

## 2. The Architectural Change

| | Paper 1 | Paper 2 |
|---|---|---|
| State | one graph; corpus + generated mixed | corpus **frozen**; a separate shared board |
| What evolves | the whole graph | **only the board** |
| Specialization | hard: environment blocks reads outside a scope | soft: `k` topics in the agent's prompt |
| Environment filtering | search/browse results scope-filtered | **none — everything is readable** |
| Actions | 3 navigation + 3 write | 3 per store (6) + 3 write |

The corpus becomes **static external background knowledge**. The board is the only
thing that changes, so the dynamical system under study is the board's trajectory.

### 2.1 Framing

The board is where agents publish to each other. Mechanically it is a
**collectively editable** network, not a forum in the strict sense: any agent may
add or remove links on any post (wiki semantics, §4.2). The paper must say this
plainly rather than lean on the forum metaphor.

## 3. Data Model

### 3.1 Two stores

- `corpus: IdeaGraph` — built from `data/stage1/*.parquet`, then `freeze()`. Every
  mutating method raises `FrozenGraphError` thereafter. Read-only is a property of
  the type, not a convention.
- `board: IdeaGraph` — empty at the start of every run; holds all agent output.

Node ids are disjoint by construction: corpus ids are paper ids, board ids are
`gen:<run_id>:<n>`. `Workspace.store_of(node_id)` routes on that prefix.

### 3.2 Cross-store edges

A board post citing a corpus paper produces an edge whose destination is not a
board node. The board graph holds a **stub node** for each referenced corpus id
(`IdeaNode(source="corpus_ref")`, no text), so the edge is an ordinary networkx
edge and every graph algorithm runs on the board without special cases. The
existing `node_ids(source=...)` filter keeps stubs out of statistics.

The stub tag is load-bearing, not cosmetic. It separates the two structures the
paper measures:

- `post → corpus_ref` — how deeply an idea is rooted in the literature
- `post → post` — **whether agents actually connected**, the direct evidence for
  the paper's claim

### 3.3 Two indices

`corpus_index` is built from the precomputed embeddings and never changes;
`board_index` grows as posts appear. Separate indices are required by the separate
search actions (§4.1) — sharing one index would make `search` and `search_board`
the same operation with a filter.

### 3.4 Workspace

```
store_of(id) -> "corpus" | "board"
node(id) / has_node(id)                       # resolves across both stores
corpus_search(vec, k) / corpus_neighbors(id) / corpus_sample()
board_search(vec, k)  / board_neighbors(id)  / board_sample()
board_at(round: int) -> IdeaGraph             # replayed from the event log
post_idea(text, cited_ids, meta) -> node_id   # the only node-creating entry point
add_links(src, dsts) / remove_links(src, dsts)
```

All write validation lives here, in one place.

`board_at(round)` is a first-class capability, not a debugging aid. Every structural
metric in this paper is a function of round, so the analysis entry point is replay,
not the final state. Paper 1's `restore()` already does the work; this is a thin
wrapper over it.

## 4. Rules

### 4.1 Actions

| | Corpus (static background) | Board (shared, mutable) |
|---|---|---|
| Semantic search | `search` | `search_board` |
| Node and neighbors | `browse` | `browse_board` |
| Random jump | `sample_frontier` | `sample_board` |

Writes — `generate` (post), `add_links`, `remove_links` — target the board only.

The symmetry is deliberate: paper 1's navigation-channel ablations (no edges, no
jumps, no search) can be applied to each store independently, from one
implementation.

**The environment filters nothing.** Search returns whatever matches; the agent
reads what interests it and ignores the rest.

### 4.2 Writes

- The source of any edge must be a board node. Destination may be corpus or board.
- Corpus-internal edges can neither be added nor removed.
- No ownership check: any agent may add or remove links on any post.

### 4.3 Specialization

Each agent is given **`k` topics drawn uniformly at random** from a pool of 128,
recorded in `run_meta.json` and seeded. The topics appear **only in the agent's
system prompt**. Nothing in the environment enforces them.

`k` is therefore a continuous specialization spectrum, and it governs discovery on
both stores at once: a `k=1` agent searches one narrow area and rarely meets a
teammate; a `k=64` agent ranges widely and encounters many. **The probability that
collaboration occurs is a function of `k`** — which is the quantity under study.

No separate "channel on/off" switch exists. Such a switch would duplicate what `k`
already controls, by a mechanism independent of `k`, and would contaminate the dial.

`AgentScope`'s read/write gating from paper 1 (`read`, `write`, `read_anchors`,
`read_radius`, `_readable`, `_writable`) is unused here. `allow_search` and
`allow_jump` are kept — those are environmental, not dispositional.

### 4.4 Topic pool

The existing pool has 50 topics; `k=64` needs more. Regenerate at **K=128**, so the
whole sweep fits and `k=64` is still half the pool rather than degenerating into
"all of it". This is a data prerequisite: it must exist before any run.

## 5. Metrics

**Headline: acc@≥2**, identical to paper 1 — same six-level rubric, same three
recognition tiers, same `realized_min_date = 2025-06-01`.

**Mechanism: the board's structure as a function of round.** Share of `post → post`
edges among all board edges; number of connected components; round of the first
cross-agent edge; depth of the longest citation chain among posts; in-degree
distribution over posts. Paper 1 could report one scalar (1 cross-agent citation in
10,300 actions); this paper reports the trajectory.

Because nothing is filtered, the event log already records what an agent retrieved but did not use. When reporting, keep two readings apart: "the channel is closed" and "the channel is open and unused".

## 6. Experiments

**Stage 1 — the `k` sweep.** `k` in {1, 2, 4, 8, 16, 32, 64, 128} at `N = 10`. Eight runs. `k=128` is the whole pool — every agent interested in everything, the generalist endpoint and the analogue of paper 1's `m=infinity` condition.

**Stage 2 — the `N` sweep.** Shape decided after stage 1 results, from where
cross-agent edges do and do not appear. `k` and `N` are swept separately; no
Cartesian product.

Navigation-channel ablations carry over from paper 1 and can now be applied per
store.

## 7. Repository Layout

Paths in this repo are config-driven (`data_dir`, `out_dir`); only
`scripts/gen_topic_configs.py` and two test paths are hardcoded. The migration is
therefore low-risk. Python package names cannot contain hyphens, so underscores are
used throughout for consistency.

```
src/innovation/
  core/        llm, config, data/, ideas/, network/, eval/, analysis/   [shared]
  p1_dial/     env.py, runner.py, agents/            [paper 1: one graph + hard scopes]
  p2_forum/    workspace.py, env.py, runner.py, agents/   [paper 2: two stores]
  cli.py       dispatches on the config's arch field

configs/p1_dial/    stage1.yaml, topics-k50.yaml, experiments/*.yaml
configs/p2_forum/   base.yaml, topics-k128.yaml, experiments/*.yaml

data/stage1/        corpus + embeddings      [shared; NOT renamed — renaming would
                                              break paths in paper 1's released data
                                              for no benefit]
data/caches/        llm_cache, openalex_cache  [shared; prerequisite for §7]

runs/p1_dial/       paper 1's 19 run dirs + its 9 loose artifacts
                    (mass-dial-*.png, scaling-*.{png,json},
                     probe_battery.json) — moved wholesale
runs/p2_forum/      paper 2

paper/p1_dial/      paper 1 tex + figures
paper/p2_forum/     paper 2
```

**What is shared and what is not.** Graph and index data structures, embeddings, the
LLM client, the entire evaluation pipeline, and the visualization helpers are
shared. Environment semantics and runners are not, because the two papers' state
models differ fundamentally.

`eval/` must be shared — both papers score through the same `search_verify.py`
against the same `openalex_cache`.

Paper 1's anonymous mirror is pinned at commit `35d7d923`, so reviewers see the
pre-migration layout and are unaffected by this reorganization.

## 8. Acceptance

- `uv run pytest` green after the migration — paper 1's suite is the safety net.
- **Invariant test:** after a full paper-2 run, the corpus graph's node set, edge
  set, and `corpus_index.vecs` hash to their initial values. This assertion is the
  empirical backing for the paper's "the corpus is read-only" claim and belongs in
  CI.
- A `k=1, N=10` run completes end to end and produces both headline and structural
  metrics.

## 9. Open Questions

- Stage-2 `N`-sweep shape — deliberately deferred until the `k` sweep is read.
