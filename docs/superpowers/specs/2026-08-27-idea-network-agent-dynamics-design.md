# Idea Networks and Agent Dynamics of Innovation — Design Spec

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Deliverable:** A research paper. The system built here is the experimental apparatus.

## 1. Research Question

How do (a) the **structure** of a research-idea network and (b) the **dynamics** of
LLM agents operating on it — single-agent vs multi-agent — impact scientific
innovation?

We represent the literature as an **idea network**: each paper is summarized into a
one-paragraph idea (node); citations between papers become edges between ideas.
LLM agents autonomously navigate this network (browse, search, read) and, when they
choose, generate new ideas that cite existing nodes and join the network. The
growing network is a dynamical system; we study how initial structure and agent
configuration shape the innovation it produces.

**Headline evaluation:** hold out the future. Initialize the network with papers up
to cutoff T, let agents grow it, and verify generated ideas by literature search:
an idea scores only if a real paper published after T realizes it. **T is aligned
with the agent LLM's training cutoff (~early 2025)** so that hits cannot be
pretraining recall; the reference is the real ~2025–2026 literature.

## 2. Scope and Staging

- **Stage 1 (pipeline validation + first dynamics study):** the initial graph
  is ONE small, coherent subfield, up to cutoff T (~early 2025). **Admission
  rule:** within the field keyword query and year range, a paper enters the
  graph iff it was published in the top-venue list OR its citation count
  exceeds a configured floor (implemented as the union of two OpenAlex
  queries; deduped downstream). A compact, dense field makes the innovation
  dynamics observable; evaluation still searches the WHOLE literature (§3.6),
  so agents get credit wherever their ideas were realized. All code paths, metrics, and at least one full experiment run
  end-to-end.
- **Stage 2 (full study):** add ICML, ACL, CVPR, AAAI (and similar AI/ML top
  venues), ~50–100k papers. Same code, bigger config.

Non-goals (explicitly out of scope): PDF parsing (abstracts suffice), explicit
agent-to-agent messaging (future work), agent frameworks (LangGraph/AutoGen),
graph databases.

## 3. Architecture

Python monorepo, six modules under `src/`:

```
data/         OpenAlex fetch + clean          -> papers.parquet
ideas/        LLM summarization + embeddings  -> ideas.parquet, embeddings
network/      IdeaGraph + FAISS index         -> in-memory graph, snapshots
agents/       Policy interface + LLM agent + baselines
experiments/  Config, runner, event logging   -> runs/, events.jsonl
eval/         Anticipation score + structural observables
```

### 3.1 Data layer (`data/`)

- **Source:** OpenAlex API (free; provides abstracts via inverted index and
  `referenced_works` for citation edges).
- **Filter:** keep papers with a non-empty abstract; citation edges restricted to
  within-corpus (both endpoints in our venue/year slice).
- **Output:** `papers.parquet` (openalex_id, title, abstract, year, venue,
  authors, referenced_works).

### 3.2 Idea layer (`ideas/`)

- Each paper → one-paragraph idea summary via a cheap LLM (Haiku-class), fixed
  prompt template: *problem + key insight + method*, 3–4 sentences. Stored in
  `ideas.parquet` with paper metadata.
- Embeddings via a local sentence-transformers model (e.g. bge-large); stored as
  a numpy array aligned with the ideas table.
- Every LLM call is disk-cached (keyed by model + prompt hash) for
  reproducibility and cost control.
- Quality control: manual audit of a random sample of summaries before Stage 1
  experiments begin.

### 3.3 Idea network (`network/`)

- **Storage is file-based; the graph is an in-memory object.** Canonical state =
  node table + edge table (parquet) + per-run event log (JSONL). Any graph state
  is reconstructible as *initial tables + event-log replay* — this doubles as
  checkpoint/resume and makes action traces first-class research data.
- `IdeaGraph` class wraps networkx (Stage 1). Its interface hides the backend so
  it can swap to igraph if Stage 2 community detection / clustering becomes slow.
  Key operations: `neighbors(node)`, `add_idea(text, cited_ids, meta)`,
  `network_at(T)` (time slice by year), structural metrics.
- Semantic search via FAISS index over idea embeddings; generated ideas are
  embedded and added to the index as they are created.
- Nodes carry provenance: real paper (year, venue) vs generated (run_id,
  agent_id, step).

### 3.4 Agent dynamical system (`agents/`)

- **Abstract `Policy` interface:** `observe -> action`. The LLM agent is one
  policy among several; baselines plug into the identical simulation loop.
- **Action space:**
  - `search(query)` — top-k ideas by embedding similarity
  - `browse(node_id)` — read an idea plus its in/out citation neighbors
  - `sample_frontier()` — sample an entry point (random or structure-guided)
  - `generate(text, cited_ids)` — write a new idea node, citing declared bases
  - `add_links(src_id, dst_ids)` — add missing reference edges between
    existing ideas. Agent-added edges are typed `agent_link` (originals are
    `citation`, generation edges are `generated`) so analysis can always
    separate the observed literature from agent rewiring.
  - `remove_links(src_id, dst_ids)` — remove reference edges an agent judges
    unsupported. The event log records each removed edge with its etype, and
    canonical tables are immutable, so the original network is always
    reconstructible from tables + event replay.
- Agents have a step budget per episode and decide **for themselves** when to
  generate — generation timing is part of the dynamics, not forced each step.
- **Short-term memory:** each LLM agent keeps a FIFO queue of its last 20
  (action, result) pairs, rendered into its prompt each step.
- **Policies:**
  1. **LLM agent** — tool-calling loop (Sonnet-class model), bounded context
     memory of recent observations.
  2. **No-navigation LLM baseline** — sample k random ideas, generate directly.
     Isolates the contribution of network navigation.
  3. **Structural null model** — preferential-attachment-style generator (no
     LLM). Isolates the contribution of the LLM itself; text for these nodes is
     produced by a trivial template so structural metrics remain comparable.
- **Multi-agent = stigmergy:** N agents interleave (round-robin) on one shared
  network; the *only* interaction channel is reading nodes other agents wrote.
  No direct messages.
- **Fairness control:** single-agent and multi-agent runs are compared at equal
  *total generated-idea budget*.
- **Ablation:** homogeneous N agents vs heterogeneous personas (e.g.
  conservative/exploratory, differing field preferences).

### 3.5 Experiments (`experiments/`)

Two manipulated factors, crossed:

| Factor | Levels |
|---|---|
| **Structure → behavior profiles** | structural conditions are operationalized as per-agent behavioral constraints (`AgentScope`): read scope, write scope, and whether random jumps are allowed. A scope region is defined by ONE of two mechanisms: (a) a set of Louvain communities, or (b) a **semantic region** — one or more topic strings embedded as anchors plus a cosine-distance radius (node in region iff distance to the nearest anchor ≤ radius). Canonical profiles: **specialist** (read+write one region), **broad reader / local writer** (read all, write one region), **generalist** (read+write all), each with jumps on or off. Constraints are enforced by the environment (search results filtered, browse/jump restricted to readable nodes, generate/add_links/remove_links restricted to writable nodes; a semantic write scope additionally requires the generated idea's own embedding to fall inside the region). Membership of generated nodes: automatic via their own embedding for semantic regions; majority community of citations for community regions. |
| **Dynamics** | single agent / multi-N homogeneous / multi-N heterogeneous (personas and/or mixed behavior profiles) |

Plus the two baseline policies from §3.4. All conditions run with multiple random
seeds. Every agent action is one JSONL event (run_id, agent_id, step, action,
args, result summary) — the trace is the primary dataset for dynamics analysis.
Runs are resumable from the event log. Experiment configs are YAML files checked
into the repo.

### 3.6 Evaluation (`eval/`)

- **Cutoff choice:** T is aligned with the agent LLM's training cutoff (~early
  2025). Rationale: for papers published before the model's cutoff, a "hit" may
  be pretraining recall rather than innovation; only post-cutoff hits exclude
  memorization.
- **Search-verified realization (the sole evaluation channel).** No held-out
  corpus is downloaded or embedded; each generated idea is verified against the
  live literature indexes:
  - Pipeline: idea → several extracted search-query formulations → Semantic
    Scholar + OpenAlex search APIs (NOT Google Scholar — no API, anti-scraping)
    → top-k candidate papers per query → LLM judge decides whether a candidate
    genuinely realizes the idea → human spot-check of judgments.
  - All search responses cached to disk with timestamps (live indexes drift;
    this keeps the evaluation replayable).
  - **Only anticipation counts as a hit:** a hit requires the realizing paper
    to be published *after* the agent model's training cutoff, which excludes
    pretraining recall. Matches to papers published before the cutoff
    (rediscoveries) score zero — they are logged with an `excluded_pre_cutoff`
    flag for later inspection, but never counted in any metric.
- **Precision:** fraction of generated ideas judged realized by a post-cutoff
  paper.
- **Recall:** (number of *distinct* realized papers hit across all generated
  ideas) ÷ (estimated post-cutoff relevant population). The denominator needs
  no downloads: an OpenAlex metadata *count* query over (a) post-cutoff papers
  in the venue list (NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR/ICCV/AAAI/KDD/…) plus
  (b) post-cutoff arXiv-only papers above a citation threshold (threshold in
  config).
- **Anti-plagiarism-of-the-past:** generated ideas must stay below a ceiling
  similarity to the ≤T corpus (whose embeddings exist anyway for the agents'
  `search` action); near-duplicates of the past are flagged and excluded from
  precision's numerator.
- Known limitations, stated in the paper: search-engine misses deflate
  precision (mitigated by multi-query + dual APIs; miss rate estimated via
  human audit of a sample of unmatched ideas); ideas realized entirely outside
  CS can be missed.
  - Threshold τ calibrated against a real-vs-real similarity distribution;
    matched pairs sampled for human inspection.
- **Process observables (all runs):** novelty (embedding distance to cited nodes
  and to corpus), bridging (do new edges connect distant communities — atypical
  combinations), diversity/entropy of the generated set, network growth
  statistics (degree distribution, clustering evolution).
- **Judgment audit (small):** sample of generated ideas rated for
  novelty/feasibility by LLM judge + human spot-check, to guard against metric
  gaming.

## 4. Engineering Conventions

- Python managed with `uv`; `pytest` with TDD; experiments configured via YAML.
- LLM usage: Haiku-class for bulk summarization, Sonnet-class for agent
  decisions; embeddings local (sentence-transformers). Stage 1 budget target:
  tens of dollars.
- All LLM calls disk-cached; all runs seeded; event logs make every experiment
  replayable.

## 5. Risks and Mitigations

- **Summary quality drives everything** → manual audit gate before experiments;
  fixed template; cache means re-summarization is cheap if the template changes.
- **networkx too slow at Stage 2** → `IdeaGraph` interface isolates the backend;
  swap to igraph (≈1 day).
- **Anticipation score gamed by vague ideas** (generic text matches everything)
  → anti-plagiarism ceiling vs ≤T corpus + specificity check in the judgment
  audit.
- **OpenAlex venue coverage gaps** → validate venue queries early in Stage 1;
  fall back to Semantic Scholar for missing venues.
