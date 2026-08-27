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
to cutoff year T = 2022, let agents grow it, and measure how well generated ideas
anticipate the real 2023–2024 literature.

## 2. Scope and Staging

- **Stage 1 (pipeline validation):** 1–2 venues (NeurIPS + ICLR), 2013–2024,
  ~20k papers. All code paths, metrics, and at least one full experiment run
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
- Agents have a step budget per episode and decide **for themselves** when to
  generate — generation timing is part of the dynamics, not forced each step.
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
| **Structure** (initial placement) | inside a dense community / at a community boundary / cross-community mix |
| **Dynamics** | single agent / multi-N homogeneous / multi-N heterogeneous |

Plus the two baseline policies from §3.4. All conditions run with multiple random
seeds. Every agent action is one JSONL event (run_id, agent_id, step, action,
args, result summary) — the trace is the primary dataset for dynamics analysis.
Runs are resumable from the event log. Experiment configs are YAML files checked
into the repo.

### 3.6 Evaluation (`eval/`)

- **Anticipation score (headline):** network initialized at T = 2022; real
  post-T papers (summarized identically) form the held-out set.
  - **Asymmetric corpus design:** the ≤T network is narrow (selected venues,
    needs citation edges), but the >T held-out set is **broad** — a wide AI/ML
    venue list (NeurIPS/ICML/ICLR/ACL/EMNLP/CVPR/ICCV/AAAI/KDD/…) plus arXiv
    cs.LG/cs.CL/cs.CV 2023–2024. The held-out set needs no citation edges and
    never enters the graph, so widening it costs only summarization + embedding
    (same template, same embedding model, for representational comparability).
    This prevents penalizing agents for anticipating ideas published outside the
    seed venues.
  - *Precision-like:* fraction of generated ideas whose max cosine similarity to
    a held-out idea (broad set) exceeds threshold τ, **while** remaining below a
    ceiling similarity to the ≤T corpus (anti-plagiarism-of-the-past condition).
  - *Recall-like:* reported at two scopes — narrow (future papers of the seed
    venues; the fairest in-domain coverage measure) and broad (honest but
    expectedly low; reported for reference).
  - Residual limitation: ideas realized entirely outside CS (e.g. physics
    journals) can still be missed; stated as a limitation, mitigated by human
    inspection of matched/unmatched samples.
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
