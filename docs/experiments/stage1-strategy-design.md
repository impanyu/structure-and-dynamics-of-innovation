# Stage 1 — Strategy Comparison Design

**Spec:** `docs/superpowers/specs/2026-08-27-idea-network-agent-dynamics-design.md`
**Configs:** `configs/experiments/*.yaml` (each condition is directly runnable:
`uv run python -m innovation.cli run --config configs/experiments/<name>.yaml --seed K --run-id <name>-sK`)

## Controls (identical across all conditions)

- Same initial graph (small-field corpus), same seed set (3–5 seeds), same
  `total_steps: 400`, same **global generation budget: 40** — every condition
  produces the same number of ideas; quality, not quantity, is compared.
- Every agent may random-jump (`sample_frontier`) **within its own readable
  scope** — jumps never leak outside an agent's region.
- Core matrix teams are all **N = 10**.

## Core matrix (4 conditions)

| id | condition | composition (N=10) | question |
|---|---|---|---|
| C1 | 多通才 generalists | 10 × unrestricted llm | dynamics reference |
| C2 | 多专精 specialists | 2 agents × 5 subfield topics, read+write own region | deep exploitation: feasibility vs novelty |
| C3 | 多博览专写 broad readers / local writers | 2 × 5 topics, read ALL, write own region | T-shaped researchers; predicted winner (boundary-spanning) |
| C4 | 混编 mixed team | 4 specialists + 3 broad + 3 generalists | does diversity of behavior profiles beat any pure team? |

Key contrasts: C1↔C2 (constraint cost/benefit), C2↔C3 (read breadth, write
held fixed), C1/C2/C3↔C4 (composition diversity).

## Supplementary experiments

1. **Composition sweep** — `supp-mixed-spec-heavy` (7 spec + 3 gen),
   `supp-mixed-gen-heavy` (3 spec + 7 gen), against C4 (4+3+3): performance vs
   mix ratio.
2. **Scaling** — generalists with N ∈ {1, 2, 5, 10, 20} at the SAME total
   steps and generation budget (`supp-scaling-n*`; N=10 point = C1):
   performance vs agent count under equal compute. N=1 is the single-agent
   baseline; the curve shape is the stigmergy story.
3. **Initial-graph ablation** — `supp-noedges-generalists`: same corpus,
   **no edges** (independent nodes). Isolates the value of the citation
   structure itself; agents can only navigate semantically.
4. **Jump ablation** — `supp-nojump-generalists`: C1 with `allow_jump: false`
   everywhere; serendipity contribution.
5. **Validity anchors** — `supp-baseline-nonav` (LLM without navigation) and
   `supp-baseline-pa` (preferential-attachment null, no LLM; structural
   metrics only). Cheap; keeps reviewers satisfied that both the LLM and the
   navigation matter.

## Pre-registered expectations (metric signatures)

| condition | precision (anticipation) | novelty | bridging | diversity |
|---|---|---|---|---|
| C2 specialists | high | low | low | low (within-region) |
| C3 broad/local | **high** | **high** | **high** | mid |
| C1 generalists | mid | mid | mid | mid |
| C4 mixed | between C1 and C3 | mid-high | mid-high | **high** |
| nonav | low (vague, dup-flagged) | spurious-high | random | high |

If C3 wins precision AND bridging simultaneously, the headline is
"**read broadly, write narrowly**".

## Analysis plan

- Per-run metrics from `metrics.json` + `verdicts.json`; process observables
  (novelty, bridging, diversity, action-mix over time) recomputed from event
  logs.
- Seeds are paired across conditions; report effect sizes + bootstrap CIs;
  Mann-Whitney / Wilcoxon for pairwise contrasts (few seeds — do not lean on
  p-values alone).
- Topic list (5 FL subfields) and `radius: 0.3` are tentative: after the real
  corpus is fetched, inspect embedding-space cluster structure and per-topic
  region sizes (nodes within radius), then freeze them BEFORE any full run.

## Open items before running

1. Real-data fetch + summarization + human audit of ~30 summaries.
2. Freeze topics/radius from corpus inspection (above).
3. Seed count: start with 3 (21+ runs), extend to 5 for the final paper runs.
