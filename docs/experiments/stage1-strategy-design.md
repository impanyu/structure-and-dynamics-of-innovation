# Stage 1 — Strategy Comparison Design (current, 2026-08-29)

**Spec:** `docs/superpowers/specs/2026-08-27-idea-network-agent-dynamics-design.md`
**Configs:** `configs/experiments/*.yaml` — run any condition with
`uv run python -m innovation.cli run --config configs/experiments/<name>.yaml --seed K --run-id <name>-sK`
then `... evaluate --config <same> --run-id <name>-sK`.

## World

- Corpus: NeurIPS + ICLR + ICML + AAAI, 2020 – Aug 2024 (T = gpt-5 training
  cutoff Sep 2024), citations ≥ 25 → 16,208 idea nodes, 80,962 citation edges
  (6.4% isolated).
- Agent model `openai:gpt-5`; judge `openai:gpt-5-mini:low`.
- Topic pool: 50 corpus-derived subfields (`configs/topics-k50.yaml`).
  A "specialist" draws ONE topic per run — seeded random, distinct within the
  run, same topic for read and write, recorded in `run_meta.json`. Scope =
  equal-mass region: the 800 corpus papers nearest the topic anchor (per-
  anchor derived radius, so generated ideas classify naturally).

## Controls (identical across all conditions)

- Same graph, `total_steps: 400`, global `generation_budget: 40` — every
  condition emits the same number of ideas; quality is compared, not quantity.
- Seeds: core matrix runs ONE seed per condition (seed 0) — user decision
  2026-08-29 to control cost; extend to more seeds for the paper's final
  numbers once the pipeline and effect directions are confirmed.
- Jumps (`sample_frontier`) allowed everywhere, always confined to the
  agent's readable scope.
- Headline metric: **accuracy** (precision) — share of generated ideas
  judged realized by a recognized post-cutoff (≥ Oct 2024) paper
  (recognition = 58 CCF-A venues + COLM/RLC, or citations ≥ 10; in-corpus
  and pre-cutoff matches excluded). Process observables: novelty, bridging,
  diversity, action-trace statistics from `events.jsonl`.

## Core matrix — run first (4 conditions × 1 seed = 4 runs)

| id | config | team (N=10) | question |
|---|---|---|---|
| C1 | core-generalists | 10 generalists (unrestricted) | dynamics reference |
| C2 | core-specialists | 10 specialists (read+write own random topic) | deep exploitation: feasibility vs novelty |
| C3 | core-broad | 10 broad readers / local writers (read all, write own random topic) | T-shaped researcher — pre-registered winner |
| C4 | core-mixed | 4 specialists + 3 broad + 3 generalists | does behavioral diversity beat pure teams? |

Key contrasts: C1↔C2 (cost/benefit of constraints), C2↔C3 (read breadth,
write fixed), pure↔C4 (composition diversity).

## Supplementary experiments (after core results)

1. **Composition sweep** — `supp-mixed-spec-heavy` (7 spec + 3 gen),
   `supp-mixed-gen-heavy` (3 spec + 7 gen) vs C4 (4+3+3).
2. **Scaling** — `supp-scaling-n1/n2/n5/n20` (generalists, same total steps
   and budget; N=10 point = C1). N=1 is the single-agent baseline; the curve
   is the stigmergy story.
3. **Initial-graph ablation** — `supp-noedges-generalists`: same corpus,
   zero edges; isolates the value of citation structure.
4. **Jump ablation** — `supp-nojump-generalists`: serendipity contribution.
5. **Validity anchors** — `supp-baseline-nonav` (LLM without navigation),
   `supp-baseline-pa` (preferential attachment, no LLM; structural metrics
   only).

## Pre-registered expectations

| condition | accuracy | novelty | bridging | diversity |
|---|---|---|---|---|
| C2 specialists | high | low | low | low |
| C3 broad/local | **high** | **high** | **high** | mid |
| C1 generalists | mid | mid | mid | mid |
| C4 mixed | between C1 and C3 | mid-high | mid-high | **high** |
| nonav | low (vague, dup-flagged) | spurious-high | random | high |

If C3 wins accuracy AND bridging simultaneously, the headline is "read
broadly, write narrowly".

## Analysis plan

- Per-run: `metrics.json` (accuracy + counts), `verdicts.json` (per-idea
  audit trail), `run_meta.json` (seed, topic assignments), `events.jsonl`
  (full action traces).
- With one seed per condition, compare per-idea accuracy across conditions
  (each run yields 40 scored ideas; idea-level bootstrap CIs) and report the
  seed limitation; add seeds before the paper's final claims. Topic
  assignments are recorded and analyzable as a covariate.
- Cost estimate: core matrix ≈ 4 × 400 gpt-5 steps ≈ $25–50 + evaluation
  ≈ $4 (gpt-5-mini, cached).

## Gates before running

1. Human audit of 30 summary samples (sent 2026-08-29) — PENDING.
2. Everything else (corpus, edges, summaries, embeddings, topic pool,
   configs, API keys) is ready.
