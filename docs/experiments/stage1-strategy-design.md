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

- Same graph, `total_steps: 400`, NO generation budget (user decision
  2026-08-29): agents generate freely; idea count is an outcome variable.
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

## Core matrix — the specialization dial (redesign 2026-08-30)

One axis: how narrowly agents READ. All conditions are 10 agents, writes
always free; C2–C6 confine each agent's reading to its random topic's
read_mass nearest corpus papers. Same seed everywhere → identical topic
draws; only the radius varies. Motivation: measured purity of the 800-paper
region is 5–60% own topic (median ~35%), so read_mass is a continuous
specialization dial, not a fixed "one topic" scope. (The former C3
broad-reader/local-writer and C4 mixed-team conditions were withdrawn.)

| id | config | read_mass | interpretation |
|---|---|---|---|
| C2 | core-mass-400 | 400 | narrow — ≈ 1–2 natural topics |
| C3 | core-mass-800 | 800 | sub-area neighborhood (**already run** as `core-mass-800-s0`) |
| C4 | core-mass-1600 | 1600 | broad sub-area |
| C5 | core-mass-3200 | 3200 | macro-area (~20% of corpus) |
| C6 | core-mass-6400 | 6400 | ~40% of corpus |
| C1 | core-generalists | ∞ | unrestricted endpoint (already run) |

Headline question: is accuracy monotone in scope, or is there an
inverted-U (a "focused-but-not-starved" sweet spot)? C3 beating C1 on all
tiers already rules out "broader is always better".

## Supplementary experiments (after core results)

1. **Composition sweep** — `supp-mixed-spec-heavy` (7 spec + 3 gen),
   `supp-mixed-gen-heavy` (3 spec + 7 gen). NOTE: the 4+3+3 mixed anchor
   (old C4) was withdrawn; redesign this sweep before running.
2. **Scaling** — `supp-scaling-n1/n2/n5/n20` (generalists, same total steps
   and budget; N=10 point = C1). N=1 is the single-agent baseline; the curve
   is the stigmergy story.
3. **Initial-graph ablation** — `supp-noedges-generalists`: same corpus,
   zero edges; isolates the value of citation structure.
4. **Jump ablation** — `supp-nojump-generalists`: serendipity contribution.
5. **Validity anchors** — `supp-baseline-nonav` (LLM without navigation),
   `supp-baseline-pa` (preferential attachment, no LLM; structural metrics
   only).

## Pre-registered expectations (revised for the dial, 2026-08-30)

Accuracy vs read_mass is expected to be an **inverted U**: very narrow
scopes starve agents of recombination material (and diffuse topics can
produce zero output, cf. s6 at mass 800), very broad scopes reproduce the
generalist's unfocused wandering. The observed C3(800) > C1(∞) on all
tiers pins the right side of the curve; the open question is where the
peak sits and how steep the narrow side is. Secondary expectations:
novelty and per-idea citation spread grow with mass; idea count may peak
at moderate mass. nonav baseline: low accuracy (vague, dup-flagged)
regardless of scope.

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
