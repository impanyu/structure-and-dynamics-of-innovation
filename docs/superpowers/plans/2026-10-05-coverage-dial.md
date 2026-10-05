# Coverage Dial Implementation Plan (revision of the online/gating plan)

> Executed with superpowers:subagent-driven-development; per-task briefs live in the SDD workspace.

**Goal:** replace the k-topic dial with a coverage dial: a synonym-only topic list (umbrellas kept), a 2023-24
reference paper sample labeled by the experiment's tagger, and a seed-expansion algorithm that gives each agent a
nested topic set whose sample coverage is just below each target c.

**Spec:** docs/superpowers/specs/2026-09-30-online-literature-topic-gating-design.md §3, §4, §7-§9 (revised 2026-10-05).

| Task | Deliverable |
|---|---|
| C1 | AnthropicLLM effort suffix (`claude-sonnet-5:medium`); tagger rule "1-3 most specific + their umbrellas, ≤ 8"; load_topics without a fixed count |
| C2 | topics-v3 builder: synonym-only dedupe of the 428 venue areas → draft list (live); STOP for user review, then freeze |
| C3 | reference sample: 7 venues × 2023/2024 from S2, stratified ~4,000; label all with the frozen list (live); coverage table |
| C4 | coverage + seed-expansion assignment module (pure, tested) |
| C5 | runner/config/prompt integration: `coverage` per agent, `topic_draw: coverage`, 9 configs (c grid, seed 0), all-topics prompt case |
| C6 | smoke run, cost report, user approval, sweep |
