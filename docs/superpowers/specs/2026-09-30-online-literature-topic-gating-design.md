# Paper 2 — online literature, topic labels, hard topic gating

Date: 2026-09-30, revised 2026-10-01. Supersedes two parts of
`2026-09-26-forum-architecture-design.md`:

- §4.3 "soft specialization" (topics only in the prompt; the environment
  filters nothing).
- The frozen 16,208-paper corpus as the literature store.

The shared board, wiki link semantics, the action set and the evaluation
(acc@≥2, the paper-1 protocol) are unchanged.

## 1. Why

**Soft specialization failed as a dial.** Measured on the 27-run nested sweep
(`scripts/p2_forum/topic_fidelity.py`):

- At k=1, 52% of corpus reads fell in the agent's topic (chance 1%), and half
  of its posts fell outside it.
- Agents favored the topics listed first in their prompt. At k=128, 48% of
  in-topic reads went to the first 16 listed topics (uniform: 12.5%). With
  nested draws the first-listed topics are exactly the small-k topics, so
  large k barely widened what agents read.

**The k-means topic list was weak.** Its names came from gpt-5-mini from 8
samples, with no consolidation. It had heavy overlaps (five optimization
topics, five distribution-shift topics), generic modifiers ("robust" in 20+
names), malformed names, no definitions, and cluster sizes from 21 to 304.

Paper 2 therefore:

1. adopts paper 1's rule, so the environment enforces specialization on every
   action;
2. takes its topics from an authoritative source, the submission areas of the
   top AI venues;
3. replaces the frozen corpus with online search over those venues.

## 2. Venue scope

The **CCF-A artificial-intelligence conferences** (CCF recommended list,
2026, 7th edition): **AAAI, NeurIPS, ACL, CVPR, ICCV, ICML, ICLR**.

These seven venues define both the topic list (§3) and what the literature
tools can return (§5). The scope is paper 1's corpus venues (NeurIPS, ICLR,
ICML, AAAI) extended with NLP (ACL) and vision (CVPR, ICCV).

## 3. The topic list (built once, then frozen)

1. Collect each venue's official submission areas / subject areas /
   keywords from its 2024 call for papers. The ML venues list ~20-40 areas
   each; AAAI's keyword list is fine-grained (~200); ACL, CVPR and ICCV list
   their tracks or topics.
2. One consolidation pass (the tagger model, §4) merges and deduplicates them
   into **exactly 128 topics**. Each topic gets a name, a one-sentence
   definition and its provenance (which venues' areas it came from). Coarse
   areas are split along the finer keyword lists, and granularity is evened
   out.
3. The user reviews the list. It is then frozen as
   `configs/p2_forum/topics-v2.yaml` and never regenerated. Topic ids are the
   list positions 0-127.

## 4. Labels and the tagger

Every paper, every board post and every search query carries topic labels:
**1-5 topics**, ranked by relevance (first = primary).

**Tagger: an independent LLM.** Claude Sonnet 5 (`claude-sonnet-5`), a
different model family from the agents (gpt-5) and the quality judge
(gpt-5-mini). Agents never label their own work. One call per item: the
frozen list (names + definitions) as a fixed, cacheable prefix, then the
item's text (a paper's title + abstract, a post's text, or a query). It returns
a JSON list of 1-5 topic ids and is told to pick only topics that genuinely
apply, never padding to five. Replies that do not parse to 1-5 valid distinct
ids are retried up to 3 times; after that the item counts as unlabeled, which
means unreadable and, for a post, not published.

**Board stubs** (`corpus_ref` nodes that mirror a cited paper) take the labels
of the paper they mirror.

## 5. The literature tools: online search

The frozen corpus is replaced by live **Semantic Scholar** lookups behind a
shared cache.

**Scope filter, applied everywhere** (search hits, browse targets,
reference and citation lists, random jumps):

- the paper is **published at one of the seven venues OR has ≥ 50
  citations** (any venue, including journals, other conferences and arXiv).
  This is the same rule as paper 1's tier-1 recognition (venue alias OR
  `recognized_min_citations: 50`). Venues are matched on Semantic Scholar's
  venue records;
- **and** the publication date is **on or before 2024-09-30**, the cutoff of
  paper 1 and the agent model's official knowledge cutoff;
- there is no lower year bound.

The citation count is Semantic Scholar's current count, not the count as of
the cutoff, so the ≥ 50 branch admits papers partly by impact accrued after
2024-09-30. The venue branch has no such leak. Each admitted paper records
which branch admitted it, so the analysis can report the split.

Search is therefore not venue-restricted at the source. Semantic Scholar
returns any paper, and the scope filter then applies the rule. Non-AI papers
that pass the citation branch are removed by the topic gate, since every
topic is an AI topic.

**Topic gate, two layers:**

1. **Query gate.** A `search` query is labeled first. If its labels do not
   intersect the agent's topics, the search is refused:
   `{"error": "this query is outside your topics", "topics": [...]}`.
2. **Result gate.** Every candidate paper is labeled and kept only if
   `labels(paper) ∩ topics(agent) ≠ ∅`.

| action | behavior |
|---|---|
| `search` | query gate → Semantic Scholar search (over-fetch ~50) → scope filter → label each → result gate → top k, each with title, abstract snippet, year, venue and topic names |
| `browse` | target must pass the scope filter and the result gate, otherwise error. Returns title, full abstract, year, venue, topic names, and `cites` / `cited_by` lists (each scope-filtered and result-gated, up to 10) |
| `sample_frontier` | picks one of the agent's topics at random, searches its name, returns a random gated hit |

**Cache.** Every fetched paper record (metadata, abstract, references,
citations), every search response and every label is stored in a cache that
all runs share (`data/online_cache/`), with fetch timestamps. A cached item is
never refetched. Each run also logs everything an agent was shown in its event
log. `--resume` replays from the log, so a resumed run matches an
uninterrupted one even if Semantic Scholar's index drifts.

## 6. The board and the remaining actions (unchanged from the gating design)

`readable(agent, item) := labels(item) ∩ topics(agent) ≠ ∅`, with no exception
for the agent's own posts.

| action | rule |
|---|---|
| `search_board` | board search restricted to readable posts (the query gate also applies) |
| `browse_board` | error if unreadable; `cites` / `cited_by` list only readable items |
| `sample_board` | uniform over readable posts |
| `generate` | the post is labeled first. If it does not intersect the agent's topics it is **not published** (`{"error": "this idea is outside your topics", "topics": [...]}`). Otherwise it is published; unreadable cited ids are dropped and reported (`dropped_cites`), as in paper 1 |
| `add_links`, `remove_links` | error unless the source and every destination are readable |

The generate gate means every published post is readable by its author.
Filtering returned results is the primary mechanism: an agent never sees an
unreadable id. The citation and link checks are a safety net for invented or
copied ids; every firing is logged. A refused action still consumes the
agent's step.

**k = 128** makes every in-scope item readable and accepts every post: the
ungated control.

**Post format is unchanged from paper 1:** a 3-4 sentence idea paragraph
(problem, key insight, method; ~125 words, no results). Agents now read raw
abstracts (~180 words, with results) but write proposals, which have no
results yet. The judge compares idea paragraphs with realizing papers'
abstracts, exactly as in paper 1.

## 7. Topic draws and the prompt

- Topic sets stay nested (`topic_draw: nested`): an agent's k-topic set is
  the first k of its per-agent permutation.
- The display order in the system prompt is shuffled independently, with its
  own per-agent seed, which removes the first-listed bias.
- The prompt says the agent can only search, read and publish within its
  topics, and shows each topic's name and definition.

## 8. Configuration

- `literature: online` (new; `corpus` keeps the frozen-corpus mode for the old
  runs)
- `online.venues`: the seven venues; `online.min_citations_any_venue: 50`;
  `online.max_pub_date: "2024-09-30"`
- `online.cache_dir: data/online_cache`
- `topics_file: configs/p2_forum/topics-v2.yaml`
- `gating: topics` (default `none`)
- `models.tagger: claude-sonnet-5`

## 9. Experiment

- Rerun the nested sweep: k ∈ {1, 16, 32, 48, 64, 80, 96, 112, 128} × seeds
  {0, 1, 2}, N = 10 agents, 40 rounds (400 steps), resumable for extension.
- Evaluate quality with **paper 1's evaluation, unchanged**: the same judge
  (gpt-5-mini, medium), 3 queries × (Semantic Scholar + OpenAlex) top 5,
  levels 0-5, the same three cumulative tiers (tier 1 = CCF-A venue OR ≥ 50
  citations; tier 2 = CCF-A/B OR ≥ 10; tier 3 = any published), candidates
  dated ≤ 2024-09-30 excluded, headline **tier-1 acc@≥2** counting only
  realizations dated ≥ 2025-06-01, and near-duplicate flooring at cosine ≥
  0.95.
- The one adaptation is the contamination guard. Paper 1 excluded
  candidates whose title was in its frozen corpus. Paper 2 has no frozen
  corpus, so it excludes candidates whose title is in the online cache (every
  paper any agent was shown). The guard is redundant in practice, because
  cached papers all predate 2024-09-30 and the date gate already removes
  them, but it keeps paper 1's rule intact. The near-duplicate check runs
  against the cached papers' embeddings.
- The 27 soft runs on the frozen corpus are kept as the soft-specialization
  condition.

## 10. Measurements added

- Tagging quality: labels per item (a pile-up at 5 means padding, so revise
  the prompt); stability (re-label 200 papers and measure agreement); a user
  spot check.
- Gate activity per run: refused queries, refused posts (the rejection rate),
  results filtered by scope and by topic, dropped citations, blocked links. A
  sample of refused posts and queries is reviewed to separate genuine
  off-topic attempts from labeling errors.
- What each agent read: distinct papers, venues and years.

## 11. Cost and risks

- **Tagger volume.** Each search can label up to ~50 candidates, and each
  browse labels its reference and citation lists. Cost is front-loaded, and
  the shared cache makes repeated papers free. The list prefix is cached.
- **Semantic Scholar rate limits** (~1 request/s with a key). Parallel runs
  contend; the cache and the existing backoff/circuit-breaker mitigate this,
  and throughput sets the sweep's wall-clock time.
- **Venue matching.** Semantic Scholar's venue strings vary; matching uses its
  venue ids plus an alias list, tested against known papers from each venue.
- **Abstract gaps.** Papers without an abstract are shown by title and labeled
  from the title; their count is reported.

## 12. Testing

- Scope filter: out-of-venue and post-cutoff papers never appear in any
  result, list or jump.
- Gates, one test per action: refused queries, gated results, browse of an
  unreadable or out-of-scope id, sampling, refused off-topic posts that leave
  the board unchanged, dropped citations, blocked links.
- Own-post invariant: every published post is readable by its author.
- k = 128 equals no topic gating.
- Tagger: parsing, 1-5 bounds, retry, cache use.
- Cache and resume: a resumed run makes zero network and zero tagger calls
  for replayed steps. All tests use a fake Semantic Scholar client; no test
  touches the network.

## 13. Out of scope

- Full text (abstracts only).
- Listing the posts that cite a paper when browsing that paper.
- Regenerating the topic list after it is frozen.
