# Paper 2 — online literature, topic labels, hard topic gating

Date: 2026-09-30, revised 2026-10-01. Supersedes two parts of
`2026-09-26-forum-architecture-design.md`:

- §4.3 "soft specialization" (topics only in the prompt; the environment
  filters nothing).
- The frozen 16,208-paper corpus as the literature store.

The shared board, wiki link semantics, the action set and the evaluation
(acc@≥2, the paper-1 protocol) are unchanged.

> **REVISION 2026-10-06 (supersedes §3-§5 and the topic labels/tagger).**
> The user dropped LLM topic labeling and online search for cost reasons and returned to paper 1's design:
> - **Literature = a frozen corpus**: the seven CCF-A AI venues (AAAI, NeurIPS, ACL, CVPR, ICCV, ICML, ICLR), 2020-2024,
>   **citations >= 50**, built with paper 1's Semantic Scholar pipeline (cmd_fetch `mode: s2_venues`, same date rule:
>   published before the agent model's cutoff month), citation edges from S2 references + citations + OpenAlex augmentation.
> - **Paper text = the original abstract** (no LLM idea compression); embeddings (BAAI/bge-small-en-v1.5) of title + abstract.
> - **Specialization = paper 1's semantic-region mechanism, as a nearest-neighbour ball**: each agent gets a seed corpus paper
>   (seeded random); its readable region at coverage c is the c x N corpus papers most similar to the seed (exact coverage,
>   nested across c). A board post is in an agent's region iff at least 3 of its 5 nearest corpus papers (cosine, K=5, M=3) are
>   members of that region (nearest-neighbour majority), and is readable by that agent iff so (its author can always read it; since 2026-10-08
>   there is no publish gate). Why: posts are idea paragraphs and papers are abstracts, so seed similarity wrongly refused posts
>   whose nearest papers lay inside the region (live smoke, coverage 1%).
> - Every action stays gated (search = local semantic search over the agent's readable papers; browse lists only readable
>   references/citations; random jumps and `related` stay within the ball; links need readable ends). No query gate.
> - **Coverage grid: c in {1, 5, 10, 20, 30, 40, 50, 100}%**, one seed, N = 10 agents, 40 rounds; more values later if needed.
>   Figures use each run's mean achieved coverage (exact here).
> - Unchanged: shared board, wiki links, post format, evaluation (Opus 5.5 judge, tier rules, exclude_workshops), standalone framing.
> - Retired (code kept, unused): online Semantic Scholar/OpenAlex literature, topic lists v2/v3, the tagger, coverage sample.
>
> **2026-10-07 R5 (soft topics + hard gate; ranked listings).** In run forum-region-c20-s0 agents were never told their area,
> and search silently returned the best in-region papers however irrelevant, so agents kept searching out-of-area titles and
> their posts were refused. Now:
> - **Soft topic list plus the hard gate.** Each region's members are clustered with k-means (bge-small embeddings,
>   n = clamp(round(N_members / 150), 3, 30), seeded) and each cluster is named from its 8 most central papers by one
>   `models.topic_namer` call (gpt-5-mini, low effort; cached, so identical regions share names). The agent's prompt lists
>   the topics (name, one-sentence description) and states that it can only find, read, cite and publish within them.
>   Topics are recorded in run_meta and reused on resume. The region gate is unchanged.
> - **Ranked, tiered, paginated listings.** search, related, search_board and the reference/cited-by lists of browse and
>   browse_board are ranked by cosine, 10 per page (`page`, or `ref_page`/`cited_by_page`), each item tagged high/medium/low
>   relevance instead of a raw score (query tiers 0.80/0.72, paper tiers 0.85/0.78, calibrated on data/p2_corpus).
> - **Search notice.** Every search (and related) result says the results are restricted to the agent's topics, and that
>   only low-relevance hits mean the target is outside them: rephrase toward the topics rather than repeat the search.
>
> **2026-10-08 (read gate only; publishing is free).** In run forum-region-c20-s0b, 14 of 50 posts were refused although every
> refused post cited only in-region papers: combining what an agent read often lands an idea at the region's edge, where
> fewer than 3 of its 5 nearest papers are members. Researchers are limited in what they read, not in what they may publish,
> so the publish gate is removed. A post's cites must still be readable by its author (others are dropped). Its author can
> always reread it; any other agent can read it iff the nearest-neighbour majority rule places it in that agent's region.
> The prompt now says the agent may publish any idea but can cite only what it can read.
> Same day, the user unified the read rule: a post is readable by an agent under exactly the rule that makes a paper a
> member, i.e. iff its cosine to the agent's seed is at least the ball's radius (region.contains_vec), plus authors always
> read their own posts. The nearest-neighbour majority rule (K=5, M=3) is retired.
>
> **2026-10-08 R6 (one tool set over papers and posts; symmetric prompt).** In run forum-region-c20-s0c agents searched the
> literature 160 times but the board only 9 times, and never opened a teammate's post (0 cross-agent citations): the prompt
> described the board far more thinly than the literature, and board search was a separate tool to remember. A single merged
> ranking would bury the posts (simulated on that run: a readable teammate post reaches the top 10 in only 6 of 123 searches;
> median best-post rank 488). So, like a scholar search engine listing papers and preprints side by side:
> - **Unified tools with sectioned results.** Region mode documents exactly `search`, `browse`, `related`, `random`,
>   `generate`, `add_links`, `remove_links`. Papers and posts share one id space (posts start with `gen:`) and every item
>   carries `kind`. `search(query, page, post_page)` and `related(node_id, page, post_page)` return two separately ranked
>   sections, `papers` (10 per page) and `posts` (5 per page, the agent's own marked `author: "you"`), with one notice;
>   each section honours its store's search switch. Query tiers serve both sections (query -> cited papers and query -> the
>   next post both median 0.77; random post 0.67, random paper 0.64). `random(kind)` jumps to a readable paper or post with
>   the same rng draws as before. `search_board`, `browse_board`, `sample_frontier`, `sample_board` remain as undocumented
>   aliases, so old logs restore and pre-R6 runs resume.
>   Result key order is posts, a short notice, then papers (and a paper's `cited_by_posts` before `cites`/`cited_by`),
>   because the agent's rolling history keeps only the first 1500 characters of each result. The posts section also carries
>   `filtered`: the number of board posts the agent cannot read.
> - **cited_by_posts.** `browse` opens a paper (abstract, `cites`, `cited_by`, and `cited_by_posts`: readable board posts
>   citing it, ranked by cosine to the paper, 10 per page on `post_page`; hidden ones counted in `filtered.region_posts`) or
>   a post (full text, author, mixed `cites` with kinds, `cited_by`).
> - **Symmetric prompt.** The LITERATURE and the BOARD get parallel descriptions ("starts empty" is gone), one sentence
>   explains that one search covers both, the goal asks the agent to keep up with both the literature and its peers' posts
>   and cite either, and the topic rule speaks of "papers and posts". Gate rules are unchanged.

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

These seven venues define the **topic list** (§3): it is built from their
submission areas, unchanged.

What the literature tools can return (§5) uses the evaluation's tier-1 venue
rule instead, so reading and recognition share one source of truth: the venue
set is `recognized_venues` (the CCF-A list across computer science, 60
venues, inherited from paper 1's config), plus any paper with at least
`eval.recognized_min_citations` (50) citations. The topic gate (§4) keeps
reading within the AI topics.

## 3. The topic list and the coverage dial (revised 2026-10-05)

**The dial is coverage, not a topic count.** Experiments set each agent's
**coverage c** = the share of a reference paper sample its topics can read.
Topics may be of very different sizes; coverage measures the reading world
directly. (This replaces the earlier "exactly 128 topics, capacity-balanced"
design; that pipeline is retired.)

**3.1 Topic list.** The 428 official submission areas of the seven venues
(configs/p2_forum/venue_areas.yaml) are deduplicated **only for synonyms**
(e.g. "Reinforcement learning" at several venues). Umbrella areas and fine
areas both stay ("Deep Learning" next to "Graph Neural Networks"), as in real
researchers' interest lists. Expected size: about 300-400 topics, each with a
name, one-sentence definition and provenance. The user reviews it; it is then
frozen as `configs/p2_forum/topics-v3.yaml` (ids = list positions; no fixed
count).

**3.2 Reference sample.** Papers of the seven venues (AAAI, NeurIPS, ACL, CVPR,
ICCV, ICML, ICLR) published in **2023 and 2024** (ICCV only in 2023), fetched
from Semantic Scholar by venue and year, **stratified-sampled to ~4,000
papers** in proportion to each venue-year's size. Each paper is labeled with
exactly the experiment's tagger settings (§4). The result is a paper→labels
table; the coverage of any topic set S is the share of sample papers with at
least one label in S (the gate's "any label" rule).

**3.3 Assigning topics for a target coverage c (greedy seed expansion).**
1. Seed: random (per agent, seeded) among topics whose own sample coverage ≤ c.
2. Order every other topic by distance from the seed, nearest first. Distance
   is the co-labeling rate with the seed on the sample, with name+definition
   embedding similarity as the tie-breaker.
3. Scan that order once: add a topic if the set's actual coverage after adding
   it is ≤ c; otherwise skip it. Stop when every topic has been tried.
4. Nesting: an agent's targets are processed from smallest to largest c. Each
   larger target starts from the previous target's set and scans the same
   order again, so a smaller-c set is a subset of a larger-c set.
5. No tolerance parameter: the target c is only an upper bound. Coverage
   never exceeds c.
6. Analysis: a run's x-value is the **mean achieved coverage of its agents**
   (not the nominal c). Figures also show each run's min-max range across
   agents.
7. run_meta.json records, per agent and target, the seed, the topic ids and
   the achieved coverage.

## 4. Labels and the tagger

Every paper, every board post and every search query carries topic labels:
the **1-3 most specific matching topics plus every broader (umbrella) topic
that contains them, at most 8**, most specific first.

**Tagger: an independent LLM.** Claude Sonnet 5 (`claude-sonnet-5`) with **medium thinking effort** (user decision: moderate thinking), a
different model family from the agents (gpt-5) and the quality judge
(gpt-5-mini). Agents never label their own work. One call per item: the
frozen list (names + definitions) as a fixed, cacheable prefix, then the
item's text (a paper's title + abstract, a post's text, or a query). It returns
a JSON list of 1-8 topic ids following the rule above, never padding. A refusal (stop_reason=refusal) is retried, then the item is unlabeled and the refusal logged. Replies that do not parse to 1-8 valid distinct
ids are retried up to 3 times; after that the item counts as unlabeled, which
means unreadable and, for a post, not published.

**Board stubs** (`corpus_ref` nodes that mirror a cited paper) take the labels
of the paper they mirror.

## 5. The literature tools: online search

The frozen corpus is replaced by live **Semantic Scholar** lookups behind a
shared cache.

**Scope filter, applied everywhere** (search hits, browse targets,
reference and citation lists, random jumps):

- the paper is **published at a tier-1 venue (the CCF-A list across computer
  science, `recognized_venues`) OR has ≥ 50 citations** (any venue, including
  journals, other conferences and arXiv). This is exactly the evaluation's
  tier-1 rule (venue alias OR `eval.recognized_min_citations: 50`), built from
  the same config keys by one helper. Venues are matched on Semantic Scholar's
  venue records, and a venue string containing "workshop" never satisfies the
  venue branch (a workshop paper can still qualify through the ≥ 50 citation
  branch);
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
| `related` | source must be readable; returns up to k readable recommendations from Semantic Scholar's embedding-based recommender (all-cs pool), scope-filtered and result-gated; closed with the corpus search channel |
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
abstracts.

## 7. Topic assignment and the prompt

- Topic sets come from the coverage algorithm (§3.3); `topic_draw: coverage`.
- The display order in the system prompt is shuffled with its own per-agent
  seed (no first-listed bias).
- The prompt says the agent can only search, read and publish within its
  topics, and lists them (name + definition). An agent whose set is every topic
  (c = 100%) is told it may work on any AI topic instead of receiving the full
  list.

## 8. Configuration

- `literature: online` (new; `corpus` keeps the frozen-corpus mode for the old
  runs)
- `online.max_pub_date: "2024-09-30"`. The reading scope's venues and
  citation floor come from `recognized_venues` and
  `eval.recognized_min_citations`; `online.venues` and
  `online.min_citations_any_venue` are removed and a config that still sets
  them is refused
- `online.cache_dir: data/online_cache`
- `topics_file: configs/p2_forum/topics-v3.yaml`; `coverage_sample: data/p2_forum/coverage_sample/` (papers + labels)
- `gating: topics` (default `none`)
- `models.tagger: claude-sonnet-5:medium` (medium thinking effort)
- `models.judge: claude-opus-5-5`
- `models.agent: openai:gpt-5:medium` (written out explicitly; it is the API default)

## 9. Experiment

- Sweep the coverage dial: c ∈ {1%, 2%, 5%, 10%, 20%, 35%, 50%, 75%, 100%},
  **one seed (seed 0) per c**, N = 10 agents, 40 rounds (400 steps),
  resumable for extension. Figures plot outcomes against each run's mean
  achieved coverage, with the across-agent range.
- **Paper 2 is a standalone study.** Its evaluation is reported on its own
  terms, with no comparison line to paper 1 and no shared judge.
- **Judge: Claude Opus 5.5** (`claude-opus-5-5`), independent of the agents
  (gpt-5), so it cannot favor its own family's ideas. It writes the search
  queries and assigns the levels.
- **Protocol** (the existing evaluator, `innovation.core.eval`): 3 judge
  queries × (Semantic Scholar + OpenAlex) top 5; similarity levels 0-5; three
  cumulative recognition tiers (tier 1 = CCF-A venue OR ≥ 50 citations; tier 2
  = CCF-A/B OR ≥ 10; tier 3 = any published); candidates dated ≤ 2024-09-30
  excluded; headline **tier-1 acc@≥2**, counting only realizations dated ≥
  2025-06-01 (the agent model's recall horizon); ideas with cosine ≥ 0.95 to a
  paper the agents could read are floored to 0.
- **Workshops:** `eval.exclude_workshops: true` (new, default false, set in
  the online config) stops a venue string containing "workshop" from
  satisfying the tier-1 or tier-2 venue (alias) branches, matching the reading
  rule; the citation branches still apply (≥ 50 → tier 1, ≥ 10 → tier 2,
  otherwise tier 3). Paper 1's evaluation leaves the flag off.
- **Contamination guard:** candidates whose title is in the online cache
  (every paper any agent was shown) are excluded. The near-duplicate check
  runs against the cached papers' embeddings.
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
