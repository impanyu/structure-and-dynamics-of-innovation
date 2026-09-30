# Paper 2 — topic labels and hard topic gating

Date: 2026-09-30. Supersedes the "soft specialization" rule of
`2026-09-26-forum-architecture-design.md` §4.3 (topics only in the prompt, the
environment filters nothing). The shared-board architecture, the nine actions,
wiki link semantics and the evaluation are unchanged.

## 1. Why

The soft design failed as a dial. Measured on the 27-run nested sweep
(`scripts/p2_forum/topic_fidelity.py`):

- Topics steered agents only partway: at k=1, 52% of corpus reads fell in the
  agent's topic (chance 1%); half of its posts fell outside it.
- Agents favored the topics listed first in their prompt: at k=128, 48% of
  in-topic reads went to the first 16 listed topics (uniform: 12.5%). With
  nested draws the first-listed topics are exactly the small-k topics, so large
  k barely widened what agents actually read.

Paper 2 therefore adopts paper 1's rule: specialization is enforced by the
environment. Every action is gated by the agent's topics.

## 2. The topic list (built once, then frozen)

1. Cluster the 16,208 corpus idea embeddings (BAAI/bge-small-en-v1.5, the
   environment's embedder) into ~300 candidate clusters with k-means (seed 0).
   Candidates only; not topics.
2. Name each candidate with gpt-5-mini from 8 sample ideas.
3. One consolidation call (gpt-5) reads all candidate names with one-line
   samples and returns **exactly 128 topics**, each a name plus a one-sentence
   definition: merge duplicates and near-synonyms, split over-broad
   candidates, even out granularity.
4. The user reviews the list. It is then frozen as
   `configs/p2_forum/topics-v2.yaml` and never regenerated. Topic ids are the
   list positions 0-127.

## 3. Labels

Every paper and every board post carries **1-5 topic labels**, ranked by
relevance (first = primary).

**Tagger.** One LLM call per item: the frozen list (names + definitions) as a
fixed, cacheable prefix, then the item's idea text. The model returns a JSON
list of 1-5 topic ids, instructed to pick only topics that genuinely apply and
never to pad to five. Model: gpt-5-mini (effort low). Replies that do not parse
to 1-5 valid distinct ids are retried; after 3 failures the item is an error
(the corpus build stops; a post is rejected with an error, see §4).

**Corpus.** Tagged once, offline, into `data/stage1/topic_labels_v2.json`
(`{paper_id: [topic ids]}`), through the existing disk LLM cache. About 25M
input tokens, a few dollars.

**Posts.** Tagged synchronously inside `generate`, with the same tagger, before
the post exists. The labels are written into the event's result, and
`--resume` reuses logged labels instead of calling the model again, so a
resumed run is identical to an uninterrupted one.

**Board stubs** (`corpus_ref` nodes that mirror a cited paper) take the corpus
labels of the paper they mirror.

## 4. The gate (paper 1's rule, all nine actions)

`readable(agent, item) := labels(item) ∩ topics(agent) ≠ ∅`

No exception for the agent's own posts. None is needed: the generate gate
below guarantees every post an agent publishes is readable by its author.

| action | rule |
|---|---|
| `search`, `search_board` | exact search restricted to readable items; returns the top k readable |
| `browse`, `browse_board` | error if the target is unreadable; `cites` / `cited_by` list only readable items |
| `sample_frontier`, `sample_board` | uniform over readable items; error if none are readable |
| `generate` | the post is tagged first; if its labels do not intersect the agent's topics it is **not published** and the agent gets `{"error": "this idea is outside your topics", "topics": [...]}`. Otherwise it is published; cited ids that are unreadable are dropped and reported (`dropped_cites`), as in paper 1 |
| `add_links`, `remove_links` | error unless the source and every destination are readable |

Filtering the returned results is the primary mechanism: an agent never sees an
unreadable id. The checks on `generate` citations and on links are a safety net
for invented or copied ids; every firing is logged.

A rejected `generate` still consumes the agent's step, like any failed action.

**Display.** Every item returned to an agent shows its topic names, e.g.
`"topics": ["Graph neural network expressivity", "Molecular ML"]`.

**k = 128** makes every item readable (each carries at least one label) and
accepts every post: the ungated control.

## 5. Topic draws and the prompt

Topic sets stay nested (`topic_draw: nested`): an agent's k-topic set is the
first k of its per-agent permutation, so a smaller k's set is a strict subset
of a larger k's.

The **display order** in the system prompt is shuffled independently, with its
own per-agent seed. That removes the first-listed-topic bias found in §1. The
prompt drops the soft-specialization line ("Most of what you find will be
outside your interests"); it states instead that the agent can only read and
publish within its topics.

## 6. Configuration

- `topics_file: configs/p2_forum/topics-v2.yaml`
- `topic_labels: data/stage1/topic_labels_v2.json`
- `gating: topics` (new; default `none` keeps old configs reproducible)
- `models.tagger: openai:gpt-5-mini:low`

## 7. Experiment

- Rerun the nested sweep under gating: k ∈ {1, 16, 32, 48, 64, 80, 96, 112,
  128} × seeds {0, 1, 2}, N = 10 agents, 40 rounds (400 steps), resumable for
  extension.
- Evaluate quality (acc@≥2, the paper-1 protocol) after the runs.
- The 27 existing soft runs are kept as the soft-specialization condition, so
  the paper can compare soft against hard specialization.

## 8. Measurements added

- Tagging quality: labels per item (a pile-up at 5 means padding, so revise the
  prompt); stability (re-tag 200 corpus items, measure agreement); a user spot
  check.
- Gate activity per run: generate rejections (the rejection rate), dropped
  citations, blocked link operations. A sample of rejected posts is reviewed to
  separate genuinely off-topic ideas from tagging errors.
- Readable share of the corpus per k (the real size of each agent's world).

## 9. Testing

- Gate unit tests, one per action: unreadable items never appear in results,
  browse of an unreadable id errors, sampling stays readable, an off-topic
  generate is refused and leaves the board unchanged, citations are dropped,
  and links are blocked.
- Own-post invariant: every published post is readable by its author.
- k = 128 equals no gating on a fixed seed.
- Tagger: parsing, 1-5 bounds, retry, and cache use.
- Resume: a resumed gated run reproduces labels and decisions from the log
  with zero tagger calls for replayed steps.

## 10. Out of scope

- Listing the posts that cite a paper when browsing that paper (a new
  collaboration channel; not added).
- Regenerating the topic list after it is frozen.
