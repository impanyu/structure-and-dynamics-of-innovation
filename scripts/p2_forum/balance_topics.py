"""One-shot (Task 2c): venue_areas.yaml -> a uniform-granularity, capacity-
balanced 128-topic DRAFT (configs/p2_forum/topics-v2.draft.yaml) plus its
capacity table (configs/p2_forum/topics-v2.capacity.json).

Step A: one LLM call maps every venue's areas onto AAAI's keyword granularity.
Step B/C: estimate each topic's capacity (live S2 bulk-search totals x tagger
precision), then split/merge until all capacities lie in [tau/sqrt3, tau*sqrt3]
or 3 rounds have run. Every LLM reply and S2 response is cached, so a crashed
run resumes. The user reviews the draft; renaming it to topics-v2.yaml freezes it.
Run: uv run python scripts/p2_forum/balance_topics.py
"""
import argparse
import threading
import time
from pathlib import Path

import yaml

from innovation.core.config import load_env
from innovation.core.ideas.embed import Embedder
from innovation.core.llm import AnthropicLLM, CachedLLM, RoutedLLM
from innovation.p2_forum.balance import LLMRewriter, Similarity, run_balance, write_outputs
from innovation.p2_forum.capacity import BulkCounter, CapacityEstimator
from innovation.p2_forum.tagger import TopicTagger
from innovation.p2_forum.topics import backbone_prompt, parse_topic_list

N, MODEL = 128, "claude-sonnet-5"
OUT_YAML = Path("configs/p2_forum/topics-v2.draft.yaml")
OUT_JSON = Path("configs/p2_forum/topics-v2.capacity.json")
CACHE = Path("data/online_cache")
HEADER = ("# DRAFT topic list (uniform AAAI-keyword granularity, capacity-balanced), built from\n"
          "# configs/p2_forum/venue_areas.yaml by scripts/p2_forum/balance_topics.py; capacities\n"
          "# in topics-v2.capacity.json. Review, edit, then rename to topics-v2.yaml to freeze it\n"
          "# (ids = list positions).\n")


class Counting:
    """Counts calls that reach the provider (placed under CachedLLM)."""

    def __init__(self, inner):
        self.inner, self.calls, self._lock = inner, 0, threading.Lock()

    def complete(self, **kw):
        with self._lock:
            self.calls += 1
        return self.inner.complete(**kw)


def backbone_draft(llm, venues) -> list[dict]:
    names = {v["venue"] for v in venues}
    prompt, feedback = backbone_prompt(venues, N), ""
    for _ in range(3):
        reply = llm.complete(model=MODEL, system="You design research taxonomies.",
                             user=prompt + feedback, max_tokens=16000)
        try:
            return parse_topic_list(reply, N, names)
        except ValueError as e:
            feedback = (f"\n\nYour previous answer was rejected: {e}. Fix it: return "
                        f"exactly {N} objects (merge the closest topics if you have too "
                        f"many) and cite only these venues: {', '.join(sorted(names))}.")
    raise SystemExit("could not obtain a valid backbone topic list in 3 attempts")


def summary(result) -> str:
    lines = ["| round | max/min | out of band | tau | unlabeled samples |", "|---|---|---|---|---|"]
    for h in result["history"]:
        r = h["max_min_ratio"]
        lines.append(f"| {h['round']} | {f'{r:.2f}' if r else 'inf'} | {h['out_of_band']} "
                     f"| {h['tau']:.0f} | {h['unlabeled_samples']} |")
    lines += ["", "Exceptions:" if result["exceptions"] else "Exceptions: none"]
    lines += [f"- {e['id']}. {e['name']} ({e['capacity']:.0f}, {e['side']}): {e['reason']}"
              for e in result["exceptions"]]
    lines += ["", "Final list (i. name | capacity | sources):"]
    lines += [f"{i}. {t['name']} | {rec['capacity']:.0f} | {', '.join(t['sources'])}"
              for i, (t, rec) in enumerate(zip(result["topics"], result["estimate"].records))]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()
    load_env()
    venues = yaml.safe_load(open("configs/p2_forum/venue_areas.yaml"))["venues"]
    tier1 = [v["name"] for v in yaml.safe_load(open("configs/p1_dial/stage1.yaml"))["recognized_venues"]]
    # Thinking off for generation (as build_topic_list.py); the tagger keeps
    # the model default, exactly as it labels at run time.
    gen = Counting(RoutedLLM(anthropic_factory=lambda: AnthropicLLM(thinking={"type": "disabled"})))
    # Topic-list building only: thinking off for the tagger to cut cost (user decision 2026-10-04); experiment runs keep thinking.
    tag = Counting(RoutedLLM(anthropic_factory=lambda: AnthropicLLM(thinking={"type": "disabled"})))
    gen_llm, tag_llm = CachedLLM(gen, CACHE / "llm"), CachedLLM(tag, CACHE / "llm")

    topics = backbone_draft(gen_llm, venues)
    print(f"step A: backbone draft with {len(topics)} topics", flush=True)
    counter = BulkCounter(CACHE / "capacity", tier1)
    taggers = []

    def make_tagger(ts):
        taggers.append(TopicTagger(llm=tag_llm, model=MODEL, topics=ts,
                                   refusal_log=CACHE / "capacity" / "tagger_refusals.jsonl"))
        return taggers[-1]
    estimate = CapacityEstimator(
        counter=counter, llm=gen_llm, model=MODEL, workers=args.workers,
        tagger_factory=make_tagger,
        log=lambda m: print(m, flush=True))
    result = run_balance(topics, estimate, LLMRewriter(gen_llm, MODEL),
                         Similarity(Embedder().encode), n=N, max_rounds=args.max_rounds,
                         log=lambda m: print(m, flush=True))
    run = {"wall_clock_s": round(time.time() - t0), "s2_live_calls": counter.live_calls,
           "tagger_live_calls": tag.calls, "generation_live_calls": gen.calls,
           "tagger_refusals": sum(t.refusals for t in taggers),
           "unlabeled_samples_final": result["estimate"].unlabeled}
    write_outputs(result, OUT_YAML, OUT_JSON, HEADER, run)
    print(f"wrote {OUT_YAML} and {OUT_JSON}; {run}")
    print(summary(result))
    print(f"tagger refusals: {run['tagger_refusals']}")


if __name__ == "__main__":
    main()
