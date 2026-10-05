"""One-shot: venue_areas.yaml -> configs/p2_forum/topics-v3.draft.yaml.

Synonym-only deduplication of the 428 official submission areas (umbrella
and fine areas kept side by side; only literal "Other ..." bins dropped).
The user reviews the draft; renaming it to topics-v3.yaml freezes it.
Run: uv run python scripts/p2_forum/build_topics_v3.py
"""
from pathlib import Path

import yaml

from innovation.core.config import load_env
from innovation.core.llm import AnthropicLLM, CachedLLM, RoutedLLM
from innovation.p2_forum.topics import load_topics
from innovation.p2_forum.topics_v3 import (
    SYNONYM_BATCHES, VIEWS, apply_groups, build_topics, candidate_sets, draft_yaml,
    exact_duplicate_units, fallback_name, is_catch_all, load_areas, naming_prompt, parse_groups,
    parse_names, synonym_prompt, verified_groups, verify_prompt)

MODEL, ATTEMPTS, NAME_BATCH = "claude-sonnet-5", 3, 40
SYSTEM = "You curate research-area taxonomies precisely and conservatively."
OUT = Path("configs/p2_forum/topics-v3.draft.yaml")

load_env()
# Thinking off, as in build_topic_list.py: replies are short structured JSON.
llm = CachedLLM(RoutedLLM(anthropic_factory=lambda: AnthropicLLM(thinking={"type": "disabled"})),
                Path("data/online_cache/llm"))
fallbacks: list[str] = []


def ask(make_prompt, parse, label):
    """Up to ATTEMPTS replies, each prompt naming its attempt number; None if
    all fail (the caller applies its always-valid fallback)."""
    feedback = ""
    for attempt in range(1, ATTEMPTS + 1):
        reply = llm.complete(model=MODEL, system=SYSTEM,
                             user=make_prompt(attempt, feedback), max_tokens=8000)
        try:
            return parse(reply)
        except ValueError as e:
            feedback = str(e)[:500]
            print(f"  [{label}] attempt {attempt} rejected: {feedback}", flush=True)
    fallbacks.append(f"{label}: {feedback}")
    return None


def synonym_pass(units, areas, scope, cross=False):
    """Propose, then verify. Proposals: one vote per listing order (VIEWS);
    every set any vote merged is a candidate (high recall). Verification: a
    focused prompt per candidate keeps only the true synonym sets (precision).
    A reply that still fails validation after ATTEMPTS counts as 'no merges'
    (always valid) and is recorded in `fallbacks`."""
    votes = []
    for view in range(len(VIEWS)):
        groups = ask(lambda k, fb: synonym_prompt(units, areas, scope, k, fb, cross, view),
                     lambda r: parse_groups(r, len(units)), f"synonyms/{scope}/vote {view}")
        votes.append(groups if groups is not None else [[k] for k in range(len(units))])
    candidates = candidate_sets(votes, len(units))
    local = []
    for c in candidates:
        lg = ask(lambda k, fb: verify_prompt(c, units, areas, k, fb),
                 lambda r: parse_groups(r, len(c)),
                 f"verify/{scope}/" + " + ".join(areas[units[u][0]].keyword[:25] for u in c))
        local.append(lg if lg is not None else [[j] for j in range(len(c))])
    kept = sum(1 for lg in local for g in lg if len(g) > 1)
    print(f"  {scope}: {len(candidates)} candidate sets, {kept} verified", flush=True)
    return apply_groups(units, verified_groups(len(units), candidates, local))


venues = yaml.safe_load(open("configs/p2_forum/venue_areas.yaml"))["venues"]
areas = load_areas(venues)
dropped = [a for a in areas if is_catch_all(a)]
kept = [i for i, a in enumerate(areas) if not is_catch_all(a)]
units = exact_duplicate_units(areas, kept)
print(f"{len(areas)} areas, {len(dropped)} catch-alls dropped, "
      f"{len(units)} units after exact-duplicate merge", flush=True)

# 1) synonyms within each domain batch
merged = []
for scope, domains in SYNONYM_BATCHES:
    batch = [u for u in units if areas[u[0]].domain in domains]
    out = synonym_pass(batch, areas, scope)
    print(f"  batch {scope}: {len(batch)} -> {len(out)}", flush=True)
    merged += out
assert sorted(i for u in merged for i in u) == kept, "batches must cover every kept area"
merged.sort(key=min)

# 2) one cross-batch pass for synonyms that straddle batches
units = synonym_pass(merged, areas, "all domains, cross-domain pass", cross=True)
print(f"  cross-batch pass: {len(merged)} -> {len(units)}", flush=True)

# 3) names + definitions, batched; later batches see the names already used
names, taken = {}, []
for start in range(0, len(units), NAME_BATCH):
    batch = [(k, units[k]) for k in range(start, min(start + NAME_BATCH, len(units)))]
    ids = [k for k, _ in batch]
    got = ask(lambda a, fb: naming_prompt(batch, areas, taken, a, fb),
              lambda r: parse_names(r, ids, taken), f"names/{start}-{ids[-1]}")
    if got is None:
        got = {}
        for k, u in batch:
            got[k] = fallback_name(u, areas, taken + [n for n, _ in got.values()])
    names.update(got)
    taken += [names[k][0] for k in ids]

topics = build_topics(units, areas, names)
OUT.write_text(draft_yaml(topics, dropped))
load_topics(OUT, expected=None)
groups = [t for t in topics if len(t["sources"]) > 1]
print(f"wrote {OUT}: {len(topics)} topics, {len(groups)} merged groups, "
      f"{len(dropped)} drops, {len(fallbacks)} fallbacks")
for f in fallbacks:
    print(f"  FALLBACK {f}")
for a in dropped:
    print(f"  DROP {a.venue}: {a.area}")
for t in groups:
    print(f"  MERGE {t['name']} <- " + " | ".join(f"{s['venue']}: {s['area']}" for s in t["sources"]))
