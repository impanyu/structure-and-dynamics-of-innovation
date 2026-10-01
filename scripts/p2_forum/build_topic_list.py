"""One-shot: venue_areas.yaml -> configs/p2_forum/topics-v2.draft.yaml.

The user reviews the draft; renaming it to topics-v2.yaml freezes it (spec §3).
Run: uv run python scripts/p2_forum/build_topic_list.py
"""
from pathlib import Path

import yaml

from innovation.core.config import load_env
from innovation.core.llm import AnthropicLLM, CachedLLM, RoutedLLM
from innovation.p2_forum.topics import consolidation_prompt, parse_topic_list

N, MODEL = 128, "claude-sonnet-5"
load_env()
venues = yaml.safe_load(open("configs/p2_forum/venue_areas.yaml"))["venues"]
names = {v["venue"] for v in venues}
# Thinking is off here: on this long synthesis it exhausted 44k tokens
# without emitting any text.
llm = CachedLLM(RoutedLLM(anthropic_factory=lambda: AnthropicLLM(thinking={"type": "disabled"})),
                Path("data/online_cache/llm"))
prompt, feedback = consolidation_prompt(venues, N), ""
for attempt in range(3):
    reply = llm.complete(model=MODEL, system="You design research taxonomies.",
                         user=prompt + feedback, max_tokens=16000)
    try:
        topics = parse_topic_list(reply, N, names)
        break
    except ValueError as e:
        feedback = f"\n\nYour previous answer was rejected: {e}. Fix it."
else:
    raise SystemExit("could not obtain a valid topic list in 3 attempts")
out = Path("configs/p2_forum/topics-v2.draft.yaml")
out.write_text("# DRAFT topic list built from configs/p2_forum/venue_areas.yaml by\n"
               "# scripts/p2_forum/build_topic_list.py. Review, edit, then rename to\n"
               "# topics-v2.yaml to freeze it (ids = list positions).\n"
               + yaml.safe_dump({"topics": topics}, allow_unicode=True, sort_keys=False, width=100))
print(f"wrote {out} ({len(topics)} topics)")
