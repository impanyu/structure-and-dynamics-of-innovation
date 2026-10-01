"""The frozen paper-2 topic list (spec §3): built once from the seven
venues' official submission areas, reviewed by the user, then never
regenerated. Topic ids are list positions."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Topic:
    id: int
    name: str
    definition: str
    sources: list[str] = field(default_factory=list)


def load_topics(path, expected: int = 128) -> list[Topic]:
    raw = yaml.safe_load(Path(path).read_text())["topics"]
    if len(raw) != expected:
        raise ValueError(f"{path}: expected {expected} topics, found {len(raw)}")
    names = [t["name"] for t in raw]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate topic names")
    return [Topic(id=i, name=t["name"], definition=t["definition"],
                  sources=list(t.get("sources", []))) for i, t in enumerate(raw)]


def consolidation_prompt(venues: list[dict], n: int) -> str:
    lines = [f"{v['venue']}: {a}" for v in venues for a in v["areas"]]
    return (
        "Below are the official submission areas of top AI venues, one per line "
        "as '<venue>: <area>'.\n\n" + "\n".join(lines) + "\n\n"
        f"Merge and deduplicate them into exactly {n} research topics that together "
        "cover all of them. Topics must not overlap in meaning, should be of similar "
        "granularity (split broad areas along the finer keyword lists, merge "
        "near-synonyms), and must avoid generic modifiers such as 'robust', "
        "'efficient', 'scalable' or 'advances in' unless they name the topic itself.\n"
        "Reply with ONLY a JSON list of objects: "
        '{"name": "<3-8 word topic name>", "definition": "<one sentence: what is in, '
        'what is out>", "sources": ["<venue>", ...]} where sources lists every venue '
        "whose areas this topic absorbs.")


def parse_topic_list(reply: str, n: int, venue_names: set[str]) -> list[dict]:
    start, end = reply.find("["), reply.rfind("]")
    if start == -1 or end <= start:
        raise ValueError("no JSON list in reply")
    items = json.loads(reply[start:end + 1])
    if len(items) != n:
        raise ValueError(f"expected {n} topics, got {len(items)}")
    seen = set()
    for t in items:
        name = str(t.get("name", "")).strip()
        if not name or name.lower() in seen:
            raise ValueError(f"duplicate or empty topic name: {name!r}")
        seen.add(name.lower())
        if not str(t.get("definition", "")).strip():
            raise ValueError(f"topic {name!r} has no definition")
        unknown = set(t.get("sources", [])) - venue_names
        if unknown or not t.get("sources"):
            raise ValueError(f"topic {name!r} cites unknown venue(s) {sorted(unknown)}")
    return [{"name": t["name"].strip(), "definition": t["definition"].strip(),
             "sources": list(t["sources"])} for t in items]
