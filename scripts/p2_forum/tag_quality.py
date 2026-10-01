"""Tagging quality (spec §10): labels-per-item histogram, share labeled 5, and
re-label agreement on a random sample, using fresh calls through an uncached
RoutedLLM. Paper text comes from titles/abstracts logged in run event logs.
Run: uv run python scripts/p2_forum/tag_quality.py [run_dir ...]"""
import collections
import json
import random
import sys
from pathlib import Path

from innovation.core.llm import RoutedLLM
from innovation.p2_forum.tagger import TopicTagger
from innovation.p2_forum.topics import load_topics

LABELS = Path("data/online_cache/labels.jsonl")
TOPICS = Path("configs/p2_forum/topics-v2.yaml")
DRAFT = Path("configs/p2_forum/topics-v2.draft.yaml")
MODEL = "claude-sonnet-5"
SAMPLE = 200


class _StabilityLLM:
    """Appends the re-label suffix so no prompt matches an earlier call."""
    def __init__(self, llm):
        self.llm = llm

    def complete(self, *, model, system, user, max_tokens=1024):
        return self.llm.complete(model=model, system=system,
                                 user=user + "\n\n(stability)", max_tokens=max_tokens)


def read_labels(path=LABELS) -> dict[str, list[int]]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("labels"):
                out[r["paper_id"]] = r["labels"]
    return out


def histogram(labels: dict[str, list[int]]) -> dict:
    h = collections.Counter(len(v) for v in labels.values())
    n = len(labels)
    return {"n": n, "hist": {k: h[k] for k in sorted(h)},
            "share_5": h[5] / n if n else 0.0}


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def logged_texts(run_dirs) -> dict[str, str]:
    """paper_id -> 'title. text' from search hits and browse results in event logs."""
    texts = {}
    for d in run_dirs:
        p = Path(d) / "events.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            e = json.loads(line)
            r = e.get("result", {})
            items = list(r.get("hits", []))
            if e.get("action") == "browse" and "text" in r:
                items.append({**r, "node_id": e.get("args", {}).get("node_id")})
            for h in items:
                if h.get("node_id") and h.get("text"):
                    texts.setdefault(h["node_id"], f"{h.get('title', '')}\n{h['text']}".strip())
    return texts


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    labels = read_labels()
    s = histogram(labels)
    print(f"items: {s['n']}  labels-per-item histogram: {s['hist']}  share labeled 5: {s['share_5']:.3f}")
    dirs = args or sorted(str(d) for d in Path("runs/p2_forum").glob("forum-online-k*-s*"))
    texts = logged_texts(dirs)
    ids = sorted(i for i in labels if i in texts)
    if not ids:
        print("no sampled paper has logged text; skipping re-label agreement")
        return
    ids = random.Random(0).sample(ids, min(SAMPLE, len(ids)))
    path = TOPICS
    if not path.exists():
        print(f"WARNING: {TOPICS} absent; using draft {DRAFT}")
        path = DRAFT
    tagger = TopicTagger(llm=_StabilityLLM(RoutedLLM()), model=MODEL, topics=load_topics(path))
    new = tagger.label_many([texts[i] for i in ids])
    pairs = [(labels[i], n) for i, n in zip(ids, new) if n]
    if not pairs:
        print("no re-labels succeeded")
        return
    print(f"re-labeled {len(pairs)}/{len(ids)}: mean Jaccard "
          f"{sum(jaccard(a, b) for a, b in pairs) / len(pairs):.3f}, primary-label agreement "
          f"{sum(a[0] == b[0] for a, b in pairs) / len(pairs):.3f}")


if __name__ == "__main__":
    main()
