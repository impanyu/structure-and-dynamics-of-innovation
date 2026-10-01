"""Tagging quality (spec §10): labels-per-item histogram, share labeled 5, and
re-label agreement on a random sample, using fresh calls through an uncached
RoutedLLM. Paper text is rebuilt from the (disk-cached) Semantic Scholar record exactly as
the original labeling saw it (Paper.text()).
Run: uv run python scripts/p2_forum/tag_quality.py"""
import collections
import json
import random
from pathlib import Path

from innovation.core.llm import RoutedLLM
from innovation.p2_forum.literature import Paper
from innovation.p2_forum.s2_online import S2Online
from innovation.p2_forum.tagger import TopicTagger
from innovation.p2_forum.topics import load_topics

CACHE = "data/online_cache"
LABELS = Path(CACHE) / "labels.jsonl"
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
            if r.get("labels") and r["paper_id"] not in out:   # first write wins
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


def paper_text(raw: dict) -> str:
    """The labeler's original input, via the same Paper.text() the store used."""
    return Paper(paper_id=raw["paperId"], title=raw.get("title") or "",
                 abstract=raw.get("abstract") or "", year=None, venue="",
                 pub_date="", citations=0, branch="").text()


def sample_texts(client, ids, n: int, seed: int = 0) -> dict[str, str]:
    """Draw n ids from the label store; keep those the client can still fetch."""
    ids = sorted(ids)
    picked = random.Random(seed).sample(ids, min(n, len(ids)))
    out = {}
    for pid in picked:
        raw = client.paper(pid)
        if raw:
            out[pid] = paper_text(raw)
    return out


def main() -> None:
    labels = read_labels()
    s = histogram(labels)
    print(f"items: {s['n']}  labels-per-item histogram: {s['hist']}  share labeled 5: {s['share_5']:.3f}")
    texts = sample_texts(S2Online(CACHE), labels, SAMPLE)
    if not texts:
        print("no sampled paper could be fetched; skipping re-label agreement")
        return
    ids = list(texts)
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
