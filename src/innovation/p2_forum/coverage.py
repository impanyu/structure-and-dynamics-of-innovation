"""Topic coverage on a reference paper sample (spec §3.2, §4): a stratified
sample of the seven venues' 2023/2024 papers, labeled by the experiment's
tagger. A topic set's coverage is the share of labeled sample papers that
carry any topic in the set. Pure helpers; scripts/p2_forum/build_coverage_sample.py
does the fetching and labeling."""
import json
import random
from collections import Counter
from pathlib import Path

from innovation.p2_forum.topics import Topic


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split n across strata proportionally to their sizes, largest-remainder
    rounding (exact integer arithmetic; ties go to the earlier key). The result
    sums to n; an empty stratum gets 0."""
    total = sum(sizes.values())
    if n > total:
        raise ValueError(f"cannot sample {n} from a population of {total}")
    if n == 0:
        return {k: 0 for k in sizes}
    out = {k: s * n // total for k, s in sizes.items()}
    order = sorted(sizes, key=lambda k: -(sizes[k] * n % total))   # stable: ties keep key order
    for k in order[:n - sum(out.values())]:
        out[k] += 1
    return out


def sample_stratum(records: list[dict], k: int, *, seed: int, stratum: str) -> list[dict]:
    """k records drawn uniformly without replacement. Deterministic in (seed,
    stratum) and independent of the input order (records are sorted by paperId first)."""
    pool = sorted(records, key=lambda r: r["paperId"])
    return random.Random(f"{seed}|{stratum}").sample(pool, k)


def coverage(label_sets: list[set[int]], topic_ids: set[int]) -> float:
    """Share of the papers whose labels meet topic_ids (0.0 for no papers)."""
    if not label_sets:
        return 0.0
    return sum(1 for s in label_sets if s & topic_ids) / len(label_sets)


def labels_histogram(labels: list[list[int] | None]) -> dict[str, int]:
    """{number of labels: papers}, over labeled papers only; keys are strings (JSON)."""
    c = Counter(len(x) for x in labels if x is not None)
    return {str(k): c[k] for k in sorted(c)}


def coverage_table(paper_ids: list[str], labels: list[list[int] | None],
                   topics: list[Topic]) -> dict:
    """The committed per-topic coverage table. Denominator = labeled papers;
    unlabeled ones (refusals, parse failures) are only counted."""
    labeled = [set(x) for x in labels if x is not None]
    counts = Counter(t for s in labeled for t in s)
    n = len(labeled)
    return {
        "sample_size": len(paper_ids),
        "labeled": n,
        "unlabeled": len(paper_ids) - n,
        "labels_per_paper": labels_histogram(labels),
        "topics": [{"id": t.id, "name": t.name,
                    "coverage": counts[t.id] / n if n else 0.0,
                    "count": counts[t.id]} for t in topics],
    }


def load_sample_labels(sample_dir) -> tuple[list[str], list[set[int]]]:
    """The labeled sample papers (ids, label sets) from <dir>/labels.jsonl;
    unlabeled papers are left out, as they are of the coverage denominator."""
    ids, sets = [], []
    for line in (Path(sample_dir) / "labels.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("labels") is not None:
                ids.append(r["paperId"])
                sets.append(set(r["labels"]))
    return ids, sets
