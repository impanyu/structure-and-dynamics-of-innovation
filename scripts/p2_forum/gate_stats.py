"""Gate activity per online run (spec §10).
Run: uv run python scripts/p2_forum/gate_stats.py [run_dir ...]
Default run dirs: runs/p2_forum/forum-online-k*-s*."""
import collections
import json
import sys
from pathlib import Path

GATES = ("query", "result", "scope", "post", "unlabeled", "link")


def gate_counts(events) -> dict:
    c = collections.Counter()
    for e in events:
        r = e.get("result", {})
        if "gate" in r:
            c[r["gate"]] += 1
        for kind in ("scope", "topic"):
            c["filtered_" + kind] += (r.get("filtered") or {}).get(kind, 0)
        if e["action"] in ("search", "search_board"):
            c["searches"] += 1
        if r.get("source") == "openalex":
            c["openalex_fallbacks"] += 1
        c["openalex_unmapped"] += r.get("unmapped", 0)
        if e["action"] == "related":
            c["related"] += 1
        if e["action"] == "generate" and "node_id" in r:
            c["posts"] += 1
            c["dropped_cites"] += len(r.get("dropped_cites", []))
    out = {k: c.get(k, 0) for k in (*GATES, "dropped_cites", "posts", "searches", "related",
                                         "filtered_scope", "filtered_topic",
                                         "openalex_fallbacks", "openalex_unmapped")}
    tried = out["post"] + out["posts"]
    out["rejection_rate"] = out["post"] / tried if tried else 0.0
    return out


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else argv
    explicit = bool(args)
    dirs = [Path(a) for a in args] or sorted(Path("runs/p2_forum").glob("forum-online-k*-s*"))
    rows = []
    for d in dirs:
        if not (d / "events.jsonl").exists():
            continue
        ev = [json.loads(l) for l in open(d / "events.jsonl") if l.strip()]
        rows.append({"run": d.name, **gate_counts(ev)})
    if not explicit:
        Path("runs/p2_forum").mkdir(parents=True, exist_ok=True)
        json.dump(rows, open("runs/p2_forum/gate_stats.json", "w"), indent=1)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
