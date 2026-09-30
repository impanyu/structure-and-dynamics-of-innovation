"""Quality of the nested k sweep: acc@>=2 per k (3 seeds), and whether ideas
that build on a teammate's post, or that a teammate later cites, score better.

Reads runs/p2_forum/forum-nested-k{k}-s{s}/{verdicts.json,events.jsonl} for
every run that has been evaluated; writes runs/p2_forum/nested_quality.json.

Run: uv run python scripts/p2_forum/eval_nested.py
"""
import collections
import json
import os
from types import SimpleNamespace

from innovation.core.eval.metrics import idea_levels

KS = [1, 16, 32, 48, 64, 80, 96, 112, 128]
WINDOW = "2025-06-01"
TIERS = ("tier1", "tier2", "tier3")


def levels(rid):
    vs = json.load(open(f"runs/p2_forum/{rid}/verdicts.json"))
    objs = [SimpleNamespace(idea_id=v["idea_id"], best=v["best"], candidates=v["candidates"]) for v in vs]
    dup = {v["idea_id"]: v["dup_flag"] for v in vs}
    return {t: dict(zip([o.idea_id for o in objs], idea_levels(objs, dup, t, WINDOW))) for t in TIERS}


def roles(rid):
    """Per post: does it cite a teammate's post (builds_on), is it cited by a
    teammate (cited_by_mate), does it cite any board post at all."""
    ev = [json.loads(line) for line in open(f"runs/p2_forum/{rid}/events.jsonl")]
    author = {e["result"]["node_id"]: e["agent_id"] for e in ev
              if e["action"] == "generate" and "node_id" in e["result"]}
    out = {nid: {"builds_on": False, "cited_by_mate": False, "cites_board": False} for nid in author}
    for e in ev:
        if e["action"] != "generate" or "node_id" not in e["result"]:
            continue
        me = e["result"]["node_id"]
        for c in e["args"].get("cited_ids", []):
            if c in author:
                out[me]["cites_board"] = True
                if author[c] != e["agent_id"]:
                    out[me]["builds_on"] = True
                    out[c]["cited_by_mate"] = True
    return out


rows, pooled = [], collections.defaultdict(lambda: collections.defaultdict(list))
for s in (0, 1, 2):
    for k in KS:
        rid = f"forum-nested-k{k}-s{s}"
        if not os.path.exists(f"runs/p2_forum/{rid}/verdicts.json"):
            continue
        lv, rl = levels(rid), roles(rid)
        n = len(lv["tier1"])
        row = {"k": k, "seed": s, "n": n}
        for t in TIERS:
            row[f"{t}_hits"] = sum(1 for x in lv[t].values() if x >= 2)
            row[f"{t}_acc"] = row[f"{t}_hits"] / n if n else 0.0
        rows.append(row)
        for nid, r in rl.items():
            x = lv["tier1"].get(nid)
            if x is None:
                continue
            for key in ("builds_on", "cited_by_mate", "cites_board"):
                pooled[key][r[key]].append(x >= 2)

json.dump({"runs": rows, "by_role": {k: {str(b): [sum(v), len(v)] for b, v in d.items()}
                                     for k, d in pooled.items()}},
          open("runs/p2_forum/nested_quality.json", "w"), indent=1)

print(f"{'k':>4} {'seeds':>5} {'ideas':>5}  tier1 acc@>=2 (per seed)        tier2   tier3")
for k in KS:
    rs = [r for r in rows if r["k"] == k]
    if not rs:
        continue
    N = sum(r["n"] for r in rs)
    t1 = sum(r["tier1_hits"] for r in rs) / N
    per = " ".join(f"{r['tier1_acc']:.2f}" for r in rs)
    print(f"{k:>4} {len(rs):>5} {N:>5}  {t1:.3f}  ({per:<16})   "
          f"{sum(r['tier2_hits'] for r in rs) / N:.3f}   {sum(r['tier3_hits'] for r in rs) / N:.3f}")
print("\ntier-1 acc@>=2 by role (pooled over all evaluated runs):")
for key, d in pooled.items():
    for b in (True, False):
        v = d.get(b, [])
        if v:
            print(f"  {key:>14}={str(b):<5}  {sum(v):>3}/{len(v):<4} = {sum(v) / len(v):.3f}")
