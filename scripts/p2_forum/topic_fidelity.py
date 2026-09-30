"""Do agents stay inside their k topics? (nested sweep)

The environment filters nothing; topics live only in the system prompt. This
measures, per k, the share of what agents READ and CITE that falls inside the
reading agent's own topics, against the share expected if they ignored their
topics entirely (chance = corpus share of those topics).

Topic of a corpus node: its k-means cluster (the same deterministic clustering
that named the 128-topic pool; labels cached by this script's first run in
data/stage1/topic_labels_k128.npy). Topic of a board post: nearest cluster
centroid of the post's embedding.

Reads:  search hits shown, browse targets (+ neighbors are not counted), sample
        jumps; the same for the board. Cites: generate cited_ids.

Run: uv run python scripts/p2_forum/topic_fidelity.py
"""
import collections
import json
import os

import numpy as np
import yaml

from innovation.core.ideas.embed import Embedder, load_embeddings

KS = [1, 16, 32, 48, 64, 80, 96, 112, 128]
ids, _ = load_embeddings("data/stage1")
lab = np.load("data/stage1/topic_labels_k128.npy")
cent = np.load("data/stage1/topic_centroids_k128.npy")
topic_of = dict(zip(ids, lab.tolist()))
pool = [t["topic"] for t in yaml.safe_load(open("configs/p2_forum/topics-k128.yaml"))["topics"]]
tidx = {t: i for i, t in enumerate(pool)}
share = np.bincount(lab, minlength=128) / len(lab)
emb = Embedder("BAAI/bge-small-en-v1.5")


def run_stats(rid):
    meta = json.load(open(f"runs/p2_forum/{rid}/run_meta.json"))
    mine = {a: {tidx[t] for t in ts} for a, ts in meta["topic_assignments"].items()}
    ev = [json.loads(line) for line in open(f"runs/p2_forum/{rid}/events.jsonl")]
    posts = {e["result"]["node_id"]: e["args"]["text"] for e in ev
             if e["action"] == "generate" and "node_id" in e["result"]}
    pids = list(posts)
    if pids:
        pv = emb.encode([posts[p] for p in pids])
        for p, t in zip(pids, np.argmax(pv @ cent.T, axis=1).tolist()):
            topic_of[p] = t
    c = collections.Counter()

    def tally(kind, a, nid):
        t = topic_of.get(nid)
        if t is None:
            return
        c[kind + "_n"] += 1
        c[kind + "_in"] += t in mine[a]

    for e in ev:
        a, act, r = e["agent_id"], e["action"], e["result"]
        if act in ("search", "search_board"):
            for h in r.get("hits", []):
                tally("read_corpus" if h.get("store") == "corpus" else "read_board", a, h["node_id"])
        elif act in ("browse", "sample_frontier", "browse_board", "sample_board") and "node_id" in r:
            tally("read_corpus" if r.get("store") == "corpus" else "read_board", a, r["node_id"])
        elif act == "generate" and "node_id" in r:
            tally("post", a, r["node_id"])
            for x in e["args"].get("cited_ids", []):
                tally("cite_board" if x in posts else "cite_corpus", a, x)
    chance = float(np.mean([share[list(m)].sum() for m in mine.values()]))
    return c, chance


rows = []
for k in KS:
    agg, ch = collections.Counter(), []
    for s in (0, 1, 2):
        rid = f"forum-nested-k{k}-s{s}"
        if os.path.exists(f"runs/p2_forum/{rid}/events.jsonl"):
            c, chance = run_stats(rid)
            agg += c
            ch.append(chance)
    r = {"k": k, "chance": float(np.mean(ch))}
    for kind in ("read_corpus", "cite_corpus", "post", "read_board", "cite_board"):
        n = agg[kind + "_n"]
        r[kind] = agg[kind + "_in"] / n if n else None
        r[kind + "_n"] = n
    rows.append(r)
json.dump(rows, open("runs/p2_forum/topic_fidelity.json", "w"), indent=1)


def f(x):
    return "   -  " if x is None else f"{x:6.2f}"


print(f"{'k':>4} {'chance':>6} | {'readC':>6} {'citeC':>6} {'post':>6} | {'readB':>6} {'citeB':>6}   (share inside own topics)")
for r in rows:
    print(f"{r['k']:>4} {r['chance']:6.2f} | {f(r['read_corpus'])} {f(r['cite_corpus'])} {f(r['post'])} | "
          f"{f(r['read_board'])} {f(r['cite_board'])}   n={r['read_corpus_n']}/{r['cite_corpus_n']}/{r['post_n']}/{r['read_board_n']}/{r['cite_board_n']}")
