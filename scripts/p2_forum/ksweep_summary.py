import json, collections, re
ID = re.compile(r"gen:[^\"'\s,\]]+")
rows = []
for k in [1, 2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 128]:
    rid = f"forum-k{k}"
    ev = [json.loads(l) for l in open(f"runs/p2_forum/{rid}/events.jsonl")]
    f = json.load(open(f"runs/p2_forum/{rid}/board_metrics.json"))["final"]
    meta = json.load(open(f"runs/p2_forum/{rid}/run_meta.json"))
    ta = meta["topic_assignments"]
    # mean pairwise topic overlap (Jaccard) between agents
    ags = list(ta); jac = []
    for i in range(len(ags)):
        for j in range(i+1, len(ags)):
            a, b = set(ta[ags[i]]), set(ta[ags[j]]); jac.append(len(a & b) / len(a | b))
    author, born = {}, {}
    for e in ev:
        if e["action"] == "generate" and "node_id" in e["result"]:
            author[e["result"]["node_id"]] = e["agent_id"]; born[e["result"]["node_id"]] = e["step"]
    by = collections.defaultdict(list)
    for e in ev: by[e["agent_id"]].append(e)
    ov = mv = oc = mc = 0
    for a, es in by.items():
        for i, e in enumerate(es):
            if e["action"] != "generate" or "node_id" not in e["result"]: continue
            vis = set()
            for w in es[max(0, i-20):i]:
                vis |= {x for x in ID.findall(json.dumps(w["result"])[:1500]) if x in author and born[x] < e["step"]}
            vo = {x for x in vis if author[x] == a}; vm = vis - vo
            ov += len(vo); mv += len(vm)
            for c in e["args"].get("cited_ids", []):
                if c in vo: oc += 1
                elif c in vm: mc += 1
    acts = collections.Counter(e["action"] for e in ev)
    rows.append(dict(k=k, steps=len(ev), posts=f["n_posts"], overlap=sum(jac)/len(jac),
        reads=sum(acts[x] for x in ("search_board", "browse_board", "sample_board")),
        pp=f["n_post_post_edges"], cross=f["cross_agent_edges"], comps=f["n_components"],
        first=f["first_cross_agent_citation_step"], mate_vis=mv, mate_cit=mc,
        p_mate=mc / mv if mv else 0.0, p_own=oc / ov if ov else 0.0,
        errors=sum(1 for e in ev if "error" in e["result"])))
json.dump(rows, open("runs/p2_forum/ksweep_summary.json", "w"), indent=1)
print(f"{'k':>4} {'steps':>5} {'posts':>5} {'overlap':>7} {'reads':>5} {'pp':>4} {'cross':>5} {'comps':>5} {'first':>5} {'mateVis':>7} {'P(mate)':>7} {'P(own)':>6} {'err':>3}")
for r in rows:
    print(f"{r['k']:>4} {r['steps']:>5} {r['posts']:>5} {r['overlap']:>7.3f} {r['reads']:>5} {r['pp']:>4} {r['cross']:>5} {r['comps']:>5} {str(r['first']):>5} {r['mate_vis']:>7} {r['p_mate']:>7.3f} {r['p_own']:>6.3f} {r['errors']:>3}")
