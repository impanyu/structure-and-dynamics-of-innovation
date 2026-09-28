import json, collections, re, sys
rid = sys.argv[1]
ev = [json.loads(l) for l in open(f"runs/p2_forum/{rid}/events.jsonl")]
author, born = {}, {}
for e in ev:
    if e["action"] == "generate" and "node_id" in e["result"]:
        author[e["result"]["node_id"]] = e["agent_id"]; born[e["result"]["node_id"]] = e["step"]
by_agent = collections.defaultdict(list)
for e in ev: by_agent[e["agent_id"]].append(e)
ID = re.compile(r"gen:[^\"'\s,\]]+")

# ---- H1: what the agent could see when it wrote (its 20-entry memory, results truncated to 1500 chars, as the policy does)
own_vis = mate_vis = own_cite = mate_cite = own_cite_vis = mate_cite_vis = 0
own_cite_notvis = mate_cite_notvis = 0
n_gen = 0
for a, evs in by_agent.items():
    for i, e in enumerate(evs):
        if e["action"] != "generate" or "node_id" not in e["result"]: continue
        n_gen += 1
        window = evs[max(0, i-20):i]
        vis = set()
        for w in window:
            vis |= {x for x in ID.findall(json.dumps(w["result"])[:1500]) if x in author and born[x] < e["step"]}
        vo = {x for x in vis if author[x] == a}; vm = vis - vo
        own_vis += len(vo); mate_vis += len(vm)
        for c in e["args"].get("cited_ids", []):
            if c not in author: continue
            mine = author[c] == a
            if mine: own_cite += 1; own_cite_vis += c in vo; own_cite_notvis += c not in vo
            else:   mate_cite += 1; mate_cite_vis += c in vm; mate_cite_notvis += c not in vm
print(f"== {rid}: {n_gen} posts ==")
print(f"H1 visibility  — per post, avg posts in memory window: own {own_vis/n_gen:.2f}, teammates' {mate_vis/n_gen:.2f}")
print(f"   citations    — own {own_cite} (in window {own_cite_vis}, NOT in window {own_cite_notvis}); teammates' {mate_cite} (in window {mate_cite_vis})")
if own_vis: print(f"   P(cite | visible): own {own_cite_vis/own_vis:.2f}   teammate {mate_cite_vis/max(mate_vis,1):.2f}")

# ---- where do own ids in the window come from?
src = collections.Counter()
for a, evs in by_agent.items():
    for w in evs:
        for x in ID.findall(json.dumps(w["result"])[:1500]):
            if x in author and author[x] == a: src[w["action"]] += 1
print(f"   own post ids appear in memory via: {dict(src)}")

# ---- H2: search_board ranking — whose posts come back?
hit_own = hit_mate = 0; top1 = collections.Counter()
for e in ev:
    if e["action"] != "search_board": continue
    hits = [h["node_id"] for h in e["result"].get("hits", []) if h["node_id"] in author]
    avail_own = sum(1 for p in author if author[p]==e["agent_id"] and born[p] < e["step"])
    avail_mate = sum(1 for p in author if author[p]!=e["agent_id"] and born[p] < e["step"])
    for h in hits: hit_own += author[h]==e["agent_id"]; hit_mate += author[h]!=e["agent_id"]
    if hits: top1["own" if author[hits[0]]==e["agent_id"] else "mate"] += 1
n_sb = sum(1 for e in ev if e["action"]=="search_board")
print(f"H2 search_board ({n_sb} calls) — hits own {hit_own} vs teammates' {hit_mate}; top-1 hit {dict(top1)}")

# ---- H3: serial refinement — does a self-citation point at the agent's most recent post?
last_own = collections.defaultdict(list); latest = back2 = older = 0
for e in ev:
    if e["action"] == "generate" and "node_id" in e["result"]:
        mine = last_own[e["agent_id"]]
        for c in e["args"].get("cited_ids", []):
            if c in author and author[c] == e["agent_id"] and mine:
                if c == mine[-1]: latest += 1
                elif len(mine) > 1 and c == mine[-2]: back2 += 1
                else: older += 1
        mine.append(e["result"]["node_id"])
print(f"H3 self-citation target — most recent own post {latest}, 2nd most recent {back2}, older {older}")
