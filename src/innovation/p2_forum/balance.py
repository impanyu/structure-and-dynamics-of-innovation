"""Capacity balancing of the topic list (Task 2c, step C).

Each round: tau = total capacity / n, band [tau/sqrt(3), tau*sqrt(3)] (so the
largest in-band topic is at most 3x the smallest). Topics above the band are
split into max(2, round(c/tau)) sub-areas, topics below it are merged,
smallest first, into their nearest topic within the band cap (nearest = the
highest co-labeling rate on the precision samples, ties broken by embedding
cosine), the list is restored to exactly n, and every capacity is
re-estimated (labels compete across topics). Stops when all topics are in
band or after max_rounds; a topic still out of band is an exception, never
force-merged (the merge LLM may answer that no sensible merge exists)."""
import json
import math
from pathlib import Path

import numpy as np
import yaml

SQRT3 = math.sqrt(3)


def band(caps: list[float], n: int) -> tuple[float, float, float]:
    tau = sum(caps) / n
    return tau, tau / SQRT3, tau * SQRT3


def split_count(c: float, tau: float) -> int:
    return max(2, round(c / tau))


def max_min_ratio(caps: list[float]) -> float | None:
    lo = min(caps)
    return max(caps) / lo if lo > 0 else None


def out_of_band(caps: list[float], n: int) -> list[int]:
    _, lo, hi = band(caps, n)
    return [i for i, c in enumerate(caps) if not lo <= c <= hi]


def colabel_rate(labels: dict, a: frozenset, b: frozenset) -> float:
    """Share of sample papers labeled with a topic of `a` that also carry a
    topic of `b` (members are ids in the estimated list)."""
    with_a = [lab for lab in labels.values() if lab and a & set(lab)]
    if not with_a:
        return 0.0
    return sum(bool(b & set(lab)) for lab in with_a) / len(with_a)


class Similarity:
    """Cosine of name+definition embeddings, cached by text."""

    def __init__(self, embed):
        self.embed, self._vecs = embed, {}

    def _vec(self, t: dict):
        key = f"{t['name']}: {t['definition']}"
        if key not in self._vecs:
            v = np.asarray(self.embed([key])[0], dtype=np.float64)
            self._vecs[key] = v / (np.linalg.norm(v) or 1.0)
        return self._vecs[key]

    def __call__(self, a: dict, b: dict) -> float:
        return float(self._vec(a) @ self._vec(b))


def choose_merge_target(s: dict, work: list[dict], labels: dict, sim,
                        cap: float | None) -> dict | None:
    """The nearest other topic whose merged capacity stays <= cap (None = no
    cap): highest co-labeling rate, ties broken by cosine similarity."""
    best, best_key = None, None
    for t in work:
        if t is s or (cap is not None and s["capacity"] + t["capacity"] > cap):
            continue
        key = (colabel_rate(labels, s["members"], t["members"]), sim(s, t))
        if best_key is None or key > best_key:
            best, best_key = t, key
    return best


def _merged(s: dict, t: dict, named: dict) -> dict:
    return {"name": named["name"], "definition": named["definition"],
            "sources": list(dict.fromkeys(t["sources"] + s["sources"])),
            "capacity": s["capacity"] + t["capacity"],
            "members": s["members"] | t["members"]}


def _at(work: list[dict], x: dict) -> int:
    """Index by identity: two working topics may compare equal as dicts."""
    return next(i for i, w in enumerate(work) if w is x)


def _replace_pair(work: list[dict], s: dict, t: dict, merged: dict) -> None:
    work[_at(work, t)] = merged
    del work[_at(work, s)]


def _names_except(work, *skip) -> set[str]:
    return {w["name"].lower() for w in work if not any(w is s for s in skip)}


def _split(w: dict, p: int, titles: list[str], work: list[dict], rewriter) -> list[dict]:
    kids = rewriter.split(w, p, titles, _names_except(work, w))
    return [{"name": k["name"], "definition": k["definition"],
             "sources": list(w["sources"]), "capacity": w["capacity"] / p,
             "members": w["members"]} for k in kids]


def restore_count(work: list[dict], n: int, labels: dict, sim, rewriter, *,
                  cap: float, titles_of, skip: set[str] = frozenset()) -> list[dict]:
    """Back to exactly n topics: while too many, merge the smallest topic into
    its nearest (within cap when possible) — a refused pair moves on to the
    next smallest; while too few, split the largest in two."""
    work, refused = list(work), set(skip)
    while len(work) > n:
        for s in sorted(work, key=lambda w: w["capacity"]):
            if s["name"] in refused:
                continue
            t = (choose_merge_target(s, work, labels, sim, cap)
                 or choose_merge_target(s, work, labels, sim, None))
            named = rewriter.merge(s, t, _names_except(work, s, t)) if t else None
            if named is None:
                refused.add(s["name"])
                continue
            _replace_pair(work, s, t, _merged(s, t, named))
            break
        else:
            raise RuntimeError(f"cannot restore {n} topics: every merge was refused")
    while len(work) < n:
        w = max(work, key=lambda x: x["capacity"])
        i = _at(work, w)
        work[i:i + 1] = _split(w, 2, titles_of(w), work, rewriter)
    return work


def rebalance_round(topics: list[dict], est, rewriter, sim, n: int):
    """One round on an estimated list -> (new topics, flags, info). flags maps
    a topic name to why it stayed out of band."""
    caps = est.capacities
    tau, lo, hi = band(caps, n)
    titles = {i: est.titles[i] for i in range(len(topics))}

    def titles_of(w):
        return [x for m in sorted(w["members"]) for x in titles.get(m, [])]

    work = [{"name": t["name"], "definition": t["definition"],
             "sources": list(t.get("sources", [])), "capacity": c,
             "members": frozenset({i})} for i, (t, c) in enumerate(zip(topics, caps))]
    flags, info = {}, {"splits": [], "merges": [], "tau": tau, "band": [lo, hi]}
    for w in [w for w in work if w["capacity"] > hi]:
        p = split_count(w["capacity"], tau)
        i = _at(work, w)
        kids = _split(w, p, titles_of(w), work, rewriter)
        work[i:i + 1] = kids
        info["splits"].append({"topic": w["name"], "into": [k["name"] for k in kids]})
    for s in sorted([w for w in work if w["capacity"] < lo], key=lambda w: w["capacity"]):
        if not any(s is w for w in work):
            continue                      # already absorbed as a merge target
        t = choose_merge_target(s, work, est.labels, sim, hi)
        if t is None:
            flags[s["name"]] = "no merge target within the band cap"
            continue
        named = rewriter.merge(s, t, _names_except(work, s, t))
        if named is None:
            flags[s["name"]] = f"no sensible merge (with {t['name']!r})"
            continue
        m = _merged(s, t, named)
        _replace_pair(work, s, t, m)
        info["merges"].append({"topics": [s["name"], t["name"]], "into": m["name"]})
    before = len(work)
    work = restore_count(work, n, est.labels, sim, rewriter, cap=hi,
                         titles_of=titles_of, skip=set(flags))
    info["restored_from"] = before
    return ([{"name": w["name"], "definition": w["definition"], "sources": w["sources"]}
             for w in work], flags, info)


def run_balance(topics: list[dict], estimate, rewriter, sim, *, n: int = 128,
                max_rounds: int = 3, log=print) -> dict:
    est = estimate(topics)
    history = [_round_summary(0, est, n, None)]
    log(f"round 0: {history[-1]}")
    flags, rounds = {}, 0
    while rounds < max_rounds and out_of_band(est.capacities, n):
        topics, flags, info = rebalance_round(topics, est, rewriter, sim, n)
        est = estimate(topics)
        rounds += 1
        history.append(_round_summary(rounds, est, n, info))
        log(f"round {rounds}: { {k: v for k, v in history[-1].items() if k != 'changes'} }")
    tau, lo, hi = band(est.capacities, n)
    exceptions = [{"id": i, "name": topics[i]["name"], "capacity": est.capacities[i],
                   "side": "below" if est.capacities[i] < lo else "above",
                   "reason": flags.get(topics[i]["name"],
                                       f"out of band after {rounds} round(s)")}
                  for i in out_of_band(est.capacities, n)]
    return {"topics": topics, "estimate": est, "tau": tau, "band": [lo, hi],
            "rounds": rounds, "history": history, "exceptions": exceptions,
            "max_min_ratio": max_min_ratio(est.capacities)}


def _round_summary(r: int, est, n: int, info) -> dict:
    caps = est.capacities
    tau, lo, hi = band(caps, n)
    out = {"round": r, "tau": tau, "band": [lo, hi], "max": max(caps), "min": min(caps),
           "max_min_ratio": max_min_ratio(caps), "out_of_band": len(out_of_band(caps, n))}
    if info:
        out["changes"] = {k: info[k] for k in ("splits", "merges", "restored_from")}
    return out


def write_outputs(result: dict, yaml_path, json_path, header: str, run: dict | None = None):
    topics = result["topics"]
    Path(yaml_path).write_text(header + yaml.safe_dump(
        {"topics": topics}, allow_unicode=True, sort_keys=False, width=100))
    payload = {"tau": result["tau"], "band": result["band"], "rounds": result["rounds"],
               "max_min_ratio": result["max_min_ratio"], "exceptions": result["exceptions"],
               "history": result["history"], "run": run or {},
               "topics": [{"id": i, **rec} for i, rec in enumerate(result["estimate"].records)]}
    Path(json_path).write_text(json.dumps(payload, indent=1) + "\n")


# --- the LLM rewriter -----------------------------------------------------

SYSTEM = "You design research taxonomies."


def _json_span(reply: str, open_ch: str, close_ch: str):
    start, end = reply.find(open_ch), reply.rfind(close_ch)
    if start == -1 or end <= start:
        raise ValueError(f"no JSON {open_ch}{close_ch} in reply")
    return json.loads(reply[start:end + 1])


def parse_split(reply: str, p: int, taken: set[str]) -> list[dict]:
    items = _json_span(reply, "[", "]")
    if len(items) != p:
        raise ValueError(f"expected {p} subtopics, got {len(items)}")
    seen = set(taken)
    out = []
    for it in items:
        name, d = str(it.get("name", "")).strip(), str(it.get("definition", "")).strip()
        if not name or not d:
            raise ValueError("every subtopic needs a name and a definition")
        if name.lower() in seen:
            raise ValueError(f"topic name {name!r} is already used")
        seen.add(name.lower())
        out.append({"name": name, "definition": d})
    return out


def parse_merge(reply: str, taken: set[str]) -> dict | None:
    obj = _json_span(reply, "{", "}")
    if obj.get("no_merge") is True:
        return None
    name, d = str(obj.get("name", "")).strip(), str(obj.get("definition", "")).strip()
    if not name or not d:
        raise ValueError("the merged topic needs a name and a definition")
    if name.lower() in taken:
        raise ValueError(f"topic name {name!r} is already used")
    return {"name": name, "definition": d}


def split_prompt(t: dict, p: int, titles: list[str], taken: set[str]) -> str:
    sample = "\n".join(f"- {x}" for x in titles[:40]) or "(none)"
    return (
        f"The research topic below holds too many papers. Split it into exactly {p} "
        "subtopics along real sub-areas of the field (as its papers show), "
        "together covering the whole topic without overlapping.\n\n"
        f"TOPIC: {t['name']} — {t['definition']}\n\nSAMPLE PAPER TITLES:\n{sample}\n\n"
        "Names already used by other topics (do not reuse): "
        + "; ".join(sorted(taken)) + "\n\n"
        "Reply with ONLY a JSON list of objects: "
        '{"name": "<3-8 word topic name>", "definition": "<one sentence: what is in, '
        'what is out>"}.')


def merge_prompt(s: dict, t: dict, taken: set[str]) -> str:
    return (
        "Topic A holds too few papers and is a candidate for merging into its "
        "nearest topic B.\n\n"
        f"A: {s['name']} — {s['definition']}\nB: {t['name']} — {t['definition']}\n\n"
        "If A and B together form one coherent research topic, reply with ONLY "
        '{"name": "<3-8 word topic name>", "definition": "<one sentence: what is in, '
        'what is out>"} for the merged topic. If they are distinct areas with no '
        'sensible merge, reply with ONLY {"no_merge": true}.\n'
        "Names already used by other topics (do not reuse): " + "; ".join(sorted(taken)))


class LLMRewriter:
    def __init__(self, llm, model: str, attempts: int = 3):
        self.llm, self.model, self.attempts = llm, model, attempts

    def _ask(self, prompt: str, parse):
        feedback = ""
        for _ in range(self.attempts):
            reply = self.llm.complete(model=self.model, system=SYSTEM,
                                      user=prompt + feedback, max_tokens=4000)
            try:
                return parse(reply)
            except (ValueError, json.JSONDecodeError, AttributeError) as e:
                feedback = f"\n\nYour previous answer was rejected: {e}. Fix it."
        raise RuntimeError(f"no valid LLM answer in {self.attempts} attempts")

    def split(self, t: dict, p: int, titles: list[str], taken: set[str]) -> list[dict]:
        return self._ask(split_prompt(t, p, titles, taken),
                         lambda r: parse_split(r, p, taken))

    def merge(self, s: dict, t: dict, taken: set[str]) -> dict | None:
        return self._ask(merge_prompt(s, t, taken), lambda r: parse_merge(r, taken))
