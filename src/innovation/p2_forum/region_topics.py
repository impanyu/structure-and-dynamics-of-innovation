"""A region's topic list: what an agent's prompt says its research area is
(spec, REVISION 2026-10-06, R5 2026-10-07).

The region gate (region_env.py) is hard and stays as it is; this list is the
soft half. Each region's member papers are clustered with k-means over their
embeddings, and each cluster is named by one LLM call from its most central
papers. The agent reads the names and descriptions and plans inside them, so it
stops searching for things the gate will never show it.

Deterministic for a fixed region, corpus and seed. The namer goes through the
run's CachedLLM, so identical clusters (e.g. coverage 100%, where every agent
shares one region) cost one call each, however many agents share them.
"""
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans

from innovation.p2_forum.region import Region, _unit

# Topics per region: one per ~150 member papers, at least 3 and at most 30
# (never more than the region has papers). 185 -> 3, 1854 -> 12, 3708 -> 25,
# 18542 -> 30. Few enough to read in a prompt; fine enough that a topic is a
# recognisable research line rather than a whole field.
PAPERS_PER_TOPIC = 150
MIN_TOPICS = 3
MAX_TOPICS = 30
N_REPRESENTATIVES = 8
ABSTRACT_CHARS = 300


def n_topics(n_members: int) -> int:
    """min(MAX_TOPICS, max(MIN_TOPICS, round(n_members / PAPERS_PER_TOPIC)))."""
    return min(MAX_TOPICS, max(MIN_TOPICS, round(n_members / PAPERS_PER_TOPIC)))


@dataclass(frozen=True)
class Cluster:
    members: tuple[str, ...]          # sorted ids
    representatives: tuple[str, ...]  # the N_REPRESENTATIVES closest to the centroid


def cluster_region(region: Region, ids: list[str], vecs: np.ndarray,
                   seed: int) -> list[Cluster]:
    """k-means (k = n_topics, capped at the number of distinct member vectors) over the region's
    member papers, largest cluster first.

    `ids`/`vecs` are the corpus index's rows; members are taken in sorted-id
    order so the result does not depend on set iteration order. A cluster's
    representatives are its members closest (cosine) to its centroid, ties
    broken by id. Size ties are broken by the cluster's smallest member id."""
    row = {pid: i for i, pid in enumerate(ids)}
    members = sorted(region.members)
    if not members:
        return []
    unit = _unit(np.asarray(vecs)[[row[m] for m in members]])
    # Never more clusters than distinct points (duplicate abstracts embed
    # identically), so k-means cannot leave a cluster empty.
    k = min(n_topics(len(members)), len(np.unique(unit, axis=0)))
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(unit)
    clusters = []
    for c in range(k):
        idx = [i for i in range(len(members)) if labels[i] == c]
        if not idx:
            continue
        centroid = _unit(unit[idx].mean(axis=0))
        sims = unit[idx] @ centroid
        ranked = sorted(zip(idx, sims), key=lambda t: (-float(t[1]), members[t[0]]))
        clusters.append(Cluster(
            members=tuple(members[i] for i in idx),
            representatives=tuple(members[i] for i, _ in ranked[:N_REPRESENTATIVES])))
    clusters.sort(key=lambda c: (-len(c.members), c.members[0]))
    return clusters


NAMER_SYSTEM = ("You name research topics. Reply with STRICT JSON only, "
                "no other text.")


def _namer_prompt(reps: list[tuple[str, str]], attempt: int) -> str:
    papers = "\n\n".join(f"{i}. {title}\n{abstract[:ABSTRACT_CHARS]}"
                         for i, (title, abstract) in enumerate(reps, 1))
    head = "" if attempt == 1 else f"(attempt {attempt}: your previous reply was not valid JSON)\n\n"
    return (head
            + "These are the most central papers of one cluster of research papers:\n\n"
            + papers
            + "\n\nName the research topic this cluster covers. Reply with STRICT JSON:\n"
            '{"name": "<3-8 word research topic>", "description": "<one sentence>"}\n'
            "The name must be specific enough that a researcher knows which papers "
            "are in the topic and which are out. Do not mention venue names.")


def _parse_topic(reply: str) -> dict | None:
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(reply[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name, desc = obj.get("name"), obj.get("description")
    if not (isinstance(name, str) and name.strip()
            and isinstance(desc, str) and desc.strip()):
        return None
    return {"name": name.strip(), "description": desc.strip()}


def name_topic(llm, model: str, reps: list[tuple[str, str]]) -> dict:
    """{"name", "description"} for a cluster from its representatives'
    (title, abstract) pairs. One call; on an unparseable reply, one retry with
    an attempt-numbered prompt (so a disk cache does not replay the bad reply);
    then ValueError."""
    for attempt in (1, 2):
        reply = llm.complete(model=model, system=NAMER_SYSTEM,
                             user=_namer_prompt(reps, attempt), max_tokens=1000)
        topic = _parse_topic(reply)
        if topic is not None:
            return topic
    raise ValueError(f"topic namer {model} gave no valid JSON after 2 attempts: {reply[:200]!r}")


def region_topics(llm, model: str, region: Region, ids: list[str], vecs: np.ndarray,
                  seed: int, text_of: Callable[[str], tuple[str, str]]) -> list[dict]:
    """The region's topics, largest first: [{"name", "description", "n_papers"}].
    `text_of(paper_id)` gives a paper's (title, abstract)."""
    out = []
    for c in cluster_region(region, ids, vecs, seed):
        topic = name_topic(llm, model, [text_of(r) for r in c.representatives])
        out.append({**topic, "n_papers": len(c.members)})
    return out
