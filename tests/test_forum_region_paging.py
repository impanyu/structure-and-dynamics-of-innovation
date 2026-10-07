"""Region mode listings (R5, 2026-10-07): ranked by cosine, 10 per page,
tagged high/medium/low relevance; every search carries the topics notice."""
import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import Action
from innovation.p2_forum.region import build_region
from innovation.p2_forum.region_env import (PAGE_SIZE, PAPER_TIERS, QUERY_TIERS,
                                            SEARCH_NOTICE, RegionGatedEnvironment,
                                            _tier)
from innovation.p2_forum.workspace import Workspace
from test_forum_region_env import AngleEmbedder, unit

# 25 papers p00..p24 at 3-degree steps (0..72 deg); p00 cites all the others.
N = 25
ANG = {f"p{i:02d}": 3 * i for i in range(N)}


def make(tmp_path, coverage=1.0, angles=ANG, cites=None):
    cites = {"p00": [p for p in angles if p != "p00"]} if cites is None else cites
    corpus = IdeaGraph()
    for pid in sorted(angles, key=lambda p: p == "p00"):     # cited before citing
        corpus.add_idea(pid, f"Title {pid}\n\nAbstract of {pid}.", cites.get(pid, []),
                        source="corpus", year=2021, meta={"venue": "ICML"})
    corpus.freeze()
    ids = list(angles)
    vecs = np.stack([unit(angles[i]) for i in ids])
    ci = VectorIndex(4)
    ci.add(ids, vecs)
    ws = Workspace(corpus=corpus, corpus_index=ci, board_index=VectorIndex(4),
                   embedder=AngleEmbedder(), run_id="t")
    return RegionGatedEnvironment(
        run_id="t", workspace=ws, event_log=EventLog(tmp_path / "e.jsonl"),
        rng=np.random.default_rng(0), regions={"a": build_region("p00", ids, vecs, coverage)})


def search(env, page=None, query="@0", **extra):
    args = {"query": query, **extra}
    if page is not None:
        args["page"] = page
    return env.execute("a", 0, Action("search", args))


def ids(items):
    return [h["node_id"] for h in items]


def test_search_pages_are_disjoint_ordered_and_report_totals(tmp_path):
    env = make(tmp_path)
    p1, p2, p3 = search(env), search(env, 2), search(env, 3)
    assert ids(p1["hits"]) == [f"p{i:02d}" for i in range(10)]
    assert ids(p2["hits"])[0] == "p10"                       # page 2 starts at item 11
    assert ids(p1["hits"] + p2["hits"] + p3["hits"]) == list(ANG)
    assert [len(p["hits"]) for p in (p1, p2, p3)] == [10, 10, 5]
    assert all((p["total"], p["pages"]) == (25, 3) for p in (p1, p2, p3))
    assert [p["page"] for p in (p1, p2, p3)] == [1, 2, 3]
    past = search(env, 9)
    assert past["hits"] == [] and (past["page"], past["total"], past["pages"]) == (9, 25, 3)


@pytest.mark.parametrize("bad", [0, -2, "x", None, 1.0, "1"])
def test_invalid_page_is_page_one(tmp_path, bad):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "@0", "page": bad}))
    assert out["page"] == 1 and ids(out["hits"])[0] == "p00"


def test_notice_is_on_every_search_and_old_k_is_ignored(tmp_path):
    env = make(tmp_path)
    for out in (search(env), search(env, 3), search(env, 50), search(env, k=3),
                search(env, query="@180")):
        assert out["notice"] == SEARCH_NOTICE and "error" not in out
    assert len(search(env, k=3)["hits"]) == PAGE_SIZE
    assert "outside your topics" in SEARCH_NOTICE


def test_search_is_still_gated_to_the_region(tmp_path):
    env = make(tmp_path, coverage=0.5)                     # 12 papers nearest p00
    members = env.regions["a"].members
    assert len(members) == 12
    out = search(env)
    both = search(env)["hits"] + search(env, 2)["hits"]
    assert set(ids(both)) == members and out["total"] == 12 and out["pages"] == 2
    assert "gate" not in out


def test_tier_thresholds():
    assert [_tier(c, QUERY_TIERS) for c in (0.80, 0.799, 0.72, 0.719)] == \
        ["high", "medium", "medium", "low"]
    assert [_tier(c, PAPER_TIERS) for c in (0.85, 0.849, 0.78, 0.779)] == \
        ["high", "medium", "medium", "low"]


def test_search_and_related_tiers_follow_cosine(tmp_path):
    # Query @0. cos 36 = .809 (high), 37 = .799 / 43 = .731 (medium), 44 = .719 (low).
    angles = {"p00": 0, "q36": 36, "q37": 37, "q43": 43, "q44": 44}
    env = make(tmp_path, angles=angles, cites={})
    hits = search(env)["hits"]
    assert [(h["node_id"], h["relevance"]) for h in hits] == [
        ("p00", "high"), ("q36", "high"), ("q37", "medium"), ("q43", "medium"),
        ("q44", "low")]
    # Paper p00. cos 31 = .857 (high), 32 = .848 / 38 = .788 (medium), 39 = .777 (low).
    angles = {"p00": 0, "r31": 31, "r32": 32, "r38": 38, "r39": 39}
    env = make(tmp_path / "x", angles=angles, cites={"p00": ["r39", "r31", "r38", "r32"]})
    rel = env.execute("a", 0, Action("related", {"node_id": "p00"}))
    assert [(h["node_id"], h["relevance"]) for h in rel["related"]] == [
        ("r31", "high"), ("r32", "medium"), ("r38", "medium"), ("r39", "low")]
    v = env.execute("a", 1, Action("browse", {"node_id": "p00"}))
    assert [(c["node_id"], c["relevance"]) for c in v["cites"]] == [
        ("r31", "high"), ("r32", "medium"), ("r38", "medium"), ("r39", "low")]


def test_related_paginates_and_excludes_the_paper(tmp_path):
    env = make(tmp_path)
    p1 = env.execute("a", 0, Action("related", {"node_id": "p00"}))
    p3 = env.execute("a", 0, Action("related", {"node_id": "p00", "page": 3, "k": 4}))
    assert ids(p1["related"]) == [f"p{i:02d}" for i in range(1, 11)]
    assert ids(p3["related"]) == ["p21", "p22", "p23", "p24"]
    assert (p1["total"], p1["pages"]) == (24, 3) and "notice" in p3


def test_browse_ranks_and_paginates_each_neighbour_list(tmp_path):
    env = make(tmp_path)
    v1 = env.execute("a", 0, Action("browse", {"node_id": "p00"}))
    v3 = env.execute("a", 0, Action("browse", {"node_id": "p00", "ref_page": 3}))
    assert ids(v1["cites"]) == [f"p{i:02d}" for i in range(1, 11)]
    assert ids(v3["cites"]) == [f"p{i:02d}" for i in range(21, 25)]
    assert (v1["cites_total"], v1["cites_pages"]) == (24, 3)
    assert v1["filtered"] == {"region": 0}
    w = env.execute("a", 0, Action("browse", {"node_id": "p07", "cited_by_page": 2}))
    assert w["cited_by"] == [] and (w["cited_by_total"], w["cited_by_pages"]) == (1, 1)
    assert ids(env.execute("a", 0, Action("browse", {"node_id": "p07"}))["cited_by"]) == ["p00"]


def test_browse_filtered_counts_are_unchanged_by_paging(tmp_path):
    env = make(tmp_path, coverage=0.5)                     # p00..p11 readable
    v = env.execute("a", 0, Action("browse", {"node_id": "p00", "ref_page": 2}))
    assert ids(v["cites"]) == ["p11"]
    assert v["cites_total"] == 11 and v["filtered"] == {"region": 13}


def test_board_search_and_browse_board_paginate_with_tiers(tmp_path):
    env = make(tmp_path)
    posts = [env.execute("a", 0, Action("generate", {"text": f"idea @{3 * i}",
                                                      "cited_ids": list(ANG)[:12]}))["node_id"]
             for i in range(12)]
    s1 = env.execute("a", 1, Action("search_board", {"query": "@0", "k": 2}))
    s2 = env.execute("a", 1, Action("search_board", {"query": "@0", "page": 2}))
    assert ids(s1["hits"]) == posts[:10] and ids(s2["hits"]) == posts[10:]
    assert (s1["total"], s1["pages"]) == (12, 2) and s1["filtered"] == {"region": 0}
    assert "score" not in s1["hits"][0] and s1["hits"][0]["relevance"] == "high"
    assert "posts within your allowed topics" in s1["notice"]
    b1 = env.execute("a", 2, Action("browse_board", {"node_id": posts[0]}))
    b2 = env.execute("a", 2, Action("browse_board", {"node_id": posts[0], "ref_page": 2}))
    assert ids(b1["cites"]) == list(ANG)[:10] and ids(b2["cites"]) == list(ANG)[10:12]
    assert (b1["cites_total"], b1["cites_pages"]) == (12, 2)
    assert b1["cites"][0]["relevance"] == "high"
