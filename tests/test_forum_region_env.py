# tests/test_forum_region_env.py
import json
import math
import re

import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import Action, Navigation
from innovation.p2_forum.region import build_region
from innovation.p2_forum.region_env import RegionGatedEnvironment
from innovation.p2_forum.workspace import Workspace


def unit(deg):
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a), 0.0, 0.0], dtype=np.float32)


class AngleEmbedder:
    """'... @<deg> ...' embeds at that angle on the unit circle; text without
    an angle embeds at 270 degrees, outside every region below."""
    dim = 4

    def encode(self, texts):
        out = []
        for t in texts:
            m = re.search(r"@(-?\d+)", t)
            out.append(unit(int(m.group(1)) if m else 270))
        return np.array(out, dtype=np.float32)


# Corpus papers on the unit circle (degrees). At coverage 0.5 (4 of 8):
#   agent a, seed c0:  {c0, c1, c2, c3}, radius cos 30 -> posts in [-30, 30] degrees
#   agent b, seed c4:  {c4, c5, c1, c6}, radius cos 35 -> posts in [15, 85] degrees
# c1 is shared and on neither boundary; c7 is nobody's. Posts at @5 are a-only,
# @25 both, @60 b-only.
ANGLES = {"c0": 0, "c1": 20, "c2": -28, "c3": -30, "c4": 50, "c5": 75, "c6": 85, "c7": 180}
CITES = {"c0": ["c2", "c4", "c1"], "c5": ["c0"]}
VENUES = {"c0": "ICML", "c1": "NeurIPS", "c2": "ACL", "c3": "CVPR", "c4": "ICLR",
          "c5": "AAAI", "c6": "ICCV", "c7": "ICML"}
INSERT_ORDER = ["c1", "c2", "c3", "c4", "c6", "c7", "c0", "c5"]     # cited before citing


def make(tmp_path, coverage=0.5, budget=None):
    corpus = IdeaGraph()
    for pid in INSERT_ORDER:
        corpus.add_idea(pid, f"Title {pid}\n\nAbstract of {pid}. " + "word " * 100,
                        CITES.get(pid, []), source="corpus", year=2020 + int(pid[1]) % 5,
                        meta={"venue": VENUES[pid]})
    corpus.freeze()
    ids = list(ANGLES)
    vecs = np.stack([unit(ANGLES[i]) for i in ids])
    ci = VectorIndex(4)
    ci.add(ids, vecs)
    ws = Workspace(corpus=corpus, corpus_index=ci, board_index=VectorIndex(4),
                   embedder=AngleEmbedder(), run_id="t")
    regions = {"a": build_region("c0", ids, vecs, coverage),
               "b": build_region("c4", ids, vecs, coverage)}
    return RegionGatedEnvironment(run_id="t", workspace=ws,
                                  event_log=EventLog(tmp_path / "e.jsonl"),
                                  rng=np.random.default_rng(0), regions=regions,
                                  generation_budget=budget)


def post(env, agent, text, cites=()):
    return env.execute(agent, 0, Action("generate", {"text": text, "cited_ids": list(cites)}))


def test_regions_in_the_fixture_are_as_documented(tmp_path):
    env = make(tmp_path)
    assert env.regions["a"].members == {"c0", "c1", "c2", "c3"}
    assert env.regions["b"].members == {"c4", "c5", "c1", "c6"}


def test_readable_corpus_by_membership_and_posts_by_vector(tmp_path):
    env = make(tmp_path)
    assert env.readable("a", "c1") and not env.readable("a", "c4")
    assert env.readable("b", "c1") and not env.readable("b", "c7")
    only_a = post(env, "a", "idea @5")["node_id"]
    assert env.readable("a", only_a) and not env.readable("b", only_a)
    shared = post(env, "a", "idea @25")["node_id"]
    assert env.readable("b", shared)
    assert not env.readable("a", "gen:t:999") and not env.readable("a", "nope")


def test_search_returns_only_readable_papers_ranked_with_paper_fields(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "near @95", "k": 5}))
    assert [h["node_id"] for h in out["hits"]] == ["c1", "c0", "c2", "c3"]   # b's papers hidden
    h = out["hits"][0]
    assert h["store"] == "corpus" and h["title"] == "Title c1"
    assert h["text"].startswith("Abstract of c1.") and len(h["text"]) <= 300
    assert h["year"] == 2021 and h["venue"] == "NeurIPS"
    assert "gate" not in out                                  # no query gate, ever
    one = env.execute("a", 1, Action("search", {"query": "@0", "k": 1}))
    assert [h["node_id"] for h in one["hits"]] == ["c0"]


def test_closed_search_is_refused(tmp_path):
    env = make(tmp_path)
    env.nav = Navigation(corpus_search=False)
    out = env.execute("a", 0, Action("search", {"query": "@0"}))
    assert "closed" in out["error"] and "hits" not in out


def test_browse_gates_the_target_and_lists_all_readable_neighbours(tmp_path):
    env = make(tmp_path)
    assert env.execute("a", 0, Action("browse", {"node_id": "c4"}))["gate"] == "result"
    unknown = env.execute("a", 0, Action("browse", {"node_id": "zzz"}))
    assert "error" in unknown and "gate" not in unknown
    v = env.execute("a", 1, Action("browse", {"node_id": "c0"}))
    assert v["title"] == "Title c0" and v["text"].startswith("Abstract of c0.")
    assert v["text"].endswith("word ")                        # the full abstract
    assert v["year"] == 2020 and v["venue"] == "ICML"
    assert [c["node_id"] for c in v["cites"]] == ["c2", "c1"]  # c4 hidden
    assert v["cites"][1] == {"node_id": "c1", "title": "Title c1", "year": 2021}
    assert v["cited_by"] == []                                 # c5 hidden
    assert v["filtered"] == {"region": 2}
    w = env.execute("b", 2, Action("browse", {"node_id": "c1"}))
    assert w["cited_by"] == [] and w["filtered"] == {"region": 1}   # c0 is outside b


def test_browse_with_closed_edges_shows_no_neighbours(tmp_path):
    env = make(tmp_path)
    env.nav = Navigation(corpus_edges=False)
    v = env.execute("a", 0, Action("browse", {"node_id": "c0"}))
    assert v["cites"] == [] and v["cited_by"] == [] and v["title"] == "Title c0"
    assert v["filtered"] == {"region": 0}                     # same keys as the open path


def test_sample_frontier_stays_in_the_region_and_covers_it(tmp_path):
    env = make(tmp_path)
    seen = {env.execute("a", s, Action("sample_frontier", {}))["node_id"] for s in range(60)}
    assert seen == {"c0", "c1", "c2", "c3"}
    out = env.execute("a", 0, Action("sample_frontier", {}))
    assert out["store"] == "corpus" and out["title"] and out["text"].endswith("word ")
    env.nav = Navigation(corpus_jump=False)
    assert "closed" in env.execute("a", 0, Action("sample_frontier", {}))["error"]


def test_related_ranks_readable_neighbours_excluding_the_source(tmp_path):
    env = make(tmp_path)
    out = env.execute("a", 0, Action("related", {"node_id": "c1", "k": 10}))
    assert out["node_id"] == "c1"
    assert [h["node_id"] for h in out["related"]] == ["c0", "c2", "c3"]
    assert out["related"][0]["title"] == "Title c0" and out["related"][0]["venue"] == "ICML"
    one = env.execute("b", 1, Action("related", {"node_id": "c1", "k": 1}))
    assert [h["node_id"] for h in one["related"]] == ["c4"]
    assert env.execute("a", 2, Action("related", {"node_id": "c4"}))["gate"] == "result"
    assert "error" in env.execute("a", 2, Action("related", {"node_id": "zzz"}))
    env.nav = Navigation(corpus_search=False)
    assert "closed" in env.execute("a", 3, Action("related", {"node_id": "c0"}))["error"]


def test_post_outside_the_authors_ball_is_not_published(tmp_path):
    env = make(tmp_path)
    out = post(env, "a", "idea @60", ["c0"])
    assert out["gate"] == "post" and "node_id" not in out
    assert env.ws.board_post_ids() == []


def test_post_drops_unreadable_cites(tmp_path):
    env = make(tmp_path)
    out = post(env, "a", "idea @5", ["c0", "c4", "c1", "nope"])
    assert out["dropped_cites"] == ["c4", "nope"]
    assert env.ws.board_neighbors(out["node_id"])[0] == ["c0", "c1"]
    clean = post(env, "a", "idea @6", ["c2"])
    assert "dropped_cites" not in clean


def test_generation_budget_is_respected(tmp_path):
    env = make(tmp_path, budget=1)
    assert "node_id" in post(env, "a", "idea @5")
    assert "budget" in post(env, "a", "idea @6")["error"]


def test_board_reads_are_gated_by_vector(tmp_path):
    env = make(tmp_path)
    pa = post(env, "a", "idea @5", ["c0"])["node_id"]
    pb = post(env, "b", "idea @60", ["c4"])["node_id"]
    s = env.execute("a", 1, Action("search_board", {"query": "@60", "k": 5}))
    assert [h["node_id"] for h in s["hits"]] == [pa] and s["filtered"] == {"region": 1}
    assert s["hits"][0]["store"] == "board" and "score" in s["hits"][0]
    assert env.execute("a", 2, Action("browse_board", {"node_id": pb}))["gate"] == "result"
    assert all(env.execute("a", k, Action("sample_board", {}))["node_id"] == pa
               for k in range(5))
    v = env.execute("a", 3, Action("browse_board", {"node_id": pa}))
    assert v["text"] == "idea @5"
    assert v["cites"] == [{"node_id": "c0", "store": "corpus", "title": "Title c0",
                           "year": 2020}]


def test_browse_board_hides_unreadable_neighbours(tmp_path):
    env = make(tmp_path)
    pa = post(env, "a", "idea @25")["node_id"]               # readable by both
    pb = post(env, "b", "idea @60", [pa, "c4"])["node_id"]   # b only
    v = env.execute("a", 1, Action("browse_board", {"node_id": pa}))
    assert v["cited_by"] == []                               # pb hidden from a
    w = env.execute("b", 2, Action("browse_board", {"node_id": pb}))
    assert [c["node_id"] for c in w["cites"]] == [pa, "c4"]
    assert env.execute("a", 3, Action("browse_board", {"node_id": "c0"}))["error"]


def test_sample_board_with_nothing_readable(tmp_path):
    env = make(tmp_path)
    post(env, "b", "idea @60")
    assert "error" in env.execute("a", 0, Action("sample_board", {}))


def test_links_need_both_ends_readable(tmp_path):
    env = make(tmp_path)
    pa = post(env, "a", "idea @5")["node_id"]
    pb = post(env, "b", "idea @60")["node_id"]
    assert env.execute("a", 1, Action("add_links", {"src_id": pa, "dst_ids": ["c4"]}))["gate"] == "link"
    assert env.execute("a", 1, Action("add_links", {"src_id": pb, "dst_ids": ["c0"]}))["gate"] == "link"
    ok = env.execute("a", 2, Action("add_links", {"src_id": pa, "dst_ids": ["c1"]}))
    assert ok["added"] == ["c1"]
    assert env.execute("b", 3, Action("remove_links", {"src_id": pa, "dst_ids": ["c1"]}))["gate"] == "link"
    gone = env.execute("a", 4, Action("remove_links", {"src_id": pa, "dst_ids": ["c1"]}))
    assert "gate" not in gone and gone["removed"]


def test_links_to_unknown_ids_are_a_plain_error_not_a_gate(tmp_path):
    env = make(tmp_path)
    pa = post(env, "a", "idea @5")["node_id"]
    for args in ({"src_id": pa, "dst_ids": ["c1", "nope"]},
                 {"src_id": "gen:t:99", "dst_ids": ["c1"]}):
        for name in ("add_links", "remove_links"):
            out = env.execute("a", 1, Action(name, args))
            bad = "nope" if "nope" in args["dst_ids"] else "gen:t:99"
            assert out == {"error": f"{bad} is not a known paper or post"}
    # an existing but unreadable id is still a gate refusal
    assert env.execute("a", 2, Action("add_links", {"src_id": pa, "dst_ids": ["c7"]}))["gate"] == "link"


def test_a_shared_post_never_shows_a_cited_paper_outside_the_readers_region(tmp_path):
    """b can read the post (@25 is in both balls), but c2 (cited at post time)
    and c3 (linked later by a) lie outside b's region and stay hidden from b."""
    env = make(tmp_path)
    shared = post(env, "a", "idea @25", ["c2", "c1"])["node_id"]
    assert env.readable("b", shared) and not env.readable("b", "c2")
    env.execute("a", 1, Action("add_links", {"src_id": shared, "dst_ids": ["c3"]}))
    assert sorted(env.ws.board_neighbors(shared)[0]) == ["c1", "c2", "c3"]

    seen = json.dumps([
        env.execute("b", 2, Action("browse_board", {"node_id": shared})),
        env.execute("b", 3, Action("search_board", {"query": "@25"})),
        env.execute("b", 4, Action("sample_board", {})),
        env.execute("b", 5, Action("browse", {"node_id": "c1"})),
    ])
    assert "c2" not in seen and "c3" not in seen
    v = env.execute("b", 6, Action("browse_board", {"node_id": shared}))
    assert [c["node_id"] for c in v["cites"]] == ["c1"]
    a_view = env.execute("a", 7, Action("browse_board", {"node_id": shared}))
    assert sorted(c["node_id"] for c in a_view["cites"]) == ["c1", "c2", "c3"]


def test_restore_round_trip(tmp_path):
    env = make(tmp_path, budget=5)
    pa = post(env, "a", "idea @5", ["c0", "c4"])["node_id"]
    post(env, "a", "idea @60")                                 # refused, must not replay
    pb = post(env, "b", "idea @60", ["c5"])["node_id"]
    env.execute("a", 1, Action("add_links", {"src_id": pa, "dst_ids": ["c1"]}))
    events = env.event_log.read_all()
    fresh = make(tmp_path / "x", budget=5)
    fresh.restore(events)
    assert fresh.ws.board_post_ids() == [pa, pb]
    assert sorted(fresh.ws.board_neighbors(pa)[0]) == ["c0", "c1"]   # dropped c4 stays dropped
    assert fresh.generation_budget == 3
    assert fresh.readable("a", pa) and not fresh.readable("a", pb)
    assert post(fresh, "a", "idea @7")["node_id"] not in (pa, pb)
