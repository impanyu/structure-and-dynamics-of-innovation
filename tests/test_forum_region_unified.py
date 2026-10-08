"""Region mode, R6 (2026-10-08): one tool set over papers and posts. search and
related return two separately ranked sections (papers 10 to a page, posts 5);
browse opens either kind and lists the posts citing a paper; random(kind)
replaces the two jumps; the pre-R6 names stay as aliases."""
import json

import numpy as np

from innovation.p2_forum.env import Action, Navigation
from innovation.p2_forum.region_env import (POST_PAGE_SIZE, QUERY_TIERS,
                                            RELATED_NOTICE, SEARCH_NOTICE, _tier)
from test_forum_region_env import make as make_two
from test_forum_region_env import post
from test_forum_region_paging import make as make_wide


def ids(items):
    return [h["node_id"] for h in items]


def search(env, agent="a", **args):
    return env.execute(agent, 0, Action("search", {"query": "@0", **args}))


def test_search_returns_two_sections_ranked_paged_and_tiered(tmp_path):
    env = make_wide(tmp_path)                       # 25 papers, all readable by a
    angles = [60, 0, 50, 10, 40, 20, 30]            # posted out of order
    by = {d: post(env, "z", f"idea @{d}")["node_id"] for d in angles}   # another author
    p1, p2 = search(env), search(env, page=2, post_page=2)
    assert POST_PAGE_SIZE == 5
    assert ids(p1["papers"]["items"]) == [f"p{i:02d}" for i in range(10)]
    assert ids(p2["papers"]["items"]) == [f"p{i:02d}" for i in range(10, 20)]
    assert (p1["papers"]["total"], p1["papers"]["pages"]) == (25, 3)
    assert ids(p1["posts"]["items"]) == [by[d] for d in (0, 10, 20, 30, 40)]
    assert ids(p2["posts"]["items"]) == [by[d] for d in (50, 60)]          # disjoint pages
    assert (p1["posts"]["page"], p1["posts"]["total"], p1["posts"]["pages"]) == (1, 7, 2)
    assert p2["posts"]["page"] == 2
    # query tiers on posts: cos 30 = .87 high, cos 40 = .77 medium, cos 50 = .64 low
    assert [h["relevance"] for h in p1["posts"]["items"]] == ["high"] * 4 + ["medium"]
    assert [h["relevance"] for h in p2["posts"]["items"]] == ["low", "low"]
    assert all(h["relevance"] == _tier(np.cos(np.radians(3 * i)), QUERY_TIERS)
               for i, h in enumerate(p1["papers"]["items"]))
    h = p1["posts"]["items"][0]
    assert h == {"node_id": by[0], "kind": "post", "author": "z", "text": "idea @0",
                 "relevance": "high"}
    assert p1["notice"] == SEARCH_NOTICE and set(p1) == {"papers", "posts", "notice"}
    # the page arguments are independent
    mixed = search(env, page=3, post_page=1)
    assert ids(mixed["papers"]["items"]) == [f"p{i:02d}" for i in range(20, 25)]
    assert ids(mixed["posts"]["items"]) == ids(p1["posts"]["items"])


def test_post_text_in_search_is_capped_at_300_characters(tmp_path):
    env = make_wide(tmp_path)
    post(env, "a", "idea @0 " + "x" * 500)
    assert len(search(env)["posts"]["items"][0]["text"]) == 300


def test_posts_in_search_follow_readability_and_mark_own_posts(tmp_path):
    env = make_two(tmp_path)                         # a: ball of 30 deg at 0; b: 35 deg at 50
    own = post(env, "a", "idea @5")["node_id"]        # a only
    shared = post(env, "b", "idea @25")["node_id"]    # in both balls
    b_only = post(env, "b", "idea @60")["node_id"]    # outside a's ball
    b_in_a = post(env, "b", "idea @0")["node_id"]     # inside a's ball, outside b's
    seen_a = {h["node_id"]: h["author"] for h in search(env, "a")["posts"]["items"]}
    assert seen_a == {own: "you", shared: "b", b_in_a: "b"}
    seen_b = {h["node_id"]: h["author"] for h in search(env, "b")["posts"]["items"]}
    assert seen_b == {shared: "you", b_only: "you", b_in_a: "you"}     # authors read their own
    assert b_only not in str(search(env, "a"))


def test_each_navigation_flag_closes_only_its_own_section(tmp_path):
    env = make_two(tmp_path)
    post(env, "a", "idea @5", ["c0"])
    env.nav = Navigation(board_search=False)
    out = search(env)
    assert "closed" in out["posts"]["error"] and len(out["papers"]["items"]) == 4
    rel = env.execute("a", 1, Action("related", {"node_id": "c0"}))
    assert "closed" in rel["posts"]["error"] and len(rel["papers"]["items"]) == 3
    env.nav = Navigation(corpus_search=False)
    out = search(env)
    assert "closed" in out["papers"]["error"] and len(out["posts"]["items"]) == 1
    env.nav = Navigation(corpus_search=False, board_search=False)
    out = search(env)
    assert "error" in out["papers"] and "error" in out["posts"] and "notice" in out
    # browse: corpus edges close a paper's paper lists, board edges its citing posts
    env.nav = Navigation(corpus_edges=False)
    v = env.execute("a", 2, Action("browse", {"node_id": "c0"}))
    assert v["cites"] == [] and len(v["cited_by_posts"]) == 1
    env.nav = Navigation(board_edges=False)
    v = env.execute("a", 3, Action("browse", {"node_id": "c0"}))
    assert len(v["cites"]) == 2 and v["cited_by_posts"] == []


def test_browse_paper_lists_readable_citing_posts_and_counts_the_hidden_ones(tmp_path):
    env = make_two(tmp_path)
    own_far = post(env, "a", "idea @5", ["c1"])["node_id"]
    own_near = post(env, "a", "idea @25", ["c1"])["node_id"]
    hidden = post(env, "b", "idea @60", ["c1"])["node_id"]       # a cannot read it
    theirs = post(env, "b", "idea @0", ["c1"])["node_id"]        # a can
    v = env.execute("a", 0, Action("browse", {"node_id": "c1"}))  # c1 at 20 deg
    assert ids(v["cited_by_posts"]) == [own_near, own_far, theirs]   # 5, 15, 20 deg away
    assert v["cited_by_posts"][0] == {"node_id": own_near, "kind": "post", "author": "you",
                                      "text": "idea @25", "relevance": "high"}
    assert v["cited_by_posts"][2]["author"] == "b"
    assert (v["cited_by_posts_page"], v["cited_by_posts_total"],
            v["cited_by_posts_pages"]) == (1, 3, 1)
    assert v["filtered"]["region_posts"] == 1 and hidden not in str(v)
    w = env.execute("b", 1, Action("browse", {"node_id": "c1"}))
    assert set(ids(w["cited_by_posts"])) == {own_near, hidden, theirs}
    assert w["filtered"]["region_posts"] == 1                     # own_far is outside b


def test_browse_paper_paginates_citing_posts_on_post_page(tmp_path):
    env = make_wide(tmp_path)
    posts = [post(env, "a", f"idea @{3 * i}", ["p05"])["node_id"] for i in range(12)]
    v1 = env.execute("a", 0, Action("browse", {"node_id": "p05"}))
    v2 = env.execute("a", 0, Action("browse", {"node_id": "p05", "post_page": 2}))
    assert len(v1["cited_by_posts"]) == 10 and len(v2["cited_by_posts"]) == 2
    assert sorted(ids(v1["cited_by_posts"] + v2["cited_by_posts"])) == sorted(posts)
    assert v1["cited_by_posts"][0]["node_id"] == posts[5]           # same angle as p05
    assert (v1["cited_by_posts_total"], v1["cited_by_posts_pages"]) == (12, 2)
    assert v2["cited_by_posts_page"] == 2
    assert ids(v1["cited_by"]) == ["p00"]                            # paper list unaffected


def test_browse_post_shows_mixed_cites_with_kinds_and_its_citers(tmp_path):
    env = make_two(tmp_path)
    pa = post(env, "a", "idea @5", ["c0"])["node_id"]
    pb = post(env, "a", "idea @25", [pa, "c1", "c2"])["node_id"]
    pc = post(env, "b", "idea @26", [pb])["node_id"]                 # b's, in a's ball
    v = env.execute("a", 0, Action("browse", {"node_id": pb}))
    assert (v["node_id"], v["kind"], v["author"], v["text"]) == (pb, "post", "you", "idea @25")
    assert [(c["node_id"], c["kind"]) for c in v["cites"]] == [
        ("c1", "paper"), (pa, "post"), ("c2", "paper")]              # 5, 20, 53 deg away
    assert v["cites"][0] == {"node_id": "c1", "kind": "paper", "title": "Title c1",
                             "year": 2021, "relevance": "high"}
    assert v["cites"][1] == {"node_id": pa, "kind": "post", "author": "you",
                             "text": "idea @5", "relevance": "high"}
    assert v["cites"][2]["relevance"] == "low"
    assert v["cited_by"] == [{"node_id": pc, "kind": "post", "author": "b",
                              "text": "idea @26", "relevance": "high"}]
    assert (v["cites_total"], v["cited_by_total"]) == (3, 1)
    assert "cited_by_posts" not in v
    assert env.execute("b", 1, Action("browse", {"node_id": pa}))["gate"] == "result"


def test_related_on_a_paper_and_on_a_post_gives_two_sections_without_the_item(tmp_path):
    env = make_two(tmp_path)
    p5 = post(env, "a", "idea @5")["node_id"]
    p25 = post(env, "a", "idea @25")["node_id"]
    on_paper = env.execute("a", 0, Action("related", {"node_id": "c1"}))   # 20 deg
    assert on_paper["kind"] == "paper" and on_paper["notice"] == RELATED_NOTICE
    assert ids(on_paper["papers"]["items"]) == ["c0", "c2", "c3"]
    assert ids(on_paper["posts"]["items"]) == [p25, p5]
    assert on_paper["posts"]["items"][0]["kind"] == "post"
    on_post = env.execute("a", 1, Action("related", {"node_id": p25}))
    assert on_post["node_id"] == p25 and on_post["kind"] == "post"
    assert ids(on_post["papers"]["items"]) == ["c1", "c0", "c2", "c3"]   # 5, 25, 53, 55 deg
    assert ids(on_post["posts"]["items"]) == [p5]                         # itself excluded
    assert on_post["posts"]["items"][0]["relevance"] == "high"           # 20 deg, paper tiers
    pb = post(env, "b", "idea @60")["node_id"]
    assert env.execute("a", 2, Action("related", {"node_id": pb}))["gate"] == "result"
    assert env.execute("a", 2, Action("related", {"node_id": "gen:t:99"}))["error"]


def test_random_draws_exactly_as_the_old_jumps(tmp_path):
    new, old = make_two(tmp_path / "n"), make_two(tmp_path / "o")
    for env in (new, old):
        post(env, "a", "idea @5")
        post(env, "b", "idea @25")
        post(env, "b", "idea @60")                                   # hidden from a
    seq_new, seq_old = [], []
    for k in range(12):
        kind = "paper" if k % 3 else "post"
        seq_new.append(new.execute("a", k, Action("random", {"kind": kind}))["node_id"])
        old_name = "sample_frontier" if kind == "paper" else "sample_board"
        seq_old.append(old.execute("a", k, Action(old_name, {}))["node_id"])
    assert seq_new == seq_old
    # and the draws are the old handlers' formula over the same rng stream
    rng = np.random.default_rng(0)
    members = ["c0", "c1", "c2", "c3"]
    posts = [n for n in new.ws.board_post_ids() if new.readable("a", n)]
    expect = [(members if k % 3 else posts)[int(rng.integers(4 if k % 3 else len(posts)))]
              for k in range(12)]
    assert seq_new == expect
    paper = new.execute("a", 20, Action("random", {}))               # default kind is paper
    assert paper["kind"] == "paper" and paper["text"].endswith("word ")
    one = new.execute("a", 21, Action("random", {"kind": "post"}))
    assert one["kind"] == "post" and one["author"] in ("you", "b")
    bad = new.execute("a", 22, Action("random", {"kind": "papers"}))
    assert "kind must be one of" in bad["error"]
    new.nav = Navigation(board_jump=False)
    assert "closed" in new.execute("a", 23, Action("random", {"kind": "post"}))["error"]
    assert "node_id" in new.execute("a", 24, Action("random", {"kind": "paper"}))


def test_old_names_are_aliases_and_old_logs_restore(tmp_path):
    env = make_two(tmp_path)
    pa = post(env, "a", "idea @5", ["c0"])["node_id"]
    sb = env.execute("a", 1, Action("search_board", {"query": "@5", "k": 5}))
    assert set(sb) == {"papers", "posts", "notice"} and ids(sb["posts"]["items"]) == [pa]
    assert env.execute("a", 2, Action("browse_board", {"node_id": pa}))["kind"] == "post"
    assert env.execute("a", 3, Action("browse_board", {"node_id": "c0"}))["kind"] == "paper"
    assert env.execute("a", 4, Action("sample_frontier", {}))["kind"] == "paper"
    assert env.execute("a", 5, Action("sample_board", {}))["node_id"] == pa
    env.execute("a", 6, Action("add_links", {"src_id": pa, "dst_ids": ["c1"]}))
    events = env.event_log.read_all()
    # a pre-R6 read event, with its old result shape, in the middle of the log
    events.insert(1, {"run_id": "t", "agent_id": "a", "step": 1, "action": "search_board",
                      "args": {"query": "x", "k": 5},
                      "result": {"hits": [], "filtered": {"region": 0}}})
    fresh = make_two(tmp_path / "x")
    fresh.restore(events)
    assert fresh.ws.board_post_ids() == [pa]
    assert sorted(fresh.ws.board_neighbors(pa)[0]) == ["c0", "c1"]


def test_posts_come_first_so_a_truncated_history_entry_still_shows_them(tmp_path):
    """The rolling history keeps json.dumps(result)[:1500]: posts and the short
    notice lead, the long papers section comes last."""
    env = make_wide(tmp_path)                                    # 25 papers
    pid = post(env, "a", "idea @0", ["p05"])["node_id"]
    for out in (search(env), env.execute("a", 1, Action("related", {"node_id": "p03"}))):
        dumped = json.dumps(out)
        tail = dumped[dumped.index('"posts"'):]
        assert tail.startswith('"posts": {"page": 1')
        assert pid in dumped[:1500] and '"notice"' in dumped[:1500]
        keys = [k for k in out if k in ("posts", "notice", "papers")]
        assert keys == ["posts", "notice", "papers"]
    assert json.dumps(search(env)).startswith('{"posts": {')
    v = env.execute("a", 2, Action("browse", {"node_id": "p05"}))
    order = list(v)
    assert order.index("cited_by_posts") < order.index("cites") < order.index("cited_by")


def test_posts_section_counts_the_posts_the_gate_hides(tmp_path):
    env = make_two(tmp_path)
    post(env, "a", "idea @5")                                    # a only
    post(env, "b", "idea @25")                                   # both
    post(env, "b", "idea @60")                                   # b only
    post(env, "b", "idea @180")                                  # b only (author)
    assert search(env, "a")["posts"]["filtered"] == 2
    assert search(env, "b")["posts"]["filtered"] == 1
    rel = env.execute("a", 0, Action("related", {"node_id": "c0"}))
    assert rel["posts"]["filtered"] == 2 and rel["posts"]["total"] == 2
    env.nav = Navigation(board_search=False)
    assert "filtered" not in search(env, "a")["posts"]


def test_compact_history_of_a_full_search_keeps_every_post_and_paper(tmp_path):
    from innovation.p2_forum.agent import ForumAgentPolicy
    from innovation.p2_forum.runner import REGION_HISTORY_CHARS
    env = make_wide(tmp_path)                                    # 25 papers
    pids = [post(env, "z", f"idea @{i} " + "w" * 400)["node_id"] for i in range(5)]
    out = search(env)
    assert [len(h["text"]) for h in out["posts"]["items"]] == [300] * 5   # full-length posts
    assert len(out["papers"]["items"]) == 10
    raw = json.dumps(out)[:1500]
    assert not any(h["node_id"] in raw for h in out["papers"]["items"])  # the old cut lost them
    pol = ForumAgentPolicy(llm=None, model="m", topics=[], compact_history=True,
                           history_chars=REGION_HISTORY_CHARS)
    entry = pol.history_entry(out)
    assert REGION_HISTORY_CHARS == 3000 and len(entry) < 3000
    assert all(p in entry for p in pids)
    assert all(f'"{h["node_id"]}"' in entry for h in out["papers"]["items"])
    assert SEARCH_NOTICE not in entry and "Abstract of" not in entry
    assert json.loads(entry)["posts"]["filtered"] == 0
