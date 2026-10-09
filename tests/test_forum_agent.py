import json
import re

from innovation.p2_forum.agent import VALID_ACTIONS, ForumAgentPolicy


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, model, system, user, max_tokens):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_every_action_name_is_available_to_the_agent():
    assert VALID_ACTIONS == {"search", "browse", "related", "sample_frontier",
                             "search_board", "browse_board", "sample_board",
                             "generate", "add_links", "remove_links"}


def test_the_agents_topics_appear_in_its_system_prompt():
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'])
    pol = ForumAgentPolicy(llm=llm, model="m",
                           topics=["federated learning privacy",
                                   "sparse attention kernels"])

    pol.act({"step": 0, "last_result": {}})

    system, _ = llm.prompts[0]
    assert "federated learning privacy" in system
    assert "sparse attention kernels" in system


def test_board_actions_are_documented_in_the_user_prompt():
    llm = ScriptedLLM(['{"action": "search_board", "args": {"query": "x"}}'])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    action = pol.act({"step": 0, "last_result": {}})

    _, user = llm.prompts[0]
    assert "search_board" in user and "browse_board" in user
    assert action.name == "search_board"


def test_malformed_reply_falls_back_to_a_corpus_jump():
    llm = ScriptedLLM(["not json at all"])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    assert pol.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_unknown_action_name_falls_back_to_a_corpus_jump():
    llm = ScriptedLLM([json.dumps({"action": "delete_corpus", "args": {}})])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"])

    assert pol.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_gated_prompt_states_the_hard_rule_and_uses_given_order():
    from innovation.core.llm import FakeLLM
    from innovation.p2_forum.agent import GATED_ACTIONS_DOC, GATED_SYSTEM, ForumAgentPolicy
    pol = ForumAgentPolicy(llm=FakeLLM(default='{"action":"search","args":{"query":"x"}}'),
                           model="m", topics=["B — def b", "A — def a"],
                           system_template=GATED_SYSTEM, actions_doc=GATED_ACTIONS_DOC)
    assert pol.system.index("B — def b") < pol.system.index("A — def a")
    assert "will not be published" in pol.system
    pol.act({"step": 0, "last_result": {}})
    assert GATED_ACTIONS_DOC.splitlines()[1] in pol.llm.calls[0]["user"]


def test_gated_doc_asks_for_researcher_style_searches():
    from innovation.p2_forum.agent import GATED_ACTIONS_DOC
    lines = GATED_ACTIONS_DOC.splitlines()
    assert lines[1] == ('{"action": "search", "args": {"query": "<the title of a specific paper, '
                        'or a short meaningful phrase as you would type into Google Scholar>", '
                        '"k": 5}} -- search the published literature (do not paste lists of keywords)')
    assert lines[2] == ('{"action": "browse", "args": {"node_id": "<paper id>"}} -- open a paper by '
                        'its id (from a search result or a reference list) to read its abstract, '
                        'its full reference list and the papers citing it')
    assert "short keyword query" not in GATED_ACTIONS_DOC


BIG = {"node_id": "p0", "text": "x" * 3000, "cites": [{"node_id": "r12", "title": "last ref"}]}


def test_default_policy_prompt_is_unchanged():
    from innovation.p2_forum.agent import ACTIONS_DOC
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'] * 2)
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"], identity="r:a")
    pol.act({"step": 0, "last_result": {}})
    pol.act({"step": 1, "last_result": BIG})
    expected = ("[agent r:a]\n\n" + ACTIONS_DOC + "\n\nRecent history (oldest first):\n"
                + "(none) -> {}\n" + "sample_frontier -> " + json.dumps(BIG)[:1500]
                + "\n\nChoose your next action (JSON only):")
    assert llm.prompts[1][1] == expected


def test_gated_policy_shows_the_latest_result_in_full():
    from innovation.p2_forum.agent import GATED_ACTIONS_DOC, GATED_SYSTEM
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'] * 2)
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"], system_template=GATED_SYSTEM,
                           actions_doc=GATED_ACTIONS_DOC, latest_result_chars=20000)
    pol.act({"step": 0, "last_result": {}})
    pol.act({"step": 1, "last_result": BIG})
    user = llm.prompts[1][1]
    assert "Latest result (full):\n" + json.dumps(BIG) in user
    assert user.index("Recent history") < user.index("Latest result (full)")
    assert user.endswith("\n\nChoose your next action (JSON only):")
    assert pol.memory[-1] == ("sample_frontier", json.dumps(BIG)[:1500])   # history stays truncated


def test_latest_result_is_capped():
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"], latest_result_chars=100)
    pol.act({"step": 0, "last_result": BIG})
    assert "Latest result (full):\n" + json.dumps(BIG)[:100] + "…(truncated)\n\n" in llm.prompts[0][1]


def test_latest_result_within_the_cap_has_no_marker():
    llm = ScriptedLLM(['{"action": "sample_frontier", "args": {}}'])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=["t"], latest_result_chars=20000)
    pol.act({"step": 0, "last_result": BIG})
    assert "truncated" not in llm.prompts[0][1]


def test_gated_system_describes_search_hits_and_opened_papers():
    from innovation.p2_forum.agent import FORUM_SYSTEM, GATED_SYSTEM
    assert ("Search results show each paper's title, an abstract snippet, year, venue and topics. "
            "Opening a paper (browse) shows its abstract, topics, full reference list and the "
            "papers citing it.") in " ".join(GATED_SYSTEM.split())
    assert "Each search or read shows" not in GATED_SYSTEM
    assert "Each search or read shows" not in FORUM_SYSTEM and "fixed network" in FORUM_SYSTEM


def test_gated_doc_offers_related_after_browse_and_old_mode_does_not():
    from innovation.p2_forum.agent import ACTIONS_DOC, GATED_ACTIONS_DOC, VALID_ACTIONS
    lines = GATED_ACTIONS_DOC.splitlines()
    assert lines[3] == ('{"action": "related", "args": {"node_id": "<paper id>", "k": 10}} -- '
                        'list papers related to a paper (like "Related articles" in Google Scholar)')
    assert "related" in VALID_ACTIONS and "related" not in ACTIONS_DOC


def test_region_fallback_prompt_states_the_hard_rule_without_describing_the_area():
    """Runs recorded before topics existed (no topics in run_meta)."""
    from innovation.p2_forum.agent import REGION_ACTIONS_DOC, REGION_SYSTEM_NO_TOPICS
    llm = ScriptedLLM(['{"action": "related", "args": {"node_id": "p1"}}'])
    pol = ForumAgentPolicy(llm=llm, model="m", topics=[],
                           system_template=REGION_SYSTEM_NO_TOPICS,
                           actions_doc=REGION_ACTIONS_DOC)
    flat = " ".join(pol.system.split())
    assert pol.system == REGION_SYSTEM_NO_TOPICS             # nothing is filled in
    assert ("LITERATURE is your reference library: a fixed collection of published papers "
            "from top AI venues (2020-2024)") in flat
    assert ("You can only find, read and cite within your own research area; papers and "
            "other researchers' posts outside it are hidden from you. You may publish any idea, but you can "
            "cite only what you can read.") in flat
    for leak in ("topic", "{", "seed", "radius", "coverage", "Your area"):
        assert leak not in pol.system
    assert pol.act({"step": 0, "last_result": {}}).name == "related"
    assert REGION_ACTIONS_DOC in llm.prompts[0][1]


def test_region_actions_doc_lists_the_ten_actions_under_three_headings():
    from innovation.p2_forum.agent import REGION_ACTIONS_DOC
    lines = REGION_ACTIONS_DOC.splitlines()
    groups, current = {}, None
    for l in lines[1:-1]:
        m = re.match(r'\{"action": "(\w+)"', l)
        if m:
            groups[current].append(m.group(1))
        else:
            current = l
            groups[current] = []
    assert groups == {
        "Literature (your reference library):": ["search", "browse", "related", "sample_frontier"],
        "Board (your group's research results):": ["search_board", "browse_board", "sample_board"],
        "Writing:": ["generate", "add_links", "remove_links"]}
    assert sorted(sum(groups.values(), [])) == sorted(VALID_ACTIONS)
    assert '"random"' not in REGION_ACTIONS_DOC and "post_page" not in REGION_ACTIONS_DOC
    search = next(l for l in lines if '"search"' in l)
    assert '"query": "<a paper title or a short meaningful phrase>", "page": 1' in search
    assert "do not paste lists of keywords" in search
    board = next(l for l in lines if '"search_board"' in l)
    assert "your colleagues' and your own posts" in board
    assert "do not paste lists of keywords" in board
    browse = next(l for l in lines if '"browse"' in l)
    assert "references" in browse and "citing" in browse
    assert '"ref_page": 1, "cited_by_page": 1' in browse
    browse_board = next(l for l in lines if '"browse_board"' in l)
    assert "who posted it" in browse_board and '"ref_page": 1, "cited_by_page": 1' in browse_board
    assert "10 results per page, most relevant first" in lines[-1]
    assert "high/medium/low relevance" in lines[-1]
    assert '"k"' not in REGION_ACTIONS_DOC
    generate = next(l for l in lines if '"generate"' in l)
    assert "3-4 sentence" in generate and "post your new idea to the board" in generate
    assert "topic" not in REGION_ACTIONS_DOC


def test_region_prompt_gives_literature_and_board_distinct_roles():
    from innovation.p2_forum.agent import REGION_SYSTEM, REGION_SYSTEM_NO_TOPICS
    pol = ForumAgentPolicy(llm=ScriptedLLM([]), model="m", topics=["T1: d1", "T2: d2"],
                           system_template=REGION_SYSTEM)
    flat = " ".join(pol.system.split())
    assert flat.startswith("You are a research agent working in a group of researchers.")
    assert ("The LITERATURE is your reference library: a fixed collection of published papers "
            "from top AI venues (2020-2024)") in flat
    assert "Look things up in it: background, methods, results and open problems" in flat
    assert "The BOARD is where your group exchanges research results." in flat
    assert ("everyone reads, builds on and cites each other's results, as a research community "
            "does with new papers. It keeps growing as the group works.") in flat
    assert "opening a post (`browse_board`) shows its full text, who posted it" in flat
    assert ("use the literature to find useful material, and follow the results your colleagues "
            "publish on the board: check the board regularly, especially before you write an idea") in flat
    assert ("Searches of the literature and of the board return only papers and posts in "
            "these topics; papers and other researchers' posts outside them are hidden from "
            "you. You may publish any idea, but you can cite only what you can read.") in flat
    assert "- T1: d1\n- T2: d2" in pol.system
    for p in (REGION_SYSTEM, REGION_SYSTEM_NO_TOPICS):
        assert "where your group exchanges research results" in p
        assert "starts empty" not in p and "One search covers both" not in p
    no_topics = " ".join(REGION_SYSTEM_NO_TOPICS.split())
    assert "papers and other researchers' posts outside it are hidden from you" in no_topics


def test_old_prompts_do_not_mention_regions():
    from innovation.p2_forum.agent import ACTIONS_DOC, FORUM_SYSTEM, GATED_ACTIONS_DOC, GATED_SYSTEM
    for p in (ACTIONS_DOC, FORUM_SYSTEM, GATED_ACTIONS_DOC, GATED_SYSTEM):
        assert "research area" not in p


def test_default_history_is_the_raw_result_cut_at_1500_characters():
    """Corpus and online modes: history entries are byte-identical to before."""
    pol = ForumAgentPolicy(llm=ScriptedLLM([]), model="m", topics=["t"])
    big = {"hits": [{"node_id": f"n{i}", "text": "x" * 300, "notice": "y"} for i in range(9)],
           "notice": "z"}
    assert pol.history_entry(big) == json.dumps(big)[:1500]


def test_compact_history_keeps_ids_and_relevance_and_drops_snippets():
    from innovation.p2_forum.agent import compact_result
    browse = {"node_id": "p1", "store": "corpus", "title": "T", "text": "a" * 900,
              "year": 2021, "venue": "ICML",
              "cites": [{"node_id": "p2", "title": "U", "year": 2020, "relevance": "low"}],
              "cites_page": 1, "cites_total": 1, "cites_pages": 1, "cited_by": [],
              "filtered": {"region": 2}}
    c = compact_result(browse)
    assert c["text"] == "a" * 300 and c["year"] == 2021 and c["venue"] == "ICML"
    assert c["cites"] == [{"node_id": "p2", "title": "U", "relevance": "low"}]
    assert c["filtered"] == browse["filtered"] and c["cites_total"] == 1
    post = {"node_id": "gen:r:3", "store": "board", "author": "b", "text": "b" * 900,
            "cites": [{"node_id": "gen:r:1", "store": "board", "author": "you",
                       "text": "c" * 200, "relevance": "high"},
                      {"node_id": "p2", "store": "corpus", "title": "U", "year": 2020,
                       "relevance": "low"}],
            "cited_by": []}
    c = compact_result(post)
    assert c["author"] == "b" and c["text"] == "b" * 300
    assert c["cites"] == [{"node_id": "gen:r:1", "store": "board", "author": "you",
                           "relevance": "high", "text": "c" * 120},
                          {"node_id": "p2", "store": "corpus", "title": "U",
                           "relevance": "low"}]
    assert compact_result({"error": "x", "gate": "result"}) == {"error": "x", "gate": "result"}


def test_region_history_keeps_every_id_of_full_length_board_and_literature_searches():
    """A search_board result with 5 full-length posts and a search with 10
    papers keep all their ids in the history, within REGION_HISTORY_CHARS."""
    from innovation.p2_forum.region_env import BOARD_NOTICE, SEARCH_NOTICE
    from innovation.p2_forum.runner import REGION_HISTORY_CHARS
    pol = ForumAgentPolicy(llm=ScriptedLLM([]), model="m", topics=["t"],
                           compact_history=True, history_chars=REGION_HISTORY_CHARS)
    board = {"page": 1, "total": 5, "pages": 1,
             "hits": [{"node_id": f"gen:forum-region-c20-s0d:{100 + i}", "store": "board",
                       "author": f"agent-{i:02d}", "text": "w" * 300, "relevance": "high"}
                      for i in range(5)],
             "filtered": {"region": 7}, "notice": BOARD_NOTICE}
    lit = {"page": 1, "total": 812, "pages": 82,
           "hits": [{"node_id": f"{i:040x}", "store": "corpus",
                     "title": "A Long Paper Title About Federated Optimization Under "
                              "Heterogeneous Client Drift And Privacy " + str(i),
                     "text": "x" * 300, "year": 2022, "venue": "NeurIPS",
                     "relevance": "medium"} for i in range(10)],
           "notice": SEARCH_NOTICE}
    for result in (board, lit):
        entry = pol.history_entry(result)
        assert len(entry) < REGION_HISTORY_CHARS
        assert json.loads(entry)["hits"]                        # not cut: valid JSON
        assert all(h["node_id"] in entry for h in result["hits"])
        assert "notice" not in entry
    board_entry = json.loads(pol.history_entry(board))
    assert all(h["author"] and len(h["text"]) == 120 for h in board_entry["hits"])
