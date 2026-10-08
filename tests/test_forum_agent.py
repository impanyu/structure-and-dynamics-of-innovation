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
    assert ("LITERATURE is a fixed collection of published papers from top AI venues "
            "(2020-2024)") in flat
    assert ("You can only find, read and cite within your own research area; papers and "
            "posts outside it are hidden from you. You may publish any idea, but you can "
            "cite only what you can read.") in flat
    for leak in ("topic", "{", "seed", "radius", "coverage", "Your area"):
        assert leak not in pol.system
    assert pol.act({"step": 0, "last_result": {}}).name == "related"
    assert REGION_ACTIONS_DOC in llm.prompts[0][1]


def test_region_actions_doc_lists_every_action_once():
    from innovation.p2_forum.agent import REGION_ACTIONS_DOC
    lines = REGION_ACTIONS_DOC.splitlines()
    names = [re.match(r'\{"action": "(\w+)"', l).group(1) for l in lines[1:-1]]
    assert sorted(names) == sorted(VALID_ACTIONS) and len(names) == len(set(names))
    assert '"query": "<a paper title or a short meaningful phrase>", "page": 1' in lines[1]
    browse = next(l for l in lines if '"browse"' in l)
    assert "reference list" in browse and "citing" in browse
    assert '"ref_page": 1, "cited_by_page": 1' in browse
    assert "10 results per page, most relevant first" in lines[-1]
    assert "high/medium/low relevance" in lines[-1]
    assert '"k"' not in REGION_ACTIONS_DOC
    generate = next(l for l in lines if '"generate"' in l)
    assert "3-4 sentence" in generate
    assert "topic" not in REGION_ACTIONS_DOC


def test_old_prompts_do_not_mention_regions():
    from innovation.p2_forum.agent import ACTIONS_DOC, FORUM_SYSTEM, GATED_ACTIONS_DOC, GATED_SYSTEM
    for p in (ACTIONS_DOC, FORUM_SYSTEM, GATED_ACTIONS_DOC, GATED_SYSTEM):
        assert "research area" not in p
