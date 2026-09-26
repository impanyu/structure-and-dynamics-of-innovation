import json

from innovation.p2_forum.agent import VALID_ACTIONS, ForumAgentPolicy


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, *, model, system, user, max_tokens):
        self.prompts.append((system, user))
        return self.replies.pop(0)


def test_every_action_name_is_available_to_the_agent():
    assert VALID_ACTIONS == {"search", "browse", "sample_frontier",
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
