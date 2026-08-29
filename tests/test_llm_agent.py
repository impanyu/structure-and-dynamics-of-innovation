import json

from innovation.agents.llm_agent import LLMAgentPolicy
from innovation.llm import FakeLLM


def test_parses_json_action_from_reply():
    reply = 'Thinking... {"action": "search", "args": {"query": "sparse attention", "k": 3}}'
    pol = LLMAgentPolicy(llm=FakeLLM(responses=[reply]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "search"
    assert action.args == {"query": "sparse attention", "k": 3}


def test_falls_back_to_sample_frontier_on_garbage():
    pol = LLMAgentPolicy(llm=FakeLLM(responses=["I refuse to answer with JSON"]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "sample_frontier"
    pol2 = LLMAgentPolicy(llm=FakeLLM(
        responses=[json.dumps({"action": "hack_the_planet", "args": {}})]), model="m")
    assert pol2.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_memory_window_appears_in_prompt_and_is_bounded():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m", memory_size=2)
    pol.act({"step": 0, "last_result": {}})
    pol.act({"step": 1, "last_result": {"node_id": "W7", "text": "old idea"}})
    pol.act({"step": 2, "last_result": {"node_id": "W8", "text": "newer idea"}})
    prompt = llm.calls[-1]["user"]
    assert "W8" in prompt
    # memory_size=2: the step-0 empty result is beyond the window by call 3
    assert len(pol.memory) == 2


def test_persona_is_added_to_system_prompt():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m", persona="You are a risk-taking theorist.")
    pol.act({"step": 0, "last_result": {}})
    assert "risk-taking theorist" in llm.calls[0]["system"]


def test_parses_add_links_action():
    reply = json.dumps({"action": "add_links",
                        "args": {"src_id": "W1", "dst_ids": ["W2"]}})
    pol = LLMAgentPolicy(llm=FakeLLM(responses=[reply]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "add_links"
    assert action.args["dst_ids"] == ["W2"]


def test_memory_is_fifo_with_default_20():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m")
    assert pol.memory.maxlen == 20
    for i in range(25):
        pol.act({"step": i, "last_result": {"n": i}})
    assert len(pol.memory) == 20
    # FIFO: the oldest surviving entry carries the result from call 5
    assert '"n": 5' in pol.memory[0][1]


def test_malformed_json_falls_back():
    pol = LLMAgentPolicy(llm=FakeLLM(responses=["{not valid json}"]), model="m")
    assert pol.act({"step": 0, "last_result": {}}).name == "sample_frontier"


def test_parses_remove_links_action():
    reply = json.dumps({"action": "remove_links",
                        "args": {"src_id": "W1", "dst_ids": ["W2"]}})
    pol = LLMAgentPolicy(llm=FakeLLM(responses=[reply]), model="m")
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "remove_links"


def test_identity_makes_prompts_unique_across_agents():
    """Same memory state, different agents -> different prompts (prevents the
    disk cache from collapsing agent diversity at step 0)."""
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    a = LLMAgentPolicy(llm=llm, model="m", identity="run1:g0")
    b = LLMAgentPolicy(llm=llm, model="m", identity="run1:g1")
    a.act({"step": 0, "last_result": {}})
    b.act({"step": 0, "last_result": {}})
    p0, p1 = llm.calls[0]["user"], llm.calls[1]["user"]
    assert p0 != p1 and "run1:g0" in p0 and "run1:g1" in p1
    assert "step 0" in p0


def test_pace_header_shows_step_over_total():
    llm = FakeLLM(default=json.dumps({"action": "sample_frontier", "args": {}}))
    pol = LLMAgentPolicy(llm=llm, model="m", identity="r:g0", total_steps=400)
    pol.act({"step": 65, "last_result": {}})
    assert "step 65/400" in llm.calls[0]["user"]
