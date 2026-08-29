"""The navigating LLM agent: JSON tool-calling loop over the action space (spec §3.4)."""
import json
from collections import deque

from innovation.agents.policy import Policy
from innovation.experiments.env import Action
from innovation.llm import LLM

VALID_ACTIONS = {"search", "browse", "sample_frontier", "generate",
                 "add_links", "remove_links"}

AGENT_SYSTEM = """You are a research agent exploring a network of research ideas \
distilled from published papers. Ideas cite the ideas they build on. Your goal is to \
find promising unexplored directions and, when you see one, contribute a genuinely \
new idea to the network. Prefer exploring (search, browse) until you understand a \
neighborhood well enough that your new idea is specific and well-grounded. Your \
header shows the step counter and the team's live idea budget (used/total): pace \
yourself so the team has (nearly) used the whole budget by the end of the run — \
do not hoard exploration. If you \
notice an idea clearly builds on another idea it does not yet reference, you may \
record that missing link. Some of your actions may be restricted to a region of \
the network; if an action returns a restriction error, adapt your strategy \
instead of repeating it."""

ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- semantic search over all ideas
{"action": "browse", "args": {"node_id": "<id>"}} -- read an idea and its citation neighbors
{"action": "sample_frontier", "args": {}} -- jump to a random idea
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- add your new idea, citing the existing ideas it builds on
{"action": "add_links", "args": {"src_id": "<id>", "dst_ids": ["<id>", ...]}} -- add missing reference links from one existing idea to ideas it builds on
{"action": "remove_links", "args": {"src_id": "<id>", "dst_ids": ["<id>", ...]}} -- remove reference links from an idea that do not actually support it"""


class LLMAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, memory_size: int = 20,
                 persona: str = "", identity: str = "", total_steps: int = 0):
        self.llm = llm
        self.model = model
        # identity (run:agent) is embedded in every prompt so identical memory
        # states of DIFFERENT agents/runs never share a disk-cache entry.
        self.identity = identity
        self.total_steps = total_steps
        self.system = AGENT_SYSTEM + ("\n\n" + persona if persona else "")
        # FIFO short-term memory of (action, result) pairs; oldest evicted first.
        self.memory: deque[tuple[str, str]] = deque(maxlen=memory_size)
        self._last_action: str = "(none)"

    def act(self, obs: dict) -> Action:
        result_snippet = json.dumps(obs.get("last_result", {}))[:1500]
        self.memory.append((self._last_action, result_snippet))
        history = "\n".join(f"{a} -> {r}" for a, r in self.memory)
        total = f"/{self.total_steps}" if self.total_steps else ""
        budget = ""
        if obs.get("ideas_total"):
            budget = f" | team ideas {obs.get('ideas_used', 0)}/{obs['ideas_total']}"
        header = (f"[agent {self.identity} | step {obs['step']}{total}{budget}]\n\n"
                  if self.identity or self.total_steps else "")
        user = (header + ACTIONS_DOC + "\n\nRecent history (oldest first):\n" + history
                + "\n\nChoose your next action (JSON only):")
        # 2000 visible-output budget: a generate action carries a full idea
        # paragraph (~250 tokens) and must survive even a long reasoning turn
        # (OpenAI reasoning models share one max_output_tokens cap; the
        # provider client adds its own reasoning headroom on top).
        reply = self.llm.complete(model=self.model, system=self.system,
                                  user=user, max_tokens=2000)
        action = self._parse(reply)
        self._last_action = action.name
        return action

    @staticmethod
    def _parse(reply: str) -> Action:
        start, end = reply.find("{"), reply.rfind("}")
        if start == -1 or end <= start:
            return Action("sample_frontier", {})
        try:
            obj = json.loads(reply[start:end + 1])
        except json.JSONDecodeError:
            return Action("sample_frontier", {})
        name = obj.get("action")
        if name not in VALID_ACTIONS or not isinstance(obj.get("args", {}), dict):
            return Action("sample_frontier", {})
        return Action(name, obj.get("args", {}))
