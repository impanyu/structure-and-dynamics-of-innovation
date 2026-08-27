"""The navigating LLM agent: JSON tool-calling loop over the action space (spec §3.4)."""
import json

from innovation.agents.policy import Policy
from innovation.experiments.env import Action
from innovation.llm import LLM

VALID_ACTIONS = {"search", "browse", "sample_frontier", "generate"}

AGENT_SYSTEM = """You are a research agent exploring a network of research ideas \
distilled from published papers. Ideas cite the ideas they build on. Your goal is to \
find promising unexplored directions and, when you see one, contribute a genuinely \
new idea to the network. Prefer exploring (search, browse) until you understand a \
neighborhood well enough that your new idea is specific and well-grounded."""

ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- semantic search over all ideas
{"action": "browse", "args": {"node_id": "<id>"}} -- read an idea and its citation neighbors
{"action": "sample_frontier", "args": {}} -- jump to a random idea
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- add your new idea, citing the existing ideas it builds on"""


class LLMAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, memory_size: int = 6, persona: str = ""):
        self.llm = llm
        self.model = model
        self.memory_size = memory_size
        self.system = AGENT_SYSTEM + ("\n\n" + persona if persona else "")
        self.memory: list[str] = []  # rendered "(action -> result)" lines
        self._last_action: str = "(none)"

    def act(self, obs: dict) -> Action:
        result_snippet = json.dumps(obs.get("last_result", {}))[:1500]
        self.memory.append(f"step {obs['step'] - 1}: {self._last_action} -> {result_snippet}")
        self.memory = self.memory[-self.memory_size:]
        user = (ACTIONS_DOC + "\n\nRecent history:\n" + "\n".join(self.memory)
                + "\n\nChoose your next action (JSON only):")
        reply = self.llm.complete(model=self.model, system=self.system,
                                  user=user, max_tokens=600)
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
