"""Paper 2's agent: the same JSON tool-calling loop as paper 1, over two
stores instead of one.

Specialization is soft (spec §4.3). The agent's topics appear only here, in
its system prompt. Nothing in the environment enforces them: the agent sees
every result and decides for itself what is worth following.
"""
import json
from collections import deque

from innovation.core.llm import LLM
from innovation.core.policy import Policy
from innovation.p2_forum.env import Action

VALID_ACTIONS = {"search", "browse", "sample_frontier",
                 "search_board", "browse_board", "sample_board",
                 "generate", "add_links", "remove_links"}

FORUM_SYSTEM = """You are a research agent. Two things are in front of you.

The LITERATURE is a fixed network of research ideas distilled from published \
papers, each citing the ideas it builds on. You can read it but never change it.

The BOARD is a shared space where you and other agents publish new ideas. \
Everything anyone posts is visible to everyone, and anyone may adjust the \
reference links on any post. The board starts empty and grows only from what \
the agents put there.

Your goal is to find promising unexplored directions and publish genuinely new \
ideas to the board. Ground them: cite the literature they build on, and cite \
other agents' posts when your idea builds on theirs. Prefer exploring until you \
understand a neighborhood well enough that your idea is specific.

You are interested in these topics:
{topics}

You may read anything in either the literature or the board. Most of what you \
find will be outside your interests — skim it and move on."""

ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<text>", "k": 5}} -- semantic search over the literature
{"action": "browse", "args": {"node_id": "<id>"}} -- read a paper-idea and its citation neighbors
{"action": "sample_frontier", "args": {}} -- jump to a random paper-idea
{"action": "search_board", "args": {"query": "<text>", "k": 5}} -- semantic search over the board
{"action": "browse_board", "args": {"node_id": "<id>"}} -- read a post and its reference neighbors
{"action": "sample_board", "args": {}} -- jump to a random post
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- publish your new idea to the board, citing what it builds on (papers or posts)
{"action": "add_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- add reference links from a post to what it builds on
{"action": "remove_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- remove reference links from a post that do not actually support it"""


class ForumAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, topics: list[str],
                 memory_size: int = 20, identity: str = "",
                 total_steps: int = 0):
        self.llm = llm
        self.model = model
        self.topics = list(topics)
        # identity (run:agent) is embedded in every prompt so identical memory
        # states of DIFFERENT agents/runs never share a disk-cache entry.
        self.identity = identity
        self.total_steps = total_steps
        bullets = "\n".join(f"- {t}" for t in self.topics)
        self.system = FORUM_SYSTEM.format(topics=bullets)
        self.memory: deque[tuple[str, str]] = deque(maxlen=memory_size)
        self._last_action: str = "(none)"

    def act(self, obs: dict) -> Action:
        result_snippet = json.dumps(obs.get("last_result", {}))[:1500]
        self.memory.append((self._last_action, result_snippet))
        history = "\n".join(f"{a} -> {r}" for a, r in self.memory)
        header = f"[agent {self.identity}]\n\n" if self.identity else ""
        user = (header + ACTIONS_DOC + "\n\nRecent history (oldest first):\n"
                + history + "\n\nChoose your next action (JSON only):")
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
