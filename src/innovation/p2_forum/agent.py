"""Paper 2's agent: the same JSON tool-calling loop as paper 1, over two
stores instead of one.

Specialization is soft in corpus mode and hard in online mode (gated_env.py)
and region mode (region_env.py).
In soft mode, the agent's topics appear only in its system prompt. Nothing in
the environment enforces them: the agent sees every result and decides for
itself what is worth following. In hard mode, the environment gates all results
(papers and posts): in online mode to those matching the agent's topics, in
region mode to the agent's semantic region. Region mode combines both: the
prompt describes the region as a list of topics (k-means clusters of its papers,
named by an LLM; region_topics.py) and the gate enforces the region itself.
Region mode also reads both stores with one tool set (R6): search / related
return papers and posts in separate sections, browse opens either kind.
"""
import json
from collections import deque

from innovation.core.llm import LLM
from innovation.core.policy import Policy
from innovation.p2_forum.env import Action

# "random" is region mode's jump (R6); there the four pre-R6 board/jump names
# stay accepted as aliases. A reply naming anything else falls back to
# sample_frontier, a valid jump in every mode.
VALID_ACTIONS = {"search", "browse", "related", "random", "sample_frontier",
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

GATED_SYSTEM = """You are a research agent. Two things are in front of you.

The LITERATURE is published research you can search online: papers from top AI \
venues, or highly cited papers, up to September 2024. Search results show each \
paper's title, an abstract snippet, year, venue and topics. Opening a paper \
(browse) shows its abstract, topics, full reference list and the papers citing it.

The BOARD is a shared space where you and other agents publish new ideas. Anyone \
may adjust the reference links on any post. It starts empty.

Your goal is to find promising unexplored directions and publish genuinely new \
ideas to the board. Ground them: cite the papers they build on, and cite other \
agents' posts when your idea builds on theirs.

You work ONLY within your topics. Searches outside them are refused, papers and \
posts outside them are hidden from you, and an idea outside them will not be \
published.

Your topics:
{topics}"""

GATED_ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<the title of a specific paper, or a short meaningful phrase as you would type into Google Scholar>", "k": 5}} -- search the published literature (do not paste lists of keywords)
{"action": "browse", "args": {"node_id": "<paper id>"}} -- open a paper by its id (from a search result or a reference list) to read its abstract, its full reference list and the papers citing it
{"action": "related", "args": {"node_id": "<paper id>", "k": 10}} -- list papers related to a paper (like "Related articles" in Google Scholar)
{"action": "sample_frontier", "args": {}} -- jump to a random paper in one of your topics
{"action": "search_board", "args": {"query": "<text>", "k": 5}} -- semantic search over the board
{"action": "browse_board", "args": {"node_id": "<post id>"}} -- read a post and its reference neighbors
{"action": "sample_board", "args": {}} -- jump to a random post
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- publish your new idea to the board, citing what it builds on (papers or posts)
{"action": "add_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- add reference links from a post to what it builds on
{"action": "remove_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- remove reference links from a post that do not actually support it"""

# Region mode (region_env.py): the prompt describes the agent's area as its
# region's topics (region_topics.py, recorded in run_meta) and states the rule
# the gate enforces. REGION_SYSTEM_NO_TOPICS is the fallback for runs recorded
# before topics existed: it states the rule without describing the area.
# Since R6 (2026-10-08) the literature and the board are described in parallel
# and read with one tool set (search / browse / related / random) whose
# results list papers and posts in separate sections; following peers' posts
# is part of the goal, on a par with reading the literature.
_REGION_INTRO = """You are a research agent. Two things are in front of you.

The LITERATURE is a fixed collection of published papers from top AI venues \
(2020-2024), each citing the papers it builds on. You can read it but never \
change it.

The BOARD is a growing collection of new research ideas posted by you and other \
researchers working alongside you, each a short paragraph citing the papers and \
posts it builds on. Anyone may adjust a post's reference links.

One search covers both: it returns matching papers and matching posts in \
separate lists. Opening an item (browse) shows a paper's abstract or a post's \
full text, what it cites, and what cites it, including posts that cite a paper.

Your goal is to find promising unexplored directions and publish genuinely new \
ideas to the board. Like a real researcher, keep up with both the published \
literature and the new ideas your peers post; build on either, and cite the \
papers and posts your idea builds on.

"""

REGION_SYSTEM = _REGION_INTRO + """Your research area is defined by the following \
topics (each a cluster of papers you can read):
{topics}

You can only find, read and cite within these topics. Searches return only \
papers and posts in these topics; papers and other researchers' posts outside \
them are hidden from you. You may publish any idea, but you can cite only what you can read."""

REGION_SYSTEM_NO_TOPICS = _REGION_INTRO + """You can only find, read and cite \
within your own research area; papers and other researchers' posts outside it \
are hidden from you. You may publish any idea, but you can cite only what you can read."""

REGION_ACTIONS_DOC = """Available actions (reply with EXACTLY one JSON object, nothing else):
{"action": "search", "args": {"query": "<a paper title or a short meaningful phrase>", "page": 1, "post_page": 1}} -- semantic search over papers and posts (do not paste lists of keywords); returns two lists: "papers" (10 per page, `page`) and "posts" (5 per page, `post_page`)
{"action": "browse", "args": {"node_id": "<paper or post id>", "ref_page": 1, "cited_by_page": 1, "post_page": 1}} -- open a paper or a post by its id (from any result or reference list): a paper's abstract or a post's full text, what it cites (`ref_page`) and what cites it (`cited_by_page`); for a paper also the posts citing it (`post_page`)
{"action": "related", "args": {"node_id": "<paper or post id>", "page": 1, "post_page": 1}} -- the papers and posts most similar to a paper or post, in the same two lists as search
{"action": "random", "args": {"kind": "paper"}} -- jump to a random paper ("kind": "paper") or a random post ("kind": "post")
{"action": "generate", "args": {"text": "<3-4 sentence new idea paragraph>", "cited_ids": ["<id>", ...]}} -- publish your new idea to the board, citing what it builds on (papers or posts)
{"action": "add_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- add reference links from a post to what it builds on
{"action": "remove_links", "args": {"src_id": "<post id>", "dst_ids": ["<id>", ...]}} -- remove reference links from a post that do not actually support it
Every item says its "kind" ("paper" or "post"); post ids start with "gen:". Every list is most relevant first and each item is tagged high/medium/low relevance; ask for a later page to see more. Search and related show 10 papers and 5 posts per page; browse lists show 10 per page."""


# Compact history (region mode, R6): with posts listed before papers, a
# 1500-character cut of a raw search result held ~3.5 posts and no papers. A
# history entry instead keeps, for every item of every list in the result,
# only what identifies it and how relevant it was (plus a post's opening);
# abstract snippets and the notice are dropped, and a browsed item's own text
# is cut. The newest result is still shown in full.
_HISTORY_ITEM_KEYS = ("node_id", "kind", "store", "title", "author", "relevance")
HISTORY_POST_CHARS = 120
HISTORY_OWN_TEXT_CHARS = 300


def _compact_item(item: dict) -> dict:
    out = {k: item[k] for k in _HISTORY_ITEM_KEYS if k in item}
    if (item.get("kind") == "post" or item.get("store") == "board") and "text" in item:
        out["text"] = str(item["text"])[:HISTORY_POST_CHARS]
    return out


def compact_result(result, top: bool = True):
    """The history form of a result: lists of items (dicts with a node_id)
    are compacted item by item, nested sections recursively; the notice is
    dropped and the top-level (browsed item's) text cut."""
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        if k == "notice":
            continue
        if top and k == "text" and isinstance(v, str):
            out[k] = v[:HISTORY_OWN_TEXT_CHARS]
        elif isinstance(v, list) and v and all(isinstance(x, dict) and "node_id" in x
                                               for x in v):
            out[k] = [_compact_item(x) for x in v]
        elif isinstance(v, dict):
            out[k] = compact_result(v, top=False)
        else:
            out[k] = v
    return out


class ForumAgentPolicy(Policy):
    def __init__(self, *, llm: LLM, model: str, topics: list[str],
                 memory_size: int = 20, identity: str = "",
                 total_steps: int = 0,
                 system_template: str = FORUM_SYSTEM, actions_doc: str = ACTIONS_DOC,
                 latest_result_chars: int | None = None,
                 compact_history: bool = False, history_chars: int = 1500):
        self.llm = llm
        self.model = model
        self.topics = list(topics)
        # identity (run:agent) is embedded in every prompt so identical memory
        # states of DIFFERENT agents/runs never share a disk-cache entry.
        self.identity = identity
        self.total_steps = total_steps
        bullets = "\n".join(f"- {t}" for t in self.topics)
        self.system = system_template.format(topics=bullets)
        self.actions_doc = actions_doc
        # None: the newest result appears only in the (truncated) history.
        # An int: it is also shown in full, up to this many characters, so a
        # long read (abstract plus a full reference list) is not cut off.
        self.latest_result_chars = latest_result_chars
        # How a past result appears in the history: by default the raw JSON
        # cut to history_chars; with compact_history, compact_result() first.
        self.compact_history = compact_history
        self.history_chars = history_chars
        self.memory: deque[tuple[str, str]] = deque(maxlen=memory_size)
        self._last_action: str = "(none)"

    def history_entry(self, result: dict) -> str:
        """A past result as the history shows it (also used by resume)."""
        if self.compact_history:
            return json.dumps(compact_result(result),
                              separators=(",", ":"))[:self.history_chars]
        return json.dumps(result)[:self.history_chars]

    def act(self, obs: dict) -> Action:
        latest = json.dumps(obs.get("last_result", {}))
        self.memory.append((self._last_action, self.history_entry(obs.get("last_result", {}))))
        history = "\n".join(f"{a} -> {r}" for a, r in self.memory)
        header = f"[agent {self.identity}]\n\n" if self.identity else ""
        user = header + self.actions_doc + "\n\nRecent history (oldest first):\n" + history
        if self.latest_result_chars is not None:
            cut = latest[:self.latest_result_chars]
            if len(latest) > self.latest_result_chars:
                cut += "…(truncated)"
            user += "\n\nLatest result (full):\n" + cut
        user += "\n\nChoose your next action (JSON only):"
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
