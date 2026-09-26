"""Paper 2's environment: six navigation actions over two stores, three writes
that target the board (spec §4).

The environment filters nothing. Every agent may read every node in either
store; what an agent cares about is decided by the topics in its prompt, and a
result it does not want is a result it ignores.
"""
from dataclasses import dataclass

from innovation.core.action import Action  # re-exported: this module's public name

CHANNELS = ("search", "edges", "jump")
STORES = ("corpus", "board")


@dataclass(frozen=True)
class Navigation:
    """Which navigation channels are open, PER STORE (spec §4.1).

    Paper 1's three navigation-channel ablations — no search, no edges, no
    jumps — apply to each store independently here, from this one
    implementation. That is the point of the action table's symmetry, and it is
    what makes "ablate the board channel the paper is about while leaving the
    literature intact" expressible.

    The three channels, and what closing one means:

    - `search` refuses `search` / `search_board` outright. The agent is told,
      because a semantic index either answers or does not exist.
    - `edges` leaves `browse` / `browse_board` readable but returns no
      neighbours. This is the faithful analogue of paper 1's edge ablation
      (`run.init_edges: none`), which leaves the nodes in place and removes the
      links between them: the agent loses the channel, not the node's text.
      `run.init_edges: none` still removes the corpus's edges from the DATA;
      `navigation.corpus.edges: false` closes the channel over data that has
      them.
    - `jump` refuses `sample_frontier` / `sample_board`.

    These are environmental, not dispositional (§4.3): one board is shared, so
    a channel is open or closed for the whole run. They are therefore
    configured at the top level, never per agent — see
    ForumRunConfig.__post_init__.
    """
    corpus_search: bool = True
    corpus_edges: bool = True
    corpus_jump: bool = True
    board_search: bool = True
    board_edges: bool = True
    board_jump: bool = True

    @classmethod
    def from_config(cls, section: dict | None) -> "Navigation":
        """Build from a config's top-level `navigation:` section:

            navigation:
              search: false            # both stores (shorthand)
              corpus: {jump: false}    # one store, overrides the shorthand
              board:  {edges: false}
              board:  false            # per-store shorthand: close every
                                        # channel on this store at once
              board:  true             # per-store shorthand: open every
                                        # channel on this store (explicit,
                                        # matches the default)

        A per-store value must be a mapping OR a bare bool. The bool is not a
        default-y falsy/truthy check: `false` means "close every channel on
        this store", full stop — it is the whole-store analogue of Experiment
        3's per-channel ablation, and it must never be mistaken for "no
        overrides given" (a config that reads as an ablation but silently
        runs unablated is the failure mode this guards against). Anything
        else that isn't a mapping or a bool (a string, a list, `None`
        written explicitly as `board: null`, ...) is rejected by name rather
        than ignored.

        Unknown keys raise rather than being ignored, so a typo in an
        ablation config cannot quietly produce an unablated run."""
        _unset = object()
        section = dict(section or {})
        shared = {}
        for channel in CHANNELS:
            if channel in section:
                shared[channel] = bool(section.pop(channel))
        per_store = {}
        for store in STORES:
            block = section.pop(store, _unset)
            if block is _unset:
                block = {}
            elif isinstance(block, bool):
                # Per-store shorthand: set every channel on this store to the
                # same value. `false` must disable the whole store, not be
                # swallowed as "nothing to override".
                block = {channel: block for channel in CHANNELS}
            elif not isinstance(block, dict):
                raise ValueError(
                    f"navigation.{store} must be a mapping of "
                    f"{list(CHANNELS)} to booleans, or a single bool to set "
                    f"all of them at once; got {block!r}")
            unknown = sorted(set(block) - set(CHANNELS))
            if unknown:
                raise ValueError(
                    f"unknown navigation.{store} channel(s) {unknown}; "
                    f"known channels are {list(CHANNELS)}")
            per_store[store] = block
        if section:
            raise ValueError(
                f"unknown navigation key(s) {sorted(section)}; expected "
                f"{list(CHANNELS)} and/or {list(STORES)}")
        kwargs = {}
        for store in STORES:
            for channel in CHANNELS:
                kwargs[f"{store}_{channel}"] = bool(
                    per_store[store].get(channel, shared.get(channel, True)))
        return cls(**kwargs)

    def is_open(self, store: str, channel: str) -> bool:
        return bool(getattr(self, f"{store}_{channel}"))


class ForumEnvironment:
    def __init__(self, *, run_id, workspace, event_log, rng,
                 navigation: Navigation | None = None,
                 generation_budget: int | None = None):
        self.run_id = run_id
        self.ws = workspace
        self.event_log = event_log
        self.rng = rng
        self.nav = navigation or Navigation()
        self.generation_budget = generation_budget

    # --- entry point ---
    def execute(self, agent_id: str, step: int, action: Action) -> dict:
        handler = getattr(self, f"_do_{action.name}", None)
        if handler is None:
            result = {"error": f"unknown action: {action.name}"}
        else:
            try:
                result = handler(agent_id=agent_id, step=step, **action.args)
            except (KeyError, TypeError, ValueError) as exc:
                result = {"error": str(exc)}
        self.event_log.append({"run_id": self.run_id, "agent_id": agent_id,
                               "step": step, "action": action.name,
                               "args": action.args, "result": result})
        return result

    def _meta(self, agent_id, step) -> dict:
        return {"run_id": self.run_id, "agent_id": agent_id, "step": step}

    def _hits(self, raw, store: str) -> dict:
        return {"hits": [{"node_id": nid, "store": store,
                          "text": self.ws.node(nid).text[:300], "score": score}
                         for nid, score in raw]}

    def _view(self, node_id: str, store: str, out_ids, in_ids) -> dict:
        def preview(nid):
            return {"node_id": nid, "store": self.ws.store_of(nid),
                    "text": self.ws.node(nid).text[:200]}
        return {"node_id": node_id, "store": store,
                "text": self.ws.node(node_id).text,
                "cites": [preview(n) for n in out_ids[:10]],
                "cited_by": [preview(n) for n in in_ids[:10]]}

    # --- corpus navigation ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.corpus_search:
            return {"error": "semantic search over the literature is closed"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.corpus_search(vec, k=k), "corpus")

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "corpus" or not self.ws.corpus.has_node(node_id):
            return {"error": f"{node_id} is not a corpus node"}
        if not self.nav.corpus_edges:
            return self._view(node_id, "corpus", [], [])
        out_ids, in_ids = self.ws.corpus_neighbors(node_id)
        return self._view(node_id, "corpus", out_ids, in_ids)

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.nav.corpus_jump:
            return {"error": "random jumps into the literature are closed"}
        nid = self.ws.corpus_sample(self.rng)
        return {"node_id": nid, "store": "corpus", "text": self.ws.node(nid).text}

    # --- board navigation ---
    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.board_search:
            return {"error": "semantic search over the board is closed"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.board_search(vec, k=k), "board")

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        if not self.nav.board_edges:
            return self._view(node_id, "board", [], [])
        out_ids, in_ids = self.ws.board_neighbors(node_id)
        return self._view(node_id, "board", out_ids, in_ids)

    def _do_sample_board(self, *, agent_id, step) -> dict:
        if not self.nav.board_jump:
            return {"error": "random jumps into the board are closed"}
        nid = self.ws.board_sample(self.rng)
        if nid is None:
            return {"error": "the board is empty"}
        return {"node_id": nid, "store": "board", "text": self.ws.node(nid).text}

    # --- writes ---
    def _do_generate(self, *, agent_id, step, text: str,
                     cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        node_id = self.ws.post_idea(text, cited_ids, meta=self._meta(agent_id, step))
        if self.generation_budget is not None:
            self.generation_budget -= 1
        return {"node_id": node_id}

    def _do_add_links(self, *, agent_id, step, src_id: str,
                      dst_ids: list[str]) -> dict:
        return self.ws.add_links(src_id, dst_ids, meta=self._meta(agent_id, step))

    def _do_remove_links(self, *, agent_id, step, src_id: str,
                         dst_ids: list[str]) -> dict:
        return self.ws.remove_links(src_id, dst_ids)

    # --- replay ---
    def restore(self, events: list[dict]) -> None:
        """Rebuild the board from the event log; no logging. Events replay in
        order, so a post citing an earlier post resolves."""
        for e in events:
            result = e.get("result", {})
            if e["action"] == "generate" and "node_id" in result:
                self.ws.post_idea(e["args"]["text"], e["args"]["cited_ids"],
                                  meta={"run_id": e["run_id"],
                                        "agent_id": e["agent_id"],
                                        "step": e["step"]},
                                  node_id=result["node_id"])
                if self.generation_budget is not None:
                    self.generation_budget -= 1
            elif e["action"] == "add_links" and "added" in result:
                self.ws.add_links(e["args"]["src_id"], result["added"],
                                  meta={"run_id": e["run_id"],
                                        "agent_id": e["agent_id"],
                                        "step": e["step"]})
            elif e["action"] == "remove_links" and "removed" in result:
                self.ws.remove_links(e["args"]["src_id"],
                                     [r["dst_id"] for r in result["removed"]])

    def generated_ids(self) -> list[str]:
        return self.ws.board_post_ids()
