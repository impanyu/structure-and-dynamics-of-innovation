"""Paper 2's environment: six navigation actions over two stores, three writes
that target the board (spec §4).

The environment filters nothing. Every agent may read every node in either
store; what an agent cares about is decided by the topics in its prompt, and a
result it does not want is a result it ignores.
"""
from innovation.core.action import Action  # re-exported: this module's public name


class ForumEnvironment:
    def __init__(self, *, run_id, workspace, event_log, rng,
                 allow_jump: bool = True, allow_search: bool = True,
                 generation_budget: int | None = None):
        self.run_id = run_id
        self.ws = workspace
        self.event_log = event_log
        self.rng = rng
        self.allow_jump = allow_jump
        self.allow_search = allow_search
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
        if not self.allow_search:
            return {"error": "semantic search is not allowed for this agent"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.corpus_search(vec, k=k), "corpus")

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "corpus" or not self.ws.corpus.has_node(node_id):
            return {"error": f"{node_id} is not a corpus node"}
        out_ids, in_ids = self.ws.corpus_neighbors(node_id)
        return self._view(node_id, "corpus", out_ids, in_ids)

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.allow_jump:
            return {"error": "random jump is not allowed for this agent"}
        nid = self.ws.corpus_sample(self.rng)
        return {"node_id": nid, "store": "corpus", "text": self.ws.node(nid).text}

    # --- board navigation ---
    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.allow_search:
            return {"error": "semantic search is not allowed for this agent"}
        vec = self.ws.embedder.encode([query])[0]
        return self._hits(self.ws.board_search(vec, k=k), "board")

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        out_ids, in_ids = self.ws.board_neighbors(node_id)
        return self._view(node_id, "board", out_ids, in_ids)

    def _do_sample_board(self, *, agent_id, step) -> dict:
        if not self.allow_jump:
            return {"error": "random jump is not allowed for this agent"}
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
