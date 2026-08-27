"""The simulation environment: executes agent actions on the living idea network (spec §3.4)."""
from dataclasses import dataclass, field


def _preview(graph, node_id: str) -> dict:
    return {"node_id": node_id, "text": graph.node(node_id).text[:200]}


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


class Environment:
    def __init__(self, *, run_id, graph, index, embedder, event_log, rng,
                 generation_budget: int):
        self.run_id = run_id
        self.graph = graph
        self.index = index
        self.embedder = embedder
        self.event_log = event_log
        self.rng = rng
        self.generation_budget = generation_budget
        self._gen_counter = 0

    # --- public entry point: execute + log ---
    def execute(self, agent_id: str, step: int, action: Action) -> dict:
        handler = getattr(self, f"_do_{action.name}", None)
        if handler is None:
            result = {"error": f"unknown action: {action.name}"}
        else:
            try:
                result = handler(agent_id=agent_id, step=step, **action.args)
            except (KeyError, TypeError) as exc:
                result = {"error": str(exc)}
        self.event_log.append({"run_id": self.run_id, "agent_id": agent_id,
                               "step": step, "action": action.name,
                               "args": action.args, "result": result})
        return result

    # --- actions (spec §3.4) ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        vec = self.embedder.encode([query])[0]
        hits = [{"node_id": nid, "text": self.graph.node(nid).text[:300],
                 "score": score} for nid, score in self.index.search(vec, k=k)]
        return {"hits": hits}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        node = self.graph.node(node_id)  # KeyError -> error result
        return {"node_id": node_id, "text": node.text,
                "cites": [_preview(self.graph, n) for n in self.graph.citations_out(node_id)[:10]],
                "cited_by": [_preview(self.graph, n) for n in self.graph.citations_in(node_id)[:10]]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        node_id = str(self.rng.choice(self.graph.node_ids()))
        return _preview(self.graph, node_id) | {"text": self.graph.node(node_id).text}

    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        node_id = self._apply_generate(text, cited_ids,
                                       meta={"run_id": self.run_id,
                                             "agent_id": agent_id, "step": step})
        self.generation_budget -= 1
        return {"node_id": node_id}

    def _do_add_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self.graph.add_links(src_id, dst_ids,  # KeyError -> error result
                                    meta={"run_id": self.run_id,
                                          "agent_id": agent_id, "step": step})

    def _do_remove_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self.graph.remove_links(src_id, dst_ids)  # KeyError -> error result

    # --- shared by generate and restore ---
    def _apply_generate(self, text: str, cited_ids: list[str], meta: dict) -> str:
        node_id = f"gen:{self.run_id}:{self._gen_counter}"
        self.graph.add_idea(node_id, text, cited_ids, meta=meta)  # KeyError propagates
        self.index.add([node_id], self.embedder.encode([text]))
        self._gen_counter += 1
        return node_id

    def restore(self, events: list[dict]) -> None:
        """Replay graph-mutating events (generate, add_links) to rebuild state
        (spec §3.3); no logging. Events replay in order, so links to nodes
        generated earlier in the trace resolve correctly."""
        for e in events:
            if e["action"] == "generate" and "node_id" in e.get("result", {}):
                self._apply_generate(e["args"]["text"], e["args"]["cited_ids"],
                                     meta={"run_id": e["run_id"],
                                           "agent_id": e["agent_id"],
                                           "step": e["step"]})
                self.generation_budget -= 1
            elif e["action"] == "add_links" and "added" in e.get("result", {}):
                self.graph.add_links(e["args"]["src_id"], e["result"]["added"],
                                     meta={"run_id": e["run_id"],
                                           "agent_id": e["agent_id"],
                                           "step": e["step"]})
            elif e["action"] == "remove_links" and "removed" in e.get("result", {}):
                self.graph.remove_links(
                    e["args"]["src_id"],
                    [r["dst_id"] for r in e["result"]["removed"]])

    def generated_ids(self) -> list[str]:
        return self.graph.node_ids(source="generated")
