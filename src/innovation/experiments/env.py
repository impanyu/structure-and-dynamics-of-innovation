"""The simulation environment: executes agent actions on the living idea network (spec §3.4)."""
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


def _preview(graph, node_id: str) -> dict:
    return {"node_id": node_id, "text": graph.node(node_id).text[:200]}


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)


@dataclass
class AgentScope:
    """Behavioral constraints modeling researcher profiles (spec §3.4-3.5).
    Each of read/write is defined by ONE mechanism: a set of community ids
    (read/write) or a semantic region (topic anchor embeddings + cosine-distance
    radii — a scalar, or a per-anchor array for equal-mass regions). None
    everywhere means unrestricted."""
    read: set | None = None            # community ids
    write: set | None = None
    allow_jump: bool = True
    read_anchors: object | None = None   # (m, d) topic embeddings
    read_radius: object = 0.0            # float or (m,) array
    write_anchors: object | None = None
    write_radius: object = 0.0


def _within_region(anchors, radii, v) -> bool:
    """Inside the region iff within radius of ANY anchor (radii broadcastable)."""
    sims = anchors @ v
    return bool(np.any(sims >= 1.0 - np.asarray(radii)))


class Environment:
    def __init__(self, *, run_id, graph, index, embedder, event_log, rng,
                 generation_budget: int | None = None, scopes: dict | None = None,
                 communities: dict | None = None):
        self.run_id = run_id
        self.graph = graph
        self.index = index
        self.embedder = embedder
        self.event_log = event_log
        self.rng = rng
        self.generation_budget = generation_budget
        self.scopes = scopes or {}          # agent_id -> AgentScope
        self.community_of = dict(communities or {})  # node_id -> community id
        self._gen_counter = 0

    # --- scope checks ---
    def _in_semantic_region(self, anchors, radius, node_id: str) -> bool:
        v = self.index.vec(node_id)
        if v is None:
            return False
        return _within_region(anchors, radius, v)

    def _readable(self, agent_id: str, node_id: str) -> bool:
        scope = self.scopes.get(agent_id)
        if scope is None:
            return True
        if scope.read is not None:
            return self.community_of.get(node_id) in scope.read
        if scope.read_anchors is not None:
            return self._in_semantic_region(scope.read_anchors,
                                            scope.read_radius, node_id)
        return True

    def _writable(self, agent_id: str, node_id: str) -> bool:
        scope = self.scopes.get(agent_id)
        if scope is None:
            return True
        if scope.write is not None:
            return self.community_of.get(node_id) in scope.write
        if scope.write_anchors is not None:
            return self._in_semantic_region(scope.write_anchors,
                                            scope.write_radius, node_id)
        return True

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
        raw = self.index.search(vec, k=k * 4)  # over-fetch, then scope-filter
        hits = [{"node_id": nid, "text": self.graph.node(nid).text[:300],
                 "score": score} for nid, score in raw
                if self._readable(agent_id, nid)][:k]
        return {"hits": hits}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        if not self._readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your readable region"}
        node = self.graph.node(node_id)  # KeyError -> error result
        cites = [n for n in self.graph.citations_out(node_id)
                 if self._readable(agent_id, n)][:10]
        cited_by = [n for n in self.graph.citations_in(node_id)
                    if self._readable(agent_id, n)][:10]
        return {"node_id": node_id, "text": node.text,
                "cites": [_preview(self.graph, n) for n in cites],
                "cited_by": [_preview(self.graph, n) for n in cited_by]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        scope = self.scopes.get(agent_id)
        if scope is not None and not scope.allow_jump:
            return {"error": "random jump is not allowed for this agent"}
        pool = [n for n in self.graph.node_ids() if self._readable(agent_id, n)]
        if not pool:
            return {"error": "no readable nodes to sample"}
        node_id = str(self.rng.choice(pool))
        return _preview(self.graph, node_id) | {"text": self.graph.node(node_id).text}

    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        scope = self.scopes.get(agent_id)
        kept, dropped = list(cited_ids), []
        if scope is not None:
            if scope.write is not None:
                # Community write scope has no content check, so citations
                # remain the binding proxy: any out-of-region cite is fatal.
                blocked = [c for c in kept if not self._writable(agent_id, c)]
                if blocked:
                    return {"error": f"cannot write outside your region; "
                                     f"blocked cites: {blocked}"}
            else:
                # Citation admissibility follows the READ scope (citing means
                # having read): a broad reader may cite anywhere; a specialist
                # only its readable region. Out-of-scope cites are DROPPED,
                # not fatal — the generate proceeds with the rest.
                kept = [c for c in cited_ids if self._readable(agent_id, c)]
                dropped = [c for c in cited_ids if c not in kept]
            if scope.write_anchors is not None:
                # The write scope binds the new idea's OWN content.
                v = self.embedder.encode([text])[0]
                if not _within_region(scope.write_anchors, scope.write_radius, v):
                    return {"error": "the idea itself is outside your writable topic region"}
        node_id = self._apply_generate(text, kept,
                                       meta={"run_id": self.run_id,
                                             "agent_id": agent_id, "step": step})
        if self.generation_budget is not None:
            self.generation_budget -= 1
        result = {"node_id": node_id}
        if dropped:
            result["dropped_cites"] = dropped
        return result

    def _do_add_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        blocked = [n for n in [src_id, *dst_ids] if not self._writable(agent_id, n)]
        if blocked:
            return {"error": f"cannot write outside your region; blocked: {blocked}"}
        return self.graph.add_links(src_id, dst_ids,  # KeyError -> error result
                                    meta={"run_id": self.run_id,
                                          "agent_id": agent_id, "step": step})

    def _do_remove_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        blocked = [n for n in [src_id, *dst_ids] if not self._writable(agent_id, n)]
        if blocked:
            return {"error": f"cannot write outside your region; blocked: {blocked}"}
        return self.graph.remove_links(src_id, dst_ids)  # KeyError -> error result

    # --- shared by generate and restore ---
    def _apply_generate(self, text: str, cited_ids: list[str], meta: dict,
                        node_id: str | None = None) -> str:
        # restore passes the RECORDED id so replayed nodes keep their identity
        # even if the run was later renamed (run_id != original prefix).
        if node_id is None:
            node_id = f"gen:{self.run_id}:{self._gen_counter}"
        self.graph.add_idea(node_id, text, cited_ids, meta=meta)  # KeyError propagates
        self.index.add([node_id], self.embedder.encode([text]))
        self._gen_counter += 1
        # New node inherits the majority community of its citations (smallest
        # id on ties, deterministic) so scoped regions stay meaningful as the
        # network grows.
        comms = [self.community_of[c] for c in cited_ids if c in self.community_of]
        if comms:
            counts = Counter(comms)
            best = max(counts.values())
            self.community_of[node_id] = min(c for c, n in counts.items() if n == best)
        return node_id

    def restore(self, events: list[dict]) -> None:
        """Replay graph-mutating events (generate, add_links) to rebuild state
        (spec §3.3); no logging. Events replay in order, so links to nodes
        generated earlier in the trace resolve correctly."""
        for e in events:
            if e["action"] == "generate" and "node_id" in e.get("result", {}):
                dropped = set(e["result"].get("dropped_cites") or [])
                kept = [c for c in e["args"]["cited_ids"] if c not in dropped]
                self._apply_generate(e["args"]["text"], kept,
                                     meta={"run_id": e["run_id"],
                                           "agent_id": e["agent_id"],
                                           "step": e["step"]},
                                     node_id=e["result"]["node_id"])
                if self.generation_budget is not None:
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
