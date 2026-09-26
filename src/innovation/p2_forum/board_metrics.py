"""The board's structure as a function of round (spec §5).

Paper 1 could report one scalar — one cross-agent citation in 10,300 actions.
Here every structural quantity is a function of time, so the analysis entry
point is replay of the event log, not the final state.
"""
import networkx as nx
import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.workspace import CORPUS_REF, Workspace


class _NullLog(EventLog):
    """Replay must not re-log; board_at is read-only analysis."""

    def __init__(self):
        pass

    def append(self, event: dict) -> dict:
        return event

    def read_all(self) -> list[dict]:
        return []


def board_at(events: list[dict], round_index: int, *, corpus, corpus_index,
             embedder, run_id: str) -> Workspace:
    """Rebuild the board as it stood after the first `round_index` events."""
    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=run_id)
    env = ForumEnvironment(run_id=run_id, workspace=ws, event_log=_NullLog(),
                           rng=np.random.default_rng(0))
    env.restore(events[:round_index])
    return ws


def board_structure(ws: Workspace, events: list[dict]) -> dict:
    """Structural summary of one board state. `events` supplies authorship,
    which the graph itself does not carry.

    Two of the returned keys look similar but differ in scope, and the
    difference matters for the paper:

    - `first_cross_agent_citation_step` is the step of the first citation of
      another agent's post made AT POST TIME — i.e. it only looks at
      `cited_ids` on `generate` events, in event order. It ignores
      cross-agent edges added later via `add_links`.
    - `cross_agent_edges` counts every cross-agent post-post edge currently on
      the board, INCLUDING ones added later by `add_links` (not only the ones
      present at post time).
    """
    posts = set(ws.board_post_ids())
    author = {e["result"]["node_id"]: e["agent_id"] for e in events
              if e["action"] == "generate" and "node_id" in e.get("result", {})}

    post_post, post_corpus, cross = 0, 0, 0
    first_cross = None
    for src in posts:
        for dst in ws.board.citations_out(src):
            if ws.board.node(dst).source == CORPUS_REF:
                post_corpus += 1
                continue
            post_post += 1
            if author.get(src) is not None and author.get(src) != author.get(dst):
                cross += 1

    for e in events:
        if e["action"] != "generate" or "node_id" not in e.get("result", {}):
            continue
        src = e["result"]["node_id"]
        for dst in e["args"].get("cited_ids", []):
            if dst in author and author[dst] != e["agent_id"]:
                first_cross = e["step"] if first_cross is None else first_cross
                break
        if first_cross is not None:
            break

    sub = nx.DiGraph()
    sub.add_nodes_from(posts)
    sub.add_edges_from((s, d) for s in posts
                       for d in ws.board.citations_out(s) if d in posts)
    total = post_post + post_corpus

    return {"n_posts": len(posts),
            "n_post_post_edges": post_post,
            "n_post_corpus_edges": post_corpus,
            "post_post_share": (post_post / total) if total else 0.0,
            "n_components": nx.number_weakly_connected_components(sub),
            "longest_chain": nx.dag_longest_path_length(sub)
            if nx.is_directed_acyclic_graph(sub) else -1,
            "first_cross_agent_citation_step": first_cross,
            "cross_agent_edges": cross}
