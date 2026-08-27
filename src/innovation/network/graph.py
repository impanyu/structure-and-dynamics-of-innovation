"""IdeaGraph: networkx behind our own interface so the backend can swap (spec §3.3)."""
from dataclasses import dataclass, field

import networkx as nx
import pandas as pd


@dataclass
class IdeaNode:
    node_id: str
    text: str
    year: int | None
    source: str  # "corpus" | "generated"
    meta: dict = field(default_factory=dict)


class IdeaGraph:
    def __init__(self):
        self._g = nx.DiGraph()  # edge src -> dst means "src cites dst"

    @classmethod
    def from_tables(cls, ideas: pd.DataFrame, edges: pd.DataFrame) -> "IdeaGraph":
        g = cls()
        for r in ideas.itertuples():
            g._g.add_node(r.paper_id, data=IdeaNode(
                node_id=r.paper_id, text=r.idea_text, year=int(r.year),
                source="corpus", meta={"venue": r.venue}))
        for r in edges.itertuples():
            g._g.add_edge(r.src, r.dst)
        return g

    # --- reads ---
    def node(self, node_id: str) -> IdeaNode:
        return self._g.nodes[node_id]["data"]

    def has_node(self, node_id: str) -> bool:
        return self._g.has_node(node_id)

    def node_ids(self, source: str | None = None) -> list[str]:
        if source is None:
            return list(self._g.nodes)
        return [n for n in self._g.nodes if self._g.nodes[n]["data"].source == source]

    def citations_out(self, node_id: str) -> list[str]:
        return list(self._g.successors(node_id))

    def citations_in(self, node_id: str) -> list[str]:
        return list(self._g.predecessors(node_id))

    def in_degree(self, node_id: str) -> int:
        return self._g.in_degree(node_id)

    @property
    def num_nodes(self) -> int:
        return self._g.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._g.number_of_edges()

    # --- writes ---
    def add_idea(self, node_id: str, text: str, cited_ids: list[str], *,
                 source: str = "generated", year: int | None = None,
                 meta: dict | None = None) -> None:
        if self._g.has_node(node_id):
            raise ValueError(f"duplicate node_id: {node_id}")
        missing = [c for c in cited_ids if not self._g.has_node(c)]
        if missing:
            raise KeyError(f"cited ids not in graph: {missing}")
        self._g.add_node(node_id, data=IdeaNode(
            node_id=node_id, text=text, year=year, source=source, meta=meta or {}))
        for c in cited_ids:
            self._g.add_edge(node_id, c)

    # --- analysis ---
    def network_at(self, year: int) -> "IdeaGraph":
        keep = [n for n in self._g.nodes
                if self._g.nodes[n]["data"].source == "corpus"
                and self._g.nodes[n]["data"].year is not None
                and self._g.nodes[n]["data"].year <= year]
        sub = IdeaGraph()
        sub._g = self._g.subgraph(keep).copy()
        return sub

    def communities(self, seed: int = 0) -> dict[str, int]:
        parts = nx.community.louvain_communities(self._g.to_undirected(), seed=seed)
        return {n: i for i, part in enumerate(parts) for n in part}
