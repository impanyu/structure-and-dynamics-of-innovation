"""Two stores behind one facade (spec §3).

The corpus is frozen external background; the board is the only thing that
evolves. Every write rule lives here, in one place: an edge's source must be a
board node, so corpus-internal structure can never be added to or removed.
"""
import numpy as np

from innovation.core.network.graph import IdeaGraph, IdeaNode
from innovation.core.network.index import VectorIndex

BOARD_PREFIX = "gen:"
CORPUS_REF = "corpus_ref"


class Workspace:
    def __init__(self, *, corpus: IdeaGraph, corpus_index: VectorIndex,
                 board_index: VectorIndex, embedder, run_id: str):
        if not corpus.frozen:
            raise ValueError("the corpus must be frozen before use")
        self.corpus = corpus
        self.board = IdeaGraph()
        self.corpus_index = corpus_index
        self.board_index = board_index
        self.embedder = embedder
        self.run_id = run_id
        self._counter = 0

    # --- routing ---
    def store_of(self, node_id: str) -> str:
        return "board" if str(node_id).startswith(BOARD_PREFIX) else "corpus"

    def has_node(self, node_id: str) -> bool:
        if self.store_of(node_id) == "board":
            return self.board.has_node(node_id)
        return self.corpus.has_node(node_id)

    def node(self, node_id: str) -> IdeaNode:
        if self.store_of(node_id) == "board":
            return self.board.node(node_id)
        return self.corpus.node(node_id)

    def board_post_ids(self) -> list[str]:
        """Board nodes that are real posts — stubs excluded."""
        return [n for n in self.board.node_ids()
                if self.board.node(n).source != CORPUS_REF]

    # --- reads ---
    def corpus_search(self, vec, k: int = 5) -> list[tuple[str, float]]:
        return self.corpus_index.search(vec, k=k)

    def board_search(self, vec, k: int = 5) -> list[tuple[str, float]]:
        return self.board_index.search(vec, k=k)

    def corpus_neighbors(self, node_id: str) -> tuple[list[str], list[str]]:
        return (self.corpus.citations_out(node_id),
                self.corpus.citations_in(node_id))

    def board_neighbors(self, node_id: str) -> tuple[list[str], list[str]]:
        return (self.board.citations_out(node_id),
                self.board.citations_in(node_id))

    def corpus_sample(self, rng) -> str:
        return str(rng.choice(self.corpus.node_ids()))

    def board_sample(self, rng) -> str | None:
        posts = self.board_post_ids()
        return str(rng.choice(posts)) if posts else None

    # --- writes (all validation lives here) ---
    def _require_board_source(self, src_id: str) -> None:
        if self.store_of(src_id) != "board":
            raise ValueError(
                f"edge source must be a board node; {src_id} is in the corpus")
        if not self.board.has_node(src_id):
            raise KeyError(f"unknown board node: {src_id}")

    def _ensure_stub(self, node_id: str) -> None:
        """Corpus ids referenced from the board get a text-free stub so the
        edge is an ordinary networkx edge and board algorithms need no
        special cases."""
        if self.board.has_node(node_id):
            return
        if not self.corpus.has_node(node_id):
            raise KeyError(f"unknown node: {node_id}")
        self.board.add_idea(node_id, "", [], source=CORPUS_REF, year=None)

    def post_idea(self, text: str, cited_ids: list[str], meta: dict,
                  node_id: str | None = None) -> str:
        missing = [c for c in cited_ids if not self.has_node(c)]
        if missing:
            raise KeyError(f"cited ids not found: {missing}")
        for c in cited_ids:
            self._ensure_stub(c)
        if node_id is None:
            node_id = f"{BOARD_PREFIX}{self.run_id}:{self._counter}"
        self.board.add_idea(node_id, text, cited_ids, source="generated",
                            meta=meta)
        self.board_index.add([node_id],
                             np.asarray(self.embedder.encode([text])))
        self._counter += 1
        return node_id

    def add_links(self, src_id: str, dst_ids: list[str], meta: dict) -> dict:
        self._require_board_source(src_id)
        missing = [d for d in dst_ids if not self.has_node(d)]
        if missing:
            raise KeyError(f"link targets not found: {missing}")
        for d in dst_ids:
            self._ensure_stub(d)
        return self.board.add_links(src_id, dst_ids, meta=meta)

    def remove_links(self, src_id: str, dst_ids: list[str]) -> dict:
        self._require_board_source(src_id)
        present = [d for d in dst_ids if self.board.has_node(d)]
        missing = [d for d in dst_ids if d not in present]
        out = self.board.remove_links(src_id, present) if present else {
            "removed": [], "skipped": []}
        out["skipped"] = list(out.get("skipped", [])) + missing
        return out
