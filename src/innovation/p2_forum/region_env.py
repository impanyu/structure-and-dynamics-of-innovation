"""Paper 2's environment over the frozen corpus with semantic-region gating
(spec, REVISION 2026-10-06).

Each agent has a Region (region.py): a nearest-neighbour ball around its seed
paper. A corpus paper is readable iff it is a member; a board post is readable
iff its embedding lies inside the ball. Every action is gated: results are
filtered to readable items, and reads, posts and links that would leave the
region are refused with a "gate" field (as in gated_env.py), so gate activity
can be counted from the event log. There is no query gate.

Corpus node text is "title\\n\\nabstract"; year and venue come from the node.
"""
import numpy as np

from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.region import Region, contains_vec


def _split(text: str) -> tuple[str, str]:
    title, _, abstract = text.partition("\n\n")
    return title, abstract


class RegionGatedEnvironment(ForumEnvironment):
    def __init__(self, *, regions: dict[str, Region], **kw):
        super().__init__(**kw)
        self.regions = dict(regions)
        # Sorted, because frozenset order depends on the string hash seed and
        # random jumps must replay identically across processes.
        self._members_sorted = {a: sorted(r.members) for a, r in self.regions.items()}

    # --- the gate ---
    def readable(self, agent_id: str, node_id: str) -> bool:
        region = self.regions[agent_id]
        if self.ws.store_of(node_id) == "board":
            vec = self.ws.board_index.vec(node_id)
            return vec is not None and contains_vec(region, vec)
        # Corpus papers, and the corpus_ref stubs the board keeps for them,
        # share the paper's id.
        return node_id in region.members

    def _is_paper(self, node_id: str) -> bool:
        return self.ws.store_of(node_id) == "corpus" and self.ws.corpus.has_node(node_id)

    # --- views ---
    def _paper_hit(self, node_id: str) -> dict:
        node = self.ws.node(node_id)
        title, abstract = _split(node.text)
        return {"node_id": node_id, "store": "corpus", "title": title,
                "text": abstract[:300], "year": node.year,
                "venue": node.meta.get("venue")}

    def _paper_full(self, node_id: str) -> dict:
        return {**self._paper_hit(node_id), "text": _split(self.ws.node(node_id).text)[1]}

    def _ref_entry(self, node_id: str) -> dict:
        """A reference-list line: enough to recognise a paper and decide to open it."""
        node = self.ws.node(node_id)
        return {"node_id": node_id, "title": _split(node.text)[0], "year": node.year}

    def _ranked_papers(self, agent_id: str, vec, k: int, exclude: str | None = None):
        ranked = self.ws.corpus_search(np.asarray(vec), k=len(self.ws.corpus_index.ids))
        members = self.regions[agent_id].members
        return [n for n, _ in ranked if n in members and n != exclude][:k]

    # --- literature ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.corpus_search:
            return {"error": "semantic search over the literature is closed"}
        vec = self.ws.embedder.encode([query])[0]
        return {"hits": [self._paper_hit(n) for n in self._ranked_papers(agent_id, vec, k)]}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        if not self._is_paper(node_id):
            return {"error": f"{node_id} is not an available paper"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        view = self._paper_full(node_id)
        if not self.nav.corpus_edges:
            return {**view, "cites": [], "cited_by": [], "filtered": {"region": 0}}
        out_ids, in_ids = self.ws.corpus_neighbors(node_id)
        cites = [n for n in out_ids if self.readable(agent_id, n)]
        cited_by = [n for n in in_ids if self.readable(agent_id, n)]
        hidden = len(out_ids) - len(cites) + len(in_ids) - len(cited_by)
        return {**view, "cites": [self._ref_entry(n) for n in cites],
                "cited_by": [self._ref_entry(n) for n in cited_by],
                "filtered": {"region": hidden}}

    def _do_related(self, *, agent_id, step, node_id: str, k: int = 10) -> dict:
        if not self.nav.corpus_search:
            return {"error": "finding related papers is closed"}
        if not self._is_paper(node_id):
            return {"error": f"{node_id} is not an available paper"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        vec = self.ws.corpus_index.vec(node_id)
        return {"node_id": node_id,
                "related": [self._paper_hit(n) for n in
                            self._ranked_papers(agent_id, vec, k, exclude=node_id)]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.nav.corpus_jump:
            return {"error": "random jumps into the literature are closed"}
        members = self._members_sorted[agent_id]
        return self._paper_full(members[int(self.rng.integers(len(members)))])

    # --- board ---
    def _post_view(self, nid) -> dict:
        if self.ws.store_of(nid) == "board":
            return {"node_id": nid, "store": "board", "text": self.ws.node(nid).text[:200]}
        node = self.ws.node(nid)
        return {"node_id": nid, "store": "corpus", "title": _split(node.text)[0],
                "year": node.year}

    def _readable_posts(self, agent_id) -> list[str]:
        return [n for n in self.ws.board_post_ids() if self.readable(agent_id, n)]

    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.board_search:
            return {"error": "semantic search over the board is closed"}
        vec = self.ws.embedder.encode([query])[0]
        ranked = self.ws.board_search(vec, k=len(self.ws.board_post_ids()) or 1)
        gated = [(n, s) for n, s in ranked if self.readable(agent_id, n)]
        return {"hits": [{**self._post_view(n), "text": self.ws.node(n).text[:300],
                          "score": s} for n, s in gated[:k]],
                "filtered": {"region": len(ranked) - len(gated)}}

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        out_ids, in_ids = (self.ws.board_neighbors(node_id) if self.nav.board_edges
                           else ([], []))
        keep = lambda ids: [self._post_view(n) for n in ids if self.readable(agent_id, n)][:10]
        return {**self._post_view(node_id), "text": self.ws.node(node_id).text,
                "cites": keep(out_ids), "cited_by": keep(in_ids)}

    def _do_sample_board(self, *, agent_id, step) -> dict:
        if not self.nav.board_jump:
            return {"error": "random jumps into the board are closed"}
        posts = self._readable_posts(agent_id)
        if not posts:
            return {"error": "no readable posts on the board"}
        nid = posts[int(self.rng.integers(len(posts)))]
        return {**self._post_view(nid), "text": self.ws.node(nid).text}

    # --- writes ---
    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
        vec = self.ws.embedder.encode([text])[0]
        if not contains_vec(self.regions[agent_id], vec):
            return {"error": "this idea is outside your research area; it was not published",
                    "gate": "post"}
        kept = [c for c in cited_ids if self.ws.has_node(c) and self.readable(agent_id, c)]
        dropped = [c for c in cited_ids if c not in kept]
        node_id = self.ws.post_idea(text, kept, meta=self._meta(agent_id, step))
        if self.generation_budget is not None:
            self.generation_budget -= 1
        out = {"node_id": node_id}
        if dropped:
            out["dropped_cites"] = dropped
        return out

    def _links_gate(self, agent_id, src_id, dst_ids):
        # An id that names nothing is a mistake, not a gate refusal: only
        # existing-but-unreadable ids count towards gate activity.
        for n in [src_id, *dst_ids]:
            if not self.ws.has_node(n):
                return {"error": f"{n} is not a known paper or post"}
        blocked = [n for n in [src_id, *dst_ids] if not self.readable(agent_id, n)]
        if blocked:
            return {"error": f"outside your research area: {blocked}", "gate": "link"}
        return None

    def _do_add_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_add_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    def _do_remove_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_remove_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    # restore(): the base class's replay is exact here. Posts replay with their
    # kept cites (_kept_cites honours dropped_cites); refused posts and links
    # carry no node_id / added / removed and are skipped. Regions are fixed
    # per run, so no gate state needs rebuilding and no LLM is called.
