"""Paper 2's environment over the frozen corpus with semantic-region gating
(spec, REVISION 2026-10-06; ranking and pagination R5 2026-10-07).

Each agent has a Region (region.py): a nearest-neighbour ball around its seed
paper. A corpus paper is readable iff it is a member; a board post is readable
iff at least 3 of its 5 nearest corpus papers are members (region.post_in_region). Every action is gated: results are
filtered to readable items, and reads, posts and links that would leave the
region are refused with a "gate" field (as in gated_env.py), so gate activity
can be counted from the event log. There is no query gate.

Every listing (search, related, search_board, and the reference / cited-by
lists of browse and browse_board) is ranked by cosine, paginated PAGE_SIZE to
a page and tagged with a coarse relevance tier, never a raw score. Searches
carry a notice that results are restricted to the agent's topics, so an agent
whose query lies outside its area learns that from the low tiers instead of
repeating the search.

Corpus node text is "title\\n\\nabstract"; year and venue come from the node.
"""
import math

import numpy as np

from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.region import NN_K, Region, _unit, post_in_region

PAGE_SIZE = 10

# Relevance tiers over bge-small cosine, (high, medium) lower bounds; below
# medium is low. Calibrated on data/p2_corpus (2026-10-07):
# - query -> paper: over the c20 run's queries, the global top-1 match has median
#   cosine 0.85, the 10th 0.80, the 100th 0.75; a random paper is 0.64. High is a
#   top-10-grade match, medium a top-few-hundred one, low is near random.
# - paper -> paper: cited pairs have median 0.81 (p25 0.78), random pairs 0.70,
#   and a paper's 10th nearest neighbour is at 0.86. High is nearest-neighbour
#   grade, medium is typical of a real citation, low is near random.
QUERY_TIERS = (0.80, 0.72)     # search, search_board
PAPER_TIERS = (0.85, 0.78)     # related, browse / browse_board neighbour lists

SEARCH_NOTICE = (
    "These are the papers within your allowed topics that best match your query, "
    "most relevant first. Papers outside your topics are never shown: if nothing "
    "here is relevant (e.g. only low relevance), what you are looking for is outside "
    "your topics — rephrase toward your topics rather than repeating the search.")
RELATED_NOTICE = (
    "These are the papers within your allowed topics most related to this paper, "
    "most relevant first. Papers outside your topics are never shown: if nothing "
    "here is relevant (e.g. only low relevance), what you are looking for is outside "
    "your topics — turn toward your topics rather than repeating the request.")
BOARD_NOTICE = (
    "These are the posts within your allowed topics that best match your query, "
    "most relevant first. Posts outside your topics are never shown: if nothing "
    "here is relevant (e.g. only low relevance), what you are looking for is outside "
    "your topics — rephrase toward your topics rather than repeating the search.")


def _split(text: str) -> tuple[str, str]:
    title, _, abstract = text.partition("\n\n")
    return title, abstract


def _tier(cos: float, tiers: tuple[float, float]) -> str:
    high, medium = tiers
    return "high" if cos >= high else "medium" if cos >= medium else "low"


def _page_number(page) -> int:
    """1-based; anything invalid or < 1 is page 1."""
    try:
        p = int(page)
    except (TypeError, ValueError):
        return 1
    return p if p >= 1 else 1


def _paginate(ranked: list, page) -> tuple[int, list, dict]:
    """(page, the page's items, {"page", "total", "pages"}). A page past the
    end is empty but still reports the totals."""
    p = _page_number(page)
    total = len(ranked)
    items = ranked[(p - 1) * PAGE_SIZE: p * PAGE_SIZE]
    return p, items, {"page": p, "total": total, "pages": math.ceil(total / PAGE_SIZE)}


class RegionGatedEnvironment(ForumEnvironment):
    def __init__(self, *, regions: dict[str, Region], **kw):
        super().__init__(**kw)
        self.regions = dict(regions)
        # Sorted, because frozenset order depends on the string hash seed and
        # random jumps must replay identically across processes.
        self._members_sorted = {a: sorted(r.members) for a, r in self.regions.items()}
        # Unit corpus vectors for cosine ranking, and each agent's member rows
        # in the same id-sorted order (so stable ranking breaks ties by id).
        ci = self.ws.corpus_index
        self._corpus_row = {pid: i for i, pid in enumerate(ci.ids)}
        self._corpus_unit = _unit(ci.vecs)
        self._member_rows = {a: np.array([self._corpus_row[m] for m in ms], dtype=int)
                             for a, ms in self._members_sorted.items()}
        # post id -> its NN_K nearest corpus paper ids, fixed when the post is
        # created. A pure function of the post text, so restore() refills it
        # lazily and identically (no LLM).
        self._post_nn: dict[str, tuple[str, ...]] = {}

    def _nearest_papers(self, vec) -> tuple[str, ...]:
        return tuple(n for n, _ in self.ws.corpus_search(np.asarray(vec), k=NN_K))

    def _post_nearest(self, node_id: str) -> tuple[str, ...] | None:
        if node_id not in self._post_nn:
            vec = self.ws.board_index.vec(node_id)
            if vec is None:
                return None
            self._post_nn[node_id] = self._nearest_papers(vec)
        return self._post_nn[node_id]

    # --- the gate ---
    def readable(self, agent_id: str, node_id: str) -> bool:
        region = self.regions[agent_id]
        if self.ws.store_of(node_id) == "board":
            nearest = self._post_nearest(node_id)
            return nearest is not None and post_in_region(region, nearest)
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

    # --- ranking ---
    def _vec(self, node_id: str):
        """Unit vector of a corpus paper or board post (None if unindexed)."""
        row = self._corpus_row.get(node_id)
        if row is not None:
            return self._corpus_unit[row]
        v = self.ws.board_index.vec(node_id)
        return None if v is None else _unit(v)

    def _rank(self, node_ids, vec) -> list[tuple[str, float]]:
        """node_ids ranked by cosine to vec, most similar first, ties by id."""
        q = _unit(vec)
        scored = []
        for n in node_ids:
            v = self._vec(n)
            scored.append((n, -1.0 if v is None else float(v @ q)))
        return sorted(scored, key=lambda t: (-t[1], t[0]))

    def _ranked_papers(self, agent_id: str, vec, exclude: str | None = None):
        """The agent's readable papers ranked by cosine to vec (vectorised:
        regions can hold the whole corpus)."""
        rows = self._member_rows[agent_id]
        sims = self._corpus_unit[rows] @ _unit(vec)
        order = np.argsort(-sims, kind="stable")      # members are id-sorted: ties by id
        members = self._members_sorted[agent_id]
        return [(members[i], float(sims[i])) for i in order if members[i] != exclude]

    # --- literature ---
    def _do_search(self, *, agent_id, step, query: str, page=1, **_ignored) -> dict:
        if not self.nav.corpus_search:
            return {"error": "semantic search over the literature is closed"}
        vec = self.ws.embedder.encode([query])[0]
        ranked = self._ranked_papers(agent_id, vec)
        _, items, info = _paginate(ranked, page)
        return {**info, "hits": [{**self._paper_hit(n), "relevance": _tier(s, QUERY_TIERS)}
                                 for n, s in items],
                "notice": SEARCH_NOTICE}

    def _do_browse(self, *, agent_id, step, node_id: str, ref_page=1, cited_by_page=1,
                   **_ignored) -> dict:
        if not self._is_paper(node_id):
            return {"error": f"{node_id} is not an available paper"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        view = self._paper_full(node_id)
        if not self.nav.corpus_edges:
            out_ids, in_ids = [], []
        else:
            out_ids, in_ids = self.ws.corpus_neighbors(node_id)
        cites = [n for n in out_ids if self.readable(agent_id, n)]
        cited_by = [n for n in in_ids if self.readable(agent_id, n)]
        hidden = len(out_ids) - len(cites) + len(in_ids) - len(cited_by)
        return {**view,
                **self._neighbour_pages(node_id, cites, cited_by, ref_page, cited_by_page,
                                        self._ref_entry),
                "filtered": {"region": hidden}}

    def _neighbour_pages(self, node_id, cites, cited_by, ref_page, cited_by_page, view):
        """cites / cited_by ranked by cosine to node_id (PAPER_TIERS), each
        paginated on its own page argument."""
        vec = self._vec(node_id)
        out = {}
        for key, ids, page in (("cites", cites, ref_page), ("cited_by", cited_by, cited_by_page)):
            ranked = self._rank(ids, vec) if vec is not None else [(n, -1.0) for n in ids]
            _, items, info = _paginate(ranked, page)
            out[key] = [{**view(n), "relevance": _tier(s, PAPER_TIERS)} for n, s in items]
            out[f"{key}_page"] = info["page"]
            out[f"{key}_total"] = info["total"]
            out[f"{key}_pages"] = info["pages"]
        return out

    def _do_related(self, *, agent_id, step, node_id: str, page=1, **_ignored) -> dict:
        if not self.nav.corpus_search:
            return {"error": "finding related papers is closed"}
        if not self._is_paper(node_id):
            return {"error": f"{node_id} is not an available paper"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        ranked = self._ranked_papers(agent_id, self._vec(node_id), exclude=node_id)
        _, items, info = _paginate(ranked, page)
        return {"node_id": node_id, **info,
                "related": [{**self._paper_hit(n), "relevance": _tier(s, PAPER_TIERS)}
                            for n, s in items],
                "notice": RELATED_NOTICE}

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

    def _do_search_board(self, *, agent_id, step, query: str, page=1, **_ignored) -> dict:
        if not self.nav.board_search:
            return {"error": "semantic search over the board is closed"}
        vec = self.ws.embedder.encode([query])[0]
        posts = self.ws.board_post_ids()
        gated = self._readable_posts(agent_id)
        _, items, info = _paginate(self._rank(gated, vec), page)
        return {**info,
                "hits": [{**self._post_view(n), "text": self.ws.node(n).text[:300],
                          "relevance": _tier(s, QUERY_TIERS)} for n, s in items],
                "filtered": {"region": len(posts) - len(gated)},
                "notice": BOARD_NOTICE}

    def _do_browse_board(self, *, agent_id, step, node_id: str, ref_page=1,
                         cited_by_page=1, **_ignored) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your research area", "gate": "result"}
        out_ids, in_ids = (self.ws.board_neighbors(node_id) if self.nav.board_edges
                           else ([], []))
        keep = lambda ids: [n for n in ids if self.readable(agent_id, n)]
        return {**self._post_view(node_id), "text": self.ws.node(node_id).text,
                **self._neighbour_pages(node_id, keep(out_ids), keep(in_ids),
                                        ref_page, cited_by_page, self._post_view)}

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
        nearest = self._nearest_papers(vec)
        if not post_in_region(self.regions[agent_id], nearest):
            return {"error": "this idea is outside your research area; it was not published",
                    "gate": "post"}
        kept = [c for c in cited_ids if self.ws.has_node(c) and self.readable(agent_id, c)]
        dropped = [c for c in cited_ids if c not in kept]
        node_id = self.ws.post_idea(text, kept, meta=self._meta(agent_id, step))
        self._post_nn[node_id] = nearest
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
