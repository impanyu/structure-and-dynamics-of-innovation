"""Paper 2's environment over the frozen corpus with semantic-region gating
(spec, REVISION 2026-10-06; ranking and pagination R5 2026-10-07; one tool set
over papers and posts R6 2026-10-08).

Each agent has a Region (region.py): a nearest-neighbour ball around its seed
paper. A corpus paper is readable iff it is a member. A board post is readable
by its author, and by any other agent under the same rule as a paper: iff it
lies inside that agent's ball, i.e. its cosine to the seed is at least the
radius (region.contains_vec; user decision 2026-10-08). Reading is
gated, publishing is not (user decision 2026-10-08): results are filtered to
readable items, and reads and links that would leave the region are refused
with a "gate" field (as in gated_env.py), so gate activity can be counted from
the event log. An agent may publish any idea; only its cites are restricted to
what it can read. There is no query gate.

One tool set covers both stores (R6). Papers (S2 hex ids) and posts (`gen:`
ids) share one id space and every result item carries `"kind": "paper"` or
`"kind": "post"`:
- search / related return two separately ranked sections, `papers` (PAGE_SIZE
  to a page, `page`) and `posts` (POST_PAGE_SIZE to a page, `post_page`), plus
  one notice. A merged ranking would bury the posts (R6 simulation: a readable
  teammate post reached the top 10 in 6 of 123 searches). Each section honours
  its own store's Navigation search flag.
- browse opens a paper (abstract, `cites`, `cited_by`, and `cited_by_posts`:
  the board posts citing it) or a post (full text, mixed `cites`, `cited_by`).
- random(kind) jumps to a random readable paper or post.
The pre-R6 names stay as undocumented aliases (search_board -> search,
browse_board -> browse, sample_frontier -> random(paper), sample_board ->
random(post)), so old logs, resumed runs and stray replies still work.

Every listing is ranked by cosine and tagged with a coarse relevance tier,
never a raw score.

Corpus node text is "title\\n\\nabstract"; year and venue come from the node.
"""
import math

import numpy as np

from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.region import Region, _unit, contains_vec

PAGE_SIZE = 10          # papers per page, and every browse list
POST_PAGE_SIZE = 5      # posts per page in search / related

# Relevance tiers over bge-small cosine, (high, medium) lower bounds; below
# medium is low. Calibrated on data/p2_corpus (2026-10-07):
# - query -> paper: over the c20 run's queries, the global top-1 match has median
#   cosine 0.85, the 10th 0.80, the 100th 0.75; a random paper is 0.64. High is a
#   top-10-grade match, medium a top-few-hundred one, low is near random.
# - paper -> paper: cited pairs have median 0.81 (p25 0.78), random pairs 0.70,
#   and a paper's 10th nearest neighbour is at 0.86. High is nearest-neighbour
#   grade, medium is typical of a real citation, low is near random.
# - query -> post (R6, run forum-region-c20-s0c): a query against the papers the
#   agent's next post cited has median 0.77, against that next post itself also
#   0.77; against a random post 0.67 (a random paper: 0.64). Queries meet posts
#   at the same cosines as papers, so the query tiers serve both sections.
QUERY_TIERS = (0.80, 0.72)     # search: both sections
PAPER_TIERS = (0.85, 0.78)     # related (both sections), browse neighbour lists

SEARCH_NOTICE = (
    "These are the papers and posts within your allowed topics that best match your "
    "query, most relevant first, in two separate lists. Papers and posts outside your "
    "topics are never shown: if nothing here is relevant (e.g. only low relevance), "
    "what you are looking for is outside your topics — rephrase toward your topics "
    "rather than repeating the search. The posts list shows what other researchers "
    "(and you) have recently proposed in your topics.")
RELATED_NOTICE = (
    "These are the papers and posts within your allowed topics most related to this "
    "item, most relevant first, in two separate lists. Papers and posts outside your "
    "topics are never shown: if nothing here is relevant (e.g. only low relevance), "
    "what you are looking for is outside your topics — turn toward your topics rather "
    "than repeating the request. The posts list shows what other researchers (and you) "
    "have recently proposed in your topics.")

KINDS = ("paper", "post")


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


def _paginate(ranked: list, page, size: int = PAGE_SIZE) -> tuple[int, list, dict]:
    """(page, the page's items, {"page", "total", "pages"}). A page past the
    end is empty but still reports the totals."""
    p = _page_number(page)
    total = len(ranked)
    items = ranked[(p - 1) * size: p * size]
    return p, items, {"page": p, "total": total, "pages": math.ceil(total / size)}


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

    # --- the gate ---
    def readable(self, agent_id: str, node_id: str) -> bool:
        region = self.regions[agent_id]
        if self.ws.store_of(node_id) == "board":
            if (self.ws.board.has_node(node_id)
                    and self.ws.node(node_id).meta.get("agent_id") == agent_id):
                return True   # an author can always reread its own posts
            vec = self.ws.board_index.vec(node_id)
            return vec is not None and contains_vec(region, vec)
        # Corpus papers, and the corpus_ref stubs the board keeps for them,
        # share the paper's id.
        return node_id in region.members

    def _is_paper(self, node_id: str) -> bool:
        return self.ws.store_of(node_id) == "corpus" and self.ws.corpus.has_node(node_id)

    def _is_post(self, node_id: str) -> bool:
        return self.ws.store_of(node_id) == "board" and self.ws.board.has_node(node_id)

    def _open(self, agent_id: str, node_id) -> tuple[str | None, dict | None]:
        """(kind, None) for a readable paper or post, else (None, the error)."""
        node_id = str(node_id)
        kind = "paper" if self._is_paper(node_id) else "post" if self._is_post(node_id) else None
        if kind is None:
            return None, {"error": f"{node_id} is not a known paper or post"}
        if not self.readable(agent_id, node_id):
            return None, {"error": f"{node_id} is outside your research area", "gate": "result"}
        return kind, None

    # --- views ---
    def _author(self, agent_id: str, node_id: str) -> str:
        author = self.ws.node(node_id).meta.get("agent_id")
        return "you" if author == agent_id else author

    def _paper_hit(self, node_id: str) -> dict:
        node = self.ws.node(node_id)
        title, abstract = _split(node.text)
        return {"node_id": node_id, "kind": "paper", "title": title,
                "text": abstract[:300], "year": node.year,
                "venue": node.meta.get("venue")}

    def _paper_full(self, node_id: str) -> dict:
        return {**self._paper_hit(node_id), "text": _split(self.ws.node(node_id).text)[1]}

    def _post_hit(self, agent_id: str, node_id: str, chars: int | None = 300) -> dict:
        text = self.ws.node(node_id).text
        return {"node_id": node_id, "kind": "post", "author": self._author(agent_id, node_id),
                "text": text if chars is None else text[:chars]}

    def _ref_entry(self, agent_id: str, node_id: str) -> dict:
        """A reference-list line: enough to recognise a paper or post and
        decide to open it."""
        if self.ws.store_of(node_id) == "board":
            return self._post_hit(agent_id, node_id, chars=200)
        node = self.ws.node(node_id)
        return {"node_id": node_id, "kind": "paper", "title": _split(node.text)[0],
                "year": node.year}

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
        if vec is None:
            return [(n, -1.0) for n in node_ids]
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

    def _readable_posts(self, agent_id) -> list[str]:
        return [n for n in self.ws.board_post_ids() if self.readable(agent_id, n)]

    def _sections(self, agent_id, vec, tiers, page, post_page, exclude=None) -> dict:
        """The two sections of search / related: readable papers (PAGE_SIZE to
        a page) and readable posts (POST_PAGE_SIZE), each ranked by cosine to
        vec on its own, or an error where that store's search is closed."""
        if self.nav.corpus_search:
            _, items, info = _paginate(self._ranked_papers(agent_id, vec, exclude), page)
            papers = {**info, "items": [{**self._paper_hit(n), "relevance": _tier(s, tiers)}
                                        for n, s in items]}
        else:
            papers = {"error": "semantic search over the literature is closed"}
        if self.nav.board_search:
            posts = [n for n in self._readable_posts(agent_id) if n != exclude]
            _, items, info = _paginate(self._rank(posts, vec), post_page, POST_PAGE_SIZE)
            posts = {**info, "items": [{**self._post_hit(agent_id, n),
                                        "relevance": _tier(s, tiers)} for n, s in items]}
        else:
            posts = {"error": "semantic search over the board is closed"}
        return {"papers": papers, "posts": posts}

    def _list(self, agent_id, ids, vec, page, key) -> dict:
        """One browse list: ids ranked by cosine to vec (PAPER_TIERS), PAGE_SIZE
        to a page, as `key` plus `key`_page / _total / _pages."""
        _, items, info = _paginate(self._rank(ids, vec), page)
        return {key: [{**self._ref_entry(agent_id, n), "relevance": _tier(s, PAPER_TIERS)}
                      for n, s in items],
                f"{key}_page": info["page"], f"{key}_total": info["total"],
                f"{key}_pages": info["pages"]}

    # --- reading: search, browse, related, random ---
    def _do_search(self, *, agent_id, step, query: str, page=1, post_page=1,
                   **_ignored) -> dict:
        vec = self.ws.embedder.encode([query])[0]
        return {**self._sections(agent_id, vec, QUERY_TIERS, page, post_page),
                "notice": SEARCH_NOTICE}

    def _do_browse(self, *, agent_id, step, node_id: str, ref_page=1, cited_by_page=1,
                   post_page=1, **_ignored) -> dict:
        kind, err = self._open(agent_id, node_id)
        if err:
            return err
        vec = self._vec(node_id)
        keep = lambda ids: [n for n in ids if self.readable(agent_id, n)]
        if kind == "post":
            out_ids, in_ids = (self.ws.board_neighbors(node_id) if self.nav.board_edges
                               else ([], []))
            return {**self._post_hit(agent_id, node_id, chars=None),
                    **self._list(agent_id, keep(out_ids), vec, ref_page, "cites"),
                    **self._list(agent_id, keep(in_ids), vec, cited_by_page, "cited_by")}
        out_ids, in_ids = (self.ws.corpus_neighbors(node_id) if self.nav.corpus_edges
                           else ([], []))
        # Posts citing a paper are in-edges of its corpus_ref stub on the board
        # (every board edge has a post as its source).
        citing_posts = (self.ws.board_neighbors(node_id)[1]
                        if self.nav.board_edges and self.ws.board.has_node(node_id) else [])
        cites, cited_by, by_posts = keep(out_ids), keep(in_ids), keep(citing_posts)
        return {**self._paper_full(node_id),
                **self._list(agent_id, cites, vec, ref_page, "cites"),
                **self._list(agent_id, cited_by, vec, cited_by_page, "cited_by"),
                **self._list(agent_id, by_posts, vec, post_page, "cited_by_posts"),
                # region: hidden papers (cites + cited_by); region_posts: hidden citing posts
                "filtered": {"region": len(out_ids) - len(cites) + len(in_ids) - len(cited_by),
                             "region_posts": len(citing_posts) - len(by_posts)}}

    def _do_related(self, *, agent_id, step, node_id: str, page=1, post_page=1,
                    **_ignored) -> dict:
        kind, err = self._open(agent_id, node_id)
        if err:
            return err
        return {"node_id": node_id, "kind": kind,
                **self._sections(agent_id, self._vec(node_id), PAPER_TIERS, page, post_page,
                                 exclude=node_id),
                "notice": RELATED_NOTICE}

    def _do_random(self, *, agent_id, step, kind="paper", **_ignored) -> dict:
        # One rng draw per call, exactly as the pre-R6 sample_frontier /
        # sample_board handlers, so replays and resumed runs draw the same.
        if kind == "paper":
            if not self.nav.corpus_jump:
                return {"error": "random jumps into the literature are closed"}
            members = self._members_sorted[agent_id]
            return self._paper_full(members[int(self.rng.integers(len(members)))])
        if kind == "post":
            if not self.nav.board_jump:
                return {"error": "random jumps into the board are closed"}
            posts = self._readable_posts(agent_id)
            if not posts:
                return {"error": "no readable posts on the board"}
            return self._post_hit(agent_id, posts[int(self.rng.integers(len(posts)))],
                                  chars=None)
        return {"error": f"kind must be one of {list(KINDS)}; got {kind!r}"}

    # --- pre-R6 names: undocumented aliases ---
    def _do_search_board(self, *, agent_id, step, query: str, page=1, **_ignored) -> dict:
        # Its `page` paged the posts, so it becomes post_page.
        return self._do_search(agent_id=agent_id, step=step, query=query, post_page=page)

    def _do_browse_board(self, **kw) -> dict:
        return self._do_browse(**kw)

    def _do_sample_frontier(self, *, agent_id, step, **_ignored) -> dict:
        return self._do_random(agent_id=agent_id, step=step, kind="paper")

    def _do_sample_board(self, *, agent_id, step, **_ignored) -> dict:
        return self._do_random(agent_id=agent_id, step=step, kind="post")

    # --- writes ---
    def _do_generate(self, *, agent_id, step, text: str, cited_ids: list[str]) -> dict:
        if self.generation_budget is not None and self.generation_budget <= 0:
            return {"error": "generation budget exhausted"}
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

    # restore(): the base class's replay is exact here. Only generate /
    # add_links / remove_links are replayed, so reads under the pre-R6 names
    # (search_board, browse_board, sample_frontier, sample_board) need nothing.
    # Posts replay with their kept cites (_kept_cites honours dropped_cites);
    # refused posts and links carry no node_id / added / removed and are
    # skipped. Regions are fixed per run, so no gate state needs rebuilding
    # and no LLM is called.
