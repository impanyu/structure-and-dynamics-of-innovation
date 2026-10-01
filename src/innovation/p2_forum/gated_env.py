"""Paper 2's environment with online literature and hard topic gating
(spec §5-§6). An item is readable by an agent iff its labels intersect the
agent's topics. Filtering what is RETURNED is the primary mechanism; the
checks on citations and links are a safety net. Every refusal carries a
"gate" field so gate activity can be counted from the event log."""
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.tagger import UnlabeledError


class GatedForumEnvironment(ForumEnvironment):
    def __init__(self, *, literature, tagger, agent_topics: dict[str, set[int]],
                 topic_names: list[str], **kw):
        super().__init__(**kw)
        self.lit, self.tagger = literature, tagger
        self.agent_topics = {a: set(t) for a, t in agent_topics.items()}
        self.topic_names = topic_names
        self.post_labels: dict[str, list[int]] = {}

    # --- labels and the gate ---
    def _labels(self, node_id: str) -> list[int] | None:
        if self.ws.store_of(node_id) == "board":
            return self.post_labels.get(node_id)
        return self.lit.labels([node_id]).get(node_id)

    def _ok(self, agent_id: str, labels) -> bool:
        return bool(labels) and bool(set(labels) & self.agent_topics[agent_id])

    def readable(self, agent_id: str, node_id: str) -> bool:
        return self._ok(agent_id, self._labels(node_id))

    def _names(self, labels) -> list[str]:
        return [self.topic_names[i] for i in labels or []]

    def _gated_papers(self, agent_id, papers):
        labs = self.lit.labels([p.paper_id for p in papers])
        return [(p, labs[p.paper_id]) for p in papers if self._ok(agent_id, labs[p.paper_id])]

    def _paper_hit(self, p, labels) -> dict:
        return {"node_id": p.paper_id, "store": "corpus", "title": p.title,
                "text": p.abstract[:300], "year": p.year, "venue": p.venue,
                "topics": self._names(labels)}

    def _query_gate(self, agent_id, query):
        try:
            labels = self.tagger.label(query)
        except UnlabeledError:
            return {"error": "this query could not be labeled", "gate": "unlabeled"}
        if not self._ok(agent_id, labels):
            return {"error": "this query is outside your topics", "gate": "query",
                    "topics": self._names(labels)}
        return None

    # --- literature ---
    def _do_search(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        refused = self._query_gate(agent_id, query)
        if refused:
            return refused
        kept = self._gated_papers(agent_id, self.lit.search(query))[:k]
        return {"hits": [self._paper_hit(p, l) for p, l in kept]}

    def _do_browse(self, *, agent_id, step, node_id: str) -> dict:
        p = self.lit.get(node_id)
        if p is None:
            return {"error": f"{node_id} is not an available paper", "gate": "result"}
        labels = self.lit.labels([node_id])[node_id]
        if not self._ok(agent_id, labels):
            return {"error": f"{node_id} is outside your topics", "gate": "result"}
        view = {**self._paper_hit(p, labels), "text": p.abstract}
        if not self.nav.corpus_edges:
            return {**view, "cites": [], "cited_by": []}
        cites = self._gated_papers(agent_id, self.lit.references(node_id))[:10]
        cited_by = self._gated_papers(agent_id, self.lit.citations(node_id))[:10]
        return {**view, "cites": [self._paper_hit(q, l) for q, l in cites],
                "cited_by": [self._paper_hit(q, l) for q, l in cited_by]}

    def _do_sample_frontier(self, *, agent_id, step) -> dict:
        if not self.nav.corpus_jump:
            return {"error": "random jumps into the literature are closed"}
        topic = self.topic_names[int(self.rng.choice(sorted(self.agent_topics[agent_id])))]
        kept = self._gated_papers(agent_id, self.lit.search(topic))
        if not kept:
            return {"error": "no paper found for a random jump"}
        p, l = kept[int(self.rng.integers(len(kept)))]
        return {**self._paper_hit(p, l), "text": p.abstract}

    # --- board ---
    def _post_view(self, nid) -> dict:
        if self.ws.store_of(nid) == "board":
            return {"node_id": nid, "store": "board", "text": self.ws.node(nid).text[:200],
                    "topics": self._names(self.post_labels.get(nid))}
        p = self.lit.get(nid)
        return {"node_id": nid, "store": "corpus", "title": p.title if p else "",
                "topics": self._names(self._labels(nid))}

    def _readable_posts(self, agent_id) -> list[str]:
        return [n for n in self.ws.board_post_ids() if self.readable(agent_id, n)]

    def _do_search_board(self, *, agent_id, step, query: str, k: int = 5) -> dict:
        if not self.nav.board_search:
            return {"error": "semantic search over the board is closed"}
        refused = self._query_gate(agent_id, query)
        if refused:
            return refused
        vec = self.ws.embedder.encode([query])[0]
        ranked = self.ws.board_search(vec, k=len(self.ws.board_post_ids()) or 1)
        kept = [(n, s) for n, s in ranked if self.readable(agent_id, n)][:k]
        return {"hits": [{**self._post_view(n), "text": self.ws.node(n).text[:300],
                          "score": s} for n, s in kept]}

    def _do_browse_board(self, *, agent_id, step, node_id: str) -> dict:
        if self.ws.store_of(node_id) != "board" or not self.ws.board.has_node(node_id):
            return {"error": f"{node_id} is not a board node"}
        if not self.readable(agent_id, node_id):
            return {"error": f"{node_id} is outside your topics", "gate": "result"}
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
        try:
            labels = self.tagger.label(text)
        except UnlabeledError:
            return {"error": "this idea could not be labeled; it was not published",
                    "gate": "unlabeled"}
        if not self._ok(agent_id, labels):
            return {"error": "this idea is outside your topics; it was not published",
                    "gate": "post", "topics": labels}
        kept = [c for c in cited_ids if self.ws.has_node(c) and self.readable(agent_id, c)]
        dropped = [c for c in cited_ids if c not in kept]
        node_id = self.ws.post_idea(text, kept, meta=self._meta(agent_id, step))
        self.post_labels[node_id] = labels
        if self.generation_budget is not None:
            self.generation_budget -= 1
        out = {"node_id": node_id, "topics": labels}
        if dropped:
            out["dropped_cites"] = dropped
        return out

    def _links_gate(self, agent_id, src_id, dst_ids):
        blocked = [n for n in [src_id, *dst_ids] if not self.readable(agent_id, n)]
        if blocked:
            return {"error": f"outside your topics: {blocked}", "gate": "link"}
        return None

    def _do_add_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_add_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    def _do_remove_links(self, *, agent_id, step, src_id: str, dst_ids: list[str]) -> dict:
        return self._links_gate(agent_id, src_id, dst_ids) or super()._do_remove_links(
            agent_id=agent_id, step=step, src_id=src_id, dst_ids=dst_ids)

    # --- replay ---
    def restore(self, events: list[dict]) -> None:
        """Replay the board; post labels come from the log, so a resumed run
        makes no tagger calls for replayed steps."""
        for e in events:
            r = e.get("result", {})
            if e["action"] == "generate" and "node_id" in r:
                self.post_labels[r["node_id"]] = r["topics"]
                kept = [c for c in e["args"]["cited_ids"] if c not in r.get("dropped_cites", [])]
                self.ws.post_idea(e["args"]["text"], kept,
                                  meta={"run_id": e["run_id"], "agent_id": e["agent_id"],
                                        "step": e["step"]}, node_id=r["node_id"])
                if self.generation_budget is not None:
                    self.generation_budget -= 1
            else:
                super().restore([e])
