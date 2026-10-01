from innovation.p2_forum.literature import OnlineLiterature, Scope

ALIASES = ("neurips", "neural information processing", "computer vision and pattern recognition")
SCOPE = Scope(venue_aliases=ALIASES, min_citations=50, max_date="2024-09-30")


def raw(pid, venue="NeurIPS", date="2023-12-01", year=2023, cites=0, abstract="abs"):
    return {"paperId": pid, "title": f"T{pid}", "abstract": abstract, "year": year,
            "venue": venue, "publicationVenue": None, "publicationDate": date,
            "citationCount": cites}


def test_scope_branches():
    assert SCOPE.admit(raw("a")) == "venue"
    assert SCOPE.admit(raw("b", venue="Nature", cites=80)) == "citations"
    assert SCOPE.admit(raw("c", venue="Nature", cites=49)) is None
    assert SCOPE.admit(raw("d", date="2024-10-01")) is None
    assert SCOPE.admit(raw("e", date="2024-09-30")) == "venue"


def test_scope_workshops_and_undated_boundary_year():
    assert SCOPE.admit(raw("w", venue="CVPR Workshops")) is None
    assert SCOPE.admit(raw("w2", venue="2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)")) is None
    assert SCOPE.admit(raw("u", date=None, year=2024)) is None   # cannot prove <= cutoff
    assert SCOPE.admit(raw("v", date=None, year=2023)) == "venue"


def test_scope_reads_publication_venue_names():
    r = raw("p", venue="")
    r["publicationVenue"] = {"name": "Neural Information Processing Systems",
                             "alternate_names": ["NeurIPS"]}
    assert SCOPE.admit(r) == "venue"


class FakeClient:
    def __init__(self, papers, refs=None, cits=None):
        self.papers, self.refs, self.cits = papers, refs or {}, cits or {}
        self.search_calls = 0

    def search(self, q, *, limit, max_date):
        self.search_calls += 1
        return list(self.papers.values())[:limit]

    def paper(self, pid):
        return self.papers.get(pid)

    def references(self, pid):
        return self.refs.get(pid, [])

    def citations(self, pid):
        return self.cits.get(pid, [])


class FakeTagger:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def label_many(self, texts, workers=8):
        self.calls += len(texts)
        return [self.table.get(t.split("\n")[0]) for t in texts]


def lit(tmp_path, papers, **kw):
    tagger = FakeTagger({f"T{k}": [i] for i, k in enumerate(papers)})
    return OnlineLiterature(client=FakeClient(papers, **kw), scope=SCOPE,
                            tagger=tagger, cache_dir=tmp_path), tagger


def test_search_drops_out_of_scope_and_remembers_branch(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a"), "b": raw("b", venue="Nature", cites=99),
                          "c": raw("c", date="2025-01-01")})
    got = L.search("q")
    assert [(p.paper_id, p.branch) for p in got] == [("a", "venue"), ("b", "citations")]
    assert L.has("a") and not L.has("c")


def test_labels_are_computed_once_and_persisted(tmp_path):
    papers = {"a": raw("a"), "b": raw("b")}
    L, tagger = lit(tmp_path, papers)
    L.search("q")
    assert L.labels(["a", "b"]) == {"a": [0], "b": [1]}
    L.labels(["a", "b"])
    assert tagger.calls == 2
    L2, tagger2 = lit(tmp_path, papers)        # a fresh process reads the store
    L2.search("q")
    assert L2.labels(["a"]) == {"a": [0]} and tagger2.calls == 0


def test_get_and_neighbors_are_scope_filtered(tmp_path):
    papers = {"a": raw("a"), "x": raw("x", venue="Nature", cites=1)}
    L, _ = lit(tmp_path, papers, refs={"a": [raw("r", date="2022-01-01"), raw("late", date="2025-02-02")]})
    assert L.get("x") is None
    assert [p.paper_id for p in L.references("a")] == ["r"]


def test_text_uses_title_when_abstract_missing(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a", abstract=None)})
    assert L.get("a").text() == "Ta"
