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
    def __init__(self, papers, refs=None, cits=None, by_query=None, titles=None):
        self.papers, self.refs, self.cits = papers, refs or {}, cits or {}
        self.by_query, self.titles = by_query, titles or {}   # by_query: query -> raw list
        self.search_calls = 0
        self.queries = []

    def search(self, q, *, limit, max_date):
        self.search_calls += 1
        self.queries.append(("search", q))
        if self.by_query is not None:
            return self.by_query.get(q, [])[:limit]
        return list(self.papers.values())[:limit]

    def match(self, title, *, max_date):
        self.queries.append(("match", title))
        return self.titles.get(title, [])

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


def test_corrupt_lines_skipped_at_load(tmp_path):
    (tmp_path / "labels.jsonl").write_text(
        '{"paper_id": "a", "labels": [7]}\n{"paper_id": "b", "lab\n{"x": 1}\nnot json\n')
    L, tagger = lit(tmp_path, {"a": raw("a")})
    L.search("q")
    assert L.labels(["a"]) == {"a": [7]} and tagger.calls == 0


def test_second_instance_picks_up_later_appends(tmp_path):
    papers = {"a": raw("a"), "b": raw("b")}
    L1, t1 = lit(tmp_path, papers)
    L2, t2 = lit(tmp_path, papers)
    L1.search("q"); L2.search("q")
    L1.labels(["a", "b"])
    assert L2.labels(["a", "b"]) == {"a": [0], "b": [1]} and t2.calls == 0


def test_first_write_wins(tmp_path):
    (tmp_path / "labels.jsonl").write_text(
        '{"paper_id": "a", "labels": [1]}\n{"paper_id": "a", "labels": [2]}\n')
    L, _ = lit(tmp_path, {"a": raw("a")})
    L.search("q")
    assert L.labels(["a"]) == {"a": [1]}


def test_none_label_not_persisted(tmp_path):
    papers = {"a": raw("a")}
    L, _ = lit(tmp_path, papers)
    L.search("q")
    L.tagger.table = {}
    assert L.labels(["a"]) == {"a": None}
    assert L.labels(["a"]) == {"a": None} and L.tagger.calls == 1   # no retry in-process
    L2, t2 = lit(tmp_path, papers)
    L2.search("q")
    assert L2.labels(["a"]) == {"a": [0]} and t2.calls == 1


def test_null_paper_id_skipped(tmp_path):
    bad = raw("z"); bad["paperId"] = None
    L, _ = lit(tmp_path, {"a": raw("a")}, refs={"a": [bad, raw("r")]})
    assert [p.paper_id for p in L.references("a")] == ["r"]


def test_record_appended_between_lock_phases_is_not_appended_again(tmp_path):
    papers = {"a": raw("a")}
    L, tagger = lit(tmp_path, papers)
    L.search("q")
    orig = tagger.label_many

    def racing(texts, workers=8):        # another run writes "a" while we tag
        with (tmp_path / "labels.jsonl").open("a") as f:
            f.write('{"paper_id": "a", "labels": [9]}\n')
        return orig(texts, workers)
    tagger.label_many = racing
    assert L.labels(["a"]) == {"a": [9]}          # theirs wins, ours not appended
    lines = (tmp_path / "labels.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_last_scope_dropped_counts_per_call(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a"), "c": raw("c", date="2025-01-01"),
                          "n": {**raw("n"), "paperId": None}},
               refs={"a": [raw("r"), raw("late", date="2025-02-02")]})
    assert L.last_scope_dropped == 0
    L.search("q")
    assert L.last_scope_dropped == 2
    L.references("a")
    assert L.last_scope_dropped == 1
    L.citations("a")
    assert L.last_scope_dropped == 0


def test_label_store_is_keyed_by_tagger_system(tmp_path):
    from innovation.p2_forum.literature import label_store_path

    def mk(system):
        t = FakeTagger({})
        if system is not None:
            t.system = system
        return OnlineLiterature(client=FakeClient({}), scope=SCOPE, tagger=t, cache_dir=tmp_path)

    a, a2, b, plain = mk("list A"), mk("list A"), mk("list B"), mk(None)
    assert a._store == a2._store != b._store
    assert a._store.name.startswith("labels-") and a._store.suffix == ".jsonl"
    assert plain._store == tmp_path / "labels.jsonl"
    t = FakeTagger({})
    t.system = "list A"
    assert label_store_path(tmp_path, t) == a._store


SAE = "how to use sparse autoencoders to find interpretable features in language models"


def test_search_uses_the_query_as_is_when_it_finds_records(tmp_path):
    L, _ = lit(tmp_path, {"a": raw("a")})
    assert [p.paper_id for p in L.search(SAE)] == ["a"]
    assert L.last_query_used is None and L.client.queries == [("search", SAE)]


def test_search_falls_back_to_a_title_match(tmp_path):
    title = "LRM: Large Reconstruction Model for Single Image to 3D"
    client = FakeClient({}, by_query={}, titles={title: [raw("lrm")]})
    L = OnlineLiterature(client=client, scope=SCOPE, tagger=FakeTagger({}), cache_dir=tmp_path)
    assert [p.paper_id for p in L.search(title)] == ["lrm"]
    assert L.last_query_used is None
    assert client.queries == [("search", title), ("match", title)]


def test_search_falls_back_to_dropping_stopwords(tmp_path):
    short = "sparse autoencoders find interpretable features language models"
    client = FakeClient({}, by_query={short: [raw("s"), raw("late", date="2025-01-01")]})
    L = OnlineLiterature(client=client, scope=SCOPE, tagger=FakeTagger({}), cache_dir=tmp_path)
    assert [p.paper_id for p in L.search(SAE)] == ["s"]
    assert L.last_query_used == short and L.last_scope_dropped == 1
    assert client.queries == [("search", SAE), ("match", SAE), ("search", short)]
    L.search(short)                       # a later plain search resets it
    assert L.last_query_used is None


def test_stopword_rewrite_keeps_the_first_eight_words(tmp_path):
    from innovation.p2_forum.literature import drop_stopwords
    assert drop_stopwords("The role of attention in a model") == "role attention model"
    assert drop_stopwords("one two three four five six seven eight nine ten") == \
        "one two three four five six seven eight"
    assert drop_stopwords("How, using what?") == ""


def test_no_retry_when_the_rewrite_changes_nothing(tmp_path):
    client = FakeClient({}, by_query={})
    L = OnlineLiterature(client=client, scope=SCOPE, tagger=FakeTagger({}), cache_dir=tmp_path)
    assert L.search("diffusion transformers") == []
    assert client.queries == [("search", "diffusion transformers"), ("match", "diffusion transformers")]
    assert L.last_query_used is None
