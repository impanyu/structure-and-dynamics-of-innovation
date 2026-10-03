from innovation.p2_forum.literature import OnlineLiterature
from innovation.p2_forum.openalex_online import OpenAlexOnline, map_to_s2, s2_lookup_id
from innovation.p2_forum.s2_online import S2Online
from test_forum_literature import FakeClient, FakeTagger, SCOPE, raw
from test_forum_s2_online import Resp, fake_get

OA = {"results": [
    {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/abc", "title": "One"},
    {"id": "https://openalex.org/W2", "doi": None, "title": "Two",
     "ids": {"arxiv": "https://arxiv.org/abs/2301.00001"}},
    {"id": "https://openalex.org/W3", "doi": None, "title": "Three", "ids": {}},
    {"id": "https://openalex.org/W4", "doi": None, "title": None, "ids": {}}]}


def test_search_params_and_cache_exclude_api_key(tmp_path, monkeypatch):
    get = fake_get({"openalex.org/works": Resp(200, OA)})
    monkeypatch.setenv("OPENALEX_API_KEY", "k1")
    c = OpenAlexOnline(tmp_path, http_get=get, delay=0)
    assert len(c.search("q", max_date="2024-09-30")) == 4
    p = get.calls[0][1]
    assert p["api_key"] == "k1" and p["search"] == "q" and p["per_page"] == 25
    assert p["filter"] == "to_publication_date:2024-09-30"
    monkeypatch.setenv("OPENALEX_API_KEY", "other")        # same cache entry
    assert len(c.search("q", max_date="2024-09-30")) == 4
    assert len(get.calls) == 1
    assert not any("k1" in f.read_text() or "other" in f.read_text()
                   for f in (tmp_path / "openalex").iterdir())


def test_lookup_ids():
    assert s2_lookup_id(OA["results"][0]) == "DOI:10.1/abc"
    assert s2_lookup_id(OA["results"][1]) == "ARXIV:2301.00001"
    assert s2_lookup_id(OA["results"][2]) is None


def test_batch_cache_is_order_safe(tmp_path):
    calls = []

    def post(url, params=None, json=None, headers=None, timeout=None):
        calls.append((url, params, json))
        return Resp(200, [raw("x"), raw("y")] if json["ids"] == ["DOI:x", "DOI:y"] else [])
    c = S2Online(tmp_path, http_get=None, delay=0, http_post=post)
    assert [r["paperId"] for r in c.batch(["DOI:x", "DOI:y"])] == ["x", "y"]
    assert [r["paperId"] for r in c.batch(["DOI:y", "DOI:x"])] == ["y", "x"]   # cached, realigned
    assert len(calls) == 1 and calls[0][0].endswith("/paper/batch")


def test_batch_chunks_at_500_ids(tmp_path):
    sizes = []

    def post(url, params=None, json=None, headers=None, timeout=None):
        sizes.append(len(json["ids"]))
        return Resp(200, [None] * len(json["ids"]))
    c = S2Online(tmp_path, delay=0, http_post=post)
    assert c.batch([f"DOI:{i:04d}" for i in range(501)]) == [None] * 501
    assert sizes == [500, 1]


class MapClient:
    def __init__(self):
        self.batches, self.matched = [], []

    def batch(self, ids):
        self.batches.append(ids)
        return [raw("a") if i == "DOI:10.1/abc" else None for i in ids]

    def match(self, title, *, max_date):
        self.matched.append(title)
        return [raw("m")] if title == "Three" else []


def test_map_to_s2_batches_then_matches_title_and_counts_unmapped():
    c = MapClient()
    recs, unmapped = map_to_s2(OA["results"], c, max_date="2024-09-30")
    assert len(c.batches) == 1 and c.batches[0] == ["DOI:10.1/abc", "ARXIV:2301.00001"]
    assert c.matched == ["Two", "Three"]        # no title, no id: dropped without a call
    assert [r["paperId"] for r in recs] == ["a", "m"]
    assert unmapped == 2


class OAClient(FakeClient):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.batch_ids = []

    def batch(self, ids):
        self.batch_ids.append(ids)
        return [raw("z")]


class FakeOA:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, q, *, max_date, limit=25):
        self.calls.append(q)
        return self.hits


def make_lit(tmp_path, client, oa):
    return OnlineLiterature(client=client, scope=SCOPE, tagger=FakeTagger({}),
                            cache_dir=tmp_path, openalex=oa)


HIT = [{"id": "W9", "doi": "https://doi.org/10.9/z", "title": "Tz"}]


def test_openalex_runs_only_after_every_s2_stage_admits_nothing(tmp_path):
    oa = FakeOA(HIT)
    client = OAClient({}, by_query={}, titles={})
    L = make_lit(tmp_path, client, oa)
    got = L.search("the quick privacy query")
    assert [p.paper_id for p in got] == ["z"]
    assert oa.calls == ["the quick privacy query"]          # the ORIGINAL query
    assert L.last_source == "openalex" and L.last_unmapped == 0
    assert client.batch_ids == [["DOI:10.9/z"]]
    assert [k for k, _ in client.queries] == ["search", "match", "search"]  # all S2 stages first


def test_openalex_skipped_when_s2_admits_or_client_absent(tmp_path):
    oa = FakeOA(HIT)
    L = make_lit(tmp_path, OAClient({"a": raw("a")}), oa)
    assert [p.paper_id for p in L.search("q")] == ["a"]
    assert oa.calls == [] and L.last_source == "s2"
    L2 = make_lit(tmp_path, OAClient({}, by_query={}), None)
    assert L2.search("q") == [] and L2.last_source == "s2"


def test_unmapped_counted_and_scope_drops_summed(tmp_path):
    class C(OAClient):
        def batch(self, ids):
            return [raw("late", date="2025-01-01")]
    L = make_lit(tmp_path, C({}, by_query={}), FakeOA(HIT + [{"id": "W8", "title": None}]))
    assert L.search("q") == []
    assert L.last_unmapped == 1 and L.last_scope_dropped == 1 and L.last_source == "s2"


class BoomOA:
    def search(self, q, *, max_date, limit=25):
        import requests
        raise requests.RequestException("down")


def test_failing_openalex_stage_returns_empty_and_logs(tmp_path):
    L = make_lit(tmp_path, OAClient({}, by_query={}), BoomOA())
    assert L.search("my query") == [] and L.last_source == "s2"
    assert "my query" in (tmp_path / "openalex_errors.log").read_text()


def test_errors_never_carry_the_api_key(tmp_path, monkeypatch):
    import requests
    monkeypatch.setenv("OPENALEX_API_KEY", "sekrit")
    url = "https://api.openalex.org/works?search=q&api_key=sekrit"

    class Bad:
        status_code = 403
        headers = {}

        def raise_for_status(self):
            raise requests.HTTPError(f"403 Client Error: Forbidden for url: {url}", response=self)

    def conn(u, params=None, **kw):
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")
    for get in (lambda u, params=None, **kw: Bad(), conn):
        c = OpenAlexOnline(tmp_path / str(id(get)), http_get=get, delay=0)
        try:
            c.search("q", max_date="2024-09-30")
            raise AssertionError("expected an error")
        except requests.RequestException as e:
            assert "sekrit" not in str(e) and "***" in str(e)
            assert e.__cause__ is None and e.__suppress_context__


def test_429_gives_up_after_few_attempts(tmp_path, monkeypatch):
    import innovation.core.data.s2 as s2m
    monkeypatch.setattr(s2m.time, "sleep", lambda s: None)
    n = []

    def get(u, params=None, **kw):
        n.append(1)
        return Resp(429, {})
    import requests
    c = OpenAlexOnline(tmp_path, http_get=get, delay=0)
    try:
        c.search("q", max_date="2024-09-30")
    except requests.RequestException:
        pass
    assert len(n) == 4
