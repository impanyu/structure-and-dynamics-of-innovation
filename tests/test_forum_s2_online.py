from innovation.p2_forum.s2_online import S2Online


class Resp:
    def __init__(self, status, payload):
        self.status_code, self._p = status, payload
        self.headers = {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(response=self)


def fake_get(routes):
    calls = []

    def get(url, params=None, headers=None, timeout=None):
        calls.append((url, dict(params or {})))
        for key, resp in routes.items():
            if key in url:
                return resp
        raise AssertionError(url)
    get.calls = calls
    return get


P = {"paperId": "a", "title": "A", "abstract": "x", "year": 2023,
     "venue": "NeurIPS", "publicationVenue": None,
     "publicationDate": "2023-12-01", "citationCount": 3}


def test_search_passes_date_cap_and_caches(tmp_path):
    get = fake_get({"/paper/search": Resp(200, {"data": [P]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.search("q", limit=50, max_date="2024-09-30") == [P]
    assert c.search("q", limit=50, max_date="2024-09-30") == [P]
    assert len(get.calls) == 1
    params = get.calls[0][1]
    assert params["publicationDateOrYear"] == ":2024-09-30"
    assert params["limit"] == 50


def test_references_and_citations_unwrap(tmp_path):
    get = fake_get({"/references": Resp(200, {"data": [{"citedPaper": P}]}),
                    "/citations": Resp(200, {"data": [{"citingPaper": P}]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.references("z") == [P] and c.citations("z") == [P]


def test_missing_paper_is_none(tmp_path):
    c = S2Online(tmp_path, http_get=fake_get({"/paper/": Resp(404, {})}), delay=0)
    assert c.paper("nope") is None


def test_null_entries_are_dropped(tmp_path):
    get = fake_get({"/references": Resp(200, {"data": [{"citedPaper": {"paperId": None}}, {"citedPaper": P}]})})
    assert S2Online(tmp_path, http_get=get, delay=0).references("z") == [P]


def test_null_data_and_null_entries_are_tolerated(tmp_path):
    get = fake_get({"/search": Resp(200, {"data": None}),
                    "/references": Resp(200, {"data": [None, {"citedPaper": None}, {"citedPaper": P}]}),
                    "/citations": Resp(200, {"data": None})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.search("q", limit=5, max_date="2024-09-30") == []
    assert c.references("z") == [P]
    assert c.citations("z") == []


def test_match_returns_the_data_list_and_caches(tmp_path):
    get = fake_get({"/paper/search/match": Resp(200, {"data": [{**P, "matchScore": 99.0}]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    got = c.match("A", max_date="2024-09-30")
    assert [p["paperId"] for p in got] == ["a"]
    assert c.match("A", max_date="2024-09-30") == got
    assert len(get.calls) == 1
    url, params = get.calls[0]
    assert url.endswith("/paper/search/match") and params["query"] == "A"


def test_match_without_a_title_match_is_empty(tmp_path):
    c = S2Online(tmp_path, http_get=fake_get({"/paper/search/match": Resp(404, {})}), delay=0)
    assert c.match("no such title", max_date="2024-09-30") == []


def test_match_caches_a_404_as_empty(tmp_path):
    get = fake_get({"/paper/search/match": Resp(404, {})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.match("t", max_date="2024-09-30") == []
    assert c.match("t", max_date="2024-09-30") == []
    assert len(get.calls) == 1


def test_other_404s_are_still_not_cached(tmp_path):
    get = fake_get({"/paper/": Resp(404, {})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.paper("nope") is None and c.paper("nope") is None
    assert len(get.calls) == 2


def test_recommend_uses_all_cs_pool_unwraps_tolerates_nulls_and_caches(tmp_path):
    get = fake_get({"/recommendations/v1/papers/forpaper/z": Resp(
        200, {"recommendedPapers": [None, {"paperId": None}, P]})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.recommend("z") == [P] and c.recommend("z") == [P]
    assert len(get.calls) == 1
    url, params = get.calls[0]
    assert url == "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/z"
    assert params["limit"] == 500 and params["from"] == "all-cs" and "paperId" in params["fields"]


def test_recommend_null_and_missing(tmp_path):
    get = fake_get({"/forpaper/n": Resp(200, {"recommendedPapers": None}),
                    "/forpaper/m": Resp(404, {})})
    c = S2Online(tmp_path, http_get=get, delay=0)
    assert c.recommend("n") == [] and c.recommend("m") == []
    assert c.recommend("m") == []
    assert [u for u, _ in get.calls].count(
        "https://api.semanticscholar.org/recommendations/v1/papers/forpaper/m") == 1
