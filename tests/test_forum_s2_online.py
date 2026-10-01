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
