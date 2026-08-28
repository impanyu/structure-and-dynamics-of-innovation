"""Tests for the Semantic Scholar venue-corpus module."""

from innovation.data.s2 import (build_s2_corpus, s2_bulk_venue_search,
                                s2_fetch_references)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_bulk_venue_search_paginates_and_caches(tmp_path):
    pages = [
        {"total": 3, "token": "T1",
         "data": [{"paperId": "p1", "title": "A", "abstract": "aa",
                   "year": 2020, "venue": "NeurIPS", "citationCount": 60}]},
        {"total": 3, "token": None,
         "data": [{"paperId": "p2", "title": "B", "abstract": "bb",
                   "year": 2021, "venue": "NeurIPS", "citationCount": 55}]},
    ]
    calls = []

    def fake_get(url, params=None, headers=None):
        calls.append(params)
        return FakeResponse(pages[len(calls) - 1])

    papers = s2_bulk_venue_search("NeurIPS", "2016-2025", min_citations=50,
                                  cache_dir=tmp_path, http_get=fake_get, delay=0)
    assert [p["paperId"] for p in papers] == ["p1", "p2"]
    assert calls[0]["minCitationCount"] == 50
    assert calls[0]["venue"] == "NeurIPS"
    assert "token" not in calls[0] or calls[0].get("token") is None
    assert calls[1]["token"] == "T1"
    # fully cached second pass: zero HTTP
    calls2 = []

    def fake_get2(url, params=None, headers=None):
        calls2.append(1)
        return FakeResponse({})

    papers2 = s2_bulk_venue_search("NeurIPS", "2016-2025", min_citations=50,
                                   cache_dir=tmp_path, http_get=fake_get2, delay=0)
    assert papers2 == papers and calls2 == []


def test_fetch_references_batches_and_caches(tmp_path):
    def fake_post(url, params=None, json=None, headers=None):
        return FakeResponse([
            {"paperId": pid, "references": [{"paperId": f"r_{pid}"}, {"paperId": None}]}
            for pid in json["ids"]])

    refs = s2_fetch_references(["p1", "p2"], cache_dir=tmp_path,
                               http_post=fake_post, delay=0, batch_size=1)
    assert refs == {"p1": ["r_p1"], "p2": ["r_p2"]}
    # cached
    refs2 = s2_fetch_references(["p1", "p2"], cache_dir=tmp_path,
                                http_post=lambda **k: (_ for _ in ()).throw(AssertionError),
                                delay=0, batch_size=1)
    assert refs2 == refs


def test_build_s2_corpus_filters_and_edges():
    raw = [
        {"paperId": "p1", "title": "A", "abstract": "aa", "year": 2020,
         "venue": "NeurIPS", "citationCount": 60},
        {"paperId": "p2", "title": "B", "abstract": None, "year": 2021,
         "venue": "ICLR", "citationCount": 55},          # dropped: no abstract
        {"paperId": "p3", "title": "C", "abstract": "cc", "year": 2022,
         "venue": "ICLR", "citationCount": 70},
        {"paperId": "p3", "title": "C", "abstract": "cc", "year": 2022,
         "venue": "ICLR", "citationCount": 70},          # duplicate
    ]
    refs = {"p3": ["p1", "external"], "p1": []}
    papers, edges = build_s2_corpus(raw, refs)
    assert list(papers["paper_id"]) == ["p1", "p3"]
    assert list(papers.columns) == ["paper_id", "title", "abstract", "year", "venue"]
    assert [(r.src, r.dst) for r in edges.itertuples()] == [("p3", "p1")]
