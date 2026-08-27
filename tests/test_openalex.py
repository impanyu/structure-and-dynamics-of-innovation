from innovation.data.openalex import (fetch_field_works, fetch_source_works,
                                      find_source_id, reconstruct_abstract)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def make_fake_get(pages):
    """pages: list of payloads returned in call order. Records params."""
    calls = []

    def fake_get(url, params=None):
        calls.append({"url": url, "params": params})
        return FakeResponse(pages[len(calls) - 1])

    fake_get.calls = calls
    return fake_get


def test_reconstruct_abstract_orders_words_by_position():
    inv = {"networks": [2], "idea": [1], "grow": [3], "the": [0]}
    assert reconstruct_abstract(inv) == "the idea networks grow"
    assert reconstruct_abstract(None) == ""


def test_fetch_source_works_paginates_and_caches(tmp_path):
    page1 = {"results": [{"id": "W1"}, {"id": "W2"}],
             "meta": {"next_cursor": "abc"}}
    page2 = {"results": [{"id": "W3"}], "meta": {"next_cursor": None}}
    fake_get = make_fake_get([page1, page2])
    works = fetch_source_works("S123", 2013, 2024, mailto="a@b.c",
                               cache_dir=tmp_path, http_get=fake_get)
    assert [w["id"] for w in works] == ["W1", "W2", "W3"]
    assert len(fake_get.calls) == 2
    assert "S123" in fake_get.calls[0]["params"]["filter"]
    # Second call: everything comes from the disk cache, zero HTTP.
    fake_get2 = make_fake_get([])
    works2 = fetch_source_works("S123", 2013, 2024, mailto="a@b.c",
                                cache_dir=tmp_path, http_get=fake_get2)
    assert works2 == works
    assert len(fake_get2.calls) == 0


def test_find_source_id_returns_top_search_hit(tmp_path):
    payload = {"results": [{"id": "https://openalex.org/S999",
                            "display_name": "NeurIPS"}]}
    fake_get = make_fake_get([payload])
    sid = find_source_id("NeurIPS", mailto="a@b.c", cache_dir=tmp_path,
                         http_get=fake_get)
    assert sid == "S999"


def test_fetch_field_works_filters_by_search_and_citations(tmp_path):
    page1 = {"results": [{"id": "W1"}], "meta": {"next_cursor": None}}
    fake_get = make_fake_get([page1])
    works = fetch_field_works("federated learning", 2016, 2024,
                              mailto="a@b.c", cache_dir=tmp_path,
                              http_get=fake_get, min_citations=5)
    assert [w["id"] for w in works] == ["W1"]
    f = fake_get.calls[0]["params"]["filter"]
    assert "title_and_abstract.search:federated learning" in f
    assert "publication_year:2016-2024" in f
    assert "cited_by_count:>5" in f
    # min_citations=0 omits the citation clause
    fake_get2 = make_fake_get([page1])
    fetch_field_works("federated learning", 2016, 2024, mailto="a@b.c",
                      cache_dir=tmp_path / "b", http_get=fake_get2)
    assert "cited_by_count" not in fake_get2.calls[0]["params"]["filter"]
