"""Tests for OpenAlex edge augmentation."""
import pandas as pd

from innovation.core.data.edge_augment import augment_edges, s2_fetch_external_ids


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def test_fetch_external_ids_caches(tmp_path):
    def fake_post(url, params=None, json=None, headers=None):
        return FakeResponse([{"paperId": pid, "externalIds": {"MAG": "1"}}
                             for pid in json["ids"]])

    ext = s2_fetch_external_ids(["a", "b"], cache_dir=tmp_path,
                                http_post=fake_post, delay=0, batch_size=1)
    assert ext == {"a": {"MAG": "1"}, "b": {"MAG": "1"}}
    ext2 = s2_fetch_external_ids(["a", "b"], cache_dir=tmp_path,
                                 http_post=lambda **k: (_ for _ in ()).throw(AssertionError),
                                 delay=0, batch_size=1)
    assert ext2 == ext


def test_augment_edges_unions_openalex_refs(tmp_path):
    papers = pd.DataFrame([
        {"paper_id": "s1", "title": "A", "abstract": "a", "year": 2020, "venue": "V"},
        {"paper_id": "s2", "title": "B", "abstract": "b", "year": 2021, "venue": "V"},
        {"paper_id": "s3", "title": "C", "abstract": "c", "year": 2022, "venue": "V"},
    ])
    s2_edges = pd.DataFrame([{"src": "s2", "dst": "s1"}])

    def fake_post(url, params=None, json=None, headers=None):
        ext = {"s1": {"MAG": "100"},                       # -> W100 (openalex arm)
               "s2": {"DOI": "10.1/x"},                    # -> doi arm
               "s3": {"ArXiv": "2101.00001"}}              # -> datacite doi arm
        return FakeResponse([{"paperId": pid, "externalIds": ext[pid]}
                             for pid in json["ids"]])

    def fake_get(url, params=None, headers=None):
        f = params["filter"]
        if f.startswith("openalex:"):
            results = [{"id": "https://openalex.org/W100", "doi": None,
                        "referenced_works": []}]
        else:  # doi chunk: s2 cites s1 (via W100); s3 cites s2
            results = [
                {"id": "https://openalex.org/W200",
                 "doi": "https://doi.org/10.1/x",
                 "referenced_works": ["https://openalex.org/W100"]},
                {"id": "https://openalex.org/W300",
                 "doi": "https://doi.org/10.48550/arxiv.2101.00001",
                 "referenced_works": ["https://openalex.org/W200"]},
            ]
        return FakeResponse({"results": results})

    edges = augment_edges(papers, s2_edges, cache_dir=tmp_path, mailto="t@t",
                          http_get=fake_get, http_post=fake_post, delay=0)
    got = {(r.src, r.dst) for r in edges.itertuples()}
    # s2->s1 exists in both sources (deduped); s3->s2 comes only from OpenAlex
    assert got == {("s2", "s1"), ("s3", "s2")}


def _one_paper():
    return pd.DataFrame([{"paper_id": "s1", "title": "A", "abstract": "a",
                          "year": 2020, "venue": "V"}])


def _post(url, params=None, json=None, headers=None):
    return FakeResponse([{"paperId": pid, "externalIds": {"DOI": "10.1/x"}}
                         for pid in json["ids"]])


def test_openalex_key_sent_but_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRETKEY")
    seen = []

    def fake_get(url, params=None, headers=None):
        seen.append(dict(params))
        return FakeResponse({"results": []})

    augment_edges(_one_paper(), pd.DataFrame(columns=["src", "dst"]),
                  cache_dir=tmp_path, mailto="t@t", http_get=fake_get,
                  http_post=_post, delay=0)
    assert seen and all(p["api_key"] == "SECRETKEY" for p in seen)
    assert not any("SECRETKEY" in f.name or "SECRETKEY" in f.read_text()
                   for f in tmp_path.iterdir())


def test_openalex_key_redacted_in_errors(tmp_path, monkeypatch):
    import pytest
    import requests
    monkeypatch.setenv("OPENALEX_API_KEY", "SECRETKEY")

    class Bad:
        status_code = 403

        def raise_for_status(self):
            raise requests.HTTPError("403 for url: https://x?api_key=SECRETKEY")

    with pytest.raises(requests.RequestException) as ei:
        augment_edges(_one_paper(), pd.DataFrame(columns=["src", "dst"]),
                      cache_dir=tmp_path, mailto="t@t",
                      http_get=lambda url, params=None, **k: Bad(),
                      http_post=_post, delay=0)
    assert "SECRETKEY" not in str(ei.value)
