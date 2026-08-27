import json

from innovation.eval.search_verify import (Verdict, extract_queries,
                                           judge_realization, s2_search,
                                           verify_idea)
from innovation.llm import FakeLLM


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def s2_payload(papers):
    return {"data": [{"paperId": p["id"], "title": p["title"],
                      "abstract": p.get("abstract", ""),
                      "publicationDate": p.get("date")} for p in papers]}


def test_extract_queries_splits_lines():
    llm = FakeLLM(responses=["sparse attention transformers\nefficient long context\n"])
    qs = extract_queries(llm, model="m", idea_text="An idea about sparse attention.")
    assert qs == ["sparse attention transformers", "efficient long context"]


def test_s2_search_normalizes_and_caches(tmp_path):
    payload = s2_payload([{"id": "p1", "title": "T", "date": "2025-06-01"}])
    calls = []

    def fake_get(url, params=None):
        calls.append(url)
        return FakeResponse(payload)

    hits = s2_search("q", cache_dir=tmp_path, http_get=fake_get)
    assert hits[0] == {"paper_id": "p1", "title": "T", "abstract": "",
                       "pub_date": "2025-06-01", "source_api": "s2"}
    s2_search("q", cache_dir=tmp_path, http_get=fake_get)
    assert len(calls) == 1  # cached
    cache_files = list(tmp_path.glob("*.json"))
    assert "fetched_at" in json.loads(cache_files[0].read_text())


def test_judge_realization_yes_no():
    assert judge_realization(FakeLLM(responses=["YES"]), model="m",
                             idea_text="i", candidate={"title": "t", "abstract": "a"})
    assert not judge_realization(FakeLLM(responses=["NO, unrelated"]), model="m",
                                 idea_text="i", candidate={"title": "t", "abstract": "a"})


def test_verify_idea_anticipation_only(tmp_path):
    # Two candidates realize the idea: one pre-cutoff (excluded), one post-cutoff (hit).
    payload = s2_payload([{"id": "old", "title": "Old paper", "date": "2024-01-01"},
                          {"id": "new", "title": "New paper", "date": "2025-08-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["one query"] + ["YES", "YES"])
    v = verify_idea(llm, model="m", idea_id="gen:r:0", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1, top_k=5)
    assert isinstance(v, Verdict)
    assert v.hit and v.paper["paper_id"] == "new"
    assert [p["paper_id"] for p in v.excluded_pre_cutoff] == ["old"]


def test_verify_idea_no_hit_when_only_pre_cutoff(tmp_path):
    payload = s2_payload([{"id": "old", "title": "Old", "date": "2020-01-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", "YES"])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert not v.hit and v.paper is None
    assert len(v.excluded_pre_cutoff) == 1


def test_verify_idea_unknown_date_not_hit_not_excluded(tmp_path):
    # One judged-YES candidate with no date; should not be excluded or hit.
    payload = s2_payload([{"id": "unknown", "title": "Unknown date paper"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", "YES"])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert not v.hit and v.paper is None
    assert v.excluded_pre_cutoff == []
    assert [p["paper_id"] for p in v.unknown_date] == ["unknown"]
