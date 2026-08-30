import json

from innovation.eval.search_verify import (Verdict, extract_queries,
                                           judge_level, s2_search,
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
                       "pub_date": "2025-06-01", "venue": "", "citations": 0,
                       "source_api": "s2"}
    s2_search("q", cache_dir=tmp_path, http_get=fake_get)
    assert len(calls) == 1  # cached
    cache_files = list(tmp_path.glob("*.json"))
    assert "fetched_at" in json.loads(cache_files[0].read_text())


def test_judge_level_parses_json_and_clamps():
    lvl, ev = judge_level(FakeLLM(responses=['{"level": 4, "evidence": "same core"}']),
                          model="m", idea_text="i",
                          candidate={"title": "t", "abstract": "a"})
    assert (lvl, ev) == (4, "same core")
    lvl, _ = judge_level(FakeLLM(responses=["garbage"]), model="m",
                         idea_text="i", candidate={"title": "t", "abstract": "a"})
    assert lvl == 0
    lvl, _ = judge_level(FakeLLM(responses=['{"level": 9}']), model="m",
                         idea_text="i", candidate={"title": "t", "abstract": "a"})
    assert lvl == 5


def test_verify_idea_anticipation_only(tmp_path):
    # Two candidates realize the idea: one pre-cutoff (excluded), one post-cutoff (hit).
    payload = s2_payload([{"id": "old", "title": "Old paper", "date": "2024-01-01"},
                          {"id": "new", "title": "New paper", "date": "2025-08-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["one query",
                             '{"level": 4, "evidence": "e"}',
                             '{"level": 3, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="gen:r:0", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1, top_k=5)
    assert isinstance(v, Verdict)
    assert v.best["tier1"]["level"] == 3
    assert v.best["tier1"]["paper"]["paper_id"] == "new"
    assert [p["paper_id"] for p in v.excluded_pre_cutoff] == ["old"]
    assert v.excluded_pre_cutoff[0]["level"] == 4


def test_verify_idea_no_hit_when_only_pre_cutoff(tmp_path):
    payload = s2_payload([{"id": "old", "title": "Old", "date": "2020-01-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", '{"level": 5, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert v.best["tier3"]["level"] == 0
    assert len(v.excluded_pre_cutoff) == 1


def test_verify_idea_unknown_date_not_hit_not_excluded(tmp_path):
    # One judged-YES candidate with no date; should not be excluded or hit.
    payload = s2_payload([{"id": "unknown", "title": "Unknown date paper"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", '{"level": 4, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert v.best["tier3"]["level"] == 0
    assert v.excluded_pre_cutoff == []
    assert [p["paper_id"] for p in v.unknown_date] == ["unknown"]


def test_verify_idea_keeps_first_of_multiple_post_cutoff_hits(tmp_path):
    payload = s2_payload([{"id": "n1", "title": "New A", "date": "2025-06-01"},
                          {"id": "n2", "title": "New B", "date": "2025-07-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", '{"level": 3, "evidence": "e"}',
                             '{"level": 4, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert v.best["tier1"]["level"] == 4
    assert v.best["tier1"]["paper"]["paper_id"] == "n2"


def test_tier_of_classification():
    from innovation.eval.search_verify import tier_of
    t1 = ["neural information processing", "iclr"]
    t2 = ["emnlp"]
    assert tier_of({"venue": "Advances in Neural Information Processing Systems",
                    "citations": 0}, t1, t2, 50) == "tier1"
    assert tier_of({"venue": "Workshop on X", "citations": 80}, t1, t2, 50) == "tier1"
    assert tier_of({"venue": "Proceedings of EMNLP 2025", "citations": 3},
                   t1, t2, 50) == "tier2"
    assert tier_of({"venue": "Obscure Journal", "citations": 3}, t1, t2, 50) == "tier3"
    assert tier_of({"venue": "", "citations": 0}, None, None, 50) == "tier1"  # off


def test_verify_idea_hit_requires_recognition(tmp_path):
    payload = {"data": [
        {"paperId": "lowq", "title": "Fringe paper", "abstract": "a",
         "publicationDate": "2025-06-01", "venue": "Obscure Journal",
         "citationCount": 2},
        {"paperId": "top", "title": "ICLR paper", "abstract": "a",
         "publicationDate": "2025-07-01",
         "venue": "International Conference on Learning Representations",
         "citationCount": 0}]}

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", '{"level": 4, "evidence": "e"}',
                             '{"level": 4, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1,
                    recognized_aliases=["international conference on learning representations"],
                    recognized_min_citations=10)
    # top (ICLR alias) -> tier1; lowq (obscure venue, 2 cites) -> tier3
    assert v.best["tier1"]["level"] == 4
    assert v.best["tier1"]["paper"]["paper_id"] == "top"
    assert [p["paper_id"] for p in v.candidates["tier3"]] == ["lowq"]
    assert v.best["tier3"]["level"] == 4  # cumulative: tier1 paper counts too


def test_verify_idea_excludes_in_corpus_titles(tmp_path):
    payload = s2_payload([{"id": "known", "title": "A Known Graph Paper",
                           "date": "2025-06-01"}])

    def fake_get(url, params=None):
        return FakeResponse(payload if "semanticscholar" in url
                            else {"results": [], "meta": {}})

    llm = FakeLLM(responses=["q", '{"level": 5, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1,
                    corpus_titles={"a known graph paper"})
    assert v.best["tier3"]["level"] == 0
    assert [p["paper_id"] for p in v.excluded_in_corpus] == ["known"]


def test_cached_get_retries_on_429(tmp_path):
    from innovation.eval.search_verify import _cached_get

    calls = []

    class Resp:
        def __init__(self, code):
            self.status_code = code

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": []}

    def flaky_get(url, params=None, **kw):
        calls.append(1)
        return Resp(429 if len(calls) < 3 else 200)

    out = _cached_get("http://x", {"q": 1}, tmp_path, flaky_get, delay=0)
    assert out == {"data": []} and len(calls) == 3


def test_verify_idea_degrades_when_openalex_unavailable(tmp_path):
    """A dead OpenAlex channel (persistent 429/budget) must not kill the
    evaluation — verify proceeds on S2 results alone."""
    import requests

    payload = s2_payload([{"id": "new", "title": "New paper", "date": "2025-08-01"}])

    class Resp429:
        status_code = 429
        def raise_for_status(self):
            raise requests.HTTPError("429")
        def json(self):
            return {}

    def fake_get(url, params=None, **kw):
        if "semanticscholar" in url:
            return FakeResponse(payload)
        return Resp429()

    llm = FakeLLM(responses=["q", '{"level": 4, "evidence": "e"}'])
    v = verify_idea(llm, model="m", idea_id="x", idea_text="idea",
                    cutoff_date="2025-01-01", mailto="a@b.c",
                    cache_dir=tmp_path, http_get=fake_get, n_queries=1)
    assert v.best["tier1"]["level"] == 4  # no alias lists -> everything tier1
