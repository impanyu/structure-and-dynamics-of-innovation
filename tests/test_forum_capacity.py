import pytest

from innovation.core.llm import FakeLLM
from innovation.p2_forum.capacity import (BulkCounter, CapacityEstimator, capacity,
                                          parse_query, precision, union_count,
                                          venue_chunks)

COMMA = "International Conference for High Performance Computing, Networking, Storage and Analysis"
VENUES = ["Neural Information Processing Systems", COMMA, "AAAI Conference on Artificial Intelligence"]


def test_union_is_inclusion_exclusion():
    assert union_count(n_v=100, n_c=300, n_vc=40) == 360


def test_capacity_counts_venue_pool_and_citation_only_pool():
    # 100 venue papers at 50% + (300 - 40) citation-only papers at 10%.
    assert capacity(100, 300, 40, 0.5, 0.1) == pytest.approx(50 + 26)


def test_venue_chunks_isolate_comma_names():
    chunks = venue_chunks(VENUES)
    assert chunks == [["Neural Information Processing Systems",
                       "AAAI Conference on Artificial Intelligence"], [COMMA]]
    assert all(len(c) == 1 for c in chunks if any("," in v for v in c))


def test_precision_skips_unlabeled_and_empty_sample_is_zero():
    labels = {"a": [3, 1], "b": [2], "c": None}
    assert precision(["a", "b", "c"], labels, 3) == (0.5, 2)
    assert precision([], labels, 3) == (0.0, 0)


@pytest.mark.parametrize("reply, want", [
    ('"graph neural network" | "message passing"', '"graph neural network" | "message passing"'),
    ("```\ngraph | self-supervised | \"node embedding\"\n```",
     'graph | "self-supervised" | "node embedding"'),
])
def test_parse_query_normalizes(reply, want):
    assert parse_query(reply) == want


@pytest.mark.parametrize("reply", ["only one", "a | b | c | d | e | f", "a | (b + c)", "a || b"])
def test_parse_query_rejects(reply):
    with pytest.raises(ValueError):
        parse_query(reply)


class FakeResp:
    def __init__(self, payload, status=200):
        self.status_code, self._p = status, payload

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _rec(pid, venue, title="t"):
    return {"paperId": pid, "title": f"{title} {pid}", "abstract": "abs", "venue": venue}


class FakeS2:
    """Totals per (venue chunk, cited). The comma name is split by the real
    API: its page carries an unrelated venue except for `comma_hits` records."""

    def __init__(self, comma_hits=1):
        self.calls, self.comma_hits = [], comma_hits

    def __call__(self, url, params, headers):
        self.calls.append(dict(params))
        venue, cited = params.get("venue"), "minCitationCount" in params
        if venue is None:
            data = [_rec("c1", "AAAI Conference on Artificial Intelligence"),
                    _rec("c2", "arXiv.org"), _rec("c3", "IEEE Access")]
            return FakeResp({"total": 500, "data": data})
        if venue == COMMA:
            data = ([_rec(f"h{i}", COMMA) for i in range(self.comma_hits)]
                    + [_rec(f"x{i}", "Languages") for i in range(4 - self.comma_hits)])
            return FakeResp({"total": 40 if cited else 80, "data": data})
        assert "," in venue and COMMA not in venue      # the plain chunk, comma-joined
        data = [_rec("v1", "Neural Information Processing Systems"), _rec("v2", "AAAI")]
        return FakeResp({"total": 30 if cited else 200, "data": data})


def test_counter_sums_chunks_scales_comma_names_and_samples(tmp_path):
    s2 = FakeS2(comma_hits=1)
    c = BulkCounter(tmp_path, VENUES, http_get=s2, delay=0)
    out = c.count('"x y" | z')
    # n_v = 200 + 80 * 1/4; n_vc = 30 + 40 * 1/4.
    assert (out.n_v, out.n_c, out.n_vc) == (220, 500, 40)
    assert [r["paperId"] for r in out.venue_sample] == ["v1", "v2"]
    # citation-only sample drops records whose venue is in the list
    assert [r["paperId"] for r in out.citation_sample] == ["c2", "c3"]
    assert all(p["publicationDateOrYear"] == ":2024-09-30" for p in s2.calls)
    assert all(p["minCitationCount"] == "50" for p in s2.calls if "minCitationCount" in p)
    n = len(s2.calls)
    assert c.live_calls == n == 5
    c.count('"x y" | z')                       # every response is cached
    assert len(s2.calls) == n


def test_counter_skips_both_query_when_chunk_is_empty(tmp_path):
    s2 = FakeS2(comma_hits=0)
    out = BulkCounter(tmp_path, VENUES, http_get=s2, delay=0).count("q | r")
    assert out.n_v == 200 and out.n_vc == 30
    comma_cited = [p for p in s2.calls if p.get("venue") == COMMA and "minCitationCount" in p]
    assert comma_cited == []


class FakeCounter:
    live_calls = 0

    def count(self, q):
        from innovation.p2_forum.capacity import Counts
        k = 1 if "alpha" in q else 2
        return Counts(n_v=100 * k, n_c=50, n_vc=10,
                      venue_sample=[_rec(f"v{k}a", "V"), _rec(f"v{k}b", "V")],
                      citation_sample=[_rec(f"c{k}", "arXiv.org")])


class FakeTagger:
    def __init__(self, topics):
        self.topics = topics

    def label_many(self, texts, workers=8):
        # papers of topic 0's query: v1a->[0], v1b->[1], c1->[0]; topic 1's: all [1]
        table = {"t v1a": [0], "t v1b": [1], "t c1": [0]}
        return [table.get(t.split("\n")[0], [1]) for t in texts]


def test_estimator_measures_precision_with_current_list():
    llm = FakeLLM(['"alpha one" | two', '"beta one" | three'])
    topics = [{"name": "A", "definition": "a", "sources": ["AAAI"]},
              {"name": "B", "definition": "b", "sources": ["AAAI"]}]
    est = CapacityEstimator(counter=FakeCounter(), llm=llm, model="m",
                            tagger_factory=FakeTagger, log=lambda *_: None)(topics)
    a, b = est.records
    assert (a["p_v"], a["p_c"], a["sample_v"], a["sample_c"]) == (0.5, 1.0, 2, 1)
    assert a["capacity"] == pytest.approx(100 * 0.5 + 40 * 1.0)
    assert b["capacity"] == pytest.approx(200 * 1.0 + 40 * 1.0)
    assert a["query"] == '"alpha one" | two'
    assert est.titles[0] == ["t v1a", "t c1"]
    assert est.labels["v1b"] == [1]


def test_estimator_retries_a_bad_query_then_gives_up():
    llm = FakeLLM(["nope", '"alpha one" | two'])
    e = CapacityEstimator(counter=FakeCounter(), llm=llm, model="m",
                          tagger_factory=FakeTagger, log=lambda *_: None)
    assert e.query({"name": "A", "definition": "a"}) == '"alpha one" | two'
    assert "rejected" in llm.calls[1]["user"]
    with pytest.raises(RuntimeError, match="no valid query"):
        CapacityEstimator(counter=FakeCounter(), llm=FakeLLM(default="bad"), model="m",
                          tagger_factory=FakeTagger).query({"name": "A", "definition": "a"})



def test_citation_only_cache_key_depends_on_the_venue_list(tmp_path):
    s2 = FakeS2()
    BulkCounter(tmp_path, VENUES, http_get=s2, delay=0).count("q | r")
    n = len(s2.calls)
    BulkCounter(tmp_path, [VENUES[0], VENUES[2]], http_get=s2, delay=0).count("q | r")
    cited_only = [p for p in s2.calls[n:] if "venue" not in p]
    assert len(cited_only) == 1        # a different list cannot reuse the old sample
