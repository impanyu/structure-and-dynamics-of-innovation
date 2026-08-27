import numpy as np
import pytest

from innovation.eval.metrics import (aggregate_run, arxiv_population_filter,
                                     bridging, diversity, novelty,
                                     openalex_population_count, past_dup_flag,
                                     precision, recall,
                                     venue_population_filter)
from innovation.eval.search_verify import Verdict


def verdicts():
    return [Verdict("g0", True, {"paper_id": "P1"}),
            Verdict("g1", True, {"paper_id": "P1"}),   # same paper twice
            Verdict("g2", False, None),
            Verdict("g3", True, {"paper_id": "P2"})]


def test_precision_excludes_dup_flagged_from_numerator():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    # hits g0,g1 count; g3 is a hit but dup-flagged -> excluded; 2/4
    assert precision(verdicts(), dup) == pytest.approx(0.5)


def test_recall_counts_distinct_papers():
    assert recall(verdicts(), population=10) == pytest.approx(0.2)  # {P1,P2}/10


def test_population_filters_and_count():
    f = venue_population_filter(["S1", "S2"], "2025-01-01")
    assert f == "primary_location.source.id:S1|S2,from_publication_date:2025-01-01"
    f2 = arxiv_population_filter("S99", "2025-01-01", 10)
    assert f2 == ("primary_location.source.id:S99,"
                  "from_publication_date:2025-01-01,cited_by_count:>10")

    class FakeResponse:
        def json(self):
            return {"meta": {"count": 1234}, "results": []}

        def raise_for_status(self):
            pass

    n = openalex_population_count(f, mailto="a@b.c",
                                  http_get=lambda url, params=None: FakeResponse())
    assert n == 1234


def test_embedding_metrics():
    corpus = np.array([[1, 0], [0, 1]], dtype=np.float32)
    v_new = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)], dtype=np.float32)
    assert novelty(v_new, corpus) == pytest.approx(1 - np.cos(np.pi / 4), abs=1e-5)
    assert past_dup_flag(np.array([1, 0], dtype=np.float32), corpus, ceiling=0.95)
    assert not past_dup_flag(v_new, corpus, ceiling=0.95)
    assert diversity(np.array([[1, 0], [0, 1]], dtype=np.float32)) == pytest.approx(1.0)
    assert bridging(["a", "b", "c"], {"a": 0, "b": 0, "c": 2}) == 2


def test_aggregate_run():
    dup = {"g0": False, "g1": False, "g2": False, "g3": True}
    agg = aggregate_run(verdicts(), dup, population=10)
    assert agg == {"precision": 0.5, "recall": 0.2, "n_ideas": 4,
                   "n_hits": 3, "n_dup_flagged": 1}
