import json

import pytest

from innovation.p2_forum.coverage import (allocate, coverage, coverage_table, labels_histogram,
                                          load_sample_labels, sample_stratum)
from innovation.p2_forum.topics import Topic


def test_allocate_proportional_largest_remainder_sums_to_n():
    sizes = {"a": 500, "b": 300, "c": 200}
    assert allocate(sizes, 10) == {"a": 5, "b": 3, "c": 2}
    # quotas 3.33/3.33/3.33 -> floors 3,3,3; one leftover breaks the tie by key order
    out = allocate({"x": 100, "y": 100, "z": 100}, 10)
    assert sum(out.values()) == 10 and sorted(out.values()) == [3, 3, 4]


def test_allocate_largest_remainder_picks_biggest_fractions():
    # quotas: a=6.5, b=2.7, c=0.8 -> floors 6,2,0 (8); two leftovers to c (.8) then b (.7)
    assert allocate({"a": 650, "b": 270, "c": 80}, 10) == {"a": 6, "b": 3, "c": 1}


def test_allocate_zero_size_strata_get_zero():
    out = allocate({"a": 10, "empty": 0, "b": 30}, 8)
    assert out == {"a": 2, "empty": 0, "b": 6}


def test_allocate_rejects_n_above_population():
    with pytest.raises(ValueError):
        allocate({"a": 2, "b": 1}, 4)


def _recs(n, prefix="p"):
    return [{"paperId": f"{prefix}{i:03d}"} for i in range(n)]


def test_sample_stratum_deterministic_and_order_independent():
    recs = _recs(50)
    a = sample_stratum(recs, 7, seed=0, stratum="NeurIPS|2023")
    b = sample_stratum(list(reversed(recs)), 7, seed=0, stratum="NeurIPS|2023")
    assert [r["paperId"] for r in a] == [r["paperId"] for r in b]
    assert len({r["paperId"] for r in a}) == 7
    c = sample_stratum(recs, 7, seed=1, stratum="NeurIPS|2023")
    assert {r["paperId"] for r in a} != {r["paperId"] for r in c}


def test_sample_stratum_edge_sizes():
    assert sample_stratum(_recs(5), 0, seed=0, stratum="s") == []
    assert len(sample_stratum(_recs(5), 5, seed=0, stratum="s")) == 5


def test_coverage_of_union():
    sets = [{0, 1}, {2}, {3}, set()]
    assert coverage(sets, {0}) == 0.25
    assert coverage(sets, {1, 2}) == 0.5          # a paper counts once
    assert coverage(sets, {0, 1, 2, 3}) == 0.75
    assert coverage(sets, set()) == 0.0
    assert coverage([], {0}) == 0.0


def test_labels_histogram():
    assert labels_histogram([[0], [1, 2], [3, 4], None]) == {"1": 1, "2": 2}


def test_coverage_table_schema():
    topics = [Topic(0, "A", "a"), Topic(1, "B", "b"), Topic(2, "C", "c")]
    table = coverage_table(["p1", "p2", "p3"], [[0, 1], [0], None], topics)
    assert table["sample_size"] == 3
    assert table["labeled"] == 2 and table["unlabeled"] == 1
    assert table["labels_per_paper"] == {"1": 1, "2": 1}
    assert table["topics"] == [
        {"id": 0, "name": "A", "coverage": 1.0, "count": 2},
        {"id": 1, "name": "B", "coverage": 0.5, "count": 1},
        {"id": 2, "name": "C", "coverage": 0.0, "count": 0},
    ]
    json.dumps(table)


def test_load_sample_labels_keeps_labeled_only(tmp_path):
    lines = [{"paperId": "p1", "labels": [3, 1]}, {"paperId": "p2", "labels": None},
             {"paperId": "p3", "labels": [2]}]
    (tmp_path / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    ids, sets = load_sample_labels(tmp_path)
    assert ids == ["p1", "p3"]
    assert sets == [{1, 3}, {2}]
