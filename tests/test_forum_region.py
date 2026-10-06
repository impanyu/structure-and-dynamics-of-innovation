# tests/test_forum_region.py
import numpy as np
import pytest

from innovation.p2_forum.region import build_region, contains_vec, draw_seeds


def _corpus(n=200, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return [f"p{i}" for i in range(n)], vecs


@pytest.mark.parametrize("c", [0.001, 0.01, 0.05, 0.1, 0.2, 0.33, 0.5, 1.0])
def test_member_count_is_exact(c):
    ids, vecs = _corpus()
    r = build_region("p7", ids, vecs, c)
    assert len(r.members) == max(1, round(c * len(ids)))
    assert "p7" in r.members and r.seed_id == "p7"


def test_regions_nest_across_coverages_for_one_seed():
    ids, vecs = _corpus()
    grid = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    regions = [build_region("p3", ids, vecs, c) for c in grid]
    for small, big in zip(regions, regions[1:]):
        assert small.members < big.members
        assert small.radius >= big.radius


def test_radius_separates_members_from_the_rest():
    ids, vecs = _corpus()
    r = build_region("p11", ids, vecs, 0.2)
    sims = vecs @ vecs[ids.index("p11")]
    inside = [s for i, s in zip(ids, sims) if i in r.members]
    outside = [s for i, s in zip(ids, sims) if i not in r.members]
    assert min(inside) == pytest.approx(r.radius)
    assert max(outside) <= r.radius
    assert all(contains_vec(r, vecs[ids.index(i)]) for i in r.members)


def test_contains_vec_normalizes_and_uses_the_radius():
    ids, vecs = _corpus()
    r = build_region("p0", ids, vecs, 0.1)
    assert contains_vec(r, 5.0 * vecs[0])                  # the seed's direction, any length
    assert not contains_vec(r, -vecs[0])                   # the opposite direction


def test_unnormalized_corpus_vectors_rank_by_cosine():
    ids = ["a", "b", "c"]
    vecs = np.array([[1, 0], [10, 1], [0.5, 0.5]], dtype=np.float32)   # b: long but near a
    r = build_region("a", ids, vecs, 2 / 3)
    assert r.members == frozenset({"a", "b"})


def test_full_coverage_is_everything_and_every_post_is_inside():
    ids, vecs = _corpus()
    r = build_region("p5", ids, vecs, 1.0)
    assert r.members == frozenset(ids)
    assert contains_vec(r, -vecs[ids.index("p5")])


def test_tiny_coverage_is_the_seed_alone():
    ids, vecs = _corpus()
    r = build_region("p9", ids, vecs, 1e-6)
    assert r.members == frozenset({"p9"})


def test_boundary_ties_are_broken_by_corpus_order():
    ids = ["s", "x", "y", "z"]
    vecs = np.array([[1, 0], [0, 1], [0, 1], [-1, 0]], dtype=np.float32)   # x and y tie
    r = build_region("s", ids, vecs, 0.5)
    assert r.members == frozenset({"s", "x"})
    r2 = build_region("s", ["s", "y", "x", "z"], vecs[[0, 2, 1, 3]], 0.5)
    assert r2.members == frozenset({"s", "y"})


def test_the_seed_is_always_a_member_even_with_a_duplicate_vector():
    ids = ["dup", "s", "z"]
    vecs = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    assert build_region("s", ids, vecs, 0.01).members == frozenset({"s"})


def test_bad_inputs_raise():
    ids, vecs = _corpus(n=10)
    for c in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            build_region("p0", ids, vecs, c)
    with pytest.raises(KeyError):
        build_region("nope", ids, vecs, 0.5)


def test_seeds_are_deterministic_distinct_and_prefix_stable():
    ids = [f"p{i}" for i in range(50)]
    a = draw_seeds(10, ids, seed=1)
    assert a == draw_seeds(10, ids, seed=1)
    assert len(set(a)) == 10 and set(a) <= set(ids)
    assert draw_seeds(4, ids, seed=1) == a[:4]          # adding agents keeps the first ones
    assert a != draw_seeds(10, ids, seed=2)


def test_seeds_stay_distinct_when_agents_fill_the_corpus():
    ids = [f"p{i}" for i in range(5)]
    assert sorted(draw_seeds(5, ids, seed=0)) == ids
    with pytest.raises(ValueError):
        draw_seeds(6, ids, seed=0)
