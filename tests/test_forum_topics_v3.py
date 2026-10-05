import json

import pytest

from innovation.p2_forum.topics import load_topics
from innovation.p2_forum.topics_v3 import (
    VIEWS, apply_groups, build_topics, candidate_sets, draft_yaml, exact_duplicate_units, fallback_name,
    is_catch_all, load_areas, majority_groups, naming_prompt, normalize, parse_groups, parse_names,
    split_prefix, synonym_prompt, validate_partition, verified_groups, verify_prompt)

VENUES = [
    {"venue": "NeurIPS", "areas": ["Optimization (e.g., convex and non-convex)"]},
    {"venue": "ICML", "areas": ["Deep Learning (architectures, etc.)"]},
    {"venue": "ICLR", "areas": ["optimization", "learning theory"]},
    {"venue": "AAAI", "areas": ["ML: Optimization", "ML: Learning Theory",
                                "ML: Other Foundations of Machine Learning",
                                "CV: Applications", "NLP: Applications",
                                "DMKM: Conversational Systems for Recommendation & Retri"]},
    {"venue": "ACL", "areas": ["NLP Applications"]},
    {"venue": "CVPR", "areas": ["Self-& semi-& meta-& unsupervised learning", "Vision + graphics",
                                "Machine learning (other than deep learning)"]},
    {"venue": "ICCV", "areas": ["Self-, semi-, meta-, unsupervised learning", "Vision and graphics"]},
]


def _areas():
    return load_areas(VENUES)


def _kept(areas):
    return [i for i, a in enumerate(areas) if not is_catch_all(a)]


def test_split_prefix_and_normalize():
    assert split_prefix("ML: Clustering") == ("ML", "Clustering")
    assert split_prefix("Biometrics") == ("", "Biometrics")
    assert normalize("Self-& semi-& meta-& unsupervised learning") == \
        normalize("Self-, semi-, meta-, unsupervised learning")
    assert normalize("Vision + graphics") == normalize("Vision and graphics")
    assert normalize("Medical and biological vision, cell microscopy") == \
        normalize("Medical and biological vision; cell microscopy")


def test_load_areas_orders_by_venue_and_strips_aaai_prefix_only():
    areas = _areas()
    assert [a.venue for a in areas][:4] == ["NeurIPS", "ICML", "ICLR", "ICLR"]
    ml = next(a for a in areas if a.area == "ML: Optimization")
    assert (ml.domain, ml.keyword) == ("ML", "Optimization")
    acl = next(a for a in areas if a.venue == "ACL")
    assert (acl.domain, acl.keyword) == ("NLP", "NLP Applications")


def test_load_areas_rejects_missing_venue():
    with pytest.raises(ValueError, match="expected venues"):
        load_areas(VENUES[:-1])


def test_catch_alls_are_only_literal_other_bins():
    dropped = [a.area for a in _areas() if is_catch_all(a)]
    assert dropped == ["ML: Other Foundations of Machine Learning"]


def test_exact_duplicates_merge_within_a_domain_only():
    areas = _areas()
    units = exact_duplicate_units(areas, _kept(areas))
    as_text = [{areas[i].area for i in u} for u in units]
    assert {"optimization", "ML: Optimization"} in as_text
    assert {"learning theory", "ML: Learning Theory"} in as_text
    assert {"Self-& semi-& meta-& unsupervised learning",
            "Self-, semi-, meta-, unsupervised learning"} in as_text
    assert {"Vision + graphics", "Vision and graphics"} in as_text
    # identical keyword, different domains: not auto-merged
    assert {"CV: Applications"} in as_text and {"NLP: Applications"} in as_text
    # NeurIPS' annotated "Optimization (e.g. ...)" is not an exact duplicate
    assert {"Optimization (e.g., convex and non-convex)"} in as_text
    assert sum(len(u) for u in units) == len(_kept(areas))
    assert [min(u) for u in units] == sorted(min(u) for u in units)


@pytest.mark.parametrize("groups, msg", [
    ([[0], [1]], r"missing.*\[2\]"),
    ([[0, 1], [1, 2]], "more than once"),
    ([[0, 1, 2, 3]], "out of range"),
    ([[0], [], [1, 2]], "non-empty"),
    ({"a": 1}, "list of lists"),
])
def test_validate_partition_rejects_bad_groups(groups, msg):
    with pytest.raises(ValueError, match=msg):
        validate_partition(groups, 3)


def test_parse_groups_completes_singletons_from_merge_sets():
    assert parse_groups("```json\n[[0, 2]]\n```", 3) == [[0, 2], [1]]
    reasoning = "Items [0] and [2] are one area.\n<answer>[[0, 2]]</answer>"
    assert parse_groups(reasoning, 3) == [[0, 2], [1]]
    assert parse_groups("<answer>[]</answer>", 2) == [[0], [1]]


@pytest.mark.parametrize("reply, msg", [
    ("no idea", "no JSON"),
    ("<answer>[[0, 1], [1, 2]]</answer>", "more than once"),
    ("<answer>[[0, 5]]</answer>", "out of range"),
    ("<answer>[[0]]</answer>", "at least two"),
    ("<answer>[0, 1]</answer>", "list of lists"),
])
def test_parse_groups_rejects_bad_merge_sets(reply, msg):
    with pytest.raises(ValueError, match=msg):
        parse_groups(reply, 3)


def test_apply_groups_keeps_every_absorbed_area_in_stable_order():
    units = [[0], [3, 5], [1], [2]]
    merged = apply_groups(units, [[1, 2], [0], [3]])
    assert merged == [[0], [1, 3, 5], [2]]


def test_majority_groups_keeps_only_agreed_merges():
    votes = [[[0, 1], [2, 3], [4]],
             [[0, 1], [2], [3], [4]],
             [[0, 1, 4], [2], [3]]]
    assert majority_groups(votes, 5, min_votes=2) == [[0, 1], [2], [3], [4]]
    assert majority_groups(votes, 5, min_votes=1) == [[0, 1, 4], [2, 3]]
    with pytest.raises(ValueError, match="missing"):
        majority_groups([[[0]]], 2, min_votes=1)


def test_candidates_are_union_of_votes_and_verification_maps_back():
    votes = [[[0, 1], [2], [3], [4]],
             [[0], [1, 4], [2], [3]],
             [[0], [1], [2], [3], [4]]]
    cands = candidate_sets(votes, 5)
    assert cands == [[0, 1, 4]]
    # verifier keeps 1+4 (local 1, 2) and splits off 0
    assert verified_groups(5, cands, [[[0], [1, 2]]]) == [[0], [1, 4], [2], [3]]
    with pytest.raises(ValueError, match="missing"):
        verified_groups(5, cands, [[[0], [1]]])


def test_verify_prompt_renumbers_the_candidate_items():
    areas = _areas()
    units = exact_duplicate_units(areas, _kept(areas))
    cv = next(k for k, u in enumerate(units) if areas[u[0]].area == "CV: Applications")
    nlp = next(k for k, u in enumerate(units) if areas[u[0]].area == "NLP: Applications")
    p = verify_prompt([cv, nlp], units, areas)
    assert "0. AAAI [Computer Vision]: Applications" in p
    assert "1. AAAI [Speech & Natural Language Processing]: Applications" in p
    assert "0..1" in p and "Attempt" not in p
    assert "Attempt 2" in verify_prompt([cv, nlp], units, areas, attempt=2, feedback="x")


def test_synonym_prompt_views_list_the_same_numbered_items():
    areas = _areas()
    units = exact_duplicate_units(areas, _kept(areas))
    prompts = [synonym_prompt(units, areas, "t", view=v) for v in range(len(VIEWS))]
    assert len(set(prompts)) == 3
    line = lambda p: sorted(l for l in p.splitlines() if l[:1].isdigit() and ". " in l)
    assert line(prompts[0]) == line(prompts[1]) == line(prompts[2])


def test_synonym_prompt_numbers_items_and_varies_with_attempt():
    areas = _areas()
    units = exact_duplicate_units(areas, _kept(areas))
    p1 = synonym_prompt(units, areas, "test")
    assert "AAAI [Computer Vision]: Applications" in p1
    assert "ACL [Speech & Natural Language Processing venue]: NLP Applications" in p1
    assert f"0..{len(units) - 1}" in p1
    # alphabetical listing: CV's and NLP's "Applications" sit next to each other
    cv = p1.index("AAAI [Computer Vision]: Applications")
    nlp = p1.index("AAAI [Speech & Natural Language Processing]: Applications")
    assert "Optimization" not in p1[min(cv, nlp):max(cv, nlp)]
    p2 = synonym_prompt(units, areas, "test", attempt=2, feedback="index 3 missing")
    assert p2 != p1 and "Attempt 2" in p2 and "index 3 missing" in p2
    assert "different domains" in synonym_prompt(units, areas, "x", cross=True)
    assert "different domains" not in p1


def test_parse_names_validates_ids_uniqueness_and_prefixes():
    ok = json.dumps([{"id": 4, "name": "Optimization", "definition": "In: x. Out: y."},
                     {"id": 7, "name": "Learning Theory", "definition": "d"}])
    assert parse_names(ok, [4, 7], taken=[]) == {4: ("Optimization", "In: x. Out: y."),
                                                 7: ("Learning Theory", "d")}
    with pytest.raises(ValueError, match="duplicate"):
        parse_names(ok, [4, 7], taken=["optimization"])
    with pytest.raises(ValueError, match="missing"):
        parse_names(ok, [4, 7, 9], taken=[])
    with pytest.raises(ValueError, match="prefix"):
        parse_names(json.dumps([{"id": 1, "name": "ML: Clustering", "definition": "d"}]),
                    [1], taken=[])
    with pytest.raises(ValueError, match="mentions a venue"):
        parse_names(json.dumps([{"id": 1, "name": "Biometrics (CVPR/ICCV)", "definition": "d"}]),
                    [1], taken=[])
    with pytest.raises(ValueError, match="mentions a venue"):
        parse_names(json.dumps([{"id": 1, "name": "Biometrics (Vision Venue)", "definition": "d"}]),
                    [1], taken=[])
    with pytest.raises(ValueError, match="no definition"):
        parse_names(json.dumps([{"id": 1, "name": "Clustering", "definition": " "}]),
                    [1], taken=[])


def test_truncated_area_gets_clean_name_but_verbatim_provenance(tmp_path):
    areas = _areas()
    units = exact_duplicate_units(areas, _kept(areas))
    trunc = next(k for k, u in enumerate(units)
                 if areas[u[0]].area.endswith("& Retri"))
    p = naming_prompt([(trunc, units[trunc])], areas, taken=[])
    assert "Conversational Systems for Recommendation & Retri" in p and "cut off" in p
    names = {k: (f"T{k}", f"d{k}") for k in range(len(units))}
    names[trunc] = ("Conversational Systems for Recommendation & Retrieval", "d")
    topics = build_topics(units, areas, names)
    t = next(t for t in topics if t["name"].endswith("& Retrieval"))
    assert t["sources"] == [{"venue": "AAAI",
                             "area": "DMKM: Conversational Systems for Recommendation & Retri"}]


def test_draft_yaml_loads_with_no_fixed_count_and_keeps_provenance(tmp_path):
    areas = _areas()
    kept = _kept(areas)
    units = exact_duplicate_units(areas, kept)
    names = {k: (f"Topic {k}", f"definition {k}") for k in range(len(units))}
    topics = build_topics(units, areas, names)
    path = tmp_path / "topics-v3.draft.yaml"
    path.write_text(draft_yaml(topics, [a for a in areas if is_catch_all(a)]))
    text = path.read_text()
    assert text.startswith("# DRAFT (synonym-only dedupe; umbrellas kept).")
    assert "#   AAAI: ML: Other Foundations of Machine Learning" in text
    loaded = load_topics(path, expected=None)
    assert len(loaded) == len(units)
    all_sources = [s for t in loaded for s in t.sources]
    assert sorted((s["venue"], s["area"]) for s in all_sources) == \
        sorted((areas[i].venue, areas[i].area) for i in kept)
    # stable order: first topic comes from NeurIPS, the last from ICCV/CVPR
    assert loaded[0].sources[0]["venue"] == "NeurIPS"


def test_build_topics_rejects_case_insensitive_duplicate_names():
    areas = _areas()
    units = [[0], [1]]
    with pytest.raises(ValueError, match="duplicate"):
        build_topics(units, areas, {0: ("Optimization", "d"), 1: ("optimization", "d")})


def test_fallback_name_is_unique_and_cites_source():
    areas = _areas()
    i = next(k for k, a in enumerate(areas) if a.area == "CV: Applications")
    name, definition = fallback_name([i], areas, taken=["applications"])
    assert name == "Applications (Computer Vision)"
    assert "CV: Applications" in definition
