import json

import pytest
import yaml

from innovation.p2_forum.topics import (Topic, backbone_prompt, consolidation_prompt,
                                        load_topics, parse_topic_list)

VENUES = [{"venue": "NeurIPS", "areas": ["Deep learning", "Theory"]},
          {"venue": "ACL", "areas": ["Machine translation"]}]


def _topics(n):
    return [{"name": f"T{i}", "definition": f"d{i}", "sources": ["NeurIPS"]}
            for i in range(n)]


def test_prompt_lists_every_area_with_its_venue():
    p = consolidation_prompt(VENUES, n=128)
    assert "NeurIPS: Deep learning" in p and "ACL: Machine translation" in p
    assert "exactly 128" in p


def test_backbone_prompt_groups_aaai_keywords_and_lists_other_areas():
    venues = VENUES + [{"venue": "AAAI", "areas": ["ML: Clustering", "ML: Ensemble Methods",
                                                    "CV: Biometrics"]}]
    p = backbone_prompt(venues, n=128)
    assert "[ML] Clustering; Ensemble Methods" in p and "[CV] Biometrics" in p
    assert "NeurIPS: Deep learning" in p and "ACL: Machine translation" in p
    assert "AAAI: ML: Clustering" not in p
    assert "exactly 128" in p


def test_parse_accepts_a_valid_list_inside_fences():
    reply = "```json\n" + json.dumps(_topics(4)) + "\n```"
    out = parse_topic_list(reply, n=4, venue_names={"NeurIPS", "ACL"})
    assert [t["name"] for t in out] == ["T0", "T1", "T2", "T3"]


@pytest.mark.parametrize("bad, msg", [
    (_topics(3), "expected 4"),
    (_topics(3) + [{"name": "T0", "definition": "x", "sources": ["ACL"]}], "duplicate"),
    (_topics(3) + [{"name": "T9", "definition": "", "sources": ["ACL"]}], "definition"),
    (_topics(3) + [{"name": "T9", "definition": "x", "sources": ["KDD"]}], "unknown venue"),
])
def test_parse_rejects_bad_lists(bad, msg):
    with pytest.raises(ValueError, match=msg):
        parse_topic_list(json.dumps(bad), n=4, venue_names={"NeurIPS", "ACL"})


def test_load_topics_assigns_ids_by_position(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump({"topics": _topics(3)}))
    ts = load_topics(f, expected=3)
    assert ts[2] == Topic(id=2, name="T2", definition="d2", sources=["NeurIPS"])


def test_load_topics_refuses_wrong_count(tmp_path):
    f = tmp_path / "t.yaml"
    f.write_text(yaml.safe_dump({"topics": _topics(3)}))
    with pytest.raises(ValueError, match="128"):
        load_topics(f)
