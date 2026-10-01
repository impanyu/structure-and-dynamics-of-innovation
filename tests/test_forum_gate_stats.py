import importlib.util
from pathlib import Path


def load(name="gate_stats"):
    spec = importlib.util.spec_from_file_location(name, Path(f"scripts/p2_forum/{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_gate_counts():
    ev = [{"action": "search", "result": {"gate": "query"}},
          {"action": "search", "result": {"hits": []}},
          {"action": "generate", "result": {"gate": "post"}},
          {"action": "generate", "result": {"node_id": "gen:r:0", "dropped_cites": ["x", "y"]}},
          {"action": "add_links", "result": {"gate": "link"}},
          {"action": "browse", "result": {"gate": "scope"}}]
    c = load().gate_counts(ev)
    assert c["query"] == 1 and c["post"] == 1 and c["link"] == 1 and c["scope"] == 1
    assert c["dropped_cites"] == 2 and c["posts"] == 1 and c["searches"] == 2
    assert c["rejection_rate"] == 0.5


def test_tag_quality_helpers_import_clean():
    m = load("tag_quality")
    assert m.jaccard([1, 2], [2, 3]) == 1 / 3
    h = m.histogram({"a": [1], "b": [1, 2, 3, 4, 5]})
    assert h["hist"] == {1: 1, 5: 1} and h["share_5"] == 0.5
