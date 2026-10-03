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


def test_gate_counts_sums_filtered():
    ev = [{"action": "search", "result": {"hits": [], "filtered": {"scope": 2, "topic": 3}}},
          {"action": "browse", "result": {"filtered": {"scope": 1, "topic": 4}}},
          {"action": "search", "result": {"gate": "query"}}]
    c = load().gate_counts(ev)
    assert c["filtered_scope"] == 3 and c["filtered_topic"] == 7


def test_tag_quality_helpers_import_clean():
    m = load("tag_quality")
    assert m.jaccard([1, 2], [2, 3]) == 1 / 3
    h = m.histogram({"a": [1], "b": [1, 2, 3, 4, 5]})
    assert h["hist"] == {1: 1, 5: 1} and h["share_5"] == 0.5


def test_tag_quality_reconstructs_paper_text(tmp_path):
    from innovation.p2_forum.literature import Paper
    m = load("tag_quality")

    class Client:
        raws = {"a": {"paperId": "a", "title": "T", "abstract": "A long abstract."},
                "b": {"paperId": "b", "title": "Only title", "abstract": None}}

        def paper(self, pid):
            return self.raws.get(pid)

    def expect(r):
        return Paper(paper_id=r["paperId"], title=r["title"], abstract=r["abstract"] or "",
                     year=None, venue="", pub_date="", citations=0, branch="").text()

    out = m.sample_texts(Client(), ["a", "b", "gone"], 10)
    assert out == {"a": expect(Client.raws["a"]), "b": "Only title"}
    assert out["a"] == "T\n\nA long abstract."


def test_read_labels_first_wins(tmp_path):
    m = load("tag_quality")
    p = tmp_path / "l.jsonl"
    p.write_text('{"paper_id":"a","labels":[1]}\n{"paper_id":"a","labels":[2]}\n')
    assert m.read_labels(p) == {"a": [1]}


def test_gate_stats_main_skips_missing_events_and_no_json_for_explicit(tmp_path, monkeypatch):
    m = load()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "r1").mkdir()
    (tmp_path / "r2").mkdir()
    (tmp_path / "r2" / "events.jsonl").write_text('{"action":"search","result":{}}\n')
    m.main(["r1", "r2"])
    assert not (tmp_path / "runs/p2_forum/gate_stats.json").exists()


def test_gate_counts_counts_related_calls():
    ev = [{"action": "related", "result": {"related": []}},
          {"action": "related", "result": {"gate": "result"}},
          {"action": "search", "result": {"hits": []}}]
    c = load().gate_counts(ev)
    assert c["related"] == 2 and c["searches"] == 1 and c["result"] == 1


def test_gate_counts_openalex_fallbacks():
    ev = [{"action": "search", "result": {"hits": [], "source": "openalex", "unmapped": 2}},
          {"action": "search", "result": {"hits": [], "source": "openalex", "unmapped": 1}},
          {"action": "search", "result": {"hits": []}}]
    c = load().gate_counts(ev)
    assert c["openalex_fallbacks"] == 2 and c["openalex_unmapped"] == 3 and c["searches"] == 3
