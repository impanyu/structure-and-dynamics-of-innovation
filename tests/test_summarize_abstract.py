"""cmd_summarize with corpus.idea_text: abstract (paper 2): no LLM."""
import pandas as pd

from innovation.cli import cmd_summarize
from innovation.core.data.corpus import load_corpus, save_corpus
from innovation.core.ideas.embed import FakeEmbedder, load_embeddings
from innovation.core.ideas.summarize import load_ideas


def _cfg(tmp_path, **corpus):
    return {"data_dir": str(tmp_path), "embedding_model": "m", "corpus": corpus}


def _save(tmp_path):
    papers = pd.DataFrame([
        {"paper_id": "a", "title": "T-a", "abstract": "Abs a.", "year": 2020, "venue": "V"},
        {"paper_id": "b", "title": "T-b", "abstract": "  ", "year": 2021, "venue": "V"},
        {"paper_id": "c", "title": "T-c", "abstract": "Abs c.", "year": 2022, "venue": "W"},
    ])
    edges = pd.DataFrame([{"src": "c", "dst": "a"}, {"src": "b", "dst": "a"},
                          {"src": "c", "dst": "b"}])
    save_corpus(papers, edges, tmp_path)


def test_abstract_ideas_no_llm_drops_abstractless(tmp_path, monkeypatch, capsys):
    _save(tmp_path)
    monkeypatch.setattr("innovation.cli.Embedder", lambda name: FakeEmbedder())
    monkeypatch.setattr("innovation.cli._llm", lambda cfg: (_ for _ in ()).throw(
        AssertionError("LLM must not be used")))
    cmd_summarize(_cfg(tmp_path, idea_text="abstract"))
    ideas = load_ideas(tmp_path)
    assert list(ideas["paper_id"]) == ["a", "c"]
    assert ideas["idea_text"][0] == "T-a\n\nAbs a."
    assert list(ideas.columns) == ["paper_id", "idea_text", "year", "venue"]
    papers, edges = load_corpus(tmp_path)
    assert list(papers["paper_id"]) == ["a", "c"]
    assert {(r.src, r.dst) for r in edges.itertuples()} == {("c", "a")}
    ids, vecs = load_embeddings(tmp_path)
    assert ids == ["a", "c"] and len(vecs) == 2
    assert "dropped_no_abstract=1" in capsys.readouterr().out


def test_default_path_still_uses_llm(tmp_path, monkeypatch):
    _save(tmp_path)
    monkeypatch.setattr("innovation.cli.Embedder", lambda name: FakeEmbedder())
    called = []

    def fake_summarize(llm, papers, **kw):
        called.append(len(papers))
        return pd.DataFrame({"paper_id": papers["paper_id"], "idea_text": "x",
                             "year": papers["year"], "venue": papers["venue"]})

    monkeypatch.setattr("innovation.cli.summarize_corpus", fake_summarize)
    monkeypatch.setattr("innovation.cli._llm", lambda cfg: None)
    cfg = _cfg(tmp_path)
    cfg["models"] = {"summarizer": "m"}
    cmd_summarize(cfg)
    assert called == [3]
