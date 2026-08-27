from innovation.ideas.summarize import (load_ideas, save_ideas,
                                        summarize_corpus, summarize_paper)
from innovation.llm import FakeLLM
import pandas as pd


def papers_df():
    return pd.DataFrame([
        {"paper_id": "W1", "title": "T1", "abstract": "A1", "year": 2020, "venue": "NeurIPS"},
        {"paper_id": "W2", "title": "T2", "abstract": "A2", "year": 2021, "venue": "ICLR"},
    ])


def test_summarize_paper_fills_template():
    llm = FakeLLM(responses=["An idea."])
    out = summarize_paper(llm, model="m", title="Attention", abstract="We propose...")
    assert out == "An idea."
    assert "Attention" in llm.calls[0]["user"]
    assert "We propose..." in llm.calls[0]["user"]


def test_summarize_corpus_and_roundtrip(tmp_path):
    llm = FakeLLM(responses=["Idea one.", "Idea two."])
    ideas = summarize_corpus(llm, papers_df(), model="m")
    assert list(ideas.columns) == ["paper_id", "idea_text", "year", "venue"]
    assert list(ideas["idea_text"]) == ["Idea one.", "Idea two."]
    save_ideas(ideas, tmp_path)
    pd.testing.assert_frame_equal(load_ideas(tmp_path), ideas)
