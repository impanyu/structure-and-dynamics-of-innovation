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


def test_summarize_corpus_parallel_preserves_order():
    import threading

    class SlowFake:
        def __init__(self):
            self.lock = threading.Lock()
            self.count = 0

        def complete(self, *, model, system, user, max_tokens=1024):
            # echo the title back so order is verifiable
            title = user.split("Title: ")[1].split("\n")[0]
            with self.lock:
                self.count += 1
            return f"idea for {title}"

    papers = pd.DataFrame([
        {"paper_id": f"W{i}", "title": f"T{i}", "abstract": f"A{i}",
         "year": 2020, "venue": "V"} for i in range(20)])
    llm = SlowFake()
    ideas = summarize_corpus(llm, papers, model="m", workers=8)
    assert llm.count == 20
    assert list(ideas["paper_id"]) == [f"W{i}" for i in range(20)]
    assert list(ideas["idea_text"]) == [f"idea for T{i}" for i in range(20)]
