import pandas as pd

from innovation.data.corpus import build_corpus, load_corpus, save_corpus


def make_work(wid, year, refs=(), abstract=True):
    return {
        "id": f"https://openalex.org/{wid}",
        "title": f"Paper {wid}",
        "publication_year": year,
        "abstract_inverted_index": {"hello": [0], "world": [1]} if abstract else None,
        "referenced_works": [f"https://openalex.org/{r}" for r in refs],
    }


def test_build_corpus_filters_and_keeps_within_corpus_edges():
    works = {
        "NeurIPS": [make_work("W1", 2020),
                    make_work("W2", 2021, refs=["W1", "W999"]),  # W999 external
                    make_work("W3", 2022, abstract=False)],      # dropped: no abstract
        "ICLR": [make_work("W4", 2021, refs=["W1", "W3"])],      # W3 dropped upstream
    }
    papers, edges = build_corpus(works)
    assert set(papers["paper_id"]) == {"W1", "W2", "W4"}
    assert papers.set_index("paper_id").loc["W2", "venue"] == "NeurIPS"
    assert papers.set_index("paper_id").loc["W1", "abstract"] == "hello world"
    got = {(r.src, r.dst) for r in edges.itertuples()}
    assert got == {("W2", "W1"), ("W4", "W1")}


def test_build_corpus_dedupes_papers_across_venues():
    works = {"A": [make_work("W1", 2020)], "B": [make_work("W1", 2020)]}
    papers, _ = build_corpus(works)
    assert len(papers) == 1


def test_save_and_load_roundtrip(tmp_path):
    works = {"A": [make_work("W1", 2020), make_work("W2", 2021, refs=["W1"])]}
    papers, edges = build_corpus(works)
    save_corpus(papers, edges, tmp_path)
    p2, e2 = load_corpus(tmp_path)
    pd.testing.assert_frame_equal(papers.reset_index(drop=True), p2)
    pd.testing.assert_frame_equal(edges.reset_index(drop=True), e2)
