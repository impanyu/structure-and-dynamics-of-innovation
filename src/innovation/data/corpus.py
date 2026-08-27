"""Turn raw OpenAlex works into the canonical papers/edges tables (spec §3.1)."""
from pathlib import Path

import pandas as pd

from innovation.data.openalex import reconstruct_abstract


def _short_id(openalex_url: str) -> str:
    return openalex_url.rsplit("/", 1)[-1]


def build_corpus(works_by_venue: dict[str, list[dict]]):
    rows, refs_by_paper = [], {}
    seen: set[str] = set()
    for venue, works in works_by_venue.items():
        for w in works:
            pid = _short_id(w["id"])
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            if not abstract or pid in seen:
                continue
            seen.add(pid)
            rows.append({"paper_id": pid, "title": w.get("title") or "",
                         "abstract": abstract, "year": w["publication_year"],
                         "venue": venue})
            refs_by_paper[pid] = [_short_id(r) for r in w.get("referenced_works", [])]
    papers = pd.DataFrame(rows, columns=["paper_id", "title", "abstract", "year", "venue"])
    in_corpus = set(papers["paper_id"])
    edge_rows = [{"src": pid, "dst": ref}
                 for pid, refs in refs_by_paper.items()
                 for ref in refs if ref in in_corpus]
    edges = pd.DataFrame(edge_rows, columns=["src", "dst"])
    return papers, edges


def save_corpus(papers: pd.DataFrame, edges: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    papers.to_parquet(out / "papers.parquet", index=False)
    edges.to_parquet(out / "edges.parquet", index=False)


def load_corpus(out_dir):
    out = Path(out_dir)
    return (pd.read_parquet(out / "papers.parquet"),
            pd.read_parquet(out / "edges.parquet"))
