"""Paper -> one-paragraph idea (spec §3.2). Fixed template; caching lives in CachedLLM."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from innovation.llm import LLM

SUMMARY_SYSTEM = (
    "You summarize research papers into a single self-contained idea paragraph.")

SUMMARY_TEMPLATE = """Summarize the following paper as ONE paragraph of 3-4 sentences \
covering: (1) the problem it addresses, (2) the key insight, (3) the method. \
Write only the paragraph, no preamble.

Title: {title}

Abstract: {abstract}"""


def summarize_paper(llm: LLM, *, model: str, title: str, abstract: str) -> str:
    user = SUMMARY_TEMPLATE.format(title=title, abstract=abstract)
    return llm.complete(model=model, system=SUMMARY_SYSTEM, user=user, max_tokens=400).strip()


def summarize_corpus(llm: LLM, papers: pd.DataFrame, *, model: str,
                     workers: int = 1) -> pd.DataFrame:
    records = list(papers.itertuples())

    def one(p):
        idea = summarize_paper(llm, model=model, title=p.title, abstract=p.abstract)
        return {"paper_id": p.paper_id, "idea_text": idea,
                "year": p.year, "venue": p.venue}

    if workers <= 1:
        rows = [one(p) for p in records]
    else:  # order-preserving parallel map; disk cache writes are per-file, safe
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(one, records))
    return pd.DataFrame(rows, columns=["paper_id", "idea_text", "year", "venue"])


def save_ideas(ideas: pd.DataFrame, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ideas.to_parquet(out / "ideas.parquet", index=False)


def load_ideas(out_dir) -> pd.DataFrame:
    return pd.read_parquet(Path(out_dir) / "ideas.parquet")
