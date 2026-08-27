"""Headline metrics (precision/recall, spec §3.6) + process observables (novelty,
bridging, diversity)."""
import numpy as np
import requests

OPENALEX_WORKS = "https://api.openalex.org/works"


# --- headline ---
def precision(verdicts, dup_flags: dict[str, bool]) -> float:
    if not verdicts:
        return 0.0
    good = sum(1 for v in verdicts if v.hit and not dup_flags.get(v.idea_id, False))
    return good / len(verdicts)


def recall(verdicts, population: int) -> float:
    papers = {v.paper["paper_id"] for v in verdicts if v.hit}
    return len(papers) / population if population else 0.0


def venue_population_filter(source_ids: list[str], from_date: str) -> str:
    return (f"primary_location.source.id:{'|'.join(source_ids)},"
            f"from_publication_date:{from_date}")


def arxiv_population_filter(arxiv_source_id: str, from_date: str,
                            min_citations: int) -> str:
    return (f"primary_location.source.id:{arxiv_source_id},"
            f"from_publication_date:{from_date},cited_by_count:>{min_citations}")


def openalex_population_count(filter_str: str, *, mailto: str, http_get=None) -> int:
    """Recall denominator via a metadata count — no downloads (spec §3.6)."""
    http_get = http_get or requests.get
    r = http_get(OPENALEX_WORKS,
                 params={"filter": filter_str, "per-page": 1, "mailto": mailto})
    r.raise_for_status()
    return r.json()["meta"]["count"]


# --- process observables ---
def novelty(vec: np.ndarray, corpus_vecs: np.ndarray) -> float:
    return float(1.0 - np.max(corpus_vecs @ vec))


def past_dup_flag(vec: np.ndarray, corpus_vecs: np.ndarray,
                  ceiling: float = 0.95) -> bool:
    """Anti-plagiarism-of-the-past (spec §3.6): near-duplicate of the <=T corpus."""
    return bool(np.max(corpus_vecs @ vec) >= ceiling)


def bridging(cited_ids: list[str], communities: dict[str, int]) -> int:
    return len({communities[c] for c in cited_ids if c in communities})


def diversity(vecs: np.ndarray) -> float:
    if len(vecs) < 2:
        return 0.0
    sims = vecs @ vecs.T
    n = len(vecs)
    off_diag = sims[np.triu_indices(n, k=1)]
    return float(np.mean(1.0 - off_diag))


def aggregate_run(verdicts, dup_flags: dict[str, bool], population: int) -> dict:
    return {"precision": precision(verdicts, dup_flags),
            "recall": recall(verdicts, population),
            "n_ideas": len(verdicts),
            "n_hits": sum(1 for v in verdicts if v.hit),
            "n_dup_flagged": sum(1 for f in dup_flags.values() if f)}
