"""Headline metric (precision/accuracy, spec §3.6) + process observables
(novelty, bridging, diversity)."""
import numpy as np


# --- headline ---
def precision(verdicts, dup_flags: dict[str, bool]) -> float:
    if not verdicts:
        return 0.0
    good = sum(1 for v in verdicts if v.hit and not dup_flags.get(v.idea_id, False))
    return good / len(verdicts)


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


def aggregate_run(verdicts, dup_flags: dict[str, bool]) -> dict:
    return {"precision": precision(verdicts, dup_flags),
            "n_ideas": len(verdicts),
            "n_hits": sum(1 for v in verdicts if v.hit),
            "n_dup_flagged": sum(1 for f in dup_flags.values() if f)}
