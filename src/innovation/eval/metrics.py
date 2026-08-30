"""Headline metric (precision/accuracy, spec §3.6) + process observables
(novelty, bridging, diversity)."""
import numpy as np


# --- headline (graded realization, levels 0-5, three recognition tiers) ---
def idea_levels(verdicts, dup_flags: dict[str, bool], tier: str = "tier1") -> list[int]:
    """Per-idea realization level at a cumulative recognition tier;
    near-duplicates of the past corpus score 0."""
    return [0 if dup_flags.get(v.idea_id, False) else v.best[tier]["level"]
            for v in verdicts]


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
    out = {"n_ideas": len(verdicts),
           "n_dup_flagged": sum(1 for f in dup_flags.values() if f)}
    for tier in ("tier1", "tier2", "tier3"):
        lv = idea_levels(verdicts, dup_flags, tier)
        n = len(lv)
        out[tier] = {"mean_level": (sum(lv) / n) if n else 0.0,
                     "acc_ge3": (sum(1 for x in lv if x >= 3) / n) if n else 0.0,
                     "acc_ge4": (sum(1 for x in lv if x >= 4) / n) if n else 0.0,
                     "acc_eq5": (sum(1 for x in lv if x == 5) / n) if n else 0.0}
    return out
