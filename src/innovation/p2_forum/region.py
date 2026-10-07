"""Semantic regions: each agent's readable world is a nearest-neighbour ball
around a seed corpus paper (spec, REVISION 2026-10-06).

At coverage c over an N-paper corpus, an agent's region is the round(c*N)
corpus papers most similar (cosine) to its seed, the seed itself first. The
ball's radius is the similarity of its farthest member. A board post belongs
to a region by nearest-neighbour majority: iff at least NN_MAJORITY of its
NN_K nearest corpus papers are members (post_in_region).

Pure: numpy only, no I/O, no environment.
"""
from dataclasses import dataclass

import numpy as np

# Mixed into each agent's seed stream so it never coincides with the run's
# other per-agent streams (runner._DISPLAY_SALT, runner._NESTED_SALT) or the
# environment's rng, which is seeded from the bare run seed.
_SEED_SALT = 0x5EEDBA11

# A post is in a region iff at least NN_MAJORITY of its NN_K nearest corpus
# papers (cosine, over the vectors the regions are built from) are members.
# Posts are idea paragraphs and papers are abstracts, so a post is rarely as
# close to a seed as a paper is; with small, sharp balls a seed-similarity test
# refused posts whose neighbourhood lay inside the region. Judging a post by
# where its nearest papers fall does not depend on the ball's radius.
NN_K = 5
NN_MAJORITY = 3


@dataclass(frozen=True, eq=False)
class Region:
    seed_id: str
    seed_vec: np.ndarray          # unit length
    radius: float                 # cosine of the farthest member; -1.0 at full coverage
    members: frozenset[str]


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def build_region(seed_id: str, ids: list[str], vecs: np.ndarray,
                 coverage: float) -> Region:
    """The round(coverage * N) corpus papers nearest to `seed_id` (at least 1).

    Ranking is by cosine to the seed, the seed always first. Ties are broken
    by position in `ids` (a stable sort), so at the boundary the paper listed
    earlier is included; for a fixed seed and `ids` order the members at a
    smaller coverage are therefore a prefix of those at a larger one (nested).

    Full coverage (every paper a member) sets radius to -1.0, the minimum
    cosine, so the 100% condition gates no post either."""
    if not 0 < coverage <= 1:
        raise ValueError(f"coverage must be in (0, 1]; got {coverage}")
    row = {pid: i for i, pid in enumerate(ids)}
    if seed_id not in row:
        raise KeyError(f"seed {seed_id} is not a corpus paper")
    unit = _unit(vecs)
    seed_vec = unit[row[seed_id]]
    sims = unit @ seed_vec
    order = np.argsort(-sims, kind="stable")
    order = [row[seed_id]] + [int(i) for i in order if i != row[seed_id]]
    n = min(len(ids), max(1, round(coverage * len(ids))))
    top = order[:n]
    radius = -1.0 if n == len(ids) else float(min(sims[i] for i in top))
    return Region(seed_id=seed_id, seed_vec=seed_vec, radius=radius,
                  members=frozenset(ids[i] for i in top))


def post_in_region(region: Region, nearest_ids) -> bool:
    """Nearest-neighbour majority: are at least NN_MAJORITY of a post's NN_K
    nearest corpus paper ids members of the region?"""
    return sum(n in region.members for n in nearest_ids) >= NN_MAJORITY


def draw_seeds(n_agents: int, ids: list[str], seed: int) -> list[str]:
    """One distinct seed paper per agent. Agent i draws uniformly from the
    papers not already taken by agents 0..i-1, with its own stream
    default_rng(SeedSequence([seed, i, SALT])), so adding agents never
    changes the seeds of the existing ones."""
    if n_agents > len(ids):
        raise ValueError(f"{n_agents} agents need distinct seeds; the corpus has {len(ids)}")
    taken: list[str] = []
    used: set[str] = set()
    for i in range(n_agents):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), i, _SEED_SALT]))
        free = [p for p in ids if p not in used]
        pick = free[int(rng.integers(len(free)))]
        taken.append(pick)
        used.add(pick)
    return taken
