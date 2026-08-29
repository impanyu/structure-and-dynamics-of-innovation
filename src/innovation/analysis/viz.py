"""Post-run visualization: agent read/write trajectories over a 2D map of the
idea space (initial corpus + generated ideas).

- The corpus embedding is projected once (PCA-50 -> t-SNE) and cached beside
  the data; every run reuses the same base map.
- Generated ideas cannot be transformed by t-SNE, so they are placed at the
  cosine-weighted average of their k nearest corpus points (standard trick).
- Trajectories come from events.jsonl: browse/sample_frontier = READ visits,
  generate = WRITE. Search actions touch many nodes and are omitted from the
  path (the follow-up browse shows where the agent actually went).
"""
import json
from pathlib import Path

import numpy as np

from innovation.experiments.events import load_events
from innovation.ideas.embed import load_embeddings


def project_corpus_2d(data_dir, cache_name: str = "proj_tsne.npy",
                      seed: int = 0) -> tuple[list[str], np.ndarray]:
    """(ids, (n,2) coords), cached on disk after the first computation."""
    ids, vecs = load_embeddings(data_dir)
    cache = Path(data_dir) / cache_name
    if cache.exists():
        return ids, np.load(cache)
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    reduced = PCA(n_components=50, random_state=seed).fit_transform(vecs)
    coords = TSNE(n_components=2, random_state=seed, init="pca",
                  perplexity=30).fit_transform(reduced).astype(np.float32)
    np.save(cache, coords)
    return ids, coords


def place_new_points(new_vecs: np.ndarray, corpus_vecs: np.ndarray,
                     corpus_coords: np.ndarray, k: int = 10) -> np.ndarray:
    """Place new embeddings on the corpus map via k-NN cosine-weighted average."""
    out = np.zeros((len(new_vecs), 2), dtype=np.float32)
    for i, v in enumerate(new_vecs):
        sims = corpus_vecs @ v
        top = np.argsort(-sims)[:k]
        w = np.clip(sims[top], 1e-6, None)
        out[i] = (corpus_coords[top] * w[:, None]).sum(0) / w.sum()
    return out


def extract_trajectories(events: list[dict]) -> dict[str, list[dict]]:
    """agent_id -> ordered [{step, node_id, kind: read|write}] visits."""
    traj: dict[str, list[dict]] = {}
    for e in events:
        res = e.get("result") or {}
        node = None
        kind = "read"
        if e["action"] in ("browse", "sample_frontier") and "node_id" in res:
            node = res["node_id"]
        elif e["action"] == "generate" and "node_id" in res:
            node, kind = res["node_id"], "write"
        if node:
            traj.setdefault(e["agent_id"], []).append(
                {"step": e["step"], "node_id": node, "kind": kind})
    return traj


def plot_run(run_dir, data_dir, embedder, out_name: str = "trajectories.png"):
    """Write <run_dir>/trajectories.png and trajectories.json; returns paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    events = load_events(run_dir / "events.jsonl")
    ids, coords = project_corpus_2d(data_dir)
    pos = dict(zip(ids, coords))
    _, corpus_vecs = load_embeddings(data_dir)

    gen_texts = {e["result"]["node_id"]: e["args"]["text"]
                 for e in events
                 if e["action"] == "generate" and "node_id" in (e.get("result") or {})}
    if gen_texts:
        new_ids = list(gen_texts)
        new_coords = place_new_points(embedder.encode([gen_texts[n] for n in new_ids]),
                                      corpus_vecs, coords)
        pos.update(zip(new_ids, new_coords))

    traj = extract_trajectories(events)
    (run_dir / "trajectories.json").write_text(json.dumps(traj, indent=1))

    fig, ax = plt.subplots(figsize=(14, 12))
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c="#d0d0d0", linewidths=0,
               rasterized=True, label="corpus (16k ideas)")
    cmap = plt.get_cmap("tab10")
    for i, (aid, visits) in enumerate(sorted(traj.items())):
        color = cmap(i % 10)
        pts = np.array([pos[v["node_id"]] for v in visits if v["node_id"] in pos])
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], "-", color=color, alpha=0.35, lw=1.0)
        reads = np.array([pos[v["node_id"]] for v in visits
                          if v["kind"] == "read" and v["node_id"] in pos])
        if len(reads):
            ax.scatter(reads[:, 0], reads[:, 1], s=14, color=color, alpha=0.8,
                       linewidths=0)
        writes = np.array([pos[v["node_id"]] for v in visits
                           if v["kind"] == "write" and v["node_id"] in pos])
        if len(writes):
            ax.scatter(writes[:, 0], writes[:, 1], s=180, color=color,
                       marker="*", edgecolors="black", linewidths=0.6,
                       label=f"{aid}: {len(writes)} ideas")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Agent read/write trajectories — {run_dir.name} "
                 f"(dots=reads, stars=generated ideas)")
    ax.set_xticks([]), ax.set_yticks([])
    out = run_dir / out_name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out, run_dir / "trajectories.json"
