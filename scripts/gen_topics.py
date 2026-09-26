"""Generate a topic pool by clustering corpus idea embeddings and naming each
cluster with an LLM. Paper 2 uses K=128 (configs/p2_forum/topics-k128.yaml).

Run: uv run python scripts/gen_topics.py --k 128
"""
import argparse
import json
from pathlib import Path

import yaml

from innovation.core.ideas.topics import cluster_topics

NAME_SYSTEM = ("You name research topics. Given several paper-idea paragraphs "
               "from one cluster, reply with a single noun phrase of 4-8 words "
               "naming what they have in common. Reply with the phrase only.")


def name_cluster(llm, model: str, texts: list[str]) -> str:
    sample = "\n\n".join(t[:400] for t in texts[:8])
    reply = llm.complete(model=model, system=NAME_SYSTEM,
                         user=sample, max_tokens=60)
    return reply.strip().strip('."')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--data-dir", default="data/stage1")
    ap.add_argument("--config", default="configs/p2_forum/topics-k128.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from innovation.core.config import load_config, load_env
    from innovation.core.ideas.embed import load_embeddings
    from innovation.core.ideas.summarize import load_ideas
    from innovation.core.llm import CachedLLM, RoutedLLM

    load_env()
    cfg = load_config("configs/p1_dial/stage1.yaml")
    ids, vecs = load_embeddings(args.data_dir)
    ideas = load_ideas(args.data_dir)
    text_by_id = dict(zip(ideas["paper_id"], ideas["idea_text"]))

    clusters = cluster_topics(vecs, args.k, args.seed)
    llm = CachedLLM(RoutedLLM(), Path(args.data_dir) / "llm_cache")
    model = cfg["models"]["summarizer"]

    pool = []
    for rows in clusters:
        texts = [text_by_id[ids[i]] for i in rows]
        pool.append({"topic": name_cluster(llm, model, texts), "size": len(rows)})

    out = Path(args.data_dir) / f"topics_k{args.k}.json"
    out.write_text(json.dumps(pool, indent=1))
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump({"topics": pool}, open(args.config, "w"), allow_unicode=True)
    print(f"{len(pool)} topics -> {out} and {args.config}")


if __name__ == "__main__":
    main()
