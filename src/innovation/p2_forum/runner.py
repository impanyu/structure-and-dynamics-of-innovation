"""Round-robin runner for paper 2 (spec §6).

Each agent draws k topics uniformly at random from the pool. Draws are seeded
and recorded in run_meta.json. Agents may share topics — with N agents and a
128-topic pool that is unavoidable above a certain N, and it is the substrate
collaboration needs, not a defect.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from innovation.core.events import EventLog
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.agent import ForumAgentPolicy
from innovation.p2_forum.env import ForumEnvironment
from innovation.p2_forum.workspace import Workspace


@dataclass
class ForumRunConfig:
    run_id: str
    seed: int
    total_steps: int
    agents: list[dict] = field(default_factory=list)
    topic_pool: list[str] = field(default_factory=list)
    generation_budget: int | None = None


def draw_topics(agents: list[dict], pool: list[str], rng) -> dict[str, list[str]]:
    """k distinct topics per agent, drawn independently. Different agents may
    draw the same topic; that overlap is what makes collaboration possible."""
    if not pool:
        raise ValueError("no topic_pool given")
    out: dict[str, list[str]] = {}
    for a in agents:
        k = int(a.get("k_topics", 1))
        if k > len(pool):
            raise ValueError(
                f"k_topics={k} exceeds the {len(pool)}-topic pool")
        picks = rng.choice(len(pool), size=k, replace=False)
        out[a["agent_id"]] = [pool[int(i)] for i in picks]
    return out


def run_forum(cfg: ForumRunConfig, *, corpus, corpus_index, embedder, llm,
              model, out_dir) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assignments = draw_topics(cfg.agents, cfg.topic_pool, rng)
    # run_meta is written BEFORE driving: an interrupted run must stay
    # reproducible from its recorded draws rather than re-sampling them.
    (run_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": cfg.run_id, "seed": cfg.seed, "arch": "p2_forum",
         "topic_assignments": assignments}, indent=1))

    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=cfg.run_id)
    env = ForumEnvironment(run_id=cfg.run_id, workspace=ws,
                           event_log=EventLog(run_dir / "events.jsonl"),
                           rng=rng,
                           generation_budget=cfg.generation_budget,
                           allow_jump=all(a.get("allow_jump", True)
                                          for a in cfg.agents),
                           allow_search=all(a.get("allow_search", True)
                                            for a in cfg.agents))

    policies = {
        a["agent_id"]: ForumAgentPolicy(
            llm=llm, model=model, topics=assignments[a["agent_id"]],
            memory_size=a.get("memory_size", 20),
            identity=f"{cfg.run_id}:{a['agent_id']}",
            total_steps=cfg.total_steps)
        for a in cfg.agents}

    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}
    for step in range(cfg.total_steps):
        agent_id = order[step % len(order)]
        action = policies[agent_id].act(
            {"step": step, "last_result": last_result[agent_id]})
        last_result[agent_id] = env.execute(agent_id, step, action)

    return {"run_id": cfg.run_id, "steps": cfg.total_steps,
            "generated": env.generated_ids(),
            "topic_assignments": assignments}
