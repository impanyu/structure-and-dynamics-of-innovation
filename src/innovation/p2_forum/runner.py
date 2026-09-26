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

from innovation.core.events import EventLog, load_events
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


def _build_env(cfg: ForumRunConfig, *, run_dir, corpus, corpus_index, embedder,
               rng) -> ForumEnvironment:
    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=cfg.run_id)
    return ForumEnvironment(run_id=cfg.run_id, workspace=ws,
                            event_log=EventLog(run_dir / "events.jsonl"),
                            rng=rng,
                            generation_budget=cfg.generation_budget,
                            allow_jump=all(a.get("allow_jump", True)
                                           for a in cfg.agents),
                            allow_search=all(a.get("allow_search", True)
                                             for a in cfg.agents))


def _build_policies(cfg: ForumRunConfig, *, llm, model,
                    assignments: dict) -> dict[str, ForumAgentPolicy]:
    return {
        a["agent_id"]: ForumAgentPolicy(
            llm=llm, model=model, topics=assignments[a["agent_id"]],
            memory_size=a.get("memory_size", 20),
            identity=f"{cfg.run_id}:{a['agent_id']}",
            total_steps=cfg.total_steps)
        for a in cfg.agents}


def _drive(cfg: ForumRunConfig, env, policies, order, last_result,
           start_step: int) -> dict:
    for step in range(start_step, cfg.total_steps):
        agent_id = order[step % len(order)]
        action = policies[agent_id].act(
            {"step": step, "last_result": last_result[agent_id]})
        last_result[agent_id] = env.execute(agent_id, step, action)
    return {"run_id": cfg.run_id, "steps": cfg.total_steps,
            "generated": env.generated_ids()}


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
         "total_steps": cfg.total_steps,
         "topic_assignments": assignments}, indent=1))

    env = _build_env(cfg, run_dir=run_dir, corpus=corpus,
                     corpus_index=corpus_index, embedder=embedder, rng=rng)
    policies = _build_policies(cfg, llm=llm, model=model,
                               assignments=assignments)

    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}
    out = _drive(cfg, env, policies, order, last_result, 0)
    out["topic_assignments"] = assignments
    return out


def resume_forum(cfg: ForumRunConfig, *, corpus, corpus_index, embedder, llm,
                 model, out_dir) -> dict:
    """Continue an existing paper-2 run up to cfg.total_steps (raise it in the
    config to extend), mirroring p1_dial.runner.resume_simulation.

    The board is rebuilt by replaying the event log, so the workspace's id
    counter continues past every id already on disk — without this, a second
    `run_forum` at the same run_id appends `gen:<run>:0` a second time and the
    log stops replaying at all. The recorded topic draws are reused, never
    re-sampled. Each agent's rolling memory and last result are reconstructed.
    The resumed segment draws from a fresh rng stream seeded by
    (seed, start_step): reproducible, but not bit-identical to an
    uninterrupted run."""
    run_dir = Path(out_dir) / cfg.run_id
    events = load_events(run_dir / "events.jsonl")
    if not events:
        raise SystemExit(f"nothing to resume at {run_dir}")
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"cannot resume {run_dir}: run_meta.json is missing, so the "
            "recorded topic draws are gone and resuming would re-sample them")
    meta = json.loads(meta_path.read_text())
    assignments = meta.get("topic_assignments") or {}
    missing = [a["agent_id"] for a in cfg.agents
               if a["agent_id"] not in assignments]
    if missing:
        raise SystemExit(
            f"cannot resume {run_dir}: no recorded topic draw for {missing}")
    start_step = max(e["step"] for e in events) + 1
    if start_step >= cfg.total_steps:
        raise SystemExit(f"run already has {start_step} steps; "
                         f"raise total_steps beyond that to extend")

    rng = np.random.default_rng((cfg.seed, start_step))
    env = _build_env(cfg, run_dir=run_dir, corpus=corpus,
                     corpus_index=corpus_index, embedder=embedder, rng=rng)
    env.restore(events)

    policies = _build_policies(cfg, llm=llm, model=model,
                               assignments=assignments)
    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {}
    by_agent = {aid: [e for e in events if e["agent_id"] == aid] for aid in order}
    for aid in order:
        mine = by_agent[aid]
        pol = policies[aid]
        if mine:
            # entry i is (action_{i-1}, result_{i-1}); the newest result feeds
            # back through last_result instead.
            pol.memory.append(("(none)", "{}"))
            for e in mine[:-1]:
                pol.memory.append((e["action"], json.dumps(e["result"])[:1500]))
            pol._last_action = mine[-1]["action"]
        last_result[aid] = mine[-1]["result"] if mine else {}

    out = _drive(cfg, env, policies, order, last_result, start_step)
    out["topic_assignments"] = assignments
    out["resumed_from_step"] = start_step
    meta_path.write_text(json.dumps(
        {**meta, "total_steps": cfg.total_steps,
         "resumed_from": meta.get("resumed_from", []) + [start_step]}, indent=1))
    return out
