"""Round-robin simulation runner (spec §3.4-3.5): stigmergy via the shared network."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from innovation.agents.baselines import (NoNavLLMPolicy,
                                         PreferentialAttachmentPolicy)
from innovation.agents.llm_agent import LLMAgentPolicy
from innovation.experiments.env import Environment
from innovation.experiments.events import EventLog


@dataclass
class RunConfig:
    run_id: str
    seed: int
    total_steps: int
    generation_budget: int  # GLOBAL: the fairness control of spec §3.4
    agents: list[dict] = field(default_factory=list)


def build_policy(spec: dict, *, llm, model, graph, rng):
    kind = spec["policy"]
    if kind == "llm":
        return LLMAgentPolicy(llm=llm, model=model,
                              memory_size=spec.get("memory_size", 6),
                              persona=spec.get("persona", ""))
    if kind == "nonav":
        return NoNavLLMPolicy(llm=llm, model=model, graph=graph, rng=rng,
                              k=spec.get("k", 3))
    if kind == "pa":
        return PreferentialAttachmentPolicy(graph=graph, rng=rng, m=spec.get("m", 3))
    raise ValueError(f"unknown policy kind: {kind}")


def run_simulation(cfg: RunConfig, *, graph, index, embedder, llm, model,
                   out_dir) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    env = Environment(run_id=cfg.run_id, graph=graph, index=index,
                      embedder=embedder, event_log=EventLog(run_dir / "events.jsonl"),
                      rng=rng, generation_budget=cfg.generation_budget)
    policies = {a["agent_id"]: build_policy(a, llm=llm, model=model,
                                            graph=graph, rng=rng)
                for a in cfg.agents}
    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}

    for step in range(cfg.total_steps):
        agent_id = order[step % len(order)]
        obs = {"step": step, "last_result": last_result[agent_id]}
        action = policies[agent_id].act(obs)
        last_result[agent_id] = env.execute(agent_id, step, action)

    return {"run_id": cfg.run_id, "steps": cfg.total_steps,
            "generated": env.generated_ids()}
