"""Round-robin simulation runner (spec §3.4-3.5): stigmergy via the shared network."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from innovation.agents.baselines import (NoNavLLMPolicy,
                                         PreferentialAttachmentPolicy)
from innovation.agents.llm_agent import LLMAgentPolicy
from innovation.experiments.env import AgentScope, Environment
from innovation.experiments.events import EventLog


@dataclass
class RunConfig:
    run_id: str
    seed: int
    total_steps: int
    generation_budget: int  # GLOBAL: the fairness control of spec §3.4
    agents: list[dict] = field(default_factory=list)
    topic_pool: list[str] | None = None  # for read/write_topics: "random"


def build_policy(spec: dict, *, llm, model, graph, rng):
    kind = spec["policy"]
    if kind == "llm":
        return LLMAgentPolicy(llm=llm, model=model,
                              memory_size=spec.get("memory_size", 20),
                              persona=spec.get("persona", ""),
                              identity=spec.get("_identity", ""),
                              total_steps=spec.get("_total_steps", 0))
    if kind == "nonav":
        return NoNavLLMPolicy(llm=llm, model=model, graph=graph, rng=rng,
                              k=spec.get("k", 3))
    if kind == "pa":
        return PreferentialAttachmentPolicy(graph=graph, rng=rng, m=spec.get("m", 3))
    raise ValueError(f"unknown policy kind: {kind}")


def _mass_radii(anchors, corpus_vecs, k: int):
    """Per-anchor radius = cosine distance to the k-th nearest corpus paper,
    so each anchor's region holds exactly k corpus papers (equal mass)."""
    sims = corpus_vecs @ anchors.T                     # (n_corpus, m)
    kth = np.partition(sims, -k, axis=0)[-k]           # k-th largest per anchor
    return 1.0 - kth


def build_scope(spec: dict, embedder=None, corpus_vecs=None) -> AgentScope | None:
    """Agent behavioral profile from its config dict (spec §3.4-3.5). Each of
    read/write uses ONE mechanism: community ids (read_communities /
    write_communities) or a semantic region (read_topics + read_radius /
    write_topics + write_radius; topics are embedded as anchors). allow_jump
    defaults to True."""
    read = spec.get("read_communities")
    write = spec.get("write_communities")
    read_topics = spec.get("read_topics")
    write_topics = spec.get("write_topics")
    if read_topics == "random" or write_topics == "random":
        raise ValueError("unresolved 'random' topic sentinel reached build_scope")
    if read is not None and read_topics:
        raise ValueError("read scope: give communities OR topics, not both")
    if write is not None and write_topics:
        raise ValueError("write scope: give communities OR topics, not both")
    allow_jump = spec.get("allow_jump", True)
    if (read is None and write is None and not read_topics
            and not write_topics and allow_jump):
        return None
    read_anchors = embedder.encode(read_topics) if read_topics else None
    write_anchors = embedder.encode(write_topics) if write_topics else None

    def radii(anchors, mass_key, radius_key):
        mass = spec.get(mass_key)
        if anchors is not None and mass:
            if corpus_vecs is None:
                raise ValueError(f"{mass_key} requires corpus_vecs")
            return _mass_radii(anchors, corpus_vecs, int(mass))
        return spec.get(radius_key, 0.3)

    return AgentScope(
        read=set(read) if read is not None else None,
        write=set(write) if write is not None else None,
        allow_jump=allow_jump,
        read_anchors=read_anchors,
        read_radius=radii(read_anchors, "read_mass", "read_radius"),
        write_anchors=write_anchors,
        write_radius=radii(write_anchors, "write_mass", "write_radius"))


def _resolve_random_topics(agents: list[dict], pool, rng) -> dict:
    """Replace read/write_topics: "random" with one pool topic per agent —
    distinct within the run, drawn with the run's seeded rng. A specialist
    with both random reads and writes gets the SAME topic for both."""
    specs = [a for a in agents
             if a.get("read_topics") == "random" or a.get("write_topics") == "random"]
    if not specs:
        return {}
    if not pool:
        raise ValueError("agents request random topics but no topic_pool given")
    picks = rng.choice(len(pool), size=len(specs), replace=False)
    assignments = {}
    for a, k in zip(specs, picks):
        topic = [pool[int(k)]]
        if a.get("read_topics") == "random":
            a["read_topics"] = topic
        if a.get("write_topics") == "random":
            a["write_topics"] = topic
        assignments[a["agent_id"]] = topic[0]
    return assignments


def run_simulation(cfg: RunConfig, *, graph, index, embedder, llm, model,
                   out_dir) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    agents = [dict(a) for a in cfg.agents]
    for a in agents:
        a["_identity"] = f"{cfg.run_id}:{a['agent_id']}"
        a["_total_steps"] = cfg.total_steps
    assignments = _resolve_random_topics(agents, cfg.topic_pool, rng)
    cfg = RunConfig(run_id=cfg.run_id, seed=cfg.seed, total_steps=cfg.total_steps,
                    generation_budget=cfg.generation_budget, agents=agents,
                    topic_pool=cfg.topic_pool)
    # index.vecs holds exactly the corpus at run start (nothing generated yet)
    scopes = {a["agent_id"]: s for a in cfg.agents
              if (s := build_scope(a, embedder, corpus_vecs=index.vecs)) is not None}
    needs_communities = any(s.read is not None or s.write is not None
                            for s in scopes.values())
    communities = graph.communities() if needs_communities else None
    env = Environment(run_id=cfg.run_id, graph=graph, index=index,
                      embedder=embedder, event_log=EventLog(run_dir / "events.jsonl"),
                      rng=rng, generation_budget=cfg.generation_budget,
                      scopes=scopes, communities=communities)
    policies = {a["agent_id"]: build_policy(a, llm=llm, model=model,
                                            graph=graph, rng=rng)
                for a in cfg.agents}
    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}

    for step in range(cfg.total_steps):
        agent_id = order[step % len(order)]
        obs = {"step": step, "last_result": last_result[agent_id],
               "ideas_used": cfg.generation_budget - env.generation_budget,
               "ideas_total": cfg.generation_budget}
        action = policies[agent_id].act(obs)
        last_result[agent_id] = env.execute(agent_id, step, action)

    out = {"run_id": cfg.run_id, "steps": cfg.total_steps,
           "generated": env.generated_ids(),
           "topic_assignments": assignments}
    (run_dir / "run_meta.json").write_text(json.dumps(
        {"run_id": cfg.run_id, "seed": cfg.seed,
         "topic_assignments": assignments}, indent=1))
    return out
