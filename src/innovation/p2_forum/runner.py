"""Round-robin runner for paper 2 (spec §6).

Each agent draws k topics uniformly at random from the pool. Draws are seeded
and recorded in run_meta.json. Agents may share topics — with N agents and a
128-topic pool that is unavoidable above a certain N, and it is the substrate
collaboration needs, not a defect.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from innovation.core.events import EventLog, load_events
from innovation.core.network.graph import IdeaGraph
from innovation.core.network.index import VectorIndex
from innovation.p2_forum.agent import (GATED_ACTIONS_DOC, GATED_SYSTEM,
                                       ForumAgentPolicy)
from innovation.p2_forum.env import ForumEnvironment, Navigation
from innovation.p2_forum.gated_env import GatedForumEnvironment
from innovation.p2_forum.workspace import Workspace


TOPIC_DRAWS = ("independent", "nested")
# "corpus"/"none" is the original mode (frozen corpus, soft specialization);
# "online"/"topics" is online literature behind a hard topic gate. The two
# axes are not independent: the gate needs labelled literature, and the online
# literature is only in scope through the gate.
LITERATURES = ("corpus", "online")
GATINGS = ("none", "topics")


@dataclass
class ForumRunConfig:
    run_id: str
    seed: int
    total_steps: int
    agents: list[dict] = field(default_factory=list)
    topic_pool: list[str] = field(default_factory=list)
    generation_budget: int | None = None
    # Navigation ablations are environmental, not dispositional (spec §4.3):
    # one board is shared, so a channel is open or closed for the whole run.
    navigation: Navigation = field(default_factory=Navigation)
    # "independent": each run draws each agent's k topics afresh (the original
    # k sweep). "nested": each agent gets a seed-determined permutation of the
    # pool and takes its first k, so for one seed the topic sets are strict
    # prefixes of each other across k -- adjacent k differ ONLY by the added
    # topics, not by a reshuffle.
    topic_draw: str = "independent"
    literature: str = "corpus"
    gating: str = "none"
    # topic name -> one-sentence definition, shown next to each topic in the
    # gated prompt. Unused in corpus mode.
    topic_definitions: dict[str, str] = field(default_factory=dict)

    @property
    def online(self) -> bool:
        return self.literature == "online"

    def __post_init__(self):
        if self.topic_draw not in TOPIC_DRAWS:
            raise ValueError(f"topic_draw must be one of {TOPIC_DRAWS}, "
                             f"got {self.topic_draw!r}")
        if self.literature not in LITERATURES:
            raise ValueError(f"literature must be one of {LITERATURES}, "
                             f"got {self.literature!r}")
        if self.gating not in GATINGS:
            raise ValueError(f"gating must be one of {GATINGS}, "
                             f"got {self.gating!r}")
        if (self.literature == "online") != (self.gating == "topics"):
            raise ValueError(
                "literature: online requires gating: topics and vice versa; "
                f"got literature={self.literature!r}, gating={self.gating!r}")
        # Per-agent allow_jump/allow_search used to be collapsed with all(...)
        # into one env-wide flag, so one agent opting out silently disabled the
        # channel for every agent. Refuse the key rather than mean something
        # else by it.
        stray = sorted({key for a in self.agents for key in a
                        if key in ("allow_jump", "allow_search")})
        if stray:
            raise ValueError(
                f"{stray} is per-agent in run.agents, but navigation ablations "
                "are environment-wide (spec §4.3); move them to the top-level "
                "`navigation:` section, e.g. navigation: {board: {jump: false}}")


_DISPLAY_SALT = 0xD15B1A


def display_order(topics: list[str], seed: int, agent_index: int) -> list[str]:
    """Prompt display order, shuffled independently of the nested draw so the
    first-listed topics are not the small-k topics (spec §7)."""
    own = np.random.default_rng(
        np.random.SeedSequence([int(seed), agent_index, _DISPLAY_SALT]))
    return [topics[int(j)] for j in own.permutation(len(topics))]


def _display_orders(cfg: "ForumRunConfig", assignments: dict) -> dict[str, list[str]]:
    return {a["agent_id"]: display_order(assignments[a["agent_id"]], cfg.seed, i)
            for i, a in enumerate(cfg.agents)}


def _topic_ids(cfg: "ForumRunConfig", assignments: dict) -> dict[str, list[int]]:
    return {aid: sorted(cfg.topic_pool.index(t) for t in names)
            for aid, names in assignments.items()}


# Mixed into the nested scheme's seed so its per-agent streams never coincide
# with the environment's rng, which is seeded from the bare run seed.
_NESTED_SALT = 0x70C1C5


def draw_topics(agents: list[dict], pool: list[str], rng, *,
                scheme: str = "independent",
                seed: int | None = None) -> dict[str, list[str]]:
    """k distinct topics per agent. Different agents may share topics; that
    overlap is what makes collaboration possible.

    independent -- each agent's k topics are drawn from `rng` in turn, so the
      draw for one k says nothing about the draw for another k.
    nested      -- agent i gets a fixed permutation of the pool from its own
      stream, seeded by (seed, i), and takes its first k. The permutation does
      not depend on k, so for one seed topics(k) is a strict prefix of
      topics(k') whenever k < k'. `rng` is not consumed.
    """
    if not pool:
        raise ValueError("no topic_pool given")
    if scheme not in TOPIC_DRAWS:
        raise ValueError(f"unknown topic draw scheme {scheme!r}")
    if scheme == "nested" and seed is None:
        raise ValueError("the nested scheme needs the run seed")
    out: dict[str, list[str]] = {}
    for i, a in enumerate(agents):
        k = int(a.get("k_topics", 1))
        if k > len(pool):
            raise ValueError(
                f"k_topics={k} exceeds the {len(pool)}-topic pool")
        if scheme == "nested":
            own = np.random.default_rng(
                np.random.SeedSequence([int(seed), i, _NESTED_SALT]))
            picks = own.permutation(len(pool))[:k]
        else:
            picks = rng.choice(len(pool), size=k, replace=False)
        out[a["agent_id"]] = [pool[int(j)] for j in picks]
    return out


def _build_env(cfg: ForumRunConfig, *, run_dir, corpus, corpus_index, embedder,
               rng, literature=None, tagger=None,
               topic_ids: dict | None = None) -> ForumEnvironment:
    if cfg.online:
        if literature is None or tagger is None:
            raise ValueError("literature: online needs a literature and a tagger")
        # No corpus: papers live in the online literature, which also vouches
        # for the paper ids posts cite (Workspace.external_papers).
        empty = IdeaGraph()
        empty.freeze()
        ws = Workspace(corpus=empty, corpus_index=VectorIndex(embedder.dim),
                       board_index=VectorIndex(embedder.dim),
                       embedder=embedder, run_id=cfg.run_id,
                       external_papers=literature)
        return GatedForumEnvironment(
            run_id=cfg.run_id, workspace=ws,
            event_log=EventLog(run_dir / "events.jsonl"), rng=rng,
            generation_budget=cfg.generation_budget, navigation=cfg.navigation,
            literature=literature, tagger=tagger,
            agent_topics={a: set(t) for a, t in topic_ids.items()},
            topic_names=cfg.topic_pool)
    ws = Workspace(corpus=corpus, corpus_index=corpus_index,
                   board_index=VectorIndex(corpus_index.dim),
                   embedder=embedder, run_id=cfg.run_id)
    return ForumEnvironment(run_id=cfg.run_id, workspace=ws,
                            event_log=EventLog(run_dir / "events.jsonl"),
                            rng=rng,
                            generation_budget=cfg.generation_budget,
                            navigation=cfg.navigation)


def _build_policies(cfg: ForumRunConfig, *, llm, model, assignments: dict,
                    display_orders: dict | None = None) -> dict[str, ForumAgentPolicy]:
    if cfg.online:
        return {
            a["agent_id"]: ForumAgentPolicy(
                llm=llm, model=model,
                topics=[f"{t} — {cfg.topic_definitions.get(t, '')}"
                        for t in display_orders[a["agent_id"]]],
                memory_size=a.get("memory_size", 20),
                identity=f"{cfg.run_id}:{a['agent_id']}",
                total_steps=cfg.total_steps,
                system_template=GATED_SYSTEM, actions_doc=GATED_ACTIONS_DOC)
            for a in cfg.agents}
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


def run_forum(cfg: ForumRunConfig, *, corpus=None, corpus_index=None, embedder,
              llm, model, out_dir, literature=None, tagger=None) -> dict:
    rng = np.random.default_rng(cfg.seed)
    run_dir = Path(out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    assignments = draw_topics(cfg.agents, cfg.topic_pool, rng,
                              scheme=cfg.topic_draw, seed=cfg.seed)
    meta = {"run_id": cfg.run_id, "seed": cfg.seed, "arch": "p2_forum",
            "total_steps": cfg.total_steps,
            "navigation": asdict(cfg.navigation),
            "topic_draw": cfg.topic_draw,
            "topic_assignments": assignments}
    topic_ids = display_orders = None
    if cfg.online:
        # Corpus-mode run_meta stays exactly as before (old runs reproduce
        # byte for byte); a missing "literature" key means corpus/none.
        topic_ids = _topic_ids(cfg, assignments)
        display_orders = _display_orders(cfg, assignments)
        meta.update(literature=cfg.literature, gating=cfg.gating,
                    topic_ids=topic_ids, display_orders=display_orders)
    # run_meta is written BEFORE driving: an interrupted run must stay
    # reproducible from its recorded draws rather than re-sampling them.
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))

    env = _build_env(cfg, run_dir=run_dir, corpus=corpus,
                     corpus_index=corpus_index, embedder=embedder, rng=rng,
                     literature=literature, tagger=tagger, topic_ids=topic_ids)
    policies = _build_policies(cfg, llm=llm, model=model,
                               assignments=assignments,
                               display_orders=display_orders)

    order = [a["agent_id"] for a in cfg.agents]
    last_result: dict[str, dict] = {aid: {} for aid in order}
    out = _drive(cfg, env, policies, order, last_result, 0)
    out["topic_assignments"] = assignments
    return out


def resume_forum(cfg: ForumRunConfig, *, corpus=None, corpus_index=None,
                 embedder, llm, model, out_dir, literature=None,
                 tagger=None) -> dict:
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
    recorded = meta.get("literature", "corpus")
    if recorded != cfg.literature:
        raise SystemExit(
            f"cannot resume {run_dir}: it was run with literature: {recorded}, "
            f"the config says {cfg.literature}")
    topic_ids = display_orders = None
    if cfg.online:
        # Reuse what the run recorded; never re-derive the gate's topic sets.
        topic_ids = meta["topic_ids"]
        display_orders = meta["display_orders"]
    start_step = max(e["step"] for e in events) + 1
    if start_step >= cfg.total_steps:
        raise SystemExit(f"run already has {start_step} steps; "
                         f"raise total_steps beyond that to extend")

    rng = np.random.default_rng((cfg.seed, start_step))
    env = _build_env(cfg, run_dir=run_dir, corpus=corpus,
                     corpus_index=corpus_index, embedder=embedder, rng=rng,
                     literature=literature, tagger=tagger, topic_ids=topic_ids)
    env.restore(events)

    policies = _build_policies(cfg, llm=llm, model=model,
                               assignments=assignments,
                               display_orders=display_orders)
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

    # Written BEFORE driving, same rule as run_forum: if this resumed segment
    # is interrupted, the run's own record of how far it got (total_steps,
    # resumed_from) must still be on disk, or a later resume has no way to
    # know this segment was ever attempted. The topic draws survive either
    # way (run_forum already wrote them), but that is not the same as the
    # run recording its own progress.
    meta_path.write_text(json.dumps(
        {**meta, "total_steps": cfg.total_steps,
         "resumed_from": meta.get("resumed_from", []) + [start_step]}, indent=1))

    out = _drive(cfg, env, policies, order, last_result, start_step)
    out["topic_assignments"] = assignments
    out["resumed_from_step"] = start_step
    return out
