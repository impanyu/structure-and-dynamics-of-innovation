import json

import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.p2_forum.board_metrics import (board_at, board_structure,
                                                board_trajectory)
from innovation.p2_forum.env import Action, ForumEnvironment


def build_events(make_workspace, tmp_path):
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "e.jsonl"),
                           rng=np.random.default_rng(0))
    a = env.execute("a0", 0, Action("generate", {"text": "one", "cited_ids": ["p1"]}))["node_id"]
    env.execute("a1", 1, Action("generate", {"text": "two", "cited_ids": [a]}))
    return env.event_log.read_all()


def test_board_structure_separates_post_post_from_post_corpus_edges(
        make_workspace, fake_embedder, tmp_path):
    events = build_events(make_workspace, tmp_path)
    ws = make_workspace()
    rebuilt = board_at(events, len(events), corpus=ws.corpus,
                       corpus_index=ws.corpus_index, embedder=fake_embedder,
                       run_id="t")

    s = board_structure(rebuilt, events)

    assert s["n_posts"] == 2
    assert s["n_post_corpus_edges"] == 1   # post -> p1
    assert s["n_post_post_edges"] == 1     # post -> post
    assert s["post_post_share"] == 0.5


def test_first_cross_agent_citation_step_is_the_step_of_the_first_a_to_b_citation(
        make_workspace, fake_embedder, tmp_path):
    events = build_events(make_workspace, tmp_path)
    ws = make_workspace()
    rebuilt = board_at(events, len(events), corpus=ws.corpus,
                       corpus_index=ws.corpus_index, embedder=fake_embedder,
                       run_id="t")

    s = board_structure(rebuilt, events)

    assert s["first_cross_agent_citation_step"] == 1
    assert s["cross_agent_edges"] == 1


def test_board_at_truncates_to_a_prefix_of_the_log(
        make_workspace, fake_embedder, tmp_path):
    events = build_events(make_workspace, tmp_path)
    ws = make_workspace()
    early = board_at(events, 1, corpus=ws.corpus,
                     corpus_index=ws.corpus_index, embedder=fake_embedder,
                     run_id="t")

    assert len(early.board_post_ids()) == 1


def test_an_isolated_board_has_one_component_per_post(
        make_workspace, fake_embedder, tmp_path):
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "iso.jsonl"),
                           rng=np.random.default_rng(0))
    env.execute("a0", 0, Action("generate", {"text": "x", "cited_ids": []}))
    env.execute("a1", 1, Action("generate", {"text": "y", "cited_ids": []}))
    events = env.event_log.read_all()

    fresh = make_workspace()
    rebuilt = board_at(events, len(events), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index,
                       embedder=fake_embedder, run_id="t")
    s = board_structure(rebuilt, events)

    assert s["n_components"] == 2
    assert s["n_post_post_edges"] == 0
    assert s["first_cross_agent_citation_step"] is None
    assert s["longest_chain"] == 0


def test_in_degree_distribution_over_posts_is_reported(
        make_workspace, fake_embedder, tmp_path):
    """Spec §5 names the in-degree distribution among the mechanism metrics."""
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "deg.jsonl"),
                           rng=np.random.default_rng(0))
    hub = env.execute("a0", 0, Action("generate", {"text": "hub", "cited_ids": []}))["node_id"]
    env.execute("a1", 1, Action("generate", {"text": "one", "cited_ids": [hub]}))
    env.execute("a2", 2, Action("generate", {"text": "two", "cited_ids": [hub]}))
    events = env.event_log.read_all()

    fresh = make_workspace()
    rebuilt = board_at(events, len(events), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index,
                       embedder=fake_embedder, run_id="t")
    s = board_structure(rebuilt, events)

    # the hub is cited twice; the two citing posts are cited by nobody
    assert s["post_in_degree_distribution"] == {"0": 2, "2": 1}
    assert s["max_post_in_degree"] == 2
    assert s["mean_post_in_degree"] == pytest.approx(2 / 3)


def test_longest_chain_is_well_defined_on_a_cyclic_board(
        make_workspace, fake_embedder, tmp_path):
    """§4.2 lets any agent link any post, so a 2-cycle is reachable. The metric
    is the longest chain over the DAG condensation — never a -1 sentinel that
    would silently poison an averaged column."""
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "cyc.jsonl"),
                           rng=np.random.default_rng(0))
    a = env.execute("a0", 0, Action("generate", {"text": "a", "cited_ids": []}))["node_id"]
    b = env.execute("a1", 1, Action("generate", {"text": "b", "cited_ids": [a]}))["node_id"]
    added = env.execute("a1", 2, Action("add_links", {"src_id": a, "dst_ids": [b]}))
    assert added["added"] == [b]          # the cycle really is on the board
    events = env.event_log.read_all()

    fresh = make_workspace()
    rebuilt = board_at(events, len(events), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index,
                       embedder=fake_embedder, run_id="t")
    s = board_structure(rebuilt, events)

    assert s["n_post_post_edges"] == 2
    assert s["post_post_is_acyclic"] is False
    # a and b are one strongly connected component: one group, no chain between groups
    assert s["longest_chain"] == 0


def test_longest_chain_counts_edges_on_an_acyclic_chain(
        make_workspace, fake_embedder, tmp_path):
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "chain.jsonl"),
                           rng=np.random.default_rng(0))
    prev = None
    for i in range(4):
        prev = env.execute(f"a{i}", i, Action(
            "generate", {"text": f"p{i}",
                         "cited_ids": [prev] if prev else []}))["node_id"]
    events = env.event_log.read_all()

    fresh = make_workspace()
    rebuilt = board_at(events, len(events), corpus=fresh.corpus,
                       corpus_index=fresh.corpus_index,
                       embedder=fake_embedder, run_id="t")
    s = board_structure(rebuilt, events)

    assert s["post_post_is_acyclic"] is True
    assert s["longest_chain"] == 3
    assert s["n_components"] == 1


def test_every_structural_value_survives_a_json_round_trip(
        make_workspace, fake_embedder, tmp_path):
    events = build_events(make_workspace, tmp_path)
    ws = make_workspace()
    rebuilt = board_at(events, len(events), corpus=ws.corpus,
                       corpus_index=ws.corpus_index, embedder=fake_embedder,
                       run_id="t")

    s = board_structure(rebuilt, events)

    assert json.loads(json.dumps(s)) == s


def test_board_trajectory_steps_by_one_round_not_one_event(
        make_workspace, fake_embedder, tmp_path):
    """One round is N events with N agents round-robin, so a per-round series
    must stride by N. Each entry says which it is."""
    ws = make_workspace()
    env = ForumEnvironment(run_id="t", workspace=ws,
                           event_log=EventLog(tmp_path / "traj.jsonl"),
                           rng=np.random.default_rng(0))
    for step in range(6):
        env.execute(f"a{step % 2}", step,
                    Action("generate", {"text": f"idea {step}", "cited_ids": []}))
    events = env.event_log.read_all()

    fresh = make_workspace()
    series = board_trajectory(events, corpus=fresh.corpus,
                              corpus_index=fresh.corpus_index,
                              embedder=fake_embedder, run_id="t", n_agents=2)

    assert [r["round"] for r in series] == [1, 2, 3]
    assert [r["n_events"] for r in series] == [2, 4, 6]
    assert [r["n_posts"] for r in series] == [2, 4, 6]


def test_the_trajectorys_last_entry_matches_a_one_shot_replay(
        make_workspace, fake_embedder, tmp_path):
    """Incremental replay must give the same answer as board_at on the whole
    log — the incremental pass exists only to pay for embedding once."""
    events = build_events(make_workspace, tmp_path)
    fresh = make_workspace()
    series = board_trajectory(events, corpus=fresh.corpus,
                              corpus_index=fresh.corpus_index,
                              embedder=fake_embedder, run_id="t", n_agents=1)

    other = make_workspace()
    one_shot = board_structure(
        board_at(events, len(events), corpus=other.corpus,
                 corpus_index=other.corpus_index, embedder=fake_embedder,
                 run_id="t"), events)

    assert {k: v for k, v in series[-1].items()
            if k not in ("round", "n_events")} == one_shot


def test_board_trajectory_rejects_a_zero_agent_stride(make_workspace,
                                                      fake_embedder, tmp_path):
    events = build_events(make_workspace, tmp_path)
    ws = make_workspace()
    with pytest.raises(ValueError, match="n_agents"):
        board_trajectory(events, corpus=ws.corpus,
                         corpus_index=ws.corpus_index, embedder=fake_embedder,
                         run_id="t", n_agents=0)
