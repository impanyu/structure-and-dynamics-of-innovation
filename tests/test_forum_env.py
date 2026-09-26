import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.p2_forum.env import Action, ForumEnvironment


def make_env(make_workspace, tmp_path, **kw) -> ForumEnvironment:
    return ForumEnvironment(run_id="t", workspace=make_workspace(),
                            event_log=EventLog(tmp_path / "events.jsonl"),
                            rng=np.random.default_rng(0), **kw)


def test_search_and_search_board_hit_different_stores(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    env.execute("a", 0, Action("generate", {"text": "board idea", "cited_ids": []}))

    corpus_hits = env.execute("a", 1, Action("search", {"query": "paper", "k": 5}))
    board_hits = env.execute("a", 2, Action("search_board", {"query": "board idea", "k": 5}))

    assert all(h["node_id"].startswith("p") for h in corpus_hits["hits"])
    assert all(h["node_id"].startswith("gen:") for h in board_hits["hits"])


def test_nothing_is_filtered_search_returns_every_hit(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    out = env.execute("a", 0, Action("search", {"query": "anything", "k": 5}))
    assert len(out["hits"]) == 2  # the whole two-paper corpus


def test_generate_posts_to_the_board_and_leaves_the_corpus_alone(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    before = (env.ws.corpus.num_nodes, env.ws.corpus.num_edges)

    out = env.execute("a", 0, Action("generate", {"text": "new", "cited_ids": ["p1"]}))

    assert out["node_id"].startswith("gen:")
    assert (env.ws.corpus.num_nodes, env.ws.corpus.num_edges) == before


def test_add_links_from_a_corpus_node_is_rejected(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    out = env.execute("a", 0, Action("add_links", {"src_id": "p1", "dst_ids": ["p2"]}))
    assert "error" in out
    assert "board node" in out["error"]


def test_browse_board_on_an_unknown_id_returns_an_error_not_a_crash(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    out = env.execute("a", 0, Action("browse_board", {"node_id": "gen:t:99"}))
    assert "error" in out


def test_sample_board_is_an_error_while_the_board_is_empty(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    out = env.execute("a", 0, Action("sample_board", {}))
    assert "error" in out


def test_allow_jump_false_blocks_both_jump_actions(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path, allow_jump=False)
    assert "error" in env.execute("a", 0, Action("sample_frontier", {}))
    assert "error" in env.execute("a", 1, Action("sample_board", {}))


def test_allow_search_false_blocks_both_search_actions(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path, allow_search=False)
    assert "error" in env.execute("a", 0, Action("search", {"query": "x"}))
    assert "error" in env.execute("a", 1, Action("search_board", {"query": "x"}))


def test_restore_rebuilds_the_board_from_the_event_log(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path)
    nid = env.execute("a", 0, Action("generate", {"text": "one", "cited_ids": ["p1"]}))["node_id"]
    env.execute("a", 1, Action("generate", {"text": "two", "cited_ids": [nid]}))
    events = env.event_log.read_all()

    fresh = make_env(make_workspace, tmp_path / "other")
    fresh.restore(events)

    assert fresh.generated_ids() == env.generated_ids()
    assert fresh.ws.board.citations_out(env.generated_ids()[1]) == [nid]
