import numpy as np
import pytest

from innovation.core.events import EventLog
from innovation.p2_forum.env import Action, ForumEnvironment, Navigation


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


def test_closing_the_jump_channel_blocks_both_jump_actions(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"jump": False}))
    assert "error" in env.execute("a", 0, Action("sample_frontier", {}))
    assert "error" in env.execute("a", 1, Action("sample_board", {}))


def test_closing_the_search_channel_blocks_both_search_actions(make_workspace,
                                                               tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"search": False}))
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


# --- spec §4.1: the three ablations apply to each store independently ---

def _post_and_cite(env):
    """Two board posts, the second citing the first, so both stores have edges."""
    a = env.execute("a", 0, Action("generate", {"text": "one", "cited_ids": ["p1"]}))["node_id"]
    b = env.execute("a", 1, Action("generate", {"text": "two", "cited_ids": [a]}))["node_id"]
    return a, b


def test_search_can_be_closed_on_the_board_alone(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"board": {"search": False}}))
    _post_and_cite(env)

    assert "error" in env.execute("a", 2, Action("search_board", {"query": "x"}))
    assert "hits" in env.execute("a", 3, Action("search", {"query": "x"}))


def test_search_can_be_closed_on_the_corpus_alone(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"corpus": {"search": False}}))
    _post_and_cite(env)

    assert "error" in env.execute("a", 2, Action("search", {"query": "x"}))
    assert "hits" in env.execute("a", 3, Action("search_board", {"query": "x"}))


def test_jumps_can_be_closed_on_one_store_alone(make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"corpus": {"jump": False}}))
    _post_and_cite(env)

    assert "error" in env.execute("a", 2, Action("sample_frontier", {}))
    assert "error" not in env.execute("a", 3, Action("sample_board", {}))


def test_closing_the_board_edge_channel_hides_board_neighbours_only(
        make_workspace, tmp_path):
    """The channel this paper is about. Closing it leaves the post readable and
    removes the links, exactly as paper 1's init_edges: none does for the
    corpus."""
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"board": {"edges": False}}))
    a, b = _post_and_cite(env)

    board_view = env.execute("a", 2, Action("browse_board", {"node_id": b}))
    corpus_view = env.execute("a", 3, Action("browse", {"node_id": "p2"}))

    assert board_view["text"] == "two"          # still readable
    assert board_view["cites"] == [] and board_view["cited_by"] == []
    assert [c["node_id"] for c in corpus_view["cites"]] == ["p1"]  # corpus intact


def test_closing_the_corpus_edge_channel_hides_corpus_neighbours_only(
        make_workspace, tmp_path):
    env = make_env(make_workspace, tmp_path,
                   navigation=Navigation.from_config({"corpus": {"edges": False}}))
    a, b = _post_and_cite(env)

    corpus_view = env.execute("a", 2, Action("browse", {"node_id": "p2"}))
    board_view = env.execute("a", 3, Action("browse_board", {"node_id": b}))

    assert corpus_view["text"] == "paper two"
    assert corpus_view["cites"] == [] and corpus_view["cited_by"] == []
    assert [c["node_id"] for c in board_view["cites"]] == [a]


def test_a_per_store_setting_overrides_the_shared_shorthand():
    nav = Navigation.from_config({"search": False, "corpus": {"search": True}})

    assert nav.corpus_search is True and nav.board_search is False


def test_navigation_defaults_to_every_channel_open():
    nav = Navigation.from_config(None)

    assert all(nav.is_open(store, channel)
               for store in ("corpus", "board")
               for channel in ("search", "edges", "jump"))


def test_a_misspelled_ablation_key_is_an_error_not_a_no_op():
    with pytest.raises(ValueError, match="unknown navigation key"):
        Navigation.from_config({"serach": False})
    with pytest.raises(ValueError, match="unknown navigation.board channel"):
        Navigation.from_config({"board": {"jumps": False}})


# --- a bare bool on a per-store key must not be a silent no-op (finding 1) ---

def test_a_false_store_shorthand_closes_every_channel_on_that_store():
    nav = Navigation.from_config({"board": False})

    assert nav.board_search is False
    assert nav.board_edges is False
    assert nav.board_jump is False
    # the other store is untouched
    assert nav.corpus_search is True
    assert nav.corpus_edges is True
    assert nav.corpus_jump is True


def test_a_true_store_shorthand_opens_every_channel_on_that_store():
    nav = Navigation.from_config({"board": True})

    assert nav.board_search is True
    assert nav.board_edges is True
    assert nav.board_jump is True


def test_a_malformed_store_value_is_rejected_not_ignored():
    with pytest.raises(ValueError, match="navigation.board must be a mapping"):
        Navigation.from_config({"board": "off"})
    with pytest.raises(ValueError, match="navigation.corpus must be a mapping"):
        Navigation.from_config({"corpus": ["jump"]})
    with pytest.raises(ValueError, match="navigation.board must be a mapping"):
        Navigation.from_config({"board": None})
