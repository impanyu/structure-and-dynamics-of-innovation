import pytest

from innovation.core.network.graph import FrozenGraphError


def test_store_of_routes_by_id_prefix(make_workspace):
    ws = make_workspace()
    nid = ws.post_idea("a new idea", ["p1"], meta={})

    assert ws.store_of("p1") == "corpus"
    assert ws.store_of(nid) == "board"


def test_post_idea_never_touches_the_corpus(make_workspace):
    ws = make_workspace()
    before_nodes, before_edges = ws.corpus.num_nodes, ws.corpus.num_edges

    ws.post_idea("grounded idea", ["p1", "p2"], meta={})

    assert (ws.corpus.num_nodes, ws.corpus.num_edges) == (before_nodes, before_edges)


def test_cross_store_citation_becomes_a_stub_node_on_the_board(make_workspace):
    ws = make_workspace()
    nid = ws.post_idea("grounded idea", ["p1"], meta={})

    assert ws.board.citations_out(nid) == ["p1"]
    assert ws.board.node("p1").source == "corpus_ref"
    # stubs are not posts
    assert ws.board_post_ids() == [nid]


def test_board_post_is_searchable_and_corpus_is_not_in_the_board_index(make_workspace, fake_embedder):
    ws = make_workspace()
    nid = ws.post_idea("searchable idea", [], meta={})

    hits = ws.board_search(fake_embedder.encode(["searchable idea"])[0], k=5)
    assert [h[0] for h in hits] == [nid]


def test_link_source_must_be_on_the_board(make_workspace):
    ws = make_workspace()
    nid = ws.post_idea("an idea", [], meta={})

    assert ws.add_links(nid, ["p1"], meta={})["added"] == ["p1"]
    with pytest.raises(ValueError, match="source must be a board node"):
        ws.add_links("p1", ["p2"], meta={})
    with pytest.raises(ValueError, match="source must be a board node"):
        ws.remove_links("p2", ["p1"], )


def test_any_agent_may_edit_any_post_wiki_semantics(make_workspace):
    ws = make_workspace()
    a = ws.post_idea("post by agent A", [], meta={"agent_id": "a"})
    b = ws.post_idea("post by agent B", [], meta={"agent_id": "b"})

    assert ws.add_links(b, [a], meta={"agent_id": "a"})["added"] == [a]
    removed = ws.remove_links(b, [a])["removed"]
    assert [r["dst_id"] for r in removed] == [a]


def test_unknown_link_target_raises(make_workspace):
    ws = make_workspace()
    nid = ws.post_idea("an idea", [], meta={})

    with pytest.raises(KeyError):
        ws.add_links(nid, ["nope"], meta={})


def test_corpus_stays_frozen_through_the_facade(make_workspace):
    ws = make_workspace()
    with pytest.raises(FrozenGraphError):
        ws.corpus.add_idea("x", "x", [])


def test_external_papers_back_corpus_ids_for_stubs():
    import numpy as np
    from innovation.core.network.graph import IdeaGraph
    from innovation.core.network.index import VectorIndex
    from innovation.p2_forum.workspace import Workspace
    from conftest import FakeEmbedder

    class Known:
        def has(self, pid):
            return pid == "s2:abc"

    empty = IdeaGraph()
    empty.freeze()
    ws = Workspace(corpus=empty, corpus_index=VectorIndex(4), board_index=VectorIndex(4),
                   embedder=FakeEmbedder(), run_id="t", external_papers=Known())
    nid = ws.post_idea("idea", ["s2:abc"], meta={})
    assert ws.has_node("s2:abc")
    assert ws.board.has_node("s2:abc")          # stub created
    assert ws.board.citations_out(nid) == ["s2:abc"]
    with pytest.raises(KeyError):
        ws.post_idea("idea2", ["unknown"], meta={})
