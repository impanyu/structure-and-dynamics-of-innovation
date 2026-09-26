import pandas as pd
import pytest

from innovation.core.network.graph import FrozenGraphError, IdeaGraph


def small_graph():
    ideas = pd.DataFrame([
        {"paper_id": "W1", "idea_text": "i1", "year": 2019, "venue": "V"},
        {"paper_id": "W2", "idea_text": "i2", "year": 2020, "venue": "V"},
        {"paper_id": "W3", "idea_text": "i3", "year": 2021, "venue": "V"},
    ])
    edges = pd.DataFrame([{"src": "W2", "dst": "W1"}, {"src": "W3", "dst": "W2"}])
    return IdeaGraph.from_tables(ideas, edges)


def test_from_tables_builds_nodes_and_citations():
    g = small_graph()
    assert g.num_nodes == 3 and g.num_edges == 2
    assert g.node("W1").text == "i1"
    assert g.node("W1").source == "corpus"
    assert g.citations_out("W2") == ["W1"]
    assert g.citations_in("W2") == ["W3"]
    assert g.in_degree("W1") == 1


def test_add_idea_appends_generated_node_with_provenance():
    g = small_graph()
    g.add_idea("gen:r1:0", "new idea", ["W1", "W3"],
               meta={"run_id": "r1", "agent_id": "a0", "step": 4})
    assert g.num_nodes == 4
    assert set(g.citations_out("gen:r1:0")) == {"W1", "W3"}
    assert g.node("gen:r1:0").source == "generated"
    assert g.node("gen:r1:0").meta["agent_id"] == "a0"
    assert g.node_ids(source="generated") == ["gen:r1:0"]
    with pytest.raises(KeyError):
        g.add_idea("gen:r1:1", "bad", ["W_missing"])


def test_add_idea_rejects_duplicate_node_id():
    g = small_graph()
    g.add_idea("gen:r1:0", "new idea", ["W1"])
    with pytest.raises(ValueError):
        g.add_idea("gen:r1:0", "new idea again", ["W2"])


def test_network_at_slices_by_year():
    g = small_graph()
    g2020 = g.network_at(2020)
    assert set(g2020.node_ids()) == {"W1", "W2"}
    assert g2020.num_edges == 1


def test_communities_cover_all_nodes():
    g = small_graph()
    comm = g.communities()
    assert set(comm) == {"W1", "W2", "W3"}
    assert all(isinstance(c, int) for c in comm.values())


def test_add_links_typed_dedup_and_validation():
    g = small_graph()
    # W3 already cites W2; W3->W1 is new; W3->W3 is a self-loop
    res = g.add_links("W3", ["W1", "W2", "W3"], meta={"agent_id": "a0"})
    assert res == {"added": ["W1"], "skipped": ["W2", "W3"]}
    assert g.edge_type("W3", "W1") == "agent_link"
    assert g.edge_type("W3", "W2") == "citation"
    with pytest.raises(KeyError):
        g.add_links("W3", ["missing"])
    with pytest.raises(KeyError):
        g.add_links("missing", ["W1"])


def test_generate_edges_are_typed():
    g = small_graph()
    g.add_idea("gen:r:0", "t", ["W1"])
    assert g.edge_type("gen:r:0", "W1") == "generated"


def test_remove_links_records_etype_and_validates():
    g = small_graph()
    # W3->W2 exists (citation); W3->W1 does not
    res = g.remove_links("W3", ["W2", "W1"])
    assert res == {"removed": [{"dst_id": "W2", "etype": "citation"}],
                   "skipped": ["W1"]}
    assert "W2" not in g.citations_out("W3")
    with pytest.raises(KeyError):
        g.remove_links("missing", ["W1"])
    with pytest.raises(KeyError):
        g.remove_links("W3", ["missing"])


def _two_node_graph() -> IdeaGraph:
    g = IdeaGraph()
    g.add_idea("a", "idea a", [], source="corpus", year=2020)
    g.add_idea("b", "idea b", ["a"], source="corpus", year=2021)
    return g


def test_freeze_blocks_every_mutation():
    g = _two_node_graph()
    g.freeze()

    assert g.frozen is True
    with pytest.raises(FrozenGraphError):
        g.add_idea("c", "idea c", ["a"])
    with pytest.raises(FrozenGraphError):
        g.add_links("b", ["a"])
    with pytest.raises(FrozenGraphError):
        g.remove_links("b", ["a"])


def test_freeze_leaves_reads_working():
    g = _two_node_graph()
    g.freeze()

    assert g.num_nodes == 2
    assert g.citations_out("b") == ["a"]
    assert g.node("a").text == "idea a"


def test_an_unfrozen_graph_hands_out_the_live_node():
    """Paper 1 never freezes, so node() must keep returning the stored node."""
    g = IdeaGraph()
    g.add_idea("a", "idea a", [], meta={"agent_id": "a0"})

    assert g.node("a") is g.node("a")


def test_freeze_detaches_node_payload_from_the_store():
    g = IdeaGraph()
    g.add_idea("a", "idea a", [], source="corpus", year=2020,
               meta={"venue": "V"})
    g.freeze()

    g.node("a").text = "REWRITTEN"
    g.node("a").meta["venue"] = "forged"
    g.node("a").year = 1999

    assert (g.node("a").text, g.node("a").year, g.node("a").meta) == (
        "idea a", 2020, {"venue": "V"})
