import pandas as pd
import pytest

from innovation.network.graph import IdeaGraph


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
