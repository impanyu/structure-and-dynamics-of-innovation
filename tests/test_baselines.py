import numpy as np
import pandas as pd

from innovation.agents.baselines import (NoNavLLMPolicy,
                                         PreferentialAttachmentPolicy)
from innovation.llm import FakeLLM
from innovation.network.graph import IdeaGraph


def graph_with_hub():
    ideas = pd.DataFrame(
        [{"paper_id": f"W{i}", "idea_text": f"idea {i}", "year": 2020, "venue": "V"}
         for i in range(5)])
    # W0 is a hub: cited by everyone else.
    edges = pd.DataFrame([{"src": f"W{i}", "dst": "W0"} for i in range(1, 5)])
    return IdeaGraph.from_tables(ideas, edges)


def test_nonav_policy_generates_citing_sampled_corpus_ideas():
    g = graph_with_hub()
    llm = FakeLLM(responses=["A brand new idea."])
    pol = NoNavLLMPolicy(llm=llm, model="m", graph=g,
                         rng=np.random.default_rng(0), k=3)
    action = pol.act({"step": 0, "last_result": {}})
    assert action.name == "generate"
    assert action.args["text"] == "A brand new idea."
    assert len(action.args["cited_ids"]) == 3
    assert all(cid.startswith("W") for cid in action.args["cited_ids"])
    # The sampled idea texts were shown to the LLM.
    assert "idea" in llm.calls[0]["user"]


def test_pa_policy_prefers_high_in_degree_nodes():
    g = graph_with_hub()
    pol = PreferentialAttachmentPolicy(graph=g, rng=np.random.default_rng(0), m=2)
    hub_hits = 0
    for _ in range(50):
        action = pol.act({"step": 0, "last_result": {}})
        assert action.name == "generate"
        assert len(set(action.args["cited_ids"])) == 2
        if "W0" in action.args["cited_ids"]:
            hub_hits += 1
    assert hub_hits > 40  # in_degree(W0)=4 vs 0 elsewhere -> nearly always picked
