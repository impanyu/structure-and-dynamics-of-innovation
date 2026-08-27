"""Baseline policies: isolate the contributions of navigation and of the LLM (spec §3.4)."""
import numpy as np

from innovation.experiments.env import Action
from innovation.agents.policy import Policy
from innovation.llm import LLM

NONAV_TEMPLATE = """Here are {k} ideas from the research literature:

{ideas}

Propose ONE new research idea that builds on them. Write a single 3-4 sentence \
paragraph covering the problem, the key insight, and the method. Write only the paragraph."""


class NoNavLLMPolicy(Policy):
    """Baseline 2: LLM without navigation — random sample of corpus ideas, generate."""

    def __init__(self, *, llm: LLM, model: str, graph, rng: np.random.Generator, k: int = 3):
        self.llm, self.model, self.graph, self.rng, self.k = llm, model, graph, rng, k

    def act(self, obs: dict) -> Action:
        corpus = self.graph.node_ids(source="corpus")
        cited = [str(c) for c in self.rng.choice(corpus, size=self.k, replace=False)]
        ideas = "\n\n".join(f"- {self.graph.node(c).text}" for c in cited)
        text = self.llm.complete(
            model=self.model, system="You propose new research ideas.",
            user=NONAV_TEMPLATE.format(k=self.k, ideas=ideas), max_tokens=400).strip()
        return Action("generate", {"text": text, "cited_ids": cited})


class PreferentialAttachmentPolicy(Policy):
    """Baseline 3: structural null model — no LLM, cites ∝ in-degree, template text."""

    def __init__(self, *, graph, rng: np.random.Generator, m: int = 3):
        self.graph, self.rng, self.m = graph, rng, m

    def act(self, obs: dict) -> Action:
        nodes = self.graph.node_ids()
        weights = np.array([self.graph.in_degree(n) + 1 for n in nodes], dtype=float)
        weights /= weights.sum()
        cited = [str(c) for c in self.rng.choice(nodes, size=self.m,
                                                 replace=False, p=weights)]
        text = "A recombination of the ideas in: " + ", ".join(cited) + "."
        return Action("generate", {"text": text, "cited_ids": cited})
