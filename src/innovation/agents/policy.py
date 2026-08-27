"""Policy ABC: the LLM agent is one policy among several (spec §3.4)."""
from abc import ABC, abstractmethod

from innovation.experiments.env import Action


class Policy(ABC):
    @abstractmethod
    def act(self, obs: dict) -> Action:
        """obs = {"step": int, "last_result": dict}; returns the next Action."""
