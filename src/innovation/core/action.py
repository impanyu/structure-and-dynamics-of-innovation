"""The one Action type both papers' environments speak (spec §7: core is shared).

`core.policy.Policy.act` is annotated with this type, so it must live in the
shared layer: a `core` module that imported a paper track would make every
paper depend on paper 1. Both `p1_dial.env` and `p2_forum.env` re-export it,
so `from innovation.p1_dial.env import Action` keeps working and there is now
exactly ONE Action class rather than two identically-shaped ones.
"""
from dataclasses import dataclass, field


@dataclass
class Action:
    name: str
    args: dict = field(default_factory=dict)
