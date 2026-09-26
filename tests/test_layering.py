"""core/ is shared (spec §7): it must not import either paper track.

A `core` module importing `p1_dial` would make paper 2 depend transitively on
paper 1 through `core.policy`, and it is how the repo ended up with two
identically-shaped `Action` dataclasses.
"""
import ast
from pathlib import Path

import innovation.core.action
import innovation.p1_dial.env
import innovation.p2_forum.env

CORE = Path("src/innovation/core")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_core_never_imports_a_paper_track():
    offenders = {}
    for py in sorted(CORE.rglob("*.py")):
        bad = sorted(m for m in _imported_modules(py)
                     if m.startswith(("innovation.p1_dial", "innovation.p2_forum")))
        if bad:
            offenders[str(py)] = bad
    assert offenders == {}


def test_there_is_exactly_one_action_type():
    assert innovation.p1_dial.env.Action is innovation.core.action.Action
    assert innovation.p2_forum.env.Action is innovation.core.action.Action


def test_the_shared_action_keeps_paper_ones_shape():
    a = innovation.p1_dial.env.Action("search")
    assert (a.name, a.args) == ("search", {})
    assert innovation.p1_dial.env.Action("generate", {"text": "x"}).args == {"text": "x"}
    # default args must not be a shared mutable
    first, second = innovation.core.action.Action("a"), innovation.core.action.Action("b")
    first.args["k"] = 1
    assert second.args == {}
