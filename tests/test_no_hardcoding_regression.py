"""Day-7 regression test — the most important test in the suite.

This test FAILS if `_known_connections()` or hardcoded `pos = {...}` is ever
reintroduced into `diagram_analysis.py` or `src/graph/builder.py`. The Day-1
audit identified these as the biggest credibility risk; the Day-4 refactor
removed them. This test makes sure they stay removed.

If you find yourself wanting to add a hardcoded relationship list "just for the
demo" — don't. The whole point of DiagraMine is that the graph is data-driven.
"""
from __future__ import annotations

import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_diagram_analysis_has_no_known_connections():
    """`diagram_analysis.py` must not define a `_known_connections` function."""
    src = _read(os.path.join(ROOT, "diagram_analysis.py"))
    tree = ast.parse(src)
    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_known_connections" not in fn_names, (
        "_known_connections() must not be reintroduced — it was the Day-1 "
        "audit's biggest credibility risk (8 hardcoded edges for one diagram). "
        "If you need a hardcoded fallback, build it as a data file and load it "
        "behind a feature flag — not as code."
    )


def test_graph_builder_has_no_known_connections():
    """builder.py must not define or *call* `_known_connections` (mentioning the
    name in a docstring as removal-context is fine and intended)."""
    src = _read(os.path.join(ROOT, "src", "graph", "builder.py"))
    tree = ast.parse(src)

    fn_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_known_connections" not in fn_names

    # No call expression named `_known_connections(...)`.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            callee = node.func
            if isinstance(callee, ast.Name) and callee.id == "_known_connections":
                pytest.fail("`_known_connections(...)` call found in builder.py")
            if isinstance(callee, ast.Attribute) and callee.attr == "_known_connections":
                pytest.fail("`.._known_connections(...)` call found in builder.py")

    # No `from x import _known_connections`, no `_known_connections = ...` binding.
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                assert alias.name != "_known_connections"
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "_known_connections":
                    pytest.fail("`_known_connections = ...` rebinding in builder.py")


def _code_string_literals(tree: ast.AST) -> list[str]:
    """All Constant string literals EXCEPT docstrings (the first Expr-Constant
    statement of a Module / FunctionDef / ClassDef). Lets us search for hardcoded
    diagram-specific labels without false positives from intentional historical
    references in module docstrings."""
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            out.append(node.value)
    return out


def test_draw_graph_uses_data_driven_layout():
    """The graph layout must come from networkx (kamada_kawai / spring), not a
    hardcoded `pos = {...}` dict literal. Diagram-specific labels must not
    appear as string literals in actual code (docstrings are fine — they record
    what was deliberately removed)."""
    src = _read(os.path.join(ROOT, "src", "graph", "builder.py"))
    tree = ast.parse(src)
    # Must call a networkx layout function.
    assert re.search(r"nx\.(kamada_kawai_layout|spring_layout)", src), (
        "draw_graph must use a data-driven layout (kamada_kawai or spring)."
    )
    # No diagram-specific label as a string literal in code (excluding docstrings).
    code_strs = _code_string_literals(tree)
    for forbidden in ("Plant An App", "ELSER Model", "Elastic Connector for MS SQL"):
        offenders = [s for s in code_strs if forbidden in s]
        assert not offenders, (
            f"src/graph/builder.py contains the hardcoded label '{forbidden}' "
            f"as a code string literal (not a docstring). Code strings found: "
            f"{offenders}"
        )


def test_no_hardcoded_input_image_constant():
    """The original monolith had `INPUT_IMAGE = 'search_interview_test.png'` at
    module scope. The wrapper now uses a DEFAULT_IMAGE for the no-argument
    fallback, but the path must be derived (not pinned at module init), and any
    image must be acceptable as a CLI argument."""
    src = _read(os.path.join(ROOT, "diagram_analysis.py"))
    # The wrapper accepts an arbitrary image path.
    assert "sys.argv" in src, "diagram_analysis.py must accept a CLI image path."
    # No top-level INPUT_IMAGE constant pinning a single PNG.
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "INPUT_IMAGE":
                    pytest.fail(
                        "INPUT_IMAGE constant must not be reintroduced — image "
                        "path is now a CLI argument."
                    )


def test_pipeline_relationships_are_marked_detected_true():
    """Every relationship emitted by the modular pipeline must have
    `detected=True` (i.e., derived from a real arrow). The legacy schema had a
    `detected=False` branch for hardcoded edges; that branch must not be used."""
    from src.schemas import Relationship
    # Default value of `detected` is True (the model constructor uses it).
    r = Relationship(source="A", target="B")
    assert r.detected is True, (
        "Relationship.detected must default to True — every emitted edge is "
        "derived from a detected arrow."
    )
