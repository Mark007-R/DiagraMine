"""Tests for `src.graph.builder` — the data-driven relationship builder."""
from __future__ import annotations


def test_build_relationships_empty_on_no_arrows():
    from src.graph import builder
    boxes = [{"x": 0, "y": 0, "w": 100, "h": 100, "label": "A"},
             {"x": 200, "y": 0, "w": 100, "h": 100, "label": "B"}]
    rels = builder.build_relationships([], boxes)
    assert rels == [], "No arrows must produce no relationships."


def test_build_relationships_maps_arrow_to_nearest_boxes():
    """An arrow whose endpoints sit on two labeled boxes must produce one
    (source, target) edge between those labels."""
    from src.graph import builder
    boxes = [{"x": 0, "y": 0, "w": 100, "h": 100, "label": "Source"},
             {"x": 300, "y": 0, "w": 100, "h": 100, "label": "Target"}]
    # Arrow from box-A right edge (100, 50) to box-B left edge (300, 50)
    arrows = [{"x1": 100, "y1": 50, "x2": 300, "y2": 50,
               "direction": "right", "line_style": "dashed"}]
    rels = builder.build_relationships(arrows, boxes, outside_box_gate=True,
                                       max_dist=120.0)
    assert len(rels) == 1
    r = rels[0]
    assert {r["source"], r["target"]} == {"Source", "Target"}
    assert r["detected"] is True, "Every relationship must be marked detected=True."


def test_outside_box_gate_drops_inside_box_segments():
    """An arrow whose midpoint sits INSIDE a box (i.e. an internal-edge or
    border artifact) must be dropped by the outside-box gate."""
    from src.graph import builder
    boxes = [{"x": 0, "y": 0, "w": 200, "h": 200, "label": "Big"},
             {"x": 400, "y": 0, "w": 100, "h": 100, "label": "Far"}]
    # Arrow midpoint = (50, 50) — inside the "Big" box.
    arrows = [{"x1": 10, "y1": 50, "x2": 90, "y2": 50}]
    rels = builder.build_relationships(arrows, boxes, outside_box_gate=True,
                                       max_dist=120.0)
    assert rels == [], "outside_box_gate must drop arrows whose midpoint is in a box."


def test_no_hardcoded_edges_for_search_interview_diagram(test_image: str):
    """Regression: when the legacy search_interview_test.png is processed, the
    relationships list must not contain the 8 ELSER/Plant-An-App edges that
    `_known_connections()` used to inject. They were data-coded into the source
    on Day 1 and removed on Day 4."""
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    search_img = os.path.join(repo_root, "search_interview_test.png")
    if not os.path.isfile(search_img):
        return  # Image not in repo — skip silently.
    from src import pipeline as pl
    from src.schemas import PipelineConfig
    result = pl.extract(search_img, PipelineConfig())
    sources_targets = {(r.source, r.target) for r in result.relationships}
    # The 8 hardcoded pairs that used to be returned by _known_connections().
    forbidden = {
        ("Server Website", "Plant An App - AWS"),
        ("Search UI", "Elastic Language Client"),
        ("Elastic Language Client", "Database"),
        ("Database", "Elastic Connector for MS SQL"),
        ("Elastic Connector for MS SQL", "DB"),
        ("Elastic Connector for MS SQL", "Indices"),
        ("Indices", "ELSER Model"),
        ("Elastic Language Client", "Elasticsearch Serverless"),
    }
    # It's OK if SOME of these survive (they happen to also be data-driven
    # detections), but the system must never inject all 8 as a block — that
    # would mean `_known_connections` was re-added. Require strict subset.
    intersection = sources_targets & forbidden
    assert len(intersection) < 8, (
        f"All 8 legacy `_known_connections` edges were present in the output. "
        f"This strongly suggests _known_connections() was reintroduced. "
        f"Forbidden block: {forbidden}"
    )


def test_draw_graph_returns_false_on_empty():
    from src.graph import builder
    import os
    out = os.path.join(os.path.dirname(__file__), "_empty_graph_should_not_exist.png")
    result = builder.draw_graph([], out)
    assert result is False
    if os.path.exists(out):
        os.unlink(out)
