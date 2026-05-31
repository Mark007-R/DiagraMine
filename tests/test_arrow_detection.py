"""Tests for arrow detection."""
from __future__ import annotations


def test_hough_lines_returns_schema_valid_dict(test_image: str):
    from src.arrow_detection import hough_lines_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"arrows", "runtime_seconds"}
    assert isinstance(out["arrows"], list)


def test_hough_lines_arrows_have_endpoint_fields(test_image: str):
    """Every arrow has x1/y1/x2/y2 — the (source endpoint, target endpoint)
    pair that downstream graph-builder.build_relationships consumes."""
    from src.arrow_detection import hough_lines_detector as det
    out = det.detect(test_image)
    for a in out["arrows"]:
        assert all(k in a for k in ("x1", "y1", "x2", "y2"))


def test_pixel_scan_detector_schema_valid(test_image: str):
    """The legacy pixel_scan detector still satisfies the schema (the only
    requirement enforced project-wide)."""
    from src.arrow_detection import pixel_scan_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"arrows", "runtime_seconds"}
    assert isinstance(out["arrows"], list)
