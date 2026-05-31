"""Tests for icon detection."""
from __future__ import annotations


def test_template_matching_returns_schema_valid_dict(test_image: str):
    from src.icon_detection import template_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"icons", "runtime_seconds"}
    assert isinstance(out["icons"], list)


def test_template_matching_no_false_positives_on_iconless_diagram(test_image: str):
    """diagram_01.png has no icons in ground truth — the template detector
    should not over-detect on it. Template matching with curated templates is
    expected to return 0 (or at most a small number) on this clean diagram."""
    from src.icon_detection import template_detector as det
    out = det.detect(test_image)
    assert len(out["icons"]) <= 2, (
        f"template_matching over-detected on diagram_01 (no GT icons): "
        f"got {len(out['icons'])} > 2"
    )


def test_hsv_detector_schema_valid(test_image: str):
    from src.icon_detection import hsv_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"icons", "runtime_seconds"}
    assert isinstance(out["icons"], list)
