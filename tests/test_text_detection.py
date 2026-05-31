"""Tests for text detection backends."""
from __future__ import annotations

import os

import pytest


def test_easyocr_returns_schema_valid_dict(test_image: str):
    """EasyOCR detector returns a schema-valid dict with the expected keys."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    assert set(out.keys()) >= {"texts", "runtime_seconds"}
    assert isinstance(out["texts"], list)
    assert out["runtime_seconds"] >= 0.0


def test_easyocr_text_elements_have_required_fields(test_image: str):
    """Every detected text element must have x/y/w/h/text/conf fields."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    for t in out["texts"]:
        assert "text" in t
        assert all(k in t for k in ("x", "y", "w", "h"))
        assert "conf" in t
        assert isinstance(t["text"], str)


def test_easyocr_finds_some_text_on_benchmark(test_image: str):
    """diagram_01.png is a 3-tier web app with 'Browser', 'Web Server',
    'App Server', 'Database' — we expect at least 2 detected labels."""
    from src.text_detection import easyocr_detector as det
    out = det.detect(test_image)
    assert len(out["texts"]) >= 2, (
        f"Expected EasyOCR to find at least 2 text labels on diagram_01, "
        f"got {len(out['texts'])}"
    )
