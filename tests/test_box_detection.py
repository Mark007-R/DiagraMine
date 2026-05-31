"""Tests for the canny+contours box detector — synthetic 4-box image."""
from __future__ import annotations


def test_canny_contours_finds_four_boxes_on_synthetic(synthetic_4box_image: str):
    from src.box_detection import canny_contours_detector as det
    out = det.detect(synthetic_4box_image)
    assert "boxes" in out
    assert isinstance(out["boxes"], list)
    # On a clean 2x2 grid the detector should find at least 4 boxes; the
    # outer page rectangle may show up too but the candidate count must >= 4.
    boxes = [b for b in out["boxes"] if not b.get("is_region")]
    assert len(boxes) >= 4, f"Expected >=4 boxes, got {len(boxes)}"
    assert "runtime_seconds" in out
    assert out["runtime_seconds"] >= 0.0


def test_schema_valid_dict_on_empty_image(tmp_path):
    """Even on a blank white image the detector must return a schema-valid dict
    (boxes=[], runtime_seconds>=0) — the reliability guarantee."""
    import cv2
    import numpy as np
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    p = str(tmp_path / "blank.png")
    cv2.imwrite(p, blank)
    from src.box_detection import canny_contours_detector as det
    out = det.detect(p)
    assert set(out.keys()) >= {"boxes", "runtime_seconds"}
    assert isinstance(out["boxes"], list)
