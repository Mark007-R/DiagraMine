"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def repo_root() -> str:
    return ROOT


@pytest.fixture(scope="session")
def diagrams_dir(repo_root: str) -> str:
    return os.path.join(repo_root, "data", "eval", "diagrams_15")


@pytest.fixture(scope="session")
def test_image(diagrams_dir: str) -> str:
    """The simplest real benchmark diagram — 3-tier web app, 4 boxes, 3 arrows."""
    return os.path.join(diagrams_dir, "diagram_01.png")


@pytest.fixture(scope="session")
def synthetic_4box_image(tmp_path_factory) -> str:
    """A clean 4-box synthetic test image: a 2x2 grid of rectangles with text.

    Used by box-detection tests so they don't depend on EasyOCR weights or
    matplotlib rendering. Pure OpenCV draw.
    """
    import cv2
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    # 2x2 grid of boxes
    boxes = [(50, 50, 200, 120),
             (350, 50, 500, 120),
             (50, 250, 200, 320),
             (350, 250, 500, 320)]
    labels = ["Alpha", "Beta", "Gamma", "Delta"]
    for (x1, y1, x2, y2), lbl in zip(boxes, labels):
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.putText(img, lbl, (x1 + 10, y1 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    path = str(tmp_path_factory.mktemp("synth") / "four_box.png")
    cv2.imwrite(path, img)
    return path
