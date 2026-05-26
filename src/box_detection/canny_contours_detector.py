"""Canny + RETR_TREE contour box detection — equivalent to the current
diagram_analysis.detect_boxes_and_regions but stripped down to just the
box-detection portion (no text labeling, no region/parent nesting). Returns
the same (x, y, w, h) the legacy pipeline would call a 'box'."""
from __future__ import annotations

import time
from typing import List

import cv2


def _dedup_boxes(boxes: List[dict], iou_thresh: float = 0.6) -> List[dict]:
    keep = []
    used = set()
    order = sorted(range(len(boxes)), key=lambda i: boxes[i]["w"] * boxes[i]["h"], reverse=True)
    for i in order:
        if i in used:
            continue
        keep.append(boxes[i])
        for j in order:
            if j in used or j == i:
                continue
            a, b = boxes[i], boxes[j]
            ix1 = max(a["x"], b["x"])
            iy1 = max(a["y"], b["y"])
            ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
            iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = a["w"] * a["h"] + b["w"] * b["h"] - inter
            if union > 0 and inter / union > iou_thresh:
                used.add(j)
        used.add(i)
    return keep


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h_img, w_img = img.shape[:2]

    start = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1200 or w < 40 or h < 20:
            continue
        rect_area = w * h
        if rect_area == 0:
            continue
        rectangularity = cv2.contourArea(cnt) / rect_area
        if rectangularity < 0.3:
            continue
        if w > w_img * 0.95 and h > h_img * 0.95:
            continue
        # The legacy split: REGIONS (area > 25000) vs BOXES (area <= 25000).
        # For Phase 2a evaluation we want all rectangular objects, so we
        # keep both, but flag them.
        boxes.append(
            {
                "x": int(x), "y": int(y),
                "w": int(w), "h": int(h),
                "area": int(area),
                "is_region": area > 25000,
            }
        )

    boxes = _dedup_boxes(boxes, iou_thresh=0.6)
    elapsed = time.perf_counter() - start
    return {"boxes": boxes, "runtime_seconds": round(elapsed, 3)}
