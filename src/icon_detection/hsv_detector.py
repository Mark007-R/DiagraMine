"""HSV-segmentation icon detector — wraps the legacy detect_icons logic
from diagram_analysis.py.

Detects coloured regions that don't overlap text bounding boxes. Returns
generic label `"icon"` — has no notion of icon CLASS (docker vs aws etc.).
That class blindness is the credibility gap; template_matching and CLIP
detectors below close it.
"""
from __future__ import annotations

import time
from typing import List

import cv2


def _dedup_boxes(boxes: List[dict], iou_thresh: float = 0.5) -> List[dict]:
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
    max_area = h_img * w_img * 0.08

    start = time.perf_counter()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    icons: List[dict] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 200 or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue
        icons.append({
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "label": "icon",
        })

    icons = _dedup_boxes(icons)
    elapsed = time.perf_counter() - start
    return {"icons": icons, "runtime_seconds": round(elapsed, 3)}
