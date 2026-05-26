"""Hough-rectangle box detection.

Strategy: cv2.HoughLinesP on the Canny edge map, then group axis-aligned
horizontal and vertical line segments by proximity, and reconstruct rectangles
where four sides (two H + two V) co-localise (pairwise corner intersections).

This is an alternative to RETR_TREE contour scanning that tends to be more
robust to slightly broken rectangles (where edges aren't perfectly closed),
but more sensitive to dashed-arrow line segments masquerading as box sides.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np


def _segments(lines, axis: str, slope_eps: int = 3) -> List[Tuple[int, int, int, int]]:
    """Filter HoughLinesP output to nearly-horizontal or nearly-vertical."""
    out = []
    if lines is None:
        return out
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if axis == "h" and abs(y2 - y1) <= slope_eps:
            out.append((min(x1, x2), (y1 + y2) // 2, max(x1, x2), (y1 + y2) // 2))
        elif axis == "v" and abs(x2 - x1) <= slope_eps:
            out.append(((x1 + x2) // 2, min(y1, y2), (x1 + x2) // 2, max(y1, y2)))
    return out


def _cluster(values: List[int], tol: int = 6) -> List[List[int]]:
    """Greedy 1-D cluster: group values within `tol` of each other."""
    if not values:
        return []
    s = sorted(values)
    clusters = [[s[0]]]
    for v in s[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return clusters


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    h_img, w_img = img.shape[:2]

    start = time.perf_counter()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=60,
        minLineLength=40,
        maxLineGap=8,
    )

    h_segs = _segments(lines, "h")
    v_segs = _segments(lines, "v")

    # Cluster horizontals by y (top/bottom edges) and verticals by x (left/right)
    h_by_y = _cluster([s[1] for s in h_segs], tol=6)
    v_by_x = _cluster([s[0] for s in v_segs], tol=6)

    # Collapse each cluster to its mean y/x, keep the spanning x/y range
    h_lines = []  # (y, x_min, x_max)
    for cluster in h_by_y:
        members = [s for s in h_segs if s[1] in cluster]
        if not members:
            continue
        y_mean = int(round(sum(m[1] for m in members) / len(members)))
        x_min = min(m[0] for m in members)
        x_max = max(m[2] for m in members)
        h_lines.append((y_mean, x_min, x_max))

    v_lines = []  # (x, y_min, y_max)
    for cluster in v_by_x:
        members = [s for s in v_segs if s[0] in cluster]
        if not members:
            continue
        x_mean = int(round(sum(m[0] for m in members) / len(members)))
        y_min = min(m[1] for m in members)
        y_max = max(m[3] for m in members)
        v_lines.append((x_mean, y_min, y_max))

    # Reconstruct boxes: every pair of horizontal lines (top/bottom) and every
    # pair of vertical lines (left/right) where each corner falls near a line
    # segment endpoint.
    boxes: List[dict] = []
    for i, (y1, _, _) in enumerate(h_lines):
        for y2, _, _ in h_lines[i + 1 :]:
            if y2 - y1 < 20:
                continue
            for j, (x1, _, _) in enumerate(v_lines):
                for x2, _, _ in v_lines[j + 1 :]:
                    if x2 - x1 < 40:
                        continue
                    # corners must each be inside both an h-segment span and a v-segment span
                    def corner_supported(x, y):
                        for hy, hxmin, hxmax in h_lines:
                            if abs(hy - y) <= 8 and hxmin - 4 <= x <= hxmax + 4:
                                break
                        else:
                            return False
                        for vx, vymin, vymax in v_lines:
                            if abs(vx - x) <= 8 and vymin - 4 <= y <= vymax + 4:
                                return True
                        return False

                    if not (
                        corner_supported(x1, y1)
                        and corner_supported(x2, y1)
                        and corner_supported(x1, y2)
                        and corner_supported(x2, y2)
                    ):
                        continue
                    w, h = x2 - x1, y2 - y1
                    if w > w_img * 0.95 and h > h_img * 0.95:
                        continue
                    boxes.append({"x": x1, "y": y1, "w": w, "h": h, "area": w * h})

    # Dedup by IoU
    boxes_sorted = sorted(boxes, key=lambda b: b["area"], reverse=True)
    kept: List[dict] = []
    for b in boxes_sorted:
        keep = True
        for k in kept:
            ix1 = max(b["x"], k["x"])
            iy1 = max(b["y"], k["y"])
            ix2 = min(b["x"] + b["w"], k["x"] + k["w"])
            iy2 = min(b["y"] + b["h"], k["y"] + k["h"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = b["area"] + k["area"] - inter
            if union > 0 and inter / union > 0.5:
                keep = False
                break
        if keep:
            kept.append(b)
    elapsed = time.perf_counter() - start
    return {"boxes": kept, "runtime_seconds": round(elapsed, 3)}
