"""Scan-line dashed arrow detector — wraps the legacy `_find_dash_groups`
logic from `diagram_analysis.detect_arrows`.

Pure detector: returns endpoint coordinates only. Does NOT apply the
legacy `box_inside_text` / `inside_box` filters — those need texts + boxes
+ icons and would conflate the arrow-detector's quality with the box-detector's
quality. For Phase 2b we want to isolate the arrow detector itself.

Output format (shared across all arrow detectors):
    {
      "arrows": [{"x1", "y1", "x2", "y2", "direction"}],
      "runtime_seconds": float,
    }
"""
from __future__ import annotations

import time
from typing import List

import cv2
import numpy as np


def _find_dash_groups(line_pixels, dark_thresh=145, min_dash=3, max_dash=15,
                      min_gap=5, max_gap=16):
    """Identical to diagram_analysis._find_dash_groups — duplicated here so
    the detector module is self-contained and can be re-tuned in Phase 4
    without mutating the legacy file."""
    n = len(line_pixels)
    if n < 15:
        return []
    dark_runs = []
    in_dark = False
    start = 0
    for i in range(n):
        if line_pixels[i] < dark_thresh:
            if not in_dark:
                start = i
                in_dark = True
        else:
            if in_dark:
                dark_runs.append((start, i, i - start))
                in_dark = False
    if in_dark:
        dark_runs.append((start, n, n - start))
    dashes = [(s, e, l) for s, e, l in dark_runs if min_dash <= l <= max_dash]
    if len(dashes) < 3:
        return []
    groups = [[dashes[0]]]
    for k in range(1, len(dashes)):
        gap = dashes[k][0] - dashes[k-1][1]
        if gap > 30:
            groups.append([dashes[k]])
        else:
            groups[-1].append(dashes[k])
    result = []
    for grp in groups:
        if len(grp) < 2:
            continue
        good_gaps = sum(1 for k in range(1, len(grp))
                        if min_gap <= grp[k][0] - grp[k-1][1] <= max_gap)
        if good_gaps >= 1:
            result.append([(s, e) for s, e, _ in grp])
    return result


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    start = time.perf_counter()

    # ── Horizontal scan ──
    h_hits = []
    for y in range(h_img):
        for dashes in _find_dash_groups(gray[y, :].astype(int)):
            x_start, x_end = dashes[0][0], dashes[-1][1]
            span = x_end - x_start
            if 20 <= span <= 800:
                h_hits.append((y, x_start, x_end, len(dashes)))

    h_hits.sort(key=lambda x: x[0])
    h_arrows = []
    if h_hits:
        cluster = [h_hits[0]]
        for hit in h_hits[1:]:
            prev = cluster[-1]
            if hit[0] - prev[0] <= 3 and abs(hit[1] - prev[1]) < 50:
                cluster.append(hit)
            else:
                y_avg = int(np.mean([c[0] for c in cluster]))
                x_min = min(c[1] for c in cluster)
                x_max = max(c[2] for c in cluster)
                h_arrows.append((x_min, y_avg, x_max, y_avg))
                cluster = [hit]
        y_avg = int(np.mean([c[0] for c in cluster]))
        x_min = min(c[1] for c in cluster)
        x_max = max(c[2] for c in cluster)
        h_arrows.append((x_min, y_avg, x_max, y_avg))

    # ── Vertical scan ──
    v_hits = []
    for x in range(w_img):
        for dashes in _find_dash_groups(gray[:, x].astype(int)):
            y_start, y_end = dashes[0][0], dashes[-1][1]
            span = y_end - y_start
            if 20 <= span <= 200:
                v_hits.append((x, y_start, y_end, len(dashes)))

    v_hits.sort(key=lambda x: x[0])
    v_arrows = []
    if v_hits:
        cluster = [v_hits[0]]
        for hit in v_hits[1:]:
            prev = cluster[-1]
            if hit[0] - prev[0] <= 3 and abs(hit[1] - prev[1]) < 50:
                cluster.append(hit)
            else:
                x_avg = int(np.mean([c[0] for c in cluster]))
                y_min = min(c[1] for c in cluster)
                y_max = max(c[2] for c in cluster)
                v_arrows.append((x_avg, y_min, x_avg, y_max))
                cluster = [hit]
        x_avg = int(np.mean([c[0] for c in cluster]))
        y_min = min(c[1] for c in cluster)
        y_max = max(c[2] for c in cluster)
        v_arrows.append((x_avg, y_min, x_avg, y_max))

    # ── Light filters: drop tiny v-arrows + dedupe near-duplicates ──
    v_arrows = [a for a in v_arrows if (a[3] - a[1]) >= 25]

    arrows: List[dict] = []
    for x1, y1, x2, y2 in h_arrows:
        arrows.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "direction": "horizontal",
        })
    for x1, y1, x2, y2 in v_arrows:
        arrows.append({
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "direction": "vertical",
        })

    elapsed = time.perf_counter() - start
    return {"arrows": arrows, "runtime_seconds": round(elapsed, 3)}
