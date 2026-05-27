"""Hough-lines arrow detector.

Pipeline:
  1. Adaptive-threshold + thin the image (`cv2.ximgproc.thinning`) so dashed
     and solid lines collapse to 1-pixel skeletons.
  2. Run `cv2.HoughLinesP` to find line segments.
  3. Group collinear segments (same y for horizontal, same x for vertical)
     that lie close together — dashed arrows produce many short collinear
     segments because each dash becomes a separate segment.
  4. Filter:
       - drop very short segments (< 20 px) — likely text strokes / box corners.
       - drop segments that lie ON a box border by ignoring detections inside
         a small dilation of the high-contrast page mask.
       (The latter requires the box detector; we do a cheap proxy via a
        morphological mask of closed dark contours so this detector stays
        self-contained.)

This detector exists to test the hypothesis: replacing the brittle pixel-row
dash scanner with a thinning+Hough pipeline generalises better across dash
spacings that vary per matplotlib rendering pass.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np


def _angle_bucket(x1, y1, x2, y2) -> str:
    """horizontal, vertical, or diagonal."""
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dy <= 3:
        return "horizontal"
    if dx <= 3:
        return "vertical"
    return "diagonal"


def _cluster_collinear(segments: List[Tuple[int, int, int, int]],
                       direction: str,
                       tol: int = 6,
                       gap_tol: int = 60) -> List[dict]:
    """Cluster collinear segments into single arrows.

    For 'horizontal': cluster by y (within tol px), merge along x if gap<=gap_tol.
    For 'vertical':   cluster by x (within tol px), merge along y if gap<=gap_tol.
    """
    if not segments:
        return []

    if direction == "horizontal":
        normed = [(min(x1, x2), max(x1, x2), (y1 + y2) // 2)
                  for x1, y1, x2, y2 in segments]
        normed.sort(key=lambda s: (s[2], s[0]))
        out: List[dict] = []
        cluster = [normed[0]]
        cluster_y = normed[0][2]
        for s in normed[1:]:
            x0, x1_, y = s
            if abs(y - cluster_y) <= tol:
                last = cluster[-1]
                if x0 - last[1] <= gap_tol:
                    cluster.append(s)
                    continue
            # close current cluster
            xs0 = min(c[0] for c in cluster)
            xs1 = max(c[1] for c in cluster)
            ys = int(np.mean([c[2] for c in cluster]))
            if xs1 - xs0 >= 20 and len(cluster) >= 1:
                out.append({"x1": xs0, "y1": ys, "x2": xs1, "y2": ys,
                            "direction": "horizontal",
                            "n_segments": len(cluster)})
            cluster = [s]
            cluster_y = y
        xs0 = min(c[0] for c in cluster)
        xs1 = max(c[1] for c in cluster)
        ys = int(np.mean([c[2] for c in cluster]))
        if xs1 - xs0 >= 20:
            out.append({"x1": xs0, "y1": ys, "x2": xs1, "y2": ys,
                        "direction": "horizontal",
                        "n_segments": len(cluster)})
        return out
    else:
        normed = [((x1 + x2) // 2, min(y1, y2), max(y1, y2))
                  for x1, y1, x2, y2 in segments]
        normed.sort(key=lambda s: (s[0], s[1]))
        out = []
        cluster = [normed[0]]
        cluster_x = normed[0][0]
        for s in normed[1:]:
            x, y0, y1_ = s
            if abs(x - cluster_x) <= tol:
                last = cluster[-1]
                if y0 - last[2] <= gap_tol:
                    cluster.append(s)
                    continue
            xs = int(np.mean([c[0] for c in cluster]))
            ys0 = min(c[1] for c in cluster)
            ys1 = max(c[2] for c in cluster)
            if ys1 - ys0 >= 20:
                out.append({"x1": xs, "y1": ys0, "x2": xs, "y2": ys1,
                            "direction": "vertical",
                            "n_segments": len(cluster)})
            cluster = [s]
            cluster_x = x
        xs = int(np.mean([c[0] for c in cluster]))
        ys0 = min(c[1] for c in cluster)
        ys1 = max(c[2] for c in cluster)
        if ys1 - ys0 >= 20:
            out.append({"x1": xs, "y1": ys0, "x2": xs, "y2": ys1,
                        "direction": "vertical",
                        "n_segments": len(cluster)})
        return out


def _box_border_mask(gray: np.ndarray) -> np.ndarray:
    """Cheap proxy for 'is this pixel on a box border?'. Closed dark
    rectangular contours in the image; we morphologically dilate them so
    a Hough segment lying on the border is filtered out."""
    _, bw = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    # Erode/dilate to get only solid rectangle outlines (no text glyphs)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray, dtype=np.uint8)
    h, w = gray.shape
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        area = ww * hh
        if area < 1500 or ww < 40 or hh < 20:
            continue
        if ww > w * 0.95 and hh > h * 0.95:
            continue
        # draw the rectangle border with a small dilation
        cv2.rectangle(mask, (x, y), (x + ww, y + hh), 255, thickness=4)
    return mask


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()

    # invert: lines become 255, background becomes 0
    _, inv = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Close small dash gaps so the thinning produces continuous skeletons
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    closed_h = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    close_kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    closed_v = cv2.morphologyEx(inv, cv2.MORPH_CLOSE, close_kernel_v, iterations=1)
    closed = cv2.bitwise_or(closed_h, closed_v)

    thin = cv2.ximgproc.thinning(closed, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)

    # Suppress segments lying on box borders
    border_mask = _box_border_mask(gray)
    thin_filt = cv2.bitwise_and(thin, cv2.bitwise_not(border_mask))

    lines = cv2.HoughLinesP(thin_filt, rho=1, theta=np.pi / 180,
                            threshold=18, minLineLength=15, maxLineGap=12)

    horiz_segs: List[Tuple[int, int, int, int]] = []
    vert_segs: List[Tuple[int, int, int, int]] = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            bucket = _angle_bucket(x1, y1, x2, y2)
            if bucket == "horizontal":
                horiz_segs.append((int(x1), int(y1), int(x2), int(y2)))
            elif bucket == "vertical":
                vert_segs.append((int(x1), int(y1), int(x2), int(y2)))
            # diagonals dropped — diagrams here are axis-aligned

    arrows = []
    arrows += _cluster_collinear(horiz_segs, "horizontal")
    arrows += _cluster_collinear(vert_segs, "vertical")

    # Drop arrows shorter than 30 px (text underlines, box corners)
    arrows = [
        a for a in arrows
        if abs(a["x2"] - a["x1"]) + abs(a["y2"] - a["y1"]) >= 30
    ]

    elapsed = time.perf_counter() - start
    out_arrows = [
        {"x1": int(a["x1"]), "y1": int(a["y1"]),
         "x2": int(a["x2"]), "y2": int(a["y2"]),
         "direction": a["direction"]}
        for a in arrows
    ]
    return {"arrows": out_arrows, "runtime_seconds": round(elapsed, 3)}
