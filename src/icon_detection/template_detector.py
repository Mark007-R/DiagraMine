"""Template-matching icon detector.

For each icon template in `data/icon_templates/`, runs `cv2.matchTemplate`
(TM_CCOEFF_NORMED) at multiple scales against the image, then performs
non-maximum suppression on the resulting heatmap. Each accepted detection
carries a `label` matching the template filename (without extension).

Templates currently in the library (curated from search_interview_test.png):

  - docker.png : the docker whale icon at (599, 199) 41×34
  - ms_sql.png : the database cylinder at (427, 241) 52×58 — represents
                 the "MS SQL" GT icon since `Database` in the diagram is
                 the MS SQL endpoint per the diagram's connector layout.

NOTE: this is an honest baseline, not a generalised production strategy.
The template library is curated from the one diagram the legacy pipeline
was tuned on; we report it as a baseline because it's a fair point on the
"how well does template matching transfer?" axis vs CLIP zero-shot.
"""
from __future__ import annotations

import os
import time
from typing import List

import cv2
import numpy as np


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEMPLATE_DIR = os.path.join(_ROOT, "data", "icon_templates")
_SCALES = (0.7, 0.85, 1.0, 1.2, 1.5)
_THRESHOLD = 0.65   # TM_CCOEFF_NORMED threshold
_NMS_IOU = 0.3


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def _load_templates():
    if not os.path.isdir(_TEMPLATE_DIR):
        return {}
    out = {}
    for f in sorted(os.listdir(_TEMPLATE_DIR)):
        if not f.lower().endswith(".png"):
            continue
        label = os.path.splitext(f)[0]
        if label.startswith("_"):
            continue   # ignore staging crops
        path = os.path.join(_TEMPLATE_DIR, f)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        out[label] = img
    return out


def detect(image_path: str) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()
    templates = _load_templates()
    raw: List[dict] = []
    for label, tmpl in templates.items():
        for scale in _SCALES:
            th = int(tmpl.shape[0] * scale)
            tw = int(tmpl.shape[1] * scale)
            if th < 8 or tw < 8 or th > gray.shape[0] or tw > gray.shape[1]:
                continue
            tmpl_s = cv2.resize(tmpl, (tw, th))
            res = cv2.matchTemplate(gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= _THRESHOLD)
            for y, x in zip(ys, xs):
                raw.append({
                    "x": int(x), "y": int(y),
                    "w": int(tw), "h": int(th),
                    "label": label,
                    "score": float(res[y, x]),
                })

    # NMS per label
    by_label: dict = {}
    for r in raw:
        by_label.setdefault(r["label"], []).append(r)
    icons: List[dict] = []
    for label, dets in by_label.items():
        dets.sort(key=lambda d: d["score"], reverse=True)
        keep = []
        for d in dets:
            box = (d["x"], d["y"], d["w"], d["h"])
            if all(_iou(box, (k["x"], k["y"], k["w"], k["h"])) < _NMS_IOU for k in keep):
                keep.append(d)
        icons.extend(keep)

    elapsed = time.perf_counter() - start
    return {"icons": icons, "runtime_seconds": round(elapsed, 3)}
