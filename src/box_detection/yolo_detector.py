"""YOLOv8 zero-shot box detector using COCO-pretrained weights.

This is an honest negative-result baseline: COCO has no "rectangle" class,
so YOLOv8 will return whatever rectangle-shaped objects it learned from
natural photos (tv, laptop, microwave, book, etc). On synthetic architecture
diagrams it is expected to detect essentially nothing.

We keep the COMPLETE detection set (no class filter) so the reader can see
exactly what YOLO thinks an architecture diagram contains. The headline:
zero-shot transfer from COCO photos to abstract diagrams fails.
"""
from __future__ import annotations

import time
from typing import List

import cv2

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        # nano model, COCO pretrained — small, fast download.
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def detect(image_path: str, conf_thresh: float = 0.10) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    model = _get_model()

    start = time.perf_counter()
    results = model(img, verbose=False, conf=conf_thresh)
    elapsed = time.perf_counter() - start

    boxes: List[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        names = r.names
        for b in r.boxes:
            xyxy = b.xyxy[0].tolist()
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            boxes.append(
                {
                    "x": x1, "y": y1,
                    "w": x2 - x1, "h": y2 - y1,
                    "area": (x2 - x1) * (y2 - y1),
                    "yolo_class": names.get(cls_id, str(cls_id)),
                    "conf": round(conf, 3),
                }
            )
    return {"boxes": boxes, "runtime_seconds": round(elapsed, 3)}
