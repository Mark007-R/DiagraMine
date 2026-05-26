"""EasyOCR-based text detector. Mirrors the current diagram_analysis.detect_text
behaviour (Reader(en, gpu=False), conf>=0.10) but without the legacy
_merge_nearby_texts post-processing so that comparison across OCR backends is
apples-to-apples on raw per-fragment detections."""
from __future__ import annotations

import time
from typing import List

import cv2
import easyocr
import numpy as np

_READER: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    global _READER
    if _READER is None:
        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def detect(image_path: str, min_conf: float = 0.10) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    reader = _get_reader()
    start = time.perf_counter()
    raw = reader.readtext(img)
    elapsed = time.perf_counter() - start

    texts: List[dict] = []
    for bbox, text, conf in raw:
        if conf < min_conf:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        texts.append(
            {
                "text": text.strip(),
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "conf": float(conf),
            }
        )
    return {"texts": texts, "runtime_seconds": round(elapsed, 3)}
