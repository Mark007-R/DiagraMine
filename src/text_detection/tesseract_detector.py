"""Tesseract-based text detector via pytesseract. Requires the tesseract.exe
binary on PATH (or pytesseract.tesseract_cmd set). If the binary is missing,
detect() returns {"texts": [], "runtime_seconds": None, "skipped": True,
"reason": "..."}.
"""
from __future__ import annotations

import shutil
import time
from typing import List

import cv2

try:
    import pytesseract
    from pytesseract import Output
    _PYTESSERACT_OK = True
except Exception as exc:  # pragma: no cover
    _PYTESSERACT_OK = False
    _IMPORT_ERR = repr(exc)


def _binary_available() -> bool:
    if not _PYTESSERACT_OK:
        return False
    cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    return shutil.which(cmd) is not None


def detect(image_path: str, min_conf: float = 10.0) -> dict:
    if not _PYTESSERACT_OK:
        return {
            "texts": [],
            "runtime_seconds": None,
            "skipped": True,
            "reason": f"pytesseract import failed: {_IMPORT_ERR}",
        }
    if not _binary_available():
        return {
            "texts": [],
            "runtime_seconds": None,
            "skipped": True,
            "reason": "tesseract.exe binary not on PATH (install from "
                       "https://github.com/UB-Mannheim/tesseract/wiki)",
        }

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()
    data = pytesseract.image_to_data(gray, output_type=Output.DICT)
    elapsed = time.perf_counter() - start

    texts: List[dict] = []
    for i, txt in enumerate(data["text"]):
        if not txt.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < min_conf:
            continue
        texts.append(
            {
                "text": txt.strip(),
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
                "conf": conf / 100.0,
            }
        )
    return {"texts": texts, "runtime_seconds": round(elapsed, 3)}
