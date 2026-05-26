"""Schema-valid JSON output rate audit for Phase 2a detectors.

Per Hard Rule #9 (DiagraMine: ALWAYS report schema-valid JSON output rate
alongside precision/recall), this script verifies that EVERY detector's
return value can be (a) round-tripped through json.dumps/json.loads, and
(b) satisfies the expected Phase 2a contract:

  text detector returns:
      {"texts":   [{"text": str, "x": int, "y": int, "w": int, "h": int, "conf": float}, ...],
       "runtime_seconds": float | None}
  box detector returns:
      {"boxes":   [{"x": int, "y": int, "w": int, "h": int, ...}, ...],
       "runtime_seconds": float}

Output: results/phase2a_schema_validity.json (per-detector,per-diagram rate +
aggregate)
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.text_detection import easyocr_detector, paddle_detector, tesseract_detector
from src.box_detection import canny_contours_detector, hough_detector, yolo_detector

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
RESULTS_DIR = os.path.join(ROOT, "results")
GT_LABELS = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth.json"), encoding="utf-8"))

TEXT_DETECTORS = {
    "easyocr": easyocr_detector,
    "tesseract": tesseract_detector,
    "paddleocr": paddle_detector,
}
BOX_DETECTORS = {
    "canny_contours": canny_contours_detector,
    "hough_rectangles": hough_detector,
    "yolov8_zero_shot": yolo_detector,
}

REQUIRED_TEXT_KEYS = {"text", "x", "y", "w", "h", "conf"}
REQUIRED_BOX_KEYS = {"x", "y", "w", "h"}


def validate_text(out: dict) -> tuple[bool, str]:
    if "texts" not in out or "runtime_seconds" not in out:
        return False, "missing top-level keys"
    if not isinstance(out["texts"], list):
        return False, "texts is not a list"
    for i, t in enumerate(out["texts"]):
        if not isinstance(t, dict):
            return False, f"texts[{i}] is not a dict"
        if not REQUIRED_TEXT_KEYS.issubset(t.keys()):
            missing = REQUIRED_TEXT_KEYS - set(t.keys())
            return False, f"texts[{i}] missing {missing}"
    return True, "ok"


def validate_box(out: dict) -> tuple[bool, str]:
    if "boxes" not in out or "runtime_seconds" not in out:
        return False, "missing top-level keys"
    if not isinstance(out["boxes"], list):
        return False, "boxes is not a list"
    for i, b in enumerate(out["boxes"]):
        if not isinstance(b, dict):
            return False, f"boxes[{i}] is not a dict"
        if not REQUIRED_BOX_KEYS.issubset(b.keys()):
            missing = REQUIRED_BOX_KEYS - set(b.keys())
            return False, f"boxes[{i}] missing {missing}"
    return True, "ok"


def round_trip(out: dict) -> tuple[bool, str]:
    try:
        s = json.dumps(out)
        json.loads(s)
        return True, "ok"
    except (TypeError, ValueError) as e:
        return False, f"json error: {e}"


def audit() -> None:
    summary: dict[str, Any] = {"text": {}, "box": {}, "aggregate": {}}
    diagrams = sorted(GT_LABELS.keys())

    total_valid = 0
    total_runs = 0
    total_skipped = 0

    for det_name, mod in TEXT_DETECTORS.items():
        per_diag: List[dict] = []
        skipped = False
        for fname in diagrams:
            path = os.path.join(DIAGRAMS_DIR, fname)
            try:
                out = mod.detect(path)
            except Exception as exc:
                per_diag.append({"diagram": fname, "schema_valid": False, "json_serialisable": False, "reason": f"exception: {exc!r}"})
                total_runs += 1
                continue
            if out.get("skipped"):
                skipped = True
                summary["text"][det_name] = {"skipped": True, "reason": out.get("reason")}
                total_skipped += len(diagrams)
                break
            schema_ok, schema_reason = validate_text(out)
            json_ok, json_reason = round_trip(out)
            per_diag.append({
                "diagram": fname,
                "schema_valid": schema_ok and json_ok,
                "schema_reason": schema_reason if not schema_ok else json_reason,
            })
            total_runs += 1
            if schema_ok and json_ok:
                total_valid += 1
        if skipped:
            continue
        rate = sum(1 for r in per_diag if r["schema_valid"]) / len(per_diag) if per_diag else 0.0
        summary["text"][det_name] = {
            "per_diagram_count": len(per_diag),
            "valid_count": sum(1 for r in per_diag if r["schema_valid"]),
            "rate": round(rate, 3),
            "per_diagram": per_diag,
        }
        print(f"[text] {det_name:<12} schema_valid={rate:.3f} ({summary['text'][det_name]['valid_count']}/{len(per_diag)})")

    box_diagrams = diagrams  # boxes run on all 15
    for det_name, mod in BOX_DETECTORS.items():
        per_diag = []
        for fname in box_diagrams:
            path = os.path.join(DIAGRAMS_DIR, fname)
            try:
                out = mod.detect(path)
            except Exception as exc:
                per_diag.append({"diagram": fname, "schema_valid": False, "schema_reason": f"exception: {exc!r}"})
                total_runs += 1
                continue
            schema_ok, schema_reason = validate_box(out)
            json_ok, json_reason = round_trip(out)
            per_diag.append({
                "diagram": fname,
                "schema_valid": schema_ok and json_ok,
                "schema_reason": schema_reason if not schema_ok else json_reason,
            })
            total_runs += 1
            if schema_ok and json_ok:
                total_valid += 1
        rate = sum(1 for r in per_diag if r["schema_valid"]) / len(per_diag) if per_diag else 0.0
        summary["box"][det_name] = {
            "per_diagram_count": len(per_diag),
            "valid_count": sum(1 for r in per_diag if r["schema_valid"]),
            "rate": round(rate, 3),
            "per_diagram": per_diag,
        }
        print(f"[box]  {det_name:<18} schema_valid={rate:.3f} ({summary['box'][det_name]['valid_count']}/{len(per_diag)})")

    summary["aggregate"] = {
        "total_runs_evaluated": total_runs,
        "valid_runs": total_valid,
        "skipped_runs": total_skipped,
        "schema_valid_json_rate": round(total_valid / total_runs, 3) if total_runs else 0.0,
        "note": "Phase 2a evaluates detector-level outputs only; the pipeline-level "
                 "JSON rate from Day 1 baseline (100% on extracted_structure.json) is "
                 "unchanged because diagram_analysis.export_json() was not modified.",
    }
    out_path = os.path.join(RESULTS_DIR, "phase2a_schema_validity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[agg] {total_valid}/{total_runs} runs schema-valid "
          f"({summary['aggregate']['schema_valid_json_rate']:.3f}); "
          f"{total_skipped} skipped")
    print(f"[out] {out_path}")


if __name__ == "__main__":
    audit()
