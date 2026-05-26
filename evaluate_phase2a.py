"""Phase 2a benchmark harness — Day 2 of the DiagraMine sprint.

Runs every text detector (EasyOCR, Tesseract, PaddleOCR) and every box
detector (Canny+contours, Hough rectangles, YOLOv8 zero-shot) against the
15-diagram benchmark, scoring against:

  • Text:   ground_truth.json["components"]    (fuzzy match @ ratio>=60)
  • Boxes:  ground_truth_boxes.json["boxes"]   (IoU@0.5, generated diagrams only)

Outputs:
  results/phase2a_text_box.csv          (per-detector aggregate)
  results/phase2a_per_diagram.csv       (per-detector per-diagram)
  results/phase2a_detail.json           (all raw matches/runtimes)
  results/samples/text/<detector>_<diagram>.txt   (5 samples)
  results/samples/box/<detector>_<diagram>.png    (5 samples annotated)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from typing import Dict, List

import cv2
from rapidfuzz import fuzz

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.text_detection import easyocr_detector, paddle_detector, tesseract_detector
from src.box_detection import canny_contours_detector, hough_detector, yolo_detector

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_LABELS = json.load(
    open(os.path.join(ROOT, "data", "eval", "ground_truth.json"), encoding="utf-8")
)
GT_BOXES = json.load(
    open(os.path.join(ROOT, "data", "eval", "ground_truth_boxes.json"), encoding="utf-8")
)
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_DIR_TEXT = os.path.join(RESULTS_DIR, "samples", "text")
SAMPLE_DIR_BOX = os.path.join(RESULTS_DIR, "samples", "box")
os.makedirs(SAMPLE_DIR_TEXT, exist_ok=True)
os.makedirs(SAMPLE_DIR_BOX, exist_ok=True)

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

FUZZ_THRESHOLD = 60  # match yesterday's baseline
IOU_THRESHOLD = 0.5
SAMPLE_DIAGRAMS = [
    "diagram_01.png",
    "diagram_05.png",
    "diagram_07.png",
    "diagram_10.png",
    "diagram_14.png",
]


# ---------------------------------------------------------------------------
# Text scoring
# ---------------------------------------------------------------------------

def score_text(detected: List[dict], gt_labels: List[str]) -> dict:
    """Bipartite fuzzy match. Each GT label may match at most one detected
    string; we greedily pick the best-scoring pair until no pair >= threshold.
    Per-char accuracy = mean fuzz ratio across matched pairs (0..1 scale).
    """
    det_texts = [t["text"] for t in detected if t["text"].strip()]
    used_d = set()
    used_g = set()
    matches = []  # (gt, det, ratio)

    pairs = []
    for gi, gt in enumerate(gt_labels):
        for di, dt in enumerate(det_texts):
            r = fuzz.ratio(dt.lower(), gt.lower())
            pairs.append((r, gi, di))
    pairs.sort(reverse=True)
    for r, gi, di in pairs:
        if r < FUZZ_THRESHOLD:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gt_labels[gi], det_texts[di], r))

    tp = len(matches)
    fp = len(det_texts) - tp
    fn = len(gt_labels) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    per_char = (sum(m[2] for m in matches) / len(matches) / 100.0) if matches else 0.0
    return {
        "detected": len(det_texts),
        "ground_truth": len(gt_labels),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "per_char_accuracy": round(per_char, 3),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Box scoring (IoU @ 0.5)
# ---------------------------------------------------------------------------

def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def score_boxes(detected: List[dict], gt_boxes: List[dict]) -> dict:
    """Bipartite IoU@0.5 matching. Each GT box matches at most one detected."""
    used_d, used_g = set(), set()
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, d in enumerate(detected):
            pairs.append((iou(d, g), gi, di))
    pairs.sort(reverse=True)
    matches = []
    for v, gi, di in pairs:
        if v < IOU_THRESHOLD:
            break
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)
        matches.append((gi, di, round(v, 3)))

    tp = len(matches)
    fp = len(detected) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fp_rate = fp / len(detected) if detected else 0.0
    return {
        "detected": len(detected),
        "ground_truth": len(gt_boxes),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "fp_rate": round(fp_rate, 3),
        "matches": matches,
    }


# ---------------------------------------------------------------------------
# Sample emitters
# ---------------------------------------------------------------------------

def emit_text_sample(name: str, fname: str, detected_texts: List[dict],
                     score: dict) -> None:
    out = os.path.join(SAMPLE_DIR_TEXT, f"{name}__{fname.replace('.png','.txt')}")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# {name} on {fname}\n")
        f.write(f"# precision={score['precision']} recall={score['recall']} "
                f"f1={score['f1']} per_char_acc={score['per_char_accuracy']}\n\n")
        f.write("# Detected texts:\n")
        for t in detected_texts:
            txt = t['text']
            f.write(f"  ({t['x']:4d},{t['y']:4d}) {txt!r:<40} conf={t.get('conf', 0):.2f}\n")
        f.write("\n# Matches (gt -> detected, fuzz_ratio):\n")
        for gt, dt, r in score['matches']:
            f.write(f"  {gt!r:<35} -> {dt!r:<35} {r}\n")


def emit_box_sample(name: str, fname: str, image_path: str, detected: List[dict],
                    gt_boxes: List[dict], score: dict) -> None:
    img = cv2.imread(image_path)
    if img is None:
        return
    # GT in green
    for g in gt_boxes:
        cv2.rectangle(img, (g["x"], g["y"]),
                      (g["x"] + g["w"], g["y"] + g["h"]),
                      (0, 200, 0), 2)
        cv2.putText(img, g["label"], (g["x"] + 4, g["y"] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 120, 0), 1)
    # Detected in red
    for d in detected:
        cv2.rectangle(img, (d["x"], d["y"]),
                      (d["x"] + d["w"], d["y"] + d["h"]),
                      (0, 0, 255), 2)
        if "yolo_class" in d:
            cv2.putText(img, d["yolo_class"], (d["x"] + 4, d["y"] + d["h"] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    label = (f"{name} p={score['precision']} r={score['recall']} "
             f"f1={score['f1']} det={len(detected)} gt={len(gt_boxes)}")
    cv2.putText(img, label, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    out = os.path.join(SAMPLE_DIR_BOX, f"{name}__{fname}")
    cv2.imwrite(out, img)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run() -> None:
    detail: Dict[str, dict] = {"text": {}, "box": {}, "skipped": []}
    per_diagram_rows: List[dict] = []

    diagram_files = sorted(GT_LABELS.keys())

    # ── TEXT ─────────────────────────────────────────────────────────────
    for det_name, mod in TEXT_DETECTORS.items():
        per_diag = {}
        skipped_reason = None
        agg = {"tp": 0, "fp": 0, "fn": 0, "per_char_sum": 0.0, "n_matches": 0,
               "runtime_sum": 0.0, "n_runs": 0}
        for fname in diagram_files:
            path = os.path.join(DIAGRAMS_DIR, fname)
            t0 = time.perf_counter()
            try:
                out = mod.detect(path)
            except Exception as exc:
                print(f"[text] {det_name} on {fname} FAILED: {exc!r}")
                continue
            wall = time.perf_counter() - t0
            if out.get("skipped"):
                skipped_reason = out.get("reason")
                print(f"[text] {det_name} SKIPPED: {skipped_reason}")
                break
            gt = GT_LABELS[fname]["components"]
            score = score_text(out["texts"], gt)
            per_diag[fname] = {**score, "runtime_seconds": out["runtime_seconds"]}
            agg["tp"] += score["tp"]; agg["fp"] += score["fp"]; agg["fn"] += score["fn"]
            if score["matches"]:
                agg["per_char_sum"] += sum(m[2] for m in score["matches"])
                agg["n_matches"] += len(score["matches"])
            agg["runtime_sum"] += out["runtime_seconds"]
            agg["n_runs"] += 1
            per_diagram_rows.append({
                "stage": "text", "detector": det_name, "diagram": fname,
                "detected": score["detected"], "gt": score["ground_truth"],
                "tp": score["tp"], "fp": score["fp"], "fn": score["fn"],
                "precision": score["precision"], "recall": score["recall"],
                "f1": score["f1"], "per_char_accuracy": score["per_char_accuracy"],
                "runtime_seconds": out["runtime_seconds"],
            })
            if fname in SAMPLE_DIAGRAMS:
                emit_text_sample(det_name, fname, out["texts"], score)
        if skipped_reason:
            detail["text"][det_name] = {"skipped": True, "reason": skipped_reason}
            detail["skipped"].append({"stage": "text", "detector": det_name,
                                       "reason": skipped_reason})
            continue
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_char = (agg["per_char_sum"] / agg["n_matches"] / 100.0) if agg["n_matches"] else 0.0
        avg_rt = agg["runtime_sum"] / agg["n_runs"] if agg["n_runs"] else 0.0
        detail["text"][det_name] = {
            "per_diagram": per_diag,
            "aggregate": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "per_char_accuracy": round(per_char, 3),
                "avg_runtime_seconds": round(avg_rt, 3),
            },
        }
        print(f"[text] {det_name:<11} p={precision:.3f} r={recall:.3f} "
              f"f1={f1:.3f} per_char_acc={per_char:.3f} avg_rt={avg_rt:.2f}s")

    # ── BOX ──────────────────────────────────────────────────────────────
    # Pixel-IoU evaluation only on the 14 generated diagrams (search_interview_test
    # has no machine-readable GT bboxes).
    box_files = [f for f in diagram_files if f in GT_BOXES]
    for det_name, mod in BOX_DETECTORS.items():
        per_diag = {}
        agg = {"tp": 0, "fp": 0, "fn": 0, "fp_total": 0, "det_total": 0,
               "runtime_sum": 0.0, "n_runs": 0}
        for fname in box_files:
            path = os.path.join(DIAGRAMS_DIR, fname)
            try:
                out = mod.detect(path)
            except Exception as exc:
                print(f"[box] {det_name} on {fname} FAILED: {exc!r}")
                continue
            gt = GT_BOXES[fname]["boxes"]
            score = score_boxes(out["boxes"], gt)
            per_diag[fname] = {**score, "runtime_seconds": out["runtime_seconds"]}
            agg["tp"] += score["tp"]; agg["fp"] += score["fp"]; agg["fn"] += score["fn"]
            agg["fp_total"] += score["fp"]; agg["det_total"] += score["detected"]
            agg["runtime_sum"] += out["runtime_seconds"]
            agg["n_runs"] += 1
            per_diagram_rows.append({
                "stage": "box", "detector": det_name, "diagram": fname,
                "detected": score["detected"], "gt": score["ground_truth"],
                "tp": score["tp"], "fp": score["fp"], "fn": score["fn"],
                "precision": score["precision"], "recall": score["recall"],
                "f1": score["f1"], "fp_rate": score["fp_rate"],
                "runtime_seconds": out["runtime_seconds"],
            })
            if fname in SAMPLE_DIAGRAMS:
                emit_box_sample(det_name, fname, path, out["boxes"], gt, score)
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        fp_rate = agg["fp_total"] / agg["det_total"] if agg["det_total"] else 0.0
        avg_rt = agg["runtime_sum"] / agg["n_runs"] if agg["n_runs"] else 0.0
        detail["box"][det_name] = {
            "per_diagram": per_diag,
            "aggregate": {
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1": round(f1, 3),
                "fp_rate": round(fp_rate, 3),
                "avg_runtime_seconds": round(avg_rt, 3),
            },
        }
        print(f"[box]  {det_name:<18} p={precision:.3f} r={recall:.3f} "
              f"f1={f1:.3f} fp_rate={fp_rate:.3f} avg_rt={avg_rt:.2f}s")

    # ── Write CSVs ───────────────────────────────────────────────────────
    leaderboard_path = os.path.join(RESULTS_DIR, "phase2a_text_box.csv")
    with open(leaderboard_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "detector", "precision", "recall", "f1",
                    "per_char_accuracy", "fp_rate", "avg_runtime_seconds", "notes"])
        for det_name, info in detail["text"].items():
            if info.get("skipped"):
                w.writerow(["text", det_name, "", "", "", "", "", "",
                            "SKIPPED: " + info["reason"]])
                continue
            agg = info["aggregate"]
            w.writerow(["text", det_name, agg["precision"], agg["recall"],
                        agg["f1"], agg["per_char_accuracy"], "",
                        agg["avg_runtime_seconds"], "fuzz>=60 over 15 diagrams"])
        for det_name, info in detail["box"].items():
            agg = info["aggregate"]
            w.writerow(["box", det_name, agg["precision"], agg["recall"],
                        agg["f1"], "", agg["fp_rate"],
                        agg["avg_runtime_seconds"],
                        f"IoU>=0.5 over {len(box_files)} generated diagrams"])

    per_diag_path = os.path.join(RESULTS_DIR, "phase2a_per_diagram.csv")
    with open(per_diag_path, "w", newline="", encoding="utf-8") as f:
        if per_diagram_rows:
            all_keys: List[str] = []
            seen = set()
            for r in per_diagram_rows:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        all_keys.append(k)
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            for row in per_diagram_rows:
                w.writerow({k: row.get(k, "") for k in all_keys})

    detail_path = os.path.join(RESULTS_DIR, "phase2a_detail.json")
    # Strip the verbose match lists from text detail to keep file size sane;
    # they're already in samples/text/.
    for det_name, info in detail["text"].items():
        if info.get("skipped"):
            continue
        for fname, pd in info["per_diagram"].items():
            pd.pop("matches", None)
    for det_name, info in detail["box"].items():
        for fname, pd in info["per_diagram"].items():
            pd.pop("matches", None)
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    print("\nLeaderboard ->", leaderboard_path)
    print("Per-diagram  ->", per_diag_path)
    print("Detail JSON  ->", detail_path)


if __name__ == "__main__":
    run()
