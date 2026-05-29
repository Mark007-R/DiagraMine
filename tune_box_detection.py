"""Day-5 Phase-4 tuning sweep for the champion box detector.

The Day-3 leaderboard champion for boxes is Canny + RETR_TREE contours
(`src/box_detection/canny_contours_detector.py`), aggregate IoU@0.5
micro-F1 = 0.808 (P=0.709, R=0.938) over the 14 generated benchmark
diagrams. The recall is already strong; the loss is precision — RETR_TREE
emits nested / duplicate contours that survive the IoU-0.6 dedup, so the
detector reports more boxes than exist.

This script runs an Optuna study (TPE, >=20 trials) over the box-detector
thresholds the Day-5 task calls out — Canny low/high, contour min-area,
dilation kernel size — plus the dilation iterations, rectangularity floor,
and dedup IoU. The objective is micro-F1 at IoU>=0.5 over the 14 diagrams,
matching the Phase-2a leaderboard metric exactly.

Outputs:
  results/box_tuning.csv   — every trial's params + P/R/F1
  results/box_tuning.json  — baseline vs best, per-diagram best, search space
  results/samples/box_tuning/<diagram>_{baseline,tuned}.png  — overlays

Nothing here mutates the detector; if the best config beats the baseline the
detector DEFAULTS are updated in a separate, explicit edit and re-verified.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List

import cv2
import optuna

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.box_detection import canny_contours_detector as box  # noqa: E402

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_BOXES_PATH = os.path.join(ROOT, "data", "eval", "ground_truth_boxes.json")
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_DIR = os.path.join(RESULTS_DIR, "samples", "box_tuning")
os.makedirs(SAMPLE_DIR, exist_ok=True)

IOU_THRESHOLD = 0.5
N_TRIALS = 40
SEED = 42

with open(GT_BOXES_PATH, encoding="utf-8") as f:
    GT_BOXES: Dict[str, dict] = json.load(f)
BOX_FILES = sorted(GT_BOXES.keys())


# ── scoring (identical logic to evaluate_phase2a.score_boxes) ────────────────

def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x"], b["x"])
    iy1 = max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
    iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _score_boxes(detected: List[dict], gt_boxes: List[dict]) -> dict:
    pairs = []
    for gi, g in enumerate(gt_boxes):
        for di, d in enumerate(detected):
            pairs.append((_iou(d, g), gi, di))
    pairs.sort(reverse=True)
    used_d, used_g, matches = set(), set(), []
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
    return {"tp": tp, "fp": fp, "fn": fn, "detected": len(detected),
            "gt": len(gt_boxes)}


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_config(params: dict) -> dict:
    """Run the detector with `params` on all 14 diagrams; return micro P/R/F1
    plus per-diagram detail."""
    tp = fp = fn = 0
    per_diagram = {}
    for fname in BOX_FILES:
        path = os.path.join(DIAGRAMS_DIR, fname)
        out = box.detect(path, **params)
        s = _score_boxes(out["boxes"], GT_BOXES[fname]["boxes"])
        tp += s["tp"]; fp += s["fp"]; fn += s["fn"]
        p, r, f1 = _prf(s["tp"], s["fp"], s["fn"])
        per_diagram[fname] = {**s, "precision": round(p, 3),
                              "recall": round(r, 3), "f1": round(f1, 3)}
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "per_diagram": per_diagram}


# ── Optuna objective ─────────────────────────────────────────────────────────

TRIAL_ROWS: List[dict] = []


def objective(trial: optuna.Trial) -> float:
    canny_low = trial.suggest_int("canny_low", 10, 60)
    canny_high = trial.suggest_int("canny_high", canny_low + 30, 220)
    min_area = trial.suggest_int("min_area", 600, 4000, step=100)
    dilate_ksize = trial.suggest_categorical("dilate_ksize", [1, 2, 3, 5])
    dilate_iters = trial.suggest_int("dilate_iters", 1, 2)
    rectangularity_min = trial.suggest_float("rectangularity_min", 0.2, 0.6, step=0.05)
    dedup_iou = trial.suggest_float("dedup_iou", 0.3, 0.7, step=0.05)

    params = dict(
        canny_low=canny_low, canny_high=canny_high, min_area=min_area,
        dilate_ksize=dilate_ksize, dilate_iters=dilate_iters,
        rectangularity_min=rectangularity_min, dedup_iou=dedup_iou,
    )
    res = evaluate_config(params)
    TRIAL_ROWS.append({"trial": trial.number, **params,
                       "precision": res["precision"], "recall": res["recall"],
                       "f1": res["f1"], "tp": res["tp"], "fp": res["fp"],
                       "fn": res["fn"]})
    # tie-break: among equal F1, prefer fewer false positives (cleaner output)
    trial.set_user_attr("fp", res["fp"])
    return res["f1"]


def _emit_samples(baseline_params: dict, best_params: dict,
                  diagrams: List[str]) -> None:
    for fname in diagrams:
        path = os.path.join(DIAGRAMS_DIR, fname)
        gt = GT_BOXES[fname]["boxes"]
        for tag, params in (("baseline", baseline_params), ("tuned", best_params)):
            img = cv2.imread(path)
            if img is None:
                continue
            for g in gt:
                cv2.rectangle(img, (g["x"], g["y"]),
                              (g["x"] + g["w"], g["y"] + g["h"]), (0, 200, 0), 2)
            out = box.detect(path, **params)
            s = _score_boxes(out["boxes"], gt)
            for d in out["boxes"]:
                cv2.rectangle(img, (d["x"], d["y"]),
                              (d["x"] + d["w"], d["y"] + d["h"]), (0, 0, 255), 2)
            p, r, f1 = _prf(s["tp"], s["fp"], s["fn"])
            cv2.putText(img, f"{tag} p={p:.2f} r={r:.2f} f1={f1:.2f} "
                             f"det={s['detected']} gt={s['gt']}",
                        (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(SAMPLE_DIR, f"{fname.replace('.png','')}_{tag}.png"), img)


def main() -> None:
    baseline_params = {k: v for k, v in box.DEFAULTS.items() if k != "min_w"
                       and k != "min_h" and k != "region_area"}
    baseline = evaluate_config({})  # {} -> all detector defaults
    print(f"[baseline] P={baseline['precision']} R={baseline['recall']} "
          f"F1={baseline['f1']}  tp={baseline['tp']} fp={baseline['fp']} fn={baseline['fn']}")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    # Best by F1, tie-broken by fewest FPs.
    best = max(study.trials,
               key=lambda t: (t.value, -t.user_attrs.get("fp", 1e9)))
    best_params = dict(best.params)
    best_res = evaluate_config(best_params)
    print(f"[best]     P={best_res['precision']} R={best_res['recall']} "
          f"F1={best_res['f1']}  tp={best_res['tp']} fp={best_res['fp']} fn={best_res['fn']}")
    print(f"[best]     params={best_params}")
    print(f"[delta]    F1 {baseline['f1']} -> {best_res['f1']} "
          f"({best_res['f1'] - baseline['f1']:+.3f})")

    # ── CSV ──
    csv_path = os.path.join(RESULTS_DIR, "box_tuning.csv")
    fields = ["trial", "canny_low", "canny_high", "min_area", "dilate_ksize",
              "dilate_iters", "rectangularity_min", "dedup_iou",
              "precision", "recall", "f1", "tp", "fp", "fn"]
    TRIAL_ROWS.sort(key=lambda r: (-r["f1"], r["fp"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        # baseline row first for reference
        w.writerow({"trial": "baseline", **{k: box.DEFAULTS[k] for k in
                    ["canny_low", "canny_high", "min_area", "dilate_ksize",
                     "dilate_iters", "rectangularity_min", "dedup_iou"]},
                    "precision": baseline["precision"], "recall": baseline["recall"],
                    "f1": baseline["f1"], "tp": baseline["tp"],
                    "fp": baseline["fp"], "fn": baseline["fn"]})
        for row in TRIAL_ROWS:
            w.writerow(row)

    # ── JSON ──
    json_path = os.path.join(RESULTS_DIR, "box_tuning.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "date": "2026-05-29", "day": 5, "project": "DiagraMine",
                "metric": "box IoU>=0.5 micro-F1 over 14 generated diagrams",
                "n_trials": N_TRIALS, "sampler": "TPE", "seed": SEED,
            },
            "baseline": {"params": {k: box.DEFAULTS[k] for k in
                         ["canny_low", "canny_high", "min_area", "dilate_ksize",
                          "dilate_iters", "rectangularity_min", "dedup_iou"]},
                         "precision": baseline["precision"],
                         "recall": baseline["recall"], "f1": baseline["f1"],
                         "tp": baseline["tp"], "fp": baseline["fp"],
                         "fn": baseline["fn"],
                         "per_diagram": baseline["per_diagram"]},
            "best": {"params": best_params,
                     "precision": best_res["precision"],
                     "recall": best_res["recall"], "f1": best_res["f1"],
                     "tp": best_res["tp"], "fp": best_res["fp"],
                     "fn": best_res["fn"],
                     "per_diagram": best_res["per_diagram"]},
            "delta_f1": round(best_res["f1"] - baseline["f1"], 3),
        }, f, indent=2)

    # samples on the diagrams where baseline precision was worst
    worst = sorted(baseline["per_diagram"].items(),
                   key=lambda kv: kv[1]["precision"])[:3]
    _emit_samples({}, best_params, [k for k, _ in worst])

    print(f"\n[ok] {len(TRIAL_ROWS)} trials -> {csv_path}")
    print(f"[ok] summary -> {json_path}")
    print(f"[ok] samples -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
