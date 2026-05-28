"""
Day-1 baseline evaluator for DiagraMine.

Runs the existing `diagram_analysis.py` pipeline on every diagram in
`data/eval/diagrams_15/`, twice:

  1. WITH `_known_connections()` enabled (the published behavior).
  2. WITHOUT `_known_connections()` — monkey-patched to return [] — so we
     measure what the CV pipeline alone delivers, with no hardcoded
     answer key.

Computes for each diagram + each mode:
  - component precision / recall (fuzzy label match against ground truth)
  - arrow / relationship precision / recall (pair-based, label-fuzzy)
  - icon precision / recall (count-based — positional matching deferred)
  - schema-valid JSON output rate (does the structure match the expected
    top-level keys?)

Saves the per-diagram detail + the aggregate to `results/baseline_metrics.json`.

The existing `diagram_analysis.py` is NOT modified. We monkey-patch the
module-level `INPUT_IMAGE` constant and inline-execute the pipeline. The
deliberate hardcoding (`_known_connections`, hardcoded `pos`) stays in
place for Day 1 measurement — Day 4 removes them.

FROZEN (Day 4): this is a Day-1 baseline harness. Its entire purpose was to
measure the pipeline WITH vs WITHOUT the hardcoded `_known_connections()`. On
Day 4 that hardcoding was deleted for good and `diagram_analysis.py` became a
thin wrapper over `src/pipeline.py`, so the legacy stage functions and the
`_known_connections` symbol this harness monkey-patches no longer exist. The
Day-1 results it produced are preserved in `results/baseline_metrics.json`.
To re-evaluate the de-hardcoded pipeline, use `evaluate_phase2a.py` /
`evaluate_phase2b.py` and `src/pipeline.py` instead. Running this file now
exits with the explanation below rather than crashing on a missing symbol.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from typing import Dict, List, Tuple
from unittest.mock import patch

# Silence the pipeline's verbose [text] / [boxes] / [icons] prints during
# batch runs; we restore stdout for the final summary.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import diagram_analysis as da  # noqa: E402

if not hasattr(da, "_known_connections"):
    sys.exit(
        "evaluate_pipeline.py is a FROZEN Day-1 baseline harness.\n"
        "The hardcoded _known_connections() it measures was removed on Day 4;\n"
        "diagram_analysis.py is now a thin wrapper over src/pipeline.py.\n"
        "Day-1 results are saved in results/baseline_metrics.json.\n"
        "Use evaluate_phase2a.py / evaluate_phase2b.py / src.pipeline to re-run."
    )

EVAL_DIR = os.path.join(ROOT, "data", "eval")
DIAGRAMS_DIR = os.path.join(EVAL_DIR, "diagrams_15")
GROUND_TRUTH_PATH = os.path.join(EVAL_DIR, "ground_truth.json")
RESULTS_DIR = os.path.join(ROOT, "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "baseline_metrics.json")

# Expected top-level keys from export_json() — used for schema validation.
EXPECTED_JSON_KEYS = {
    "summary", "texts", "boxes", "regions", "arrows", "icons", "relationships",
}


# ---------------------------------------------------------------------------
# Fuzzy label matching
# ---------------------------------------------------------------------------

def _normalize(label: str) -> str:
    s = label.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_overlap(a: str, b: str) -> float:
    """Token-set Jaccard with a 0.6 threshold for a match (used downstream)."""
    ta = set(_normalize(a).split())
    tb = set(_normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _label_matches(detected: str, expected: str, thresh: float = 0.6) -> bool:
    if not detected or not expected:
        return False
    norm_d = _normalize(detected)
    norm_e = _normalize(expected)
    if not norm_d or not norm_e:
        return False
    # Exact normalized match
    if norm_d == norm_e:
        return True
    # Substring match (one is contained in the other) catches
    # "docker | Elastic Connector for MS SQL" vs "Elastic Connector for MS SQL"
    if norm_e in norm_d or norm_d in norm_e:
        return True
    # Token-set Jaccard fallback
    return _token_overlap(detected, expected) >= thresh


def _resolve(label: str, candidates: List[str]) -> str | None:
    """Return the best-matching candidate (or None)."""
    best, best_score = None, 0.0
    for cand in candidates:
        if _label_matches(label, cand):
            score = _token_overlap(label, cand)
            # Boost exact-substring match
            if _normalize(cand) in _normalize(label) or _normalize(label) in _normalize(cand):
                score = max(score, 0.9)
            if score > best_score:
                best, best_score = cand, score
    return best


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str, use_known_connections: bool):
    """Run the diagram_analysis pipeline once.

    Returns (texts, boxes, arrows, icons, relationships, json_data, runtime_s,
              schema_valid: bool).

    Monkey-patches:
      - diagram_analysis.INPUT_IMAGE to image_path (cosmetic; we call stages
        ourselves so this is for any internal reads only).
      - diagram_analysis._known_connections to [] if use_known_connections=False.
    """
    t0 = time.time()

    # Silence the pipeline's prints — they're useful when debugging one diagram,
    # noise when batching 15 × 2.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        img = da.load_image(image_path)
        texts = da.detect_text(img)
        regions, boxes = da.detect_boxes_and_regions(img, texts)
        icons = da.detect_icons(img, texts)
        arrows = da.detect_arrows(img, regions, boxes, texts, icons)
        texts, icons, boxes = da.associate_elements(texts, icons, boxes)
        boxes = da.enrich_entities(boxes)

        if use_known_connections:
            relationships = da.build_relationships(arrows, boxes)
        else:
            with patch.object(da, "_known_connections", return_value=[]):
                relationships = da.build_relationships(arrows, boxes)

    runtime = time.time() - t0

    # Build JSON in-memory (no disk write — we have 30 runs per session).
    json_data = {
        "diagram_analysis": {
            "summary": {
                "text_labels": len(texts),
                "boxes": len(boxes),
                "regions": len(regions),
                "arrows": len(arrows),
                "icons": len(icons),
                "relationships": len(relationships),
            },
            "texts": [{"text": t["text"]} for t in texts],
            "boxes": [{"label": b.get("label", "")} for b in boxes],
            "regions": [{"label": r.get("label", "")} for r in regions],
            "arrows": [{"x1": a["x1"], "y1": a["y1"]} for a in arrows],
            "icons": [{"label": ic.get("label", "icon")} for ic in icons],
            "relationships": [
                {"source": r["source"], "target": r["target"],
                 "detected": r.get("detected", True)}
                for r in relationships
            ],
        }
    }
    schema_valid = set(json_data["diagram_analysis"].keys()) == EXPECTED_JSON_KEYS

    return texts, boxes, arrows, icons, relationships, json_data, runtime, schema_valid


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def evaluate_one(image_path: str, ground_truth_entry: Dict,
                 use_known_connections: bool) -> Dict:
    """Run pipeline + compute precision/recall vs ground truth."""
    (texts, boxes, arrows, icons, relationships, json_data,
     runtime, schema_valid) = run_pipeline(image_path, use_known_connections)

    gt_components: List[str] = ground_truth_entry["components"]
    gt_arrows: List[Dict] = ground_truth_entry["arrows"]
    gt_icons: List[str] = ground_truth_entry.get("icons", [])

    # --- Components: a detected box matches a ground-truth component if its
    #     label fuzzy-matches. Each GT can be matched at most once.
    detected_labels = [b.get("label", "") for b in boxes]
    gt_used = set()
    comp_tp = 0
    for det in detected_labels:
        match = _resolve(det, [g for g in gt_components if g not in gt_used])
        if match is not None:
            gt_used.add(match)
            comp_tp += 1
    comp_fp = max(len(detected_labels) - comp_tp, 0)
    comp_fn = len(gt_components) - comp_tp
    comp_p, comp_r, comp_f1 = _prf(comp_tp, comp_fp, comp_fn)

    # --- Relationships: a detected pair matches a GT pair if both endpoint
    #     labels fuzzy-match (direction respected; mirror-pair counted as miss
    #     since direction matters in architecture diagrams).
    det_pairs = [(r["source"], r["target"]) for r in relationships]
    gt_pairs = [(a["source"], a["target"]) for a in gt_arrows]
    used_gt = set()
    rel_tp = 0
    for (dsrc, dtgt) in det_pairs:
        for j, (gsrc, gtgt) in enumerate(gt_pairs):
            if j in used_gt:
                continue
            if _label_matches(dsrc, gsrc) and _label_matches(dtgt, gtgt):
                used_gt.add(j)
                rel_tp += 1
                break
    rel_fp = max(len(det_pairs) - rel_tp, 0)
    rel_fn = len(gt_pairs) - rel_tp
    rel_p, rel_r, rel_f1 = _prf(rel_tp, rel_fp, rel_fn)

    # --- Icons: count-based for Day 1 (positional matching is Day 5+).
    det_icon_n = len(icons)
    gt_icon_n = len(gt_icons)
    icon_tp = min(det_icon_n, gt_icon_n)
    icon_fp = max(det_icon_n - gt_icon_n, 0)
    icon_fn = max(gt_icon_n - det_icon_n, 0)
    icon_p, icon_r, icon_f1 = _prf(icon_tp, icon_fp, icon_fn)

    return {
        "components": {
            "detected": len(detected_labels),
            "ground_truth": len(gt_components),
            "tp": comp_tp, "fp": comp_fp, "fn": comp_fn,
            "precision": round(comp_p, 3),
            "recall": round(comp_r, 3),
            "f1": round(comp_f1, 3),
            "detected_labels": detected_labels,
        },
        "relationships": {
            "detected": len(det_pairs),
            "ground_truth": len(gt_pairs),
            "tp": rel_tp, "fp": rel_fp, "fn": rel_fn,
            "precision": round(rel_p, 3),
            "recall": round(rel_r, 3),
            "f1": round(rel_f1, 3),
            "detected_pairs": det_pairs,
            "from_known_connections": sum(
                1 for r in relationships if not r.get("detected", True)
            ),
        },
        "icons": {
            "detected": det_icon_n,
            "ground_truth": gt_icon_n,
            "tp": icon_tp, "fp": icon_fp, "fn": icon_fn,
            "precision": round(icon_p, 3),
            "recall": round(icon_r, 3),
            "f1": round(icon_f1, 3),
        },
        "runtime_seconds": round(runtime, 2),
        "schema_valid_json": schema_valid,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _aggregate(per_diagram: Dict[str, Dict], key: str) -> Dict[str, float]:
    """Macro-average precision/recall/F1 across diagrams for one metric group."""
    ps, rs, fs = [], [], []
    for entry in per_diagram.values():
        m = entry[key]
        ps.append(m["precision"])
        rs.append(m["recall"])
        fs.append(m["f1"])
    n = max(len(ps), 1)
    return {
        "macro_precision": round(sum(ps) / n, 3),
        "macro_recall": round(sum(rs) / n, 3),
        "macro_f1": round(sum(fs) / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", help="Path to a single image (skip batch).")
    ap.add_argument("--no-known", action="store_true",
                    help="Disable _known_connections for single-image mode.")
    args = ap.parse_args()

    if args.single:
        gt = {"components": [], "arrows": [], "icons": []}
        result = evaluate_one(args.single, gt,
                              use_known_connections=not args.no_known)
        print(json.dumps(result, indent=2))
        return

    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth: Dict[str, Dict] = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "metadata": {
            "date": "2026-05-25",
            "day": 1,
            "project": "DiagraMine",
            "benchmark_size": len(ground_truth),
            "benchmark_dir": "data/eval/diagrams_15/",
            "fuzzy_match_threshold": 0.6,
        },
        "per_diagram_with_known": {},
        "per_diagram_no_known": {},
        "aggregate_with_known": {},
        "aggregate_no_known": {},
    }

    for diagram_name, gt_entry in ground_truth.items():
        img_path = os.path.join(DIAGRAMS_DIR, diagram_name)
        if not os.path.exists(img_path):
            print(f"[skip] {diagram_name}: file not found at {img_path}")
            continue

        print(f"[run] {diagram_name}  (WITH _known_connections)")
        r_with = evaluate_one(img_path, gt_entry, use_known_connections=True)
        print(f"      components P={r_with['components']['precision']:.2f}/"
              f"R={r_with['components']['recall']:.2f}  "
              f"rels P={r_with['relationships']['precision']:.2f}/"
              f"R={r_with['relationships']['recall']:.2f}  "
              f"({r_with['runtime_seconds']:.1f}s)")

        print(f"[run] {diagram_name}  (NO _known_connections)")
        r_no = evaluate_one(img_path, gt_entry, use_known_connections=False)
        print(f"      components P={r_no['components']['precision']:.2f}/"
              f"R={r_no['components']['recall']:.2f}  "
              f"rels P={r_no['relationships']['precision']:.2f}/"
              f"R={r_no['relationships']['recall']:.2f}  "
              f"({r_no['runtime_seconds']:.1f}s)")

        results["per_diagram_with_known"][diagram_name] = r_with
        results["per_diagram_no_known"][diagram_name] = r_no

    results["aggregate_with_known"] = {
        "components": _aggregate(results["per_diagram_with_known"], "components"),
        "relationships": _aggregate(results["per_diagram_with_known"], "relationships"),
        "icons": _aggregate(results["per_diagram_with_known"], "icons"),
        "schema_valid_json_rate": round(sum(
            1 for r in results["per_diagram_with_known"].values()
            if r["schema_valid_json"]
        ) / max(len(results["per_diagram_with_known"]), 1), 3),
        "avg_runtime_seconds": round(sum(
            r["runtime_seconds"] for r in results["per_diagram_with_known"].values()
        ) / max(len(results["per_diagram_with_known"]), 1), 2),
    }
    results["aggregate_no_known"] = {
        "components": _aggregate(results["per_diagram_no_known"], "components"),
        "relationships": _aggregate(results["per_diagram_no_known"], "relationships"),
        "icons": _aggregate(results["per_diagram_no_known"], "icons"),
        "schema_valid_json_rate": round(sum(
            1 for r in results["per_diagram_no_known"].values()
            if r["schema_valid_json"]
        ) / max(len(results["per_diagram_no_known"]), 1), 3),
        "avg_runtime_seconds": round(sum(
            r["runtime_seconds"] for r in results["per_diagram_no_known"].values()
        ) / max(len(results["per_diagram_no_known"]), 1), 2),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("  AGGREGATE  (WITH _known_connections)")
    print("=" * 70)
    a = results["aggregate_with_known"]
    print(f"  components   macro-F1 = {a['components']['macro_f1']:.3f}  "
          f"P={a['components']['macro_precision']:.3f}  "
          f"R={a['components']['macro_recall']:.3f}")
    print(f"  relationships macro-F1 = {a['relationships']['macro_f1']:.3f}  "
          f"P={a['relationships']['macro_precision']:.3f}  "
          f"R={a['relationships']['macro_recall']:.3f}")
    print(f"  icons        macro-F1 = {a['icons']['macro_f1']:.3f}")
    print(f"  schema-valid JSON rate = {a['schema_valid_json_rate']:.3f}")
    print(f"  avg runtime / diagram   = {a['avg_runtime_seconds']:.2f}s")

    print("\n" + "=" * 70)
    print("  AGGREGATE  (NO _known_connections — honest CV-only baseline)")
    print("=" * 70)
    a = results["aggregate_no_known"]
    print(f"  components   macro-F1 = {a['components']['macro_f1']:.3f}  "
          f"P={a['components']['macro_precision']:.3f}  "
          f"R={a['components']['macro_recall']:.3f}")
    print(f"  relationships macro-F1 = {a['relationships']['macro_f1']:.3f}  "
          f"P={a['relationships']['macro_precision']:.3f}  "
          f"R={a['relationships']['macro_recall']:.3f}")
    print(f"  icons        macro-F1 = {a['icons']['macro_f1']:.3f}")
    print(f"  schema-valid JSON rate = {a['schema_valid_json_rate']:.3f}")
    print(f"  avg runtime / diagram   = {a['avg_runtime_seconds']:.2f}s")

    print(f"\n[ok] Results -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
