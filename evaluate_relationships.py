"""Day-5 Phase-4 relationship re-evaluation: before vs after the targeted fix.

The error analysis (results/error_analysis.json) named ARROW-MAPPING the
dominant failure mode (63% of failures), and split the missed connections into
~28 not-detected (a Day-3 detection-recall ceiling, out of the mapper's reach)
and ~18 mis-mapped (short segments whose endpoints land in whitespace beyond
the snap radius — mapper-fixable). It also surfaced a region/component mislabel:
genuine components whose detected area drifts above `region_area` get filed as
regions and pulled out of the relationship candidate pool.

This script measures the relationship-level P/R/F1 (final pipeline output vs
ground-truth connections) under three configs, as an ablation, on the 14
generated diagrams. Detection (EasyOCR / Canny / Hough) is run ONCE per
diagram and cached; only the graph-builder settings change between configs:

  A  baseline champion        region_area=25000, max_dist=120, ray off
  B  + region-flag fix        region_area=60000, max_dist=120, ray off
  C  + arrow-mapping fix      region_area=60000, max_dist=160, ray ON  (champion)

We also confirm every emitted relationship is schema-valid (validates against
the Pydantic `Relationship` model) — the headline reliability metric stays 1.0.

Outputs:
  results/relationship_fix.csv     — per-config aggregate + per-diagram
  results/relationship_fix.json    — full detail incl. champion choice
  results/samples/day05/<diagram>_{before,after}_graph.png
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.graph import builder  # noqa: E402
from src.schemas import Relationship  # noqa: E402
from src.text_detection import easyocr_detector  # noqa: E402
from src.box_detection import canny_contours_detector as boxdet  # noqa: E402
from src.arrow_detection import hough_lines_detector  # noqa: E402

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_LABELS = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth.json"), encoding="utf-8"))
GT_BOXES = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth_boxes.json"), encoding="utf-8"))
RESULTS_DIR = os.path.join(ROOT, "results")
SAMPLE_DIR = os.path.join(RESULTS_DIR, "samples", "day05")
os.makedirs(SAMPLE_DIR, exist_ok=True)

CONFIGS = [
    ("A_baseline", dict(region_area=25000, max_dist=120.0, ray_intersection=False)),
    ("B_region_fix", dict(region_area=60000, max_dist=120.0, ray_intersection=False)),
    ("C_mapper_fix", dict(region_area=60000, max_dist=160.0, ray_intersection=True)),
]
SAMPLE_DIAGRAMS = ["diagram_02.png", "diagram_07.png", "diagram_10.png"]


def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _match(a: str, b: str, thresh: float = 0.6) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb or nb in na or na in nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    return len(ta & tb) / len(ta | tb) >= thresh if (ta and tb) else False


def score_rels(rels: List[dict], gt_arrows: List[dict]) -> Tuple[int, int, int]:
    det = [(r["source"], r["target"]) for r in rels]
    gt = [(a["source"], a["target"]) for a in gt_arrows]
    matched, used_d = set(), set()
    for di, (ds, dt) in enumerate(det):
        for gi, (gs, gt_) in enumerate(gt):
            if gi in matched:
                continue
            if ((_match(ds, gs) and _match(dt, gt_)) or
                    (_match(ds, gt_) and _match(dt, gs))):
                matched.add(gi)
                used_d.add(di)
                break
    tp = len(matched)
    fp = len(det) - len(used_d)
    fn = len(gt) - tp
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f1, 3)


def cache_detection(fname: str) -> dict:
    path = os.path.join(DIAGRAMS_DIR, fname)
    texts = easyocr_detector.detect(path)["texts"]
    boxes = boxdet.detect(path)["boxes"]   # champion (tuned) params
    arrows = hough_lines_detector.detect(path)["arrows"]
    return {"texts": texts, "boxes": boxes, "arrows": arrows}


def build_for_config(cache: dict, cfg: dict) -> List[dict]:
    # Re-flag is_region from area under this config's region_area, then label
    # and build relationships with the config's mapper settings.
    boxes = [dict(b) for b in cache["boxes"]]
    for b in boxes:
        b["is_region"] = b["area"] > cfg["region_area"]
    boxes = builder.label_boxes(boxes, cache["texts"])
    rels = builder.build_relationships(
        cache["arrows"], boxes,
        outside_box_gate=True,
        max_dist=cfg["max_dist"],
        ray_intersection=cfg["ray_intersection"],
    )
    return rels


def main() -> None:
    diagrams = [f for f in sorted(GT_LABELS.keys()) if f in GT_BOXES]
    caches = {f: cache_detection(f) for f in diagrams}

    # Instrument the ray-intersection path so we can report how often the
    # "expand radius + intersect" fallback actually triggers (and succeeds).
    ray_stats = {"taken": 0, "hit": 0}
    _orig_ray = builder._box_along_ray

    def _counting_ray(*a, **k):
        ray_stats["taken"] += 1
        r = _orig_ray(*a, **k)
        if r is not None:
            ray_stats["hit"] += 1
        return r
    builder._box_along_ray = _counting_ray

    results: Dict[str, dict] = {}
    per_diagram_rows: List[dict] = []
    schema_valid_runs = 0
    schema_total_runs = 0
    ray_stats_by_config: Dict[str, dict] = {}

    for cfg_name, cfg in CONFIGS:
        ray_stats["taken"] = 0
        ray_stats["hit"] = 0
        TP = FP = FN = 0
        per_diag = {}
        for f in diagrams:
            rels = build_for_config(caches[f], cfg)
            # schema validation of every emitted relationship
            schema_total_runs += 1
            try:
                for r in rels:
                    Relationship.model_validate(r)
                schema_valid_runs += 1
            except Exception:
                pass
            tp, fp, fn = score_rels(rels, GT_LABELS[f]["arrows"])
            TP += tp; FP += fp; FN += fn
            p, r, f1 = _prf(tp, fp, fn)
            per_diag[f] = {"detected": len(rels), "tp": tp, "fp": fp, "fn": fn,
                           "precision": p, "recall": r, "f1": f1}
            per_diagram_rows.append({"config": cfg_name, "diagram": f,
                                     "detected": len(rels), "tp": tp, "fp": fp,
                                     "fn": fn, "precision": p, "recall": r, "f1": f1})
        p, r, f1 = _prf(TP, FP, FN)
        results[cfg_name] = {"precision": p, "recall": r, "f1": f1,
                             "tp": TP, "fp": FP, "fn": FN, "per_diagram": per_diag}
        ray_stats_by_config[cfg_name] = dict(ray_stats)
        print(f"[{cfg_name:14s}] P={p:.3f} R={r:.3f} F1={f1:.3f}  "
              f"(tp={TP} fp={FP} fn={FN})  "
              f"ray_path_taken={ray_stats['taken']} ray_hits={ray_stats['hit']}")

    champion = max(results, key=lambda k: results[k]["f1"])
    base_f1 = results["A_baseline"]["f1"]
    champ_f1 = results[champion]["f1"]
    schema_rate = round(schema_valid_runs / schema_total_runs, 3) if schema_total_runs else 0.0
    print(f"\nChampion: {champion}  F1 {base_f1} -> {champ_f1} ({champ_f1 - base_f1:+.3f})")
    print(f"Schema-valid relationship rate: {schema_valid_runs}/{schema_total_runs} = {schema_rate}")

    # ── samples: before (A) vs after (champion) relationship graphs ──
    for f in SAMPLE_DIAGRAMS:
        for tag, cfg_name in (("before", "A_baseline"), ("after", champion)):
            cfg = dict(CONFIGS)[cfg_name]
            rels = build_for_config(caches[f], cfg)
            builder.draw_graph(rels, os.path.join(SAMPLE_DIR, f"{f.replace('.png','')}_{tag}_graph.png"))

    # ── write CSV / JSON ──
    with open(os.path.join(RESULTS_DIR, "relationship_fix.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "precision", "recall", "f1", "tp", "fp", "fn"])
        for cfg_name, _ in CONFIGS:
            r = results[cfg_name]
            w.writerow([cfg_name, r["precision"], r["recall"], r["f1"],
                        r["tp"], r["fp"], r["fn"]])
        w.writerow([])
        w.writerow(["--- per diagram ---"])
        w.writerow(["config", "diagram", "detected", "tp", "fp", "fn",
                    "precision", "recall", "f1"])
        for row in per_diagram_rows:
            w.writerow([row["config"], row["diagram"], row["detected"], row["tp"],
                        row["fp"], row["fn"], row["precision"], row["recall"], row["f1"]])

    with open(os.path.join(RESULTS_DIR, "relationship_fix.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "metadata": {"date": "2026-05-29", "day": 5, "project": "DiagraMine",
                         "metric": "relationship unordered-pair micro-F1 over 14 diagrams",
                         "configs": {n: c for n, c in CONFIGS}},
            "results": {n: {k: v for k, v in results[n].items() if k != "per_diagram"}
                        for n, _ in CONFIGS},
            "per_diagram": {n: results[n]["per_diagram"] for n, _ in CONFIGS},
            "champion": champion,
            "delta_f1_vs_baseline": round(champ_f1 - base_f1, 3),
            "schema_valid_relationship_rate": schema_rate,
            "ray_path_stats": ray_stats_by_config,
        }, fh, indent=2)

    print(f"\n[ok] -> results/relationship_fix.csv / .json ; samples -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
