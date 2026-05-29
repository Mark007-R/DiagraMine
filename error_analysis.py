"""Day-5 Phase-4 diverse-diagram error analysis.

Runs the champion pipeline (`src.pipeline.extract` with default config) on the
14 generated benchmark diagrams and categorises every failure into one of four
buckets, so we can name the *dominant* failure mode and target the fix:

  • text     — a ground-truth component label the OCR stage missed (FN).
  • box      — a GT box missed (FN, recall loss), a detected box that MERGED
               two GT boxes into one, or an extra detected box (FP).
  • arrow    — a GT connection not present in the final relationships (FN) or a
               spurious relationship (FP). This is the end-to-end arrow→
               relationship "mapping" quality the Day-5 fix targets.
  • icon     — a GT icon missed (only search_interview_test has icon GT, which
               has no machine-readable boxes, so it is reported separately).

For the arrow bucket we additionally split each MISSED GT connection into:
  • not_detected   — no detected arrow segment runs between the two GT boxes
                     (a detection-recall loss; cannot be fixed in the mapper).
  • mis_mapped     — a segment DOES run between them but the relationship was
                     dropped/misrouted (snap radius too small, gate too strict,
                     wrong nearest box). THIS is what "expand search radius +
                     intersection-with-box logic" addresses.

Outputs:
  results/error_analysis.json   — per-diagram + aggregate counts
  results/error_analysis.csv    — flat per-diagram failure tally
  results/error_analysis_by_category.png  — bar chart of failures by category
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.pipeline import extract  # noqa: E402
from src.schemas import PipelineConfig  # noqa: E402

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_LABELS = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth.json"), encoding="utf-8"))
GT_BOXES = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth_boxes.json"), encoding="utf-8"))
RESULTS_DIR = os.path.join(ROOT, "results")


# ── fuzzy label match (mirrors evaluate_pipeline) ────────────────────────────

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


def _iou(a: dict, b: dict) -> float:
    ix1 = max(a["x"], b["x"]); iy1 = max(a["y"], b["y"])
    ix2 = min(a["x"] + a["w"], b["x"] + b["w"]); iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    u = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / u if u > 0 else 0.0


def _contains_frac(inner: dict, outer: dict) -> float:
    """Fraction of `inner`'s area covered by `outer` (for merge detection)."""
    ix1 = max(inner["x"], outer["x"]); iy1 = max(inner["y"], outer["y"])
    ix2 = min(inner["x"] + inner["w"], outer["x"] + outer["w"])
    iy2 = min(inner["y"] + inner["h"], outer["y"] + outer["h"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a = inner["w"] * inner["h"]
    return inter / a if a > 0 else 0.0


def _seg_runs_between(arrow: dict, b1: dict, b2: dict, margin: int = 90) -> bool:
    """Does a detected arrow segment plausibly run between GT boxes b1 and b2?
    True if one endpoint is near b1 and the other near b2 (point within an
    expanded rect), order-agnostic."""
    def near(pt, b):
        x, y = pt
        return (b["x"] - margin <= x <= b["x"] + b["w"] + margin and
                b["y"] - margin <= y <= b["y"] + b["h"] + margin)
    p1 = (arrow["x1"], arrow["y1"]); p2 = (arrow["x2"], arrow["y2"])
    return (near(p1, b1) and near(p2, b2)) or (near(p1, b2) and near(p2, b1))


# ── per-diagram analysis ─────────────────────────────────────────────────────

def analyse(fname: str, cfg: PipelineConfig) -> Dict:
    path = os.path.join(DIAGRAMS_DIR, fname)
    res = extract(path, cfg)
    gt = GT_LABELS[fname]
    gt_components: List[str] = gt["components"]
    gt_arrows = gt["arrows"]
    gt_boxes = GT_BOXES.get(fname, {}).get("boxes", [])

    # ---- TEXT failures: GT component label missed by OCR ----
    det_texts = [t.text for t in res.texts]
    text_fn = [g for g in gt_components if not any(_match(d, g) for d in det_texts)]

    # ---- BOX failures: missed / merged / extra (IoU@0.5) ----
    # Two lenses:
    #  (a) raw detection — ALL detected boxes (components + regions). This is
    #      the box-detector quality, consistent with the Phase-2a / Day-5
    #      tuning metric.
    #  (b) component-only — boxes the pipeline emits under "boxes" (non-region).
    #      The gap between (a) and (b) is the region/component mislabel: genuine
    #      components whose detected area drifts above region_area get filed as
    #      regions and disappear from the component output + relationship pool.
    def _match_gt(det_boxes: List[dict]) -> set:
        used_d, matched = set(), set()
        for gi, g in enumerate(gt_boxes):
            best_di, best_v = None, 0.5
            for di, d in enumerate(det_boxes):
                if di in used_d:
                    continue
                v = _iou(d, g)
                if v >= best_v:
                    best_v, best_di = v, di
            if best_di is not None:
                used_d.add(best_di)
                matched.add(gi)
        return matched

    raw_boxes = ([{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in res.boxes] +
                 [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in res.regions])
    comp_boxes = [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in res.boxes]
    matched_raw = _match_gt(raw_boxes)
    matched_comp = _match_gt(comp_boxes)
    box_fn = len(gt_boxes) - len(matched_raw)           # true missed boxes (raw)
    box_fp = max(len(raw_boxes) - len(matched_raw), 0)  # extra boxes (raw)
    region_misflag = len(matched_raw) - len(matched_comp)  # comps filed as regions
    # merge: a detected box that covers >=2 GT boxes at >=60% of each GT's area
    box_merges = 0
    for d in raw_boxes:
        covered = sum(1 for g in gt_boxes if _contains_frac(g, d) >= 0.6)
        if covered >= 2:
            box_merges += 1

    # ---- ARROW-MAPPING failures: final relationships vs GT connections ----
    det_pairs = [(_norm(r.source), _norm(r.target)) for r in res.relationships]
    gt_pairs = [(g["source"], g["target"]) for g in gt_arrows]
    matched_gt = set()
    used_dp = set()
    for di, (ds, dt) in enumerate(det_pairs):
        for gi, (gs, gt_) in enumerate(gt_pairs):
            if gi in matched_gt:
                continue
            if ((_match(ds, gs) and _match(dt, gt_)) or
                    (_match(ds, gt_) and _match(dt, gs))):
                matched_gt.add(gi)
                used_dp.add(di)
                break
    arrow_tp = len(matched_gt)
    arrow_fn = len(gt_pairs) - arrow_tp
    arrow_fp = max(len(det_pairs) - len(used_dp), 0)

    # split missed connections: not_detected vs mis_mapped
    label_to_box = {b["label"]: b for b in gt_boxes}
    raw_arrows = [{"x1": a.x1, "y1": a.y1, "x2": a.x2, "y2": a.y2} for a in res.arrows]
    not_detected = 0
    mis_mapped = 0
    for gi, (gs, gt_) in enumerate(gt_pairs):
        if gi in matched_gt:
            continue
        b1, b2 = label_to_box.get(gs), label_to_box.get(gt_)
        if not b1 or not b2:
            not_detected += 1
            continue
        if any(_seg_runs_between(a, b1, b2) for a in raw_arrows):
            mis_mapped += 1
        else:
            not_detected += 1

    # ---- ICON failures ----
    gt_icons = gt.get("icons", [])
    det_icon_labels = [ic.label for ic in res.icons]
    icon_fn = sum(1 for g in gt_icons if not any(_match(d, g) for d in det_icon_labels))

    return {
        "diagram": fname,
        "text": {"gt": len(gt_components), "fn": len(text_fn), "missed": text_fn},
        "box": {"gt": len(gt_boxes), "detected_raw": len(raw_boxes),
                "detected_components": len(comp_boxes),
                "fn": box_fn, "fp": box_fp, "merges": box_merges,
                "region_misflag": region_misflag},
        "arrow": {"gt": len(gt_pairs), "detected": len(det_pairs),
                  "tp": arrow_tp, "fn": arrow_fn, "fp": arrow_fp,
                  "missed_not_detected": not_detected,
                  "missed_mis_mapped": mis_mapped},
        "icon": {"gt": len(gt_icons), "fn": icon_fn},
        "failures": {  # one number per category = FN + FP (+ merges for box)
            "text": len(text_fn),
            "box": box_fn + box_fp,
            "arrow": arrow_fn + arrow_fp,
            "icon": icon_fn,
        },
    }


def main() -> None:
    cfg = PipelineConfig()  # champion defaults (now tuned box detector)
    diagrams = [f for f in sorted(GT_LABELS.keys()) if f in GT_BOXES]  # 14 scorable

    per_diagram = []
    for f in diagrams:
        r = analyse(f, cfg)
        per_diagram.append(r)
        print(f"{f:16s} text_fn={r['text']['fn']} "
              f"box(fn={r['box']['fn']},fp={r['box']['fp']},"
              f"region_misflag={r['box']['region_misflag']}) "
              f"arrow(fn={r['arrow']['fn']},fp={r['arrow']['fp']}; "
              f"not_det={r['arrow']['missed_not_detected']},"
              f"mis_map={r['arrow']['missed_mis_mapped']}) icon_fn={r['icon']['fn']}")

    agg = {"text": 0, "box": 0, "arrow": 0, "icon": 0}
    arrow_split = {"not_detected": 0, "mis_mapped": 0, "fp": 0, "fn": 0, "tp": 0, "gt": 0}
    box_split = {"fn": 0, "fp": 0, "merges": 0, "region_misflag": 0}
    for r in per_diagram:
        for k in agg:
            agg[k] += r["failures"][k]
        arrow_split["not_detected"] += r["arrow"]["missed_not_detected"]
        arrow_split["mis_mapped"] += r["arrow"]["missed_mis_mapped"]
        arrow_split["fp"] += r["arrow"]["fp"]
        arrow_split["fn"] += r["arrow"]["fn"]
        arrow_split["tp"] += r["arrow"]["tp"]
        arrow_split["gt"] += r["arrow"]["gt"]
        box_split["fn"] += r["box"]["fn"]
        box_split["fp"] += r["box"]["fp"]
        box_split["merges"] += r["box"]["merges"]
        box_split["region_misflag"] += r["box"]["region_misflag"]

    total = sum(agg.values()) or 1
    dominant = max(agg, key=agg.get)
    shares = {k: round(v / total, 3) for k, v in agg.items()}

    print("\n=== Aggregate failures by category (FN+FP over 14 diagrams) ===")
    for k in ["text", "box", "arrow", "icon"]:
        print(f"  {k:6s} {agg[k]:3d}  ({shares[k]*100:.0f}%)")
    print(f"  DOMINANT: {dominant}")
    print(f"  box note: raw detection fn={box_split['fn']} fp={box_split['fp']}; "
          f"region_misflag={box_split['region_misflag']} "
          f"(genuine components filed as regions — schema/candidate bug, not a miss)")
    print(f"  arrow missed split: not_detected={arrow_split['not_detected']} "
          f"mis_mapped={arrow_split['mis_mapped']} (mis-mapped are mapper-fixable)")

    out = {
        "metadata": {"date": "2026-05-29", "day": 5, "project": "DiagraMine",
                     "diagrams_scored": len(diagrams),
                     "champion_config": cfg.model_dump()},
        "aggregate_failures": agg,
        "failure_shares": shares,
        "dominant_failure_mode": dominant,
        "arrow_breakdown": arrow_split,
        "box_breakdown": box_split,
        "per_diagram": per_diagram,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "error_analysis.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "error_analysis.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["diagram", "text_fn", "box_fn", "box_fp", "box_region_misflag",
                    "arrow_fn", "arrow_fp", "arrow_not_detected",
                    "arrow_mis_mapped", "icon_fn"])
        for r in per_diagram:
            w.writerow([r["diagram"], r["text"]["fn"], r["box"]["fn"],
                        r["box"]["fp"], r["box"]["region_misflag"], r["arrow"]["fn"],
                        r["arrow"]["fp"], r["arrow"]["missed_not_detected"],
                        r["arrow"]["missed_mis_mapped"], r["icon"]["fn"]])
        w.writerow([])
        w.writerow(["TOTAL", agg["text"], box_split["fn"], box_split["fp"],
                    box_split["region_misflag"], arrow_split["fn"], arrow_split["fp"],
                    arrow_split["not_detected"], arrow_split["mis_mapped"], agg["icon"]])

    # bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ["text", "box", "arrow", "icon"]
    vals = [agg[c] for c in cats]
    colors = ["#7fb8e6", "#7fd6a0", "#e67f7f", "#d6b87f"]
    bars = ax.bar(cats, vals, color=colors, edgecolor="#333")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v),
                ha="center", fontweight="bold")
    ax.set_ylabel("Failures (FN + FP) across 14 diagrams")
    ax.set_title(f"DiagraMine failure modes — dominant: {dominant.upper()}",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "error_analysis_by_category.png"),
                dpi=150, facecolor="white")
    plt.close(fig)

    print(f"\n[ok] -> results/error_analysis.json / .csv / _by_category.png")


if __name__ == "__main__":
    main()
