"""Build phase2_leaderboard.csv combining Phase 2a (text+box) and Phase 2b
(arrow+icon) winners into one master champion stack.

Run from repo root:  python results/_build_leaderboard.py
"""
from __future__ import annotations

import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")


def load_phase2a():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "phase2a_text_box.csv"),
                                     encoding="utf-8")))
    return rows


def load_phase2b():
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "phase2b_arrow_icon.csv"),
                                     encoding="utf-8")))
    return rows


def main() -> None:
    out_path = os.path.join(RESULTS, "phase2_leaderboard.csv")

    # Champion picks (highest F1 per stage)
    stages: dict = {"text": [], "box": [], "arrow": [], "icon": []}
    for r in load_phase2a():
        if r["stage"] in stages:
            stages[r["stage"]].append(r)
    for r in load_phase2b():
        if r["stage"] in stages:
            stages[r["stage"]].append(r)

    def f1(r):
        try:
            return float(r["f1"])
        except (ValueError, TypeError):
            return 0.0

    champions = {}
    for stage, rows in stages.items():
        if not rows:
            continue
        # Filter SKIPPED entries
        rows = [r for r in rows if r["precision"] != "" and r["f1"] != ""]
        rows.sort(key=f1, reverse=True)
        champions[stage] = rows[0] if rows else None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "champion", "precision", "recall", "f1",
                    "secondary", "avg_runtime_seconds", "notes"])
        for stage in ("text", "box", "arrow", "icon"):
            c = champions.get(stage)
            if c is None:
                w.writerow([stage, "(no champion)", "", "", "", "", "", ""])
                continue
            w.writerow([stage, c["detector"], c["precision"], c["recall"],
                        c["f1"],
                        c.get("per_char_accuracy") or c.get("fp_rate")
                        or c.get("secondary") or "",
                        c["avg_runtime_seconds"], c["notes"]])

        # Add a runners-up block for transparency
        w.writerow([])
        w.writerow(["# Full leaderboard below for transparency"])
        w.writerow(["stage", "rank", "detector", "precision", "recall", "f1",
                    "avg_runtime_seconds", "notes"])
        for stage in ("text", "box", "arrow", "icon"):
            rows = [r for r in stages[stage] if r["precision"] != ""]
            rows.sort(key=f1, reverse=True)
            for i, r in enumerate(rows, 1):
                w.writerow([stage, i, r["detector"], r["precision"],
                            r["recall"], r["f1"],
                            r["avg_runtime_seconds"], r["notes"]])

    print("Wrote", out_path)
    print("\nChampion stack:")
    for stage in ("text", "box", "arrow", "icon"):
        c = champions.get(stage)
        if c:
            print(f"  {stage:<6} -> {c['detector']:<20} "
                  f"F1={c['f1']} runtime={c['avg_runtime_seconds']}s")


if __name__ == "__main__":
    main()
