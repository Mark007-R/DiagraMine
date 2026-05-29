"""Day-5 end-to-end integration check.

Confirms the Day-5 champion config (Optuna-tuned box detector, region_area
fix, arrow-mapping fix wired on) still returns schema-valid JSON on all 15
benchmark diagrams and that every relationship is data-driven. Writes to a
DAY-5 file so the Day-4 record is left untouched.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.pipeline import extract  # noqa: E402
from src.schemas import ExtractionResult, PipelineConfig  # noqa: E402

DIAGRAMS_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT = json.load(open(os.path.join(ROOT, "data", "eval", "ground_truth.json"), encoding="utf-8"))
OUT = os.path.join(ROOT, "results", "day05_integration.json")


def main() -> None:
    cfg = PipelineConfig()
    rows = []
    valid = 0
    total_rels = 0
    all_data_driven = True
    for fname in sorted(GT.keys()):
        res = extract(os.path.join(DIAGRAMS_DIR, fname), cfg)
        ok = True
        try:
            ExtractionResult.model_validate(json.loads(json.dumps(res.model_dump())))
        except Exception:
            ok = False
        valid += int(ok)
        total_rels += len(res.relationships)
        if any(not r.detected for r in res.relationships):
            all_data_driven = False
        rows.append({"diagram": fname, "schema_valid": ok,
                     "boxes": len(res.boxes), "regions": len(res.regions),
                     "relationships": len(res.relationships)})
        print(f"{fname:30s} valid={ok} boxes={len(res.boxes)} "
              f"regions={len(res.regions)} rels={len(res.relationships)}")

    summary = {
        "date": "2026-05-29", "day": 5, "n_diagrams": len(rows),
        "schema_valid_count": valid, "schema_valid_rate": round(valid / len(rows), 3),
        "all_relationships_data_driven": all_data_driven,
        "total_relationships": total_rels,
        "champion_config": cfg.model_dump(),
        "per_diagram": rows,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in
                      ["n_diagrams", "schema_valid_count", "schema_valid_rate",
                       "all_relationships_data_driven", "total_relationships"]}, indent=2))
    print(f"[day05] -> {OUT}")


if __name__ == "__main__":
    main()
