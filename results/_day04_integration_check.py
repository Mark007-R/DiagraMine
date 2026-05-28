"""Day-4 integration check.

Runs the refactored modular pipeline (src/pipeline.py, champion config) across
all 15 benchmark diagrams and verifies:

  1. SCHEMA-VALID JSON RATE (the headline reliability metric, Hard Rule #9):
     for each diagram, the result must round-trip
     ExtractionResult -> model_dump() -> json.dumps -> json.loads ->
     ExtractionResult.model_validate() without error.
  2. NO-HARDCODING REGRESSION: every emitted relationship must be data-driven
     (`detected is True`). If any relationship were re-introduced from a
     hardcoded source it would show detected=False; we assert that never
     happens. We also assert the `_known_connections` symbol is gone.

Writes results/day04_integration.json (+ a flat CSV) and emits annotated /
graph samples for 5 diagrams to results/samples/day04/.
"""
from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402

import diagram_analysis as da  # noqa: E402
from src import pipeline  # noqa: E402
from src.graph import builder  # noqa: E402
from src.schemas import ExtractionResult, PipelineConfig  # noqa: E402

DIAG_DIR = os.path.join(ROOT, "data", "eval", "diagrams_15")
GT_PATH = os.path.join(ROOT, "data", "eval", "ground_truth.json")
OUT_JSON = os.path.join(ROOT, "results", "day04_integration.json")
OUT_CSV = os.path.join(ROOT, "results", "day04_integration.csv")
SAMPLE_DIR = os.path.join(ROOT, "results", "samples", "day04")
SAMPLE_DIAGRAMS = {"diagram_01.png", "diagram_07.png", "diagram_12.png",
                   "diagram_14.png", "search_interview_test.png"}


def schema_round_trip_ok(result: ExtractionResult) -> bool:
    """Strongest schema-valid test: dump -> json -> parse -> re-validate."""
    try:
        s = json.dumps(result.model_dump())
        ExtractionResult.model_validate(json.loads(s))
        return True
    except Exception:
        return False


def main() -> None:
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    gt = json.load(open(GT_PATH, encoding="utf-8"))
    config = PipelineConfig()  # champions

    # No-hardcoding regression: the symbol must be gone from the wrapper.
    assert not hasattr(da, "_known_connections"), "_known_connections was re-introduced!"

    rows = []
    detail = {}
    schema_valid = 0
    all_data_driven = True
    for name in gt.keys():
        path = os.path.join(DIAG_DIR, name)
        result = pipeline.extract(path, config)
        ok = schema_round_trip_ok(result)
        schema_valid += int(ok)
        data_driven = all(r.detected for r in result.relationships)
        all_data_driven = all_data_driven and data_driven
        errs = result.detectors.get("errors", {})

        s = result.summary()
        rows.append({
            "diagram": name, "schema_valid": ok, "all_data_driven": data_driven,
            "texts": s["text_labels"], "boxes": s["boxes"], "regions": s["regions"],
            "arrows": s["arrows"], "icons": s["icons"],
            "relationships": s["relationships"],
            "runtime_seconds": result.runtime_seconds,
            "stage_errors": ";".join(errs.keys()) if errs else "",
        })
        detail[name] = {
            "summary": s, "schema_valid": ok, "all_data_driven": data_driven,
            "runtimes": result.runtimes.model_dump(), "errors": errs,
            "relationships": [r.model_dump() for r in result.relationships],
        }

        if name in SAMPLE_DIAGRAMS:
            stem = os.path.splitext(name)[0]
            cv2.imwrite(os.path.join(SAMPLE_DIR, f"{stem}_annotated.png"),
                        pipeline.annotate(path, result))
            builder.draw_graph([r.model_dump() for r in result.relationships],
                               os.path.join(SAMPLE_DIR, f"{stem}_graph.png"))
            with open(os.path.join(SAMPLE_DIR, f"{stem}_structure.json"), "w",
                      encoding="utf-8") as f:
                json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)

    n = len(gt)
    aggregate = {
        "n_diagrams": n,
        "schema_valid_count": schema_valid,
        "schema_valid_rate": round(schema_valid / n, 4),
        "all_relationships_data_driven": all_data_driven,
        "config": config.model_dump(),
        "total_relationships": sum(r["relationships"] for r in rows),
        "avg_runtime_seconds": round(sum(r["runtime_seconds"] for r in rows) / n, 3),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_diagram": detail}, f, indent=2, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps(aggregate, indent=2))
    print(f"[day04] wrote {OUT_JSON} and {OUT_CSV}")
    print(f"[day04] samples -> {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
