"""
Diagram Understanding & Structure Extraction — backward-compatible entrypoint.
==============================================================================
The original 1,257-line monolith has been refactored (Day 4) into the modular
`src/` package:

    src/text_detection/   src/box_detection/   src/arrow_detection/
    src/icon_detection/    src/graph/builder.py   src/pipeline.py   src/schemas.py

This file is now a thin wrapper that delegates to `src.pipeline`. It exists so
that `python diagram_analysis.py [image.png]` keeps working and reproduces the
original output filenames for the test diagram.

What was removed on Day 4 (the credibility fixes from the Day-1 CV audit):
  - `_known_connections()`   — 8 relationships hand-coded for the "Plant An App
    AWS" diagram, silently merged into every graph. The graph is now built
    purely from detected arrows (`src/graph/builder.build_relationships`).
  - hardcoded `pos = {...}`   — node coordinates pinned to that one diagram.
    Layout is now `kamada_kawai` / `spring`, computed from graph structure.
  - `INPUT_IMAGE` constant     — the single supported PNG. Image path is now a
    CLI argument / function parameter.

Run:  python diagram_analysis.py [path/to/diagram.png]
"""
from __future__ import annotations

import json
import os
import sys

import cv2

from src.graph import builder
from src.pipeline import annotate, extract, write_csv
from src.schemas import PipelineConfig

# Fallback image for `python diagram_analysis.py` with no argument — NOT a
# hardcoded sole-source; any path can be passed as the first CLI argument.
DEFAULT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_interview_test.png")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main(image_path: str | None = None, output_dir: str | None = None) -> dict:
    image_path = image_path or (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE)
    output_dir = output_dir or OUTPUT_DIR

    print("=" * 60)
    print("  DIAGRAM UNDERSTANDING & STRUCTURE EXTRACTION (modular pipeline)")
    print(f"  Input: {image_path}")
    print("=" * 60)

    result = extract(image_path, PipelineConfig())

    # Preserve the original output filenames for backward compatibility.
    annotated_path = os.path.join(output_dir, "annotated_diagram.png")
    graph_path = os.path.join(output_dir, "relationship_graph.png")
    json_path = os.path.join(output_dir, "extracted_structure.json")
    csv_path = os.path.join(output_dir, "extracted_data.csv")

    cv2.imwrite(annotated_path, annotate(image_path, result))
    builder.draw_graph([r.model_dump() for r in result.relationships], graph_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
    write_csv(result, csv_path)

    s = result.summary()
    print("\n" + "=" * 60)
    print("  EXTRACTION SUMMARY")
    print("=" * 60)
    for k, v in s.items():
        print(f"  {k:14s}: {v}")
    print("-" * 60)
    print("  All relationships are data-driven from detected arrows")
    print("  (no hardcoded connections, no hardcoded node positions).")
    print("=" * 60)
    print(f"  Annotated image : {annotated_path}")
    print(f"  JSON output     : {json_path}")
    print(f"  Graph image     : {graph_path}")
    print(f"  CSV output      : {csv_path}")
    print("=" * 60)
    print("  Done.")
    return result.model_dump()


if __name__ == "__main__":
    main()
