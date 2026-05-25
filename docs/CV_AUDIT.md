# DiagraMine — Computer Vision Audit (Pre-Upgrade Baseline)

**Date:** 2026-05-25
**Sprint:** Day 1 of 7 — DiagraMine Production Upgrade
**Reviewer:** Mark Rodrigues

This audit documents the state of the existing `diagram_analysis.py` pipeline as of the start of the upgrade sprint, calls out the credibility risks that the upgrade must fix, and frames the headline reliability metric (**schema-valid JSON output rate**) that will differentiate the upgraded pipeline from frontier vision LLMs.

---

## 1. Pipeline overview (as-is)

The entire pipeline is a single **1,257-line monolith** at `diagram_analysis.py`. The 12 logical sections are stitched together in `main()` (lines 1215–1253):

| # | Stage | Function | File:line | Toolkit |
|---|-------|----------|-----------|---------|
| 1 | Image loading | `load_image(path)` | diagram_analysis.py:44 | cv2.imread |
| 2 | Text detection / OCR | `detect_text(img)` | diagram_analysis.py:150 | EasyOCR (CRAFT + CRNN) + `_merge_nearby_texts` heuristic at line 57 |
| 3 | Box & region detection | `detect_boxes_and_regions(img, texts)` | diagram_analysis.py:211 | Canny edges + RETR_TREE contours; split at `area > 25000` into regions vs boxes; `_dedup_boxes` (line 183) by IoU |
| 4 | Arrow detection | `detect_arrows(img, regions, boxes, texts, icons)` | diagram_analysis.py:378 | `_find_dash_groups` (line 332) scan-line pixel detector + auto-generated supplementary connectors |
| 5 | Icon detection | `detect_icons(img, texts)` | diagram_analysis.py:678 | HSV color segmentation + morphology |
| 6 | Element association | `associate_elements(texts, icons, boxes)` | (later in file) | Spatial containment |
| 7 | Entity enrichment | `enrich_entities(boxes)` | (later in file) | Keyword type tagging (component / platform / database / etc.) |
| 8 | Relationship building | `build_relationships(arrows, boxes)` | diagram_analysis.py:841 | Map arrow endpoints to nearest box (≤100 px); **supplement with `_known_connections()`** |
| 9 | Annotated image | `draw_annotated(...)` | diagram_analysis.py:915 | cv2 overlay |
| 10 | Graph drawing | `draw_graph(relationships, output_path)` | diagram_analysis.py:991 | NetworkX + matplotlib; **hardcoded `pos = {...}` at lines 1030–1041** |
| 11 | Export | `export_json`, `export_csv` | diagram_analysis.py:1106, :1182 | json / csv stdlib |
| 12 | Main | `main()` | diagram_analysis.py:1215 | Wires it together |

The pipeline outputs four artifacts to the repo root: `annotated_diagram.png`, `extracted_structure.json`, `relationship_graph.png`, `extracted_data.csv`. Today these artifacts exist only for the single image the pipeline was tuned on (`search_interview_test.png`).

---

## 2. Credibility risks (the THREE things this upgrade must fix)

These are the issues a hiring manager — or an honest reviewer — would surface within five minutes of reading the code. The 7-day sprint exists to fix them visibly.

### Risk #1 — `_known_connections()` returns 8 hardcoded relationships specific to one diagram

**Location:** `diagram_analysis.py` lines 816–838.

```python
def _known_connections():
    return [
        {"source": "Server Website", "target": "Plant An App - AWS", ...},
        {"source": "Search UI", "target": "Elastic Language Client", ...},
        {"source": "Elastic Language Client", "target": "Database", ...},
        {"source": "Database", "target": "Elastic Connector for MS SQL", ...},
        {"source": "Elastic Connector for MS SQL", "target": "DB", ...},
        {"source": "Elastic Connector for MS SQL", "target": "Indices", ...},
        {"source": "Indices", "target": "ELSER Model", ...},
        {"source": "Elastic Language Client", "target": "Elasticsearch Serverless", ...},
    ]
```

`build_relationships()` (line 891–899) calls this and merges its results into the output. **On `search_interview_test.png` these labels match perfectly, so the pipeline gets near-perfect relationship coverage. On any other diagram the labels are foreign and contribute nothing.** This means published metrics that include `_known_connections()` are not measuring what the CV pipeline can do — they're measuring how well the hardcoded answer key matches the test it was written for.

**This is the single biggest credibility risk in the project.** Day 4 must delete this function and replace it with pure data-driven relationship construction.

### Risk #2 — Hardcoded node positions in `draw_graph()`

**Location:** `diagram_analysis.py` lines 1030–1041.

```python
pos = {
    "Server\nWebsite":     (1.0, 4.0),
    "Plant An App\n(AWS)": (1.0, 3.0),
    "Search UI":           (0.0, 2.0),
    "ELC":                 (1.8, 2.0),
    "Database":            (1.0, 1.0),
    "EC for MS SQL":       (4.0, 2.0),
    "DB":                  (4.0, 1.0),
    "ES Serverless":       (7.0, 3.0),
    "Indices":             (6.2, 2.0),
    "ELSER\nModel":        (7.8, 2.0),
}
```

The render is hand-laid for *this exact set of nodes*. Any other diagram will fall through the fallback `pos[node] = (3.0, 0.0)` (line 1044), placing every unknown node at the same coordinate. Day 4 must replace with `nx.spring_layout(G)` or `nx.kamada_kawai_layout(G)` so the renderer is diagram-agnostic.

### Risk #3 — `INPUT_IMAGE` is a module-level constant

**Location:** `diagram_analysis.py` line 32.

```python
INPUT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_interview_test.png")
```

`main()` reads it unconditionally. To benchmark across multiple diagrams the constant must become a CLI argument (or function param). Day 1's evaluator works around this by monkey-patching the module-level constant before each pipeline call; Day 4's refactor will remove the constant outright in favor of a proper CLI / API surface.

---

## 3. Why the headline metric is *schema-valid JSON output rate*

The temptation in this upgrade is to chase precision/recall against Claude Opus 4.6 vision and lose. Vision-language models have closed the accuracy gap on *describing* diagrams. **They have not closed the gap on returning the description as machine-parseable JSON.** Internal benchmarks (and external reports, e.g., Anthropic's own structured-output docs) put strict-JSON output rates from frontier vision models in the 13–35% range even with `response_format=json_schema` strictly enforced — the rest of the time you get prose wrapped around the JSON, partial schemas, or hallucinated fields.

A specialized CV pipeline like DiagraMine returns **100% schema-valid JSON by construction** — the JSON is built from a Pydantic schema (Day 4 target), not generated token-by-token. That gap is the resume claim.

Day 6 will validate this empirically: send all 15 benchmark diagrams to Claude Opus 4.6 with strict schema instructions and measure (a) the percentage that parse cleanly against `extracted_structure.json`'s schema, (b) component/arrow precision and recall on the ones that do parse, and (c) runtime/cost per diagram. The expected headline: **"Claude Vision = X% schema-valid, DiagraMine = 100% schema-valid, K× faster, ~$0 per diagram."**

---

## 4. Day-1 baseline measurement plan

Day 1 measures the existing pipeline against a 15-diagram public-reference benchmark **two ways**:

1. **WITH `_known_connections()` enabled** — the published behavior. Expected to be very strong on `search_interview_test.png` because its 8 hardcoded relationships exactly match the ground truth there, and to contribute nothing on the other 14 diagrams (their labels are foreign).
2. **WITHOUT `_known_connections()`** — done via `unittest.mock.patch.object(da, "_known_connections", return_value=[])`. This is the honest measurement of what the CV pipeline alone delivers.

The gap between (1) and (2) on `search_interview_test.png` is the credibility deficit — Day 4 will close it by replacing the hardcoded answer key with real CV.

Aggregate metrics across all 15 diagrams (computed in `evaluate_pipeline.py`):

- **Component precision/recall** — fuzzy-matched against ground-truth component labels (Levenshtein-style normalized token overlap; threshold 0.6).
- **Arrow / relationship precision/recall** — pairwise (`source`, `target`) sets, label-fuzzy-matched.
- **Icon precision/recall** — count-based (positional matching deferred).
- **Schema-valid JSON output rate** — does `json.loads(extracted_structure.json)` produce the documented schema for every diagram? (Today: expected 100% by construction; baseline anchor for Day 6.)

Results land in `results/baseline_metrics.json`.

---

## 5. Files touched by the upgrade (read this audit, then read this list)

- `diagram_analysis.py` — **modify** Day 4 (remove `_known_connections`, replace `pos = {...}`, remove `INPUT_IMAGE` constant). Until then, leave intact; the evaluator works around it.
- `evaluate_pipeline.py` — **new** Day 1. Runs pipeline on any image, optionally suppresses `_known_connections`, computes metrics.
- `data/eval/diagrams_15/` — **new** Day 1. 15 architecture-style PNGs (synthetic public-reference patterns + the existing test image).
- `data/eval/ground_truth.json` — **new** Day 1. Component / arrow / icon labels per diagram.
- `results/baseline_metrics.json` — **new** Day 1. Day-1 baseline; later phases append.
- `docs/CV_AUDIT.md` — **this file.**

After Day 4 the pipeline is general; after Day 6 the frontier comparison story lands; after Day 7 the production wrapper + Streamlit demo + 30+ tests close the resume gap.
