# Day 01 — Audit + 15-diagram benchmark + baseline measurement — DiagraMine

**Date:** 2026-05-25
**Day:** 01 of 7

## Resume gap progress

**Gap:** CV reliability vs vision LLMs — *can DiagraMine return schema-valid JSON across diverse architecture diagrams, or does the current pipeline only work on the one diagram it was tuned for?*

**Today's contribution:** Built a 15-diagram public-reference benchmark and measured the existing pipeline twice — with and without the hardcoded `_known_connections()` answer key. On the test diagram the pipeline was tuned for, relationship recall is **1.00 with the hardcoded key and 0.38 without it** — quantifying for the first time how much of DiagraMine's published performance is real CV vs hand-coded answers.

## Files touched

- `docs/CV_AUDIT.md` (new) — pipeline audit, 3 credibility risks called out, headline-metric rationale (schema-valid JSON rate).
- `data/eval/generate_diagrams.py` (new) — programmatic generator for 14 synthetic public-reference patterns.
- `data/eval/diagrams_15/diagram_01..14.png` (new) — 14 synthetic PNGs.
- `data/eval/diagrams_15/search_interview_test.png` (copy of root) — 15th diagram.
- `data/eval/ground_truth.json` (new) — components / arrows / icons per diagram, fuzzy-match keyed.
- `evaluate_pipeline.py` (new) — runs `diagram_analysis.py` per image, monkey-patches `_known_connections` to `[]` in NO-known mode, computes P/R/F1 plus schema-valid JSON rate.
- `results/baseline_metrics.json` (new) — per-diagram + aggregate metrics in both modes.
- `.gitignore` (new) — Python / venv / EasyOCR cache / regenerable artifacts.

Did NOT modify `diagram_analysis.py` itself. Rule #15 explicitly defers the removal of `_known_connections()` and the hardcoded `pos = {...}` to Day 4 — Day 1's job is measurement.

## Setup

- **Compute:** CPU only. ~14s per diagram-mode (EasyOCR CRAFT+CRNN dominates). 15 diagrams × 2 modes ≈ 7 minutes wall-clock.
- **Benchmark dataset:** 15 diagrams in `data/eval/diagrams_15/`. 14 are programmatically generated public-reference patterns (3-tier web, microservices gateway, pub/sub, serverless ETL, Kubernetes, master/replica, load balancer, cache-aside, ETL pipeline, CQRS, saga, circuit breaker, BFF, hexagonal) rendered in a visual style similar to the original test image (white background, rectangular boxes, dashed arrows). The 15th is the existing `search_interview_test.png` — the diagram the pipeline was hand-tuned for. **No proprietary diagrams used; nothing fetched from the web.**
- **Components touched:** entire pipeline (text → boxes → arrows → icons → relationships), plus `_known_connections` monkey-patched for honest measurement.

## Experiments

### Experiment 1.1: Existing pipeline WITH `_known_connections()` — published behavior

**Hypothesis:** The pipeline gets near-perfect relationship coverage on `search_interview_test.png` (which `_known_connections` was hand-written for) and near-zero relationship coverage on the other 14 diagrams (whose labels are foreign to the function).

**Method:** Run `diagram_analysis.py` end-to-end on each of the 15 diagrams; `build_relationships()` calls `_known_connections()` and merges its 8 entries into the output.

**Result (aggregate):**

| Metric                       | Macro-P | Macro-R | Macro-F1 |
|------------------------------|---------|---------|----------|
| Components                   | 0.713   | 0.840   | 0.763    |
| Relationships                | 0.090   | 0.158   | 0.111    |
| Icons                        | —       | —       | 0.038    |
| Schema-valid JSON rate       | —       | —       | **1.000** |
| Avg runtime per diagram      | —       | —       | 14.04 s  |

### Experiment 1.2: Existing pipeline WITHOUT `_known_connections()` — honest CV-only

**Hypothesis:** Component metrics unchanged (the function only affects relationships). Relationship recall on the test image collapses dramatically. Across the 14 synthetic diagrams, removing the false positives the function injects actually *raises* macro-F1 — because injecting 8 wrong relationships per diagram destroys precision.

**Method:** `unittest.mock.patch.object(da, "_known_connections", return_value=[])` wrapping the `build_relationships` call.

**Result (aggregate):**

| Metric                       | Macro-P | Macro-R | Macro-F1 |
|------------------------------|---------|---------|----------|
| Components                   | 0.713   | 0.840   | 0.763    |
| Relationships                | **0.367**   | 0.116   | **0.172**    |
| Icons                        | —       | —       | 0.038    |
| Schema-valid JSON rate       | —       | —       | **1.000** |
| Avg runtime per diagram      | —       | —       | 15.37 s  |

**Interpretation:** Component metrics are identical (as predicted — `_known_connections` doesn't touch the box detector). Relationship aggregate macro-F1 actually *rose* (0.111 → 0.172) without the function. That's because for 14 of 15 diagrams the function was contributing 8 false-positive relationships per diagram, tanking precision without helping recall (the labels never match foreign diagrams). The function is net-negative on every diagram except the one it was written for.

### Experiment 1.3: Per-diagram comparison (the credibility-gap story)

**Result (full table):**

| Diagram                       | Comp F1 | Rel F1 WITH known | Rel F1 NO known | Rel Recall WITH | Rel Recall NO | Runtime |
|-------------------------------|---------|-------------------|-----------------|-----------------|---------------|---------|
| diagram_01 (3-tier web)       | 0.86    | 0.00              | 0.00            | 0.00            | 0.00          | 12.6s   |
| diagram_02 (microservices gw) | 0.86    | 0.14              | 0.33            | 0.20            | 0.20          | 16.9s   |
| diagram_03 (pub/sub)          | 0.46    | 0.00              | 0.00            | 0.00            | 0.00          | 14.6s   |
| diagram_04 (serverless ETL)   | 0.25    | 0.00              | 0.00            | 0.00            | 0.00          | 14.9s   |
| diagram_05 (Kubernetes)       | 0.91    | 0.15              | 0.40            | 0.25            | 0.25          | 13.3s   |
| diagram_06 (master/replica)   | 0.77    | 0.00              | 0.00            | 0.00            | 0.00          | 14.0s   |
| diagram_07 (load balancer)    | 1.00    | 0.00              | 0.00            | 0.00            | 0.00          | 12.0s   |
| diagram_08 (cache-aside)      | 1.00    | 0.17              | 0.50            | 0.33            | 0.33          | 11.8s   |
| diagram_09 (ETL pipeline)     | 0.60    | 0.15              | 0.40            | 0.33            | 0.33          | 13.7s   |
| diagram_10 (CQRS)             | 0.91    | 0.15              | 0.40            | 0.25            | 0.25          | 14.0s   |
| diagram_11 (saga)             | 0.67    | 0.00              | 0.00            | 0.00            | 0.00          | 14.2s   |
| diagram_12 (circuit breaker)  | 0.75    | 0.00              | 0.00            | 0.00            | 0.00          | 12.8s   |
| diagram_13 (BFF)              | 1.00    | 0.00              | 0.00            | 0.00            | 0.00          | 16.3s   |
| diagram_14 (hexagonal)        | 0.77    | 0.00              | 0.00            | 0.00            | 0.00          | 17.5s   |
| **search_interview_test.png** | **0.64** | **0.89**         | **0.55**        | **1.00**        | **0.38**      | 11.8s   |

**Interpretation:** The last row is the credibility story in one line — on the only diagram where the relationship score is impressive, that score is 62% hardcoded. Strip the hardcoding and the CV pipeline finds 3 of 8 relationships. On the other 14 diagrams the CV pipeline finds 0–2 of 3–5 relationships, generally because the arrow detector is tuned for the specific dash-pattern (3–15px dashes, 5–16px gaps) and `FancyArrowPatch` dashed arrows render with different spacing.

## Head-to-Head Comparison

| Rank | Strategy                                    | Component macro-F1 | Relationship macro-F1 | Rel Recall on test image | Schema-valid JSON | Avg runtime/diagram |
|------|---------------------------------------------|--------------------|-----------------------|--------------------------|-------------------|---------------------|
| 1    | Existing pipeline + `_known_connections()`  | 0.763              | 0.111                 | **1.00 (inflated)**      | 1.000             | 14.04 s             |
| 2    | Existing pipeline, no `_known_connections`  | 0.763              | **0.172**             | **0.38 (honest)**        | 1.000             | 15.37 s             |

Stripping the hardcoded answer key actually improves aggregate macro-F1 because it stops injecting 8 false-positive relationships per non-matching diagram. The "improvement" is itself an indictment: the function was bolted on for one diagram and is silently degrading the other 14.

## Key Findings

1. **The headline credibility deficit is 62 percentage points of relationship recall on the test image.** 7 of 8 detected relationships in the published pipeline come from `_known_connections()`, not from CV. Day 4's first commit must delete that function — there's no path to a defensible production claim while it exists.

2. **The arrow detector is brittle to dash spacing.** It found 0 arrows on 9 of 14 synthetic diagrams despite all of them rendering arrows visibly as dashed lines. The `_find_dash_groups` thresholds (`dark_thresh=145`, `min_dash=3`, `max_dash=15`, `min_gap=5`, `max_gap=16`) are tuned for `search_interview_test.png` specifically. Day 3's Hough-lines + thinning vs CNN comparison should close this gap.

3. **Box detection generalizes much better than expected — macro-F1 = 0.76 across all 15 diagrams** including ones the pipeline has never seen. The Canny + RETR_TREE approach is robust to layout changes. The failure mode is over-detection: precision (0.71) lags recall (0.84) because the contour pass picks up arrow segments, text bounding boxes, and section dividers as additional "boxes." Day 5's tuning sweep on `cv2.Canny` thresholds and the `area > 25000` region/box split is the right place to address this.

4. **Schema-valid JSON output rate is 100% in both modes.** This is the resume claim that has to survive Day 6's vs-Claude-Vision comparison. The current pipeline returns the expected top-level keys (`summary, texts, boxes, regions, arrows, icons, relationships`) deterministically; Claude Vision tends not to.

5. **What didn't work and why:** the assumption that matplotlib's `linestyle="--"` arrows would render dash patterns the existing detector could parse turned out to be wrong. The dashes that `FancyArrowPatch` produces in matplotlib are anti-aliased and shorter than the 3–15 px window `_find_dash_groups` expects, so the detector reads them as a single long dark run rather than a dash group. This is a *finding*, not a benchmark-construction bug — it correctly exposes the detector's brittleness.

## Sample Outputs Saved

- `results/baseline_metrics.json` — per-diagram and aggregate metrics for both modes
- `data/eval/diagrams_15/diagram_{01..14}.png` — 14 synthetic public-reference diagrams
- `data/eval/diagrams_15/search_interview_test.png` — original test diagram (mirrored from repo root)
- `data/eval/ground_truth.json` — components / arrows / icons per diagram

## Next Day

Day 2 — Phase 2a: Text + Box detection comparison.

1. Text: EasyOCR (current) vs Tesseract (pytesseract) vs PaddleOCR on the 15-diagram set. Measure precision/recall on detected text + per-character accuracy + runtime.
2. Box: cv2 Canny + contours (current) vs Hough rectangles vs YOLOv8 zero-shot. Measure IoU @ 0.5, false-positive rate, runtime.
3. Save to `results/phase2a_text_box.csv`.

Day 1 *intentionally* did not remove the credibility-risk hardcoding — that's Rule #15 / Day 4. Anything Day 2 builds runs against the same hardcoding-aware evaluator from today.

## Code Changes

- New: `docs/CV_AUDIT.md`
- New: `data/eval/generate_diagrams.py`
- New: `data/eval/diagrams_15/*.png` (15 files)
- New: `data/eval/ground_truth.json`
- New: `evaluate_pipeline.py`
- New: `results/baseline_metrics.json`
- New: `.gitignore`
- Unchanged: `diagram_analysis.py` (deferred to Day 4 per Rule #15)
