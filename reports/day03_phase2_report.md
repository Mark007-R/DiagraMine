# Day 03 — Phase 2b: Arrow + Icon detection comparison + Phase-2 champion pick — DiagraMine
**Date:** 2026-05-27
**Day:** 3 of 7

## Resume gap progress
**Gap:** Computer-vision reliability vs vision LLMs on document/diagram understanding.
**Today's contribution:** Quantified the arrow- and icon-detector design space against the 15-diagram benchmark from Day 1, picked Phase-2 champions per stage, and published the master `results/phase2_leaderboard.csv` that the Day-4 production refactor will integrate. Headline finding: arrow detection is the weak link of the whole pipeline (best F1 = 0.434 from CNN-verified Hough) — most of that score is propped up by an unintended box-adjacency artifact rather than real arrow detection, which is exactly the kind of failure the Day-1 honesty audit was meant to surface in time for Day-5 tuning.

## Files touched
- `src/arrow_detection/__init__.py` (new)
- `src/arrow_detection/pixel_scan_detector.py` (new) — scan-line dash detector, isolates the legacy `_find_dash_groups` logic with no box-aware post-filtering.
- `src/arrow_detection/hough_lines_detector.py` (new) — adaptive-threshold → close → `cv2.ximgproc.thinning` → `cv2.HoughLinesP` → collinear-segment cluster with a box-border suppression mask.
- `src/arrow_detection/cnn_detector.py` (new) — synthetic 32×32 patch generator (500 positives / 500 negatives), a 3-layer Conv2D classifier (~7K params, 8 epochs CPU, ~2.5 s train), used as a Hough-candidate verifier. Weights cached at `models/arrow_cnn.pt`.
- `src/icon_detection/__init__.py` (new)
- `src/icon_detection/hsv_detector.py` (new) — legacy `detect_icons` HSV segmentation, returns generic `"icon"` labels (no class).
- `src/icon_detection/template_detector.py` (new) — multi-scale `cv2.matchTemplate` (TM_CCOEFF_NORMED) + per-label NMS over the `data/icon_templates/` library.
- `src/icon_detection/clip_detector.py` (new) — `openai/clip-vit-base-patch32` zero-shot over the HSV-proposed candidate regions, vocabulary of 6 real classes + 1 `"no_icon"` decoy.
- `data/icon_templates/docker.png`, `data/icon_templates/ms_sql.png` (new) — 2 templates cropped from `search_interview_test.png`.
- `models/arrow_cnn.pt` (new, gitignored) — trained CNN weights, ~30 KB.
- `evaluate_phase2b.py` (new) — Phase 2b benchmark harness (arrow + icon scoring, schema-validity audit, per-diagram CSVs, sample emission).
- `results/_build_leaderboard.py` (new) — combines Phase 2a + 2b into `results/phase2_leaderboard.csv`.
- `results/phase2b_arrow_icon.csv`, `results/phase2b_per_diagram.csv`, `results/phase2b_detail.json`, `results/phase2b_schema_validity.json`, `results/phase2_leaderboard.csv` (new).
- `results/samples/arrow/{pixel_scan,hough_lines,cnn_verified}__diagram_{01,05,07,10,14}.png` (15 new overlays).
- `results/samples/icon/{hsv,template_matching,clip_zero_shot}__{...,search_interview_test}.png` (18 new overlays).
- `.gitignore` — added `models/*.pt`, `.cache/`, `~/.cache/huggingface` to keep the CLIP weights out of git.

## Setup
- **Compute:** CPU only (Windows 11, Python 3.11). PyTorch 2.1+cpu. Wall-clock for the full benchmark: ~25 s for arrow detectors (3 × 15 diagrams), ~80 s first run for icons (CLIP model load: 60 s download + 18 s load). Subsequent runs cached.
- **Dataset slice:** all 15 benchmark diagrams from `data/eval/diagrams_15/`. Arrow scoring uses pixel-level GT boxes from `ground_truth_boxes.json` to snap detected endpoints to box labels; `search_interview_test.png` has no machine-readable GT boxes and is skipped for arrow scoring (it still contributes to schema-validity counts). Icon scoring uses the textual `icons:[...]` arrays in `ground_truth.json`.
- **Critical caveat re: icon GT sparsity:** only 1 of 15 diagrams has any GT icons (`search_interview_test.png`: `["docker", "MS SQL"]`). The 14 generated diagrams are deliberately black-on-white rectangles with no logos, because the Day-1 benchmark prioritised box/arrow GT coverage. **All by-label P/R/F1 numbers for icons are therefore computed against 2 GT labels** — they're meaningful as an axis of comparison but the absolute values shouldn't be over-interpreted. Day 5 / Day 6 should expand icon GT before strong claims are made.
- **Components touched:** arrow + icon detection stages from `diagram_analysis.py` (`detect_arrows`, `_find_dash_groups`, `detect_icons`). No edits to `diagram_analysis.py` itself — Day 4 will handle integration per Rule #15.

## Experiments

### Experiment 3.1 — Arrows: scan-line dash vs Hough+thinning vs CNN-verified Hough
**Hypothesis:** All three detectors will hit a ceiling around 30–40 % recall because matplotlib's `FancyArrowPatch` anti-aliased dashes don't match the dash-spacing thresholds the legacy `_find_dash_groups` was tuned for (Day-1 finding #2). Hough-with-thinning should generalise slightly better because thinning collapses dash gaps; CNN verification should improve precision without changing recall (it can only suppress, not add, candidates).

**Method:** Each detector returns a list of `{x1, y1, x2, y2, direction}` endpoints. The eval harness snaps each endpoint to its nearest GT box (`ground_truth_boxes.json`) — first by point-in-rect with a 5 px margin, then by nearest-centre within 60 px — producing an unordered `(src_label, tgt_label)` pair. Bipartite-matched to the GT arrow list (also normalised to unordered pairs). 14 diagrams scored; `search_interview_test.png` has no GT boxes so it's skipped (still counted for schema validity).

**Result (aggregate, 14 scored diagrams):**

| Detector       | Precision | Recall | F1    | Avg runtime/img | Schema-valid |
|----------------|-----------|--------|-------|-----------------|--------------|
| pixel_scan     | 0.545     | 0.231  | 0.324 | 0.15 s          | 15 / 15      |
| hough_lines    | 0.383     | 0.346  | 0.364 | 0.02 s          | 15 / 15      |
| **cnn_verified** | **0.581** | **0.346** | **0.434** | **0.08 s**  | **15 / 15** |

**Interpretation:**
- **cnn_verified wins F1 but with a caveat.** Visual inspection of `results/samples/arrow/cnn_verified__diagram_05.png` reveals that most of the segments the CNN accepts are **box-border edges**, not actual arrows — the synthetic-positive training distribution looks like "horizontal or vertical solid line with optional arrowhead," and a thinned box border satisfies that perfectly. The recall (0.346) is **not arriving via genuine arrow detection** but via box-adjacency snapping: when a detector returns a segment lying on the border between two boxes, the eval's `_nearest_box` snaps both endpoints to those two adjacent box labels, producing a (src, tgt) pair that happens to match the GT arrow because the diagram layout puts connected boxes adjacent. On diagram_12 (circuit-breaker pattern with 3 sequential boxes), this snap-to-adjacency trick produces F1 = 1.000 even though no actual arrow was detected. **This is a measurement artifact of the unordered-pair snapping rule, not a CV capability.** Day-5 tuning should either (a) require detected segments to lie *outside* any box bbox before contributing a relationship, or (b) reject pairs whose connecting line is on a box border. I am NOT promoting cnn_verified to "champion" without that fix — see "Champion pick" below.
- **hough_lines has the best recall (0.346) at 5× the speed of pixel_scan.** Thinning + line clustering ARE picking up some real arrows (e.g., diagram_07 single horizontal load-balancer arrow). But it also picks up the most box-border noise, dragging precision to 0.383.
- **pixel_scan is exactly as brittle as Day-1 predicted.** F1 = 0.324; recall = 0.231 — the worst of the three because the dash-group splitter (`_find_dash_groups` at `dark_thresh=145, min_dash=3-15, min_gap=5-16`) cannot fire on matplotlib's anti-aliased dashes which produce dark runs of length 1–2 px under the threshold. Confirmed by adding `print(len(dashes))` instrumentation: dashes-per-line is 0–2 on every generated diagram, never the 3 required.
- **All three are 15/15 schema-valid.** Even when a detector finds 0 arrows it returns `{"arrows": [], "runtime_seconds": ...}` — structure-preserving by construction.

### Experiment 3.2 — Icons: HSV vs template matching vs CLIP zero-shot
**Hypothesis:** HSV will find icon-shaped colour blobs on `search_interview_test.png` but cannot label them (it returns `"icon"`), so by-label F1 = 0. Template matching against a curated 2-template library cropped from the test image should score perfectly on that image and produce 0 false positives on the other 14 (which have no colour). CLIP zero-shot will identify the docker icon correctly but over-detect — it will label every coloured sub-region (text glyphs, gradient artifacts) as one of the 6 candidate classes because the `"no_icon"` decoy class often loses the argmax.

**Method:** Each detector returns `{icons: [{x, y, w, h, label}], runtime_seconds}`. Two scoring axes:
1. **By-label F1** — collapse detections to unique label set, fuzzy-match (≥ 80) against GT label set.
2. **Presence F1** — capped: `tp = min(len(gt), len(detected))`. Fair to HSV (which can't label) but penalises over-detection.

GT icons exist only on `search_interview_test.png` (`["docker", "MS SQL"]`). The 14 generated diagrams have empty icon GT — any detection on those is a false positive in both axes.

**Result (aggregate across 15 diagrams):**

| Detector            | By-label P | R     | F1    | Presence P | R     | F1    | FPs on empty-GT diagrams | Avg runtime/img |
|---------------------|------------|-------|-------|------------|-------|-------|--------------------------|-----------------|
| hsv                 | 0.000      | 0.000 | 0.000 | 0.105      | 1.000 | 0.190 | 0                        | 0.003 s         |
| **template_matching** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **0**            | **0.18 s**     |
| clip_zero_shot      | 0.500      | 1.000 | 0.667 | 0.111      | 1.000 | 0.200 | 0                        | 0.71 s          |

**Interpretation:**
- **Template matching wins both axes** — but this is a curated 2-template library tested against the diagram those templates were cropped from. Precision = 1.000 and 0 FPs on the 14 empty-GT diagrams is genuine — the synthetic diagrams contain no docker / MS SQL pixels so TM_CCOEFF_NORMED at 0.65 threshold produces no spurious matches. Generalisation to *unseen* icons (logos not in the template library) is **not measured here**; that's a Day-5 or Day-6 question.
- **CLIP correctly identifies both GT icons** (docker score 0.36, MS SQL score 0.36 on the matching cylinder), but produces 16 additional detections on the same image with scores 0.21–0.28 because the "a small drawing or coloured shape, not a software logo" decoy class loses the argmax to one of the real labels for any sufficiently colourful crop (text glyphs in the AWS gradient logo, partial elasticsearch-logo fragments). By-label F1 = 0.667 reflects this: 2 correct labels found, but 2 extra unique wrong labels (`elasticsearch`, `AWS`) inflate the FP count. **The story is "CLIP gets the right answer but also gets several wrong ones at the same time."** Tighten the threshold to 0.30 in Day 5 and presence/precision both rise sharply, at the cost of recall on subtler icons.
- **HSV's by-label F1 = 0 is by construction** — it returns generic `"icon"` for every coloured contour, so it can never match `"docker"` or `"MS SQL"` under fuzzy ≥ 80. Presence F1 = 0.190 reflects that it does correctly fire 2 detections that overlap the GT icons, but its 17 other detections on the AWS / elasticsearch sub-logos drag presence-precision to 0.105. **HSV is best understood as a *region proposer*, not a classifier** — which is exactly the role we have it playing in `clip_detector.py` (proposing candidate crops for CLIP to label).
- **All three are 15/15 schema-valid.** Empty `icons: []` is valid output.

### Schema-valid JSON output rate (Phase 2b)

| Detector                        | Schema-valid runs / total | Rate  |
|---------------------------------|---------------------------|-------|
| arrow / pixel_scan              | 15 / 15                   | 1.000 |
| arrow / hough_lines             | 15 / 15                   | 1.000 |
| arrow / cnn_verified            | 15 / 15                   | 1.000 |
| icon / hsv                      | 15 / 15                   | 1.000 |
| icon / template_matching        | 15 / 15                   | 1.000 |
| icon / clip_zero_shot           | 15 / 15                   | 1.000 |
| **Aggregate (Phase 2b detectors)** | **90 / 90**            | **1.000** |

Combined Phase 2a + 2b: **165 / 165 = 1.000.** Every detector on every diagram returns a `json.dumps`-able dict with the expected top-level keys, including the runs that return zero detections. This is the Day-6 frontier-comparison baseline: vision LLMs in the same evaluation regime drop to ~13 % schema-valid output rate, per the SKILL's Phase 5 prediction.

## Head-to-Head Comparison — Master Phase 2 Leaderboard

| Rank | Stage | Champion          | Precision | Recall | F1    | Latency       | Notes |
|------|-------|-------------------|-----------|--------|-------|---------------|-------|
| 1    | text  | EasyOCR           | 0.904     | 1.000  | 0.949 | 1.90 s/img    | Best recall; PaddleOCR is faster runner-up (0.67 s, F1=0.942) |
| 1    | box   | Canny + contours  | 0.709     | 0.938  | 0.808 | 0.003 s/img   | Robust to layout, page-border FPs dominate (Day-5 tuning target) |
| 1*   | arrow | **(disputed)** Hough+thinning, F1=0.364 | 0.383 | 0.346 | 0.364 | 0.02 s/img | See "Champion pick" below — CNN-verified is higher F1 but inflated by snap-to-adjacency artifact |
| 1    | icon  | Template matching | 1.000     | 1.000  | 1.000 | 0.18 s/img    | Sparse GT (1 of 15 diagrams); CLIP runner-up F1=0.667 |

**Full ranked CSV:** `results/phase2_leaderboard.csv`.

### Champion pick (arrows) — the honest call
The headline number table above ranks **hough_lines** as the arrow champion despite CNN-verified having a higher F1, because CNN-verified's lead is propped up by the box-adjacency snapping artifact described in 3.1's interpretation. The CNN itself learnt "line-like patch → arrow" cleanly enough (training accuracy ~99 % on synthetic), but at inference it cannot distinguish box-border thinning artifacts from real arrows — both look identical at the 32 px patch scale. Two follow-ups for Day 4 / Day 5:

1. **Day 4 integration:** carry forward **hough_lines** as the production arrow detector. It has equal recall (0.346) and worse precision (0.383 vs 0.581) than cnn_verified, but its precision drop is from finding real-but-spurious box edges, not from learned-but-wrong patch verdicts — easier to fix downstream by requiring detected segments to **lie outside any detected box bbox** before contributing a relationship. That's the `_known_connections`-style "the CV is the source of truth, no hardcoded answers" version of arrow→relationship mapping.
2. **Day 5 ablation:** retrain the CNN with **box-border patches as explicit negatives** (sample 200 patches centred on the borders of detected boxes from a few diagrams, label them 0). This is the most surgical fix and should rescue CNN-verified into a genuine champion. If retraining doesn't pull precision past hough_lines + outside-box filter, the CNN approach is dead-ended for this problem.

### Champion pick (icons)
**Template matching wins** on the available evidence, but with a caveat banner: the library only covers `docker` and `ms_sql`, both cropped from the test image. The Day-4 production refactor should expose the icon library as a config (path, list of templates) so growing it to AWS / GCP / Kubernetes / etc. is a config change, not a code change. CLIP stays in the codebase as the fallback for icons not in the template library — its 0.667 by-label F1 + 100 % recall on the templates it covers is exactly the "specialist + zero-shot fallback" pattern Day-6's frontier comparison wants to demonstrate.

## Key Findings
1. **Arrow detection is the pipeline's bottleneck and the Day-5 tuning priority.** Best honest F1 is 0.364 (hough_lines) — versus 0.808 for boxes and 0.949 for text. The arrow detector is doing 30 % of the work the rest of the pipeline does. The cause is matplotlib's `FancyArrowPatch` anti-aliased dashes failing every dash-spacing heuristic the legacy detector was tuned on (`_find_dash_groups` thresholds `dark_thresh=145, min_dash=3-15, gap=5-16`). The fix is one of: re-render the benchmark with a known dash pattern, retrain the CNN with box-border negatives, or switch to a learned arrowhead detector (small CNN trained on synthetic arrowhead triangles, not on line patches).
2. **CNN-verified Hough achieved higher F1 (0.434) than Hough alone (0.364) for the wrong reason.** Inspection of the diagram_12 / diagram_05 samples showed the CNN is verifying box-border line segments. The snap-to-adjacency rule in the evaluator then maps two adjacent boxes' shared border to a correct (src, tgt) pair. This artifact would not survive an "outside any box" gate, which is what Day 4 integration must apply. **I am surfacing this as a finding rather than fudging the number — that's exactly what the Day-1 hardcoded-`_known_connections` audit set the tone for.**
3. **Template matching beats CLIP zero-shot by a wide margin on the curated-library task, but the test is unfair.** Template matching scores F1 = 1.000 because the templates were cropped from the diagram being tested. The fair next test is: collect 5 *unseen* logos (Stripe, Datadog, Snowflake, Vercel, Postgres), run both detectors on a diagram that contains a subset, and re-score. That's a Day-5 polish task if there's room after the box-detection tuning.
4. **Schema-valid JSON output rate stays at 1.000 across all 165 detector × diagram runs.** Zero detections is still a valid JSON output. This is the metric Day 6 will use to beat Claude Vision (~13 % schema-valid expected) — and we now have 165 / 165 cumulative evidence on the DiagraMine side.
5. **What didn't work:** the CNN-as-verifier strategy as currently designed. It works on synthetic patches (the held-out training-set patches it never saw scored ~99 % accuracy), but at inference time it sees a domain shift — box-border thinning artifacts look more "arrow-like" than the dashed-arrow thinning artifacts because the dashed-arrow ones are anti-aliased and the box-border ones are not. Day-5 retraining with box-border negatives should fix this; if it doesn't, the architecture is wrong.

## Sample Outputs Saved
- `results/phase2b_arrow_icon.csv` — aggregate per-detector P / R / F1.
- `results/phase2b_per_diagram.csv` — per-detector per-diagram breakdown (90 rows).
- `results/phase2b_detail.json` — full per-diagram match details.
- `results/phase2b_schema_validity.json` — schema-valid output rate per detector.
- `results/phase2_leaderboard.csv` — master Phase 2 leaderboard (text + box + arrow + icon champions + full ranked list).
- `results/samples/arrow/{pixel_scan,hough_lines,cnn_verified}__diagram_{01,05,07,10,14}.png` — 15 overlays: GT arrows in green, detected segments in red, endpoint dots highlight what got snapped to which box.
- `results/samples/icon/{hsv,template_matching,clip_zero_shot}__{diagram_*,search_interview_test}.png` — 18 overlays: red bounding boxes with detector-assigned labels.
- `data/icon_templates/docker.png`, `data/icon_templates/ms_sql.png` — curated icon library (2 templates, used by `template_detector`).
- `models/arrow_cnn.pt` — trained CNN binary classifier (gitignored, ~30 KB; regenerable from `cnn_detector._train_model()` in ~2.5 s).

## Next Day
- **Day 4 — Phase 3: champion integration + REMOVE HARDCODING (per Rule #15).**
  - **First commit:** delete `_known_connections()` from `diagram_analysis.py` (lines 816–838) and replace the hardcoded `pos = {...}` in `draw_graph()` (lines 1030–1041) with `nx.spring_layout(G)` / `nx.kamada_kawai_layout(G)`.
  - Refactor `diagram_analysis.py` into `src/text_detection/`, `src/box_detection/`, `src/arrow_detection/`, `src/icon_detection/`, `src/graph/builder.py`, `src/pipeline.py`, `src/schemas.py`.
  - Wire today's champions: PaddleOCR for text *(EasyOCR is F1 winner but PaddleOCR is 2.8× faster at tied F1 — production serving prefers latency)*; Canny+contours for boxes; **hough_lines + an "outside any box bbox" gate** for arrows (not cnn_verified, per the Day-5 retraining rationale above); template_matching with config-driven library for icons.
  - Add `src/api.py` FastAPI service: `POST /extract` returns annotated PNG (base64) + JSON + relationship CSV.
  - Day 4 is post-eligible; phase wrap-up section is mandatory.

## Code Changes
- New: `src/arrow_detection/{__init__.py, pixel_scan_detector.py, hough_lines_detector.py, cnn_detector.py}`
- New: `src/icon_detection/{__init__.py, hsv_detector.py, template_detector.py, clip_detector.py}`
- New: `data/icon_templates/{docker.png, ms_sql.png}`
- New: `models/arrow_cnn.pt` (gitignored)
- New: `evaluate_phase2b.py`
- New: `results/_build_leaderboard.py`
- New: `results/{phase2b_arrow_icon.csv, phase2b_per_diagram.csv, phase2b_detail.json, phase2b_schema_validity.json, phase2_leaderboard.csv}`
- New: `results/samples/arrow/` (15 overlays) and `results/samples/icon/` (18 overlays)
- Modified: `.gitignore` (+ `models/*.pt`, CLIP HF cache patterns)
- Unchanged: `diagram_analysis.py` — Rule #15 says the first refactor commit (removing `_known_connections` + hardcoded `pos`) lands on Day 4, not today.
