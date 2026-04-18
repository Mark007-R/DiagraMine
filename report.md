# Diagram Understanding & Structure Extraction - Report

## What I Built

A Python script that takes an architecture diagram image and automatically finds all the text, boxes, arrows, icons and how they connect to each other.

---

## Tools I Used

| What | Tool | Why |
|------|------|-----|
| Reading text from image | **EasyOCR** | Works out of the box, good accuracy, no setup needed |
| Finding boxes and regions | **OpenCV** (edges + contours) | Standard CV library, reliable for detecting rectangles |
| Finding dashed arrows | **OpenCV** (scan-line technique) | Scans every pixel row/column looking for dash patterns |
| Finding icons/logos | **OpenCV** (colour detection) | Icons are colourful — easy to pick out by colour |
| Building the relationship graph | **NetworkX + Matplotlib** | Simple graph library, good for drawing directed graphs |
| Saving results | **Python json/csv** | Standard, nothing fancy needed |

---

## How It Works

1. **Read text** — EasyOCR scans the image and finds all text labels. Some get split across lines so I merge nearby fragments back together.

2. **Find boxes** — Edge detection finds all the rectangles. Big ones are background regions (like the AWS section or Elasticsearch section). Smaller ones are the actual component boxes.

3. **Find arrows** — This was the hardest part. I scan every pixel row looking for a specific pattern: short dark segments with regular gaps between them (that's what a dashed line looks like at pixel level). Then I filter out false positives that sit on box borders or inside text. For a few arrows that were too faint to detect, the code auto-generates them by looking at which boxes should logically connect based on their positions.

4. **Find icons** — Colour-based detection picks out the logos (Plant An App, Docker, database cylinders, Elasticsearch).

5. **Connect everything** — Each text and icon gets assigned to its nearest box. Each arrow gets mapped to the boxes it connects. This gives us the full relationship graph.

---

## What It Detects

| Thing | Count |
|-------|-------|
| Text labels | 15 |
| Component boxes | 15 |
| Background regions | 4 |
| Dashed arrows | 14 (10 found by scanning + 4 auto-generated from box positions) |
| Icons | 5 |
| Relationships | 10 |

---

## What Was Tricky

The biggest challenge was **arrows vs box borders**. Both are dark lines on a light background. Dashed arrows have gaps between segments, but some box borders in this diagram are also dashed. I tried 4 different approaches before settling on the scan-line technique:

1. **Hough Line Transform** — found 100+ lines, mostly box borders. Too noisy.
2. **Residual-Skeleton** — subtract borders then skeletonize. Lost arrows near box edges.
3. **Box-pair corridor scanning** — scan between each pair of boxes. Hard to filter false positives from overlapping boxes.
4. **Scan-line dash detection** (final approach) — scan raw pixels row by row looking for strict dash patterns. Works best because it detects the actual dash pattern directly.

Even with the best approach, 4 arrows were too faint at pixel level (2-3 dashes only). The code handles these by auto-generating connector lines based on spatial relationships between detected boxes and existing arrows.