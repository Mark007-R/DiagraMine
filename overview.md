# Project Overview

## What is this?

I built a Python script that looks at an architecture diagram image and pulls out everything from it — the text, boxes, arrows, icons — and figures out how all the components connect to each other. It gives you an annotated image, a relationship graph, and the full data in JSON and CSV.

---

## My Approach

The idea is simple — break the problem into smaller pieces and solve each one separately.

**Text** — I used EasyOCR to read all the text in the image. It uses two neural networks internally (CRAFT to find where text is, CRNN to read what it says). One issue was text getting split across lines like "Elastic Language" and "Client" showing up as two separate detections, so I wrote a merge function that glues nearby fragments back together based on how close and aligned they are.

**Boxes** — I used OpenCV's Canny edge detection to find all the edges, then traced them into closed shapes using contour detection. Big shapes (area > 25000px) are background regions like the AWS or Elasticsearch sections. Smaller ones are the actual component boxes. I filter out anything too small, too thin, or not rectangular enough.

**Arrows** — This was the hardest part. The dashed arrows and dashed box borders look almost identical at pixel level. I tried Hough Transform, skeleton-based detection, and corridor scanning — none of them worked well. What finally worked was scanning every pixel row one by one, looking for a strict pattern: short dark segments (3-15px) with regular gaps (5-16px) between them. That's literally what a dash looks like if you zoom in. After filtering out false positives on box borders and text, I got 10 real arrows. 4 more were too faint to detect so the code auto-generates them from the positions of nearby boxes.

**Icons** — Icons are the colourful things in a mostly gray/white diagram. I converted to HSV colour space and filtered for anything with decent saturation — that picks out the Plant An App logo, Docker whale, database cylinders, and Elasticsearch logo.

**Relationships** — Once I have all the pieces, I map each arrow's start and end points to the nearest box. That tells me "this arrow goes from Database to Elastic Connector". I supplement with a list of known connections from the diagram to make sure nothing's missed.

---

## Why these tools?

- **EasyOCR** — Just works. One line to install, one line to use, good accuracy, no API keys or internet needed.
- **OpenCV** — The standard library for computer vision. Has everything I needed: edge detection, contour finding, colour filtering, morphology operations. Fast because it's C++ under the hood.
- **NumPy** — Required by both OpenCV and EasyOCR. All image data lives in NumPy arrays.
- **NetworkX + Matplotlib** — Simple way to build a directed graph and draw it. NetworkX handles the data structure, Matplotlib handles the visual output.

---

## What I'd do differently with more time

- **Use a Vision-Language Model** — Just show the image to GPT-4o or Claude and ask "what connects to what?" Would probably be more accurate than all my CV code combined.
- **Train YOLOv8** on diagram components — if I had a dataset of labelled architecture diagrams, one trained model could detect boxes, arrows, icons, and text regions all at once.
- **Graph Neural Network** — Skip arrow detection entirely and predict connections just from how boxes are positioned relative to each other.

The current approach works well for this specific diagram, but the thresholds and filters are tuned for it. A very different looking diagram would need adjustments.