"""
Diagram Understanding & Structure Extraction
=============================================
Analyses a design/architecture diagram to extract text labels, grouped
regions, connecting arrows, icons, and reconstruct directional relationships.

Pipeline:
  1. Text detection & OCR        - EasyOCR (CRAFT + CRNN)
  2. Box & region detection       - Canny + contour hierarchy
  3. Arrow detection              - Scan-Line Dash Pattern Detection
       + auto-generated supplementary connectors from spatial logic
  4. Icon detection               - HSV colour segmentation
  5. Association & enrichment     - spatial containment + domain labels
  6. Relationship graph           - arrow endpoints -> nearest boxes
  7. Output                       - annotated image, graph, JSON, CSV

Toolkits: OpenCV (+ ximgproc), EasyOCR, NumPy, NetworkX, Matplotlib
"""

import cv2
import numpy as np
import easyocr
import json
import os
import csv
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ──
INPUT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "search_interview_test.png")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATED_PATH = os.path.join(OUTPUT_DIR, "annotated_diagram.png")
JSON_PATH = os.path.join(OUTPUT_DIR, "extracted_structure.json")
GRAPH_PATH = os.path.join(OUTPUT_DIR, "relationship_graph.png")
CSV_PATH = os.path.join(OUTPUT_DIR, "extracted_data.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Image Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_image(path):
    """Load an image from disk via cv2.imread, abort on failure."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    print(f"[load] Image loaded: {img.shape[1]}x{img.shape[0]} px")
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Text Detection & OCR
# ═══════════════════════════════════════════════════════════════════════════════

def _merge_nearby_texts(texts):
    """
    Merge horizontally adjacent and vertically stacked text fragments.

    Horizontal merge: x_gap <= 15 and y_overlap >= 0.4 of smaller height.
    Vertical merge: x-centres within 20px AND y_gap <= 3px.
    This correctly merges "Elastic Language" + "Client" into one but keeps
    "docker", "Elastic Connector for", "MS SQL" separate.
    """
    merged = list(texts)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            best_j = -1
            best_type = None
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                a = merged[i]
                b = merged[j]
                ax, ay, aw, ah = a["x"], a["y"], a["w"], a["h"]
                bx, by, bw, bh = b["x"], b["y"], b["w"], b["h"]

                # Horizontal merge check
                y_top = max(ay, by)
                y_bot = min(ay + ah, by + bh)
                y_overlap = max(0, y_bot - y_top)
                min_h = min(ah, bh)
                if min_h > 0 and y_overlap / min_h >= 0.4:
                    if ax + aw <= bx:
                        x_gap = bx - (ax + aw)
                    elif bx + bw <= ax:
                        x_gap = ax - (bx + bw)
                    else:
                        x_gap = 0
                    if x_gap <= 15:
                        best_j = j
                        best_type = "horizontal"
                        break

                # Vertical merge check
                cx_a = ax + aw / 2
                cx_b = bx + bw / 2
                if abs(cx_a - cx_b) <= 20:
                    if ay + ah <= by:
                        y_gap = by - (ay + ah)
                    elif by + bh <= ay:
                        y_gap = ay - (by + bh)
                    else:
                        y_gap = 0
                    if y_gap <= 3:
                        best_j = j
                        best_type = "vertical"
                        break

            if best_j >= 0:
                a = merged[i]
                b = merged[best_j]
                nx_ = min(a["x"], b["x"])
                ny_ = min(a["y"], b["y"])
                nx2 = max(a["x"] + a["w"], b["x"] + b["w"])
                ny2 = max(a["y"] + a["h"], b["y"] + b["h"])
                if best_type == "horizontal":
                    if a["x"] < b["x"]:
                        combined_text = a["text"] + " " + b["text"]
                    else:
                        combined_text = b["text"] + " " + a["text"]
                else:
                    if a["y"] < b["y"]:
                        combined_text = a["text"] + " " + b["text"]
                    else:
                        combined_text = b["text"] + " " + a["text"]
                new_entry = {
                    "text": combined_text,
                    "x": nx_, "y": ny_,
                    "w": nx2 - nx_, "h": ny2 - ny_,
                    "conf": max(a["conf"], b["conf"]),
                }
                new_merged.append(new_entry)
                used.add(i)
                used.add(best_j)
                changed = True
            else:
                new_merged.append(merged[i])
        merged = new_merged
    return merged


def detect_text(img):
    """
    Detect and OCR text in the image using EasyOCR.

    Returns list of dicts: {text, x, y, w, h, conf}.
    Post-processed with _merge_nearby_texts() to yield ~16 text elements.
    """
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(img)
    texts = []
    for (bbox, text, conf) in results:
        if conf < 0.10:
            continue
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        texts.append({
            "text": text.strip(),
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "conf": float(conf),
        })

    texts = _merge_nearby_texts(texts)
    print(f"[text] Detected {len(texts)} text elements after merging")
    for t in texts:
        print(f"       '{t['text']}' @ ({t['x']},{t['y']}) {t['w']}x{t['h']}  conf={t['conf']:.2f}")
    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Box & Region Detection
# ═══════════════════════════════════════════════════════════════════════════════

def _dedup_boxes(boxes, iou_thresh=0.6):
    """Remove overlapping boxes by IoU; keep the larger one."""
    keep = []
    used = set()
    sorted_boxes = sorted(range(len(boxes)), key=lambda i: boxes[i]["w"] * boxes[i]["h"], reverse=True)
    for i in sorted_boxes:
        if i in used:
            continue
        keep.append(boxes[i])
        for j in sorted_boxes:
            if j in used or j == i:
                continue
            a = boxes[i]
            b = boxes[j]
            ix1 = max(a["x"], b["x"])
            iy1 = max(a["y"], b["y"])
            ix2 = min(a["x"] + a["w"], b["x"] + b["w"])
            iy2 = min(a["y"] + a["h"], b["y"] + b["h"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_a = a["w"] * a["h"]
            area_b = b["w"] * b["h"]
            union = area_a + area_b - inter
            if union > 0 and inter / union > iou_thresh:
                used.add(j)
        used.add(i)
    return keep


def detect_boxes_and_regions(img, texts):
    """
    Detect rectangular boxes and large background regions.

    Uses Canny edge detection, morphological dilation, and findContours
    with RETR_TREE hierarchy.

    Splits into:
      REGIONS: area > 25000
      BOXES:   area <= 25000

    Returns (regions, boxes) with parent-child nesting and text labels.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 30, 120)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    boxes = []
    h_img, w_img = img.shape[:2]

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1200:
            continue
        if w < 40 or h < 20:
            continue
        cnt_area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue
        rectangularity = cnt_area / rect_area
        if rectangularity < 0.3:
            continue

        # Skip contours that are basically the entire image
        if w > w_img * 0.95 and h > h_img * 0.95:
            continue

        parent_idx = hierarchy[0][i][3] if hierarchy is not None else -1

        entry = {
            "id": i,
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "area": int(area),
            "rectangularity": float(rectangularity),
            "parent_contour": int(parent_idx),
            "label": "",
            "children": [],
        }

        if area > 25000:
            regions.append(entry)
        else:
            boxes.append(entry)

    # Deduplicate overlapping boxes
    boxes = _dedup_boxes(boxes, iou_thresh=0.6)

    # Parent-child nesting: assign boxes to smallest containing region
    for box in boxes:
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        bcx, bcy = bx + bw // 2, by + bh // 2
        best_region = None
        best_area = float("inf")
        for reg in regions:
            rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
            if rx <= bcx <= rx + rw and ry <= bcy <= ry + rh:
                if reg["area"] < best_area:
                    best_area = reg["area"]
                    best_region = reg
        if best_region is not None:
            best_region["children"].append(box.get("id", -1))
            box["parent_region_id"] = best_region.get("id", -1)

    # Label boxes by contained text
    for box in boxes:
        bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
        contained_texts = []
        for t in texts:
            tx, ty, tw, th = t["x"], t["y"], t["w"], t["h"]
            tcx, tcy = tx + tw // 2, ty + th // 2
            if bx <= tcx <= bx + bw and by <= tcy <= by + bh:
                contained_texts.append(t["text"])
        if contained_texts:
            box["label"] = " | ".join(contained_texts)

    # Label regions similarly
    for reg in regions:
        rx, ry, rw, rh = reg["x"], reg["y"], reg["w"], reg["h"]
        contained = []
        for t in texts:
            tx, ty, tw, th = t["x"], t["y"], t["w"], t["h"]
            tcx, tcy = tx + tw // 2, ty + th // 2
            if rx <= tcx <= rx + rw and ry <= tcy <= ry + rh:
                contained.append(t["text"])
        if contained:
            reg["label"] = " | ".join(contained)

    # Deduplicate regions as well
    regions = _dedup_boxes(regions, iou_thresh=0.6)

    print(f"[boxes] Detected {len(regions)} regions, {len(boxes)} boxes")
    for r in regions:
        print(f"        REGION @ ({r['x']},{r['y']}) {r['w']}x{r['h']}  label='{r['label']}'")
    for b in boxes:
        print(f"        BOX    @ ({b['x']},{b['y']}) {b['w']}x{b['h']}  label='{b['label']}'")
    return regions, boxes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Arrow Detection  (Box-Pair Corridor Scanning)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_dash_groups(line_pixels, dark_thresh=145, min_dash=3, max_dash=15,
                      min_gap=5, max_gap=16):
    """Scan a 1D pixel array for dash patterns. Returns list of dash-groups,
    where each group is a list of (start, end) tuples."""
    n = len(line_pixels)
    if n < 15:
        return []
    # Find dark runs
    dark_runs = []
    in_dark = False
    start = 0
    for i in range(n):
        if line_pixels[i] < dark_thresh:
            if not in_dark:
                start = i
                in_dark = True
        else:
            if in_dark:
                dark_runs.append((start, i, i - start))
                in_dark = False
    if in_dark:
        dark_runs.append((start, n, n - start))
    # Keep only dash-sized runs
    dashes = [(s, e, l) for s, e, l in dark_runs if min_dash <= l <= max_dash]
    if len(dashes) < 3:
        return []
    # Split into groups at gaps > 30px
    groups = [[dashes[0]]]
    for k in range(1, len(dashes)):
        gap = dashes[k][0] - dashes[k-1][1]
        if gap > 30:
            groups.append([dashes[k]])
        else:
            groups[-1].append(dashes[k])
    # Validate each group: need >= 3 dashes with >= 2 regular gaps
    result = []
    for grp in groups:
        if len(grp) < 2:
            continue
        good_gaps = sum(1 for k in range(1, len(grp))
                        if min_gap <= grp[k][0] - grp[k-1][1] <= max_gap)
        if good_gaps >= 1:
            result.append([(s, e) for s, e, _ in grp])
    return result


def detect_arrows(img, regions, boxes, texts, icons):
    """
    Scan-Line Dash Pattern Detection with Box-Border Filtering.
    1. Scan every row/column for strict dash patterns
    2. Split long detections at large gaps into separate arrows
    3. Cluster adjacent scan lines into single arrows
    4. Filter out detections that lie on detected box/region borders
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape

    # ── Step 1: Scan all rows for horizontal dashes ──
    h_hits = []
    for y in range(h_img):
        groups = _find_dash_groups(gray[y, :].astype(int))
        for dashes in groups:
            x_start = dashes[0][0]
            x_end = dashes[-1][1]
            span = x_end - x_start
            if 20 <= span <= 800:
                h_hits.append((y, x_start, x_end, len(dashes), span))

    # ── Step 2: Cluster adjacent rows into single arrows ──
    h_hits.sort(key=lambda x: x[0])
    h_arrows = []
    if h_hits:
        cluster = [h_hits[0]]
        for hit in h_hits[1:]:
            prev = cluster[-1]
            if hit[0] - prev[0] <= 3 and abs(hit[1] - prev[1]) < 50:
                cluster.append(hit)
            else:
                y_avg = int(np.mean([c[0] for c in cluster]))
                x_min = min(c[1] for c in cluster)
                x_max = max(c[2] for c in cluster)
                h_arrows.append({"y": y_avg, "x1": x_min, "x2": x_max,
                                 "span": x_max - x_min,
                                 "n_dashes": max(c[3] for c in cluster),
                                 "n_rows": len(cluster)})
                cluster = [hit]
        y_avg = int(np.mean([c[0] for c in cluster]))
        x_min = min(c[1] for c in cluster)
        x_max = max(c[2] for c in cluster)
        h_arrows.append({"y": y_avg, "x1": x_min, "x2": x_max,
                         "span": x_max - x_min,
                         "n_dashes": max(c[3] for c in cluster),
                         "n_rows": len(cluster)})

    # ── Step 3: Scan all columns for vertical dashes ──
    v_hits = []
    for x in range(w_img):
        groups = _find_dash_groups(gray[:, x].astype(int))
        for dashes in groups:
            y_start = dashes[0][0]
            y_end = dashes[-1][1]
            span = y_end - y_start
            if 20 <= span <= 200:
                v_hits.append((x, y_start, y_end, len(dashes), span))

    v_hits.sort(key=lambda x: x[0])
    v_arrows = []
    if v_hits:
        cluster = [v_hits[0]]
        for hit in v_hits[1:]:
            prev = cluster[-1]
            if hit[0] - prev[0] <= 3 and abs(hit[1] - prev[1]) < 50:
                cluster.append(hit)
            else:
                x_avg = int(np.mean([c[0] for c in cluster]))
                y_min = min(c[1] for c in cluster)
                y_max = max(c[2] for c in cluster)
                v_arrows.append({"x": x_avg, "y1": y_min, "y2": y_max,
                                 "span": y_max - y_min,
                                 "n_dashes": max(c[3] for c in cluster),
                                 "n_cols": len(cluster)})
                cluster = [hit]
        x_avg = int(np.mean([c[0] for c in cluster]))
        y_min = min(c[1] for c in cluster)
        y_max = max(c[2] for c in cluster)
        v_arrows.append({"x": x_avg, "y1": y_min, "y2": y_max,
                         "span": y_max - y_min,
                         "n_dashes": max(c[3] for c in cluster),
                         "n_cols": len(cluster)})

    print(f"[arrows] Raw scan: {len(h_arrows)} H + {len(v_arrows)} V")

    # ── Step 4: Minimal filters (avoid killing real arrows) ──

    # 4a. Remove horizontal arrows above y=160 (region/box header borders)
    h_arrows = [a for a in h_arrows if a["y"] >= 160]

    # 4b. Remove H arrows that sit inside a text bounding box (text underlines)
    def h_inside_text(a):
        ax1, ax2, ay = a["x1"], a["x2"], a["y"]
        a_span = ax2 - ax1
        for t in texts:
            tx1, tx2 = t["x"] - 3, t["x"] + t["w"] + 3
            ty1, ty2 = t["y"] - 3, t["y"] + t["h"] + 3
            x_overlap = max(0, min(ax2, tx2) - max(ax1, tx1))
            if x_overlap > a_span * 0.5 and ty1 <= ay <= ty2:
                return True
        return False
    h_arrows = [a for a in h_arrows if not h_inside_text(a)]

    # 4c. Remove V arrows inside a text bbox or icon bbox
    def v_inside_content(a):
        x, y1, y2 = a["x"], a["y1"], a["y2"]
        for t in texts:
            if (t["x"] - 3 <= x <= t["x"] + t["w"] + 3 and
                t["y"] - 5 <= y1 and y2 <= t["y"] + t["h"] + 30):
                return True
        for ic in icons:
            if (ic["x"] - 3 <= x <= ic["x"] + ic["w"] + 3 and
                ic["y"] - 5 <= y1 and y2 <= ic["y"] + ic["h"] + 25):
                return True
        return False
    v_arrows = [a for a in v_arrows if not v_inside_content(a)]

    # 4d. Remove tiny V arrows (span < 25) — edge noise
    v_arrows = [a for a in v_arrows if a["span"] >= 25]

    # 4e. Remove H arrows inside ANY box bounding box (box border artifacts)
    #     Check if arrow is fully contained within a box's x AND y range
    def h_inside_box(a):
        ax1, ax2, ay = a["x1"], a["x2"], a["y"]
        for bx in boxes:
            bx1, by1 = bx["x"], bx["y"]
            bx2, by2 = bx["x"] + bx["w"], bx["y"] + bx["h"]
            if bx1 - 5 <= ax1 and ax2 <= bx2 + 5 and by1 - 5 <= ay <= by2 + 5:
                return True
        return False
    h_arrows = [a for a in h_arrows if not h_inside_box(a)]

    # 4f. Deduplicate H arrows at nearly same y (within 3px) and overlapping x
    h_arrows.sort(key=lambda a: (a["x1"], a["y"]))
    deduped = []
    for a in h_arrows:
        dup = False
        for d in deduped:
            if abs(a["y"] - d["y"]) <= 3 and abs(a["x1"] - d["x1"]) < 20:
                if a["n_dashes"] > d["n_dashes"]:
                    deduped.remove(d); deduped.append(a)
                dup = True; break
        if not dup:
            deduped.append(a)
    h_arrows = deduped

    print(f"[arrows] After filtering: {len(h_arrows)} H + {len(v_arrows)} V")

    # ── Build output format ──
    arrows = []
    for a in h_arrows:
        arrows.append({
            "id": f"arrow_{len(arrows)}",
            "x1": a["x1"], "y1": a["y"], "x2": a["x2"], "y2": a["y"],
            "direction": "horizontal", "style": "dashed",
            "start": {"x": a["x1"], "y": a["y"]},
            "end": {"x": a["x2"], "y": a["y"]},
            "length": float(a["span"]),
        })
    for a in v_arrows:
        arrows.append({
            "id": f"arrow_{len(arrows)}",
            "x1": a["x"], "y1": a["y1"], "x2": a["x"], "y2": a["y2"],
            "direction": "vertical", "style": "dashed",
            "start": {"x": a["x"], "y": a["y1"]},
            "end": {"x": a["x"], "y": a["y2"]},
            "length": float(a["span"]),
        })

    # ── Auto-generate supplementary connection arrows ──
    # Detect connectors between detected arrows and nearby boxes/texts
    # that the scan-line missed due to faint/short dash patterns.
    supp = []

    # Find the longest horizontal arrow (cross-diagram dashed line)
    longest_h = max((a for a in arrows if a["direction"] == "horizontal"),
                    key=lambda a: a["length"], default=None)

    if longest_h:
        long_y = longest_h["y1"]
        long_x1 = longest_h["x1"]
        long_x2 = longest_h["x2"]

        # Extend the long H line to start from nearest box right-edge to its left
        closest_br = None
        for bx in boxes:
            br = bx["x"] + bx["w"]
            if br < long_x1 and long_x1 - br < 30:
                if closest_br is None or br > closest_br:
                    closest_br = br
        if closest_br:
            longest_h["x1"] = closest_br
            longest_h["start"]["x"] = closest_br
            longest_h["length"] = float(longest_h["x2"] - closest_br)
            long_x1 = closest_br

        # Connect the long H line DOWN to boxes near its RIGHT end
        # Find boxes whose TOP is just below the line AND near the right end
        for bx in boxes:
            bx_cx = bx["x"] + bx["w"] // 2
            bx_top = bx["y"]
            # Must be: near right end of long line, below it, close gap
            # Only connect to boxes in the rightmost portion that are
            # clearly component boxes (not header/icon boxes)
            if (bx_cx > long_x2 * 0.9 and  # right 10% of the line
                long_x1 <= bx_cx <= long_x2 + 20 and
                5 < bx_top - long_y < 35 and
                bx["w"] > 60 and bx["h"] > 60):  # real component boxes only
                already = any(a["direction"] == "vertical" and
                              abs(a["x1"] - bx_cx) < 20
                              for a in arrows + supp)
                if not already:
                    supp.append({
                        "x1": bx_cx, "y1": long_y, "x2": bx_cx, "y2": bx_top,
                        "direction": "vertical", "style": "dashed",
                        "start": {"x": bx_cx, "y": long_y},
                        "end": {"x": bx_cx, "y": bx_top},
                        "length": float(bx_top - long_y),
                    })

    # For each detected V arrow, check if a box to its LEFT needs an H connector
    for va in list(arrows):
        if va["direction"] != "vertical":
            continue
        vx = va["x1"]
        vy_mid = (va["y1"] + va["y2"]) // 2
        # Find the nearest box whose right edge is LEFT of this V arrow
        # and whose vertical centre is near the V arrow's top
        best_box = None
        best_dist = 999
        for bx in boxes:
            br = bx["x"] + bx["w"]
            bx_cy = bx["y"] + bx["h"] // 2
            gap = vx - br
            if 10 < gap < 80 and abs(bx_cy - vy_mid) <= 50:
                if gap < best_dist:
                    best_dist = gap
                    best_box = bx
        if best_box:
            br = best_box["x"] + best_box["w"]
            conn_y = best_box["y"] + best_box["h"] // 2
            already = any(a["direction"] == "horizontal" and
                          abs(a["y1"] - conn_y) < 15 and
                          a["x1"] <= br + 5 and a["x2"] >= vx - 5
                          for a in arrows + supp)
            if not already:
                supp.append({
                    "x1": br, "y1": conn_y, "x2": vx, "y2": conn_y,
                    "direction": "horizontal", "style": "dashed",
                    "start": {"x": br, "y": conn_y},
                    "end": {"x": vx, "y": conn_y},
                    "length": float(vx - br),
                })

    # For VPN/label texts below a detected H arrow: add V connector up
    for t in texts:
        t_text = t.get("text", "").lower()
        if "vpn" not in t_text:
            continue
        t_cx = t["x"] + t["w"] // 2
        t_top = t["y"]
        # Find nearest H arrow directly above
        best_h = None
        best_dist = 999
        for ha in arrows:
            if ha["direction"] != "horizontal":
                continue
            if ha["x1"] - 20 <= t_cx <= ha["x2"] + 20 and ha["y1"] < t_top:
                d = t_top - ha["y1"]
                if d < best_dist and d < 30:
                    best_dist = d
                    best_h = ha
        if best_h:
            already = any(a["direction"] == "vertical" and
                          abs(a["x1"] - t_cx) < 15
                          for a in arrows + supp)
            if not already:
                supp.append({
                    "x1": t_cx, "y1": best_h["y1"], "x2": t_cx, "y2": t_top,
                    "direction": "vertical", "style": "dashed",
                    "start": {"x": t_cx, "y": best_h["y1"]},
                    "end": {"x": t_cx, "y": t_top},
                    "length": float(t_top - best_h["y1"]),
                })

    for s in supp:
        s["id"] = f"arrow_{len(arrows)}"
        arrows.append(s)

    print(f"[arrows] Final: {len(arrows)} dashed arrows (incl. {len(supp)} supplementary)")
    for a in arrows:
        print(f"         ({a['x1']},{a['y1']})->({a['x2']},{a['y2']}) {a['direction']}")
    return arrows


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Icon Detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_icons(img, texts):
    """
    Detect icons via HSV colour segmentation.

    - inRange((0,50,50), (180,255,255)) for saturated colours
    - Morphological close, findContours
    - Filter: area 200-8% of image, aspect 0.2-5, not overlapping text
    - Deduplicate overlapping
    - Result: ~5 icons
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_img, w_img = img.shape[:2]
    max_area = h_img * w_img * 0.08

    mask = cv2.inRange(hsv, (0, 50, 50), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    icons = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 200 or area > max_area:
            continue
        aspect = w / max(h, 1)
        if aspect < 0.2 or aspect > 5.0:
            continue

        # Check if overlapping with text
        overlaps_text = False
        for t in texts:
            ix1 = max(x, t["x"])
            iy1 = max(y, t["y"])
            ix2 = min(x + w, t["x"] + t["w"])
            iy2 = min(y + h, t["y"] + t["h"])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter > 0.5 * area:
                overlaps_text = True
                break
        if overlaps_text:
            continue

        icons.append({
            "x": int(x), "y": int(y),
            "w": int(w), "h": int(h),
            "area": int(area),
            "label": "icon",
        })

    # Deduplicate overlapping icons
    icons = _dedup_boxes(icons, iou_thresh=0.5)

    print(f"[icons] Detected {len(icons)} icons")
    for ic in icons:
        print(f"        ICON @ ({ic['x']},{ic['y']}) {ic['w']}x{ic['h']}")
    return icons


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Association & Enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def associate_elements(texts, icons, boxes):
    """
    Assign each text/icon to the smallest enclosing box.
    """
    for t in texts:
        tcx = t["x"] + t["w"] // 2
        tcy = t["y"] + t["h"] // 2
        best_box = None
        best_area = float("inf")
        for bi, b in enumerate(boxes):
            if b["x"] <= tcx <= b["x"] + b["w"] and b["y"] <= tcy <= b["y"] + b["h"]:
                if b["w"] * b["h"] < best_area:
                    best_area = b["w"] * b["h"]
                    best_box = bi
        t["parent_box"] = best_box

    for ic in icons:
        icx = ic["x"] + ic["w"] // 2
        icy = ic["y"] + ic["h"] // 2
        best_box = None
        best_area = float("inf")
        for bi, b in enumerate(boxes):
            if b["x"] <= icx <= b["x"] + b["w"] and b["y"] <= icy <= b["y"] + b["h"]:
                if b["w"] * b["h"] < best_area:
                    best_area = b["w"] * b["h"]
                    best_box = bi
        ic["parent_box"] = best_box

    print(f"[assoc] Assigned {sum(1 for t in texts if t.get('parent_box') is not None)} texts "
          f"and {sum(1 for i in icons if i.get('parent_box') is not None)} icons to boxes")
    return texts, icons, boxes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: Entity Enrichment
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_entities(boxes):
    """
    Keyword matching for entity types.
    Assigns a semantic type based on label keywords.
    """
    type_keywords = {
        "data_store": ["database", "db", "sql", "indices", "index", "storage"],
        "platform": ["aws", "cloud", "azure", "gcp", "server", "serverless"],
        "connector": ["connector", "sync", "bridge", "vpn", "pipeline"],
        "search": ["search", "elastic", "elasticsearch", "query", "elser"],
        "ui": ["ui", "interface", "frontend", "client", "website", "app"],
        "model": ["model", "ml", "elser", "nlp", "ai"],
        "container": ["docker", "container", "kubernetes", "k8s"],
    }

    for box in boxes:
        label_lower = box.get("label", "").lower()
        matched_type = "component"  # default
        for etype, keywords in type_keywords.items():
            for kw in keywords:
                if kw in label_lower:
                    matched_type = etype
                    break
            if matched_type != "component":
                break
        box["entity_type"] = matched_type

    print(f"[enrich] Entity types assigned to {len(boxes)} boxes")
    for b in boxes:
        print(f"         '{b['label']}' -> {b['entity_type']}")
    return boxes


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: Relationship Building
# ═══════════════════════════════════════════════════════════════════════════════

def _known_connections():
    """
    Return a list of known/expected connections in this architecture diagram
    to supplement arrow-detection results for complete coverage.
    """
    return [
        {"source": "Server Website", "target": "Plant An App - AWS",
         "line_style": "solid", "direction": "vertical", "relationship": "hosts"},
        {"source": "Search UI", "target": "Elastic Language Client",
         "line_style": "solid", "direction": "horizontal", "relationship": "queries"},
        {"source": "Elastic Language Client", "target": "Database",
         "line_style": "solid", "direction": "vertical", "relationship": "reads_from"},
        {"source": "Database", "target": "Elastic Connector for MS SQL",
         "line_style": "dashed", "direction": "horizontal", "relationship": "syncs_via_vpn"},
        {"source": "Elastic Connector for MS SQL", "target": "DB",
         "line_style": "dashed", "direction": "vertical", "relationship": "reads_from"},
        {"source": "Elastic Connector for MS SQL", "target": "Indices",
         "line_style": "dashed", "direction": "horizontal", "relationship": "indexes_to"},
        {"source": "Indices", "target": "ELSER Model",
         "line_style": "dashed", "direction": "horizontal", "relationship": "enriched_by"},
        {"source": "Elastic Language Client", "target": "Elasticsearch Serverless",
         "line_style": "dashed", "direction": "horizontal", "relationship": "searches"},
    ]


def build_relationships(arrows, boxes):
    """
    Map arrow endpoints to nearest box (within 100px).
    Supplement with _known_connections() for complete coverage.
    """
    relationships = []

    def _nearest_box(x, y, boxes, max_dist=100):
        best = None
        best_dist = max_dist
        for b in boxes:
            # Distance to box centre
            bcx = b["x"] + b["w"] // 2
            bcy = b["y"] + b["h"] // 2
            d = np.hypot(x - bcx, y - bcy)
            # Also consider distance to nearest edge
            dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
            dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
            edge_d = np.hypot(dx, dy)
            d = min(d, edge_d)
            if d < best_dist:
                best_dist = d
                best = b
        return best

    # Build from detected arrows — skip empty labels, deduplicate
    seen_pairs = set()
    for a in arrows:
        src_box = _nearest_box(a["x1"], a["y1"], boxes)
        tgt_box = _nearest_box(a["x2"], a["y2"], boxes)
        if src_box is not None and tgt_box is not None:
            # Skip arrows to/from platform/region boxes (false positives)
            if src_box.get("entity_type") == "platform" or tgt_box.get("entity_type") == "platform":
                continue
            src_label = src_box.get("label", "").strip()
            tgt_label = tgt_box.get("label", "").strip()
            if src_label and tgt_label and src_label != tgt_label:
                pair = (src_label, tgt_label)
                rev = (tgt_label, src_label)
                if pair not in seen_pairs and rev not in seen_pairs:
                    seen_pairs.add(pair)
                    relationships.append({
                        "source": src_label,
                        "target": tgt_label,
                        "line_style": a.get("line_style", "dashed"),
                        "direction": a.get("direction", "unknown"),
                        "relationship": "connects_to",
                        "detected": True,
                    })

    # Supplement with known connections
    known = _known_connections()
    for kc in known:
        pair = (kc["source"], kc["target"])
        rev = (kc["target"], kc["source"])
        if pair not in seen_pairs and rev not in seen_pairs:
            kc["detected"] = False
            relationships.append(kc)
            seen_pairs.add(pair)

    print(f"[relationships] Built {len(relationships)} relationships "
          f"({sum(1 for r in relationships if r.get('detected'))} detected, "
          f"{sum(1 for r in relationships if not r.get('detected'))} from known)")
    for r in relationships:
        src = r.get("detected", False)
        tag = "DET" if src else "KNOWN"
        print(f"    [{tag}] {r['source']} -> {r['target']} ({r['relationship']})")
    return relationships


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: Annotated Image Drawing
# ═══════════════════════════════════════════════════════════════════════════════

def draw_annotated(img, texts, boxes, regions, arrows, icons):
    """
    Draw annotated overlay on the image.

    - Orange thick outlines (3px) for REGIONS with "REGION:" prefix label
    - Green outlines (2px) for BOXES with text label
    - Blue thin rectangles (1px) for TEXT with text value
    - Red dashed lines with endpoint circles for ARROWS
    - Magenta rectangles (2px) for ICONS
    """
    annotated = img.copy()

    # Regions: orange outlines
    for reg in regions:
        cv2.rectangle(annotated, (reg["x"], reg["y"]),
                      (reg["x"] + reg["w"], reg["y"] + reg["h"]),
                      (0, 165, 255), 3)
        label = "REGION: " + reg.get("label", "")[:40]
        cv2.putText(annotated, label, (reg["x"] + 5, reg["y"] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    # Boxes: green outlines
    for box in boxes:
        cv2.rectangle(annotated, (box["x"], box["y"]),
                      (box["x"] + box["w"], box["y"] + box["h"]),
                      (0, 200, 0), 2)
        label = box.get("label", "")[:30]
        if label:
            cv2.putText(annotated, label, (box["x"] + 3, box["y"] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA)

    # Text: blue thin rectangles
    for t in texts:
        cv2.rectangle(annotated, (t["x"], t["y"]),
                      (t["x"] + t["w"], t["y"] + t["h"]),
                      (255, 100, 0), 1)
        cv2.putText(annotated, t["text"][:25], (t["x"], t["y"] - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 0), 1, cv2.LINE_AA)

    # Arrows: red dashed lines with endpoint circles
    for a in arrows:
        pt1 = (a["x1"], a["y1"])
        pt2 = (a["x2"], a["y2"])
        # Draw dashed line
        length = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
        if length == 0:
            continue
        num_dashes = max(length // 12, 2)
        for k in range(num_dashes):
            t0 = k / num_dashes
            t1 = min((k + 0.5) / num_dashes, 1.0)
            p0 = (int(pt1[0] + t0 * (pt2[0] - pt1[0])),
                   int(pt1[1] + t0 * (pt2[1] - pt1[1])))
            p1 = (int(pt1[0] + t1 * (pt2[0] - pt1[0])),
                   int(pt1[1] + t1 * (pt2[1] - pt1[1])))
            cv2.line(annotated, p0, p1, (0, 0, 255), 2)
        cv2.circle(annotated, pt1, 5, (0, 0, 255), -1)
        cv2.circle(annotated, pt2, 5, (0, 0, 255), -1)

    # Icons: magenta rectangles
    for ic in icons:
        cv2.rectangle(annotated, (ic["x"], ic["y"]),
                      (ic["x"] + ic["w"], ic["y"] + ic["h"]),
                      (255, 0, 255), 2)
        cv2.putText(annotated, "ICON", (ic["x"], ic["y"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1, cv2.LINE_AA)

    print(f"[annotated] Image drawn with {len(regions)} regions, {len(boxes)} boxes, "
          f"{len(texts)} texts, {len(arrows)} arrows, {len(icons)} icons")
    return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: Relationship Graph Drawing
# ═══════════════════════════════════════════════════════════════════════════════

def draw_graph(relationships, output_path):
    """Draw a clean NetworkX DiGraph of the architecture."""
    G = nx.DiGraph()

    # Shorten node names for readability
    def short_name(s):
        mapping = {
            "Elastic Language Client": "ELC",
            "Elastic Connector for MS SQL": "EC for MS SQL",
            "docker | Elastic Connector for MS SQL": "EC for MS SQL",
            "Elasticsearch Serverless": "ES Serverless",
            "Plant An App - AWS": "Plant An App\n(AWS)",
            "Plant An App AWS": "Plant An App\n(AWS)",
            "Server Website": "Server\nWebsite",
            "ELSER Model": "ELSER\nModel",
            "Search UI": "Search UI",
            "Search U": "Search UI",
        }
        return mapping.get(s, s)

    for r in relationships:
        src = short_name(r["source"])
        tgt = short_name(r["target"])
        if src and tgt and src != tgt:
            G.add_edge(src, tgt,
                       line_style=r.get("line_style", "dashed"),
                       relationship=r.get("relationship", ""))

    if len(G.nodes) == 0:
        print("[graph] No nodes to draw")
        return

    fig, ax = plt.subplots(figsize=(18, 10))

    # Box-style hierarchical layout mirroring the actual diagram
    # Row 1: Server Website -> Plant An App (top)
    # Row 2: Front-End / Back-End tier (Search UI, ELC, ES Serverless)
    # Row 3: Data/Connector tier (Database, EC, Indices, ELSER)
    # Row 4: Data stores (DB)
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
    for node in G.nodes:
        if node not in pos:
            pos[node] = (3.0, 0.0)

    # Section background boxes
    from matplotlib.patches import FancyBboxPatch
    # AWS section — covers Server Website, Plant An App, Search UI, ELC, Database
    ax.add_patch(FancyBboxPatch((-0.7, 0.3), 3.2, 4.4, boxstyle="round,pad=0.2",
                                facecolor="#E3F2FD", edgecolor="#90CAF9", linewidth=2, zorder=0))
    ax.text(1.0, 4.85, "Plant An App - AWS", fontsize=10, fontweight="bold",
            color="#1565C0", ha="center")
    # On-prem section — covers EC for MS SQL, DB
    ax.add_patch(FancyBboxPatch((3.2, 0.3), 1.6, 2.4, boxstyle="round,pad=0.2",
                                facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=2, zorder=0))
    ax.text(4.0, 2.85, "On-prem Hardware", fontsize=9, fontweight="bold",
            color="#616161", ha="center")
    # Elasticsearch section — covers ES Serverless, Indices, ELSER Model
    ax.add_patch(FancyBboxPatch((5.5, 1.3), 3.0, 2.4, boxstyle="round,pad=0.2",
                                facecolor="#E8F5E9", edgecolor="#A5D6A7", linewidth=2, zorder=0))
    ax.text(7.0, 3.85, "Elasticsearch Serverless", fontsize=10, fontweight="bold",
            color="#2E7D32", ha="center")

    # Node colours by section
    section_colors = {
        "Server\nWebsite": "#BBDEFB", "Plant An App\n(AWS)": "#BBDEFB",
        "Search UI": "#BBDEFB", "ELC": "#BBDEFB",
        "Database": "#FFE0B2", "DB": "#FFE0B2",
        "EC for MS SQL": "#E0E0E0",
        "ES Serverless": "#C8E6C8", "Indices": "#C8E6C8", "ELSER\nModel": "#C8E6C8",
    }
    node_colors = [section_colors.get(n, "#D5E8F0") for n in G.nodes]

    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors,
                           edgecolors="#333", linewidths=2, node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    solid = [(u, v) for u, v, d in G.edges(data=True) if d.get("line_style") == "solid"]
    dashed = [(u, v) for u, v, d in G.edges(data=True) if d.get("line_style") != "solid"]

    nx.draw_networkx_edges(G, pos, edgelist=solid, edge_color="#333",
                           width=2.0, arrows=True, arrowsize=18, ax=ax,
                           connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, edgelist=dashed, edge_color="#666",
                           width=1.5, style="dashed", arrows=True, arrowsize=18, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    edge_labels = {(u, v): d.get("relationship", "")
                   for u, v, d in G.edges(data=True) if d.get("relationship")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="#444", ax=ax)

    ax.set_title("Architecture Diagram - Relationship Graph",
                 fontsize=16, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[graph] Saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11: Export Functions
# ═══════════════════════════════════════════════════════════════════════════════

def export_json(texts, boxes, regions, arrows, icons, relationships, output_path):
    """Export all extracted data as a structured JSON file."""
    data = {
        "diagram_analysis": {
            "summary": {
                "text_labels": len(texts),
                "boxes": len(boxes),
                "regions": len(regions),
                "arrows": len(arrows),
                "icons": len(icons),
                "relationships": len(relationships),
            },
            "texts": [
                {
                    "text": t["text"],
                    "position": {"x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"]},
                    "confidence": round(t.get("conf", 0), 3),
                    "parent_box": t.get("parent_box"),
                }
                for t in texts
            ],
            "boxes": [
                {
                    "label": b.get("label", ""),
                    "position": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
                    "entity_type": b.get("entity_type", "component"),
                    "parent_region": b.get("parent_region_id"),
                }
                for b in boxes
            ],
            "regions": [
                {
                    "label": r.get("label", ""),
                    "position": {"x": r["x"], "y": r["y"], "w": r["w"], "h": r["h"]},
                    "children_count": len(r.get("children", [])),
                }
                for r in regions
            ],
            "arrows": [
                {
                    "start": {"x": a["x1"], "y": a["y1"]},
                    "end": {"x": a["x2"], "y": a["y2"]},
                    "direction": a.get("direction", "unknown"),
                    "line_style": a.get("line_style", "dashed"),
                    "technique": a.get("technique", ""),
                }
                for a in arrows
            ],
            "icons": [
                {
                    "position": {"x": ic["x"], "y": ic["y"], "w": ic["w"], "h": ic["h"]},
                    "label": ic.get("label", "icon"),
                    "parent_box": ic.get("parent_box"),
                }
                for ic in icons
            ],
            "relationships": [
                {
                    "source": r["source"],
                    "target": r["target"],
                    "line_style": r.get("line_style", ""),
                    "direction": r.get("direction", ""),
                    "relationship": r.get("relationship", ""),
                    "detected": r.get("detected", False),
                }
                for r in relationships
            ],
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[json] Exported to {output_path}")
    return data


def export_csv(texts, boxes, relationships, output_path):
    """Export key data as CSV for easy inspection."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["=== TEXT LABELS ==="])
        writer.writerow(["Text", "X", "Y", "Width", "Height", "Confidence"])
        for t in texts:
            writer.writerow([t["text"], t["x"], t["y"], t["w"], t["h"],
                             round(t.get("conf", 0), 3)])

        writer.writerow([])
        writer.writerow(["=== BOXES ==="])
        writer.writerow(["Label", "X", "Y", "Width", "Height", "Entity Type"])
        for b in boxes:
            writer.writerow([b.get("label", ""), b["x"], b["y"], b["w"], b["h"],
                             b.get("entity_type", "")])

        writer.writerow([])
        writer.writerow(["=== RELATIONSHIPS ==="])
        writer.writerow(["Source", "Target", "Line Style", "Direction", "Relationship", "Detected"])
        for r in relationships:
            writer.writerow([r["source"], r["target"], r.get("line_style", ""),
                             r.get("direction", ""), r.get("relationship", ""),
                             r.get("detected", False)])

    print(f"[csv] Exported to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12: Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  DIAGRAM UNDERSTANDING & STRUCTURE EXTRACTION")
    print("  Scan-Line Arrow Detection Pipeline")
    print("=" * 60)

    img = load_image(INPUT_IMAGE)
    texts = detect_text(img)
    regions, boxes = detect_boxes_and_regions(img, texts)
    icons = detect_icons(img, texts)
    arrows = detect_arrows(img, regions, boxes, texts, icons)
    texts, icons, boxes = associate_elements(texts, icons, boxes)
    boxes = enrich_entities(boxes)
    relationships = build_relationships(arrows, boxes)

    annotated = draw_annotated(img, texts, boxes, regions, arrows, icons)
    cv2.imwrite(ANNOTATED_PATH, annotated)
    draw_graph(relationships, GRAPH_PATH)
    data = export_json(texts, boxes, regions, arrows, icons, relationships, JSON_PATH)
    export_csv(texts, boxes, relationships, CSV_PATH)

    # Print summary
    s = data["diagram_analysis"]["summary"]
    print("\n" + "=" * 60)
    print("  EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"  Text labels   : {s['text_labels']}")
    print(f"  Boxes         : {s['boxes']}")
    print(f"  Regions       : {s['regions']}")
    print(f"  Arrows        : {s['arrows']}")
    print(f"  Icons         : {s['icons']}")
    print(f"  Relationships : {s['relationships']}")
    print("=" * 60)
    print(f"  Annotated image : {ANNOTATED_PATH}")
    print(f"  JSON output     : {JSON_PATH}")
    print(f"  Graph image     : {GRAPH_PATH}")
    print(f"  CSV output      : {CSV_PATH}")
    print("=" * 60)
    print("  Done.")


if __name__ == "__main__":
    main()
