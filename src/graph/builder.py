"""Data-driven graph construction.

This module is the heart of the Day-4 de-hardcoding. The legacy monolith built
its relationship graph by *merging in* `_known_connections()` — 8 edges hand-
coded for the "Plant An App - AWS" diagram — so the graph looked complete even
when arrow detection found nothing. That baked the answer key into the code.

Here, EVERY relationship is derived from a detected arrow:
  1. `label_boxes`   — assign each box a label from the text contained in it.
  2. `build_relationships` — map each arrow's endpoints to the nearest boxes,
     gated so segments lying inside/on a box border (the Day-3 snap-to-adjacency
     artifact) are rejected.
  3. `draw_graph`    — render with a general `kamada_kawai` / `spring` layout.
     No hardcoded node coordinates.
"""
from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def label_boxes(boxes: List[dict], texts: List[dict]) -> List[dict]:
    """Assign each box a text label from the OCR fragments whose centre falls
    inside it (smallest enclosing box wins, mirroring the legacy
    associate_elements). Fragments are joined in reading order (top-to-bottom,
    left-to-right). Boxes that contain no text keep an empty label."""
    contained: dict = {i: [] for i in range(len(boxes))}
    for t in texts:
        tcx = t["x"] + t["w"] // 2
        tcy = t["y"] + t["h"] // 2
        best_box = None
        best_area = float("inf")
        for bi, b in enumerate(boxes):
            if b["x"] <= tcx <= b["x"] + b["w"] and b["y"] <= tcy <= b["y"] + b["h"]:
                area = b["w"] * b["h"]
                if area < best_area:
                    best_area = area
                    best_box = bi
        if best_box is not None:
            contained[best_box].append(t)

    for bi, b in enumerate(boxes):
        frags = sorted(contained[bi], key=lambda t: (t["y"], t["x"]))
        label = " ".join(f["text"].strip() for f in frags if f.get("text", "").strip())
        b["label"] = label.strip()
    return boxes


def _point_in_box(x: int, y: int, b: dict) -> bool:
    return b["x"] <= x <= b["x"] + b["w"] and b["y"] <= y <= b["y"] + b["h"]


def _nearest_box(x: int, y: int, boxes: List[dict], max_dist: float = 100.0) -> Optional[int]:
    """Index of the box nearest to (x, y) within max_dist, by min(centre, edge)
    distance. Returns None if nothing is close enough."""
    best_i = None
    best_d = max_dist
    for i, b in enumerate(boxes):
        bcx = b["x"] + b["w"] / 2
        bcy = b["y"] + b["h"] / 2
        d_centre = float(np.hypot(x - bcx, y - bcy))
        dx = max(b["x"] - x, 0, x - (b["x"] + b["w"]))
        dy = max(b["y"] - y, 0, y - (b["y"] + b["h"]))
        d_edge = float(np.hypot(dx, dy))
        d = min(d_centre, d_edge)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _midpoint_inside_any_box(arrow: dict, boxes: List[dict]) -> bool:
    """The Day-3 'outside any box bbox' gate. A genuine connector runs through
    whitespace BETWEEN boxes, so its midpoint is not inside a box. A box-border
    / internal-edge segment (the snap-to-adjacency artifact that inflated CNN-
    verified recall on Day 3) has its midpoint on or inside a box bbox."""
    mx = (arrow["x1"] + arrow["x2"]) // 2
    my = (arrow["y1"] + arrow["y2"]) // 2
    return any(_point_in_box(mx, my, b) for b in boxes)


def build_relationships(
    arrows: List[dict],
    boxes: List[dict],
    outside_box_gate: bool = True,
    max_dist: float = 120.0,
) -> List[dict]:
    """Build relationships purely from detected arrows. No hardcoded edges.

    For each arrow, snap each endpoint to the nearest labelled box and emit an
    unordered, de-duplicated (source -> target) edge. With `outside_box_gate`,
    arrows whose midpoint lies inside/on a box are dropped (border artifacts).
    """
    # Endpoints map to component boxes (skip region/container boxes and
    # platform background boxes — they swallow endpoints and create spurious
    # parent->child edges).
    candidates = [
        b for b in boxes
        if not b.get("is_region") and b.get("entity_type") != "platform"
    ]
    if len(candidates) < 2:
        candidates = [b for b in boxes if not b.get("is_region")]
    if len(candidates) < 2:
        candidates = boxes

    relationships: List[dict] = []
    seen = set()
    for a in arrows:
        if outside_box_gate and _midpoint_inside_any_box(a, candidates):
            continue
        si = _nearest_box(a["x1"], a["y1"], candidates, max_dist)
        ti = _nearest_box(a["x2"], a["y2"], candidates, max_dist)
        if si is None or ti is None or si == ti:
            continue
        src = candidates[si].get("label", "").strip()
        tgt = candidates[ti].get("label", "").strip()
        if not src or not tgt or src == tgt:
            continue
        pair = (src, tgt)
        if pair in seen or (tgt, src) in seen:
            continue
        seen.add(pair)
        relationships.append({
            "source": src,
            "target": tgt,
            "line_style": a.get("line_style", "solid"),
            "direction": a.get("direction", "unknown"),
            "relationship": "connects_to",
            "detected": True,
        })
    return relationships


def build_graph(relationships: List[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for r in relationships:
        src, tgt = (r.get("source") or "").strip(), (r.get("target") or "").strip()
        if src and tgt and src != tgt:
            G.add_edge(src, tgt,
                       line_style=r.get("line_style", "solid"),
                       relationship=r.get("relationship", ""))
    return G


def _short_label(s: str, max_chars: int = 18) -> str:
    s = (s or "").strip()
    if len(s) <= max_chars:
        return s
    words, line1, line2 = s.split(), "", ""
    for w in words:
        if len(line1) + len(w) + 1 <= max_chars or not line1:
            line1 = (line1 + " " + w).strip()
        else:
            line2 = (line2 + " " + w).strip()
    return line1 + ("\n" + line2 if line2 else "")


def draw_graph(relationships: List[dict], output_path: str, layout: str = "kamada_kawai") -> bool:
    """Render the relationship DiGraph with a general, data-driven layout.

    No hardcoded `pos = {...}`: node positions come from the graph structure, so
    this works on any diagram. Returns False if there is nothing to draw."""
    G = nx.DiGraph()
    for r in relationships:
        src = _short_label(r.get("source", ""))
        tgt = _short_label(r.get("target", ""))
        if src and tgt and src != tgt:
            G.add_edge(src, tgt,
                       line_style=r.get("line_style", "solid"),
                       relationship=r.get("relationship", ""))
    if len(G.nodes) == 0:
        return False

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42, k=1.2)
    else:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=1.2)

    fig, ax = plt.subplots(figsize=(16, 9))
    nx.draw_networkx_nodes(G, pos, node_size=3600, node_color="#D5E8F0",
                           edgecolors="#333", linewidths=2, node_shape="s", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold", ax=ax)

    solid = [(u, v) for u, v, d in G.edges(data=True) if d.get("line_style") == "solid"]
    dashed = [(u, v) for u, v, d in G.edges(data=True) if d.get("line_style") != "solid"]
    nx.draw_networkx_edges(G, pos, edgelist=solid, edge_color="#333", width=2.0,
                           arrows=True, arrowsize=18, ax=ax,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edges(G, pos, edgelist=dashed, edge_color="#666", width=1.5,
                           style="dashed", arrows=True, arrowsize=18, ax=ax,
                           connectionstyle="arc3,rad=0.08")
    edge_labels = {(u, v): d.get("relationship", "")
                   for u, v, d in G.edges(data=True) if d.get("relationship")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=7, font_color="#444", ax=ax)
    ax.set_title("Relationship Graph (data-driven)", fontsize=15, fontweight="bold", pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True
