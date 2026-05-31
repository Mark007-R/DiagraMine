"""DiagraMine Streamlit demo.

Upload an architecture diagram → see annotated overlay, structured JSON, and
the relationship graph side-by-side. The point: show that the same upload
returns the same schema every call (the 1.000 schema-valid-JSON-rate story).

Run:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import time

import cv2
import streamlit as st
from PIL import Image

from src import pipeline as pl
from src.graph import builder
from src.schemas import PipelineConfig

st.set_page_config(page_title="DiagraMine", page_icon=":triangular_ruler:", layout="wide")

st.title("DiagraMine — Diagram Structure Extraction")
st.caption(
    "Upload an architecture diagram. The pipeline returns schema-valid JSON, "
    "an annotated overlay, and a data-driven relationship graph. Compared to "
    "vision LLMs (which return strict-parseable JSON ~13% of the time under "
    "structured prompting), DiagraMine returns it 100% of the time by "
    "construction — every output field is a typed Pydantic model."
)

# ── Sidebar config ───────────────────────────────────────────────────────────
st.sidebar.header("Pipeline configuration")
text_detector = st.sidebar.selectbox("Text detector", ["easyocr", "paddleocr"], index=0)
box_detector = st.sidebar.selectbox("Box detector", ["canny_contours", "hough", "yolo"], index=0)
arrow_detector = st.sidebar.selectbox("Arrow detector", ["hough_lines", "pixel_scan", "cnn"], index=0)
icon_detector = st.sidebar.selectbox("Icon detector", ["template_matching", "clip", "hsv"], index=0)
outside_box_gate = st.sidebar.checkbox(
    "Outside-box gate (Day-3 fix)", value=True,
    help="Rejects arrow segments whose midpoint sits inside a box (border artifacts). "
         "Lifts real-diagram rel-F1 0.545→0.889. Costs ~0.077 on clean synthetic diagrams.",
)
graph_layout = st.sidebar.selectbox("Graph layout", ["kamada_kawai", "spring"], index=0)
st.sidebar.markdown(
    "**Champion config (defaults)** — picked by Day-3 Phase-2 leaderboard, "
    "tuned Day-5. EasyOCR + Canny+contours + Hough lines + template-matching."
)

# ── Upload ───────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an architecture diagram (PNG, JPG, WebP)",
    type=["png", "jpg", "jpeg", "webp"],
)

if uploaded is None:
    st.info(
        "Tip: try one of the 15 benchmark diagrams in `data/eval/diagrams_15/`. "
        "diagram_01.png is the simplest (3-tier web app, 4 boxes, 3 arrows); "
        "search_interview_test.png is the original interview diagram."
    )
    st.stop()

# Save upload to a temp file for the pipeline.
suffix = os.path.splitext(uploaded.name)[1] or ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

config = PipelineConfig(
    text_detector=text_detector, box_detector=box_detector,
    arrow_detector=arrow_detector, icon_detector=icon_detector,
    outside_box_gate=outside_box_gate, graph_layout=graph_layout,
)

with st.spinner("Extracting structure (EasyOCR weights download on first run)..."):
    t0 = time.perf_counter()
    result = pl.extract(tmp_path, config)
    annotated = pl.annotate(tmp_path, result)
    rt_total = time.perf_counter() - t0

# Render the relationship graph to a temp PNG.
graph_path = os.path.join(tempfile.gettempdir(), f"diagramine_graph_{os.getpid()}.png")
graph_ok = builder.draw_graph(
    [r.model_dump() for r in result.relationships], graph_path, layout=graph_layout
)

# ── Metrics row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Texts", len(result.texts))
c2.metric("Boxes", len(result.boxes))
c3.metric("Arrows", len(result.arrows))
c4.metric("Relationships", len(result.relationships))
c5.metric("Runtime", f"{rt_total:.2f}s")

# Schema-valid badge — the headline reliability claim.
st.success(
    f"Schema-valid JSON: ✓ (every field validated by Pydantic v2). "
    f"`result.model_dump()` round-trips through `json.dumps` without loss."
)

# ── Two-column visual ────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Annotated overlay")
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(annotated_rgb, use_container_width=True,
             caption="Green = boxes, blue = text, red = arrows, magenta = icons, orange = regions")

with right:
    st.subheader("Relationship graph (data-driven layout)")
    if graph_ok and os.path.exists(graph_path):
        st.image(graph_path, use_container_width=True)
    else:
        st.warning("No relationships detected — the data-driven graph is empty. "
                   "(Previously the legacy pipeline would have injected 8 hardcoded "
                   "ELSER/Plant-An-App edges here; that was removed on Day 4.)")

# ── JSON / CSV downloads ────────────────────────────────────────────────────
st.subheader("Structured JSON output")
structured = result.model_dump()
st.json(structured, expanded=False)

dl1, dl2 = st.columns(2)
dl1.download_button(
    "Download structure.json",
    data=json.dumps(structured, indent=2, ensure_ascii=False),
    file_name=f"{os.path.splitext(uploaded.name)[0]}_structure.json",
    mime="application/json",
)
import csv as _csv
csv_buf = io.StringIO()
w = _csv.writer(csv_buf)
w.writerow(["source", "target", "line_style", "direction", "relationship"])
for r in result.relationships:
    w.writerow([r.source, r.target, r.line_style, r.direction, r.relationship])
dl2.download_button(
    "Download relationships.csv",
    data=csv_buf.getvalue(),
    file_name=f"{os.path.splitext(uploaded.name)[0]}_relationships.csv",
    mime="text/csv",
)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "DiagraMine v4.0 · modular pipeline (Day-4 refactor) · "
    "data-driven relationships (no `_known_connections` hardcoding) · "
    "schema-valid by construction (typed Pydantic models)"
)

try:
    os.unlink(tmp_path)
except OSError:
    pass
