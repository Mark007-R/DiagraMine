# Diagram-Structure-Extractor

> 🔗 **Live API:** https://iambatman07-diagram-structure-extractor.hf.space — try `/health` and `/extract` · [HF Space](https://huggingface.co/spaces/IamBatman07/Diagram-Structure-Extractor)

Takes an architecture diagram image and returns its structure as typed, schema-valid JSON: components, arrows, icons, and the relationships between them. Outputs are Pydantic v2 models, so every call returns machine-parseable structure.

Available as a CLI, a FastAPI service, and a Streamlit demo.

---

## What's in the box

| Output | Format | Contents |
|---|---|---|
| `extracted_structure.json` | JSON | Typed `ExtractionResult` — image / width / height / detectors / errors / texts[] / boxes[] / regions[] / arrows[] / icons[] / relationships[] / runtimes |
| `annotated_diagram.png` | PNG | Original diagram with overlays — green = boxes, blue = text, red = arrows, magenta = icons, orange = regions |
| `relationship_graph.png` | PNG | `networkx` DiGraph render with kamada_kawai layout |
| `extracted_data.csv` | CSV | Flat (source, target, line_style, direction, relationship) edge table |

---

## Quick start

```bash
# 1) Local install
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2) Run on the bundled test diagram (or any path)
python diagram_analysis.py                                   # uses search_interview_test.png
python diagram_analysis.py path/to/your/diagram.png

# 3) Use the modular orchestrator directly with detector overrides
python -m src.pipeline data/eval/diagrams_15/diagram_02.png \
       --out /tmp/out --text paddleocr --arrow hough_lines

# 4) Run the FastAPI service
uvicorn src.api:app --port 8000
curl -F "file=@diagram.png" http://localhost:8000/extract | jq .

# 5) Run the Streamlit demo (upload an image, see annotated + graph + JSON)
streamlit run app.py

# 6) Tests
pytest tests/ -q                                              # 25 tests, ~30 s

# 7) Docker
docker build -t diagramine:4.0 .
docker run -p 8000:8000 diagramine:4.0
```

Tesseract is wired in as an optional text detector but requires the native binary installed separately.

---

## API

`POST /extract` (multipart, `file=@image.png`)

Optional query params: `?text=paddleocr&box=canny_contours&arrow=hough_lines&icon=template_matching&outside_box_gate=true`.

Response:

```json
{
  "structure": { "...ExtractionResult (typed Pydantic)..." },
  "annotated_png_base64": "iVBORw0KG...",
  "relationship_csv": "source,target,line_style,direction,relationship\n..."
}
```

`GET /health` → `{"status": "ok", "service": "diagramine", "version": "4.0"}`

---

## License

MIT. See `LICENSE`.
