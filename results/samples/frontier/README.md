# Frontier-comparison sample outputs

This folder holds representative paired outputs for the Day-6 Phase-5 frontier
comparison.

- `diagramine_diagram_02_structure.json` — actual machine-parseable JSON the
  DiagraMine pipeline emits for `diagram_02.png`. Schema-valid by construction:
  every field is a typed Pydantic model.
- `claude_vision_diagram_02_PROJECTED.txt` — illustrative LLM response. In the
  Day-6 autonomous run the live Claude Vision call was blocked by a missing
  `ANTHROPIC_API_KEY` and we used the SKILL-cited prior (~13% schema-valid). The
  text in this file is a representative *example* of the prose-wrapped JSON that
  causes vision LLMs to fail strict parsing — not a real captured response. The
  benchmark harness (`benchmark_claude_vision.py`) at the repo root captures
  real responses when run with credentials.

When the harness is re-run with a valid key, this folder will be regenerated
with actual paired outputs.
