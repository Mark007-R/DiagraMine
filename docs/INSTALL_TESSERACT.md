# Installing Tesseract on Windows for Phase 2a benchmark

The Phase 2a OCR comparison benchmarks EasyOCR, PaddleOCR, **and Tesseract**.
EasyOCR + PaddleOCR are pip-installed by `requirements.txt`. Tesseract requires
a separate binary install because its Python binding (`pytesseract`) is a thin
wrapper that shells out to the native `tesseract.exe`.

## Status on the original Day-2 run host

Tesseract was skipped on the original Day-2 run. The host had `pytesseract`
installed but no `tesseract.exe` binary, and every install path attempted
required administrator elevation that was not available in the automated run
environment:

| Method                                              | Result                                       |
|-----------------------------------------------------|----------------------------------------------|
| `winget install UB-Mannheim.TesseractOCR --scope user` | "No applicable installer found"           |
| `winget install tesseract-ocr.tesseract`            | UAC prompt, cancelled                        |
| NSIS `setup.exe /S /D=<user-dir>` direct            | "The requested operation requires elevation" |
| `Start-Process` from PowerShell                     | Same elevation error                         |
| 7-Zip extraction of installer payload               | 7-Zip not present                            |

`src/text_detection/tesseract_detector.py` graceful-degrades when the binary
is missing — it returns `{"texts": [], "skipped": True, "reason": "..."}` so
the benchmark continues with the available detectors.

## Recommended install path (Windows, admin available)

1. Download the official installer from UB-Mannheim:
   <https://github.com/UB-Mannheim/tesseract/wiki>
   (or `winget install tesseract-ocr.tesseract` and accept the UAC prompt)
2. Install with defaults to `C:\Program Files\Tesseract-OCR\`.
3. Add the install directory to `PATH`, OR set the binding in code:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
4. Verify:
   ```
   tesseract --version
   ```
5. Re-run the Phase 2a benchmark — it will auto-include Tesseract:
   ```
   python evaluate_phase2a.py
   ```

## Recommended install path (Linux / WSL / macOS)

```
# Ubuntu / Debian / WSL
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## Why not bundle a portable build?

UB-Mannheim does not publish an official portable zip. Third-party portable
builds exist on GitHub but are unverified — bundling unverified binaries in
the benchmark repo would compromise reproducibility. The `requirements.txt`
+ this install doc is the audit trail.
