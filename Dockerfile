# DiagraMine — production FastAPI image.
#
# Build:  docker build -t diagramine:4.0 .
# Run:    docker run -p 8000:8000 diagramine:4.0
# Probe:  curl http://localhost:8000/health
# Use:    curl -F "file=@diagram.png" http://localhost:8000/extract | jq .
#
# Image philosophy: slim Python base + only the OS libs OpenCV / EasyOCR /
# matplotlib require. EasyOCR's English model weights (~64 MB) download on
# first /extract call rather than at build, so the image stays compact for CI.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# OS libs: OpenCV needs libGL + libglib; matplotlib needs libfreetype/libpng.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libfreetype6 \
        libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for cacheability.
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Source.
COPY src/ /app/src/
COPY diagram_analysis.py /app/
COPY data/icon_templates/ /app/data/icon_templates/

EXPOSE 8000

# Healthcheck pings /health every 30 s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
