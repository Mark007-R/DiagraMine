"""Tiny CNN arrow detector.

Training data is generated synthetically with PIL: 1000 32×32 grayscale
patches, ~500 positive (a horizontal/vertical dashed or solid line, with
optional arrowhead) and ~500 negative (empty white, solid blob, text-like
glyph, box-corner). The CNN is a 3-layer Conv2d binary classifier (Tiny by
design: ~7K params; trains in <30 s on CPU).

At inference time the CNN does NOT do a dense sliding-window scan — that
would be prohibitively slow on CPU for a 1000×700 image with 32×32 patches
(20K+ patches/image). Instead we run it as a **verifier on Hough-line
candidates**: take the segments produced by the Hough detector, extract a
32×32 patch centred at multiple points along each segment, classify each
patch, and keep segments whose median classification is "arrow."

This is the "small CNN trained on synthetic data" baseline described in the
Day-3 task: it tests whether a learnt patch classifier generalises across
the dash-spacing variations that fool the legacy `_find_dash_groups`.

The trained model weights are cached at `models/arrow_cnn.pt` after first
use and re-loaded on subsequent calls.
"""
from __future__ import annotations

import io
import os
import random
import time
from typing import List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

_PATCH = 32
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WEIGHTS_PATH = os.path.join(_ROOT, "models", "arrow_cnn.pt")
_MODEL: "ArrowCNN | None" = None
_HOUGH_MODULE = None


class ArrowCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# Synthetic patch generation
# ---------------------------------------------------------------------------

def _draw_arrow_patch(rng: random.Random) -> np.ndarray:
    """Generate a 32×32 grayscale patch containing a horizontal or vertical
    line. Random style (dashed vs solid), random dash spacing (3-8 px on),
    random gap (2-6 px off), random thickness (1-2 px), random offset along
    the orthogonal axis."""
    p = _PATCH
    img = Image.new("L", (p, p), color=255)
    draw = ImageDraw.Draw(img)
    horizontal = rng.random() < 0.5
    thickness = rng.choice([1, 1, 2])
    offset = rng.randint(8, p - 8)
    dashed = rng.random() < 0.7
    if dashed:
        on = rng.randint(3, 9)
        off = rng.randint(2, 7)
        pos = rng.randint(-5, 5)
        while pos < p:
            x0, y0 = (pos, offset) if horizontal else (offset, pos)
            x1, y1 = (pos + on, offset) if horizontal else (offset, pos + on)
            draw.line([(x0, y0), (x1, y1)], fill=0, width=thickness)
            pos += on + off
    else:
        x0, y0 = (0, offset) if horizontal else (offset, 0)
        x1, y1 = (p - 1, offset) if horizontal else (offset, p - 1)
        draw.line([(x0, y0), (x1, y1)], fill=0, width=thickness)

    # Optional arrowhead near one end (50% chance)
    if rng.random() < 0.5:
        if horizontal:
            tip_x = p - 2 if rng.random() < 0.5 else 1
            for dy in (-2, -1, 0, 1, 2):
                if rng.random() < 0.7:
                    draw.point((tip_x, offset + dy), fill=0)
        else:
            tip_y = p - 2 if rng.random() < 0.5 else 1
            for dx in (-2, -1, 0, 1, 2):
                if rng.random() < 0.7:
                    draw.point((offset + dx, tip_y), fill=0)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _draw_negative_patch(rng: random.Random) -> np.ndarray:
    """Generate a 32×32 grayscale "not an arrow" patch.
    Variants: empty, solid blob (icon-like), text-glyph imitation,
    box-corner (two perpendicular short lines)."""
    p = _PATCH
    img = Image.new("L", (p, p), color=255)
    draw = ImageDraw.Draw(img)
    variant = rng.choice(["empty", "blob", "glyph", "corner", "noise"])
    if variant == "empty":
        # 80% truly empty, 20% with a few stray noise pixels
        if rng.random() < 0.2:
            for _ in range(rng.randint(1, 8)):
                draw.point((rng.randint(0, p - 1), rng.randint(0, p - 1)), fill=0)
    elif variant == "blob":
        x = rng.randint(6, p - 12)
        y = rng.randint(6, p - 12)
        r = rng.randint(4, 10)
        draw.ellipse([(x, y), (x + r, y + r)], fill=0)
    elif variant == "glyph":
        for _ in range(rng.randint(3, 8)):
            x0 = rng.randint(4, p - 8)
            y0 = rng.randint(8, p - 8)
            draw.line([(x0, y0), (x0 + rng.randint(-4, 4),
                                  y0 + rng.randint(-4, 4))],
                      fill=0, width=1)
    elif variant == "corner":
        cx = rng.randint(8, p - 8)
        cy = rng.randint(8, p - 8)
        draw.line([(cx, cy), (cx + rng.randint(6, 14), cy)],
                  fill=0, width=rng.choice([1, 2]))
        draw.line([(cx, cy), (cx, cy + rng.randint(6, 14))],
                  fill=0, width=rng.choice([1, 2]))
    else:  # noise
        for _ in range(rng.randint(5, 20)):
            draw.point((rng.randint(0, p - 1), rng.randint(0, p - 1)), fill=0)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _build_training_set(n_positive: int = 500, n_negative: int = 500,
                        seed: int = 42):
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n_positive):
        X.append(_draw_arrow_patch(rng))
        y.append(1)
    for _ in range(n_negative):
        X.append(_draw_negative_patch(rng))
        y.append(0)
    X = np.stack(X)[:, None, :, :]
    y = np.asarray(y, dtype=np.int64)
    return torch.from_numpy(X), torch.from_numpy(y)


def _train_model() -> ArrowCNN:
    torch.manual_seed(0)
    X, y = _build_training_set()
    model = ArrowCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bs = 64
    n_epochs = 8
    n = X.shape[0]
    for _epoch in range(n_epochs):
        idx = torch.randperm(n)
        for i in range(0, n, bs):
            sel = idx[i:i + bs]
            logits = model(X[sel])
            loss = F.cross_entropy(logits, y[sel])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def _load_or_train_model() -> ArrowCNN:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if os.path.isfile(_WEIGHTS_PATH):
        m = ArrowCNN()
        m.load_state_dict(torch.load(_WEIGHTS_PATH, map_location="cpu"))
        m.eval()
        _MODEL = m
        return m
    m = _train_model()
    os.makedirs(os.path.dirname(_WEIGHTS_PATH), exist_ok=True)
    torch.save(m.state_dict(), _WEIGHTS_PATH)
    _MODEL = m
    return m


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _hough_candidates(image_path: str) -> List[dict]:
    """Reuse the Hough detector's output as candidate segments to verify."""
    global _HOUGH_MODULE
    if _HOUGH_MODULE is None:
        from src.arrow_detection import hough_lines_detector
        _HOUGH_MODULE = hough_lines_detector
    return _HOUGH_MODULE.detect(image_path)["arrows"]


def _patches_along(gray: np.ndarray, a: dict, n_samples: int = 5) -> torch.Tensor:
    """Sample n_samples 32×32 patches along the segment of arrow `a`.
    Patches are clamped to image bounds; if the segment is too short for
    n_samples, sample at endpoints + midpoint only."""
    p = _PATCH
    h_img, w_img = gray.shape
    pts = []
    for k in range(n_samples):
        t = k / max(n_samples - 1, 1)
        x = int(a["x1"] + t * (a["x2"] - a["x1"]))
        y = int(a["y1"] + t * (a["y2"] - a["y1"]))
        x0 = max(0, min(w_img - p, x - p // 2))
        y0 = max(0, min(h_img - p, y - p // 2))
        crop = gray[y0:y0 + p, x0:x0 + p]
        if crop.shape != (p, p):
            crop = cv2.resize(crop, (p, p))
        pts.append(crop.astype(np.float32) / 255.0)
    return torch.from_numpy(np.stack(pts)[:, None, :, :])


def detect(image_path: str, accept_threshold: float = 0.5) -> dict:
    """Run CNN-verified arrow detection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    start = time.perf_counter()
    model = _load_or_train_model()
    candidates = _hough_candidates(image_path)

    accepted: List[dict] = []
    if candidates:
        with torch.no_grad():
            for a in candidates:
                patches = _patches_along(gray, a)
                logits = model(patches)
                probs = F.softmax(logits, dim=1)[:, 1].numpy()
                # Accept if the median probability ≥ threshold (robust to
                # patches that fall on box-corners at endpoints of a real arrow).
                if float(np.median(probs)) >= accept_threshold:
                    accepted.append({
                        "x1": int(a["x1"]), "y1": int(a["y1"]),
                        "x2": int(a["x2"]), "y2": int(a["y2"]),
                        "direction": a["direction"],
                        "cnn_score": float(np.median(probs)),
                    })
    elapsed = time.perf_counter() - start
    return {"arrows": accepted, "runtime_seconds": round(elapsed, 3)}
