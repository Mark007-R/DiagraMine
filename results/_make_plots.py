"""Generate phase2a comparison charts from the leaderboard CSV."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "phase2a_text_box.csv")

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
text_rows = [r for r in rows if r["stage"] == "text" and r["precision"]]
box_rows = [r for r in rows if r["stage"] == "box"]

# Plot 1: F1 comparison (text + box)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
names = [r["detector"] for r in text_rows]
f1s = [float(r["f1"]) for r in text_rows]
runtimes = [float(r["avg_runtime_seconds"]) for r in text_rows]
ax.bar(names, f1s, color=["#3776ab", "#76b900"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 (component-label fuzz match)")
ax.set_title("Text detection — 15 diagrams")
for i, (f1, rt) in enumerate(zip(f1s, runtimes)):
    ax.text(i, f1 / 2, f"F1={f1:.3f}\n{rt:.2f}s/img",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")

ax = axes[1]
names = [r["detector"] for r in box_rows]
f1s = [float(r["f1"]) for r in box_rows]
runtimes = [float(r["avg_runtime_seconds"]) for r in box_rows]
ax.bar(names, f1s, color=["#3776ab", "#e67e22", "#c0392b"])
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 (box IoU@0.5)")
ax.set_title("Box detection — 14 generated diagrams")
for i, (f1, rt) in enumerate(zip(f1s, runtimes)):
    label = f"F1={f1:.3f}\n{rt*1000:.0f}ms/img"
    y = max(f1 / 2, 0.06)
    color = "white" if f1 > 0.1 else "black"
    ax.text(i, y, label, ha="center", va="center", fontsize=10, color=color,
            fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

fig.suptitle("Phase 2a — DiagraMine Day 2", fontweight="bold")
fig.tight_layout()
out = os.path.join(HERE, "phase2a_f1_comparison.png")
fig.savefig(out, dpi=130)
print(f"wrote {out}")
plt.close(fig)
