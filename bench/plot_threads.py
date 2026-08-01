#!/usr/bin/env python3
"""Thread-scaling chart for the playbook: pp/tg throughput vs core count on Cobalt 100."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/k2_udq2_threads.csv")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "playbook/artifacts/threads_sweep.png")

SURFACE, PRIMARY, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1–2

pp, tg = {}, {}
with open(CSV) as f:
    for row in csv.DictReader(f):
        t = int(row["n_threads"])
        ts = float(row["avg_ts"])
        (pp if int(row["n_prompt"]) > 0 else tg)[t] = ts

threads = sorted(pp)
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

ax.plot(threads, [pp[t] for t in threads], color=BLUE, lw=2, marker="o", ms=7,
        markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5,
        label="Prompt processing (pp512)", zorder=3)
ax.plot(threads, [tg[t] for t in threads], color=ORANGE, lw=2, marker="o", ms=7,
        markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=1.5,
        label="Token generation (tg128)", zorder=3)

# Selective direct labels: the endpoints and the tg peak
ax.annotate(f"{pp[96]:.1f} tok/s", (96, pp[96]), textcoords="offset points",
            xytext=(-4, 10), color=PRIMARY, fontsize=10, fontweight="bold", ha="right")
ax.annotate(f"{tg[48]:.1f} tok/s", (48, tg[48]), textcoords="offset points",
            xytext=(0, 12), color=PRIMARY, fontsize=10, fontweight="bold", ha="center")
ax.annotate("generation peaks at 48 threads —\none NUMA node, bandwidth-bound",
            (48, tg[48]), textcoords="offset points", xytext=(14, -34),
            color=SECONDARY, fontsize=9, ha="left")

ax.set_title("Kimi K2 (1.04T params, 2-bit MoE) on Azure Cobalt 100 — CPU only",
             color=PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16)
ax.text(0, 1.02, "llama.cpp + KleidiAI · E96ps_v6 (96× Neoverse-N2) · tokens/sec, higher is better",
        transform=ax.transAxes, color=SECONDARY, fontsize=10)

ax.set_xlabel("Threads", color=MUTED, fontsize=10)
ax.set_xticks(threads)
ax.set_ylim(0, 24)
ax.tick_params(colors=MUTED, labelsize=10)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)

ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=SECONDARY)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {OUT}")
