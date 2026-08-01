#!/usr/bin/env python3
"""Refined, editorial-style charts for NightShift. Restrained palette, strong type
hierarchy, generous whitespace, minimal chartjunk — designed to read as human, not template."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---- typography: prefer a real sans the system has ----
for fam in ("Inter", "Helvetica Neue", "Helvetica", "Arial"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        FONT = fam
        break
    except Exception:
        FONT = "DejaVu Sans"
mpl.rcParams.update({
    "font.family": FONT,
    "font.size": 13,
    "axes.edgecolor": "#d9d7d1",
    "axes.linewidth": 1.0,
    "figure.dpi": 200,
})

INK = "#1a1a17"          # near-black text
SUB = "#6b6a64"          # muted secondary text
FAINT = "#ededea"        # hairline grid
PAPER = "#ffffff"
ACCENT = "#c2410c"       # a single confident accent (burnt orange)
STEEL = "#1f4e79"        # a single deep supporting color
BAR2 = "#c9c6bd"         # neutral gray for the "reference" series


def _finish(fig, ax, title, subtitle, source):
    ax.set_title("")  # we place titles manually for control
    fig.text(0.02, 0.965, title, ha="left", va="top", fontsize=19,
             fontweight="bold", color=INK)
    fig.text(0.02, 0.895, subtitle, ha="left", va="top", fontsize=12.5, color=SUB)
    fig.text(0.02, 0.03, source, ha="left", va="bottom", fontsize=10, color=SUB)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d9d7d1")
    ax.spines["bottom"].set_color("#d9d7d1")
    ax.tick_params(colors=SUB, length=0, labelsize=12)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)


# ============ Chart 1: two-tier throughput (grouped, horizontal, clean) ============
def two_tier():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.subplots_adjust(left=0.30, right=0.95, top=0.74, bottom=0.16)

    rows = [
        ("Prompt processing", "Qwen3-30B", 232.5, ACCENT),
        ("Prompt processing", "Kimi K2", 20.6, STEEL),
        ("Token generation", "Qwen3-30B", 48.0, ACCENT),
        ("Token generation", "Kimi K2", 11.0, STEEL),
    ]
    ys = [3.4, 2.6, 1.2, 0.4]
    for (grp, model, val, col), y in zip(rows, ys):
        ax.barh(y, val, height=0.66, color=col, zorder=3)
        ax.text(val + 3, y, f"{val:g}", va="center", ha="left",
                fontsize=13, color=INK, fontweight="bold")
        ax.text(-6, y, model, va="center", ha="right", fontsize=12, color=SUB)
    # group labels
    ax.text(-6, 3.0, "PROMPT", va="center", ha="right", fontsize=10.5,
            color=INK, fontweight="bold")
    ax.text(-6, 0.8, "GENERATION", va="center", ha="right", fontsize=10.5,
            color=INK, fontweight="bold")

    ax.axvline(30, color=SUB, lw=1, ls=(0, (3, 3)), zorder=2)
    ax.text(30, 4.0, "30 tok/s\ninteractive comfort", fontsize=9.5, color=SUB,
            ha="center", va="bottom")
    ax.set_xlim(0, 250)
    ax.set_ylim(-0.2, 4.3)
    ax.set_yticks([])
    ax.set_xlabel("tokens / second  ·  higher is better", color=SUB, fontsize=11.5)
    ax.grid(axis="x", color=FAINT, lw=1, zorder=0)
    _finish(fig, ax,
            "Two model tiers, one Arm CPU VM",
            "Kimi K2 (1.04T params) for depth · Qwen3-30B for interactive speed",
            "Azure E96ps_v6 · 96× Neoverse-N2 (Cobalt 100) · llama.cpp + KleidiAI · CPU only, no GPU")
    fig.savefig("playbook/artifacts/pro_two_tier.png", facecolor=PAPER)
    print("wrote pro_two_tier.png")


# ============ Chart 2: cost per PR review (the money chart) ============
def cost():
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.subplots_adjust(left=0.06, right=0.95, top=0.72, bottom=0.20)
    items = [("Kimi K2 · 1.04T", 0.081, STEEL), ("Qwen3-30B", 0.012, ACCENT)]
    ys = [1, 0]
    for (label, val, col), y in zip(items, ys):
        ax.barh(y, val, height=0.5, color=col, zorder=3)
        ax.text(val + 0.002, y, f"${val:.3f}", va="center", ha="left",
                fontsize=15, color=INK, fontweight="bold")
        ax.text(0.001, y + 0.36, label, va="bottom", ha="left", fontsize=12, color=SUB)
    ax.set_xlim(0, 0.11)
    ax.set_ylim(-0.5, 1.7)
    ax.set_yticks([])
    ax.set_xlabel("USD per pull-request review  ·  spot pricing", color=SUB, fontsize=11.5)
    ax.grid(axis="x", color=FAINT, lw=1, zorder=0)
    _finish(fig, ax,
            "A trillion-parameter code review costs 8¢",
            "3,000-token diff, 1,200-token review · 100 reviews overnight ≈ $8",
            "Measured throughput × Azure retail spot ($1.15/hr, eastus2) · your code never leaves your tenant")
    fig.savefig("playbook/artifacts/pro_cost.png", facecolor=PAPER)
    print("wrote pro_cost.png")


# ============ Chart 3: thread scaling / the bandwidth wall ============
def scaling():
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.74, bottom=0.14)
    threads = [24, 48, 64, 96]
    pp = [6.6, 12.7, None, 20.6]
    tg = [7.4, 10.4, 11.0, 9.6]
    # prompt processing (compute-bound, scales)
    xp = [t for t, v in zip(threads, pp) if v is not None]
    yp = [v for v in pp if v is not None]
    ax.plot(xp, yp, color=STEEL, lw=2.4, marker="o", ms=7, mfc=STEEL, mec=PAPER, mew=1.6, zorder=3)
    ax.plot(threads, tg, color=ACCENT, lw=2.4, marker="o", ms=7, mfc=ACCENT, mec=PAPER, mew=1.6, zorder=3)
    ax.text(96, 20.6 + 0.7, "prompt processing", color=STEEL, fontsize=12, fontweight="bold", ha="right")
    ax.text(64, 11.0 + 0.9, "token generation — peaks at 64, then the\nmemory-bandwidth wall bites", color=ACCENT, fontsize=11, ha="center")
    ax.set_xticks(threads)
    ax.set_xlim(18, 102)
    ax.set_ylim(0, 24)
    ax.set_xlabel("CPU threads", color=SUB, fontsize=11.5)
    ax.set_ylabel("tokens / second", color=SUB, fontsize=11.5)
    ax.grid(color=FAINT, lw=1, zorder=0)
    _finish(fig, ax,
            "More cores stop helping generation",
            "Kimi K2 (1.04T, 2-bit) — generation is bound by memory bandwidth, not compute",
            "Azure E96ps_v6 · two NUMA nodes of 48 cores · llama.cpp llama-bench")
    fig.savefig("playbook/artifacts/pro_scaling.png", facecolor=PAPER)
    print("wrote pro_scaling.png")


two_tier(); cost(); scaling()
