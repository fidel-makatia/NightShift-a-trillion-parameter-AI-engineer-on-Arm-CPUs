#!/usr/bin/env python3
"""Round-2 playbook charts: two-tier serving + the NUMA cliff. Data from bench/results/."""
import matplotlib.pyplot as plt

SURFACE, PRIMARY, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # K2 = blue, Qwen3 = orange, everywhere
plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10)

# ---------- Chart 1: two-tier serving, one Arm VM ----------
# measured: K2 tg 11.0 (t=64), pp 20.6 (t=96) | Qwen3 tg 48.0 (t=48), pp 232.5 (t=96)
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
cats = ["Token generation", "Prompt processing"]
k2, qw = [11.0, 20.6], [48.0, 232.5]
x, w = [0, 1], 0.36
b1 = ax.bar([i - w / 2 - 0.01 for i in x], k2, w, color=BLUE, zorder=3,
            label="Kimi K2 — 1.04T params (deep/async tier)")
b2 = ax.bar([i + w / 2 + 0.01 for i in x], qw, w, color=ORANGE, zorder=3,
            label="Qwen3-30B-A3B (interactive tier)")
for bars in (b1, b2):
    for r in bars:
        ax.annotate(f"{r.get_height():g}", (r.get_x() + r.get_width() / 2, r.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    color=PRIMARY, fontsize=11, fontweight="bold")
ax.axhline(30, color=MUTED, lw=1, ls=(0, (4, 4)))
ax.annotate("30 tok/s — interactive comfort line", (1.38, 30), xytext=(0, 6),
            textcoords="offset points", color=SECONDARY, fontsize=9, ha="right")
style(ax)
ax.set_title("Two serving tiers, one Arm VM — tokens/sec, best measured config",
             color=PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=16)
ax.text(0, 1.02, "Azure E96ps_v6 (Cobalt 100, 96× Neoverse-N2) · llama.cpp + KleidiAI · CPU only",
        transform=ax.transAxes, color=SECONDARY, fontsize=10)
ax.set_xticks(x, cats)
ax.set_ylim(0, 265)
ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=SECONDARY)
fig.tight_layout()
fig.savefig("playbook/artifacts/two_tier.png", facecolor=SURFACE, bbox_inches="tight")

# ---------- Chart 2: the NUMA cliff (generation vs threads, both models) ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200)
fig.patch.set_facecolor(SURFACE)
k2_t = [24, 48, 56, 64, 80, 96]
k2_tg = [7.37, 10.44, 10.48, 10.96, 10.39, 9.57]
qw_t = [24, 48, 96]
qw_tg = [47.4, 48.0, 3.2]

for ax, ts, tgs, color, name, peak_t in (
    (axes[0], k2_t, k2_tg, BLUE, "Kimi K2 (1.04T, 2-bit)", 64),
    (axes[1], qw_t, qw_tg, ORANGE, "Qwen3-30B-A3B (Q4_K_M)", 48),
):
    ax.plot(ts, tgs, color=color, lw=2, marker="o", ms=7,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    peak = max(tgs)
    ax.annotate(f"peak {peak:g} tok/s @ {peak_t}t", (peak_t, peak), xytext=(0, 8),
                textcoords="offset points", ha="center", color=PRIMARY,
                fontsize=10, fontweight="bold")
    style(ax)
    ax.set_title(name, color=PRIMARY, fontsize=11, fontweight="bold", loc="left")
    ax.set_xticks(ts)
    ax.set_xlabel("Threads", color=MUTED, fontsize=10)
    ax.set_ylim(0, peak * 1.28)

axes[1].annotate("the NUMA cliff:\n96 threads → 3.2 tok/s", (96, 3.2),
                 xytext=(-8, 32), textcoords="offset points", ha="right",
                 color=SECONDARY, fontsize=9)
fig.suptitle("More cores ≠ more tokens: generation is bandwidth-bound — token generation tok/s vs threads",
             color=PRIMARY, fontsize=12.5, fontweight="bold", x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("playbook/artifacts/numa_cliff.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote two_tier.png, numa_cliff.png")
