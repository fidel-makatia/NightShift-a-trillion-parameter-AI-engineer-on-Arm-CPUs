#!/usr/bin/env python3
"""Cost per PR review, from measured tok/s and official Azure retail prices."""
import matplotlib.pyplot as plt

SURFACE, PRIMARY, SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

SPOT, PAYGO = 1.148547, 4.426          # $/hr, Azure retail API, eastus2, 2026-07-31
PROMPT_TOK, GEN_TOK = 3000, 1200        # a substantial PR review

def task_hours(pp, tg):
    return (PROMPT_TOK / pp + GEN_TOK / tg) / 3600

tiers = [
    ("Deep review\nKimi K2 — 1.04T params", task_hours(20.6, 11.0), BLUE),
    ("Quick review\nQwen3-30B-A3B", task_hours(232.5, 48.0), ORANGE),
]

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

labels = [t[0] for t in tiers]
costs = [t[1] * SPOT for t in tiers]
bars = ax.barh(labels, costs, 0.5, color=[t[2] for t in tiers], zorder=3)
for r, (label, hrs, _) in zip(bars, tiers):
    ax.annotate(f"  ${r.get_width():.3f}   ({hrs * 3600:.0f}s on spot @ ${SPOT:.2f}/hr; pay-go ${hrs * PAYGO:.2f})",
                (r.get_width(), r.get_y() + r.get_height() / 2),
                va="center", color=PRIMARY, fontsize=10.5, fontweight="bold")

ax.set_title("What one PR review costs on NightShift (spot pricing)",
             color=PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=18)
ax.text(0, 1.04, f"{PROMPT_TOK:,}-token diff, {GEN_TOK:,}-token review · measured throughput · "
        "Azure retail prices, eastus2 · overnight batch of 100 deep reviews ≈ $8",
        transform=ax.transAxes, color=SECONDARY, fontsize=9.5)
ax.set_xlim(0, max(costs) * 1.9)
ax.set_xlabel("USD per review", color=MUTED, fontsize=10)
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASELINE)
ax.tick_params(colors=MUTED, labelsize=10)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig("playbook/artifacts/cost_per_review.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote cost_per_review.png")
