#!/usr/bin/env python3
"""ExpertAtlas analysis: turn an expert-activation CSV into a skew report, a heat map,
and a memory-placement policy for llama.cpp.

  python3 analyze.py activations.csv --keep-frac 0.5 --outdir .

Inputs : activations.csv with columns layer,expert,count (from expert-capture).
Outputs: expert_heatmap.png, skew_curve.png, placement.json, override-tensor.txt
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load(path):
    cells = {}
    layers, experts = set(), set()
    with open(path) as f:
        for row in csv.DictReader(f):
            l, e, c = int(row["layer"]), int(row["expert"]), int(row["count"])
            cells[(l, e)] = c
            layers.add(l)
            experts.add(e)
    return cells, sorted(layers), sorted(experts)


def gini(values):
    xs = sorted(values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = 0
    for i, x in enumerate(xs, 1):
        cum += i * x
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def analyze(cells, layers, experts, keep_frac):
    total = sum(cells.values())
    per_expert = defaultdict(int)          # global expert load (summed over layers)
    for (l, e), c in cells.items():
        per_expert[e] += c

    ranked = sorted(per_expert.items(), key=lambda kv: kv[1], reverse=True)
    # cumulative share of activations covered as we add experts hottest-first
    cum, curve = 0, []
    for i, (e, c) in enumerate(ranked, 1):
        cum += c
        curve.append((i / len(experts), cum / total))

    # placement is per-(layer,expert): keep the hottest keep_frac of cells resident
    ranked_cells = sorted(cells.items(), key=lambda kv: kv[1], reverse=True)
    n_keep = int(round(keep_frac * len(ranked_cells)))
    hot = dict(ranked_cells[:n_keep])
    cold = dict(ranked_cells[n_keep:])
    hot_share = sum(hot.values()) / total if total else 0

    return {
        "total_activations": total,
        "n_layers": len(layers),
        "n_experts": len(experts),
        "gini": round(gini(list(per_expert.values())), 4),
        "keep_frac": keep_frac,
        "hot_cells": n_keep,
        "cold_cells": len(cold),
        "hot_activation_share": round(hot_share, 4),
        "curve": curve,
        "ranked_experts": ranked,
        "hot_keys": list(hot.keys()),
    }


def emit_overrides(report, path):
    """llama.cpp --override-tensor patterns pushing cold experts' FFN tensors off-RAM.
    Cold experts are matched by their per-layer FFN-exps tensor name and pinned to CPU/disk;
    hot ones stay in the default (fast) buffer."""
    cold_by_layer = defaultdict(list)
    hot = set(tuple(k) for k in report["hot_keys"])
    # any (layer,expert) not hot is cold
    with open(path, "w") as f:
        f.write("# Feed these to llama-server as repeated --override-tensor args.\n")
        f.write("# Hot experts stay resident; cold expert FFN tensors are pinned to a\n")
        f.write("# memory-mapped (disk-backed) buffer so they don't consume RAM.\n")
        f.write(f"# Policy: keep hottest {report['keep_frac']:.0%} of cells "
                f"({report['hot_activation_share']:.1%} of all activations).\n")
        # llama.cpp exposes per-layer expert tensors as blk.<il>.ffn_*_exps.weight
        f.write("# Example (regex, disk buffer):\n")
        f.write("#   --override-tensor 'blk\\.(3|7|11)\\.ffn_.*_exps=CPU'\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--keep-frac", type=float, default=0.5)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cells, layers, experts = load(args.csv)
    report = analyze(cells, layers, experts, args.keep_frac)

    json_report = {k: v for k, v in report.items() if k not in ("curve", "ranked_experts", "hot_keys")}
    (outdir / "placement.json").write_text(json.dumps(json_report, indent=2))
    emit_overrides(report, outdir / "override-tensor.txt")

    try:
        make_charts(cells, layers, experts, report, outdir)
    except ImportError:
        print("matplotlib not available; skipped charts")

    print(json.dumps(json_report, indent=2))
    print(f"\nHeadline: keeping the hottest {args.keep_frac:.0%} of expert cells in RAM "
          f"covers {report['hot_activation_share']:.1%} of all activations "
          f"(Gini {report['gini']}).")


def make_charts(cells, layers, experts, report, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE, PRIMARY, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, BASELINE, BLUE, ORANGE = "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834"
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    # skew curve: cumulative activation share vs fraction of experts (hottest first)
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    xs = [0] + [p[0] for p in report["curve"]]
    ys = [0] + [p[1] for p in report["curve"]]
    ax.plot(xs, ys, color=BLUE, lw=2.5, zorder=3)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=(0, (4, 4)), zorder=2)  # uniform baseline
    kf = report["keep_frac"]
    ax.axvline(kf, color=ORANGE, lw=1.5, zorder=2)
    ax.annotate(f"keep hottest {kf:.0%} of experts\n→ {report['hot_activation_share']:.0%} of activations",
                (kf, report["hot_activation_share"]), xytext=(12, -30),
                textcoords="offset points", color=PRIMARY, fontsize=10, fontweight="bold")
    g = report["gini"]
    verdict = "heavily skewed" if g >= 0.35 else "moderately skewed" if g >= 0.15 else "near-uniform"
    ax.set_title(f"Expert activation is {verdict} (Gini {g})",
                 color=PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.02, f"{report['n_experts']} experts × {report['n_layers']} layers · "
            "solid = measured, dashed = uniform routing", transform=ax.transAxes,
            color=SECONDARY, fontsize=9.5)
    ax.set_xlabel("fraction of experts (hottest first)", color=MUTED, fontsize=10)
    ax.set_ylabel("cumulative share of activations", color=MUTED, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "skew_curve.png", facecolor=SURFACE, bbox_inches="tight")

    # heat map: layer × expert activation
    import numpy as np
    grid = np.zeros((len(layers), len(experts)))
    li = {l: i for i, l in enumerate(layers)}
    ei = {e: i for i, e in enumerate(experts)}
    for (l, e), c in cells.items():
        grid[li[l], ei[e]] = c
    grid = grid / (grid.max() or 1)
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(grid, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_title("Expert activation heat map — layer × expert",
                 color=PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("expert id", color=MUTED, fontsize=10)
    ax.set_ylabel("layer", color=MUTED, fontsize=10)
    ax.tick_params(colors=MUTED, labelsize=9)
    fig.colorbar(im, ax=ax, label="relative activation")
    fig.tight_layout()
    fig.savefig(outdir / "expert_heatmap.png", facecolor=SURFACE, bbox_inches="tight")


if __name__ == "__main__":
    main()
