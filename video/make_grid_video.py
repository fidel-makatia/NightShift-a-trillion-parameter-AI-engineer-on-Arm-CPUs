#!/usr/bin/env python3
"""Render a 2x2 grid of terminal panes filling in concurrently, then encode to MP4.
Each pane streams its REAL captured output at a rate proportional to its measured tok/s,
so the video honestly shows four dev jobs sharing one Kimi K3 endpoint on Arm.

Usage: make_grid_video.py panes.json out.mp4
panes.json = [{"title","lines":[...],"cps":<chars/sec>}, x4]
"""
import json
import os
import subprocess
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1600, 900, 30
FRAMES = Path("video/_grid_frames")
FF = os.environ.get("FFMPEG", "ffmpeg")

BG = (10, 11, 13)
PANE = (18, 20, 24)
HEADER = (30, 33, 39)
FG = (216, 219, 224)
GREEN = (94, 214, 130)
CYAN = (86, 182, 224)
DIM = (120, 124, 132)
ACCENT = (232, 120, 70)
TITLES_COL = {"codegen": CYAN, "webapp": ACCENT, "embedded": GREEN, "devtest": (200, 160, 240)}


def fnt(sz, mono=True, bold=False):
    for c in (["/System/Library/Fonts/Menlo.ttc"] if mono else
              ["/System/Library/Fonts/HelveticaNeue.ttc"]):
        try:
            return ImageFont.truetype(c, sz)
        except OSError:
            pass
    return ImageFont.load_default()


MONO = fnt(14)
HFONT = fnt(15, mono=False, bold=True)
BIG = fnt(30, mono=False, bold=True)
SUB = fnt(17, mono=False)


def wrap(line, width=64):
    out = []
    while len(line) > width:
        out.append(line[:width]); line = line[width:]
    out.append(line)
    return out


def pane_box(d, x, y, w, h, title, key, shown_lines):
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=PANE)
    d.rounded_rectangle([x, y, x + w, y + 30], radius=8, fill=HEADER)
    d.rectangle([x, y + 20, x + w, y + 30], fill=HEADER)
    dot = TITLES_COL.get(key, DIM)
    d.ellipse([x + 12, y + 11, x + 22, y + 21], fill=dot)
    d.text((x + 34, y + 15), title, font=HFONT, fill=FG, anchor="lm")
    ty = y + 42
    lh = 18
    maxrows = (h - 50) // lh
    flat = []
    for ln in shown_lines:
        flat.extend(wrap(ln, (w - 28) // 8))
    for ln in flat[-maxrows:]:
        col = FG
        if ln.startswith("$ "):
            col = GREEN
        elif ln.startswith("//") or ln.startswith("#"):
            col = DIM
        d.text((x + 14, ty), ln[: (w - 24)//8], font=MONO, fill=col)
        ty += lh


def render(panes, out):
    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*.png"):
        f.unlink()
    # per-pane: how many characters revealed over time
    total = [sum(len(l) + 1 for l in p["lines"]) for p in panes]
    cps = [max(p.get("cps", 40), 12) for p in panes]
    duration = max(total[i] / cps[i] for i in range(len(panes))) + 2.0
    nframes = int(duration * FPS)

    gx, gy, gw, gh, gap = 24, 96, None, None, 18
    gw = (W - 2 * gx - gap) // 2
    gh = (H - gy - 24 - gap) // 2
    coords = [(gx, gy), (gx + gw + gap, gy), (gx, gy + gh + gap), (gx + gw + gap, gy + gh + gap)]

    for fi in range(nframes):
        t = fi / FPS
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((gx, 26), "One Arm CPU VM · Kimi K3 (2.8T) · 4 concurrent dev sessions",
                font=BIG, fill=(245, 246, 248))
        d.text((gx, 66), "Azure Cobalt 100 · continuous batching · no GPU",
                font=SUB, fill=DIM)
        for i, p in enumerate(panes):
            nchars = int(t * cps[i])
            # reveal lines up to nchars
            shown, acc = [], 0
            for ln in p["lines"]:
                if acc + len(ln) + 1 <= nchars:
                    shown.append(ln); acc += len(ln) + 1
                elif acc < nchars:
                    shown.append(ln[:nchars - acc] + "█"); acc = nchars; break
                else:
                    break
            x, y = coords[i]
            pane_box(d, x, y, gw, gh, p["title"], p["key"], shown)
        img.save(FRAMES / f"g{fi:06d}.png")
    subprocess.run([FF, "-y", "-r", str(FPS), "-i", str(FRAMES / "g%06d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
                   check=True)
    print(f"wrote {out} ({nframes} frames, {duration:.1f}s)")


if __name__ == "__main__":
    panes = json.load(open(sys.argv[1]))
    render(panes, sys.argv[2])
