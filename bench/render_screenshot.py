#!/usr/bin/env python3
"""Render captured terminal output as a clean dark-terminal PNG for the gallery.
Usage: render_screenshot.py <title> <out.png> < input.txt"""
import sys
from PIL import Image, ImageDraw, ImageFont

title, out = sys.argv[1], sys.argv[2]
text = sys.stdin.read().rstrip()

# terminal chrome
BG, HEADER, FG = (13, 13, 13), (30, 30, 30), (230, 230, 225)
GREEN, MUTED = (80, 220, 120), (137, 135, 129)
PAD, LINE_H, FONT_SIZE, HDR_H = 28, 26, 15, 44

def load_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()

mono = load_font(["Menlo.ttc", "Monaco.ttf", "/System/Library/Fonts/Menlo.ttc"], FONT_SIZE)
mono_b = load_font(["/System/Library/Fonts/Menlo.ttc"], FONT_SIZE)

lines = text.split("\n")
width = max(760, PAD * 2 + int(max(mono.getlength(l) for l in lines)) + 20)
height = HDR_H + PAD + LINE_H * len(lines) + PAD

img = Image.new("RGB", (width, height), BG)
d = ImageDraw.Draw(img)
d.rectangle([0, 0, width, HDR_H], fill=HEADER)
for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
    d.ellipse([16 + i * 22, HDR_H // 2 - 6, 28 + i * 22, HDR_H // 2 + 6], fill=c)
d.text((width // 2, HDR_H // 2), title, font=mono, fill=MUTED, anchor="mm")

y = HDR_H + PAD
for l in lines:
    color = GREEN if l.startswith(("$", ">")) else (MUTED if l.startswith("#") else FG)
    d.text((PAD, y), l, font=mono, fill=color)
    y += LINE_H

img.save(out)
print(f"wrote {out} ({width}x{height})")
