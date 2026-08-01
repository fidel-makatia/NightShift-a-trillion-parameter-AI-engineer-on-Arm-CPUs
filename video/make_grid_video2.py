#!/usr/bin/env python3
"""Enhanced 4-session grid video: each pane shows the PROMPT, then streams the real K3
generation, with a live tok/s badge. Ends with a full-screen reveal of the web UI K3 built.

Usage: make_grid_video2.py panes.json out.mp4 [ui_image.png]
"""
import json, os, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1600, 900, 30
FRAMES = Path("video/_grid2_frames")
FF = os.environ.get("FFMPEG", "ffmpeg")

BG=(10,11,13); PANE=(18,20,24); HEADER=(30,33,39); FG=(216,219,224)
GREEN=(94,214,130); CYAN=(86,182,224); DIM=(120,124,132); ACCENT=(232,120,70)
PROMPT_COL=(240,190,90); BADGE_BG=(40,44,52)
DOTCOL={"codegen":CYAN,"webapp":ACCENT,"embedded":GREEN,"devtest":(200,160,240)}

def fnt(sz, mono=True, bold=False):
    p = "/System/Library/Fonts/Menlo.ttc" if mono else "/System/Library/Fonts/HelveticaNeue.ttc"
    try: return ImageFont.truetype(p, sz)
    except OSError: return ImageFont.load_default()
MONO=fnt(13); MONO_SM=fnt(12); HFONT=fnt(15,mono=False,bold=True); BIG=fnt(30,mono=False,bold=True)
SUB=fnt(17,mono=False); BADGE=fnt(13,mono=False,bold=True); HUGE=fnt(46,mono=False,bold=True)

fno=0
def emit(img):
    global fno; img.save(FRAMES/f"f{fno:06d}.png"); fno+=1
def hold(img, s):
    for _ in range(int(s*FPS)): emit(img.copy())

def wrap(t, w):
    out=[]
    while len(t)>w: out.append(t[:w]); t=t[w:]
    out.append(t); return out

def draw_pane(d, x,y,w,h, p, prompt_chars, out_chars):
    d.rounded_rectangle([x,y,x+w,y+h], radius=8, fill=PANE)
    d.rounded_rectangle([x,y,x+w,y+34], radius=8, fill=HEADER); d.rectangle([x,y+24,x+w,y+34], fill=HEADER)
    d.ellipse([x+13,y+12,x+24,y+23], fill=DOTCOL.get(p["key"],DIM))
    d.text((x+36,y+17), p["title"], font=HFONT, fill=FG, anchor="lm")
    # tok/s badge (right)
    btxt=f"K3 · {p['tps']:.2f} tok/s"; bw=int(BADGE.getlength(btxt))+18
    d.rounded_rectangle([x+w-bw-10,y+9,x+w-10,y+27], radius=9, fill=BADGE_BG)
    d.text((x+w-bw-1,y+18), btxt, font=BADGE, fill=GREEN, anchor="lm")
    # body
    cw=(w-28)//7; ty=y+44; lh=16.5
    # prompt block (revealed by prompt_chars)
    ptext="❯ "+p["prompt"]
    shown_p=ptext[:prompt_chars]
    plines=wrap(shown_p, cw)
    for ln in plines:
        d.text((x+14,ty), ln, font=MONO_SM, fill=PROMPT_COL); ty+=lh
    if prompt_chars>=len(ptext):
        ty+=6
        # output (revealed by out_chars), scrolling to keep last lines
        flat=[]
        acc=0; body=""
        for ch in p["outfull"]:
            if acc>=out_chars: break
            body+=ch; acc+=1
        for ln in body.split("\n"):
            flat.extend(wrap(ln, cw) or [""])
        maxrows=int((y+h-ty-8)//lh)
        for ln in flat[-maxrows:]:
            col=GREEN if ln.startswith("$") else FG
            d.text((x+14,ty), ln[:cw], font=MONO, fill=col); ty+=lh

def render(panes, out, ui_img):
    FRAMES.mkdir(parents=True, exist_ok=True)
    for f in FRAMES.glob("*.png"): f.unlink()
    for p in panes:
        p["prompt_len"]=len("❯ "+p["prompt"]); p["out_len"]=len(p["outfull"])
    PROMPT_CPS=55
    prompt_dur=max(p["prompt_len"] for p in panes)/PROMPT_CPS + 0.5
    gen_dur=max(p["out_len"]/p["cps"] for p in panes)+1.2
    total=prompt_dur+gen_dur
    nf=int(total*FPS)
    gx,gy,gap=24,96,18; gw=(W-2*gx-gap)//2; gh=(H-gy-24-gap)//2
    coords=[(gx,gy),(gx+gw+gap,gy),(gx,gy+gh+gap),(gx+gw+gap,gy+gh+gap)]
    for fi in range(nf):
        t=fi/FPS
        img=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(img)
        d.text((gx,24),"One Arm CPU VM · Kimi K3 (2.8T) · 4 concurrent dev sessions",font=BIG,fill=(245,246,248))
        d.text((gx,64),"prompt → live generation · tokens/sec per session · Azure Cobalt 100 · no GPU",font=SUB,fill=DIM)
        for i,p in enumerate(panes):
            pc=int(min(t,prompt_dur)*PROMPT_CPS)
            oc=int(max(0,t-prompt_dur)*p["cps"])
            x,y=coords[i]; draw_pane(d,x,y,gw,gh,p,pc,oc)
        emit(img)
    # hold last grid frame
    last=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(last)
    d.text((gx,24),"One Arm CPU VM · Kimi K3 (2.8T) · 4 concurrent dev sessions",font=BIG,fill=(245,246,248))
    d.text((gx,64),"prompt → live generation · tokens/sec per session · Azure Cobalt 100 · no GPU",font=SUB,fill=DIM)
    for i,p in enumerate(panes):
        x,y=coords[i]; draw_pane(d,x,y,gw,gh,p,p["prompt_len"],p["out_len"])
    hold(last,2.5)
    # web UI finale
    if ui_img and Path(ui_img).exists():
        card=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(card)
        d.text((W//2,70),"…and the web app K3 wrote, actually rendered:",font=SUB,fill=DIM,anchor="mm")
        src=Image.open(ui_img).convert("RGB")
        sc=min((W-260)/src.width,(H-220)/src.height); src=src.resize((int(src.width*sc),int(src.height*sc)),Image.LANCZOS)
        card.paste(src,((W-src.width)//2,120))
        hold(card,3.5)
    subprocess.run([FF,"-y","-r",str(FPS),"-i",str(FRAMES/"f%06d.png"),"-c:v","libx264","-pix_fmt","yuv420p","-movflags","+faststart",out],check=True)
    print(f"wrote {out} ({fno} frames, {fno/FPS:.1f}s)")

if __name__=="__main__":
    panes=json.load(open(sys.argv[1])); out=sys.argv[2]
    ui=sys.argv[3] if len(sys.argv)>3 else None
    render(panes,out,ui)
