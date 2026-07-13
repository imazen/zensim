#!/usr/bin/env python3
"""Ladder v4: weight-compensation sweep. Columns: engine | SDF c=0 |
SDF c=0.2 | SDF c=0.35, where the SDF threshold is dilated by
shift = c * max(0, 1 - k) texels (k = out_h/BASE). Also prints mean
linear-light ink per cell as a weight proxy."""
import numpy as np, subprocess
from scipy.ndimage import distance_transform_edt as edt

W, H, CW, CH = 2496, 54, 26, 54
strip = np.fromfile("strip.raw", dtype=np.uint8).reshape(H, W).astype(np.float32) / 255.0
SS, BASE = 8, 27
WORD = "Rag7"
OUTD = "/mnt/v/output/zensim-regress/sdf-explainer"
CS = [0.0, 0.2, 0.35]

def bilinear(img, oh, ow):
    ih, iw = img.shape
    ys = (np.arange(oh) + 0.5) * ih / oh - 0.5
    xs = (np.arange(ow) + 0.5) * iw / ow - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ih-1); y1 = np.clip(y0+1, 0, ih-1)
    x0 = np.clip(np.floor(xs).astype(int), 0, iw-1); x1 = np.clip(x0+1, 0, iw-1)
    fy = np.clip(ys - y0, 0, 1)[:, None]; fx = np.clip(xs - x0, 0, 1)[None, :]
    return (img[np.ix_(y0, x0)]*(1-fy)*(1-fx) + img[np.ix_(y0, x1)]*(1-fy)*fx +
            img[np.ix_(y1, x0)]*fy*(1-fx) + img[np.ix_(y1, x1)]*fy*fx)

def srgb_encode(a):
    a = np.clip(a, 0, 1)
    return np.where(a <= 0.0031308, 12.92*a, 1.055*np.power(a, 1/2.4) - 0.055)

def srgb_decode(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.04045, v/12.92, np.power((v + 0.055)/1.055, 2.4))

def gcov(ch):
    i = ord(ch) - 0x20
    return strip[:, i*CW:(i+1)*CW]

_c = {}
def sdf_atlas(ch):
    if ch not in _c:
        hi = bilinear(gcov(ch), CH*SS, CW*SS) >= 0.5
        sd = np.where(hi, edt(hi) - 0.5, -(edt(~hi) - 0.5)).astype(np.float32)
        s = bilinear(sd, BASE, round(CW*BASE/CH)) / (CH*SS/BASE)
        _c[ch] = np.clip((s + 4) / 8 * 255, 0, 255).astype(np.uint8)
    return _c[ch]

def load_pgm(path):
    data = open(path, "rb").read()
    hdr, rest = data.split(b"\n", 1)
    _, w, h, _ = hdr.split()
    return np.frombuffer(rest, np.uint8).reshape(int(h), int(w)).astype(np.float32)/255

def word_sdf(oh, comp):
    imgs = []
    k = oh / BASE
    shift = comp * max(0.0, 1.0 - k)
    for c in WORD:
        q = sdf_atlas(c)
        ow = round(q.shape[1] * oh / q.shape[0])
        f = bilinear(q.astype(np.float32), oh, ow) / 255 * 8 - 4
        imgs.append(np.clip(0.5 + (f + shift) * k, 0, 1))
    return np.hstack(imgs)          # linear coverage

canvas = np.full((820, 850), 0.12, np.float32)
def place(img, x, y):
    h, w = img.shape
    canvas[y:y+h, x:x+w] = np.maximum(canvas[y:y+h, x:x+w], img)
def label(s, x, y, h=14):
    cx = x
    for ch in s:
        g = srgb_encode(bilinear(gcov(ch), h, round(CW*h/CH))) * 0.85
        place(g, cx, y)
        cx += g.shape[1]

COLS = [130, 310, 490, 670]
label("engine", COLS[0], 10)
for x, cv in zip(COLS[1:], CS):
    label(f"SDF c={cv}", x, 10)

print(f"{'size':>5} {'engine':>7} " + " ".join(f"c={cv:>4}" for cv in CS) + "   (mean linear ink)")
y = 40
for oh in (12, 18, 27, 54):
    label(f"{oh}px", 16, y + max(0, (oh - 16)//2))
    eng = load_pgm(f"{OUTD}/engine_{oh}px.pgm")
    place(eng, COLS[0], y)
    inks = [srgb_decode(eng).mean()]
    for x, cv in zip(COLS[1:], CS):
        cov = word_sdf(oh, cv)
        inks.append(cov.mean())
        place(srgb_encode(cov), x, y)
    print(f"{oh:>4}px " + " ".join(f"{v:7.4f}" for v in inks))
    y += oh + 20

y += 6
label("3x nearest zoom:", 16, y, 15); y += 26
for oh in (12, 18):
    label(f"{oh}px", 16, y + (oh*3 - 16)//2)
    place(np.kron(load_pgm(f"{OUTD}/engine_{oh}px.pgm"), np.ones((3,3), np.float32)), COLS[0], y)
    for x, cv in zip(COLS[1:], CS):
        place(np.kron(srgb_encode(word_sdf(oh, cv)), np.ones((3,3), np.float32)), x, y)
    y += oh*3 + 22

img8 = (np.clip(canvas[:y+8], 0, 1) * 255).astype(np.uint8)
with open("ladder4.pgm", "wb") as fh:
    fh.write(f"P5 {img8.shape[1]} {img8.shape[0]} 255\n".encode())
    fh.write(img8.tobytes())
subprocess.run(["convert", "ladder4.pgm", f"{OUTD}/sdf_ladder_v4_weightcomp.png"], check=True)
print("ok")
