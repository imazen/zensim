#!/usr/bin/env python3
"""Measure real SDF atlas sizes per charset tier, baked from DejaVu Sans
Mono (free license). Pipeline per glyph = the pinned spec: render at 8x
(216px em), binarize, exact signed EDT (+/-0.5 corrected), point-sample
at 27px-em texel grid, spread +/-4 texels, quantize.

Reports per tier: glyph count, uniform-grid 4-bit bytes (exact), tight-
cropped 8-bit / 4-bit bytes (+6B metrics per glyph), and zlib of the
tight 4-bit stream (the zenflate feature tier)."""
import zlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from scipy.ndimage import distance_transform_edt as edt

TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
EM_HI, BASE_EM = 216, 27          # 8x supersample of a 27px-em base
SPREAD = 4.0

cmap = set(TTFont(TTF).getBestCmap().keys())
font = ImageFont.truetype(TTF, EM_HI)
asc, desc = font.getmetrics()
ADV_HI = int(round(font.getlength("M")))
PAD_HI = 48                        # horizontal overhang room (combining marks)
CELL_W_HI, CELL_H_HI = ADV_HI + 2*PAD_HI, asc + desc
S = EM_HI // BASE_EM               # 8

def rng(*pairs):
    out = set()
    for a, b in pairs:
        out.update(range(a, b + 1))
    return out

LATIN_CORE = rng((0x20, 0x7E), (0xA0, 0xFF), (0x100, 0x17F))
PUNCT_SYM = rng((0x2000, 0x206F), (0x20A0, 0x20BF), (0x2070, 0x209F))
LATIN_FULL = LATIN_CORE | PUNCT_SYM | rng((0x180, 0x24F), (0x1E00, 0x1EFF),
                                          (0x250, 0x2AF), (0x2B0, 0x2FF), (0x300, 0x36F))
NON_ASIAN = LATIN_FULL | rng((0x370, 0x3FF), (0x1F00, 0x1FFF),      # Greek + polytonic
                             (0x400, 0x52F),                          # Cyrillic + supp
                             (0x530, 0x58F), (0x590, 0x5FF),          # Armenian, Hebrew
                             (0x10A0, 0x10FF),                        # Georgian
                             (0x2100, 0x214F), (0x2190, 0x21FF),      # letterlike, arrows
                             (0x2200, 0x22FF),                        # math operators
                             (0x2500, 0x25FF))                        # box/blocks/geometric

TIERS = [("ascii",           rng((0x20, 0x7E))),
         ("latin-web",       LATIN_CORE | rng((0x2013, 0x2026)) | rng((0x20AC, 0x20AC))),
         ("latin-complete",  LATIN_FULL),
         ("non-asian",       NON_ASIAN)]

def bake(cp):
    img = Image.new("L", (CELL_W_HI, CELL_H_HI), 0)
    ImageDraw.Draw(img).text((PAD_HI, 0), chr(cp), font=font, fill=255)
    a = np.asarray(img)
    ink = a >= 128
    if not ink.any():
        return None                                    # metrics-only (spaces)
    sd = np.where(ink, edt(ink) - 0.5, -(edt(~ink) - 0.5)).astype(np.float32)
    th, tw = CELL_H_HI // S, CELL_W_HI // S
    ys = (np.arange(th) * S + S // 2).clip(0, CELL_H_HI - 1)
    xs = (np.arange(tw) * S + S // 2).clip(0, CELL_W_HI - 1)
    tex = sd[np.ix_(ys, xs)] / S
    q = np.clip((tex + SPREAD) / (2 * SPREAD) * 255, 0, 255).astype(np.uint8)
    zero = np.uint8(round((0 + SPREAD) / (2 * SPREAD) * 255 / 17) * 17)  # 4-bit empty level
    cols = np.where((q > 8).any(axis=0))[0]            # tight-crop where field ~ -spread
    rows = np.where((q > 8).any(axis=1))[0]
    return q[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]

print(f"font em metrics @{BASE_EM}px: advance {ADV_HI//S} texels, cell "
      f"{(ADV_HI//S)}x{CELL_H_HI//S} = {(ADV_HI//S)*(CELL_H_HI//S)} texels/glyph")
UNIF4 = ((ADV_HI // S) * (CELL_H_HI // S) + 1) // 2

baked = {}
name_pad = max(len(n) for n, _ in TIERS)
print(f"{'tier':>{name_pad}} {'glyphs':>6} {'uniform4':>9} {'tight8':>9} "
      f"{'tight4':>9} {'tight4+zlib':>11}")
for name, want in TIERS:
    cps = sorted(want & cmap)
    t8 = t4 = 0
    stream4 = bytearray()
    for cp in cps:
        if cp not in baked:
            baked[cp] = bake(cp)
        q = baked[cp]
        if q is None:
            continue
        t8 += q.size
        n4 = (q.size + 1) // 2
        t4 += n4
        f = (q.reshape(-1).astype(np.uint16) * 15 // 255).astype(np.uint8)
        if len(f) % 2:
            f = np.append(f, 0)
        stream4 += (f[0::2] << 4 | f[1::2]).astype(np.uint8).tobytes()
    metrics = 6 * len(cps)
    z = len(zlib.compress(bytes(stream4), 9)) + metrics
    print(f"{name:>{name_pad}} {len(cps):>6} {UNIF4*len(cps):>9} "
          f"{t8+metrics:>9} {t4+metrics:>9} {z:>11}")
print("(tight sizes include 6 B/glyph metrics; uniform4 is exact cell arithmetic)")
