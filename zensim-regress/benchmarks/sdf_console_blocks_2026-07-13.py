#!/usr/bin/env python3
"""Per-block ablation + console-dev tier sizes (companion to
sdf_charset_sizes_2026-07-13.py; same bake pipeline, DejaVu Sans Mono,
27px em, +/-4 spread). Results table lives in
sdf_font_atlas_exploration_2026-07-13.md."""
import zlib, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools import subset
from scipy.ndimage import distance_transform_edt as edt

TTF = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
EM_HI, BASE_EM, SPREAD = 216, 27, 4.0
cmap = set(TTFont(TTF).getBestCmap().keys())
font = ImageFont.truetype(TTF, EM_HI)
asc, desc = font.getmetrics()
ADV = int(round(font.getlength("M"))); PAD = 48; S = EM_HI // BASE_EM
CW_HI, CH_HI = ADV + 2*PAD, asc + desc

def rng(*p):
    o = set()
    for a, b in p:
        o.update(range(a, b + 1))
    return o

BLOCKS = [
    ("ascii", rng((0x20, 0x7E))), ("latin-1", rng((0xA0, 0xFF))),
    ("latin-ext-A", rng((0x100, 0x17F))), ("latin-ext-B", rng((0x180, 0x24F))),
    ("IPA", rng((0x250, 0x2AF))), ("modifiers", rng((0x2B0, 0x2FF))),
    ("combining", rng((0x300, 0x36F))), ("latin-ext-add", rng((0x1E00, 0x1EFF))),
    ("gen-punct", rng((0x2000, 0x206F))), ("super/sub", rng((0x2070, 0x209F))),
    ("currency", rng((0x20A0, 0x20BF))),
]
CONSOLE_EXTRA = [
    ("letterlike", rng((0x2100, 0x214F))), ("arrows", rng((0x2190, 0x21FF))),
    ("math-ops", rng((0x2200, 0x22FF))), ("misc-technical", rng((0x2300, 0x23FF))),
    ("control-pics", rng((0x2400, 0x243F))), ("box-drawing", rng((0x2500, 0x257F))),
    ("block-elems", rng((0x2580, 0x259F))), ("geometric", rng((0x25A0, 0x25FF))),
    ("misc-symbols", rng((0x2600, 0x26FF))), ("dingbats", rng((0x2700, 0x27BF))),
    ("misc-math-A", rng((0x27C0, 0x27EF))), ("suppl-arrows-A", rng((0x27F0, 0x27FF))),
    ("braille", rng((0x2800, 0x28FF))),
]

_baked = {}
def bake(cp):
    if cp in _baked:
        return _baked[cp]
    img = Image.new("L", (CW_HI, CH_HI), 0)
    ImageDraw.Draw(img).text((PAD, 0), chr(cp), font=font, fill=255)
    a = np.asarray(img); ink = a >= 128
    if not ink.any():
        _baked[cp] = None; return None
    sd = np.where(ink, edt(ink) - 0.5, -(edt(~ink) - 0.5)).astype(np.float32)
    th, tw = CH_HI // S, CW_HI // S
    ys = (np.arange(th)*S + S//2).clip(0, CH_HI-1)
    xs = (np.arange(tw)*S + S//2).clip(0, CW_HI-1)
    q = np.clip((sd[np.ix_(ys, xs)]/S + SPREAD)/(2*SPREAD)*255, 0, 255).astype(np.uint8)
    cols = np.where((q > 8).any(axis=0))[0]; rows = np.where((q > 8).any(axis=1))[0]
    _baked[cp] = q[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    return _baked[cp]

def measure(cps):
    stream = bytearray(); t4 = 0
    for cp in cps:
        q = bake(cp)
        if q is None:
            continue
        t4 += (q.size + 1)//2
        f = (q.reshape(-1).astype(np.uint16)*15//255).astype(np.uint8)
        if len(f) % 2:
            f = np.append(f, 0)
        stream += (f[0::2] << 4 | f[1::2]).astype(np.uint8).tobytes()
    m = 6*len(cps)
    return t4 + m, len(zlib.compress(bytes(stream), 9)) + m

print(f"{'block':>15} {'glyphs':>6} {'raw4':>8} {'zlib4':>8}")
total = set()
for name, want in BLOCKS + CONSOLE_EXTRA:
    cps = sorted(want & cmap); total |= set(cps)
    r, z = measure(cps)
    print(f"{name:>15} {len(cps):>6} {r:>8} {z:>8}")
r, z = measure(sorted(total))
print(f"{'CONSOLE-DEV':>15} {len(total):>6} {r:>8} {z:>8}")
f = TTFont(TTF)
ss = subset.Subsetter(subset.Options(hinting=False, glyph_names=False))
ss.populate(unicodes=sorted(total)); ss.subset(f)
b = io.BytesIO(); f.save(b)
print(f"console-dev subset TTF (no hinting): {len(b.getvalue())} bytes")
