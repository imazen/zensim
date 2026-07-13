#!/usr/bin/env python3
"""Bake src/sdf_atlas.bin from src/font_strip.png — the `sdf-font`
feature's embedded atlas (prototype: same Consolas-derived glyphs as
the strip, so output is directly comparable; the production bake tool
will regenerate from an OFL face's vector outlines).

Pipeline = the validated spec (benchmarks/sdf_font_atlas_exploration_
2026-07-13.md): 8x supersampled binarization, exact signed EDT with
+/-0.5 center correction, point-sample at 27px-base texel centers,
spread +/-4 texels, 4-bit quantization (round), high-nibble-first pack.

Bin layout (little-endian):
  u32 cell_w (13) | u32 cell_h (27) | u32 n_glyphs (96) |
  f32 spread (4.0) | u32 packed_bytes_per_glyph (ceil(w*h/2)) |
  n_glyphs * packed_bytes_per_glyph nibble data (row-major per glyph,
  first pixel in the HIGH nibble, odd tail padded with 0)

Usage: python3 benchmarks/bake_sdf_atlas_2026-07-13.py  (from crate root)
"""
import struct
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt as edt

W, H, CW, CH, N = 2496, 54, 26, 54, 96
SS, BASE, SPREAD = 8, 27, 4.0

strip = np.asarray(Image.open("src/font_strip.png").convert("L"),
                   dtype=np.float32) / 255.0
assert strip.shape == (H, W)

def bilinear(img, oh, ow):
    ih, iw = img.shape
    ys = (np.arange(oh) + 0.5) * ih / oh - 0.5
    xs = (np.arange(ow) + 0.5) * iw / ow - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, ih - 1); y1 = np.clip(y0 + 1, 0, ih - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, iw - 1); x1 = np.clip(x0 + 1, 0, iw - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]; fx = np.clip(xs - x0, 0, 1)[None, :]
    return (img[np.ix_(y0, x0)] * (1-fy) * (1-fx) + img[np.ix_(y0, x1)] * (1-fy) * fx +
            img[np.ix_(y1, x0)] * fy * (1-fx) + img[np.ix_(y1, x1)] * fy * fx)

cell_w = round(CW * BASE / CH)          # 13
per_glyph = (cell_w * BASE + 1) // 2    # 176
out = bytearray(struct.pack("<IIIfI", cell_w, BASE, N, SPREAD, per_glyph))

for i in range(N):
    cov = strip[:, i*CW:(i+1)*CW]
    hi = bilinear(cov, CH*SS, CW*SS) >= 0.5
    if hi.any():
        sd = np.where(hi, edt(hi) - 0.5, -(edt(~hi) - 0.5)).astype(np.float32)
    else:
        sd = np.full((CH*SS, CW*SS), -SPREAD * SS, np.float32)
    tex = bilinear(sd, BASE, cell_w) / (CH * SS / BASE)
    q8 = np.clip((tex + SPREAD) / (2*SPREAD) * 255, 0, 255)
    nib = np.clip(np.round(q8 / 17.0), 0, 15).astype(np.uint8).reshape(-1)
    if len(nib) % 2:
        nib = np.append(nib, 0)
    out += (nib[0::2] << 4 | nib[1::2]).astype(np.uint8).tobytes()

with open("src/sdf_atlas.bin", "wb") as f:
    f.write(out)
print(f"wrote src/sdf_atlas.bin: {len(out)} bytes "
      f"(cell {cell_w}x{BASE}, {N} glyphs, {per_glyph} B/glyph packed)")
