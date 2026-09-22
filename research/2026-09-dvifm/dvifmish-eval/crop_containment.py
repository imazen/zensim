#!/usr/bin/env python3
"""Crop-containment check: is image A an unscaled crop of image B?

dHash is crop-blind (the stage-1 audit hashes whole frames). For each pair
(small A, large B) this slides A over B at scale 1 on a 4x-downsampled luma
grid, takes the best normalised cross-correlation, then refines at full
resolution around the best coarse offset. NCC ~1.0 at some offset = A is a
crop of B. Stdlib PNG reader (8-bit RGB/RGBA/gray, non-interlaced) + numpy.

Usage: crop_containment.py <small_dir_or_glob> <large_dir_or_glob> <out.tsv>
"""
import glob
import struct
import sys
import zlib

import numpy as np


def read_png_luma(path):
    data = open(path, "rb").read()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    pos, idat, w = 8, [], None
    while pos < len(data):
        ln, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + ln]
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, inter = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and inter == 0, (path, depth, inter)
            ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
        elif typ == b"IDAT":
            idat.append(body)
        pos += 12 + ln
    raw = zlib.decompress(b"".join(idat))
    stride = w * ch
    img = np.zeros((h, stride), np.int32)
    prev = np.zeros(stride, np.int32)
    for y in range(h):
        f = raw[y * (stride + 1)]
        row = np.frombuffer(raw, np.uint8, stride, y * (stride + 1) + 1).astype(np.int32)
        if f == 0:
            cur = row
        elif f == 2:
            cur = (row + prev) & 255
        else:
            cur = np.zeros(stride, np.int32)
            for x in range(stride):
                a = cur[x - ch] if x >= ch else 0
                b = prev[x]
                c = prev[x - ch] if x >= ch else 0
                if f == 1:
                    p = a
                elif f == 3:
                    p = (a + b) >> 1
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    p = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[x] = (row[x] + p) & 255
        img[y] = cur
        prev = cur
    px = img.reshape(h, w, ch).astype(np.float64)
    if ch >= 3:
        return 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
    return px[..., 0]


def down(a, k):
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def best_ncc(small, large):
    sh, sw = small.shape
    lh, lw = large.shape
    if sh > lh or sw > lw:
        return None
    s = (small - small.mean()) / (small.std() + 1e-9)
    best = (-2.0, 0, 0)
    for y in range(0, lh - sh + 1):
        for x in range(0, lw - sw + 1):
            win = large[y:y + sh, x:x + sw]
            v = float(((win - win.mean()) * s).mean() / (win.std() + 1e-9))
            if v > best[0]:
                best = (v, y, x)
    return best


def main():
    smalls = sorted(glob.glob(sys.argv[1]))
    larges = sorted(glob.glob(sys.argv[2]))
    L = {p: read_png_luma(p) for p in larges}
    with open(sys.argv[3], "w") as out:
        out.write("small\tlarge\tcoarse_ncc\tfine_ncc\toffset_y\toffset_x\n")
        for sp in smalls:
            a = read_png_luma(sp)
            ad = down(a, 4)
            rows = []
            for lp, b in L.items():
                r = best_ncc(ad, down(b, 4))
                if r is None:
                    continue
                v, y, x = r
                # refine at full resolution around the coarse hit
                fb = (-2.0, 0, 0)
                for dy in range(-4, 5):
                    for dx in range(-4, 5):
                        yy, xx = 4 * y + dy, 4 * x + dx
                        if 0 <= yy <= b.shape[0] - a.shape[0] and 0 <= xx <= b.shape[1] - a.shape[1]:
                            win = b[yy:yy + a.shape[0], xx:xx + a.shape[1]]
                            c = float(np.corrcoef(win.ravel(), a.ravel())[0, 1])
                            if c > fb[0]:
                                fb = (c, yy, xx)
                rows.append((fb[0], v, lp, fb[1], fb[2]))
            rows.sort(reverse=True)
            fine, coarse, lp, y, x = rows[0]
            out.write(f"{sp}\t{lp}\t{coarse:.4f}\t{fine:.4f}\t{y}\t{x}\n")
            print(f"{sp.split('/')[-1]:10} best {lp.split('/')[-1]:12} coarse {coarse:.4f} fine {fine:.4f} at ({y},{x})", flush=True)


if __name__ == "__main__":
    main()
