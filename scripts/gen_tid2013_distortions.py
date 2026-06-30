#!/usr/bin/env python3
"""TID2013-style artificial-distortion generator for the zensim OOD-safety
training group.

Applies the TID2013 distortion taxonomy (a representative subset) at 5
severity levels to CID22-safe synthetic source images. The output (ref,
distorted) pairs are scored by ssim2 and fed the 372-feature extractor to
build an artificial-distortion training parquet — the data that teaches the
metric "blur/noise/etc. = low quality" (OOD safety) WITHOUT the strict
monotone-cbc constraint that craters CID22 on codec-distortion rank.

KADID/TID stay held-out validation guards; this set is generated fresh on the
synthetic sources, so there's no train/val leak.

Usage:
  gen_tid2013_distortions.py --src DIR --out DIR [--levels 5] [--limit N]
Emits <out>/<stem>__<dist>__L<level>.png per distorted variant + a manifest
TSV (ref_path, dist_path, dist_type, level).
"""
import argparse, os, io, sys
import numpy as np
import cv2
from PIL import Image

# Per-distortion severity ladders (level 0 = mildest .. 4 = strongest),
# calibrated so ssim2 spans roughly the full quality range across the set.
def _f(img):  # uint8 HWC -> float32 [0,1]
    return img.astype(np.float32) / 255.0
def _u(a):    # float -> uint8
    return np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)

def gaussian_noise(img, L):
    s = [0.02, 0.05, 0.09, 0.15, 0.25][L]
    return _u(_f(img) + np.random.normal(0, s, img.shape).astype(np.float32))

def color_noise(img, L):  # noise mostly in chroma (YCbCr)
    s = [0.04, 0.09, 0.16, 0.26, 0.40][L]
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y[..., 1:] += np.random.normal(0, s * 255, y[..., 1:].shape).astype(np.float32)
    return cv2.cvtColor(np.clip(y, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def impulse_noise(img, L):  # salt & pepper
    p = [0.01, 0.03, 0.06, 0.12, 0.22][L]
    out = img.copy()
    m = np.random.random(img.shape[:2])
    out[m < p / 2] = 0
    out[m > 1 - p / 2] = 255
    return out

def mult_noise(img, L):  # multiplicative (speckle)
    s = [0.03, 0.07, 0.13, 0.22, 0.35][L]
    return _u(_f(img) * (1 + np.random.normal(0, s, img.shape).astype(np.float32)))

def quantization(img, L):  # bit-depth reduction
    bits = [6, 5, 4, 3, 2][L]
    step = 256 // (2 ** bits)
    return (img // step * step).astype(np.uint8)

def gaussian_blur(img, L):
    s = [0.7, 1.3, 2.2, 3.5, 5.5][L]
    return cv2.GaussianBlur(img, (0, 0), s)

def jpeg(img, L):
    q = [38, 24, 15, 9, 5][L]
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def mean_shift(img, L):  # brightness offset
    d = [10, 22, 36, 54, 78][L] * (1 if np.random.random() < 0.5 else -1)
    return np.clip(img.astype(np.int16) + d, 0, 255).astype(np.uint8)

def contrast(img, L):
    c = [0.85, 0.72, 0.58, 1.5, 1.9][L]
    return _u(np.clip((_f(img) - 0.5) * c + 0.5, 0, 1))

def saturation(img, L):
    f = [0.6, 0.3, 0.0, 1.6, 2.2][L]
    h = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    h[..., 1] = np.clip(h[..., 1] * f, 0, 255)
    return cv2.cvtColor(h.astype(np.uint8), cv2.COLOR_HSV2RGB)

def sharpen(img, L):  # over-sharpening (HF emphasis)
    a = [0.4, 0.9, 1.6, 2.6, 4.0][L]
    blur = cv2.GaussianBlur(img, (0, 0), 1.5).astype(np.float32)
    return _u(np.clip((_f(img) * 255 + a * (img.astype(np.float32) - blur)) / 255, 0, 1))

def chromatic_aberration(img, L):
    d = [1, 2, 3, 5, 8][L]
    b, g, r = cv2.split(img)
    M = np.float32([[1, 0, d], [0, 1, 0]])
    r = cv2.warpAffine(r, M, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    M2 = np.float32([[1, 0, -d], [0, 1, 0]])
    b = cv2.warpAffine(b, M2, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b, g, r])

def block_distortion(img, L):  # local block-wise corruption
    n = [4, 10, 20, 40, 80][L]
    out = img.copy(); h, w = img.shape[:2]; bs = 16
    for _ in range(n):
        y, x = np.random.randint(0, h - bs), np.random.randint(0, w - bs)
        out[y:y+bs, x:x+bs] = np.random.randint(0, 256, (bs, bs, 3), np.uint8)
    return out

DISTORTIONS = {
    "gnoise": gaussian_noise, "cnoise": color_noise, "impulse": impulse_noise,
    "mnoise": mult_noise, "quant": quantization, "blur": gaussian_blur,
    "jpeg": jpeg, "meanshift": mean_shift, "contrast": contrast,
    "saturate": saturation, "sharpen": sharpen, "chroma": chromatic_aberration,
    "block": block_distortion,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    srcs = []
    for dp, _, fns in os.walk(a.src):
        for fn in fns:
            if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                srcs.append(os.path.join(dp, fn))
    srcs.sort()
    if a.limit:
        srcs = srcs[:a.limit]
    man = open(os.path.join(a.out, "manifest.tsv"), "w")
    man.write("ref_path\tdist_path\tdist_type\tlevel\n")
    n = 0
    for sp in srcs:
        try:
            img = np.array(Image.open(sp).convert("RGB"))
        except Exception as e:
            print(f"  skip {sp}: {e}", file=sys.stderr); continue
        stem = os.path.splitext(os.path.basename(sp))[0]
        for dname, dfn in DISTORTIONS.items():
            for L in range(a.levels):
                try:
                    d = dfn(img, L)
                except Exception as e:
                    print(f"  fail {stem} {dname} L{L}: {e}", file=sys.stderr); continue
                op = os.path.join(a.out, f"{stem}__{dname}__L{L}.png")
                Image.fromarray(d).save(op)
                man.write(f"{sp}\t{op}\t{dname}\t{L}\n")
                n += 1
        if n % 1000 < len(DISTORTIONS) * a.levels:
            print(f"  {n} variants ({stem})")
    man.close()
    print(f"done: {n} distorted variants from {len(srcs)} sources -> {a.out}")

if __name__ == "__main__":
    main()
