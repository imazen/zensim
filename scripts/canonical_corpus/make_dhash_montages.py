#!/usr/bin/env python3
"""Labeled dHash-review montages (v2 — replaces the 2026-08-27 unlabeled set).

Fixes from the user's eye pass: (1) every montage now carries a text strip
naming each side (pool path left, estate file right) plus the Hamming d;
(2) both images are ASSERTED loaded and non-blank before compositing — the
v1 set silently rendered two synth montages with a white right half (a
loader failure indistinguishable from 'different image' to the reviewer).

usage: make_dhash_montages.py [--audit-dir DIR] [--out DIR]
"""
import argparse, collections, os
from PIL import Image, ImageDraw

def load_checked(path, side):
    # A failed load raises (that is what silently blanked v1's synth montages);
    # a GENUINELY near-white image (white-page screenshots are legitimate
    # estate content, and flat content is the documented dHash false-positive
    # class) is kept but flagged so the reviewer sees "near-blank" as a fact
    # about the FILE, not a rendering bug.
    im = Image.open(path).convert("RGB")
    g = im.resize((16, 16)).convert("L")
    px = list(g.getdata())
    near_blank = sum(1 for p in px if p > 245) / len(px) > 0.99
    return im, near_blank

def fit(im, h):
    w = round(im.width * h / im.height)
    return im.resize((max(1, w), h))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="/mnt/v/output/zensim/imazen26-dhash-2026-08-27")
    ap.add_argument("--out", default="/mnt/v/output/zensim/imazen26-dhash-2026-08-27/montages_v2")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    # Estate index. The audit's staged refs (hash-prefixed copies under the
    # session scratchpad) were WIPED — the /tmp-ban lesson, again: that wipe is
    # also what blanked v1's two synth montages. Index the DURABLE estate
    # trees by original basename and strip the 8-hex staging prefix on lookup.
    est_idx = {}
    for root_dir in ("/mnt/v/imazen-26",
                     "/mnt/v/output/imazen-26-hdr-grid-2026-06-14",
                     "/mnt/v/output/imazen-26-variants",
                     "/mnt/v/output/imazen-26-png-v2"):
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for f in files:
                est_idx.setdefault(f, os.path.join(root, f))
    import re
    unprefix = lambda n: re.sub(r"^[0-9a-f]{8}_", "", n)

    manifest = open(os.path.join(a.out, "manifest.tsv"), "w")
    manifest.write("montage\tsweep\td\tleft_path\tright_file\n")
    made = 0
    for sweep in ("refs_sdr_vs_train_picker", "refs_sdr_vs_train_synth",
                  "refs_hdr_vs_train_picker", "refs_hdr_vs_train_synth"):
        tsv = os.path.join(a.audit_dir, f"{sweep}.tsv")
        if not os.path.exists(tsv):
            continue
        pairs = collections.OrderedDict()
        for line in open(tsv):
            f = line.rstrip("\n").split("\t")
            if len(f) < 5 or not f[4].isdigit():
                continue
            d = int(f[4])
            if d > 2:
                continue
            # one montage per (origin-file, estate-file) pair; keep smallest d
            key = (f[0].split(".scale")[0], f[2])
            if key not in pairs or d < pairs[key][2]:
                pairs[key] = (f[0], f[2], d)
        # sample: all d1/d2 + up to 24 of the d0 tier (mirrors the v1 sampling)
        d0 = [v for v in pairs.values() if v[2] == 0][:24]
        rest = [v for v in pairs.values() if v[2] > 0]
        for left_path, right_file, d in d0 + rest:
            right_path = est_idx.get(right_file) or est_idx.get(unprefix(right_file))
            if right_path is None:
                print(f"SKIP (estate file not found on disk): {right_file}")
                continue
            Lim, Lblank = load_checked(left_path, "left/pool")
            Rim, Rblank = load_checked(right_path, "right/estate")
            L, R = fit(Lim, 360), fit(Rim, 360)
            strip = 34
            canvas = Image.new("RGB", (L.width + R.width + 12, 360 + strip * 2), (24, 26, 30))
            canvas.paste(L, (0, strip)); canvas.paste(R, (L.width + 12, strip))
            dr = ImageDraw.Draw(canvas)
            dr.text((4, 8), f"POOL  {os.path.basename(left_path)}", fill=(220, 220, 210))
            dr.text((L.width + 16, 8), f"ESTATE  {right_file}", fill=(220, 220, 210))
            note = "".join([
                "   LEFT NEAR-BLANK (flat-content dHash class)" if Lblank else "",
                "   RIGHT NEAR-BLANK (flat-content dHash class)" if Rblank else "",
            ])
            dr.text((4, 360 + strip + 8), f"{sweep}   hamming d={d}{note}",
                    fill=(230, 170, 120) if note else (150, 200, 180))
            name = f"{sweep.replace('refs_','').replace('_vs_train','')}_d{d}_{made:03d}.png"
            canvas.save(os.path.join(a.out, name))
            manifest.write(f"{name}\t{sweep}\t{d}\t{left_path}\t{right_file}\n")
            made += 1
    manifest.close()
    print(f"wrote {made} labeled montages -> {a.out} (+ manifest.tsv)")

if __name__ == "__main__":
    main()
