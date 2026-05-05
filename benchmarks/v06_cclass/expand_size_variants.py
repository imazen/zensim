#!/usr/bin/env python3
"""Expand each gen-* 1024sq synthetic source into the full 6-size augmentation set.

Inputs:  /mnt/v/input/zensim/sources/gen-*_1024sq.png
Outputs: gen-*_512sq.png (already exist, skip), gen-*_513x769.png, gen-*_769x513.png,
         gen-*_818x1022.png, gen-*_1022x818.png.

Resampling: Lanczos (PIL Image.Resampling.LANCZOS). For non-square targets we
center-crop the 1024sq image to the target aspect ratio first, then resize.
This matches the safe-synthetic-v2 pipeline (which used CLIC 1024sq sources
center-cropped to the target).

Updates the sidecar TSV to add new source paths.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

from PIL import Image

SRC_DIR = Path("/mnt/v/input/zensim/sources")
SIDECAR = Path("/mnt/v/output/zensim/v06-rebalance/synth_sources.tsv")
LICENSE = "CC0-1.0"

# The 6 standard size buckets used by the safe-synthetic pipeline:
#   512sq, 1024sq, 769x513, 513x769, 1022x818, 818x1022
# We start from 1024sq, derive the others.
SIZES = [
    ("512sq", (512, 512)),
    ("769x513", (769, 513)),
    ("513x769", (513, 769)),
    ("1022x818", (1022, 818)),
    ("818x1022", (818, 1022)),
]


def center_crop_to_aspect(img: Image.Image, tw: int, th: int) -> Image.Image:
    sw, sh = img.size
    target_ar = tw / th
    src_ar = sw / sh
    if abs(src_ar - target_ar) < 1e-6:
        return img
    if src_ar > target_ar:
        # Source wider — crop sides
        new_w = int(round(sh * target_ar))
        x0 = (sw - new_w) // 2
        return img.crop((x0, 0, x0 + new_w, sh))
    else:
        new_h = int(round(sw / target_ar))
        y0 = (sh - new_h) // 2
        return img.crop((0, y0, sw, y0 + new_h))


def derive_from_1024sq(p: Path) -> int:
    """Return number of new files written."""
    n_new = 0
    img = Image.open(p).convert("RGB")
    base = p.name.removesuffix("_1024sq.png")
    for label, (tw, th) in SIZES:
        out = SRC_DIR / f"{base}_{label}.png"
        if out.exists():
            continue
        cropped = center_crop_to_aspect(img, tw, th)
        resized = cropped.resize((tw, th), Image.Resampling.LANCZOS)
        resized.save(out, optimize=True, compress_level=6)
        n_new += 1
    return n_new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    sources_1024sq = sorted(SRC_DIR.glob("gen-*_1024sq.png"))
    if args.limit:
        sources_1024sq = sources_1024sq[: args.limit]
    print(f"found {len(sources_1024sq)} gen-* 1024sq sources")

    n_total_new = 0
    for i, p in enumerate(sources_1024sq):
        n_total_new += derive_from_1024sq(p)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sources_1024sq)} processed; {n_total_new} new files")

    print(f"done: {n_total_new} new size-variant files written")

    # Rebuild sidecar to include the derived variants
    rows = []
    if SIDECAR.exists():
        with SIDECAR.open() as f:
            header = f.readline()
            for ln in f:
                if not ln.strip():
                    continue
                rows.append(ln.rstrip("\n").split("\t"))

    # Convert each row to also include all derived size variants for that base
    new_rows = []
    seen_paths = set()
    for r in rows:
        new_rows.append(r)
        seen_paths.add(r[0])
        # Derive parent base name
        path = Path(r[0])
        if "_1024sq.png" in path.name:
            base = path.name.removesuffix("_1024sq.png")
            for label, _ in SIZES:
                out = SRC_DIR / f"{base}_{label}.png"
                if out.exists() and str(out) not in seen_paths:
                    new_rows.append([str(out), r[1], r[2], r[3], r[4]])
                    seen_paths.add(str(out))

    with SIDECAR.open("w") as f:
        f.write("source_path\tcontent_class\tsubset\tseed\tlicense\n")
        for r in new_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    print(f"updated sidecar with {len(new_rows)} rows")

    from collections import Counter
    c = Counter(r[1] for r in new_rows)
    print("class totals:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
