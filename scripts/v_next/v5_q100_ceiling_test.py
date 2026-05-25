#!/usr/bin/env python3
"""Sanity-check: what score does v5 produce at q=100 (lossy-max) for each codec?
Uses PIL for JPEG/WebP encoding (libjpeg-turbo / libwebp underneath)."""
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path("/home/lilith/work/zen/zensim")
BAKE = REPO / "zensim/weights/v_tuner_v11_2026-05-24.bin"
SOURCE_DIR = Path("/mnt/v/zen/picker-training/2026-05-19")
SRC_LIST = sorted(Path("/mnt/v/input/zensim/sources").glob("*_512sq.png"))[:20]
ZENSIM_BIN = REPO / "target/release/predict_features_with_bake"

# We need to decode the codec output back to RGB, then extract zensim features,
# then score. To avoid the feature-extraction step, just use the existing
# v_tuner_v11_2026-05-24 packed score via the full Zensim::compute path —
# easier to do via the zensim library's Python bindings... wait, no Python
# bindings exist. Let me use zen-metrics CLI which can score (ref, distorted).

ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zen-metrics")

def main():
    if not ZEN_METRICS.exists():
        print(f"zen-metrics not built at {ZEN_METRICS}")
        return 1
    # Encode each source at q=100 for JPEG and WebP via PIL, then use
    # zen-metrics batch --metric zensim to score (ref, encoded-decoded).
    work = Path(tempfile.mkdtemp(prefix="v5_q100_"))
    pairs_tsv = work / "pairs.tsv"
    rows = []
    for src in SRC_LIST[:20]:
        img = Image.open(src).convert("RGB")
        for codec, ext, save_kwargs in [
            ("jpeg_q100", "jpg", {"format": "JPEG", "quality": 100, "subsampling": 0}),
            ("webp_q100", "webp", {"format": "WebP", "quality": 100, "method": 6}),
            ("webp_lossless", "webp", {"format": "WebP", "lossless": True, "method": 6}),
        ]:
            out_path = work / f"{src.stem}_{codec}.{ext}"
            img.save(out_path, **save_kwargs)
            decoded = work / f"{src.stem}_{codec}_decoded.png"
            Image.open(out_path).convert("RGB").save(decoded)
            rows.append((src, decoded, codec))

    with open(pairs_tsv, "w") as f:
        f.write("ref_path\tdist_path\tcodec\n")
        for ref, dist, codec in rows:
            f.write(f"{ref}\t{dist}\t{codec}\n")
    print(f"wrote {len(rows)} pairs to {pairs_tsv}")

    # Score via zen-metrics zensim with the v5 packed bake. The metric will
    # extract features + forward through bake + return score.
    out_tsv = work / "scores.tsv"
    cmd = [
        str(ZEN_METRICS),
        "batch", "--metric", "zensim",
        "--pairs", str(pairs_tsv),
        "--output", str(out_tsv),
    ]
    print(f"running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"zen-metrics failed: {e.stderr}")
        return 2

    # Parse results.
    import csv
    by_codec = {}
    with open(out_tsv) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score_col = next((k for k in row if k.startswith("zensim")), None)
            if not score_col:
                continue
            by_codec.setdefault(row["codec"], []).append(float(row[score_col]))

    print()
    print("=== q=100 (or lossless) ceiling per codec ===")
    print(f"{'codec':<20} n  median  p25  p75  min  max")
    for codec, scores in by_codec.items():
        arr = np.array(scores)
        print(f"{codec:<20} {len(arr)}  {np.median(arr):.2f}  "
              f"{np.quantile(arr, 0.25):.2f}  {np.quantile(arr, 0.75):.2f}  "
              f"{arr.min():.2f}  {arr.max():.2f}")

if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
