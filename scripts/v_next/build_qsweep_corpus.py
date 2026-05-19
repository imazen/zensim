#!/usr/bin/env python3
"""Build a q-sweep evaluation corpus for the PreviewV0_5Tuner experiment
(2026-05-18). Picks N source PNGs, encodes JPEG at q ∈ [5..95 step 5],
decodes back to PNG, and emits a TSV manifest with columns:

    ref_path  dist_path  image_id  codec  q

The TSV is consumed by `extract_features_372col --corpus qsweep` to
produce a 372-feature CSV that bake_verdict (or this experiment's
custom monotonicity-eval) can score.

Selection: alphabetical-sorted set of `_512sq.png` files from
`/mnt/v/input/zensim/sources/`, evenly strided to give a representative
N-sample subset. _512sq is chosen because every source has it (uniform
size, makes cjpeg/djpeg straightforward).
"""

import io
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from PIL import Image

SRC_DIR = Path("/mnt/v/input/zensim/sources")
OUT_DIR = Path("/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep")
N_SOURCES = 50
Q_VALUES = list(range(5, 100, 5))  # 5,10,15,...,95 → 19 values


def encode_one(args):
    src_path, image_id, q = args
    jpeg_path = OUT_DIR / f"{image_id}_q{q:03d}.jpg"
    dist_png = OUT_DIR / f"{image_id}_q{q:03d}.png"
    if not dist_png.exists():
        img = Image.open(src_path).convert("RGB")
        # JPEG encode via PIL → libjpeg, 4:2:0 chroma, baseline.
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, subsampling=2, optimize=False, progressive=False)
        # Write jpeg if we want it on disk too (skip for cache speed unless asked).
        if not jpeg_path.exists():
            with open(jpeg_path, "wb") as f:
                f.write(buf.getvalue())
        # Decode back via PIL.
        buf.seek(0)
        dist = Image.open(buf).convert("RGB")
        dist.save(dist_png, "PNG")
    return (str(src_path), str(dist_png), image_id, "jpeg420", q)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_512 = sorted(p for p in SRC_DIR.glob("*_512sq.png"))
    primary = [p for p in all_512 if "_512sq_" not in p.name]
    if len(primary) < N_SOURCES:
        print(f"FATAL: only {len(primary)} _512sq sources available", file=sys.stderr)
        return 1
    step = max(1, len(primary) // N_SOURCES)
    selected = primary[::step][:N_SOURCES]
    print(f"selected {len(selected)} sources (every {step}th of {len(primary)})", file=sys.stderr)

    tasks = []
    for src in selected:
        image_id = src.stem.replace("_512sq", "")
        for q in Q_VALUES:
            tasks.append((str(src), image_id, q))

    print(f"queueing {len(tasks)} (src,q) tasks", file=sys.stderr)

    manifest_rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        for i, row in enumerate(ex.map(encode_one, tasks, chunksize=16)):
            manifest_rows.append(row)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(tasks)}", file=sys.stderr)

    manifest_path = OUT_DIR / "qsweep_manifest.tsv"
    with open(manifest_path, "w") as f:
        f.write("ref_path\tdist_path\timage_id\tcodec\tq\n")
        for r in manifest_rows:
            f.write("\t".join(str(c) for c in r) + "\n")
    print(f"wrote {manifest_path} ({len(manifest_rows)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
