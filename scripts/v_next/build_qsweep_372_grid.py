#!/usr/bin/env python3
"""Build a multi-codec q-sweep grid TSV for the dial-dynamic-range + codec-reach
eval (qsweep_eval). 2026-05-29.

Reuses the on-disk decoded codec/q variants at /mnt/v/input/zensim/images/<src>/
<codec>/q<N>.png (no re-encoding) + source refs at /mnt/v/input/zensim/sources/.
Emits a TSV in the format both `extract_features_372col --corpus qsweep` and
`qsweep_eval --manifest` consume:

    ref_path <TAB> dist_path <TAB> image_id <TAB> codec <TAB> q

One clean engine per codec family (JPEG/WebP/JXL/AVIF) so cross-codec
equivalence (G4) compares families, not encoder-effort knobs.

Pipeline:
  python3 scripts/v_next/build_qsweep_372_grid.py --out grid.tsv --n-images 40
  ./target/release/examples/extract_features_372col --corpus qsweep \
      --path grid.tsv --out grid_features.csv
  ./target/release/qsweep_eval --features grid_features.csv --manifest grid.tsv \
      --bake A=zensim/weights/v47_strict_qat_native_2026-05-27.bin:clamp \
      --bake Cell5=zensim/weights/v02_372feat_cell5_2026-05-28.bin:clamp \
      --out dialreach.md

A 300-input bake (Balanced/Compression) scores this 372-feature grid fine —
bake_runtime takes min(n_inputs, row.len()) = its f0..f299 prefix.
"""
from __future__ import annotations
import argparse, os, sys

SRCROOT = "/mnt/v/input/zensim/sources"
IMGROOT = "/mnt/v/input/zensim/images"
# codec dir -> family name (one engine per family for clean cross-codec comparison)
CODECS = {
    "mozjpeg-rs-420-e4": "jpeg",
    "zenwebp-default-m4": "webp",
    "zenjxl-e7": "jxl",
    "zenavif-s5-e6": "avif",
}
QS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 87, 90, 95, 100]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/qsweep_372_grid.tsv")
    ap.add_argument("--n-images", type=int, default=40)
    ap.add_argument("--codecs", default=",".join(CODECS), help="comma-list of codec dir names")
    args = ap.parse_args()
    codec_dirs = args.codecs.split(",")
    fam = {c: CODECS.get(c, c) for c in codec_dirs}

    all_srcs = sorted(os.listdir(IMGROOT))
    eligible = [
        s for s in all_srcs
        if os.path.exists(f"{SRCROOT}/{s}.png")
        and all(os.path.isdir(f"{IMGROOT}/{s}/{c}") for c in codec_dirs)
    ]
    stride = max(1, len(eligible) // args.n_images)
    sel = eligible[::stride][: args.n_images]
    print(f"eligible={len(eligible)} selected={len(sel)}", file=sys.stderr)

    rows = []
    for s in sel:
        ref = f"{SRCROOT}/{s}.png"
        for cdir in codec_dirs:
            for q in QS:
                dist = f"{IMGROOT}/{s}/{cdir}/q{q}.png"
                if os.path.exists(dist):
                    rows.append(f"{ref}\t{dist}\t{s}\t{fam[cdir]}\t{q}")
    with open(args.out, "w") as f:
        f.write("ref_path\tdist_path\timage_id\tcodec\tq\n")
        f.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} pairs to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
