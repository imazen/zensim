#!/usr/bin/env bash
# Encode the gen-* synthetic non-photo references through zenjpeg-420-e1
# at the same quality grid as the e1 fill (training.csv) so the rows merge
# cleanly with the existing extended CSV.
#
# Pipeline:
#   1. Source: /mnt/v/input/zensim/sources/gen-*.png
#   2. Encode via the prebuilt generate_zensim_training binary at:
#        /home/lilith/work/coefficient/target/release/examples/generate_zensim_training
#   3. Output rows append to /mnt/v/output/zensim/training_v06_rebalance.csv
#
# Filter to ONLY the gen-* sources by passing a curated --sources directory.
# Otherwise the binary would re-encode all 4653 existing photo sources.
#
# Usage: bash encode_synth_via_zenjpeg.sh

# ============================================================================
# Salvaged from parked v06-rebalanced-corpus branch (af957010, 2026-05-05).
# Status on main: working, with caveats documented below.
#
# What this shell does:
#   1. Builds a curated symlink dir at /mnt/v/input/zensim/sources_gen_v06rb_raw/
#      pointing only at gen-*_1024sq.png (suffix stripped -- needed to bypass
#      the encoder's recursive-retile guard which rejects pre-cropped tiles).
#   2. Invokes coefficient's generate_zensim_training binary with
#      --codecs zenjpeg-420-e1 against that curated dir.
#   3. Output rows append to /mnt/v/output/zensim/v06-rebalance/training.csv
#      and encoded variants land at /mnt/v/input/zensim/images/<ref>/<codec>/q*.
#
# CAVEAT 1 (encoder location): The GENERATOR= path assumes you've built
# coefficient/examples/generate_zensim_training. Build with:
#   cd ~/work/coefficient && cargo build --release --features gpu,zenavif,zenjxl,zenwebp \
#                                          --example generate_zensim_training
#
# CAVEAT 2 (q-grid): This shell only invokes zenjpeg-420-e1. To reproduce the
# FULL safesyn corpus, you need to invoke for each of the 7 codec configs
# generate_zensim_training builds (mozjpeg-rs-420-e4, zenjpeg-420-{e1,e2,xyb-e2},
# zenwebp-default-m4, zenavif-s5-e6, zenjxl-e7). See coefficient/examples/
# generate_zensim_training.rs:build_codecs() for the canonical list. Pass
# `--codecs <name>` to filter, or omit for all-7.
#
# CAVEAT 3 (per-crop audit 2026-05-26): The encoder will produce 6 size-bucket
# variants per gen-* ref (512sq, 1024sq, 769x513, 513x769, 1022x818, 818x1022).
# Per the per-crop audit, the 4 boundary-aspect variants (the non-square ones)
# add no measurable training signal vs 512sq/1024sq. For new training runs,
# consider patching generate_zensim_training.rs:SIZE_BUCKETS to drop the
# 4 aspect-ratio entries.
#
# CAVEAT 4 (reproducibility): For full (ref, dist) reproducibility you also
# need the source PNGs. They're at /mnt/v/input/zensim/sources/ (mirrored
# to s3://codec-corpus/synthetic-v2/ as of 2026-05-22). Regenerate via
# python3 ../synth_nonphoto.py --seed 20260505 (same default seed as the
# original 2026-05-05 run).
#
# See ~/work/zen/_ml-inventory-2026-05-20/09-data-pipeline-design.md for
# the layered-store design that supersedes this ad-hoc shell pipeline.
# ============================================================================

set -euo pipefail

SRC_DIR=${SRC_DIR:-/mnt/v/input/zensim/sources}
GEN_DIR=/mnt/v/input/zensim/sources_gen_v06rb_raw
OUT_DIR=${OUT_DIR:-/mnt/v/output/zensim/v06-rebalance}
REMOTE_DIR=${REMOTE_DIR:-/mnt/v/input/zensim/images}
GENERATOR=${GENERATOR:-/home/lilith/work/coefficient/target/release/examples/generate_zensim_training}

[ -x "$GENERATOR" ] || { echo "missing $GENERATOR"; exit 1; }

# Build a curated source dir of symlinks pointing only at gen-*_1024sq.png,
# renamed to drop the bucket suffix. The encoder rejects any file whose stem
# ends with a known bucket label (treats it as a pre-cropped tile and refuses
# to re-tile to prevent recursion). By feeding it the 1024sq sources without
# the suffix, the encoder treats each as a raw source and tiles into all 6
# buckets at the e1 q-grid — same shape as the existing safe-synthetic CSV.
mkdir -p "$GEN_DIR"
find "$GEN_DIR" -maxdepth 1 -type l -delete  # clean stale links

n=0
for f in "$SRC_DIR"/gen-*_1024sq.png; do
  bn=$(basename "$f")
  bare="${bn%_1024sq.png}"
  ln -sf "$f" "$GEN_DIR/${bare}.png"
  n=$((n + 1))
done
echo "linked $n gen-*_1024sq sources (suffix stripped) into $GEN_DIR"

mkdir -p "$OUT_DIR"

# Match the e1 fill quality grid. zenjpeg-420-e1 only.
TS=$(date -u +%Y%m%dT%H%M%S)
LOG=/tmp/v06_rebalance_encode_${TS}.log
echo "encoding to $OUT_DIR (codec: zenjpeg-420-e1)"
echo "log: $LOG"

"$GENERATOR" \
  --sources "$GEN_DIR" \
  --output "$OUT_DIR" \
  --remote "$REMOTE_DIR" \
  --codecs zenjpeg-420-e1 \
  > "$LOG" 2>&1

echo "encoding done"
tail -20 "$LOG"
