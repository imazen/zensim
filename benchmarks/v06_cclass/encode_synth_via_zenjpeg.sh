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

set -euo pipefail

SRC_DIR=${SRC_DIR:-/mnt/v/input/zensim/sources}
GEN_DIR=/mnt/v/input/zensim/sources_gen_v06rb
OUT_DIR=${OUT_DIR:-/mnt/v/output/zensim/v06-rebalance}
REMOTE_DIR=${REMOTE_DIR:-/mnt/v/input/zensim/images}
GENERATOR=${GENERATOR:-/home/lilith/work/coefficient/target/release/examples/generate_zensim_training}

[ -x "$GENERATOR" ] || { echo "missing $GENERATOR"; exit 1; }

# Build a curated source dir of symlinks pointing only at gen-* PNGs.
# The generator scans the --sources directory non-recursively.
mkdir -p "$GEN_DIR"
find "$GEN_DIR" -maxdepth 1 -type l -delete  # clean stale links

n=0
for f in "$SRC_DIR"/gen-*.png; do
  ln -sf "$f" "$GEN_DIR/$(basename "$f")"
  n=$((n + 1))
done
echo "linked $n gen-* sources into $GEN_DIR"

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
