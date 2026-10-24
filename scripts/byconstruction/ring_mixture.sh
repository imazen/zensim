#!/bin/bash
# Label-free damage-mixture test for every cell of the given arms, scored under ZENSIM_RING_FORM=FORM with the
# given scorer build. Exploration record: benchmarks/byconstruction_2026-10-22.md.
# Usage: ring_mixture.sh BIN_DIR FORM JOBS ARM [ARM ...]; MIX selects another mixture set (default: the standard one).
set -euo pipefail
BIN=$1; FORM=$2; J=$3; shift 3
H=$(cd "$(dirname "$0")" && pwd)
MIX=${MIX:-/mnt/v/output/zensim/byconstruction-2026-10-22/mixture}
for arm in "$@"; do
  for d in "$HOME"/tmp/byc/cells/"$arm"/without_*_s*; do
    cell=$(basename "$d"); cell=${cell#without_}
    bake=$(ls "$d"/dense/*.bin | head -1)
    ZENSIM_RING_FORM=$FORM RAYON_NUM_THREADS=1 python3 "$H/mixture_test.py" score --out "$MIX" \
      --bake "${arm}_${cell}=$bake" --bin-dir "$BIN" --jobs "$J"
  done
done
