#!/usr/bin/env bash
# Build a DATED 372-col eval root with the CURRENT extractor.
#
# WHY: the 2026-05-15 root's masked/IW block (f228..371) was a function of
# RAYON_NUM_THREADS and does not reproduce at its own build commit — see
# benchmarks/v1_extractor_drift_2026-08-30.md + docs/DATASET_HISTORY.md §3.27.
# The old root is NEVER overwritten; this writes a NEW dated root so a verdict
# can be taken on either era by swapping `--features-root`.
#
# FILE NAMES ARE DELIBERATELY THE OLD ONES. `bake_verdict`'s CORPORA table
# hardcodes each corpus's filename, so a drop-in root must reuse them; the ROOT
# directory carries the date, not the files.
#
# Usage: scripts/canonical_corpus/build_eval372_root.sh [OUT_ROOT]
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-/mnt/v/zen/zensim-training/2026-08-30-full-features-372}"
WORK="${WORK:-$HOME/tmp/eval372root}"
ZV="$REPO/target/release/zensim-validate"
EX="$REPO/zensim-bench/target/release/examples/extract_features_372col"

mkdir -p "$OUT" "$WORK"
for b in "$ZV" "$EX"; do
  [ -x "$b" ] || { echo "missing binary: $b" >&2; exit 2; }
done

zv() { # zv <name> <dataset> <format>
  echo "=== $1 (zensim-validate --format $3) ==="
  "$ZV" --dataset "$2" --format "$3" --extract-only \
        --extended-features --iw-features --recompute \
        --features-csv "$WORK/$1.csv"
}
ex() { # ex <name> <corpus> <path>
  echo "=== $1 (extract_features_372col --corpus $2) ==="
  "$EX" --corpus "$2" --path "$3" --out "$WORK/$1.csv"
}

zv cid22 /mnt/v/dataset/cid22           cid22
zv kadid /mnt/v/dataset/kadid10k        kadid10k
zv tid   /mnt/v/dataset/tid2013         tid2013
zv pipal /mnt/v/dataset/pipal           pipal
ex konjnd konjnd    /mnt/v/datasets/KonJND-1k/KonJND-1k
ex aic3   aic3      /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv
ex csiq   pairs-tsv /mnt/v/dataset/csiq/csiq_pairs.tsv
ex live   pairs-tsv /mnt/v/datasets/LIVE/live_r2_pairs.tsv

wc -l "$WORK"/*.csv
echo "CSVs in $WORK — convert + manifest with scripts/canonical_corpus/pack_eval372_root.py"
