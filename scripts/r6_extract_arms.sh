#!/usr/bin/env bash
# R6: extract every 372-col table at one F4 luminance arm.
#
# ONE binary, four arms. The arm is chosen at RUNTIME with `ZENSIM_SSIM_LUMA`
# (`ssim_form::active_luma_form`'s measurement override), never by a rebuild:
# this repo has measured a rebuild alone moving a 2304^2 timing ~10 %, and a
# rebuild between arms would put the same class of confound into the FEATURES.
#
# Corpora and invocations are the ones `build_eval372_root.sh` already owns —
# same datasets, same flags, same output filenames — so an arm's directory is a
# drop-in `--features-root` for `bake_verdict` and the `ssim2` arm is a
# bit-exact reproduction control against the registered postC root.
#
# Usage: r6_extract_arms.sh <arm> [OUT_ROOT] [--eval-only|--safesyn-only]
set -euo pipefail

ARM="${1:?usage: r6_extract_arms.sh <ssim2|c1|lorentz|clamp> [OUT] [--eval-only|--safesyn-only]}"
case "$ARM" in ssim2|c1|lorentz|clamp) ;; *) echo "bad arm: $ARM" >&2; exit 2;; esac
OUT="${2:-/mnt/v/output/zensim/rev2-2026-09-05/r6/tables}/$ARM"
MODE="${3:-all}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZV="$REPO/target/release/zensim-validate"
EX="$REPO/zensim-bench/target/release/examples/extract_features_372col"
SAFESYN_TSV="${SAFESYN_TSV:-$HOME/tmp/r6/safesyn_full.tsv}"
for b in "$ZV" "$EX"; do [ -x "$b" ] || { echo "missing binary: $b" >&2; exit 2; }; done
mkdir -p "$OUT"

export ZENSIM_SSIM_LUMA="$ARM"
say() { printf '[%s] %s %s\n' "$(date -u +%H:%M:%S)" "$ARM" "$*"; }

zv() { say "$1"; nice -n19 ionice -c3 "$ZV" --dataset "$2" --format "$3" --extract-only \
        --extended-features --iw-features --recompute --features-csv "$OUT/$1.csv" >/dev/null; }
ex() { say "$1"; nice -n19 ionice -c3 "$EX" --corpus "$2" --path "$3" --out "$OUT/$1.csv" >/dev/null; }

if [ "$MODE" != "--safesyn-only" ]; then
  # NOTE (inherited from build_eval372_root.sh, learned the hard way): the CID22
  # *validation set* is a SUBDIR. Pointing at the parent yields "0 valid pairs"
  # with rc=0 and a 34-byte cache.
  zv cid22 /mnt/v/dataset/cid22/CID22_validation_set cid22
  zv kadid /mnt/v/dataset/kadid10k                   kadid10k
  zv tid   /mnt/v/dataset/tid2013                    tid2013
  ex konjnd konjnd    /mnt/v/datasets/KonJND-1k/KonJND-1k
  ex aic3   aic3      /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv
  ex csiq   pairs-tsv /mnt/v/dataset/csiq/csiq_pairs.tsv
  ex live   pairs-tsv /mnt/v/datasets/LIVE/live_r2_pairs.tsv
fi
if [ "$MODE" != "--eval-only" ]; then
  ex safesyn pairs-tsv "$SAFESYN_TSV"
fi
say "done -> $OUT"
wc -l "$OUT"/*.csv
