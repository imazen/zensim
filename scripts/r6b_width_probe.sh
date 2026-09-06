#!/usr/bin/env bash
# R6b: does F17's blast radius vary with WIDTH or POOL STATE? Measure it.
#
# Four shapes, because F4's own answer varies with pool state (132 at 372 and at
# a pools-live 944, 36 at the zeroed ext944/ext924 roots) and a wave that assumed
# F17 behaved the same way would over- or under-declare it. MEASURED: it does not
# — the same twelve at `944full` (pools live), `924` (pools zeroed, the ext944
# class), `372` (v1-only) and `156` (basic-only).
#
# The registry DERIVES 12 slots at every layout (`contrast_inc` is a basic slot
# and every layout keeps the basic block) and `feature_defs`'s
# `f17_moves_exactly_the_twelve_contrast_inc_slots` gates that derivation. This
# measures it instead, because F4's own blast radius was understated by 60 slots
# for exactly as long as it was only derived: R6 found 132 where the audit had
# reasoned 72.
#
# One binary, arms selected at runtime, `to_bits()` dumps diffed. The synthetic
# pair keeps four of the twelve at exactly 0.0 (its Y channel), which is a free
# check on H4: every arm must leave those four untouched, so the moved set comes
# out as {the twelve} INTERSECT {nonzero} rather than {the twelve}.
#
# Usage: r6b_width_probe.sh [OUT_DIR] [W] [H]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/mnt/v/output/zensim/rev2-2026-09-05/r6b/widthprobe}"
W="${2:-700}"; H="${3:-500}"
EX="$REPO/target/release/examples/foldapp_stream_bigpair"
[ -x "$EX" ] || { echo "build it: cargo build --release -p zensim --example foldapp_stream_bigpair --features feature-regime-v2" >&2; exit 2; }
mkdir -p "$OUT"
for arm in ratio satexcess cap log1p bexcess; do
  for shape in 944full 924 372 156; do
    ZENSIM_HF_GAIN=$arm ZENSIM_BIGPAIR_TOGGLES=$shape ZENSIM_BIGPAIR_ITERS=2 \
      ZENSIM_BIGPAIR_DUMP="$OUT/w_${shape}_${arm}.tsv" \
      nice -n19 ionice -c3 "$EX" "$W" "$H" >/dev/null 2>&1
  done
done
python3 - "$OUT" <<'PY'
import sys
out = sys.argv[1]
bits = lambda p: [l.split("\t")[2].strip() for l in open(p)]
vals = lambda p: [float(l.split("\t")[1]) for l in open(p)]
exp = [c * 13 + 12 for c in range(12)]
for shape in ("944full", "924", "372", "156"):
    b, bv = bits(f"{out}/w_{shape}_ratio.tsv"), vals(f"{out}/w_{shape}_ratio.tsv")
    nz = [i for i in exp if bv[i] != 0.0]
    print(f"\n=== shape {shape}: {len(b)} emitted slots; "
          f"{len(nz)} of the 12 contrast_inc slots are nonzero here ===")
    for arm in ("satexcess", "cap", "log1p", "bexcess"):
        g = bits(f"{out}/w_{shape}_{arm}.tsv")
        moved = [i for i, (x, y) in enumerate(zip(b, g)) if x != y]
        inside = set(moved) <= set(exp)
        print(f"  {arm:10s} moved {len(moved):3d}  inside the twelve: {inside}  "
              f"zeros untouched: {not (set(moved) & (set(exp) - set(nz)))}  {moved}")
PY
