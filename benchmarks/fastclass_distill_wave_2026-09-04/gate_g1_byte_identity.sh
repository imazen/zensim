#!/bin/bash
# GATE G1 (fastclass wave, run BEFORE any arm): the extended
# train_156_student.sh, with every new lever UNSET, must reproduce A4b's own
# bake BYTE-IDENTICALLY. If it does not, the four new env vars are not the
# no-ops they claim to be and every Δ-vs-control in this wave would be
# measuring the script edit instead of the arm.
set -euo pipefail
REPO="${FCD_REPO:-/home/lilith/work/zen/zensim}"
cd "$REPO"
export ZL_TRAIN=/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
O=/mnt/v/output/zensim/fastclass-2026-09-04
export WR4_OUT="$O/bakes"
mkdir -p "$WR4_OUT"
REF=/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A4b_156_s4004.bin
OUT="$WR4_OUT/C0_s4004.bin"
if [ ! -f "$OUT" ]; then
  "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
    "$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh" distill 4004 "$OUT"
fi
A=$(sha256sum "$REF" | cut -d' ' -f1)
B=$(sha256sum "$OUT" | cut -d' ' -f1)
echo "A4b_156_s4004.bin  $A"
echo "C0_s4004.bin       $B"
if [ "$A" = "$B" ]; then echo "GATE G1: PASS (byte-identical)"; exit 0; fi
echo "GATE G1: FAIL — the script extension is NOT a no-op when unset"; exit 1
