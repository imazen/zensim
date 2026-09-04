#!/bin/bash
# fastclass wave — LABELLED EXTRA ARM G1 (class-C free slots), added 2026-09-04
# after the class-C lane landed and BEFORE any G1 result existed.
#
# A separate driver on purpose: the frozen 7-arm driver is RUNNING, and editing
# a script bash is still reading is a corruption hazard. This one waits for the
# frozen set to finish, then trains G1 with the ONLY change being the input
# slice (289 coords = the 265 this wave's arms use + the 24 class-C slots).
#
# Registration: benchmarks/fastclass_distill_wave_2026-09-04.md §6g (AMENDMENT A3).
set -euo pipefail
REPO="${FCD_REPO:-/home/lilith/work/zen/zensim}"
cd "$REPO"
export ZL_TRAIN=/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free_classc.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
O=/mnt/v/output/zensim/fastclass-2026-09-04
export WR4_OUT="$O/bakes"; export WR4_SCORE="$O"
mkdir -p "$WR4_OUT"
HB="${FCD_HB:-$HOME/tmp/fastclass/classc}"
mkdir -p "$(dirname "$HB")"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$HB.log"; }
[ -f "$WR4_KEEP" ] || { say "ABORT: slice missing $WR4_KEEP"; exit 2; }

# Wait for the frozen 7-arm driver so nothing runs concurrently (CLAUDE.md).
if [ -n "${FCD_WAIT_FOR:-}" ]; then
  say "waiting for the frozen set: $FCD_WAIT_FOR"
  while ! grep -q 'ALL ARMS COMPLETE' "$FCD_WAIT_FOR" 2>/dev/null; do sleep 60; done
  say "frozen set complete — starting G1"
fi

for SEED in ${FCD_SEEDS:-4004 4005 4006}; do
  NAME="G1_s${SEED}"; OUT="$WR4_OUT/${NAME}.bin"
  if [ -f "$O/${NAME}.fulleval.json" ]; then say "SKIP (scored): $NAME"; continue; fi
  if [ ! -f "$OUT" ]; then
    say "TRAIN $NAME (289-coord class-C slice)"
    "$HOME/work/zen/scripts/run-heavy" --mem 16G --jobs 8 -- \
      "$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh" distill "$SEED" "$OUT" \
      >>"$HB.train.log" 2>&1 || { say "TRAIN FAILED $NAME"; touch "$O/${NAME}.TRAIN_FAILED"; continue; }
  fi
  say "SCORE $NAME"
  "$REPO/benchmarks/wave_r4_2026-09-01/score_arm.sh" "$OUT" "$NAME" 944 >>"$HB.score.log" 2>&1 \
    || { say "SCORE FAILED $NAME"; touch "$O/${NAME}.HARVEST_FAILED"; continue; }
  say "DONE $NAME"
done
say "CLASS-C ARM COMPLETE"
