#!/bin/bash
# fastclass2 (2026-09-05) — SET x WIDTH x HEAD sweep on the fast class.
# Registration: docs/PLAN_FASTCLASS2_2026-09-05.md (committed BEFORE the first
# fit). Sets, widths, seeds, base recipe and gates are frozen there.
#
# Reuses the wave-r4 owners verbatim (train_156_student.sh + score_arm.sh) and
# the wave-r4 ROOT, so every cell reads byte-identical features to FC_D3's own
# — the control this campaign is read against. Nothing here computes a
# statistic; every number comes from bake_dial_refit (pack) and bake_verdict.
#
# Each bake is SCORED the moment it lands (playbook step 4), so a late wake-up
# costs review latency only.
set -euo pipefail
REPO="${FC2_REPO:-/home/lilith/work/zen/zensim--fastclass2}"
cd "$REPO"
BIN="${FC2_BIN:-/mnt/v/zen/cargo-targets/fastclass2/release}"
export ZL_TRAIN="$BIN/zensim_mlp_train"
export ZL_BIN="$BIN"                       # score_arm.sh honours this (2026-09-05)
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
# base recipe = FC_D3 (plan section 4): A4b + both within-ref ladders
export WR4_KON_WITHINREF=1
export WR4_HF_WITHINREF=1
O="${FC2_OUT:-/mnt/v/output/zensim/fastclass2-2026-09-05}"
export WR4_OUT="$O/bakes"
export WR4_SCORE="$O"
mkdir -p "$WR4_OUT" "$O"
TRAIN_SH="$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh"
SCORE_SH="$REPO/benchmarks/wave_r4_2026-09-01/score_arm.sh"
RH="$HOME/work/zen/scripts/run-heavy"
HB="${FC2_HB:-$HOME/tmp/fastclass2/run}"
mkdir -p "$(dirname "$HB")"
S="$REPO/scripts/sota944"

# SORACLE is the COMPUTE-CEILING instrument (plan Phase A-ORACLE): WR4_KEEP is
# left UNSET, so the trainer runs --max-features 944 with no --keep-features and
# the model may read all 944 coordinates. Not a ship candidate -- it prices at
# the full 944 walk by construction.
slice_of() {
  case "$1" in
    SORACLE) echo "" ;;
    S156) echo "$S/slice_basic156.txt" ;;
    S228) echo "$S/slice_basic156_peaks.txt" ;;
    S261) echo "$S/slice_basic156_free_nolumaref.txt" ;;
    S265) echo "$S/slice_basic156_free.txt" ;;
    S289) echo "$S/slice_basic156_free_classc.txt" ;;
    *) echo "ABORT: unknown set $1" >&2; exit 2 ;;
  esac
}

# CELLS is a whitespace list of SET:WIDTH:HEAD triples; HEAD in {p,a1,a2,sk}
#   p  = plain path, depth 1           (the control head)
#   a1 = --per-sample-alpha-head, depth 1
#   a2 = --per-sample-alpha-head --n-hidden-layers 2
#   sk = plain path + --skip-connection
CELLS="${FC2_CELLS:?FC2_CELLS required, e.g. 'S265:128:p S265:32:p'}"
SEEDS="${FC2_SEEDS:-4004 4005 4006}"

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$HB.log"; }
say "START cells=[$CELLS] seeds=[$SEEDS] bin=$BIN"
for CELL in $CELLS; do
  SET="${CELL%%:*}"; REST="${CELL#*:}"; W="${REST%%:*}"; HEAD="${REST#*:}"
  KEEP=$(slice_of "$SET")
  if [ -n "$KEEP" ]; then [ -f "$KEEP" ] || { say "ABORT: missing slice $KEEP"; exit 2; }; fi
  for SEED in $SEEDS; do
    NAME="F2_${SET}_H${W}_${HEAD}_s${SEED}"
    OUT="$WR4_OUT/${NAME}.bin"
    if [ -f "$O/${NAME}.fulleval.json" ]; then say "SKIP (scored): $NAME"; continue; fi
    if [ ! -f "$OUT" ]; then
      say "TRAIN $NAME"
      (
        unset WR4_HIDDEN WR4_ALPHA_HEAD WR4_N_HIDDEN_LAYERS WR4_SKIP WR4_FULL944 || true
        if [ -n "$KEEP" ]; then export WR4_KEEP="$KEEP"
        else unset WR4_KEEP || true; export WR4_FULL944=1; fi
        # H128 is the trainer's own default: the flag stays OMITTED so the cell's
        # argv is byte-identical to a pre-2026-09-05 run (gate G1).
        [ "$W" = 128 ] || export WR4_HIDDEN="$W"
        case "$HEAD" in
          p)  : ;;
          a1) export WR4_ALPHA_HEAD=1 ;;
          a2) export WR4_ALPHA_HEAD=1 WR4_N_HIDDEN_LAYERS=2 ;;
          sk) export WR4_SKIP=1 ;;
          *)  echo "ABORT: unknown head $HEAD"; exit 2 ;;
        esac
        "$RH" --mem 16G --jobs 8 -- "$TRAIN_SH" distill "$SEED" "$OUT"
      ) >>"$HB.train.log" 2>&1 || { say "TRAIN FAILED $NAME"; touch "$O/${NAME}.TRAIN_FAILED"; continue; }
    else
      say "SKIP (bake exists): $NAME"
    fi
    say "SCORE $NAME"
    "$SCORE_SH" "$OUT" "$NAME" 944 >>"$HB.score.log" 2>&1 \
      || { say "SCORE FAILED $NAME"; touch "$O/${NAME}.HARVEST_FAILED"; continue; }
    say "DONE $NAME"
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$NAME" >> "$HB.status"
  done
done
say "ALL CELLS COMPLETE"
touch "$O/PHASE.done"
