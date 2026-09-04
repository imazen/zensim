#!/bin/bash
# fastclass distillation wave (2026-09-04) — train + HARVEST-INLINE every arm.
#
# Registration: benchmarks/fastclass_distill_wave_2026-09-04.md (committed
# BEFORE the first fit). Arms, seeds, mechanisms and bars are frozen there.
#
# Reuses the wave-r4 owners verbatim (train_156_student.sh + score_arm.sh) and
# the wave-r4 BUILD + ROOT, because that is the only way a new arm reads
# byte-identical features to A4b's own — the control this whole wave is read
# against. Nothing here computes a statistic.
#
# Each bake is SCORED the moment it lands (playbook step 4): a missed wake-up
# then costs review latency only.
set -euo pipefail
REPO="${FCD_REPO:-/home/lilith/work/zen/zensim}"
cd "$REPO"
export ZL_TRAIN=/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
O=/mnt/v/output/zensim/fastclass-2026-09-04
export WR4_OUT="$O/bakes"
export WR4_SCORE="$O"
mkdir -p "$WR4_OUT" "$O"
TRAIN_SH="$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh"
SCORE_SH="$REPO/benchmarks/wave_r4_2026-09-01/score_arm.sh"
RH="$HOME/work/zen/scripts/run-heavy"
HB="${FCD_HB:-$HOME/tmp/fastclass/run}"
mkdir -p "$(dirname "$HB")"
SEEDS="${FCD_SEEDS:-4004 4005 4006}"
ARMS="${FCD_ARMS:-C0 D1 D2 D3 D4 E1 F1}"

# arm -> the env deltas that DEFINE it (everything else is A4b verbatim)
arm_env() {
  unset WR4_KON_WITHINREF WR4_HF_WITHINREF WR4_HIGH_Q_BOOST WR4_N_HIDDEN_LAYERS WR4_KADIS || true
  case "$1" in
    C0) : ;;                                                        # control
    D1) export WR4_KON_WITHINREF=1 ;;                               # kon ladder within-image
    D2) export WR4_HF_WITHINREF=1 ;;                                # near-lossless ladder within-image
    D3) export WR4_KON_WITHINREF=1 WR4_HF_WITHINREF=1 ;;            # both
    D4) export WR4_HIGH_Q_BOOST=3.0 ;;                              # B3 sampling mass
    E1) export WR4_N_HIDDEN_LAYERS=2 ;;                             # capacity
    F1) export WR4_KADIS=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/ext_kadis.parquet ;;
    *)  echo "ABORT: unknown arm $1"; exit 2 ;;
  esac
}

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$HB.log"; }
say "START arms=[$ARMS] seeds=[$SEEDS]"
for ARM in $ARMS; do
  for SEED in $SEEDS; do
    NAME="${ARM}_s${SEED}"
    OUT="$WR4_OUT/${NAME}.bin"
    if [ -f "$O/${NAME}.fulleval.json" ]; then say "SKIP (scored): $NAME"; continue; fi
    if [ ! -f "$OUT" ]; then
      say "TRAIN $NAME"
      ( arm_env "$ARM"; "$RH" --mem 16G --jobs 8 -- "$TRAIN_SH" distill "$SEED" "$OUT" ) \
        >>"$HB.train.log" 2>&1 || { say "TRAIN FAILED $NAME"; touch "$O/${NAME}.TRAIN_FAILED"; continue; }
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
say "ALL ARMS COMPLETE"
