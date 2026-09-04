#!/bin/bash
# fastclass wave: run the exam's OWN W1/W2 instrument over every corpus for
# every arm. No new statistics -- this only sets the env vars
# `paired_perref_boot.py` already documents. konjnd is now in the loop because
# that script gained a ref_basename JOIN path for it this lane (the registered
# instrument gap wave_r4 §19 named).
set -euo pipefail
REPO="${FCD_REPO:-/home/lilith/work/zen/zensim}"
cd "$REPO"
export O=/mnt/v/output/zensim/fastclass-2026-09-04
export ZEN_PANEL_BIN=/mnt/v/zen/cargo-targets/waver4/release/panel
ARMS_DEFAULT=""
for a in ${FCD_ARMS:-C0 D1 D2 D3 D4 E1 F1}; do
  for s in ${FCD_SEEDS:-4004 4005 4006}; do
    [ -f "$O/pp_${a}_s${s}_cid22.tsv" ] && ARMS_DEFAULT="$ARMS_DEFAULT ${a}_s${s}"
  done
done
export ARMS="${ARMS:-$ARMS_DEFAULT}"
OUTLOG="$O/paired_boot_fastclass.txt"
: > "$OUTLOG"
echo "# ARMS=$ARMS" | tee -a "$OUTLOG"
for c in cid22 csiq aic3 live aic4 konjnd; do
  echo "##### CORPUS=$c #####" | tee -a "$OUTLOG"
  CORPUS="$c" python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py 2>&1 | tee -a "$OUTLOG"
  echo "" | tee -a "$OUTLOG"
done
echo "##### CORPUS=cid22 BAND=[0.8,inf) (hfnl_cid22band) #####" | tee -a "$OUTLOG"
CORPUS=cid22 BAND_LO=0.8 python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py 2>&1 | tee -a "$OUTLOG"
echo "WROTE $OUTLOG"
