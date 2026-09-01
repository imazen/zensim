#!/bin/bash
# a4bkon lane: run the exam's OWN W1/W2 instrument (paired_perref_boot.py,
# unmodified) over every corpus for K1/K2/K4, plus the hfnl_cid22band mode.
# No new statistics -- this only sets O/CORPUS/ARMS/BAND_* env vars per the
# script's own documented interface.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
cd "$REPO"
export O=/mnt/v/output/zensim/a4bkon-2026-09-01
export ARMS="K1_w1.8_s4004 K1_w1.8_s4005 K1_w2.4_s4004 K1_w2.4_s4005 K2_s4004 K2_s4005 K4"
OUTLOG=/mnt/v/output/zensim/a4bkon-2026-09-01/paired_boot_a4bkon.txt
: > "$OUTLOG"

for c in cid22 csiq aic3 live aic4; do
  echo "##### CORPUS=$c #####" | tee -a "$OUTLOG"
  CORPUS="$c" python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py 2>&1 | tee -a "$OUTLOG"
  echo "" | tee -a "$OUTLOG"
done

echo "##### CORPUS=cid22 BAND=[0.8,inf) (hfnl_cid22band) #####" | tee -a "$OUTLOG"
CORPUS=cid22 BAND_LO=0.8 python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py 2>&1 | tee -a "$OUTLOG"

echo "WROTE $OUTLOG"
