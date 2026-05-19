#!/usr/bin/env bash
# EX-PERCENTILE-POOL: re-extract 372-col features for all validation
# corpora with P²-quantile pooling enabled (--p2-pool flag).
#
# Output: /mnt/v/zen/zensim-training/2026-05-18-percentile-pool/
#
# Validation corpora (372 col):
#   - cid22  (4292 pairs)  via zensim-validate --extract-only --p2-pool
#   - kadid  (10125 pairs) via zensim-validate --extract-only --p2-pool
#   - tid    (3000 pairs)  via zensim-validate --extract-only --p2-pool
#   - konjnd (1008 pairs)  via extract_features_372col --p2-pool
#   - aic3   (600 pairs)   via extract_features_372col --p2-pool
#
# Train corpora (300 col, no IW): re-extract kadid/tid/konjnd with --p2-pool
# but compute_iw_features=false; safesyn skipped (too expensive without
# joined target columns).
set -euo pipefail

OUT="/mnt/v/zen/zensim-training/2026-05-18-percentile-pool"
mkdir -p "$OUT"
LOG_DIR=/tmp
ROOT=/home/lilith/work/zen/zensim--exp-percentile-pool
VALIDATE="$ROOT/target/release/zensim-validate"
EXTRACT372="$ROOT/target/release/examples/extract_features_372col"

date_started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[${date_started}] EX-PERCENTILE-POOL feature extraction" | tee "$OUT/extraction.log"

# --- Validation corpora (372 col, p2-pool ON) -----------------------------
# CID22
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] CID22 (372col, p2)" | tee -a "$OUT/extraction.log"
"$VALIDATE" \
    --dataset /mnt/v/dataset/cid22/CID22_validation_set \
    --format cid22 \
    --extract-only \
    --extended-features \
    --iw-features \
    --p2-pool \
    --features-csv "$OUT/cid22_features_372col_p2.csv" \
    2>&1 | tee "$LOG_DIR/exp_percentile_pool_cid22.log" >/dev/null &
PID_CID22=$!

# KADID
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] KADID (372col, p2)" | tee -a "$OUT/extraction.log"
"$VALIDATE" \
    --dataset /mnt/v/dataset/kadid10k \
    --format kadid10k \
    --extract-only \
    --extended-features \
    --iw-features \
    --p2-pool \
    --features-csv "$OUT/kadid_features_372col_p2.csv" \
    2>&1 | tee "$LOG_DIR/exp_percentile_pool_kadid.log" >/dev/null &
PID_KADID=$!

# TID
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TID (372col, p2)" | tee -a "$OUT/extraction.log"
"$VALIDATE" \
    --dataset /mnt/v/dataset/tid2013 \
    --format tid2013 \
    --extract-only \
    --extended-features \
    --iw-features \
    --p2-pool \
    --features-csv "$OUT/tid_features_372col_p2.csv" \
    2>&1 | tee "$LOG_DIR/exp_percentile_pool_tid.log" >/dev/null &
PID_TID=$!

# KonJND (1008 pairs)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] KonJND (372col, p2)" | tee -a "$OUT/extraction.log"
"$EXTRACT372" \
    --corpus konjnd \
    --path /mnt/v/datasets/KonJND-1k/KonJND-1k \
    --out "$OUT/konjnd_features_372col_p2.csv" \
    --p2-pool \
    2>&1 | tee "$LOG_DIR/exp_percentile_pool_konjnd.log" >/dev/null &
PID_KONJND=$!

# AIC-3 (600 pairs)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] AIC-3 (372col, p2)" | tee -a "$OUT/extraction.log"
"$EXTRACT372" \
    --corpus aic3 \
    --path /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
    --out "$OUT/aic3_features_372col_p2.csv" \
    --p2-pool \
    2>&1 | tee "$LOG_DIR/exp_percentile_pool_aic3.log" >/dev/null &
PID_AIC3=$!

# Wait for all
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for all extractions..." | tee -a "$OUT/extraction.log"
wait $PID_CID22 || echo "CID22 failed (exit $?)" | tee -a "$OUT/extraction.log"
wait $PID_KADID || echo "KADID failed (exit $?)" | tee -a "$OUT/extraction.log"
wait $PID_TID   || echo "TID failed (exit $?)"   | tee -a "$OUT/extraction.log"
wait $PID_KONJND || echo "KonJND failed (exit $?)" | tee -a "$OUT/extraction.log"
wait $PID_AIC3  || echo "AIC-3 failed (exit $?)" | tee -a "$OUT/extraction.log"

date_done=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[${date_done}] EX-PERCENTILE-POOL extraction complete" | tee -a "$OUT/extraction.log"
ls -lh "$OUT/"*.csv | tee -a "$OUT/extraction.log"
