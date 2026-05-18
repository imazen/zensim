#!/usr/bin/env bash
# EX-MIX3: continuously run bake_verdict on new bakes as they appear.
# Designed to run alongside the training matrix so verdicts are ready
# when all 15 trainings finish.

set -uo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18"
BAKE_VERDICT="/home/lilith/work/zen/zensim/target/release/bake_verdict"
mkdir -p "${OUT_DIR}/verdicts"

VARIANTS=("cv33_iw33_sm33" "cv30_iw40_sm30" "cv40_iw40_sm20")
SEEDS=(1 2 3 4 5)

total=15
done_count=0
loops=0
MAX_LOOPS=480  # 480 * 30s = 4 hours

while [ $done_count -lt $total ] && [ $loops -lt $MAX_LOOPS ]; do
    done_count=0
    for v in "${VARIANTS[@]}"; do
        for s in "${SEEDS[@]}"; do
            bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
            out="${OUT_DIR}/verdicts/exmix3_${v}_s${s}.md"
            if [ -f "${out}" ]; then
                done_count=$((done_count + 1))
                continue
            fi
            if [ -f "${bake}" ]; then
                echo "[$(date +%H:%M:%S)] verdict: ${v} s${s}"
                "${BAKE_VERDICT}" --bake "${bake}" --corpora cid22,kadid,tid,konjnd,aic3 --output "${out}" 2>&1 | tail -1
                done_count=$((done_count + 1))
            fi
        done
    done
    echo "[$(date +%H:%M:%S)] verdicts done: ${done_count}/${total}"
    if [ $done_count -lt $total ]; then
        sleep 30
    fi
    loops=$((loops + 1))
done

echo "[$(date +%H:%M:%S)] verdict-as-ready DONE (${done_count}/${total})"
