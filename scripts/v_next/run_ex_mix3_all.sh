#!/usr/bin/env bash
# EX-MIX3: launch all 3 variants × 5 seeds = 15 runs
#
# 7950X has 16C/32T. parallel_batch=true means each trainer uses
# many threads via rayon. Co-tenant: EX-DUAL agent running in parallel
# workspace ../zensim--dual-target (one trainer ~30% CPU).
#
# Throttle to 2 EX-MIX3 trainers in parallel to stay friendly.
# 15 jobs / 2 concurrent = 8 batches × ~50min ≈ 7h wall.
# Variant rotation prevents single variant blocking on a slow seed.

set -euo pipefail

WORKSPACE="/home/lilith/work/zen/zensim--ex-mix3"
RUN_SCRIPT="${WORKSPACE}/scripts/v_next/run_ex_mix3_seed.sh"

# Job list (variant, seed) pairs interleaved by variant
JOBS=(
    "cv33_iw33_sm33 1" "cv30_iw40_sm30 1" "cv40_iw40_sm20 1"
    "cv33_iw33_sm33 2" "cv30_iw40_sm30 2" "cv40_iw40_sm20 2"
    "cv33_iw33_sm33 3" "cv30_iw40_sm30 3" "cv40_iw40_sm20 3"
    "cv33_iw33_sm33 4" "cv30_iw40_sm30 4" "cv40_iw40_sm20 4"
    "cv33_iw33_sm33 5" "cv30_iw40_sm30 5" "cv40_iw40_sm20 5"
)

CONCURRENCY="${EX_MIX3_CONCURRENCY:-2}"

run_one() {
    local v="$1" s="$2"
    echo "[$(date +%H:%M:%S)] START: variant=${v} seed=${s}"
    if bash "${RUN_SCRIPT}" "${v}" "${s}" >/dev/null 2>&1; then
        echo "[$(date +%H:%M:%S)] DONE:  variant=${v} seed=${s} OK"
    else
        echo "[$(date +%H:%M:%S)] FAIL:  variant=${v} seed=${s} exit=$?"
    fi
}

i=0
total=${#JOBS[@]}
while [ $i -lt $total ]; do
    pids=()
    end=$((i + CONCURRENCY))
    [ $end -gt $total ] && end=$total
    for ((j=i; j<end; j++)); do
        pair="${JOBS[$j]}"
        v="${pair% *}"
        s="${pair##* }"
        run_one "${v}" "${s}" &
        pids+=($!)
    done
    wait "${pids[@]}"
    i=$end
done

echo "[$(date +%H:%M:%S)] === all 15 trains complete ==="
