#!/usr/bin/env bash
# sota944_armC_chain.sh <seed...> — run arm-C seeds SEQUENTIALLY (one heavy
# job at a time), each via sota944_armC_seed.sh; per-seed log + verdict via
# the shared invocation; heartbeat to stdout per seed. A failed seed is
# logged and the chain continues (honest-loss row, never a silent stop).
set -uo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOGD=${SOTA944_LOGD:-$HOME/tmp/sota944}
mkdir -p "$LOGD"
for SEED in "$@"; do
    echo "[chain] seed $SEED start $(date -u +%H:%M:%SZ)"
    if bash "$REPO_ROOT/scripts/sota944_armC_seed.sh" "$SEED" \
        > "$LOGD/armC_s${SEED}.log" 2>&1; then
        echo "[chain] seed $SEED trained OK"
    else
        echo "[chain] seed $SEED FAILED rc=$? (see $LOGD/armC_s${SEED}.log)"
        continue
    fi
    BAKE=/mnt/v/output/zensim/bakes/sota944/bakes/C_em944_s${SEED}.bin
    if [[ -f "$BAKE" ]]; then
        bash "$REPO_ROOT/scripts/sota944_verdict.sh" "$BAKE" "C_em944_s${SEED}" \
            > "$LOGD/armC_s${SEED}.verdict.log" 2>&1 \
            && echo "[chain] seed $SEED verdict OK" \
            || echo "[chain] seed $SEED verdict FAILED"
    fi
done
echo "[chain] all seeds processed $(date -u +%H:%M:%SZ)"
