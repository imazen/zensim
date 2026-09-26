#!/usr/bin/env bash
# Wait for all fit and CI receipts, then pool outer-fold predictions.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=/var/tmp/rev4-featpot
stop=${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md}
log="$out/run_mlp_ci_after_grid.log"
while [[ ! -f "$log" ]] || ! rg -q 'REGISTERED_CORE_GATES_DONE' "$log"; do
    printf 'WAIT core %s\n' "$(date -u +%FT%TZ)"
    sleep 60
done

attempt=0
while true; do
    while [[ -e "$stop" ]]; do
        printf 'WAIT quota %s\n' "$(date -u +%FT%TZ)"
        sleep 30
    done
    attempt=$((attempt + 1))
    progress="$out/mlp_aggregate_resume_${attempt}.log"
    printf 'START mlp_aggregate attempt=%s %s\n' "$attempt" "$(date -u +%FT%TZ)"
    if bash scripts/rev4_featpot/run_mlp_aggregate.sh > "$progress" 2>&1; then
        printf 'DONE mlp_aggregate attempt=%s %s\n' "$attempt" "$(date -u +%FT%TZ)"
        sha256sum "$progress"
        tail -n 2 "$progress"
        break
    else
        rc=$?
    fi
    printf 'EXIT mlp_aggregate attempt=%s rc=%s %s\n' "$attempt" "$rc" "$(date -u +%FT%TZ)"
    sha256sum "$progress"
    tail -n 2 "$progress"
    if (( rc != 75 )) || [[ ! -e "$stop" ]]; then
        exit "$rc"
    fi
done
printf 'REGISTERED_MLP_AGGREGATES_DONE %s\n' "$(date -u +%FT%TZ)"
