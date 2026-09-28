#!/usr/bin/env bash
# Wait for the registered grid, then compute and verify MLP reference CIs.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=/var/tmp/rev4-featpot
stop=${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md}
log="$out/resume_when_quota_clears.log"
while [[ ! -f "$log" ]] || ! rg -q 'REGISTERED_BASELINE_BATCHES_DONE' "$log"; do
    printf 'WAIT grid %s\n' "$(date -u +%FT%TZ)"
    sleep 60
done

attempt=0
while true; do
    while [[ -e "$stop" ]]; do
        printf 'WAIT quota %s\n' "$(date -u +%FT%TZ)"
        sleep 30
    done
    attempt=$((attempt + 1))
    progress="$out/mlp_ci_resume_${attempt}.log"
    printf 'START mlp_ci attempt=%s %s\n' "$attempt" "$(date -u +%FT%TZ)"
    if bash scripts/rev4_featpot/run_mlp_ci.sh > "$progress" 2>&1; then
        printf 'DONE mlp_ci attempt=%s %s\n' "$attempt" "$(date -u +%FT%TZ)"
        sha256sum "$progress"
        tail -n 2 "$progress"
        break
    else
        rc=$?
    fi
    printf 'EXIT mlp_ci attempt=%s rc=%s %s\n' "$attempt" "$rc" "$(date -u +%FT%TZ)"
    sha256sum "$progress"
    tail -n 2 "$progress"
    if (( rc != 75 )) || [[ ! -e "$stop" ]]; then
        exit "$rc"
    fi
done
python scripts/rev4_featpot/summarize.py --out "$out/baseline_summary.json"
python scripts/rev4_featpot/check_core_gates.py
printf 'REGISTERED_CORE_GATES_DONE %s\n' "$(date -u +%FT%TZ)"
