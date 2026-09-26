#!/usr/bin/env bash
# Resume only registered baseline cells after the coordinator lifts the quota gate.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=/var/tmp/rev4-featpot
stop=$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md
mkdir -p "$out"

wait_for_gate() {
    while [[ -e "$stop" ]]; do
        printf 'WAIT quota %s\n' "$(date -u +%FT%TZ)"
        sleep 30
    done
}

run_batch() {
    local script=$1 tag=$2 attempt=0 rc log
    while true; do
        wait_for_gate
        attempt=$((attempt + 1))
        log="$out/${tag}_resume_${attempt}.log"
        printf 'START %s attempt=%s %s\n' "$tag" "$attempt" "$(date -u +%FT%TZ)"
        if bash "scripts/rev4_featpot/$script" > "$log" 2>&1; then
            printf 'DONE %s attempt=%s %s\n' "$tag" "$attempt" "$(date -u +%FT%TZ)"
            sha256sum "$log"
            tail -n 2 "$log"
            return 0
        else
            rc=$?
        fi
        printf 'EXIT %s attempt=%s rc=%s %s\n' "$tag" "$attempt" "$rc" "$(date -u +%FT%TZ)"
        sha256sum "$log"
        tail -n 2 "$log"
        if (( rc != 75 )) || [[ ! -e "$stop" ]]; then
            return "$rc"
        fi
    done
}

run_batch run_cell_ci.sh cell_ci
run_batch run_sham.sh sham
run_batch run_enrich.sh enrich
run_batch run_mlp_baseline.sh mlp_baseline
wait_for_gate
python scripts/rev4_featpot/summarize.py --out "$out/baseline_summary.json"
printf 'REGISTERED_BASELINE_BATCHES_DONE %s\n' "$(date -u +%FT%TZ)"
