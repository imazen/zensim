#!/usr/bin/env bash
# Publish only after fit, CI, and pooled aggregate stages signal success.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=/var/tmp/rev4-featpot
log="$out/run_mlp_aggregate_after_core.log"
while [[ ! -f "$log" ]] || ! rg -q 'REGISTERED_MLP_AGGREGATES_DONE' "$log"; do
    printf 'WAIT aggregate %s\n' "$(date -u +%FT%TZ)"
    sleep 60
done
while [[ -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]; do
    printf 'WAIT quota before report %s\n' "$(date -u +%FT%TZ)"
    sleep 30
done
python scripts/rev4_featpot/finalize_core_reports.py --verify-only
python scripts/rev4_featpot/finalize_core_reports.py
printf 'CORE_REPORTS_UPDATED %s\n' "$(date -u +%FT%TZ)"
