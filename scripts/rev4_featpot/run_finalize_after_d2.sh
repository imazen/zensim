#!/usr/bin/env bash
# Refresh both receipt-only reports after core P0 and paired D2 MLP finish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root=/var/tmp/rev4-featpot
stop=${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md}

while [[ ! -f "$root/p2/d2_mlp/compare_done" ]] || \
      [[ ! -f "$root/run_finalize_after_aggregates.log" ]] || \
      ! rg -q 'CORE_REPORTS_UPDATED' "$root/run_finalize_after_aggregates.log"; do
    printf 'WAIT D2/core reports %s\n' "$(date -u +%FT%TZ)"
    sleep 60
done
while [[ -e "$stop" ]]; do
    printf 'WAIT quota before D2 report %s\n' "$(date -u +%FT%TZ)"
    sleep 30
done
free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
(( free_gb >= 20 ))
PYTHONPYCACHEPREFIX="$root/pycache" python scripts/rev4_featpot/finalize_core_reports.py --verify-only
PYTHONPYCACHEPREFIX="$root/pycache" python scripts/rev4_featpot/finalize_core_reports.py
PYTHONPYCACHEPREFIX="$root/pycache" python scripts/rev4_featpot/p2_report.py
touch "$root/p2/d2_mlp/reports_done"
printf 'D2_REPORTS_UPDATED %s\n' "$(date -u +%FT%TZ)"
