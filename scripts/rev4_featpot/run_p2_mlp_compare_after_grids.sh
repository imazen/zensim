#!/usr/bin/env bash
# Join saved P0/P2 five-seed panel draws only after both complete grids.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
root=/var/tmp/rev4-featpot
while [[ ! -f "$root/p2/mlp_done" ]] ||
      ! rg -q 'REGISTERED_MLP_AGGREGATES_DONE' "$root/run_mlp_aggregate_after_core.log" 2>/dev/null; do
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    sleep 30
done
for set_name in kadid_train tid2013 konfig_train cid22_a25 aic3 kadid_select konfig_val; do
    for hidden in 32 128; do
        out="$root/p2/mlp_compare/POT_${set_name}_mlp${hidden}.json"
        if [[ ! -f "$out" ]]; then
            python scripts/rev4_featpot/p2_mlp_compare.py --set "$set_name" --hidden "$hidden"
        fi
    done
done
python scripts/rev4_featpot/p2_report.py
touch "$root/p2/mlp_compare_done"
printf 'P2 pooled MLP comparison complete %s\n' "$(date -u +%FT%TZ)"
