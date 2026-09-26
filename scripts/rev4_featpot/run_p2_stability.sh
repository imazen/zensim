#!/usr/bin/env bash
# Registered 200 half-reference draws for P2 and its matched permutation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

while [[ ! -f /var/tmp/rev4-featpot/p2/deterministic_done ]]; do
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    sleep 30
done
for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
    for arm in p2 p2_perm; do
        out="/var/tmp/rev4-featpot/p2/stability/POT_${set_name}_${arm}_lasso/result.json"
        if [[ ! -f "$out" ]]; then
            [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
            free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
            (( free_gb >= 20 ))
            log="/var/tmp/rev4-featpot/p2_stability_${set_name}_${arm}.log"
            printf 'START P2 STABILITY %s %s %s\n' "$set_name" "$arm" "$(date -u +%FT%TZ)"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/p2_stability.py --set "$set_name" --arm "$arm" > "$log" 2>&1
            sha256sum "$out" "$log"
        fi
    done
done
touch /var/tmp/rev4-featpot/p2/stability_done
printf 'P2 stability complete %s\n' "$(date -u +%FT%TZ)"
