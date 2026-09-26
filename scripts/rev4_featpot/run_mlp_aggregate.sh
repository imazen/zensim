#!/usr/bin/env bash
# Pooled out-of-fold H32/H128 estimates after all registered fits exist.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target
export CARGO_HOME=/var/tmp/rev4-featpot/cargo_home
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
    for arm in r0 minus_basic; do
        for hidden in 32 128; do
            dest="/var/tmp/rev4-featpot/mlp_aggregate/POT_${set_name}_${arm}_mlp${hidden}/result.json"
            if [[ -f "$dest" ]]; then
                printf 'SKIP AGGREGATE %s existing result\n' "$dest"
                continue
            fi
            if [[ -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]; then
                printf 'QUOTA_STOP at %s\n' "$(date -u +%FT%TZ)"
                exit 75
            fi
            free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
            if (( free_gb < 20 )); then
                printf 'DISK_STOP /home=%sG at %s\n' "$free_gb" "$(date -u +%FT%TZ)"
                exit 75
            fi
            log="/var/tmp/rev4-featpot/mlp_aggregate_${set_name}_${arm}_h${hidden}.log"
            printf 'START AGGREGATE %s %s H%s %s\n' "$set_name" "$arm" "$hidden" "$(date -u +%FT%TZ)"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/mlp_aggregate.py --set "$set_name" \
                    --arm "$arm" --hidden "$hidden" > "$log" 2>&1
            printf 'END AGGREGATE %s %s H%s %s\n' "$set_name" "$arm" "$hidden" "$(date -u +%FT%TZ)"
            sha256sum "$log" "$dest"
            tail -n 2 "$log"
        done
    done
done
