#!/usr/bin/env bash
# POTENTIAL — ceiling, not a model score. E1 and within-reference owner stats.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target
export CARGO_HOME=/var/tmp/rev4-featpot/cargo_home
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
    for model in bvls linear; do
        for arm in r0 minus_basic; do
            source="/var/tmp/rev4-featpot/fits/POT_${set_name}_${arm}_${model}/result.json"
            result="/var/tmp/rev4-featpot/stats/${set_name}_${arm}_${model}.json"
            if [[ ! -f "$source" ]]; then
                printf 'MISSING %s %s %s fit\n' "$set_name" "$model" "$arm"
                continue
            fi
            if [[ -f "$result" ]]; then
                printf 'SKIP %s %s %s existing stats\n' "$set_name" "$model" "$arm"
                continue
            fi
            if [[ -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]; then
                printf 'QUOTA_STOP before %s %s %s at %s\n' "$set_name" "$model" "$arm" "$(date -u +%FT%TZ)"
                exit 75
            fi
            free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
            if (( free_gb < 20 )); then
                printf 'DISK_STOP /home=%sG before %s %s %s\n' "$free_gb" "$set_name" "$model" "$arm"
                exit 75
            fi
            log="/var/tmp/rev4-featpot/${set_name}_${arm}_${model}_stats.log"
            printf 'START %s %s %s %s\n' "$set_name" "$model" "$arm" "$(date -u +%FT%TZ)"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/enrich_stats.py --set "$set_name" --model "$model" --arm "$arm" > "$log" 2>&1
            printf 'END %s %s %s %s\n' "$set_name" "$model" "$arm" "$(date -u +%FT%TZ)"
            sha256sum "$log" "$result"
            tail -1 "$log"
        done
    done
done
