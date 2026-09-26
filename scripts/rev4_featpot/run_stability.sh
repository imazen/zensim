#!/usr/bin/env bash
# POTENTIAL — ceiling, not a model score. Frozen-lambda 200 reference-half draws.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target
export CARGO_HOME=/var/tmp/rev4-featpot/cargo_home
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

for set_name in tid2013 konfig_train konjnd_bpg_train cid22_a25 kadid_select konfig_val; do
    result="/var/tmp/rev4-featpot/stability/POT_${set_name}_r0_lasso/result.json"
    if [[ -f "$result" ]]; then
        printf 'SKIP %s existing result\n' "$set_name"
        continue
    fi
    if [[ ! -f "/var/tmp/rev4-featpot/fits/POT_${set_name}_r0_linear/result.json" ]]; then
        printf 'MISSING %s R0 lasso cell\n' "$set_name"
        continue
    fi
    if [[ -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]; then
        printf 'QUOTA_STOP before %s at %s\n' "$set_name" "$(date -u +%FT%TZ)"
        exit 75
    fi
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    if (( free_gb < 20 )); then
        printf 'DISK_STOP /home=%sG before %s\n' "$free_gb" "$set_name"
        exit 75
    fi
    log="/var/tmp/rev4-featpot/${set_name}_stability_full.log"
    printf 'START %s %s\n' "$set_name" "$(date -u +%FT%TZ)"
    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
        python scripts/rev4_featpot/stability_lasso.py --set "$set_name" > "$log" 2>&1
    printf 'END %s %s\n' "$set_name" "$(date -u +%FT%TZ)"
    sha256sum "$log" "$result"
    tail -1 "$log"
done
