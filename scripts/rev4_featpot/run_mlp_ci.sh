#!/usr/bin/env bash
# Reference-clustered intervals for all registered H32/H128 replicate results.
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
            for outer in 0 1 2 3 4 full; do
                for rep in 0 1 2 3 4; do
                    if [[ "$outer" == full ]]; then
                        dest="/var/tmp/rev4-featpot/fits/POT_${set_name}_${arm}_mlp${hidden}/full_r${rep}"
                        outer_args=()
                    else
                        dest="/var/tmp/rev4-featpot/fits/POT_${set_name}_${arm}_mlp${hidden}/o${outer}_r${rep}"
                        outer_args=(--outer "$outer")
                    fi
                    [[ -f "$dest/result.json" ]] || { printf 'MISSING FIT %s\n' "$dest"; exit 1; }
                    if [[ -f "$dest/ci.json" ]]; then
                        printf 'SKIP CI %s existing result\n' "$dest"
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
                    log="/var/tmp/rev4-featpot/mlp_ci_${set_name}_${arm}_h${hidden}_${outer}_r${rep}.log"
                    printf 'START CI %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                        python scripts/rev4_featpot/mlp_ci.py --set "$set_name" --arm "$arm" \
                            --hidden "$hidden" "${outer_args[@]}" --rep "$rep" > "$log" 2>&1
                    printf 'END CI %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                    sha256sum "$log" "$dest/ci.json"
                    tail -n 2 "$log"
                done
            done
        done
    done
done
