#!/usr/bin/env bash
# POTENTIAL — ceiling, not a model score. Idempotent 5x5-Latin H32/H128 runner.
# Reacquire the shared lock per replicate and per outer-fold importance call.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target
export CARGO_HOME=/var/tmp/rev4-featpot/cargo_home
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

gate() {
    if [[ -e $HOME/tmp/zensim-paper/rev4/FLEET_FITS_GO.md ]]; then
        python scripts/rev4_featpot/p0_fleet_pause.py --runner serial
        exit 0
    fi
    if [[ -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]; then
        printf 'QUOTA_STOP at %s\n' "$(date -u +%FT%TZ)"
        exit 75
    fi
    local free_gb
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    if (( free_gb < 20 )); then
        printf 'DISK_STOP /home=%sG at %s\n' "$free_gb" "$(date -u +%FT%TZ)"
        exit 75
    fi
}

for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
    for arm in r0 minus_basic; do
        for hidden in 32 128; do
            for outer in 0 1 2 3 4 full; do
                for rep in 0 1 2 3 4; do
                    dest="/var/tmp/rev4-featpot/fits/POT_${set_name}_${arm}_mlp${hidden}/${outer}_r${rep}"
                    outer_args=(--outer "$outer")
                    if [[ "$outer" == full ]]; then
                        outer_args=()
                    else
                        dest="/var/tmp/rev4-featpot/fits/POT_${set_name}_${arm}_mlp${hidden}/o${outer}_r${rep}"
                    fi
                    if [[ ! -f "$dest/result.json" ]]; then
                        gate
                        log="/var/tmp/rev4-featpot/mlp_${set_name}_${arm}_h${hidden}_${outer}_r${rep}.log"
                        printf 'START FIT %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/mlp_probe.py --set "$set_name" --arm "$arm" \
                                --hidden "$hidden" "${outer_args[@]}" --rep "$rep" > "$log" 2>&1
                        printf 'END FIT %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                        sha256sum "$log" "$dest/result.json"
                        tail -2 "$log"
                    else
                        printf 'SKIP FIT %s %s H%s %s r%s existing result\n' "$set_name" "$arm" "$hidden" "$outer" "$rep"
                    fi
                    if [[ "$outer" != full && ! -f "$dest/importance.json" ]]; then
                        gate
                        log="/var/tmp/rev4-featpot/mlp_${set_name}_${arm}_h${hidden}_${outer}_r${rep}_importance.log"
                        printf 'START IMPORTANCE %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/mlp_importance.py --set "$set_name" --arm "$arm" \
                                --hidden "$hidden" --outer "$outer" --rep "$rep" > "$log" 2>&1
                        printf 'END IMPORTANCE %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                        sha256sum "$log" "$dest/importance.json"
                        tail -1 "$log"
                    fi
                done
            done
        done
    done
done
