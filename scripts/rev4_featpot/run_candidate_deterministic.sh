#!/usr/bin/env bash
# POTENTIAL C1-C4: paired D1/D2 lasso and BVLS. Input-pin commit required.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export REV4_FIT_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/bake_dial_refit
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot/candidates

gate() {
    [[ -f $HOME/tmp/zensim-paper/rev4/PARTB_C1C4_DONE.md ]]
    [[ -f "$root/input_pin_committed.json" ]]
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    local free_gb
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
    [[ -x "$REV4_FIT_BIN" && -x "$ZEN_PANEL_BIN" ]]
    [[ -f /var/tmp/rev4-featpot/main_build_20260924/binary_meta.json ]]
}

for arm in c1 c4 c2 c3 all; do
    for model in bvls linear; do
        for variant in "$arm" "${arm}_perm"; do
            for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
                out="$root/d1/POT_${set_name}_${variant}_${model}/result.json"
                if [[ ! -f "$out" ]]; then
                    gate
                    log="$root/log_d1_${set_name}_${variant}_${model}.log"
                    printf 'START D1 %s %s %s %s\n' "$set_name" "$variant" "$model" "$(date -u +%FT%TZ)"
                    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                        python scripts/rev4_featpot/candidate_linear.py --set "$set_name" \
                            --arm "$variant" --model "$model" > "$log" 2>&1
                    printf 'END D1 %s %s %s %s\n' "$set_name" "$variant" "$model" "$(date -u +%FT%TZ)"
                    sha256sum "$out" "$log"
                fi
            done
        done
        for set_name in kadid_train tid2013 konfig_train cid22_a25 aic3 kadid_select konfig_val; do
            out="$root/d1/POT_${set_name}_${arm}_${model}_compare.json"
            if [[ ! -f "$out" ]]; then
                gate
                log="$root/log_d1_${set_name}_${arm}_${model}_compare.log"
                $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                    python scripts/rev4_featpot/candidate_compare.py --scope d1 \
                        --set "$set_name" --arm "$arm" --model "$model" > "$log" 2>&1
                sha256sum "$out" "$log"
            fi
        done
        for variant in "$arm" "${arm}_perm"; do
            out="$root/d2/LODO_${variant}_${model}/result.json"
            if [[ ! -f "$out" ]]; then
                gate
                log="$root/log_d2_${variant}_${model}.log"
                $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                    python scripts/rev4_featpot/candidate_lodo.py --arm "$variant" \
                        --model "$model" > "$log" 2>&1
                sha256sum "$out" "$log"
            fi
        done
        out="$root/d2/LODO_${arm}_${model}_compare.json"
        if [[ ! -f "$out" ]]; then
            gate
            log="$root/log_d2_${arm}_${model}_compare.log"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/candidate_compare.py --scope d2 \
                    --arm "$arm" --model "$model" > "$log" 2>&1
            sha256sum "$out" "$log"
        fi
    done
done
touch "$root/deterministic_done"
printf 'C1-C4 deterministic D1/D2 complete %s\n' "$(date -u +%FT%TZ)"
