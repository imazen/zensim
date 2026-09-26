#!/usr/bin/env bash
# POTENTIAL C1-C4 D2 H32/H128 source-held-out transfer and matched controls.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export REV4_FIT_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/bake_dial_refit
export REV4_TRAINER_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/zensim_mlp_train
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot/candidates

gate() {
    [[ -f $HOME/tmp/zensim-paper/rev4/PARTB_C1C4_DONE.md ]]
    [[ -f "$root/input_pin_committed.json" ]]
    [[ -f /var/tmp/rev4-featpot/main_build_20260924/binary_meta.json ]]
    [[ -x "$REV4_FIT_BIN" && -x "$REV4_TRAINER_BIN" && -x "$ZEN_PANEL_BIN" ]]
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    local free_gb
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
}

for arm in c1 c4 c2 c3 all; do
    for variant in "$arm" "${arm}_perm"; do
        receipt="$root/d2_mlp/tables/$variant/receipt.json"
        if [[ ! -f "$receipt" ]]; then
            gate
            log="$root/log_d2_mlp_tables_${variant}.log"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/candidate_lodo_mlp_tables.py --arm "$variant" > "$log" 2>&1
            sha256sum "$receipt" "$log"
        fi
        for hidden in 32 128; do
            for heldout in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select; do
                for rep in 0 1 2 3 4; do
                    dest="$root/d2_mlp/LODO_${variant}_mlp${hidden}/without_${heldout}_r${rep}"
                    if [[ ! -f "$dest/result.json" ]]; then
                        gate
                        log="$root/log_d2_mlp_${variant}_h${hidden}_${heldout}_r${rep}.log"
                        printf 'START D2 MLP %s H%s %s r%s %s\n' "$variant" "$hidden" "$heldout" "$rep" "$(date -u +%FT%TZ)"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/candidate_lodo_mlp.py --arm "$variant" \
                                --hidden "$hidden" --heldout "$heldout" --rep "$rep" > "$log" 2>&1
                        sha256sum "$dest/result.json" "$log"
                    fi
                done
            done
        done
    done
    for hidden in 32 128; do
        for heldout in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select; do
            out="$root/d2_mlp_compare/LODO_${heldout}_${arm}_mlp${hidden}.json"
            if [[ ! -f "$out" ]]; then
                for rep in 0 1 2 3 4; do
                    baseline="/var/tmp/rev4-featpot/p2/d2_mlp/LODO_r0_mlp${hidden}/without_${heldout}_r${rep}/result.json"
                    while [[ ! -f "$baseline" ]]; do
                        gate
                        sleep 30
                    done
                done
                gate
                log="$root/log_d2_mlp_compare_${arm}_h${hidden}_${heldout}.log"
                $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                    python scripts/rev4_featpot/candidate_lodo_mlp_compare.py --arm "$arm" \
                        --hidden "$hidden" --heldout "$heldout" > "$log" 2>&1
                sha256sum "$out" "$log"
            fi
        done
    done
done
touch "$root/mlp_d2_done"
printf 'C1-C4 MLP D2 grid complete %s\n' "$(date -u +%FT%TZ)"
