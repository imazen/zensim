#!/usr/bin/env bash
# P2 and two-column matched control: registered H32/H128 five-seed Latin grid.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

gate() {
    local review=$HOME/tmp/zensim-paper/rev4/REVIEW_GMSBANK.md
    [[ -f "$review" ]] && rg -q '^\*\*PROMOTE WITH CORRECTIONS\*\*' "$review"
    rg -q '^\- \*\*Peer GMSD/GMSM:\*\*' "$review"
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    local free_gb
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
}

pilot=/var/tmp/rev4-featpot/p2/mlp/POT_konfig_train_p2_mlp32/o0_r0/result.json
while [[ ! -f "$pilot" ]]; do
    gate
    sleep 20
done

for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
    for arm in p2 p2_perm; do
        for hidden in 32 128; do
            for outer in 0 1 2 3 4 full; do
                for rep in 0 1 2 3 4; do
                    tag="o${outer}"
                    outer_args=(--outer "$outer")
                    if [[ "$outer" == full ]]; then
                        tag=full
                        outer_args=()
                    fi
                    dest="/var/tmp/rev4-featpot/p2/mlp/POT_${set_name}_${arm}_mlp${hidden}/${tag}_r${rep}"
                    if [[ ! -f "$dest/result.json" ]]; then
                        gate
                        log="/var/tmp/rev4-featpot/p2_mlp_${set_name}_${arm}_h${hidden}_${outer}_r${rep}.log"
                        printf 'START FIT %s %s H%s %s r%s %s\n' "$set_name" "$arm" "$hidden" "$outer" "$rep" "$(date -u +%FT%TZ)"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/p2_mlp.py --set "$set_name" --arm "$arm" \
                                --hidden "$hidden" "${outer_args[@]}" --rep "$rep" > "$log" 2>&1
                        sha256sum "$dest/result.json" "$log"
                    fi
                    if [[ "$outer" != full && ! -f "$dest/importance.json" ]]; then
                        gate
                        log="/var/tmp/rev4-featpot/p2_mlp_${set_name}_${arm}_h${hidden}_${outer}_r${rep}_importance.log"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/p2_mlp_importance.py --set "$set_name" --arm "$arm" \
                                --hidden "$hidden" --outer "$outer" --rep "$rep" > "$log" 2>&1
                        sha256sum "$dest/importance.json" "$log"
                    fi
                    if [[ ! -f "$dest/ci.json" ]]; then
                        gate
                        log="/var/tmp/rev4-featpot/p2_mlp_${set_name}_${arm}_h${hidden}_${outer}_r${rep}_ci.log"
                        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                            python scripts/rev4_featpot/p2_mlp_ci.py --set "$set_name" --arm "$arm" \
                                --hidden "$hidden" "${outer_args[@]}" --rep "$rep" > "$log" 2>&1
                        sha256sum "$dest/ci.json" "$log"
                    fi
                done
            done
            out="/var/tmp/rev4-featpot/p2/mlp_aggregate/POT_${set_name}_${arm}_mlp${hidden}/result.json"
            if [[ ! -f "$out" ]]; then
                gate
                log="/var/tmp/rev4-featpot/p2_mlp_aggregate_${set_name}_${arm}_h${hidden}.log"
                $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                    python scripts/rev4_featpot/p2_mlp_aggregate.py --set "$set_name" --arm "$arm" \
                        --hidden "$hidden" > "$log" 2>&1
                sha256sum "$out" "$log"
            fi
        done
    done
done
touch /var/tmp/rev4-featpot/p2/mlp_done
printf 'P2 MLP D1 grid complete %s\n' "$(date -u +%FT%TZ)"
