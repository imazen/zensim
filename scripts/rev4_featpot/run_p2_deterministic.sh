#!/usr/bin/env bash
# Reviewed P2 peer control, D1/D2, with its two-column within-ref permutation.
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

while [[ ! -f /var/tmp/rev4-featpot/p2/standalone.json ]]; do
    gate
    sleep 20
done

for model in bvls linear; do
    for arm in p2 p2_perm; do
        for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
            out="/var/tmp/rev4-featpot/p2/d1/POT_${set_name}_${arm}_${model}/result.json"
            if [[ ! -f "$out" ]]; then
                gate
                log="/var/tmp/rev4-featpot/p2_d1_${set_name}_${arm}_${model}.log"
                printf 'START D1 %s %s %s %s\n' "$set_name" "$arm" "$model" "$(date -u +%FT%TZ)"
                $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                    python scripts/rev4_featpot/p2_linear.py --set "$set_name" --arm "$arm" --model "$model" > "$log" 2>&1
                printf 'END D1 %s %s %s %s\n' "$set_name" "$arm" "$model" "$(date -u +%FT%TZ)"
                sha256sum "$out" "$log"
            fi
        done
    done
done

for model in bvls linear; do
    for set_name in kadid_train tid2013 konfig_train cid22_a25 aic3 kadid_select konfig_val; do
        out="/var/tmp/rev4-featpot/p2/d1/POT_${set_name}_${model}_compare.json"
        if [[ ! -f "$out" ]]; then
            gate
            log="/var/tmp/rev4-featpot/p2_d1_${set_name}_${model}_compare.log"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/p2_compare.py --scope d1 --set "$set_name" --model "$model" > "$log" 2>&1
            sha256sum "$out" "$log"
        fi
    done
done

for model in bvls linear; do
    for arm in p2 p2_perm; do
        out="/var/tmp/rev4-featpot/p2/d2/LODO_${arm}_${model}/result.json"
        if [[ ! -f "$out" ]]; then
            gate
            log="/var/tmp/rev4-featpot/p2_d2_${arm}_${model}.log"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/p2_lodo.py --arm "$arm" --model "$model" > "$log" 2>&1
            sha256sum "$out" "$log"
        fi
    done
    out="/var/tmp/rev4-featpot/p2/d2/LODO_${model}_compare.json"
    if [[ ! -f "$out" ]]; then
        gate
        log="/var/tmp/rev4-featpot/p2_d2_${model}_compare.log"
        $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
            python scripts/rev4_featpot/p2_compare.py --scope d2 --model "$model" > "$log" 2>&1
        sha256sum "$out" "$log"
    fi
done
touch /var/tmp/rev4-featpot/p2/deterministic_done
printf 'P2 deterministic D1/D2 complete %s\n' "$(date -u +%FT%TZ)"
