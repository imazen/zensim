#!/usr/bin/env bash
# Registered D2 seven-fold, five-seed H32/H128 P0/P2/perm MLP transfer.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot

gate() {
    local review=$HOME/tmp/zensim-paper/rev4/REVIEW_GMSBANK.md
    [[ -f "$review" ]] && rg -q '^\*\*PROMOTE WITH CORRECTIONS\*\*' "$review"
    rg -q '^\- \*\*Peer GMSD/GMSM:\*\*' "$review"
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    local free_gb
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
}

pilot="$root/p2/d2_mlp/LODO_p2_mlp32/without_konfig_train_r0/result.json"
while [[ ! -f "$pilot" ]]; do
    gate
    sleep 20
done

for arm in r0 p2 p2_perm; do
    for hidden in 32 128; do
        for heldout in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select; do
            for rep in 0 1 2 3 4; do
                out="$root/p2/d2_mlp/LODO_${arm}_mlp${hidden}/without_${heldout}_r${rep}/result.json"
                if [[ ! -f "$out" ]]; then
                    gate
                    log="$root/p2_d2_mlp_${arm}_h${hidden}_without_${heldout}_r${rep}.log"
                    printf 'START D2 MLP %s H%s without=%s rep=%s %s\n' "$arm" "$hidden" "$heldout" "$rep" "$(date -u +%FT%TZ)"
                    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                        python scripts/rev4_featpot/p2_lodo_mlp.py --arm "$arm" --hidden "$hidden" \
                            --heldout "$heldout" --rep "$rep" > "$log" 2>&1
                    sha256sum "$out" "$log"
                fi
            done
        done
    done
done
touch "$root/p2/d2_mlp/fits_done"
for hidden in 32 128; do
    for heldout in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select; do
        out="$root/p2/d2_mlp_compare/LODO_${heldout}_mlp${hidden}.json"
        if [[ ! -f "$out" ]]; then
            gate
            log="$root/p2_d2_mlp_compare_h${hidden}_without_${heldout}.log"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/p2_lodo_mlp_compare.py --hidden "$hidden" \
                    --heldout "$heldout" > "$log" 2>&1
            sha256sum "$out" "$log"
        fi
    done
done
touch "$root/p2/d2_mlp/compare_done"
python scripts/rev4_featpot/p2_report.py
printf 'P2 D2 MLP complete %s\n' "$(date -u +%FT%TZ)"
