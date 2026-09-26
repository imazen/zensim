#!/usr/bin/env bash
# Registered 200 half-reference lasso stability draws for the candidate/restore arms (frozen 1-SE lambda).
# VARIANTS=real runs the arms, VARIANTS=perm their matched controls. Needs the arm's D1 lasso full fit.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export REV4_FIT_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/bake_dial_refit
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/main_build_20260924/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot/candidates
mkdir -p "$root/stability"
for base in ${ARMS:-p1 p3 c7 csfw a1 a1w b2 a1m b2m c1 c2 c3 c4 all b1 b1s c8n rall}; do
    arm=$base; [[ "${VARIANTS:-real}" == perm ]] && arm=${base}_perm
    for set_name in kadid_train tid2013 konfig_train konjnd_bpg_train cid22_a25 aic3 kadid_select konfig_val; do
        out="$root/stability/POT_${set_name}_${arm}_lasso/result.json"
        if [[ ! -f "$out" ]]; then
            [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
            free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
            (( free_gb >= 20 ))
            log="$root/log_stability_${set_name}_${arm}.log"
            printf 'START STABILITY %s %s %s\n' "$set_name" "$arm" "$(date -u +%FT%TZ)"
            $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
                python scripts/rev4_featpot/candidate_stability.py --set "$set_name" --arm "$arm" > "$log" 2>&1
            printf 'END STABILITY %s %s %s\n' "$set_name" "$arm" "$(date -u +%FT%TZ)"
            sha256sum "$out" "$log"
        fi
    done
done
touch "$root/stability_${VARIANTS:-real}_done"
printf 'arm stability (%s) complete %s\n' "${VARIANTS:-real}" "$(date -u +%FT%TZ)"
