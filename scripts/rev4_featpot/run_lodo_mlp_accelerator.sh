#!/usr/bin/env bash
# Bounded concurrent D2 P0/P2/perm fits; original serial runner remains live.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot
pilot="$root/p2_d2_mlp_accel_pilot.log"

while [[ ! -f "$pilot" ]] || ! rg -q 'run-heavy: done rc=0' "$pilot"; do
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    sleep 20
done
batch=0
while [[ -e "$root/p2/d2_mlp_accel/batch_$(printf '%04d' "$((batch + 1))").log" ]]; do
    batch=$((batch + 1))
done
while true; do
    count=$(find "$root/p2/d2_mlp" -path '*/result.json' | wc -l)
    printf 'COUNT d2_mlp=%s/210 %s\n' "$count" "$(date -u +%FT%TZ)"
    if (( count >= 210 )); then
        break
    fi
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
    batch=$((batch + 1))
    log="$root/p2/d2_mlp_accel/batch_$(printf '%04d' "$batch").log"
    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
        python scripts/rev4_featpot/accelerate_lodo_mlp.py --batch 8 > "$log" 2>&1
    sha256sum "$log"
    tail -2 "$log"
done
touch "$root/p2/d2_mlp/accelerated_fits_done"
printf 'D2 MLP replicate grids complete %s\n' "$(date -u +%FT%TZ)"
