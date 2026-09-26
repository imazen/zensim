#!/usr/bin/env bash
# Run disjoint P0/P2 fit cells concurrently, within the shared heavy lock.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot
pilot="$root/mlp_accel_pilot.log"

while [[ ! -f "$pilot" ]] || ! rg -q 'run-heavy: done rc=0' "$pilot"; do
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    sleep 20
done

batch=0
while [[ -e "$root/mlp_accel/batch_$(printf '%04d' "$((batch + 1))").log" ]]; do
    batch=$((batch + 1))
done
while true; do
    if [[ -e $HOME/tmp/zensim-paper/rev4/FLEET_FITS_GO.md ]]; then
        python scripts/rev4_featpot/p0_fleet_pause.py --runner accelerator
        exit 0
    fi
    p0=$(find "$root/fits" -path '*mlp*/result.json' | wc -l)
    p2=$(find "$root/p2/mlp" -path '*mlp*/result.json' | wc -l)
    printf 'COUNT p0=%s/960 p2=%s/960 %s\n' "$p0" "$p2" "$(date -u +%FT%TZ)"
    if (( p0 >= 960 && p2 >= 960 )); then
        break
    fi
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
    batch=$((batch + 1))
    log="$root/mlp_accel/batch_$(printf '%04d' "$batch").log"
    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
        python scripts/rev4_featpot/accelerate_mlp.py --batch 8 > "$log" 2>&1
    sha256sum "$log"
    tail -2 "$log"
done
touch "$root/mlp_accel/fits_done"
printf 'P0/P2 fit grids complete %s\n' "$(date -u +%FT%TZ)"
