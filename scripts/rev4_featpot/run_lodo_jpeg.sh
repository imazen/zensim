#!/usr/bin/env bash
# Registered R0 D2 JPEG-versus-other transfer contrast; saved LODO fits only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot
for model in bvls linear; do
    output="$root/lodo_jpeg/LODO_r0_${model}/result.json"
    if [[ -f "$output" ]]; then
        sha256sum "$output"
        continue
    fi
    [[ ! -e ${QUOTA_STOP_FILE:-$HOME/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md} ]]
    free_gb=$(df -BG --output=avail /home | tail -1 | tr -dc '0-9')
    (( free_gb >= 20 ))
    log="$root/lodo_r0_${model}_jpeg_gap.log"
    printf 'START %s %s\n' "$model" "$(date -u +%FT%TZ)"
    $HOME/tmp/devin/heavy --mem 16G --jobs 8 -- \
        python scripts/rev4_featpot/lodo_jpeg_gap.py --model "$model" > "$log" 2>&1
    printf 'END %s %s\n' "$model" "$(date -u +%FT%TZ)"
    sha256sum "$output" "$log"
done
printf 'LODO JPEG contrasts complete %s\n' "$(date -u +%FT%TZ)"
