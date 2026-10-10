#!/usr/bin/env bash
# E33 post-fit: waits for the harvest chain's all-126 marker, then runs the registered E21 assessment (9.1), the
# label-free gates on full-data seed 0 (9.2), the runtime grid (9.3), seeds 1-2 report-only gates and the summary.
# Never schedules fleet work. Heavy steps take heavy.lock + run-heavy one at a time. Runs in its own herdr tab.
set -euo pipefail
here=$(dirname "$(realpath "$0")")
repo=$(realpath "$here/../..")
E=/mnt/v/output/zensim/e33-impl-2026-10-09
G=$E/gates
log=$E/launch/postfit.log
lock=$HOME/tmp/zensim-paper/rev4/heavy.lock
heavy=(flock "$lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 --)
# Timing and RSS take heavy.lock per segment themselves (speedq_run segment_lock), so only run-heavy wraps them.
capped=("$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 --)
analyzer=/mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer
export TMPDIR="${TMPDIR:-$HOME/tmp/e33}" MPLCONFIGDIR="$HOME/tmp/e33/mpl"
export REV4_V2_BIN_DIR=$E/bin-final2 ZEN_PANEL_BIN=$E/bin-final2/panel
export PYTHONPATH="$repo/scripts:$repo/scripts/rev4_featpot:$repo/scripts/demos"
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
exec > >(tee -a "$log") 2>&1
step() { printf '==== E33 POSTFIT %s %s\n' "$1" "$(date -u +%FT%TZ)"; }
step wait
until grep -q '==== E33 ALL 126 CELLS VERIFIED AND INSTALLED' "$E/launch/harvest-chain.log"; do
    if grep -q 'E33 HARVEST FAILED' "$E/launch/harvest-chain.log"; then
        printf 'E33 POSTFIT STOPPED: harvest chain failed\n'
        exit 1
    fi
    sleep 120
done
cd "$repo"
step e21
"${heavy[@]}" python3 "$here/e33_score.py" e21 --out "$E/assessment-e33"
step gates-seed0
gates() {
    local seed=$1
    for arm in a c; do
        "${heavy[@]}" python3 "$here/e33_gates.py" nearid --arm "$arm" --seed "$seed"
        for grid in standard ladder; do
            "${heavy[@]}" python3 "$here/e33_gates.py" verdict --arm "$arm" --seed "$seed" --grid "$grid"
        done
        "${heavy[@]}" python3 "$here/e33_gates.py" steer --arm "$arm" --seed "$seed"
    done
    "${heavy[@]}" python3 "$here/e33_gates.py" identity --seed "$seed"
}
gates 0
step runtime
candidates=$(python3 "$here/e33_gates.py" runtime-candidates)
binary=$G/bin/costcmp_instrument
"${heavy[@]}" python3 scripts/demos/costcmp_run.py parity --binary "$binary" --dest "$G/runtime/parity" --e33-grid "$candidates"
"${capped[@]}" python3 scripts/demos/costcmp_run.py timing --binary "$binary" --dest "$G/runtime/timing" \
    --parity "$G/runtime/parity/PREFLIGHT_PASS.json" --analyzer "$analyzer" --lock "$lock" --e33-grid "$candidates"
"${capped[@]}" python3 scripts/demos/costcmp_run.py rss --binary "$binary" --dest "$G/runtime/rss" \
    --parity "$G/runtime/parity/PREFLIGHT_PASS.json" --lock "$lock" --e33-grid "$candidates"
step summary
python3 "$here/e33_gates.py" summary
step report-only-seeds-1-2
gates 1
gates 2
printf '==== E33 POSTFIT COMPLETE %s\n' "$(date -u +%FT%TZ)"
