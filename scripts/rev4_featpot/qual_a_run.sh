#!/usr/bin/env bash
# QUAL-A (E33 A release gates, C and production seed 0 report-only): after the gate runner's queue, run the
# Rust-surface audit, the scorecard runtime grid, Rev5 correctness and the label-free workspace tests.
# Heavy steps take heavy.lock + run-heavy; timing/RSS lock per segment. Never reads labels.
set -euo pipefail
Q=${1:?usage: qual_a_run.sh QUAL_ROOT}
repo=$(realpath "$(dirname "$0")/../..")
lock=$HOME/tmp/zensim-paper/rev4/heavy.lock
heavy=(flock "$lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 --)
capped=("$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 --)
export TMPDIR="$HOME/tmp/qual-a" SPEEDQ_LOCK_OWNER=e33-opus ZEN_PANEL_BIN=/mnt/v/output/zensim/e33-impl-2026-10-09/bin-final2/panel
mkdir -p "$TMPDIR"
exec > >(tee -a "$Q/run.log") 2>&1
step() { printf '==== QUAL-A %s %s\n' "$1" "$(date -u +%FT%TZ)"; }
cd "$repo"
step wait-gates
until grep -q QUAL-GATES-DONE "$Q/gates.log"; do sleep 60; done
models=$(python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(' '.join(m[k]['path'] for k in ('a','c','seed0')))" "$Q/MODELS.json")
step surface
if [[ ! -s $Q/native-synthetic.json ]]; then
    # shellcheck disable=SC2086
    "${heavy[@]}" env ZENSIM_FORMULA_REV=5 "$Q/bin/serve_custom_bake" --prodqual $models > "$Q/native-synthetic.json.partial"
    mv "$Q/native-synthetic.json.partial" "$Q/native-synthetic.json"
    python3 scripts/prodqual_label_free.py --native "$Q/native-synthetic.json" --output "$Q/native-summary.json"
fi
step runtime
binary=$Q/bin/costcmp_instrument
[[ -f $Q/runtime/parity/PREFLIGHT_PASS.json ]] || "${heavy[@]}" python3 scripts/demos/costcmp_run.py parity \
    --binary "$binary" --dest "$Q/runtime/parity" --qual-grid "$Q/runtime-candidates.json"
"${capped[@]}" python3 scripts/demos/costcmp_run.py timing --binary "$binary" --dest "$Q/runtime/timing" \
    --parity "$Q/runtime/parity/PREFLIGHT_PASS.json" --analyzer /mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer \
    --lock "$lock" --qual-grid "$Q/runtime-candidates.json"
"${capped[@]}" python3 scripts/demos/costcmp_run.py rss --binary "$binary" --dest "$Q/runtime/rss" \
    --parity "$Q/runtime/parity/PREFLIGHT_PASS.json" --lock "$lock" --qual-grid "$Q/runtime-candidates.json"
step rev5-correctness
[[ -f $Q/rev5-parity.rc ]] || { "${heavy[@]}" just prodqual-rev5 > "$Q/rev5-parity.log" 2>&1; echo $? > "$Q/rev5-parity.rc"; } || true
step workspace-tests
[[ -f $Q/workspace-tests.rc ]] || { "${heavy[@]}" just prodqual-workspace-tests > "$Q/workspace-tests.log" 2>&1; echo $? > "$Q/workspace-tests.rc"; } || true
printf '==== QUAL-A RUN COMPLETE %s\n' "$(date -u +%FT%TZ)"
