#!/usr/bin/env bash
# Measure-first gate: the first N SafeSyn pairs through (1) the baseline extractor at --full-gmsbank
# (main@origin), (2) the candidate at --full-gmsbank (families off), (3) the candidate with all four
# restored families plus the whole existing surface requested (`prefix`). Reports wall time, pairs/s, and the f0..f1501 bit identity between all three.
# Run through the shared lock. Usage: gate_sample.sh [N=2000]
set -euo pipefail
N=${1:-2000}
ROOT=/var/tmp/restore-cuts
REPO=/home/lilith/work/zen/zensim--restore-cuts
export ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=8
pairs=$ROOT/pairs/safesyn_gate$N.tsv
head -n $((N + 1)) "$ROOT/pairs/safesyn.tsv" > "$pairs"
run() { # name binary extra-args...
    local name=$1 bin=$2; shift 2
    rm -f "$ROOT/raw/gate_$name.csv" "$ROOT/raw/gate_$name.csv".*
    /usr/bin/time -f "WALL_SECONDS=%e MAXRSS_KB=%M" "$bin" --corpus pairs-tsv --path "$pairs" \
        --out "$ROOT/raw/gate_$name.csv" --input-contract legacy-rgb8 "$@" 2>&1 | grep -E '^(scored|Wrote|WALL_SECONDS)'
}
echo "GATE_START n=$N utc=$(date -u +%FT%TZ)"
run base "$ROOT/bin/extract_base" --full-gmsbank
run candoff "$ROOT/bin/extract_cand" --full-gmsbank
run candall "$ROOT/bin/extract_cand" --restore-cuts prefix,mapdev,z1max,gmsnative,dvifmgate
python3 "$REPO/scripts/restore_cuts/compare_prefix.py" "$ROOT/raw/gate_base.csv" "$ROOT/raw/gate_candoff.csv" 1502
python3 "$REPO/scripts/restore_cuts/compare_prefix.py" "$ROOT/raw/gate_base.csv" "$ROOT/raw/gate_candall.csv" 1502
sha256sum "$pairs" "$ROOT"/raw/gate_{base,candoff,candall}.csv "$ROOT/bin/extract_base" "$ROOT/bin/extract_cand"
echo "GATE_END utc=$(date -u +%FT%TZ)"
