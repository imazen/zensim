#!/usr/bin/env bash
# Extract the restored-cut families for one bank set and bind them (Part B conventions).
# Usage (through the shared lock): ~/tmp/devin/heavy --mem 16G --jobs 8 -- bash scripts/restore_cuts/run_set.sh <set> [limit]
# With a limit the run is a measure-first sample: pairs are cut to the first <limit> rows and NO sidecar is bound.
set -euo pipefail
set_name=$1
limit=${2:-}
# SIDECAR_SPEC=signed: the SIGNEDFEAT families (texgain, satsign) at Rev4 under /var/tmp/signedfeat/sidecar
# (same bank keys, same conventions; bank_sidecar.py reads the same variable).
if [[ ${SIDECAR_SPEC:-restore} == signed ]]; then
    ROOT=${RESTORE_ROOT:-/var/tmp/signedfeat/sidecar}; TOKENS=texgain,satsign; REV=4
else
    ROOT=${RESTORE_ROOT:-/var/tmp/restore-cuts}; TOKENS=mapdev,z1max,gmsnative,dvifmgate; REV=3
fi
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)  # repo root, derived at run time
bin=$ROOT/bin/extract_cand
suffix=
pairs=$ROOT/pairs/$set_name.tsv
if [[ -n $limit ]]; then
    suffix=_sample$limit
    head -n $((limit + 1)) "$pairs" > "$ROOT/pairs/$set_name$suffix.tsv"
    pairs=$ROOT/pairs/$set_name$suffix.tsv
fi
csv=$ROOT/raw/$set_name$suffix.csv
audit=$ROOT/raw/$set_name$suffix.audit.jsonl
[[ -x $bin && -s $pairs ]] || { echo "missing binary or pairs: $set_name"; exit 65; }
[[ ! -e $csv && ! -e $audit ]] || { echo "output already exists: $set_name$suffix"; exit 66; }
export ZENSIM_FORMULA_REV=$REV ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS=8
echo "RESTORE_START set=$set_name$suffix utc=$(date -u +%FT%TZ)"
(while :; do
    date -u +"%FT%TZ claude-sonnet-restore-cuts extracting $set_name$suffix" > "$REPO/.workongoing"
    sleep 90
done) &
marker_pid=$!
trap 'kill "$marker_pid" 2>/dev/null || true' EXIT
/usr/bin/time -f 'WALL_SECONDS=%e MAXRSS_KB=%M' "$bin" --corpus pairs-tsv --path "$pairs" --out "$csv" \
    --restore-cuts $TOKENS --audit-jsonl "$audit" --input-contract legacy-rgb8
if [[ -z $limit ]]; then
    python3 "$REPO/scripts/restore_cuts/bank_sidecar.py" bind --set "$set_name" --binary "$bin" \
        --build-meta "$ROOT/build_meta.json"
fi
sha256sum "$pairs" "$csv" "$audit" "$csv.manifest.json"
echo "RESTORE_END set=$set_name$suffix utc=$(date -u +%FT%TZ)"
