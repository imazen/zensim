#!/usr/bin/env bash
# geometry lane runner — stages 1-3 (post-build). Idempotent per stage:
# each stage checks its outputs before doing heavy work. Everything heavy
# goes through ~/tmp/devin/heavy (shared lock).
set -uo pipefail

OUT=/mnt/v/output/zensim/geometry-2026-09-21
WS=/home/lilith/work/zen/zensim--geometry
GM=$HOME/tmp/devin/geometry-target/release/examples/geometry_matrix
HEAVY=$HOME/tmp/devin/heavy
LOG=$HOME/tmp/devin/lane_geometry.log

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { echo "[$(ts)] $*" | tee -a "$OUT/runner.log"; }

# The 14 Stage-1 cells: baseline + 13 OFAT arms. `;`-separated (the
# `@oy,ox` token contains a comma).
CELLS="bin121.local.n5"
CELLS="$CELLS;box2.local.n5"
CELLS="$CELLS;bin1331.local.n5"
CELLS="$CELLS;mitchell.local.n5"
CELLS="$CELLS;lanczos3.local.n5"
CELLS="$CELLS;bin121.lap.n5"
CELLS="$CELLS;bin121.boxres.n5"
CELLS="$CELLS;bin121.local.n3"
CELLS="$CELLS;bin121.local.n7"
CELLS="$CELLS;bin121.local.n4"
CELLS="$CELLS;bin121.local.n8"
CELLS="$CELLS;box2.boxres.n5.xyb"
CELLS="$CELLS;bin121.local.n5@2,2"

stage() {
  local name=$1; shift
  if [ -f "$OUT/done/$name" ]; then say "SKIP $name (done)"; return 0; fi
  say "START $name"
  echo "$(ts) devin-geometry stage $name start" >> "$LOG"
  "$@"
  local rc=$?
  echo "$(ts) devin-geometry stage $name rc=$rc" >> "$LOG"
  if [ $rc -eq 0 ]; then touch "$OUT/done/$name"; say "DONE $name"; else say "FAIL $name rc=$rc"; fi
  return $rc
}

mkdir -p "$OUT/done"

# Stage 0: release build of the driver + library (also runs geom tests).
build() {
  cd "$WS/zensim" && $HEAVY --mem 24G --jobs 8 -- env \
    CARGO_TARGET_DIR=$HOME/tmp/devin/geometry-target \
    cargo test --release -p zensim --features training dvifm::geom -- --nocapture \
  && cd "$WS/zensim-bench" && $HEAVY --mem 24G --jobs 8 -- env \
    CARGO_TARGET_DIR=$HOME/tmp/devin/geometry-target \
    cargo build --release --example geometry_matrix \
    --features 'training zen-decode'
}

# Stage 1a: per-(cell,plane,level) gate-kappa census on the TRAIN-fit
# subset (label-free; the stability screen uses it to normalise f1_gate).
kappa() {
  $HEAVY -- "$GM" --mode kappa --pairs "$OUT/pairs/train_fit4k.tsv" \
        --cells "$CELLS" --rows 4096 --out "$OUT/kappa"
}

# Stage 1b: label-free stability on 192 TRAIN pairs x 11 transforms.
stability() {
  $HEAVY -- "$GM" --mode stability --pairs "$OUT/pairs/stability192.tsv" \
        --cells "$CELLS" --rows 192 --kappa "$OUT/kappa/kappa.json" \
        --out "$OUT/stability"
}

# Stage 1c: 1024^2 one-thread cost screen, >=20 interleaved rounds.
# dvifm_geometry_run is serial — the "1 pinned thread" protocol holds
# inside the measured region regardless of the harness's thread count.
cost1() {
  $HEAVY -- "$GM" --mode cost --cells "$CELLS" --sizes 1024 --rounds 20 \
        --threads 1 --out "$OUT/cost"
}

# Stage 1 analysis: pick survivors (emits stage1.json/.md; survivors are
# decided from the stability gate — the cost column is information).
analyze1() {
  python3 "$OUT/tools/stage1_analyze.py" "$OUT"
}

# Stage 2a: extract the fit4k + dev-leg caches for the survivor cells.
# SURVIVORS is a `;`-separated cell list read from stage1.json by the
# caller (e.g. `SURVIVORS="$(jq -r '.survivors|join(";")' stage1.json)"`).
extract() {
  local cells="$1"
  for leg in train_fit4k codec_dev human_dev kadid135 konfig_val safesyn_sub; do
    $HEAVY -- "$GM" --mode extract \
          --pairs "$OUT/pairs/$leg.tsv" --cells "$cells" \
          --out "$OUT/cache/$leg" || return 1
  done
}

# Stage 2b: cells.json for fit_geometry.py — baseline uses the existing
# joint-core caches via the index views in cache/baseline_view/.
cells_json() {
  local cells="$1" ; local out="$2"
  python3 - "$cells" "$out" <<'PY'
import json, sys
from pathlib import Path
OUT = Path('/mnt/v/output/zensim/geometry-2026-09-21')
cells = sys.argv[1].split(';')
PLANES = ('ycbcr_y', 'ycbcr_cb', 'ycbcr_cr')
BV = OUT / 'cache' / 'baseline_view'
j = {"__pairs__": {
    "train_fit4k": str(OUT / 'pairs' / 'train_fit4k.tsv'),
    "codec_dev": str(OUT / 'pairs' / 'codec_dev.tsv'),
    "human_dev": str(OUT / 'pairs' / 'human_dev.tsv'),
    "kadid135": str(OUT / 'pairs' / 'kadid135.tsv'),
    "konfig_val": str(OUT / 'pairs' / 'konfig_val.tsv'),
    "safesyn_sub": str(OUT / 'pairs' / 'safesyn_sub.tsv')}}
XYB = ('xyb_x', 'xyb_y', 'xyb_b')
DEV_LEGS = ('codec_dev', 'human_dev', 'kadid135', 'konfig_val',
            'safesyn_sub')
for c in cells:
    planes = XYB if c.endswith('.xyb') else PLANES
    if c == 'bin121.local.n5':
        fit = {p: str(BV / f'fit4k_{p}.bin') for p in planes}
        dev = {leg: {p: str(BV / f'{leg}_{p}.bin') for p in planes}
               for leg in DEV_LEGS}
    else:
        fit = {p: str(OUT / 'cache' / 'train_fit4k' / f'{c}_{p}.bin')
               for p in planes}
        dev = {leg: {p: str(OUT / 'cache' / leg / f'{c}_{p}.bin')
                     for p in planes}
               for leg in DEV_LEGS}
    j[c] = {"planes": list(planes), "fit": fit, "dev": dev}
Path(sys.argv[2]).write_text(json.dumps(j, indent=1) + '\n')
PY
}

# Stage 2c: gate fits, 3 paired seeds, histogram-grid stage only.
fit2() {
  local cells_json="$1"
  $HEAVY -- python3 "$WS/tools/joint_core/fit_geometry.py" \
        "$cells_json" "$OUT/pairs/train_fit4k.tsv" "$OUT/fits" \
        --seeds 9201,9207,9211 --rows 1024
}

# Stage 3: full cost protocol on the recommended cell + baseline.
cost3() {
  local cells="${1:-bin121.local.n5}"
  $HEAVY -- "$GM" --mode cost --cells "$cells" \
        --sizes 64,256,1024,2048,4096 --rounds 30 --threads 1 \
        --out "$OUT/cost3" || return 1
  $HEAVY -- "$GM" --mode cost --cells "$cells" \
        --sizes 64,256,1024,2048,4096 --rounds 30 --threads 8 \
        --out "$OUT/cost3"
}

case "${1:-all}" in
  build) stage build build ;;
  kappa) stage kappa kappa ;;
  stability) stage stability stability ;;
  cost1) stage cost1 cost1 ;;
  analyze1) stage analyze1 analyze1 ;;
  extract) stage extract extract "$2" ;;
  cells_json) cells_json "$2" "$3" ;;
  fit2) stage fit2 fit2 "$2" ;;
  cost3) stage cost3 cost3 "$2" ;;
  all)
    stage build build || exit 1
    stage kappa kappa || exit 1
    stage stability stability || exit 1
    stage cost1 cost1 || exit 1
    stage analyze1 analyze1 || exit 1
    echo "Stage 1 done — pick survivors, then: $0 extract <cells>; $0 cells_json <cells> <json>; $0 fit2 <json>; $0 cost3 <cells>"
    ;;
  *) echo "usage: $0 [build|kappa|stability|cost1|analyze1|extract <cells>|cells_json <cells> <json>|fit2 <json>|cost3 <cells>|all]"; exit 2 ;;
esac
