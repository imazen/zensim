#!/usr/bin/env bash
# Phase 2d Part D standalone fits — Amendment-1 rerun.
# Domains: cid22a, tidkadid, imazen26, pooled(cid22a+tidkadid).
# Variants: native3 (Y'+Cb+Cr — PRIMARY), luma (Y' only), xyb-y (ablation).
# Order (supervisor 2026-09-19): cid22a native3 -> luma -> xyb-y, then the
# TID/KADID domain same order, then imazen26, then pooled. K1 eval-consts
# (Amendment-1 map) per domain after its variants.
# Pooled domains use virtual CatCache concat (comma-joined bins) so no
# duplicate block bytes are materialised. Resumable: skips existing fits.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
K1=/mnt/v/output/zensim/dvifm-screen-2026-09-19/specs/dvifm-local-fitted-final.json
# fork-pool safety: pin BLAS to one thread so forked fit workers never
# inherit/contend a threaded OpenBLAS pool (the workload is elementwise,
# not BLAS-bound — no numeric effect)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1
FIT="python3 $OUT/tools/fit_standalone.py"
mkdir -p "$OUT/fits" "$OUT/surfaces" "$OUT/logs"

declare -A PAIRS=(
  [cid22a]=$OUT/pairs/cid22a.tsv
  [tidkadid]=$OUT/pairs/tidkadid.tsv
  [imazen26]=$OUT/pairs/imazen26_pairs.tsv
  [pooled]=$OUT/pairs/pooled_cid22a_tidkadid.tsv
)
# bin list per (domain, plane), comma-joined for CatCache
bin_for() { # dom plane
  case $1 in
    cid22a|imazen26) echo "$OUT/cache/${1}_${2}.bin";;
    tidkadid) echo "$OUT/cache/tid_${2}.bin,$OUT/cache/kadid_train_${2}.bin";;
    pooled)   echo "$OUT/cache/cid22a_${2}.bin,$OUT/cache/tid_${2}.bin,$OUT/cache/kadid_train_${2}.bin";;
  esac
}

fails=0
run_one() { # dom var
  local dom=$1 var=$2
  local tsv="${PAIRS[$dom]}"
  local outj=$OUT/fits/${dom}_${var}.json
  [ -s "$outj" ] && { echo "SKIP $dom $var"; return 0; }
  local caches
  case $var in
    luma)    caches="--cache ycbcr_y=$(bin_for $dom ycbcr_y)";;
    xyb-y)   caches="--cache xyb_y=$(bin_for $dom xyb_y)";;
    native3) caches="--cache ycbcr_y=$(bin_for $dom ycbcr_y) \
--cache ycbcr_cb=$(bin_for $dom ycbcr_cb) \
--cache ycbcr_cr=$(bin_for $dom ycbcr_cr)";;
  esac
  echo "=== FIT $dom $var ==="
  $FIT fit --variant "$var" $caches --pairs "$tsv" \
    --init-spec "$K1" --out "$outj" \
    --surfaces-dir "$OUT/surfaces/${dom}_${var}" \
    > "$OUT/logs/fit_${dom}_${var}.log" 2>&1
  rc=$?
  tail -3 "$OUT/logs/fit_${dom}_${var}.log"
  if [ $rc -ne 0 ]; then
    echo "FAIL $dom $var (rc=$rc, see log)"
    if grep -q "SANITY FAIL" "$OUT/logs/fit_${dom}_${var}.log"; then
      echo "SANITY GATE TRIPPED on $dom $var — stopping the driver"
      exit 3
    fi
    fails=$((fails+1))
  fi
}

# K1 control per domain (xyb_y plane) — eval-consts, cheap
k1_control() { # dom
  local dom=$1
  local k1j=$OUT/fits/${dom}_k1.json
  [ -s "$k1j" ] && return 0
  $FIT eval-consts --spec "$K1" --variant xyb-y \
    --cache "$(bin_for $dom xyb_y)" --pairs "${PAIRS[$dom]}" \
    > "$k1j" || { echo "FAIL $dom k1"; fails=$((fails+1)); }
  # the cid22a control doubles as the B-read scoring artefact's map source
  if [ "$dom" = cid22a ] && [ -s "$k1j" ]; then
    python3 "$OUT/tools/k1_to_fit.py" "$K1" "$k1j" "$OUT/fits/k1_control.json"
  fi
}

# --- supervisor order: cid22a native3 -> luma -> xyb-y, then tidkadid ---
for dom in cid22a tidkadid imazen26 pooled; do
  for var in native3 luma xyb-y; do
    run_one "$dom" "$var"
  done
  k1_control "$dom"
done
echo "FIT DRIVER DONE fails=$fails"
