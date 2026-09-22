#!/usr/bin/env bash
# Phase 2d Part D scoring driver — runs ONLY after all fits are frozen.
# For each fitted artefact: score TRAIN-side dev legs (kadid_dev refs {1,3,5},
# konfig SRC01/03/31/45) and the cross-domain transfer matrix
# (each fit scored on all four fit domains). CID22-B is scored by
# score_cid22b.sh exactly once, separately, after this completes.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
FIT="python3 $OUT/tools/fit_standalone.py"
mkdir -p "$OUT/scores"

dompairs() {
  case $1 in
    cid22a)    echo "$OUT/pairs/cid22a.tsv";;
    tidkadid)  echo "$OUT/pairs/tidkadid.tsv";;
    imazen26)  echo "$OUT/pairs/imazen26_pairs.tsv";;
    pooled)    echo "$OUT/pairs/pooled_cid22a_tidkadid.tsv";;
    kadid_dev) echo "$OUT/pairs/kadid_dev.tsv";;
    konfig)    echo "$OUT/pairs/konfig_val.tsv";;
  esac
}
bin_for() {
  case $1 in
    cid22a|imazen26|kadid_dev|konfig)
      echo "$OUT/cache/${1}_${2}.bin";;
    tidkadid)
      echo "$OUT/cache/tid_${2}.bin,$OUT/cache/kadid_train_${2}.bin";;
    pooled)
      echo "$OUT/cache/cid22a_${2}.bin,$OUT/cache/tid_${2}.bin,$OUT/cache/kadid_train_${2}.bin";;
  esac
}
caches_for() { # variant domain -> --cache args
  case $1 in
    luma)    echo "--cache ycbcr_y=$(bin_for $2 ycbcr_y)";;
    xyb-y)   echo "--cache xyb_y=$(bin_for $2 xyb_y)";;
    native3) echo "--cache ycbcr_y=$(bin_for $2 ycbcr_y) \
--cache ycbcr_cb=$(bin_for $2 ycbcr_cb) \
--cache ycbcr_cr=$(bin_for $2 ycbcr_cr)";;
  esac
}

fails=0
for dom in cid22a tidkadid imazen26 pooled; do
  for var in luma native3 xyb-y; do
    fit=$OUT/fits/${dom}_${var}.json
    [ -s "$fit" ] || { echo "MISSING FIT $dom $var"; fails=$((fails+1)); continue; }
    for evaldom in cid22a tidkadid imazen26 pooled kadid_dev konfig; do
      out=$OUT/scores/${dom}_${var}__on__${evaldom}.csv
      [ -s "$out" ] && continue
      $FIT score --fit "$fit" $(caches_for $var $evaldom) \
        --pairs "$(dompairs $evaldom)" --out "$out" \
        || { echo "SCORE-FAIL $dom $var on $evaldom"; fails=$((fails+1)); }
    done
  done
done
# K1 controls on every domain (xyb_y plane — its native plane).
# Output map comes from the CID22-A eval-consts artefact ONLY (train-side);
# eval legs get the frozen artefact. Rank stats are map-free; PLCC is
# computed downstream under a monotone map.
K1=/mnt/v/output/zensim/dvifm-screen-2026-09-19/specs/dvifm-local-fitted-final.json
python3 "$OUT/tools/k1_to_fit.py" "$K1" "$OUT/fits/cid22a_k1.json" \
  "$OUT/fits/k1_control.json"
for evaldom in cid22a tidkadid imazen26 pooled kadid_dev konfig; do
  out=$OUT/scores/k1__on__${evaldom}.csv
  [ -s "$out" ] && continue
  $FIT score --fit "$OUT/fits/k1_control.json" \
    --cache xyb_y="$(bin_for $evaldom xyb_y)" --pairs "$(dompairs $evaldom)" \
    --out "$out" || { echo "K1-FAIL $evaldom"; fails=$((fails+1)); }
done
echo "SCORE DRIVER DONE fails=$fails"
