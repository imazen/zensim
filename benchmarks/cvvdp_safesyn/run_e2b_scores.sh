#!/usr/bin/env bash
# cvvdp-safesyn E2b: score 7 display arms x 3 TRAIN legs with the quarantine
# binary. One `zenmetrics batch` run per (leg, display); output column carries
# the display suffix (standard_4k keeps the plain `cvvdp_cpu_imazen_v0_1_0`).
# --group-by-ref amortizes the reference decode across each ladder.
set -euo pipefail

BIN="${BIN:-/var/tmp/cvvdp-safesyn/target-zenmetrics/release/zenmetrics}"
OUT="${OUT:-/var/tmp/cvvdp-safesyn/e2b/scores}"
JOBS="${JOBS:-8}"
mkdir -p "$OUT" /var/tmp/cvvdp-safesyn/logs

declare -A LEGS=(
  [kadid]=/var/tmp/cvvdp-safesyn/e2b/kadid_train_pairs.tsv
  [tid]=/var/tmp/cvvdp-safesyn/e2b/tid_train_pairs.tsv
  [konfig]=/var/tmp/cvvdp-safesyn/e2b/konfig_train_pairs.tsv
)

DISPLAYS=(standard_4k sdr_4k_30 standard_fhd sdr_fhd_24 standard_phone iphone_14_pro modern_oled_phone_indoor)

for leg in kadid tid konfig; do
  for disp in "${DISPLAYS[@]}"; do
    out="$OUT/${leg}__${disp}.tsv"
    log="/var/tmp/cvvdp-safesyn/logs/e2b_${leg}__${disp}.log"
    if [[ "$disp" == "standard_4k" ]]; then
      dargs=()
    else
      dargs=(--display-model "$disp")
    fi
    if [[ -s "$out" ]]; then
      echo "[skip] $out exists" >&2
      continue
    fi
    /usr/bin/time -v "$BIN" batch --metric cvvdp "${dargs[@]}" \
      --pairs "${LEGS[$leg]}" --output "$out" \
      --group-by-ref --jobs "$JOBS" >"$log" 2>&1
    echo "[done] $leg $disp" >&2
  done
done
