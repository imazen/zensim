#!/usr/bin/env bash
# VALIDATE-slice selection rerun (user directive 2026-08-28; methodology-audit
# fix). Builds validate-family slices, family-filters them, stages a validate
# features-root, rescores the eligibility set, paired CIs vs incumbent.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
VS=/mnt/v/zen/zensim-training/valsel-2026-08-28
HB=$HOME/tmp/valsel/heartbeat
mkdir -p "$VS" "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
say "build validate slices"
nice -n19 ionice -c3 python3 "$REPO/scripts/canonical_corpus/build_eval_slices_944.py" \
    --views-root "$ROOT/bigcodec" --split validate --out-root "$VS" >> "$HB" 2>&1 \
    || { say "BUILD FAILED"; exit 6; }
say "family-filter to validate bucket + channel-A exclusion"
nice -n19 python3 "$REPO/scripts/validate_slice_family_filter.py" "$VS" >> "$HB" 2>&1 \
    || { say "FILTER FAILED"; exit 6; }
say "stage features-root"
mkdir -p "$VS/root"
for f in "$ROOT"/*.parquet; do ln -sf "$f" "$VS/root/$(basename "$f")"; done
for n in imazen26 nonphoto hfnlproxy; do cp -f "$VS/ext_${n}.parquet" "$VS/root/ext_${n}.parquet"; done
say "rescore candidates"
declare -A BAKES=(
  [A_PH_s4004]=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin
  [B_e060]=/mnt/v/output/zensim/bakes/htraj-2026-08-28/ckpt_epoch060_packed_stamped.bin
  [incumbent]=/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin
  [f054]=/mnt/v/output/zensim/bakes/htraj-fine-2026-08-28/ckpt_epoch054_packed.bin
  [s4005P]=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9P_s4005_packed.bin
  [s4010P]=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9P_s4010_packed.bin
)
for tag in "${!BAKES[@]}"; do
  for ax in hfnlproxy imazen26 nonphoto; do
    nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "${BAKES[$tag]}" \
      --regime 944 --cross-regime --corpora $ax --features-root "$VS/root" \
      --per-pair-output "$VS/pp_${ax}_${tag}.tsv" > /dev/null 2>&1 || say "RESCORE FAIL $tag $ax"
  done
  say "rescored $tag"
done
say "VALSEL DONE"
