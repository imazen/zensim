#!/usr/bin/env bash
# SDR purity wave seed extension (registered in
# benchmarks/sdr_pure_retrain_wave_2026-08-28.md "SEED EXTENSION").
# Trains seeds 4006..4011 on the frozen argv, packs (campaign parity
# invocation), harvests. Serialized; heartbeats to $HB; writes $HB.done on
# every exit path.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARGV_FILE=${SDRPURE_ARGV:-$HOME/tmp/sdrpure_argv.txt}
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${SDRPURE_HB:-$HOME/tmp/sdrpure_ext/heartbeat}
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$ARGV_FILE"
for seed in 4006 4007 4008 4009 4010 4011; do
    stem="W10L9P_s${seed}"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
            --seed "$seed" --out "$OUT/$stem.bin" >> "$OUT/train_ext.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; continue; }
    fi
    if [ ! -f "$OUT/${stem}_packed.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_packed.bin" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/pack_ext.log" 2>&1 || { say "PACK FAILED $stem"; continue; }
    fi
    say "harvest ${stem}_packed"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_packed.bin" --regime 944 \
        >> "$OUT/harvest_ext.log" 2>&1 || say "HARVEST FAILED ${stem}_packed"
done
say "ALL DONE"
