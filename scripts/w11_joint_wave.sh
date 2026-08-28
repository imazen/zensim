#!/usr/bin/env bash
# W11 joint-selection wave (registered in
# benchmarks/sdr_pure_retrain_wave_2026-08-28.md "NEXT-GEN PROGRAM / W11").
# Purity argv + the family-clean tbig HF leg as an extra group; 3 seeds.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARGV_FILE=${SDRPURE_ARGV:-$HOME/tmp/sdrpure_argv.txt}
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${W11_HB:-$HOME/tmp/w11/heartbeat}
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$ARGV_FILE"
# the argv file's first line is the trainer path itself — strip it
[[ "${ARGV[0]}" == */zensim_mlp_train ]] && ARGV=("${ARGV[@]:1}")
FAILS=0
for seed in 4012 4013 4014; do
    stem="W11J_s${seed}"
    mkdir -p "$OUT/${stem}_ckpts"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
            --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
            --seed "$seed" --out "$OUT/$stem.bin" \
            --dump-checkpoints-every 10 --dump-checkpoints-dir "$OUT/${stem}_ckpts" >> "$OUT/w11_train.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    if [ ! -f "$OUT/${stem}_packed.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_packed.bin" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/w11_pack.log" 2>&1 || { say "PACK FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    say "harvest ${stem}_packed"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_packed.bin" --regime 944 \
        >> "$OUT/w11_harvest.log" 2>&1 || { say "HARVEST FAILED ${stem}_packed"; FAILS=$((FAILS+1)); }
done
say "ALL DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
