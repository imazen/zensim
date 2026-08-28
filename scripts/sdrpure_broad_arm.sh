#!/usr/bin/env bash
# SPH1-BROAD arm (registered in
# benchmarks/sdr_pure_retrain_wave_2026-08-28.md "SPH1-BROAD ARM").
# Purity argv + the family-clean tbig HF leg as an extra group; 3 seeds.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARGV_FILE=${SDRBROAD_ARGV:-$HOME/tmp/sdrbroad_argv.txt}
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${SPH1BR_HB:-$HOME/tmp/sph1broad/heartbeat}
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$ARGV_FILE"
# the argv file's first line is the trainer path itself — strip it
[[ "${ARGV[0]}" == */zensim_mlp_train ]] && ARGV=("${ARGV[@]:1}")
FAILS=0
for seed in 4003 4004 4005; do
    stem="W10L9PBR_s${seed}"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
            --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
            --seed "$seed" --out "$OUT/$stem.bin" >> "$OUT/sph1broad_train.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    if [ ! -f "$OUT/${stem}_packed.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_packed.bin" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/sph1broad_pack.log" 2>&1 || { say "PACK FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    say "harvest ${stem}_packed"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_packed.bin" --regime 944 \
        >> "$OUT/sph1broad_harvest.log" 2>&1 || { say "HARVEST FAILED ${stem}_packed"; FAILS=$((FAILS+1)); }
done
say "ALL DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
