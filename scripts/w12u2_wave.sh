#!/usr/bin/env bash
# W12-U "lodestar" unified-ship-candidate wave (registered in
# benchmarks/balance_campaign_2026-08-28.md "W12-U LODESTAR" BEFORE launch).
# W11J recipe + doubled jpeg/webp top-zone HF mass (tbig_hf_jw); 3 seeds.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARGV_FILE=${SDRPURE_ARGV:-$HOME/tmp/sdrpure_argv.txt}
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${W12U2_HB:-$HOME/tmp/w12u2/heartbeat}
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$ARGV_FILE"
[[ "${ARGV[0]}" == */zensim_mlp_train ]] && ARGV=("${ARGV[@]:1}")
FAILS=0
for seed in 4031 4032 4033; do
    stem="LSTAR2_s${seed}"
    mkdir -p "$OUT/${stem}_ckpts"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
            --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
            --group "tbig_hf_jw:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_jw.parquet:0.5:0.0:both" \
            --seed "$seed" --out "$OUT/$stem.bin" \
            --dump-checkpoints-every 10 --dump-checkpoints-dir "$OUT/${stem}_ckpts" >> "$OUT/w12u2_train.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    if [ ! -f "$OUT/${stem}_packed.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_packed.bin" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/w12u2_pack.log" 2>&1 || { say "PACK FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    say "harvest ${stem}_packed"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_packed.bin" --regime 944 \
        >> "$OUT/w12u2_harvest.log" 2>&1 || { say "HARVEST FAILED ${stem}_packed"; FAILS=$((FAILS+1)); }
done
for seed in 4031 4032 4033; do
    stem="LSTAR2C_s${seed}"
    mkdir -p "$OUT/${stem}_ckpts"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem (jw05+cd)"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
            --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
            --group "tbig_hf_jw:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_jw.parquet:0.5:0.0:both" \
            --coarse-decay 1e-5 \
            --seed "$seed" --out "$OUT/$stem.bin" \
            --dump-checkpoints-every 10 --dump-checkpoints-dir "$OUT/${stem}_ckpts" >> "$OUT/w12u2_train.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    if [ ! -f "$OUT/${stem}_packed.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_packed.bin" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/w12u2_pack.log" 2>&1 || { say "PACK FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    say "harvest ${stem}_packed"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_packed.bin" --regime 944 \
        >> "$OUT/w12u2_harvest.log" 2>&1 || { say "HARVEST FAILED ${stem}_packed"; FAILS=$((FAILS+1)); }
done
say "ALL DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
