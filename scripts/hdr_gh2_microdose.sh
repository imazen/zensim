#!/usr/bin/env bash
# GH2 micro-dose arms (registered in benchmarks/hdr944_retrain_wave_2026-08-28.md).
# GH1 recipe at hf2 weight 0.10 (GH2a) / 0.15 (GH2b); seeds {4003,4005} each.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=/mnt/v/output/zensim/bakes/hdr944-retrain-2026-08-28
TRD=/mnt/v/zen/zensim-training
ROOT=$TRD/ext944-canonical-2026-08-01
HB=${GH2_HB:-$HOME/tmp/gh2/heartbeat}
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
FAILS=0
for spec in a:0.10:4003 a:0.10:4005 b:0.15:4003 b:0.15:4005; do
    dose="${spec%%:*}"; rest="${spec#*:}"; w="${rest%%:*}"; seed="${rest#*:}"
    stem="HDR944_GH2${dose}_s${seed}"
    if [ ! -f "$OUT/$stem.bin" ]; then
        say "train $stem"
        nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" \
            --group "hdrmc:$TRD/hdrgrid-mc944-t1-2026-08-27/hdrgrid_mc944_t2_train.parquet:1.0:0.0:both" \
            --group "hdrmc_hf:$TRD/hdr944-retrain-2026-08-28/t1_train_hf.parquet:1.0:0.0:both" \
            --group "hdrmc_hf2:$TRD/hdr944-retrain-2026-08-28/t2_train_hf.parquet:${w}:0.0:both" \
            --group "hdrmc_val:$TRD/hdrgrid-mc944-t1-2026-08-27/hdrgrid_mc944_t2_val.parquet:0.0:1.0:both" \
            --n-hidden-layers 0 --target-column human_score --target-scale 100 \
            --epochs 120 --pairs-per-epoch 50000 --seed "$seed" --max-features 944 \
            --coarse-decay 1e-5 --out "$OUT/$stem.bin" >> "$OUT/gh2_train.log" 2>&1 \
            || { say "TRAIN FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    if [ ! -f "$OUT/${stem}_hfpack.bin" ]; then
        say "pack $stem"
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$OUT/$stem.bin" --out "$OUT/${stem}_hfpack.bin" --neg-tail \
            --anchor "$TRD/hdr944-retrain-2026-08-28/anchor_hf_t1.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$OUT/gh2_pack.log" 2>&1 || { say "PACK FAILED $stem"; FAILS=$((FAILS+1)); continue; }
    fi
    say "harvest ${stem}_hfpack"
    "$REPO/scripts/harvest_bakes.sh" --bake "$OUT/${stem}_hfpack.bin" --regime 944 \
        >> "$OUT/gh2_harvest.log" 2>&1 || { say "HARVEST FAILED ${stem}_hfpack"; FAILS=$((FAILS+1)); }
done
say "GH2 DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
