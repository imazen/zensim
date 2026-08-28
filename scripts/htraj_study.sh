#!/usr/bin/env bash
# H-TRAJ epoch-M3a trajectory study (registered: balance_campaign_2026-08-28.md).
# Retrace SPH1 s4004 with checkpoint dumps; per checkpoint: parity pack +
# M3a (m3a_sweep owner) + cid22/hfnl rank. Emits trajectory TSV.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=/mnt/v/output/zensim/bakes/htraj-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${HTRAJ_HB:-$HOME/tmp/htraj/heartbeat}
mkdir -p "$D" "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$HOME/tmp/sdrpure_argv.txt"
[[ "${ARGV[0]}" == */zensim_mlp_train ]] && ARGV=("${ARGV[@]:1}")
if [ ! -f "$D/W10L9PH_s4004_retrace.bin" ]; then
    say "retrace train (dumps every 10)"
    nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
        --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
        --seed 4004 --out "$D/W10L9PH_s4004_retrace.bin" \
        --dump-checkpoints-every 10 --dump-checkpoints-dir "$D" \
        >> "$D/train.log" 2>&1 || { say "TRAIN FAILED"; exit 6; }
fi
orig=$(sha256sum /mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004.bin | cut -d' ' -f1)
retr=$(sha256sum "$D/W10L9PH_s4004_retrace.bin" | cut -d' ' -f1)
say "REPRO CHECK: original=$orig retrace=$retr match=$([ "$orig" = "$retr" ] && echo YES || echo NO)"
echo -e "epoch\tm3a\tm3\tcid22\thfnl" > "$D/trajectory.tsv"
DM="$REPO/target/release/examples/diffmap_block_coherence"
for ck in "$D"/ckpt_epoch*.bin "$D/W10L9PH_s4004_retrace.bin"; do
    base=$(basename "$ck" .bin)
    ep=$(echo "$base" | grep -o '[0-9]*$' || echo 999)
    [ "$base" = "W10L9PH_s4004_retrace" ] && ep=final
    pk="$D/${base}_packed.bin"
    if [ ! -f "$pk" ]; then
        nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
            --in "$ck" --out "$pk" --neg-tail \
            --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
            --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
            >> "$D/pack.log" 2>&1 || { say "PACK FAILED $base"; continue; }
    fi
    say "m3a $base"
    "$REPO/scripts/m3a_sweep.sh" --bake "$pk" --bin "$DM" --grid full \
        --label "$base" --logdir "$D" --tsv "$D/${base}.m3a_cells.tsv" > "$D/${base}.m3a.kv" 2>>"$D/m3a.log" || say "M3A FAILED $base"
    m3a=$(awk -F= '$1=="M3A_MEAN"{print $2; exit}' "$D/${base}.m3a.kv" 2>/dev/null || echo NA)
    m3=$(awk -F= '$1=="M3_MEAN"{print $2; exit}' "$D/${base}.m3a.kv" 2>/dev/null || echo NA)
    for ax in cid22 hfnlproxy; do
        nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "$pk" \
            --regime 944 --cross-regime --corpora $ax \
            --features-root "$ROOT" --json "$D/${base}.${ax}.json" \
            > /dev/null 2>&1 || say "VERDICT FAILED $base $ax"
    done
    c22=$(python3 -c "import json;print(round(json.load(open('$D/${base}.cid22.json'))['corpora']['cid22']['srocc_signed'],4))" 2>/dev/null || echo NA)
    hfn=$(python3 -c "import json;print(round(json.load(open('$D/${base}.hfnlproxy.json'))['corpora']['hfnlproxy']['srocc_signed'],4))" 2>/dev/null || echo NA)
    echo -e "${ep}\t${m3a}\t${m3}\t${c22}\t${hfn}" >> "$D/trajectory.tsv"
    say "cell $base m3a=$m3a cid22=$c22 hfnl=$hfn"
done
say "H-TRAJ DONE"
