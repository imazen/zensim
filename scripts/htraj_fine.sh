#!/usr/bin/env bash
# H-TRAJ FINE pass (user call 2026-08-28): val+dumps every 2 epochs; evaluate
# the 44-80 window for a checkpoint with tied<=0.335 AND m3a>=0.83.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
D=/mnt/v/output/zensim/bakes/htraj-fine-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
HB=${HTRAJF_HB:-$HOME/tmp/htrajf/heartbeat}
mkdir -p "$D" "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
mapfile -t ARGV < "$HOME/tmp/sdrpure_argv.txt"
[[ "${ARGV[0]}" == */zensim_mlp_train ]] && ARGV=("${ARGV[@]:1}")
if [ ! -f "$D/retrace_fine.bin" ]; then
    say "fine retrace (val+dumps every 2)"
    nice -n19 ionice -c3 "$REPO/target/release/zensim_mlp_train" "${ARGV[@]}" \
        --group "tbig_hf:/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/tbig_hf_pure.parquet:1.0:0.0:both" \
        --seed 4004 --out "$D/retrace_fine.bin" --log-every 2 \
        --dump-checkpoints-every 2 --dump-checkpoints-dir "$D" \
        >> "$D/train.log" 2>&1 || { say "TRAIN FAILED"; exit 6; }
fi
DM="$REPO/target/release/examples/diffmap_block_coherence"
echo -e "epoch\tm3a\tcid22\thfnl" > "$D/trajectory_fine.tsv"
for ep in 044 046 048 050 052 054 056 058 060 062 064 066 068 070 072 074 076 078 080; do
    ck="$D/ckpt_epoch${ep}.bin"
    [ -f "$ck" ] || { say "missing $ck"; continue; }
    pk="$D/ckpt_epoch${ep}_packed.bin"
    [ -f "$pk" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
        --in "$ck" --out "$pk" --neg-tail \
        --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
        --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
        >> "$D/pack.log" 2>&1 || { say "PACK FAILED $ep"; continue; }
    "$REPO/scripts/m3a_sweep.sh" --bake "$pk" --bin "$DM" --grid full \
        --label "e${ep}" --logdir "$D" --tsv "$D/e${ep}.m3a_cells.tsv" > "$D/e${ep}.m3a.kv" 2>>"$D/m3a.log" || say "M3A FAILED $ep"
    for ax in cid22 hfnlproxy; do
        nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "$pk" \
            --regime 944 --cross-regime --corpora $ax --features-root "$ROOT" \
            --json "$D/e${ep}.${ax}.json" > /dev/null 2>&1 || true
    done
    m3a=$(awk -F= '$1=="M3A_MEAN"{print $2; exit}' "$D/e${ep}.m3a.kv" 2>/dev/null || echo NA)
    c22=$(python3 -c "import json;print(round(json.load(open('$D/e${ep}.cid22.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    hfn=$(python3 -c "import json;print(round(json.load(open('$D/e${ep}.hfnlproxy.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    echo -e "${ep}\t${m3a}\t${c22}\t${hfn}" >> "$D/trajectory_fine.tsv"
    say "e${ep} m3a=$m3a cid22=$c22 hfnl=$hfn"
done
say "FINE DONE"
