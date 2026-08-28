#!/usr/bin/env bash
# W11 selection under VALIDATE-slice eligibility (stage 4 of the chain).
# Per frozen W11 rule + the validate migration: per seed, m3a the 40-80-window
# checkpoints + final, then rescore m3a-strong cells on the validate root.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
VROOT=/mnt/v/zen/zensim-training/valsel-2026-08-28/root
HB=$HOME/tmp/w11sel/heartbeat
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
DM="$REPO/target/release/examples/diffmap_block_coherence"
echo -e "seed\tepoch\tm3a\tcid22\tvhfnl" > "$OUT/w11_selection.tsv"
for seed in 4012 4013 4014; do
  CK="$OUT/W11J_s${seed}_ckpts"
  for ck in "$CK"/ckpt_epoch0{4,5,6,7,8}0.bin "$OUT/W11J_s${seed}.bin"; do
    [ -f "$ck" ] || continue
    base=$(basename "$ck" .bin)
    ep=$(echo "$base" | grep -o '[0-9]*$' || echo final); [ "$base" = "W11J_s${seed}" ] && ep=final
    pk="$CK/${base}_s${seed}_packed.bin"
    [ -f "$pk" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
        --in "$ck" --out "$pk" --neg-tail \
        --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
        --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
        >> "$HOME/tmp/w11sel/pack.log" 2>&1 || { say "PACK FAIL $seed $ep"; continue; }
    "$REPO/scripts/m3a_sweep.sh" --bake "$pk" --bin "$DM" --grid full \
        --label "w11_${seed}_${ep}" --logdir "$HOME/tmp/w11sel" \
        --tsv "$HOME/tmp/w11sel/${seed}_${ep}.m3a_cells.tsv" > "$HOME/tmp/w11sel/${seed}_${ep}.kv" 2>>"$HOME/tmp/w11sel/m3a.log" || say "M3A FAIL $seed $ep"
    m3a=$(awk -F= '$1=="M3A_MEAN"{print $2; exit}' "$HOME/tmp/w11sel/${seed}_${ep}.kv" 2>/dev/null || echo NA)
    for spec in "cid22:$ROOT" "hfnlproxy:$VROOT"; do
      ax="${spec%%:*}"; fr="${spec#*:}"
      nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "$pk" \
        --regime 944 --cross-regime --corpora $ax --features-root "$fr" \
        --json "$HOME/tmp/w11sel/${seed}_${ep}.${ax}.json" > /dev/null 2>&1 || true
    done
    c22=$(python3 -c "import json;print(round(json.load(open('$HOME/tmp/w11sel/${seed}_${ep}.cid22.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    vhf=$(python3 -c "import json;print(round(json.load(open('$HOME/tmp/w11sel/${seed}_${ep}.hfnlproxy.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    echo -e "${seed}\t${ep}\t${m3a}\t${c22}\t${vhf}" >> "$OUT/w11_selection.tsv"
    say "cell s${seed} e${ep} m3a=$m3a cid22=$c22 vhfnl=$vhf"
  done
done
say "W11SEL DONE"
