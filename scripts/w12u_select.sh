#!/usr/bin/env bash
# W12-U lodestar selection (frozen protocol, balance_campaign W12-U section).
# Per frozen W12-U rule + the validate migration: per seed, m3a the 40-80-window
# checkpoints + final, then rescore m3a-strong cells on the validate root.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
VROOT=/mnt/v/zen/zensim-training/valsel-2026-08-28/root
HB=$HOME/tmp/w12usel/heartbeat
mkdir -p "$(dirname "$HB")"
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
DM="$REPO/target/release/examples/diffmap_block_coherence"
echo -e "seed\tepoch\tm3a\tcid22\tvhfnl" > "$OUT/w12u_selection.tsv"
for seed in 4021 4022 4023; do
  CK="$OUT/LSTAR_s${seed}_ckpts"
  for ck in "$CK"/ckpt_epoch0{4,5,6,7,8}0.bin "$OUT/LSTAR_s${seed}.bin"; do
    [ -f "$ck" ] || continue
    base=$(basename "$ck" .bin)
    ep=$(echo "$base" | grep -o '[0-9]*$' || echo final); [ "$base" = "LSTAR_s${seed}" ] && ep=final
    pk="$CK/${base}_s${seed}_packed.bin"
    [ -f "$pk" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_dial_refit" pack \
        --in "$ck" --out "$pk" --neg-tail \
        --anchor "$ROOT/anchor944_dial.parquet" --target-col target_score \
        --verify "$ROOT/ext_cid22val.parquet" --verify-col human_score --verify-scale 100 \
        >> "$HOME/tmp/w12usel/pack.log" 2>&1 || { say "PACK FAIL $seed $ep"; continue; }
    "$REPO/scripts/m3a_sweep.sh" --bake "$pk" --bin "$DM" --grid full \
        --label "w12u_${seed}_${ep}" --logdir "$HOME/tmp/w12usel" \
        --tsv "$HOME/tmp/w12usel/${seed}_${ep}.m3a_cells.tsv" > "$HOME/tmp/w12usel/${seed}_${ep}.kv" 2>>"$HOME/tmp/w12usel/m3a.log" || say "M3A FAIL $seed $ep"
    m3a=$(awk -F= '$1=="M3A_MEAN"{print $2; exit}' "$HOME/tmp/w12usel/${seed}_${ep}.kv" 2>/dev/null || echo NA)
    for spec in "cid22:$ROOT" "hfnlproxy:$VROOT"; do
      ax="${spec%%:*}"; fr="${spec#*:}"
      nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "$pk" \
        --regime 944 --cross-regime --corpora $ax --features-root "$fr" \
        --json "$HOME/tmp/w12usel/${seed}_${ep}.${ax}.json" > /dev/null 2>&1 || true
    done
    c22=$(python3 -c "import json;print(round(json.load(open('$HOME/tmp/w12usel/${seed}_${ep}.cid22.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    vhf=$(python3 -c "import json;print(round(json.load(open('$HOME/tmp/w12usel/${seed}_${ep}.hfnlproxy.json'))['corpora'][0]['srocc'],4))" 2>/dev/null || echo NA)
    echo -e "${seed}\t${ep}\t${m3a}\t${c22}\t${vhf}" >> "$OUT/w12u_selection.tsv"
    say "cell s${seed} e${ep} m3a=$m3a cid22=$c22 vhfnl=$vhf"
  done
done
say "W11SEL DONE"
