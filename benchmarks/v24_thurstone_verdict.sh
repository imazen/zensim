#!/usr/bin/env bash
# Run bake_verdict on every Thurstone seed bake and summarise.
set -e
OUT_DIR=/mnt/v/output/zensim/ex1_thurstone_2026-05-18
VERDICT_DIR="$OUT_DIR/verdicts"
SUMMARY="$OUT_DIR/v24_thurstone_5seed_panel.md"
BAKE_VERDICT=/home/lilith/work/zen/zensim/target/release/bake_verdict

for SEED in 1 2 3 4 5; do
  BAKE="$OUT_DIR/bakes/v24_thurstone_konjnd_002_LARGE_iwssim_s${SEED}_h128.bin"
  if [[ ! -f "$BAKE" ]]; then
    echo "missing $BAKE — skipping" >&2
    continue
  fi
  V="$VERDICT_DIR/seed${SEED}.md"
  "$BAKE_VERDICT" --bake "$BAKE" --output "$V" --corpora cid22,kadid,tid,konjnd,aic3 2>&1 | tail -3
done

# Aggregate.
{
  echo "# V_24-thurstone + konjnd@0.02 + LARGE+iwssim — 5-seed CI"
  echo
  echo "Baseline (V_22-mix-LARGE+iwssim packed): bake_verdict on"
  echo "\`/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin\`"
  echo
  echo "| Corpus | V_22 SROCC | V_24-thurstone seed1..5 mean ± stdev | Δ vs V_22 |"
  echo "|---|---:|---:|---:|"
  for CORPUS in "CID22 (n=4292)" "KADIK10k (n=10125)" "TID2013 (n=3000)" "KonJND-1k (full) (n=1008)" "AIC-3 CTC (n=600)"; do
    V22=$(awk -v c="## $CORPUS" '
      $0==c {f=1}
      f && /^\| V_X bake \|/{ split($0,a,"|"); gsub(/ /,"",a[3]); print a[3]; exit }
    ' /tmp/v22_baseline_verdict.md)
    SEED_VALS=""
    for SEED in 1 2 3 4 5; do
      V="$VERDICT_DIR/seed${SEED}.md"
      [[ -f "$V" ]] || continue
      S=$(awk -v c="## $CORPUS" '
        $0==c {f=1}
        f && /^\| V_X bake \|/{ split($0,a,"|"); gsub(/ /,"",a[3]); print a[3]; exit }
      ' "$V")
      SEED_VALS="$SEED_VALS $S"
    done
    MEAN_SD=$(echo "$SEED_VALS" | awk '{n=NF; m=0; for(i=1;i<=n;i++) m+=$i; m/=n; s=0; for(i=1;i<=n;i++) s+=($i-m)^2; s=(n>1)?sqrt(s/(n-1)):0; printf "%.4f ± %.4f", m, s}')
    MEAN_ONLY=$(echo "$SEED_VALS" | awk '{n=NF; m=0; for(i=1;i<=n;i++) m+=$i; m/=n; printf "%.4f", m}')
    if [[ -n "$V22" && -n "$MEAN_ONLY" ]]; then
      DELTA=$(awk -v a="$MEAN_ONLY" -v b="$V22" 'BEGIN{printf "%+0.4f", a-b}')
    else
      DELTA="?"
    fi
    echo "| $CORPUS | $V22 | $MEAN_SD | $DELTA |"
  done
  echo
  echo "Raw per-seed SROCC by corpus:"
  echo
  for CORPUS in "CID22 (n=4292)" "KADIK10k (n=10125)" "TID2013 (n=3000)" "KonJND-1k (full) (n=1008)" "AIC-3 CTC (n=600)"; do
    SEED_VALS=""
    for SEED in 1 2 3 4 5; do
      V="$VERDICT_DIR/seed${SEED}.md"
      [[ -f "$V" ]] || continue
      S=$(awk -v c="## $CORPUS" '
        $0==c {f=1}
        f && /^\| V_X bake \|/{ split($0,a,"|"); gsub(/ /,"",a[3]); print a[3]; exit }
      ' "$V")
      SEED_VALS="$SEED_VALS s$SEED:$S"
    done
    echo "- $CORPUS:$SEED_VALS"
  done
} > "$SUMMARY"
echo "Wrote summary to $SUMMARY"
cat "$SUMMARY"
