#!/usr/bin/env bash
# W12-U2 battery stage (generated from ~/tmp/w12u2bat_cells.json)
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ROOT=/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01
VROOT=/mnt/v/zen/zensim-training/valsel-2026-08-28/root
WD=$HOME/tmp/w12u2bat; mkdir -p "$WD"; HB=$WD/heartbeat
trap 'echo "$(date -u +%FT%TZ) EXIT rc=$?" >> "$HB"; touch "$HB.done"' EXIT
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$HB"; }
FAILS=0
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4031_e050.fulleval.json" ]; then
  say "harvest lstar2_4031_e050"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch050_s4031_packed.bin" --stem "lstar2_4031_e050" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4031_e050"; FAILS=$((FAILS+1)); }
fi
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4031_e060.fulleval.json" ]; then
  say "harvest lstar2_4031_e060"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch060_s4031_packed.bin" --stem "lstar2_4031_e060" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4031_e060"; FAILS=$((FAILS+1)); }
fi
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4032_e080.fulleval.json" ]; then
  say "harvest lstar2_4032_e080"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch080_s4032_packed.bin" --stem "lstar2_4032_e080" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4032_e080"; FAILS=$((FAILS+1)); }
fi
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4032_e070.fulleval.json" ]; then
  say "harvest lstar2_4032_e070"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch070_s4032_packed.bin" --stem "lstar2_4032_e070" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4032_e070"; FAILS=$((FAILS+1)); }
fi
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4033_e080.fulleval.json" ]; then
  say "harvest lstar2_4033_e080"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch080_s4033_packed.bin" --stem "lstar2_4033_e080" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4033_e080"; FAILS=$((FAILS+1)); }
fi
if [ ! -f "/mnt/v/output/zensim/reports/fulleval/lstar2_4033_e070.fulleval.json" ]; then
  say "harvest lstar2_4033_e070"
  "$REPO/scripts/harvest_bakes.sh" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch070_s4033_packed.bin" --stem "lstar2_4033_e070" --regime 944 >> "$WD/harvest.log" 2>&1 || { say "HARVEST FAIL lstar2_4033_e070"; FAILS=$((FAILS+1)); }
fi
[ -f "$WD/pp_cid22_lstar2_4031_e050.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch050_s4031_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4031_e050.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e050 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4031_e050.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch050_s4031_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4031_e050.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e050 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4031_e050.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch050_s4031_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4031_e050.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e050 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4031_e050"
[ -f "$WD/pp_cid22_lstar2_4031_e060.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch060_s4031_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4031_e060.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e060 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4031_e060.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch060_s4031_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4031_e060.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e060 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4031_e060.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4031_ckpts/ckpt_epoch060_s4031_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4031_e060.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4031_e060 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4031_e060"
[ -f "$WD/pp_cid22_lstar2_4032_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch080_s4032_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4032_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e080 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4032_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch080_s4032_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4032_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e080 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4032_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch080_s4032_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4032_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e080 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4032_e080"
[ -f "$WD/pp_cid22_lstar2_4032_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch070_s4032_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4032_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e070 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4032_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch070_s4032_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4032_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e070 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4032_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4032_ckpts/ckpt_epoch070_s4032_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4032_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4032_e070 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4032_e070"
[ -f "$WD/pp_cid22_lstar2_4033_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch080_s4033_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4033_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e080 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4033_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch080_s4033_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4033_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e080 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4033_e080.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch080_s4033_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4033_e080.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e080 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4033_e080"
[ -f "$WD/pp_cid22_lstar2_4033_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch070_s4033_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_lstar2_4033_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e070 cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_lstar2_4033_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch070_s4033_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_lstar2_4033_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e070 hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_lstar2_4033_e070.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR2_s4033_ckpts/ckpt_epoch070_s4033_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_lstar2_4033_e070.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL lstar2_4033_e070 tid"; FAILS=$((FAILS+1)); }
say "rescored lstar2_4033_e070"
[ -f "$WD/pp_cid22_A.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_A.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL A cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_A.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_A.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL A hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_A.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_A.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL A tid"; FAILS=$((FAILS+1)); }
say "rescored A"
[ -f "$WD/pp_cid22_incumbent.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin" --regime 944 --cross-regime --corpora cid22 --features-root "$ROOT" --per-pair-output "$WD/pp_cid22_incumbent.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL incumbent cid22"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_hfnlproxy_incumbent.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin" --regime 944 --cross-regime --corpora hfnlproxy --features-root "$VROOT" --per-pair-output "$WD/pp_hfnlproxy_incumbent.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL incumbent hfnlproxy"; FAILS=$((FAILS+1)); }
[ -f "$WD/pp_tid_incumbent.tsv" ] || nice -n19 ionice -c3 "$REPO/target/release/bake_verdict" --bake "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin" --regime 944 --cross-regime --corpora tid --features-root "$ROOT" --per-pair-output "$WD/pp_tid_incumbent.tsv" > /dev/null 2>&1 || { say "RESCORE FAIL incumbent tid"; FAILS=$((FAILS+1)); }
say "rescored incumbent"
say "BATTERY-STAGE DONE fails=$FAILS"
[ "$FAILS" = 0 ] || exit 6
