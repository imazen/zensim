#!/bin/bash
# fastclass2 GATE G4 — close the era between the 944 leaders and the fast class.
#
# The two families do NOT share a features root, so the plan section 1 table is
# not apples-to-apples until this runs. The move is NOT "score the leaders on
# the fast class's root": the leaders train on ext944-canonical-2026-08-01,
# whose registered set is `basic+v2+append+append2@w944/ext944` -- pools
# ZEROED -- while the fast class's root is the pools-LIVE
# `basic+peaks+masked+iw+v2+append+append2@w944/era2r4`. Feeding a
# pools-zeroed-trained model live f156..371 is the registered wrong-regime
# silent-mis-score class, not an era A/B.
#
# So each family is read on ITS OWN COMPUTE at the SAME ERA:
#   leaders  -> ext944-era2r4-2026-09-01/foldapp2_views
#               (`basic+v2+append+append2@w944/era2r4` -- same compute as their
#                native ext944 set, era2r4 era)
#   fastclass-> ext944-era2r4-2026-09-01 (its own native pools root)
# Era is then the ONLY thing that changed for the leaders, and regime purity
# holds for both.
set -euo pipefail
BIN="${FC2_BIN:-/mnt/v/zen/cargo-targets/fastclass2/release}"
FA2=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views
DIAL="${FC2_DIAL:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/dial_grid_944col_foldapp2_2026-09-01.parquet}"
O="${FC2_OUT:-/mnt/v/output/zensim/fastclass2-2026-09-05}/g4era"
mkdir -p "$O"
SDR=/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28
REP=/mnt/v/output/zensim/replication-2026-09-05/bakes
for B in \
  "$SDR/W10L9PH_s4003_packed.bin" "$SDR/W10L9PH_s4004_packed.bin" "$SDR/W10L9PH_s4005_packed.bin" \
  "$SDR/W10L9PH_s4006_packed.bin" "$SDR/W10L9PH_s4007_packed.bin" "$SDR/W10L9PH_s4008_packed.bin" \
  "$SDR/W11J_s4012_packed.bin" "$SDR/W11J_s4013_packed.bin" "$SDR/W11J_s4014_packed.bin" \
  "$REP/W11J__I__i5011_p4013_packed.bin" "$REP/W11J__I__i5012_p4013_packed.bin" \
  "$REP/W11J__S__i4013_p5001_packed.bin" "$REP/W11J__S__i4013_p5002_packed.bin" ; do
  N="$(basename "$B" .bin)@era2r4"
  [ -f "$O/$N.fulleval.json" ] && { echo "SKIP $N"; continue; }
  echo "== $N"
  nice -n19 ionice -c3 "$BIN/bake_verdict" --bake "$B" --regime 944 \
    --features-root "$FA2" --dial-grid "$DIAL" --name "$N" \
    --full-json "$O/$N.fulleval.json" --output "$O/$N.verdict.md" 2>&1 | tail -4
done
echo "G4 COMPLETE"
