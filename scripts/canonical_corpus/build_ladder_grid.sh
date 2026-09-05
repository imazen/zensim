#!/usr/bin/env bash
# Build the LADDER dial-grid instrument: five per-codec ladders whose bottom is the
# codec's own lowest DISTINCT settings.
#
# WHY THIS EXISTS (measured 2026-09-05, docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md §1):
# the previous grid sampled q 0/5/10 at the floor, and zenjpeg emits ONE bitstream for
# every q in 0..10 — so its three lowest "settings" were one setting sampled three
# times, and the mentor's floor bar for jpeg was a vacuous 0.0000. Every codec has its
# own plateau (avif/svt-rs ties pairwise because quality 0..100 maps onto QP 0..63), so
# the grid is DENSE at the floor and duplicates are flagged `saturated` downstream by
# encode_sha rather than guessed at with a per-codec step table.
#
# Owner discipline: this script only ORCHESTRATES `zenmetrics sweep` (the owner). It
# computes no statistics and encodes nothing itself.
#
# Usage:  build_ladder_grid.sh <OUT_DIR> <SOURCES_DIR> [ZM_BIN]
set -euo pipefail

OUT="${1:?usage: build_ladder_grid.sh <OUT_DIR> <SOURCES_DIR> [ZM_BIN]}"
SRC="${2:?missing SOURCES_DIR}"
ZM="${3:-/mnt/v/output/zensim/ladder-2026-09-05/bin/zenmetrics_svtnew}"

[ -x "$ZM" ] || { echo "FATAL: zenmetrics binary not executable: $ZM" >&2; exit 2; }
[ -d "$SRC" ] || { echo "FATAL: sources dir missing: $SRC" >&2; exit 2; }

# The 66-step floor-dense q grid. 0..30 step 1 is the NEW part: it guarantees three
# DISTINCT lowest settings exist for every codec (jpeg's first distinct step is q=11).
# LADDER_QGRID overrides the q axis. The ANCHOR run uses a floor-preserving trim
# (0..30 step 1 kept verbatim — the whole point of a new anchor is that the shipped
# one's max(ssim2,0) clamp collapsed the floor — with the top thinned), because an
# anchor needs score-range coverage, not the instrument's grading density.
QG="${LADDER_QGRID:-}"
[ -n "$QG" ] || QG="$(python3 -c '
q =[float(i) for i in range(0,31)]
q+=[float(i) for i in range(35,71,5)]
q+=[float(i) for i in range(72,91,2)]
q+=[float(i) for i in range(91,97)]
q+=[96.5,97,97.5,98,98.5,99,99.25,99.5,99.75,99.9,100]
print(",".join("%g"%x for x in sorted(set(q))))')"

# JXL rides its distance knob. 25.0 IS the floor (26/30/40/50 are byte-identical to
# 25.0 — measured); 24 and 22 are added so the three lowest DISTINCT settings exist.
JXL_KNOB='{"distance":[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.2,1.4,1.6,1.8,2.0,2.2,2.4,2.6,2.8,3.0,3.5,4.0,5.0,6.5,8.0,10.0,13.0,15.0,17.0,19.0,21.0,22.0,23.0,24.0,25.0]}'

# Every metric variant this build can produce. Incremental cost is microseconds;
# recovery cost is days (ML-pipeline discipline §4).
# `--metric` is a REPEATED flag, not comma-separated (a comma-joined value is
# rejected at parse time with the possible-values list).
IFS=, read -ra METRICS <<< "${LADDER_METRICS:-ssim2,butteraugli,dssim}"
METRIC_ARGS=(); for m in "${METRICS[@]}"; do METRIC_ARGS+=(--metric "$m"); done

mkdir -p "$OUT"/{tsv,dist,encoded,pairs,logs}

run_leg () {                       # run_leg <label> <codec> <qgrid> [knob-json]
  local label="$1" codec="$2" qg="$3" knob="${4:-}"
  local t0 t1
  echo "== leg $label ($codec) start $(date -u +%H:%M:%SZ)" | tee -a "$OUT/logs/progress.log"
  t0=$(date +%s)
  # LADDER_JOBS bounds sweep parallelism. Load-bearing on large sources: the
  # anchor leg (6.27 MP mean, up to 12 MP) was SIGKILLed at 38 s with an empty log
  # at the default job count — each in-flight cell holds the reference, the
  # distorted image and butteraugli's own buffers, so the burst scales with
  # jobs x megapixels, not with the corpus size.
  local args=(sweep --codec "$codec" --sources "$SRC" --q-grid "$qg"
              --jobs "${LADDER_JOBS:-0}"
              "${METRIC_ARGS[@]}"
              --output       "$OUT/tsv/$label.tsv"
              --distorted-out-dir "$OUT/dist/$label"
              --encoded-out-dir   "$OUT/encoded/$label"
              --pairs-tsv    "$OUT/pairs/$label.tsv")
  [ -n "$knob" ] && args+=(--knob-grid "$knob")
  nice -n19 ionice -c3 "$ZM" "${args[@]}" > "$OUT/logs/$label.log" 2>&1 || {
      echo "!! leg $label FAILED rc=$? (see $OUT/logs/$label.log)" | tee -a "$OUT/logs/progress.log"; return 1; }
  t1=$(date +%s)
  echo "== leg $label done in $((t1-t0))s :: $(grep -o 'done: .*' "$OUT/logs/$label.log" | tail -1)" \
      | tee -a "$OUT/logs/progress.log"
}

# LADDER_LEGS selects which families run (default: all five). The ANCHOR run uses
# this to drop `avif_rav1e`: MEASURED 2026-09-05 at ~10 s/cell on the anchor's
# 6.27-MP-mean sources (vs ~1.2 s/cell on the grid's 0.84 MP), i.e. 5.9 h for that
# leg alone. The anchor shapes the output SPLINE; the INSTRUMENT — which does carry
# avif_rav1e — is what grades. Re-including it is registered, not run.
LEGS="${LADDER_LEGS:-jpeg,webp,avif_svt,avif_rav1e,jxl}"
want () { [[ ",$LEGS," == *",$1,"* ]]; }

want jpeg       && run_leg jpeg        zenjpeg "$QG"
want webp       && run_leg webp        zenwebp "$QG"
want avif_svt   && run_leg avif_svt    zenavif "$QG" '{"backend":["svt-rs"]}'
want avif_rav1e && run_leg avif_rav1e  zenavif "$QG" '{"backend":["zenravif"]}'
want jxl        && run_leg jxl         zenjxl  50    "$JXL_KNOB"

echo "LADDER-GRID-ALL-DONE $(date -u +%H:%M:%SZ)" | tee -a "$OUT/logs/progress.log"
touch "$OUT/COMPLETE"
