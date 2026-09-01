#!/usr/bin/env bash
# Free-feature A/B: does the free superset walk cost anything over the 156 walk?
#   benchmarks/free_features_2026-09-01.md
#
# THE PROTOCOL (era-2 §22.5 — the noise floor at these sizes is ASLR, and it is
# 10 %, so none of this is optional):
#   1. ONE binary, arms selected at RUNTIME (`ZENSIM_BIGPAIR_TOGGLES`).
#   2. BYTE-IDENTICAL environment blocks between arms — every arm name is
#      exactly three characters, and no variable is set for one arm and not
#      another. The env block's SIZE is itself a layout input.
#   3. Arms INTERLEAVED, never `base×N` then `arm×N`.
#   4. `min` of N walks inside a process (removes interference).
#   5. `min` over >= 15 process starts with ASLR on (removes layout).
#   6. A bit-identical CONTROL arm carried throughout: `15c` is `15f` with the
#      four accumulators off and everything else — including the output
#      vector's width — the same, so `15c` vs `15f` isolates exactly the
#      question and `156` vs `15c` prices the wider layout separately.
#
# Usage: scripts/freefeats_ab.sh [OUTDIR]
# Env:   FF_STARTS (default 15)  FF_ITERS (default 7)
#        FF_SIZES  (default "576 1152 2304")
#        FF_THREADS(default "1 8 16")
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/mnt/v/output/zensim/freefeats-2026-09-01}"
STARTS="${FF_STARTS:-15}"
ITERS="${FF_ITERS:-7}"
SIZES="${FF_SIZES:-576 1152 2304}"
THREADS="${FF_THREADS:-1 8 16}"
ARMS="156 15c 15f"
mkdir -p "$OUT"
TSV="$OUT/ab_raw.tsv"
LOG="$OUT/ab_progress.log"

: "${CARGO_TARGET_DIR:=$REPO/target-ff}"
export CARGO_TARGET_DIR
BIN="$CARGO_TARGET_DIR/release/examples/foldapp_stream_bigpair"

echo "[build] $BIN" | tee -a "$LOG"
( cd "$REPO" && cargo build --release -p zensim \
    --features custom-profiles,feature-regime-v2,threads,training \
    --example foldapp_stream_bigpair >>"$LOG" 2>&1 )
[[ -x "$BIN" ]] || { echo "missing $BIN" >&2; exit 2; }

printf 'size\tthreads\tarm\tstart\tmin_ms\tmedian_ms\tload1\n' > "$TSV"
echo "[run] starts=$STARTS iters=$ITERS sizes='$SIZES' threads='$THREADS' arms='$ARMS'" | tee -a "$LOG"
# jj workspaces have no .git of their own, so ask jj first and fall back to git.
REV=$( (cd "$REPO" && jj log -r @- --no-graph -T 'commit_id.short()' 2>/dev/null) \
       || (cd "$REPO" && git rev-parse --short HEAD 2>/dev/null) || echo unknown )
echo "[run] commit $REV  host $(hostname)  $(date -u +%FT%TZ)" | tee -a "$LOG"

for size in $SIZES; do
  for th in $THREADS; do
    # CCD0 pin (9950X3D: 96 MiB L3 on CCD0) — removes one variable for free.
    PIN=(taskset -c 0-7,16-23)
    PAR=1; [[ "$th" == "1" ]] && PAR=0
    for start in $(seq 1 "$STARTS"); do
      for arm in $ARMS; do          # arms INTERLEAVED inside each start
        line=$(env -i \
          PATH=/usr/bin:/bin \
          HOME="$HOME" \
          RAYON_NUM_THREADS="$th" \
          ZENSIM_BIGPAIR_TOGGLES="$arm" \
          ZENSIM_BIGPAIR_PARALLEL="$PAR" \
          ZENSIM_BIGPAIR_ITERS="$ITERS" \
          "${PIN[@]}" "$BIN" "$size" "$size" 2>&1 | grep '^arm=' || true)
        mn=$(sed -E 's/.*min ([0-9.]+).*/\1/' <<<"$line")
        md=$(sed -E 's/.*median ([0-9.]+) ms.*/\1/' <<<"$line")
        l1=$(cut -d' ' -f1 /proc/loadavg)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$size" "$th" "$arm" "$start" "$mn" "$md" "$l1" >> "$TSV"
      done
      echo "[$(date -u +%T)] ${size}^2 ${th}T start $start/$STARTS load $(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
    done
    echo "[$(date -u +%T)] DONE ${size}^2 ${th}T" | tee -a "$LOG"
  done
done
echo "[done] $TSV" | tee -a "$LOG"
