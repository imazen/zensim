#!/usr/bin/env bash
# Copyright (c) Imazen LLC.
# Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
#
# Fast-class extraction sweep driver — the ASLR protocol from
# `benchmarks/era2_perf_break_2026-08-31.md` §22.5, mechanised so a cell cannot
# be produced the wrong way by hand.
#
# What the protocol requires, and what this script does about each:
#
#   one binary + RUNTIME arms  -> every arm is a `ZENSIM_BIGPAIR_TOGGLES` value
#                                 against ONE `foldapp_stream_bigpair` build.
#   identical env byte length  -> every arm name here is exactly 3 characters,
#                                 and the script REFUSES a longer one. The env
#                                 block's size is itself a layout input; an arm
#                                 selected by a longer string is measuring a
#                                 different address space, not a different walk.
#   arms interleaved           -> the arm loop is INSIDE the start loop, so arm
#                                 order rotates per start rather than running
#                                 all of one arm then all of the next.
#   min of N walks in-process  -> ZENSIM_BIGPAIR_ITERS, which prints its own min.
#   min over >=15 starts       -> STARTS (default 15); the reduction is min.
#   ASLR ON                    -> no `setarch -R`. That is a second opinion,
#                                 never the primary.
#   bit-identical control arm  -> CONTROL (default `15c`) runs once per
#                                 start, so its own min-vs-median spread across
#                                 the starts IS the cell's measured noise floor.
#                                 If that spread exceeds the effect being
#                                 claimed, the cell is UNESTABLISHED, and the
#                                 report must say so rather than round it away.
#
# And the hazard the protocol does NOT cover, which bit this repo once:
# own-process CPU contention. `nice` lowers priority, it does not isolate.
# A sweep run beside another lane's extraction or `cargo` is contaminated, and
# `min()` does not save you because contention only adds time to SOME starts.
# So every cell self-checks load first and REFUSES rather than emitting a
# number it cannot stand behind (`--force` overrides, and stamps the output).
#
# Usage:
#   scripts/kernel_fastclass_sweep.sh --arms 156,15o --sizes 576,1152 \
#       --threads 1,8 --starts 15 --iters 7 --out /mnt/v/output/.../cell.tsv
#   ZEN_S2_CAP_V3=1 ... same, for the v3 (AVX2) tier.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ZEN_BIGPAIR_BIN:-$REPO/target/release/examples/foldapp_stream_bigpair}"

ARMS="156,15o"
SIZES="576,1152,2304"
THREADS="1,8"
STARTS=15
ITERS=7
CONTROL="15c"
OUT=""
FORCE=0
# Refuse above this 1-minute load average. The box is 32 threads; a sweep
# wants the machine, so anything much above the sweep's own thread count means
# somebody else is on it.
MAXLOAD="${ZEN_SWEEP_MAXLOAD:-3.0}"

while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift 2;;
    --sizes) SIZES="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --starts) STARTS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --control) CONTROL="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --maxload) MAXLOAD="$2"; shift 2;;
    --force) FORCE=1; shift;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

[ -x "$BIN" ] || { echo "missing $BIN — cargo build --release -p zensim --example foldapp_stream_bigpair" >&2; exit 2; }

# ---- the env-byte-length invariant, enforced rather than trusted.
for a in ${ARMS//,/ } "$CONTROL"; do
  if [ ${#a} -ne 3 ]; then
    echo "REFUSED: arm '$a' is ${#a} characters, not 3." >&2
    echo "  The ASLR protocol requires byte-identical environment blocks between" >&2
    echo "  interleaved arms. Add a 3-character alias in foldapp_stream_bigpair.rs" >&2
    echo "  (that is what '15o' exists for against '156off')." >&2
    exit 3
  fi
done

# ---- load self-check. Returns 0 when the box is ours.
box_is_idle() {
  local la busy
  la=$(awk '{print $1}' /proc/loadavg)
  # Named heavy neighbours, matched on the EXECUTABLE name only. `pgrep -f`
  # would match this script's own command line (CLAUDE.md: never -f on a
  # pattern that can match yourself); `comm` truncates at 15 chars, so these
  # are the truncated forms.
  busy=0
  for n in extract_feature zensim_mlp_trai rustc cargo bake_verdict; do
    if pgrep -x "$n" >/dev/null 2>&1; then
      echo "  busy: $n running" >&2
      busy=1
    fi
  done
  if awk -v a="$la" -v b="$MAXLOAD" 'BEGIN{exit !(a>b)}'; then
    echo "  busy: loadavg $la > $MAXLOAD" >&2
    busy=1
  fi
  [ "$busy" -eq 0 ]
}

TIER="native(v4x)"
[ "${ZEN_S2_CAP_V3:-0}" = "1" ] && TIER="capped(v3/AVX2)"

emit() { if [ -n "$OUT" ]; then echo -e "$1" >> "$OUT"; fi; echo -e "$1"; }

if [ -n "$OUT" ]; then
  mkdir -p "$(dirname "$OUT")"
  : > "$OUT"
fi
emit "# kernel_fastclass_sweep $(date -u +%FT%TZ) host=$(hostname) tier=$TIER"
emit "# bin=$BIN sha=$(sha256sum "$BIN" | cut -c1-16) starts=$STARTS iters=$ITERS control=$CONTROL"
emit "# git=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')"
emit "size\tthreads\ttier\tarm\tmin_ms\tmed_of_start_mins\tn_starts\tstatus"

for size in ${SIZES//,/ }; do
for thr in ${THREADS//,/ }; do
  # One waiting point per cell. Foreground, bounded, no background watcher.
  waited=0
  while ! box_is_idle; do
    if [ "$FORCE" = "1" ]; then
      echo "  --force: proceeding on a busy box; cell will be stamped CONTAMINATED" >&2
      break
    fi
    if [ "$waited" -ge 60 ]; then
      emit "$size\t$thr\t$TIER\t-\t-\t-\t0\tSKIPPED_BOX_BUSY_60min"
      continue 2
    fi
    echo "[$(date -u +%T)] box busy; cell ${size}/${thr}T waiting 60s (waited ${waited}m)" >&2
    sleep 60
    waited=$((waited+1))
  done

  declare -A MINS=()
  for a in ${ARMS//,/ } "$CONTROL"; do MINS[$a]=""; done

  for s in $(seq 1 "$STARTS"); do
    # Interleave: rotate arm order per start so no arm is systematically first.
    IFS=',' read -ra AR <<< "$ARMS"
    SEQ=("${AR[@]}" "$CONTROL")
    n=${#SEQ[@]}
    for ((k=0;k<n;k++)); do
      idx=$(( (k + s) % n ))
      a="${SEQ[$idx]}"
      out=$(RAYON_NUM_THREADS="$thr" \
            ZENSIM_BIGPAIR_ITERS="$ITERS" \
            ZENSIM_BIGPAIR_TOGGLES="$a" \
            ZENSIM_BIGPAIR_PARALLEL=$([ "$thr" -gt 1 ] && echo 1 || echo 0) \
            nice -n19 "$BIN" "$size" "$size" 2>&1 | grep -oP 'min \K[0-9.]+' | head -1)
      [ -n "$out" ] && MINS[$a]="${MINS[$a]} $out"
    done
  done

  for a in "${!MINS[@]}"; do
    vals=$(echo "${MINS[$a]}" | tr ' ' '\n' | grep -v '^$' | sort -g)
    [ -z "$vals" ] && continue
    mn=$(echo "$vals" | head -1)
    cnt=$(echo "$vals" | wc -l)
    med=$(echo "$vals" | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
    st="OK"; [ "$FORCE" = "1" ] && st="CONTAMINATED"
    emit "$size\t$thr\t$TIER\t$a\t$mn\t$med\t$cnt\t$st"
  done
done
done
echo "done -> ${OUT:-stdout}" >&2
