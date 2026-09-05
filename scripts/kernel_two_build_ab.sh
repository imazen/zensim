#!/usr/bin/env bash
# Copyright (c) Imazen LLC.
# Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
#
# TWO-BUILD interleaved wall-clock A/B, for a bit-exact kernel lever.
#
# WHY THIS EXISTS, AND WHAT IT CANNOT DO.
# `scripts/kernel_fastclass_sweep.sh` implements the repo's ASLR protocol for
# runtime arms inside ONE binary. A bit-exact kernel lever cannot use it: the
# lever IS the code, so the two arms are two BUILDS. CLAUDE.md is explicit that
# this shape is untrustworthy below ~10 % because any edit reshuffles the
# binary's own layout by about that much.
#
# So this driver does the most that shape allows, and no more:
#   * ASLR ON, min of ITERS walks in-process, min over >=STARTS process starts,
#     with the two binaries INTERLEAVED start-by-start (never all of A then all
#     of B) so a drift in machine state hits both arms equally.
#   * a load gate before EVERY invocation, which WAITS rather than contaminating
#     (own-process contention is real even niced; `min()` does not save you,
#     because contention adds time to only some starts).
#   * a DIRECTIONAL control instead of a bit-identical one, because no arm in
#     this walk skips the kernel under test. The prediction is stated up front:
#     a lever in a phase that is a LARGER share of arm X's walk than of arm Y's
#     must improve X by MORE than Y. If it does not, the reading is layout
#     noise and the cell is UNESTABLISHED — which is a result, not a failure.
#
# The deterministic instrument (callgrind Ir) remains PRIMARY for any claim
# about a bit-exact lever. This is corroboration with a stated ceiling.
#
# Usage:
#   scripts/kernel_two_build_ab.sh --old <bin> --new <bin> \
#       --arms 156,944full --sizes 576,1152,2304 --threads 1,8 \
#       --starts 15 --iters 7 --out /mnt/v/output/.../ab.tsv
set -uo pipefail

OLD=""; NEW=""; ARMS="156,944full"; SIZES="576,1152,2304"; THREADS="1,8"
STARTS=15; ITERS=7; OUT=""; MAXLOAD="${ZEN_SWEEP_MAXLOAD:-3.0}"; MAXWAIT=3600
while [ $# -gt 0 ]; do
  case "$1" in
    --old) OLD="$2"; shift 2;; --new) NEW="$2"; shift 2;;
    --arms) ARMS="$2"; shift 2;; --sizes) SIZES="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;; --starts) STARTS="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;; --out) OUT="$2"; shift 2;;
    --max-wait) MAXWAIT="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
[ -x "$OLD" ] && [ -x "$NEW" ] || { echo "need --old and --new executables" >&2; exit 2; }
[ -n "$OUT" ] || { echo "need --out" >&2; exit 2; }
mkdir -p "$(dirname "$OUT")"

# Wait for an idle window. Uses `pgrep -x` (exact executable name) — NEVER
# `pgrep -f`, which matches this script's own command line (CLAUDE.md).
#
# AND `comm` IS TRUNCATED TO 15 CHARACTERS, so `pgrep -x zensim_mlp_train`
# (16) can never match and the gate silently never fires. This driver shipped
# with exactly that bug for one run and produced a table measured beside a live
# trainer (record §L2.4). The names are therefore truncated HERE, in one place,
# rather than being written pre-truncated at the call site the way
# `kernel_fastclass_sweep.sh` does it — so a caller can write the real name and
# still be gated. CLAUDE.md warns about this; knowing the warning was not
# enough, having the code do it is.
wait_idle() {
  local waited=0
  while :; do
    local l busy=0
    l=$(awk '{print $1}' /proc/loadavg)
    for p in zensim_mlp_train v2_ab_extract extract_features_372col cargo rustc bake_verdict; do
      pgrep -x "${p:0:15}" >/dev/null 2>&1 && busy=1
    done
    if [ "$busy" = 0 ] && awk -v a="$l" -v b="$MAXLOAD" 'BEGIN{exit !(a<b)}'; then return 0; fi
    [ "$waited" -ge "$MAXWAIT" ] && { echo "# GAVE UP waiting: load=$l busy=$busy after ${waited}s" >&2; return 1; }
    sleep 30; waited=$((waited+30))
  done
}

echo -e "arm\tsize\tthreads\tbuild\tstart\tmin_ms" > "$OUT"
IFS=, read -ra A <<< "$ARMS"; IFS=, read -ra S <<< "$SIZES"; IFS=, read -ra T <<< "$THREADS"
for sz in "${S[@]}"; do for th in "${T[@]}"; do for arm in "${A[@]}"; do
  par=$([ "$th" -gt 1 ] && echo 1 || echo 0)
  for i in $(seq 1 "$STARTS"); do
    # INTERLEAVE: both builds inside the start loop, order rotated per start.
    if [ $((i % 2)) -eq 0 ]; then order="OLD NEW"; else order="NEW OLD"; fi
    for b in $order; do
      bin=$OLD; [ "$b" = NEW ] && bin=$NEW
      wait_idle || { echo -e "$arm\t$sz\t$th\t$b\t$i\tSKIPPED_BOX_BUSY" >> "$OUT"; continue; }
      ms=$(RAYON_NUM_THREADS=$th ZENSIM_BIGPAIR_TOGGLES=$arm ZENSIM_BIGPAIR_ITERS=$ITERS \
           ZENSIM_BIGPAIR_PARALLEL=$par "$bin" "$sz" "$sz" 2>&1 \
           | sed -n 's/.* min *\([0-9.]*\).*/\1/p' | tail -1)
      echo -e "$arm\t$sz\t$th\t$b\t$i\t${ms:-NA}" >> "$OUT"
    done
    echo "# $arm ${sz}^2 T$th start $i/$STARTS done $(date -u +%H:%M:%S)" >&2
  done
done; done; done
echo "# COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
touch "$OUT.done"
