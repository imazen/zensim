#!/usr/bin/env bash
#
# wave9_pull.sh — copy finished lianli bakes into the canonical local bake dir
# so the ONE harvest daemon (scripts/harvest_bakes.sh) sees every cell,
# whichever lane produced it. Not a waiter: it does work, and the waiter still
# owns the terminal condition.
#
# Pulls only bakes whose remote size has been STABLE across two polls, so a
# file still being written can never be harvested half-copied. Copies to a
# .part and renames, so the local appearance is atomic.
set -uo pipefail
L=${WAVE9_REMOTE:-lilith@r7900x}
RDIR=${WAVE9_REMOTE_OUT:-/home/lilith/sota944/out/w9}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
LOG=${WAVE9_LOG:-$HOME/tmp/wave9}
COUNT=${1:?usage: wave9_pull.sh <n-remote-bakes> [timeout-s]}
TIMEOUT=${2:-21600}
mkdir -p "$OUT" "$LOG"
say() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG/pull.log"; }
declare -A seen_size
t0=$(date +%s); n=0
say "START want=$COUNT from $L:$RDIR -> $OUT"
while [ "$n" -lt "$COUNT" ]; do
    [ $(( $(date +%s) - t0 )) -lt "$TIMEOUT" ] || { say "TIMEOUT n=$n"; exit 3; }
    while read -r sz name; do
        [ -n "${name:-}" ] || continue
        [ -f "$OUT/$name" ] && continue
        if [ "${seen_size[$name]:-}" = "$sz" ] && [ "$sz" -gt 0 ]; then
            if scp -q "$L:$RDIR/$name" "$OUT/.$name.part" 2>>"$LOG/pull.log"; then
                scp -q "$L:$RDIR/$name.spec.json" "$OUT/$name.spec.json" 2>>"$LOG/pull.log" || true
                mv "$OUT/.$name.part" "$OUT/$name"
                n=$((n+1)); say "PULLED $name ($sz bytes) [$n/$COUNT]"
            else
                say "scp failed $name"; rm -f "$OUT/.$name.part"
            fi
        fi
        seen_size[$name]=$sz
    done < <(ssh -o BatchMode=yes -o ConnectTimeout=10 "$L" \
                "stat -c '%s %n' $RDIR/*.bin 2>/dev/null | sed 's|$RDIR/||'" 2>>"$LOG/pull.log")
    [ "$n" -lt "$COUNT" ] && sleep 45
done
say "DONE n=$n"
