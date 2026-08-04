#!/usr/bin/env bash
# await_artifacts.sh — the ONE way to wait for detached compute in this repo.
#
# WHY THIS EXISTS (measured, benchmarks/rnd_cycle_audit_2026-08-04.md):
# on 2026-08-03/04 the campaign lost 6.77 h of wall-clock to whole-session idle
# in which nothing was computing OR finished work sat unharvested — 46% of all
# idle time. The two worst single events were a hand-rolled wait each:
#
#   * wave-6 arm F: every bake was verdicted + fullevaled by 03:08:40Z; the
#     results commit landed 05:20:40Z. 125.6 min of finished compute sitting
#     on disk with nobody looking.
#   * coherence wave: lianli's last bake at 19:11:30Z, first agent action at
#     20:32:05Z. 80.6 min.
#
# In both cases the compute was fine and a bespoke `while sleep` waiter was
# the thing that failed — it exited, timed out, or its `tail -f` lost the file,
# and it left NO evidence that it had stopped. A wait that dies silently is
# indistinguishable from a wait that is still waiting, so nobody re-checks.
#
# THE CONTRACT this script provides:
#   1. It ALWAYS writes a terminal sentinel file. Normal exit, timeout, `set -e`
#      failure, SIGTERM/INT/HUP — an EXIT trap writes `<heartbeat>.done` with a
#      status word and an exit code. There is no path where it stops without
#      saying so. That is the property hand-rolled loops lack.
#   2. The sentinel is the terminal condition to watch — a FILE, not a log line.
#      File existence survives log rotation, buffering, and a dropped `tail -f`.
#   3. The heartbeat file carries a fresh timestamp on every poll, so "still
#      working" and "died 40 minutes ago" are distinguishable at a glance.
#   4. Nonzero exit on timeout, so a driver chain stops instead of continuing
#      on absent inputs.
#
# USAGE
#   scripts/await_artifacts.sh --count 12 --glob '/path/to/bakes/C_w8_s*.bin' \
#       --heartbeat ~/tmp/wave8/await --timeout 14400 [--interval 60] \
#       [--label 'wave-8 bakes'] [--also-glob '<path>' --also-count N] \
#       [--ready-cmd '<shell test>']
#
#   # detached (the normal case — survives the launching shell):
#   setsid nohup scripts/await_artifacts.sh ... >/dev/null 2>&1 &
#
#   # then watch ONE file, and only that file:
#   #   ~/tmp/wave8/await.done      appears exactly once, whatever happens
#   #   ~/tmp/wave8/await.status    one line, rewritten each poll
#   #   ~/tmp/wave8/await.log       append-only timestamped history
#
# EXIT CODES
#   0  terminal condition met
#   3  timeout
#   4  bad usage
#   5  killed by a signal (recorded in the sentinel)
set -uo pipefail

COUNT=""; GLOB=""; ALSO_GLOB=""; ALSO_COUNT=""; READY_CMD=""
HEARTBEAT=""; TIMEOUT=""; INTERVAL=60; LABEL="artifacts"

die() { echo "await_artifacts: $*" >&2; exit 4; }

while [ $# -gt 0 ]; do
    case "$1" in
        --count)      COUNT=${2:?};      shift 2 ;;
        --glob)       GLOB=${2:?};       shift 2 ;;
        --also-glob)  ALSO_GLOB=${2:?};  shift 2 ;;
        --also-count) ALSO_COUNT=${2:?}; shift 2 ;;
        --ready-cmd)  READY_CMD=${2:?};  shift 2 ;;
        --heartbeat)  HEARTBEAT=${2:?};  shift 2 ;;
        --timeout)    TIMEOUT=${2:?};    shift 2 ;;
        --interval)   INTERVAL=${2:?};   shift 2 ;;
        --label)      LABEL=${2:?};      shift 2 ;;
        -h|--help)    sed -n '2,60p' "$0"; exit 0 ;;
        *)            die "unknown arg: $1" ;;
    esac
done

[ -n "$HEARTBEAT" ] || die "--heartbeat <path> is required (it is the evidence trail)"
[ -n "$TIMEOUT" ]   || die "--timeout <seconds> is required (an unbounded wait is the bug)"
[ -n "$GLOB$READY_CMD" ] || die "need --glob (+--count) or --ready-cmd"
[ -z "$GLOB" ] || [ -n "$COUNT" ] || die "--glob requires --count"
case "$TIMEOUT$INTERVAL" in *[!0-9]*) die "--timeout/--interval must be integer seconds" ;; esac

mkdir -p "$(dirname "$HEARTBEAT")" || die "cannot create heartbeat dir"
STATUS="$HEARTBEAT.status"
LOG="$HEARTBEAT.log"
DONE="$HEARTBEAT.done"
rm -f "$DONE"

now()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
# Count matches without tripping over an unmatched glob (which bash leaves
# literal) or over filenames containing spaces.
nmatch() {
    local pat=$1 n=0 f
    for f in $pat; do [ -e "$f" ] && n=$((n + 1)); done
    printf '%s' "$n"
}
say() { printf '%s %s\n' "$(now)" "$*" >> "$LOG"; }

STATE=RUNNING
RC=5
# The whole point: every exit path leaves the sentinel behind.
finish() {
    local rc=$RC
    printf '%s %s rc=%s label=%s pid=%s\n' "$(now)" "$STATE" "$rc" "$LABEL" "$$" > "$DONE"
    say "TERMINAL $STATE rc=$rc"
    exit "$rc"
}
trap 'STATE=SIGNAL; RC=5; finish' TERM INT HUP
trap 'finish' EXIT

START=$(date +%s)
DEADLINE=$((START + TIMEOUT))
say "START label='$LABEL' pid=$$ glob='$GLOB' count=${COUNT:-–} ready_cmd='${READY_CMD:-–}' timeout=${TIMEOUT}s interval=${INTERVAL}s deadline=$(date -u -d "@$DEADLINE" +%Y-%m-%dT%H:%M:%SZ)"

while :; do
    T=$(date +%s)
    ok=1
    prog=""

    if [ -n "$GLOB" ]; then
        n=$(nmatch "$GLOB")
        prog="$prog $n/$COUNT"
        [ "$n" -ge "$COUNT" ] || ok=0
    fi
    if [ -n "$ALSO_GLOB" ]; then
        n2=$(nmatch "$ALSO_GLOB")
        prog="$prog also=$n2/${ALSO_COUNT:-1}"
        [ "$n2" -ge "${ALSO_COUNT:-1}" ] || ok=0
    fi
    if [ -n "$READY_CMD" ]; then
        if bash -c "$READY_CMD" >/dev/null 2>&1; then prog="$prog ready=yes"
        else prog="$prog ready=no"; ok=0; fi
    fi

    printf '%s %s elapsed=%ss remaining=%ss%s\n' \
        "$(now)" "$LABEL" "$((T - START))" "$((DEADLINE - T))" "$prog" > "$STATUS"
    say "poll$prog elapsed=$((T - START))s"

    if [ "$ok" = 1 ]; then STATE=COMPLETE; RC=0; finish; fi
    if [ "$T" -ge "$DEADLINE" ]; then STATE=TIMEOUT; RC=3; finish; fi
    # `sleep &` + `wait` rather than a plain `sleep`: bash defers a trap until
    # the current FOREGROUND command finishes, so a plain `sleep 600` would
    # swallow SIGTERM for up to ten minutes. `wait` is interrupted immediately,
    # so the sentinel gets written the moment the signal lands.
    sleep "$INTERVAL" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null || true
done
