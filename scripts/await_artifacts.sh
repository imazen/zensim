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
#   5. RESTART-PROOF ENDGAME (--then): the wave's endgame runs HERE, in this
#      detached OS process, the moment the terminal condition is met — not in
#      an agent woken by a notification. Agent wake-chains do not survive a
#      Claude Code host restart (background bashes, Monitor watches, and
#      pending subagent notifications are never restored on resume); a setsid
#      process is indifferent to it. The 2026-08-05 four-orphan incident
#      (featsub, wave-11, hygiene2, HDR) lost only supervisor latency because
#      per-bake harvest kept state on disk — --then removes even that: by the
#      time anyone resumes, tables + doc draft already exist.
#
# USAGE
#   scripts/await_artifacts.sh --count 12 --glob '/path/to/bakes/C_w8_s*.bin' \
#       --heartbeat ~/tmp/wave8/await --timeout 14400 [--interval 60] \
#       [--label 'wave-8 bakes'] [--also-glob '<path>' --also-count N] \
#       [--ready-cmd '<shell test>'] \
#       [--then 'scripts/endgame_w8.sh'] [--then-always]
#
#   # detached (the normal case — survives the launching shell AND any
#   # Claude Code host restart):
#   setsid nohup scripts/await_artifacts.sh ... >/dev/null 2>&1 &
#
#   # then watch ONE file, and only that file:
#   #   ~/tmp/wave8/await.done          appears exactly once, whatever happens
#   #   ~/tmp/wave8/await.status        one line, rewritten each poll
#   #   ~/tmp/wave8/await.log           append-only timestamped history
#   #   ~/tmp/wave8/await.endgame.done  (--then only) endgame terminal record
#   #   ~/tmp/wave8/await.then.log      (--then only) endgame stdout+stderr
#
# --then CONTRACT
#   * The command MUST be a COMMITTED script (pre-register discipline; an
#     uncommitted endgame under ~/tmp is warned about loudly — that is the
#     wave-6 `process.sh` failure class).
#   * It MUST be idempotent — a re-run of await re-runs it.
#   * It runs on COMPLETE; with --then-always also on TIMEOUT (draft what
#     exists — a partial wave still needs its tables); NEVER on SIGNAL (a
#     deliberate kill means a human took over).
#   * Env: AA_STATE, AA_RC, AA_HEARTBEAT describe the terminal condition.
#   * Its terminal record is written to <heartbeat>.endgame.done on EVERY
#     endgame exit path (success, failure, signal-during-endgame). A missing
#     .endgame.done next to a .done that says COMPLETE means the endgame is
#     still running — check `<heartbeat>.then.log`.
#
# EXIT CODES
#   0  terminal condition met (and endgame, if any, succeeded)
#   3  timeout
#   4  bad usage
#   5  killed by a signal (recorded in the sentinel)
#   7  terminal condition met but the --then endgame FAILED (see .endgame.done)
set -uo pipefail

COUNT=""; GLOB=""; ALSO_GLOB=""; ALSO_COUNT=""; READY_CMD=""
HEARTBEAT=""; TIMEOUT=""; INTERVAL=60; LABEL="artifacts"
THEN_CMD=""; THEN_ALWAYS=0

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
        --then)       THEN_CMD=${2:?};   shift 2 ;;
        --then-always) THEN_ALWAYS=1;    shift ;;
        -h|--help)    sed -n '2,90p' "$0"; exit 0 ;;
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
ENDGAME_DONE="$HEARTBEAT.endgame.done"
THEN_LOG="$HEARTBEAT.then.log"
rm -f "$DONE"
[ -z "$THEN_CMD" ] || rm -f "$ENDGAME_DONE"

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
WROTE_DONE=0
THEN_STARTED=0
# The whole point: every exit path leaves the sentinel behind. Guarded so a
# trap re-entering after terminal() cannot rewrite an already-written sentinel
# (the .done must describe the WATCH terminal state, not the endgame).
write_done_once() {
    if [ "$WROTE_DONE" = 0 ]; then
        WROTE_DONE=1
        printf '%s %s rc=%s label=%s pid=%s\n' "$(now)" "$STATE" "$RC" "$LABEL" "$$" > "$DONE"
        say "TERMINAL $STATE rc=$RC"
    fi
}
finish() {
    write_done_once
    # Endgame-layer sentinel discipline: if a signal lands MID-endgame, the
    # endgame record still gets written — a missing .endgame.done must mean
    # exactly one thing ("still running"), never "died silently".
    if [ "$THEN_STARTED" = 1 ] && [ ! -f "$ENDGAME_DONE" ]; then
        printf '%s SIGNAL rc=5 cmd=%s\n' "$(now)" "$THEN_CMD" > "$ENDGAME_DONE"
        say "ENDGAME SIGNAL (killed mid-run; see $THEN_LOG)"
    fi
    exit "$RC"
}
trap 'STATE=SIGNAL; RC=5; finish' TERM INT HUP
trap 'finish' EXIT

# Run the committed endgame IN THIS PROCESS, after the watch sentinel is on
# disk. Restart-proofness lives here: this is a detached OS process, so the
# endgame executes even if every Claude Code session on the box is gone.
maybe_then() {
    [ -n "$THEN_CMD" ] || return 0
    case "$STATE" in
        COMPLETE) ;;
        TIMEOUT) [ "$THEN_ALWAYS" = 1 ] || { say "ENDGAME SKIPPED (state=$STATE without --then-always)"; return 0; } ;;
        *) say "ENDGAME SKIPPED (state=$STATE)"; return 0 ;;
    esac
    case "$THEN_CMD" in
        /tmp/*|"$HOME"/tmp/*|~/tmp/*)
            say "WARNING: --then command '$THEN_CMD' is not a committed script — the wave-6 uncommitted-process.sh failure class. Commit it." ;;
    esac
    THEN_STARTED=1
    say "ENDGAME START cmd='$THEN_CMD' (log: $THEN_LOG)"
    local trc=0
    AA_STATE="$STATE" AA_RC="$RC" AA_HEARTBEAT="$HEARTBEAT" \
        nice -n 19 ionice -c 3 bash -c "$THEN_CMD" >>"$THEN_LOG" 2>&1 || trc=$?
    if [ "$trc" = 0 ]; then
        printf '%s COMPLETE rc=0 cmd=%s\n' "$(now)" "$THEN_CMD" > "$ENDGAME_DONE"
        say "ENDGAME COMPLETE"
    else
        printf '%s FAILED rc=%s cmd=%s\n' "$(now)" "$trc" "$THEN_CMD" > "$ENDGAME_DONE"
        say "ENDGAME FAILED rc=$trc (see $THEN_LOG)"
        RC=7
    fi
}

# Terminal-condition path: write the watch sentinel FIRST (watchers unblock on
# the artifacts immediately), then run the endgame, then exit.
terminal() {
    write_done_once
    maybe_then
    exit "$RC"
}

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

    if [ "$ok" = 1 ]; then STATE=COMPLETE; RC=0; terminal; fi
    if [ "$T" -ge "$DEADLINE" ]; then STATE=TIMEOUT; RC=3; terminal; fi
    # `sleep &` + `wait` rather than a plain `sleep`: bash defers a trap until
    # the current FOREGROUND command finishes, so a plain `sleep 600` would
    # swallow SIGTERM for up to ten minutes. `wait` is interrupted immediately,
    # so the sentinel gets written the moment the signal lands.
    sleep "$INTERVAL" &
    SLEEP_PID=$!
    wait "$SLEEP_PID" 2>/dev/null || true
done
