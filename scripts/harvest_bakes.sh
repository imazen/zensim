#!/usr/bin/env bash
# harvest_bakes.sh — verdict + fulleval every bake AS IT LANDS, and fail loud.
#
# WHY THIS EXISTS (measured, benchmarks/rnd_cycle_audit_2026-08-04.md).
# Two distinct failures cost 2026-08-03/04 several hours each, and this script
# is the committed owner that removes both:
#
#   (A) FINISHED WORK SAT UNHARVESTED. 5.31 h of the day's 6.77 h of dead
#       wall-clock was compute that had completed while every agent was asleep.
#       If the verdict + fulleval are already on disk the moment the bake
#       lands, an agent waking late costs *nothing* — it reads results instead
#       of starting a 35 s/bake serial eval chain. Wave 6 proved the pattern
#       works: its ad-hoc `process.sh` had all six bakes verdicted+fullevaled
#       by 03:08:40Z. It was never committed, so wave 7 hand-rolled it again.
#
#   (B) A POST-BAKE HOOK THAT FAILED SILENTLY. The coherence wave's local lane
#       *did* auto-eval each bake — and every call exited 2 on a missing
#       corpus file (`ext_sdr25.parquet`), printing the error into a log
#       nobody read. 3 h 24 min of training produced zero usable verdicts, and
#       the whole set had to be re-run by hand at 20:53Z. A hook whose failure
#       is invisible is worse than no hook: it manufactures false confidence.
#       So here, a failed harvest writes a `.HARVEST_FAILED` marker next to the
#       bake, appends to a FAILURES file, and makes the final exit nonzero.
#
# USAGE
#   # one bake
#   scripts/harvest_bakes.sh --bake <bake.bin> [--stem NAME] [--regime 944]
#
#   # daemon: harvest everything matching a glob, until --count are done
#   scripts/harvest_bakes.sh --glob '/mnt/v/.../C_w8_s*.bin' --count 12 \
#       --regime 944 --heartbeat ~/tmp/wave8/harvest --timeout 21600
#
#   # detached (the normal case):
#   setsid nohup scripts/harvest_bakes.sh --glob ... >/dev/null 2>&1 &
#
# Idempotent: a bake whose fulleval JSON already exists is skipped, so it is
# safe to re-run, and safe to run alongside a puller that is still copying.
#
# EXIT CODES
#   0  every targeted bake harvested cleanly
#   2  bad usage
#   3  timeout before --count bakes were harvested
#   6  at least one bake failed to harvest (see the FAILURES file)
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VD=${ZENSIM_VERDICT_DIR:-/mnt/v/output/zensim/bakes/sota944/verdicts}
FE=${ZENSIM_FULLEVAL_OUT:-/mnt/v/output/zensim/reports/fulleval}

BAKE=""; STEM=""; GLOB=""; COUNT=""; REGIME=944
HEARTBEAT=""; TIMEOUT=21600; INTERVAL=60; SKIP_FULLEVAL=0

die() { echo "harvest_bakes: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --bake)          BAKE=${2:?};      shift 2 ;;
        --stem)          STEM=${2:?};      shift 2 ;;
        --glob)          GLOB=${2:?};      shift 2 ;;
        --count)         COUNT=${2:?};     shift 2 ;;
        --regime)        REGIME=${2:?};    shift 2 ;;
        --heartbeat)     HEARTBEAT=${2:?}; shift 2 ;;
        --timeout)       TIMEOUT=${2:?};   shift 2 ;;
        --interval)      INTERVAL=${2:?};  shift 2 ;;
        --no-fulleval)   SKIP_FULLEVAL=1;  shift ;;
        -h|--help)       sed -n '2,40p' "$0"; exit 0 ;;
        *)               die "unknown arg: $1" ;;
    esac
done
[ -n "$BAKE$GLOB" ] || die "need --bake <file> or --glob <pattern>"

HEARTBEAT=${HEARTBEAT:-$HOME/tmp/harvest/$(date -u +%Y%m%dT%H%M%SZ)}
mkdir -p "$(dirname "$HEARTBEAT")" "$VD" "$FE"
LOG="$HEARTBEAT.log"
STATUS="$HEARTBEAT.status"
DONE="$HEARTBEAT.done"
FAILURES="$HEARTBEAT.failures"
rm -f "$DONE"

now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
say() { printf '%s %s\n' "$(now)" "$*" | tee -a "$LOG" >&2; }

STATE=RUNNING; RC=5; NOK=0; NFAIL=0; NO_M3A=0
finish() {
    printf '%s %s rc=%s harvested=%s failed=%s no_m3a=%s\n' \
        "$(now)" "$STATE" "$RC" "$NOK" "$NFAIL" "$NO_M3A" > "$DONE"
    say "TERMINAL $STATE rc=$RC harvested=$NOK failed=$NFAIL no_m3a=$NO_M3A"
    exit "$RC"
}
trap 'STATE=SIGNAL; RC=5; finish' TERM INT HUP
trap 'finish' EXIT

# --- harvest one bake -----------------------------------------------------
# Returns 0 on success (or already-done), 1 on failure. NEVER swallows an
# error: that is failure (B) above.
harvest_one() {
    local bake=$1 stem=$2 rc=0
    if [ -f "$FE/$stem.fulleval.json" ] || { [ "$SKIP_FULLEVAL" = 1 ] && [ -f "$VD/$stem.full.json" ]; }; then
        return 0
    fi
    # A bake still being written by the trainer, or mid-rsync, is not ready.
    # This MUST fall through to the accounting block below rather than
    # early-returning: an early `return 1` here skipped the NFAIL increment and
    # the .HARVEST_FAILED marker, so a missing bake reported "COMPLETE rc=0" —
    # the exact silent-success class this script exists to prevent. Caught by
    # the H2 test on 2026-08-04; the test is the reason the bug is not shipped.
    if [ ! -s "$bake" ]; then
        rc=1; say "NOT READY $stem — bake is empty or absent"
    fi

    if [ "$rc" = 0 ] && [ ! -f "$VD/$stem.full.json" ]; then
        say "verdict $stem"
        if ! "$REPO_ROOT/scripts/sota944_verdict.sh" "$bake" "$stem" >>"$LOG" 2>&1; then
            rc=1; say "VERDICT FAILED $stem"
        fi
    fi
    if [ "$rc" = 0 ] && [ "$SKIP_FULLEVAL" = 0 ] && [ ! -f "$FE/$stem.fulleval.json" ]; then
        say "fulleval $stem (regime $REGIME)"
        if ! nice -n 19 ionice -c 3 "$REPO_ROOT/scripts/run_full_eval.sh" \
                "$bake" "$stem" "$REGIME" >>"$LOG" 2>&1; then
            rc=1; say "FULLEVAL FAILED $stem"
        fi
    fi

    # M3a coverage guard (campaign appendix E.6). M3a is now a FIRST-CLASS
    # selection input: `freeze_check --select` treats a missing M3a as
    # UNMEASURED and therefore NOT SELECTABLE. A bake that harvests "clean"
    # but silently carries no M3a would drop out of selection at the end of
    # the wave, which is the same invisible-failure class this script exists
    # to prevent — so it is loud, on disk, next to the artifact.
    #
    # It is a WARNING, not a harvest failure: era-bridge widths and ensembles
    # legitimately have no M3a (the coherence instrument loads one ZNPR), and
    # failing those would be a false alarm.
    if [ "$rc" = 0 ] && [ "$SKIP_FULLEVAL" = 0 ] && [ -f "$FE/$stem.fulleval.json" ]; then
        m3a=$(jq -r '.m3a_coherence // "null"' "$FE/$stem.fulleval.json" 2>/dev/null)
        if [ "$m3a" = "null" ] || [ -z "$m3a" ]; then
            NO_M3A=$((NO_M3A + 1))
            printf '%s no m3a_coherence in %s — UNMEASURED, so NOT SELECTABLE under freeze_check --select\n' \
                "$(now)" "$stem" > "$bake.NO_M3A"
            say "WARNING $stem — NO M3a (not selectable; see $bake.NO_M3A)"
        else
            rm -f "$bake.NO_M3A"
        fi
    fi

    if [ "$rc" = 0 ]; then
        rm -f "$bake.HARVEST_FAILED"
        NOK=$((NOK + 1))
        say "OK $stem"
    else
        # Loud, on disk, next to the artifact — not only in a log nobody reads.
        printf '%s harvest failed: %s (see %s)\n' "$(now)" "$stem" "$LOG" \
            > "$bake.HARVEST_FAILED"
        printf '%s %s\n' "$(now)" "$stem" >> "$FAILURES"
        NFAIL=$((NFAIL + 1))
    fi
    return "$rc"
}

stem_of() { basename "$1" .bin; }

# --- single-bake mode -----------------------------------------------------
if [ -n "$BAKE" ]; then
    harvest_one "$BAKE" "${STEM:-$(stem_of "$BAKE")}" || true
    if [ "$NFAIL" -gt 0 ]; then STATE=FAILED; RC=6; else STATE=COMPLETE; RC=0; fi
    finish
fi

# --- daemon mode ----------------------------------------------------------
START=$(date +%s)
DEADLINE=$((START + TIMEOUT))
say "START glob='$GLOB' count=${COUNT:-all-present} regime=$REGIME timeout=${TIMEOUT}s pid=$$"
declare -A SEEN=()
while :; do
    T=$(date +%s)
    for b in $GLOB; do
        [ -e "$b" ] || continue
        s=$(stem_of "$b")
        [ -n "${SEEN[$s]:-}" ] && continue
        if harvest_one "$b" "$s"; then SEEN[$s]=ok; else SEEN[$s]=fail; fi
    done
    printf '%s harvested=%s failed=%s target=%s elapsed=%ss\n' \
        "$(now)" "$NOK" "$NFAIL" "${COUNT:-?}" "$((T - START))" > "$STATUS"

    if [ -n "$COUNT" ] && [ "$((NOK + NFAIL))" -ge "$COUNT" ]; then
        if [ "$NFAIL" -gt 0 ]; then STATE=FAILED; RC=6; else STATE=COMPLETE; RC=0; fi
        finish
    fi
    if [ -z "$COUNT" ] && [ "${#SEEN[@]}" -gt 0 ]; then
        if [ "$NFAIL" -gt 0 ]; then STATE=FAILED; RC=6; else STATE=COMPLETE; RC=0; fi
        finish
    fi
    if [ "$T" -ge "$DEADLINE" ]; then STATE=TIMEOUT; RC=3; finish; fi
    sleep "$INTERVAL" &
    wait $! 2>/dev/null || true
done
