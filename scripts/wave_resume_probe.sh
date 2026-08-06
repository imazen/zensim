#!/usr/bin/env bash
# wave_resume_probe.sh — orient a (resumed) session in ONE glance: every piece
# of detached wave state on disk right now, classified.
#
# WHY THIS EXISTS. Agent wake-chains do not survive a Claude Code host
# restart: background bashes, Monitor watches, and pending subagent
# notifications are never restored on resume (code.claude.com/docs/en/
# scheduled-tasks.md "Background Bash and monitor tasks are never restored").
# The 2026-08-05 four-orphan incident (featsub, wave-11, hygiene2, HDR lanes)
# lost nothing on disk — sentinels + per-bake harvest held all state — but
# every lane needed a HUMAN to notice and nudge, because the resumed session
# had no idea which waits existed. This probe is the missing orientation: run
# it (or wire it as a SessionStart hook — see docs/WAVE_PLAYBOOK.md) and the
# session starts knowing what finished, what failed, what is still running,
# and what died silently.
#
# USAGE
#   scripts/wave_resume_probe.sh [--hours 48] [--stale-min 10] [--root DIR ...]
#
# Scans ~/tmp (default; add --root for others) up to 3 levels deep for wave
# evidence younger than --hours:
#   TERMINAL   *.done            (await/harvest/lane sentinels — with content)
#   ENDGAME    *.endgame.done    (driver-executed endgame records)
#   FAILURE    *.failures non-empty, *.FAILED, *.HARVEST_FAILED markers
#   LIVE       *.status heartbeat fresher than --stale-min with no .done yet
#   DEAD?      *.status heartbeat OLDER than --stale-min with no .done — a
#              waiter/driver that stopped without a terminal record
#
# ALWAYS exits 0 (a probe must never block a session), prints nothing when
# there is nothing to say, and reads no stdin (safe as a hook command).
set -uo pipefail

HOURS=48; STALE_MIN=10; ROOTS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --hours)     HOURS=${2:?};     shift 2 ;;
        --stale-min) STALE_MIN=${2:?}; shift 2 ;;
        --root)      ROOTS+=("${2:?}"); shift 2 ;;
        -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
        *)           echo "wave_resume_probe: ignoring unknown arg $1" >&2; shift ;;
    esac
done
[ "${#ROOTS[@]}" -gt 0 ] || ROOTS=("$HOME/tmp")
MMIN=$((HOURS * 60))

lines=0
emit() { printf '%s\n' "$*"; lines=$((lines + 1)); }

for root in "${ROOTS[@]}"; do
    [ -d "$root" ] || continue

    # Terminal sentinels, newest first, endgame records separated out.
    while IFS= read -r f; do
        case "$f" in *.endgame.done) tag=ENDGAME ;; *) tag=TERMINAL ;; esac
        emit "$tag $f :: $(head -c 200 "$f" 2>/dev/null | tr '\n' ' ')"
    done < <(find "$root" -maxdepth 3 -type f -name '*.done' -mmin "-$MMIN" \
                -printf '%T@ %p\n' 2>/dev/null | sort -rn | cut -d' ' -f2- | head -30)

    # Failure evidence: markers and non-empty failures files.
    while IFS= read -r f; do
        emit "FAILURE $f :: $(head -c 160 "$f" 2>/dev/null | tr '\n' ' ')"
    done < <(find "$root" -maxdepth 3 -type f -mmin "-$MMIN" \
                \( -name '*.FAILED' -o -name '*.HARVEST_FAILED' \
                   -o \( -name '*.failures' -size +0 \) \) 2>/dev/null | head -20)

    # Heartbeats without a terminal record: live vs died-silently.
    while IFS= read -r s; do
        hb=${s%.status}
        [ -f "$hb.done" ] && continue
        if find "$s" -mmin "-$STALE_MIN" 2>/dev/null | grep -q .; then
            emit "LIVE $s :: $(head -c 160 "$s" 2>/dev/null | tr '\n' ' ')"
        else
            emit "DEAD? $s :: no .done and heartbeat stale >${STALE_MIN}m — waiter/driver stopped without a terminal record"
        fi
    done < <(find "$root" -maxdepth 3 -type f -name '*.status' -mmin "-$MMIN" 2>/dev/null | head -20)
done

if [ "$lines" -gt 0 ]; then
    emit "wave_resume_probe: ${lines} item(s). ENDGAME COMPLETE => review + push only. DEAD? => re-run the (idempotent) driver. Never re-message closed agent lanes about these."
fi
exit 0
