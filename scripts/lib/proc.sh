# proc.sh — C7 (opus-review, campaign appendix W): safe process-matching
# helpers. Source this instead of hand-rolling pgrep.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/proc.sh"   # from scripts/*
#
# THE TRAP (documented in ~/.claude/CLAUDE.md, and it STILL bit an agent this
# window): `pgrep -f PATTERN` matches the FULL COMMAND LINE of every process —
# including the `bash -c '... pgrep -f PATTERN ...'` that is asking, and any
# wrapper shell whose argv contains your script text. Wait-loops built on it
# never exit (a queued job once waited EIGHT HOURS on its own wrapper), and
# `pkill -f` kills the wrong thing. Also: pgrep name-match truncates comm to
# 15 chars, so `pgrep -x a-very-long-binary-name` silently never matches.
#
# Preference order: (1) exact name, (2) recorded PID, (3) completion marker.

# Exact-name match (never the command line). Returns 0 if any such process
# exists. Comm is truncated to 15 chars by the kernel — this truncates the
# pattern to match, loudly.
proc_alive_exact() {
    local name=$1
    if [ "${#name}" -gt 15 ]; then
        echo "proc_alive_exact: '$name' >15 chars — matching kernel-truncated '${name:0:15}'" >&2
        name=${name:0:15}
    fi
    pgrep -x "$name" >/dev/null
}

# Count of exact-name processes (0 when none, never miscounts wrappers).
proc_count_exact() {
    local name=$1
    [ "${#name}" -gt 15 ] && name=${name:0:15}
    pgrep -cx "$name" 2>/dev/null || echo 0
}

# Is a recorded PID still alive? Unambiguous by construction — capture `$!`
# when you launch, use this to wait/kill.
pid_alive() { kill -0 "$1" 2>/dev/null; }

# Wait until PID exits; poll every $2 s (default 5), timeout $3 s (default
# 3600, rc 124 on timeout). For processes you launched yourself.
pid_wait() {
    local pid=$1 step=${2:-5} timeout=${3:-3600} t=0
    while pid_alive "$pid"; do
        sleep "$step"; t=$((t + step))
        [ "$t" -ge "$timeout" ] && return 124
    done
    return 0
}

# Wait for a completion marker file — the most robust cross-host wait: the
# JOB says when it is done instead of being inferred from the process table.
# rc 124 on timeout. Pair with scripts/await_artifacts.sh --heartbeat, whose
# .done fires on EVERY exit path.
marker_wait() {
    local f=$1 step=${2:-15} timeout=${3:-7200} t=0
    until [ -e "$f" ]; do
        sleep "$step"; t=$((t + step))
        [ "$t" -ge "$timeout" ] && return 124
    done
    return 0
}

# Kill by exact name — REFUSES patterns that look like -f abuse (spaces /
# slashes mean you wanted a command line, and that is the footgun).
proc_kill_exact() {
    local name=$1 sig=${2:-TERM}
    case $name in
        *' '*|*/*)
            echo "proc_kill_exact: '$name' contains space/slash — kill by PID instead (pkill -f is banned here)" >&2
            return 2 ;;
    esac
    [ "${#name}" -gt 15 ] && name=${name:0:15}
    pkill "-$sig" -x "$name"
}
