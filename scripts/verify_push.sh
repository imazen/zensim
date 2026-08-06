#!/usr/bin/env bash
# verify_push.sh — C4 (opus-review, campaign appendix W): the mechanical
# "is it REALLY on the remote" check.
#
# Sub-agent reports have claimed "verified pushed" for commits that were never
# on the tracked branch (the cvvdp Path-A orphan, 2026-05-29; an orphaned
# "verified pushed" commit again in this session's window). Words are not
# evidence. This script fetches, tests ancestry, and emits ONE line the report
# must paste VERBATIM — a supervisor greps for `VERIFY-PUSH OK <sha>` and
# re-runs the same command to confirm; anything else is not verification.
#
#   scripts/verify_push.sh <commit-ish> [remote-ref]   # default origin/main
#
# Works from the primary colocated checkout AND from a secondary jj workspace
# (which has no .git — the git path fails there, so we fall back to jj's view
# of the same store; that fallback failing silently is exactly how empty
# provenance shipped in 3 of 4 appendix-R .meta sidecars).
set -euo pipefail

commit=${1:?usage: verify_push.sh <commit-ish> [remote-ref=origin/main]}
remote_ref=${2:-origin/main}
now=$(date -u +%Y-%m-%dT%H:%M:%SZ)

emit_ok() { # $1 = full sha
    printf 'VERIFY-PUSH OK %s is-ancestor-of %s checked=%s\n' "$1" "$remote_ref" "$now"
}
emit_fail() { # $1 = what we know
    printf 'VERIFY-PUSH FAIL %s NOT confirmed on %s checked=%s (%s)\n' \
        "$commit" "$remote_ref" "$now" "$1"
    exit 1
}

if git rev-parse --git-dir >/dev/null 2>&1; then
    git fetch -q "${remote_ref%%/*}" "${remote_ref#*/}" 2>/dev/null \
        || emit_fail "git fetch ${remote_ref} failed"
    full=$(git rev-parse --verify "${commit}^{commit}" 2>/dev/null) \
        || emit_fail "unknown commit locally"
    if git merge-base --is-ancestor "$full" "$remote_ref"; then
        emit_ok "$full"
    else
        emit_fail "merge-base --is-ancestor says NO"
    fi
elif command -v jj >/dev/null 2>&1 && jj --ignore-working-copy root >/dev/null 2>&1; then
    # Secondary jj workspace: no .git here. jj sees the same commits; the
    # remote bookmark form of origin/main is main@origin.
    jjref=${remote_ref#origin/}@origin
    jj git fetch >/dev/null 2>&1 || emit_fail "jj git fetch failed"
    full=$(jj --ignore-working-copy log --no-graph -r "present(${commit})" \
            -T 'commit_id' 2>/dev/null | head -1) || true
    [ -n "${full:-}" ] || emit_fail "unknown commit in jj store"
    anc=$(jj --ignore-working-copy log --no-graph \
            -r "present(${commit}) & ::${jjref}" -T 'commit_id' 2>/dev/null | head -1) || true
    if [ -n "${anc:-}" ]; then
        emit_ok "$full"
    else
        emit_fail "not an ancestor of ${jjref} per jj"
    fi
else
    emit_fail "neither git nor jj available here"
fi
