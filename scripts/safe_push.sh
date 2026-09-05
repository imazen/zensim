#!/usr/bin/env bash
# safe_push.sh — the ONLY sanctioned way to advance a tracked bookmark in this repo.
#
# WHY THIS EXISTS
# ---------------
# `jj bookmark set main -r @ && jj git push --bookmark main` is a NON-fast-forward
# push whenever `@` does not descend from `main@origin`. jj performs it without a
# prompt, and the push succeeds: the bookmark simply MOVES SIDEWAYS and every commit
# that was only reachable from the old tip is silently unreachable from the new one.
# Nothing errors. Nothing warns. The commits still exist as local objects, so
# `jj log` on the pushing lane looks fine — the loss is only visible to whoever
# later asks "is my commit an ancestor of origin/main?".
#
# MEASURED, 2026-09-04: origin/main moved sideways TWICE in one afternoon (16:58:53
# and 17:08:29 UTC-6), and the second move dropped NINE commits from six lanes,
# including `d3a948ca` (the G-ADDR board-coverage feature, +555/-23 across six
# files). The boards on /mnt/v had been generated WITH that code, so the next
# regen from main would have silently un-drawn 46 NOT-SHIPPABLE badges. Full
# incident record: benchmarks/push_clobber_2026-09-05.md.
#
# WHAT THIS DOES
# --------------
#   fetch -> assert <bookmark>@origin is an ANCESTOR of the target -> set -> push -> verify
#
# The assertion is not a warning. If any commit is reachable from <bookmark>@origin
# but NOT from the target, this script prints every one of them and exits 3 WITHOUT
# touching the bookmark. There is no --force. Recover with:
#     jj git fetch && jj rebase -d <bookmark>@origin      # then re-run safe_push
#
# USAGE
#   scripts/safe_push.sh                       # push @ (or @- if @ is empty) to main
#   scripts/safe_push.sh -r <rev>              # push an explicit revision
#   scripts/safe_push.sh -b <bookmark>         # a bookmark other than main
#   scripts/safe_push.sh --dry-run             # run every check, push nothing
#   scripts/safe_push.sh --self-test           # build a throwaway repo and prove
#                                              #   a sideways target is REFUSED
set -uo pipefail

BOOKMARK=main
REV=""
DRY_RUN=0
SELF_TEST=0

while [ $# -gt 0 ]; do
  case "$1" in
    -b|--bookmark) BOOKMARK="$2"; shift 2 ;;
    -r|--rev)      REV="$2";      shift 2 ;;
    -n|--dry-run)  DRY_RUN=1;     shift ;;
    --self-test)   SELF_TEST=1;   shift ;;
    -h|--help)     sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "safe_push: unknown argument '$1' (see --help)" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------- core

jjq() { jj log --no-graph --ignore-working-copy -r "$1" -T "${2:-commit_id}"; }

do_push() {
  local bookmark="$1" rev="$2" dry="$3"

  echo "safe_push: fetching..."
  jj git fetch || { echo "safe_push: FETCH FAILED — refusing to push blind." >&2; return 4; }

  # Resolve the target. Default: @, or @- when @ is the usual empty working commit.
  if [ -z "$rev" ]; then
    if [ -z "$(jj log --no-graph --ignore-working-copy -r '@ & ~empty()' -T 'commit_id')" ]; then
      rev='@-'
      echo "safe_push: @ is empty; targeting @- instead."
    else
      rev='@'
    fi
  fi

  local target
  target=$(jjq "$rev") || { echo "safe_push: cannot resolve revision '$rev'." >&2; return 2; }
  if [ -z "$target" ]; then
    echo "safe_push: revision '$rev' resolved to nothing." >&2; return 2
  fi

  local remote_rev="${bookmark}@origin"
  local remote_tip
  remote_tip=$(jjq "$remote_rev" 2>/dev/null || true)

  if [ -z "$remote_tip" ]; then
    echo "safe_push: ${remote_rev} does not exist yet — treating this as a NEW bookmark."
  else
    # THE GATE. `::A ~ ::B` is the set of commits reachable from the remote tip
    # but NOT from the target — exactly what this push would drop. It is empty
    # if and only if the remote tip is an ancestor of the target, so the same
    # expression is both the test and the diagnostic. (`A ~ ::B` would be a
    # correct test too, but names only the tip and hides the rest of the loss.)
    local dropped
    dropped=$(jj log --no-graph --ignore-working-copy \
                -r "::${remote_rev} ~ ::${target}" \
                -T 'commit_id.short() ++ " " ++ description.first_line() ++ "\n"' 2>/dev/null || true)
    if [ -n "$dropped" ]; then
      echo "" >&2
      echo "safe_push: REFUSED — this push is NOT a fast-forward." >&2
      echo "  bookmark : ${bookmark}" >&2
      echo "  target   : ${target}" >&2
      echo "  ${remote_rev} : ${remote_tip}" >&2
      echo "" >&2
      echo "  ${remote_rev} is NOT an ancestor of the target. Pushing would make these" >&2
      echo "  commits unreachable from ${bookmark} (they would NOT be deleted, but nothing" >&2
      echo "  would point at them and the next lane to regenerate from ${bookmark} would" >&2
      echo "  silently lose their content):" >&2
      echo "" >&2
      printf '%s\n' "$dropped" | head -40 | sed 's/^/      /' >&2
      local n_dropped; n_dropped=$(printf '%s\n' "$dropped" | grep -c .)
      [ "$n_dropped" -gt 40 ] && echo "      ... and $((n_dropped - 40)) more ($n_dropped total)" >&2
      echo "" >&2
      echo "  Fix it, do not force it:" >&2
      echo "      jj git fetch && jj rebase -d ${remote_rev}" >&2
      echo "      # resolve any conflicts in @, keeping BOTH lanes' hunks, then re-run:" >&2
      echo "      scripts/safe_push.sh -b ${bookmark}" >&2
      echo "" >&2
      return 3
    fi
    echo "safe_push: OK — ${remote_rev} (${remote_tip:0:12}) is an ancestor of ${target:0:12}."
  fi

  if [ "$dry" = 1 ]; then
    echo "safe_push: --dry-run, stopping before 'bookmark set'."
    return 0
  fi

  jj bookmark set "$bookmark" -r "$target" || return 5
  jj git push --bookmark "$bookmark" || {
    echo "safe_push: PUSH FAILED. The bookmark was moved locally but not published;" >&2
    echo "  re-run after 'jj git fetch && jj rebase -d ${remote_rev}'." >&2
    return 5
  }

  # VERIFY. A push that reports success but does not land is the failure mode the
  # 2026-05-29 orphaned-bookmark incident taught us to check for explicitly.
  jj git fetch >/dev/null 2>&1
  remote_tip=$(jjq "$remote_rev" 2>/dev/null || true)
  local unlanded
  unlanded=$(jj log --no-graph --ignore-working-copy -r "${target} ~ ::${remote_rev}" -T 'commit_id' 2>/dev/null || true)
  if [ -n "$unlanded" ]; then
    echo "safe_push: VERIFY FAILED — ${target:0:12} is NOT an ancestor of ${remote_rev} after the push." >&2
    return 6
  fi
  echo "safe_push: VERIFIED — ${target:0:12} is on ${remote_rev} (now ${remote_tip:0:12})."
  return 0
}

# ---------------------------------------------------------------- self-test

self_test() {
  local rc fails=0
  # SP_TMP is intentionally GLOBAL: the EXIT trap fires after self_test's locals
  # have gone out of scope, so a `local tmp` would be unbound there under `set -u`.
  SP_TMP=$(mktemp -d "${TMPDIR:-$HOME/tmp}/safe_push_selftest.XXXXXX") || return 1
  trap 'rm -rf "$SP_TMP"' EXIT
  local tmp="$SP_TMP"

  local script; script=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")

  git init --bare -q "$tmp/remote.git" || return 1
  jj git clone --quiet "$tmp/remote.git" "$tmp/work" >/dev/null 2>&1 || return 1
  cd "$tmp/work" || return 1
  jj config set --repo user.name "safe-push selftest" >/dev/null 2>&1
  jj config set --repo user.email "selftest@example.invalid" >/dev/null 2>&1

  # base commit, published
  echo base > f.txt
  jj describe -m base >/dev/null 2>&1
  jj bookmark create main -r @ >/dev/null 2>&1 || jj bookmark set main -r @ >/dev/null 2>&1
  jj git push --bookmark main >/dev/null 2>&1 || return 1
  local base; base=$(jj log --no-graph --ignore-working-copy -r 'main@origin' -T 'commit_id')

  # --- CASE 1: a fast-forward target is ACCEPTED -------------------------
  jj new "main@origin" -m "ff-child" >/dev/null 2>&1
  echo ff > g.txt
  # Capture the target BEFORE the push: a successful `jj git push` makes the
  # working-copy commit immutable and creates a fresh empty `@` on top of it,
  # so reading `@` afterwards would read the wrong commit.
  jj status >/dev/null 2>&1   # force a snapshot so g.txt is in the commit
  local ff_tip; ff_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  "$script" -r "$ff_tip" >"$tmp/case1.log" 2>&1; rc=$?
  if [ "$rc" -eq 0 ] && [ -z "$(jj log --no-graph --ignore-working-copy -r "${ff_tip} ~ ::main@origin" -T 'commit_id')" ]; then
    echo "  CASE 1 fast-forward ACCEPTED and landed .......... PASS"
  else
    echo "  CASE 1 fast-forward ACCEPTED and landed .......... FAIL (rc=$rc)"; sed 's/^/      /' "$tmp/case1.log"; fails=$((fails+1))
  fi

  # --- CASE 2: a SIDEWAYS target is REFUSED ------------------------------
  # A second lane's commit, built on the OLD base, ignorant of ff-child.
  jj new "$base" -m "sideways-lane" >/dev/null 2>&1
  echo sideways > h.txt
  jj status >/dev/null 2>&1
  local side_tip; side_tip=$(jj log --no-graph --ignore-working-copy -r '@' -T 'commit_id')
  local before; before=$(jj log --no-graph --ignore-working-copy -r 'main@origin' -T 'commit_id')
  "$script" -r "$side_tip" >"$tmp/case2.log" 2>&1; rc=$?
  local after; after=$(jj log --no-graph --ignore-working-copy -r 'main@origin' -T 'commit_id')
  if [ "$rc" -eq 3 ] && [ "$before" = "$after" ]; then
    echo "  CASE 2 sideways REFUSED (rc=3), remote UNMOVED ... PASS"
  else
    echo "  CASE 2 sideways REFUSED (rc=3), remote UNMOVED ... FAIL (rc=$rc, before=${before:0:12} after=${after:0:12})"
    sed 's/^/      /' "$tmp/case2.log"; fails=$((fails+1))
  fi

  # --- CASE 3: the refusal NAMES the commit that would be dropped --------
  if grep -q "ff-child" "$tmp/case2.log"; then
    echo "  CASE 3 refusal names the dropped commit .......... PASS"
  else
    echo "  CASE 3 refusal names the dropped commit .......... FAIL"; fails=$((fails+1))
  fi

  # --- CASE 4: after rebasing onto the remote tip, it is ACCEPTED --------
  jj rebase -r "$side_tip" -d 'main@origin' >/dev/null 2>&1
  local fixed; fixed=$(jj log --no-graph --ignore-working-copy -r 'description(glob:"sideways-lane*")' -T 'commit_id')
  "$script" -r "$fixed" >"$tmp/case4.log" 2>&1; rc=$?
  # Assert by DESCRIPTION, not by the pre-rebase commit id: `jj rebase` rewrites
  # the commit, and jj resolves an obsolete id to its successor, so a stale id
  # would compare the wrong object.
  local landed_side landed_ff
  landed_side=$(jj log --no-graph --ignore-working-copy -r 'description(glob:"sideways-lane*") & ::main@origin' -T 'commit_id')
  landed_ff=$(jj log --no-graph --ignore-working-copy -r 'description(glob:"ff-child*") & ::main@origin' -T 'commit_id')
  local miss_fixed="" miss_ff=""
  [ -z "$landed_side" ] && miss_fixed="sideways-lane"
  [ -z "$landed_ff" ]   && miss_ff="ff-child"
  if [ "$rc" -eq 0 ] && [ -n "$landed_side" ] && [ -n "$landed_ff" ]; then
    echo "  CASE 4 rebase-then-push lands BOTH lanes ......... PASS"
  else
    echo "  CASE 4 rebase-then-push lands BOTH lanes ......... FAIL (rc=$rc not_landed='${miss_fixed} ${miss_ff}')"
    sed 's/^/      /' "$tmp/case4.log"; fails=$((fails+1))
  fi

  echo ""
  if [ "$fails" -eq 0 ]; then
    echo "self-test safe_push: PASS (0 failure(s))"; return 0
  fi
  echo "self-test safe_push: FAIL ($fails failure(s))"; return 1
}

if [ "$SELF_TEST" = 1 ]; then
  self_test; exit $?
fi

do_push "$BOOKMARK" "$REV" "$DRY_RUN"; exit $?
