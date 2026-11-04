#!/usr/bin/env bash
# Enforce CPU placement and the caller's run-heavy memory cap on one job scope.
set -euo pipefail
if test "${1-}" = --verify; then
    shift
    python3 - <<'PY'
import json
import subprocess

result = subprocess.run(["zenbench", "cpu-info"], check=True, capture_output=True, text=True)
print(result.stdout, end="", flush=True)
placement = json.loads(result.stdout)["placement"].get("Ok")
if placement is None or placement["status"] != "enforced_uniform":
    raise SystemExit("CPU boundary was not verified; command was not started")
PY
    exec "$@"
fi
if test "$#" -lt 2; then
    echo 'usage: profile_test_cpuset.sh CPU_IDS COMMAND [ARGS...]' >&2
    exit 2
fi
profile_cpu_ids=$1
shift
profile_scope=$(awk -F/ '$1 == "0::" {print $NF}' /proc/self/cgroup)
case "$profile_scope" in
    run-*.scope) ;;
    *) echo 'run this command inside run-heavy on systemd Linux' >&2; exit 2 ;;
esac
profile_memory_max=$(systemctl --user show "$profile_scope" --property=MemoryMax --value)
case "$profile_memory_max" in
    ''|*[!0-9]*) echo 'the caller scope must have a finite MemoryMax' >&2; exit 2 ;;
esac
# User managers may lack the cpuset controller. A job-specific system scope can
# enforce it; copy the finite memory limit and run the payload as the caller.
profile_user=$(id -un)
profile_script=$(realpath "$0")
printf 'job MemoryMax=%s CPU_IDS=%s\n' "$profile_memory_max" "$profile_cpu_ids"
exec sudo -n systemd-run --scope --quiet \
    --property="AllowedCPUs=$profile_cpu_ids" \
    --property="MemoryMax=$profile_memory_max" --property=MemorySwapMax=0 \
    /usr/sbin/runuser -u "$profile_user" -- /usr/bin/env \
    PATH="$PATH" TMPDIR="${TMPDIR:?set TMPDIR to a disk path}" \
    CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:?run through run-heavy}" \
    RAYON_NUM_THREADS="${RAYON_NUM_THREADS:?run through run-heavy}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:?run through run-heavy}" \
    ZENBENCH_CPU_SET="$profile_cpu_ids" ZENBENCH_CPU_POLICY=auto \
    /bin/bash "$profile_script" --verify "$@"
