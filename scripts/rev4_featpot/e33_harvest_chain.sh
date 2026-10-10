#!/usr/bin/env bash
# E33 harvest chain: verify and install every DONE cell of each queued E33 jobset until all 126 are installed.
# Never schedules work. Runs in its own herdr tab; heartbeat and results go to <packet>/../launch/harvest-chain.log.
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'usage: %s PACKET\n' "$0" >&2
    exit 2
fi
packet=$(realpath "$1")
here=$(dirname "$(realpath "$0")")
log="$packet/../launch/harvest-chain.log"
export TMPDIR="${TMPDIR:-$HOME/tmp/e33}"
mkdir -p "$TMPDIR"
python3 "$here/e33_launch.py" verify --packet "$packet" > /dev/null
# The credential file is supplied by the host, outside the source tree.
# shellcheck source=/dev/null
. "$HOME/.config/zen/s3env.sh" >/dev/null 2>&1
export EP
exec > >(tee -a "$log") 2>&1
printf 'E33 harvest chain start %s packet %s\n' "$(date -u +%FT%TZ)" "$packet"
while true; do
    complete=0
    for arm in control a c full; do
        js="fite33-$arm-20261010"
        if ! grep -q "^$js " /var/tmp/fitv2/fleet_queue; then
            continue
        fi
        if ! out=$(nice -n19 python3 "$here/e33_launch.py" harvest --packet "$packet" --arm "$arm" --install 2>&1); then
            printf '%s\nE33 HARVEST FAILED %s %s\n' "$out" "$js" "$(date -u +%FT%TZ)"
            exit 1
        fi
        line=$(printf '%s\n' "$out" | grep -m1 '^{"jobset"' || true)
        printf '%s %s\n' "$(date -u +%FT%TZ)" "$line"
        if printf '%s' "$line" | python3 -c 'import json,sys; r=json.load(sys.stdin); sys.exit(0 if r["installed"]==r["total"] else 1)'; then
            complete=$((complete + 1))
        fi
    done
    if [[ $complete == 4 ]]; then
        for arm in control a c full; do
            python3 "$here/e33_launch.py" harvest --packet "$packet" --arm "$arm" --install --require-all
        done
        printf '==== E33 ALL 126 CELLS VERIFIED AND INSTALLED %s\n' "$(date -u +%FT%TZ)"
        break
    fi
    sleep 300
done
