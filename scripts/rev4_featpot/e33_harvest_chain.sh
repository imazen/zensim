#!/usr/bin/env bash
# E33 harvest chain: verify and install every DONE cell of each queued E33 jobset until all 126 are installed.
# Never schedules work. Runs in its own herdr tab; heartbeat and results go to <packet>/../launch/harvest-chain.log.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 3 ]]; then
    printf 'usage: %s PACKET [E34_PACKET_SHA E34_QUEUE]\n' "$0" >&2
    exit 2
fi
packet=$(realpath "$1")
here=$(dirname "$(realpath "$0")")
log="$packet/../launch/harvest-chain.log"
export TMPDIR="${TMPDIR:-$HOME/tmp/e33}"
mkdir -p "$TMPDIR"
campaign=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["schema"])' "$packet/PACKET.json")
if [[ $campaign == e34-fleet-packet-v1 ]]; then
    [[ $# == 3 ]] || { echo 'E34 requires its frozen packet SHA and queue'; exit 2; }
    pin=$2
    queue=$3
    python3 "$here/e33_launch.py" verify --packet "$packet" --packet-sha "$pin" >/dev/null
    . "$HOME/.config/zen/s3env.sh" >/dev/null 2>&1
    export EP
    exec > >(tee -a "$log") 2>&1
    while true; do
        complete=0
        while read -r js manifest image; do
            [[ -n $js ]] || continue
            out=$(nice -n19 ionice -c3 python3 "$here/e33_launch.py" harvest --packet "$packet" --packet-sha "$pin" --arm "$js" --install) || { echo "E34 HARVEST FAILED $js"; exit 1; }
            line=$(printf '%s\n' "$out" | grep -m1 '^{"jobset"')
            printf '%s %s\n' "$(date -u +%FT%TZ)" "$line"
            n=$(printf '%s' "$line" | python3 -c 'import json,sys;r=json.load(sys.stdin);print(r["installed"])')
            complete=$((complete+n))
        done < "$queue"
        if [[ $complete == 305 ]]; then
            echo "E34 ALL305 REGISTERED CELLS VERIFIED AND INSTALLED $(date -u +%FT%TZ)"
            exit 0
        fi
        echo "E34 installed $complete/305; conditional branches remain held until registered pilot selection"
        sleep 60
    done
fi
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
