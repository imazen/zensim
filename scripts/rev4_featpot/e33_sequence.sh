#!/usr/bin/env bash
# E33 jobset sequencer: appends A, C and full in order (e33_launch.py advance), then requires real filler placements
# within 5 minutes of each append. Zero placements stops the sequence for diagnosis. Runs in its own herdr tab.
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'usage: %s PACKET\n' "$0" >&2
    exit 2
fi
packet=$(realpath "$1")
here=$(dirname "$(realpath "$0")")
log="$packet/../launch/sequence.log"
# shellcheck source=/dev/null
. "$HOME/.config/zen/s3env.sh" >/dev/null 2>&1
export EP
exec > >(tee -a "$log") 2>&1
printf 'E33 sequencer start %s\n' "$(date -u +%FT%TZ)"
while true; do
    out=$(python3 "$here/e33_launch.py" advance --packet "$packet" 2>&1) || { printf '%s\nE33 SEQUENCER REFUSED\n' "$out"; exit 1; }
    line=$(printf '%s\n' "$out" | tail -1)
    printf '%s\n' "$line"
    action=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["action"])')
    if [[ $action == none ]]; then
        printf '==== E33 ALL FOUR JOBSETS QUEUED %s\n' "$(date -u +%FT%TZ)"
        break
    fi
    if [[ $action == launched ]]; then
        arm=$(printf '%s' "$line" | python3 -c 'import json,sys; print(json.load(sys.stdin)["next"])')
        sleep 240
        check=$(python3 "$here/e33_launch.py" placements --packet "$packet" --arm "$arm")
        printf '%s\n' "$check"
        if printf '%s' "$check" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin)["started"] > 0 else 1)'; then
            continue
        fi
        printf 'E33 PLACEMENT FAILURE %s: no filler placement within 4 minutes of the append; stopped for diagnosis\n' "$arm"
        exit 1
    fi
    sleep 300
done
