#!/usr/bin/env bash
# Workstation-side helper for zensim Hetzner train boxes (docs/DATA_SPLITS.md §6).
# Cache-friendly by design: launch fire-and-forget, poll with ONE tiny status
# fetch, pull results in one rsync. Never ships root R2 keys (scoped temp creds).
#
#   hz.sh provision <name>              create ccx63 + print IP
#   hz.sh bootstrap <ip> <zensim-commit>  scoped-creds + bootstrap (nohup, on box)
#   hz.sh push-eval <ip>                rsync eval features + grids + TV pairs
#   hz.sh push-manifests <ip> <m1> [m2..]  rsync manifests + write cells.txt
#   hz.sh run <ip> [PAR]                start runcells.sh under nohup
#   hz.sh status <ip>                   cat /data/out/status.tsv (tiny)
#   hz.sh pull <ip>                     rsync /data/out -> probe dir hetzner-out/
#   hz.sh destroy <name>                delete the server (results pulled first!)
set -euo pipefail
CMD="${1:?cmd}"; shift
export HCLOUD_TOKEN=$(grep -oP 'api_token=\K.*' ~/.config/hetzner/credentials)
SSH="ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new root@"
PROBE=/mnt/v/output/zensim-multicodec-probe

mint_scoped_ro() { # scoped read-only zentrain creds, 48h TTL
  set -a; . ~/.config/cloudflare/r2-credentials; set +a
  body=$(python3 -c "import json,os;print(json.dumps({'bucket':'zentrain','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-only','ttlSeconds':172800}))")
  curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" \
    -d "$body" "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" \
    | python3 -c "import json,sys;r=json.load(sys.stdin)['result'];print(f\"export AWS_ACCESS_KEY_ID={r['accessKeyId']} AWS_SECRET_ACCESS_KEY={r['secretAccessKey']} AWS_SESSION_TOKEN={r['sessionToken']} R2_ENDPOINT=https://{'$R2_ACCOUNT_ID'}.r2.cloudflarestorage.com\")" \
    | sed "s|\$R2_ACCOUNT_ID|$R2_ACCOUNT_ID|"
}

case "$CMD" in
  provision)
    NAME="${1:?name}"
    trap 'NEWIP=$(hcloud server ip "$NAME" 2>/dev/null); [ -n "$NEWIP" ] && ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$NEWIP" >/dev/null 2>&1 || true' EXIT
    hcloud server create --name "$NAME" --type ccx63 --image ubuntu-24.04 \
      --location fsn1 --ssh-key zen-arm-dev-20260528 -o columns=ipv4 | tail -1
    ;;
  bootstrap)
    IP="${1:?ip}"; COMMIT="${2:?zensim commit}"
    CREDS=$(mint_scoped_ro)
    scp -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new \
      "$(dirname "$0")/bootstrap_trainbox.sh" "$(dirname "$0")/rebuild_derived.py" \
      "$(dirname "$0")/runcells.sh" root@"$IP":/root/
    $SSH"$IP" "mkdir -p /root/scripts-hetzner && mv /root/rebuild_derived.py /root/scripts-hetzner/ 2>/dev/null; \
      $CREDS; export ZENSIM_COMMIT=$COMMIT; nohup bash /root/bootstrap_trainbox.sh > /root/bootstrap.log 2>&1 & echo bootstrap-launched"
    ;;
  push-eval)
    IP="${1:?ip}"
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new" /mnt/v/zen/zensim-training/2026-05-15-full-features/ root@"$IP":/data/evalfeat/
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new" "$PROBE"/kadis_test_safetygrid.parquet "$PROBE"/hq_codec_grid_2026-07-01.parquet \
      "$PROBE"/hq_codec_refs_2026-07-01.parquet "$PROBE"/kadis_tv_pairs_clean.tsv root@"$IP":/data/grids/
    ;;
  push-manifests)
    IP="${1:?ip}"; shift
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new" /home/lilith/work/zen/zensim/zensim/weights/manifests/ root@"$IP":/root/work/zensim/zensim/weights/manifests/
    printf '%s\n' "$@" | $SSH"$IP" "cat > /root/cells.txt && wc -l /root/cells.txt"
    ;;
  run)
    IP="${1:?ip}"; PAR="${2:-6}"
    $SSH"$IP" "grep -q 'boot] DONE' /root/bootstrap.log"       || { echo "REFUSED: bootstrap not DONE on $IP (hz.sh status $IP)" >&2; exit 1; }
    $SSH"$IP" "nohup bash /root/work/zensim/scripts/hetzner/runcells.sh /root/cells.txt $PAR > /data/out/runcells.log 2>&1 & echo run-launched"
    ;;
  status)  $SSH"${1:?ip}" "cat /data/out/status.tsv 2>/dev/null | tail -20; tail -2 /root/bootstrap.log 2>/dev/null" ;;
  pull)    IP="${1:?ip}"; mkdir -p "$PROBE/hetzner-out"
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new" root@"$IP":/data/out/ "$PROBE/hetzner-out/"
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev -o StrictHostKeyChecking=accept-new" root@"$IP":/data/derived/*.bin "$PROBE/" 2>/dev/null
    # THE RULE (2026-07-02): every pulled result gets a visual report in
    # /mnt/v/output/zensim/reports/ (viewer index regenerates each run).
    for B in "$PROBE"/*.bin; do
      L=$(basename "$B" .bin)
      [ -d "/mnt/v/output/zensim/reports/$(date +%F)_$L" ] ||         nice -n19 python3 "$(dirname "$0")/../v_next/bake_report.py" --bake "$B" --label "$L" 2>&1 | tail -1
    done
    echo pulled ;;
  autoretire) # detached watcher: pull + retire when work completes OR box idles
    # usage: hz.sh autoretire <ip> <name> [idle_min=25]
    # Boxes bill by the STARTED hour (user 2026-07-03): after work completes
    # the box stays reusable through the paid hour and retires at minute >=45
    # (uptime mod 60). Runs ON THE WORKSTATION (hcloud token never leaves).
    # Veto: touch $PROBE/.box_hold. Watcher is a real script file (inline
    # quoting mangled variables in the first version).
    IP="${1:?ip}"; NAME="${2:?server name}"; IDLE_MIN="${3:-25}"
    PROBE=/mnt/v/output/zensim-multicodec-probe
    W="$PROBE/autoretire_${NAME}.sh"
    REPO="$(cd "$(dirname "$0")/../.." && pwd)"
    cat > "$W" <<'WATCH'
#!/usr/bin/env bash
IP="$1"; NAME="$2"; IDLE_MIN="$3"; REPO="$4"
PROBE=/mnt/v/output/zensim-multicodec-probe
SSHC="ssh -i $HOME/.ssh/zen-arm-dev -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new"
idle=0
while true; do
  sleep 180
  [ -f "$PROBE/.box_hold" ] && { idle=0; continue; }
  TRAINERS=$($SSHC root@"$IP" "ps -e | grep -c zensim_mlp" 2>/dev/null)
  UNITS=$($SSHC root@"$IP" "systemctl list-units --state=active --plain --no-legend 'wave*' 'ablation*' 2>/dev/null | wc -l" 2>/dev/null)
  if [ -z "$TRAINERS" ]; then idle=$((idle+3))
  elif [ "${TRAINERS:-0}" -gt 0 ] || [ "${UNITS:-0}" -gt 0 ]; then idle=0; continue
  else idle=$((idle+3)); fi
  DONE=$($SSHC root@"$IP" "grep -c ALLDONE /data/out/status.tsv 2>/dev/null" 2>/dev/null)
  if [ "${DONE:-0}" -ge 1 ] || [ "$idle" -ge "$IDLE_MIN" ]; then
    UPMIN=$($SSHC root@"$IP" "awk '{print int(\$1/60)}' /proc/uptime" 2>/dev/null)
    if [ -n "$UPMIN" ] && [ $((UPMIN % 60)) -lt 45 ]; then
      echo "[autoretire] $(date -u +%FT%TZ) work done (done=${DONE:-0} idle=${idle}m) — paid-hour reuse window open (uptime ${UPMIN}m; retire at min>=45)"
      continue
    fi
    echo "[autoretire] $(date -u +%FT%TZ) pulling + retiring $NAME (done=${DONE:-0} idle=${idle}m uptime=${UPMIN:-?}m)"
    cd "$REPO"
    bash scripts/hetzner/hz.sh pull "$IP" 2>&1 | tail -1
    rsync -az -e "ssh -i $HOME/.ssh/zen-arm-dev" root@"$IP":/data/derived/*.bin "$PROBE/" 2>/dev/null
    export HCLOUD_TOKEN=$(grep -oP "api_token=\K.*" "$HOME/.config/hetzner/credentials")
    bash scripts/hetzner/hz.sh retire "$NAME" --force 2>&1 | tail -2
    exit 0
  fi
done
WATCH
    chmod +x "$W"
    nohup "$W" "$IP" "$NAME" "$IDLE_MIN" "$REPO" >> "$PROBE/autoretire_${NAME}.log" 2>&1 &
    echo "autoretire watcher armed for $NAME ($IP, idle=${IDLE_MIN}m; veto: touch $PROBE/.box_hold)"
    ;;
  retire)  # snapshot then delete — MANDATORY end-state for x86 big boxes
    NAME="${1:?name}"
    if [ "${2:-}" != "--force" ]; then
      RIP=$(hcloud server ip "$NAME" 2>/dev/null || true)
      if [ -n "$RIP" ] && $SSH"$RIP" "test -f /data/out/status.tsv && ! grep -q ALLDONE /data/out/status.tsv" 2>/dev/null; then
        echo "REFUSED: $NAME has a cell queue without ALLDONE — pull/inspect first or retire $NAME --force" >&2; exit 1
      fi
    fi
    SNAP="${NAME}-$(date +%s)"
    hcloud server create-image --type snapshot --description "$SNAP" "$NAME"
    hcloud server delete "$NAME"
    echo "retired $NAME -> snapshot $SNAP"
    ;;
  restore) # recreate from the newest snapshot matching the name
    NAME="${1:?name}"
    # a restored box reuses IPs with a fresh host key — scrub stale entries
    # after create (see the 2026-07-02 BOX-SSH-FAILED incident)
    IMG=$(hcloud image list --type snapshot -o noheader -o columns=id,description | grep "$NAME" | sort -k2 | tail -1 | awk '{print $1}')
    [ -n "$IMG" ] || { echo "no snapshot matching $NAME" >&2; exit 1; }
    trap 'NEWIP=$(hcloud server ip "$NAME" 2>/dev/null); [ -n "$NEWIP" ] && ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$NEWIP" >/dev/null 2>&1 || true' EXIT
    hcloud server create --name "$NAME" --type ccx63 --image "$IMG" --location fsn1 --ssh-key zen-arm-dev-20260528
    ;;
  destroy) hcloud server delete "${1:?name}" ;;
  *) echo "unknown cmd $CMD" >&2; exit 2 ;;
esac
