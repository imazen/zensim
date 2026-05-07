#!/bin/bash
#
# Launch a vast.ai zen-metrics sweep for one of v16{w,a,j} (cross-codec).
#
# Adapts /home/lilith/work/zen/zenmetrics/scripts/sweep/v15/launch_gpu.sh for
# our v_next plan TODO §4.2.
#
# Prerequisites (one-time):
#   1. v16 chunks generated:
#        python3 scripts/v_next/generate_v16_chunks.py
#   2. Chunk JSONL files uploaded to:
#        s3://coefficient/jobs/sweep-v16{w,a,j}-2026-05-07/chunks.jsonl
#   3. Source corpus mirrored to:
#        s3://zentrain/sweep-v16{w,a,j}-2026-05-07/sources/
#   4. R2 credentials in env (~/.config/cloudflare/r2-env.sh)
#
# Usage:
#   bash scripts/v_next/launch_v16_sweep.sh w   # zenwebp,  ~330k cells
#   bash scripts/v_next/launch_v16_sweep.sh a   # zenavif,   ~41k cells
#   bash scripts/v_next/launch_v16_sweep.sh j   # zenjxl,   ~110k cells
#
# Env knobs (defaults shown):
#   N_BOXES=20   MAX_DPH=0.20   MIN_CORES=8   MIN_RAM_GB=12   MIN_DISK_GB=25
#   IMAGE=ghcr.io/imazen/zen-metrics-sweep:0.6.3
#   SWEEP_BIN=s3://coefficient/binaries/zen-metrics-0.6.8-linux-x86_64-gpu
#   DRY_RUN=1   # only print picked offers, don't actually launch
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 {w|a|j}" >&2
  exit 1
fi

CODEC_TAG="$1"
case "$CODEC_TAG" in
  w) CODEC=zenwebp ;;
  a) CODEC=zenavif ;;
  j) CODEC=zenjxl  ;;
  *) echo "unknown codec tag '$CODEC_TAG' (expected w|a|j)" >&2; exit 1 ;;
esac
SWEEP_RUN_ID="sweep-v16${CODEC_TAG}-2026-05-07"

# Source R2 credentials.
if [ -f ~/.config/cloudflare/r2-env.sh ]; then
  source ~/.config/cloudflare/r2-env.sh
elif [ -f ~/.config/cloudflare/r2-credentials ]; then
  source ~/.config/cloudflare/r2-credentials
else
  echo "ERROR: ~/.config/cloudflare/r2-env.sh not found" >&2
  exit 1
fi

IMAGE="${IMAGE:-ghcr.io/imazen/zen-metrics-sweep:0.6.3}"
N_BOXES="${N_BOXES:-20}"
MAX_DPH="${MAX_DPH:-0.20}"
MIN_CORES="${MIN_CORES:-8}"
MIN_RAM_GB="${MIN_RAM_GB:-12}"
MIN_DISK_GB="${MIN_DISK_GB:-25}"
SWEEP_BIN="${SWEEP_BIN:-s3://coefficient/binaries/zen-metrics-0.6.8-linux-x86_64-gpu}"
DRY_RUN="${DRY_RUN:-0}"

GHCR_TOKEN="$(gh auth token)"
GHCR_USER="lilithriver"

echo "[v16/$CODEC] run_id=$SWEEP_RUN_ID"
echo "[v16/$CODEC] image=$IMAGE  sweep-bin=$SWEEP_BIN"
echo "[v16/$CODEC] target $N_BOXES boxes @ <\$$MAX_DPH/hr"

QUERY="rentable=true reliability>0.95 dph_total<${MAX_DPH} cpu_cores>=${MIN_CORES} cpu_ram>=${MIN_RAM_GB} disk_space>${MIN_DISK_GB} cuda_max_good>=12 num_gpus=1"
echo "[v16/$CODEC] querying: $QUERY"
OFFERS_JSON=$(vastai search offers "$QUERY" --order 'dph_total' --raw)
OFFER_IDS=$(echo "$OFFERS_JSON" | python3 -c "
import json, sys, os
d = json.loads(sys.stdin.read())
offers = d if isinstance(d, list) else d.get('offers', [])
seen, picked = set(), []
for o in offers:
    mid = o.get('machine_id')
    if mid in seen: continue
    seen.add(mid)
    picked.append(str(o['id']))
    if len(picked) >= int(os.environ.get('N_BOXES', '20')): break
print('\n'.join(picked))
")
n=$(echo "$OFFER_IDS" | wc -w)
echo "[v16/$CODEC] picked $n distinct offers (need $N_BOXES)"
if [[ "$DRY_RUN" == "1" ]]; then echo "$OFFER_IDS" | head -10; exit 0; fi
[[ "$n" -lt 5 ]] && { echo "Not enough offers; relax filters." >&2; exit 1; }

INSTANCE_LOG="/mnt/v/zen/zensim-training/2026-05-07/logs/${SWEEP_RUN_ID}.instances.txt"
mkdir -p "$(dirname "$INSTANCE_LOG")"
> "$INSTANCE_LOG"
i=0
for offer_id in $OFFER_IDS; do
    i=$((i+1))
    WORKER_ID="${SWEEP_RUN_ID}-w${i}"
    LABEL="zen-v16${CODEC_TAG}-${i}"
    ENV_STR="-e SWEEP_BIN_OVERRIDE=${SWEEP_BIN} -e R2_ACCOUNT_ID=$R2_ACCOUNT_ID -e R2_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID -e R2_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY -e SWEEP_RUN_ID=$SWEEP_RUN_ID -e WORKER_ID=$WORKER_ID -e SWEEP_GPU_RUNTIME=cuda"
    LOGIN_STR="-u ${GHCR_USER} -p ${GHCR_TOKEN} ghcr.io"
    OUT=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --login "$LOGIN_STR" \
        --onstart-cmd "/usr/local/bin/zen-metrics-worker" \
        --disk "$MIN_DISK_GB" --label "$LABEL" --env "$ENV_STR" \
        --raw 2>&1) || { echo "  $i fail: $(echo "$OUT" | head -c 200)"; continue; }
    ID=$(echo "$OUT" | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(d.get('new_contract', d.get('id','')))" 2>/dev/null || echo "")
    [[ -z "$ID" ]] && { echo "  $i parse-fail: $(echo "$OUT" | head -c 200)"; continue; }
    echo "$ID $offer_id $WORKER_ID" >> "$INSTANCE_LOG"
    echo "  $i -> instance $ID"
done
echo
echo "[v16/$CODEC] launched $(wc -l < "$INSTANCE_LOG") instances → $INSTANCE_LOG"
echo "[v16/$CODEC] watch progress:"
echo "  vastai show instances"
echo "  aws s3 ls --endpoint-url=https://\${R2_ACCOUNT_ID}.r2.cloudflarestorage.com s3://zentrain/${SWEEP_RUN_ID}/${CODEC}/"
echo "  python3 ~/work/zen/zenmetrics/scripts/sweep/sweep_diag.py ${SWEEP_RUN_ID}"
