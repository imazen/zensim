#!/usr/bin/env bash
# strategy_fleet.sh — fan strategy-ablation trainer cells across cheap Hetzner
# shared-vCPU boxes (cx53 ≈ €0.05/hr, well under the $0.30 ceiling; user
# directive 2026-07-02). Each box pulls the prebuilt trainer binary + pinned
# inputs from R2 (glibc matches: workstation and hetzner are both ubuntu-24.04
# / glibc 2.39), runs its assigned manifests serially, pushes each bake +
# train log to the run's results prefix, then idles (reap from the
# workstation — the hcloud token never leaves this machine).
#
#   usage: strategy_fleet.sh launch <run_id> <cells.txt> <N_boxes>
#          strategy_fleet.sh status <run_id> <cells.txt>
#          strategy_fleet.sh reap   <run_id>            # delete all run boxes
#
# Inputs expected on R2 (upload once):
#   s3://zentrain/strategy-fleet-2026-07-02/bin/zensim_mlp_train
#   s3://zentrain/strategy-fleet-2026-07-02/derived/*   (pinned parquets etc.)
#   s3://zentrain/strategy-fleet-2026-07-02/manifests/* (uploaded at launch)
#   s3://zentrain/strategy-fleet-2026-07-02/benchmarks/* (mask + transforms tsv)
#   s3://zentrain/canonical-2026-05-21/train/*           (already canonical)
set -uo pipefail
CMD="${1:?launch|status|reap}"; RUN="${2:?run id}"
FLEET_PREFIX="strategy-fleet-2026-07-02"
RESULTS_PREFIX="strategy-results/$RUN"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

case "$CMD" in
launch)
  CELLS_FILE="${3:?cells file}"; N="${4:?n boxes}"
  # upload manifests + benchmark sidecars fresh (tiny)
  cd "$(dirname "$0")/../.."
  for m in $(cat "$CELLS_FILE"); do
    r2 cp "zensim/weights/manifests/${m}.toml" "s3://zentrain/$FLEET_PREFIX/manifests/${m}.toml" >/dev/null
  done
  r2 cp benchmarks/feature_sign_mask_2026-05-26.tsv "s3://zentrain/$FLEET_PREFIX/benchmarks/feature_sign_mask_2026-05-26.tsv" >/dev/null
  r2 cp benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv "s3://zentrain/$FLEET_PREFIX/benchmarks/screen_results_cross_corpus_safe.tsv" >/dev/null
  # scoped 24h creds: read fleet inputs + canonical, write results
  body=$(python3 -c "import json,os;print(json.dumps({'bucket':'zentrain','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':86400,'prefixes':['$FLEET_PREFIX/','canonical-2026-05-21/','$RESULTS_PREFIX/']}))")
  curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d "$body" \
    "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials" > /tmp/sf_fleet_cred.json
  read -r AK SK ST < <(python3 -c 'import json;r=json.load(open("/tmp/sf_fleet_cred.json"))["result"];print(r["accessKeyId"],r["secretAccessKey"],r["sessionToken"])')
  [ -n "${AK:-}" ] || { echo "cred mint failed"; cat /tmp/sf_fleet_cred.json; exit 1; }
  mapfile -t CELLS < "$CELLS_FILE"
  for i in $(seq 0 $((N-1))); do
    ASSIGNED=""
    for j in $(seq 0 $((${#CELLS[@]}-1))); do
      [ $((j % N)) -eq "$i" ] && ASSIGNED="$ASSIGNED ${CELLS[$j]}"
    done
    [ -z "$ASSIGNED" ] && continue
    ci=$(mktemp)
    cat > "$ci" <<EOF
#cloud-config
runcmd:
  - |
    set -x
    exec > /root/worker.log 2>&1
    curl -sSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
    export AWS_ACCESS_KEY_ID='$AK' AWS_SECRET_ACCESS_KEY='$SK' AWS_SESSION_TOKEN='$ST' AWS_REGION=auto
    S5="s5cmd --endpoint-url $EP"
    mkdir -p /data/derived /data/canonical-2026-05-21/train /root/manifests /root/benchmarks
    \$S5 cp "s3://zentrain/$FLEET_PREFIX/bin/zensim_mlp_train" /usr/local/bin/zensim_mlp_train && chmod +x /usr/local/bin/zensim_mlp_train
    \$S5 cp "s3://zentrain/$FLEET_PREFIX/derived/*" /data/derived/
    \$S5 cp "s3://zentrain/canonical-2026-05-21/train/*" /data/canonical-2026-05-21/train/
    \$S5 cp "s3://zentrain/$FLEET_PREFIX/manifests/*" /root/manifests/
    \$S5 cp "s3://zentrain/$FLEET_PREFIX/benchmarks/*" /root/benchmarks/
    mkdir -p /mnt/v/zen/zensim-training /mnt/v/output
    ln -sfn /data/canonical-2026-05-21 /mnt/v/zen/zensim-training/canonical-2026-05-21
    ln -sfn /data/derived /mnt/v/output/zensim-multicodec-probe
    # manifests reference benchmarks/ relative to the repo root three levels up
    mkdir -p /root/repo/zensim/weights/manifests
    cp /root/manifests/*.toml /root/repo/zensim/weights/manifests/
    cp -r /root/benchmarks /root/repo/benchmarks
    mv /root/repo/benchmarks/screen_results_cross_corpus_safe.tsv /root/repo/benchmarks/screen_results_tmp.tsv 2>/dev/null || true
    mkdir -p /root/repo/benchmarks/yeo_johnson_screen_widest_2026-05-25
    mv /root/repo/benchmarks/screen_results_tmp.tsv /root/repo/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv 2>/dev/null || true
    cd /root/repo/zensim/weights/manifests
    for m in$ASSIGNED; do
      echo "=== cell \$m ==="
      ( cd /root/repo && ZENSIM_ALLOW_TRAINER_DRIFT=1 zensim_mlp_train --manifest "zensim/weights/manifests/\$m.toml" --manifest-allow-sha-drift > "/root/\$m.train.log" 2>&1 )
      rc=\$?
      BAKE=\$(grep -oE 'file *= *"[^"]+"' "\$m.toml" | head -1 | grep -oE '/[^"]+')
      [ -f "\$BAKE" ] && \$S5 cp "\$BAKE" "s3://zentrain/$RESULTS_PREFIX/\$m/\$(basename \$BAKE)"
      \$S5 cp "/root/\$m.train.log" "s3://zentrain/$RESULTS_PREFIX/\$m/train.log"
      echo "rc=\$rc" > /root/done.txt && \$S5 cp /root/done.txt "s3://zentrain/$RESULTS_PREFIX/\$m/done.txt"
    done
    echo "BOX-ALL-CELLS-DONE"
EOF
    name="sfb-$RUN-$i"
    for typ in cx53 cx43 cpx62; do
      for loc in fsn1 nbg1 hel1; do
        hcloud server create --name "$name" --type "$typ" --image ubuntu-24.04 --location "$loc" \
          --ssh-key "$SSH_KEY" --label group="sf-$RUN" --user-data-from-file "$ci" >/dev/null 2>&1 \
          && { echo "$name launched ($typ/$loc):$ASSIGNED"; break 2; }
      done
    done
    rm -f "$ci"
  done
  echo "### fleet launched. status: $0 status $RUN $CELLS_FILE ; reap: $0 reap $RUN"
  ;;
status)
  CELLS_FILE="${3:?cells file}"
  for m in $(cat "$CELLS_FILE"); do
    D=$(r2 cat "s3://zentrain/$RESULTS_PREFIX/$m/done.txt" 2>/dev/null | tr -d '\n')
    echo "$m: ${D:-pending}"
  done
  ;;
reap)
  hcloud server list -l group="sf-$RUN" -o noheader | awk '{print $2}' | xargs -r -n1 hcloud server delete >/dev/null && echo reaped
  ;;
esac
