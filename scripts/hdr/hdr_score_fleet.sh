#!/usr/bin/env bash
# hdr_score_fleet.sh — score an HDR datagen's (ref .hdr.png, dist .jxl) pairs
# across cheap cx boxes (the fleet norm; local GPU runs are the exception, not
# the rule — user directive 2026-07-03). Mirrors strategy_fleet.sh's proven
# pattern: plain ubuntu VMs, scoped R2 creds, per-box chunk of the pairs list,
# self-reported sidecars; reaped from the workstation.
#
# Each box: pulls the CPU-only HDR binary (zentrain/hdr/bin/), the datagen's
# ref/ + variants.tar, extracts, rewrites pairs paths, then per metric runs
#   zenmetrics score-pairs --hdr --hdr-transfer pu-rescale
# with --feature-output on the zensim pass (the 372 PU21 features). The
# first-pair NaN gate from datagen_score_hdr.sh is applied per metric.
#
#   usage: hdr_score_fleet.sh launch <run_id> <datagen_prefix> <N_boxes> [metrics]
#          hdr_score_fleet.sh status <run_id> <N_boxes>
#          hdr_score_fleet.sh reap   <run_id>
#   e.g.:  hdr_score_fleet.sh launch hdrhq1 picker-sweep-2026-06-22/datagen-2026-07-03-hdr-hq 6
set -uo pipefail
CMD="${1:?launch|status|reap}"; RUN="${2:?run id}"
BIN_KEY="hdr/bin/zenmetrics-hdr-cpu-2026-07-03"
RESULTS_PREFIX="hdr/runs/$RUN"
SSH_KEY="${SSH_KEY:-zen-arm-dev-20260528}"
set -a; . ~/.config/cloudflare/r2-credentials; set +a
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
r2(){ AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto s5cmd --endpoint-url "$EP" "$@"; }

case "$CMD" in
launch)
  DGP="${3:?datagen prefix (codec-corpus)}"; N="${4:?n boxes}"
  METRICS="${5:-zensim ssim2 iwssim dssim cvvdp}"
  # scoped creds: read datagen (codec-corpus) needs its own token; read bin +
  # write results on zentrain needs another (R2 temp creds are single-bucket).
  mk_cred(){ python3 -c "import json,os;print(json.dumps({'bucket':'$1','parentAccessKeyId':os.environ['R2_ACCESS_KEY_ID'],'parentSecretAccessKey':os.environ['R2_SECRET_ACCESS_KEY'],'permission':'object-read-write','ttlSeconds':43200,'prefixes':$2}))" \
    | curl -sS -X POST -H "Authorization: Bearer $R2_API_TOKEN" -H "Content-Type: application/json" -d @- \
      "https://api.cloudflare.com/client/v4/accounts/$R2_ACCOUNT_ID/r2/temp-access-credentials"; }
  CC=$(mk_cred codec-corpus "['$DGP/']"); ZT=$(mk_cred zentrain "['hdr/']")
  read -r CAK CSK CST < <(python3 -c "import json;r=json.loads('''$CC''')['result'];print(r['accessKeyId'],r['secretAccessKey'],r['sessionToken'])")
  read -r ZAK ZSK ZST < <(python3 -c "import json;r=json.loads('''$ZT''')['result'];print(r['accessKeyId'],r['secretAccessKey'],r['sessionToken'])")
  [ -n "$CAK" ] && [ -n "$ZAK" ] || { echo "cred mint failed"; exit 1; }
  for i in $(seq 0 $((N-1))); do
    ci=$(mktemp)
    cat > "$ci" <<EOF
#cloud-config
runcmd:
  - |
    set -x
    exec > /root/worker.log 2>&1
    curl -sSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz | tar -xz -C /usr/local/bin s5cmd
    export AWS_REGION=auto
    S5C="s5cmd --endpoint-url $EP"
    # zentrain (bin + results)
    export AWS_ACCESS_KEY_ID='$ZAK' AWS_SECRET_ACCESS_KEY='$ZSK' AWS_SESSION_TOKEN='$ZST'
    \$S5C cp "s3://zentrain/$BIN_KEY" /usr/local/bin/zenmetrics && chmod +x /usr/local/bin/zenmetrics
    # codec-corpus (datagen)
    export AWS_ACCESS_KEY_ID='$CAK' AWS_SECRET_ACCESS_KEY='$CSK' AWS_SESSION_TOKEN='$CST'
    mkdir -p /data/ref /data/variants
    \$S5C cp "s3://codec-corpus/$DGP/ref/*" /data/ref/
    \$S5C cp "s3://codec-corpus/$DGP/zenjxl/variants.tar" /data/variants.tar
    tar -xf /data/variants.tar -C /data/variants && rm /data/variants.tar
    \$S5C cat "s3://codec-corpus/$DGP/zenjxl/pairs.tsv" > /data/pairs_all.tsv
    # my 1/N shard (header + every Nth row)
    awk -v n=$N -v i=$i 'NR==1 || (NR-1)%n==i' /data/pairs_all.tsv > /data/pairs.tsv
    wc -l /data/pairs.tsv
    export AWS_ACCESS_KEY_ID='$ZAK' AWS_SECRET_ACCESS_KEY='$ZSK' AWS_SESSION_TOKEN='$ZST'
    mkdir -p /out
    for m in $METRICS; do
      feat=""
      [ "\$m" = "zensim" ] && feat="--feature-output /out/zensim_features.parquet --zensim-features-regime with-iw"
      # first-pair gate
      head -2 /data/pairs.tsv > /data/one.tsv
      if ! zenmetrics score-pairs --metric "\$m" --hdr --hdr-transfer pu-rescale --pairs-tsv /data/one.tsv --out-parquet /out/gate.parquet > /out/gate.log 2>&1 \
         || grep -qiE "failed:|requires" /out/gate.log; then
        echo "GATE-FAILED \$m"; \$S5C cp /out/gate.log "s3://zentrain/$RESULTS_PREFIX/box-$i/GATE-FAILED-\$m.log"; continue
      fi
      zenmetrics score-pairs --metric "\$m" --hdr --hdr-transfer pu-rescale \
        --pairs-tsv /data/pairs.tsv --out-parquet "/out/\$m.parquet" \$feat >> /root/worker.log 2>&1
      \$S5C cp "/out/\$m.parquet" "s3://zentrain/$RESULTS_PREFIX/box-$i/\$m.parquet"
      [ -f /out/zensim_features.parquet ] && [ "\$m" = "zensim" ] && \$S5C cp /out/zensim_features.parquet "s3://zentrain/$RESULTS_PREFIX/box-$i/zensim_features.parquet"
    done
    echo done > /out/done.txt && \$S5C cp /out/done.txt "s3://zentrain/$RESULTS_PREFIX/box-$i/done.txt"
    echo "BOX-DONE"
EOF
    name="hdrsf-$RUN-$i"
    for typ in cx53 cx43 cpx62; do for loc in fsn1 nbg1 hel1; do
      hcloud server create --name "$name" --type "$typ" --image ubuntu-24.04 --location "$loc" \
        --ssh-key "$SSH_KEY" --label group="hdrsf-$RUN" --user-data-from-file "$ci" >/dev/null 2>&1 \
        && { echo "$name launched ($typ/$loc)"; break 2; }
    done; done
    rm -f "$ci"
  done
  echo "### launched. status: $0 status $RUN $N ; reap: $0 reap $RUN"
  ;;
status)
  N="${3:-6}"
  for i in $(seq 0 $((N-1))); do
    D=$(r2 cat "s3://zentrain/$RESULTS_PREFIX/box-$i/done.txt" 2>/dev/null | tr -d '\n')
    G=$(r2 ls "s3://zentrain/$RESULTS_PREFIX/box-$i/" 2>/dev/null | grep -c GATE-FAILED)
    echo "box-$i: ${D:-pending} (gate-failures: ${G:-0})"
  done
  ;;
reap)
  hcloud server list -l group="hdrsf-$RUN" -o noheader | awk '{print $2}' | xargs -r -n1 hcloud server delete >/dev/null && echo reaped
  ;;
esac
