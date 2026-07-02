#!/usr/bin/env bash
# Bootstrap a Hetzner CCX (dedicated-CPU) box as a zensim TRAIN box.
# Idempotent; fail-loud. Run ON THE BOX (via ssh). See docs/DATA_SPLITS.md §6.
#
# Expects env (injected by scripts/hetzner/hz.sh provision|bootstrap):
#   ZENSIM_COMMIT   git commit to build the trainer at (trainer_commit gate)
#   R2_ENDPOINT     https://<acct>.r2.cloudflarestorage.com
#   AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN
#                   SCOPED temp creds (object-read-only on zentrain) — NEVER root keys.
set -euo pipefail
trap 'echo "[boot] FAILED at line $LINENO (rc=$?) $(date -u +%FT%TZ)"' ERR
echo "[boot] $(date -u +%FT%TZ) start"
# disk guard: refuse to start with <100G free (the encodes-runaway lesson)
AVAIL_G=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "$AVAIL_G" -ge 100 ] || { echo "[boot] FAILED: only ${AVAIL_G}G free"; exit 1; }

# 1. base tools (dev box, not a fleet image — bootstrap once, idempotent)
if ! command -v s5cmd >/dev/null; then
  curl -sSL https://github.com/peak/s5cmd/releases/download/v2.3.0/s5cmd_2.3.0_Linux-64bit.tar.gz \
    | tar -xz -C /usr/local/bin s5cmd
fi
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git build-essential pkg-config)
python3 -c "import pyarrow, toml, numpy, scipy" 2>/dev/null || (apt-get update -qq && apt-get install -y -qq python3-pip && pip3 install --break-system-packages -q pyarrow toml numpy scipy)
if ! command -v cargo >/dev/null; then
  curl -sSf https://sh.rustup.rs | sh -s -- -y -q --default-toolchain stable
fi
source "$HOME/.cargo/env" 2>/dev/null || true

# 2. clone + build the trainer AT THE PINNED COMMIT (byte-reproducible per
#    the 2026-07-01 verification; trainer_commit gate enforces it at run time)
mkdir -p ~/work && cd ~/work
[ -d zensim ] || git clone --filter=blob:none https://github.com/imazen/zensim
cd zensim
git fetch -q origin && git checkout -q "${ZENSIM_COMMIT:?set ZENSIM_COMMIT}"
cargo build --release -p zensim-validate --bin zensim_mlp_train --bin bake_verdict -j "$(nproc)"
echo "[boot] trainer built at $(git rev-parse HEAD)"

# 3. pull data from R2 (Hetzner<->Cloudflare is fast; R2 egress free)
export S5="s5cmd --endpoint-url $R2_ENDPOINT"
mkdir -p /data/{canonical-2026-05-21/train,canonical-2026-06-27,kadis,evalfeat,grids,out}
$S5 sync "s3://zentrain/canonical-2026-05-21/train/*" /data/canonical-2026-05-21/train/
# parquets ONLY — the encodes/ prefix is millions of objects (runaway 2026-07-02)
for ds in zenjpeg_lossy zenwebp_lossy zenwebp_lossless zenpng_lossless zenjxl_lossy zenjxl_lossless zenavif_lossy; do
  mkdir -p /data/canonical-2026-06-27/$ds
  for sp in train validate test; do
    $S5 cp "s3://zentrain/canonical/2026-06-27/$ds/$sp.parquet" /data/canonical-2026-06-27/$ds/ || true
  done
  $S5 cp "s3://zentrain/canonical/2026-06-27/$ds/_MANIFEST.json" /data/canonical-2026-06-27/$ds/ || true
done
$S5 cp   "s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet" /data/kadis/
# eval features + grids are rsynced from the workstation by hz.sh push-eval
echo "[boot] R2 data pulled"

# 4. rebuild derived training parquets ON-BOX (same scripts as local; the
#    manifests' sha256 gates verify byte-equality with the local builds)
cd ~/work/zensim
python3 scripts/hetzner/rebuild_derived.py /data
# mandatory data-contract validation (fleet/versioning errors die HERE)
python3 scripts/v_next/validate_parquet.py /data/derived/*.parquet --kind train --contracts /data/derived/_CONTRACTS.json
echo "[boot] DONE $(date -u +%FT%TZ)"
