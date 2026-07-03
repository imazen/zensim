#!/usr/bin/env bash
# Reproduce the t1dro51 candidate (2026-07-03) from pinned inputs, bit-for-bit.
#
#   bash scripts/reproduce_t1dro51.sh [seed]        # default 31 (best CID22)
#
# What it does:
#   1. builds the trainer at the RECORDED commit (training is deterministic;
#      the manifest's trainer_commit gate enforces source identity),
#   2. fetches every pinned input from R2 if not present locally (sha256s are
#      enforced by the trainer's manifest gates at load time),
#   3. trains zensim/weights/manifests/w3_t1dro51_s<seed>.toml,
#   4. verifies the produced bake's sha256 against the recorded value.
#
# Recorded bakes (2026-07-03, trainer 78ec8e614e46, machine-independent for a
# given SIMD tier — AVX-512 vs AVX2 reduction order can differ, see
# benchmarks/strategy_ablation_2026-07-02.md "cross-machine determinism"):
#   s17 556a08317f500fa26f4b89deb011faebc79c24e8ae5eb5a8eba7aded76b87155
#   s7  a326ed22751a8d8230d361a9b9f7bfd8dc0bfa6f10a174179adc5b184b543fc3
#   s31 c2ffc04452a61de670cfab54ae3b45c59676e5cc0ccf4989621194d61c486a60
#   s47 2a8af7ada1e1923e702f4e64db270cdae1ce605ff70ee5aae6e5fbd4d3cd58dc
#   s63 02bd7a722887fd70a0e4d7e65d96e2f55d6cb9e1bb2ba15371b3b3ebb40a0b8d
#
# Recipe (the "t1dro" strategy stack on the v51 base):
#   ema_decay=0.9  hard_pair_frac=0.5  hard_pair_max_delta=0.05
#   stratified_bands=10  dro_eta=0.5
#   (hard_pair_max_delta is a CID22<->KonJND dial: 0.03 rank-leaning,
#    0.08 threshold-leaning — see the wave-4 section of the campaign doc.)
set -euo pipefail
SEED="${1:-31}"
TRAINER_COMMIT="78ec8e614e46b326975680f629c7e25bd20f22fc"
declare -A SHAS=(
  [17]=556a08317f500fa26f4b89deb011faebc79c24e8ae5eb5a8eba7aded76b87155
  [7]=a326ed22751a8d8230d361a9b9f7bfd8dc0bfa6f10a174179adc5b184b543fc3
  [31]=c2ffc04452a61de670cfab54ae3b45c59676e5cc0ccf4989621194d61c486a60
  [47]=2a8af7ada1e1923e702f4e64db270cdae1ce605ff70ee5aae6e5fbd4d3cd58dc
  [63]=02bd7a722887fd70a0e4d7e65d96e2f55d6cb9e1bb2ba15371b3b3ebb40a0b8d
)
REPO="$(cd "$(dirname "$0")/.." && pwd)"
P=/mnt/v/output/zensim-multicodec-probe
CANON=/mnt/v/zen/zensim-training/canonical-2026-05-21/train
MANIFEST="$REPO/zensim/weights/manifests/w3_t1dro51_s${SEED}.toml"
[ -f "$MANIFEST" ] || { echo "no manifest for seed $SEED"; exit 1; }

echo "== 1/4 trainer at $TRAINER_COMMIT =="
cd "$REPO"
git fetch -q origin 2>/dev/null || true
git stash -u -q 2>/dev/null || true
git checkout -q "$TRAINER_COMMIT" -- zensim-validate/src zensim-train-core/src zensim/src 2>/dev/null \
  || git checkout -q "$TRAINER_COMMIT"
cargo build --release -p zensim-validate --bin zensim_mlp_train -j "$(( $(nproc) > 8 ? 8 : $(nproc) ))"

echo "== 2/4 inputs (fetch from R2 if missing; shas enforced by manifest gates) =="
if [ ! -f "$P/bigcodec_traindigits_2026-07-02.parquet" ] || [ ! -f "$CANON/safesyn.parquet" ]; then
  set -a; . ~/.config/cloudflare/r2-credentials; set +a
  EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
  export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto
  mkdir -p "$P" "$CANON"
  s5cmd --endpoint-url "$EP" cp "s3://zentrain/canonical-2026-05-21/train/*" "$CANON/"
  s5cmd --endpoint-url "$EP" cp "s3://zentrain/strategy-fleet-2026-07-02/derived/*" "$P/"
fi
python3 "$REPO/scripts/v_next/validate_parquet.py" --manifest "$MANIFEST"

echo "== 3/4 train (deterministic; ~10 min on 8 threads) =="
OUT="$P/repro_t1dro51_s${SEED}.bin"
RAYON_NUM_THREADS=6 nice -n10 "$REPO/target/release/zensim_mlp_train" \
  --manifest "$MANIFEST" --out "$OUT"

echo "== 4/4 verify =="
GOT=$(sha256sum "$OUT" | awk '{print $1}')
WANT="${SHAS[$SEED]}"
if [ "$GOT" = "$WANT" ]; then
  echo "REPRODUCED BIT-FOR-BIT: $OUT ($GOT)"
else
  echo "sha differs: got $GOT want $WANT"
  echo "(expected on a different SIMD tier — AVX-512 vs AVX2 changes fp reduction order;"
  echo " verify equivalence with: bake_verdict --bake $OUT and compare the panel)"
fi
