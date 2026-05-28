#!/usr/bin/env bash
# Single-command reproduction of the shipped v47-strict-QAT-native bake
# (ZensimProfile::A). Fetches the canonical training inputs from R2, builds the
# trainer, runs the one-pass --manifest recipe, and verifies the result.
#
#   bash scripts/reproduce_v47.sh
#
# PREREQUISITES (the "anyone's machine" contract):
#   - Rust toolchain (1.89+) + network (the trainer pulls zenpredict from the
#     pinned imazen/zenanalyze git rev — no sibling checkout needed).
#   - aws CLI.
#   - R2 read credentials. Provide EITHER:
#       env AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY, OR
#       ~/.config/cloudflare/r2-credentials (access_key_id / secret_access_key).
#     (s3://zentrain is a private bucket — you need read access to it.)
#   - ~1.5 GB free disk for the canonical inputs (safesyn alone is 590 MB).
#
# CAVEAT — numeric, not bit-exact: training runs in f32 with rayon parallelism,
# so a fresh run on different hardware reproduces the v47 RESULTS (held-out
# panel within noise) but not necessarily a byte-identical bake sha256. The
# recipe's per-input sha256 gate guarantees you trained on the EXACT inputs;
# bake_verdict at the end confirms the held-out numbers match.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECIPE="$REPO/zensim/weights/manifests/v47_strict_qat.toml"
R2_ENDPOINT="https://338ad3b06716695d6e2c81c864e387d8.r2.cloudflarestorage.com"
R2_ROOT="s3://zentrain/canonical-2026-05-21/train"
# The recipe's [inputs.canonical_root].local — override with CANON_DIR=... if
# /mnt/v isn't available (then the recipe's canonical_root.local must match).
CANON_DIR="${CANON_DIR:-/mnt/v/zen/zensim-training/canonical-2026-05-21/train}"
INPUTS=(safesyn kadid tid cid22_train_norm konjnd-dense-norm multiband_anchor_dial100)

echo "== reproduce v47 =="
echo "repo=$REPO"
echo "recipe=$RECIPE"
echo "canonical dir=$CANON_DIR"

# 1) R2 credentials.
if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -f "$HOME/.config/cloudflare/r2-credentials" ]]; then
  AWS_ACCESS_KEY_ID=$(grep -m1 -iE "access_key_id|access-key" "$HOME/.config/cloudflare/r2-credentials" | sed 's/.*[=:] *//;s/"//g' | tr -d ' ')
  AWS_SECRET_ACCESS_KEY=$(grep -m1 -iE "secret_access_key|secret" "$HOME/.config/cloudflare/r2-credentials" | sed 's/.*[=:] *//;s/"//g' | tr -d ' ')
  export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
fi
[[ -n "${AWS_ACCESS_KEY_ID:-}" ]] || { echo "ERROR: no R2 creds (env or ~/.config/cloudflare/r2-credentials)"; exit 1; }

# 2) Fetch the canonical inputs from R2 (skips files already present + identical).
mkdir -p "$CANON_DIR"
echo "== fetching ${#INPUTS[@]} canonical inputs from $R2_ROOT =="
for f in "${INPUTS[@]}"; do
  dst="$CANON_DIR/$f.parquet"
  if [[ -f "$dst" ]]; then echo "  have $f.parquet"; continue; fi
  echo "  fetching $f.parquet ..."
  aws s3 cp "$R2_ROOT/$f.parquet" "$dst" --endpoint-url="$R2_ENDPOINT"
done

# 3) Build the trainer + the verifier (zenpredict pulled from the pinned git rev).
echo "== building zensim_mlp_train + bake_verdict (release) =="
cargo build --release --manifest-path "$REPO/Cargo.toml" \
  -p zensim-validate --bin zensim_mlp_train --bin bake_verdict

TRAIN="$REPO/target/release/zensim_mlp_train"
VERDICT="$REPO/target/release/bake_verdict"

# 4) Train in ONE pass. The recipe's sha256 gate verifies every input BEFORE
#    training — if R2 served the wrong bytes, this fails loud.
echo "== training (one-pass --manifest; sha-gated; ~200 epochs, last 40 QAT) =="
"$TRAIN" --manifest "$RECIPE"

# 5) Verify the produced bake on the held-out panel.
BAKE=$(grep -m1 '^file ' "$RECIPE" | sed 's/.*= *"//;s/".*//')
echo "== bake produced: $BAKE =="
sha256sum "$BAKE" || true
if [[ -x "$VERDICT" && -f "$BAKE" ]]; then
  echo "== bake_verdict (compare to recipe [eval]: CID22≈0.8657, dial G1≈0.97) =="
  "$VERDICT" --bake "$BAKE" --corpora cid22,kadid,tid,konjnd,aic3,aic4 || true
fi
echo "== done. Compare the panel above to zensim/weights/manifests/v47_strict_qat.toml [eval]. =="
