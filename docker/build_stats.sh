#!/usr/bin/env bash
# Build the zensim-stats Docker image.
#
# Per CLAUDE.md "BAKE EVERYTHING": the binaries are pre-built on the
# host (not inside the Dockerfile RUN). The cargo build is cached
# via the workspace's target/, so subsequent docker builds only pay
# the COPY cost.
set -euo pipefail

cd "$(dirname "$0")/.."

ARTIFACTS="docker/build_artifacts"
mkdir -p "$ARTIFACTS/val_parquets"

echo "===== Phase 1: build Rust binaries (cargo release) ====="
cargo build --release \
    --bin bake_verdict \
    --bin qsweep_eval \
    --bin predict_features_with_bake \
    -p zensim-validate

cp -f target/release/bake_verdict           "$ARTIFACTS/"
cp -f target/release/qsweep_eval            "$ARTIFACTS/"
cp -f target/release/predict_features_with_bake "$ARTIFACTS/"

echo "===== Phase 2: stage val parquets ====="
VAL_ROOT="/mnt/v/zen/zensim-training/2026-05-15-full-features"
# bake_verdict reads features-root and expects the canonical naming
# scheme `<corpus>_features_372col_<date>.parquet`. Copy the
# fair-holdout set (CID22 + AIC-3 + AIC-4 + KonJND) + the training
# anchors (KADID + TID — these were in v0.3 training, but useful as
# in-training reference points).
for f in \
    cid22_features_372col_2026-05-15.parquet \
    kadid_features_372col_2026-05-15.parquet \
    tid_features_372col_2026-05-15.parquet \
    konjnd_features_372col_2026-05-15.parquet \
    aic3_features_372col_2026-05-15.parquet \
    aic4_features_372col_2026-05-20.parquet ; do
    if [ ! -f "$VAL_ROOT/$f" ]; then
        echo "WARN: missing $VAL_ROOT/$f — skipping"
        continue
    fi
    cp -f "$VAL_ROOT/$f" "$ARTIFACTS/val_parquets/"
done

echo "===== Phase 3: stage qsweep fixture ====="
cp -f /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv \
      "$ARTIFACTS/qsweep_features.csv"
cp -f /mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv \
      "$ARTIFACTS/qsweep_manifest.tsv"

echo "===== Phase 4: docker build ====="
docker build -f docker/Dockerfile.stats -t zensim-stats:latest .

echo
echo "===== Build complete ====="
docker image ls zensim-stats:latest
echo
echo "Run: docker run --rm zensim-stats:latest"
echo "Or:  docker run --rm zensim-stats:latest --json > stats.json"
