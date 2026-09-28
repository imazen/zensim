#!/usr/bin/env bash
# Worker 2; invoked through its shared heavy lock. Immutable per-attempt log.
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export PYTHONDONTWRITEBYTECODE=1 RUSTFLAGS='--cfg gmsbank_calibration_instrument'
export ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt
unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
cd "$root/c8/impl-src"
date -u +%FT%TZ
cargo fmt -p zensim
export GMSBANK_CHROMA_ORACLE_DIR="$root/octave"
export GMSBANK_CHROMA_ORACLE_REPORT="$root/c8/author_cs_$1.json"
cargo test -p zensim --release --all-features --lib gmsbank_chroma_author_map -- --nocapture
for test in gmsbank_strict_contrast_reduction_routes_only_to_loss gmsbank_identity_and_constant_monotonicity gmsbank_materialized_matches_streaming gmsbank_stride_matches_tight; do
    cargo test -p zensim --release --all-features --lib "$test" -- --nocapture
done
cargo test -p zensim --release --all-features --lib feature_defs::tests -- --nocapture
env -u ZENSIM_FORMULA_REV -u ZENSIM_ROOT_FORM cargo test -p zensim --release --all-features --lib feature_plan:: -- --nocapture
export GMSBANK_CHROMA_CALIB_DIR="$root/c8/calibration"
export GMSBANK_CHROMA_FEATURES="$root/c8/features8_$1.csv"
cargo test -p zensim --release --all-features --lib gmsbank_chroma_features_dump -- --nocapture
"$root/stats-env/bin/python" "$root/c8/scripts/numpy_reference.py" --chroma \
    "$root/c8/calibration/chroma_xyb8.tsv" "$GMSBANK_CHROMA_FEATURES" \
    "$root/c8/calibration/chroma_report.json"
date -u +%FT%TZ
