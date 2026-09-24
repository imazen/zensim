#!/usr/bin/env bash
# Run through worker 2's serialized heavy wrapper.
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
export RUSTFLAGS='--cfg gmsbank_calibration_instrument'
export GMSBANK_CHROMA_CALIB_DIR="$root/c8/calibration"
unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
cd "$root/c8/src"
date -u +%FT%TZ
cargo fmt -p zensim
sha256sum zensim/src/gmsbank_calibration_instrument.rs Cargo.toml zensim/Cargo.toml
cargo test -p zensim --release --all-features --lib gmsbank_chroma_calibration_dump -- --nocapture
date -u +%FT%TZ
