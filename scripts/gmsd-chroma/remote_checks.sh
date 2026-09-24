#!/usr/bin/env bash
set -uo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS ZENSIM_FORMULA_REV ZENSIM_ROOT_FORM
cd "$root/c8/check-src" || exit 1
date -u +%FT%TZ
cargo fmt -p zensim || exit 1
cargo fmt --all -- --check
fmt_status=$?
cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings
clippy_status=$?
cargo test -p zensim --release --all-features --lib
test_status=$?
printf '{"fmt":%s,"clippy":%s,"release_lib_tests":%s}\n' "$fmt_status" "$clippy_status" "$test_status" > "$root/c8/checks_$1.json"
cat "$root/c8/checks_$1.json"
sha256sum Cargo.toml Cargo.lock zensim/Cargo.toml zensim/src/feature_v2.rs zensim/src/feature_defs.rs zensim/src/gmsbank_constants.rs
date -u +%FT%TZ
test "$fmt_status" -eq 0 && test "$clippy_status" -eq 0 && test "$test_status" -eq 0
