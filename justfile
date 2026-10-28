# Run compute-heavy recipes through the environment's resource-limited runner.

fmt:
    cargo fmt --all --check

check:
    cargo check --offline -p zensim --all-targets --all-features --message-format=short

test:
    cargo test --offline --workspace --features zensim/training,zensim/classification,zensim/zenpixels --no-fail-fast

clippy:
    cargo clippy --offline --workspace --all-targets --all-features -- -D warnings
