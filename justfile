# Run compute-heavy recipes through the environment's resource-limited runner.

fmt:
    cargo fmt --all --check

check:
    cargo check --offline -p zensim --all-targets --all-features --message-format=short

test:
    cargo test --offline --workspace --features zensim/training,zensim/classification,zensim/zenpixels --no-fail-fast

clippy:
    cargo clippy --offline --workspace --all-targets --all-features -- -D warnings


path-test:
    cargo test --offline -p zensim --test cross_path --features training,classification,zenpixels -- --test-threads=1

path-audit *args:
    cargo run --offline --release -p zensim --example path_audit --features training,classification -- {{args}}
