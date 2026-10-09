# Steering correctness diagnostics; run under shared flock/run-heavy.
set working-directory := '..'

[positional-arguments]
steerfix-baseline root:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo fmt -p zensim -- --check > "$1/provenance/fmt-before.log" 2>&1
    just clippy > "$1/provenance/clippy-before.log" 2>&1

steerfix-build:
    cargo build -p zensim --release --all-features --example diffmap_block_coherence

steerfix-prepare root:
    python3 scripts/prodqual_label_free.py --steerfix-prepare "{{root}}"

steerfix-report root:
    python3 scripts/prodqual_label_free.py --steerfix-report "{{root}}"

# Custom cfg makes external packet access explicitly caller-selected.
steerfix-instrument-build:
    RUSTFLAGS='--cfg steerfix_instrument' cargo test -p zensim --all-features --release --lib --no-run --message-format=json-render-diagnostics

steerfix-regress:
    cargo test -p zensim --all-features --lib prepared_floor_recovery_preserves_serving_and_public_gradient -- --nocapture
    cargo test -p zensim --all-features --lib rev5_replay_defaults_without_environment_override -- --nocapture
    cargo test -p zensim --all-features --lib rev5_entry_steering_env_on_and_off -- --nocapture
    cargo test -p zensim --all-features --lib neighbour_exact_env_gate_and_gain -- --nocapture

[positional-arguments]
steerfix-validate root:
    #!/usr/bin/env bash
    set -euo pipefail
    cargo fmt -p zensim -- --check > "$1/provenance/fmt-final.log" 2>&1
    just --justfile benchmarks/steerfix.just steerfix-regress > "$1/provenance/regression.log" 2>&1
    cargo test -p zensim --all-features > "$1/provenance/all-features-test.log" 2>&1
    cargo check -p zensim --no-default-features --features custom-profiles > "$1/provenance/minimal-build.log" 2>&1
    cargo check -p zensim --no-default-features --features custom-profiles,feature-regime-v2 > "$1/provenance/v2-build.log" 2>&1
    just clippy > "$1/provenance/clippy-final.log" 2>&1
    just lint-scripts > "$1/provenance/lint-scripts.log" 2>&1

[positional-arguments]
steerfix-packet binary packet neighbour:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT="$3" STEERFIX_PACKET="$2" "$1" metric::bake::steerfix_packet::engineering_packet --exact --nocapture
