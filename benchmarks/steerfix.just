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

# Custom cfg makes external packet access explicitly caller-selected.
steerfix-instrument-build:
    RUSTFLAGS='--cfg steerfix_instrument' cargo test -p zensim --all-features --release --lib --no-run --message-format=json-render-diagnostics

[positional-arguments]
steerfix-packet binary packet neighbour:
    ZENSIM_FORMULA_REV=5 ZENSIM_NEIGHBOUR_EXACT="$3" STEERFIX_PACKET="$2" "$1" metric::bake::steerfix_packet::engineering_packet --exact --nocapture
