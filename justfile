# zensim dev commands

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

# CI-exact clippy
clippy:
    cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings

# THE default metric evaluation: run every eval/stat/bucket on a bake and
# emit console (markdown) + a self-contained big HTML report. Optional REF
# bake enables the per-zone dial-agreement panel; RAMP grid enables the
# severity-ramp monotonicity section (point it at a regime-matched parquet).
#   just metric-eval zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin
#   just metric-eval <bake> <ref-bake> <ramp-grid.parquet>
metric-eval bake ref="" ramp="" out="/mnt/v/output/zensim/reports":
    cargo build --release -p zensim-validate --bin bake_verdict
    mkdir -p {{out}}
    ./target/release/bake_verdict --bake {{bake}} \
        {{ if ref != "" { "--compare " + ref } else { "" } }} \
        {{ if ramp != "" { "--ramp-grid " + ramp } else { "" } }} \
        --output {{out}}/$(basename {{bake}} .bin).md \
        --html {{out}}/$(basename {{bake}} .bin).html
    @echo "report: {{out}}/$(basename {{bake}} .bin).html"
