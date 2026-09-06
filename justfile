# zensim dev commands

# The rustdoc-JSON nightly is PINNED (keep in sync with the `api-doc-check`
# job in .github/workflows/ci.yml): an unpinned tracking nightly churns
# cross-crate path rendering with zero repo changes — MEASURED 2026-09-06,
# regenerating zensim-regress.txt against a newer nightly than the one that
# produced the committed snapshot rewrote 11 lines of `std::io::error::Error`
# to `core::io::error::Error` (the core::io re-homing), which is exactly the
# false-diff class this pin exists to prevent. Bump the pin deliberately, in
# the same commit as a `just api-doc` regen.
apidoc_toolchain := "nightly-2026-09-02"

# Format + regenerate the public-API surface snapshots (docs/public-api/).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test`, nor by any OTHER CI job — only
# the dedicated `api-doc-check` CI job (which sets ZEN_API_DOC=check) runs it.
fmt:
    cargo fmt --all
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots only
api-doc:
    ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current (what CI's api-doc-check job runs)
api-doc-check:
    ZEN_API_DOC=check ZEN_API_DOC_TOOLCHAIN={{apidoc_toolchain}} cargo test --manifest-path apidoc/Cargo.toml

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

# Fail on scripts that cannot run: pinned to a deleted sibling worktree, or
# hardcoding a binary with no source anywhere. On 2026-07-15 an audit found 25
# of 130 scripts in scripts/v_next/ pointing into worktrees that had been
# cleaned up weeks earlier, plus one that had not PARSED since a bulk sed.
# Nobody noticed because nobody ran them. This is the check that notices.
lint-scripts:
    python3 scripts/lint_scripts.py

# Report only, never fails — for a quick survey.
lint-scripts-list:
    python3 scripts/lint_scripts.py --list

# Data-integrity gates for a training mix. `lint-scripts` asks "can this script
# still run?"; these ask "is the DATA this recipe trains on structurally sound?".
# Added 2026-08-04 after the first-ever orientation gate found a six-week-old
# inverted KADID target on its first run, and the follow-up audit
# (benchmarks/data_integrity_audit_2026-08-04.md) found a teacher twin ranking its
# own rows at rho=0.25 at 7.9% of sampling mass. Both classes are silent without
# a gate. Pass SPEC=<bake>.bin.spec.json to audit a different recipe.
SPEC := "/mnt/v/output/zensim/bakes/sota944/bakes/H_co3abpg_s2507.bin.spec.json"
EXT944 := "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01"

# Target orientation vs raw human labels, every known eval root (exits 1 on any inversion).
check-orientation:
    python3 scripts/canonical_corpus/check_target_orientation.py --all-roots

# Which mix legs an external orientation check can reach at all (2 of 11 today).
check-provenance:
    python3 scripts/canonical_corpus/check_target_orientation.py --provenance

# Full structural audit of a mix from a bake's embedded repro: target + feature
# integrity, duplicate rows, teacher-twin correspondence/agreement, eval leakage.
# ~4 min, peak RSS ~3.6 GiB for the 11-table SOTA-944 mix.
check-mix:
    python3 scripts/canonical_corpus/check_table_integrity.py \
        --mix-from-spec {{SPEC}} \
        --data-root '/home/lilith/sota944/data/ext944={{EXT944}}' \
        --data-root '/home/lilith/sota944/data/teacher=/mnt/v/output/zensim/bakes/sota944/teacher' \
        --data-root '/home/lilith/sota944/data/kadis944=/mnt/v/zen/zensim-training/kadis-944-2026-08-01' \
        --data-root '/home/lilith/sota944/data/tbig_944_200k.parquet=/mnt/v/zen/zensim-training/tbig_944_200k.parquet' \
        --twin tsafesyn=safesyn --twin ttbig=bigcodec --twin tkadis=kadis \
        --leak-eval-root {{EXT944}}

# Everything above, in the order a pre-training check should run them.
check-data: check-provenance check-orientation check-mix

# THE CROSS-LIBC GATE (F18 + F19, `zensim::det_math`): build the feature dump
# for glibc AND static musl from THIS commit and compare `to_bits()` over the
# 20-cell parity matrix + 200 ladder cells. Sweeps the 2x2 of the two era knobs
# (`ZENSIM_ROOT_FORM` x `ZENSIM_POW_FORM`) as RUNTIME env vars on the same pair
# of binaries, so only one thing varies per cell, and gates the FEATURES and the
# SCORE separately -- they are two independent defects, and the `sqrt`+`libm`
# cell is what MEASURES that (F18's fix leaves the score exactly as
# libc-dependent as it found it: 1 of 220). Revision 1 MUST differ on BOTH
# columns (the negative controls) and revision 2 MUST be bit-identical on both.
# Needs `rustup target add x86_64-unknown-linux-musl`; no container.
check-cross-libc:
    ./scripts/verify_cross_libc_features.sh

# SAME-CLASS golden check for zensim-validate/tests/legacy_bake_sha.rs: the
# PINNED sha256 digests in that file were measured on THIS box (Zen 4 /
# AVX-512) and are not bit-reproducible on other SIMD tiers/platforms — CI runs
# the in-process A/B in that file's default (env-unset) path instead. Run this
# ONLY on the Zen 4 / AVX-512 dev box that captured PINNED, after touching
# anything in mlp_train's polarity-sensitive sites, to confirm the same-class
# digests still hold.
legacy-bake-zen4-golden:
    ZENSIM_ZEN4_GOLDEN_BAKE_SHA=1 cargo test -p zensim-validate --test legacy_bake_sha -- --nocapture
