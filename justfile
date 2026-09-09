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

# Quick offline rank/dial report (not full-eval or product qualification).
# Emits markdown plus a self-contained HTML report. Optional REF
# bake enables the per-zone dial-agreement panel; RAMP grid enables the
# severity-ramp monotonicity section (point it at a regime-matched parquet).
#   just metric-eval zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin
#   just metric-eval <bake> <ref-bake> <ramp-grid.parquet>
[doc("Quick offline rank/dial report; not full-eval or product qualification")]
[positional-arguments]
metric-eval bake ref="" ramp="" out="/mnt/v/output/zensim/reports":
    #!/usr/bin/env bash
    set -euo pipefail
    bake=$1
    ref=$2
    ramp=$3
    out=$4
    cargo build --release -p zensim-validate --bin bake_verdict
    mkdir -p "$out"
    stem=$(basename "$bake" .bin)
    args=(--bake "$bake" --output "$out/$stem.md" --html "$out/$stem.html")
    if [[ -n "$ref" ]]; then args+=(--compare "$ref"); fi
    if [[ -n "$ramp" ]]; then args+=(--ramp-grid "$ramp"); fi
    "${CARGO_TARGET_DIR:-target}/release/bake_verdict" "${args[@]}"
    echo "report: $out/$stem.html"

# Fail on scripts that cannot run: pinned to a deleted sibling worktree, or
# hardcoding a binary with no source anywhere. On 2026-07-15 an audit found 25
# of 130 scripts in scripts/v_next/ pointing into worktrees that had been
# cleaned up weeks earlier, plus one that had not PARSED since a bulk sed.
# Nobody noticed because nobody ran them. This is the check that notices.
lint-scripts:
    python3 scripts/lint_scripts.py

# Paired arithmetic-revision cost A/B (issue #61). Builds the interleaved
# extraction instrument ONCE and runs it in alternating revision-1/revision-3
# blocks on one pinned core, with `fast_ssim2` as the cross-block drift anchor.
# Do NOT wrap in run-heavy — its cgroup makes zenbench's gating refuse the run.
#   just rev3-cost ~/tmp/rev3-cost 2 1024,2048 8
[positional-arguments]
rev3-cost out blocks="2" sizes="1024,2048" cpu="8":
    cargo bench --no-run --locked --bench extract_paths_bench -p zensim \
        --features custom-profiles,feature-regime-v2,threads,training
    ./scripts/bench/rev3_cost_ab.sh "$1" "$2" "$3" "$4"
    python3 scripts/bench/rev3_cost_report.py "$1"

# Peak-RSS half of the revision cost question: `/usr/bin/time -v` max RSS per
# arm per revision, one arm per process so the reading is attributable.
#   just rev3-rss ~/tmp/rev3-rss
[positional-arguments]
rev3-rss out sizes="1024 2048" arms="buf_v1_372 fold372_full fold944_full" cpu="8":
    ./scripts/bench/rev3_rss.sh "$1" "$2" "$3" "$4"

# Replay the registered spatial coherence cells at revision 1 and revision 3.
# ARMS THE CROSS-REVISION DIAGNOSTIC BYPASS: the bakes were fit at revision 1,
# so this measures what the extraction change does to a FIXED model. Not a
# qualification, not model quality. See the script header.
[positional-arguments]
rev3-spatial out:
    ./scripts/bench/rev3_spatial_replay.sh "$1"

# Report only, never fails — for a quick survey.
lint-scripts-list:
    python3 scripts/lint_scripts.py --list

# Target orientation vs raw human labels, every registered eval root.
check-orientation:
    python3 scripts/canonical_corpus/check_target_orientation.py --all-roots

# Which mix legs an external orientation check can reach.
check-provenance:
    python3 scripts/canonical_corpus/check_target_orientation.py --provenance

# Audit THIS recipe's data. Pass owner options explicitly (root remaps, twins,
# leakage roots); no hardcoded August campaign is silently substituted.
# Example: just check-mix path/to/bake.bin.spec.json --leak-eval-root /data/eval
[doc("Audit the supplied recipe's data; pass root/remap/leakage options explicitly")]
[positional-arguments]
check-mix spec *options:
    #!/usr/bin/env bash
    set -euo pipefail
    spec=$1
    shift
    exec python3 scripts/canonical_corpus/check_table_integrity.py --mix-from-spec "$spec" "$@"

# Orientation/provenance plus the supplied recipe's structural data audit.
[positional-arguments]
check-data spec *options:
    #!/usr/bin/env bash
    set -euo pipefail
    spec=$1
    shift
    just check-provenance
    just check-orientation
    just check-mix "$spec" "$@"

# Run the existing offline+coherence owner; this does not run the codec exam.
# Regime is explicit; an optional fourth argument overrides the feature root.
[doc("Offline + coherence evaluation: bake, name, regime, optional feature root")]
[positional-arguments]
full-eval bake name regime root="":
    #!/usr/bin/env bash
    set -euo pipefail
    exec scripts/run_full_eval.sh "$@"

# Render recorded verdicts with the current summer-gauntlet owner.
[positional-arguments]
compare fulleval_dir out *options:
    #!/usr/bin/env bash
    set -euo pipefail
    fulleval_dir=$1
    out=$2
    shift 2
    python3 scripts/v_next/gauntlet.py --fulleval-dir "$fulleval_dir" --out "$out" "$@"
    scripts/v_next/gauntlet_gates.sh "$out" "$fulleval_dir"

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
