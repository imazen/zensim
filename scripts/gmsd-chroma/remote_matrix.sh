#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export PYTHONDONTWRITEBYTECODE=1 RUSTFLAGS='--cfg gmsbank_calibration_instrument'
export ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt
unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
export GMSBANK_MATRIX_DIR="$root/c8/identity_v3"
date -u +%FT%TZ
for arm in ${2:-base off on}; do
    export GMSBANK_MATRIX_ARM=$arm
    if [[ $arm == base ]]; then cd "$root/c8/base-src"; else cd "$root/c8/check-src"; fi
    # Copied workspaces can share Cargo package IDs and cached fingerprint
    # paths. Separate targets are mandatory for the parent/candidate contrast.
    if [[ $arm == base ]]; then
        export CARGO_TARGET_DIR="$root/c8/base-target"
    else
        export CARGO_TARGET_DIR="$root/c8/candidate-target"
    fi
    cargo fmt -p zensim
    cargo test -p zensim --release --all-features --lib --no-run --message-format=json > "$root/c8/matrix_${1}_${arm}_build.jsonl"
    binary=$(python3 -c 'import json,sys; a=[json.loads(s) for s in open(sys.argv[1])]; print(next(r["executable"] for r in a if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="zensim" and r.get("executable")))' "$root/c8/matrix_${1}_${arm}_build.jsonl")
    frozen="$root/c8/binaries/matrix_${1}_${arm}"
    test ! -e "$frozen"
    cp "$binary" "$frozen"
    if [[ $arm != base ]]; then
        "$frozen" --list > "$root/c8/matrix_${1}_${arm}_tests.txt"
        grep -q '^feature_v2::tests::gmsbank_chroma_author_map:' "$root/c8/matrix_${1}_${arm}_tests.txt"
    fi
    sha256sum "$frozen" Cargo.toml Cargo.lock zensim/src/feature_v2.rs zensim/src/feature_defs.rs
    for tier in native v3 scalar; do
        export GMSBANK_MATRIX_TIER=$tier
        for threads in 1 8; do
            export RAYON_NUM_THREADS=$threads
            export GMSBANK_MATRIX_OUT="$root/c8/identity_v3/$1_${arm}_${tier}_mt${threads}"
            "$frozen" --exact feature_v2::tests::gmsbank_chroma_prefix_dump --nocapture
        done
    done
    if [[ $arm == on ]]; then
        "$frozen" --exact feature_v2::tests::gmsbank_constant_chroma_shift_is_visible_without_gradients --nocapture
        export GMSBANK_CHROMA_ORACLE_DIR="$root/octave"
        export GMSBANK_CHROMA_ORACLE_REPORT="$root/c8/author_cs_$1.json"
        "$frozen" --exact feature_v2::tests::gmsbank_chroma_author_map --nocapture
        export GMSBANK_CHROMA_CALIB_DIR="$root/c8/calibration"
        export GMSBANK_CHROMA_FEATURES="$root/c8/features8_$1.csv"
        "$frozen" --exact feature_v2::tests::gmsbank_chroma_features_dump --nocapture
        "$root/stats-env/bin/python" "$root/c8/scripts/numpy_reference.py" --chroma \
            "$root/c8/calibration/chroma_xyb8.tsv" "$GMSBANK_CHROMA_FEATURES" \
            "$root/c8/calibration/chroma_report.json"
    fi
done
date -u +%FT%TZ
