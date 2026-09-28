#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
export RUSTFLAGS='--cfg gmsbank_calibration_instrument'
export ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt
unset CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS
cd "$root/c8/impl-src"
date -u +%FT%TZ
cargo bench -p zensim --all-features --bench extract_paths_bench --no-run --message-format=json > "$root/c8/cost_build_$1.jsonl"
binary=$(python3 -c 'import json,sys; a=[json.loads(s) for s in open(sys.argv[1])]; print(next(r["executable"] for r in a if r.get("reason")=="compiler-artifact" and r.get("target",{}).get("name")=="extract_paths_bench" and r.get("executable")))' "$root/c8/cost_build_$1.jsonl")
sha256sum "$binary"
export ZEN_XP_SIZES=256,1024,2048,4096 ZEN_XP_ROUNDS=8 ZEN_XP_MIN_ROUNDS=8 ZEN_XP_WALL_S=180
cd "$root/remote-src"
for threads in 1 8; do
    export RAYON_NUM_THREADS=$threads ZENBENCH_RESULT_PATH="$root/c8/cost_${1}_mt${threads}.zenbench"
    # sudo clears the caller environment; pass all benchmark settings as
    # command arguments INSIDE the privacy namespace.
    bash "$root/remote-src/scripts/gmsd-chroma/speed_namespace.sh" env \
        ZEN_XP_SIZES=256,1024,2048,4096 ZEN_XP_ROUNDS=8 ZEN_XP_MIN_ROUNDS=8 ZEN_XP_WALL_S=180 \
        ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt RAYON_NUM_THREADS="$threads" \
        ZENBENCH_RESULT_PATH="$ZENBENCH_RESULT_PATH" "$binary"
    test -s "$ZENBENCH_RESULT_PATH"
done
date -u +%FT%TZ
