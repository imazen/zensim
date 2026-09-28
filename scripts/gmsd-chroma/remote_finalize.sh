#!/usr/bin/env bash
set -euo pipefail
root=/var/tmp/gmsd-chroma
export PATH="$root/toolchain/bin:$PATH"
export TMPDIR="$root/tmp" XDG_CACHE_HOME="$root/cache"
export CARGO_HOME="$root/cache/cargo" CARGO_TARGET_DIR="$root/c8/target"
export UV_CACHE_DIR="$root/cache/uv" PYTHONDONTWRITEBYTECODE=1
unset RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_RUSTFLAGS ZENSIM_FORMULA_REV ZENSIM_ROOT_FORM
cd "$root/c8/check-src"
date -u +%FT%TZ
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --exclude zensim-wasm-tests -- -D warnings
sha256sum Cargo.toml Cargo.lock zensim/Cargo.toml zensim/src/feature_v2.rs zensim/src/feature_defs.rs zensim/src/gmsbank_constants.rs
cp --no-clobber "$root/c8/target/release/deps/extract_paths_bench-7220be076ceaaeb9" "$root/c8/binaries/cost_v3"
sha256sum "$root/c8/binaries/cost_v3"
python3 - "$root/c8/remote_inventory_$1.json" <<'PY'
import datetime,json,os,sys
root='/var/tmp/gmsd-chroma'
records=[]
def fail(error):
    raise error
for directory,dirs,files in os.walk(root,followlinks=False,onerror=fail):
    for name in dirs+files:
        path=os.path.join(directory,name)
        stat=os.lstat(path)
        records.append(dict(path=path,bytes=stat.st_size,blocks=stat.st_blocks,symlink=os.path.islink(path)))
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),worker='worker2',records=records)
with open(sys.argv[1],'x') as f:json.dump(out,f,sort_keys=True)
print('remote_inventory_paths',len(records))
PY
date -u +%FT%TZ
