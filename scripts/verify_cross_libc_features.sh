#!/usr/bin/env bash
# THE CROSS-LIBC GATE for F18 (`zensim::det_math`, era `v1detroot`).
#
# THE CONTRACT: two builds of the SAME COMMIT, one linked against glibc and
# one statically against musl, must emit bit-identical v1-372 feature vectors.
# Under revision 1 they DO NOT — `powf` is not correctly rounded and no
# standard makes two libcs agree on it, which is why the fleet's Feature
# executor had to be rebuilt against glibc
# (benchmarks/libm_pow_nondeterminism_2026-09-06.md). Under the deterministic
# root form the pooled 4th/8th roots become `sqrt` compositions, and IEEE-754
# REQUIRES `sqrt` to be correctly rounded.
#
# So this script measures BOTH arms from the SAME pair of binaries — the arm
# is a runtime env var, not a rebuild, because this repo has measured a
# rebuild alone shifting a 2304^2 timing ~10 % and the same reasoning applies
# to any two-build comparison: vary one thing.
#
#   libm arm (revision 1, shipped): expected to DIFFER
#   sqrt arm (revision 2):          expected to be BIT-IDENTICAL
#
# The SCORE is dumped alongside the features and compared SEPARATELY, because
# `metric.rs`'s raw-distance -> score mapping calls `powf` at exponents that
# are not powers of two. That exposure is real, named, and NOT fixed by this
# era; reporting it apart from the features is what keeps the feature result
# honest.
#
# Fails loud and nonzero.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="${XLIBC_OUT:-$HOME/tmp/libcfix}"
GNU=x86_64-unknown-linux-gnu
MUSL=x86_64-unknown-linux-musl
mkdir -p "$OUT"

for T in "$GNU" "$MUSL"; do
  echo "== building example libc_feature_dump for $T"
  ( cd "$REPO" && nice -n19 ionice -c3 cargo build --release -p zensim \
      --features training --example libc_feature_dump --target "$T" ) || exit 2
done

TD="${CARGO_TARGET_DIR:-$REPO/target}"
BIN_GNU="$TD/$GNU/release/examples/libc_feature_dump"
BIN_MUSL="$TD/$MUSL/release/examples/libc_feature_dump"
echo "== linkage"
file "$BIN_GNU" | sed 's/^/   gnu : /'
file "$BIN_MUSL" | sed 's/^/   musl: /'
ldd "$BIN_GNU" 2>&1 | sed 's/^/   gnu ldd: /'

rc=0
for ARM in libm sqrt; do
  ZENSIM_ROOT_FORM=$ARM "$BIN_GNU"  > "$OUT/dump.$ARM.gnu.tsv"  2>"$OUT/dump.$ARM.gnu.err"  || exit 2
  ZENSIM_ROOT_FORM=$ARM "$BIN_MUSL" > "$OUT/dump.$ARM.musl.tsv" 2>"$OUT/dump.$ARM.musl.err" || exit 2
  for KIND in feat score; do
    if [ "$KIND" = feat ]; then FILT='$2 != "score"'; else FILT='$2 == "score"'; fi
    awk -F'\t' "$FILT" "$OUT/dump.$ARM.gnu.tsv"  > "$OUT/$KIND.$ARM.gnu"
    awk -F'\t' "$FILT" "$OUT/dump.$ARM.musl.tsv" > "$OUT/$KIND.$ARM.musl"
    tot=$(wc -l < "$OUT/$KIND.$ARM.gnu")
    dif=$(diff --unchanged-line-format= --old-line-format='%L' --new-line-format= \
            "$OUT/$KIND.$ARM.gnu" "$OUT/$KIND.$ARM.musl" | wc -l)
    printf '%-5s %-5s  differing %6d / %6d\n' "$ARM" "$KIND" "$dif" "$tot"
    # THE GATE: features under the deterministic arm must be zero.
    if [ "$ARM" = sqrt ] && [ "$KIND" = feat ] && [ "$dif" -ne 0 ]; then
      echo "FAIL: the deterministic arm still disagrees across libcs" >&2
      rc=1
    fi
  done
done
# A gate that can only pass is not a gate: revision 1 MUST show the defect,
# or the instrument is not sensitive to what it claims to measure.
d1=$(diff --unchanged-line-format= --old-line-format='%L' --new-line-format= \
       "$OUT/feat.libm.gnu" "$OUT/feat.libm.musl" | wc -l)
if [ "$d1" -eq 0 ]; then
  echo "FAIL: revision 1 shows NO cross-libc difference — the negative" >&2
  echo "control did not fire, so a zero on the sqrt arm proves nothing." >&2
  rc=1
fi
[ "$rc" -eq 0 ] && echo "PASS: features bit-identical across libcs on the deterministic arm"
exit $rc
