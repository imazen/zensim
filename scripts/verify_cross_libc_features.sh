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
# The SCORE is dumped alongside the features and gated SEPARATELY, because it
# is a SEPARATE defect with a SEPARATE owner: `metric.rs`'s raw-distance ->
# score mapping calls `powf` at exponents that are not powers of two (F19,
# `det_math::PowForm`, era `scorepow`). F18's `sqrt` derivation cannot reach
# it -- x^0.7 has no correctly-rounded closed form -- so the two axes are two
# env vars and this script sweeps the 2x2, which is what MEASURES that fixing
# the features did not fix the score rather than merely asserting it.
#
#   root=libm pow=libm : revision 1, both defects live      (negative control)
#   root=sqrt pow=libm : F18 fixed, F19 live                (the cell that proves
#                                                            they are independent)
#   root=libm pow=pure : F19 fixed, F18 live
#   root=sqrt pow=pure : revision 2                          (THE gate: both zero)
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
declare -A DIF
for ROOT in libm sqrt; do
for POW in libm pure; do
  ARM="r${ROOT}_p${POW}"
  ZENSIM_ROOT_FORM=$ROOT ZENSIM_POW_FORM=$POW "$BIN_GNU" \
      > "$OUT/dump.$ARM.gnu.tsv"  2>"$OUT/dump.$ARM.gnu.err"  || exit 2
  ZENSIM_ROOT_FORM=$ROOT ZENSIM_POW_FORM=$POW "$BIN_MUSL" \
      > "$OUT/dump.$ARM.musl.tsv" 2>"$OUT/dump.$ARM.musl.err" || exit 2
  for KIND in feat score; do
    if [ "$KIND" = feat ]; then FILT='$2 != "score"'; else FILT='$2 == "score"'; fi
    awk -F'\t' "$FILT" "$OUT/dump.$ARM.gnu.tsv"  > "$OUT/$KIND.$ARM.gnu"
    awk -F'\t' "$FILT" "$OUT/dump.$ARM.musl.tsv" > "$OUT/$KIND.$ARM.musl"
    tot=$(wc -l < "$OUT/$KIND.$ARM.gnu")
    dif=$(diff --unchanged-line-format= --old-line-format='%L' --new-line-format= \
            "$OUT/$KIND.$ARM.gnu" "$OUT/$KIND.$ARM.musl" | wc -l)
    DIF[$KIND.$ARM]=$dif
    printf 'root=%-4s pow=%-4s %-5s  differing %6d / %6d\n' "$ROOT" "$POW" "$KIND" "$dif" "$tot"
  done
done
done

# THE GATES: under revision 2 (root=sqrt, pow=pure) BOTH columns must be zero.
if [ "${DIF[feat.rsqrt_ppure]}" -ne 0 ]; then
  echo "FAIL: features still disagree across libcs under revision 2" >&2; rc=1
fi
if [ "${DIF[score.rsqrt_ppure]}" -ne 0 ]; then
  echo "FAIL: the SCORE still disagrees across libcs under revision 2" >&2; rc=1
fi

# A gate that can only pass is not a gate: revision 1 MUST show BOTH defects,
# or the instrument is not sensitive to what it claims to measure.
if [ "${DIF[feat.rlibm_plibm]}" -eq 0 ]; then
  echo "FAIL: revision 1 shows NO cross-libc FEATURE difference — the negative" >&2
  echo "control did not fire, so a zero on the deterministic arm proves nothing." >&2
  rc=1
fi
if [ "${DIF[score.rlibm_plibm]}" -eq 0 ]; then
  echo "FAIL: revision 1 shows NO cross-libc SCORE difference — the F19" >&2
  echo "negative control did not fire, so its zero proves nothing." >&2
  rc=1
fi

# And the INDEPENDENCE claim, which is the reason the two forms are two knobs:
# fixing the features must NOT have fixed the score.
if [ "${DIF[feat.rsqrt_plibm]}" -ne 0 ]; then
  echo "FAIL: F18's fix did not make the features deterministic on its own" >&2; rc=1
fi
if [ "${DIF[score.rsqrt_plibm]}" -eq 0 ]; then
  echo "NOTE: the score is libc-clean with F18 alone on this grid — F19's" >&2
  echo "exposure is real (measured in det_math's own tests) but this grid" >&2
  echo "did not reach it; widen the grid before citing a score number." >&2
fi

[ "$rc" -eq 0 ] && echo "PASS: features AND score bit-identical across libcs under revision 2"
exit $rc
