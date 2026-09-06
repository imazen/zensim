#!/usr/bin/env bash
# THE cross-feature-set serving gate.
#
# Builds `zensim/examples/serving_matrix.rs` under a matrix of cargo feature
# sets and asserts that every shipped profile scores BIT-IDENTICALLY to the
# reference arm of its environment, or refuses with a named error. A third
# outcome — a different number, no error — is the failure this gate exists to
# catch, and it is invisible from inside any single build.
#
# WHY a script and not a `#[test]`: a test cannot rebuild its own crate under
# different features. The in-process half of this gate is
# `zensim::serving::tests::every_shipped_profile_scores_its_pinned_value`,
# which runs under every permutation the CI matrix builds; this is the exact
# cross-build diff.
#
# Background: 2026-09-06. `cb2f412d` made the shipped A/B/BHdr/D bakes DENSE
# while the gather that serves a dense declaration was gated on
# `feature-regime-v2`, so every build without that feature served them the
# positional prefix — silently wrong by 3.3 to 260 zensim points.
# benchmarks/dense_serving_ungate_2026-09-06.md
#
# Usage: scripts/serving_matrix.sh [outdir]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-${TMPDIR:-$HOME/tmp}/zensim-serving-matrix}"
mkdir -p "$OUT"
cd "$REPO" || exit 2

# Each environment fixes the SIMD/threading axes, because those legitimately
# move the last few ULPs (MEASURED 2026-09-06: up to 1.05e-5 between avx512+
# threads and neither). Only the profile/regime axes vary within an
# environment, so any difference inside one is a serving defect, full stop.
#   name | environment features | reference arm extras | arm extras...
run_env() {
  local env_name="$1" env_feats="$2" ref_extra="$3"; shift 3
  local arms=("$@")
  local ref_file="$OUT/${env_name}__REF.tsv"
  local failed=0
  # VACUITY GUARD. `candidate-profiles` REQUIRES `feature-regime-v2` (a
  # profile a build can name it must be able to serve), so an arm that names
  # `candidate-profiles` silently gets v2 back through feature unification.
  # At least one arm per environment must therefore be genuinely v2-free, or
  # this matrix tests nothing at all — which is the failure mode that let the
  # original defect ship.
  local v2_free=0 a
  for a in "${arms[@]}"; do
    case "${a#*=},$env_feats," in
      *feature-regime-v2*|*candidate-profiles*) ;;
      *) v2_free=1 ;;
    esac
  done
  if [ "$v2_free" -eq 0 ]; then
    echo "  FAIL $env_name: no arm is free of \`feature-regime-v2\` — the matrix is vacuous"
    return 1
  fi

  # Join the environment's fixed features with an arm's extras into one
  # cargo `--features` string, tolerating either half being empty.
  join_feats() { printf '%s\n' "$env_feats" "$1" | paste -sd, - | sed 's/^,//; s/,$//; s/,,/,/g'; }

  build_run() { # <label> <extra-features>
    local label="$1" extra="$2" feats
    feats="$(join_feats "$extra")"
    local args=(--no-default-features)
    [ -n "$feats" ] && args+=(--features "$feats")
    if ! cargo build --release -p zensim --example serving_matrix "${args[@]}" \
         >"$OUT/${env_name}__${label}.build.log" 2>&1; then
      echo "BUILD FAILED: $env_name/$label (features: '$feats')"
      tail -20 "$OUT/${env_name}__${label}.build.log"
      return 1
    fi
    ./target/release/examples/serving_matrix > "$OUT/${env_name}__${label}.tsv" 2>&1
  }

  echo "=== environment '$env_name' (fixed: '${env_feats:-<none>}')"
  build_run REF "$ref_extra" || return 1
  local n_ref; n_ref=$(( $(wc -l < "$ref_file") - 1 ))
  echo "    reference arm: $n_ref rows (features: '$(join_feats "$ref_extra")')"

  for arm in "${arms[@]}"; do
    local label="${arm%%=*}" extra="${arm#*=}"
    build_run "$label" "$extra" || { failed=1; continue; }
    local bad n_refused=0
    bad=$(awk -F'\t' '
      NR==FNR { if (FNR>1) ref[$1"\t"$2"\t"$3] = $4"\t"$5; next }
      FNR==1  { next }
      {
        k = $1"\t"$2"\t"$3
        if (!(k in ref)) next                      # profile absent from REF: impossible, REF is the superset
        if ($4 == "REFUSED") next                  # a NAMED refusal is an accepted outcome
        if (ref[k] != $4"\t"$5) printf "      %s: REF=%s  ARM=%s\n", k, ref[k], $4"\t"$5
      }' "$ref_file" "$OUT/${env_name}__${label}.tsv")
    local n_rows; n_rows=$(( $(wc -l < "$OUT/${env_name}__${label}.tsv") - 1 ))
    local n_ref_refused; n_ref_refused=$(awk -F'\t' 'FNR>1 && $4=="REFUSED"' "$ref_file" | wc -l)
    if [ -n "$bad" ]; then
      echo "  FAIL $label ($n_rows rows, features '$(join_feats "$extra")'): scores differ from the reference arm"
      echo "$bad"
      failed=1
    else
      n_refused=$(awk -F'\t' 'FNR>1 && $4=="REFUSED"' "$OUT/${env_name}__${label}.tsv" | wc -l)
      echo "  ok   $label ($n_rows rows, $n_refused named refusals)"
    fi
    # A named refusal is accepted, but an arm that refuses MORE than the
    # reference is reported so a silent capability loss is still visible.
    if [ "$n_refused" -gt "$n_ref_refused" ] 2>/dev/null; then
      echo "       note: $n_refused refusals vs $n_ref_refused in the reference arm"
    fi
  done
  return $failed
}

rc=0
# NATIVE: what a `cargo add zensim` consumer gets, with the profile/regime
# axes swept. The reference is the full default set.
# Arm labels say whether `feature-regime-v2` is RESOLVED on, not merely
# whether it was typed: `candidate-profiles` pulls it in.
run_env native "avx512,threads,imgref" "deprecated-profiles,candidate-profiles,feature-regime-v2" \
  "dep_NOv2=deprecated-profiles" \
  "bare_NOv2=" \
  "cand_v2=candidate-profiles" \
  "v2_only=feature-regime-v2" \
  "dep_v2=deprecated-profiles,feature-regime-v2" \
  || rc=1

# BARE: no avx512, no rayon — the wasm / embedded / `--no-default-features`
# shape, which is the one that was broken. `classification` mirrors what
# `zensim-wasm-tests` asks for.
run_env bare "" "deprecated-profiles,candidate-profiles,feature-regime-v2" \
  "dep_NOv2=deprecated-profiles" \
  "wasm_like_NOv2=imgref,classification,deprecated-profiles" \
  "minimal_NOv2=" \
  || rc=1

if [ $rc -eq 0 ]; then
  echo "SERVING MATRIX: PASS — every arm agrees with its environment's reference arm."
else
  echo "SERVING MATRIX: FAIL — see above. Artifacts in $OUT"
fi
exit $rc
