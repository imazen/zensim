#!/usr/bin/env bash
# R6: the monotone-linear fits, one arm at a time, one owner for every step.
#
# `bake_dial_refit gram` -> `bake_dial_refit fit-lasso` -> `bake_verdict`.
# Nothing here computes a fit, a spline or a statistic; this file only decides
# WHICH tables go in.
#
# Flags are the did100 `ctl` recipe -- the one that reproduces shipped Profile D
# BYTE-IDENTICALLY -- with the input tables swapped for the arm's own:
#   --space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10
# so any difference between two arms is attributable to the luminance form.
#
# Three slices x two solvers. 0..155 is the ADD156 / Profile-D lineage and
# 0..227 is basic+peaks; 0..371 is added because the plan's R6 line asks for the
# fit "per feature family" and the masked + IW families are only REACHABLE at
# 372 width — F4 moves 36 slots of the first slice, 60 of the second and all
# 132 of the third, so the third is the most exposed instrument of the three.
# (§7.4 of the pre-registration narrowed this to two slices; the third is added
# back as a STRENGTHENING and is reported separately from the pre-registered
# decision, never in place of it.)
#
# two solvers (`lasso` = the shipped recipe, `bvls` = the sign-constrained
# monotone-linear class the user's directive names). The sign mask is held FIXED
# across arms on purpose: it encodes the structural direction of an error
# feature, and re-deriving it per arm would vary two things at once.
#
# Usage: r6_fit_arms.sh <arm> [ROOT]
set -euo pipefail
ARM="${1:?usage: r6_fit_arms.sh <ssim2|c1|lorentz|clamp> [ROOT]}"
ROOT="${2:-/mnt/v/output/zensim/rev2-2026-09-05/r6}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BDR="$REPO/target/release/bake_dial_refit"
MASK="$REPO/benchmarks/feature_sign_mask_2026-05-26.tsv"
T="$ROOT/tables/$ARM"
mkdir -p "$ROOT"/{grams,fits,bakes,slices}

for n in 156 228 372; do
  [ -s "$ROOT/slices/a$n.idx" ] || seq 0 $((n-1)) > "$ROOT/slices/a$n.idx"
done

say() { printf '[%s] %s %s\n' "$(date -u +%H:%M:%S)" "$ARM" "$*"; }

say "gram (safesyn, $(wc -l < "$T/safesyn.csv") csv lines)"
nice -n19 ionice -c3 "$BDR" gram --parquet "$T/safesyn.parquet" \
    --target human_score --space raw --out "$ROOT/grams/${ARM}_safesyn.npz"

for slice in 156 228 372; do
  for solver in lasso bvls; do
    L="${ARM}_s${slice}_${solver}"
    say "fit $L"
    BOUNDS=(); [ "$solver" = bvls ] && BOUNDS=(--bounds-tsv "$MASK")
    nice -n19 ionice -c3 "$BDR" fit-lasso \
        --gram "$ROOT/grams/${ARM}_safesyn.npz" --weight 1.0 \
        --space raw --target human_score --solver "$solver" "${BOUNDS[@]}" \
        --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10 \
        --slice-file "$ROOT/slices/a${slice}.idx" \
        --anchor-parquet "$T/anchor.parquet" --anchor-target human_score \
        --emit-fit-npz "$ROOT/fits/${L}.fit.npz" \
        --out "$ROOT/bakes/${L}.bin"
  done
done
say "done"
