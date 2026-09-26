# featcanon WORKLOG — canonical feature arithmetic lane (2026-09-25)

Lane: `quarantine/devin/featcanon`, workspace `~/work/zen/zensim--featcanon`
(based on main@origin 25edb379). Brief: `~/tmp/devin/fitopt/BRIEF_featcanon.md`
+ user ADDENDUM: FULL Rev4 set = all 18 families, width 1825
(`research::extract(Request::everything(), parallel=false)` + restore-cuts).
Target: new arithmetic revision making the whole 1825-vector bit-identical
across tiers; Rev1/Rev2/Rev3 values MUST NOT move.

## Scratch/artifacts (all under ~/tmp/devin/featcanon/, NOT /tmp)

- `audit_pairs.tsv` — 11 pairs (9 real TRAIN + derived `kadid64crop` 64x64
  textured crop + `mosaic4096` = 2x2 mosaic of disjoint ~2048² real AIC3
  crops; no real ≥4MP images exist in the corpora — verified by scan).
- `pairs_derived/` — generated PNGs.
- `audit_rev1.{tsv,log}`, `audit_rev3.{tsv,log}` — tier-diff audit at
  main@origin BEFORE canon work. Tool: `zensim-validate/src/bin/featcanon_audit.rs`
  (copied from tierparity lane's tier_audit_features.rs; usage:
  `RAYON_NUM_THREADS=1 ./featcanon_audit audit_pairs.tsv`; cumulative tier
  disables → columns v4x,v4,v3,scalar vs v3 baseline).
- Build: `CARGO_TARGET_DIR=/var/tmp/featcanon/target ~/tmp/devin/heavy --mem 16G --jobs 8 -- cargo ...`

## Baseline audit results (main@origin, Rev3 audit similar)

- v4x ≡ v4 (feature dispatch has no material v4x-only leaf); v4 diffs ONLY in
  basic/iw/masked/peaks (~0–297 slots/pair, rel ≤ 6e-5) — v1 SSIM pools.
- scalar: EVERY family dirty (~400–1600 slots, up to rel 9e290 denormal /
  1.49 on gridblk). Root causes: (a) magetypes scalar `mul_add` unfused
  (XYB matrix, fused_blur_h_ssim sliding sums, ext weights), (b) `reduce_add`
  tree per backend width, (c) tail/chunk geometry (XYB n%8, width%16/8).

## Design decided (era-2 precedent)

Canonical body = ONE plain-Rust body per leaf, inherent `f32::mul_add`
(fused on every target), fixed 8-virtual-lane pools (lane = x mod 8) closed
by `era2_reduce8`, no `reduce_add`. Candidates per brief:
- (d) c32 = LanesF32 (f32 lanes + fixed tree) — era-2 shape.
- (e) c64 = LanesF64 (f64 lanes, same tree).
- (f) neum = Neum64 (Neumaier-compensated f64, sequential).
- exact = f64 elements (fused mul_add mirrors) + Neum64 accumulation,
  f32 rounding only at plane stores — the oracle arm.

Switch: `ZENSIM_FEATCANON=off|exact|c32|c64|neum` (OnceLock) in
`zensim/src/featcanon.rs` (new module: Mode, LanesF32/LanesF64/Neum64,
`Pool` trait {zero,add(lane,f32),add64,fin}, CANON_LANES=8).
Leaf wrappers consult `crate::featcanon::mode()` and early-return the
canon/exact body BEFORE `incant!`.

## State of the code (ALL UNCOMMITTED in @)

Converted/hooked:
- `color.rs`: `opsin_px_canon`/`opsin_px_exact` + `srgb_xyb_canon<POSITIVE>` /
  `linear_xyb_canon<CLAMP>` drivers; hooked at `srgb_to_positive_xyb_planar_into`,
  `srgb_to_xyb_planar_into`, `linear_to_positive_xyb_planar_into`,
  `linear_to_positive_xyb_planar_into_unclamped`. Canon = OpsinChunk formula
  w/ fused mul_add + `magetypes::nostd_math::cbrt_midp_f32` + always-padded tail
  (cbrt_midp_f32 == vector cbrt_midp lanes, verified same ops/seed).
  NOTE: pu_xyb (`linear_to_pu_xyb_planar_into`, log2/exp2_midp_precise) NOT yet
  canon — HDR path only; assess whether SDR everything() reaches it.
- `blur.rs`: `fused_blur_h_ssim_canon` (f32 arm + exact f64 arm inline) —
  replicates the generic sliding-moment chains exactly (`ssq = s.mul_add(s,
  d.mul_add(d, ssq))` etc.); hooked in `fused_blur_h_ssim_at_revision` AND
  `fused_blur_h_ssim3` (covers both entries). Plain `box_blur_*`,
  `downscale_2x*`, `box_spread_*`, `compute_xyb_mean_offset` verified
  tier-stable already (plain sequential adds; no mul_add) — no canon needed.
- `ssim_form.rs`: `ssim_dissim_exact(form, m1,m2,ssq,s12,direct) -> f64`
  (f64 mirror of `ssim_dissim_raw_scalar`/`ssim_direct_raw_scalar`).
- `fused.rs`: `VblurPools<P>`/`BandPools<P>` structs;
  `fused_vblur_ssim_canon<P>` (row-major; col sliding-sum vecs; row pools
  fin() once per row; band pools fm_*/be_* fin() at last inner row — matches
  production one-reduce-per-band lifetime); `fused_vblur_ssim_exact`
  (f64 + Neum64, uses ssim_dissim_exact); `fused_vblur_edge_canon<P>` +
  `fused_vblur_edge_exact`; hooked in `fused_vblur_features_ssim` and
  `fused_vblur_features_edge` (mode match before incant!).
  CAVEAT: canon kernel allocates `vec![f32; width]` per call — fine for
  measurement; revisit for perf.

Tier-stable already (verified by code reading; elementwise or plain-seq f64):
`abs_diff_into`, `mul_into`, `box_blur_v_from_copy`, `box_blur_h(_untiled)`,
`box_blur_1pass_into`, `box_blur_h_into_abs_diff*`, `downscale_2x*`,
`box_spread_*`, `compute_xyb_mean_offset`. era-2 dense kernel is canon by
design (`dense_block_kernel_era2`, default-on via ZENSIM_ERA2_DENSE != 0).

Still to canonicalize (Rev3-path walk leaves + restore families):
- `feature_v2.rs` block kernels: `gradient_block_kernel` 6 entry variants,
  `append_block_kernel` 3 variants, `csfw_block_kernel`, `gridblk_strip_wide`,
  `dst_y_edge_mask`, `n::*` families (ringbasis/tailhist/arttype via `n::run`
  + `finish_n_cell`), `gmsbank` (gradient `_gmsbank` variants), restore-cuts
  `mapdev`/`z1max`/`gmsnative`/`dvifmgate` (restore_maps kernel).
- `dvifm.rs`: `dvifm_push_rows_*`, `dvifm_finish_*`.
- `simd_ops.rs` ssim_signal_inline_*/ssim_channel_*: only on the Rev1 walk arm
  (`!fused_ext`); NOT reachable under Rev3+ — canon rev sits on Rev3 → skip.
- Dense: under exact mode route `dense_block_kernel` → exact oracle
  (`feature_v2::oracle::dense_reference` or f64 sibling of era2 body incl.
  `r4` extras — CHECK r4 coverage in oracle).
- Check for any other incant leaves reachable: `attr_pass_b_*` (attribution —
  verify whether everything() calls it).

## Revision plumbing (NOT yet done)

- `feature_defs.rs`: add `FormulaRevision::Rev4` = Rev3 tokens + new era
  `"tiercanon"`; `Revision` entries with era "tiercanon" on every signal def
  whose value moves (add to each family const array — REV4BANK pattern);
  `ssim_form.rs`: `active_revision()` accept "4"; ensure Rev4 selects
  Rev3 semantics (`effective_revision`, `SsimLumaForm::for_revision`,
  `direct`, `stable` flags all branch on `== Rev3` — they must treat Rev4
  as Rev3-semantics → audit every `== FormulaRevision::Rev3` comparison and
  make it >= Rev3 or add Rev4 arm).
- After measurement decides the winner, leaf hooks become
  `if active_revision() >= Rev4 || featcanon::canon_active_for_rev()`.

## Exact-oracle caveat to record in DONE report

"exact" = every leaf's arithmetic in f64 (Neumaier sums) with f32 rounding
only at plane stores — a real-arithmetic model, not bit-true reals. Elementwise
single-op leaves are exact already (correctly rounded) — skip hooks there.

## Next step (verbatim)

`cargo check -p zensim` was running when context ended (first check had
Pool-trait scope errors in fused.rs, fixed via
`use crate::featcanon::{Neum64, Pool as _};` in the two exact bodies).
Then: build `featcanon_audit`, run audit under ZENSIM_FEATCANON=c32 to see
remaining divergent families → continue leaf-by-leaf (block kernels next),
then exact-mode vector → per-family error table for a/b/c/d/e/f,
then zenbench cost, then revision wiring + tests.

## Update — canon c32 smoke (konjnd 640x480 + aic3-853x945, Rev3)

With ONLY opsin-XYB + fused_blur_h_ssim + fused_vblur_{ssim,edge} canonicalized:
- v4x: 0/1825 diffs; v4: 0/1825 diffs (both pairs)
- scalar: 8/1825 diffs — ALL `csfw_w_global_*` slots (worst rel 3.3e-5)

Interpretation: every other block kernel (gradient, append, n::*,
restore-cuts, gridblk, dst_y_edge, dvifm) is already tier-stable once its
INPUT PLANES are canonical — their divergences were inherited from XYB/blur/
fused-vblur. The only divergent leaf left is `csfw_block_kernel` (Horner
`w` via unfused scalar mul_add + f32 lanes + an f64-formula scalar tail).
Canonical csfw body written (`csfw_block_kernel_canon` + `csfw_row_canon<P>`,
exact arm in-line) — pending verification.

Note: canon mode NOT yet wired to a revision; all hooks gate on
`ZENSIM_FEATCANON` only.

## Update 2 — candidate decision + Rev4 wiring (measurements)

Full 11-pair audit (all 1825 slots), ZENSIM_FORMULA_REV=3 + mode override:
- `c32` / `c64` / `neum`: **0 diffs on every tier (v4x,v4,v3,scalar), every pair**
- `exact`: 0 diffs across tiers too (self-consistent oracle)
- Vectors dumped per (mode,tier,pair) to `~/tmp/devin/featcanon/vecs/*.f64bin`.

Error vs exact (aggregated 20075 slots×pairs; rel err):
- prod_v3 / prod_v4 / c32 / c64 / neum: max ≈ 1.1e290 (degens), p99 ≈ 1.4e-2,
  median ≈ 5.2e-7 — IDENTICAL profiles. The remaining error is element-level
  f32 rounding (XYB/SSIM formulas), not accumulation order; accumulation
  choice moves nothing measurable. prod_scalar slightly worse medians.
- Worst "errors" are saturate() degenerate slots (exact ~1e-11 vs 0.0) —
  abs values ≤ ~2e-4; all candidates identical there.

Cost (min-of-3, taskset c9, threads=1): 2000x2496 pair → prod_v3 3.38s,
prod_v4 3.05s, c32 6.15s, c64 6.58s, neum 7.29s; 4096² → 11.35/10.27/20.74/
21.53/26.75s; 384x512 → 0.101/0.086/0.211/0.186/0.255s.
**c32 is ~1.8× prod-v3** and cheapest of the candidates → CHOSEN (era-2
pattern; lane structure is SIMD-vectorizable later without arithmetic change).

c32 vs prod_v3: 837/1825 slots move (union over pairs) — era "tiercanon"
registered as "any slot may move" (era_moved_slots special case).

## Rev4 wiring done
- `FormulaRevision::Rev4` (era_tokens = Rev3 + "tiercanon");
  `ZENSIM_FORMULA_REV=4`; `paired_global_contrast` incl Rev4; all
  `== Rev3` semantics switches → `>= Rev3` (feature_v2/fused/blur/streaming/
  ssim_form check_route + for_revision arms in det_math/hf_gain_form).
- `featcanon::mode()`: env override else Rev4→Canon32.
- Verified: FORMULA_REV=4 vectors == c32 vectors on all pairs; tiers
  identical; rev4 ≠ rev3 (4177 slot-diffs summed over pairs).
- New test zensim/tests/featcanon_tier_parity.rs (rev4 bit-identical gate,
  rev3 negative control, rev4≠rev3, remainder+small geometries).

Remaining: csfw/edge libs compile clean; full `cargo test -p zensim`,
clippy, fmt; FEATCANON_DONE.md; jj commit on quarantine/devin/featcanon.

## Update 3 — verification state

- `ZENSIM_FORMULA_REV=4` (no env override) → all tiers 0/1825 diffs on all
  11 pairs (vecs bit-match the c32 dumps exactly; differs from prod_v3).
- `cargo test -p zensim --lib`: 439+446 pass, 0 fail.
- `zensim/tests/featcanon_tier_parity.rs`: 3/3 — rev4 whole-vector
  bit-identical over all token permutations at 64²/97×63/131×65/255×129;
  rev3 negative control diverges (control fires); rev4≠rev3.
- clippy `-p zensim --all-targets --all-features -D warnings`: clean
  (manual_clamp allows on canon bodies — max/min ordering is deliberate,
  clamp has different NaN/-0.0 semantics).
- Fixed downstream exhaustive matches: `det_math::RootForm`/`PowForm`,
  `hf_gain_form`, `feature_plan` test, `streaming.rs` dump test,
  `bake_verdict` revision map.

Notes for the DONE report:
- pu_xyb (HDR/log2 path) deliberately NOT canonicalized — the SDR
  everything() walk never reaches it; it stays tier-dispatched.
- `attr_pass_b` (attribution) is not on the everything() path.
- All other block kernels were already tier-stable once their input planes
  were canonical — divergence was inherited, not generated, downstream.

## Final state (2026-09-25)

- Rev4 (`tiercanon` era) lands the canonical form = candidate (d): fused
  mul_add elements + LanesF32 + era2_reduce8 — chosen because error-vs-exact
  is identical across candidates (element-dominated) and c32 is cheapest.
- Verification: FORMULA_REV=4 bit-identical across all tiers on all 11 pairs;
  featcanon_tier_parity tests 3/3; lib 446 + integration suites pass;
  workspace clippy clean; fmt clean.
- Deliverables: this worklog; FEATCANON_DONE.md; audit harness
  `zensim-validate/src/bin/featcanon_audit.rs`; dumps in
  ~/tmp/devin/featcanon/vecs/.
- Open: canon bodies are scalar-shaped (~1.8× v3); LanesF32 is SIMD-
  vectorizable later. pu_xyb/attr_pass_b intentionally not canonical.
