# Folded-720 extraction — fold v1-basic into the v2 walk, skip the ablated pools (2026-07-24)

**Ask (user):** fold the v1 features the latest models actually use into v2,
skipping the ablated ones, for performance.

**Result: ONE v2-walk pass now emits the 720-layout the latest models consume —
`[f0..156) = v1 basic | [156..372) = 0.0 (deprecated pools) | [372..720) = v2-348`
— at 1.86× ext-720 single-thread (59.9 vs 111.3 ms/pair, aic3-100).** The v2
block is bit-identical to the plain v2 path; the v1 basic block is
**bit-identical to the frozen v1 path when `simd_padded_width(w) == w`**, and
elsewhere diverges only by v1's own padded-width semantics — measured through
the true foldable clean model at **0.058 dial-points mean / 0.37 max** (inside
the dial's noise floor) with **0/100 corruption-gate flips**.

Host: 7950X (Zen4, AVX-512), WSL2. Single-thread runs serialized. Commit: (this
change; parent `0189ecaa`). Pairs: first 100 of
`/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv` (~1–5 MP photos).

## Why this feature set

Every current-generation model reads exactly **v1 basic-156 and none of v1's
f156..371** (peaks/masked/IW pools):

- ideal clean perceptual model (dial+diffmap): basic-156 ++ foldable v2
  (`ideal_clean_model_2026-07-24.md`);
- corruption head: foldable-384 = the same subset
  (`corruption_head_2026-07-24.md` — "the f156..371 block can be dropped
  entirely; neither head needs it");
- top5 replication (winner_dial / Ebothg / ADD156): 504 = basic-156 ++ v2-348
  (`top5_v2_replication_2026-07-23.md`).

The fold makes that the *extraction* reality: pools are never computed, and the
v1 basic block rides the v2 walk's shared work.

## v1 cost anatomy (what skipping buys) — aic3-100, 1 thread, decode ≈ 2.10 s

| v1 mode | wall | compute ms/pair |
|---|--:|--:|
| 372 (extended+IW, the fleet config) | 7.57 s | 54.7 |
| 300 (extended only) | 7.46 s | 53.6 |
| 228 (basic+peaks; masked+IW skipped) | 5.70 s | 36.0 |

IW is nearly free (+1.1 ms — it shares the activity blurs); the
**masked/activity path costs 17.6 ms/pair**; the remaining 36 ms duplicates
blur machinery the v2 walk already runs on the same pyramid. Flag-only skipping
(v1s + v2 two-pass) would land ≈ 84.5 ms/pair = 1.32×; the true fold reaches
59.9 ms/pair by sharing decode, XYB, pyramid, ref-prep, and the H-blur.

## The fold (implementation)

`Zensim::compute_folded720_features[_with_ref_and_scratch]`
(`feature-regime-v2`-gated; `FeatureRegime::Folded720`): inside
`compute_channel_scale_v2`'s strip loop, after the v2 blur pass fills the
H-planes (`fused_blur_h_ssim` — the SAME kernel v1 uses, same radius 5), v1's
own `fused_vblur_features_ssim` runs over those H-planes and accumulates the 13
basic sums per channel-scale; finalize replicates v1's pooling + assembly
bit-for-bit (`V1BasicSums::finalize_into`). The moments-cached blur pass swaps
`fused_blur_h_ssim3 → fused_blur_h_ssim` when folding (the fold kernel needs
`mu1_h`, whose chain ssim3 compiles out; the other 3 outputs are bit-identical).

**The load-bearing subtlety — v1's band tiling is part of its numerics.** v1
V-blurs every scale in 32-row bands (`STRIP_INNER`), each from a buffer
extending `overlap = blur_passes×radius = 5` rows past the band, clamped at the
plane; the f32 sliding V-window state re-initializes at every band's buffer
top. A first fold attempt ran one kernel call per 128-row v2 strip: per-pixel
mu/σ then differ at f32-ULP scale, and the C2-stabilized SSIM division
amplifies ~1000× → observed rel diffs up to 5e-3 at deep scales. The fix,
`fold_v1_basic_bands`, **replays v1's exact band tiling** (128 = 4×32, bands
nest in strips; same buffer extents, same init points, same merge order) over
the shared H-planes — after which 16-multiple-width fixtures are **BIT-EXACT**
(`folded720_v1_basic_matches_v1_path`, `to_bits` equality).

## Parity contract (the honest boundary)

v1's frozen semantics compute over `simd_padded_width(w)`: mirrored pad columns
participate in pooling and the downscale chain halves the PADDED width, shifting
every deeper scale's grid (`metric` walk: `let mut w = padded_width`). A
true-width shared walk cannot reproduce that — and `simd_padded_width` bumps
≥512 even-16-multiples (1024→1040, 1920→1936), so **most production images are
in the divergent class** (aic3-100: 100/100). This is v1's pad wart, not fold
noise; the fold's basic block is v1-basic-at-true-width.

Measured consequences (aic3-100, all divergent-class):

| consumer | ext-vs-fold Δ | verdict |
|---|---|---|
| clean foldable model, raw (`ideal_smoothpow_p0p2.bin`) | mean 1.6e-3, max 6.1e-3 raw | — |
| same, [0,100] dial-splined | **mean 0.058 pts, p95 0.16, max 0.37** | inside dial noise (median backward step 0.28 pts) |
| corruption head foldable-384, P(corruption) | p95 0.000, max 1.4e-2 | **0/100 gate flips** at T=0.95 |
| winner_m504 MLP (rank reference) | mean 8.7e-3, max 3.2e-2 raw (≈0.9/3.2 pts) | MLPs amplify — refit/revalidate before serving folded |

Per-feature basic-block diffs on this content: p95 2.3e-2 rel (deep-scale
ssim features carry the padded-grid shift; near-zero cells dominate max-rel
stats meaninglessly). Report JSON:
`/mnt/v/output/zensim/fold-parity-2026-07-24/fold_parity_report.json`;
analysis: `scripts/v_next/fold_parity_check_2026-07-24.py`.

**Regime rule:** folded rows are their own extraction regime. NEVER mix them
into v1-extracted corpora (same columns, different padded-width semantics +
zeroed pools). New training/calibration rounds should extract folded;
existing bakes keep ext-720 until refit or re-validated.

## ⚠ Side-find: the RESOLUTION perceptual artifact is NOT foldable-only

`ideal_p0p2_L0p003_F0p005.bin` (the mono-regularized "clean perceptual model"
pick of `ideal_clean_model_2026-07-24.md` RESOLUTION) measures **heavy pool
dependence**: zeroing f300..372 (IW) alone moves its raw output by mean 0.40
(≈40 dial points); peaks 0.088, masked 0.19. The frontier row claims "foldable
BVLS" — either the artifact on disk is not the documented fit, or the
regularized refit silently dropped the foldable mask. Its M3=0.58 claim needs
re-scrutiny for the same reason. By contrast `ideal_smoothpow_p0p2.bin`
measures **exactly 0.0** on all three pool blocks (truly foldable-only). Until
resolved, the L0p003 artifact must not be treated as the one-shared-extraction
perceptual model.

## Timings (aic3-100, grouped + moments)

| mode | 1-thread | ms/pair | 8-thread driver wall |
|---|--:|--:|--:|
| ext-720 (v1-372 ++ v2-348, two passes) | 11.13 s | 111.3 | 2.03 s |
| **folded-720 (one pass)** | **5.99 s** | **59.9** | **1.52 s** |
| (v2-348 alone, reference) | 4.85 s | 48.5 | — |

Fold marginal cost for the v1-basic block ≈ 11.4 ms/pair (vs 36.0 for v1's own
228 pass — the shared H-blur/pyramid/decode/ref-prep is the win). 8-thread
wall is contention-bound (1.34×); single-thread is the per-pair truth (1.86×).
Driver: `ZENSIM_AB_MODE=fold` in `v2_ab_extract` (also gained `v1e`/`v1s`
timing modes).

## Tests

- `folded720_v1_basic_matches_v1_path` — BIT-EXACT basic block at
  `simd_padded_width(w)==w` (96×64, 64×300, 208×144); documented divergence
  class asserted finite + printed (127×93, 200×150); pools all-zero + v2 block
  bitwise, both classes.
- `folded720_sub64_matches_padding_v1_entry` — sub-64 reflect-pad path,
  bit-exact vs the padding v1 entry.
- `folded720_ref_paths_bit_identical` — pair / prepared / prepared+moments all
  720 slots bitwise-equal; folded `view()` = v2 tail.
- Full suite green incl. `v1_golden_bytes` (v1 path untouched).

## Follow-ons

1. Resolve the `ideal_p0p2_L0p003_F0p005.bin` pool-dependence contradiction
   (refit foldable-masked with the mono-regularizer, or correct the doc).
2. Wire zenmetrics' jobexec/sidecar path to `compute_folded720_features_with_
   ref_and_scratch` for production extraction (the "ONE set → content-addressed
   sidecar" plan) once a fold-extracted training round lands.
3. If a bit-exact-everywhere fold is ever demanded: the only route is replaying
   padded-width planes for the v1 block (own pyramid, no H sharing) ≈ 1.3×
   total — measured not worth it vs 1.86× + sub-noise model deltas.
