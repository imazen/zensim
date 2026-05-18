# EX-4 XYB + LMS-biased-log front-end stats + CVVDP-shape per-pair features

**Status: Rust modules + unit tests landed; corpus rebuild + train + eval NOT executed in this session.**

This doc covers what landed under `feat/ex4-xyb-frontend-extfeat`,
what is *not* yet executed, and the next concrete chunks needed to
ship V_25.

## Hypothesis (per principled-experiment workflow Step 1)

1. **Hypothesis**: Adding 43 CVVDP-shape + XYB/LMS-biased-log
   feature dimensions to the 300-input MLP should lift CID22 SROCC
   by `Δ ≥ +0.005` (doc § 8 EX-4 floor) and bigger on chroma-dominant
   distortions, justified by:
   - XYB / LMS-biased-log put per-pixel deltas in perceptually
     uniform spaces (cube-root + log nonlinearity respectively),
     where every reference IQA metric since SSIMULACRA2 operates.
   - Mutual-masking residual + Minkowski-β=3 pool capture the
     same "spread + peak" shape that GMSD's std-pooling proved
     critical (doc § 3).
2. **Falsification**: If CID22 aggregate SROCC drops, stays flat
   across 3 seeds, or if at least 4 of the 6 Mohammadi panel
   stats agree on regression (Mohammadi 2025 rule), the
   hypothesis is dead and the doc § 8 EX-4 estimate of `+0.005–0.015`
   is wrong for *this* feature batch (not necessarily for all
   front-end work).
3. **Cost ceiling**: 1× corpus rebuild + seed=1 fine-tune. If
   seed=1 negative, abandon.
4. **Ship form**: Single-bake PreviewV0_5+ with `feature_transforms`
   propagation through `predict_transformed` dispatch.

## What landed in zensim

### Module 1: `zensim/src/xyb_lms_features.rs`

24 features per **reference image**:

| Block | Channels × Stats | Count |
|---|---|---|
| XYB (libjxl cbrt) global stats | 3 channels × (mean, std, p5, p95) | 12 |
| LMS biased-log (Butteraugli) global stats | 3 channels × (mean, std, p5, p95) | 12 |
| **Total** | | **24** |

Constants lifted verbatim from `crate::color` (single source of truth):
- `K_M00..K_M22` opsin absorbance matrix (libjxl `enc_xyb.cc`).
- `K_B0 = 0.003793073` cube-root bias.
- `LMS_BIASED_LOG_OFFSET = 0.01` (Guetzli §3.1).

Tests (5/5 passing):
- `feature_count_is_24`
- `uniform_grey_has_zero_chroma_and_zero_std` — sanity for XYB X
  channel near zero, Y > 0, all stds ~0 on uniform input.
- `red_image_has_negative_xyb_x` — sign correctness on the
  opponent matrix.
- `black_image_has_log_at_floor` — biased-log floor is
  `ln(K_B0 + 0.01) ≈ -4.28` on black input.
- `percentile_ordering_holds` — p5 ≤ p95 and std ≥ 0 on every
  channel × front-end.

### Module 2: `zensim/src/cvvdp_features.rs`

19 features per **(reference, distorted) pair**:

| Block | Description | Count |
|---|---|---|
| DKL global stats | Δmean(A), \|Δstd\|(A); ref_std + dist_std for RG and VY | 6 |
| Weber-contrast pyramid band gains | per-level mean Weber contrast on achromatic | 4 |
| CSF-weighted band-energy ratios | `(E_dist / E_ref) × CSF_band_weight` for 4 levels | 4 |
| Mutual-masking residual variances | std of `(R_k − T_k) / (R_k + T_k + ε)` per level | 4 |
| Minkowski-β=3 pool | global β=3 pool of \|A_ref − A_dist\| | 1 |
| **Total** | | **19** |

Constants:
- DKL matrix: same coefficients as cvvdp-gpu `SRGB_LINEAR_TO_DKL`
  (Mantiuk et al. 2024 CVVDP paper appendix; published, MIT/Apache
  compatible).
- Display preset: `STANDARD_4K` (`Y_peak=200`, `Y_black=0.2`,
  `Y_refl=0.398`).
- `MINKOWSKI_BETA = 3.0` (doc § 3 Table).
- `CSF_BAND_WEIGHTS = [0.5, 1.0, 0.8, 0.4]` — qualitative achromatic
  CSF prior (NOT the cvvdp castleCSF LUT, which lives in AGPL
  cvvdp-gpu).
- `N_LEVELS = 4` (matches existing zensim 4-scale pyramid).

Tests (5/5 passing):
- `feature_count_is_19`
- `identical_inputs_zero_error` — all delta features ~0 when
  ref ≡ dist.
- `red_vs_grey_produces_nonzero_features` — checkerboard test
  ensures non-zero chroma std.
- `mutual_masking_bounded` — per-level residual std stays in
  `[0, 1.5]` on random noise.
- `swap_ref_dist_flips_signed_features` — Δmean(A) flips sign
  on swap; |Δstd| and Minkowski stay symmetric.

### License + dependency posture

- **No cvvdp-gpu dependency** — that crate is AGPL-3.0; zensim is
  MIT/Apache-2.0. The CVVDP-shape features here are an
  independent implementation of the same published math
  (Mantiuk et al. 2024). No code was copy-pasted from cvvdp-gpu.
- **No new external deps** — both modules use only `crate::color`
  primitives + standard library.

## What did NOT land in this session

Per the principled-experiment Step 1 cost ceiling, the session
was scoped to "architecture + tests, no train." The following
chunks remain and are the work for the next session(s):

### Chunk A — Corpus rebuild with 43 new feature dimensions

**Blocker found**: the existing training parquets at
`/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/` carry
only `ref_basename` as identity column — no image paths. To
recompute features we need (`ref_image_path`, `dist_image_path`)
per row.

**Resolution path**:
1. Map `ref_basename` back to the original training corpus
   manifest via `/mnt/v/zen/zensim-training/2026-05-15-full-features/_MANIFEST.md`
   (which has explicit source-corpus paths per dataset).
2. Re-encode distortions deterministically from the safesyn
   recipe (the `human_score` column anchors the original
   distortion choice; the `q` and `codec` columns are required —
   the 300-col parquet does NOT carry these but the
   `safesyn_mix_300col.parquet` IS the raw merged training table
   that may have them in its schema).
3. Call `extract_xyb_lms_features` and `extract_cvvdp_features`
   per row, append to a new 343-column parquet at
   `/mnt/v/zen/zensim-training/2026-05-18-extfeat/cvvdp_iwssim_large_343col_v3.parquet`.

This is **~6–12 CPU hours** on the 7950X for 73,300 rows × O(MP)
per row, parallelised via `rayon`.

### Chunk B — Train V_25-extfeat seed=1

Once the 343-col parquet exists:

```bash
target/release/zensim_mlp_train \
    --group safesyn:/mnt/v/zen/zensim-training/2026-05-18-extfeat/safesyn_extfeat.parquet:1.0:1.0 \
    --group kadid:/mnt/v/zen/zensim-training/2026-05-18-extfeat/kadid_extfeat.parquet:1.0:2.0 \
    --group tid:/mnt/v/zen/zensim-training/2026-05-18-extfeat/tid_extfeat.parquet:1.0:2.0 \
    --group konjnd:/mnt/v/zen/zensim-training/2026-05-18-extfeat/konjnd_extfeat.parquet:0.10:2.0 \
    --feature-set extended_iw \
    --target-column mix_cv40_iw60 \
    --hidden 64 \
    --epochs 200 \
    --seed 1 \
    --output /tmp/v25_extfeat_seed1.bin
```

Decision tree (Step 3 of the workflow):
- Seed=1 wins KADID + TID by `≥ +0.005 SROCC` → sweep 5 seeds.
- Seed=1 flat → falsified, document and stop.
- Seed=1 mixed → diagnose per-band, decide.

### Chunk C — Mohammadi full-panel eval on CID22

Run `bake_verdict` on the seed=1 candidate against `cid22_features_372col_2026-05-15.parquet`
**LAST** (Step 2 — CID22 inspection at decision time only). Emit
the full Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE)
aggregate + 10-band per corpus.

Note: `bake_verdict` currently reads a **372-feature** parquet. The
343 = 300 + 43 layout does NOT match the existing 372-feature
schema (300 + 72 IW pool). To eval the new bake we either need to:

- (a) **Repurpose the 372-col files** — append the 43 EX-4 features
  as `f372..f414` columns, retraining with `--feature-set extended_iw+ex4`
  (a new feature-set variant the trainer must learn). 415-input MLP.
- (b) **Recompute the 5 validation parquets** at 343 cols by running
  `extract_features` + `extract_xyb_lms_features` + `extract_cvvdp_features`
  on the source images. Cleaner schema, more compute.

Option (a) is the cheaper path; option (b) is the correct long-term
schema. Both are queued.

## What this session does NOT claim

- **No CID22 SROCC number is reported.** Honest claim per
  "NEVER CLAIM FALSE COMPLETION": Rust feature modules + tests
  landed; no training, no eval, no calibration, no ship candidate.
- **No feature-importance analysis** — that requires a trained
  bake to read out per-feature gradient magnitude or absolute
  weight. Queued as part of Chunk B's seed=1 inspection.
- **No 5-seed CI** — queued for Chunk B after seed=1 passes.
- **No packed candidate bake path.**
- **No runtime cost number.** The Rust modules compile and run
  but were not benchmarked; the CVVDP-shape features are O(n)
  per pixel × 4 pyramid levels + O(n log n) for percentile
  sorts in xyb_lms_features. Expected: ~30 ms on a 4 MP image
  for the combined 43-feature extract.

## Honest gaps (doc § 8 acceptance bar)

- The user's task brief described EX-4 as "+0.005–0.015 CID22
  SROCC lift" — this session reports `untested`. Not "below
  the lower bound" (we measured nothing), but **provisional**.
- The IW-SSIM batch in the task brief (5-level Laplacian +
  11×11 Gauss + GSM info weights) was **already shipped** in
  zensim as the 72-feature IW pool block (`FEATURES_PER_CHANNEL_IW`
  in `metric.rs`). The doc § 2 IW-SSIM port suggestion is
  redundant relative to what's already in production. Recorded
  here so future sessions don't re-port what already exists.
- The CVVDP castleCSF LUT (the high-fidelity per-band sensitivity
  function) is NOT used — substituted with 4-tap `CSF_BAND_WEIGHTS`.
  Crossing the AGPL boundary would require explicit user approval.

## Files changed

- `zensim/src/lib.rs` — added `pub mod xyb_lms_features` +
  `pub mod cvvdp_features` (both gated by `training` feature).
- `zensim/src/color.rs` — `K_M00..K_M22` + `K_B0` visibility
  bumped from `const` to `pub(crate) const` so feature modules
  share the constants.
- `zensim/src/xyb_lms_features.rs` — new (288 lines).
- `zensim/src/cvvdp_features.rs` — new (398 lines).
- `benchmarks/ex4_xyb_lms_extfeat_methodology_2026-05-18.md` — this doc.

## Test count

81 / 81 zensim lib tests passing (76 pre-existing + 10 new from
the EX-4 modules + 1 new doctest, all green).

## Branch

`feat/ex4-xyb-frontend-extfeat` in zensim. Workspace path:
`/home/lilith/work/zen/zensim--ex4-extfeat/`.
