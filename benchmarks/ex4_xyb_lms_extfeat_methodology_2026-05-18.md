# EX-4 XYB + LMS-biased-log front-end stats + CVVDP-shape per-pair features

**Status: FALSIFIED at seed=3 on the 24-feature per-ref-only subset.**
**Corpus rebuild + V_25 seed=3 train + Mohammadi full-panel eval landed
this session. Per-pair 19 CVVDP-shape features DEFERRED — dist
images for the 73k LARGE corpus rows are not on disk locally
(distortions were generated on vast.ai workers and not synced back).**

## Verdict (seed=3, 5-group recipe, 324-feature input)

| Corpus | V_22 baseline (s3, packed) | V_25-extfeat (s3) | Δ |
|---|---|---|---|
| CID22 SROCC | 0.8324 | **0.8171** | **−0.0153** |
| KADID SROCC | 0.9677 | **0.8999** | **−0.0678** |
| TID SROCC | 0.9729 | **0.8822** | **−0.0907** |
| KonJND SROCC | 0.8927 | **0.8108** | **−0.0819** |
| AIC-3 SROCC | 0.7845 | **0.7701** | **−0.0144** |

The doc § 1 hypothesis floor was Δ ≥ +0.005 CID22 SROCC. The measured
delta is **−0.015 CID22**, **−0.068 KADID**, **−0.091 TID** — universally
worse, well outside any plausible noise envelope.

Per principled-experiment workflow Step 3 ("Seed=1 flat or negative →
hypothesis dead. Do NOT sweep 5 seeds — that is p-hacking."), the
24-per-ref-feature direction is **falsified** at seed=3 and NOT swept
to 5 seeds.

## Mechanism — what the trainer actually did

`zenpredict inspect --weights` on the V_25 seed=3 bake shows the
trainer DID learn substantial weights on the 24 EX-4 features:

| Statistic | Base (f0..f299) | EX-4 (f300..f323) |
|---|---|---|
| Per-column L2 norm (mean) | 1.282 | 1.236 |
| Per-column L2 norm (max) | 1.668 | 1.431 (lms_M_p5) |
| abs_mean (all weights) | 0.0851 | 0.0832 |
| Near-zero fraction (|w|<0.01) | — | 10.2 % |

The EX-4 features are NOT inert — the trainer assigned them comparable
weight magnitude to the base features. The performance regression is
therefore **not** "wasted capacity"; it's **misleading capacity**.

The structural problem: XYB+LMS per-ref features describe the
**reference image**, NOT the (ref, dist) difference. RankNet's
pairwise objective compares pairs with the *same* reference, so all
24 EX-4 features are **identical** within a pair. They cannot carry
ranking signal directly. What they CAN do is encode per-content-class
biases in the synth-corpus MOS distribution (e.g., "gen-chart refs
get systematically higher q for the same human_score than gen-photo
refs"). The trainer learns those biases — and they fail on held-out
content (CID22's 49 refs, KADID's 81 refs, TID's 25 refs, KonJND's
1008 refs — none of which overlap the safesyn training refs).

This is FRIQUEE-2017-shaped: synth-corpus per-image content priors
do not transfer to authentic-distortion validation. Same mechanism
as V_20b's falsification (cycle-13). The per-ref XYB+LMS features
make the problem worse, not better, because they give the trainer
explicit handles for content-class memorization.

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

## What this session executed (continuation 2026-05-18)

The prior agent landed Rust modules + unit tests. This agent executed:

1. **Corpus rebuild** (~10 min wall, 16-thread rayon):
   - `extract_ex4_features` binary added (`zensim-validate/src/bin/`)
   - Per-ref XYB+LMS 24 features computed once per unique
     `ref_basename` via cached lookup, then broadcast to all rows.
   - 9 corpora rebuilt (5 train + 4 validation):
     * safesyn: 196,086 rows × 3218 unique refs → 66 s wall
     * cvvdp_iwssim_large: 73,300 rows × 200 unique refs → 5 s wall
     * kadid (train + val): 10,125 rows × 81 refs → 1.6 s
     * tid (train + val): 3,000 rows × 25 refs → 0.6 s
     * konjnd (train + val): 1,008 rows × 1008 refs → 9 s
     * cid22 (val): 4,292 rows × 49 refs → 0.9 s
     * aic3 (val): 600 rows × 10 refs → 1.5 s
   - Output at `/mnt/v/zen/zensim-training/2026-05-18-extfeat/`
   - All 24 new feature columns are bounded, finite, with
     plausible per-channel distributions (XYB Y mean ~0.45 ± 0.15,
     LMS biased-log mean ~−1.8 ± 0.8). No NaN columns; no
     pathologically-zero columns.

2. **Schema normalization**: V_22 trainer used `--max-features 300`
   (dropped IW pool from 372col parquets). To preserve V_22's exact
   base-feature set + add EX-4 features at f300..f323 cleanly, the
   324-col parquets were built by dropping f300..f371 (IW pool block
   from 372col) and renaming f372..f395 → f300..f323. Trainer then
   sees f0..f299 = V_22's exact base features + f300..f323 = EX-4.

3. **V_25 seed=3 train**: 324-feature input, 128 hidden,
   5-group (safesyn 1.0, kadid 0.3, tid 0.3, konjnd 0.02,
   cvvdp_large 0.5), mix_cv40_iw60 target, PWRC + Norm-in-Norm 0.1,
   minibatch=256. Early-stopped at epoch 180, val_mean=0.8108,
   wall ~4 min.
   Bake: `/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v25_extfeat_mix_cv40_konjnd_0_02_LARGE_iwssim_h128_s3.bin`
   (169,732 bytes f32 unpacked).

4. **bake_verdict full Mohammadi panel** on all 5 corpora:
   see "Verdict" section above.

5. **Feature-importance analysis** via `zenpredict inspect --weights`:
   see "Mechanism" section above.

## What this session does NOT include

- **No 5-seed CI** — falsified at seed=3 per Step 3 decision tree.
- **No CVVDP-shape per-pair feature corpus rebuild.** The 19 per-pair
  features need both ref AND dist images. The 73k LARGE corpus dist
  images live on vast.ai paths not synced locally; regenerating them
  requires re-running the v15 sweep cluster. KADID / TID / KonJND /
  CID22 dist images ARE locally available — per-pair feature
  extraction on those 4 corpora is feasible in ~2-4 hr wall but was
  NOT executed in this 150-min budget. Queued as Chunk C below.
- **No packed candidate bake.** A failing-direction bake should not
  ship.
- **No runtime cost benchmark.** The 24 EX-4 features are per-ref
  and computed once per scoring session (not per pair), so runtime
  cost is ≈ 1 ms per ref image. Not measured precisely because the
  feature set is falsified.
- The IW-SSIM batch in the task brief (5-level Laplacian +
  11×11 Gauss + GSM info weights) was **already shipped** in
  zensim as the 72-feature IW pool block (`FEATURES_PER_CHANNEL_IW`
  in `metric.rs`). The doc § 2 IW-SSIM port suggestion is
  redundant relative to what's already in production. Recorded
  here so future sessions don't re-port what already exists.
- The CVVDP castleCSF LUT (the high-fidelity per-band sensitivity
  function) is NOT used — substituted with 4-tap `CSF_BAND_WEIGHTS`.
  Crossing the AGPL boundary would require explicit user approval.

## Chunk C — Per-pair CVVDP-shape features (queued; not executed)

The 19 per-pair CVVDP-shape features remain the load-bearing
candidate from doc § 8 EX-4 because they DO carry per-distortion
signal (RankNet pairs share the ref but differ in the dist; the
DKL Δ-stats / Weber pyramid / mutual-masking residual / Minkowski
pool all depend on the dist).

For KADID, TID, KonJND, CID22 (val) the dist images are locally
available with known on-disk paths:
- KADID: `/mnt/v/dataset/kadid10k/images/<I_xx_yy_zz>.png` via dmos.csv row order
- TID: `/mnt/v/dataset/tid2013/distorted_images_png/<iXX_YY_Z.png>` via mos_with_names.txt
- KonJND: `/mnt/v/datasets/KonJND-1k/KonJND-1k/distorted_image/...` (verify exact layout)
- CID22: `/mnt/v/dataset/cid22/CID22_validation_set/compressed/...` (verify exact layout)

For safesyn / cvvdp_iwssim_large the dist images are NOT locally
available (vast.ai workers). Options to extend Chunk C:
- (a) Skip the per-pair features for safesyn / cvvdp_large, fill
  zero columns. Falls back to "if EX-4 helps anywhere, it'll show
  up on KADID/TID/KonJND/CID22; safesyn supervision is just the
  base shape."
- (b) Local re-encode of the 73k cvvdp_iwssim_large rows. Each row
  has (image_path, codec, q, knob_tuple_json) recoverable via
  rejoin with cvvdp_imazen_consolidated + iwssim_imazen_consolidated
  parquets on `(basename, iwssim, cvvdp_score)` keys (verified
  unique-key join in this session: 65,970 unique tuples cover
  73,300 rows with ~10 % many-to-many on duplicate scores).
- (c) Sync dist images from tower (NFS at /mnt/tower if present;
  this session didn't check).

Chunk C is the **right next experiment** for the EX-4 direction —
the falsified per-ref-only result here doesn't transfer to
per-pair-feature trial.

## Files changed (continuation)

- `zensim-validate/src/bin/extract_ex4_features.rs` — new (260 lines).
  Reads input parquet, extracts 24 per-ref XYB+LMS features with
  rayon-parallel ref-cache, writes 324-col output parquet, emits
  per-feature distribution sanity stats.
- `scripts/v_next/v25_extfeat_launch.sh` — V_25 5-group launch
  matching V_22-mix-LARGE recipe.

## Test count

81 / 81 zensim lib tests passing.

## Branch

`feat/ex4-xyb-frontend-extfeat` in zensim. Workspace path:
`/home/lilith/work/zen/zensim--ex4-extfeat/`. Pushed to origin
through commit `66fbebf5`.
