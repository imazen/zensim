# EXP-CROSS-CODEC-V11 ssim2-anchored substrate (task #189, 2026-05-20)

**Status: PARTIAL FALSIFICATION — substrate built honestly, ship gate failed.**

User directive 2026-05-20: "why is butter involved in this? ssim2 and cvvdp
are more reliable". Rebuild the V_CrossCodec → V6 → V9 → V10 cross-codec
substrate using ssim2 as the primary anchor and cvvdp as the cross-validation
anchor, in place of the butter_pnorm3-based substrate.

This document captures the substrate build, the structural findings about
the available data, and the ship-gate evaluation against V10 BalancedV3.

## Phase 1+2+3 substrate — built honestly

### Anchor parquet (ssim2-anchored)

**Output**: `/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/anchors_ssim2_300col.parquet`

- Rows: 8,893 × 311 cols (300 features + 11 metadata cols).
- Coverage by codec × band (anchor landings, see build log for full table):
  - **zenjpeg** (via direct ssim2 match on v15r 1.78M-row substrate, 979 imgs × 19 q): all 10 bands populated, 24-951 rows per band.
  - **zenwebp** (via cvvdp→ssim2 conversion, 200 imgs × 5 q): collapsed at high q (saturates at ssim2≈85), 4-173 rows per band at score≤65; 157 each at score≥90 (cvvdp saturated).
  - **zenavif** (via cvvdp→ssim2, 200 imgs × 5 q): 43-190 rows per band.
  - **zenjxl** (via cvvdp→ssim2, 200 imgs × 5 q): **0 rows below score=20**, 3-200 at score≥35. zenjxl saturates BOTH ssim2 (≈87 across q=10..90) AND cvvdp (≈9.985 across q=10..90) on this image corpus.

### Cross-codec equivalence (ssim2-pivoted)

**Output**: `/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/cross_codec_equivalence_ssim2.parquet`

- 3,169 pairs × 612 cols (300 features × 2 + 12 metadata).
- Pair counts:
  - zenjpeg ↔ zenavif: 573
  - zenjpeg ↔ zenjxl: 531
  - zenjpeg ↔ zenwebp: 508
  - zenwebp ↔ zenjxl: 514
  - zenavif ↔ zenjxl: 563
  - zenwebp ↔ zenavif: 480
- Uses `unified_v13_zenjpeg_cvvdp.parquet` for zenjpeg (200 imgs shared with v12 others) rather than `unified_v15r` (979 imgs, disjoint corpus).

### cvvdp cross-validation substrate (Phase 3)

**Output**: `/mnt/v/zen/zensim-training/2026-05-20-cvvdp-anchors/anchors_cvvdp_300col.parquet`

- 3,517 anchor rows + 1,365 cvvdp-pivoted equivalence pairs.
- All 4 codecs covered at 5 q-levels each (zenjxl saturated below score=35).

### cvvdp → ssim2 calibration curve

`/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/cvvdp_to_ssim2_map.json`

- 80-knot piecewise-linear median curve fit on zenjpeg unified_v15r rows
  where both ssim2 (`score_ssim2`) and cvvdp (`cvvdp_imazen_v0_0_1`) are
  non-null (1,096,100 rows used).
- Range: cvvdp 8.675 → ssim2 8.30 up to cvvdp 10.000 → ssim2 95.41.
- Anchor band landings:
  - ssim2=100 → cvvdp=10.000
  - ssim2=90 → cvvdp=9.995
  - ssim2=75 (JND) → cvvdp=9.917
  - ssim2=45 (JOD) → cvvdp=9.532
  - ssim2=18 → cvvdp=8.925
  - ssim2=3 (worst floor) → cvvdp=8.675

## Structural findings about the available data

### 1. ssim2 is computed only for zenjpeg in the high-coverage substrate

| Source | rows | q levels | imgs | ssim2 nn | cvvdp nn |
|---|--:|--:|--:|--:|--:|
| unified_v15r_zenjpeg_cvvdp | 1,785,696 | 19 (q=5..95) | 979 | 100% | 61% |
| unified_v15rc_zenjpeg_cvvdp | 513,570 | 19 (q=5..95) | 901 | 100% | 0% |
| unified_v12_zenavif_cvvdp | 4,000 | 5 (q=10..90) | 200 | **0%** | 95% |
| unified_v12_zenwebp_cvvdp | 1,000 | 5 (q=10..90) | 200 | **0%** | 100% |
| unified_v12_zenjxl_cvvdp | 32,000 | 5 (q=10..90) | 200 | **0%** | 98% |
| unified_v13_zenjpeg_cvvdp | 36,000 | 5 (q=10..90) | 200 | 100% | 97% |
| picker-training/butter/<codec> | 19,000 | 19 (q=5..95) | 1,000 | NO | NO |

**ssim2 was NEVER computed on the 3 non-jpeg codecs in this image corpus.**
The original butter substrate worked because butter_pnorm3 was computed
across all 4 codecs uniformly via the picker-training butter sweep.

### 2. zenjxl saturates BOTH ssim2 and cvvdp

Per-codec ssim2 medians at the 5 q levels (canonical scores parquet):

| codec | q=10 | q=30 | q=60 | q=80 | q=90 |
|---|--:|--:|--:|--:|--:|
| zenjpeg | 33.1 | 55.2 | 72.2 | 81.0 | 86.2 |
| zenwebp | 58.6 | 71.1 | 80.0 | 83.9 | 86.9 |
| zenavif | 17.4 | 55.3 | 81.3 | 88.0 | 90.7 |
| **zenjxl** | **87.4** | **87.6** | **87.7** | **87.6** | **87.4** |

zenjxl ssim2 is **flat at ~87 across the full q range**. Below-JND bands
for zenjxl have effectively zero rows. Same structural pattern in cvvdp
(zenjxl saturates at cvvdp ≈ 9.985 across q=10..90). This is a data
property, not solvable by changing the anchor metric.

### 3. v15r and v12 use disjoint image corpora

- v15r_zenjpeg: 979 wikimedia 512sq images (e.g. `00b13be94a4867dd_512sq.png`)
- v12_zen{webp,avif,jxl}: 200 `gen-*_1024sq.png` synthetic images.
- Intersection: **0 images**.

So zenjpeg cross-codec pairs CANNOT come from v15r — we have to use
v13_zenjpeg (200 imgs shared with v12 others) for the equivalence pool.
The v15r dense ssim2 substrate is usable for anchor rows but NOT for
cross-codec pairing.

## Phase 4 — V11-A' Balanced retrain (FAILED)

### Recipe from task brief

```bash
zensim_mlp_train \
  --group safesyn:.../safesyn.parquet:1.0:0.0 \
  --group kadid:.../kadid.parquet:0.6:0.4 \
  --group tid:.../tid.parquet:0.6:0.4 \
  --group konjnd:.../konjnd-dense.parquet:0.6:0.0 \
  --group large:.../cvvdp_iwssim_LARGE.parquet:1.0:0.0 \
  --group cid22_train:.../cid22_train.parquet:0.5:0.0 \
  --group pipal:.../pipal.parquet:0.3:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 32 \
  --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min \
  --early-stop-patience 0 \
  --max-features 300 --target-column mix_cv35_iw65 \
  --anchor-parquet .../anchors_ssim2_300col.parquet \
  --anchor-loss-weight 1.0 --anchor-step-p 0.30 \
  --cross-codec-eq-parquet .../cross_codec_equivalence_ssim2.parquet \
  --cross-codec-eq-weight 1.0 --cross-codec-rank-preserve-weight 0.2 \
  --seed S --out cc4v11a_ssim2_sS.bin
```

### Critical finding: anchor/equiv aux losses NOT WIRED on plain MLP

The trainer emits this warning at startup:

```
WARNING: --anchor-loss-weight is only wired on the per-sample-α head;
anchor data ignored on this head.
```

Code path (`zensim-validate/src/mlp_train.rs:915-921`): the anchor loss
+ cross-codec-eq + rank-preserve aux losses are gated on the
`--per-sample-alpha-head` flag. The V11-A' Balanced recipe does NOT
enable that head (it's plain MLP). Result: the anchor and equiv
parquets are loaded into memory but ignored during training.

**Implication**: V11-A' as specified cannot use the new substrate
during training. The substrate is only usable for offline spline
refitting (see Phase 5 below).

### V11-A' s1 single-seed result (plain MLP, anchor IGNORED)

Bake: `/mnt/v/zen/zensim-eval/exp_cross_codec_v11a_ssim2_2026-05-20/cc4v11a_ssim2_s1.bin`

| Corpus | V10 BalancedV3 | V11-A' s1 (plain MLP) | Δ |
|---|--:|--:|--:|
| CID22 SROCC | 0.8324 | 0.8157 | **−0.0167** |
| KADID SROCC | 0.9664 | 0.9139 | **−0.0525** |
| TID SROCC | 0.9712 | 0.8908 | **−0.0804** |
| KonJND SROCC | 0.8927 | 0.4306 | **−0.4621 (catastrophic)** |
| AIC-3 SROCC | 0.7845 | 0.8102 | +0.0257 |

The catastrophic KonJND regression is because V11-A' s1 was trained
without spline calibration metadata. KonJND uses PJND thresholds (small
positive numbers) while the bake's raw output is distance-shaped. The
mismatch destroys ranking on the PJND corpus.

Even after refitting V11 ssim2-anchored spline ON the V11-A' s1 bake
(which adds the metadata but cannot change SROCC since PCHIP is
monotone), the SROCC numbers are identical:
`benchmarks/v11_verdicts/v11a_s1_with_spline.md`.

The structural conclusion: the V11-A' recipe with new training groups
(`cid22_train`, `pipal`) and different target column (`mix_cv35_iw65`
vs V_22-mix's setup) produces a fundamentally weaker network than
V10's underlying V_22-mix-LARGE+iwssim bake. The substrate swap did
not cause the regression — the recipe change did.

### Other recipes not attempted (out of session scope)

Per-sample-α head version (`--per-sample-alpha-head`) WOULD use the
anchor + equiv substrate during training. But that's the **Compression
trail architecture**, not Balanced. Shipping it would replace V10
CompressionV3, not BalancedV3. Not attempted here per the task brief's
"V11-A' = Balanced retrain" framing.

## Phase 5 — V11 spline refit on V10 BalancedV3 base (PARTIAL WIN)

Refitted the V10 spline mechanism ONTO the existing V10 BalancedV3
bake's source network (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`)
using the new V11 ssim2-anchored parquet. This isolates the substrate
swap: same network, different spline calibration.

Output: `zensim/weights/v11_candidates/v_balanced_v11_2026-05-20.bin`

### V11 spline knots (10, sorted by raw_pred)

```
raw = -20.7717 → score = 100.0  (lossless)
raw = -20.5965 → score =  95.0
raw = -18.5952 → score =  90.0  (visually identical)
raw =  -9.2722 → score =  80.0  (JND)
raw =  -3.9632 → score =  65.0
raw =   0.5472 → score =  50.0  (JOD)
raw =   4.4304 → score =  35.0
raw =   7.9375 → score =  20.0
raw =   9.4009 → score =  10.0
raw =  10.6996 → score =   0.0  (worst-floor)
```

All 10 ssim2 bands kept monotone. Bit-exact anchor landing at each band.

### V11-spline-refit vs V10 BalancedV3

| Corpus | V10 BalancedV3 | V11 spline-refit | Δ |
|---|--:|--:|--:|
| CID22 SROCC | 0.8324 | 0.8324 | +0.0000 |
| CID22 PLCC | 0.8256 | 0.8281 | +0.0025 |
| CID22 Z-RMSE | 0.564 | 0.561 | −0.003 |
| KADID SROCC | 0.9664 | 0.9677 | +0.0013 |
| KADID PLCC | 0.9562 | 0.9654 | **+0.0092** |
| KADID Z-RMSE | 0.293 | 0.261 | **−0.032** |
| TID SROCC | 0.9712 | 0.9729 | +0.0017 |
| TID PLCC | 0.9379 | 0.9692 | **+0.0313** |
| TID Z-RMSE | 0.347 | 0.246 | **−0.101** |
| KonJND SROCC | 0.8927 | 0.8927 | 0.0000 |
| AIC-3 SROCC | 0.7845 | 0.7845 | 0.0000 |

**SROCC is invariant** (PCHIP spline is monotone). PLCC and Z-RMSE
improve on KADID and TID (the spline lands closer to the human MOS
distribution shape).

## Ship gate verdict (FAIL)

Per task brief gate:

| Metric | V10 BalancedV3 | V11 spline-refit target | Result |
|---|--:|---|---|
| CID22 SROCC | 0.8324 | ≥ 0.8374 (+0.005) | **FAIL** (0.8324, no SROCC change) |
| CID22 Z-RMSE | 0.560 | ≤ 0.530 | **FAIL** (0.561, no Z-RMSE win) |
| Other corpora SROCC | base | within −0.10 | **PASS** (all within ±0.005) |
| Anchor landing JND@ssim2≈75 | exact 80 | exact 80 | **PASS** (bit-exact at knots) |
| Anchor landing JOD@ssim2≈45 | exact 50 | exact 50 | **PASS** (bit-exact) |

**Do not ship V11 spline-refit as PreviewV0_5BalancedV4.** SROCC is
unchanged (PCHIP cannot move rank), Z-RMSE on CID22 is unchanged. The
KADID/TID PLCC wins are real but the brief's primary CID22 SROCC gate
fails.

## What the substrate IS good for

1. **Documenting the JND/JOD calibration in ssim2 terms** — useful for
   building the user-facing dial doc. Spline knots in the bake metadata
   provide a traceable ssim2 ↔ score map.
2. **Independent cross-validation** — Phase 3 cvvdp-anchored substrate
   gives a second opinion on cross-codec consistency, useful for future
   work that retrains with anchor loss enabled.
3. **Future per-sample-α head training** — when a per-sample-α head
   network is trained with `--anchor-loss-weight > 0` and
   `--cross-codec-eq-weight > 0`, this substrate plugs in directly.

## What does NOT work

- **V11-A' Balanced retrain as specified in the brief**: anchor data
  ignored, recipe produces weaker network than V_22-mix-LARGE+iwssim.
- **Spline-only refit on V10 BalancedV3**: improves PLCC modestly but
  cannot move SROCC. Fails the brief's primary +0.005 CID22 SROCC gate.
- **ssim2 as a cross-codec anchor** — zenjxl saturates ssim2 at ≈87
  across all q-levels, so cross-codec equivalence at ssim2 < 60 has
  ZERO zenjxl coverage. cvvdp has identical saturation problem.

## Honest assessment for the user

The user's critique ("ssim2 and cvvdp are more reliable than butter")
is correct **for per-pair score precision**, but **does not survive
the cross-codec multi-q substrate requirement** because ssim2 was only
computed for zenjpeg in our existing data, AND because zenjxl saturates
both ssim2 and cvvdp on the available image corpus. The butter-based
substrate worked despite butter's perceptual weakness BECAUSE it had
uniform 4-codec × 19-q coverage from the picker-training butter sweep.

To genuinely improve the cross-codec metric, the right path is:
1. **Score ssim2 on the picker-training butter sweep's 1000 images**
   for all 4 codecs (GPU compute job, ~4-6 hr on vast.ai).
2. **Replace the corpus** (drop saturating images so each codec actually
   covers the q range with non-saturated ssim2/cvvdp output).
3. **Train with the per-sample-α head** so anchor loss + cross-codec-eq
   aux losses actually fire during training.

Until those happen, the existing V10 BalancedV3 ship remains the
defensible Balanced bake.

## Outputs in this commit

- Substrate build:
  - `scripts/v_next/v11_ssim2/build_v11_ssim2_substrate.py`
  - `scripts/v_next/v11_ssim2/build_v11_cvvdp_substrate.py`
- Training driver:
  - `scripts/v_next/v11_ssim2/run_v11a_seed.sh`
- Spline calibrator:
  - `scripts/v_next/v11_ssim2/calibrate_v11_balanced_spline.py`
- Per-band CSVs:
  - `benchmarks/v11_balanced_spline_2026-05-20.csv` (V10 base + V11 spline)
  - `benchmarks/v11_balanced_v11a_s1_spline.csv` (V11-A' s1 base + V11 spline)
- bake_verdict outputs:
  - `benchmarks/v11_verdicts/v10_balanced_v3.md` (V10 baseline)
  - `benchmarks/v11_verdicts/v11_balanced.md` (V10 base + V11 spline)
  - `benchmarks/v11_verdicts/v11a_s1_plainmlp.md` (V11-A' s1 plain MLP)
  - `benchmarks/v11_verdicts/v11a_s1_with_spline.md` (V11-A' s1 + V11 spline)
- Candidate bakes (NOT shipped):
  - `zensim/weights/v11_candidates/v_balanced_v11_2026-05-20.bin`
  - `zensim/weights/v11_candidates/v_balanced_v11a_s1_spline_2026-05-20.bin`

## Substrate files on /mnt/v (preserved for future use)

- `/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/`
  - `anchors_ssim2_300col.parquet` (12 MB, 8893 rows)
  - `cross_codec_equivalence_ssim2.parquet` (6 MB, 3169 pairs)
  - `cvvdp_to_ssim2_map.json` (calibration curve)
- `/mnt/v/zen/zensim-training/2026-05-20-cvvdp-anchors/`
  - `anchors_cvvdp_300col.parquet` (5 MB, 3517 rows)
  - `cross_codec_equivalence_cvvdp.parquet` (3 MB, 1365 pairs)
