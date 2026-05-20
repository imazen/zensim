# PreviewV0_5TunerV3 (V7 empirical anchor target) methodology — 2026-05-19

**Status:** FALSIFIED (does NOT ship). See
`benchmarks/v_tuner_v7_2026-05-19_verdict.md` for the full verdict.
**Bake:** N/A — V6 ship retained as `PreviewV0_5TunerV2`. V7 bakes
remain at `/mnt/v/zen/zensim-eval/exp_cross_codec_v7_2026-05-19/`
for forensic reference.
**Ship gate (per Tuner trail in `SOTA_TRAILS.md`):**
1. strict monotonicity ≥ 1 pp better than every V0_5 rank-trail ship
   on the JPEG 50-image × 19-q sweep (V6 reference bar: 0.9522).
2. tied rate ≤ 5 %.
3. median range ≥ 50 score units.
4. T=63 butter_p3 < 2.5.
5. PJND cross-codec cc_std_median ≤ 5.
6. All-band cross-codec cc_std_median ≤ 5 at every band.

PLUS V7-specific advisory check:
7. Per-(codec, band) predicted achieved_mean within ±5 of the
   empirical target (sanity that the trainer actually learned the
   empirical targets, not its own internal shape).

## Hypothesis

V6 (`PreviewV0_5TunerV2`, commit `1dd61fc`) passed all 6 Tuner-trail
gates using a multi-band anchor parquet with rule-of-thumb
`target_score` ∈ {90, 75, 63, 45, 25, 10}. Only the 63 anchor at
butter=1.5 had empirical grounding (CID22 paper, PJND ≈ 63 in
ssim2 units). The other 5 were vibes-from-memory.

**V7 hypothesis**: replacing rule-of-thumb anchors with empirical
per-(codec, band) median ssim2 (from the canonical score parquets)
should produce a bake whose predicted output at each band lands
within ±5 of human-validated metric medians at that band. V6's
predicted band means (per V6 methodology table 1) sit near the
rule-of-thumb values, which empirically are systematically LOW —
V7 should shift the bake up at bands 0.8 / 1.5 / 2.5 / 4.0 / 6.0
to match what ssim2 actually returns there.

If V7 passes all 6 V6 gates AND the new V7-specific within-±5
check: ship as `PreviewV0_5TunerV3`. If V7 fails: document that V6
architecture is good but the recipe is sensitive to anchor target
shape — V8 design space opens up.

## V6 rule-of-thumb vs V7 empirical anchor targets

From `benchmarks/v_tuner_v7_anchor_target_comparison_2026-05-19.md`,
aggregate (median across codecs):

| butter_pnorm3 | V6 rule | V7 empirical ssim2 | Δ |
|---:|---:|---:|---:|
| 0.3 | 90.0 | 87.58 | -2.42 |
| 0.8 | 75.0 | 85.33 | +10.33 |
| 1.5 | 63.0 | 77.69 | +14.69 |
| 2.5 | 45.0 | 62.42 | +17.42 |
| 4.0 | 25.0 | 55.14 | +30.14 |
| 6.0 | 10.0 | 34.64 | +24.64 |

V6's targets are systematically low across all bands except 0.3.
At butter=4.0 the gap is 30 score units. The structural cause:
butter saturates earlier than ssim2 (a butter of 6.0 is well past
butter PJND but ssim2 still scores ~35 on average). The V6
rule-of-thumb embedded an assumption that butter-bands correspond
to ssim2-bands of the same "label intensity" — which is false.

Per-codec divergence is also large. At butter=1.5:
- zenjpeg: 79.43
- zenwebp: 75.38
- zenavif: 75.95
- zenjxl: 87.85 (anomalous — see "Open questions")

The V7 anchor parquet emits per-(codec, band) targets so the
trainer learns these codec-specific differences directly.

## Trainer matrix

3 seeds × 1 anchor weight (1.0) = 3 bakes. Identical to V6 ship
recipe except the anchor parquet pointer.

| Bake | seed | anchor_w | anchor_step_p |
|---|---:|---:|---:|
| cc4v7_s1 | 1 | 1.0 | 0.30 |
| cc4v7_s2 | 2 | 1.0 | 0.30 |
| cc4v7_s3 | 3 | 1.0 | 0.30 |

Out dir: `/mnt/v/zen/zensim-eval/exp_cross_codec_v7_2026-05-19/`.

Driver: `scripts/v_next/run_cross_codec_v7_seed.sh <seed>`.

## Full trainer command (per bake, only seed differs)

```sh
target/release/zensim_mlp_train \
    --group safesyn:/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet:1.0:0.0 \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet /mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet /mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed ${SEED} --out ${OUT_BAKE} --log-path ${LOG}
```

## Inputs (md5 / row count)

- Train corpus: `/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet`
  (196,086 rows × 372 features, sha256 prefix `1ee0565fb6cb` per
  canonical CLAUDE.md).
- Multi-band empirical anchor:
  `/mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet`
  (18,457 rows × 381 cols, 6 butter bands × 4 codecs × ~1000
  sources, per-(codec, band) `target_score` = empirical ssim2
  median from canonical score parquets).
- Cross-codec equivalence pool: identical to V6 ship,
  `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`.

## Anchor parquet build (V7-specific)

Source: `scripts/v_next/build_empirical_anchor_parquet.py` (commit
`68886694`).

Algorithm per (codec, band):
1. Read the butter parquet for the codec; filter to rows with
   `butter_pnorm3 ∈ [band ± 0.5]`.
2. For each row, look up `ssim2_gpu` from
   `canonical-2026-05-18/scores/ssim2_imazen.parquet` by
   `(basename, codec, q)`. Drop rows with no ssim2.
3. Compute the median ssim2 over the cell → that's the band's
   per-codec `target_score`.
4. Emit per-(image, codec, band) anchor rows with the per-(codec, band)
   target_score (so the same image at the same butter for different
   codecs gets a slightly different target — which is what we want).

cvvdp is computed in parallel for the comparison table (using the
safesyn-corpus -log(10-cvvdp) min-max normalization) but is NOT used
as the trainer's target_score, because cvvdp_log_norm in safesyn
units lives in ~10-40 across the compression regime — structurally
below ssim2's 0-100 range. Mixing them as a 50/50 joint would bias
the anchor down for no good reason.

## Per-bake gate results

| Bake | mono ≥ 0.9522 | tied ≤ 5% | medRange ≥ 50 | T63 butter_p3 < 2.5 | PJND cc_std ≤ 5 | All-band cc_std ≤ 5 | per-band ±5 | passed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| cc4v7_s1 | PASS (0.9722) | PASS (0.0000) | PASS (60.58) | **FAIL (3.637)** | PASS (0.70) | PASS (max 4.22) | FAIL (b≥2.5) | 5/7 |
| cc4v7_s2 | PASS (0.9611) | PASS (0.0000) | PASS (65.13) | **FAIL (3.355)** | PASS (0.73) | PASS (max 4.24) | FAIL (b≥2.5) | 5/7 |
| cc4v7_s3 | PASS (0.9767) | PASS (0.0000) | PASS (67.47) | **FAIL (3.893)** | PASS (0.69) | PASS (max 4.56) | FAIL (b≥2.5) | 5/7 |

Two gate failures across all 3 seeds:
1. Gate 4 (T=63 cross-codec): V7 outputs score=63 at butter≈2.5 instead
   of V6's butter≈1.5, because V7's empirical PJND anchor is at
   score=77.7-not-63. Cross-codec butter agreement at score=63 is
   correspondingly worse (mean butter_p3 = 3.4–3.9 vs V6 ship 1.73).
2. Gate 7 (V7-specific per-(codec, band) within ±5): at butter≥2.5,
   per-codec empirical targets diverge by 30-70 score units (e.g.,
   zenjxl=86 vs zenwebp=17 at butter=4.0). The trainer chose
   cross-codec parity (cc_std ≤ 5) over per-codec target chasing —
   the same prioritization V6 uses, just exposed because V7's anchor
   targets MADE the conflict visible.

## Per-band achievement (codec-pooled, cc4v7_s3 representative)

| band | empirical target (med) | achieved_mean | Δ |
|---:|---:|---:|---:|
| 0.30 | 88.0 | 87.19 | -0.81 |
| 0.80 | 87.4 | 85.03 | -2.37 |
| 1.50 | 75.9 | 78.56 | +2.66 |
| 2.50 | 66.2 | 67.24 | +1.04 |
| 4.00 | 26.4 | 48.81 | **+22.41** |
| 6.00 | 20.0 | 28.46 | **+8.46** |

## Mohammadi panel (held-out validation)

| Corpus | n | V6 ship (cc4v6_w1p0_p0p30_s1) | V7 s1 | V7 s2 | V7 s3 |
|---|--:|---:|---:|---:|---:|
| CID22 | 4292 | 0.8770 | 0.8729 | 0.8581 | 0.8600 |
| KADIK10k | 10125 | 0.7179 | 0.5493 | 0.5519 | 0.6756 |
| TID2013 | 3000 | 0.7542 | 0.6824 | 0.6354 | 0.7495 |
| KonJND-1k | 1008 | 0.1962 | 0.4585 | 0.3205 | 0.3656 |
| AIC-3 | 600 | 0.7961 | 0.7827 | 0.7761 | 0.7877 |

Seed s1 has the largest KonJND improvement (+0.262 vs V6 ship) —
the empirical anchor at butter=1.5 (score=77.7 vs V6's 63) pushes
the PJND-region calibration toward where KonJND humans actually
sit. But this comes at the cost of the V6 score=63-at-PJND
convention that gate 4 measures.

## Open questions surfaced by V7 anchor build

1. **zenjxl ssim2 stays at 86-88 across ALL butter bands** including
   butter=6.0 where zenjpeg/zenwebp/zenavif fall to 34/-/-/20. Either:
   - The (image × q) intersection between butter's 1000 refs and the
     ssim2 score parquet is selecting easy zenjxl images, OR
   - zenjxl produces ssim2-saturated output even at heavy butter
     distortion (which would mean butter and ssim2 disagree on
     zenjxl's perceptual quality at the same q).
2. **zenwebp band=6.0 has 0 emitted anchors** (all 1000 sources
   filtered — zenwebp can't reach butter=6.0 within ±0.5 in its
   q∈[5..95] sweep). The trainer sees no zenwebp at the heavy
   distortion band; downstream gate measurements there will rely on
   the 3 surviving codecs.

These are properties of the underlying butter/ssim2/codec relationship,
not artifacts of the V7 build. Document for future investigation but
do NOT block the V7 ship on them.

## Predecessor docs

- V6 (last ship): `benchmarks/v_tuner_v6_methodology_2026-05-19.md`
- V5 (range-failing predecessor): `benchmarks/v_tuner_v5_falsification_2026-05-19.md`
- V7 anchor target comparison: `benchmarks/v_tuner_v7_anchor_target_comparison_2026-05-19.md`
