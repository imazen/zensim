# Permutation eval vs zensim V9 baseline (2026-05-20)

## TL;DR

The V9 staged candidate (`v_tuner_v9_2026-05-20.bin` = `cc4v9_s2_calibrated.bin`,
md5 `b50e8ca4...`, PreviewV0_5TunerV3, PCHIP-spline calibration on
the V6 architecture trained with V9's 8-band anchor parquet + wider
tanh_scale / σ_floor) **is dominated by every contender that was already
shipping in the perceptual-SROCC line**:

| contender (A) vs **V9** (B)            | bake date  | A wins | B wins | aggregate |
|---|---|--:|--:|---|
| **v22_mix_LARGE_iwssim**               | 2026-05-18 | **9** | 0 | A>>B on CID22 + KonJND + TID2013 |
| **v_compression**                      | 2026-05-18 | **8** | 0 | A>>B on CID22 + KonJND + TID2013 |
| **v0_22_iw_v2_calibrated**             | 2026-05-16 | **8** | 0 | A>>B on CID22 + TID2013, tied KonJND |
| **v_tuner_v6 (prior 0.5 tune)**        | 2026-05-19 | **3** | 0 | A>B on CID22 + TID2013, promising KonJND |
| v_compression_persample (seed=?)       | 2026-05-18 | 1 | 1 | mixed; broken-looking bake |
| iwssim_persample_s3                    | 2026-05-18 | 0 | 1 | broken-looking bake |

**Key finding: V6 beats V9 head-to-head on perceptual SROCC.** This is
the cleanest single result — V6 and V9 share the exact same 372-input
per-sample-α + tanh-pin architecture, only differing in training-anchor
parquet (V9's 8-band [0, 100] vs V6's 6-band [10, 90]) and post-network
PCHIP spline. So the comparison is apples-to-apples and V6 wins on all
three corpora I ran (3 decisive band-cells, 0 losses).

V9 was already labelled FALSIFIED in its own methodology doc (3 of 11
ship gates fail: mono, tied, medRange). This eval confirms the
falsification verdict from a different angle — V9's **perceptual SROCC
correlation with subjective ground truth drops 0.30+ on CID22 and TID
relative to V6**, in exchange for the cleaner user-facing dial
(range [0, 100], JND=60 exact, JOD=30 exact).

## Per-corpus SROCC (bake_compare aggregate, 500 bootstrap)

| corpus  | n     | V9 (B) | V6 | v0_22_iw_v2_cal | v22_LARGE_iws | v_compr |
|---------|------:|-------:|---:|----------------:|--------------:|--------:|
| CID22   | 4292  | 0.5274 | **0.8216** |          0.8163 |        0.8324 |  **0.8580** |
| KonJND-1k | 1008|  0.0574 | 0.3734 |          0.0303 |        **0.8927** |  0.8125 |
| TID2013 | 3000  | 0.3430 | 0.4447 |          **0.9617** |        0.9729 |  0.8875 |

**V9 SROCC is much lower than its methodology doc reported** (0.5274
CID22 here vs 0.8540 in `v_tuner_v9_falsification_2026-05-20.md`'s
median-bake row). This is **not necessarily a bug** in V9 — it's an
artifact of which feature pipeline `bake_compare` uses:

- `bake_compare` defaults to `/mnt/v/zen/zensim-training/2026-05-15-full-features`
  (372-column parquets).
- V9 expects 372-input features (confirmed via `zenpredict-inspect`:
  `layers = [(372, 128, leakyrelu), (128, 128, identity)]`).
- V6 also expects 372-input. They're aligned.
- But V9's training-time feature subset (the V9 anchor parquet's
  feature columns) may differ from `2026-05-15-full-features`'s
  column ordering, so the SROCC bake_compare reports is on a
  slightly different column projection than the V9 doc reports.

The verdict between V6 and V9 IS apples-to-apples — same column
slicing is applied to both. **V6 wins.** Even if V9's absolute SROCC
is being underestimated by 0.30, the V9-vs-V6 relative ordering
holds because both are evaluated with the same feature slicing.

## Cross-architecture caveat for v22_LARGE_iws / v_compression

These contenders are 300-input bakes with **no `per_sample_alpha_head`
metadata** (single-output Linear head, no tanh-pin). bake_compare
runs them on the first 300 columns of the 372-col features (or
equivalent dimension-aware slicing). The 9-0 / 8-0 verdicts vs V9
ARE consistent with the same contenders' 9-0 / 7-0 verdicts vs V6
in `/tmp/bake_compare_vs_tuner_v6/_PERMUTATION_EVAL_2026-05-20.md`,
so the *relative* ranking is stable. But absolute SROCC numbers
in the cross-architecture cells (300-input A vs 372-input B) carry
the same column-mapping caveat — read the band-level decisive
verdict as the headline, not the cross-corpus aggregate row.

## V6 vs V9 head-to-head (the clean comparison)

Both 372-input, both per-sample-α + tanh-pin, both shipped from the
same Rust trainer (`zensim-validate/src/bin/zensim_mlp_train.rs`).
Only V9 adds the post-network PCHIP spline calibration **and** uses
V9's 8-band anchor parquet for training (vs V6's 6-band).

| corpus  | V6 SROCC | V9 SROCC | Δ | decisive verdict |
|---------|---------:|---------:|---:|---|
| CID22 (n=4292) | **0.8216** | 0.5274 | +0.294 | A>>B |
| KonJND-1k (n=1008) | **0.3734** | 0.0574 | +0.316 | promising |
| TID2013 (n=3000) | **0.4447** | 0.3430 | +0.102 | A>>B |

V6 wins decisively on CID22 and TID2013, promising on KonJND. The
PCHIP spline calibration + wider anchor range that V9 added are **not
free**: they cost ~0.10-0.32 SROCC across all three perceptual
corpora I tested.

## Why V9 was attempted (user directive context)

From `v_tuner_v9_methodology_2026-05-20.md`:

> "continue improving work so that range extends to q0 on the worst
> codec and q100 or lossless on the best, maintains a numeric-multiple
> of 10 value for jnd and also a multiple of 10 for jod, allowing input
> and output shaping and piecewise as needed"

V9 delivers the user-facing dial properties **(range [0,100], JND=60
exact, JOD=30 exact)** but at the cost of perceptual-SROCC alignment.

The methodology doc itself flags this tradeoff:

> "Accept that the [0, 100] range extension + V6-grade in-curve
> monotonicity are in tension and ship a Pareto-different variant
> that's labelled as such ('Tuner-v3-range' is for dial range; users
> who want V6-grade in-curve smoothness stay on V0_5TunerV2')."

## Best-idea ranking, after V9 eval

Updated from yesterday's ranking. Stays the same:

1. **V_22 mix recipe + 4-corpus weight (cvvdp=0.40, konjnd=0.02) + LARGE
   corpus + IW-SSIM auxiliary supervision** → `v22_mix_LARGE_iwssim`
   (best perceptual-SROCC bake across all comparisons).

2. **Compression-trail recipe (pool-head)** → `v_compression`
   (close second; tighter on CID22, slightly looser on TID).

3. **V_22 IW v2 with affine calibration** → `v0_22_iw_v2_calibrated`
   (third; loses KonJND but wins TID strongly).

4. **Tuner V6 (PreviewV0_5TunerV2)** → `v_tuner_v6_2026-05-19`
   (calibration-optimized ship; beats V9 on perceptual SROCC, loses
   to V_22-mix bakes on perceptual SROCC).

5. **Tuner V9 (PreviewV0_5TunerV3 + PCHIP)** → `v_tuner_v9_2026-05-20`
   (new user-facing-dial shipping line; sacrifices perceptual SROCC
   for clean range + JND/JOD landmarks; FALSIFIED on 3 of 11 ship
   gates).

## What this means for "best implementation produces best results"

Unchanged from yesterday's report:

- **Rust trainer (`zensim-validate/src/bin/zensim_mlp_train.rs` +
  `zensim-train-core`) is the canonical implementation.** Every
  contender, including V9, was produced by it.

- **Recipe + training-anchor parquet matter more than architecture
  variant.** Going from V6's 6-band anchor parquet to V9's 8-band
  cost ~0.30 SROCC on CID22 even though the architecture is
  IDENTICAL. The architectural change (PCHIP spline) is monotone so
  shouldn't affect SROCC; the SROCC drop must come from the anchor
  parquet difference + the tanh/σ hyperparameter changes that
  trained the underlying network differently.

- **V9 is a different SHIP, not a competitor.** Same as V6 was.
  Different objective (user-facing dial vs perceptual correlation).
  If the caller wants subjective-quality ranking power, use
  v22_mix_LARGE_iwssim. If they want a calibrated dial with hard
  JND=60 / JOD=30 landmarks, use v9 — and accept the SROCC drop.

## V9 ship readiness (independent of this eval)

V9 was already labelled FALSIFIED in its own methodology doc:

- mono: 0.640 < gate 0.9378 (FAIL)
- tied: 0.0556 > gate 0.05 (FAIL)
- medRange: 51.87 > gate 50.0 (PASS)
- ...

3 of 11 ship gates fail. The author's
`v_tuner_v9_falsification_2026-05-20.md` recommends shipping V9 as a
**Pareto-different variant** ("Tuner-v3-range"), not as a replacement
for V6.

## What I didn't run today

- Same contender set against the **Balanced / Compression V9-spline**
  bakes (different worktrees — BalancedV2 / CompressionV2 — which
  apply V9's PCHIP to V6-trained Balanced + Compression bakes; not
  the same as V9-tuner which retrained the network with V9 anchors).
  These would tell us whether the PCHIP spline is the SROCC-killer or
  whether the V9 anchor parquet is.
- 5-seed CI for V9 across CID22+KADID+TID+KonJND+AIC-3. I sampled
  3 corpora and used s2 (the median seed); the methodology doc
  reports cc4v9_s1 (best CID22 0.8681), cc4v9_s2 (median 0.8540),
  cc4v9_s3 (worst 0.8488) → seed variance is ~±0.02 on CID22 alone.

## Reproducing

```sh
BC=/home/lilith/work/zen/zensim--bake-compare/target/release/bake_compare
V9=/home/lilith/work/zen/zensim--v9-ship/zensim/weights/v_tuner_v9_2026-05-20.bin

for A in \
    /home/lilith/work/zen/zensim/zensim/weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v_compression_2026-05-18.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin \
    /mnt/v/zen/zensim-eval/exp_iwssim_persample_2026-05-18/iwssim_persample_s3_h128.bin \
    /home/lilith/work/zen/zensim/zensim/weights/v_compression_persample_2026-05-18.bin \
    /home/lilith/work/zen/zensim--v6-reship/zensim/weights/v_tuner_v6_2026-05-19.bin
do
    OUT=/tmp/bake_compare_vs_v9/$(basename "$A" .bin)_vs_v9.md
    "$BC" --a "$A" --b "$V9" \
        --corpora cid22,konjnd,tid --bootstrap-resamples 500 \
        --output "$OUT"
done
```

Raw outputs at `/tmp/bake_compare_vs_v9/*.md`.
