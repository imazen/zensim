# V13-CVVDP-DISTILL — FALSIFIED (task #200, 2026-05-20)

**Verdict: cvvdp distillation produces a saturated metric with universal KonJND collapse across all 5 seeds.** The root cause is distributional skew of the cvvdp_score training corpus (>70 % at JOD ≥ 9.5, >27 % maxed at 10.0 in safesyn alone), not the Basin B cross-codec-eq pair-loss mechanism the brief hypothesized about. PCHIP spline calibration produces degenerate 2-knot fits because 8 of 10 anchor bands are non-monotone in raw predictions.

**The brief's "biggest swing" experiment closes with a different finding than expected:** cvvdp distillation isn't *more* informative than ssim2 — it's *less* informative on this corpus because the JOD distribution is right-skewed past the perceptibility threshold. Basin B may be specific to cross-codec-eq pair loss after all (consistent with V11/V12 findings); V13's KonJND collapse has an unrelated cause (target saturation).

V10 BalancedV3 remains the Balanced ship. V_24 per-sample-α remains the Compression ship. No new ship from V13.

## Hypothesis (task #200 brief, condensed)

> Every V11/V12 variant fires cross-codec-eq pair loss `(y_a − y_b)²` between codec pairs at matched anchor levels and hits KonJND collapse. cvvdp distillation removes that loss entirely — the training signal becomes pure `(predicted − cvvdp_score)²`. No equivalence-pair term, no anchor parquet beyond the global PCHIP spline. **Should not go through Basin B** (different mechanism than cross-codec-eq pair-equiv enforcement).

Ship gate (Balanced trail):
- CID22 SROCC ≥ 0.8324 + 0.005 = **≥ 0.8374**
- CID22 Z-RMSE ≤ **0.500** (V10 was 0.564)
- KADID/TID/KonJND within −0.10 of V10 baseline
- AIC-3/AIC-4 within −0.005 of V10 baseline
- Anchor landings bit-exact via PCHIP knots

## Method (deviations from brief, documented)

### Recipe deviation 1: per-sample-α head required for MSE

The brief specifies `--mse-weight 0.5 --monotonicity-reg 0.5` with the V_22-mix architecture (plain MLP). **These aux losses only fire on the per-sample-α head path** (zensim-validate/src/mlp_train.rs:419 — "trainer panics if set on other heads"). Pure RankNet on cvvdp targets is not distillation; the experiment's intent — `(predicted − cvvdp_score)²` per pair — requires MSE. We use per-sample-α head with cvvdp_score target. Architecture diverges from V_22-mix-LARGE; the *target shape* is what's being tested.

### Recipe deviation 2: 3 of 7 brief-specified groups dropped (zero cvvdp coverage)

cvvdp coverage audit on `canonical-2026-05-21/train/*.parquet`:

| group | rows | cvvdp_score coverage |
|---|--:|---:|
| safesyn | 196,086 | 100 % |
| kadid | 10,125 | 100 % |
| tid | 3,000 | 100 % |
| cvvdp_iwssim_LARGE | 73,300 | 100 % |
| konjnd-dense | 20,160 | **0 %** |
| cid22_train | 17,611 | **0 %** |
| pipal | 21,800 | **0 %** |

The 3 subjective-IQA sets (konjnd-dense, cid22_train, pipal) carry `human_score` + `pjnd_target` (KonJND) or human MOS (CID22/PIPAL) but were never scored with cvvdp. The brief's fallback options were: (a) backfill via `zen-metrics batch --metric cvvdp` on the underlying images, or (b) drop the groups. Backfilling 60k images would blow the 3-hour budget; we dropped. Net training rows with cvvdp coverage: **282,511** across 4 groups.

### Recipe deviation 3: target column + scale per trainer docs

Brief specifies `--target-column cvvdp_target_score` (a derived column). The actual canonical schema has `cvvdp_score` ∈ [0, 10] JOD scale and `cvvdp_log_norm` ∈ [0, 100] precomputed. Per the trainer's documented pattern (`--target-column cvvdp_jod --target-scale 10.0`), we use `cvvdp_score --target-scale 10.0` to bring JOD into 0..100 band-cutoff space. Pearson correlation with `human_score` (ssim2-derived) on safesyn: **cvvdp_score r=0.956, cvvdp_log_norm r=0.660** — `cvvdp_score` is the better-shaped target.

### Trainer invocation

```bash
target/release/zensim_mlp_train \
    --group "safesyn:.../safesyn.parquet:1.0:0.0" \
    --group "kadid:.../kadid.parquet:0.6:0.4" \
    --group "tid:.../tid.parquet:0.6:0.4" \
    --group "large:.../cvvdp_iwssim_LARGE.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --minibatch-size 32 \
    --lr 5.66e-3 --l2 1e-5 --leaky-alpha 0.01 \
    --val-policy min --early-stop-patience 0 \
    --max-features 300 --target-column cvvdp_score --target-scale 10.0 \
    --per-sample-alpha-head --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.5 --mse-weight 0.5 --monotonicity-reg 0.5 \
    --gpu-runtime cuda \
    --seed <S> --out cc4v13_s<S>.bin
```

GPU trainer floors `--minibatch-size` to `max(32, 512) = 512` on the GPU path. 29,400 batches in 28-30 s per seed.

NO `--anchor-parquet`, NO `--cross-codec-eq-parquet`, NO PJND passthrough. Pure distillation per brief.

## Results (5-seed CI, pre-spline)

Bakes at `/mnt/v/zen/zensim-eval/exp_v13_cvvdp_distill_2026-05-20/cc4v13_s{1..5}.bin` (223,911 bytes each, ZNPR v3).

| Seed | CID22 SROCC | CID22 Z-RMSE | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.8152 | 0.566 | 0.8369 | 0.8535 | **0.0958** | 0.8216 | 0.9456 |
| 2 | 0.8302 | 0.552 | 0.8350 | 0.8534 | **0.1498** | 0.8196 | 0.9574 |
| 3 | 0.8332 | 0.546 | 0.8371 | 0.8583 | **0.1178** | 0.8239 | 0.9469 |
| 4 | 0.8545 | 0.509 | 0.8427 | 0.8588 | **0.0795** | 0.8216 | 0.9459 |
| 5 | 0.8423 | 0.534 | 0.8338 | 0.8496 | **0.0942** | 0.8170 | 0.9340 |
| **median** | **0.8332** | **0.546** | 0.8369 | 0.8553 | **0.0958** | 0.8216 | 0.9459 |
| V10 BalancedV3 | 0.8324 | 0.564 | — | — | (ref ~0.78-0.89) | — | — |
| brief gate | ≥ 0.8374 | ≤ 0.500 | within −0.10 | within −0.10 | within −0.10 | within −0.005 | within −0.005 |

**Ship gate status (median seed):**

- CID22 SROCC: **0.8332 vs gate 0.8374 → FAIL by −0.0042**
- CID22 Z-RMSE: **0.546 vs gate 0.500 → FAIL by +0.046**
- KonJND: **0.0958 vs V10 ~0.78 → CATASTROPHIC FAIL by −0.68**
- KADID/TID near V10 baseline (no big move there)

**No seed passes the ship gate.** Best CID22 seed (s4: 0.8545) still has KonJND 0.0795 (still collapsed) and is +0.009 CID22 vs ship — would only be a marginal CID22 win even at best-of-5 cherry-pick, and the brief explicitly forbids best-seed cherry-pick for ship.

## Root cause: cvvdp_score training-corpus distribution

cvvdp_score (JOD ∈ [0, 10]) distribution in the V13 training groups:

| corpus | n | cvvdp p10 | p50 | p90 | %≥9.5 | %≥9.95 |
|---|--:|--:|--:|--:|--:|--:|
| safesyn | 196,086 | 8.91 | 9.80 | 9.99 | **73 %** | **27 %** |
| kadid | 10,125 | 6.43 | 8.75 | 9.84 | 25 % | 5 % |
| tid | 3,000 | 7.14 | 9.07 | 9.88 | 32 % | 3 % |
| cvvdp_iwssim_LARGE | 73,300 | 9.46 | 9.96 | 10.00 | **89 %** | **54 %** |

In score units (×10), safesyn + LARGE comprise the bulk of training data and have 73-89 % of pairs at score ≥ 95. MSE loss minimizes `(predicted − ~98)²` across most pairs → network predicts ~98 for everything → tanh-output-head-scale 20.0 saturates the output near the maximum → the metric loses discriminating power at the just-perceptible regime (~63 ssim2-equivalent, where KonJND pairs live).

### Diagnostic: per-target-band median predictions (seed=1)

V10 PCHIP anchor parquet has 10 target bands × ~2,000 rows each. Per-band median of the V13 bake's raw output:

| target | n | median raw_pred |
|--:|--:|--:|
| 0.0 | 1,159 | 49.5 |
| 10.0 | 39 | 53.5 *(out of order!)* |
| 20.0 | 349 | 47.5 *(non-monotone)* |
| 35.0 | 1,043 | 53.1 |
| 50.0 | 2,272 | 56.5 |
| 65.0 | 3,956 | 61.9 |
| 80.0 | 3,937 | 65.7 |
| 90.0 | 3,790 | **67.4** *(saturating)* |
| 95.0 | 3,602 | **67.6** *(saturating)* |
| 100.0 | 3,967 | **67.8** *(saturating)* |

The high-quality bands (90, 95, 100) all median around **67.4-67.8** — the network cannot distinguish "visually lossless" from "borderline noticeable." The full output range is compressed into ~[47, 68] — 21 score-units out of 100.

### Spline calibration confirms structural failure

`calibrate_balanced_v9_spline.py` builds PCHIP knots from per-band medians, dropping bands that violate monotonicity. On seed=1: **8 of 10 bands dropped** (target=10 inverts with target=20; targets 35..100 all merge into the saturated regime). The spline collapses to **2 knots**:

```
xs=[47.54, 49.53]  ys=[20.0, 0.0]  (direction=-1, decreasing)
```

This is a near-constant line — the spline cannot recover the missing dynamic range. Calibrated bake_verdict on seed=1 yields **identical** stats to the pre-spline version (rank-preserving spline doesn't change SROCC; Z-RMSE doesn't improve because the post-spline output stays in a degenerate range).

Seed=4 (best CID22) shows the same 8/10-bands-dropped degeneracy — the saturation is structural, not seed-specific.

## Mechanism analysis: why V13 KonJND collapse ≠ V11/V12 Basin B

The brief's hypothesis: cvvdp distillation removes the cross-codec-eq pair-loss → should escape Basin B.

**The actual mechanism for V13's KonJND collapse is different from Basin B:**

| Aspect | V11/V12 Basin B (cross-codec-eq) | V13 (cvvdp distillation) |
|---|---|---|
| Cause | `(y_codec_A − y_codec_B)²` enforces codec-pair agreement at anchor levels | MSE on right-skewed cvvdp targets pushes predictions into saturation |
| Manifestation | Network learns to ignore per-image variation (Basin B = "all codecs equal") | Network learns to predict ~98 always (saturation = "all pairs lossless") |
| KonJND collapse | Yes — codec-pair enforcement kills just-perceptible discrimination | Yes — saturated output kills just-perceptible discrimination |
| Output dynamic range | Compressed to anchor mean | Compressed to right-tail mode (~67/100) |
| CID22 SROCC impact | ~−0.03 to −0.06 | ~−0.001 to +0.022 (basically null) |
| Recovery via PJND passthrough? | Tested in V11-D, FALSIFIED | Untested — but distributional skew is the root cause, not pair loss |

**The "biggest swing" outcome:** Both V11/V12 cross-codec-eq AND V13 cvvdp-distillation produce KonJND collapse via *different* mechanisms. Basin B isn't "broader than V11/V12 suggested" — it's actually specific to pair-equiv loss formulations. But V13 reveals a *second*, independent KonJND-collapse mechanism (target-distribution saturation). Both mechanisms must be addressed to ship a cvvdp-informed metric.

## Why the V14 follow-up is non-trivial

Brief's "If fails" fallback: "Propose V14 with denser cvvdp (vast.ai backfill)."

V14 would need:

1. **cvvdp backfill on konjnd-dense + cid22_train + pipal** — 60k images × cvvdp ≈ 6-12 hr GPU time on a vast.ai box. This restores subjective-IQA pairs to the training mix, providing the just-perceptible coverage that V13 lacks.

2. **Distribution rebalancing on safesyn/LARGE** — even with konjnd backfilled, the safesyn corpus is fundamentally right-skewed. Options: (a) sample-weight pairs by inverse cvvdp density during training (re-weight away from saturation regime); (b) include pairs with `human_score < 0.5` more heavily; (c) collect new low-q cvvdp pairs at the low-quality end where existing corpora are thin.

3. **Different target shape than raw cvvdp_score** — `cvvdp_log_norm` (already in canonical schema, 0..100, mean 27.8 instead of 95.9) compresses the high-quality tail and expands the low-quality range. A V14 retrain with `--target-column cvvdp_log_norm --target-scale 1.0` would be a 30-min ablation testing whether the log-norm target avoids the saturation trap without any new data collection.

Option 3 is the cheapest test and can run inside this session. Item for next-session backlog (out of scope here per Step 10 — falsification is data, ship the negative result).

## Artifacts

- Bakes: `/mnt/v/zen/zensim-eval/exp_v13_cvvdp_distill_2026-05-20/cc4v13_s{1..5}.bin`
- Calibrated (degenerate): `/mnt/v/zen/zensim-eval/exp_v13_cvvdp_distill_2026-05-20/cc4v13_calibrated_s{1,4}.bin`
- Pre-spline verdicts: `cc4v13_s{1..5}_PRESPLINE_verdict.md`
- Post-spline verdict: `cc4v13_calibrated_s1_verdict.md` (identical to pre-spline)
- Spline knots CSV: `cc4v13_s{1,4}_spline.csv`
- Training script: `scripts/v_next/v13_cvvdp_distill/run_v13_cvvdp_distill_seed.sh`

## What V13 ruled out

1. **cvvdp distillation as a free upgrade from ssim2-target training** — given the current canonical corpus's distributional skew, cvvdp distillation produces a worse-calibrated metric, not a better one.

2. **The "Basin B is broader than cross-codec-eq" hypothesis** — V13's KonJND collapse has a different mechanism (target saturation, not pair-equiv enforcement). The cross-codec-eq frontier may still be specifically closeable by removing pair-equiv loss.

3. **The current `cvvdp_score` column as a drop-in MSE target** — the distribution is too right-skewed for direct MSE training without a re-weighting or log-norm strategy.

## V14 follow-up ablation: cvvdp_log_norm target (also FALSIFIED, in same session)

To rule out "the saturation is the only failure mode," ran V14 with identical recipe but `--target-column cvvdp_log_norm --target-scale 1.0` instead of `cvvdp_score × 10`. The `cvvdp_log_norm` column is precomputed in canonical schema as a logarithmic remap of cvvdp into [0, 100] with mean 27.8 / median 23.5 — much flatter distribution than `cvvdp_score × 10` (mean 95.9).

5-seed CI:

| Seed | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|--|--:|--:|--:|--:|--:|--:|
| 1 | 0.7103 | 0.8145 | 0.8511 | 0.3280 | 0.7637 | 0.9091 |
| 2 | 0.7624 | 0.8174 | 0.8554 | 0.2310 | 0.7291 | 0.8778 |
| 3 | 0.7557 | 0.8048 | 0.8553 | 0.1455 | 0.7299 | 0.8838 |
| 4 | 0.7113 | 0.7989 | 0.8462 | **0.4007** | 0.7315 | 0.8661 |
| 5 | 0.7480 | 0.8131 | 0.8382 | 0.2754 | 0.7044 | 0.8900 |
| **median** | **0.7480** | **0.8131** | **0.8511** | **0.2754** | **0.7299** | **0.8838** |
| **delta vs V13** | **−0.085** | −0.024 | −0.004 | +0.180 | −0.092 | −0.062 |

V14 partially recovers KonJND (+0.18 median, best seed +0.30) — confirming target-distribution saturation is part of the V13 failure mode — but at a **catastrophic CID22 cost** (−0.085 median, far worse than the V13 ship gate failure). Best seed (s4) has KonJND 0.4007 but CID22 0.7113 — still −0.12 below V10 BalancedV3.

**Two simultaneous gates**: cvvdp_log_norm avoids saturation but its log-remap doesn't track human MOS — Pearson `r(cvvdp_log_norm, human_score)=0.66` vs `r(cvvdp_score, human_score)=0.96` on safesyn. The log transform is "the wrong shape" relative to perceptual ranking.

Combined V13 + V14 findings: **direct cvvdp distillation with current canonical corpus is a closed direction.** Both the linear `cvvdp_score × 10` and the log-compressed `cvvdp_log_norm` targets fail (in different ways). Recovery requires NEW DATA, not a different target column.

Bakes: `/mnt/v/zen/zensim-eval/exp_v14_cvvdp_lognorm_2026-05-20/cc4v14_s{1..5}.bin`.

## What V13 + V14 leave open

- V15: cvvdp backfill on konjnd-dense + cid22_train + pipal (6-12 hr GPU on vast.ai) → retrain with the konjnd group restored at high weight. KonJND collapse cause is likely *training-corpus omission* of just-perceptible pairs, not anchor metric choice. This is the highest-value next experiment, but expensive.
- V16: 4-corpus distribution rebalancing — sample-weight safesyn + LARGE pairs by inverse cvvdp density during training. ~30-min trainer change + 30-min retrain. Free; tests whether re-weighting recovers KonJND without new data.
- V17: balanced cvvdp + ssim2 multi-target training (`--target-mix cvvdp_score:0.5,human_score:0.5`) — the trainer's flag set doesn't yet support this; would need new trainer code (~2 hr). Tests whether the ssim2-trained KonJND retention can be preserved while gaining cvvdp's calibration on CID22.

## Decision

**No ship.** Task #200 closed as FALSIFIED on both V13 (cvvdp_score) and V14 (cvvdp_log_norm) ablations. V15-V17 follow-ups queued for next session.

V10 BalancedV3 stays the Balanced ship. V_24 per-sample-α s4 stays the Compression ship. SOTA_TRAILS.md unchanged.
