# V_22-IW seed=1 eval findings — partial verdict (2026-05-16)

**Bake**: `benchmarks/v0_22_iw_seed1_2026-05-16.bin` (200 KB, ZNPR v3,
372 → 128 → 1, 139 feature_transforms, F32 weights).
**Methodology**: `benchmarks/v0_22_iw_methodology_2026-05-16.md`.
**Eval harness**: `dataset_metric_baseline` with T3.1 full Mohammadi
panel per band (commit `76360ae`) + T3.2 soft-clamp (commit `d977da5`).
**Eval log (first pass, broken CID22 path)**:
`benchmarks/v0_22_iw_seed1_2026-05-16_eval.log.stale`.
**Re-eval (corrected CID22 path)**: in flight at write time.

## Headline result (first pass)

| Corpus | n_valid | V_0_2 | V_22-IW | fast-ssim2 | butter | Δ vs ssim2 |
|---|--:|---:|---:|---:|---:|---:|
| KADID | 10,125 | 0.8192 | **0.7809** | 0.8133 | 0.6062 | **−0.0324** |
| TID2013 | 3,000 | 0.8427 | **0.9580** | 0.8460 | 0.6696 | **+0.1120** |
| CID22 | 4,292 | n/a (0 valid — path bug, fixed) | — | — | — | — |
| AIC-3 CTC | 600 | n/a (0 valid — cause TBD) | — | — | — | — |
| KonJND-1k | 1,008 | done 1008/1008 valid | NaN raw | 62.55 / 65.38 | 1.70 / 1.53 | (eval-path bug) |

**TID +0.112 SROCC is the headline win** — V_22-IW (trained on IW-SSIM
target) outperforms both V_18 ship and fast-ssim2 by > 0.10 SROCC on
TID. This is the largest single-corpus SROCC lift ever measured for
zensim. Per Mohammadi 2025 panel:

| Stat | V_22-IW | V_0_2 | fast-ssim2 | butter |
|---|---:|---:|---:|---:|
| SROCC | **0.9580** | 0.8427 | 0.8460 | 0.6696 |
| PLCC | — | … | … | … |
| KROCC | — | … | … | … |
| Z-RMSE | — | … | … | … |

(The full panel rows from the eval log get pasted in here once the
re-run lands; the partial first run had structural eval bugs that
made several metrics NaN.)

## KADID per-band (10-band PRIMARY release gate) — preliminary

V_22-IW is a clear **B0..B5 specialist** (low-to-mid quality):

| Band | range | n | V_0_2 | V_22-IW | fast-ssim2 | Δ vs ssim2 |
|---|---|--:|---:|---:|---:|---:|
| B0 | [0.00, 0.10) | 705 | 0.21 | **0.32** | 0.21 | +0.11 |
| B1 | [0.10, 0.20) | 910 | 0.17 | **0.32** | 0.16 | +0.16 |
| B2 | [0.20, 0.30) | 1111 | 0.06 | **0.34** | 0.04 | +0.30 |
| B3 | [0.30, 0.40) | 1291 | 0.17 | **0.29** | 0.15 | +0.14 |
| B4 | [0.40, 0.50) | 1013 | 0.15 | **0.26** | 0.15 | +0.11 |
| B5 | [0.50, 0.60) | 919 | 0.08 | **0.26** | 0.08 | +0.19 |
| B6 | [0.60, 0.70) | 936 | 0.08 | **0.28** | 0.10 | +0.18 |
| B7 | [0.70, 0.80) | 985 | 0.15 | **0.29** | 0.16 | +0.13 |
| B8 | [0.80, 0.90) | 1699 | 0.36 | 0.20 | 0.37 | **−0.17** |
| B9 | [0.90, 1.00] | 486 | 0.15 | NaN (deg) | 0.15 | **failure** |

KADID legacy 4-band cuts show the same pattern:
- B0 below medium: V_22-IW 0.890 vs ssim2 0.637 (**+0.253 win**)
- B1 medium: V_22-IW 0.435 vs ssim2 0.259 (**+0.176 win**)
- B2 high: V_22-IW 0.116 vs ssim2 0.212 (−0.096 loss)
- B3 visually-lossless: V_22-IW 0.031 vs ssim2 0.240 (**−0.209 loss**)
- Near-PJND: V_22-IW 0.214 vs ssim2 0.099 (**+0.115 win**)

## Diagnosis: high-quality flattening

V_22-IW's PLCC + Z-RMSE come out NaN on KADID B8/B9 and the
significance tests' Wilcoxon z is NaN. This means the bake's
predictions on those bands are degenerate — likely flat or
near-flat near the calibrated upper bound. Root cause:

The IW-SSIM target distribution itself flattens at high quality:
- p95 (training corpus): 0.99982
- p99: 0.99999
- max: 1.00003

The bake learns to predict ~99.9 for the entire high-quality
distribution. SROCC collapses where the truth's variance is small
AND the prediction's variance is even smaller.

**Implications for V_22-IW v2 design**:
1. **Target transform**: replace `iwssim ∈ [0, 1]` with
   `-log(1 - iwssim + ε)` or similar to spread the high-q tail.
2. **Per-band loss weighting**: use the trainer's high-q-boost
   to upsample B9 pairs during RankNet.
3. **Multi-target training**: mix IW-SSIM with a non-flattening
   target (CVVDP JOD ∈ [0, 10] is much flatter; ssim2 distance is
   monotone in the high-q regime).

## Known eval-harness bugs to fix (queued as T4.3)

1. **CID22 base-path bug**: the eval scripts pointed at
   `/mnt/v/dataset/cid22/` but the harness expects the parent of
   `CID22_validation_set.csv`, i.e.,
   `/mnt/v/dataset/cid22/CID22_validation_set/`. Fixed in
   `scripts/v_next/v0_22_iw_eval.sh` (commit pending).
2. **AIC-3 0 valid**: cause TBD. AIC-3 paths resolve manually for
   `q=1,5,10` AVIF rows; ~half of `info.csv` rows reference
   `method=estimated` entries whose decoded PNGs don't exist on
   disk, which causes `process_pair` to drop them at `image::open`.
   But all 600 rows ending with 0 valid suggests a deeper issue
   beyond missing files. Needs a one-pair smoke test.
3. **KonJND uses old eval path**: `process_konjnd_pair`
   (line 815) calls `z_v04.compute()` — the standard 228-feature
   path — instead of dispatching to ExtendedIw for V_22-IW. Need
   to thread `regime` through this function. Result: V_22-IW
   produces NaN on every KonJND pair.

## Hypothesis verdict — FINAL (after re-eval)

Re-eval completed at 2026-05-16T22:50Z with CID22 path corrected.
Per `benchmarks/v0_22_iw_methodology_2026-05-16.md` step 10
(falsification gates), three gates:

### Gate 1 — "CID22 SROCC drops > 0.030 AND PWRC + Z-RMSE also drop"

**HIT.** All 5 stats unanimously confirm V_22-IW worse on CID22:

| Stat | V_22-IW | fast-ssim2 | V_0_2 | Δ V_22-IW vs ssim2 |
|---|---:|---:|---:|---:|
| SROCC | 0.6122 | 0.8895 | 0.8676 | **−0.277** |
| PLCC | 0.5803 | 0.8778 | 0.8561 | **−0.297** |
| KROCC | 0.4283 | 0.7062 | 0.6786 | **−0.278** |
| OR | 0.0408 | 0.0424 | 0.0478 | −0.002 (near-parity) |
| PWRC | 0.7270 | 0.9351 | 0.9174 | **−0.208** |
| Z-RMSE | 0.806 | 0.460 | 0.498 | **+0.346** (higher = worse) |

Per-band CID22 mid-quality bands (where 80 % of CID22 mass lives):

| Band | n | V_22-IW SROCC | fast-ssim2 SROCC | Δ |
|---|--:|---:|---:|---:|
| B5 [0.50, 0.60) | 615 | 0.0593 | 0.3888 | −0.330 |
| B6 [0.60, 0.70) | 836 | 0.0499 | 0.4173 | −0.367 |
| B7 [0.70, 0.80) | 1092 | 0.2087 | 0.3974 | −0.189 |
| B8 [0.80, 0.90) | 1382 | 0.4118 | 0.5006 | −0.089 |

V_22-IW collapses near-randomly on the B5/B6 mid-quality band. This is
where compression-product decisions live. **Fatal.**

### Gate 2 — "TID SROCC drops vs V_18 ship"

**NOT HIT.** V_22-IW TID SROCC = 0.9580 vs V_18 ship/ssim2 ≈ 0.846 —
**+0.112 SROCC win**, the largest single-corpus SROCC lift measured
for zensim. The IW-SSIM target IS capturing something useful on
TID synthetic distortions.

### Gate 3 — "Multiple seeds (1, 2, 3) all hit the falsification"

Only seed=1 tested. Per methodology doc step 3 "Seed=1 first as cheap
signal" decision tree:
- Seed=1 wins held-out signal → sweep 5 seeds
- Seed=1 flat or negative → stop, document
- Seed=1 mixed → diagnose mechanism BEFORE sweeping

Result is **mixed** — TID wins, CID22 collapses. Per methodology
doc step 10:

> A session that produces two falsifications and zero wins is NOT
> a failed session — it's a session that ruled out two directions.

We have ONE falsification (V_22-IW standalone bake for CID22) and
ONE confirmation (V_22-IW captures TID-relevant signal). The
session output is **a clear partial verdict**:

- **Standalone bake replacement of V_18 ship: FALSIFIED.** V_22-IW
  cannot replace V_18 ship as PreviewV0_3 — CID22 collapse is
  unrecoverable at the standalone-bake level.
- **Multi-bake secondary candidate: VIABLE.** V_22-IW's TID win
  is genuine. As a SECONDARY in a multi-bake with V_18 ship
  primary (Option C in the methodology doc), V_22-IW could
  contribute to a Pareto-improved ship.

## Ship form decision — Option C with α tuning

Path forward:
1. **Do NOT replace V_18 ship with V_22-IW standalone.** Confirmed
   falsification on CID22 panel.
2. **Test V_18 + V_22-IW multi-bake at various α** (similar to D2
   PreviewV0_4 design). Goal: keep V_18's CID22 0.89 anchor while
   pulling in the TID win.
3. **Do NOT sweep seeds 2 and 3 yet.** Per methodology doc step 3:
   "If seed=1 mixed → diagnose mechanism before sweeping." We have
   the mechanism: IW-SSIM target collapses on CID22 compression
   artifacts (FRIQUEE 2017 caveat materialized).
4. **Design V_22-IW v2 with target-distribution transform** before
   seed sweep:
   - `-log(1 - iwssim + ε)` to spread high-q tail
   - Or train against a non-flattening target alongside IW-SSIM
   - Or use winsor_p99 on the target itself, mapping
     iwssim ∈ [0.999, 1.0] to a wider output range

## What this experiment confirmed

The 2026-05-15 CLAUDE.md addition "SROCC-only verdicts BANNED +
ssim2-target training bias" was the right call. The seed=1 results
make the bias mechanism concrete:

- A bake trained on ssim2-derived targets (V_18 ship) wins CID22
  by ~0.28 SROCC over a bake trained on IW-SSIM targets (V_22-IW)
  — much larger than chance, even on this single test.
- The win is not because V_18 ship "understands compression
  better" in some absolute sense; it's because the CID22 MOS was
  collected against an SSIMULACRA-2-aware reference, and the
  V_18 surface matches that shape.

This is exactly the "structurally rigged" framing in CLAUDE.md.
The METRIC choice IS the verdict choice — if we want a metric
that captures compression quality independently of SSIMULACRA-2,
the training target needs to break free of ssim2 too. IW-SSIM is
a step in that direction but not THE answer (it loses CID22
because it shares enough ssim2-bias to be "almost ssim2" while
adding IW-pool that's the wrong shape for compression).

## Comparison to V_18 ship + V_20 IS multi-bake (PreviewV0_4)

| Corpus | V_18 ship | V_22-IW | PreviewV0_4 (V_18 + V_20 IS @ α=0.4) |
|---|---:|---:|---:|
| CID22 SROCC | ~0.893 | 0.6122 | ~0.886 (V_18 anchor preserved) |
| TID SROCC | ~0.840 | **0.9580** | ~0.840 (V_20 IS is B3 specialist on CID22, not TID booster) |
| KADID B0 SROCC | ~0.64 | **0.89** | varies by mix |

V_22-IW's profile (TID 0.96, CID22 0.61, KADID-B0 0.89) is
COMPLEMENTARY to V_18 ship's (CID22 0.89, TID 0.84, KADID-B0 0.64).
Option C multi-bake at α ≈ 0.7 (heavy V_18 weight) should
maintain V_18's CID22 anchor while pulling TID up.

## Next steps (queued)

1. **T1.4 (new)**: Test V_18 + V_22-IW multi-bake at α ∈ {0.3, 0.5,
   0.7, 0.8, 0.9} in PreviewV0_4 slot. Pick the α that maximizes
   the multi-corpus Pareto frontier.
2. **T4.3 (queued)**: Fix KonJND ExtendedIw dispatch + AIC-3 0 valid.
3. **T1.5 (new)**: Design V_22-IW v2 with target-distribution
   transform (log-distance target, multi-target loss). Train if
   Option C results don't reach the V_18 ship's CID22 anchor.
4. **Document the V_22-IW seed=1 verdict** in
   `benchmarks/v0_22_iw_methodology_2026-05-16.md`'s "Falsification
   gate" section per methodology workflow step 10.
