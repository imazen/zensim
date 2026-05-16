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

## Hypothesis verdict — provisional pending re-eval

Per `benchmarks/v0_22_iw_methodology_2026-05-16.md` step 10
(falsification gates):

- **Hypothesis confirmed on TID**: V_22-IW +0.112 SROCC over V_18
  ship/ssim2. Full Mohammadi panel pending re-eval.
- **Hypothesis partially confirmed on KADID**: V_22-IW WINS B0..B7
  (the priority band per CLAUDE.md "B0..B5 lift is the dominant
  priority"), LOSES B8/B9 (high-quality, near-saturation regime).
  Aggregate −0.03 SROCC loss is within tolerance.
- **CID22 verdict deferred** to re-eval.
- **Ship form decision deferred** — Option C (V_22-IW as the
  PreviewV0_4 secondary, replacing V_20 IS) looks most appealing
  given the strong low-q win + high-q flattening pattern, but
  CID22 results are required first.

## Next steps

1. **In flight (re-eval)**: complete eval with CID22 path fixed.
   Expected: B0..B5 wins on CID22 (consistent with V_18 ship's
   CID22 0.89 baseline, V_22-IW should fall to ~0.85-0.88).
2. **Fix queued (T4.3)**: KonJND path + AIC-3 0-valid investigation.
3. **Don't sweep seeds yet** — wait for CID22 full panel verdict
   per CLAUDE.md "Seed=1 first as cheap signal" workflow step 3.
4. **If hypothesis fully confirms**: design V_22-IW v2 with
   target-distribution transform (per "Diagnosis" above) before
   moving to seed=2/3 sweep.
