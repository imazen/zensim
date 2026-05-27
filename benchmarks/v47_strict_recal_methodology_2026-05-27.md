# V47-strict-recal — methodology + ship candidate (2026-05-27)

Monotone-by-construction zensim bake with a working [0,100] dial AND a
negative corruption tail. **Candidate to replace V39 at `Profile::A`** —
fixes the #1 correctness defect (blur > identity) that V39 violates.

## What it is (lineage)

Two-step construction, both rank-invariant from the base network:

1. **Base network = v47-masked-strict** (trained earlier 2026-05-27;
   methodology: `benchmarks/v47_masked_monotone_2026-05-27.md`). The
   V32-faithful 2-layer per-sample-α recipe + `--monotone-cbc` +
   `--monotone-feature-mask benchmarks/feature_sign_mask_2026-05-26.tsv`
   (300 sign-safe pinned W1≥0 / 72 free) + `--monotone-strict`. W1≥0 on
   the sign-safe set + rank_w≤0 + tanh output pin ⇒ score monotone-↓ in
   every error feature, identity is the unique max, output bounded ≤100 —
   the blur>identity defect is fixed BY CONSTRUCTION.
   Base bake: `/mnt/v/output/zensim/bakes/v47_masked_strict_2026-05-26.bin`.

2. **Dial recalibration (this doc)** — the base bake's auto-spline
   degenerated (2 knots → all-negative output → G1=0.00). Replaced via the
   V10 spline-retrofit path (`scripts/v_next/recal_v47_dial.py`, NEG_TAIL=1):
   strip the old spline → predict the pre-spline tanh-pin output on the
   multiband anchor → fit a 17-knot monotone PCHIP `tanh-pin → target_score`
   → inject via `zenpredict bake`. Monotone spline ⇒ **SROCC unchanged**.

- **Ship artifact (packed)**: `v47_strict_recal_negtail_packed30k_2026-05-27.bin`
  (29,995 bytes, md5 `4c6cfc67769132f01bc8cca81cc6d597`) — f16 + global
  zerobias 0.005 + lz4 + spline refit on the packed net, via the standard
  `pack_and_calibrate.py` path (`benchmarks/standard_bake_packing_2026-05-27.md`).
  f32-equivalent on the full panel (CID22 0.8564), identity 97.5, 0
  above-identity, 6.6× smaller. **This is the bake to ship.**
- **f32 reference**: `v47_strict_recal_negtail_2026-05-27.bin`
  (198,520 bytes, md5 `1792624baeebd5ed1868403becbd34e0`). The clean-[0,100]
  variant (flat-0 below the anchor, no negative tail) is
  `v47_strict_recal_2026-05-27.bin` (md5 `9f58881a52771290c2ae024dbc370e63`).
  All on /mnt/v (>30 KB binaries).
- **Architecture**: 372→…→64 per-sample-α head + tanh output pin +
  feature_transforms, ZNPR v3, `n_layers=3`.

## Calibration (the recal spline)

- Fit corpus: `multiband_anchor_dial100.parquet` (2000 rows, per-row
  `target_score` ∈ [0,97.4], CID22-val-clean training anchor — V39's anchor).
- Pre-spline (tanh-pin) range on the anchor: **[49.37, 49.76], spread 0.39**,
  corr(tanh-pin, target_score)=0.877. 17 monotone knots map it to [0, 95.4].
- **NEG_TAIL=1**: the bottom keeps a single dial=0 knot so PCHIP extrapolates
  with the (steep, positive) bottom-segment slope BELOW the anchor's worst
  honest encode → worse-than-honest inputs score < 0 (corruption resolution),
  instead of flat-0.

## Held-out panel (bake_verdict, full Mohammadi)

| Corpus | n | SROCC | PLCC | KROCC | PWRC | Z-RMSE | DS-AUC |
|---|--:|---:|---:|---:|---:|---:|---:|
| CID22 | 4292 | **0.8547** | 0.8410 | 0.6619 | 0.9754 | 0.541 | 0.8078 |
| KADIK10k | 10125 | 0.8030 | 0.8021 | 0.6081 | 0.9522 | 0.597 | 0.7322 |
| TID2013 | 3000 | 0.7965 | 0.8253 | 0.6107 | 0.9686 | 0.565 | 0.7812 |
| KonJND-1k | 1008 | **0.4850** | 0.4167 | 0.3353 | 0.8144 | 0.909 | 0.5310 |
| AIC-3 CTC | 600 | 0.7700 | 0.7860 | 0.5999 | 0.9341 | 0.618 | 0.7222 |
| AIC-4 sample | 300 | 0.8902 | 0.8774 | 0.7136 | 0.9758 | 0.480 | 0.8268 |

Scorecard: **G1 dial 0.99** (pooled p5=−21.8 p95=84.5), G7 CID22 1.00
(0.8547 ≥ 0.85), G5 HF 0.23 (KonJND 0.485, still < 0.70 floor — the
characterized HF Pareto limit), G8 Z-RMSE 0.61, G9 DS-AUC 0.15.
**Weighted goal score 0.644** (v47-strict pre-recal: 0.265).

## Correctness (the #1 goal) — blur ladder, OOD synthetic

0 inversions, 0 above-identity across all 4 contents; identity = 97.8 (max);
score range [−190.1, 97.8]. Heavy blur degrades gracefully to −190/−92/
−65/−44 (negative-tail resolution). **V39: 27 inversions, 31 above-identity
(blur scores ≥ identity).** This is the defect this bake fixes.

## vs V39 — the tradeoff (advisory gates, user's call per shipping policy)

| | v47-strict-recal-negtail | V39 (shipped A) |
|---|--:|--:|
| blur > identity defect | **FIXED** (0 above-id) | VIOLATED (31 above-id) |
| monotone-by-construction | **YES** | no |
| negative corruption tail | **YES** ([−190,98]) | no (OOD inversions) |
| dial G1 | 0.99 | 1.00 |
| CID22 SROCC | 0.8547 | **0.8793** (+0.024) |
| CID22 Z-RMSE / DS-AUC | **0.541 / 0.808** | 0.584 / 0.739 |
| KADID SROCC | 0.8030 | **0.9251** (+0.122) |
| TID SROCC | 0.7965 | **0.9317** (+0.135) |
| KonJND SROCC | **0.4850** (+0.065) | 0.4197 |

**Gains**: correctness (the #1 non-speed goal), monotonicity, corruption
resolution, CID22 calibration, KonJND. **Costs**: CID22 −0.024 (above the
0.85 floor), KADID −0.122 / TID −0.135 — the analytic-distortion-ranking
signal in the 72 dropped sign-flip features. Per the training goals KADID/
TID are integrity guards, not the compression target; CID22 (the compression
gold standard) stays competitive, and V39 is the outlier-high on KADID/TID
vs conventional baselines (ssim2/cvvdp).

## Honest gaps

1. **Top-of-dial hypersensitivity.** The honest dial 64→95 is crammed into a
   ~0.002-wide tanh-pin window (the tanh-pin compresses real content to a
   0.39-wide band). Rank is unaffected (SROCC preserved), but fine high-q
   codec targeting (q resolving to <1 dial point) is brittle. The q-sweep
   monotonicity goal (G3) must be measured before relying on the high-q dial.
2. **KADID/TID rank cost** (above) — strict-monotonicity price.
3. **G5 KonJND HF still 0.485 < 0.70** — the characterized HF Pareto limit,
   unchanged by recal (it's a representation gap, not a calibration one).
4. **Corruption-corpus gate not yet run** — the negative tail is verified on
   the OOD blur ladder, not yet on the structural-corruption corpus
   (codec-corpus#7 / PR#8). Run `score(corruption) < score(q20)` once the
   corpus lands.

## Decision

Ship gate is advisory (2026-05-14 policy): CID22 −0.024 + KADID/TID rank
loss are surfaced, not blocking. This is a **user call**:
- **Replace V39 at `Profile::A`** — gains correctness + dial + corruption
  resolution + better calibration; accepts −0.024 CID22 + KADID/TID. Needs
  the 198 KB weight committed to `zensim/weights/` (binary >30 KB → user
  confirmation) + the `include_bytes!` flip in `zensim/src/profile.rs`.
- **Ship as a sibling profile** (`Profile::A_Strict` / "monotone similarity")
  — keep V39 as `Profile::A` for the codec dial; the strict bake serves the
  regression-test / general-similarity use case (zensim-regress: broken <
  honest-lq, which V39 cannot guarantee).
