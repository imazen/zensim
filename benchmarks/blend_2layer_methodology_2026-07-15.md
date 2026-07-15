# 2-layer diverse MLP — optimal-blend "best output" (2026-07-15)

User directive (2026-07-15): *"keep iterating to get the optimal input blend and best
output, make html output do bandwise reports and every graph possible."* This documents the
best-output candidate the blend search produced. **It is a CANDIDATE — Profile B is NOT
swapped.** The `include_bytes!` in `zensim/src/profile.rs` is untouched; swapping is user-gated
(new architecture vs the deliberately-linear B; a real KonJND regression to weigh — see §H).

## H. Honest gaps first (per NEVER CLAIM FALSE COMPLETION)

- **KonJND −0.038** (0.5466 → 0.5086): the 2-layer loses the HF/PJND regime to B. This is the
  standing **G5 structural limit** (falsified across architectures per zensim CLAUDE.md); no
  bake in the program clears the 0.70 floor, and the diverse MLP sits slightly below B here.
- **Size 272 KB** vs B's 7 KB (272 KB f32 2×128-wide MLP vs B's linear 372→1). A future f16
  repack (`pack_and_calibrate.py`) would ~halve it; not done here.
- **Dial top-end p95 94.7 vs B's 99.7** — B reaches higher on the codec dial. Both pass G1;
  the 2-layer's dial spline could be top-extended (`bake_dial_refit extend-top`) if desired.
- **G9 DS-AUC (AIC-3) 0.711** and **G5** remain unmet — unchanged from B; not a regression the
  blend can fix (needs an HF feature representation, not more/different training data).

Everything else is a win (§E).

## A. Architecture + identity

- **Arch**: `372 → 128 → 128 → 1`, LeakyReLU(0.01) on both hidden layers, identity output.
  Two hidden layers is the load-bearing change (§C).
- **Runtime transforms**: 372 per-feature `winsor_p99` clips (de-poison the bigcodec IW/masked
  block, f228–371) applied pre-scaler, identical to training; standardization scaler; 18-knot
  PCHIP output-calibration dial spline (ssim2-anchored, range **[−18.1, 95.8]**).
- **dtype** f32. **Size** 271,900 B. **md5** `8c689610126071993559c85908910df2`.
- **Bake** `/mnt/v/output/zensim/reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin`
  (pointer: `benchmarks/mlp_2layer_diverse_2026-07-15.bin.pointer.md`; >30 KB, not committed).

## B. Trainer command + hyperparameters

`scripts/v_next/blend_search.py --round 3 --seeds 1,7,13,17,23`, config **`r3-2L-H128`**:

```
spec = {safesyn:1, cid22_train:1, kadid:1, tid:1, bigcodec:1.0, kadis:0.3}
hp   = {layers:2, hidden:128, epochs:400, lr:1e-3, weight_decay:1e-5,
        winsor_pct:0.1, div_cap:120000, hq_band:85, hq_weight:0.3, safesyn_cap:90000}
loss = weighted smooth_l1 on standardized ssim2 target; per-corpus 75/25 train/val split.
```

The saved/baked payload is **seed 13** (blend_search saves the median seed of the 5); the
reported 5-seed-mean CID22 is 0.8825, the baked seed-13 bin verifies at **0.8807** (§E — the
0.002 gap is seed + f32 + spline, all rank-near-invariant). Trainer/scorer:
`scripts/v_next/blend_lib.py` (numpy forward matches the baked runtime; self-verified by the
bake round-trip). Bake via `scripts/v_next/bake_mlp_negatives.py` (extended for 2-layer today)
→ `zenpredict bake` (JSON-pipeline mandate).

## C. Why 2 layers — the trade-break (the core finding)

The **input blend is saturated on the ssim2 signal** (round 1: adding the now-clean kadid/tid
gives no gain — composite flat within seed noise; 13k kadid/tid ssim2 labels are negligible vs
safesyn's 90k). The real axis is the **CID22 ↔ non-photo trade**: bigcodec/imazen-26 (the
diverse real-codec corpus) buys non-photo coverage but a *single-layer* net pays ~0.03 CID22 for
it, because one hidden layer cannot fit the photographic and non-photographic distortion
manifolds simultaneously. **A second hidden layer has the capacity to fit both** → CID22 returns
to the photo-only ceiling AND non-photo jumps. Measured (5-seed mean, `benchmarks/blend_search_r3_2026-07-15.tsv`):

| variant | CID22 | non-photo | Δ mechanism |
|---|---|---|---|
| ssim2-only 2-layer (photo ceiling) | 0.8874 | 0.8624 | no diverse data → non-photo-blind |
| **1-layer diverse** (r3-1L-ref) | 0.8551 | 0.9297 | pays 0.032 CID22 for non-photo |
| **2-layer diverse (r3-2L-H128)** | 0.8825 | 0.9527 | recovers CID22 AND keeps non-photo |

The 2-layer sits within 0.005 of the photo-only CID22 ceiling while scoring non-photo 0.95.
Confirmed across 7 two-layer variants (H64/96/128, div 0.3–1.0, ep700) — not a seed artifact.

## D. Calibration

18-knot PCHIP dial spline, fit raw-MLP-output → `ssim2_gpu` on
`canonical-2026-05-21/train/multiband_anchor_dial100.parquet`, negative-capable (bottom knot
−18.1). SROCC is rank-invariant under the monotone spline (verified by the numpy↔bake match).

## E. Held-out panel (baked bin, measured by `bake_verdict`, vs shipped B)

`bake_verdict` on the 6 canonical val parquets + the imazen-26 non-photo held-out. **Bold = win.**

| corpus | n | shipped B SROCC | 2L-H128 SROCC | Δ |
|---|---|---|---|---|
| **CID22** (primary holdout) | 4292 | 0.8764 | **0.8807** | +0.0043 |
| **TID2013** | 3000 | 0.7868 | **0.8430** | **+0.0562** |
| **non-photo** (imazen-26) | 10000 | 0.8606 | **0.9495** | **+0.0889** |
| **AIC-3 CTC** | 600 | 0.7774 | **0.7865** | +0.0091 |
| **AIC-4 sample** | 300 | 0.8906 | **0.8940** | +0.0034 |
| KADIK10k (integrity guard) | 10125 | 0.8201 | 0.8169 | −0.0032 |
| KonJND-1k (HF/G5) | 1008 | 0.5466 | 0.5086 | −0.0380 |

Full Mohammadi panel (PLCC/KROCC/OR/PWRC/Z-RMSE) + 10-band per corpus in
`verdict_2L_H128_2026-07-15.md`. CID22 10-band (n per band, SROCC): B3 57/0.079, B4 266/0.248,
B5 615/0.407, B6 836/0.410, B7 1092/0.376, B8 1382/0.479, B9 43/0.050 — the aggregate 0.881 is
the release-gate number (narrow-band SROCC is rank-compressed by construction, same for all bakes).

Goals scorecard (baked): **G1 dial range 1.00 ✓, G3 monotonicity ✓, G7 CID22 ≥0.85 → 0.881 ✓,
G-NP non-photo ≥0.85 (target 0.93) → 0.9495 ✓ (exceeds target)**; G5 HF 0.29 ✗ (structural),
G8/G9 AIC-3 soft (unchanged from B).

## F. Dial panel (codec-target monotonicity — the make-or-break)

The reason B is linear is dial monotonicity; a 2-layer MLP with great SROCC but a bumpy dial
would be useless for "user types a target score, codec binary-searches q." Measured on the
quarantined densified multi-codec dial grid (4457 rows, 106 curves, 4 codec families):

| metric | shipped B | 2L-H128 | gate |
|---|---|---|---|
| monotonicity (1 − inversions >0.5pt) | 0.9740 | **0.9683** | G3 ≥ 0.93 ✓ |
| flat / clamp dead-zone | 0.0000 | 0.0000 | G3 ≤ 0.05 ✓ |
| dial p5 / p95 | 13.7 / 99.7 | 10.9 / 94.7 | G1 ✓ |

**The 2-layer keeps a monotonic, dead-zone-free dial (0.968 ≥ 0.93).** It is usable for
codec-targeting; the −0.006 vs B is within the gate and the top-end p95 gap is spline-fixable.

## G. Data lineage (all training inputs)

| corpus | path | target | rows | CID22-contam |
|---|---|---|---|---|
| safesyn | `canonical-2026-05-21/train/safesyn.parquet` | ssim2_gpu | 196,086 (cap 90k) | purged (§3.x) |
| cid22_train | `canonical-2026-05-21/train/cid22_train.parquet` | ssim2_gpu (NOT MOS) | ~17k | ssim2-anchored, disjoint from 49-ref holdout |
| kadid | `canonical-2026-05-21/train/kadid.parquet` | ssim2_gpu (**FIXED §3.18**) | 10,125 | I01–I81, d≤10 clean |
| tid | `canonical-2026-05-21/train/tid.parquet` | ssim2_gpu (**FIXED §3.18**) | 3,000 | I01–I25, d≤10 clean |
| bigcodec (DIV) | `bigcodec_hqdedup_traindigits_2026-07-02.parquet` | human_score(ssim2/100)×100 | 2.32M (cap 120k, HQ>85 ×0.3) | val-origin held out |
| kadis (NEG) | `kadis_sample_negrich.parquet` | score_ssim2_gpu | ~266k (cap 90k, w0.3) | Konstanz KADIS refs |

CID22-49 human MOS is **never** a training target (validation-only, per CLAUDE.md). All targets
are ssim2-derived (NOT `score_zensim` — that would distill Profile A).

## Reproduce

```
python3 scripts/v_next/blend_search.py --round 3 --seeds 1,7,13,17,23   # -> blend_r3_3_r3-2L-H128.npz
python3 scripts/v_next/bake_mlp_negatives.py --npz .../blend_r3_3_r3-2L-H128.npz \
    --out .../mlp_2L_diverse_H128_2026-07-15.bin --dtype f32
ZENSIM_DIAL_GRID=.../dial_grid_372col_2026-05-29_quarantined.parquet \
    ./target/release/bake_verdict --bake .../mlp_2L_diverse_H128_2026-07-15.bin
```

Dashboard: `scripts/v_next/bandwise_dashboard.py` →
`http://172.23.240.1:3300/zensim/dashboards/bandwise_dashboard_2026-07-15.html`.
