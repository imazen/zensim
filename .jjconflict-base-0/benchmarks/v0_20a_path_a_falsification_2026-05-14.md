# V0_20a Path A (IW + V0_18-style 3-way concat with TV) — falsification + B3-specialist finding

**Date**: 2026-05-14 evening
**Falsification gate (per docs/v0_20_path_evaluation_2026-05-14.md)**:
> if the 3-way concat CID22 SROCC < 0.880, stop pushing on single-MLP
> IW. iw_k1 standalone is at 0.8657; TV regularization + 3-way concat
> should add 0.01-0.02. Below 0.880 means the IW signal isn't
> materially compatible with this architecture.

**Result**: 3-way concat aggregate **0.8582** < 0.880 → **gate reached.**

## Bakes trained for Path A

| Bake | Type | val_mean SROCC |
|---|---|---:|
| iw_k1_full_s1 | base, no TV | 0.9342 |
| iw_k1_tv_s1 | cycle-14 TV seed=1 | **0.9371** |
| iw_k1_tv_s42 | cycle-14 TV seed=42 | 0.9354 |

TV regularization gave +0.003 over base on s1. Three components concatenated
at V0_18's 0.65/0.30/0.05 coefficients via `concat_three_way`
(generalized for 372-input bakes in commit `6d5bce0`), affine-calibrated
with V0_18's α/β.

## Per-band CID22 result (iw_k1 3-way concat, affine calibrated)

| Band | n | iw_k1 concat | baseline_228_full_s1 | V0_18 ship |
|---|---:|---:|---:|---:|
| Aggregate | 4292 | **0.8582** | 0.8644 | 0.8933 |
| B0 [<50] | 324 | 0.3733 | 0.3455 (+0.028) | 0.4309 |
| B1 [50,65) | 1010 | 0.4207 | 0.3852 (+0.035) | 0.4556 |
| B2 [65,90) | 2915 | 0.7237 | 0.7316 (−0.008) | 0.7842 |
| B3 [≥90] | 43 | 0.0879 | 0.0202 (+0.067) | 0.1545 |

The 3-way concat wins 3 of 4 bands over the baseline (B0, B1, B3) but
loses on B2 — and B2 dominates the aggregate (n=2915 vs the others'
324/1010/43).

## --high-q-boost=3 result (Path F#1, no IW)

| Bake | CID22 agg | B0 | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|
| baseline_228_full_s1 (no hq) | 0.8644 | 0.3455 | 0.3852 | 0.7316 | 0.0202 |
| baseline_228_hq_s1 (hq=3) | 0.8141 | 0.2739 | 0.3385 | 0.6708 | **0.1294** |

**hq=3 boosts B3 by +0.109 SROCC** at the cost of −0.050 on aggregate.
The boost works as designed — but hq=3 is too aggressive for a
single-bake aggregate ship.

## ENSEMBLE: V0_18 ship + baseline_hq (multi-output Pareto)

Output-mix simplex search:

```
best agg: α_v0_18_ship=1.0, α_iw_k1_concat=0.0, α_baseline_hq=0.0
          → SROCC 0.8933 (= V0_18 alone)
best B3:  α_v0_18_ship=0.6, α_iw_k1_concat=0.0, α_baseline_hq=0.4
          → SROCC 0.3349 (+0.180 over V0_18 alone B3 = 0.1545)
```

**The cleanest V0_20a ship form is multi-output**:
- V0_18 ship alone for aggregate (0.8933, unchanged).
- 60/40 mix of V0_18 + baseline_hq for B3 (0.3349, +0.180 lift).
- IW features (iw_k1, iw_k4) DO NOT contribute to either Pareto front
  on CID22.

The IW signal IS real (Wang & Li 2011 + Mohammadi 2025 both
demonstrate it independently), but our MLP architecture cannot
extract more lift from it than the cheaper "B3-row-boost" trick
achieves. Hypothesis: the V0_18 ship's 228 basic+peaks features
already saturate the IW signal at our MLP capacity.

## Verdict

- **Path A is falsified for aggregate ship.** IW features at single
  strength (k=1 or k=4) + V0_18-style 3-way concat + TV don't break
  the V0_18 ship's CID22 aggregate ceiling (0.8933).
- **B3 lift IS achievable**: via the `--high-q-boost` trainer flag,
  no IW features needed. The hq-boosted baseline (228 features)
  works as a B3 specialist in a multi-output ensemble.
- **Multi-output ship is the V0_20a path forward**: V0_18 + B3
  specialist. Architecture support is the work item — runtime needs
  to forward TWO MLPs and mix outputs.

## Tasks queued out of this

- V0_20a-ship-form: implement multi-output ensemble (V0_18 + B3
  specialist) in `Zensim::compute` — adds ~10 µs/call. Profile slot
  `PreviewV0_4_with_b3_head` or similar.
- V0_20b distortion-manifold (Su 2023): now the highest-conviction
  path for aggregate CID22 lift since IW is exhausted at this MLP
  capacity. Task #48.
- V0_22 CVVDP distillation: highest absolute upside per
  Mohammadi 2025. Task #49.

## Falsification math (per the rigor policy)

- Wang 2011 paper claim: IW-SSIM +0.006 SROCC vs MS-SSIM on TID2008.
- Our delta achieved: −0.0351 on aggregate (0.8582 vs 0.8933 V0_18).
- Sweep depth: 6 single-MLP cells + 3 full-recipe cells + 1 3-way
  concat = 10 configurations across k ∈ {1, 4, 8}, seed ∈ {1, 42},
  hidden ∈ {128, 256}, with/without TV.
- Falsification rigor: paper-claimed configuration explored at
  multiple scales of our recipe. The architecture mismatch
  (Wang 2011: weighted-pool MS-SSIM; ours: MLP-over-features) is
  the most likely root cause.

## Reference

- Wang & Li 2011 IW-SSIM (IEEE TIP)
- Mohammadi 2025 (arXiv:2509.13150) — IW-SSIM as the BEST classical
  metric at HF, but on a DIFFERENT corpus (JPEG AIC-3) than ours
- This session: `benchmarks/v0_20a_sweep_methodology_2026-05-14.md`,
  `benchmarks/v0_20a_smoke_methodology_2026-05-14.md`,
  `docs/v0_20_path_evaluation_2026-05-14.md`
