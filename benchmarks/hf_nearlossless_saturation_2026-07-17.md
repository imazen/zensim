# Depth-MLP near-lossless (HF) saturation — TWO bugs (2026-07-17)

`fs_minlift020` (min-lift 0.20 psa+tanh depth-MLP, `depth_v2.toml`) is the ssim2
ceiling — imazen-26 0.949 / nonphoto 0.949, clears KonJND G5 (0.723 @ s13) — but
**HF near-lossless SROCC = 0.000**. The user asked to fix it by "training with
hf_nearlossless". Diagnosis first corrected the premise, then split the bug in two.

## Premise correction: HF is ALREADY in the training

`depth_v2.toml` already carries `hf_nearlossless_train.parquet` as a group with
`within_ref = true`, train_w 0.5 (added for exactly the near-lossless ladder — its
note cites "the ladder moves ~0.92 ssim2 pts within-image vs ~6 between"). So the
0.000 is NOT a missing-data problem.

## Decisive diagnosis — spline collapse ON TOP OF a broken raw ranking

Corrupted the `zentrain.output_calibration_spline` metadata key (1-byte rename →
runtime ignores it → raw psa-head output), re-evaluated HF:

| | HF pred spread (n=300) | HF SROCC |
|---|---|---|
| WITH spline | 188.00 — all identical (1 distinct) | 0.000 |
| RAW (no spline) | 28.7 – 44.5 (282 distinct) | **0.027** |

Two bugs, in series:
1. **Spline collapse (top dead-zone).** The dial spline (fit on
   `multiband_anchor_dial100`, target ≤97) maps the ENTIRE near-lossless raw range
   [28.7, 44.5] to a flat 188 — near-lossless raw is out-of-distribution above the
   anchor's top knot, so it all extrapolates to the flat top. A monotone spline
   preserves rank EXCEPT where it's flat; this flat plateau is where it destroys it.
   (This is also the 27% dial dead-zone and the p95=188 overshoot — same bug.)
2. **Raw ranking is the real bottleneck.** Even pre-spline, HF SROCC is 0.027 — the
   psa-head output VARIES on near-lossless (282 distinct) but does NOT order it. The
   MLP's ssim2-agreement (imazen-26 0.949 on mid-quality) COLLAPSES in the
   near-lossless zone despite the HF withinref group. Fixing the spline alone
   recovers essentially nothing (0.027 ceiling).

So the near-lossless weak zone is a RAW-RANKING failure first, spline-collapse
second. The HF group at train_w 0.5 + no high-q-boost is too weak a signal for a
subtle-feature zone.

## Experiment (in flight) — strengthen the near-lossless signal

`fs_minlift020` recipe + the trainer's documented HF levers:
- `fshf_base` — reproduce (control, expect HF raw ~0.03).
- `fshf_hqb4` — `--high-q-boost 4.0` (upweight human_score≥90 rows in RankNet).
- `fshf_hqb4_w3` — + `hf_nearlossless` train_w 0.5→3.0.

Gate: does RAW HF SROCC lift meaningfully off 0.027 while imazen-26/nonphoto hold
(≥0.94)? If yes → then refit the spline over the extended raw range to un-collapse
the top. If RAW HF stays ~0 even at high-q-boost 4 + w3, the near-lossless FEATURES
don't separate quality at this resolution — a feature-extraction limit, not a
training-weight one (the honest "near-lossless is the metric's weak zone" verdict).
Results: `/mnt/v/output/zensim/fs-hf/`.
