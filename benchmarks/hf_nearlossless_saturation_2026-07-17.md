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

## RESULT — the feature TRANSFORMS are the near-lossless killer (decisive)

Raw HF SROCC vs feature-shaping (strip spline; original depth-iter bakes):

| bake | raw HF | imazen-26 | nonphoto | CID22 | KonJND |
|---|---|---|---|---|---|
| **fs_none (NO transforms)** | **0.188** | 0.865 | 0.877 | 0.868 | 0.768 |
| fs_minlift020 (few transforms) | 0.027 | 0.949 | 0.949 | 0.858 | 0.723 |
| fs_minlift002 (many transforms) | 0.036 | 0.942 | 0.947 | 0.828 | 0.254 |

Removing the winsor_p99 / yeo_johnson transforms lifts raw HF **7×** (0.027→0.188).
The transforms compress the tiny near-lossless feature differences into a flat
region → the MLP can't rank near-lossless. They are the SAME mechanism that buys
the ssim2 ceiling (imazen-26 0.949): a direct trade of near-lossless (and CID22 +
KonJND) FOR mid-quality ssim2-agreement. `--high-q-boost 4` made HF WORSE (0.011),
not better — you can't upweight rows the features don't separate.

**Verdict:** the ssim2 ceiling and a working near-lossless dial are mutually
exclusive on this recipe; even transform-free, HF tops out at 0.19 (the metric's
genuine weak zone). fs_minlift020's 0.949 is an ssim2-specialist number bought
specifically at the cost of the high-fidelity zone. The only real lever measured is
"less shaping" (fs_none / raise --auto-transforms-min-lift) — it recovers HF + CID22
+ KonJND but forfeits the ssim2 ceiling, landing back near B. Consistent with the
feature_shaping "less is more" verdict + "nothing robustly beats B".

Follow-on (untested): a PER-FEATURE transform exemption — apply transforms only to
features that don't carry near-lossless signal — could in principle keep both, but
needs a near-lossless-discriminability analysis to pick the exempt set.

## STRATEGIC REFRAME — near-lossless is NOT universal; it's a deep-MLP+transforms failure

Raw HF SROCC across architectures (strip spline):

| bake | raw HF | imazen-26 | arch |
|---|---|---|---|
| B (shipped linear) | **0.614** | 0.841 | linear + winsor |
| A (v47) | 0.622 | 0.862 | small MLP |
| min-max k24 | 0.568 | 0.880 | monotone, same transforms |
| fs_none | 0.188 | 0.865 | deep MLP, no transforms |
| fs_minlift020 | 0.027 | 0.949 | deep MLP + transforms |

**B, A, and the min-max all rank near-lossless ~0.6** — near-lossless is NOT a
universal weak zone. It is SPECIFICALLY the deep MLP + transforms that breaks it.
So fs_minlift020's imazen-26 0.949 is a **mid-quality OVERFIT**: the depth + shaping
fit the bulk of the imazen-26 distribution but fail to generalize to the near-lossless
tail (0.027). The min-max uses the SAME transforms yet keeps HF 0.568 — so it's the
depth/capacity (non-monotone overfit), compounded by transforms, not the transforms
alone. The simpler / monotone models sacrifice the ssim-2 ceiling but stay coherent
across the WHOLE quality range — which is what a dial needs.

## Retrain confirmation (deterministic)

- `fshf_base` (control) = fs_minlift020 EXACTLY: HFraw 0.0270, imazen-26 0.9490
  (recipe reproduces).
- `fshf_hqb4` (--high-q-boost 4): HFraw 0.0108 (WORSE), imazen-26 0.9092 (dropped).
  high-q-boost destabilizes — you can't upweight rows the features don't separate.

**Bottom line:** fs_minlift020 is an ssim-2-mid-quality specialist that cannot serve
as a general dial (dead near-lossless zone, and the levers to fix it forfeit the
ceiling → back to B). This is WHY the depth-MLP campaign concluded "nothing beats B":
B's linear simplicity is what keeps it coherent across the full range including the
near-lossless tail the depth MLPs overfit past.
