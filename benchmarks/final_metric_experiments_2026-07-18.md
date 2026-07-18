# Final dial metric — running experiment log (2026-07-18)

Campaign to validate/build the additive-core dial metric per
`docs/FINAL_DIAL_METRIC_DESIGN_2026-07-18.md`. Every experiment gets a hypothesis, a
result, and a verdict. Accumulate; don't overwrite.

## Baselines (context)
- **B (shipped linear)**: CID22 0.876, near-lossless HF 0.614, corruption gate (DIALED) 18%,
  diffmap spatial ~0.87–0.91.
- **basic-156 additive core** (simple 5-group linear, seed 13, no spline): CID22 0.8978,
  imazen26 0.8353, nonphoto 0.8531, HF 0.4446, corruption gate (RAW) 85.6%, diffmap ~0.987.
- Confirmed prior: additive→exact diffmap 0.987 (`diffmap_block_coherence`); basic≥full quality.

## Experiment queue
- E1  spline-preservation: full-range monotone spline on additive core preserves corruption+CID22+HF?
- E2  verify 85.6% + per-region anomaly on full corruption corpus (not the 222-subset)
- E3  winsor additive core: recover B's HF 0.614 with winsor transforms on basic block?
- E4  ModelCoherent diffmap for the SHIPPED path (additive core's own weights → 0.987 diffmap)
- E5  negative tail (kadis_negrich) through the additive core + full-range spline
- E6  dial monotonicity of the additive core on a q-sweep (Eval A foundation)
- E7  L2 max-supplement: peak/max severe term fixes edge_border w/o disturbing honest?
- E8  monotone GAM core: beat linear on CID22 while staying additive?
- E9  one-shot targeting residual (Eval C) — q_hat regressor

---
## E1 — spline-preservation → OVERTURNED: it's the WINSOR transforms, not the spline
Hypothesis: B's 18% corruption gate was the dial spline flattening the corruption range.
FALSIFIED. Measured B's RAW (pre-spline) corruption gate = **18.0%** (= its dialed gate;
`mapped` post = 0%). So B's MODEL doesn't rank corruption low — the spline is innocent.
The difference from basic-156 (85.6% raw) is the RECIPE. Prime suspect: B's `winsor_p99`
transforms clip the extreme distortion features corruption produces (which is *why* winsor
lifts near-lossless HF 0.614 vs basic-156's 0.445 — same outlier-taming mechanism). So
corruption-gating vs near-lossless is a **WINSOR tradeoff**, located in the feature transforms,
not the spline/pooling. This supersedes the "preserve-through-dial" reframe. → test E3.
