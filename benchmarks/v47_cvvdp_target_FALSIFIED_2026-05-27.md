# v47-CVVDP-target — FALSIFIED (reproduces V41), 2026-05-27

**Hypothesis** (user, 2026-05-27): "try learning from cvvdp instead of ssim2
scores" — train the v47-strict recipe on the CVVDP target instead of the
ssim2-derived human_score, hoping for a better metric / diffmap for
jxl-encoder.

**Falsification gate**: if the cvvdp-trained bake regresses across the full
Mohammadi panel on the held-out corpora, scalar-CVVDP-target is dead (this is
the V41 result; re-confirm before pursuing).

## Result — worse on EVERY corpus, EVERY stat (full panel, not SROCC-only)

| Corpus | CVVDP-target | ssim2-target (shipped v47) | Δ SROCC |
|---|--:|--:|--:|
| CID22 | 0.6963 | 0.8657 | **−0.169** |
| KADIK10k | 0.7101 | 0.7933 | −0.083 |
| TID2013 | 0.6799 | 0.7927 | −0.113 |
| KonJND-1k | 0.3288 | 0.4185 | −0.090 |
| AIC-3 CTC | 0.6098 | 0.7680 | −0.158 |
| AIC-4 sample | 0.5810 | 0.8854 | −0.304 |

PLCC, KROCC, PWRC, Z-RMSE, DS-AUC are all worse on every corpus too — the
panel agrees with SROCC (≥3 stats agree on regression on every corpus, so the
"don't trust SROCC alone" guard is satisfied: this is a real regression). The
dial is also worse: G1 0.58 (p5=−47.1, p95=70.3 — doesn't reach the top of
the dial) vs ssim2's 0.97.

CID22 0.6963 reproduces the documented **V41 finding** (CID22 0.66 vs 0.88)
almost exactly: **training a scalar metric to predict CVVDP's output produces
a worse human-MOS predictor.** Emulating CVVDP's scalar OUTPUT ≠ having its CSF
mechanism. Same recipe, same data (minus konjnd which has no cvvdp), only the
target column changed (cvvdp_log_norm) — so the regression is attributable
purely to the target.

## The distinction that matters for the diffmap goal

This falsifies **scalar-CVVDP-target** (train the metric to predict cvvdp's
*number*). It does NOT test **spatial-CVVDP-diffmap-target** (train the
metric's per-pixel DIFFMAP against cvvdp's per-pixel visual-difference map) —
which is the version that could plausibly help the jxl-encoder diffmap, since
CVVDP's spatial error localization is its strength.

**We can't run the spatial version yet**: the canonical training data carries
cvvdp SCALAR scores (`cvvdp_log_norm`), not cvvdp DIFFMAPS. Producing spatial
cvvdp targets needs a cvvdp-diffmap generation pass (zenmetrics cvvdp with
per-pixel output) over the training pairs — a data-generation effort, not a
recipe tweak.

## Verdict + recommendation

- **Do NOT ship the cvvdp-scalar bake** (`v47_cvvdp_2026-05-27.bin`, 34.9 KB) —
  it is a strictly worse metric than the shipped v47-ssim2 on every held-out
  measure. Kept on disk for the record, not wired into any profile.
- **Do NOT pursue scalar-CVVDP-target further** — V41 + this run are two
  independent confirmations of the same dead end.
- For the diffmap goal (#38): the shipped v47-ssim2 diffmap stays the
  baseline. If CVVDP is worth pursuing for the diffmap, the path is
  **spatial-cvvdp-diffmap supervision** (needs cvvdp diffmaps generated first),
  NOT scalar-target. Surface this to the user before investing in the
  diffmap-generation pass.

Recipe: `zensim/weights/manifests/v47_cvvdp.toml`. Train log:
`/mnt/v/output/zensim/cvvdp-experiment-2026-05-27/train.log`. Verdict:
`/mnt/v/output/zensim/bakes/v47_cvvdp_2026-05-27.verdict.md`.
