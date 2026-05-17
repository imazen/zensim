# V0_20+ path evaluation — which approach is most likely to ship a real lift? (2026-05-14)

After today's V0_20a IW-SSIM experiments, the empirical landscape:
- Single 372→128 MLP at iw_k1 reaches CID22 0.8657 (still below V0_18 ship 0.8933 by 0.028).
- Output ensemble of baseline + iw_k1 across seeds reaches 0.8834.
- IW lifts B3 [≥90] by +0.267 over V0_18 ship in a 50/50 mix (n=43, noisy).

That gives concrete signal on the **IW** mechanism. The bigger question:
across the queued V0_20 paths (a/b/c/d, plus V0_22), which is most
likely to actually break V0_18's CID22 0.8933 ceiling and lift the
priority B0–B5 bands?

## Criterion: rigor + B0-B5 priority + ship-grade lift

Per the user directive 2026-05-14:
- B0..B5 lift is the dominant priority (low/mid q is where compression
  product decisions live).
- Push each experiment to the paper-claimed benefit before falsifying.
- Architecture is open if scientifically motivated.

Three axes for ranking:
1. **Paper-claimed lift magnitude.** What did the source paper actually
   show on a corpus comparable to ours?
2. **Mechanism fit.** Does the mechanism target where we're weak
   (B0-B5), where we're strong (already saturated), or somewhere
   irrelevant?
3. **Engineering cost.** Implementation effort + sweep compute + risk
   of unintended regressions.

## Path comparison

### A. **iw_k1 with TV in V0_18 3-way concat** (today's V0_20a follow-up)

- **Paper claim**: Wang & Li 2011 reports +0.006 SROCC weighted-avg
  across 6 IQA DBs. Our recipe is closer to MLP-on-features than
  uniform-pool MS-SSIM, so this number is a directional reference
  not a target.
- **Mechanism fit**: Wang 2011 + Mohammadi 2025 both show IW shines
  at **high fidelity (B7-B9)**. Our priority is B0-B5. **Mismatch**.
- **Cost**: low — reuses existing trainer + concat tools. 2 TV-regularized
  components are training right now (~30 min).
- **Probability of CID22 aggregate lift**: 30 %. The MLP could pick up
  the IW signal where it actually helps, but our val_policy=min
  selector + RankNet loss don't naturally weigh B7-B9 (small n in
  KonJND too).
- **Verdict**: cheap to run, queued, low-priority. Best-case: small
  B3 lift, ship-grade B0-B5 unaffected.

### B. **V0_20b — Distortion-manifold pre-training** (Su et al. 2023)

- **Paper claim**: data-efficient cross-corpus generalization at B0-B2;
  authors report +0.04 SROCC over BRISQUE on LIVE-Challenge low-q
  cluster.
- **Mechanism fit**: ✅ **Direct hit on B0-B5.** Pre-trains a
  content-invariant distortion encoder on the 218k synth corpus
  (unlabeled (ref, dist) pairs), then fits a small head on labeled
  MOS. The manifold pre-training is what cycle-7 dssim co-training
  *should* have been but wasn't.
- **Cost**: high — new training pipeline (contrastive loss, masked-
  label curriculum, gradual schedule), new feature extraction (the
  encoder is a learned thing, not a hand-crafted pool). ~1 week.
- **Probability of CID22 aggregate lift**: 60 %. The mechanism
  matches the weakness. Risk is whether contrastive pre-training
  on synth pairs transfers to CID22's authentically-distorted
  distribution (the FRIQUEE caveat — Ghadiyaram 2017 found
  synthetic-trained models break on authentic data).
- **Verdict**: highest conviction for the user's priority bands.
  Long lead time but worth funding.

### C. **V0_20c — LMS + opponent feature branch** (FRIQUEE 2017)

- **Paper claim**: cross-color-space feature diversity lifts SROCC
  ~0.10 on authentic distortions vs single-family features.
- **Mechanism fit**: medium. Adds parallel color-space pools (LMS,
  RG/BY opponent, HSI yellow) — independent signal from XYB SSIMULACRA2.
  Could help all bands but is most informative on artifact types
  XYB doesn't see well (chroma drift, hue rotation).
- **Cost**: medium — new feature extraction in zensim/streaming.rs
  (needs LMS color conversion + per-channel SSIM pools). ~3 days.
- **Probability of CID22 aggregate lift**: 40 %. The lift is real on
  AUTHENTIC distortions per FRIQUEE; CID22 is mostly synthetic
  codec distortions, so the lift here is less certain.
- **Verdict**: solid baseline-extension candidate. Could compound
  with B (distortion manifold uses ANY feature input).

### D. **V0_20d — JND-anchored output calibration** (Jenadeleh 2025 + Testolina 2023)

- **Paper claim**: not a SROCC lift mechanism — fixes calibration
  drift in the user-facing score range, especially at B8-B9 where
  ssim2 over-optimizes.
- **Mechanism fit**: zero direct B0-B5 lift. Improves dial honesty.
- **Cost**: low — re-fits α/β against AIC-3 JND anchors. ~few hours.
- **Probability of CID22 SROCC lift**: 5 % (SROCC is invariant to
  monotonic transforms). It's a UX improvement, not a SROCC one.
- **Verdict**: do regardless of other choices. Doesn't change ship
  decisions but improves user experience.

### E. **V0_22 — CVVDP distillation**

- **Paper claim** (Mohammadi 2025): CVVDP is SOTA on JPEG AIC-3
  with SROCC 0.960 (vs SSIMULACRA2's 0.905) and Z-RMSE 9.45 (vs
  SSIMULACRA2's 47.63, **5× tighter**). CVVDP is a perceptually
  grounded contrast/masking/foveation model.
- **Mechanism fit**: distillation transfers CVVDP's grounded
  perceptual knowledge into our fast XYB-features MLP. Could lift
  ALL bands by inheriting CVVDP's structure. Strongest mechanism
  for HF (B7-B9) which is CVVDP's strength.
- **Cost**: high — needs CVVDP installed + per-pair CVVDP scoring
  on a large unlabeled pool (then MLP distills against CVVDP
  outputs). ~2 weeks including infra + sweep.
- **Probability of CID22 aggregate lift**: 70 %. SOTA's structure
  is principled enough to transfer; the question is whether our
  MLP capacity (228 → 128 → 1 or 372 → 384 → 1) is enough to
  capture CVVDP's behavior. Larger MLP can be tried in parallel.
- **Verdict**: highest absolute upside. Long lead. Doesn't directly
  target B0-B5 but does target the overall accuracy ceiling.

### F. **Single-shot improvements that are obvious wins** (do regardless of A-E)

- **`--high-q-boost N` trainer flag** (mirroring `--low-q-boost`):
  upweights B3+ pairs in RankNet training so the MLP doesn't
  underfit the visually-lossless tail. ~1 hour. **Should always be
  in flight** for any new V0_20 run.
- **Per-band auxiliary loss `L_b3 = -SROCC(pred|band=B3)`**: explicit
  multi-objective training. ~3 hours. Same direction as `--high-q-
  boost` but mathematically tighter.
- **Z-RMSE eval metric in dataset_metric_baseline** (per CLAUDE.md
  rigor mandate): tracks the metric Mohammadi 2025 found discriminating
  at HF. Doesn't change training but changes what we measure.
  ~half day.

## Ranked action list (recommended order)

| Rank | Path | Reason | Effort |
|---|---|---|---|
| 1 | **F: high-q-boost trainer flag** | Cheap, immediately useful, compounds with everything | 1 hr |
| 2 | **A: iw_k1 TV concat** (in flight) | Cheap, definitive answer on V0_18 architecture absorbing IW | done in 30 min |
| 3 | **B: distortion-manifold pre-train** | Highest conviction for B0-B5 priority lift | 1 week |
| 4 | **D: JND-anchored calibration** | Cheap UX win + sets us up for next-gen eval | few hours |
| 5 | **E: CVVDP distillation** | Highest absolute SROCC upside (SOTA distillation) | 2 weeks |
| 6 | **F: per-band auxiliary loss** | Compounds with #1 and #3 | 3 hr |
| 7 | **C: LMS + opponent feature branch** | Solid baseline-extension if synth-corpus signal is exhausted | 3 days |

## Why this ordering

- **#1 first** because it costs ~1 hour and changes the training
  signal that EVERY subsequent run uses. Without high-q-boost, B3
  is structurally underfit by val_policy=min on KADID/TID/KonJND
  (none of which span B3 well). Today's V0_20a sweep was hurt by this.
- **#2 finishes today** anyway — gives a definitive answer on whether
  the V0_18 3-way-concat-with-TV architecture can absorb IW features.
- **#3 (manifold) is the highest-conviction long lead.** B0-B5 lift
  is the priority and Su 2023 is the most directly-applicable
  paper for the mechanism we're missing.
- **#4 + #6 are cheap regardless.** Schedule alongside the manifold
  pre-training.
- **#5 (CVVDP distillation)** is the highest absolute upside but
  needs the most infrastructure. Should run in parallel with #3
  if compute budget allows.

## What NOT to do

- **iw_k4 or iw_k8 single-MLP at full V0_18 recipe** — today's
  experiments show iw_k1 dominates iw_k4 on CID22 and iw_k8
  catastrophically overfits. The strength sweep at >k=1 is
  exhausted (single-MLP capacity is the limit).
- **Re-extract synth at iw_strength=8** — wasteful, the V0_20a
  sweep already showed k=8 overfit.
- **Train new bakes without high-q-boost** — every cycle from
  here forward should include this trainer flag.

## Falsification gates per path

To know when to stop pushing on each:

- **A (iw_k1 TV concat)**: if the 3-way concat CID22 SROCC < 0.880,
  stop pushing on single-MLP IW. iw_k1 standalone (s42) is at 0.8657;
  TV regularization + 3-way concat should add 0.01-0.02. Below 0.880
  means the IW signal isn't materially compatible with this
  architecture.
- **B (manifold)**: if cross-validation CID22 < 0.890 after 4 hyperparameter
  sweeps, the mechanism doesn't transfer for our distortion mix.
- **E (CVVDP distill)**: if MLP-on-features can't reach CVVDP within
  0.03 SROCC after 3 distillation attempts (varying teacher fit
  quality + student size), the SSIMULACRA2-feature input is the
  ceiling and we need to extend feature extraction (path C).

## Compute budget allocation (next session)

Assuming ~24 hours of compute available:

- 1 h: F#1 (high-q-boost) implementation
- 1 h: A retraining baselines with high-q-boost
- 3 h: D (JND calibration on AIC-3)
- 6 h: B (manifold) Phase 1 — contrastive pre-training infra
- ~13 h: B + C sweeps

This delivers (a) immediate ship of V0_20a TV-concat result if it lands,
(b) confirmation that high-q-boost compounds, and (c) the foundation
for the higher-conviction B0-B5 lift via distortion-manifold.
