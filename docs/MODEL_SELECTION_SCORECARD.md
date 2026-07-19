# Model-selection scorecard — the five-gate closed-loop exam (2026-07-18)

**Why this exists.** Until 2026-07-18, zensim model selection was a two-panel offline exam
(rank + dial). But zensim ships to make *codecs hit targets*, and that ability was never
measured — and could not be: `DiffmapResult::score()` returned the legacy V0_2 score for every
profile (fixed `834b4387`), so no encoder loop ever actually tracked a candidate. With the fix
+ the coherence instrumentation + the codec-in-the-loop probe, every candidate bake now takes
a five-gate exam. **A bake without all five rows is not a ship candidate.** SDR and HDR use
the same gates (HDR swaps the corpora/judges — see §HDR).

| gate | question | instrument | pass bar (SDR) | cost |
|---|---|---|---|---|
| **G-RANK** | ranks like humans? | `bake_verdict` (CID22 + LIVE/CSIQ/PIPAL + nonphoto + KonJND + AIC) | ≥ incumbent on CID22-band, no holdout collapse | ~1 min |
| **G-DIAL** | monotone calibrated dial? | dial panel (quarantined_v2 grid) | G1 p5≤25 ∧ p95≥85; G3 mono ≥0.93 | incl. |
| **G-STEER** | can its diffmap steer? | `diffmap_block_coherence --bake` (M2 ceiling + M3 deployable map; fold per family) | M2 ≥0.99; M3 ≥0.70 | ~3 min |
| **G-RD** | saves real bytes at equal *judged* quality? | probe matrix + independent judge panel (`rd_probe_2026-07-18.sh` + analyze) | ≥0% on ALL judges (no gaming regression), photos | ~30 min |
| **G-TARGET** | codec hits its dial fast? | probe residuals | med \|achieved−T\| ≤2 within ≤3 passes | incl. |

Operational notes (learned the hard way — see `benchmarks/rd_probe_results_2026-07-18.md`):

- **G-RD/G-TARGET require a dial spline in the bake bytes.** A raw-scale bake cannot take
  targets (the winner MLP was excluded from zenjpeg targeting until `bake_dial_refit
  add-spline` existed — the generic, MLP-capable spline injector; rank-invariance of the
  spline MUST be verified: SROCC identical pre/post on the full panel).
- **G-RD is judged by OTHER metrics** (ssim2 + butteraugli + a fixed zensim build) — a
  candidate cannot win by grading its own homework. A regression on any independent judge at
  equal claimed quality is the gaming signature (measured: B+Trained-map −0.7% butteraugli).
- **G-STEER's fold is per-family**: signed fold for MLP gradients, abs fold for sign-mixing
  additive solves (`−|s|` through the signed path ≡ abs). The M2 ceiling is 1.0 for both
  additive and piecewise-linear (LeakyReLU) models — the axis that matters is **basic-input
  spatializable mass**, not additivity (`benchmarks/mlp_diffmap_coherence_2026-07-18.md`).
- **Steer-mass pre-screen (free):** `closed_loop.diffmap_basic_fraction` in the metrics
  sidecar. A candidate with low basic-block mass is structurally capped as a steerer BEFORE
  any training investment (B = 0.62 → 0.66 M3 ceiling; **BHdr = 0.43** — worse).
- Both codecs' distance/starting-q tables are legacy-V0_2-seeded until re-seeded; G-TARGET
  numbers improve (fewer passes) after re-seeding, but equal-quality byte comparisons are
  unaffected (same tables for every candidate).

## Tuning with the scorecard (not just picking)

G-RD/G-TARGET are objectives, not only gates: recipe/blend iterations can be selected by
bytes-saved-at-equal-judged-quality + residual, with G-RANK as the guard — e.g. the
screen-content column (every zensim driver negative on screens in the 2026-07-18 probe) is a
*measurable* retrain target: add screen/nonphoto mass, re-probe, watch the column. Steering
quality is also trainable: a mean-pooled-only basic-feature variant makes the M3 fold exact
by construction.

## HDR variant of the exam

Same five gates with: G-RANK → UPIQ/HDR panels (`upiq_panel.py` guard per the BHdr ship
policy); G-DIAL → the BHdr dial grid; G-STEER → PU-linear pairs through the coherence tool
(extension pending); G-RD/G-TARGET → jxl HDR ladder (intensity_target path) judged by
`zenmetrics --hdr` (cvvdp/HDR judges). The steer-mass pre-screen applies immediately:
**screen every BHdr candidate on `diffmap_basic_fraction` before training** — the 2026-07-18
audit measured shipped BHdr at 0.43 (57% of its steer mass unspatializable).

**HDR steer-mass landscape (measured 2026-07-18, 63 bakes across the linear-probe HDR
families — family medians of `diffmap_basic_fraction`):**

| family | med steer mass | n | note |
|---|--|--|---|
| hdrbroadplh1 (shaped, lasso) | **0.963** | 1 | most steerable HDR bake measured |
| hdriwmix (iwssim-teacher mixes) | 0.762 | 7 | steerable family |
| canonhdr40 / canonhdr15 / canonkjhdr15 | 0.58–0.65 | 18 | canonhdr15-bvls = the KonJND-HDR record holder (0.6696) |
| **bhdr_linear_shaped_cvvdpmix (SHIPPED)** | **0.435** | 1 | 57% unspatializable |
| hdriw | 0.359 | 7 | |
| hdrmix (shaped/anchored lineage) | 0.161 | 16 | |
| hdr / bhdr_anchored2 / hdrcodc | 0.01–0.07 | 6 | effectively unsteerable |

Implication: the shipped BHdr's whole shaped/anchored lineage is a steering dead-end; if the
HDR closed loop matters, the next BHdr campaign should start from (or constrain toward) the
hdrbroadplh1/hdriwmix/canonhdr families and hold the UPIQ guard — do NOT train more hdrmix-
shaped variants and hope the map follows. Rank-vs-steer for these families must be settled by
the HDR G-RD leg, not another proxy.

## Provenance

Scorecard rows live in each bake's `.metrics.json` sidecar (+ probe TSVs under
`/mnt/v/output/zensim/rd-target-eval-*/`); the dashboards read sidecars. Instruments:
`zensim/examples/diffmap_block_coherence.rs` (--bake), `scripts/v_next/rd_probe_2026-07-18.sh`
(+ `rd_probe_analyze_2026-07-18.py`), `bake_dial_refit add-spline`, `emit_bake_metrics.py`.
