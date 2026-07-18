# MLP diffmap coherence — measured (2026-07-18)

**User ask:** "measure the MLP's true diffmap coherence." Answered — and the measurement
**rewrites the additive-vs-MLP diffmap story**. Tool: `zensim/examples/diffmap_block_coherence.rs`
`--bake` mode (new; requires `--features custom-profiles`), which mounts any bake as a
`ZensimProfile::Custom`, scores through the production features→score runtime
(`score_features_with_profile`), and per 32-px block: copy reference pixels in → recompute
features → ΔS_bake. n=9 pairs per bake (gb82 {city, dog, girl} × ImageMagick JPEG q{20,50,75},
576×576, 324 blocks). Pairs + logs: `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`.

## The four numbers per (pair, bake)

- **M1** — SROCC(shipped default diffmap blocks, ΔS_bake). What a codec consuming today's
  `compute_with_diffmap(…, Trained)` (ssim-only signals, V0_2 profile weights) gets.
- **M1b** — same weight source, ALL per-pixel signals on (`include_edge_mse` + `include_hf`).
- **M3** — `DiffmapWeighting::ModelSensitivity(s_k)` (new, `custom-profiles`-gated): per-pixel
  signals weighted by the BAKE'S own gradient `s_k = ∂score/∂f_k` (numerical central differences
  through the full runtime at the base image), signed fold (v2), |mass| normalization.
- **M2** — SROCC(Σ_k s_k·Δf_k(block), ΔS_bake): the gradient applied to TRUE per-block feature
  deltas — the **linearization ceiling** for ANY gradient-based diffmap of that scalar.
- **SSE** — the codec PSNR default, as the bar.

## Headline result: M2 = 1.000 for EVERYONE — the old ceiling story was wrong

Two fold variants of the ModelSensitivity map were measured (medians of 9):

| bake | M1 ship | M1b all-sig | M3 abs-fold | M3 signed-fold | **best deployable** | M2 ceiling | SSE |
|---|--|--|--|--|--|--|--|
| winner MLP (`Ebothg`, 156→128→1) | +0.746 | +0.686 | +0.446 | **+0.759** | signed **0.759** | **+1.000** | +0.123 |
| B shipped (additive-372) | +0.243 | +0.344 | **+0.660** | +0.156 | abs **0.660** | **+1.000** | +0.423 |
| ADD156 (additive basic-156) | +0.244 | +0.405 | **+0.849** | +0.571 | abs **0.849** | **+1.000** | +0.816 |

(abs run: `matrix_m3_2026-07-18.log`; signed run: `matrix_m3v2_2026-07-18.log`.)

**The fold split is itself diagnostic.** The signed triple-fold (Σ s over {mean,4th,2nd}, sign
kept) helps the MLP — its gradient's pooling-triple members agree in sign, and beneficial-
artifact signals (positive s, e.g. winsorized `hf_gain`) correctly get negative map weight (v2
wins 8/9 winner cells; dog q20 0.430→0.759). But it HURTS the additive solves — a linear solver
freely sign-mixes WITHIN a pooling triple to shape its response (mean +w, 4th −w), so the signed
sum cancels real information; the abs fold preserves their emphasis. No API change needed to
pick per-driver: passing `−|s_k|` through the signed fold reproduces the abs fold exactly.

1. **The MLP's gradient is EXACT (M2 = 1.000).** A LeakyReLU MLP is piecewise linear; per-block
   refinements (1 of 324 blocks moving mean-pooled features) almost never flip a hidden unit's
   activation region, so the first-order Taylor is exact. The prior belief — "a non-additive
   scalar caps its diffmap at ~0.87–0.91" (`FINAL_DIAL_METRIC_DESIGN` §1) — conflated two things.
   The coherence loss is **entirely in spatialization** (mapping the gradient onto per-pixel
   signals), never in the scalar's functional form. **Additive-vs-MLP is NOT the diffmap axis.**
2. **Basic-input vs non-basic-input IS the axis.** B's structural ~0.66 M3 ceiling reproduces
   the old measurement exactly — its 38% weight mass on peak/max/IW features (f156–371) cannot
   be expressed by any per-pixel map (those poolings are non-additive across blocks). Both
   156-input models (winner MLP, ADD156) have fully-spatializable gradients.
3. **The shipped default diffmap serves the winner MLP best of all bakes (0.746)** — its dense
   basic-feature gradient happens to align with the V0_2-ish ssim weighting — and it serves
   B/ADD156 terribly (0.24!). Today's `Trained` map + B scalar (the shipping combination) is the
   single WORST-aligned pairing measured.
4. **SSE (the codec default) actively fights the winner's scalar** on textured content:
   −0.45 (dog q20), −0.19 (city q50). Any codec optimizing PSNR per block is anti-correlated
   with where the winner metric says quality lives. The RD value of a zensim diffmap is real.
5. **`ModelSensitivity` works as designed for additive-basic**: ADD156 0.244 → **0.849**
   (up to 0.93 on girl), deployable TODAY with fixed s_k (= its solve weights — no per-image
   gradient needed). This vindicates a scoped version of the "ModelCoherent" idea whose
   *fixed-weight-swap* form was falsified 2026-07-16 (that test only swapped V0_2↔Balanced on B).

## The residual gap to M2=1.0 (all bakes)

The remaining spatialization loss (best deployable 0.66–0.85 vs ceiling 1.0) is the
one-scalar-weight-per-signal fold itself: the {mean, 4th, 2nd} poolings of one per-pixel signal
have DIFFERENT per-pixel gradients (∂‖x‖₄ ∝ x³ concentrates on high-error pixels; the mean is
uniform), and no single fold represents both — city (most textured, widest error distribution)
is where this costs most, girl (smooth) least, matching the per-image pattern for every bake.
The principled fix (not yet built): weight per-pixel signals by the pooling's local gradient
(signal³ term for the 4th norm — the fusion already has the per-pixel values). Alternative:
train a steering-oriented mean-only variant so the fold is exact by construction.

## Deployable driver menu for the codec-RD experiment (as measured)

| driver | steer-map coherence w/ its scalar | scalar CID22 | notes |
|---|--|--|--|
| ADD156 + ModelSensitivity map (abs fold) | **0.849** | 0.863 | fixed s_k; best-aligned steer |
| winner MLP + ModelSensitivity map (signed fold) | 0.759 | **0.894** | best rank; per-image s_k (µs) |
| winner MLP + shipped Trained map | 0.746 | 0.894 | zero new plumbing |
| B + ModelSensitivity map (abs fold) | 0.660 | 0.876 | structural ceiling (38% non-basic mass) |
| B + shipped map (TODAY'S SHIP) | 0.243 | 0.876 | the incumbent pairing — worst measured |
| SSE (codec default) | 0.12–0.82 by bake | — | anti-correlated with winner on texture |

Open follow-ups: per-image s_k for the MLP in-loop (recompute per iteration — 2·156 tiny
forwards, µs-scale; regions shift as the image improves); norm-aware fold (above); measuring
M3 with butteraugli-style masking off remains the honest config (masking assumes non-negative
maps — keep it off with signed weights).

Runtime/API changes landed with this measurement (all `custom-profiles`-gated, no default-build
surface change): `DiffmapWeighting::ModelSensitivity(&'static [f64])` + signed
`model_sensitivity_weights` fold in `zensim/src/diffmap.rs`; `--bake` mode in
`zensim/examples/diffmap_block_coherence.rs` (Custom-profile mount, numerical gradient, M1/M1b/
M3/M2/SSE panel).
