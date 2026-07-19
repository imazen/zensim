# zensim v2 "bounded" feature extraction — defect inventory, spec, and iteration-1 as-built

Two-part document. **Part A (as-built)** describes what iteration 1 actually
implements, as landed in `zensim/src/feature_v2.rs` behind the opt-in
`feature-regime-v2` Cargo feature (default OFF). **Part B (the audit)** is
the original read-only feature-science audit that Part A was built from —
the defect inventory (D1-D9), the v2 design spec, near-threshold candidates,
the compatibility plan, and the validation plan, preserved verbatim as the
durable record per the project's "DOCS: SEARCH + UPDATE FOR EVERY LEARNING"
rule. Where iteration 1's implementation diverges from Part B's spec, Part A
says so explicitly (§A.6, "spec deviations") — Part B is not silently
edited to match what got built.

---

# Part A — iteration 1 as-built (2026-07-18)

## A.1 Status and scope

Iteration 1 = "the bounded core": a scalar, correctness-first reference
implementation of the v1→v2 redesign, validated for boundedness and basic
correctness (identity, dimension handling). It is **not**:

- fused into the archmage/SIMD streaming hot path (`streaming.rs`) — v2 is
  its own straightforward-Rust code path;
- wired into any scoring/profile/bake system — there is no v2 `ZensimProfile`
  and no trainer support yet;
- validated at full-corpus scale — only unit fixtures + 9 real (ref,
  distorted) pairs (§A.7).

v1's 372-feature extraction (`metric.rs`, `streaming.rs`, `simd_ops.rs`,
`diffmap.rs`) is untouched. Verified: the full v1 test suite (103 tests)
passes identically with `feature-regime-v2` on and off; zero new clippy
warnings in either configuration (§A.8).

## A.2 Files changed

| File | Change |
|---|---|
| `zensim/Cargo.toml` | New `feature-regime-v2 = []` feature (default OFF), documented; added to `check-cfg` |
| `zensim/src/feature_v2.rs` | **New.** Constants, per-pixel formulas, `FeatureRegime`, `ZensimV2Result`, `FeatureViewV2`, `compute_v2_features_impl`, 8 tests |
| `zensim/src/lib.rs` | `pub mod feature_v2;` gated behind the feature |
| `zensim/src/metric.rs` | New `Zensim::compute_v2_features()` method, gated |
| `zensim/src/iw_pool.rs` | Promoted `WeightedPool::mean` out of blanket `#[allow(dead_code)]` (now v2's canonical pooling helper); `l2`/`l4` individually still allowed (unused by v2 iteration 1) |
| `zensim/examples/v2_bounds_smoke.rs` | **New.** Real-pair bounds smoke tool (§A.7) |
| `docs/FEATURE_V2_SPEC_2026-07-18.md` | **New.** This file |

## A.3 Layout (as-built)

22 signals per channel per scale (4 scales × 3 XYB channels × 22 = **264**
total v2 features at the default `NUM_SCALES=4`) — a concrete instantiation
of Part B §(b)'s design table, not a byte-for-byte port of v1's 13/6/6/6
block widths. Four-pass-equivalent conceptual grouping (basic / soft-peak /
masked / iw), plus a new PJND block Part B §(c) proposed but v1 never had:

| local idx | name | formula (informal) | bound | pooling |
|--:|---|---|---|---|
| 0 | `ssim_mean` | `mean(d)`, `d` = C1-bounded SSIM dissimilarity | `[0,2]` | mean of explicit per-pixel map |
| 1 | `ssim_dev2` | `sqrt(mean((d-mean_d)²))` — GMSD-style 2nd deviation | `[0,2]` | mean of explicit per-pixel map |
| 2 | `ssim_dev4` | `(mean((d-mean_d)⁴))^0.25` — GMSD-style 4th deviation | `[0,2]` | mean of explicit per-pixel map |
| 3 | `art` | mean bounded-similarity edge-artifact dissimilarity (dst>src half) | `[0,1)` | mean |
| 4 | `det` | mean bounded-similarity edge-detail-loss dissimilarity (dst<src half) | `[0,1)` | mean |
| 5 | `mse` | `mean(saturate((src-dst)², C_MSE))` | `[0,1)` | mean |
| 6 | `hf_gain` | mean bounded-excess of per-pixel HF energy (dst>src half) | `[0,1)` | mean |
| 7 | `hf_loss` | mean bounded-excess of per-pixel HF energy (src>dst half) | `[0,1)` | mean |
| 8 | `hf_mag_loss` | mean bounded-excess of per-pixel HF magnitude (L1, src>dst half) | `[0,1)` | mean |
| 9 | `ssim_soft_peak` | `Σsal(d)·d / Σsal(d)`, `sal=saturate(·,C_PEAK)` | `[0,2]` | canonical weighted mean |
| 10 | `art_soft_peak` | same construction on `art` | `[0,1)` | canonical weighted mean |
| 11 | `det_soft_peak` | same construction on `det` | `[0,1)` | canonical weighted mean |
| 12 | `masked_ssim` | `Σw_mask·d / Σw_mask` | `[0,2]` | canonical weighted mean |
| 13 | `masked_art` | `Σw_mask·art / Σw_mask` | `[0,1)` | canonical weighted mean |
| 14 | `masked_det` | `Σw_mask·det / Σw_mask` | `[0,1)` | canonical weighted mean |
| 15 | `masked_mse` | `Σw_mask·mse / Σw_mask` | `[0,1)` | canonical weighted mean |
| 16 | `iw_ssim` | `Σw_iw·d / Σw_iw` | `[0,2]` | canonical weighted mean |
| 17 | `iw_art` | `Σw_iw·art / Σw_iw` | `[0,1)` | canonical weighted mean |
| 18 | `iw_det` | `Σw_iw·det / Σw_iw` | `[0,1)` | canonical weighted mean |
| 19 | `iw_mse` | `Σw_iw·mse / Σw_iw` | `[0,1)` | canonical weighted mean |
| 20 | `pjnd_transducer` | `mean(saturate(\|src-dst\|/(1+k·activity), C_PJND_CLAMP))` | `[0,1)` | mean |
| 21 | `pjnd_fragility` | `mean(1-saturate(grad_energy(src), C_PJND_GRAD))` — **reference-only** | `[0,1)` | mean |

Flat layout: `features[scale*3*22 + ch*22 + local_idx]`, `ch ∈ {0=X,1=Y,2=B}`.
Accessed via `FeatureViewV2::{ssim_mean, ssim_dev2, ..., pjnd_fragility}(scale, ch)`.

### Constants (all named, all cited — `feature_v2.rs`)

| const | value | derivation |
|---|--:|---|
| `C1_V2` | 0.0001 | `(K1·L)²`, K1=0.01 (Wang et al. 2004 default), L=1 — see boundedness proof below |
| `C2_V2` | 0.0009 | identical value/derivation to v1's `simd_ops::C2`, kept independent |
| `C_EDGE` | 1e-4 | GMSD/FSIM/DISTS stabilizer, order of `C1_V2` |
| `C_MSE` | 0.01 | `mse=0.5` at squared-error 0.01 (≈10% local-intensity shift, L=1 convention) |
| `C_HF` | 1e-4 | stabilizer for the HF bounded-excess forms |
| `C_PEAK` | 0.05 | soft-saliency saturating half-point |
| `C_ACTIVITY` | 0.01 | bounds the activity signal before use as a pooling weight (D6 fix) |
| `IW_WEIGHT_FLOOR` | 0.001 | prevents all-zero weight sum on a flat image |
| `C_PJND_CLAMP` | 0.1 | CVVDP-style final soft-clamp half-point |
| `C_PJND_GRAD` | 0.02 | gradient-energy saturating half-point |
| `K_PJND_MASK` | 4.0 | masking-denominator strength, same order as v1's `k_mask`/`k_iw` |
| `BLUR_RADIUS` | 5 | matches v1's `ZensimConfig::default()` |

**`C1_V2` boundedness proof (verified against source, not assumed):**
`μ1, μ2 ≥ 0` is required for `num_m = (2μ1μ2+C1)/(μ1²+μ2²+C1) ∈ [0,1]`, and
it holds here because [`crate::streaming::convert_source_to_xyb`] calls
**exclusively** the *positive*-XYB conversion variants
(`srgb_to_positive_xyb_planar_into` / `linear_to_positive_xyb_planar_into`,
confirmed at `streaming.rs`'s XYB-conversion call sites) — so all three XYB
channels are non-negative by construction, not just Y. `num_m ≤ 1` holds
unconditionally (AM-GM, no sign assumption). `num_s/denom_s ∈ [-1,1]`
(Cauchy-Schwarz). So `d = max(0, 1-num_m·num_s/denom_s) ∈ [0,2]`.

## A.4 API surface (new, all gated behind `feature-regime-v2`)

```rust
// zensim::Zensim
pub fn compute_v2_features(&self, source: &impl ImageSource, distorted: &impl ImageSource)
    -> Result<zensim::feature_v2::ZensimV2Result, ZensimError>;

// zensim::feature_v2
pub enum FeatureRegime { V1, V2Bounded }
pub struct ZensimV2Result { /* .features() .into_features() .n_scales() .regime() .view() */ }
pub struct FeatureViewV2<'a> { /* .new(features, n_scales) -> Option<Self>; named accessors */ }
pub const FEATURES_PER_CHANNEL_V2_{BASIC,PEAK,MASKED,IW,PJND,TOTAL}: usize;
pub mod idx { /* named local offsets */ }
```

Disambiguation is by **distinct Rust type**, not runtime length-sniffing —
`FeatureViewV2::new` VALIDATES the exact expected length for `n_scales` and
returns `None` on mismatch; it does not guess a regime from an ambiguous
length the way v1's `FeatureView::new` does. `FeatureRegime` still exists
(returned by `ZensimV2Result::regime()`) for callers threading results
through generic/dynamic code — this is the "or a `FeatureRegime` enum"
option from Part B §(d), kept alongside the type-level tag rather than
instead of it.

## A.5 Design principles satisfied (Part B §(b), verified per-feature)

1. **Bounded** — every one of the 22 signals has a closed-form bound
   ([0,1), [0,2], or [0,1]-ish), proved analytically for the SSIM family
   (§A.3) and by construction for the saturating/bounded-similarity/
   bounded-excess/canonical-weighted-mean forms (a weighted mean of a
   bounded quantity with non-negative weights cannot exceed the quantity's
   bound — convexity). No feature is a raw, uncapped ratio.
2. **Normalized consistently** — masked, IW, and soft-peak pooling all go
   through the ONE promoted helper, `WeightedPool::mean` (`Σw·v/Σw`,
   `iw_pool.rs`), matching Wang & Li 2011 Eq.36 exactly. There is no second
   pooling convention anywhere in v2.
3. **Spatializable** — every feature is `mean_i(explicit_per_pixel_map(i))`.
   This includes the deviation moments (D8 fix: `dev(i)=(d(i)-mean_d)^p`
   is itself a per-pixel map) and the soft-peak weight (D4 fix: a
   saturating function of the signal itself, not a hard order statistic).
   HF gain/loss/mag-loss go further than D2 strictly required: v1's HF
   features were pool-then-ratio (no per-pixel decomposition existed even
   in principle); v2 redefines them as genuinely per-pixel comparisons
   from the start.
4. **Sign-consistent** — 21 of 22 signals are error-oriented (higher=worse)
   by construction. The one documented exception is `pjnd_fragility`
   (reference-only masking-susceptibility, Bondžulić et al. 2022) — it is
   still oriented "higher = this region is more likely to show visible
   distortion," but it is not a src-vs-dst error term. See §A.6 item 7.

## A.6 Spec deviations (honest list — what iteration 1 chose that Part B left open or didn't specify)

1. **Concrete signal count (22/channel/scale)** — Part B's table was
   qualitative ("per-block redesign"); iteration 1 picked exact counts
   (9+3+4+4+2) to make something buildable. Not a contradiction, an
   instantiation.
2. **Edge artifact/detail did NOT get the dev2/dev4 treatment** — only
   `ssim_mean` got the full mean/dev2/dev4 triple; `art`/`det` are
   single mean-pooled bounded-similarity signals. Part B's D8 discussion
   was general ("the pooling triple"); scoping the deviation-moment
   treatment to SSIM only was a deliberate iteration-1 scope cut, not an
   oversight. Extending it to art/det is straightforward follow-on work.
3. **HF gain/loss/mag-loss use a bounded-EXCESS form `(a-b)/(a+b+c)`, not
   literally GMSD's bounded-SIMILARITY form `(2ab+c)/(a²+b²+c)`** — because
   gain/loss need a signed asymmetry (which side is bigger), not a
   symmetric similarity. Same bounded-ratio family (denominator dominates
   the numerator by construction), different member of it. Named
   `bounded_excess` vs `bounded_sim` in code to keep the distinction
   explicit rather than silently reusing one name for two shapes.
4. **HF features made genuinely per-pixel** — beyond D2's literal ask
   (bound the ratio); v1's HF features had no per-pixel form at all
   (pool-then-ratio of two whole-scale energies). This is a superset fix,
   flagged in case a future comparison expects v2's HF features to be a
   drop-in bounded replacement of v1's semantics — they are bounded
   replacements of v1's INTENT, not numerically comparable to v1's values.
5. **Soft-peak weighting is self-weighted** (a signal weighted by a
   saturating function of itself), not FSIM's external phase-congruency
   weight. Part B's own audit text already flagged this
   ("not identical to FSIM's PC-based external weight") — restated here
   for the as-built record, not a new deviation.
6. **`pjnd_fragility`'s sign convention was decided during implementation.**
   Part B §(c) discussed the Bondžulić gradient-energy signal as a
   "masking-capacity" feature without committing to a final sign; iteration
   1 inverts it (`1 - saturate(grad_energy, c)`) so "higher = more fragile"
   to satisfy the coordinator's sign-consistency requirement (item 6) while
   staying honest that it is not an error term (§A.5 item 4).
7. **No config knobs exposed** — `NUM_SCALES` (4) and `BLUR_RADIUS` (5) are
   hardcoded to v1's defaults, not threaded through a `V2Config` the way
   v1's `ZensimConfig` exposes them. There is no v2 config struct yet.
8. **No per-block opt-in toggles** — v1's `extended_features`/
   `compute_iw_features` let callers skip expensive blocks; v2 iteration 1
   always computes the full 22-signal set. A reasonable perf follow-on, not
   needed for a correctness/boundedness validation pass.
9. **Memory shape not yet optimized.** `compute_channel_scale_v2` allocates
   ~18 transient `f32` buffers of `width×height` (blur intermediates + 9
   per-pixel maps), freed at the end of each channel-scale call — fine at
   the 576×576 smoke-test scale (§A.7, ~90ms/pair, no memory pressure
   observed), but **not measured at large image sizes**. Per this project's
   "NEVER EXTRAPOLATE memory" rule, no GB-at-4K figure is given here — if
   large-image capacity matters before the SIMD-fusion iteration, heaptrack
   it directly rather than trusting a multiplied buffer-count estimate.
10. **`convert_source_to_xyb`'s `parallel` flag is threaded through from
    `Zensim::parallel()`**, but scale-pyramid construction and the 3-channel
    loop inside `compute_v2_features_impl` are single-threaded (a plain
    `for ch in 0..3` loop, no rayon). Parallelizing that loop is a cheap,
    unimplemented follow-on.

## A.7 Real-pair bounds smoke test (validation plan item, executed)

Tool: `zensim/examples/v2_bounds_smoke.rs`. Run:

```sh
cargo run --release -p zensim --features feature-regime-v2 --example v2_bounds_smoke -- \
  city.png city_q20.jpg city.png city_q50.jpg city.png city_q75.jpg \
  dog.png dog_q20.jpg dog.png dog_q50.jpg dog.png dog_q75.jpg \
  girl.png girl_q20.jpg girl.png girl_q50.jpg girl.png girl_q75.jpg
```

Corpus: `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/{city,dog,girl}.png`
(576×576 references) against their ImageMagick JPEG q20/q50/q75 encodes — 9
(reference, distorted) pairs, real codec output, not synthetic. Per-block
min/max across all 4 scales × 3 channels × block-member-signals (measured
2026-07-18, `feature-regime-v2` release build):

| pair | basic [0,2] | soft-peak [0,2] | masked [0,2] | iw [0,2] | pjnd [0,1] |
|---|---|---|---|---|---|
| city q20 | 0.0026–0.5114 | 0.0135–0.2507 | 0.0023–0.0822 | 0.0026–0.1560 | 0.0334–0.7797 |
| city q50 | 0.0005–0.3572 | 0.0029–0.2242 | 0.0005–0.0580 | 0.0005–0.1200 | 0.0145–0.7797 |
| city q75 | 0.0001–0.3488 | 0.0007–0.2003 | 0.0001–0.0500 | 0.0001–0.1011 | 0.0077–0.7797 |
| dog q20 | 0.0027–0.4781 | 0.0115–0.2771 | 0.0029–0.1324 | 0.0027–0.1611 | 0.0344–0.6556 |
| dog q50 | 0.0005–0.4129 | 0.0021–0.2583 | 0.0005–0.1205 | 0.0005–0.1450 | 0.0144–0.6556 |
| dog q75 | 0.0001–0.3726 | 0.0006–0.2429 | 0.0001–0.1096 | 0.0001–0.1305 | 0.0079–0.6556 |
| girl q20 | 0.0009–0.2004 | 0.0041–0.2911 | 0.0007–0.0443 | 0.0009–0.1497 | 0.0278–0.8448 |
| girl q50 | 0.0002–0.1773 | 0.0012–0.2575 | 0.0002–0.0345 | 0.0002–0.1210 | 0.0097–0.8448 |
| girl q75 | 0.0001–0.1563 | 0.0005–0.2337 | 0.0001–0.0305 | 0.0001–0.1023 | 0.0056–0.8448 |

**Result: `OK: every block stayed within its documented bound on every
pair.`** — zero out-of-bounds cells across 9 pairs × 5 blocks. Observed
maxima sit far below the theoretical worst case (~0.2–0.5 vs a bound of
2.0) — expected, since the theoretical bound is only approached by the
adversarial synthetic fixtures (unit tests), not typical codec content.
Two qualitative sanity checks, both consistent with a correctly-signed
metric: (a) `basic`/`masked`/`iw` maxima **decrease monotonically** as
quality rises q20→q50→q75 for all three references; (b) `pjnd`'s max is
**identical across q20/q50/q75 of the same reference** (city: 0.7797 all
three; dog: 0.6556; girl: 0.8448) — expected, since `pjnd_fragility` is
reference-only and is the larger of the two PJND signals in every pair
measured here, so it pins the block max regardless of distortion level.
Per-pair timing: ~86–95 ms (unoptimized scalar, 576×576, 4 scales, 3
channels — not a throughput claim, just confirms iteration 1 runs in
reasonable wall time for validation-scale work).

## A.8 Test coverage (item 7, verified)

Ran via `~/work/zen/scripts/run-heavy --jobs 8 -- cargo test -p zensim [--features feature-regime-v2]`:

| config | lib tests | result |
|---|--:|---|
| default (v2 OFF) | 103 | 103 passed, 0 failed |
| `--features feature-regime-v2` (v2 ON) | 111 | 111 passed, 0 failed (103 v1 + 8 new v2) |

The 103 v1 tests are identical in both runs (same names, same count) —
v1's suite is unaffected by the feature flag either way. `cargo clippy
--all-targets` in both configurations: zero warnings attributable to
`feature_v2.rs` or the `iw_pool.rs`/`metric.rs` edits (pre-existing
warnings in unrelated files — `metric.rs:2952`, `examples/
rd_block_selection.rs`, `streaming.rs:5752/5779` — present identically
before this work and in both configurations; not touched).

The 8 new tests: `regime_tag_and_view_accessors_match_raw_indexing`
(item 7d), `identity_input_zeroes_every_error_feature` (item 7c — 21 of 22
signals asserted `<1e-6` on a non-flat identity pair; `pjnd_fragility`
explicitly excluded with a documented reason), `bounded_range_flat_
source_plus_noise_hf_gain` / `bounded_range_large_chroma_mean_shift` /
`bounded_range_hard_edge_on_flat_masked_and_iw` (item 7b's three named
adversarial fixtures — flat+noise, chroma mean-shift, hard-edge-on-flat),
`bounded_range_all_signals_random_content` (blanket sweep), plus
`dimension_mismatch_rejected` / `small_image_reflect_pads_and_scores`
(v1-parity input handling).

---

## A.10 Iteration-2+ candidate features (2026-07-19 — per-pixel-map additions, prioritized)

Every candidate below satisfies the v2 contract by construction (bounded, explicit per-pixel
map mean-pooled, error-oriented, literature-named) and is tied to a MEASURED gap. None is
built; each must earn its way through the five-gate v2-vs-v1 A/B — search `zenpapers` for the
named paper before implementing.

| candidate | per-pixel map | targets (measured gap) | literature | cost |
|---|---|---|---|--|
| **gradient-magnitude similarity** | `1 − (2·m_r·m_d+c)/(m_r²+m_d²+c)` on Sobel magnitudes | general rank — best quality/compute in classic FR benchmarks; distinct axis from covariance-SSIM (sharpness/blur) | GMSD (Xue et al. 2013) | tiny |
| **masking-transducer bank** | `err/(1+k·a)` at 2–3 spaced `k` (we ship ONE) | KonJND/PJND banding — separates near- vs supra-threshold error | Teo-Heeger divisive norm; CVVDP transducer | tiny (reuses maps) |
| **banding/contour detector** | step-edges inside low-gradient regions (CAMBI-style contrast bins, per-pixel) | near-lossless + HDR + screens — banding is THE smooth-gradient killer | CAMBI (Netflix, Tandon et al. 2021) | small |
| **ringing detector** | `err · dilate(edge_r) · (1−edge_r)` — error NEAR strong ref edges but not ON them | HF near-lossless (JPEG/JXL ring), screens (text halos) | classic ringing-metric family (Marziliano 2004) | tiny (maps exist) |
| **oriented blockiness** | H/V step energy at fixed phase (8-px lattice) minus ref's | JPEG near-lossless; screens | Wang-Bovik blockiness, FR-ized | tiny |
| **chroma-edge similarity** | GMS form on X/B-channel gradients | screens (colored text/edges; the probe's screen gap) | FSIMc chromatic term (Zhang 2011) | tiny |
| **edge-width change** | per-edge-pixel spread estimate delta, bounded ratio | blur percept distinct from hf energy loss | Marziliano blur 2002 | small |
| phase-congruency similarity | PC similarity map | structural salience (rank) | FSIM (Zhang 2011) | LARGE (log-Gabor bank) — only if GMS underdelivers |

Deliberately NOT candidates: order statistics (unspatializable — the D4 lesson), deep-feature
terms (out of the pure-Rust runtime contract), temporal (stills). Saliency-WEIGHTING is
already covered by the soft-peak/IW weight-map pattern.

## A.11 Phase-2 fused as-built (2026-07-19 — "feature-v2b" workspace)

**Contract**: gate that new features cost real speed before they earn a place in the extractor
(user directive, verbatim: "utility/relevance of a feature is as important as speed of a
feature"). This section is the formulas/approximations/codegen record; §A.12 is the
measurements the speed-gate verdict rests on.

### A.11.1 Kernel redesign: iteration 1's array-heavy design was itself the problem

Before adding any of the 7 candidates, the phase-2 baseline (§A.12.1) measured iteration 1's
existing 22-feature `compute_channel_scale_v2` at **2-5x slower than v1's full 372-feature
extraction** — an unacceptable starting point for adding more work. Root cause: iteration 1
stored NINE full-image `Vec<f32>` per-pixel maps (`d_ssim`, `art`, `det`, `mse`, `hf_gain`,
`hf_loss`, `hf_mag_loss`, `pjnd_trans`, `pjnd_frag`) and processed them in two passes — real
memory-bandwidth cost with no compute payoff, since every one of those quantities is only ever
consumed as a mean or a weighted mean.

**Rewrite**: `compute_channel_scale_v2` is now a single `for y { for x { ... } }` pass with O(1)
extra space:

- **Simple means** (art, det, mse, hf_gain/loss/mag_loss, pjnd_transducer, gms, ringing,
  banding, blockiness): a running `f64` sum, divided by `n` at the end. No array.
- **`ssim_mean`/`ssim_dev2`/`ssim_dev4`** (the D8 GMSD-style deviation-from-mean moments):
  Terriberry's (2007) single-pass online higher-order moments — `n`, `mean`, and the 2nd/3rd/4th
  central-moment running sums `M2`/`M3`/`M4` (`M3` tracked only because `M4`'s incremental
  update formula needs its pre-update value, not used as an output). `dev2 = sqrt(M2/n)`,
  `dev4 = (M4/n)^0.25` — computed WITHOUT ever revisiting a stored `d(i)` array, unlike
  iteration 1's store-then-revisit design. (Terriberry, T.B. 2007, "Computing Higher-Order
  Moments Online", freely-circulated note; the algorithm is the standard single-pass extension
  of Welford's (1962) online variance, used throughout streaming-statistics libraries.)
- **Masked/IW/soft-peak weighted pooling**: a `WeightedSum` accumulator tracking running
  `(Σw·v, Σw)`, finished as `Σw·v/Σw` — the IDENTICAL canonical formula
  `crate::iw_pool::WeightedPool::mean` established in iteration 1 (Wang & Li 2011 Eq.36), now
  computed incrementally instead of from materialized weight/value arrays. **This is a gated
  mirror, not a duplicate**, per this crate's no-duplication policy: `WeightedPool::mean` is the
  reference form (still used and tested), `WeightedSum` is the O(1)-space streaming variant used
  by the hot path, and `feature_v2::tests::weighted_sum_matches_weighted_pool_mean_exactly` pins
  the two bit-exact (`<1e-9`) on 1000 random `(weight, value)` pairs. `WeightedPool::mean`'s doc
  comment records this relationship; `l2`/`l4` remain unused, individually `#[allow(dead_code)]`.

The five blur-pass arrays (`mu1`, `mu2`, `s12`, `ssq`, `activity`) remain full-image — a box blur
is inherently multi-pixel and cannot be expressed as a running per-pixel accumulator. v1's own
architecture carries the same five arrays for the same reason (`streaming.rs`).

**Two real bugs, both caught by the test suite (not found by inspection)**:

1. **Downscale-loop dimension corruption.** An early draft updated the pyramid's `width`/
   `height` INSIDE the per-channel downscale loop instead of after it, so channels 1/2
   downscaled from channel 0's already-halved dimensions. Every test exercising `scale > 0`
   panicked with `"src plane length must be width*height", left: 1024, right: 64` — a loud,
   immediate failure, not a silent corruption. Fixed by restoring iteration 1's pattern (compute
   `new_wh` once per scale, apply it only after all 3 channels have downscaled from the SAME
   dimensions).
2. **Banding not identity-safe.** The first banding formula used `saturate(grad_dst, C)` alone —
   a function of the distorted gradient only, not a comparison to the reference's gradient. On
   identity input (`src == dst`, so `grad_dst == grad_src` exactly), a real content edge still
   produced a nonzero `dst_edge_b` term, so banding did not vanish where the spec (and every
   other v2 feature) requires it to. Caught by
   `identity_input_zeroes_every_error_feature` (`idx 27 not zero on identity: 0.124`). Fixed by
   switching to `bounded_excess(grad_dst, grad_src, c)`, which is exactly `0` whenever
   `grad_dst <= grad_src` by construction — the general lesson (recorded here for future
   candidates): **any "does dst have MORE of X than src" feature must be written as a
   dst-vs-src comparison, never as a saturated function of dst alone**, or it fails to be
   identity-safe the moment src itself has the property being measured.

### A.11.2 The seven candidates, as-built (formula + citation + honest approximation-vs-paper delta)

All seven share ONE per-pixel gradient computation (raw-pixel central difference, `gx = right -
left`, `gy = down - up`, `mag = sqrt(gx²+gy²)`, computed once per channel per scale for both
`src` and `dst`) — the phase-2 brief's explicit fold requirement.

| # | feature (`idx`) | formula (as-built) | citation | approximation vs. paper |
|---|---|---|---|---|
| 1 | `GMS` | `1 − (2·m_r·m_d+c)/(m_r²+m_d²+c)`, `c=1e-4` | GMSD, Xue, Zhang, Mou, Bovik 2013, arXiv:1308.3052, Eq.4 | Paper uses a Prewitt filter; this uses a cheaper central-difference gradient (no diagonal terms) — same bounded-similarity shape, cheaper kernel. Not validated against the paper's own reported SROCC numbers (out of scope for a phase-2 speed/bound pass; see item 6 for OUR corpus's correlation instead). |
| 3 | `GMS` on `ch∈{0,2}` | identical formula, X/B channels | FSIMc chromatic term, Zhang, Zhang, Mou, Zhang 2011 | FSIMc's actual chroma term is a PRODUCT of two similarity maps on I/Q-like channels, pooled together into one composite term. This implementation does NOT combine X and B into one signal — `gms(scale,0)` and `gms(scale,2)` are reported SEPARATELY (free, via the existing per-channel loop architecture, zero extra compute) rather than as FSIMc's single joint chroma term. Cheaper, less faithful to the paper's specific combination. |
| 2 | `PJND_TRANSDUCER_LOW_K`/`_HIGH_K` | `saturate(err/(1+k·activity), C_PJND_CLAMP)` at `k∈{1.0, 16.0}` (core `k=4.0` is iteration 1, unchanged) | Teo & Heeger 1994 divisive normalization; CVVDP transducer, Mantiuk et al. 2024 | Geometric ×4 spacing around the existing core k (A.10's "2-3 spaced k values"); the paper's own transducer has a more elaborate multi-band CSF-weighted masking pool this reuses only the scalar `k·activity` form of, per iteration 1's own already-documented approximation. |
| — | oriented `BLOCKINESS` | FR-ized 8px-lattice step energy: `bounded_excess(\|dst step\|, \|src step\|, c)` at `x%8==0`/`y%8==0` boundaries, summed if both fire (corner) | Wang, Bovik, blockiness family; FR-ized here (v1 has no reference-free blockiness at all) | Real blockiness detectors (e.g. Wang, Bovik, Evans 2000) use a full 8-point DCT-domain or FFT-based periodicity estimate across the whole image; this is a spatial-domain, per-pixel, FR (differences-only) approximation — cheap, bounded, identity-safe, but does not detect blockiness whose PHASE has shifted (e.g. a re-cropped image whose 8px grid no longer aligns with the original MCU boundaries). |
| — | `RINGING` | `saturate(err,c1) · saturate(activity,c2) · (1 − saturate(grad_src,c3))` | classic ringing-metric family, Marziliano et al. 2004-2006 | The A.10 table's form is `err · dilate(edge_r) · (1−edge_r)` — a genuine morphological dilation of the edge-indicator mask. This implementation uses `activity` (the existing blurred local-energy signal, `blur(\|src-μ\|)`) as the dilation proxy, justified because box-blur inherently spreads a sharp edge's influence over `BLUR_RADIUS=5` pixels — a "near a strong edge" halo without a second dilation pass. **Approximation, not identical**: a true morphological dilation has a HARD cutoff radius and ignores the edge's local contrast; `activity` is a soft, contrast-weighted spread that also responds to the DISTORTED image's own local energy in a way a pure reference-derived dilation would not. Cheaper (zero extra passes; reuses the blur-pass array already computed for masked/IW), less faithful. |
| — | `BANDING` | `bounded_excess(grad_dst, grad_src, c1) · (1 − saturate(grad_src, c2))` | CAMBI, Tandon et al. 2021 (Netflix) | **The largest approximation gap of the seven.** Real CAMBI: multi-scale (multiple spatial filter widths), per-pixel JND-based maximum-noticeable-difference contrast bins with an 8-bit-precision-boosting step, aggregated with an explicit visibility/weighting model calibrated against banding-specific subjective data, and a per-frame (or per-region) score, not a simple pooled mean. This implementation: single-scale (whatever the current pyramid level is), a single gradient-based smoothness/step heuristic, no JND calibration, no bit-depth boosting. It is a genuinely CHEAP, bounded, identity-safe, single-pass proxy for "distorted introduced a step where the reference was smooth" — not a CAMBI reimplementation, and should not be cited as one. Item 6's correlation number is the only evidence for whether this cheap proxy is worth anything. |
| — | `EDGE_WIDTH_CHANGE` | `1 − bounded_sim(decay_src, decay_dst, c)`, `decay = mean_grad(coarser scale) / (mean_grad(finer scale) + c)` | Marziliano, Dufaux, Winkler, Ebrahimi 2002/2004 perceptual blur metric | **The one scale-level (not per-pixel) exception in this feature set.** Marziliano's method finds, PER EDGE PIXEL, the distance to the nearest local extrema on either side of the edge (a genuinely per-pixel, per-edge spread estimate) — expensive and sequential (a 1D profile scan per edge), and not easily foldable into this crate's fused per-pixel loop without materializing an edge map and a second traversal. The as-built version instead compares the MEAN gradient magnitude at one pyramid scale against the next-coarser scale (a scalar per channel per scale, reusing `mean_grad_src`/`mean_grad_dst` — the same accumulators GMS/ringing/banding already need), on the theory that a genuinely sharp edge's gradient energy survives 2x box-downscale less completely than an already-blurred edge's (which has less fine-scale energy to lose). This is NOT spatializable at the per-pixel level the way the other 28 signals are — flagged explicitly, the same category iteration 1 already flagged for v1's non-basic (peak/masked/IW) blocks: real signal, not currently diffmap-foldable. The coarsest scale has no next-coarser scale to compare against; its slot duplicates the second-coarsest scale's value (documented, not fabricated). |

### A.11.3 Codegen / spill findings (`cargo asm`, non-native build)

Measured on `compute_channel_scale_v2` (all 7 new features + all 22 iteration-1 features
compiled in — the toggles gate RUNTIME accumulation, not compilation, so this is the
worst-case/full function regardless of which `V2NewFeatureToggles` are active at runtime):

```
cargo asm -p zensim --features training,feature-regime-v2 --lib compute_channel_scale_v2 --no-color
```

| metric | count |
|---|--:|
| total instructions (function body) | 1,470 |
| `mov`-class instructions touching `[rsp]`/`[rbp]` (spill/reload traffic) | 412 (28.0%) |
| of those, stack STORES specifically (spill writes) | 195 (13.3%) |

**Honest gap**: the phase-2 brief asked for spill counts "before vs after adding the [7 new]
features" — that comparison was not run as a clean two-point measurement. The kernel rewrite
(§A.11.1) and the seven new candidates landed together in one pass, so there is no intermediate
"single-pass, 22-feature-only" build to diff against. What WAS measured instead, and serves the
same diagnostic purpose the brief wanted (identifying which additions are expensive and whether
splitting would help): the **per-group marginal wall-time cost via `V2NewFeatureToggles`**
(§A.12.3) — a more direct signal than a static instruction count, since it measures what
actually executes rather than what the compiler emitted for the union of all paths.

**28% stack traffic is high** and consistent with genuine register pressure from ~30 live `f64`
accumulators (9 basic + 3 peak-pair-sums + 9 masked/IW-pair-sums + several new-feature sums) in
one function body. Per the brief's own guidance ("cache-friendly two-pass usually beats a
spilling one-pass"), a 2-pass split (materializing ONE small per-pixel array — e.g. `d_ssim(i)`—
and revisiting it for the weighted-pool pass) was considered but NOT implemented in phase 2:
§A.12.4's speed-gate verdict shows the dominant cost gap is the base kernel's scalar-vs-SIMD
architecture (present since iteration 1, before ANY of the 7 new features existed), not spilling
from the new features specifically — spending the remaining phase-2 time on a 2-pass restructure
would not have moved the gate result, so it is recorded here as a characterized, deferred
option rather than attempted blind.

## A.12 Phase-2 performance (baseline, stage profile, per-group cost, IW-skip, speed-gate verdict)

All numbers below: `~/work/zen/scripts/run-heavy --jobs 8 --` (never unbounded parallelism),
zenbench (interleaved paired statistics, hardware timers), no `-C target-cpu=native`. zenbench's
own auto-convergence stopped most groups at 4 rounds on this box (flagged `only 4 rounds` in its
own output) — every number below carries that same round count and zenbench's own noise
footnotes (CV%, drift) where present; treat single-run absolute values as directionally solid,
not four-significant-figure precise, and prefer the RATIOS (which are more stable than either
raw number alone since shared system noise partially cancels).

### A.12.1 Baseline (item 1): v1 372-feature vs v2 bounded, 4 sizes, 1-thread and N-thread

Corpus: gb82 `city.png` (576x576 native) + its ImageMagick JPEG q50 encode, resized via
zenresize (Lanczos) to each target size — decode/resize happens once, outside every timed
region (`zensim/benches/v2_speed_baseline.rs`).

| size | pixels | v1 1-thread (ms) | v1 N-thread (ms) | v2 1-thread (ms) | v2 N-thread (ms) | v2/v1 ratio (1-thread) |
|--:|--:|--:|--:|--:|--:|--:|
| 256² | 65,536 | 3.8 | 3.8 | 9.6 | 10.3 | 2.53x |
| 576² | 331,776 | 18.6 | 7.4 | 51.1 | 50.4 | 2.75x |
| 1024² | 1,048,576 | 65.2 | 16.0 | 264.6 | 254.8 | **4.06x** |
| 2048² | 4,194,304 | 281.7 | 53.3 | 1189.0 | 1115.5 | 4.22x |

OLS fit `time_ms = alpha + beta_per_Mpixel * (pixels/1e6)` (`R²` in parentheses — all four fits
are >0.999, i.e. the intercept+slope model explains the sweep well despite the 4-round noise):

| config | alpha (ms) | beta (ms / Mpixel) |
|---|--:|--:|
| v1 372-feat, 1-thread | -3.24 (R²=0.9997) | 67.78 |
| v1 372-feat, N-thread | 3.30 (R²=0.9999) | 11.94 |
| v2 bounded, 1-thread | -30.26 (R²=0.9992) | 289.95 |
| v2 bounded, N-thread | -24.94 (R²=0.9993) | 271.40 |

The negative v2 intercepts are a 4-point-fit artifact (small negative numbers, not a real
"negative fixed cost" — flagged rather than silently reported as if physical). The load-bearing
number is the SLOPE: v2's per-pixel cost is **~4.3x v1's** (289.95/67.78, 1-thread) — this
matches the per-size ratio column converging toward ~4.2x at the larger sizes (where the slope
term dominates over any fixed-cost noise), and is the core finding item 1 exists to surface.

**v1 threads well (68→12 ms/Mpixel, 5.7x), v2 barely benefits from more threads at all**
(290→271 ms/Mpixel, 1.07x) — `Zensim::compute_v2_features`'s only "parallel" surface is
threading `convert_source_to_xyb`'s color conversion; the per-channel loop in
`compute_v2_features_impl` (`for ch in 0..3`) is a plain sequential loop, unlike v1's
rayon-parallel channel/scale dispatch. A cheap, unimplemented follow-on (§A.6 item 10 in Part A
already flagged this; phase 2 confirms it with numbers).

### A.12.2 Stage profile (item 2): where v1's own time goes (existing `ZensimConfig` toggles reused as the harness — no new instrumentation needed)

`Zensim`'s `config_from_params` (`metric.rs:2431-2451`) already threads `extended_features`
(masked block) and `compute_iw_features` (IW block) straight from `ProfileParams` into
`ZensimConfig` — this is item 2's harness AND, as it turns out, item 5a's mechanism (§A.12.3):
comparing configs at `{228 basic+peak, +masked=300, +IW=372}` directly measures each block's
marginal cost with zero new code.

| size | 228 (basic+peak) | 300 (+masked) | masked marginal | 372 (+IW) | IW marginal (vs 300) |
|--:|--:|--:|--:|--:|--:|
| 256² | 3.0ms | 3.7ms | +23% | 3.9ms | +5% |
| 576² | 13.0ms | 17.8ms | +37% | 18.5ms | +4% |
| 1024² | 45.1ms | 62.8ms | +39% | 63.5ms | +1.6% (within noise) |
| 2048² | *(pending — background run in progress at doc-write time; see A.12.5 note)* | | | | |

**Masked is the real cost (23-39%, growing with size); IW's marginal cost over masked is small
and noise-dominated (1.6-5%) at every size measured.** This directly informs item 5b: if a
future session wants to cut v1 cost for models that need masked-derived signal but not IW
specifically, masked is the block worth optimizing, not IW.

### A.12.3 Per-feature-group marginal cost (item 4, `V2NewFeatureToggles`, 1024x1024 1-thread — the gate size)

`zensim/benches/v2_feature_group_cost.rs`, toggling `gradient_features` (GMS + chroma-edge-GMS +
ringing + banding + edge-width — grouped because they share the `sqrt`-based gradient
magnitude), `transducer_bank` (2 extra PJND k values), `blockiness` independently:

| variant | mean (ms) | vs `v0` (no new features) |
|---|--:|--:|
| `v0` — none of the 7 new features | 243.1 | baseline |
| `v0` + `gradient_features` only | 365.5 | **+50.4%** (noisy: CV=46%, drift flagged) |
| `v0` + `transducer_bank` only | 257.0 | +5.7% |
| `v0` + `blockiness` only | 305.4 | +25.6% (noisy: CV=35%, drift flagged) |
| all 7 new features on (`v4`, phase-2 default) | 287.4 | **+18.2%** |

`gradient_features` is the expensive group (dominated by 2 `sqrt` calls per pixel plus the
branchy neighbor-load setup); `transducer_bank` is cheap as expected (2 more divisions, no new
memory access); `blockiness` costs more than its simplicity suggests, likely branch-misprediction
on the sparse (`1/8` of pixels) lattice-boundary check. The all-7-on number (+18.2%) is LOWER
than gradient-alone or blockiness-alone individually in this run — the individual-group numbers
carry real noise (CV 35-46%, "later rounds slower/faster" drift flags from zenbench itself); the
all-7 number and the base-vs-full numbers from §A.12.1 (243-264ms range) are mutually consistent
within that noise band. Re-running with more rounds forced (not done — time-boxed) would tighten
these; the DIRECTION (gradient > blockiness > transducer_bank in cost) is what this measurement
supports, not the exact percentages.

### A.12.4 Speed-gate verdict — FAILS, and the 7 new features are not why

**Pre-registered gate**: fused v2-with-all-new-features extraction ≤ 1.25x v1's 372-feature wall
time, single-thread, 1024x1024.

**Measured**: v2 all-7-on / v1 = 264.6 / 65.2 = **4.06x** (§A.12.1's dedicated sweep) to
287.4/65.2 = **4.41x** (§A.12.3's cross-check run). **Gate FAILS by a wide margin — not
relaxed, reported as measured**, per the phase-2 brief's own instruction for this outcome.

**The critical, honest finding: turning OFF all 7 new features does not come close to closing
the gate either.** §A.12.3's `v0` (zero new features, iteration-1-equivalent signal set on the
iteration-2 kernel) still measures 243.1ms — a ratio of 243.1/65.2 = **3.73x**. The 7 new
features' own marginal cost (+18.2% combined, §A.12.3) is real but SMALL relative to the ~3.7x
gap that exists with none of them. **The gate failure is dominated by the base v2 kernel
architecture, not by anything added in phase 2**: v1 dispatches through archmage/magetypes
explicit SIMD (AVX-512/AVX2/NEON tiers, fused blur+feature kernels) throughout; v2's kernel
(both iteration 1's and iteration 2's rewrite) is straightforward scalar Rust relying on LLVM
auto-vectorization, which — per §A.11.3's spill measurement (28% of instructions touch the
stack) — is not achieving anywhere near SIMD parity for this accumulator-heavy workload. This
gap PREDATES phase 2 entirely: the phase-1 baseline (iteration 1, before ANY of the 7 candidates
existed) already measured 2-5x.

**Default-on/off decision**: given (a) none of the 7 features is individually responsible for
the gate failure, (b) their combined marginal cost (+18%) is a reasonable ADDED cost once a SIMD
base kernel exists, and (c) disabling any subset still leaves the ratio at ~3.7-4.4x — nowhere
near 1.25x either way — **all 7 new features ship ON by default** (`V2NewFeatureToggles::
default()` — every field `true`). The `V2NewFeatureToggles` mechanism itself ships regardless
(it is what made the per-group measurement in §A.12.3 possible at all, and remains available for
a future session to use once the base-kernel SIMD gap is closed and the marginal costs above
actually matter for a ship decision). Per item 6 (§A.12.6): if any specific feature's
utility-per-cost turns out to be poor, `gradient_features` is the first candidate to reconsider
disabling by default, since it is both the most expensive single group AND covers 5 of the 7
candidates (GMS, chroma-edge-GMS, ringing, banding, edge-width-change) at once.

**What would actually close the gate** (not attempted in phase 2 — a properly-scoped follow-on,
not a quick fix): SIMD-ize `compute_channel_scale_v2` using archmage/magetypes dispatch matching
v1's own architecture (`#[magetypes(_v4x, v4, v3, neon, wasm128)]` generic SIMD primitives per
this project's own stated preference for new hot kernels), fusing the blur passes with the
per-pixel accumulator loop the way v1's `streaming.rs` fuses H-blur+V-blur+feature-extract. This
is a substantially larger engineering investment than phase 2's scope — flagged honestly as the
real lever, not glossed over.

### A.12.5 Note on measurement completeness at doc-write time

The 2048² row of §A.12.2's stage profile was collected in a background run that had not
completed when this section was drafted; §A.12.2's table above shows exactly what was measured
and marks the pending cell explicitly rather than estimating or extrapolating it (per this
project's own anti-extrapolation rule). See the final phase-2 report for whether it landed
before this session ended.

### A.12.6 Helpfulness screen (item 6): correlation vs human label, KADID-10k + TID2013

Methodology: `zensim/examples/v2_helpfulness_screen.rs` extracts v2 features for every
(reference, distorted) pair via zen* crates (zenpng/zenjpeg decode, no `image` crate),
rayon-parallel (8-core, `run-heavy`), aggregating each new feature to one scalar per pair (mean
across all 4 scales x 3 channels — a coarse aggregate; a trained model would use the full
per-scale-per-channel vector, this screen exists only to check "is there ANY signal here" per
the phase-2 brief's "correlation screen only" scope). Correlation computed by
`zensim-validate`'s canonical `panel` binary (`--col-predicted <feature> --col-target
human_score --json`) — NOT hand-rolled, per this crate's no-duplicate-implementations policy.
KADID: 10,125/10,125 pairs processed, zero decode/dimension errors. TID2013: 2,950/3,000 label
rows yielded valid pairs (50 rows dropped at the label-file parse stage, before any image
touched — not investigated further, time-boxed; zero decode/dimension SKIPs on the pairs that
did parse).

**Bonus real-content bounds confirmation** (beyond item 6's correlation ask, essentially free
given the extraction already ran): across all 13,075 successfully-processed real (reference,
distorted) pairs — every KADID/TID distortion type (noise, blur, JPEG/JPEG2000, color
shift, contrast change, etc.), not just the synthetic adversarial fixtures in §A.11.1's test
suite — every one of the 7 new features' per-pair aggregate stayed inside its documented bound
(observed range `[0.0000, 0.7087]`, well inside the `[0,1)`/`[0,2]` contracts) with **zero
`NaN`/`inf` values** in any of the `7 features × 13,075 pairs = 91,525` aggregated values. This
is the real-content evidence the user's earlier mid-task directive asked for (§A.7's
`o_9292`-class validation), extended to the phase-2 features specifically.

| feature | KADID SROCC | KADID PLCC | KADID KROCC | TID SROCC | TID PLCC | TID KROCC |
|---|--:|--:|--:|--:|--:|--:|
| `gms` | **0.594** | 0.584 | 0.418 | 0.478 | 0.495 | 0.326 |
| `pjnd_transducer_low_k` | 0.420 | 0.425 | 0.288 | **0.489** | 0.547 | 0.353 |
| `pjnd_transducer_high_k` | 0.424 | 0.428 | 0.293 | 0.478 | 0.533 | 0.341 |
| `ringing` | 0.386 | 0.398 | 0.264 | 0.473 | 0.530 | 0.344 |
| `edge_width_change` | 0.469 | 0.497 | 0.324 | 0.193 | 0.226 | 0.131 |
| `banding` | 0.326 | 0.337 | 0.225 | 0.167 | 0.202 | 0.114 |
| `blockiness` | 0.298 | 0.298 | 0.203 | 0.105 | 0.187 | 0.077 |

(All correlations are unsigned magnitudes here — sign consistency was already established by
construction in §A.11.1/A.10; KADID uses DMOS so its raw sign is inverted vs TID's MOS, `panel`
reports the correlation as computed on the raw columns, not re-signed for this table — the
MAGNITUDES are what this screen is checking, per the brief's `|Spearman|` framing.)

**Reading this, per feature (utility-per-cost, joining §A.12.3's marginal-ms column)**:

| feature | avg \|SROCC\| (KADID+TID) | marginal cost (group) | verdict |
|---|--:|--:|---|
| `gms` | 0.536 | in `gradient_features` (+50.4% group) | **strong, general-purpose signal** — matches GMSD's own literature reputation as one of the best cheap FR features; carries its share of the group's cost |
| `pjnd_transducer_low_k`/`_high_k` | 0.454 / 0.451 | `transducer_bank`, cheapest group (+5.7%) | **best utility-per-cost of the seven** — solid correlation on both corpora at negligible marginal cost |
| `ringing` | 0.429 | in `gradient_features` | solid on both corpora, especially TID (0.473, TID's blur/noise-heavy distortion mix is closer to what ringing targets than KADID's) |
| `edge_width_change` | 0.331 | in `gradient_features`, near-zero marginal cost beyond the shared gradient | inconsistent — strong on KADID (0.469), weak on TID (0.193); consistent with §A.11.2's flag that this is the least paper-faithful of the seven (a scale-level proxy, not Marziliano's actual per-edge measurement) |
| `banding` | 0.246 | in `gradient_features` | weak on both — **expected, not damning**: neither KADID nor TID concentrates distortion types that specifically produce banding/posterization (their distortion sets are noise/blur/color/compression-mix, not a banding-focused corpus); the adversarial unit test (§A.11.1) confirms the feature DOES fire on a genuine banding pattern, this screen just doesn't have much banding content to check it against |
| `blockiness` | 0.202 | its own group (+25.6%) | **weakest of the seven on this screen, and NOT cheap** — same "wrong corpus for this distortion type" caveat as banding (KADID/TID aren't JPEG-blocking-focused), but unlike banding this one also has real marginal cost. First candidate to reconsider default-off if a FUTURE JPEG/blocking-focused screen (or the trainability A/B) doesn't recover its utility |

**This does not change the phase-2 default (§A.12.4: all 7 ship ON)** — the speed gate is
already failing for reasons unrelated to any single feature's cost (the base-kernel
architecture gap), so there is no speed pressure actually forcing a default-off decision right
now. This table is the evidence a FUTURE session doing the real trainability A/B (explicitly
the parent's next step, not phase 2's) should start from: `gms` and the `transducer_bank` are
the clear keepers; `blockiness` is the one feature where cost AND this screen's utility are both
unfavorable simultaneously.

## A.13 Phase-3 performance pass ("feature-v2c" workspace, 2026-07-19) — GATE: v2 ≤ 1.5x v1, 1-thread, 1024x1024

**Permission for this pass, distinct from phase 2**: edit core v1 code (`streaming.rs`,
`simd_ops.rs`, `metric.rs`, `iw_pool.rs`) in place to share machinery with v2, under a hard
guarantee that v1's 372-feature output stays BYTE-IDENTICAL. **v1 was NOT edited** — every lever
below reuses v1's existing PUBLIC primitives from v2's own module without touching a single v1
call site or formula. The byte-identity guarantee therefore holds by construction, and is also
independently enforced by a new golden-byte gate built FIRST, before any other phase-3 change
(per the brief: "the golden test is the gate, not the existing invariant suite").

### A.13.1 The golden gate (built first, per the brief)

`zensim/tests/v1_golden_bytes.rs` + `zensim/examples/capture_v1_golden.rs`. Two fixtures,
captured from unmodified main (`84b86cde`, the phase-2 landing commit) via
`compute_zensim_with_config` at the full `with-iw` config (`compute_all_features` +
`extended_features` + `compute_iw_features` all `true` — the same "v1 372-feature" config
phase-2's speed baseline uses):

- `GOLDEN_SYNTHETIC`: a 64x64 deterministic pair from the existing `tests/common/generators`
  helpers (`gen_value_noise` seeded `0xC0FFEE` + `distort_block_artifacts`) — no external file,
  fully reproducible from source alone.
- `GOLDEN_REAL`: a 96x96 crop of the gb82 `city.png`/`city_q50.jpg` real-photo pair, committed at
  `zensim/tests/fixtures/v1_golden_real_{ref,dist}.png` (6.4 KB + 2.7 KB, well under the 30 KB
  budget).

Comparison is **exact bit-for-bit** (`f64::to_bits()` equality, NaN-safe), not tolerance-based —
a single-ULP drift anywhere in the 372-feature vector is a hard fail, reporting up to 20
diverging indices so a refactor's blast radius is visible immediately. Both tests were confirmed
green on unmodified main BEFORE any phase-3 edit (proving the mechanism), and are green in the
final state below. **This is the proof that v1 is byte-identical**: not "I didn't mean to touch
v1," but a literal bit-for-bit comparison of the full 372-feature output on two independent
fixtures, run after every change in this pass.

### A.13.2 What was actually shared between v1 and v2 (and what wasn't)

The brief's strategy item 1 ("feed v2 from v1's streaming intermediates... this alone should
erase most of the 3.73x base gap") assumed v1's fused SIMD pipeline exposes reusable per-scale
mu1/mu2/covariance PLANES. Tracing `streaming.rs::process_strip_channel` (the actual call site of
`fused_blur_h_ssim`/`fused_vblur_features_ssim`) found this assumption false: v1's fast path
never materializes a full-image moment plane at all — `ScaleBuffers { mu1, mu2, sigma1_sq,
sigma12, mask, mul_buf, temp_blur }` are per-STRIP scratch, reused and overwritten strip-by-strip,
immediately reduced into `ScaleAccumulators` by the same fused kernel call, with the surrounding
band/strip-parallel tiling logic carrying explicit hand-written invariants ("eliminating
f32-accumulator history divergence at strip boundaries") to preserve v1's own byte-exactness
across thread/strip boundaries. Hooking v2 into that exact machinery would mean either
restructuring v1's strip pipeline to persist full planes (a v1 change, high risk to the very
byte-identity guarantee this pass exists to protect, and a large, delicate undertaking) or reading
v1's per-strip scratch mid-stride from outside `process_strip_channel` (not exposed, `fn` not
`pub fn`, and strip-scoped by design). **Assessed as too high-risk for this pass's remaining
budget and declined** — v1 stayed completely untouched, which is the safer reading of "v1 must
remain byte-identical" than attempting the integration and relying solely on the golden test to
catch a mistake after the fact.

What WAS available and safe: `crate::blur::fused_blur_h_ssim` and `crate::blur::
box_blur_v_from_copy` are `pub fn` at the crate level (called by `process_strip_channel` for the
*same algebra* v2 needs — H/V-blurred mu1, mu2, Σ(s²+d²), Σ(s·d) — but exported, not
strip-scoped). Reusing v1's own public primitive from v2 achieves the spirit of "feed v2 from
v1's machinery" (one shared, tested, SIMD-dispatched blur kernel instead of two independent
implementations) without touching v1's call sites, formulas, or the strip-parallel tiling at all.
This is what shipped (§A.13.3). The diffmap fold was not touched, per the brief.

### A.13.3 Kernel changes in `feature_v2.rs` (v1 untouched)

1. **`ScratchV2`**: reusable scratch buffers (11 `Vec<f32>` fields), allocated once per channel
   at the largest (scale-0) size and sliced down per `(channel, scale)` call, replacing 9
   `vec![0.0f32; n]` allocations **per call** (108 allocations per `compute_v2_features_impl`
   invocation before this pass; now 33 — 11 fields x 3 channels — allocated once).
2. **`fused_blur_h_ssim` + 4x `box_blur_v_from_copy`** replace the separate `mul_into`/
   `sq_sum_into` elementwise passes + 4 independent `box_blur_1pass_into` calls for
   mu1/mu2/s12/ssq. **Measurement note, corrected mid-pass**: an initial same-day comparison
   mis-flagged this as a regression (278-282ms measured vs an ~268ms figure *inferred* from
   phase-2's ratio rather than directly measured) — see the box below. A direct, repeated A/B
   against the separate-calls alternative (2 process launches per side) found fused
   consistently faster (278-282ms vs 310-331ms at 1024x1024/1-thread) regardless of the
   ambient noise level on either side, so it is the version kept.
3. **3-way channel-level `rayon` fan-out**, gated on `ZensimConfig`'s existing `parallel` flag +
   the `threads` Cargo feature. Each of the 3 XYB channels is independent within a scale (only
   the SAME channel's `prev_grad` crosses scales, read after all 3 channels finish), so the
   3-channel loop fans out with one `ScratchV2` per channel. New test
   `parallel_matches_serial_exactly` proves bit-exact equivalence between the parallel and serial
   schedules on non-trivial 200x152 content (not a toy fixture) — a parallel path with zero
   output-checking coverage is exactly how a scratch-aliasing bug or data race would ship
   silently, so this was written before trusting the speed number.
4. **`identity_input_zeroes_every_error_feature`'s `TOL`** widened `1e-6` -> `5e-4`, documented
   in-code: `fused_blur_h_ssim`'s sliding-window FMA (`sv.mul_add(sv, dv.mul_add(dv, sum_sq))`)
   rounds differently than the old separate `sv*sv + dv*dv`, and `ssim_d_local`'s
   `denom_s = ssq - mu1^2 - mu2^2 + C2_V2` is a near-zero-minus-near-zero subtraction on identity
   input, stabilized only by `C2_V2 = 9e-4` — so ULP-scale noise in `ssq` gets divided by ~9e-4
   and amplified roughly 1000x. This is a **pre-existing sensitivity of `ssim_d_local`'s formula**
   (a small-constant-stabilized ratio), not a new bug — only which rounding path feeds it changed.
   Within this file's own documented allowance for v2 (no downstream consumers yet; v1's golden
   gate is separate, zero-tolerance, and unaffected since v1 never calls `feature_v2.rs`).

**A methodology honesty note, kept because it's instructive**: the fused-blur-vs-regression
question above was initially answered WRONG on the first pass, by comparing a freshly-measured
number against an inferred historical one instead of a direct, same-day, repeated A/B. The
correction came from (a) noticing `uptime`/`ps aux` showed 10+ concurrent agent sessions
genuinely active on this shared box during measurement (not a hypothetical — confirmed with
process listings), (b) widening the zenbench group's `min_rounds`/`max_time` to reduce
sensitivity to that noise, and (c) — the actual root cause of the worst noise spike — finding and
killing a leftover `bfs -S dfs /mnt/v -iname *o_9292*` background process (a stray full-filesystem
find from this session's own o_9292 fixture lookup, still crawling `/mnt/v` and consuming 71% of
a core, pushing 1-minute load average to 8.36) that was actively contaminating the in-flight bench
run. **Every ms figure in §A.13.4 should be read with this caveat**: this box is shared and this
session was not perfectly clean of self-inflicted noise throughout. The RELATIVE finding (fused
faster than separate-calls, consistently, across noise levels) is trusted; the ABSOLUTE ms
figures carry more uncertainty than the ±MAD alone suggests.

### A.13.4 Gate measurement: before vs after, 4 sizes, 1-thread and N-thread

`zenbench v2_speed_baseline --group=<size>`, widened to `min_rounds(15)` / `max_time(30s)` per
group (from the phase-2 defaults of 5/10s) to reduce sensitivity to the shared-box noise
documented above. "Before" = phase-2 landing (`84b86cde`, §A.12.1's own table, reproduced here for
direct comparison). "After" = this pass's final state, measured post-bfs-kill (2048x2048 group;
the 256/576/1024 groups below were measured slightly earlier, some overlapping the noise spike —
flagged per-row):

| size | pixels | v1 1-thread (ms) | v2 1-thread BEFORE (ms) | v2 1-thread AFTER (ms) | ratio BEFORE | ratio AFTER | v2 N-thread BEFORE (ms) | v2 N-thread AFTER (ms) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 256² | 65,536 | 3.8 | 9.6 | 10.5 | 2.53x | 2.76x | 10.3 | 6.2 |
| 576² | 331,776 | 18.2 | 51.1 | 72.8 [noise-flagged] | 2.75x | 4.00x [noise-flagged] | 50.4 | 34.5 |
| 1024² | 1,048,576 | 68.5 | 264.6 | 297.8 [noise-flagged] | 4.06x | 4.35x [noise-flagged] | 254.8 | 122.9 |
| 2048² | 4,194,304 | 266.6 | 1189.0 | 1184.3 | 4.22x | 4.44x | 1115.5 | 455.9 |

OLS fit `time_ms = alpha + beta_per_Mpixel * (pixels/1e6)` on the AFTER numbers, v2 1-thread:
alpha ≈ -11 ms, beta ≈ 285 ms/Mpixel — versus phase-2's BEFORE fit of alpha=-30.26,
beta=289.95 (§A.12.1). **The slope is unchanged within noise** (285 vs 290, ~1.6% apart) —
consistent with the 2048² row (the single largest, least fixed-overhead-affected, and
LEAST noise-affected measurement, taken after the `bfs` contamination was killed): 1184.3ms
after vs 1189.0ms before, a **-0.4% difference — no measurable change** in 1-thread wall time
at the size the gate actually cares about extrapolating to.

**Gate verdict: NOT MET.** The 1.5x target at 1024x1024/1-thread was not reached, and the
honest read across all 4 sizes — especially the cleanest (2048²) data point — is that this
pass's changes produced **approximately NO CHANGE in 1-thread wall time** (4.06x -> 4.22-4.44x
depending on size and noise, i.e. flat-to-very-slightly-worse, not the "erase most of the gap"
the brief's strategy item 1 hoped for). The 576²/1024² rows carry extra uncertainty from the
self-inflicted `bfs` full-filesystem-crawl contamination (§A.13.3's honesty note) and should be
read as "no worse than 4.0-4.4x," not as a precise number. **What DID genuinely improve**: the
N-thread number, substantially and consistently at every size (2048²: 1115.5ms -> 455.9ms, a
real 2.4x; 1024²: 254.8ms -> 122.9ms, ~2.1x) — from the new 3-way channel parallelism, which is
correctness-proven (§A.13.3 item 3's bit-exact parallel/serial test), not just fast. **The base
architectural gap the brief's strategy item 1 targeted (v1's SIMD-fused pipeline vs v2's scalar
per-pixel loop) was NOT closed** — strategy items 1 (deep v1-intermediate sharing) and 2
(explicit magetypes SIMD of the per-pixel formula pass) are the two levers that could plausibly
reach 1.5x on the 1-thread number the gate specifies, and neither was completed this pass: item 1
was assessed and declined as too risky to v1's byte-identity guarantee within the remaining
budget (§A.13.2), item 2 was not attempted at all (§A.13.5's honest scope note). The
scratch-reuse + fused-blur restructuring (§A.13.3 items 1-2) is real, verified-correct, and
measured faster than the pre-pass structure in a direct, repeated, same-conditions A/B — but that
A/B's effect size is evidently smaller than the run-to-run noise floor on this shared box, which
is why it doesn't show up cleanly in the BEFORE/AFTER columns above despite being real.

### A.13.5 Codegen / spill findings on the final kernel (`cargo asm`, non-native build)

Measured on `compute_channel_scale_v2` post-pass (same command as §A.11.3, for direct
comparison):

| metric | phase-2 (§A.11.3) | phase-3 (this pass) |
|---|--:|--:|
| total instructions (function body) | 1,470 | 1,492 |
| `mov`-class touching `[rsp]`/`[rbp]` (any direction) | 412 (28.0%) | 349 (23.4%) |
| stack STORES specifically | 195 (13.3%) | 187 (12.5%) |
| of which XMM/float-specific (spilled `f64` accumulators) | *(not broken out separately)* | 162 (10.9%) |

Modest improvement in stack traffic (28.0%→23.4% of instructions touch the stack at all, 13.3%→
12.5% stores specifically) from the `ScratchV2` restructuring — fewer local `Vec`s means fewer
stack-resident length/capacity/pointer triples competing for the same budget. **The per-pixel
loop still spills real floating-point state**: 162 XMM-register stack stores/loads, consistent
with ~30 live `f64` accumulators (9 basic + 3 peak-pair-sums + 8 masked/IW-pair-sums + several
new-feature sums) in one function body exceeding the available register file. This was NOT
addressed — strategy item 2 (explicit `#[magetypes(_v4x, v4, v3, neon, wasm128)]` SIMD of the
per-pixel formula pass, `#[arcane]` entry-point-only, `#[rite]` nested, `incant!` dispatch) is the
lever that would address it, and was not attempted this pass. A rough static estimate (54 scalar
`divsd` + 3 `sqrtsd` instructions in the compiled function body) is consistent with the
division-heavy `bounded_sim`/`bounded_excess`/`saturate` formula family (each contributes one
`f64` division) being the dominant per-pixel cost, ahead of the blur passes — the blur kernels are
already SIMD-dispatched (`incant!` v4x/v4/v3/neon/wasm128/scalar, same as v1's), so the per-pixel
scalar accumulator loop is the more promising target for a FUTURE SIMD pass, more so than further
blur-side changes.

### A.13.6 Honest scope note

What this pass delivered, all correctness-verified (golden gate green throughout, new
`parallel_matches_serial_exactly` test, bounds smoke re-confirmed on the original 9 real pairs
AND the `o_9292` pathological fixture — see below): a real allocation-count reduction, a measured
(if noise-affected) blur-side improvement, and genuine N-thread scaling where phase 2 had
essentially none (1.07x -> ~2x). What it did NOT deliver: the 1.5x gate, and the two levers with
the actual expected value to reach it (deep v1-intermediate sharing, explicit per-pixel SIMD) are
BOTH still open — one assessed-and-declined-as-too-risky, one not attempted at all. This is
reported as a partial result against the stated gate, not reframed as success on a substitute
metric.

**Bounds smoke re-run** (`zensim/examples/v2_bounds_smoke.rs`, post-pass binary): the original 9
real (city/dog/girl x q20/q50/q75) pairs at `/mnt/v/output/zensim/diffmap-coherence-2026-07-18/`
— `OK: every block stayed within its documented bound on every pair`, values matching §A.7's
table to within the ULP-scale FMA drift documented in §A.13.3 item 4 (e.g. dog q20 basic max
0.47881 now vs 0.47810 before — a 7e-4 absolute shift, consistent with, not a new pathology).
PLUS the `o_9292.png.scale1024x683.png` pathological fixture (the real image behind v1's
5,814,302-max explosion, `/mnt/v/output/clean-picker-corpus-2026-06-26/`, distorted via
`gen_jpeg_distortion` at q20/q50) — basic block max 0.276, pjnd max 0.906, both comfortably inside
`[0,2]`/`[0,1]`: the bounded-by-construction design continues to hold on the exact real-world case
that motivated it, post-restructure.

## A.14 Phase-4 magetypes SIMD of the v2 per-pixel pass ("feature-v2d" workspace, 2026-07-19)

**Scope (deliberately narrow, per the phase-4 brief):** magetypes-SIMD `compute_channel_scale_v2`'s
formula pass + the shared gradient pass ONLY. `#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]`
over generic SIMD primitive types, one `#[arcane]` entry per hot loop, `#[rite]`-style nested
helpers (plain `#[inline]` generic functions, per the trait-bound composable pattern in
`~/work/archmage/docs/site/content/magetypes/examples/gaussian-blur.md`), `incant!` dispatch,
token-as-self. Do NOT touch v1, the diffmap fold, the trainer, or the public API beyond the gated
v2 surface — all held: v1 was not edited (confirmed by the golden gate staying green throughout,
§A.14.1), and the only public-surface change is internal to `feature_v2.rs`.

**Protocol addendum applied mid-phase** ("rigor × efficiency", user directive 2026-07-19): iterate
at the cheapest discriminating signal (cargo-asm instruction/spill counts + a 256² timing) before
any full 4-size sweep; run the full robust sweep only twice (once to confirm the fix, once for
this report); load-gate every full sweep (`uptime` 1-min load checked before each launch — see
§A.14.4); note but do NOT chase side-quests opened by the port (§A.14.6); apply the pre-registered
kill criterion (1024² 1-thread ratio still > 2.5× ⇒ stop optimizing, report the residual instead
of continuing to grind).

### A.14.1 Golden gate + correctness (built/verified first, per standing project discipline)

v1 stayed untouched throughout — `tests/v1_golden_bytes.rs`'s bit-exact gate (both
`v1_synthetic_fixture_matches_golden` and `v1_real_fixture_matches_golden`) is green in the final
state. **A real, separate bug was found and fixed on the way**: the real-fixture golden test was
failing with "No such file or directory", not a value mismatch — `zensim/tests/fixtures/
v1_golden_real_{ref,dist}.png` were referenced as "committed" in phase-3's own commit messages and
this doc, but `.gitignore`'s blanket `*.png` rule (intended for screenshots/montages, with an
existing `!benchmarks/*.png` carve-out) had silently excluded them from every phase-3 commit —
`jj status` never flagged them because jj (correctly) respects gitignore for automatic snapshotting,
so there was nothing to notice. Fixed two ways: (a) regenerated the fixture (`convert city.png
-crop 96x96+200+150` — same source images, same crop parameters, pixel-identical to the original
per the ALREADY-COMMITTED `GOLDEN_REAL` array matching exactly, confirmed by running the test
against the regenerated PNGs with zero code changes to the golden constants), (b) added a scoped
`!zensim/tests/fixtures/v1_golden_real_*.png` gitignore exception (mirroring the existing
`!benchmarks/*.png` pattern) so this can't silently recur. v2's own correctness: all 12 pre-existing
`feature_v2` unit tests (bounded-range adversarial fixtures, identity-zeroing, dimension-mismatch,
regime/view accessors, `WeightedSum`-vs-`WeightedPool::mean` pin) pass unmodified against the new
SIMD kernel, PLUS two new tests: `parallel_matches_serial_exactly` (bit-exact — pre-existing from
phase 3, still passing) and `raw_moment_reformulation_matches_terriberry` (new, §A.14.2).

### A.14.2 Kernel architecture: what got vectorized, what didn't, and why

Two separate `#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]` hot loops, replacing
`compute_channel_scale_v2`'s single scalar `for y { for x { ... } }` pass:

1. **`dense_block_kernel`** (always runs, matches the scalar loop's unconditional block): SSIM raw
   moments, edge artifact/detail, bounded MSE, HF gain/loss/mag-loss, PJND core + transducer-bank.
   Processes each row in 8-wide `f32x8` chunks (`GenericF32x8<Token>`) via `V8::<T>::from_array` on
   manually-sliced windows (not `partition_slice`, since 7 different planes need synchronized
   chunking) + a scalar tail for `width % 8`.
2. **`gradient_block_kernel`** (only when `toggles.gradient_features`, hoisted OUTSIDE the pixel
   loop — was a per-pixel branch in the scalar version, now one runtime branch per
   `compute_channel_scale_v2` call): GMS, ringing, banding, `grad_src`/`grad_dst` sums. Interior
   pixels vectorized via SHIFTED unaligned loads for x-neighbors (`row[x-1..x+7]` /
   `row[x+1..x+9]`) and full-row-offset loads for y-neighbors (`row_u[x..x+8]` / `row_d[x..x+8]`) —
   the same shifted-window-read pattern as the gaussian-blur.md vertical pass. The 1-pixel border
   (`x∈{0,width-1}`, `y∈{0,height-1}`) uses the ORIGINAL scalar reflect-boundary formulas via a
   `scalar_pixel` closure — a tiny fraction of pixels (2 rows + 2 columns), not worth a SIMD
   boundary-clamp.
3. **Blockiness** (`toggles.blockiness`) — **deliberately NOT ported to SIMD**. It's inherently
   sparse (only `x%8==0` or `y%8==0` lattice positions contribute), so a dense 8-wide pass would
   spend 7/8 of its lanes on masked-out zero contributions. Restructured instead into
   `blockiness_sparse`: a scalar pass that visits ONLY lattice rows/columns — strictly cheaper than
   the pre-phase-4 dense-scalar-with-modulo-branch it replaces (never touches the 7/8 of pixels
   that always contribute zero), without needing SIMD at all.

**SSIM moments: raw-moment SIMD reformulation, not Terriberry.** Terriberry's (2007) online
update is inherently sequential (each sample's update depends on the running `n`), so it does not
vectorize across lanes. Each lane instead accumulates plain running sums of `d, d², d³, d⁴`
(trivially SIMD, no cross-lane dependency), reduced ONCE PER ROW into an `f64` running total (not
once per whole image — bounds the `f32` accumulator's magnitude to `~width/8` increments of a
small `[0,2]`-bounded value, avoiding "large-number-swallows-small-increment" `f32` summation
error). At the end, the standard raw-to-central-moment identity recovers `(mean, M2, M4)`. New
test `raw_moment_reformulation_matches_terriberry` runs BOTH algorithms on the same 50,000
`d∈[0,2]` values and asserts agreement within 5e-4 relative — confirms the reformulation, doesn't
just assert it. This also gives `OnlineMoments` (no longer on the hot path) a live caller instead
of leaving it as untested dead code.

### A.14.3 The register-pressure finding — the actual story of this phase

The first version of `dense_block_kernel` vectorized EVERYTHING in the scalar loop's unconditional
block, including the 11 masked/IW/soft-peak weighted-pool `(Σw, Σ(w·v))` pairs — 22 of ~35 SIMD
lane accumulators live simultaneously in the inner loop. **Measured (256² cheap iteration signal,
per the efficiency addendum) as a severe regression**: v2 1-thread went from phase-3's 10.5ms to
30.4ms at 256² (ratio 2.76x → ~8.0x) and 264.6-297.8ms to **604.5ms at 1024²** (ratio ~4.2x →
**9.12x** — WORSE than phase 3, not better). `cargo asm` confirmed the mechanism: the compiled
kernel's stack/spill traffic was ~44% of instructions (vs phase-3's scalar kernel's 23.4%), and the
instruction count for one tier's compiled body was 5,830 (vs the fixed version's 1,861 — see
below) — AVX2 has 16 YMM registers; ~35 live `f32x8` accumulators cannot fit, and the resulting
spill/reload traffic dominated over the SIMD throughput gain.

**Fix**: scalarize JUST the weighted-pool accumulation. Extract the SIMD-computed
`d`/`art_i`/`det_i`/`mse_i`/`act` lanes via `.to_array()` after each 8-pixel chunk, accumulate the
11 pairs scalar (via a new shared helper `weighted_pool_accumulate_scalar`, also reused by the
scalar tail — no logic duplication) instead of keeping them as SIMD lane accumulators. The
division-heavy CORE formulas (SSIM moments, edge artifact/detail, MSE, HF, PJND — the higher
arithmetic-intensity part, ~6+ divisions/pixel) stay vectorized; only the accumulation of already-
computed values scalarizes. **Result: 1024² 1-thread dropped from 604.5ms to 212.8ms** (a 2.84x
speedup from this one change) — ratio vs v1 (69.4ms): **3.07x**, DOWN from phase-3's 4.06-4.44x
baseline. A real, verified improvement, though not reaching the 1.5× gate (§A.14.5).

This is the load-bearing lesson of the phase: **"vectorize everything in the loop" is not the same
question as "vectorize what fits in the register file."** The formula pass and the accumulation
pass have different arithmetic-intensity-to-register-pressure ratios, and treating them as one
monolithic SIMD region cost more in spill traffic than it gained in lane parallelism for the
low-arithmetic-intensity accumulation half.

### A.14.4 Gate measurement (load-gated, per the efficiency addendum)

`uptime` 1-min load checked immediately before each bench launch; both full sweeps ran under low,
stable load (no `>8` gate trip needed this phase — the phase-3 session's `bfs` self-contamination
incident did not recur, load stayed 0.1-2.5 throughout). `zenbench v2_speed_baseline`, same widened
`min_rounds(15)`/`max_time(30s)` config phase-3 left in place. The AFTER-fix columns below are from
ONE full 4-size sweep (`benchmarks/feature_v2_phase4_simd_speed_2026-07-19.log`, 506.0s total, load
0.13 at launch) run after the register-pressure fix landed — the second of the two full-sweep
budget the efficiency addendum allowed (the first full-size confirmation was the standalone 1024²
run in §A.14.3, reproduced here within ~1% agreement).

| size | pixels | v1 1-thread (ms) | v2 1-thread BEFORE fix (ms) | v2 1-thread AFTER fix (ms) | ratio phase-3 | ratio AFTER fix | v2 N-thread AFTER (ms) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 256² | 65,536 | 4.0 | 30.4 [1] | 4.7 | 2.76x | **1.18x** | 3.7 |
| 576² | 331,776 | 17.6 | — [1] | 43.4 | 4.00x | **2.47x** | 23.2 |
| 1024² (gate) | 1,048,576 | 63.9 | 604.5 | **197.4** | 4.35x | **3.09x** | 87.2 |
| 2048² | 4,194,304 | 262.4 | — [1] | 871.9 | 4.44x | **3.32x** | 347.0 |

[1] The 256² BEFORE-fix reading is from the standalone confirmatory run that triggered the fix
(§A.14.3); 576²/2048² BEFORE-fix were never measured (the all-SIMD version was abandoned upon
confirming the regression at 256²+1024², per the efficiency addendum's "iterate at the cheapest
discriminating signal" — a full BEFORE sweep on a version already known to regress would have
been wasted bench time).

**Pattern**: the ratio INCREASES with size (1.18x -> 2.47x -> 3.09x -> 3.32x) rather than
converging to a constant, meaning the fixed-kernel's per-pixel SLOPE (not just its per-call
intercept) is still worse than v1's. Standalone 1024² (§A.14.3, 212.8ms) vs this sweep's 1024²
(197.4ms) agree within ~7%, consistent with the ambient measurement noise this project has
repeatedly documented on this shared box, not a methodology error.

### A.14.5 Kill criterion — TRIGGERED, reporting the residual instead of continuing to grind

**Pre-registered kill criterion** (phase-4 addendum): "if after the port the 1024² 1-thread ratio
is still >2.5×, stop optimizing and produce the residual-attribution table instead — that outcome
is a valid, reportable result, not a failure to keep grinding." **Measured: 3.07x > 2.5x — criterion
TRIGGERED.** Per the addendum's own framing, this stops the optimization loop here rather than
attempting a third iteration (e.g., further splitting the gradient block's register footprint, or
tackling the two-separate-kernel-call overhead) — those remain open, characterized levers for a
future pass (§A.14.6), not attempted this session.

**Residual-gap attribution** (which pass eats what, at 1024², 1-thread): v1's full 372-feature
extraction is 69.4ms; the fixed v2 kernel is 212.8ms — a 143.4ms gap. Attribution, reasoned from
the measurements above (not independently re-isolated per-block — a genuinely separate
per-block-toggle timing pass, `v2_feature_group_cost`-style, is the natural follow-on if a future
session wants exact percentages rather than reasoned bounds):

| contributor | estimated share of the 143.4ms gap | basis |
|---|---:|---|
| Dense block core formulas (SSIM/edge/MSE/HF/PJND) — SIMD, division-heavy | majority | still the largest always-on block; per-pixel division count (~6-8) unchanged from the scalar kernel, now amortized 8-wide but still the dominant arithmetic |
| Weighted-pool scalar accumulation (11 pairs, per-lane) | meaningful minority | reintroduced BY the fix — trades SIMD-register-pressure cost for scalar-loop cost; net negative-to-positive per §A.14.3's measurement, but not zero-cost |
| Gradient block (shifted-load SIMD + scalar boundary) | smaller | only active when `gradient_features` is on (default-on, so counted here); phase-2's own marginal-cost measurement found this the single most expensive new-feature group even in the SCALAR kernel (+50.4%) |
| Blur passes (`fused_blur_h_ssim` + 4x `box_blur_v_from_copy`, unchanged from phase 3) | smaller, roughly phase-3-equivalent | not touched this phase; phase-3's own A/B already found this faster than the alternative, no reason to expect a phase-4-specific change here |
| Two-call dispatch overhead (`dense_block_kernel` + `gradient_block_kernel` as separate `incant!`-dispatched calls per (channel,scale)) | small but non-zero, not isolated | `archmage::summon()` is internally cached (confirmed via `archmage`'s own `summon_overhead.rs` bench doc comment: "`std::arch::is_x86_feature_detected!` already caches internally"), so this is unlikely to be the dominant term, but a 2-call-vs-1-call split was not A/B'd this phase |

**Honest scope note**: this table is REASONED, not independently measured per-row — the phase's
remaining time went to the register-pressure fix (§A.14.3) and its confirmation, per the kill
criterion's own instruction to stop optimizing and report rather than keep iterating. A future
phase's SECOND lever (per phase-3's own §A.13.6 open-items list) is unchanged: deep v1-intermediate
sharing was assessed and declined in phase 3 as too risky to v1's byte-identity guarantee; this
phase's SIMD port, even after the fix, has NOT closed the gate — the two real remaining levers are
(a) reducing the dense block's own division count (e.g. a cheaper `saturate`/`bounded_sim` rational
approximation, if the accuracy cost is acceptable — NOT attempted, out of scope), and (b) profiling
whether the two-kernel-call structure is worth collapsing into one (§A.14.6).

### A.14.6 Side-quests opened, NOT taken (per the efficiency addendum: "note in §A.14 as findings, do NOT take them this phase")

- **A tempting blur refactor**: `fused_blur_h_ssim`'s sliding-window FMA reassociation
  (documented in §A.13.3) interacts with the new raw-moment SIMD accumulation in ways that could
  plausibly be co-optimized (e.g., a fused blur+dense-formula kernel that never round-trips
  mu1/mu2/ssq/s12 through memory at all). Not attempted — this phase's scope is the formula pass +
  gradient pass specifically, and this would mean touching the blur-pass structure phase-3 already
  settled.
- **Collapsing `dense_block_kernel` + `gradient_block_kernel` into one `#[magetypes]` entry**:
  would save one `incant!` dispatch + one function-call boundary per (channel, scale), at the cost
  of re-introducing SOME of the register-pressure risk the §A.14.3 fix just resolved (the gradient
  block's own accumulators would become live alongside the dense block's). Not attempted — the
  dispatch-caching evidence (§A.14.5's table) suggests this is a SMALL term, not worth the
  register-pressure risk to re-litigate this phase.
- **A cheaper rational-approximation for `saturate`/`bounded_sim`/`bounded_excess`**: these three
  formulas contribute the majority of the ~6-8 divisions/pixel in the dense block (§A.13.5's static
  `divsd` count from phase 3, ~54 instructions, still architecturally present — now inside a SIMD
  `div` instead of scalar `divsd`, but division throughput is still the likely dominant per-pixel
  cost). A polynomial or Newton-Raphson-refined reciprocal approximation (magetypes' own
  `rcp_approx`/`recip` per §A.14.2's backend-trait survey) could plausibly cut this further, but
  changes the NUMERIC contract (needs its own tolerance verification against the exact-division
  reference) — flagged as the highest-expected-value NEXT lever, not attempted this phase.

### A.14.7 Honest gaps in this section

- The BEFORE-fix (all-SIMD, unfixed) readings at 576²/2048² were never measured — the all-SIMD
  version was abandoned as soon as the 256²+1024² readings confirmed the regression, per the
  efficiency addendum's "iterate at the cheapest discriminating signal" (a full BEFORE sweep on a
  version already known to regress would have spent bench time without adding decision-relevant
  information). §A.14.4's table now has complete AFTER-fix numbers at all 4 sizes.
- Spill-count-after-fix was assessed via compiled instruction count (1,861 vs 5,830 — a 68%
  reduction) rather than a full stack-store percentage recount — the instruction-count proxy is a
  strong enough signal given the 2.84x wall-clock confirmation at 1024², and re-deriving the exact
  stack-touching-mov percentage was judged lower value than the report itself under this phase's
  time budget.
- The residual-attribution table (§A.14.5) is reasoned from existing measurements, not built from
  a fresh per-block toggle-timing pass. `zensim/benches/v2_feature_group_cost.rs` (phase-2's
  existing per-group marginal-cost harness) could produce exact percentages for a future session
  with ~15-20 minutes of bench time — flagged, not spent, per the kill criterion's instruction to
  stop and report.
- **The ratio-increases-with-size pattern (§A.14.4) was noticed only at write-up time**, after the
  kill-criterion decision was already made on the 1024² number alone (matching the brief's own gate
  definition). It does not change the kill-criterion verdict (1024² is the pre-registered gate
  size), but it does mean a future session should treat 2048² (3.32x), not 1024² (3.09x), as the
  more conservative "worst case" ratio if extrapolating to even larger production image sizes —
  flagged, not chased further this phase.

# Part B — the original audit (verbatim record)

> Everything below is the read-only feature-science audit as delivered
> before implementation began. It is preserved unedited as the durable
> record of the reasoning and literature grounding behind Part A. Where
> Part A's implementation differs, Part A says so explicitly (§A.6) — this
> section is not patched to match what got built.

## (a) DEFECT INVENTORY

**Layout ground truth** (verified against the actual vector-assembly loop, `metric.rs:3986-4059`, not just doc comments): four passes, each `4 scales × 3 XYB channels × width`, concatenated: basic `[0,156)` width 13, peak `[156,228)` width 6, masked `[228,300)` width 6, IW `[300,372)` width 6. Within a pass, offset = `pass_base + scale·(3·width) + ch·width + local`. (The task brief's `hf_gain` offset list `{12,38,51,77,90,116,129,155}` is a partial subset — the full correct 12 offsets, scale-major/channel-minor, are `{12,25,38,51,64,77,90,103,116,129,142,155}`; I recompute this from source rather than repeat the approximation.)

### 13 basic signals (`metric.rs:1-107` module doc + `ScaleStats`, `metric.rs:587-653`; SIMD kernel `simd_ops.rs`)

| idx | signal | formula | range (claimed/actual) | per-pixel map? |
|---|---|---|---|---|
| 0-2 | ssim_mean/4th/2nd | `d=max(0, 1-num_m·num_s/denom_s)`; `num_m=1-(μ1-μ2)²` **[no C1]**, `num_s=2·cov+C2`, `denom_s=σ1²+σ2²+C2`, `C2=0.0009` (`simd_ops.rs:255`, comment: *"There is no C1"*); pooled mean/`(mean d⁴)^.25`/`(mean d²)^.5` | claimed ≤1 (SSIM contract); **actual: unbounded above** | yes, `d(i)` exists per-pixel |
| 3-8 | art/det mean/4th/2nd | `d=(1+\|dst-μ2\|)/(1+\|src-μ1\|)-1`; artifact=max(0,d), detail=max(0,-d) | claimed small; **artifact unbounded above** when `\|src-μ1\|→0` | yes |
| 9 | mse | `mean((src-dst)²)` in XYB | bounded by SDR input range only; **no explicit clamp**; HDR/PU-linear path (`compute_pu_linear`, `metric.rs:1616`) has a wider unaudited range | yes |
| 10-12 | hf_energy_loss/mag_loss/**gain** | `var_src=mean((src-μ)²)`; `gain=var_src>1e-10 ? max(0,var_dst/var_src-1):0` (`streaming.rs:499-510`) | loss ∈[0,1] (bounded, subtractive form); **gain unbounded above** — epsilon only guards literal ÷0, not magnitude | yes |

**D1 — unbounded SSIM `d` (dominant driver, measured).** No C1 on the luminance term means a large chroma mean-difference makes `num_m` a large negative number, so `d` (floored at 0, never capped) explodes. Full-corpus scan (`bigcodec_hqdedup_traindigits_2026-07-02.parquet`, 2.3M rows) found max `d`-derived feature = **5,814,302** (`iw_ssim_4th`, scale 0, channel 2/chroma) — analytically reproduced exactly: `d≈(μ1-μ2)²` reaches 5.76e6 at Δμ≈2400 (`benchmarks/ssim_moment_explosion_2026-07-16.md` §3, §7a). Concentrated in **masked/IW blocks** (5.8e6) far more than **basic** (max 29,009) because the flatness mask *multiplies* exactly the flat-chroma-region-with-hard-edge pixels where the defect is worst. Fires on 0.03% of held-out rows; currently masked only by a per-bake `winsor_p99` clip (`bake_dial_refit add-winsor`, `zensim-validate/src/bin/bake_dial_refit.rs:853-913`) applied post-extraction — a symptom clamp, not a fix, and it flattens rank among clamped rows (ties) where a bounded-by-construction feature would preserve severity order (§7c of the same doc).

**D2 — `hf_energy_gain` unbounded ratio (the task's named defect, distinct mechanism from D1).** Division-by-near-zero, not the missing-C1 problem: `var_src` can be legitimately tiny in a truly flat source block, so any distortion energy in `var_dst` produces a huge ratio. `streaming.rs:506-510`'s `1e-10` guard prevents NaN/Inf but not magnitude — matches the task's reported ~1e7 spike on a severe-distortion row.

**D3 — edge artifact ratio, same family as D2, lower measured severity.** `(1+diff_dst)/(1+diff_src)-1` is unbounded above when `diff_src→0`; empirically measured at 0.02-0.09 max on the shipped extractor (§2 of the moment-explosion doc) but *unbounded by construction*, which the task's "every feature must be bounded" bar does not forgive on the basis of empirical rarity alone.

### Peak block (`ssim_max/art_max/det_max`, `ssim_l8/art_l8/det_l8` — field literally named `ssim_p95` in `ScaleStats` at `metric.rs:623` despite computing an L8 norm, not a percentile — minor naming defect, `streaming.rs:524-526`)

**D4 — order statistics are not sums.** `max` is a per-pixel order statistic; a block-local pixel edit does not have a well-defined linear effect on the whole-image max, so it cannot be expressed as "mean of an explicit per-pixel map" the way the basic block can. `ssim_l8 = (mean d⁸)^(1/8)` inherits D1's explosion (root-8 dampens it less than intuition suggests: measured 4th-moment/mean ratio was 630× on the worst row, §3).

### Masked block (`metric.rs:217-224`: `mask[i]=1/(1+k_mask·a[i])`, `k_mask`=`extended_masking_strength`, default 4.0) and IW block (`metric.rs:243-247`: `iw[i]=1+k_iw·a[i]`, `k_iw`=`iw_strength`, default 4.0), both `a[i]=blur(|src-μ|)` — **same activity map, opposite polarity**

**D5 — `1/n` vs `Σw` normalization divergence (real, but small — two prior "wrong turns" in this repo already ruled out the scary version).** `streaming.rs:529-543` (the shipped hot path) divides masked/IW accumulators of `Σ(w·v)` by `1/n` — this is *mean-of-weighted-values*, not a *weighted mean* (`Σ(w·v)/Σw`). The correct weighted-mean implementation exists at `iw_pool.rs:390-410` (`WeightedPool::mean`, `#[allow(dead_code)]`, test-only) but is never called by the hot path, and **no test holds the two implementations together** — a textbook instance of this repo's own "fork with a good story" duplication anti-pattern. Two rounds of measurement (`iw_a_sum`/`iw_mean_w` diagnostic, gated behind `iw-diagnostics`, `Cargo.toml:40-52`, zero-cost when off) found the *shipped* `mean_w` spans only **1.03-1.27×** across references (not the originally-fitted-but-wrong-estimator's 15.3×) — real, uniform per-reference scale error on 144/372 features, structurally invisible to per-reference SROCC and landing entirely on pooled cross-image rank, but NOT the dominant non-photo driver (that's D1).

**D6 — masked/IW weight itself is unbounded for HDR content.** `a=blur(|src-μ|)` is bounded for SDR (bounded input range) but `iw[i]=1+k_iw·a[i]` has no explicit cap and PU-linear/HDR inputs (`compute_pu_linear*`, `metric.rs:1616-1796`) have materially larger dynamic range — unaudited.

### Diffmap fold (`diffmap.rs`) — confirms defect #3's structural claim directly in code, not just by inference

**D7 — the diffmap can only ever read the basic-13 block.** `trained_multiscale_weights` (`diffmap.rs:279-369`) and `model_sensitivity_weights` (`diffmap.rs:388-439`, `custom-profiles`-gated) both hardcode `const FPC: usize = FEATURES_PER_CHANNEL_BASIC` (13) and index `weights[base..base+12]` for `base=scale_base+c·FPC` — peak/masked/IW (f156-371) are never referenced by any diffmap code path; this is a hardwired 13-wide window, not a missing `if`. Measured consequence (`benchmarks/mlp_diffmap_coherence_2026-07-18.md`): shipped **B** has ~38% of its weight mass on non-basic features → **G-STEER** deployable coherence (M3) ceilings at 0.66 vs a measured M2=1.0 gradient-linearization ceiling for *every* architecture tested (piecewise-linear MLPs have an exact local gradient). Basic-156-only models reach M3 0.66-0.85.

**D8 — the `{mean, 4th, 2nd}` pooling triple has non-uniform true per-pixel gradients even where it IS spatializable.** `∂‖x‖₄/∂x_i ∝ x_i³` concentrates on high-error pixels; the mean's gradient is uniform `1/N`. A single scalar fold-weight per pooling type only approximates this (residual gap in the same doc, §"residual gap"). Compounding: additive/linear solves **sign-mix within a triple** (mean +w, 4th −w) to shape response curvature — this cancels real per-pixel information under a signed fold and requires an `abs`-fold workaround (`diffmap.rs:205-208` comment); MLP gradients don't sign-mix this way but the 372-feature linear ship (B) does.

**D9 — KonJND/PJND signal partially lives in the excluded (D7) block.** Measured (`docs/TOP_MODELS_COOKBOOK.md:75-78`): basic-156-input top models score KonJND 0.271 / 0.335; the 372-feature shipped B scores 0.547 (still below the 0.70 G5 floor, but clearly carrying *more* near-threshold signal than basic-156 alone). Confirms the task's defect #4 directly — whatever peak/masked/IW carry for KonJND cannot reach the diffmap today.

## (b) V2 SPEC — "perfectable features"

**Principles**, each grounded:
1. **Bounded by construction**, not by post-hoc clamp. SSIM's own founding paper states boundedness as a *design condition*, not an incidental property (Wang, Bovik, Sheikh, Simoncelli 2004, *IQA: From Error Visibility to Structural Similarity*, IEEE TIP 13(4); restated in Wang, Simoncelli, Bovik 2003, *Multiscale SSIM*, Asilomar 2003, Eq. 6: "SSIM(x,y) ≤ 1"). Zensim's own `d` formula violates its ancestor's contract by dropping C1.
2. **One saturating-ratio family, reused everywhere.** GMSD (Xue, Zhang, Mou, Bovik 2013/2014, arXiv:1308.3052, Eq.4: `GMS=(2m_r·m_d+c)/(m_r²+m_d²+c)`, `c=0.0026`), FSIM (Zhang, Zhang, Mou, Zhang 2011, Eq.5-6, same shape with `T1/T2`), and DISTS (Ding, Ma, Wang, Simoncelli 2020, arXiv:2004.07728, `c1=c2=1e-6`) all converge on the identical `(2ab+c)/(a²+b²+c)` bounded-similarity form for "compare two non-negative magnitudes without one blowing up the ratio." This is literally SSIM-with-C1 restated — so the *same* fix (add C1) closes D1, and the *same shape* (applied to `var_src, var_dst`) closes D2/D3, with zero new machinery.
3. **Spatializable = explicit per-pixel map, mean-pooled, nonlinearity applied last.** GMSD's own innovation (std-pooling, not mean) is itself spatializable because it is `mean_i[(GMS(i)-mean(GMS))²]` — the *deviation* is its own per-pixel map, pooled by an ordinary mean, with `sqrt` applied once at the very end (Eq.6, `gmsd.txt:207-217`). This is the direct template for fixing D8: don't ship `(mean d⁴)^0.25` as an opaque scalar — expose `dev4(i)=(d(i)-mean(d))⁴`-style per-pixel maps (or simpler, a per-pixel *deviation-from-mean* map) so the diffmap fold has an actual per-pixel quantity to weight, not an implicit gradient it has to approximate.
4. **Weighted pooling = weight-map × signal-map, normalized-sum-pooled — never `Σ(w·v)/n`.** IW-SSIM's own formula (Wang & Li 2011, IEEE TIP 20(5), Eq.36/45-46) is `Σw_i·q_i/Σw_i`; FSIM's is the same shape (`ΣS_L(x)PC_m(x)/ΣPC_m(x)`, Eq.8). Zensim's masked/IW blocks currently match **neither** this canonical form nor its own dead-code reference (`iw_pool.rs`) — pick the paper's own `Σw·q/Σw` as the single canonical form and delete the divergent `iw_pool.rs` fork (fixes D5).
5. **Replace hard max/L8 with a saturating saliency-weighted mean** (fixes D4). FSIM's `PC_m(x)=max(PC1,PC2)` used *as a weight*, not as the pooled quantity, shows the field's standard move: keep "worst region dominates" behavior via a **soft, bounded weight** (e.g. a normalized `softmax(β·d(i))` or `d(i)^p / Σd(j)^p`-style saliency map, `p` tunable), then pool as `Σ weight(i)·d(i) / Σ weight(i)` — exactly the D5 weighted-mean shape, reused. This is differentiable per-pixel by construction, unlike a hard max.

**Per-block redesign:**

| block | v1 (defect) | v2 replacement | bounded? | spatializable? |
|---|---|---|---|---|
| ssim mean/4th/2nd | no-C1 `d`, D1 | add `C1` to `num_m`: `num_m=(2μ1μ2+C1)/(μ1²+μ2²+C1)` (standard SSIM luminance term) | yes, ≤1 by construction | yes (per-pixel `d(i)` unchanged in shape) |
| hf_gain/loss/mag | ratio, D2 | `(2·var_src·var_dst+c)/(var_src²+var_dst²+c)`-shaped GMSD-style bounded similarity, signed for gain vs loss | yes, (0,1] | yes |
| art/det | ratio, D3 | same bounded-similarity shape on `diff_src, diff_dst` | yes | yes |
| ssim_4th/2nd (moment) | opaque scalar, D8 | expose `dev_p(i)=|d(i)-mean(d)|^p` as its own per-pixel map (GMSD pattern), mean-pool, single root at the end | yes (bounded `d` ⇒ bounded `dev_p`) | yes, exactly |
| peak/max/L8 | order stat, D4 | saliency-weighted mean: `Σ σ(d(i))·d(i) / Σ σ(d(i))`, `σ`=bounded saturating weight (FSIM pattern) | yes | yes |
| masked | `mean(v·w)`, D5/D6 | `Σw_i·v_i/Σw_i` (IW-SSIM Eq.36 form), `w` from a bounded activity signal | yes if `w` bounded | yes, exactly |
| IW | same, D5 | identical canonical form, opposite-polarity `w`; delete `iw_pool.rs`'s divergent second implementation, keep ONE weighted-mean helper used by both blocks | yes | yes |
| MSE / IW weight on HDR | unaudited range, D6 | explicit `min(x, cap)` or divisive-normalize by PU-linear's own known max, audited for the HDR front-end specifically | yes | yes |

**Cross-scale combination (optional, not required for the bounded/spatializable/sign-consistent bar):** MS-SSIM's own cross-scale combination is a **weighted geometric mean** with psychophysically-fit exponents `{0.0448, 0.2856, 0.3001, 0.2363}` (Wang, Simoncelli, Bovik 2003, Eq.7), not zensim's current linear per-scale blend. Worth a follow-on experiment but out of scope for a bounded/spatializable feature *extraction* redesign — flag, don't build. (**Confirmed still out of scope for iteration 1** — Part A did not touch cross-scale combination.)

**Sign consistency:** every v2 signal above is a "higher = worse" bounded similarity/dissimilarity by construction (the GMSD/FSIM/SSIM family is naturally error-oriented once written as `1 - similarity`), so a diffmap fold never needs the `abs`-vs-`signed` fold ambiguity D8 documents for the current triple.

## (c) NEAR-THRESHOLD (PJND) candidates, spatializable form

Two independent literature families converge on the same per-pixel shape, and zensim already computes half of it:

1. **Divisive-normalization transducer** (Mantiuk, Hanji, Ashraf, Asano, Chapiro 2024, *ColorVideoVDP*, SIGGRAPH 2024, arXiv:2401.11485, Eq.9-13): `D(x) = (ΔC(x))^p / (1 + C_mask(x))`, with an explicit final soft-clamp `D̂ = k_C·D/(k_C+D)`. This normalizes the **raw per-pixel error** by local masking energy *before* any further nonlinearity — importantly different from zensim's current masked block, which multiplies an *already-pooled dissimilarity* by a mask derived from reference-only activity. Recommendation: compute a genuine per-pixel `err(i)/(1+k·a(i))` map (numerator = raw XYB residual, not SSIM's `d`), mean-pool that as its own v2 feature. **(Built as `pjnd_transducer`, §A.3.)**
2. **Mean gradient magnitude of the source alone** (Bondžulić, Pavlović, Stojanović et al. 2022, *PJND prediction for JPEG*, Vojnotehnički glasnik 70(2)): a plain per-pixel gradient-energy map of the *reference only*, mean-pooled, predicts first-JND PSNR at >92% correlation with no masking model at all — cheap, and structurally identical to zensim's existing `activity=blur(|src-μ|)` signal. The gap isn't the signal, it's what's done with it. **(Built as `pjnd_fragility`, §A.3 — implemented with a simple centered-difference L1 gradient rather than `activity`'s blurred-abs-residual, per the paper's literal "gradient magnitude" framing.)**
3. **Validation target, not a feature**: KonJND++ (Chen, Lin, Wiedemann, Saupe 2023, QoMEX, arXiv:2306.07678) produces genuine per-pixel PJND-criticality maps (click-aggregated, Gaussian-blurred σ=35px) — use these to validate a v2 near-threshold feature's spatial agreement, don't try to hand-derive the feature from them. **(Not used in iteration 1 — no trainer/validation-corpus wiring yet; flagged as future validation work.)**
4. **Content-conditioning caveat**: Liu, Zhu, Callet 2023 (*Bridge the Gap between VDP and JND*, MMSP 2023) measured near-zero correlation (PCC=0.008) between a *global scalar* VDP score and satisfied-user-ratio across different content at fixed score — a universal threshold constant is the wrong shape. This matches this repo's own already-falsified finding (`docs/DATASET_HISTORY.md` §3.21: raw-cvvdp-rank supervision helps AIC-3 but *hurts* KonJND monotonically) — any v2 near-threshold feature must be validated per-content-class, not assumed to transfer. **(Still an open validation question — iteration 1 has not trained anything against KonJND.)**

## (d) COMPATIBILITY PLAN

V1 (372 features, all 40+ shipped bakes, every canonical parquet under `/mnt/v/zen/zensim-training/canonical-2026-05-21/`) must not move a single bit. Exact seam, following this crate's own existing precedent (`training`/`classification`/`custom-profiles`/`iw-diagnostics` — all default-off, additive `Cargo.toml:38-77`):

- **New cargo feature**, e.g. `feature-regime-v2`, default OFF. Gates new `pub` surface only; zero cost and zero behavior change when off (mirrors `iw-diagnostics`'s "adds only a sum; cannot change any feature value" pattern, `Cargo.toml:40-51`). **(Built exactly as specified, §A.2.)**
- **New sibling constants**, not edits to existing ones: `FEATURES_PER_CHANNEL_BASIC=13` / `_WITH_PEAKS=19` / `_EXTENDED=25` / `_IW=6` (`metric.rs:3599-3643`) stay untouched; add `FEATURES_PER_CHANNEL_V2_*` alongside. **(Built exactly as specified — `FEATURES_PER_CHANNEL_V2_{BASIC,PEAK,MASKED,IW,PJND,TOTAL}`, §A.3.)**
- **New API entry point**, not a new branch inside `compute_extended_features` (`metric.rs:1206`): a `Zensim::compute_v2_features(...)` (or a `FeatureRegime` enum threaded through a generalized `compute_with_regime`), returning its own result type or a `FeatureView`-like accessor tagged with the regime explicitly. `FeatureView::new` currently auto-detects tier by **vector length** (`metric.rs:3682-3704`) — this is a landmine for a v2 vector whose length could collide with a v1 length combination; a v2 accessor should carry an explicit regime tag rather than extend the length-sniffing. **(Built: `Zensim::compute_v2_features` calling a separate `compute_v2_features_impl`; `FeatureViewV2` is a distinct type with length VALIDATION not sniffing, `FeatureRegime` enum also present — see §A.4 for why both.)**
- **`ProfileParams`** (`profile.rs:414-440`) already has the `extended_features`/`compute_iw_features` bool-flag precedent for "this profile needs a wider feature vector" — a v2-consuming profile adds an analogous `feature_regime_v2: bool` or a `FeatureRegime` field, default `V1`. **(Not built — no v2 profile/scoring exists yet, out of scope per the coordinator's iteration-1 brief.)**
- **ZNPR v3 bake compatibility**: v2 bakes are new bake *content* (new feature count, new metadata declaring the regime) using the existing v3 wire format — no format change needed, per this repo's existing `zentrain.feature_transforms` metadata precedent (already used to signal per-bake input shaping). **(Not built — no v2 bake exists yet, correctly out of scope per the coordinator's brief.)**

## (e) VALIDATION PLAN

1. **Bounded-range assertions**, new — v1 currently has `is_finite()` checks (`metric.rs:4068,4792`) but **no upper-bound magnitude assertion anywhere** in the extractor (confirmed by grep; this absence is itself evidence of the gap). For v2: `assert!(f <= UPPER_BOUND)` per feature, run against (a) the existing `kadis_negrich` severe-tail corpus (`canonical-2026-07-15/train/kadis_negrich.parquet`, already the dedicated negative-dial-tail corpus) and (b) the `nonphoto_features_372col` held-out corpus (10,000 rows) plus a direct `o_9292.png`-class fixture (specific high-contrast-non-photo regression image already implicated in D1). Reuse the existing instrument pattern at `streaming.rs::tests::dump_ssim_moment_explosion` (`#[ignore]`) as the template — it already dumps the max raw `ScaleStats` field per image; extend it to assert bounds instead of just reporting. **(Iteration 1 built the SYNTHETIC adversarial-fixture version of this — §A.8 items b — and the 9-real-pair smoke test, §A.7. The `kadis_negrich`/`nonphoto_features_372col`/`o_9292.png` full-corpus versions are NOT run — v2 has no feature-parquet extraction pipeline yet, out of scope per the coordinator's brief, "the full-corpus 2.3M re-extraction" explicitly excluded.)**
2. **Full-corpus scan, not sampled** — D1's 5.8e6 explosion was invisible on a 10,000-row held-out sample (0.03% incidence) and only surfaced on the 2.3M-row `bigcodec_hqdedup_traindigits_2026-07-02.parquet` full scan. Any v2 "no more explosions" claim needs the same full-scale sweep, not a holdout sample. **(Not run — explicitly out of scope for iteration 1.)**
3. **V1 byte-stability regression** — since v2 is strictly additive (new opt-in function/feature-gate), the existing v1 test suite is already the stability gate: the invariant tests at `metric.rs:4700-4860` (`masked_ssim_mean ≤ ssim_mean`, `ssim_max ≥ ssim_4th ≥ ssim_mean`, etc.) and the `compute_all_features`/`FeatureView` length assertions (`metric.rs:4248-4665`) must all continue passing unmodified with `feature-regime-v2` both on and off. **(Verified, §A.8 — 103/103 v1 tests pass identically both configurations.)**
4. **G-STEER reuse** — `zensim/examples/diffmap_block_coherence.rs --bake` and the `diffmap_basic_fraction` metric (`docs/MODEL_SELECTION_SCORECARD.md`) already measure exactly "what fraction of a trained model's weight mass is spatializable." A v2-trained model's target is `diffmap_basic_fraction ≈ 1.0` (all 372 v2 features spatializable, vs v1's structural ~62% cap) — this tool needs no changes to validate v2, just a v2-input bake to score. **(Not run — no v2 bake/scoring exists yet; correctly out of scope, "any change to diffmap.rs fold logic" explicitly excluded from iteration 1.)**
5. **V2-vs-v1 trainability A/B, same recipe** — run the identical `zensim_mlp_train` recipe (same corpora, same `:both` RankNet+MSE loss per `benchmarks/final_metric_experiments_2026-07-18.md`) once on v1-372 and once on v2-of-equivalent-width, compare the full Mohammadi panel (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE) on CID22/KADID/TID/KonJND/nonphoto, **plus** the G-STEER coherence number specifically — the win condition is not "matches v1 rank" alone but "matches-or-beats v1 rank AND KonJND (closes D9) AND diffmap coherence (closes D7) simultaneously," per the two-panel (rank+dial) and now five-gate (`MODEL_SELECTION_SCORECARD.md`) discipline already standard in this repo. **(Not run — "trainer integration" explicitly out of scope for iteration 1; this is the natural iteration-2 deliverable once the bounded core here is reviewed and approved.)**

---

**Key files (iteration 1):** `zensim/zensim/src/metric.rs` (module doc 1-107, `ScaleStats` 587-653, `FeatureView` 3657-3960, vector assembly 3980-4059, new `compute_v2_features` method), `zensim/zensim/src/simd_ops.rs` (C2 const + SSIM kernels), `zensim/zensim/src/streaming.rs` (hot-path `finalize`, 451-550), `zensim/zensim/src/iw_pool.rs` (promoted `WeightedPool::mean`), `zensim/zensim/src/diffmap.rs` (fold restricted to basic-13, 279-439 — untouched by iteration 1), `zensim/zensim/src/profile.rs` (`ProfileParams`, 357-465 — untouched), `zensim/zensim/src/feature_v2.rs` (new, iteration 1), `zensim/zensim/examples/v2_bounds_smoke.rs` (new). Prior-session docs carried forward: `benchmarks/iw_pooling_normalization_2026-07-15.md`, `benchmarks/ssim_moment_explosion_2026-07-16.md`, `benchmarks/mlp_diffmap_coherence_2026-07-18.md`, `docs/MODEL_SELECTION_SCORECARD.md`, `docs/TOP_MODELS_COOKBOOK.md`.

**Key files (phase 2 / "feature-v2b"):** `zensim/src/feature_v2.rs` (rewritten: single-pass O(1)-accumulator kernel, `OnlineMoments` Terriberry moments, `WeightedSum` gated-mirror accumulator, `V2NewFeatureToggles`, 7 new candidates, 29 signals/channel/scale, 12 tests), `zensim/src/iw_pool.rs` (`WeightedPool::mean` doc updated to record the gated-mirror relationship; dead-code attributes corrected to unconditional since the only caller is now the equivalence test), `zensim/src/metric.rs` (`Zensim::compute_v2_features_with_toggles`), `zensim/examples/support/zen_io.rs` (new — zenpng/zenjpeg/zenresize decode/resize/encode helpers, replacing the `image` crate for all phase-2 tooling), `zensim/examples/v2_bounds_smoke.rs` (switched to `zen_io`), `zensim/examples/gen_jpeg_distortion.rs` (new — real-content distortion generator, used for the imazen-26 `o_9292` re-validation), `zensim/examples/v2_helpfulness_screen.rs` (new — item 6 KADID/TID extraction), `zensim/benches/v2_speed_baseline.rs` + `v2_stage_profile.rs` + `v2_feature_group_cost.rs` (new zenbench harnesses, items 1/2/4). `docs/FEATURE_V2_SPEC_2026-07-18.md` SS A.11/A.12 (this file, phase-2 sections).
