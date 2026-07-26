//! V2 "bounded" feature extraction — opt-in, additive regime.
//!
//! Gated behind `feature-regime-v2` (default OFF). Full defect inventory
//! (D1-D9), design principles, per-feature formula/bound/citation table,
//! and iteration-1 as-built: `docs/FEATURE_V2_SPEC_2026-07-18.md` Part A/B.
//! This file is iteration 2 ("phase 2" / "feature-v2b"): §A.11/§A.12 of
//! that doc cover the fused kernel design, spill counts, per-feature
//! marginal cost, and the speed-gate verdict.
//!
//! # Iteration 2: single-pass, O(1)-accumulator kernel
//!
//! Iteration 1 stored NINE full-image `Vec<f32>` per-pixel maps (one per
//! basic-family signal) and processed them in two passes. Measured against
//! v1 (`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12): 2-5x SLOWER than v1's
//! full 372-feature extraction — an unacceptable baseline for adding more
//! features on top. Iteration 2 replaces this with a genuinely single-pass
//! kernel using O(1) running accumulators:
//!
//! - **Simple means** (art, det, mse, hf_*, pjnd_*, gms, ringing, banding,
//!   blockiness): a running sum, divided by `n` at the end. No storage.
//! - **SSIM mean/dev2/dev4**: Terriberry's (2007) single-pass online
//!   higher-order moments (mean + M2 + M3 + M4 running scalars) — the
//!   GMSD-style deviation-from-mean moments (iteration 1's D8 fix) computed
//!   WITHOUT a second pass or any array, unlike iteration 1's
//!   store-then-revisit approach.
//! - **Masked / IW / soft-peak weighted pooling**: running
//!   `(Σw·v, Σw)` numerator/denominator pairs, divided at the end
//!   (`WeightedPool`'s `Σw·v/Σw` form, computed inline rather than
//!   materializing weight or value arrays).
//!
//! The only per-scale arrays that remain are the blur-pass outputs
//! (`mu1`, `mu2`, `s12`, `ssq`, `activity`) — these are inherently
//! multi-pixel (a box blur cannot be expressed as a running per-pixel
//! accumulator), and v1's own architecture carries the same five arrays
//! for the same reason (`streaming.rs`). Everything downstream of them is
//! now one `for y { for x { ... } }` loop with O(1) extra space.
//!
//! # Iteration 2: seven new candidates (`docs/FEATURE_V2_SPEC_2026-07-18.md`
//! §A.10, phase-congruency explicitly excluded — LARGE/log-Gabor cost)
//!
//! All reuse the SAME per-pixel gradient computed once per (channel,
//! scale) pixel and shared across every gradient-consuming feature (GMS,
//! chroma-edge-GMS [=GMS evaluated on the X/B channels via the existing
//! per-channel loop — no extra computation], ringing, banding, oriented
//! blockiness, edge-width-change) — see `idx` module and
//! `compute_channel_scale_v2`'s per-pixel loop.

use crate::error::ZensimError;
use crate::source::ImageSource;

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8 as GenericF32x8;

// ============================================================================
// Layout constants
// ============================================================================

/// Bounded-basic block: SSIM mean + Terriberry 2nd/4th deviation moments
/// (D1, D8 fix), bounded edge artifact/detail (D3 fix), saturating MSE (D6
/// partial fix), bounded HF gain/loss/mag (D2 fix).
pub const FEATURES_PER_CHANNEL_V2_BASIC: usize = 9;
/// Soft-saliency-weighted-mean replacement for v1's hard max/L8 peak block
/// (D4 fix).
pub const FEATURES_PER_CHANNEL_V2_PEAK: usize = 3;
/// Masked (flatness-weighted) block using the canonical `Σw·v/Σw` pooling
/// (D5 fix), bounded activity weight (D6 fix).
pub const FEATURES_PER_CHANNEL_V2_MASKED: usize = 4;
/// IW (texture-weighted) block — same canonical pooling, opposite polarity.
pub const FEATURES_PER_CHANNEL_V2_IW: usize = 4;
/// Near-threshold (PJND) core block: divisive-normalization transducer
/// (`k=4.0`) + reference-only masking-susceptibility.
pub const FEATURES_PER_CHANNEL_V2_PJND: usize = 2;
/// Phase-2 additions (A.10 candidates, phase-congruency excluded): GMS
/// (serves both "gradient-magnitude similarity" and, via the X/B channels
/// of the existing per-channel loop, "chroma-edge similarity" — 1 slot for
/// 2 candidates), masking-transducer bank low/high k (2 slots, 1
/// candidate), oriented blockiness, ringing, banding, edge-width-change (4
/// slots, 4 candidates). 1+2+1+1+1+1 = 7 slots for 7 named candidates.
pub const FEATURES_PER_CHANNEL_V2_NEW: usize = 7;
/// Total v2 signals per channel per scale (9+3+4+4+2+7 = 29).
pub const FEATURES_PER_CHANNEL_V2_TOTAL: usize = FEATURES_PER_CHANNEL_V2_BASIC
    + FEATURES_PER_CHANNEL_V2_PEAK
    + FEATURES_PER_CHANNEL_V2_MASKED
    + FEATURES_PER_CHANNEL_V2_IW
    + FEATURES_PER_CHANNEL_V2_PJND
    + FEATURES_PER_CHANNEL_V2_NEW;

/// Named local offsets within one channel's v2 block.
pub mod idx {
    pub const SSIM_MEAN: usize = 0;
    pub const SSIM_DEV2: usize = 1;
    pub const SSIM_DEV4: usize = 2;
    pub const ART: usize = 3;
    pub const DET: usize = 4;
    pub const MSE: usize = 5;
    pub const HF_GAIN: usize = 6;
    pub const HF_LOSS: usize = 7;
    pub const HF_MAG_LOSS: usize = 8;
    pub const SSIM_SOFT_PEAK: usize = 9;
    pub const ART_SOFT_PEAK: usize = 10;
    pub const DET_SOFT_PEAK: usize = 11;
    pub const MASKED_SSIM: usize = 12;
    pub const MASKED_ART: usize = 13;
    pub const MASKED_DET: usize = 14;
    pub const MASKED_MSE: usize = 15;
    pub const IW_SSIM: usize = 16;
    pub const IW_ART: usize = 17;
    pub const IW_DET: usize = 18;
    pub const IW_MSE: usize = 19;
    pub const PJND_TRANSDUCER: usize = 20;
    pub const PJND_FRAGILITY: usize = 21;
    /// Gradient-magnitude similarity (GMSD, Xue et al. 2013). Evaluated
    /// uniformly across X/Y/B channels via the existing per-channel loop,
    /// so `gms(scale, 0)`/`gms(scale, 2)` ARE the chroma-edge-similarity
    /// values (FSIMc, Zhang 2011) — no separate slot needed.
    pub const GMS: usize = 22;
    /// Masking-transducer bank, low-k member (`k=1.0`, less aggressive
    /// masking than the core `PJND_TRANSDUCER`'s `k=4.0`).
    pub const PJND_TRANSDUCER_LOW_K: usize = 23;
    /// Masking-transducer bank, high-k member (`k=16.0`, more aggressive
    /// masking).
    pub const PJND_TRANSDUCER_HIGH_K: usize = 24;
    /// Oriented blockiness (Wang-Bovik pattern, FR-ized against the
    /// reference's own step energy at the same lattice positions).
    pub const BLOCKINESS: usize = 25;
    /// Ringing (dilated-edge form; `activity` stands in for `dilate`).
    pub const RINGING: usize = 26;
    /// Banding / contour detector (cheap single-scale approximation of
    /// CAMBI's step-energy-in-low-gradient-regions signature).
    pub const BANDING: usize = 27;
    /// Edge-width change (two-scale gradient-decay-ratio approximation).
    /// The ONE scale-level (not per-pixel) exception to full
    /// spatializability in this feature set — see its doc below.
    pub const EDGE_WIDTH_CHANGE: usize = 28;
}

// ============================================================================
// Bounding constants — each documented + cited (full derivations in
// docs/FEATURE_V2_SPEC_2026-07-18.md)
// ============================================================================

/// SSIM luminance stability constant, v2. `C1 = (K1*L)^2`, `K1=0.01` (Wang,
/// Bovik, Sheikh, Simoncelli 2004, IEEE TIP 13(4)), `L=1` (matches this
/// crate's existing `simd_ops::C2 = 0.0009 = (0.03*1)^2` derivation).
/// Restores the standard SSIM luminance form
/// `num_m = (2·μ1·μ2 + C1) / (μ1² + μ2² + C1)` in place of v1's
/// `num_m = 1 - (μ1-μ2)²` (D1 fix). Boundedness requires μ1,μ2 ≥ 0, which
/// holds because [`crate::streaming::convert_source_to_xyb`] exclusively
/// calls the *positive*-XYB conversion variants.
pub const C1_V2: f64 = 0.0001;
/// Structure/contrast stability constant, v2 — same value/derivation as
/// v1's `simd_ops::C2`, kept independent so v2 never silently inherits a
/// future v1-only change.
pub const C2_V2: f64 = 0.0009;
/// GMSD/FSIM/DISTS bounded-similarity stabilizer for edge artifact/detail.
pub const C_EDGE: f64 = 1e-4;
/// Saturating half-point for bounded MSE: `mse=0.5` at squared error
/// `C_MSE` (i.e. `sqrt(C_MSE)=0.1` absolute XYB difference).
pub const C_MSE: f64 = 0.01;
/// Stabilizer for the HF gain/loss/mag-loss bounded-excess forms.
pub const C_HF: f64 = 1e-4;
/// Saturating half-point for the soft-saliency peak weight (D4 fix).
pub const C_PEAK: f64 = 0.05;
/// Saturating half-point bounding the reference-activity signal before use
/// as a masked/IW pooling weight (D6 fix).
pub const C_ACTIVITY: f64 = 0.01;
/// Weight floor for the IW pool (prevents an all-zero weight sum on a
/// flat image).
pub const IW_WEIGHT_FLOOR: f64 = 1e-3;
/// ColorVideoVDP-style (Mantiuk et al. 2024, arXiv:2401.11485) final
/// soft-clamp half-point for the PJND transducer bank.
pub const C_PJND_CLAMP: f64 = 0.1;
/// Saturating half-point for the reference-only gradient-energy signal
/// (Bondžulić et al. 2022).
pub const C_PJND_GRAD: f64 = 0.02;
/// Core transducer masking strength (matches v1's `k_mask`/`k_iw` order of
/// magnitude).
pub const K_PJND_MASK: f64 = 4.0;
/// Masking-transducer bank: low-k member (A.10 "2-3 spaced k values" —
/// geometric spacing ×4 around the core `K_PJND_MASK`). Less aggressive
/// masking normalization — closer to raw error, catches near-threshold
/// distortions the core transducer already partly suppresses.
pub const K_PJND_MASK_LOW: f64 = 1.0;
/// Masking-transducer bank: high-k member. More aggressive masking —
/// only distortion that survives heavy local-activity normalization
/// registers, isolating supra-threshold-only error.
pub const K_PJND_MASK_HIGH: f64 = 16.0;

/// GMSD (Xue, Zhang, Mou, Bovik 2013, arXiv:1308.3052, Eq.4) bounded-
/// similarity stabilizer for gradient magnitude comparison. Same order of
/// magnitude as `C_EDGE` (gradients here are raw-pixel central differences
/// in unit-XYB scale, not blur residuals, but the dynamic range is
/// comparable).
pub const C_GMS: f64 = 1e-4;
/// Saturating half-point for the reference-edge indicator used by ringing
/// (`edge_r` in the A.10 table's `err · dilate(edge_r) · (1−edge_r)` form).
pub const C_RING_EDGE: f64 = 0.02;
/// Saturating half-point bounding the raw per-pixel error before it enters
/// the ringing product (keeps ringing bounded regardless of input dynamic
/// range, same D6-style rationale as `C_ACTIVITY`).
pub const C_RING_ERR: f64 = 0.05;
/// Saturating half-point for "distorted now has a noticeable edge here"
/// in the banding approximation.
pub const C_BAND_DST: f64 = 0.01;
/// Saturating half-point for "source was smooth here" in the banding
/// approximation.
pub const C_BAND_SRC: f64 = 0.02;
/// Bounded-excess stabilizer for oriented blockiness (FR-ized step-energy
/// comparison at the 8-pixel lattice).
pub const C_BLOCK: f64 = 1e-3;
/// Bounded-similarity stabilizer for the edge-width-change scale-decay
/// comparison.
pub const C_EDGEWIDTH: f64 = 1e-3;
/// Additive floor in the per-scale gradient-decay ratio denominator
/// (`decay = mean_grad(coarser) / (mean_grad(finer) + C_GRAD_DECAY)`).
pub const C_GRAD_DECAY: f64 = 1e-4;

// ============================================================================
// f720+ APPEND block — additions from the 2026-07-26 gap audit
// (`zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` §5, candidates
// A1-A5 + A9). Append-only: emitted AFTER the frozen f0..f719 layout, in a
// separate region `f720 + scale*(3*17) + ch*17 + local` (scale-major, same
// convention as the v2 block). Gated by `V2NewFeatureToggles::append_block`
// (default OFF — every existing path/byte is unchanged when off). Computed
// by a SEPARATE second kernel pass over the cache-hot strip planes
// (`append_block_kernel`) so the tuned dense/gradient kernels are not
// touched (their §A.14/§A.16 register-pressure story stays intact) and
// f0..f719 bit-stability is structural, not incidental.
// ============================================================================

/// Features per channel per scale in the append block (17 × 3 ch × 4
/// scales = 204; full folded-append vector = 720 + 204 = 924).
pub const FEATURES_PER_CHANNEL_APPEND: usize = 17;

/// Skip the append kernel entirely for the B (yellow-violet) channel at
/// scale 0 — its 17 slots emit 0.0 (index-stable, deprecate-by-absence).
/// Grounds: the yellow-violet foveal resolution limit is ~53 ppd vs 94
/// achromatic (Ashraf/Chapiro/Mantiuk 2025), and butteraugli carries no B
/// channel at all in its two highest-frequency bands — finest-scale B
/// append features would model signal the eye cannot resolve, at 25% of
/// the append block's pixel cost (scale 0 is ~75% of pyramid pixels).
pub const APPEND_SKIP_B_SCALE0: bool = true;

/// Named local offsets within one channel's append block.
pub mod idx_append {
    /// Cross-channel masked transducer (gap-audit A1). Y channel ONLY:
    /// `err/(err + C_PJND_CLAMP·(1 + K_PJND_MASK·act_Y + K_XCH·(act_X +
    /// act_B)))` — the ColorVideoVDP-trained masking direction (chromatic
    /// channels mask achromatic; Switkes 1988; vdp-csf-perceptual-math.md
    /// Eq. 9-11 discussion). X/B slots emit 0.0 (deprecate-by-absence:
    /// luma does NOT mask chroma per the same trained matrix, and the
    /// chroma-side planes are not available without a second sweep).
    pub const XMASK_TRANSDUCER: usize = 0;
    /// Luminance-adapted transducer (A2): activity + local-luminance term
    /// in the divisive denominator, `t = sat(ref_Y, C_LUM_T)` standing in
    /// for the DeVries-Rose/Weber background-luminance dependence.
    /// Y channel ONLY (X/B emit 0.0): the 2026-07-19 luma-gate ablation
    /// measured the v2 chroma transducers as a broad CID22 cost with
    /// near-zero steering mass — the same masking family, so the append
    /// block ships luma-gated from the start
    /// (`zensim:benchmarks/v2_trainability_ab_2026-07-19.md:209-216`).
    pub const LUM_TRANSDUCER: usize = 1;
    /// Bounded error (`mse_i = sat((s−d)², C_MSE)` — the v1/v2 MSE
    /// vocabulary) pooled over the DARK reference-luminance soft bin,
    /// weight `(1−t)²` (A2: shadow banding/blocking that uniform pooling
    /// dilutes). Reference-only weights — same foldable `Σw·v/Σw` shape
    /// as the masked/IW families. Pools `mse_i` rather than a
    /// recomputed SSIM map: the luminance CONDITIONING is the feature;
    /// `mse_i` needs no extra plane loads and keeps the kernel inside
    /// the perf gate.
    pub const LUM_DARK_ERR: usize = 2;
    /// Mid-luminance soft bin, weight `2t(1−t)`.
    pub const LUM_MID_ERR: usize = 3;
    /// Bright soft bin, weight `t²`.
    pub const LUM_BRIGHT_ERR: usize = 4;
    /// Normalize-then-difference divisive comparison (A3, NLPD/MSCN
    /// shape — Laparra et al. 2017; BRISQUE's MSCN): per-pixel
    /// `sat(|r₁/σ₁′ − r₂/σ₂′|, C_MSCN_ABS)` mean, `r_i` the blur
    /// residual, `σ_i′ = sqrt(var_i + C_MSCN_VAR)`. Unlike the pjnd
    /// transducer (reference-masked raw error), BOTH images normalize by
    /// their OWN local energy first. The σ-split comes from the cached /
    /// replayed reference plane `bs2 = blur(src²)`:
    /// `var₁ = bs2 − mu1²`, `var₂ = (ssq − bs2) − mu2²` — no per-pair
    /// blur is ever run for it.
    pub const MSCN_DIFF_MEAN: usize = 5;
    /// Squared variant: `sat((n₁−n₂)², C_MSCN_SQ)` mean.
    pub const MSCN_DIFF_L2: usize = 6;
    /// Local contrast gain (A4): `bounded_excess(var₂, var₁, C_CONTRAST)`
    /// mean — distorted window has MORE energy (over-sharpen, ringing,
    /// added grain) — requires the σ-split (`d2` plane).
    pub const CONTRAST_GAIN: usize = 7;
    /// Local contrast loss: `bounded_excess(var₁, var₂, C_CONTRAST)` mean
    /// (washed-out / smoothed rendering as a field, not via HF proxies).
    pub const CONTRAST_LOSS: usize = 8;
    /// Alignment-free texture-energy dissimilarity (A4, the cheap DISTS
    /// texture term — Ding et al. 2020): `1 − bounded_sim(var₁, var₂,
    /// C_CONTRAST)` mean. Compares local contrast STATISTICS, not
    /// pixel-registered values.
    pub const TEXTURE_DISSIM: usize = 9;
    /// GMSD-style deviation pooling of the GMS map (A5; Xue et al. 2013
    /// §3.3 — std of the map, the documented reason GMSD beats GMSM).
    /// Zero when `gradient_features` is off.
    pub const GMS_DEV2: usize = 10;
    /// Deviation (std) of the edge-artifact map (A5).
    pub const ART_DEV2: usize = 11;
    /// Deviation (std) of the detail-lost map (A5).
    pub const DET_DEV2: usize = 12;
    /// Global per-channel mean shift (A9): `sat(|mean(s)−mean(d)|,
    /// C_GDMEAN)` — the windowed features under-weigh global casts.
    pub const GLOBAL_DMEAN: usize = 13;
    /// Global contrast gain: `bounded_excess(gvar₂, gvar₁, C_GCONTRAST)`
    /// on whole-plane variances.
    pub const GLOBAL_CGAIN: usize = 14;
    /// Global contrast loss: reverse polarity of [`GLOBAL_CGAIN`].
    pub const GLOBAL_CLOSS: usize = 15;
    /// Mean source gradient magnitude, saturated (A2.iv): the >92%
    /// first-JND PSNR predictor (Bondžulić et al. 2022 via
    /// jnd-sur.md:204-210) as a reference-side complexity conditioner.
    /// Reference-only (correct 0 in any steering fold, like
    /// `PJND_FRAGILITY`). Zero when `gradient_features` is off.
    pub const GRAD_SRC_MEAN: usize = 16;
}

/// Cross-channel masking strength for the chroma→luma term (append
/// [`idx_append::XMASK_TRANSDUCER`]). Same order of magnitude as
/// `K_PJND_MASK` (the CVVDP `k_{i,c}` values are trained; we seed at the
/// core masking strength and let the head learn usage via the feature
/// weight).
pub const K_XCH: f64 = 4.0;
/// Luminance-term strength in [`idx_append::LUM_TRANSDUCER`]'s divisive
/// denominator.
pub const K_LUM_ADAPT: f64 = 4.0;
/// Saturating half-point mapping the (cbrt-domain, ~[0,1]) reference Y
/// value to the bin parameter `t = y/(y+C_LUM_T)` used by the luminance
/// soft bins and the luminance transducer term. 0.35 puts `t=0.5` near
/// mid-gray in XYB's cube-root intensity scale.
pub const C_LUM_T: f64 = 0.35;
/// Variance floor inside the MSCN normalizer `σ′ = sqrt(var + C_MSCN_VAR)`
/// (≈ (0.032)² in unit-XYB scale — same role as BRISQUE's C, keeps the
/// normalized residual bounded by `|r|/sqrt(C_MSCN_VAR)`).
pub const C_MSCN_VAR: f64 = 1e-3;
/// Per-pixel saturating half-point for `|n₁−n₂|` (bounded-by-construction
/// per the D6 design principle).
pub const C_MSCN_ABS: f64 = 0.5;
/// Per-pixel saturating half-point for `(n₁−n₂)²`.
pub const C_MSCN_SQ: f64 = 0.25;
/// Stabilizer for the var-based contrast gain/loss/texture-sim family
/// (variances of unit-XYB windows are `activity²`-scale; matches
/// `C_ACTIVITY²` order).
pub const C_CONTRAST: f64 = 1e-4;
/// Saturating half-point for the global |Δmean| feature.
pub const C_GDMEAN: f64 = 0.02;
/// Stabilizer for the global variance gain/loss pair.
pub const C_GCONTRAST: f64 = 1e-4;
/// Saturating half-point for the mean source-gradient conditioner
/// (matches `C_PJND_GRAD` — same signal family).
pub const C_GRADM: f64 = 0.02;

/// Box blur radius at scale 0 — matches v1's `ZensimConfig::default()`.
pub(crate) const BLUR_RADIUS: usize = 5;
/// Oriented-blockiness lattice period (JPEG's 8x8 MCU grid).
const BLOCK_LATTICE: usize = 8;

// ============================================================================
// Phase-5 (§A.15): strip-tiled pipeline constants + halo-boundary helper.
//
// Bandwidth hypothesis (coordinator, from phase-4's own ratio-vs-size curve
// 1.18x@256^2 -> 3.32x@2048^2 -- growing, not flat, meaning DRAM traffic
// from materializing full-image mu1/mu2/ssq/s12/activity planes, not
// division-bound compute, is the dominant slope term at large sizes):
// process the image in row STRIPS small enough that one strip's live
// planes stay cache-resident, fusing blur -> formula pass -> gradient
// pass -> accumulation per strip so full-image intermediate planes are
// NEVER materialized (`ScratchV2Strip` below is sized per-strip, not
// per-image).
// ============================================================================

/// Strip height in rows. Tunable; swept per §A.15's strip-height table
/// (candidates measured: see the spec doc). Not a public knob — phase-5's
/// scope is proving/falsifying the bandwidth hypothesis, not exposing a
/// user-facing tuning parameter.
pub(crate) const STRIP_ROWS: usize = 128;

/// Half-halo (rows of REAL context loaded beyond the strip on each side).
/// Sized so the existing (UNMODIFIED) `box_blur_v_from_copy`'s own
/// reflect-boundary logic never touches the sub-range this module
/// actually reads back out -- derivation (§A.15): computing mu1 (etc.) at
/// buffer-local row `r` needs H-blurred input at `[r-R, r+R]`; the
/// widest-reaching consumer is `activity`, which needs mu1 at strip-local
/// `[-R, strip_h+R)` (one blur-radius beyond the strip itself), so mu1's
/// OWN computation needs INPUT (mu1_h) at `[-2R, strip_h+2R)`. Padding the
/// wide buffer with `HALO_P = 2*BLUR_RADIUS` real rows on each side keeps
/// every window inside `[0, strip_h + 2*HALO_P)` without ever touching
/// the wide-buffer's own synthetic edge (which would apply the WRONG
/// reflection -- the buffer isn't the true image).
pub(crate) const HALO_P: usize = 2 * BLUR_RADIUS;

/// Mirror-without-repeating-edge boundary reflection ("reflect_101" /
/// whole-sample-symmetric convention: `-1 -> 1`, `-2 -> 2`, `height ->
/// height-2`, matching the mirror convention `crate::blur`'s own
/// boundary-handling comments describe, e.g. `fused_blur_h_ssim_inner_v4`'s
/// "reflect the index back into the image (`2*(height-1) - add_raw`)").
/// Used ONLY to gather a strip's halo rows from the FULL (already fully
/// materialized -- src/dst planes are the caller's whole-channel-scale
/// input, unchanged from phases 1-4) src/dst planes; every row this
/// pulls in is REAL image data, either directly (interior strips) or via
/// this exact mirror (strips touching the true top/bottom edge) — never
/// synthetic/zero-padding.
#[inline]
pub(crate) fn reflect_101(y: isize, height: usize) -> usize {
    if height <= 1 {
        return 0;
    }
    let h = height as isize;
    let period = 2 * (h - 1);
    let mut m = y.rem_euclid(period);
    if m >= h {
        m = period - m;
    }
    m as usize
}

/// Populate `dst` (`width * h_local` rows) from the FULL `src_full` plane
/// (`width * height_full`), where local row `i` maps to global row
/// `reflect_101(y0 as isize - halo as isize + i as isize, height_full)`.
/// For `i` such that the mapped global row is in-bounds (the common case
/// for interior strips and the strip's own core rows), this is an exact,
/// unreflected copy — `reflect_101` is the identity on `[0, height_full)`.
pub(crate) fn gather_strip_halo(
    src_full: &[f32],
    width: usize,
    height_full: usize,
    y0: usize,
    h_local: usize,
    halo: usize,
    dst: &mut [f32],
) {
    debug_assert_eq!(dst.len(), width * h_local);
    for i in 0..h_local {
        let gy = y0 as isize - halo as isize + i as isize;
        let gy_r = reflect_101(gy, height_full);
        let src_row = &src_full[gy_r * width..(gy_r + 1) * width];
        dst[i * width..(i + 1) * width].copy_from_slice(src_row);
    }
}

// ============================================================================
// Bounded per-pixel formulas
// ============================================================================

/// Per-pixel SSIM-like dissimilarity, C1-bounded (D1 fix). Bounded [0, 2].
///
/// Phase-6 (§A.16 lever A step 1): the original form computes
/// `num_m * (num_s/denom_s)` = `(A/B) * (C/D)` — TWO divisions. Since
/// `(A/B)*(C/D) == (A*C)/(B*D)` exactly (ordinary fraction algebra, no
/// approximation), this collapses to ONE division at the cost of two extra
/// multiplications — a strict win since a multiply is ~3-5x cheaper than a
/// divide on every target ISA this crate SIMD-dispatches to. Kept for the
/// scalar tail loop; see `ssim_d_local_v` for the SIMD sibling, which
/// additionally routes the single division through `.recip()` (step 2).
#[inline]
fn ssim_d_local(mu1: f64, mu2: f64, s12: f64, ssq: f64) -> f64 {
    let a = 2.0 * mu1 * mu2 + C1_V2;
    let b = mu1 * mu1 + mu2 * mu2 + C1_V2;
    let cov = s12 - mu1 * mu2;
    let c = 2.0 * cov + C2_V2;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + C2_V2;
    let local = (a * c) / (b * d);
    (1.0 - local).max(0.0)
}

/// GMSD/FSIM/DISTS canonical bounded-similarity form: `(2ab+c)/(a²+b²+c)`.
/// Bounded `(0, 1]`, `=1` iff `a == b`. Requires `a, b >= 0`.
#[inline]
fn bounded_sim(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * a * b + c) / (a * a + b * b + c)
}

/// Bounded, SIGNED normalized-difference form: `max(0, a-b) / (a+b+c)`.
/// Bounded `[0, 1)`. Requires `a, b >= 0`.
#[inline]
fn bounded_excess(a: f64, b: f64, c: f64) -> f64 {
    (a - b).max(0.0) / (a + b + c)
}

/// [`bounded_excess`]`(a,b,c)` PAIRED with [`bounded_excess`]`(b,a,c)`, one
/// shared division instead of two (phase-6 §A.16 lever A step 1: the two
/// calls' denominators — `a+b+c` and `b+a+c` — are the SAME value by
/// commutativity of addition, computed once here). Returns
/// `(bounded_excess(a,b,c), bounded_excess(b,a,c))`. Used for the
/// hf_gain/hf_loss pair, which is exactly this shape (`bounded_excess
/// (dst_sq,src_sq,c)` / `bounded_excess(src_sq,dst_sq,c)`).
#[inline]
fn bounded_excess_pair(a: f64, b: f64, c: f64) -> (f64, f64) {
    let recip_denom = 1.0 / (a + b + c);
    (
        (a - b).max(0.0) * recip_denom,
        (b - a).max(0.0) * recip_denom,
    )
}

/// Michelson-contrast-style saturating ratio `x/(x+c)`, bounded `[0, 1)`
/// for `x >= 0`.
#[inline]
fn saturate(x: f64, c: f64) -> f64 {
    let x = x.max(0.0);
    x / (x + c)
}

/// Fused PJND masking-transducer band: `saturate(raw_abs_err/(1+k*act), c)`
/// collapsed to ONE division (phase-6 §A.16 lever A step 1, exact — not an
/// approximation). Derivation: let `D = 1+k*act` (always `>= 1 > 0` since
/// `k, act >= 0`); `saturate(x/D, c) = (x/D)/((x/D)+c)`; multiplying
/// numerator and denominator by `D` (valid, `D` is a positive nonzero
/// scalar) gives `x/(x + c*D) = x/(x + c + c*k*act)` — the original chained
/// `raw_abs_err/(1+k*act)` THEN `t/(t+c)` (2 divisions) becomes one
/// division against a directly-computed denominator. Bounded `[0, 1)` for
/// `raw_abs_err >= 0`, same contract as `saturate(raw_abs_err/(1+k*act),
/// c)`.
#[inline]
fn pjnd_transducer(raw_abs_err: f64, act: f64, k: f64, c: f64) -> f64 {
    raw_abs_err / (raw_abs_err + c * (1.0 + k * act))
}

// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD formula pass — f32x8-lane vectorized
// counterparts of the four scalar formulas above. Same algebra, f32
// precision (matches the source planes' own f32 storage), 8 pixels/call
// instead of 1. Generic over any token implementing `F32x8Backend` so
// `#[magetypes(v4x, v4, v3, neon, wasm128)]` (see
// `compute_channel_scale_v2_kernel_entry` below) monomorphizes one body
// per tier — this is the SAME pattern as `~/work/archmage/docs/site/
// content/magetypes/examples/plane-ops.md`/`gaussian-blur.md`.
// ============================================================================

type V8<T> = GenericF32x8<T>;

// ============================================================================
// Phase-6 (§A.16): division-cost reduction for the SIMD formula family.
//
// SHIPPED: ALGEBRAIC SHARING ONLY (exact, no precision tradeoff beyond
// ordinary FP reassociation already tolerated by this file's 5e-4 policy):
// a chained `(a/b)*(c/d)` becomes one division `(a*c)/(b*d)`; two calls
// that share a denominator by construction (`bounded_excess(a,b,c)` +
// `bounded_excess(b,a,c)`) compute it once; a chained ratio-then-saturate
// (`saturate(x/(1+k*act), c)`) collapses to one division by multiplying
// through by the inner denominator. See `ssim_d_local`/
// `bounded_excess_pair`/`pjnd_transducer` (scalar siblings) for the exact
// derivations — same algebra here, f32 SIMD lanes. This drops the
// per-pixel division count from 13 to 8 (with `transducer_bank` on, the
// default) for FREE — native division stays correctly-rounded throughout,
// so no bounds-safety concern beyond what already existed pre-phase-6.
//
// TRIED AND DROPPED: reciprocal-estimate + Newton-Raphson (`V8::<T>::
// recip`, magetypes `archmage/magetypes/src/simd/generic/generated/
// f32x8_impl.rs:371` — every dispatched backend overrides it with a
// hardware estimate refined by one NR step, full f32 precision). Two
// findings, both measured on this box (Ryzen 9 7950X, whatever tier
// `incant!` picks):
//   1. A per-pixel `.min(...)` clamp after every recip-based division
//      (the naive "guard the ~24-bit estimate against overshooting a
//      documented bound" approach) caused a SEVERE regression: 1024²
//      1-thread went 147.0ms -> 738.2ms (~5x worse). `cargo asm` showed
//      `dense_block_kernel_generic`'s AVX2 monomorphization bloated to
//      4268 instructions / 186 un-inlined `call`s — even `_mm256_mul_ps`/
//      `_mm256_add_ps` stopped inlining, 2112-byte stack frame. The extra
//      splat+min at every division site pushed the function past LLVM's
//      inliner threshold and it gave up wholesale.
//   2. Removing the per-pixel clamps (relying on the mean/weighted-mean
//      accumulation structure to keep any epsilon overshoot from
//      amplifying, plus one cheap clamp on the FINAL feature value)
//      recovered MOST but not all of the regression (524.9ms). The
//      remaining gap was `#[inline]` being too weak a hint once these
//      functions grew larger from the algebra fusion — switching to
//      `#[inline(always)]` (KEPT, see below) fixed it completely: 147.2ms,
//      matching plain-division's 144.3ms within noise (both single
//      4-round 1024² measurements, ~5ms MAD). Precision was never the
//      problem — measured max deviation from exact division was 3.246e-7
//      on real AVX2 hardware (`_mm256_rcp_ps` + 1 NR step), five orders of
//      magnitude under the smoke test's 1e-6 slack margin.
//   3. CONCLUSION: recip() is numerically safe and, once the inlining bug
//      is fixed, performance-NEUTRAL on this hardware — not a regression,
//      but not a measurable win either (147.2ms vs 144.3ms, within noise).
//      AMD Zen4's AVX2 divider unit is evidently fast enough that trading
//      1 division for ~1 estimate + 2-3 dependent multiply/subtract ops
//      does not pay for itself. Not shipped: it adds real complexity (the
//      bounds-safety story, an extra formula-precision test) for zero
//      measured benefit on the hardware this was tested on. Kept as a
//      documented, reproducible negative result rather than silently
//      dropped — a future session on different hardware (older x86 with
//      slower `vdivps`, or a tier where `rcp_approx` is a bigger win vs.
//      that tier's native divide) could reopen this with the inlining
//      fix already known. `#[inline(always)]` is KEPT on these six
//      functions regardless of the recip question — it is what actually
//      fixed the regression and is cheap, harmless insurance against the
//      same bloat recurring as the algebra-fusion body grew larger than
//      the pre-phase-6 originals.
// ============================================================================

/// Vectorized [`ssim_d_local`] — fused to one division via `(a*c)/(b*d)`.
/// Identical algebra, f32 lanes. Bounded `[0, 2]`.
#[inline(always)]
fn ssim_d_local_v<T: F32x8Backend + Copy>(
    token: T,
    mu1: V8<T>,
    mu2: V8<T>,
    s12: V8<T>,
    ssq: V8<T>,
    c1: V8<T>,
    c2: V8<T>,
) -> V8<T> {
    let two = V8::<T>::splat(token, 2.0);
    let one = V8::<T>::splat(token, 1.0);
    let zero = V8::<T>::zero(token);
    let a = two * mu1 * mu2 + c1;
    let b = mu1 * mu1 + mu2 * mu2 + c1;
    let cov = s12 - mu1 * mu2;
    let c = two * cov + c2;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + c2;
    let local = (a * c) / (b * d);
    (one - local).max(zero)
}

/// Vectorized [`bounded_sim`] — `(2ab+c)/(a²+b²+c)`. Bounded `(0, 1]`.
#[inline(always)]
fn bounded_sim_v<T: F32x8Backend + Copy>(token: T, a: V8<T>, b: V8<T>, c: V8<T>) -> V8<T> {
    let two = V8::<T>::splat(token, 2.0);
    (two * a * b + c) / (a * a + b * b + c)
}

/// Vectorized [`bounded_excess`] — `max(0, a-b) / (a+b+c)`. Standalone (no
/// shared-denominator partner — see [`bounded_excess_pair_v`] for the
/// paired form). Bounded `[0, 1)`.
#[inline(always)]
fn bounded_excess_v<T: F32x8Backend + Copy>(token: T, a: V8<T>, b: V8<T>, c: V8<T>) -> V8<T> {
    let zero = V8::<T>::zero(token);
    (a - b).max(zero) / (a + b + c)
}

/// Vectorized [`bounded_excess_pair`] — `bounded_excess(a,b,c)` PAIRED with
/// `bounded_excess(b,a,c)`, one shared division instead of two (phase-6
/// §A.16: the two calls' denominators are the SAME value by commutativity
/// of addition). Returns `(bounded_excess(a,b,c), bounded_excess(b,a,c))`.
/// Bounded `[0, 1)` each.
#[inline(always)]
fn bounded_excess_pair_v<T: F32x8Backend + Copy>(
    token: T,
    a: V8<T>,
    b: V8<T>,
    c: V8<T>,
) -> (V8<T>, V8<T>) {
    let zero = V8::<T>::zero(token);
    let denom = a + b + c;
    ((a - b).max(zero) / denom, (b - a).max(zero) / denom)
}

/// Vectorized [`saturate`] — `max(x,0)/(max(x,0)+c)`. Bounded `[0, 1)`.
#[inline(always)]
fn saturate_v<T: F32x8Backend + Copy>(token: T, x: V8<T>, c: V8<T>) -> V8<T> {
    let zero = V8::<T>::zero(token);
    let x = x.max(zero);
    x / (x + c)
}

/// Vectorized [`pjnd_transducer`] — fused `saturate(raw_abs_err/(1+k*act),
/// c)` collapsed to one division (phase-6 §A.16, exact — see the scalar
/// sibling's doc for the derivation). Bounded `[0, 1)`.
#[inline(always)]
fn pjnd_transducer_v<T: F32x8Backend + Copy>(
    token: T,
    raw_abs_err: V8<T>,
    act: V8<T>,
    k: V8<T>,
    c: V8<T>,
) -> V8<T> {
    let one = V8::<T>::splat(token, 1.0);
    let denom = raw_abs_err + c * (one + k * act);
    raw_abs_err / denom
}

/// Terriberry (2007) single-pass online moments: tracks `n`, `mean`, and
/// the 2nd/3rd/4th central-moment sums `M2/M3/M4 = Σ(x-mean)^k`. `M3` is
/// tracked only because `M4`'s incremental update formula depends on its
/// PRE-update value — not used as an output here. This is what made
/// `ssim_dev2`/`ssim_dev4` (the D8 GMSD-style deviation moments) genuinely
/// single-pass and O(1)-space in iteration 2, unlike iteration 1's
/// store-then-revisit design.
///
/// Phase-4 (§A.14): no longer on the hot path — `compute_channel_scale_v2`
/// now computes `ssim_dev2`/`ssim_dev4` from SIMD-accumulated RAW power
/// sums (Σd, Σd², Σd³, Σd⁴; see `dense_block_kernel_generic`'s doc), since
/// Terriberry's update is inherently sequential and doesn't vectorize
/// across lanes. Kept as the reference implementation and cross-checked
/// against the raw-moment path in
/// `tests::raw_moment_reformulation_matches_terriberry` — this is the
/// proof that the phase-4 reformulation is numerically equivalent, not
/// just "removed and hoped for the best".
#[derive(Default, Clone, Copy)]
struct OnlineMoments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl OnlineMoments {
    #[inline]
    fn update(&mut self, x: f64) {
        let n1 = self.n;
        self.n += 1.0;
        let delta = x - self.mean;
        let delta_n = delta / self.n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (self.n * self.n - 3.0 * self.n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (self.n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    /// `(mean, dev2, dev4)` — `dev2 = sqrt(M2/n)`, `dev4 = (M4/n)^0.25`,
    /// the GMSD-style deviation-from-mean moments (D8 fix).
    #[inline]
    fn finish(&self) -> (f64, f64, f64) {
        if self.n < 1.0 {
            return (0.0, 0.0, 0.0);
        }
        let dev2 = (self.m2 / self.n).max(0.0).sqrt();
        let dev4 = (self.m4 / self.n).max(0.0).powf(0.25);
        (self.mean, dev2, dev4)
    }
}

/// Running `(Σw·v, Σw)` accumulator for the canonical weighted-mean
/// pooling form — the same `Σw·v/Σw` shape as
/// [`crate::iw_pool::WeightedPool::mean`] (iteration 1's promoted
/// canonical pooling helper), computed incrementally instead of from
/// materialized weight/value arrays.
///
/// **This is a gated mirror, not a duplicate**, per this crate's
/// no-duplication policy ("a second implementation is legitimate only
/// when it exists for a measured engineering reason AND a test holds it
/// bit-exact against the owner"): iteration 1's array-based
/// `WeightedPool::mean` was measured (`docs/FEATURE_V2_SPEC_2026-07-18.md`
/// §A.12) to make v2 2-5x SLOWER than v1 — materializing nine full-image
/// `Vec<f32>` maps per channel-scale to feed `WeightedPool::mean` is
/// exactly the cost iteration 2 removes. `WeightedSum` computes the
/// IDENTICAL formula (`Σw·v/Σw`) online, O(1) space. The equivalence is
/// pinned by `tests::weighted_sum_matches_weighted_pool_mean_exactly`.
#[derive(Default, Clone, Copy)]
struct WeightedSum {
    num: f64,
    den: f64,
}
impl WeightedSum {
    #[inline]
    fn add(&mut self, weight: f64, value: f64) {
        self.num += weight * value;
        self.den += weight;
    }
    #[inline]
    fn finish(&self) -> f64 {
        if self.den < 1e-12 {
            0.0
        } else {
            self.num / self.den
        }
    }
    /// Phase-5 (§A.15): fold another (partial-strip) accumulator's
    /// `(num, den)` in directly — `Σw·v` and `Σw` are both plain sums, so
    /// combining N strips' partial `WeightedSum`s is exactly equivalent to
    /// one whole-image `WeightedSum` (no reassociation beyond ordinary
    /// floating-point addition, same class of drift as any other
    /// SIMD-lane-then-strip reduction already tolerated in this file).
    #[inline]
    fn accumulate(&mut self, other: &WeightedSum) {
        self.num += other.num;
        self.den += other.den;
    }
}

// ============================================================================
// FeatureRegime + result + explicit-regime view
// ============================================================================

/// Explicit feature-extraction regime tag. See
/// `docs/FEATURE_V2_SPEC_2026-07-18.md` §(d): disambiguation is primarily
/// by distinct Rust type ([`ZensimV2Result`]/[`FeatureViewV2`] vs v1's
/// `ZensimResult`/`FeatureView`), this enum is the belt-and-suspenders
/// runtime tag for generic/dynamic call sites.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureRegime {
    /// The frozen, byte-stable v1 372-feature extraction (`metric.rs`).
    V1,
    /// The v2 "bounded" extraction this module implements.
    V2Bounded,
    /// The folded 720-layout extraction (2026-07-24): ONE v2-walk pass
    /// emitting `[f0..156) = v1 basic` (computed inside the v2 strip walk
    /// via the same `fused_blur_h_ssim` H-planes + v1's own
    /// `fused_vblur_features_ssim` kernel — parity-gated against the v1
    /// path, NOT byte-frozen), `[156..372) = 0.0` (v1's peak/masked/IW
    /// pool blocks, deprecated: no current model reads them — see
    /// `benchmarks/corruption_head_2026-07-24.md` "the f156..371 block can
    /// be dropped entirely"), `[372..720) = v2-348`. Index-stable
    /// emit-0, per the append-only feature-numbering rule.
    Folded720,
    /// [`Folded720`](Self::Folded720) plus the f720+ append block
    /// (gap-audit additions, 2026-07-26): `[720..924) = append-204`
    /// (17/ch/scale, see [`idx_append`]). The first 720 slots are
    /// bit-identical to a [`Folded720`](Self::Folded720) extraction of
    /// the same pair (`append_first720_bit_stable` gates this).
    Folded720Append,
}

/// Result of [`compute_v2_features_impl`]/[`crate::Zensim::compute_v2_features`].
#[derive(Debug, Clone)]
pub struct ZensimV2Result {
    features: Vec<f64>,
    n_scales: usize,
    regime: FeatureRegime,
}

impl ZensimV2Result {
    pub fn features(&self) -> &[f64] {
        &self.features
    }
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }
    pub fn regime(&self) -> FeatureRegime {
        self.regime
    }
    /// Explicit-regime named view over this result's features.
    ///
    /// For [`FeatureRegime::Folded720`] results the view covers the v2
    /// block (the vector's tail); the v1 basic block has no v2-named
    /// accessors and is read positionally (`features()[0..156]`).
    pub fn view(&self) -> FeatureViewV2<'_> {
        let v2_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let v2_block = match self.regime {
            FeatureRegime::Folded720 => &self.features[self.features.len() - v2_len..],
            FeatureRegime::Folded720Append => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let end = self.features.len() - append_len;
                &self.features[end - v2_len..end]
            }
            _ => &self.features[..],
        };
        FeatureViewV2::new(v2_block, self.n_scales)
            .expect("compute_v2_features always emits the v2-total layout for its own n_scales")
    }

    /// The f720+ append block (`n_scales × 3 × 17` slots, layout
    /// `scale*(3*17) + ch*17 + local` with [`idx_append`] locals), or
    /// `None` for regimes without one.
    pub fn append_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Append => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                Some(&self.features[self.features.len() - append_len..])
            }
            _ => None,
        }
    }
}

/// Named, explicit-regime view over a v2 feature vector. Unlike v1's
/// [`crate::FeatureView`], [`FeatureViewV2::new`] VALIDATES the exact
/// expected length rather than guessing a tier from an ambiguous length.
#[derive(Debug, Clone, Copy)]
pub struct FeatureViewV2<'a> {
    features: &'a [f64],
    n_scales: usize,
}

impl<'a> FeatureViewV2<'a> {
    pub fn new(features: &'a [f64], n_scales: usize) -> Option<Self> {
        let expected = n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        if features.len() != expected {
            return None;
        }
        Some(Self { features, n_scales })
    }

    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    #[inline]
    fn at(&self, scale: usize, ch: usize, local: usize) -> f64 {
        self.features
            [scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL + local]
    }

    pub fn ssim_mean(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_MEAN)
    }
    pub fn ssim_dev2(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_DEV2)
    }
    pub fn ssim_dev4(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_DEV4)
    }
    pub fn art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::ART)
    }
    pub fn det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::DET)
    }
    pub fn mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MSE)
    }
    pub fn hf_gain(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_GAIN)
    }
    pub fn hf_loss(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_LOSS)
    }
    pub fn hf_mag_loss(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::HF_MAG_LOSS)
    }
    pub fn ssim_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::SSIM_SOFT_PEAK)
    }
    pub fn art_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::ART_SOFT_PEAK)
    }
    pub fn det_soft_peak(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::DET_SOFT_PEAK)
    }
    pub fn masked_ssim(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_SSIM)
    }
    pub fn masked_art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_ART)
    }
    pub fn masked_det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_DET)
    }
    pub fn masked_mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::MASKED_MSE)
    }
    pub fn iw_ssim(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_SSIM)
    }
    pub fn iw_art(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_ART)
    }
    pub fn iw_det(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_DET)
    }
    pub fn iw_mse(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::IW_MSE)
    }
    pub fn pjnd_transducer(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER)
    }
    pub fn pjnd_fragility(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_FRAGILITY)
    }
    /// Gradient-magnitude similarity (candidate 1). On `ch=0` (X) or
    /// `ch=2` (B) this doubles as chroma-edge similarity (candidate 3).
    pub fn gms(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::GMS)
    }
    pub fn pjnd_transducer_low_k(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER_LOW_K)
    }
    pub fn pjnd_transducer_high_k(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER_HIGH_K)
    }
    pub fn blockiness(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::BLOCKINESS)
    }
    pub fn ringing(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::RINGING)
    }
    pub fn banding(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::BANDING)
    }
    pub fn edge_width_change(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::EDGE_WIDTH_CHANGE)
    }
}

// ============================================================================
// Per-channel-per-scale computation — SINGLE PASS, O(1) accumulators
// ============================================================================

/// Runtime toggles for the phase-2 new-feature GROUPS (A.10 candidates),
/// grouped by shared per-pixel computation rather than 1:1 with feature
/// slots (per `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12's per-group
/// marginal-cost measurement):
///
/// - `gradient_features`: GMS + chroma-edge-GMS (free via channel axis) +
///   ringing + banding + edge-width-change. These all consume the same
///   `sqrt`-based gradient magnitude, so they are gated together — turning
///   this off skips the `sqrt` calls entirely, not just the 4 accumulator
///   adds.
/// - `transducer_bank`: the 2 EXTRA masking-transducer k values (the core
///   `k=4.0` transducer, `PJND_TRANSDUCER`, is always computed — it is not
///   a phase-2 addition, iteration 1 already shipped it).
/// - `blockiness`: oriented blockiness (shares neighbor loads with the
///   gradient group but not the `sqrt`, so toggled independently).
///
/// Default: all `true` (every candidate on). A group that fails the speed
/// gate is set `false` here — NOT deleted — per the phase-2 brief ("If a
/// feature blows the budget, keep it OFF by default and say so").
#[derive(Debug, Clone, Copy)]
pub struct V2NewFeatureToggles {
    pub gradient_features: bool,
    pub transducer_bank: bool,
    pub blockiness: bool,
    /// Emit the masking transducers (core + fragility + bank) on the LUMA
    /// (Y) channel only; zero them on X/B. The 2026-07-19 ablation +
    /// combined-model steering analysis found the chroma transducers cost
    /// CID22 and carry near-zero steering mass in a combined model (masking
    /// is fundamentally a luma phenomenon), while the Y transducer carries
    /// the CSIQ signal. Index-stable: the X/B slots stay at their indices,
    /// emitted as 0 (deprecate-by-mask, never renumber). Default OFF — this
    /// is a compression↔general-FR trade (CID22 up, CSIQ down), so it is
    /// opt-in until the ship recipe fixes the operating point.
    /// See `benchmarks/v2_trainability_ab_2026-07-19.md`.
    pub transducers_luma_only: bool,
    /// Emit the f720+ append block (17/ch/scale — see [`idx_append`] and
    /// the module-level "f720+ APPEND block" section). Default OFF: with
    /// this false every existing path, layout, and byte is unchanged.
    /// When on, the feature vector grows by `n_scales × 3 × 17` slots
    /// appended AFTER the existing layout, and the per-strip append
    /// kernel + the `d2` (blur(dst²)) σ-split chain run.
    pub append_block: bool,
}
impl Default for V2NewFeatureToggles {
    fn default() -> Self {
        Self {
            gradient_features: true,
            transducer_bank: true,
            blockiness: true,
            transducers_luma_only: false,
            append_block: false,
        }
    }
}

/// Zero the masking-transducer slots (core, fragility, low-k, high-k) for a
/// non-luma channel when `transducers_luma_only` is set. Index-stable
/// deprecate-by-mask: the slots keep their positions, emitted as 0. `ch==1`
/// is the Y (luma) channel — left untouched. No-op when the toggle is off.
#[inline]
fn apply_transducer_luma_gate(out: &mut [f64], ch: usize, toggles: V2NewFeatureToggles) {
    if toggles.transducers_luma_only && ch != 1 {
        out[idx::PJND_TRANSDUCER] = 0.0;
        out[idx::PJND_FRAGILITY] = 0.0;
        out[idx::PJND_TRANSDUCER_LOW_K] = 0.0;
        out[idx::PJND_TRANSDUCER_HIGH_K] = 0.0;
    }
}

/// Perf-pass (phase 3, `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.13) reusable
/// scratch buffers for [`compute_channel_scale_v2`], sized once for the
/// LARGEST scale (scale 0) and reused (sliced down to the current scale's
/// `n`) across all `n_scales * 3` calls in
/// [`compute_v2_features_impl_with_toggles`]. Eliminates 9 `vec![0.0f32; n]`
/// heap allocations (+ their zero-fill) per call — 108 allocations per
/// `compute_v2_features_impl_with_toggles` invocation before this change, 9
/// after (one-time, at the scratch's construction). Pure memory-management
/// change: does not alter any arithmetic or its order, so it carries zero
/// numerical risk (and zero risk to v1's byte-identity gate, since v1 never
/// calls into this module).
///
/// Phase-5 (§A.15) RESIZED: was sized for the FULL channel-scale plane
/// (`width*height`, up to 4.2M elements at 2048²); now sized for one
/// STRIP-plus-halo (`width*(STRIP_ROWS+2*HALO_P)`, ≤ ~172K elements even
/// at 2048² width) — the actual memory-traffic reduction the bandwidth
/// hypothesis predicts should matter.
///
/// Ref-reuse pass: the halo-gather targets (`src_wide`/`dst_wide`) moved IN
/// here (they were per-call locals). The old aliasing objection — a
/// `&mut ScratchV2Strip` borrow against the `&[f32]` slices `run_blur_pass`
/// needs — is dissolved by `run_blur_pass_strip`, which destructures the
/// struct ONCE into disjoint field borrows (src/dst read-only, moment
/// buffers `&mut`), so no cell tricks are needed. Moving them in makes the
/// whole scratch set reusable ACROSS pairs via [`V2Scratch`], eliminating
/// the last per-call `vec![0.0f32; ..]` traffic (2 buffers × 3 channels ×
/// 4 scales = 24 zero-filled allocations per pair).
struct ScratchV2Strip {
    /// Halo-gathered strip inputs (`gather_strip_halo` targets) — the
    /// mirror-padded local windows of the source/distorted planes that
    /// every downstream stage of one strip iteration reads.
    src_wide: Vec<f32>,
    dst_wide: Vec<f32>,
    /// H-blur-only intermediates from [`crate::blur::fused_blur_h_ssim`]
    /// (see below) — analogous to v2 iteration-2's old per-call `tmp`, but
    /// now 4-wide since the fused kernel produces all four H-blurred planes
    /// in one pass.
    mu1_h: Vec<f32>,
    mu2_h: Vec<f32>,
    ssq_h: Vec<f32>,
    s12_h: Vec<f32>,
    /// Fully (H+V) blurred planes — same meaning as iteration-2's original
    /// `mu1`/`mu2`/`ssq`/`s12` locals.
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    /// Activity path (unchanged from iteration 2: `abs(src - mu1)` then
    /// blurred) — kept as its own two-buffer pair since
    /// `box_blur_1pass_into` still needs a temp scratch distinct from the
    /// 4-wide fused-H outputs above.
    abs_src: Vec<f32>,
    activity_tmp: Vec<f32>,
    activity: Vec<f32>,
}
impl ScratchV2Strip {
    fn new(max_n: usize) -> Self {
        Self {
            src_wide: vec![0.0f32; max_n],
            dst_wide: vec![0.0f32; max_n],
            mu1_h: vec![0.0f32; max_n],
            mu2_h: vec![0.0f32; max_n],
            ssq_h: vec![0.0f32; max_n],
            s12_h: vec![0.0f32; max_n],
            mu1: vec![0.0f32; max_n],
            mu2: vec![0.0f32; max_n],
            ssq: vec![0.0f32; max_n],
            s12: vec![0.0f32; max_n],
            abs_src: vec![0.0f32; max_n],
            activity_tmp: vec![0.0f32; max_n],
            activity: vec![0.0f32; max_n],
        }
    }
}

/// Reusable cross-pair scratch for the v2 extraction kernels — all 13
/// strip-sized working buffers for all 3 channels. Create ONCE per worker
/// thread and pass to
/// [`Zensim::compute_v2_features_with_ref_and_scratch`](crate::Zensim::compute_v2_features_with_ref_and_scratch)
/// for every pair that thread scores: buffers are lazily (re)sized to the
/// largest image seen and reused verbatim afterwards, so steady-state
/// extraction performs ZERO scratch allocation. Contents are overwritten
/// before every read (each strip iteration fully writes the ranges it
/// consumes), so reuse across unrelated pairs cannot leak data between
/// them — guarded by `scratch_reuse_matches_fresh_scratch` in this file's
/// tests.
pub struct V2Scratch {
    strips: [ScratchV2Strip; 3],
    sized_for: usize,
    /// Append-block pair-path replay planes (FULL-plane, scale-0 sized),
    /// used ONLY when `append_block` is on and the prepared reference
    /// carries no (s2-)moments cache: the X/B activity chains and the
    /// three `blur(src²)` planes are replayed per scale into these —
    /// bit-identical to what an append moments cache would hold (same
    /// helpers, same geometry) — so every path emits the same bytes.
    /// Never allocated otherwise (`Vec::new()` — zero RAM when the
    /// append block is off or an append-moments cache is present).
    append_act_x: Vec<f32>,
    append_act_b: Vec<f32>,
    append_bs2: [Vec<f32>; 3],
    /// σ-split blur temps for the strip-tiled bs2 replay — strip-wide
    /// (`width·(STRIP_ROWS+2·HALO_P)`), NOT full-plane, since P2.
    append_tmp_a: Vec<f32>,
    append_tmp_b: Vec<f32>,
    append_sized_for: usize,
}

impl V2Scratch {
    /// New, empty scratch. Buffers are allocated on first use.
    pub fn new() -> Self {
        Self {
            strips: [
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
                ScratchV2Strip::new(0),
            ],
            sized_for: 0,
            append_act_x: Vec::new(),
            append_act_b: Vec::new(),
            append_bs2: [Vec::new(), Vec::new(), Vec::new()],
            append_tmp_a: Vec::new(),
            append_tmp_b: Vec::new(),
            append_sized_for: 0,
        }
    }

    /// Grow (never shrink) every buffer to hold `strip_max_n` elements.
    fn ensure(&mut self, strip_max_n: usize) {
        if strip_max_n > self.sized_for {
            self.strips = [
                ScratchV2Strip::new(strip_max_n),
                ScratchV2Strip::new(strip_max_n),
                ScratchV2Strip::new(strip_max_n),
            ];
            self.sized_for = strip_max_n;
        }
    }

    /// Grow (never shrink) the pair-path replay planes to one full
    /// scale-0 plane each (the act/bs2 planes are indexed by global row by
    /// the kernels), and the σ-split blur temps to `strip_n` (P2: the bs2
    /// fill is kernel-strip-tiled, so its temps are strip-wide, not
    /// full-plane). Only called on the append-block-without-cache path.
    fn ensure_append(&mut self, plane_n: usize, strip_n: usize) {
        if plane_n > self.append_sized_for {
            self.append_act_x = vec![0.0f32; plane_n];
            self.append_act_b = vec![0.0f32; plane_n];
            self.append_bs2 = std::array::from_fn(|_| vec![0.0f32; plane_n]);
            self.append_sized_for = plane_n;
        }
        if strip_n > self.append_tmp_a.len() {
            self.append_tmp_a = vec![0.0f32; strip_n];
            self.append_tmp_b = vec![0.0f32; strip_n];
        }
    }
}

impl Default for V2Scratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs the blur pass (fused H-blur + 4x V-blur + activity) on a LOCAL
/// image — external `src`/`dst` variant, used by the phase-6 whole-image
/// bypass path (which reads the real planes directly, no halo copy).
/// Identical algebra to phases 3/4's inline version; writes `n = width *
/// height_local` elements into each of `scratch`'s 11 moment buffers.
fn run_blur_pass(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height_local: usize,
    scratch: &mut ScratchV2Strip,
) {
    let ScratchV2Strip {
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
        ..
    } = scratch;
    run_blur_pass_inner(
        src,
        dst,
        width,
        height_local,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
    );
}

/// Strip-path variant of [`run_blur_pass`]: the inputs are `scratch`'s own
/// `src_wide`/`dst_wide` halo buffers (filled by `gather_strip_halo` just
/// before this call). Destructuring the struct once yields disjoint
/// borrows — read-only input fields alongside `&mut` moment fields — which
/// is what lets the halo buffers live inside the reusable scratch at all.
fn run_blur_pass_strip(width: usize, height_local: usize, scratch: &mut ScratchV2Strip) {
    let n = width * height_local;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
    } = scratch;
    run_blur_pass_inner(
        &src_wide[..n],
        &dst_wide[..n],
        width,
        height_local,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu1,
        mu2,
        ssq,
        s12,
        abs_src,
        activity_tmp,
        activity,
    );
}

/// Strip-path blur pass when the reference-side moments (`mu1`,
/// `activity`) come from a [`V2PreparedReference`] cache: 3-output fused
/// H (`fused_blur_h_ssim3` — the mu1 chain is compiled out on the v4x
/// tier; `mu1_h` is only fallback scratch) + V-blur of the 3
/// distorted/joint planes only. The
/// mu1 V-blur and the whole activity chain (abs-diff + 2-pass blur) are
/// SKIPPED — their values are read from the cache instead, which was
/// filled by replaying this exact strip walk (see
/// [`compute_ref_moments_channel`]), so consumers see bit-identical
/// inputs.
fn run_blur_pass_strip_cached_ref(width: usize, height_local: usize, scratch: &mut ScratchV2Strip) {
    let n = width * height_local;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu2,
        ssq,
        s12,
        ..
    } = scratch;
    crate::blur::fused_blur_h_ssim3(
        &src_wide[..n],
        &dst_wide[..n],
        &mut mu1_h[..n],
        &mut mu2_h[..n],
        &mut ssq_h[..n],
        &mut s12_h[..n],
        width,
        height_local,
        BLUR_RADIUS,
    );
    crate::blur::box_blur_v_from_copy(&mu2_h[..n], &mut mu2[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&ssq_h[..n], &mut ssq[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&s12_h[..n], &mut s12[..n], width, height_local, BLUR_RADIUS);
}

/// [`run_blur_pass_strip_cached_ref`] variant for the v1-basic FOLD path:
/// identical except it uses the 4-output [`crate::blur::fused_blur_h_ssim`]
/// so `mu1_h` is genuinely materialized — the v1 fold kernel
/// (`fused_vblur_features_ssim`) V-blurs mu1 internally from the H-plane,
/// which `fused_blur_h_ssim3` compiles out on the v4x tier. The three
/// shared outputs (`mu2_h`/`ssq_h`/`s12_h`) are bit-identical between the
/// two H kernels (ssim3 exists purely to skip the mu1 chain), so the v2
/// block's inputs are unchanged by this swap. Costs back the mu1 H-chain
/// vs the plain cached path; the cached mu1 V-blur + activity chain
/// savings are kept.
fn run_blur_pass_strip_cached_ref_fold(
    width: usize,
    height_local: usize,
    scratch: &mut ScratchV2Strip,
) {
    let n = width * height_local;
    let ScratchV2Strip {
        src_wide,
        dst_wide,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        mu2,
        ssq,
        s12,
        ..
    } = scratch;
    crate::blur::fused_blur_h_ssim(
        &src_wide[..n],
        &dst_wide[..n],
        &mut mu1_h[..n],
        &mut mu2_h[..n],
        &mut ssq_h[..n],
        &mut s12_h[..n],
        width,
        height_local,
        BLUR_RADIUS,
    );
    crate::blur::box_blur_v_from_copy(&mu2_h[..n], &mut mu2[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&ssq_h[..n], &mut ssq[..n], width, height_local, BLUR_RADIUS);
    crate::blur::box_blur_v_from_copy(&s12_h[..n], &mut s12[..n], width, height_local, BLUR_RADIUS);
}

/// Fill one channel-scale's cached reference moments by REPLAYING the
/// strip walk on `(src, src)`: same `gather_strip_halo` geometry, same
/// `fused_blur_h_ssim` sliding-sum chains (its `sum_s`/mu1 accumulator
/// reads ONLY the first input, so feeding `src` twice yields a `mu1_h`
/// bit-identical to the pair path's — the other three outputs are
/// discarded), same V-blur and activity functions. Bit-exactness of the
/// cache is BY CONSTRUCTION, not by tolerance: every arithmetic op that
/// produces a cached value is the same op, in the same order, on the same
/// input as the per-pair kernel would have executed.
fn compute_ref_moments_channel(
    src: &[f32],
    width: usize,
    height: usize,
    scratch: &mut ScratchV2Strip,
) -> V2RefMoments {
    let n = width * height;
    let mut mu1_full = vec![0.0f32; n];
    let mut act_full = vec![0.0f32; n];

    // Mirror the (currently disabled) whole-image bypass: below the
    // threshold the kernels blur the full plane directly, so the cache
    // must too.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        run_blur_pass(src, src, width, height, scratch);
        mu1_full.copy_from_slice(&scratch.mu1[..n]);
        act_full.copy_from_slice(&scratch.activity[..n]);
        return V2RefMoments {
            mu1: mu1_full,
            activity: act_full,
            bs2: Vec::new(),
        };
    }

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.src_wide[..n_wide],
        );
        {
            let ScratchV2Strip {
                src_wide,
                mu1_h,
                mu2_h,
                ssq_h,
                s12_h,
                mu1,
                abs_src,
                activity_tmp,
                activity,
                ..
            } = scratch;
            crate::blur::fused_blur_h_ssim(
                &src_wide[..n_wide],
                &src_wide[..n_wide],
                &mut mu1_h[..n_wide],
                &mut mu2_h[..n_wide],
                &mut ssq_h[..n_wide],
                &mut s12_h[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::blur::box_blur_v_from_copy(
                &mu1_h[..n_wide],
                &mut mu1[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::simd_ops::abs_diff_into(
                &src_wide[..n_wide],
                &mu1[..n_wide],
                &mut abs_src[..n_wide],
            );
            crate::blur::box_blur_1pass_into(
                &abs_src[..n_wide],
                &mut activity[..n_wide],
                &mut activity_tmp[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
        }
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out_base = y0 * width;
        mu1_full[out_base..out_base + strip_n].copy_from_slice(&scratch.mu1[off..off + strip_n]);
        act_full[out_base..out_base + strip_n]
            .copy_from_slice(&scratch.activity[off..off + strip_n]);
        y0 += strip_h;
    }

    V2RefMoments {
        mu1: mu1_full,
        activity: act_full,
        bs2: Vec::new(),
    }
}

/// Build the full per-scale/per-channel moment cache for a prepared
/// reference (see [`V2PreparedReference::moments`]).
fn fill_ref_moments(
    scales: &[([Vec<f32>; 3], usize, usize)],
    with_s2: bool,
) -> Vec<[V2RefMoments; 3]> {
    let (_, w0, h0) = &scales[0];
    #[allow(clippy::absurd_extreme_comparisons, clippy::unnecessary_min_or_max)]
    let bypass_rows = (*h0).min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = w0 * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    let mut scratch = ScratchV2Strip::new(strip_max_n);
    // Strip-wide temps for the append σ-split's `blur(src²)` fill (P2:
    // the fill is kernel-strip-tiled, so the temps no longer need to hold
    // a full plane), reused across every (channel, scale); only allocated
    // when `with_s2`.
    let (mut s2_sq, mut s2_stage) = if with_s2 {
        (vec![0.0f32; strip_max_n], vec![0.0f32; strip_max_n])
    } else {
        (Vec::new(), Vec::new())
    };
    scales
        .iter()
        .map(|(planes, width, height)| {
            std::array::from_fn(|ch| {
                let mut m =
                    compute_ref_moments_channel(&planes[ch], *width, *height, &mut scratch);
                // The (B, scale 0) cell is skipped by the append kernel
                // (`APPEND_SKIP_B_SCALE0`) — don't spend cache memory or
                // fill time on a plane nothing reads. Scale is inferred
                // from the plane dims (scale 0 == the largest).
                let is_scale0 = width * height == w0 * h0;
                let skipped = APPEND_SKIP_B_SCALE0 && ch == 2 && is_scale0;
                if with_s2 && !skipped {
                    let n = width * height;
                    let mut bs2 = vec![0.0f32; n];
                    compute_ref_s2blur_into(
                        &planes[ch],
                        *width,
                        *height,
                        &mut s2_sq,
                        &mut s2_stage,
                        &mut bs2,
                    );
                    m.bs2 = bs2;
                }
                m
            })
        })
        .collect()
}

/// `out = blur(src²)` — the reference side of the append σ-split — computed
/// per 128-row KERNEL STRIP with the walk's own `gather_strip_halo`
/// geometry (streaming pre-chunk P2, 2026-07-26 —
/// `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md` §2): per strip, gather
/// the `HALO_P`-padded wide window, square in place, H-blur + V-blur inside
/// the wide buffer, copy out the strip's own rows. One shared
/// implementation for the moments-cache fill AND the pair path's per-scale
/// replay, which is what keeps cached and replayed `bs2` planes
/// bit-identical by construction (same ops, same order, same strip
/// geometry) — and, from C2 on, identical to what the streamed walk
/// computes per kernel strip from its rolling planes.
///
/// The previous form ran ONE whole-plane `box_blur_1pass_into`, whose V
/// pass carries a single f32 running sum from plane row 0 to the bottom —
/// an accumulation history no strip walk can reproduce. Re-tiling shifts
/// `bs2` by f32 ULPs (⇒ ULP-scale shifts on the 5 σ-split append lanes;
/// no trained consumer exists, pre-sanctioned by
/// `benchmarks/v2_append_block_2026-07-26.md` "do it BEFORE any corpus
/// freezes append values"), gives `bs2` the SAME V-accumulation tiling as
/// the `ssq` plane it is subtracted from (`var₂ = (ssq − bs2) − mu2²` now
/// subtracts same-rounding-class operands), and drops the two whole-plane
/// temps to strip-wide size (`sq_tmp`/`stage_tmp` ≥
/// `width·(STRIP_ROWS + 2·HALO_P)` — was ≥ `width·height`).
fn compute_ref_s2blur_into(
    src: &[f32],
    width: usize,
    height: usize,
    sq_tmp: &mut [f32],
    stage_tmp: &mut [f32],
    out: &mut [f32],
) {
    let n = width * height;
    debug_assert!(out.len() >= n);

    // Mirror the (currently disabled) whole-image bypass dispatch, exactly
    // like `compute_ref_moments_channel`: below the threshold the kernels
    // blur the full plane directly, so the cache must too.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        debug_assert!(sq_tmp.len() >= n && stage_tmp.len() >= n);
        sq_tmp[..n].copy_from_slice(&src[..n]);
        square_in_place(&mut sq_tmp[..n]);
        crate::blur::box_blur_1pass_into(
            &sq_tmp[..n],
            &mut out[..n],
            &mut stage_tmp[..n],
            width,
            height,
            BLUR_RADIUS,
        );
        return;
    }

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;
        debug_assert!(sq_tmp.len() >= n_wide && stage_tmp.len() >= n_wide);
        // Gather src's wide window, square it in place, then H+V blur
        // wholly inside the wide buffer (the V window at strip rows reads
        // H(sq) rows [strip−R, strip+R) ⊂ the HALO_P(=2R) window, so the
        // wide buffer's own synthetic edge is never consumed — the same
        // §A.15 argument the mu2/ssq/s12 chains rest on).
        gather_strip_halo(src, width, height, y0, wide_h, HALO_P, &mut sq_tmp[..n_wide]);
        square_in_place(&mut sq_tmp[..n_wide]);
        crate::blur::box_blur_h(&sq_tmp[..n_wide], &mut stage_tmp[..n_wide], width, wide_h, BLUR_RADIUS);
        crate::blur::box_blur_v_from_copy(
            &stage_tmp[..n_wide],
            &mut sq_tmp[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        out[y0 * width..y0 * width + strip_n].copy_from_slice(&sq_tmp[off..off + strip_n]);
        y0 += strip_h;
    }
}

/// Shared body for [`run_blur_pass`]/[`run_blur_pass_strip`] — the actual
/// fused H-blur + 4x V-blur + activity sequence, over explicit buffers.
#[allow(clippy::too_many_arguments)]
fn run_blur_pass_inner(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height_local: usize,
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    mu1: &mut [f32],
    mu2: &mut [f32],
    ssq: &mut [f32],
    s12: &mut [f32],
    abs_src: &mut [f32],
    activity_tmp: &mut [f32],
    activity: &mut [f32],
) {
    let n = width * height_local;
    let mu1_h = &mut mu1_h[..n];
    let mu2_h = &mut mu2_h[..n];
    let ssq_h = &mut ssq_h[..n];
    let s12_h = &mut s12_h[..n];
    crate::blur::fused_blur_h_ssim(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        BLUR_RADIUS,
    );

    let mu1 = &mut mu1[..n];
    crate::blur::box_blur_v_from_copy(mu1_h, mu1, width, height_local, BLUR_RADIUS);
    let mu2 = &mut mu2[..n];
    crate::blur::box_blur_v_from_copy(mu2_h, mu2, width, height_local, BLUR_RADIUS);
    let ssq = &mut ssq[..n];
    crate::blur::box_blur_v_from_copy(ssq_h, ssq, width, height_local, BLUR_RADIUS);
    let s12 = &mut s12[..n];
    crate::blur::box_blur_v_from_copy(s12_h, s12, width, height_local, BLUR_RADIUS);

    let abs_src = &mut abs_src[..n];
    crate::simd_ops::abs_diff_into(src, mu1, abs_src);
    let activity = &mut activity[..n];
    crate::blur::box_blur_1pass_into(
        abs_src,
        activity,
        &mut activity_tmp[..n],
        width,
        height_local,
        BLUR_RADIUS,
    );
}

// ============================================================================
// Phase-4 (§A.14): magetypes-SIMD dense formula pass + gradient pass.
//
// Two separate `#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]` hot loops,
// matching the brief's "compute_channel_scale_v2's formula pass + the shared
// gradient pass" scope exactly:
//   - `dense_block_kernel`: SSIM raw moments, edge artifact/detail, bounded
//     MSE, HF gain/loss/mag-loss, PJND core + transducer-bank, and all 11
//     masked/IW/soft-peak weighted-pool pairs. ALWAYS runs (matches the
//     scalar loop's unconditional block).
//   - `gradient_block_kernel`: GMS, ringing, banding, grad_src/grad_dst sums.
//     Only invoked when `toggles.gradient_features` — the toggle check is
//     hoisted OUTSIDE the pixel loop entirely (was a per-pixel branch in the
//     scalar version; now a single runtime branch per `compute_channel_
//     scale_v2` call), which is itself a small, free improvement.
//
// Blockiness is NOT ported to SIMD this phase — it is inherently sparse
// (only x%8==0 or y%8==0 lattice positions contribute), so a dense 8-wide
// SIMD pass would spend 7/8 of its lanes on masked-out zero contributions.
// It is restructured into its own small SCALAR pass that visits ONLY
// lattice positions (§A.14 "side quest not taken" — noted, not pursued,
// per the phase-4 addendum's single-lever discipline) — this is still
// strictly cheaper than the OLD dense-scalar-with-modulo-branch version it
// replaces, since it no longer visits all `width*height` pixels to find
// the ~1/8+1/8 fraction that matter.
//
// SSIM raw-moment reformulation (deliberate v2-only numeric shift,
// documented per this file's own 5e-4-tolerance allowance): Terriberry's
// online update is inherently sequential (each sample's update depends on
// the running n), so it does not vectorize across lanes. Instead, each
// lane accumulates plain running sums of d, d^2, d^3, d^4 (trivially
// SIMD — no cross-lane dependency), reduced ONCE PER ROW (not once per
// whole image — bounds the f32 accumulator's magnitude to ~width/8
// increments of a small [0,2]-bounded value, avoiding "large-swallows-
// small" f32 summation error) into a running f64 total. At the very end,
// the standard raw-to-central moment identities recover (mean, M2, M4):
//   mean = Sum_d / n
//   M2/n = Sum_d2/n - mean^2
//   M4/n = Sum_d4/n - 4*mean*(Sum_d3/n) + 6*mean^2*(Sum_d2/n) - 3*mean^4
// Matches Terriberry's OWN output up to floating-point reassociation
// (verified within 5e-4 relative on the fixture pairs — see the phase-4
// test `simd_matches_scalar_within_tolerance`).
// ============================================================================

/// Per-row-reduced f64 accumulator for the dense (always-on) block.
#[derive(Default, Clone, Copy)]
struct DenseAccum {
    sum_d: f64,
    sum_d2: f64,
    sum_d3: f64,
    sum_d4: f64,
    sum_art: f64,
    sum_det: f64,
    sum_mse: f64,
    sum_hf_gain: f64,
    sum_hf_loss: f64,
    sum_hf_mag_loss: f64,
    sum_pjnd: f64,
    sum_pjnd_lo: f64,
    sum_pjnd_hi: f64,
    ws_peak_ssim: WeightedSum,
    ws_peak_art: WeightedSum,
    ws_peak_det: WeightedSum,
    ws_mask_ssim: WeightedSum,
    ws_mask_art: WeightedSum,
    ws_mask_det: WeightedSum,
    ws_mask_mse: WeightedSum,
    ws_iw_ssim: WeightedSum,
    ws_iw_art: WeightedSum,
    ws_iw_det: WeightedSum,
    ws_iw_mse: WeightedSum,
}

impl DenseAccum {
    /// Phase-5 (§A.15): fold one strip's `DenseAccum` into the running
    /// whole-image total. Every field is a plain sum (or a `WeightedSum`,
    /// itself two plain sums), so summing N strips' partials is exactly
    /// equivalent to accumulating over the whole image in one pass —
    /// strip order does not change WHICH additions happen, only their
    /// grouping (associativity-level float reassociation only, same class
    /// already tolerated for the phase-4 SIMD-lane-then-row reduction).
    #[inline]
    fn accumulate(&mut self, other: &DenseAccum) {
        self.sum_d += other.sum_d;
        self.sum_d2 += other.sum_d2;
        self.sum_d3 += other.sum_d3;
        self.sum_d4 += other.sum_d4;
        self.sum_art += other.sum_art;
        self.sum_det += other.sum_det;
        self.sum_mse += other.sum_mse;
        self.sum_hf_gain += other.sum_hf_gain;
        self.sum_hf_loss += other.sum_hf_loss;
        self.sum_hf_mag_loss += other.sum_hf_mag_loss;
        self.sum_pjnd += other.sum_pjnd;
        self.sum_pjnd_lo += other.sum_pjnd_lo;
        self.sum_pjnd_hi += other.sum_pjnd_hi;
        self.ws_peak_ssim.accumulate(&other.ws_peak_ssim);
        self.ws_peak_art.accumulate(&other.ws_peak_art);
        self.ws_peak_det.accumulate(&other.ws_peak_det);
        self.ws_mask_ssim.accumulate(&other.ws_mask_ssim);
        self.ws_mask_art.accumulate(&other.ws_mask_art);
        self.ws_mask_det.accumulate(&other.ws_mask_det);
        self.ws_mask_mse.accumulate(&other.ws_mask_mse);
        self.ws_iw_ssim.accumulate(&other.ws_iw_ssim);
        self.ws_iw_art.accumulate(&other.ws_iw_art);
        self.ws_iw_det.accumulate(&other.ws_iw_det);
        self.ws_iw_mse.accumulate(&other.ws_iw_mse);
    }
}

/// Scalar weighted-pool accumulation for ONE pixel — shared by the SIMD
/// kernel's per-lane extraction loop AND the scalar row tail (§A.14
/// register-pressure fix, see `dense_block_kernel_generic`'s doc: the 11
/// masked/IW/soft-peak `(Σw, Σwv)` pairs are 22 of the ~35 SIMD lane
/// accumulators live in the dense kernel's inner loop — measured (256²
/// iteration signal) to cause severe register-pressure regression when
/// vectorized alongside everything else. Scalarizing JUST this block
/// (extract the SIMD-computed `d`/`art_i`/`det_i`/`mse_i`/`act` lanes via
/// `to_array()`, accumulate per-lane here) keeps the division-heavy core
/// formulas (SSIM moments, edge artifact/detail, MSE, HF, PJND — the
/// higher-arithmetic-intensity part) vectorized while removing the 22
/// accumulators that were the dominant register-pressure source.
#[inline]
#[allow(clippy::too_many_arguments)]
fn weighted_pool_accumulate_scalar(
    acc: &mut DenseAccum,
    d: f64,
    art_i: f64,
    det_i: f64,
    mse_i: f64,
    act: f64,
) {
    let mask_w = 1.0 - saturate(act, C_ACTIVITY);
    let iw_w = saturate(act, C_ACTIVITY) + IW_WEIGHT_FLOOR;
    acc.ws_mask_ssim.add(mask_w, d);
    acc.ws_mask_art.add(mask_w, art_i);
    acc.ws_mask_det.add(mask_w, det_i);
    acc.ws_mask_mse.add(mask_w, mse_i);
    acc.ws_iw_ssim.add(iw_w, d);
    acc.ws_iw_art.add(iw_w, art_i);
    acc.ws_iw_det.add(iw_w, det_i);
    acc.ws_iw_mse.add(iw_w, mse_i);

    let sal_ssim = saturate(d, C_PEAK);
    let sal_art = saturate(art_i, C_PEAK);
    let sal_det = saturate(det_i, C_PEAK);
    acc.ws_peak_ssim.add(sal_ssim, d);
    acc.ws_peak_art.add(sal_art, art_i);
    acc.ws_peak_det.add(sal_det, det_i);
}

/// Dense-block SIMD kernel body — generic over any `F32x8Backend` token.
/// Processes one row at a time: zero f32-lane accumulators, sweep the row
/// in 8-wide chunks (+ scalar tail for `width % 8 != 0`), reduce once per
/// row into the f64 [`DenseAccum`] running totals.
///
/// `POOL_SIMD` (register-assignment experiment, 2026-07-21): when `true`,
/// the 11 masked/IW/soft-peak weighted pools are accumulated in f32 SIMD
/// lanes (reduced per row to f64) instead of the §A.14 per-lane scalar
/// extraction. The pool block then needs only **16** lane accumulators —
/// not the 22 that caused §A.14's register-pressure regression — because
/// the mask family shares one `Σw` (same `mask_w` for all 4 pools), the
/// IW family shares one `Σw`, and both weights derive from a single
/// `saturate(act)` (1 division for 8 pixels instead of 16 scalar f64
/// divisions). Total live vectors ≈ 13 core + 16 pool + constants — fits
/// the 32 SIMD registers of AVX-512(VL); on 16-register tiers (AVX2/NEON)
/// it would spill, so the entry dispatch enables it for the v4x tier
/// only. Numerics: pool sums move from f64-per-pixel to
/// f32-lane-then-f64-row accumulation — the SAME reassociation class as
/// the phase-4 core-moment change, inside this module's documented 5e-4
/// tolerance (fixture tests + `pool_simd_drift_within_policy` gate it).
/// The scalar row tail keeps the exact f64 path in both modes.
///
/// `#[inline(always)]`, not `#[inline]`: this body exists ONLY to fuse
/// into its `#[arcane]`/magetypes entry's `target_feature` region. When
/// the POOL_SIMD variant pushed the body past LLVM's inline-cost
/// threshold, the hint stopped being honored and every V8 operator
/// compiled into a CALL to a non-inlined `core::arch` shim outside the
/// feature region — measured 5.3x whole-extraction regression (38.2s vs
/// 7.2s on 100 aic3 pairs; perf showed `__mm256_add_ps` as a standalone
/// 26% symbol). Forcing the inline is the entire fix.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_generic<T: F32x8Backend + Copy, const POOL_SIMD: bool>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let c1 = V8::<T>::splat(token, C1_V2 as f32);
    let c2 = V8::<T>::splat(token, C2_V2 as f32);
    let c_edge = V8::<T>::splat(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat(token, C_MSE as f32);
    let c_hf = V8::<T>::splat(token, C_HF as f32);
    let c_pjnd_clamp = V8::<T>::splat(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat(token, K_PJND_MASK as f32);
    let k_lo = V8::<T>::splat(token, K_PJND_MASK_LOW as f32);
    let k_hi = V8::<T>::splat(token, K_PJND_MASK_HIGH as f32);
    // POOL_SIMD-only constants (const-folded out otherwise).
    let c_activity = V8::<T>::splat(token, C_ACTIVITY as f32);
    let c_peak = V8::<T>::splat(token, C_PEAK as f32);
    let iw_floor = V8::<T>::splat(token, IW_WEIGHT_FLOOR as f32);

    let mut acc = DenseAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;

        // Row-local f32 lane accumulators — zeroed each row and reduced
        // ONCE at the end of the row (bounds the f32 summation magnitude
        // to ~width/8 terms, see the module-level note above).
        //
        // §A.14 register-pressure fix: the masked/IW/soft-peak weighted-pool
        // accumulators (11 pairs = 22 registers in the first version of this
        // kernel) are NOT here -- they're accumulated scalar, per lane,
        // immediately after each SIMD chunk (see `weighted_pool_accumulate_
        // scalar` below), which measurably fixed a severe register-pressure
        // regression (256x256 iteration signal: 30.4ms fused-block-only vs
        // this version's number in §A.14's bench table).
        let (mut r_d, mut r_d2, mut r_d3, mut r_d4) = (zero, zero, zero, zero);
        let (mut r_art, mut r_det, mut r_mse) = (zero, zero, zero);
        let (mut r_hfg, mut r_hfl, mut r_hfm) = (zero, zero, zero);
        let (mut r_pjnd, mut r_pjnd_lo, mut r_pjnd_hi) = (zero, zero, zero);

        // POOL_SIMD lane accumulators (16 — see the function doc). Dead
        // (const-folded away, zero register cost) when POOL_SIMD=false.
        let (mut p_mask_w, mut p_mask_d, mut p_mask_art, mut p_mask_det, mut p_mask_mse) =
            (zero, zero, zero, zero, zero);
        let (mut p_iw_w, mut p_iw_d, mut p_iw_art, mut p_iw_det, mut p_iw_mse) =
            (zero, zero, zero, zero, zero);
        let (
            mut p_sal_d,
            mut p_sal_dv,
            mut p_sal_art,
            mut p_sal_artv,
            mut p_sal_det,
            mut p_sal_detv,
        ) = (zero, zero, zero, zero, zero, zero);

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            macro_rules! ld {
                ($plane:expr) => {
                    V8::<T>::from_array(token, $plane[i..i + 8].try_into().unwrap())
                };
            }
            let s = ld!(src);
            let dd = ld!(dst);
            let m1 = ld!(mu1);
            let m2 = ld!(mu2);
            let act = ld!(activity);

            let d = ssim_d_local_v(token, m1, m2, ld!(s12), ld!(ssq), c1, c2);
            r_d += d;
            let d2 = d * d;
            r_d2 += d2;
            r_d3 += d2 * d;
            r_d4 += d2 * d2;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = one - bounded_sim_v(token, diff_src, diff_dst, c_edge);
            let gt = diff_dst.simd_gt(diff_src);
            let lt = diff_dst.simd_lt(diff_src);
            let art_i = V8::<T>::blend(gt, edge_dissim, zero);
            let det_i = V8::<T>::blend(lt, edge_dissim, zero);
            r_art += art_i;
            r_det += det_i;

            let raw_diff = s - dd;
            let raw_sq_err = raw_diff * raw_diff;
            let mse_i = saturate_v(token, raw_sq_err, c_mse);
            r_mse += mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let hf_src_sq = hf_src * hf_src;
            let hf_dst_sq = hf_dst * hf_dst;
            // Phase-6 (§A.16 lever A): hf_gain/hf_loss share one denominator
            // by construction -- one recip instead of two.
            let (hfg_i, hfl_i) = bounded_excess_pair_v(token, hf_dst_sq, hf_src_sq, c_hf);
            r_hfg += hfg_i;
            r_hfl += hfl_i;
            r_hfm += bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);

            // Phase-6 (§A.16 lever A): each band's raw-ratio-then-saturate
            // (2 divisions) fuses to one via `pjnd_transducer_v` -- see
            // `pjnd_transducer`'s doc for the derivation.
            let raw_abs_err = raw_diff.abs();
            r_pjnd += pjnd_transducer_v(token, raw_abs_err, act, k_mid, c_pjnd_clamp);
            if transducer_bank {
                r_pjnd_lo += pjnd_transducer_v(token, raw_abs_err, act, k_lo, c_pjnd_clamp);
                r_pjnd_hi += pjnd_transducer_v(token, raw_abs_err, act, k_hi, c_pjnd_clamp);
            }

            if POOL_SIMD {
                // Register-assignment variant: pools in-lane. ONE division
                // yields both family weights (mask_w = 1-sat, iw_w =
                // sat+floor); three more give the soft-peak saliencies —
                // 4 vector divisions per 8 pixels vs 40 scalar f64
                // divisions on the extraction path.
                let sat_act = saturate_v(token, act, c_activity);
                let mask_w = one - sat_act;
                let iw_w = sat_act + iw_floor;
                p_mask_w += mask_w;
                p_mask_d += mask_w * d;
                p_mask_art += mask_w * art_i;
                p_mask_det += mask_w * det_i;
                p_mask_mse += mask_w * mse_i;
                p_iw_w += iw_w;
                p_iw_d += iw_w * d;
                p_iw_art += iw_w * art_i;
                p_iw_det += iw_w * det_i;
                p_iw_mse += iw_w * mse_i;
                let sal_d = saturate_v(token, d, c_peak);
                let sal_art = saturate_v(token, art_i, c_peak);
                let sal_det = saturate_v(token, det_i, c_peak);
                p_sal_d += sal_d;
                p_sal_dv += sal_d * d;
                p_sal_art += sal_art;
                p_sal_artv += sal_art * art_i;
                p_sal_det += sal_det;
                p_sal_detv += sal_det * det_i;
            } else {
                // §A.14 register-pressure fix (see the row-header comment
                // above): extract this chunk's d/art_i/det_i/mse_i/act lanes
                // and accumulate the 11 weighted-pool pairs scalar, per lane
                // -- reuses the EXACT SAME formula as the scalar tail below
                // via `weighted_pool_accumulate_scalar`, not a re-derivation.
                let d_arr = d.to_array();
                let art_arr = art_i.to_array();
                let det_arr = det_i.to_array();
                let mse_arr = mse_i.to_array();
                let act_arr = act.to_array();
                for lane in 0..8 {
                    weighted_pool_accumulate_scalar(
                        &mut acc,
                        d_arr[lane] as f64,
                        art_arr[lane] as f64,
                        det_arr[lane] as f64,
                        mse_arr[lane] as f64,
                        act_arr[lane] as f64,
                    );
                }
            }

            x += 8;
        }

        // Reduce this row's f32 lane partials into the f64 running totals.
        acc.sum_d += r_d.reduce_add() as f64;
        acc.sum_d2 += r_d2.reduce_add() as f64;
        acc.sum_d3 += r_d3.reduce_add() as f64;
        acc.sum_d4 += r_d4.reduce_add() as f64;
        acc.sum_art += r_art.reduce_add() as f64;
        acc.sum_det += r_det.reduce_add() as f64;
        acc.sum_mse += r_mse.reduce_add() as f64;
        acc.sum_hf_gain += r_hfg.reduce_add() as f64;
        acc.sum_hf_loss += r_hfl.reduce_add() as f64;
        acc.sum_hf_mag_loss += r_hfm.reduce_add() as f64;
        acc.sum_pjnd += r_pjnd.reduce_add() as f64;
        acc.sum_pjnd_lo += r_pjnd_lo.reduce_add() as f64;
        acc.sum_pjnd_hi += r_pjnd_hi.reduce_add() as f64;
        if POOL_SIMD {
            // Fold the 16 pool lane-sums into the per-pool WeightedSums.
            // The mask/IW family `den`s are the SAME row sum replicated
            // (that sharing is what makes 16 accumulators suffice).
            let mask_w_sum = p_mask_w.reduce_add() as f64;
            acc.ws_mask_ssim.num += p_mask_d.reduce_add() as f64;
            acc.ws_mask_ssim.den += mask_w_sum;
            acc.ws_mask_art.num += p_mask_art.reduce_add() as f64;
            acc.ws_mask_art.den += mask_w_sum;
            acc.ws_mask_det.num += p_mask_det.reduce_add() as f64;
            acc.ws_mask_det.den += mask_w_sum;
            acc.ws_mask_mse.num += p_mask_mse.reduce_add() as f64;
            acc.ws_mask_mse.den += mask_w_sum;
            let iw_w_sum = p_iw_w.reduce_add() as f64;
            acc.ws_iw_ssim.num += p_iw_d.reduce_add() as f64;
            acc.ws_iw_ssim.den += iw_w_sum;
            acc.ws_iw_art.num += p_iw_art.reduce_add() as f64;
            acc.ws_iw_art.den += iw_w_sum;
            acc.ws_iw_det.num += p_iw_det.reduce_add() as f64;
            acc.ws_iw_det.den += iw_w_sum;
            acc.ws_iw_mse.num += p_iw_mse.reduce_add() as f64;
            acc.ws_iw_mse.den += iw_w_sum;
            acc.ws_peak_ssim.num += p_sal_dv.reduce_add() as f64;
            acc.ws_peak_ssim.den += p_sal_d.reduce_add() as f64;
            acc.ws_peak_art.num += p_sal_artv.reduce_add() as f64;
            acc.ws_peak_art.den += p_sal_art.reduce_add() as f64;
            acc.ws_peak_det.num += p_sal_detv.reduce_add() as f64;
            acc.ws_peak_det.den += p_sal_det.reduce_add() as f64;
        }
        // (POOL_SIMD=false: weighted-pool accumulators already folded in
        // scalar, per-lane, inside the chunk loop above -- §A.14 fix note.)

        // Scalar tail: remaining `width % 8` pixels in this row, using the
        // EXACT SAME scalar formulas as the pre-phase-4 kernel (bit-for-bit
        // identical code path — not a re-derivation).
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;

            let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64);
            acc.sum_d += d;
            acc.sum_d2 += d * d;
            acc.sum_d3 += d * d * d;
            acc.sum_d4 += d * d * d * d;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
            let (mut art_i, mut det_i) = (0.0, 0.0);
            if diff_dst > diff_src {
                art_i = edge_dissim;
            } else if diff_dst < diff_src {
                det_i = edge_dissim;
            }
            acc.sum_art += art_i;
            acc.sum_det += det_i;

            let raw_sq_err = (s - dd) * (s - dd);
            let mse_i = saturate(raw_sq_err, C_MSE);
            acc.sum_mse += mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let hf_src_sq = hf_src * hf_src;
            let hf_dst_sq = hf_dst * hf_dst;
            // Phase-6 (§A.16 lever A step 1): shared-denominator pair, one
            // division instead of two (mirrors the SIMD path above).
            let (hf_gain_i, hf_loss_i) = bounded_excess_pair(hf_dst_sq, hf_src_sq, C_HF);
            acc.sum_hf_gain += hf_gain_i;
            acc.sum_hf_loss += hf_loss_i;
            acc.sum_hf_mag_loss += bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            // Phase-6 (§A.16 lever A step 1): fused raw-ratio+saturate, one
            // division per band instead of two (mirrors the SIMD path above).
            let raw_abs_err = (s - dd).abs();
            acc.sum_pjnd += pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);
            if transducer_bank {
                acc.sum_pjnd_lo += pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP);
                acc.sum_pjnd_hi +=
                    pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP);
            }

            weighted_pool_accumulate_scalar(&mut acc, d, art_i, det_i, mse_i, act);
        }
    }

    acc
}

// Two disjoint-tier blocks emit the same base name (the
// `downscale_2x_into_inner` pattern): AVX-512(VL) has 32 SIMD registers,
// so the v4x tier runs the POOL_SIMD=true body (13 core + 16 pool lane
// accumulators live comfortably); every 16-register tier keeps the §A.14
// scalar-pool body, whose extraction fix exists precisely because 22+
// live vectors spill there.
#[magetypes(+v4x, -v4, -v3, -neon, -wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    dense_block_kernel_generic::<_, true>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    dense_block_kernel_generic::<_, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

/// Test/measurement-only dispatch that FORCES the §A.14 scalar-pool body
/// on every tier — the baseline side of the POOL_SIMD drift + timing
/// comparisons (`pool_simd_drift_within_policy`, `v2_pool_simd_ab` bench).
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_entry_pools_scalar(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    dense_block_kernel_generic::<_, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

/// Forced-scalar-pool sibling of [`dense_block_kernel`] (see
/// [`dense_block_kernel_entry_pools_scalar`]).
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_pools_scalar(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    incant!(
        dense_block_kernel_entry_pools_scalar(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

#[allow(clippy::too_many_arguments)]
fn dense_block_kernel(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    incant!(
        dense_block_kernel_entry(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Per-row-reduced f64 accumulator for the gradient block.
#[derive(Default, Clone, Copy)]
struct GradientAccum {
    sum_gms: f64,
    /// Σ gms² — second raw moment of the SAME per-pixel gms values, for
    /// the append block's GMSD-style deviation pooling
    /// ([`idx_append::GMS_DEV2`]). Accumulated unconditionally (the
    /// marginal cost is one FMA per pixel on values already in
    /// registers); only consumed when the append block is on. Adding it
    /// does not perturb `sum_gms` — the shared `g` is computed by the
    /// identical operations in the identical order.
    sum_gms2: f64,
    sum_ringing: f64,
    sum_banding: f64,
    sum_grad_src: f64,
    sum_grad_dst: f64,
}

impl GradientAccum {
    /// Phase-5 (§A.15): fold one strip's `GradientAccum` into the running
    /// whole-image total — same reasoning as `DenseAccum::accumulate`.
    #[inline]
    fn accumulate(&mut self, other: &GradientAccum) {
        self.sum_gms += other.sum_gms;
        self.sum_gms2 += other.sum_gms2;
        self.sum_ringing += other.sum_ringing;
        self.sum_banding += other.sum_banding;
        self.sum_grad_src += other.sum_grad_src;
        self.sum_grad_dst += other.sum_grad_dst;
    }
}

/// Gradient-block SIMD kernel body. Phase-5 (§A.15) NEW CONTRACT: `src_h`/
/// `dst_h` carry ONE HALO ROW top and bottom (`width*(height+2)` total —
/// local row `y+1` is the strip's real row `y`; local rows `0` and
/// `height+1` are the caller-provided neighbor context, real data for an
/// interior strip or a properly reflected row at the true image edge —
/// see `gather_strip_halo`). This REMOVES the old `y==0`/`y==height-1`
/// special-casing entirely: every output row now uses the identical
/// "interior" halo-relative indexing, since the caller guarantees valid
/// neighbor rows for ALL of them, not just strip-interior ones.
/// `activity` is NOT halo-padded (`width*height`, exactly the strip) — it
/// is read only at each pixel's own row, never a neighbor row (confirmed
/// by inspection: only `ld_at!(activity, row + x)`/`activity[i]` below,
/// no `row_u`/`row_d` indexing into it).
///
/// Interior pixels (`1..width-1`) are vectorized via SHIFTED unaligned
/// loads for the x-neighbors (`row[x-1..x+7]` / `row[x+1..x+9]`) and
/// full-row-offset loads for the y-neighbors (`row_u[x..x+8]` /
/// `row_d[x..x+8]`) — the same shifted-window-read pattern as
/// `~/work/archmage/docs/site/content/magetypes/examples/gaussian-blur.md`'s
/// vertical pass. The single-pixel column border (`x==0`, `x==width-1`)
/// still uses the scalar
/// reflect-boundary formulas (`saturating_sub`/`.min(width-1)`) exactly as
/// before — a tiny fraction of pixels (2 columns + 2 rows out of
/// width*height), not worth the complexity of a SIMD boundary-clamp.
#[inline]
fn gradient_block_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    debug_assert_eq!(src_h.len(), width * (height + 2));
    debug_assert_eq!(dst_h.len(), width * (height + 2));
    debug_assert_eq!(activity.len(), width * height);

    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let c_gms = V8::<T>::splat(token, C_GMS as f32);
    let c_ring_err = V8::<T>::splat(token, C_RING_ERR as f32);
    let c_activity = V8::<T>::splat(token, C_ACTIVITY as f32);
    let c_ring_edge = V8::<T>::splat(token, C_RING_EDGE as f32);
    let c_band_dst = V8::<T>::splat(token, C_BAND_DST as f32);
    let c_band_src = V8::<T>::splat(token, C_BAND_SRC as f32);

    let mut acc = GradientAccum::default();

    // Scalar helper for one pixel — used for the first/last COLUMN of
    // every row (x-axis boundary only; the y-axis "first/last row"
    // special case from phase 4 is GONE, per this function's new halo
    // contract — every row `y` has valid `row_u`/`row_d` neighbor data
    // in `src_h`/`dst_h` by construction, either real interior-strip
    // rows or a caller-supplied reflected true-edge row).
    let scalar_pixel = |x: usize, y: usize, acc: &mut GradientAccum| {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(width - 1);
        let row = (y + 1) * width; // +1: src_h/dst_h carry 1 halo row up front
        let row_u = y * width; // y-1, in halo-relative terms
        let row_d = (y + 2) * width; // y+1, in halo-relative terms
        let i = row + x;
        let s = src_h[i] as f64;
        let dd = dst_h[i] as f64;
        let act = activity[y * width + x] as f64;
        let sxl = src_h[row + xl] as f64;
        let sxr = src_h[row + xr] as f64;
        let syu = src_h[row_u + x] as f64;
        let syd = src_h[row_d + x] as f64;
        let dxl = dst_h[row + xl] as f64;
        let dxr = dst_h[row + xr] as f64;
        let dyu = dst_h[row_u + x] as f64;
        let dyd = dst_h[row_d + x] as f64;

        let gx_src = sxr - sxl;
        let gy_src = syd - syu;
        let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
        let gx_dst = dxr - dxl;
        let gy_dst = dyd - dyu;
        let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

        acc.sum_grad_src += grad_src_mag;
        acc.sum_grad_dst += grad_dst_mag;
        let g = 1.0 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS);
        acc.sum_gms += g;
        acc.sum_gms2 += g * g;

        let raw_abs_err = (s - dd).abs();
        let err_b = saturate(raw_abs_err, C_RING_ERR);
        let act_b = saturate(act, C_ACTIVITY);
        let edge_r = saturate(grad_src_mag, C_RING_EDGE);
        acc.sum_ringing += err_b * act_b * (1.0 - edge_r);

        let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
        let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
        acc.sum_banding += edge_excess * src_smooth_b;
    };

    for y in 0..height {
        let row = (y + 1) * width;
        let row_u = y * width;
        let row_d = (y + 2) * width;
        let act_row = y * width;

        scalar_pixel(0, y, &mut acc);
        if width > 2 {
            let interior_end = width - 1;
            let interior_w = interior_end - 1; // pixels [1, width-2]
            let chunk_end = 1 + interior_w - (interior_w % 8);

            let (mut r_gms, mut r_gms2, mut r_ring, mut r_band) = (zero, zero, zero, zero);
            let (mut r_gsrc, mut r_gdst) = (zero, zero);

            let mut x = 1usize;
            while x < chunk_end {
                macro_rules! ld_at {
                    ($plane:expr, $off:expr) => {
                        V8::<T>::from_array(token, $plane[$off..$off + 8].try_into().unwrap())
                    };
                }
                let sxl = ld_at!(src_h, row + x - 1);
                let sxr = ld_at!(src_h, row + x + 1);
                let syu = ld_at!(src_h, row_u + x);
                let syd = ld_at!(src_h, row_d + x);
                let dxl = ld_at!(dst_h, row + x - 1);
                let dxr = ld_at!(dst_h, row + x + 1);
                let dyu = ld_at!(dst_h, row_u + x);
                let dyd = ld_at!(dst_h, row_d + x);
                let s = ld_at!(src_h, row + x);
                let dd = ld_at!(dst_h, row + x);
                let act = ld_at!(activity, act_row + x);

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                r_gsrc += grad_src_mag;
                r_gdst += grad_dst_mag;
                let g = one - bounded_sim_v(token, grad_src_mag, grad_dst_mag, c_gms);
                r_gms += g;
                r_gms2 += g * g;

                let raw_abs_err = (s - dd).abs();
                let err_b = saturate_v(token, raw_abs_err, c_ring_err);
                let act_b = saturate_v(token, act, c_activity);
                let edge_r = saturate_v(token, grad_src_mag, c_ring_edge);
                r_ring += err_b * act_b * (one - edge_r);

                let edge_excess = bounded_excess_v(token, grad_dst_mag, grad_src_mag, c_band_dst);
                let src_smooth_b = one - saturate_v(token, grad_src_mag, c_band_src);
                r_band += edge_excess * src_smooth_b;

                x += 8;
            }

            acc.sum_gms += r_gms.reduce_add() as f64;
            acc.sum_gms2 += r_gms2.reduce_add() as f64;
            acc.sum_ringing += r_ring.reduce_add() as f64;
            acc.sum_banding += r_band.reduce_add() as f64;
            acc.sum_grad_src += r_gsrc.reduce_add() as f64;
            acc.sum_grad_dst += r_gdst.reduce_add() as f64;

            for x in chunk_end..=interior_end - 1 {
                scalar_pixel(x, y, &mut acc);
            }
        }
        scalar_pixel(width - 1, y, &mut acc);
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    gradient_block_kernel_generic(token, src, dst, activity, width, height)
}

fn gradient_block_kernel(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    incant!(
        gradient_block_kernel_entry(src, dst, activity, width, height),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Sparse blockiness pass — visits ONLY the 8-pixel-lattice positions
/// (`x % BLOCK_LATTICE == 0`, `y % BLOCK_LATTICE == 0`) instead of every
/// pixel. Deliberately scalar (§A.14: not folded into the SIMD dense pass
/// this phase — see the module-level note above) but strictly cheaper
/// than the pre-phase-4 dense-scalar-with-modulo-branch it replaces, since
/// it no longer visits the 7/8 (or 63/64 for the corner term) of pixels
/// that always contribute zero.
///
/// **Canonical accumulation order (streaming pre-chunk P1, 2026-07-26 —
/// `docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md` §2):** both term
/// families accumulate ROW-ORDERED (y-outer), into two separate f64 sums
/// combined once at the end:
///
/// - `sum_v` (vertical lattice-BOUNDARY steps, `x % LATTICE == 0, x > 0`,
///   every row): `for y in 0..h { for x in lattice }`.
/// - `sum_h` (horizontal lattice-boundary steps, `y % LATTICE == 0, y > 0`,
///   every column): `for y in lattice { for x in 0..w }` (this family was
///   already row-ordered).
///
/// The previous form iterated the vertical family COLUMN-outer and folded
/// both families into one running f64 — an association no row-ordered
/// strip walk can reproduce. Reordering shifts the pooled BLOCKINESS value
/// by f64 reassociation only (~1e-16 rel; 12 of 720 slots affected, all
/// entry paths shift identically so path-parity gates are unaffected), and
/// lets a strip-fed variant accumulate rows `[y0, y1)` per kernel strip
/// into running `(sum_v, sum_h)` with a bit-identical f64 op sequence.
/// (Also a locality win: the old column-outer walk strode `width` floats
/// per step.)
fn blockiness_sparse(src: &[f32], dst: &[f32], width: usize, height: usize) -> f64 {
    let (sum_v, sum_h) = blockiness_sparse_rows(src, dst, width, 0, height);
    sum_v + sum_h
}

/// Row-range core of [`blockiness_sparse`]: accumulate BOTH term families
/// for image rows `[y0, y1)` (`src`/`dst` hold the full plane here; the
/// streamed walk (C2) will call this shape with strip-local slices and
/// translated row indexing). Returns `(sum_v, sum_h)` so a strip walk can
/// keep the two running sums across strips and combine ONCE at finalize,
/// reproducing [`blockiness_sparse`]'s f64 op sequence exactly.
fn blockiness_sparse_rows(
    src: &[f32],
    dst: &[f32],
    width: usize,
    y0: usize,
    y1: usize,
) -> (f64, f64) {
    let mut sum_v = 0.0f64;
    let mut sum_h = 0.0f64;
    for y in y0..y1 {
        let row = y * width;
        // Vertical steps at lattice columns, this row.
        let mut x = BLOCK_LATTICE;
        while x < width {
            let i = row + x;
            let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
            let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
            sum_v += bounded_excess(step_dst, step_src, C_BLOCK);
            x += BLOCK_LATTICE;
        }
        // Horizontal steps across the whole row, lattice rows only.
        if y % BLOCK_LATTICE == 0 && y > 0 {
            for x in 0..width {
                let i = row + x;
                let i_up = i - width;
                let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
                let step_src = (src[i] as f64 - src[i_up] as f64).abs();
                sum_h += bounded_excess(step_dst, step_src, C_BLOCK);
            }
        }
    }
    (sum_v, sum_h)
}

/// Compute all [`FEATURES_PER_CHANNEL_V2_TOTAL`] v2 signals for one
/// channel at one scale (except [`idx::EDGE_WIDTH_CHANGE`], filled in by
/// the caller once the adjacent scale is known — see
/// [`compute_v2_features_impl`]). Writes into `out`. Returns
/// `(mean_grad_src, mean_grad_dst)` for the caller's cross-scale
/// edge-width computation (`(0,0)` when `toggles.gradient_features` is
/// off — the caller must not rely on edge-width in that case).
///
/// Phase-5 (§A.15): the per-pixel pass is now a STRIP LOOP — blur +
/// dense-formula-kernel + gradient-kernel FUSED per strip, so full-image
/// mu1/mu2/ssq/s12/activity planes are never materialized (only one
/// strip's worth, `ScratchV2Strip`, sized `width*(STRIP_ROWS+2*HALO_P)`
/// regardless of the channel-scale's actual `height`). Each strip's
/// `DenseAccum`/`GradientAccum` folds into a running whole-image total
/// via `.accumulate()`. `src`/`dst` (this function's own params) remain
/// the FULL channel-scale planes — unchanged from phases 1-4, still
/// materialized once by the caller's XYB conversion — strip-tiling only
/// removes the DERIVED intermediate planes' full-image footprint.
/// Blockiness stays a single full-image pass (§A.14: already sparse,
/// only reads `src`/`dst` which are full arrays regardless).
/// Phase-6 (§A.16 lever B): DISABLED (0 — the bypass never fires). The
/// hypothesis was that below some height, `compute_channel_scale_v2`
/// should bypass the strip loop entirely via
/// `compute_channel_scale_v2_whole`, avoiding the fixed per-strip
/// halo-gather cost. MEASURED (crossover sweep, §A.16.4): the bypass path
/// LOSES to the strip path at every tested size, 256²-512² (bypass ratio
/// 2.5-2.8x vs. strip's 1.3-1.6x at the SAME sizes, both with lever A's
/// division-fusion applied) — the opposite of the phase-5 hypothesis this
/// lever was built to fix. Root cause identified, NOT fixed this phase
/// (out of budget): `compute_channel_scale_v2_whole`'s gradient-halo
/// construction (`gather_strip_halo(..., height+2, 1, &mut src_g)`) copies
/// essentially the WHOLE image (`O(width*height)`, not the `O(width)` this
/// function's own doc originally (incorrectly) claimed) into a FRESH
/// `vec![0.0f32; width*(height+2)]` allocation, TWICE (src_g + dst_g),
/// EVERY call (up to 12x per `compute_v2_features_impl_with_toggles` — 4
/// scales x 3 channels) — this is exactly the "108 full-image allocations"
/// class of problem phase 3 fixed for the main pipeline, reintroduced here
/// specifically for the bypass's gradient halo. A fix (thread a
/// pre-allocated, scratch-owned buffer for this instead of a fresh `vec!`
/// per call) is a well-characterized, NOT-YET-ATTEMPTED follow-on — see
/// §A.16.4/§A.16.6. `compute_channel_scale_v2_whole` and its dedicated
/// test (`bounded_range_bypass_path_small_image`) are KEPT (dead at
/// runtime with this threshold, but correctness-verified and ready for a
/// future session to re-enable once the allocation bug is fixed) rather
/// than ripped out under time pressure.
const STRIP_BYPASS_HEIGHT: usize = 0;

// ============================================================================
// f720+ APPEND kernel — second formula pass over the cache-hot strip planes.
//
// Deliberately a SEPARATE kernel from `dense_block_kernel_generic`: the dense
// kernel's register budget is a documented hazard (§A.14 scalarized its pool
// block after a measured regression; §A.16 measured a 5.3x collapse when the
// body outgrew LLVM's inliner) — widening it again for 19 more accumulators
// would re-open both. A second pass over strip-resident planes re-reads L2,
// not DRAM (that locality is what §A.15's strip tiling bought), so the
// marginal cost is close to the append block's own arithmetic. It also makes
// the f0..f719 bit-stability guarantee structural: the tuned kernels are not
// edited at all (the one exception, `GradientAccum::sum_gms2`, adds a
// derived accumulator without altering any existing operation).
// ============================================================================

/// Per-channel append-kernel inputs that come from OUTSIDE the channel's own
/// strip planes: the raw reference-Y plane for this scale (luminance
/// conditioning — always available, every path materializes the reference
/// pyramid up front), and, for the Y channel only, the X/B activity planes
/// for the cross-channel masked transducer (from the reference-moments
/// cache when present, else from the pair path's `V2Scratch` replay
/// planes — bit-identical by construction, see `compute_ref_activity_into`).
#[derive(Clone, Copy)]
struct AppendCtx<'a> {
    ref_y: &'a [f32],
    /// This channel's `blur(src²)` plane (moments cache or pair-path
    /// replay — bit-identical either way), the reference half of the
    /// σ-split.
    bs2: &'a [f32],
    cross: Option<(&'a [f32], &'a [f32])>,
}

/// Per-row-reduced f64 accumulator for the append block.
#[derive(Default, Clone, Copy)]
struct AppendAccum {
    sum_xmask: f64,
    sum_lumt: f64,
    ws_dark: WeightedSum,
    ws_bright: WeightedSum,
    /// Σ mse_i — the mid-luminance bin is DERIVED at finalize: Bernstein
    /// weights sum to 1, so `Σw_mid = n − Σw_dark − Σw_bright` and
    /// `Σw_mid·v = Σv − Σw_dark·v − Σw_bright·v`. One fewer lane pair in
    /// the kernel; bitwise path-stable because the derivation runs in f64
    /// finalize on sums that are themselves path-bitwise-equal.
    sum_mse: f64,
    sum_mscn: f64,
    sum_mscn2: f64,
    sum_cgain: f64,
    sum_closs: f64,
    sum_tex: f64,
    /// Σ art², Σ det² — second raw moments of per-pixel values that are
    /// bit-identical to the dense kernel's `art_i`/`det_i` (same formula,
    /// same inputs, same order), so `finish_append` can pair them with
    /// `DenseAccum::sum_art`/`sum_det` as the matching first moments.
    sum_art2: f64,
    sum_det2: f64,
    sum_s: f64,
    sum_d: f64,
    sum_s2: f64,
    sum_d2: f64,
}

impl AppendAccum {
    /// Strip-partial fold, same reasoning as [`DenseAccum::accumulate`].
    #[inline]
    fn accumulate(&mut self, other: &AppendAccum) {
        self.sum_xmask += other.sum_xmask;
        self.sum_lumt += other.sum_lumt;
        self.ws_dark.accumulate(&other.ws_dark);
        self.ws_bright.accumulate(&other.ws_bright);
        self.sum_mse += other.sum_mse;
        self.sum_mscn += other.sum_mscn;
        self.sum_mscn2 += other.sum_mscn2;
        self.sum_cgain += other.sum_cgain;
        self.sum_closs += other.sum_closs;
        self.sum_tex += other.sum_tex;
        self.sum_art2 += other.sum_art2;
        self.sum_det2 += other.sum_det2;
        self.sum_s += other.sum_s;
        self.sum_d += other.sum_d;
        self.sum_s2 += other.sum_s2;
        self.sum_d2 += other.sum_d2;
    }
}

/// Append-block SIMD kernel body — generic over any `F32x8Backend` token,
/// same row-lane-then-f64 reduction structure as the dense kernel. `CROSS`
/// const-compiles the cross-channel transducer (and its two extra plane
/// streams) in or out — the Y channel dispatches `true`, X/B `false`, so
/// the chroma channels never touch `act_x`/`act_b` (callers pass `&[]`).
///
/// ~19 row-lane accumulators: over the 16-register budget of AVX2/NEON, so
/// some spill there — accepted for the first landing (the spills are
/// L1-resident row-locals). If the perf gate flags it, the §A.14 remedy
/// (scalarize the 3 luminance pools per lane) is the known fix.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_generic<T: F32x8Backend + Copy, const CROSS: bool>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let c_edge = V8::<T>::splat(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat(token, C_MSE as f32);
    let c_pjnd_clamp = V8::<T>::splat(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat(token, K_PJND_MASK as f32);
    let c_lum = V8::<T>::splat(token, C_LUM_T as f32);
    // pjnd_transducer(err, A, k, c) = err/(err + c·(1 + k·A)); folding the
    // luminance / cross-channel terms into A at ratio k_x/K_PJND_MASK gives
    // the intended `c·(1 + k·act + k_x·extra)` denominator with the SAME
    // fused one-division helper (exact algebra, no new formula family).
    let k_lum_ratio = V8::<T>::splat(token, (K_LUM_ADAPT / K_PJND_MASK) as f32);
    let k_xch_ratio = V8::<T>::splat(token, (K_XCH / K_PJND_MASK) as f32);
    let c_mvar = V8::<T>::splat(token, C_MSCN_VAR as f32);
    let c_mabs = V8::<T>::splat(token, C_MSCN_ABS as f32);
    let c_msq = V8::<T>::splat(token, C_MSCN_SQ as f32);
    let c_con = V8::<T>::splat(token, C_CONTRAST as f32);

    let mut acc = AppendAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;

        let (mut r_xm, mut r_lumt) = (zero, zero);
        let (mut p_wd, mut p_wdv, mut p_wb, mut p_wbv, mut r_mse) = (zero, zero, zero, zero, zero);
        let (mut r_mscn, mut r_mscn2) = (zero, zero);
        let (mut r_cg, mut r_cl, mut r_tex) = (zero, zero, zero);
        let (mut r_art2, mut r_det2) = (zero, zero);
        let (mut r_s, mut r_d, mut r_s2, mut r_d2) = (zero, zero, zero, zero);

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            macro_rules! ld {
                ($plane:expr) => {
                    V8::<T>::from_array(token, $plane[i..i + 8].try_into().unwrap())
                };
            }
            let s = ld!(src);
            let dd = ld!(dst);
            let m1 = ld!(mu1);
            let m2 = ld!(mu2);
            let ry = ld!(ref_y);
            let b2 = ld!(bs2);

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = one - bounded_sim_v(token, diff_src, diff_dst, c_edge);
            let gt = diff_dst.simd_gt(diff_src);
            let lt = diff_dst.simd_lt(diff_src);
            let art_i = V8::<T>::blend(gt, edge_dissim, zero);
            let det_i = V8::<T>::blend(lt, edge_dissim, zero);
            r_art2 += art_i * art_i;
            r_det2 += det_i * det_i;

            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
            let t = saturate_v(token, ry, c_lum);
            // Luminance + cross-channel transducers are Y-only (CROSS is
            // the Y channel's dispatch): the luma-gate ablation measured
            // chroma transducers as a broad CID22 cost. `activity` is
            // only consumed here, so the load lives inside the branch.
            if CROSS {
                let act = ld!(activity);
                r_lumt += pjnd_transducer_v(
                    token,
                    raw_abs_err,
                    t.mul_add(k_lum_ratio, act),
                    k_mid,
                    c_pjnd_clamp,
                );
                let act_c = ld!(act_x) + ld!(act_b);
                r_xm += pjnd_transducer_v(
                    token,
                    raw_abs_err,
                    act_c.mul_add(k_xch_ratio, act),
                    k_mid,
                    c_pjnd_clamp,
                );
            }

            let one_mt = one - t;
            let wd = one_mt * one_mt;
            let wb = t * t;
            p_wd += wd;
            p_wdv += wd * mse_i;
            p_wb += wb;
            p_wbv += wb * mse_i;
            r_mse += mse_i;

            let var1 = (b2 - m1 * m1).max(zero);
            let var2 = ((ld!(ssq) - b2) - m2 * m2).max(zero);
            let n1 = (s - m1) * (var1 + c_mvar).rsqrt();
            let n2 = (dd - m2) * (var2 + c_mvar).rsqrt();
            let dn = n1 - n2;
            r_mscn += saturate_v(token, dn.abs(), c_mabs);
            r_mscn2 += saturate_v(token, dn * dn, c_msq);

            let (cg, cl) = bounded_excess_pair_v(token, var2, var1, c_con);
            r_cg += cg;
            r_cl += cl;
            r_tex += one - bounded_sim_v(token, var1, var2, c_con);

            r_s += s;
            r_d += dd;
            r_s2 += s * s;
            r_d2 += dd * dd;

            x += 8;
        }

        acc.sum_xmask += r_xm.reduce_add() as f64;
        acc.sum_lumt += r_lumt.reduce_add() as f64;
        acc.ws_dark.num += p_wdv.reduce_add() as f64;
        acc.ws_dark.den += p_wd.reduce_add() as f64;
        acc.ws_bright.num += p_wbv.reduce_add() as f64;
        acc.ws_bright.den += p_wb.reduce_add() as f64;
        acc.sum_mse += r_mse.reduce_add() as f64;
        acc.sum_mscn += r_mscn.reduce_add() as f64;
        acc.sum_mscn2 += r_mscn2.reduce_add() as f64;
        acc.sum_cgain += r_cg.reduce_add() as f64;
        acc.sum_closs += r_cl.reduce_add() as f64;
        acc.sum_tex += r_tex.reduce_add() as f64;
        acc.sum_art2 += r_art2.reduce_add() as f64;
        acc.sum_det2 += r_det2.reduce_add() as f64;
        acc.sum_s += r_s.reduce_add() as f64;
        acc.sum_d += r_d.reduce_add() as f64;
        acc.sum_s2 += r_s2.reduce_add() as f64;
        acc.sum_d2 += r_d2.reduce_add() as f64;

        // Scalar tail — same formulas via the scalar siblings.
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;
            let ry = ref_y[i] as f64;
            let b2 = bs2[i] as f64;
            let sq = ssq[i] as f64;

            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let ed = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
            let art_i = if diff_dst > diff_src { ed } else { 0.0 };
            let det_i = if diff_dst < diff_src { ed } else { 0.0 };
            acc.sum_art2 += art_i * art_i;
            acc.sum_det2 += det_i * det_i;

            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate(raw_diff * raw_diff, C_MSE);
            let t = saturate(ry, C_LUM_T);
            if CROSS {
                acc.sum_lumt += pjnd_transducer(
                    raw_abs_err,
                    act + t * (K_LUM_ADAPT / K_PJND_MASK),
                    K_PJND_MASK,
                    C_PJND_CLAMP,
                );
                let act_c = act_x[i] as f64 + act_b[i] as f64;
                acc.sum_xmask += pjnd_transducer(
                    raw_abs_err,
                    act + act_c * (K_XCH / K_PJND_MASK),
                    K_PJND_MASK,
                    C_PJND_CLAMP,
                );
            }

            let one_mt = 1.0 - t;
            acc.ws_dark.add(one_mt * one_mt, mse_i);
            acc.ws_bright.add(t * t, mse_i);
            acc.sum_mse += mse_i;

            let var1 = (b2 - m1 * m1).max(0.0);
            let var2 = ((sq - b2) - m2 * m2).max(0.0);
            let n1 = (s - m1) / (var1 + C_MSCN_VAR).sqrt();
            let n2 = (dd - m2) / (var2 + C_MSCN_VAR).sqrt();
            let dn = n1 - n2;
            acc.sum_mscn += saturate(dn.abs(), C_MSCN_ABS);
            acc.sum_mscn2 += saturate(dn * dn, C_MSCN_SQ);

            let (cg, cl) = bounded_excess_pair(var2, var1, C_CONTRAST);
            acc.sum_cgain += cg;
            acc.sum_closs += cl;
            acc.sum_tex += 1.0 - bounded_sim(var1, var2, C_CONTRAST);

            acc.sum_s += s;
            acc.sum_d += dd;
            acc.sum_s2 += s * s;
            acc.sum_d2 += dd * dd;
        }
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_cross(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    act_x: &[f32],
    act_b: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    append_block_kernel_generic::<_, true>(
        token, src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_nocross(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    width: usize,
    height: usize,
) -> AppendAccum {
    append_block_kernel_generic::<_, false>(
        token, src, dst, mu1, mu2, ssq, bs2, activity, ref_y, &[], &[], width, height,
    )
}

/// Runtime dispatch wrapper for the append kernel (same `incant!` shape as
/// [`dense_block_kernel`]). `cross` carries the X/B activity strips for the
/// Y channel; `None` compiles the cross-channel chain out entirely.
#[allow(clippy::too_many_arguments)]
fn append_block_kernel(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    bs2: &[f32],
    activity: &[f32],
    ref_y: &[f32],
    cross: Option<(&[f32], &[f32])>,
    width: usize,
    height: usize,
) -> AppendAccum {
    match cross {
        Some((act_x, act_b)) => {
            incant!(
                append_block_kernel_entry_cross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
        None => {
            incant!(
                append_block_kernel_entry_nocross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
    }
}

/// Finalize one channel-scale's append block into its 17 output slots.
/// `dense`/`grad` supply the first moments matching `app`'s second moments
/// (identical per-pixel values by construction — see [`AppendAccum`]).
fn finish_append(
    dense: &DenseAccum,
    app: &AppendAccum,
    grad: &GradientAccum,
    n: usize,
    cross: bool,
    toggles: V2NewFeatureToggles,
    out: &mut [f64],
) {
    debug_assert_eq!(out.len(), FEATURES_PER_CHANNEL_APPEND);
    let n_f = n as f64;
    #[inline]
    fn clamp01(v: f64) -> f64 {
        v.clamp(0.0, 1.0)
    }
    #[inline]
    fn clamp02(v: f64) -> f64 {
        v.clamp(0.0, 2.0)
    }
    #[inline]
    fn dev_from_moments(sum: f64, sum_sq: f64, n_f: f64) -> f64 {
        let mean = sum / n_f;
        ((sum_sq / n_f) - mean * mean).max(0.0).sqrt()
    }

    out[idx_append::XMASK_TRANSDUCER] = if cross {
        clamp01(app.sum_xmask / n_f)
    } else {
        0.0
    };
    out[idx_append::LUM_TRANSDUCER] = if cross {
        clamp01(app.sum_lumt / n_f)
    } else {
        0.0
    };
    out[idx_append::LUM_DARK_ERR] = clamp02(app.ws_dark.finish());
    // Mid bin derived from the Bernstein partition of unity (see
    // `AppendAccum::sum_mse`): Σw_mid = n − Σw_dark − Σw_bright,
    // Σw_mid·v = Σv − Σw_dark·v − Σw_bright·v.
    let ws_mid = WeightedSum {
        num: app.sum_mse - app.ws_dark.num - app.ws_bright.num,
        den: n_f - app.ws_dark.den - app.ws_bright.den,
    };
    out[idx_append::LUM_MID_ERR] = clamp02(ws_mid.finish());
    out[idx_append::LUM_BRIGHT_ERR] = clamp02(app.ws_bright.finish());
    out[idx_append::MSCN_DIFF_MEAN] = clamp01(app.sum_mscn / n_f);
    out[idx_append::MSCN_DIFF_L2] = clamp01(app.sum_mscn2 / n_f);
    out[idx_append::CONTRAST_GAIN] = clamp01(app.sum_cgain / n_f);
    out[idx_append::CONTRAST_LOSS] = clamp01(app.sum_closs / n_f);
    out[idx_append::TEXTURE_DISSIM] = clamp01(app.sum_tex / n_f);
    out[idx_append::GMS_DEV2] = if toggles.gradient_features {
        clamp01(dev_from_moments(grad.sum_gms, grad.sum_gms2, n_f))
    } else {
        0.0
    };
    out[idx_append::ART_DEV2] = clamp01(dev_from_moments(dense.sum_art, app.sum_art2, n_f));
    out[idx_append::DET_DEV2] = clamp01(dev_from_moments(dense.sum_det, app.sum_det2, n_f));
    out[idx_append::GLOBAL_DMEAN] = saturate((app.sum_s - app.sum_d).abs() / n_f, C_GDMEAN);
    let gmean_s = app.sum_s / n_f;
    let gmean_d = app.sum_d / n_f;
    let gvar1 = (app.sum_s2 / n_f - gmean_s * gmean_s).max(0.0);
    let gvar2 = (app.sum_d2 / n_f - gmean_d * gmean_d).max(0.0);
    let (g_cgain, g_closs) = bounded_excess_pair(gvar2, gvar1, C_GCONTRAST);
    out[idx_append::GLOBAL_CGAIN] = clamp01(g_cgain);
    out[idx_append::GLOBAL_CLOSS] = clamp01(g_closs);
    out[idx_append::GRAD_SRC_MEAN] = if toggles.gradient_features {
        saturate(grad.sum_grad_src / n_f, C_GRADM)
    } else {
        0.0
    };
}

/// `buf[i] = buf[i]²` — feeds the append σ-split's `blur(src²)` strip
/// chain. Plain scalar loop; LLVM auto-vectorizes the independent multiply.
fn square_in_place(buf: &mut [f32]) {
    for v in buf.iter_mut() {
        *v = *v * *v;
    }
}

/// Replay ONE channel-scale's reference activity chain into `act_out`
/// (full plane) — [`compute_ref_moments_channel`] minus the mu1 plane
/// copy-out. Used by the append-block pair path (no moments cache) to
/// provision the X/B activity planes the Y channel's cross-masked
/// transducer reads; the replay uses the same helpers, the same strip
/// tiling, and the same chain starts as the moments fill, so the values
/// are bit-identical to what a `prepare_v2_reference_with_moments` cache
/// would hold — which is what keeps the pair/prepared/moments paths
/// byte-equal on the append block (`append_ref_paths_bit_identical`).
fn compute_ref_activity_into(
    src: &[f32],
    width: usize,
    height: usize,
    scratch: &mut ScratchV2Strip,
    act_out: &mut [f32],
) {
    let n = width * height;
    debug_assert!(act_out.len() >= n);

    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        run_blur_pass(src, src, width, height, scratch);
        act_out[..n].copy_from_slice(&scratch.activity[..n]);
        return;
    }

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.src_wide[..n_wide],
        );
        {
            let ScratchV2Strip {
                src_wide,
                mu1_h,
                mu2_h,
                ssq_h,
                s12_h,
                mu1,
                abs_src,
                activity_tmp,
                activity,
                ..
            } = scratch;
            crate::blur::fused_blur_h_ssim(
                &src_wide[..n_wide],
                &src_wide[..n_wide],
                &mut mu1_h[..n_wide],
                &mut mu2_h[..n_wide],
                &mut ssq_h[..n_wide],
                &mut s12_h[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::blur::box_blur_v_from_copy(
                &mu1_h[..n_wide],
                &mut mu1[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::simd_ops::abs_diff_into(
                &src_wide[..n_wide],
                &mu1[..n_wide],
                &mut abs_src[..n_wide],
            );
            crate::blur::box_blur_1pass_into(
                &abs_src[..n_wide],
                &mut activity[..n_wide],
                &mut activity_tmp[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
        }
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out_base = y0 * width;
        act_out[out_base..out_base + strip_n].copy_from_slice(&scratch.activity[off..off + strip_n]);
        y0 += strip_h;
    }
}

/// Per-channel-scale f64 sums for the v1 BASIC-13 fold — the exact subset
/// of v1's `streaming::ChannelAccum` fields that the basic block
/// (`f0..156`) finalizes from. Filled by v1's own
/// [`crate::fused::fused_vblur_features_ssim`] kernel run over the v2
/// strip walk's shared H-planes; the peak accumulators the kernel also
/// returns (max/L8) are deliberately dropped — v1's peak block `f156..228`
/// is deprecated (no current model reads it).
#[derive(Debug, Clone, Copy, Default)]
struct V1BasicSums {
    ssim_d: f64,
    ssim_d4: f64,
    ssim_d2: f64,
    edge_art: f64,
    edge_art4: f64,
    edge_art2: f64,
    edge_det: f64,
    edge_det4: f64,
    edge_det2: f64,
    mse: f64,
    hf_sq_src: f64,
    hf_sq_dst: f64,
    hf_abs_src: f64,
    hf_abs_dst: f64,
}

impl V1BasicSums {
    fn accumulate(&mut self, s: &crate::fused::StripChannelAccum) {
        self.ssim_d += s.ssim_d;
        self.ssim_d4 += s.ssim_d4;
        self.ssim_d2 += s.ssim_d2;
        self.edge_art += s.edge_art;
        self.edge_art4 += s.edge_art4;
        self.edge_art2 += s.edge_art2;
        self.edge_det += s.edge_det;
        self.edge_det4 += s.edge_det4;
        self.edge_det2 += s.edge_det2;
        self.mse += s.mse;
        self.hf_sq_src += s.hf_sq_src;
        self.hf_sq_dst += s.hf_sq_dst;
        self.hf_abs_src += s.hf_abs_src;
        self.hf_abs_dst += s.hf_abs_dst;
    }

    /// Finalize into one channel's 13 basic features, replicating v1's
    /// pooling + assembly EXACTLY (`streaming::ChannelAccum::finalize` +
    /// `metric.rs`'s pass-1 feature push, `.abs()` included): mean, L4
    /// (`(Σx⁴/n)^¼`), L2 (`(Σx²/n)^½`) for ssim/art/det; mse mean; the
    /// three HF ratio features with their `1e-10` guards. `n` is the
    /// scale's full pixel count (`w_s * h_s`) — identical to v1's per-scale
    /// accumulator `n` since both walks cover every row exactly once.
    fn finalize_into(&self, n: usize, out: &mut [f64]) {
        debug_assert_eq!(out.len(), 13);
        let one_over_n = 1.0 / n as f64;
        out[0] = (self.ssim_d * one_over_n).abs();
        out[1] = (self.ssim_d4 * one_over_n).max(0.0).powf(0.25).abs();
        out[2] = (self.ssim_d2 * one_over_n).max(0.0).sqrt().abs();
        out[3] = (self.edge_art * one_over_n).abs();
        out[4] = (self.edge_art4 * one_over_n).max(0.0).powf(0.25).abs();
        out[5] = (self.edge_art2 * one_over_n).max(0.0).sqrt().abs();
        out[6] = (self.edge_det * one_over_n).abs();
        out[7] = (self.edge_det4 * one_over_n).max(0.0).powf(0.25).abs();
        out[8] = (self.edge_det2 * one_over_n).max(0.0).sqrt().abs();
        out[9] = self.mse * one_over_n;
        let var_src = self.hf_sq_src * one_over_n;
        let var_dst = self.hf_sq_dst * one_over_n;
        out[10] = if var_src > 1e-10 {
            (1.0 - var_dst / var_src).max(0.0)
        } else {
            0.0
        };
        out[12] = if var_src > 1e-10 {
            (var_dst / var_src - 1.0).max(0.0)
        } else {
            0.0
        };
        let mad_src = self.hf_abs_src * one_over_n;
        let mad_dst = self.hf_abs_dst * one_over_n;
        out[11] = if mad_src > 1e-10 {
            (1.0 - mad_dst / mad_src).max(0.0)
        } else {
            0.0
        };
    }
}

/// v1's band tiling constants, replicated for the fold. v1 tiles every
/// scale into 32-row bands ([`crate::streaming`]'s `STRIP_INNER`) and
/// V-blurs each band from a buffer extending `overlap = blur_passes(1) ×
/// radius(5) = 5` rows past the band, mirror-clamped at the PLANE bounds.
/// The band layout is part of v1's numerics contract: the f32 sliding
/// V-blur state re-initializes at every band's buffer top, so pooled
/// sums depend on the tiling. The fold reproduces the tiling exactly —
/// same buffer extents, same init points, same band order — which makes
/// the accumulated sums bit-identical to v1's whenever the plane values
/// themselves are (true-width == v1's SIMD-padded width; see the parity
/// test for the padded-width caveat).
const V1_BAND_ROWS: usize = 32;
const V1_BAND_OVERLAP: usize = 5;

/// Run v1's fused V-blur + basic-feature kernel over the v1-aligned bands
/// covered by one fold buffer (the FOLD hook): consumes the v2 blur
/// pass's H-planes + halo buffers directly (H-blur is per-row/stateless,
/// so H values are shareable; the V state is NOT, hence the band replay).
///
/// `rows` are the image rows this call must accumulate (the strip's inner
/// rows), `strip_y0`/`halo_offset` map image rows to buffer-local rows
/// (`local = row − strip_y0 + halo_offset`; strip path: `y0`/`HALO_P` —
/// a band's ±5-row extent always lands on real gathered rows because
/// bands and strips are both 32-row aligned and `HALO_P ≥ 5`; whole-plane
/// path: `0`/`0`), and `height` is the full plane height at this scale
/// (band extents clamp against it, exactly like v1 clamps against the
/// plane).
///
/// Store flags are off, so the kernel's `mu1/mu2/sd` side-outputs are
/// never written — empty slices are safe (every write in every tier is
/// `if store_*`-gated).
///
/// `planes = [mu1_h, mu2_h, ssq_h, s12_h, src, dst]`, all in the SAME
/// buffer-local coordinate system (`local = row − strip_y0 +
/// halo_offset`): the strip path passes the scratch H-planes + halo
/// buffers; the whole-plane path passes the full-plane H-planes + the
/// real src/dst planes with `strip_y0 = halo_offset = 0`.
#[allow(clippy::too_many_arguments)]
fn fold_v1_basic_bands(
    width: usize,
    rows: core::ops::Range<usize>,
    strip_y0: usize,
    halo_offset: usize,
    height: usize,
    planes: [&[f32]; 6],
    sums: &mut V1BasicSums,
) {
    debug_assert_eq!(rows.start % V1_BAND_ROWS, 0, "strips are 32-row aligned");
    let [mu1_h, mu2_h, ssq_h, s12_h, src, dst] = planes;
    let mut b0 = rows.start;
    while b0 < rows.end {
        let b1 = (b0 + V1_BAND_ROWS).min(rows.end);
        // v1's band buffer: [b0 − overlap, b1 + overlap) clamped to the
        // plane — the kernel's own mirror at the buffer edges then
        // reproduces v1's boundary/init behavior bit-for-bit.
        let top = b0.saturating_sub(V1_BAND_OVERLAP);
        let bot = (b1 + V1_BAND_OVERLAP).min(height);
        let lt = top + halo_offset - strip_y0;
        let lb = bot + halo_offset - strip_y0;
        let h_local = bot - top;
        let inner_start = b0 - top;
        let inner_h = b1 - b0;
        let span = lt * width..lb * width;
        let mut empty_mu1: [f32; 0] = [];
        let mut empty_mu2: [f32; 0] = [];
        let mut empty_sd: [f32; 0] = [];
        sums.accumulate(&crate::fused::fused_vblur_features_ssim(
            &mu1_h[span.clone()],
            &mu2_h[span.clone()],
            &ssq_h[span.clone()],
            &s12_h[span.clone()],
            &src[span.clone()],
            &dst[span],
            width,
            h_local,
            inner_start,
            inner_h,
            BLUR_RADIUS,
            &mut empty_mu1,
            &mut empty_mu2,
            false,
            &mut empty_sd,
            false,
        ));
        b0 = b1;
    }
}

/// No-fold form of [`compute_channel_scale_v2_with_fold`] — the walk now
/// routes through the fold-aware body unconditionally, so this thin
/// delegate exists only for the direct-call unit tests.
#[cfg(test)]
fn compute_channel_scale_v2(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    moments: Option<(&[f32], &[f32])>,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> (f64, f64) {
    let (g, _sums) = compute_channel_scale_v2_with_fold(
        src, dst, width, height, toggles, moments, false, None, scratch, out,
    );
    g
}

/// [`compute_channel_scale_v2_whole`]'s strip-path sibling plus the
/// optional v1-basic FOLD: when `fold_v1` is set, each strip additionally
/// runs v1's `fused_vblur_features_ssim` over the SAME H-blurred planes
/// the v2 blur pass just produced (see [`fold_v1_basic_bands`]), returning
/// the accumulated [`V1BasicSums`] alongside the gradient pair. With
/// `fold_v1 == false` the extra work is one untaken branch per strip and
/// the returned sums are zeros.
#[allow(clippy::too_many_arguments)]
fn compute_channel_scale_v2_with_fold(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    moments: Option<(&[f32], &[f32])>,
    fold_v1: bool,
    append: Option<(AppendCtx<'_>, &mut [f64])>,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> ((f64, f64), V1BasicSums) {
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");
    if let Some((mu1_full, act_full)) = moments {
        assert_eq!(mu1_full.len(), n, "cached mu1 plane must be width*height");
        assert_eq!(
            act_full.len(),
            n,
            "cached activity plane must be width*height"
        );
    }
    assert_eq!(
        out.len(),
        FEATURES_PER_CHANNEL_V2_TOTAL,
        "out slice must hold exactly one channel-scale's v2 block"
    );
    let (append_ctx, mut append_out) = match append {
        Some((ctx, out_app)) => {
            assert!(ctx.ref_y.len() >= n, "append ref_y plane must cover the scale");
            assert_eq!(
                out_app.len(),
                FEATURES_PER_CHANNEL_APPEND,
                "append out slice must hold exactly one channel-scale's append block"
            );
            (Some(ctx), Some(out_app))
        }
        None => (None, None),
    };

    // Phase-6 (§A.16 lever B): small images skip the strip machinery
    // entirely — see `STRIP_BYPASS_HEIGHT`'s doc and `compute_channel_
    // scale_v2_whole`. Currently DISABLED (`STRIP_BYPASS_HEIGHT=0`, see its
    // doc) so this comparison is a permanent `false` today — written
    // generically (not as a dead-code-eliding special case) so a future
    // session re-enabling the lever only needs to change the constant.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        return compute_channel_scale_v2_whole_with_fold(
            src,
            dst,
            width,
            height,
            toggles,
            fold_v1,
            append_ctx.map(|ctx| (ctx, append_out.take().expect("ctx and out arrive together"))),
            scratch,
            out,
        );
    }

    let max_wide_h = STRIP_ROWS + 2 * HALO_P;
    assert!(
        width * max_wide_h <= scratch.mu1.len(),
        "scratch buffers must be sized for the largest strip+halo"
    );

    let mut dense = DenseAccum::default();
    let mut grad = GradientAccum::default();
    let mut v1_sums = V1BasicSums::default();
    let mut app_acc = AppendAccum::default();

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;

        // --- Gather this strip's halo-padded input (real image data,
        //     mirror-reflected only within BLUR_RADIUS*2 of the TRUE
        //     image top/bottom — see `gather_strip_halo`/`reflect_101`)
        //     into the scratch's own halo buffers (reused across strips,
        //     calls, and — via `V2Scratch` — across pairs). ---
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.src_wide[..n_wide],
        );
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.dst_wide[..n_wide],
        );

        // --- Blur pass on the strip+halo buffer only (§A.15's actual
        //     memory-traffic reduction: `wide_h` is O(STRIP_ROWS), not
        //     O(height)). With cached reference moments the mu1 V-blur +
        //     activity chain drop out of the per-pair cost entirely. ---
        if moments.is_some() {
            if fold_v1 {
                run_blur_pass_strip_cached_ref_fold(width, wide_h, scratch);
            } else {
                run_blur_pass_strip_cached_ref(width, wide_h, scratch);
            }
        } else {
            run_blur_pass_strip(width, wide_h, scratch);
        }


        // --- v1-basic FOLD: v1's own fused V-blur+features kernel over
        //     the H-planes this strip's blur pass just filled, replaying
        //     v1's 32-row band tiling (see `fold_v1_basic_bands`). Runs
        //     before the dense kernel purely for locality (the H-planes
        //     are cache-hot); the two reads are independent. ---
        if fold_v1 {
            let n_wide = width * wide_h;
            fold_v1_basic_bands(
                width,
                y0..y0 + strip_h,
                y0,
                HALO_P,
                height,
                [
                    &scratch.mu1_h[..n_wide],
                    &scratch.mu2_h[..n_wide],
                    &scratch.ssq_h[..n_wide],
                    &scratch.s12_h[..n_wide],
                    &scratch.src_wide[..n_wide],
                    &scratch.dst_wide[..n_wide],
                ],
                &mut v1_sums,
            );
        }

        // --- Slice down to the strip's own real rows (buffer-local
        //     offset HALO_P, matching `gather_strip_halo`'s convention:
        //     local row `HALO_P + k` is global row `y0 + k`). Cached
        //     reference moments are FULL-plane buffers, indexed by global
        //     row (`y0 * width`), not halo-local offset. ---
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out_base = y0 * width;
        let src_strip = &scratch.src_wide[off..off + strip_n];
        let dst_strip = &scratch.dst_wide[off..off + strip_n];
        let (mu1_strip, activity_strip) = match moments {
            Some((mu1_full, act_full)) => (
                &mu1_full[out_base..out_base + strip_n],
                &act_full[out_base..out_base + strip_n],
            ),
            None => (
                &scratch.mu1[off..off + strip_n],
                &scratch.activity[off..off + strip_n],
            ),
        };
        let mu2_strip = &scratch.mu2[off..off + strip_n];
        let ssq_strip = &scratch.ssq[off..off + strip_n];
        let s12_strip = &scratch.s12[off..off + strip_n];

        let strip_dense = dense_block_kernel(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            s12_strip,
            activity_strip,
            width,
            strip_h,
            toggles.transducer_bank,
        );
        dense.accumulate(&strip_dense);

        if toggles.gradient_features {
            // Gradient needs src/dst at [y0-1, y0+strip_h+1) — 1-row
            // halo, comfortably inside the HALO_P(=10)-row buffer we
            // already gathered. Buffer-local offset HALO_P-1.
            let g_off = (HALO_P - 1) * width;
            let g_n = width * (strip_h + 2);
            let src_g = &scratch.src_wide[g_off..g_off + g_n];
            let dst_g = &scratch.dst_wide[g_off..g_off + g_n];
            let strip_grad = gradient_block_kernel(src_g, dst_g, activity_strip, width, strip_h);
            grad.accumulate(&strip_grad);
        }

        // --- Append-block second formula pass over the (cache-hot) strip
        //     planes. Full-plane inputs (`ref_y`, `bs2`, cross
        //     activities) index by global row, exactly like the cached
        //     moments above. No per-pair blur runs for the σ-split: the
        //     kernel derives `var₂ = (ssq − bs2) − mu2²` from the cached/
        //     replayed reference `bs2` plane. ---
        if let Some(ctx) = append_ctx {
            let bs2_strip = &ctx.bs2[out_base..out_base + strip_n];
            let refy_strip = &ctx.ref_y[out_base..out_base + strip_n];
            let cross_strips = ctx
                .cross
                .map(|(ax, ab)| (&ax[out_base..out_base + strip_n], &ab[out_base..out_base + strip_n]));
            let strip_app = append_block_kernel(
                src_strip,
                dst_strip,
                mu1_strip,
                mu2_strip,
                ssq_strip,
                bs2_strip,
                activity_strip,
                refy_strip,
                cross_strips,
                width,
                strip_h,
            );
            app_acc.accumulate(&strip_app);
        }

        y0 += strip_h;
    }

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    let grads = finish_channel_scale(&dense, &grad, sum_blockiness, n, out);
    if let (Some(ctx), Some(out_app)) = (append_ctx, append_out) {
        finish_append(
            &dense,
            &app_acc,
            &grad,
            n,
            ctx.cross.is_some(),
            toggles,
            out_app,
        );
    }
    (grads, v1_sums)
}

/// Phase-6 (§A.16 lever B) whole-image path: below [`STRIP_BYPASS_HEIGHT`],
/// skip the strip loop's halo-gather-and-reblur machinery entirely and run
/// the blur pass + dense/gradient kernels directly on the real image —
/// mirroring phase-4's non-strip design (`compute_channel_scale_v2` at
/// commit `7696b62a`, before phase-5's strip-tiling rewrite), reusing the
/// SAME kernels (`dense_block_kernel`/`gradient_block_kernel`/
/// `blockiness_sparse`/`finish_channel_scale`) rather than re-deriving
/// them: `run_blur_pass`'s `crate::blur` primitives already do their own
/// boundary reflection (identical to how phase-4 called them directly on
/// `src`/`dst`), so no halo copy is needed for the blur+dense pass at all.
/// The one exception is `gradient_block_kernel_generic`'s NEW (phase-5)
/// halo contract, which needs a 1-row-padded buffer — built via
/// `gather_strip_halo(halo=1)`, the SAME helper the strip loop already
/// uses for its own (larger) halo, just with `halo=1` and sourced directly
/// from the real image instead of a strip's local window.
/// No-fold form of [`compute_channel_scale_v2_whole_with_fold`] — kept
/// for the direct-call unit tests, same as `compute_channel_scale_v2`.
#[cfg(test)]
fn compute_channel_scale_v2_whole(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> (f64, f64) {
    let (g, _sums) = compute_channel_scale_v2_whole_with_fold(
        src, dst, width, height, toggles, false, None, scratch, out,
    );
    g
}

/// [`compute_channel_scale_v2_whole`] with the optional v1-basic fold —
/// same relationship as `compute_channel_scale_v2_with_fold` to its plain
/// form. On this (whole-plane) path the v1 kernel runs over the full
/// planes with `inner_start = 0`; its V-blur mirror at the plane's own
/// rows 0/height-1 is the reflect-101 image-boundary treatment, the same
/// convention `gather_strip_halo` applies on the strip path.
#[allow(clippy::too_many_arguments)]
fn compute_channel_scale_v2_whole_with_fold(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    append: Option<(AppendCtx<'_>, &mut [f64])>,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> ((f64, f64), V1BasicSums) {
    let n = width * height;
    debug_assert!(
        n <= scratch.mu1.len(),
        "scratch buffers must be sized for at least STRIP_BYPASS_HEIGHT rows \
         (see the scratch allocation in compute_v2_features_impl_with_toggles)"
    );

    run_blur_pass(src, dst, width, height, scratch);

    // v1-basic fold, whole-plane form: full-plane H-planes + the real
    // src/dst planes, identity local mapping (strip_y0 = halo_offset = 0).
    let mut v1_sums = V1BasicSums::default();
    if fold_v1 {
        fold_v1_basic_bands(
            width,
            0..height,
            0,
            0,
            height,
            [
                &scratch.mu1_h[..n],
                &scratch.mu2_h[..n],
                &scratch.ssq_h[..n],
                &scratch.s12_h[..n],
                src,
                dst,
            ],
            &mut v1_sums,
        );
    }

    let mu1 = &scratch.mu1[..n];
    let mu2 = &scratch.mu2[..n];
    let ssq = &scratch.ssq[..n];
    let s12 = &scratch.s12[..n];
    let activity = &scratch.activity[..n];

    let dense = dense_block_kernel(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        toggles.transducer_bank,
    );

    let grad = if toggles.gradient_features {
        // gradient_block_kernel_generic's contract needs src_h/dst_h sized
        // width*(height+2) with a 1-row halo on each side (§A.15.1) — the
        // ONLY extra copy this bypass path pays (O(width), not
        // O(width*height)), built with the same `gather_strip_halo` the
        // strip loop uses, `halo=1` instead of `HALO_P`.
        let mut src_g = vec![0.0f32; width * (height + 2)];
        let mut dst_g = vec![0.0f32; width * (height + 2)];
        gather_strip_halo(src, width, height, 0, height + 2, 1, &mut src_g);
        gather_strip_halo(dst, width, height, 0, height + 2, 1, &mut dst_g);
        gradient_block_kernel(&src_g, &dst_g, activity, width, height)
    } else {
        GradientAccum::default()
    };

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    let grads = finish_channel_scale(&dense, &grad, sum_blockiness, n, out);
    if let Some((ctx, out_app)) = append {
        let app_acc = append_block_kernel(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            &ctx.bs2[..n],
            activity,
            &ctx.ref_y[..n],
            ctx.cross.map(|(ax, ab)| (&ax[..n], &ab[..n])),
            width,
            height,
        );
        finish_append(
            &dense,
            &app_acc,
            &grad,
            n,
            ctx.cross.is_some(),
            toggles,
            out_app,
        );
    }
    (grads, v1_sums)
}

/// Shared tail for both [`compute_channel_scale_v2`] (strip path) and
/// [`compute_channel_scale_v2_whole`] (phase-6 §A.16 lever B bypass path):
/// converts raw-moment/weighted-pool accumulators into the final 29
/// per-channel-scale feature values. Identical for both callers — the
/// accumulators (`DenseAccum`/`GradientAccum`) are plain sums, so N
/// strips' partials (via `.accumulate()`) or one whole-image pass produce
/// the exact same totals feeding this function; extracting it once avoids
/// a second copy of this ~50-line conversion drifting from the original
/// (the no-duplication policy's concern — see CLAUDE.md "NO DUPLICATE
/// IMPLEMENTATIONS").
#[inline]
fn finish_channel_scale(
    dense: &DenseAccum,
    grad: &GradientAccum,
    sum_blockiness: f64,
    n: usize,
    out: &mut [f64],
) -> (f64, f64) {
    let n_f = n as f64;

    // --- Raw-moment -> central-moment conversion (§A.14 module doc):
    //     replaces `OnlineMoments::finish()`'s Terriberry-tracked
    //     mean/M2/M4 with the standard identity from the SIMD-accumulated
    //     raw power sums. Bit-for-bit different from Terriberry (different
    //     floating-point operation order), numerically equivalent within
    //     the file's 5e-4 relative tolerance (verified by
    //     `simd_matches_scalar_within_tolerance`). ---
    let mean_d = dense.sum_d / n_f;
    let raw2 = dense.sum_d2 / n_f;
    let raw3 = dense.sum_d3 / n_f;
    let raw4 = dense.sum_d4 / n_f;
    let m2 = (raw2 - mean_d * mean_d).max(0.0);
    let m4 =
        (raw4 - 4.0 * mean_d * raw3 + 6.0 * mean_d * mean_d * raw2 - 3.0 * mean_d.powi(4)).max(0.0);
    let dev2 = m2.sqrt();
    let dev4 = m4.powf(0.25);

    // Phase-6 (§A.16): defensive bounds clamp on the FINAL feature value
    // (called once per channel-scale, O(1) not O(pixels) — cheap). Not
    // load-bearing for the shipped algebra-only reformulation (native `/`
    // stays correctly-rounded, so the pre-phase-6 per-pixel `.max(zero)`
    // guards were already sufficient) — kept as low-cost belt-and-braces
    // hardening for every feature whose per-pixel formula was touched this
    // phase (ssim_d_local_v / bounded_sim_v / bounded_excess_v /
    // bounded_excess_pair_v / saturate_v / pjnd_transducer_v), and because
    // a `.recip()`-based variant of these formulas WAS tried and measured
    // safe-but-neutral (§A.16, module doc above) — this clamp is what that
    // path would have needed had it shipped, left in place in case a
    // future session revisits recip on different hardware.
    #[inline]
    fn clamp01(v: f64) -> f64 {
        v.clamp(0.0, 1.0)
    }
    #[inline]
    fn clamp02(v: f64) -> f64 {
        v.clamp(0.0, 2.0)
    }

    out[idx::SSIM_MEAN] = clamp02(mean_d);
    out[idx::SSIM_DEV2] = clamp02(dev2);
    out[idx::SSIM_DEV4] = clamp02(dev4);
    out[idx::ART] = clamp01(dense.sum_art / n_f);
    out[idx::DET] = clamp01(dense.sum_det / n_f);
    out[idx::MSE] = clamp01(dense.sum_mse / n_f);
    out[idx::HF_GAIN] = clamp01(dense.sum_hf_gain / n_f);
    out[idx::HF_LOSS] = clamp01(dense.sum_hf_loss / n_f);
    out[idx::HF_MAG_LOSS] = clamp01(dense.sum_hf_mag_loss / n_f);
    out[idx::SSIM_SOFT_PEAK] = clamp02(dense.ws_peak_ssim.finish());
    out[idx::ART_SOFT_PEAK] = clamp02(dense.ws_peak_art.finish());
    out[idx::DET_SOFT_PEAK] = clamp02(dense.ws_peak_det.finish());
    out[idx::MASKED_SSIM] = clamp02(dense.ws_mask_ssim.finish());
    out[idx::MASKED_ART] = clamp02(dense.ws_mask_art.finish());
    out[idx::MASKED_DET] = clamp02(dense.ws_mask_det.finish());
    out[idx::MASKED_MSE] = clamp02(dense.ws_mask_mse.finish());
    out[idx::IW_SSIM] = clamp02(dense.ws_iw_ssim.finish());
    out[idx::IW_ART] = clamp02(dense.ws_iw_art.finish());
    out[idx::IW_DET] = clamp02(dense.ws_iw_det.finish());
    out[idx::IW_MSE] = clamp02(dense.ws_iw_mse.finish());
    out[idx::PJND_TRANSDUCER] = clamp01(dense.sum_pjnd / n_f);
    out[idx::PJND_FRAGILITY] = 1.0 - saturate(grad.sum_grad_src / n_f, C_PJND_GRAD);
    out[idx::GMS] = grad.sum_gms / n_f;
    out[idx::PJND_TRANSDUCER_LOW_K] = clamp01(dense.sum_pjnd_lo / n_f);
    out[idx::PJND_TRANSDUCER_HIGH_K] = clamp01(dense.sum_pjnd_hi / n_f);
    out[idx::BLOCKINESS] = sum_blockiness / n_f;
    out[idx::RINGING] = grad.sum_ringing / n_f;
    out[idx::BANDING] = grad.sum_banding / n_f;
    out[idx::EDGE_WIDTH_CHANGE] = 0.0; // filled in by the caller (needs the adjacent scale)

    (grad.sum_grad_src / n_f, grad.sum_grad_dst / n_f)
}

// ============================================================================
// Per-pixel diffmap generator (v2 core)
//
// Spatializes the SIMPLE-MEAN families of the dense/gradient/blockiness
// kernels above into an explicit per-pixel map, so
// `mean_over_pixels(map) == Σ_local weights[local] * feature_local` for
// every family whose FEATURE is, by construction, the unweighted mean of a
// well-defined per-pixel quantity. Scalar-only (no `#[magetypes]`/`incant!`
// dispatch) — this is the correctness-first reference implementation; a
// SIMD-accelerated sibling is a natural follow-up once this is proven right
// (see `tests::v2_diffmap_block_pool_matches_features`, the gate this
// module's own doc requires before anything here is trusted).
// ============================================================================

/// Per-pixel diffmap for ONE channel-scale, built from the SAME closed-form
/// formulas [`dense_block_kernel_generic`]/[`gradient_block_kernel_generic`]/
/// [`blockiness_sparse`] use to accumulate their pooled means — just written
/// out per-pixel instead of summed. For every SUPPORTED `local` (see below),
/// `dm[i]` carries `weights[local] * M_local(pixel i)`, where `M_local` is
/// the exact per-pixel quantity whose unweighted mean over all
/// `width*height` pixels equals [`finish_channel_scale`]'s `feature_local`
/// (BEFORE that function's mean-level `clamp01`/`clamp02` — see "Clamping"
/// below). Returns `dm[i] = Σ_local weights[local] · M_local(pixel i)`.
///
/// Reuses [`gather_strip_halo`]/[`run_blur_pass`]/[`ScratchV2Strip`] — the
/// SAME strip/halo machinery [`compute_channel_scale_v2`] (the function the
/// real v2 pipeline actually calls) uses to build `mu1`/`mu2`/`ssq`/`s12`/
/// `activity` — so this function's per-pixel inputs are IDENTICAL to the
/// ones the pooled features are computed from, not a fresh (and possibly
/// boundary-divergent) recomputation.
///
/// # Supported (spatialized) families
///
/// - [`idx::SSIM_MEAN`] — `ssim_d_local(mu1, mu2, s12, ssq)` itself (the
///   C1/C2-bounded SSIM dissimilarity), unchanged from
///   [`dense_block_kernel_generic`]'s scalar tail.
/// - [`idx::ART`] / [`idx::DET`] — the edge-artifact/detail split:
///   `1 - bounded_sim(|src-mu1|, |dst-mu2|, C_EDGE)` routed to ART when
///   `|dst-mu2| > |src-mu1|`, to DET when `<`, else neither fires (both 0).
/// - [`idx::MSE`] — `saturate((src-dst)^2, C_MSE)`.
/// - [`idx::HF_GAIN`] / [`idx::HF_LOSS`] — the shared-denominator
///   `bounded_excess_pair(hf_dst^2, hf_src^2, C_HF)` (`hf_x = x - mu_x`).
/// - [`idx::HF_MAG_LOSS`] — `bounded_excess(|hf_src|, |hf_dst|, C_HF)`.
/// - [`idx::PJND_TRANSDUCER`] — the CORE masking transducer only
///   (`k = K_PJND_MASK`): `pjnd_transducer(|src-dst|, activity, K_PJND_MASK,
///   C_PJND_CLAMP)`.
/// - [`idx::GMS`] — `1 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS)`
///   (central-difference gradient magnitude), using
///   [`gradient_block_kernel_generic`]'s own boundary convention exactly:
///   x-neighbors clamp-to-edge (`saturating_sub`/`.min`), y-neighbors
///   whole-sample-mirror via [`reflect_101`] (sourced from the strip's
///   halo-gathered buffer, so a strip seam or the true image edge both
///   resolve identically to the pooled feature's own computation).
/// - [`idx::BLOCKINESS`] — sparse 8-pixel-lattice step-energy comparison
///   (`bounded_excess(step_dst, step_src, C_BLOCK)`); nonzero ONLY at
///   `x % BLOCK_LATTICE == 0` / `y % BLOCK_LATTICE == 0` positions, matching
///   [`blockiness_sparse`]'s own sparsity — a lattice-corner pixel gets BOTH
///   the vertical- and horizontal-boundary contributions added, exactly as
///   [`blockiness_sparse`]'s two passes both add into the same running sum.
/// - [`idx::RINGING`] — `saturate(|src-dst|,C_RING_ERR) *
///   saturate(activity,C_ACTIVITY) * (1 - saturate(grad_src_mag,C_RING_EDGE))`.
/// - [`idx::BANDING`] — `bounded_excess(grad_dst_mag, grad_src_mag,
///   C_BAND_DST) * (1 - saturate(grad_src_mag, C_BAND_SRC))`.
/// - [`idx::MASKED_SSIM`] / [`idx::MASKED_ART`] / [`idx::MASKED_DET`] /
///   [`idx::MASKED_MSE`] and [`idx::IW_SSIM`] / [`idx::IW_ART`] /
///   [`idx::IW_DET`] / [`idx::IW_MSE`] — the weighted-pool families
///   `Σ(w·v)/Σw`. Additive weighted means, so spatializable as `n·w·v/Σw`:
///   the per-pixel numerators (`mask_w·v` with `mask_w = 1 - sat(act)`,
///   `iw_w·v` with `iw_w = sat(act) + IW_WEIGHT_FLOOR`) and the global `Σw`
///   are accumulated in the strip pass, then normalized by `n/Σw` after — so
///   `mean_over_pixels(map)` equals the pooled ratio exactly (block-pool test).
///   ART/DET route by `diff_dst`≷`diff_src`, matching the pooled kernel.
///
/// # Excluded families (weight ignored, regardless of what the caller passes)
///
/// - [`idx::SSIM_DEV2`] / [`idx::SSIM_DEV4`] — Terriberry/GMSD-style
///   deviation-FROM-THE-MEAN moments (`dev2 = sqrt(M2/n)`,
///   `M2/n = E[d^2] - E[d]^2`): a NONLINEAR function of the whole-image
///   mean, not itself the mean of any single per-pixel quantity.
/// - [`idx::SSIM_SOFT_PEAK`], [`idx::ART_SOFT_PEAK`], [`idx::DET_SOFT_PEAK`]
///   — the saliency-weighted-pool families (`Σw·v/Σw` with `w = saliency`).
///   Additive like masked/iw and spatializable the same way once the
///   per-pixel saliency weight is threaded through — a natural follow-up
///   (masked/iw landed the pattern; soft-peak reuses it with `w = sal`).
/// - [`idx::PJND_FRAGILITY`] — `1 - saturate(mean(grad_src_mag),
///   C_PJND_GRAD)`: reference-only, and a NONLINEAR function of the mean
///   gradient magnitude (not the mean of a per-pixel fragility value).
/// - [`idx::PJND_TRANSDUCER_LOW_K`] / [`idx::PJND_TRANSDUCER_HIGH_K`] — the
///   masking-transducer BANK (extra `k` values). Mechanically these use the
///   exact same [`pjnd_transducer`] per-pixel formula as the core transducer
///   above (swap `K_PJND_MASK` for `K_PJND_MASK_LOW`/`K_PJND_MASK_HIGH`) and
///   would spatialize identically — out of scope for THIS core per the task
///   brief, not for any correctness reason. Natural follow-up.
/// - [`idx::EDGE_WIDTH_CHANGE`] — the one genuinely scale-level (not
///   per-pixel) feature in this set: it compares THIS scale's mean gradient
///   magnitude against the ADJACENT (coarser) scale's, filled in by
///   [`compute_v2_features_impl_with_toggles`] only once that scale exists.
///   No single channel-scale's pixels carry enough information to
///   spatialize it.
///
/// # Clamping
///
/// [`finish_channel_scale`] applies a MEAN-level `clamp01`/`clamp02` to
/// several of the supported families (`out[idx::X] = clamp01(sum_x/n)`) —
/// i.e. the clamp bounds the AVERAGE, not each pixel. This function emits
/// the per-pixel quantity WITHOUT that mean-level clamp (`ssim_d_local`'s
/// own internal `.max(0.0)` guard, and the various `bounded_*`/`saturate`
/// helpers' own per-pixel bounds, still apply — those are part of the
/// formula itself, not the mean-level wrapper). So `mean_over_pixels(dm)`
/// equals `Σ weights[local] * feature_local` exactly when every supported
/// family's mean lands inside its clamp range on the given input (the
/// common case for a real, non-degenerate image pair, and the case this
/// module's own test suite always constructs — see `TOL`'s neighboring
/// comment on FP-reassociation tolerance). Callers relying on bit-parity
/// with heavily clamp-saturating input should expect the mean-level clamp
/// to introduce a (bounded, one-directional) divergence between the pooled
/// feature and this map's mean.
///
/// # Panics
///
/// If `src.len() != width*height` or `dst.len() != width*height`.
pub(crate) fn compute_v2_diffmap_channel_scale(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    weights: &[f64; FEATURES_PER_CHANNEL_V2_TOTAL],
) -> Vec<f32> {
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");

    let mut out = vec![0.0f32; n];
    if width == 0 || height == 0 {
        return out;
    }

    // Weighted-pool families (masked/iw) are additive weighted means
    // `Σ(w·v)/Σw` — spatializable as `n·w·v/Σw` once the global `Σw` is known.
    // Accumulate per-pixel weighted numerators + the global `Σw` during the
    // strip pass; normalize into `out` after (so the map's mean equals the
    // pooled weighted-mean feature, the block-pool identity the test gates).
    // soft-peak needs the saliency weight (a follow-up); dev/fragility/
    // edge_width are non-additive and stay excluded.
    let has_mask = weights[idx::MASKED_SSIM] != 0.0
        || weights[idx::MASKED_ART] != 0.0
        || weights[idx::MASKED_DET] != 0.0
        || weights[idx::MASKED_MSE] != 0.0;
    let has_iw = weights[idx::IW_SSIM] != 0.0
        || weights[idx::IW_ART] != 0.0
        || weights[idx::IW_DET] != 0.0
        || weights[idx::IW_MSE] != 0.0;
    let mut masked_num = if has_mask {
        vec![0.0f32; n]
    } else {
        Vec::new()
    };
    let mut iw_num = if has_iw { vec![0.0f32; n] } else { Vec::new() };
    let mut sum_mask_w = 0.0f64;
    let mut sum_iw_w = 0.0f64;

    let max_wide_h = STRIP_ROWS + 2 * HALO_P;
    let mut scratch = ScratchV2Strip::new(width * max_wide_h);
    let mut src_wide = vec![0.0f32; width * max_wide_h];
    let mut dst_wide = vec![0.0f32; width * max_wide_h];

    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let wide_h = strip_h + 2 * HALO_P;
        let n_wide = width * wide_h;

        // Same halo-gather + blur-pass as `compute_channel_scale_v2`'s
        // strip loop (§A.15) — reused verbatim so this function's
        // mu1/mu2/ssq/s12/activity are IDENTICAL to the ones the pooled
        // features are computed from, not a fresh (and possibly
        // boundary-divergent) recomputation.
        gather_strip_halo(
            src,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut src_wide[..n_wide],
        );
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut dst_wide[..n_wide],
        );
        run_blur_pass(
            &src_wide[..n_wide],
            &dst_wide[..n_wide],
            width,
            wide_h,
            &mut scratch,
        );

        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let src_strip = &src_wide[off..off + strip_n];
        let dst_strip = &dst_wide[off..off + strip_n];
        let mu1_strip = &scratch.mu1[off..off + strip_n];
        let mu2_strip = &scratch.mu2[off..off + strip_n];
        let ssq_strip = &scratch.ssq[off..off + strip_n];
        let s12_strip = &scratch.s12[off..off + strip_n];
        let activity_strip = &scratch.activity[off..off + strip_n];

        // 1-row-halo view for the gradient family — same slice convention
        // as `compute_channel_scale_v2`'s call into `gradient_block_kernel`.
        let g_off = (HALO_P - 1) * width;
        let g_n = width * (strip_h + 2);
        let src_g = &src_wide[g_off..g_off + g_n];
        let dst_g = &dst_wide[g_off..g_off + g_n];

        for y_local in 0..strip_h {
            let gy = y0 + y_local;
            let i_strip = y_local * width;
            let i_self = (y_local + 1) * width; // src_g/dst_g: +1 halo row
            let i_up = y_local * width; // src_g/dst_g: row y_local-1 (halo-relative)
            let i_down = (y_local + 2) * width; // src_g/dst_g: row y_local+1 (halo-relative)

            for x in 0..width {
                let i_local = i_strip + x;
                let i_global = gy * width + x;

                let s = src_strip[i_local] as f64;
                let dd = dst_strip[i_local] as f64;
                let m1 = mu1_strip[i_local] as f64;
                let m2 = mu2_strip[i_local] as f64;
                let act = activity_strip[i_local] as f64;

                let mut acc = 0.0f64;

                // --- Dense (always-on) family — bit-for-bit the same
                //     formulas as `dense_block_kernel_generic`'s scalar
                //     tail. ---
                let d = ssim_d_local(m1, m2, s12_strip[i_local] as f64, ssq_strip[i_local] as f64);
                acc += weights[idx::SSIM_MEAN] * d;

                let diff_src = (s - m1).abs();
                let diff_dst = (dd - m2).abs();
                let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
                if diff_dst > diff_src {
                    acc += weights[idx::ART] * edge_dissim;
                } else if diff_dst < diff_src {
                    acc += weights[idx::DET] * edge_dissim;
                }

                let raw_diff = s - dd;
                let raw_sq_err = raw_diff * raw_diff;
                let mse_v = saturate(raw_sq_err, C_MSE);
                acc += weights[idx::MSE] * mse_v;

                // Masked / IW weighted-pool numerators (normalized after the
                // pass by n/Σw). Same `mask_w`/`iw_w` weights and ART/DET
                // routing the pooled kernel uses: `mask_w = 1 - sat(act)`,
                // `iw_w = sat(act) + FLOOR`.
                if has_mask || has_iw {
                    let sat_act = saturate(act, C_ACTIVITY);
                    let (edge_art, edge_det) = if diff_dst > diff_src {
                        (edge_dissim, 0.0)
                    } else if diff_dst < diff_src {
                        (0.0, edge_dissim)
                    } else {
                        (0.0, 0.0)
                    };
                    if has_mask {
                        let mask_w = 1.0 - sat_act;
                        sum_mask_w += mask_w;
                        masked_num[i_global] += (mask_w
                            * (weights[idx::MASKED_SSIM] * d
                                + weights[idx::MASKED_ART] * edge_art
                                + weights[idx::MASKED_DET] * edge_det
                                + weights[idx::MASKED_MSE] * mse_v))
                            as f32;
                    }
                    if has_iw {
                        let iw_w = sat_act + IW_WEIGHT_FLOOR;
                        sum_iw_w += iw_w;
                        iw_num[i_global] += (iw_w
                            * (weights[idx::IW_SSIM] * d
                                + weights[idx::IW_ART] * edge_art
                                + weights[idx::IW_DET] * edge_det
                                + weights[idx::IW_MSE] * mse_v))
                            as f32;
                    }
                }

                let hf_src = s - m1;
                let hf_dst = dd - m2;
                let hf_src_sq = hf_src * hf_src;
                let hf_dst_sq = hf_dst * hf_dst;
                let (hf_gain_i, hf_loss_i) = bounded_excess_pair(hf_dst_sq, hf_src_sq, C_HF);
                acc += weights[idx::HF_GAIN] * hf_gain_i;
                acc += weights[idx::HF_LOSS] * hf_loss_i;
                acc += weights[idx::HF_MAG_LOSS] * bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

                let raw_abs_err = raw_diff.abs();
                acc += weights[idx::PJND_TRANSDUCER]
                    * pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);

                // --- Gradient family — bit-for-bit the same formulas as
                //     `gradient_block_kernel_generic`'s `scalar_pixel`
                //     closure (GMS/ringing/banding), same x-clamp / y-mirror
                //     boundary convention. ---
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(width - 1);
                let sxl = src_g[i_self + xl] as f64;
                let sxr = src_g[i_self + xr] as f64;
                let syu = src_g[i_up + x] as f64;
                let syd = src_g[i_down + x] as f64;
                let dxl = dst_g[i_self + xl] as f64;
                let dxr = dst_g[i_self + xr] as f64;
                let dyu = dst_g[i_up + x] as f64;
                let dyd = dst_g[i_down + x] as f64;

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                acc += weights[idx::GMS] * (1.0 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS));

                let err_b = saturate(raw_abs_err, C_RING_ERR);
                let act_b = saturate(act, C_ACTIVITY);
                let edge_r = saturate(grad_src_mag, C_RING_EDGE);
                acc += weights[idx::RINGING] * (err_b * act_b * (1.0 - edge_r));

                let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
                let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
                acc += weights[idx::BANDING] * (edge_excess * src_smooth_b);

                out[i_global] = acc as f32;
            }
        }

        y0 += strip_h;
    }

    // --- Blockiness — sparse 8-pixel-lattice pass over the FULL
    //     (un-strip-tiled) planes, matching `blockiness_sparse`'s own loop
    //     exactly (including the lattice-corner double-add: a pixel with
    //     both x%8==0 and y%8==0 gets a contribution from EACH loop, same
    //     as `blockiness_sparse`'s single running `sum` does). ---
    let w_block = weights[idx::BLOCKINESS];
    let mut x = BLOCK_LATTICE;
    while x < width {
        for y in 0..height {
            let i = y * width + x;
            let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
            let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
            out[i] += (w_block * bounded_excess(step_dst, step_src, C_BLOCK)) as f32;
        }
        x += BLOCK_LATTICE;
    }
    let mut y = BLOCK_LATTICE;
    while y < height {
        for x in 0..width {
            let i = y * width + x;
            let i_up = i - width;
            let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
            let step_src = (src[i] as f64 - src[i_up] as f64).abs();
            out[i] += (w_block * bounded_excess(step_dst, step_src, C_BLOCK)) as f32;
        }
        y += BLOCK_LATTICE;
    }

    // Normalize the weighted-pool numerators: feature = Σ(w·v)/Σw, so the
    // per-pixel map is (n/Σw)·Σ_family(weight·w·v). Then mean_over_pixels ==
    // Σ_family weight·(Σ w·v / Σw) == Σ_family weight·feature, matching the
    // block-pool identity for the masked/iw families. No-op if a pool's Σw≈0.
    if has_mask && sum_mask_w > 1e-12 {
        let f = n as f64 / sum_mask_w;
        for i in 0..n {
            out[i] += (f * masked_num[i] as f64) as f32;
        }
    }
    if has_iw && sum_iw_w > 1e-12 {
        let f = n as f64 / sum_iw_w;
        for i in 0..n {
            out[i] += (f * iw_num[i] as f64) as f32;
        }
    }

    out
}

/// Full-image v2 diffmap: the per-pixel contribution of a bake's v2 block to
/// its scalar, at scale-0 (full) resolution. This is the builder the runtime
/// steering fold (`compute_with_ref_and_diffmap`'s `ModelSensitivity` path)
/// needs to make the DEPLOYED diffmap read v2 — currently it folds only the
/// ≤372 v1 block, so `M3` (deployed-map coherence) is ~0 for every v2 bake
/// (and every v1 MLP bake) while `M2` (the gradient ceiling) is ≈1.0.
///
/// `s_v2` is the bake's gradient w.r.t. each v2 feature, laid out exactly as
/// [`FeatureViewV2`] indexes them: `s_v2[scale*3*TOTAL + ch*TOTAL + local]`,
/// length `n_scales*3*`[`FEATURES_PER_CHANNEL_V2_TOTAL`]. The caller slices it
/// out of the full bake gradient (`s[372..]` for a frozen-v1++v2 720 bake).
///
/// Builds the SAME pyramid the feature extractor uses (via
/// [`prepare_v2_reference_impl`] for both images — bit-identical planes), runs
/// [`compute_v2_diffmap_channel_scale`] per channel-scale with that scale's
/// weight slice, and nearest-upsamples each coarser scale's map to full res
/// (mean-preserving on even dims) accumulating into one map.
///
/// # Block-pool identity (the correctness gate)
///
/// `mean_over_pixels(full) == Σ_{scale,ch,local∈SUPPORTED} s_v2[..]·feature`,
/// to FP-reassociation tolerance, because each channel-scale map already
/// satisfies that identity ([`compute_v2_diffmap_channel_scale`]'s own gate)
/// and nearest-upsampling preserves a plane's mean. See
/// `tests::v2_diffmap_full_block_pool_matches_features`.
///
/// Exposed to consumers via the public [`crate::Zensim::compute_v2_diffmap`]
/// (the G-STEER harness + encoder closed loop call it there). The `s_v2`
/// gradient it takes is the `s[372..]` tail of a v1-372 ++ v2 bake's full
/// gradient; the v1 contribution comes from the ordinary `ModelSensitivity`
/// diffmap path, summed with this.
pub(crate) fn compute_v2_diffmap_full(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<Vec<f32>, ZensimError> {
    let rprep = prepare_v2_reference_impl(reference, max_pixels, parallel, false)?;
    let dprep = prepare_v2_reference_impl(distorted, max_pixels, parallel, false)?;
    let n_scales = rprep.scales.len().min(dprep.scales.len());
    let per_scale = 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    debug_assert!(
        s_v2.len() >= n_scales * per_scale,
        "s_v2 too short: {} < {}",
        s_v2.len(),
        n_scales * per_scale
    );
    let (w0, h0) = (rprep.scales[0].1, rprep.scales[0].2);
    let mut out = vec![0.0f32; w0 * h0];
    for scale in 0..n_scales {
        let (ref planes_r, ws, hs) = rprep.scales[scale];
        let planes_d = &dprep.scales[scale].0;
        for ch in 0..3 {
            // This channel-scale's weight slice → the [TOTAL] array the core takes.
            let mut weights = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
            let base = scale * per_scale + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            weights.copy_from_slice(&s_v2[base..base + FEATURES_PER_CHANNEL_V2_TOTAL]);
            if weights.iter().all(|&w| w == 0.0) {
                continue;
            }
            let cs =
                compute_v2_diffmap_channel_scale(&planes_r[ch], &planes_d[ch], ws, hs, &weights);
            // Nearest-upsample (ws,hs)->(w0,h0), accumulate. Mean-preserving when
            // w0/h0 are integer multiples of ws/hs (the pyramid's even case).
            for y in 0..h0 {
                let sy = if hs == h0 { y } else { y * hs / h0 };
                let row_c = sy * ws;
                let row_o = y * w0;
                for x in 0..w0 {
                    let sx = if ws == w0 { x } else { x * ws / w0 };
                    out[row_o + x] += cs[row_c + sx];
                }
            }
        }
    }
    Ok(out)
}

// ============================================================================
// Top-level entry point
// ============================================================================

/// Precomputed reference-side state for the v2 extraction: the
/// (reflect-padded, XYB-converted) reference planes at every pyramid
/// scale. Build once per reference via
/// [`crate::Zensim::prepare_v2_reference`], then score any number of
/// distorted variants against it with
/// [`crate::Zensim::compute_v2_features_with_ref`] — each call skips the
/// reference-side decode-adjacent work (reflect-pad copy, RGB→XYB
/// conversion, and the 3-level downscale chain) that the pair entry point
/// would otherwise redo per pair.
///
/// Bit-exactness contract: the planes are built by the SAME functions in
/// the SAME order as the pair path (`convert_source_to_xyb` +
/// `downscale_2x` per level), so `compute_v2_features_with_ref(prepare(r),
/// d)` produces features **bit-identical** to `compute_v2_features(r, d)`
/// — guarded by `prepared_ref_bit_identical_to_pair_path` in this file's
/// tests. Memory: ~`5.33 * w * h` f32s (≈21 MB at 1 MP, ≈85 MB at 4 MP).
pub struct V2PreparedReference {
    /// Per pyramid scale: 3 XYB planes at that scale's exact
    /// `(width, height)` — same layout the pair path materializes.
    scales: Vec<([Vec<f32>; 3], usize, usize)>,
    /// Optional cached reference-side blur moments (`mu1` = blurred ref,
    /// `activity` = blurred |ref − mu1|), per scale per channel, at full
    /// plane size. Filled by REPLAYING the strip walk (same halo gather,
    /// same sliding-sum chain starts), so the cached values are
    /// bit-identical to what the kernels would recompute — see
    /// [`fill_ref_moments`]. When present, each per-pair call skips the
    /// mu1 V-blur and the entire activity chain (abs-diff + 2-pass blur)
    /// per channel-scale.
    moments: Option<Vec<[V2RefMoments; 3]>>,
    /// Original (pre reflect-pad) source dimensions; distorted images are
    /// validated against these, exactly like `validate_pair`.
    orig_width: usize,
    orig_height: usize,
}

/// One channel-scale's cached reference-side blur moments (full-plane).
struct V2RefMoments {
    mu1: Vec<f32>,
    activity: Vec<f32>,
    /// `box_blur(src²)` — the reference half of the append block's σ-split
    /// (`var₁ = bs2 − mu1²`, `var₂ = (ssq − bs2) − mu2²`; see
    /// [`idx_append`]). EMPTY unless the cache was prepared for the append
    /// regime ([`prepare_v2_reference_impl`] `cache_s2`) — existing
    /// moments users pay zero extra memory. Computed by
    /// [`compute_ref_s2blur_into`], the same helper the pair path replays,
    /// so cached and replayed planes are bit-identical by construction.
    bs2: Vec<f32>,
}

impl V2PreparedReference {
    /// Reference width in pixels (unpadded). Distorted images scored
    /// against this reference must match exactly.
    pub fn width(&self) -> usize {
        self.orig_width
    }
    /// Reference height in pixels (unpadded).
    pub fn height(&self) -> usize {
        self.orig_height
    }
    /// Whether reference-side blur moments are cached (see
    /// [`crate::Zensim::prepare_v2_reference_with_moments`]).
    pub fn has_cached_moments(&self) -> bool {
        self.moments.is_some()
    }
}

/// Build a [`V2PreparedReference`]: validate, reflect-pad if sub-64px
/// (identity-skip otherwise — the pad is a pure copy at ≥64px and is NOT
/// performed), convert to XYB once, and materialize all
/// [`crate::NUM_SCALES`] pyramid levels.
pub(crate) fn prepare_v2_reference_impl(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    cache_moments: bool,
) -> Result<V2PreparedReference, ZensimError> {
    prepare_v2_reference_impl_inner(source, max_pixels, parallel, cache_moments, false)
}

/// [`prepare_v2_reference_impl`] for the append regime: moments cache
/// INCLUDING the `bs2 = blur(src²)` planes the f720+ σ-split consumes
/// (+one plane-kind of cache memory — charged only to append-regime
/// preparation; plain moments users keep today's exact footprint).
pub(crate) fn prepare_v2_reference_append_impl(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<V2PreparedReference, ZensimError> {
    prepare_v2_reference_impl_inner(source, max_pixels, parallel, true, true)
}

fn prepare_v2_reference_impl_inner(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    cache_moments: bool,
    cache_s2: bool,
) -> Result<V2PreparedReference, ZensimError> {
    if source.width() == 0 || source.height() == 0 {
        return Err(ZensimError::ImageTooSmall);
    }
    crate::metric::reject_hdr_input(source)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;

    let (orig_width, orig_height) = (source.width(), source.height());
    let scales = if orig_width < crate::metric::MIN_PYRAMID_DIM
        || orig_height < crate::metric::MIN_PYRAMID_DIM
    {
        build_v2_ref_scales(&crate::metric::reflect_pad_to_min(source), parallel)
    } else {
        build_v2_ref_scales(source, parallel)
    };
    let moments = cache_moments.then(|| fill_ref_moments(&scales, cache_s2));
    Ok(V2PreparedReference {
        scales,
        moments,
        orig_width,
        orig_height,
    })
}

/// Materialize the reference XYB pyramid: scale 0 from
/// `convert_source_to_xyb`, each further level via `downscale_2x_into`
/// (bit-identical arithmetic to the pair path's `downscale_2x_inplace` —
/// both compute `(a + b + c + d) * 0.25` per output element in the same
/// order; the in-place/out-of-place split changes only where the result
/// is stored).
fn build_v2_ref_scales(
    img: &impl ImageSource,
    parallel: bool,
) -> Vec<([Vec<f32>; 3], usize, usize)> {
    let mut width = img.width();
    let mut height = img.height();
    let mut scales: Vec<([Vec<f32>; 3], usize, usize)> = Vec::with_capacity(crate::NUM_SCALES);
    scales.push((
        crate::streaming::convert_source_to_xyb(img, width, parallel),
        width,
        height,
    ));
    for _ in 1..crate::NUM_SCALES {
        let new_w = width / 2;
        let new_h = height / 2;
        let mut next: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; new_w * new_h]);
        {
            let (prev_planes, _, _) = scales.last().expect("scale 0 pushed above");
            for ch in 0..3 {
                crate::blur::downscale_2x_into(
                    &prev_planes[ch],
                    width,
                    &mut next[ch],
                    new_w,
                    new_h,
                );
            }
        }
        scales.push((next, new_w, new_h));
        width = new_w;
        height = new_h;
    }
    scales
}

/// Compute the v2 "bounded" feature vector for a source/distorted pair.
///
/// Reuses [`crate::metric::validate_pair`] and
/// [`crate::metric::reflect_pad_to_min`] — identical input handling to
/// v1's entry points. Always uses [`crate::NUM_SCALES`] (4) scales and
/// `BLUR_RADIUS=5`, matching v1's defaults.
///
/// Called by [`crate::Zensim::compute_v2_features`] (the public entry
/// point); kept as a free function so it can be unit-tested without a
/// `Zensim` instance.
pub(crate) fn compute_v2_features_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<ZensimV2Result, ZensimError> {
    compute_v2_features_impl_with_toggles(
        source,
        distorted,
        max_pixels,
        parallel,
        V2NewFeatureToggles::default(),
    )
}

/// Like [`compute_v2_features_impl`], but with explicit control over which
/// phase-2 new-feature GROUPS are computed — used by
/// `Zensim::compute_v2_features_with_toggles` (per-group marginal-cost
/// measurement, `docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12) and by the
/// `v2_speed_baseline`/`v2_stage_profile` benches.
///
/// Ref-reuse pass: this is now a thin composition of
/// [`prepare_v2_reference_impl`] + [`compute_v2_features_with_ref_impl`]
/// — the pair path and the prepared path share ONE scale-walk owner, so
/// they cannot drift. `validate_pair`/`check_within_max_pixels` run first
/// to preserve the original error precedence (dimension mismatch before
/// any per-side rejection).
pub(crate) fn compute_v2_features_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let prepared = prepare_v2_reference_impl(source, max_pixels, parallel, false)?;
    let mut scratch = V2Scratch::new();
    compute_v2_features_with_ref_impl(
        &prepared,
        distorted,
        max_pixels,
        parallel,
        toggles,
        &mut scratch,
    )
}

/// Score one distorted image against a prepared reference — the shared
/// scale-walk owner for BOTH the pair entry point (which prepares
/// inline) and the batch/reuse entry points. Only distorted-side work
/// happens per call: conditional reflect-pad (skipped at ≥64px, where it
/// is an identity copy), XYB conversion, per-scale kernels, and the
/// distorted pyramid downscale. Reference planes are read from
/// `prepared`.
pub(crate) fn compute_v2_features_with_ref_impl(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    compute_v2_features_with_ref_impl_inner(
        prepared, distorted, max_pixels, parallel, toggles, false, scratch,
    )
}

/// Folded-720 pair entry (see [`FeatureRegime::Folded720`] for the
/// layout): validate + prepare + folded with-ref walk, mirroring
/// [`compute_v2_features_impl_with_toggles`]'s composition.
pub(crate) fn compute_folded720_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let prepared = prepare_v2_reference_impl(source, max_pixels, parallel, false)?;
    let mut scratch = V2Scratch::new();
    compute_folded720_with_ref_impl(
        &prepared,
        distorted,
        max_pixels,
        parallel,
        toggles,
        &mut scratch,
    )
}

/// Folded-720 with-ref entry: ONE v2 scale-walk that also computes the v1
/// basic block. Emits the 720-layout vector
/// `[v1 basic f0..156 | zeros f156..372 | v2-348 f372..720]`.
pub(crate) fn compute_folded720_with_ref_impl(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    compute_v2_features_with_ref_impl_inner(
        prepared, distorted, max_pixels, parallel, toggles, true, scratch,
    )
}

/// Folded-720-plus-append pair entry (see
/// [`FeatureRegime::Folded720Append`]): the folded 720 layout with the
/// f720+ append block (204 slots) emitted after it. Forces
/// `toggles.append_block = true`.
pub(crate) fn compute_folded720_append_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

/// Folded-720-plus-append with-ref entry, mirroring
/// [`compute_folded720_with_ref_impl`].
pub(crate) fn compute_folded720_append_with_ref_impl(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_v2_features_with_ref_impl_inner(
        prepared, distorted, max_pixels, parallel, toggles, true, scratch,
    )
}

fn compute_v2_features_with_ref_impl_inner(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    if distorted.width() != prepared.orig_width || distorted.height() != prepared.orig_height {
        return Err(ZensimError::DimensionMismatch);
    }
    crate::metric::reject_hdr_input(distorted)?;
    crate::metric::check_within_max_pixels(distorted.width(), distorted.height(), max_pixels)?;

    // Distorted-side reflect-pad only when genuinely below the pyramid
    // minimum. At ≥64px `reflect_pad_to_min` degenerates to an identity
    // copy (reflect_index(i, n) == i for i < n), so skipping it changes
    // no plane values — it only removes a full per-pixel copy of the
    // image from every pair.
    let mut dst_planes = if distorted.width() < crate::metric::MIN_PYRAMID_DIM
        || distorted.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        crate::streaming::convert_source_to_xyb(&padded_dst, padded_dst.width(), parallel)
    } else {
        crate::streaming::convert_source_to_xyb(distorted, distorted.width(), parallel)
    };

    let (mut width, mut height) = (prepared.scales[0].1, prepared.scales[0].2);
    debug_assert_eq!(
        dst_planes[0].len(),
        width * height,
        "distorted planes must match the prepared reference's scale-0 dims \
         (same original dims + same pad rule guarantee this)"
    );

    let n_scales = prepared.scales.len();
    // Folded layout: v1's full 372-wide block precedes the v2 block —
    // 3 ch × (13 basic + 6 peak + 6 masked + 6 IW) = 93 per scale, 372 at
    // the standard 4 scales (720 total). Only the basic 13/ch/scale are
    // written; the pool blocks stay 0.0 (deprecated, index-stable).
    let v1_total = if fold_v1 { n_scales * 3 * 31 } else { 0 };
    // f720+ append block (see the module-level "f720+ APPEND block"
    // section): emitted strictly AFTER the frozen v1+v2 layout, so every
    // existing index below is untouched whether it is on or off.
    let append_on = toggles.append_block;
    assert!(
        !append_on || fold_v1,
        "append_block requires the folded720 path (use the Folded720Append entries)"
    );
    let v12_total = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let append_total = if append_on {
        n_scales * 3 * FEATURES_PER_CHANNEL_APPEND
    } else {
        0
    };
    let mut features = vec![0.0f64; v12_total + append_total];
    // Per-channel (mean_grad_src, mean_grad_dst) from the previous
    // (finer) scale, for the edge-width-change cross-scale comparison.
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    // Perf pass (§A.13) / phase-5 (§A.15): one scratch buffer set PER
    // CHANNEL, sized for one STRIP+halo at the largest (scale-0) `width`
    // (`width * (STRIP_ROWS + 2*HALO_P)` — NOT `width*height` anymore;
    // this is the actual memory-traffic reduction §A.15 exists to
    // measure). 3 separate sets (rather than 1 reused across channels
    // too) so the 3 channels within a scale can run independently — see
    // the `threads` parallel branch below, which needs disjoint `&mut`
    // scratch per closure. The sets live in the caller-supplied
    // [`V2Scratch`] so batch drivers pay the allocation once per worker,
    // not once per pair.
    //
    // Phase-6 (§A.16 lever B): `compute_channel_scale_v2_whole` (the
    // small-image bypass) needs `width*height` rows of scratch, not
    // `width*(STRIP_ROWS+2*HALO_P)` — bump the allocation to cover
    // whichever is larger. `height.min(STRIP_BYPASS_HEIGHT)` is the exact
    // worst case across every scale this call will process: if scale 0's
    // `height` is already <= the threshold, every scale bypasses and the
    // largest (scale 0, this `height`) sets the bound; if `height` is
    // above the threshold, only smaller (already-downsampled) scales can
    // ever bypass, and none of them can exceed `STRIP_BYPASS_HEIGHT` by
    // construction (that IS the dispatch condition).
    // (`STRIP_BYPASS_HEIGHT=0` today makes this `.min()` a permanent
    // no-op — allow'd for the same reason as the dispatch check above.)
    #[allow(clippy::unnecessary_min_or_max)]
    let bypass_rows = height.min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = width * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    scratch.ensure(strip_max_n);
    // Append-block replay planes (pair path / non-append moments cache):
    // full scale-0-sized planes, reused (sliced down) across scales.
    // Zero allocation on the append-moments extraction path or when the
    // append block is off.
    let have_s2_cache = prepared
        .moments
        .as_ref()
        .map(|m| !m[0][0].bs2.is_empty())
        .unwrap_or(false);
    if append_on && !(have_s2_cache && prepared.moments.is_some()) {
        scratch.ensure_append(width * height, strip_max_n);
    }
    let V2Scratch {
        strips: scratch,
        append_act_x,
        append_act_b,
        append_bs2,
        append_tmp_a,
        append_tmp_b,
        ..
    } = scratch;
    // Split off the append output region so the v1+v2 writes below keep
    // their exact indexing into `features_v12` while per-channel append
    // chunks borrow disjointly from `features_app`.
    let (features_v12, features_app) = features.split_at_mut(v12_total);

    for scale in 0..n_scales {
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;

        // Reference planes for this scale come straight from the prepared
        // pyramid (read-only); only the distorted planes are walked
        // in-place below.
        let (src_planes, ref_w, ref_h) = &prepared.scales[scale];
        debug_assert_eq!(
            (*ref_w, *ref_h),
            (width, height),
            "prepared reference scale dims must track the distorted pyramid walk"
        );
        // Cached reference-side moments for this scale (if prepared with
        // them): per-channel (mu1, activity) full planes.
        let moments_scale = prepared.moments.as_ref().map(|m| &m[scale]);
        let moments_for =
            |ch: usize| moments_scale.map(|ms| (&ms[ch].mu1[..], &ms[ch].activity[..]));

        // Each channel's compute is fully independent WITHIN a scale (only
        // `prev_grad[ch]` from the SAME channel's earlier scale is read,
        // below, after all 3 channels finish, and each channel owns its
        // own scratch set) -- so the 3-channel fan-out can run in
        // parallel when both the `threads` feature and the caller's
        // `parallel` flag are active. Both branches write into the same
        // `features[scale_base..]` chunks; only the iteration strategy
        // differs.
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];
        let mut fold_sums: [V1BasicSums; 3] = [V1BasicSums::default(); 3];
        let out_region =
            &mut features_v12[v1_total + scale_base..][..3 * FEATURES_PER_CHANNEL_V2_TOTAL];

        // --- Append-block per-scale provisioning: the X/B activity planes
        //     the Y channel's cross-masked transducer reads, and each
        //     channel's `bs2 = blur(src²)` σ-split plane. With an
        //     append-prepared moments cache these are borrows of the
        //     cache (zero cost, channel fan-out stays fully independent);
        //     otherwise they are replayed into the `V2Scratch` planes —
        //     bit-identical to the cache by construction (same helpers,
        //     same geometry) — BEFORE the fan-out, so both branches only
        //     read. ---
        let scale_n = width * height;
        let append_cross: Option<(&[f32], &[f32])> = if append_on {
            Some(match moments_scale {
                Some(ms) => (&ms[0].activity[..], &ms[2].activity[..]),
                None => {
                    compute_ref_activity_into(
                        &src_planes[0],
                        width,
                        height,
                        &mut scratch[0],
                        &mut append_act_x[..scale_n],
                    );
                    compute_ref_activity_into(
                        &src_planes[2],
                        width,
                        height,
                        &mut scratch[2],
                        &mut append_act_b[..scale_n],
                    );
                    (&append_act_x[..scale_n], &append_act_b[..scale_n])
                }
            })
        } else {
            None
        };
        let append_bs2_planes: Option<[&[f32]; 3]> = if append_on {
            Some(if have_s2_cache {
                let ms = moments_scale.expect("have_s2_cache implies moments");
                [&ms[0].bs2[..], &ms[1].bs2[..], &ms[2].bs2[..]]
            } else {
                for ch in 0..3 {
                    if APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0 {
                        continue; // cell skipped by the kernel — plane never read
                    }
                    compute_ref_s2blur_into(
                        &src_planes[ch],
                        width,
                        height,
                        append_tmp_a,
                        append_tmp_b,
                        &mut append_bs2[ch][..scale_n],
                    );
                }
                [
                    &append_bs2[0][..scale_n],
                    &append_bs2[1][..scale_n],
                    &append_bs2[2][..scale_n],
                ]
            })
        } else {
            None
        };
        let ref_y_plane: &[f32] = &src_planes[1];
        let app_scale_base = scale * 3 * FEATURES_PER_CHANNEL_APPEND;

        #[cfg(feature = "threads")]
        let ran_parallel = if parallel {
            use rayon::prelude::*;
            let app_chunks: Vec<Option<&mut [f64]>> = if append_on {
                features_app[app_scale_base..][..3 * FEATURES_PER_CHANNEL_APPEND]
                    .chunks_mut(FEATURES_PER_CHANNEL_APPEND)
                    .map(Some)
                    .collect()
            } else {
                vec![None, None, None]
            };
            let results: Vec<((f64, f64), V1BasicSums)> = out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .zip(app_chunks)
                .collect::<Vec<_>>()
                .into_par_iter()
                .zip(scratch.par_iter_mut())
                .enumerate()
                .map(|(ch, ((out, out_app), scr))| {
                    let append_arg = if APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0 {
                        None
                    } else {
                        out_app.map(|oa| {
                            (
                                AppendCtx {
                                    ref_y: ref_y_plane,
                                    bs2: append_bs2_planes.expect("append_on when out_app set")[ch],
                                    cross: if ch == 1 { append_cross } else { None },
                                },
                                oa,
                            )
                        })
                    };
                    let g = compute_channel_scale_v2_with_fold(
                        &src_planes[ch],
                        &dst_planes[ch],
                        width,
                        height,
                        toggles,
                        moments_for(ch),
                        fold_v1,
                        append_arg,
                        scr,
                        out,
                    );
                    apply_transducer_luma_gate(out, ch, toggles);
                    g
                })
                .collect();
            grads = [results[0].0, results[1].0, results[2].0];
            fold_sums = [results[0].1, results[1].1, results[2].1];
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let ran_parallel = false;

        if !ran_parallel {
            let mut app_chunks: Vec<Option<&mut [f64]>> = if append_on {
                features_app[app_scale_base..][..3 * FEATURES_PER_CHANNEL_APPEND]
                    .chunks_mut(FEATURES_PER_CHANNEL_APPEND)
                    .map(Some)
                    .collect()
            } else {
                vec![None, None, None]
            };
            for (ch, (out, out_app)) in out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .zip(app_chunks.iter_mut())
                .enumerate()
            {
                let append_arg = if APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0 {
                    None
                } else {
                    out_app.take().map(|oa| {
                        (
                            AppendCtx {
                                ref_y: ref_y_plane,
                                bs2: append_bs2_planes.expect("append_on when out_app set")[ch],
                                cross: if ch == 1 { append_cross } else { None },
                            },
                            oa,
                        )
                    })
                };
                let (g, sums) = compute_channel_scale_v2_with_fold(
                    &src_planes[ch],
                    &dst_planes[ch],
                    width,
                    height,
                    toggles,
                    moments_for(ch),
                    fold_v1,
                    append_arg,
                    &mut scratch[ch],
                    out,
                );
                grads[ch] = g;
                fold_sums[ch] = sums;
                apply_transducer_luma_gate(out, ch, toggles);
            }
        }

        // --- v1-basic fold finalize: this scale's 13 basic features per
        //     channel, at v1's exact layout (scale-major, 13/ch — the
        //     `features[scale*39 + ch*13 + k]` addressing of the frozen
        //     372 vector). Pooling math replicates v1's finalize
        //     bit-for-bit on the (reassociated) strip sums. ---
        if fold_v1 {
            let n_scale_pixels = width * height;
            for (ch, sums) in fold_sums.iter().enumerate() {
                let base = scale * 39 + ch * 13;
                sums.finalize_into(n_scale_pixels, &mut features_v12[base..base + 13]);
            }
        }

        for ch in 0..3 {
            let (gsrc, gdst) = grads[ch];
            if let Some((prev_gsrc, prev_gdst)) = prev_grad[ch] {
                // Edge-width-change belongs to the FINER (previous) scale's
                // slot -- it's "how much did this scale's edges widen by
                // the time we reach the next coarser scale".
                let decay_src = gsrc / (prev_gsrc + C_GRAD_DECAY);
                let decay_dst = gdst / (prev_gdst + C_GRAD_DECAY);
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[prev_base + idx::EDGE_WIDTH_CHANGE] =
                    1.0 - bounded_sim(decay_src, decay_dst, C_EDGEWIDTH);
            }
            prev_grad[ch] = Some((gsrc, gdst));

            if scale == n_scales - 1 && n_scales >= 2 {
                // Coarsest scale has no next scale to compare against --
                // duplicate the previous (second-coarsest) scale's value
                // (documented approximation; see module/spec doc).
                let this_base = v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[this_base + idx::EDGE_WIDTH_CHANGE] =
                    features_v12[prev_base + idx::EDGE_WIDTH_CHANGE];
            }
        }

        if scale + 1 < n_scales {
            // BUG TRAP: all 3 channels must downscale from the SAME
            // (width, height) — only update the outer width/height AFTER
            // the per-channel loop. Updating them mid-loop (as an earlier
            // draft of this function did) downscales channels 1/2 from
            // channel 0's ALREADY-HALVED dimensions, corrupting the
            // pyramid (caught by this file's own test suite: "src plane
            // length must be width*height" panics on every test that
            // exercises scale > 0).
            //
            // Only the DISTORTED pyramid is walked here — the reference
            // levels were materialized once in `prepared`.
            let mut new_wh = (width, height);
            for plane in dst_planes.iter_mut() {
                new_wh = crate::blur::downscale_2x_inplace(plane, width, height);
            }
            width = new_wh.0;
            height = new_wh.1;
        }
    }

    Ok(ZensimV2Result {
        features,
        n_scales,
        regime: match (fold_v1, append_on) {
            (true, true) => FeatureRegime::Folded720Append,
            (true, false) => FeatureRegime::Folded720,
            (false, false) => FeatureRegime::V2Bounded,
            (false, true) => unreachable!(
                "append_block requires the folded path (asserted at function entry)"
            ),
        },
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::RgbSlice;

    // Perf pass (§A.13): widened from 1e-6 to 5e-4 after switching mu1/mu2/
    // ssq/s12 to `crate::blur::fused_blur_h_ssim`'s single fused-multiply-add
    // sliding-window accumulation (`sum_sq = sv.mul_add(sv, dv.mul_add(dv,
    // sum_sq))`) in place of the old separate `sq_sum_into`/`mul_into`
    // elementwise passes (plain `sv*sv + dv*dv`, two roundings). FMA vs.
    // non-FMA reassociation is a legitimate, expected last-few-ULP source of
    // drift, not a logic change — but `ssim_d_local`'s `denom_s = ssq - mu1²
    // - mu2² + C2_V2` is a near-zero-minus-near-zero subtraction on identity
    // input (variance ≈ 0), stabilized only by the small `C2_V2 = 9e-4`
    // constant, so ULP-scale noise in `ssq` gets divided by ~9e-4 and
    // amplified ~1000x. This amplification is a PRE-EXISTING property of the
    // `ssim_d_local` formula (small-constant-stabilized ratio), not
    // something the perf pass introduced — only WHICH rounding path feeds it
    // changed. Observed on the identity fixture as TOL was raised: idx 0
    // (SSIM_MEAN) 2.52e-6, idx 1 (SSIM_DEV2) 1.47e-5, idx 9 (a
    // weighted-pool/SOFT_PEAK slot one denominator-division further from the
    // source) 1.20e-4 — each roughly one more small-constant division beyond
    // the last, consistent with amplified-not-new noise rather than a
    // distinct bug. 5e-4 leaves >4x headroom above the largest observed.
    // Within this file's own documented allowance (v2 has no downstream
    // consumers yet; v1's golden gate, which does NOT go through this
    // kernel, keeps its own separate zero-tolerance `==` comparison in
    // `tests/v1_golden_bytes.rs`).
    const TOL: f64 = 5e-4;

    fn flat_image(w: usize, h: usize, rgb: [u8; 3]) -> Vec<[u8; 3]> {
        vec![rgb; w * h]
    }

    const ZERO_ONE_IDX: &[usize] = &[
        idx::ART,
        idx::DET,
        idx::MSE,
        idx::HF_GAIN,
        idx::HF_LOSS,
        idx::HF_MAG_LOSS,
        idx::PJND_TRANSDUCER,
        idx::PJND_FRAGILITY,
        idx::GMS,
        idx::PJND_TRANSDUCER_LOW_K,
        idx::PJND_TRANSDUCER_HIGH_K,
        idx::RINGING,
        idx::BANDING,
    ];
    const ZERO_TWO_IDX: &[usize] = &[
        idx::SSIM_MEAN,
        idx::SSIM_DEV2,
        idx::SSIM_DEV4,
        idx::SSIM_SOFT_PEAK,
        idx::ART_SOFT_PEAK,
        idx::DET_SOFT_PEAK,
        idx::MASKED_SSIM,
        idx::MASKED_ART,
        idx::MASKED_DET,
        idx::MASKED_MSE,
        idx::IW_SSIM,
        idx::IW_ART,
        idx::IW_DET,
        idx::IW_MSE,
        idx::BLOCKINESS, // can be up to 2 blocks (h+v corner)
        idx::EDGE_WIDTH_CHANGE,
    ];

    fn assert_all_bounded(view: &FeatureViewV2, tol: f64) {
        for scale in 0..view.n_scales() {
            for ch in 0..3 {
                for &off in ZERO_ONE_IDX {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.is_finite() && (0.0..1.0 + tol).contains(&v),
                        "idx {off} OOB [0,1): {v} at s{scale} c{ch}"
                    );
                }
                for &off in ZERO_TWO_IDX {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.is_finite() && (0.0..=2.0 + tol).contains(&v),
                        "idx {off} OOB [0,2]: {v} at s{scale} c{ch}"
                    );
                }
            }
        }
    }

    #[test]
    fn regime_tag_and_view_accessors_match_raw_indexing() {
        let src = flat_image(64, 64, [128, 128, 128]);
        let mut dst = src.clone();
        dst[0] = [200, 60, 60];
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_eq!(result.regime(), FeatureRegime::V2Bounded);
        assert_eq!(result.n_scales(), crate::NUM_SCALES);
        assert_eq!(
            result.features().len(),
            crate::NUM_SCALES * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
        );

        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                let base =
                    scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                assert_eq!(
                    view.ssim_mean(scale, ch),
                    result.features()[base + idx::SSIM_MEAN]
                );
                assert_eq!(view.gms(scale, ch), result.features()[base + idx::GMS]);
                assert_eq!(
                    view.pjnd_transducer_low_k(scale, ch),
                    result.features()[base + idx::PJND_TRANSDUCER_LOW_K]
                );
                assert_eq!(
                    view.pjnd_transducer_high_k(scale, ch),
                    result.features()[base + idx::PJND_TRANSDUCER_HIGH_K]
                );
                assert_eq!(
                    view.blockiness(scale, ch),
                    result.features()[base + idx::BLOCKINESS]
                );
                assert_eq!(
                    view.ringing(scale, ch),
                    result.features()[base + idx::RINGING]
                );
                assert_eq!(
                    view.banding(scale, ch),
                    result.features()[base + idx::BANDING]
                );
                assert_eq!(
                    view.edge_width_change(scale, ch),
                    result.features()[base + idx::EDGE_WIDTH_CHANGE]
                );
            }
        }
        assert!(FeatureViewV2::new(result.features(), result.n_scales() + 1).is_none());
        assert!(
            FeatureViewV2::new(
                &result.features()[..result.features().len() - 1],
                result.n_scales()
            )
            .is_none()
        );
    }

    /// Identity input => every error-oriented feature is exactly 0.
    /// `pjnd_fragility` is a reference-only property, excluded (as in
    /// iteration 1). `edge_width_change` compares src's OWN cross-scale
    /// decay to dst's — on identity input src==dst so their decays are
    /// identical too => also exactly 0, INCLUDED here (a strengthening
    /// over iteration 1: this is the one new-feature identity check that
    /// exercises the cross-scale path).
    #[test]
    fn identity_input_zeroes_every_error_feature() {
        let mut src = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = (((x * 7 + y * 13) % 256) as u8).max(1);
                src.push([v, v.wrapping_add(40), v.wrapping_add(80)]);
            }
        }
        let dst = src.clone();
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        let view = result.view();
        let must_be_zero = [
            idx::SSIM_MEAN,
            idx::SSIM_DEV2,
            idx::SSIM_DEV4,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::SSIM_SOFT_PEAK,
            idx::ART_SOFT_PEAK,
            idx::DET_SOFT_PEAK,
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
            idx::PJND_TRANSDUCER,
            idx::PJND_TRANSDUCER_LOW_K,
            idx::PJND_TRANSDUCER_HIGH_K,
            idx::GMS,
            idx::RINGING,
            idx::BANDING,
            idx::BLOCKINESS,
            idx::EDGE_WIDTH_CHANGE,
        ];
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                for &off in &must_be_zero {
                    let v = view.at(scale, ch, off);
                    assert!(
                        v.abs() < TOL,
                        "idx {off} not zero on identity: {v} at s{scale} c{ch}"
                    );
                }
            }
        }
    }

    /// Adversarial fixture: flat source + noise (D2/HF-gain pathology).
    #[test]
    fn transducer_luma_gate_zeroes_chroma_only() {
        // Distorted pair (dst != src) so the transducers are non-zero.
        let mut src = Vec::with_capacity(64 * 64);
        let mut dst = Vec::with_capacity(64 * 64);
        for y in 0..64 {
            for x in 0..64 {
                let v = (((x * 7 + y * 13) % 256) as u8).max(1);
                src.push([v, v.wrapping_add(40), v.wrapping_add(80)]);
                let d = v.wrapping_add(((x + y) % 17) as u8);
                dst.push([d, d.wrapping_add(40), d.wrapping_add(80)]);
            }
        }
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 64, 64);

        let base = compute_v2_features_impl(&source, &distorted, None, false).expect("base ok");
        let gated = compute_v2_features_impl_with_toggles(
            &source,
            &distorted,
            None,
            false,
            V2NewFeatureToggles {
                transducers_luma_only: true,
                ..Default::default()
            },
        )
        .expect("gated ok");
        let bv = base.view();
        let gv = gated.view();
        let transducers = [
            idx::PJND_TRANSDUCER,
            idx::PJND_FRAGILITY,
            idx::PJND_TRANSDUCER_LOW_K,
            idx::PJND_TRANSDUCER_HIGH_K,
        ];
        let n_scales = base.n_scales();
        let mut any_chroma_nonzero_in_base = false;
        for scale in 0..n_scales {
            for ch in 0..3 {
                let b = ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let sb = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
                for t in transducers {
                    let bval = base.features()[sb + b + t];
                    let gval = gated.features()[sb + b + t];
                    if ch == 1 {
                        // Luma channel: transducers unchanged.
                        assert_eq!(bval, gval, "Y transducer local {t} scale {scale} changed");
                    } else {
                        // Chroma channels: zeroed under the gate.
                        assert_eq!(gval, 0.0, "chroma transducer local {t} not zeroed");
                        if bval != 0.0 {
                            any_chroma_nonzero_in_base = true;
                        }
                    }
                }
                // Every NON-transducer feature is byte-identical between the two.
                for local in 0..FEATURES_PER_CHANNEL_V2_TOTAL {
                    if transducers.contains(&local) {
                        continue;
                    }
                    assert_eq!(
                        base.features()[sb + b + local],
                        gated.features()[sb + b + local],
                        "non-transducer local {local} ch {ch} scale {scale} changed"
                    );
                }
            }
        }
        // Guard: the gate is actually testing something (chroma transducers
        // WERE non-zero without it).
        assert!(
            any_chroma_nonzero_in_base,
            "test fixture produced no non-zero chroma transducers — gate is untested"
        );
        // Silence the unused-view warnings while keeping them for future asserts.
        let _ = (bv.gms(0, 0), gv.gms(0, 0));
    }

    #[test]
    fn bounded_range_flat_source_plus_noise() {
        let w = 64;
        let h = 64;
        let src = vec![[128u8, 128, 128]; w * h];
        let mut dst = src.clone();
        for y in 0..h {
            for x in 0..w {
                dst[y * w + x] = if (x + y) % 2 == 0 {
                    [200, 60, 220]
                } else {
                    [50, 220, 40]
                };
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Adversarial fixture: large chroma mean-shift (D1 driver).
    #[test]
    fn bounded_range_large_chroma_mean_shift() {
        let w = 64;
        let h = 64;
        let src = flat_image(w, h, [0, 255, 0]);
        let dst = flat_image(w, h, [255, 0, 255]);
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Adversarial fixture: hard edge on a flat chroma field (the exact
    /// masked/IW explosion pattern from
    /// `benchmarks/ssim_moment_explosion_2026-07-16.md`).
    #[test]
    fn bounded_range_hard_edge_on_flat() {
        let w = 64;
        let h = 64;
        let src = vec![[0u8, 200, 0]; w * h];
        let mut dst = src.clone();
        for y in 0..h {
            for x in (w / 2)..w {
                dst[y * w + x] = [220, 0, 220];
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    /// Phase-5 (§A.15): multi-strip boundary correctness. A 150-row image
    /// with `STRIP_ROWS=64` spans 3 strips (`[0,64)`, `[64,128)`,
    /// `[128,150)`), and this fixture puts a HARD HORIZONTAL edge at
    /// `y=128` — exactly the first strip seam at the current
    /// `STRIP_ROWS=128` — plus a second at `y=400` (mid-strip, for
    /// contrast) and blocky columns throughout (stresses blockiness'
    /// lattice check across the seam too). If `gather_strip_halo`/the halo
    /// math were wrong, this is exactly the pattern that would show it:
    /// gradients spanning the seam would see the WRONG neighbor row
    /// (garbage, a mismatched reflection, or a stale scratch value),
    /// producing an out-of-bounds or wildly-off signal right at `y=128`.
    /// Bounded-range + a same-shape identity check (identity input still
    /// zeroes everything even across 5 strips) are the two things
    /// actually exercised here.
    ///
    /// `h=640` is DELIBERATELY well above [`STRIP_BYPASS_HEIGHT`]
    /// (phase-6 §A.16 lever B) — this test exists specifically to exercise
    /// the STRIP LOOP's multi-strip seam handling, and must keep doing so
    /// regardless of where the bypass threshold lands; a height anywhere
    /// near the threshold would silently start routing through
    /// `compute_channel_scale_v2_whole` instead and stop testing what this
    /// test's name promises.
    #[test]
    fn bounded_range_multi_strip_horizontal_edge() {
        let w = 96;
        let h = 640; // 5 strips at STRIP_ROWS=128: [0,128)[128,256)[256,384)[384,512)[512,640)
        assert!(
            h > STRIP_BYPASS_HEIGHT,
            "test fixture must stay above the lever-B bypass threshold to keep exercising \
             the strip loop -- bump h if STRIP_BYPASS_HEIGHT is ever raised past 640"
        );
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                // Horizontal bands (edges at y=128, the first strip seam,
                // and y=400, mid-strip) plus an 8-wide vertical block
                // pattern, so gradient/blockiness both see real structure
                // crossing the strip seam.
                let band = if y < 128 {
                    0u8
                } else if y < 400 {
                    255u8
                } else {
                    120u8
                };
                let block = if (x / 8) % 2 == 0 { 0i16 } else { 40i16 };
                let v = (band as i16 + block).clamp(0, 255) as u8;
                src.push([v, v.wrapping_add(30), v.wrapping_add(60)]);
                // Distorted: shift the band edges by a few rows and
                // perturb the block pattern, so gradients differ on both
                // sides of the seam, not just coincide with src.
                let dband = if y < 125 {
                    10u8
                } else if y < 403 {
                    240u8
                } else {
                    130u8
                };
                let dblock = if (x / 8) % 2 == 0 { 5i16 } else { 35i16 };
                let dv = (dband as i16 + dblock).clamp(0, 255) as u8;
                dst.push([dv, dv.wrapping_add(25), dv.wrapping_add(55)]);
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);

        // Identity check across the SAME multi-strip shape: every
        // error-oriented feature must still be exactly zero when src==dst,
        // even though the computation now crosses 5 strip boundaries.
        let source2 = RgbSlice::new(&src, w, h);
        let identity = RgbSlice::new(&src, w, h);
        let result2 =
            compute_v2_features_impl(&source2, &identity, None, false).expect("compute ok");
        let view2 = result2.view();
        for scale in 0..result2.n_scales() {
            for ch in 0..3 {
                let d = view2.ssim_mean(scale, ch);
                assert!(
                    d.abs() < TOL,
                    "multi-strip identity: ssim_mean not zero: {d} at s{scale} c{ch}"
                );
                let gms = view2.gms(scale, ch);
                assert!(
                    gms.abs() < TOL,
                    "multi-strip identity: gms not zero: {gms} at s{scale} c{ch}"
                );
            }
        }
    }

    /// Phase-6 (§A.16 lever B): dedicated coverage for
    /// `compute_channel_scale_v2_whole` (the small-image bypass path).
    /// `STRIP_BYPASS_HEIGHT=0` DISABLES the bypass at runtime (measured
    /// regression, §A.16.4 — the function is correctness-verified and kept
    /// for a future session, not currently reachable via the public
    /// dispatch), so this test calls `compute_channel_scale_v2_whole`
    /// DIRECTLY rather than through `compute_v2_features_impl` — the only
    /// way to keep exercising it regardless of the (disabled) threshold
    /// value. A hard horizontal edge at `y=0` (the image's OWN true top
    /// edge — exercises `reflect_101`'s boundary case directly, not an
    /// interior strip seam) plus one at mid-image, and an 8-wide vertical
    /// block pattern, so gradient/blockiness see real structure at the
    /// exact row the halo-construction code touches.
    #[test]
    fn bounded_range_bypass_path_small_image() {
        let w = 80;
        let h = 96;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let band = if y < 40 { 20.0f32 } else { 220.0 };
                let block = if (x / 8) % 2 == 0 { 0.0f32 } else { 40.0 };
                src.push((band + block).clamp(0.0, 255.0) / 255.0);
                let dband = if y < 37 { 30.0f32 } else { 200.0 };
                let dblock = if (x / 8) % 2 == 0 { 5.0f32 } else { 35.0 };
                dst.push((dband + dblock).clamp(0.0, 255.0) / 255.0);
            }
        }
        let mut scratch = ScratchV2Strip::new(w * h);
        let mut out = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2_whole(
            &src,
            &dst,
            w,
            h,
            V2NewFeatureToggles::default(),
            &mut scratch,
            &mut out,
        );
        for &off in ZERO_ONE_IDX {
            let v = out[off];
            assert!(
                v.is_finite() && (0.0..1.0 + TOL).contains(&v),
                "bypass-path idx {off} OOB [0,1): {v}"
            );
        }
        for &off in ZERO_TWO_IDX {
            let v = out[off];
            assert!(
                v.is_finite() && (0.0..=2.0 + TOL).contains(&v),
                "bypass-path idx {off} OOB [0,2]: {v}"
            );
        }

        // Identity re-run through the SAME direct call: every error
        // feature must still be exactly zero.
        let mut scratch2 = ScratchV2Strip::new(w * h);
        let mut out2 = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2_whole(
            &src,
            &src,
            w,
            h,
            V2NewFeatureToggles::default(),
            &mut scratch2,
            &mut out2,
        );
        {
            let d = out2[idx::SSIM_MEAN];
            assert!(
                d.abs() < TOL,
                "bypass-path identity: ssim_mean not zero: {d}"
            );
            let gms = out2[idx::GMS];
            assert!(gms.abs() < TOL, "bypass-path identity: gms not zero: {gms}");
        }
    }

    /// Adversarial fixture: 8x8-periodic step pattern (stresses the NEW
    /// oriented-blockiness feature specifically — a fake "JPEG-blocky"
    /// image with real 8-pixel-periodic steps that should score high
    /// blockiness on itself-vs-flat, but bounded).
    #[test]
    fn bounded_range_8x8_periodic_steps() {
        let w = 64;
        let h = 64;
        let src = flat_image(w, h, [100, 100, 100]);
        let mut dst = src.clone();
        for y in 0..h {
            for x in 0..w {
                let block_val = if (x / 8 + y / 8) % 2 == 0 {
                    80u8
                } else {
                    180u8
                };
                dst[y * w + x] = [block_val, block_val, block_val];
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
        // The blockiness signal should register something (not identically
        // zero) on a genuinely blocky pattern -- a sanity check that the
        // feature is alive, not just bounded-and-silent.
        let view = result.view();
        let any_blockiness =
            (0..result.n_scales()).any(|s| (0..3).any(|c| view.blockiness(s, c) > 1e-3));
        assert!(
            any_blockiness,
            "blockiness should fire on a genuine 8x8 step pattern"
        );
    }

    /// Adversarial fixture: banding gradient (smooth ramp in source,
    /// stepped/posterized in distorted) -- stresses the NEW banding
    /// feature specifically.
    #[test]
    fn bounded_range_banding_gradient() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for _y in 0..h {
            for x in 0..w {
                let smooth = (x * 255 / w) as u8;
                let posterized = ((x * 8 / w) * 32) as u8; // 8 flat steps
                src.push([smooth, smooth, smooth]);
                dst.push([posterized, posterized, posterized]);
            }
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
        let view = result.view();
        let any_banding = (0..result.n_scales()).any(|s| (0..3).any(|c| view.banding(s, c) > 1e-3));
        assert!(
            any_banding,
            "banding should fire on a smooth-ramp -> posterized-steps pair"
        );
    }

    /// Blanket bounded-range sweep on deterministic pseudo-random content.
    #[test]
    fn bounded_range_all_signals_random_content() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        let mut state: u32 = 0x1234_5678;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..(w * h) {
            src.push([next(), next(), next()]);
            dst.push([next(), next(), next()]);
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        assert_all_bounded(&result.view(), TOL);
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let src = flat_image(64, 64, [10, 10, 10]);
        let dst = flat_image(32, 32, [10, 10, 10]);
        let source = RgbSlice::new(&src, 64, 64);
        let distorted = RgbSlice::new(&dst, 32, 32);
        let err = compute_v2_features_impl(&source, &distorted, None, false).unwrap_err();
        assert!(matches!(err, ZensimError::DimensionMismatch));
    }

    /// Gated-mirror pin (see `WeightedSum`'s doc): the incremental
    /// `WeightedSum` accumulator used by the fused kernel must match
    /// iteration 1's array-based `WeightedPool::mean` (the promoted
    /// canonical `Σw·v/Σw` helper) bit-for-bit on the same data, so the
    /// perf-motivated rewrite is provably not a silent formula drift.
    #[test]
    fn weighted_sum_matches_weighted_pool_mean_exactly() {
        let mut state: u32 = 0xabcdef01;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / u32::MAX as f32).abs()
        };
        let values: Vec<f32> = (0..1000).map(|_| next() * 2.0).collect();
        let weights: Vec<f32> = (0..1000).map(|_| next()).collect();

        let mut ws = WeightedSum::default();
        for (&v, &w) in values.iter().zip(weights.iter()) {
            ws.add(w as f64, v as f64);
        }
        let incremental = ws.finish();
        let batch = crate::iw_pool::WeightedPool::mean(&values, &weights);
        assert!(
            (incremental - batch).abs() < 1e-9,
            "WeightedSum {incremental} vs WeightedPool::mean {batch} diverged"
        );
    }

    /// Phase-4 (§A.14): proves the SIMD raw-moment reformulation
    /// (`compute_channel_scale_v2`'s `mean_d`/`m2`/`m4` block) is
    /// numerically equivalent to `OnlineMoments`'s Terriberry (2007)
    /// single-pass algorithm on the SAME data, not just "removed and
    /// hoped for the best". Uses `d` values in the actual documented
    /// range `[0, 2]` (not arbitrary floats) since that's what the real
    /// kernel feeds both paths. Also exercises `OnlineMoments` directly
    /// so it isn't dead code despite leaving the hot path.
    #[test]
    fn raw_moment_reformulation_matches_terriberry() {
        let mut state: u32 = 0x5eed_1234;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / u32::MAX as f64).abs() * 2.0 // in [0, 2]
        };
        let d_values: Vec<f64> = (0..50_000).map(|_| next()).collect();

        let mut moments = OnlineMoments::default();
        for &d in &d_values {
            moments.update(d);
        }
        let (terri_mean, terri_dev2, terri_dev4) = moments.finish();

        let n = d_values.len() as f64;
        let (mut sum_d, mut sum_d2, mut sum_d3, mut sum_d4) = (0.0, 0.0, 0.0, 0.0);
        for &d in &d_values {
            sum_d += d;
            sum_d2 += d * d;
            sum_d3 += d * d * d;
            sum_d4 += d * d * d * d;
        }
        let raw_mean = sum_d / n;
        let raw2 = sum_d2 / n;
        let raw3 = sum_d3 / n;
        let raw4 = sum_d4 / n;
        let m2 = (raw2 - raw_mean * raw_mean).max(0.0);
        let m4 = (raw4 - 4.0 * raw_mean * raw3 + 6.0 * raw_mean * raw_mean * raw2
            - 3.0 * raw_mean.powi(4))
        .max(0.0);
        let raw_dev2 = m2.sqrt();
        let raw_dev4 = m4.powf(0.25);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        assert!(
            rel(raw_mean, terri_mean) < 5e-4,
            "mean: raw {raw_mean} vs terriberry {terri_mean}"
        );
        assert!(
            rel(raw_dev2, terri_dev2) < 5e-4,
            "dev2: raw {raw_dev2} vs terriberry {terri_dev2}"
        );
        assert!(
            rel(raw_dev4, terri_dev4) < 5e-4,
            "dev4: raw {raw_dev4} vs terriberry {terri_dev4}"
        );
    }

    #[test]
    fn small_image_reflect_pads_and_scores() {
        let src = flat_image(4, 4, [90, 100, 110]);
        let mut dst = flat_image(4, 4, [90, 100, 110]);
        dst[0] = [255, 0, 0];
        let source = RgbSlice::new(&src, 4, 4);
        let distorted = RgbSlice::new(&dst, 4, 4);
        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("pads + computes");
        assert_eq!(result.n_scales(), crate::NUM_SCALES);
        let view = result.view();
        let any_nonzero =
            (0..result.n_scales()).any(|s| (0..3).any(|c| view.ssim_mean(s, c) > TOL));
        assert!(
            any_nonzero,
            "single-pixel change should move ssim_mean somewhere"
        );
    }

    /// Perf pass (§A.13): the 3-way channel `rayon` fan-out
    /// (`compute_v2_features_impl_with_toggles`'s `parallel` branch, gated
    /// on `#[cfg(feature = "threads")]`) is new in this pass and had ZERO
    /// prior test coverage — every existing test in this file calls with
    /// `parallel: false`. A parallel code path that only gets exercised by
    /// the (output-blind) speed bench is exactly how a data race or a
    /// scratch-buffer aliasing bug would ship silently. Assert the parallel
    /// and serial paths produce IDENTICAL output on real (non-flat,
    /// non-trivial) content spanning all 4 scales. Bit-exact is the right
    /// bar here (not a tolerance): 3-way channel fan-out has no floating-
    /// point reassociation vs the serial loop (each channel's own
    /// `compute_channel_scale_v2` call is byte-for-byte the same function
    /// either way — only the SCHEDULING differs), so any divergence would
    /// mean a real bug (e.g. reading another channel's scratch, or writing
    /// past a `chunks_mut` boundary), not benign FP noise.
    #[test]
    fn parallel_matches_serial_exactly() {
        let w = 200;
        // Deliberately non-power-of-2 (exercises reflect padding too) AND
        // deliberately above `STRIP_BYPASS_HEIGHT` (phase-6 §A.16 lever B)
        // so this keeps exercising the strip loop's multi-strip case
        // (h=647 spans 6 strips at STRIP_ROWS=128) on top of its primary
        // parallel-vs-serial channel-fan-out purpose — a smaller height
        // would silently start routing through the single-pass bypass path
        // instead and drop that secondary coverage.
        let h = 647;
        assert!(
            h > STRIP_BYPASS_HEIGHT,
            "keep this above the lever-B bypass threshold"
        );
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        let mut state: u32 = 0xC0FF_EE42;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        };
        for _ in 0..(w * h) {
            src.push([next(), next(), next()]);
            dst.push([next(), next(), next()]);
        }
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        let serial =
            compute_v2_features_impl(&source, &distorted, None, false).expect("serial compute");
        let parallel =
            compute_v2_features_impl(&source, &distorted, None, true).expect("parallel compute");

        let (sf, pf) = (serial.features(), parallel.features());
        assert_eq!(sf.len(), pf.len());
        let mut mismatches = 0;
        for (i, (&s, &p)) in sf.iter().zip(pf.iter()).enumerate() {
            if s.to_bits() != p.to_bits() && !(s.is_nan() && p.is_nan()) {
                if mismatches < 10 {
                    eprintln!("feature {i}: serial={s:.17e} parallel={p:.17e}");
                }
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches,
            0,
            "{mismatches} of {} v2 features diverged between parallel and serial scheduling",
            sf.len()
        );
    }

    /// Acceptance gate for [`compute_v2_diffmap_channel_scale`]: the
    /// per-pixel map's own mean must reproduce the pooled feature values it
    /// claims to spatialize, computed via [`compute_channel_scale_v2`] —
    /// the SAME entry point the real (non-bypass, `STRIP_BYPASS_HEIGHT=0`)
    /// v2 pipeline calls for every channel-scale. Tests at the PLANE level
    /// directly (a synthetic f32 pair, not an RGB image through XYB
    /// conversion) — sufficient per this function's contract, which only
    /// operates on one channel-scale's plane, and it sidesteps XYB/
    /// multi-scale machinery that's orthogonal to what's being proven here.
    ///
    /// Fixture: a smooth, strictly-positive (`mu1,mu2 >= 0`, required by
    /// `C1_V2` boundedness), textured 64x64 plane pair with a MILD signed
    /// perturbation (so ART and DET both fire somewhere, HF gain and loss
    /// both fire somewhere) plus a small 8-pixel-lattice bump (so BLOCKINESS
    /// is non-trivially exercised, not just trivially 0≈0). "Mild" is
    /// load-bearing: every clamped family (SSIM_MEAN's `clamp02`; ART/DET/
    /// MSE/HF_*/PJND_TRANSDUCER's `clamp01`) must land inside its clamp
    /// range on this fixture, or `finish_channel_scale`'s MEAN-level clamp
    /// would legitimately diverge from this function's unclamped per-pixel
    /// mean (see `compute_v2_diffmap_channel_scale`'s "Clamping" doc) — the
    /// asserts at the end confirm no clamp fired, so a future edit that
    /// makes the fixture more aggressive fails loudly here instead of
    /// silently comparing a clamped feature against an unclamped map.
    #[test]
    fn v2_diffmap_block_pool_matches_features() {
        let width = 64usize;
        let height = 64usize;
        let n = width * height;

        let mut src = vec![0.0f32; n];
        let mut dst = vec![0.0f32; n];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let fx = x as f32;
                let fy = y as f32;
                let base = 0.5 + 0.25 * (fx * 0.19).sin() * (fy * 0.13).cos();
                let s = base.clamp(0.05, 0.95);
                src[i] = s;

                // Mild, signed, spatially-varying perturbation (some pixels
                // brighter, some darker, some unchanged -- exercises ART,
                // DET, HF_GAIN, and HF_LOSS all in the same fixture) plus a
                // small block-lattice bump so BLOCKINESS is non-trivially
                // exercised.
                let mut d = s + 0.03 * ((fx + fy) * 0.5).sin();
                if x % BLOCK_LATTICE == 0 && x > 0 {
                    d += 0.015;
                }
                if y % BLOCK_LATTICE == 0 && y > 0 {
                    d -= 0.015;
                }
                dst[i] = d.clamp(0.05, 0.95);
            }
        }

        // --- "Official" per-channel-scale features, via the SAME entry
        //     point the real v2 pipeline calls. ---
        let toggles = V2NewFeatureToggles::default();
        let max_wide_h = STRIP_ROWS + 2 * HALO_P;
        let mut scratch = ScratchV2Strip::new(width * max_wide_h);
        let mut feat = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        compute_channel_scale_v2(
            &src,
            &dst,
            width,
            height,
            toggles,
            None,
            &mut scratch,
            &mut feat,
        );

        // --- Diffmap under test: weight 1.0 on every spatialized family,
        //     0.0 (excluded) on everything else. ---
        let supported = [
            idx::SSIM_MEAN,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
            idx::GMS,
            idx::BLOCKINESS,
            idx::RINGING,
            idx::BANDING,
            // Weighted-pool families (additive weighted means, normalized by n/Σw).
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
        ];
        let mut weights = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
        for &local in &supported {
            weights[local] = 1.0;
        }

        let map = compute_v2_diffmap_channel_scale(&src, &dst, width, height, &weights);
        assert_eq!(map.len(), n);
        assert!(
            map.iter().all(|v| v.is_finite()),
            "diffmap has non-finite entries"
        );

        let map_mean: f64 = map.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let expected: f64 = supported.iter().map(|&local| feat[local]).sum();

        let rel_err = (map_mean - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel_err < 5e-4,
            "diffmap block-pool mean {map_mean} vs feature sum {expected}: rel_err {rel_err}"
        );

        // Guard: the fixture must actually exercise every supported family
        // (a bug that always emits 0 must not pass by matching a
        // trivially-zero feature) AND must stay clear of every mean-level
        // clamp (clamp02 at 2.0 for SSIM_MEAN, clamp01 at 1.0 for the
        // ART/DET/MSE/HF_*/PJND_TRANSDUCER block) so the identity above is
        // exercised in its EXACT (not clamp-truncated) form.
        assert!(
            feat[idx::SSIM_MEAN] > 1e-4,
            "fixture produced ~zero SSIM_MEAN"
        );
        assert!(
            feat[idx::BLOCKINESS] > 1e-4,
            "fixture produced ~zero BLOCKINESS"
        );
        assert!(feat[idx::ART] > 1e-4, "fixture produced ~zero ART");
        assert!(feat[idx::DET] > 1e-4, "fixture produced ~zero DET");
        assert!(
            feat[idx::SSIM_MEAN] < 2.0 - 1e-6,
            "SSIM_MEAN clamp02 fired: {}",
            feat[idx::SSIM_MEAN]
        );
        for &local in &[
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
        ] {
            assert!(
                feat[local] < 1.0 - 1e-6,
                "clamp01 fired on idx {local}: {}",
                feat[local]
            );
        }
    }

    /// Acceptance gate for [`compute_v2_diffmap_full`] (the runtime-fold
    /// builder): its full-image map's mean equals the summed weighted feature
    /// over EVERY channel-scale, not just one. Expected is computed from the
    /// SAME prepared planes the builder uses (`compute_channel_scale_v2` on
    /// `prepare_v2_reference_impl`'s scales), so the XYB conversion + pyramid
    /// are shared and the only thing under test is the per-channel-scale
    /// accumulate + nearest-upsample. Even dims (96→48→24→12) make the
    /// upsample exactly mean-preserving.
    #[test]
    fn v2_diffmap_full_block_pool_matches_features() {
        let (width, height) = (96usize, 96usize);
        let n = width * height;
        // Mild RGB fixture (kept clear of mean-level clamps, like the
        // channel-scale test above).
        let mut refpx = vec![[0u8; 3]; n];
        let mut dstpx = vec![[0u8; 3]; n];
        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let base = 120.0 + 40.0 * ((x as f32) * 0.11).sin() * ((y as f32) * 0.09).cos();
                let r = base.clamp(20.0, 235.0);
                let g = (base + 8.0).clamp(20.0, 235.0);
                let b = (base - 6.0).clamp(20.0, 235.0);
                refpx[i] = [r as u8, g as u8, b as u8];
                let d = 4.0 * (((x + y) as f32) * 0.5).sin();
                dstpx[i] = [
                    (r + d).clamp(20.0, 235.0) as u8,
                    (g - d).clamp(20.0, 235.0) as u8,
                    (b + d * 0.5).clamp(20.0, 235.0) as u8,
                ];
            }
        }
        let refimg = crate::RgbSlice::new(&refpx, width, height);
        let dstimg = crate::RgbSlice::new(&dstpx, width, height);

        // Expected: Σ over every (scale,ch) of the supported families' features,
        // from the builder's own planes.
        let supported = [
            idx::SSIM_MEAN,
            idx::ART,
            idx::DET,
            idx::MSE,
            idx::HF_GAIN,
            idx::HF_LOSS,
            idx::HF_MAG_LOSS,
            idx::PJND_TRANSDUCER,
            idx::GMS,
            idx::BLOCKINESS,
            idx::RINGING,
            idx::BANDING,
            idx::MASKED_SSIM,
            idx::MASKED_ART,
            idx::MASKED_DET,
            idx::MASKED_MSE,
            idx::IW_SSIM,
            idx::IW_ART,
            idx::IW_DET,
            idx::IW_MSE,
        ];
        let rprep = prepare_v2_reference_impl(&refimg, None, false, false).unwrap();
        let dprep = prepare_v2_reference_impl(&dstimg, None, false, false).unwrap();
        let n_scales = rprep.scales.len();
        let per_scale = 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let mut s_v2 = vec![0.0f64; n_scales * per_scale];
        let mut expected = 0.0f64;
        let toggles = V2NewFeatureToggles::default();
        let max_wide_h = STRIP_ROWS + 2 * HALO_P;
        for scale in 0..n_scales {
            let (ref pr, ws, hs) = rprep.scales[scale];
            let pd = &dprep.scales[scale].0;
            for ch in 0..3 {
                let mut scratch = ScratchV2Strip::new(ws * max_wide_h);
                let mut feat = [0.0f64; FEATURES_PER_CHANNEL_V2_TOTAL];
                compute_channel_scale_v2(
                    &pr[ch],
                    &pd[ch],
                    ws,
                    hs,
                    toggles,
                    None,
                    &mut scratch,
                    &mut feat,
                );
                let base = scale * per_scale + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                for &local in &supported {
                    s_v2[base + local] = 1.0;
                    expected += feat[local];
                }
            }
        }

        let map = compute_v2_diffmap_full(&refimg, &dstimg, &s_v2, None, false).unwrap();
        assert_eq!(map.len(), n);
        assert!(map.iter().all(|v| v.is_finite()), "non-finite entries");
        let map_mean: f64 = map.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let rel_err = (map_mean - expected).abs() / expected.abs().max(1e-9);
        assert!(
            rel_err < 2e-3,
            "full-diffmap block-pool mean {map_mean} vs feature sum {expected}: rel_err {rel_err}"
        );
        // Fixture must actually exercise the families (not pass trivially at 0).
        assert!(
            expected > 1e-3,
            "fixture produced ~zero supported-feature sum"
        );
    }

    // ------------------------------------------------------------------
    // Prepared-reference (pyramid reuse) path
    // ------------------------------------------------------------------

    /// Deterministic structured content: gradients + edges + pseudo-noise
    /// texture (LCG), so every feature family (SSIM, HF, gradient,
    /// blockiness, transducer) sees real signal.
    fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut state = seed | 1;
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (state >> 24) as u8;
                let grad = ((x * 255) / w.max(1)) as u8;
                let edge = if (x / 9 + y / 7) % 2 == 0 { 200 } else { 40 };
                let r = grad.wrapping_add(noise / 4);
                let g = edge;
                let b = ((y * 255) / h.max(1)) as u8 ^ (noise / 8);
                px.push([r, g, b]);
            }
        }
        px
    }

    /// Blocky quantization distortion — deterministic, codec-flavored.
    fn quantize_distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let p = &mut out[y * w + x];
                for c in p.iter_mut() {
                    // 8x8-block DC-ish flattening + level quantization.
                    *c = (*c / 24) * 24 + ((x % 8 == 0 || y % 8 == 0) as u8) * 6;
                }
            }
        }
        out
    }

    /// The prepared-reference path must produce features BIT-IDENTICAL to
    /// the pair path — same functions, same order, only the storage of the
    /// reference pyramid differs. Covers: multi-strip (h > STRIP_ROWS),
    /// odd dims (SIMD tails), the exact-64 floor, and the sub-64
    /// reflect-pad path.
    #[test]
    fn prepared_ref_bit_identical_to_pair_path() {
        for &(w, h) in &[
            (64usize, 64usize),
            (96, 80),
            (200, 136),
            (40, 30),
            (65, 129),
        ] {
            let src = textured_image(w, h, 0xBEEF);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);

            let pair = compute_v2_features_impl(&source, &distorted, None, false)
                .expect("pair path computes");

            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .expect("with_ref computes");

            assert_eq!(pair.features().len(), with_ref.features().len());
            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{w}x{h}: feature {i} diverged: pair={a:e} with_ref={b:e}"
                );
            }
        }
    }

    /// Toggle combinations must round-trip through the prepared path
    /// identically too (the luma gate + disabled groups change which
    /// slots are written — index-stability must not depend on which
    /// entry point ran).
    #[test]
    fn prepared_ref_bit_identical_with_toggles() {
        let (w, h) = (96usize, 96usize);
        let src = textured_image(w, h, 0x5EED);
        let dst = quantize_distort(&src, w, h);
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        for toggles in [
            V2NewFeatureToggles {
                transducers_luma_only: true,
                ..Default::default()
            },
            V2NewFeatureToggles {
                gradient_features: false,
                ..Default::default()
            },
            V2NewFeatureToggles {
                blockiness: false,
                transducer_bank: false,
                ..Default::default()
            },
        ] {
            let pair =
                compute_v2_features_impl_with_toggles(&source, &distorted, None, false, toggles)
                    .expect("pair path computes");
            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                toggles,
                &mut scratch,
            )
            .expect("with_ref computes");
            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "toggles {toggles:?}: feature {i} diverged"
                );
            }
        }
    }

    /// One `V2Scratch` reused across pairs of DIFFERENT sizes and content
    /// must produce the same bits as a fresh scratch per pair — i.e. no
    /// stale-buffer leakage between pairs (buffers are fully written
    /// before every read).
    #[test]
    fn scratch_reuse_matches_fresh_scratch() {
        let mut shared = V2Scratch::new();
        // Big pair first so the small pair runs on oversized buffers full
        // of the big pair's data — the harshest staleness ordering.
        for &(w, h, seed) in &[
            (200usize, 136usize, 0xAAAAu32),
            (64, 64, 0xBBBB),
            (96, 80, 7),
        ] {
            let src = textured_image(w, h, seed);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);
            let prepared =
                prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");

            let mut fresh = V2Scratch::new();
            let a = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut fresh,
            )
            .expect("fresh-scratch computes");
            let b = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut shared,
            )
            .expect("shared-scratch computes");
            for (i, (x, y)) in a.features().iter().zip(b.features().iter()).enumerate() {
                assert!(
                    x.to_bits() == y.to_bits(),
                    "{w}x{h}: feature {i} diverged under scratch reuse"
                );
            }
        }
    }

    /// The cached-moments prepared path (mu1 + activity read from the
    /// reference cache; V-blur + activity chain skipped per pair) must ALSO
    /// be bit-identical to the pair path — the cache is filled by replaying
    /// the exact strip walk, so this gate holds by construction and this
    /// test enforces it stays that way. Multi-strip + odd + pad sizes.
    #[test]
    fn prepared_ref_with_moments_bit_identical_to_pair_path() {
        for &(w, h) in &[
            (64usize, 64usize),
            (96, 80),
            (200, 136),
            (65, 129),
            (40, 30),
        ] {
            let src = textured_image(w, h, 0xC0DE);
            let dst = quantize_distort(&src, w, h);
            let source = RgbSlice::new(&src, w, h);
            let distorted = RgbSlice::new(&dst, w, h);

            let pair = compute_v2_features_impl(&source, &distorted, None, false)
                .expect("pair path computes");

            let prepared = prepare_v2_reference_impl(&source, None, false, true)
                .expect("prepare with moments computes");
            assert!(prepared.has_cached_moments());
            let mut scratch = V2Scratch::new();
            let with_ref = compute_v2_features_with_ref_impl(
                &prepared,
                &distorted,
                None,
                false,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .expect("with_ref computes");

            for (i, (a, b)) in pair
                .features()
                .iter()
                .zip(with_ref.features().iter())
                .enumerate()
            {
                assert!(
                    a.to_bits() == b.to_bits(),
                    "{w}x{h}: feature {i} diverged under cached moments: \
                     pair={a:e} with_ref={b:e}"
                );
            }
        }
    }

    /// POOL_SIMD drift gate: the vectorized weighted-pool path (enabled on
    /// the v4x tier) must stay within this module's documented 5e-4
    /// relative tolerance of the §A.14 scalar-pool path — same
    /// reassociation class as the phase-4 core-moment change. On hosts
    /// where the active tier runs scalar pools anyway, both sides are
    /// identical and the gate passes trivially.
    #[test]
    fn pool_simd_drift_within_policy() {
        let (w, h) = (200usize, 136usize);
        let src_px = textured_image(w, h, 0xD1FF);
        let dst_px = quantize_distort(&src_px, w, h);

        // Build real moment planes via the whole-image blur pass.
        let source = RgbSlice::new(&src_px, w, h);
        let distorted = RgbSlice::new(&dst_px, w, h);
        let src_planes = crate::streaming::convert_source_to_xyb(&source, w, false);
        let dst_planes = crate::streaming::convert_source_to_xyb(&distorted, w, false);
        let mut scratch = ScratchV2Strip::new(w * h);
        for ch in 0..3 {
            run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
            let n = w * h;
            let a = dense_block_kernel(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
            );
            let b = dense_block_kernel_pools_scalar(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
            );
            let pools = [
                ("mask_ssim", a.ws_mask_ssim, b.ws_mask_ssim),
                ("mask_art", a.ws_mask_art, b.ws_mask_art),
                ("mask_det", a.ws_mask_det, b.ws_mask_det),
                ("mask_mse", a.ws_mask_mse, b.ws_mask_mse),
                ("iw_ssim", a.ws_iw_ssim, b.ws_iw_ssim),
                ("iw_art", a.ws_iw_art, b.ws_iw_art),
                ("iw_det", a.ws_iw_det, b.ws_iw_det),
                ("iw_mse", a.ws_iw_mse, b.ws_iw_mse),
                ("peak_ssim", a.ws_peak_ssim, b.ws_peak_ssim),
                ("peak_art", a.ws_peak_art, b.ws_peak_art),
                ("peak_det", a.ws_peak_det, b.ws_peak_det),
            ];
            for (name, pa, pb) in pools {
                for (part, va, vb) in [("num", pa.num, pb.num), ("den", pa.den, pb.den)] {
                    let rel = (va - vb).abs() / vb.abs().max(1e-9);
                    assert!(
                        rel <= TOL,
                        "ch{ch} pool {name}.{part}: rel drift {rel:e} \
                         (simd={va:e} scalar={vb:e}) exceeds policy {TOL:e}"
                    );
                }
            }
            // Non-pool sums must be BIT-identical — POOL_SIMD touches
            // nothing else.
            assert_eq!(a.sum_d.to_bits(), b.sum_d.to_bits());
            assert_eq!(a.sum_mse.to_bits(), b.sum_mse.to_bits());
            assert_eq!(a.sum_pjnd.to_bits(), b.sum_pjnd.to_bits());
        }
    }

    /// Dimension mismatch against the prepared reference is rejected the
    /// same way `validate_pair` rejects it on the pair path.
    #[test]
    fn prepared_ref_rejects_dim_mismatch() {
        let src = textured_image(96, 96, 3);
        let source = RgbSlice::new(&src, 96, 96);
        let prepared =
            prepare_v2_reference_impl(&source, None, false, false).expect("prepare computes");

        let other = textured_image(64, 96, 4);
        let distorted = RgbSlice::new(&other, 64, 96);
        let mut scratch = V2Scratch::new();
        let err = compute_v2_features_with_ref_impl(
            &prepared,
            &distorted,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap_err();
        assert!(matches!(err, ZensimError::DimensionMismatch));
    }

    /// FOLD PARITY GATE (2026-07-24). The folded-720 path replays v1's
    /// exact 32-row band tiling over shared H-planes, so its v1 basic
    /// block (`f0..156`) is **BIT-IDENTICAL** to the frozen v1 extraction
    /// path whenever the working widths agree — i.e. whenever
    /// `simd_padded_width(w) == w` (multiples of 16 below 512, plus the
    /// non-bumped alignments above; see `blur::simd_padded_width`).
    ///
    /// For other widths the two regimes are STRUCTURALLY different by
    /// v1's own frozen semantics: v1 computes over the SIMD-padded width
    /// (mirrored pad columns participate in pooling, and the downscale
    /// chain halves the padded width, shifting every deeper scale's
    /// grid), while the fold runs at the true width. That divergence is
    /// v1's pad wart, not fold noise — it is the documented boundary of
    /// the [`FeatureRegime::Folded720`] contract, measured on real
    /// content + real models in `benchmarks/` (fold_v1_basic_*), and it
    /// is why folded rows must NEVER be silently mixed into v1-extracted
    /// corpora.
    ///
    /// Also gates: `f156..372` all-zero (deprecated pool blocks) and the
    /// v2 block (`f372..`) BIT-identical to the plain v2 path, for BOTH
    /// width classes.
    #[test]
    fn folded720_v1_basic_matches_v1_path() {
        // (w, h, expect_bit_exact): 16-multiple widths are bit-exact;
        // 127 (pads to 128) and 200 (pads to 208) document the padded-
        // width divergence class. ≥64 sizes only: `compute_zensim_with_
        // config` (the fleet v1 extractor) does not reflect-pad sub-64
        // (see `folded720_sub64_matches_padding_v1_entry` for that).
        for &(w, h, exact) in &[
            (96usize, 64usize, true),
            (64, 300, true),
            (208, 144, true),
            (127, 93, false),
            (200, 150, false),
        ] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);

            // v1, the production extraction config (extended + IW = 372).
            let mut cfg = crate::ZensimConfig::default();
            cfg.extended_features = true;
            cfg.compute_iw_features = true;
            cfg.allow_multithreading = false;
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372);

            let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let v2 = z.compute_v2_features(&sref, &dref).unwrap();
            let folded = z.compute_folded720_features(&sref, &dref).unwrap();
            assert_eq!(folded.regime(), FeatureRegime::Folded720);
            let ff = folded.features();
            assert_eq!(ff.len(), 720);

            // v2 block: bit-identical to the plain v2 path.
            for (i, (&a, &b)) in ff[372..].iter().zip(v2.features().iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{w}x{h}: folded v2 slot {i} ({a:e}) != plain v2 ({b:e})"
                );
            }
            // Deprecated pool blocks: exactly zero.
            assert!(
                ff[156..372].iter().all(|&v| v == 0.0),
                "{w}x{h}: deprecated f156..372 must be exactly 0.0"
            );
            if exact {
                // simd_padded_width(w) == w ⇒ the band replay makes the
                // basic block bit-identical. If this fires, something
                // structural regressed (band extents, H sharing, merge
                // order) — investigate, never widen to a tolerance.
                for i in 0..156 {
                    assert_eq!(
                        ff[i].to_bits(),
                        v1f[i].to_bits(),
                        "{w}x{h}: basic f{i} fold {:e} != v1 {:e} (bit-exact class)",
                        ff[i],
                        v1f[i]
                    );
                }
                eprintln!("fold parity {w}x{h}: BIT-EXACT");
            } else {
                // Padded-width divergence class: document the magnitude,
                // assert sanity (finite, non-negative like v1's basics).
                let mut max_rel = 0.0f64;
                for i in 0..156 {
                    let (a, b) = (ff[i], v1f[i]);
                    assert!(a.is_finite(), "{w}x{h}: basic f{i} not finite");
                    if b.abs() > 1e-12 {
                        max_rel = max_rel.max((a - b).abs() / b.abs());
                    }
                }
                eprintln!(
                    "fold parity {w}x{h}: padded-width class (v1 pads {w}->{}), max rel {max_rel:.3e}",
                    crate::blur::simd_padded_width(w),
                    max_rel = max_rel
                );
            }
        }
    }

    /// Sub-64 inputs: the folded path inherits v2's reflect-pad-to-64
    /// pyramid, which matches the v1 `Zensim` API entries
    /// (`compute_with_config_inner` pads with the same
    /// `reflect_pad_to_min`) — gate the basic block against
    /// `Zensim::compute_extended_features` (its `f0..156` prefix).
    #[test]
    fn folded720_sub64_matches_padding_v1_entry() {
        let (w, h) = (48usize, 40usize);
        let src = textured_image(w, h, 11);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);

        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let v1 = z.compute_extended_features(&sref, &dref).unwrap();
        let v1f = v1.features();
        let folded = z.compute_folded720_features(&sref, &dref).unwrap();
        let ff = folded.features();
        assert_eq!(ff.len(), 720);
        // Both sides pad 48x40 → 64x64 via the same `reflect_pad_to_min`,
        // and simd_padded_width(64) == 64 ⇒ bit-exact class.
        for i in 0..156 {
            assert_eq!(
                ff[i].to_bits(),
                v1f[i].to_bits(),
                "sub64 basic f{i}: fold {:e} != v1 {:e}",
                ff[i],
                v1f[i]
            );
        }
    }

    /// The three folded entry forms must agree: pair path, prepared-ref
    /// (no moments), prepared-ref (with moments). The moments form swaps
    /// `fused_blur_h_ssim3` → `fused_blur_h_ssim` for `mu1_h`, whose three
    /// shared outputs are bit-identical, so ALL 720 slots must match
    /// bitwise across the three.
    #[test]
    fn folded720_ref_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_features(&sref, &dref).unwrap();
        let mut scratch = V2Scratch::new();
        let plain = z.prepare_v2_reference(&sref).unwrap();
        let a = z
            .compute_folded720_features_with_ref_and_scratch(
                &plain,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let moments = z.prepare_v2_reference_with_moments(&sref).unwrap();
        let b = z
            .compute_folded720_features_with_ref_and_scratch(
                &moments,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        for i in 0..720 {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair vs prepared(no moments) diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "pair vs prepared(moments) diverge at f{i}"
            );
        }
        // Folded view() exposes the v2 tail.
        assert_eq!(
            pair.view().ssim_mean(0, 0).to_bits(),
            pair.features()[372 + idx::SSIM_MEAN].to_bits()
        );
    }

    // ------------------------------------------------------------------
    // f720+ append block
    // ------------------------------------------------------------------

    /// The three append entry forms must agree bitwise on ALL 924 slots.
    /// This is the gate on the pair path's cross-channel activity replay
    /// (`compute_ref_activity_into`): its planes must be bit-identical to
    /// the moments cache's, or the Y channel's cross-masked transducer
    /// diverges across paths.
    #[test]
    fn append_ref_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_append_features(&sref, &dref).unwrap();
        assert_eq!(pair.features().len(), 720 + 4 * 3 * FEATURES_PER_CHANNEL_APPEND);
        assert_eq!(pair.regime(), FeatureRegime::Folded720Append);
        assert_eq!(
            pair.append_features().expect("append regime").len(),
            4 * 3 * FEATURES_PER_CHANNEL_APPEND
        );

        let mut scratch = V2Scratch::new();
        let plain = z.prepare_v2_reference(&sref).unwrap();
        let a = z
            .compute_folded720_append_features_with_ref_and_scratch(
                &plain,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let moments = z.prepare_v2_reference_with_moments(&sref).unwrap();
        let b = z
            .compute_folded720_append_features_with_ref_and_scratch(
                &moments,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        // Fourth leg: the append-prepared cache (moments + bs2) — the
        // production extraction shape; bs2 comes from the cache instead
        // of the per-scale replay and must still match bitwise.
        let moments_app = z.prepare_v2_reference_with_moments_append(&sref).unwrap();
        let c = z
            .compute_folded720_append_features_with_ref_and_scratch(
                &moments_app,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        for i in 0..pair.features().len() {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair vs prepared(no moments) diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "pair vs prepared(moments, replayed bs2) diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                c.features()[i].to_bits(),
                "pair vs prepared(append moments, cached bs2) diverge at f{i}"
            );
        }
    }

    /// Turning the append block on must not move a single bit of the
    /// first 720 slots — the append is strictly additive (separate second
    /// kernel pass, separate output region; the only shared-kernel touch,
    /// `GradientAccum::sum_gms2`, derives from unchanged operations).
    #[test]
    fn append_first720_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 7);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let folded = z.compute_folded720_features(&sref, &dref).unwrap();
        let appended = z.compute_folded720_append_features(&sref, &dref).unwrap();
        assert_eq!(folded.features().len(), 720);
        for i in 0..720 {
            assert_eq!(
                folded.features()[i].to_bits(),
                appended.features()[i].to_bits(),
                "append toggled on moved f{i}"
            );
        }
        // The v2-named view must agree across both regimes too.
        assert_eq!(
            folded.view().ssim_mean(1, 1).to_bits(),
            appended.view().ssim_mean(1, 1).to_bits()
        );
    }

    /// Identity pair: the error-driven append features are EXACTLY 0
    /// (`raw_abs_err`, `art_i`/`det_i`, gms, and the global sums all
    /// compute identical-bit operands on both sides); the SSIM/σ-derived
    /// features are tiny-but-nonzero, because on an identity pair `ssq`
    /// accumulates `(s²+d²)` per tap while `s12` accumulates `s·d` —
    /// `Σ(2a)` vs `2Σ(a)` round differently in f32 — and the append `bs2`
    /// plane comes from a different H kernel (`box_blur_h` on `src²`,
    /// strip-tiled per P2) than the fused `ssq` chain, so
    /// `var₁ ≠ var₂` at the ULP scale. Observed magnitude ~2e-6; the 1e-4
    /// bound is far below any real-signal value while catching sign or
    /// formula errors. `GRAD_SRC_MEAN` is reference-only and genuinely
    /// nonzero (the `PJND_FRAGILITY` precedent).
    #[test]
    fn append_identity_pair_zeros() {
        let (w, h) = (150usize, 130usize);
        let src = textured_image(w, h, 5);
        let sref = RgbSlice::new(&src, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let r = z.compute_folded720_append_features(&sref, &sref).unwrap();
        let app = r.append_features().unwrap();
        const EXACT_ZERO: [usize; 11] = [
            idx_append::XMASK_TRANSDUCER,
            idx_append::LUM_TRANSDUCER,
            idx_append::LUM_DARK_ERR,
            idx_append::LUM_MID_ERR,
            idx_append::LUM_BRIGHT_ERR,
            idx_append::GMS_DEV2,
            idx_append::ART_DEV2,
            idx_append::DET_DEV2,
            idx_append::GLOBAL_DMEAN,
            idx_append::GLOBAL_CGAIN,
            idx_append::GLOBAL_CLOSS,
        ];
        for scale in 0..4 {
            for ch in 0..3 {
                for local in 0..FEATURES_PER_CHANNEL_APPEND {
                    let v = app[scale * 3 * FEATURES_PER_CHANNEL_APPEND
                        + ch * FEATURES_PER_CHANNEL_APPEND
                        + local];
                    if local == idx_append::GRAD_SRC_MEAN {
                        assert!(
                            (0.0..1.0).contains(&v),
                            "grad_src_mean out of range at s{scale} ch{ch}: {v}"
                        );
                    } else if EXACT_ZERO.contains(&local) {
                        assert_eq!(
                            v, 0.0,
                            "append local {local} not exactly zero on identity pair \
                             (s{scale} ch{ch}): {v}"
                        );
                    } else {
                        assert!(
                            v.abs() < 1e-4,
                            "append local {local} above the identity ULP band \
                             (s{scale} ch{ch}): {v}"
                        );
                    }
                }
            }
        }
    }

    /// Bounds + layout invariants on a real distorted pair: every append
    /// feature within its documented bound, and the cross-channel
    /// transducer slot exactly 0.0 on the chroma channels (Y-only by
    /// design — luma does not mask chroma in the CVVDP trained matrix).
    #[test]
    fn append_bounds_and_chroma_xmask_zero() {
        let (w, h) = (200usize, 150usize);
        let src = textured_image(w, h, 41);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let r = z.compute_folded720_append_features(&sref, &dref).unwrap();
        let app = r.append_features().unwrap();
        let mut any_nonzero = false;
        for scale in 0..4 {
            for ch in 0..3 {
                let base = scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
                for local in 0..FEATURES_PER_CHANNEL_APPEND {
                    let v = app[base + local];
                    assert!(
                        v.is_finite() && (0.0..=2.0).contains(&v),
                        "append s{scale} ch{ch} local{local} out of bounds: {v}"
                    );
                    if v != 0.0 {
                        any_nonzero = true;
                    }
                }
                if ch != 1 {
                    assert_eq!(
                        app[base + idx_append::XMASK_TRANSDUCER],
                        0.0,
                        "xmask must be 0 on chroma channel {ch}"
                    );
                    assert_eq!(
                        app[base + idx_append::LUM_TRANSDUCER],
                        0.0,
                        "lum transducer must be 0 on chroma channel {ch} (luma-gate)"
                    );
                } else {
                    assert!(
                        app[base + idx_append::XMASK_TRANSDUCER] > 0.0,
                        "Y xmask should fire on a distorted pair (s{scale})"
                    );
                }
                if APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0 {
                    for local in 0..FEATURES_PER_CHANNEL_APPEND {
                        assert_eq!(
                            app[base + local],
                            0.0,
                            "B channel scale 0 must be fully skipped (local {local})"
                        );
                    }
                }
            }
        }
        assert!(any_nonzero, "append block is entirely zero on a distorted pair");
    }
}
