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

// ============================================================================
// append2 (f924+) constants — BANDVIS + luminance conditioner + HL bins.
// δ values are EMPIRICAL (test `bandvis_delta_derivation_table` prints the
// measurement; derivation committed in
// `benchmarks/append2_bandvis_gates_2026-07-27.md`). PROVISIONAL until the
// measurement lands — the derivation test brackets them against the live
// front-ends so a front-end change that invalidates them fails loudly.
// ============================================================================

/// BANDVIS band-pass lower half-point, SDR/cbrt route: ≈0.5× the Y-plane
/// |∇| of a one-8-bit-code plateau step at mid-gray.
pub(crate) const BV_DELTA_LO_SDR: f32 = 0.00169;
/// Upper half-point, SDR route: ≈5× the one-code step.
pub(crate) const BV_DELTA_HI_SDR: f32 = 0.0169;
/// BANDVIS lower half-point, HDR/PU route: ≈0.5× the PU-Y |∇| of one
/// 10-bit PQ code step at 100 cd/m² (PU uniformity makes one constant
/// serve all luminances — the derivation table shows the spread).
pub(crate) const BV_DELTA_LO_PU: f32 = 0.00124;
/// Upper half-point, HDR/PU route: ≈5× the one-step gradient.
pub(crate) const BV_DELTA_HI_PU: f32 = 0.0124;
/// Stabilizer for the BANDVIS FR `bounded_excess` pair (indicators are
/// [0,1] products — same class as `C_EDGE`).
pub(crate) const C_BV: f64 = 1e-4;
/// HL bin 1 anchor: the HDR route's Y-plane value of GRAY at 100 cd/m²
/// (SDR white in PU-normalized units) — measured by
/// `bandvis_delta_derivation_table`; the bin weighs error ABOVE SDR white.
pub(crate) const HL1_Y_ANCHOR: f32 = 1.01;
/// HL bin 2 anchor: Y-plane value of gray at 1000 cd/m².
pub(crate) const HL2_Y_ANCHOR: f32 = 1.649;
/// Softness half-point for the HL soft steps `u/(u+c)`, `u = max(y − anchor, 0)`
/// — ½ the measured anchor spacing (1.649 − 1.010 ≈ 0.64) so the two bins
/// overlap smoothly (E2 partition caveat: they also overlap the existing
/// bright bin — trainers must treat the three as non-orthogonal).
pub(crate) const C_HL: f64 = 0.32;

/// append2 slots per scale (Y-only block — no channel axis; layout
/// `f924 + scale*APPEND2_PER_SCALE + local`, 4 scales × 5 = 20 slots,
/// full vector = 944). Documented deviation from the append block's
/// scale-major×channel layout: every append2 signal is Y-only by
/// design, so a channel axis would be 2/3 structural zeros.
pub const APPEND2_PER_SCALE: usize = 5;

/// The one channel the append2 block is computed on (Y). The block has no
/// channel axis in its LAYOUT (see [`APPEND2_PER_SCALE`]), but its signals
/// are accumulated from the Y-channel kernels, so the attribution density
/// contributes on this channel only.
///
/// Layout metadata, so it stays compiled in every configuration — but its
/// only non-test reader is the `custom-profiles`-gated attribution-density
/// cluster, hence the targeted dead-code allowance.
#[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
pub(crate) const APPEND2_CHANNEL: usize = 1;

/// Named local offsets within one scale's append2 block (all Y-only).
pub mod idx_append2 {
    /// BANDVIS banding-visibility GAIN (gaps-doc §6b design, 2026-07-27;
    /// operator revised to CURVATURE during validation): per-pixel
    /// `band(|∇²Y|; δ_lo, δ_hi) · (1 − sat(act, C_ACTIVITY))` indicators
    /// on src and dst (`band(g) = sat(g, δ_lo)·(1 − sat(g, δ_hi))` —
    /// soft band-pass over the SECOND-difference magnitude, half-points
    /// at the EMPIRICAL one-code-step constants `BV_DELTA_*`, per
    /// route). Second differences, not first: |∇²| of a linear gradient
    /// is exactly 0 at any steepness, so smooth ramps stay out of the
    /// band at every scale (the first-difference form measured
    /// polarity-inverted on ramp fixtures — sub-step smooth gradients
    /// were indistinguishable from steps); a plateau step reports the
    /// full step at its flanking pixels, so the δ derivation is
    /// operator-independent. FR
    /// `bounded_excess(b_dst, b_src, C_BV)` mean-pooled. "NEW visible
    /// steps in flat regions" — the CAMBI job on existing machinery; the
    /// pyramid supplies the window sweep; the activity term supplies
    /// dither masking. Foldable (plain mean).
    ///
    /// MEASURED MECHANICS (2026-07-27 behavioral run — load-bearing for
    /// consumers): this is a COARSE-SCALE detector. At scale 0/1 a
    /// realistic gradient's own per-pixel |∇| sits inside the visibility
    /// band (a smooth ramp shallow enough to band is still ≥ ~¼ code per
    /// 2-px span), so the FR excess largely cancels; each downscale
    /// doubles smooth-gradient |∇| OUT of the band while plateau steps
    /// persist, so scales 2–3 are where posterization fires (exactly
    /// where CAMBI's 63-px windows land vs our radius-5 support). The
    /// scale-0/1 slots are kept index-stable; expect near-zero there.
    /// Steps beyond ~5 8-bit codes attenuate BY DESIGN (the band's upper
    /// edge — CAMBI's own contrast cap is 4 ten-bit levels; giant steps
    /// are edges, GMS/blockiness territory). Each step size therefore has
    /// a RESONANT scale (finer steps → finer scale); consumers should
    /// read the 4 per-scale slots as a response CURVE, not redundant
    /// copies (measured ladder matrix:
    /// `benchmarks/append2_bandvis_gates_2026-07-27.md`).
    ///
    /// OPT-IN VARIANT (`V2NewFeatureToggles::append2_dst_activity`,
    /// default OFF — 2026-08-02 P1.5 adjudication): GAIN becomes the
    /// pure-band FR excess pooled under the DST's own flatness weight
    /// (dst-activity plane). Production extraction runs with it OFF
    /// (adjudicated); rows extracted with it ON are a different feature
    /// definition for this slot — never column-mix. Record:
    /// `benchmarks/bandvis_dst_activity_2026-08-02.md`.
    pub const BANDVIS_GAIN: usize = 0;
    /// Reverse polarity: visible steps REMOVED (debanding credit).
    pub const BANDVIS_LOSS: usize = 1;
    /// Reference-only local-adaptation conditioner: `sat(mean(ref Y),
    /// C_LUM_T)` from the append kernel's existing `sum_s` accumulator
    /// (finalize-only — FREE). Correct-0 in any steering fold, like
    /// `PJND_FRAGILITY`/`GRAD_SRC_MEAN`.
    pub const LUMA_MEAN_REF: usize = 2;
    /// HDR-route-gated highlight bin 1: `Σw·mse_i/Σw` with
    /// `w = sat(max(y_ref − HL1_Y_ANCHOR, 0), C_HL)` — error weighted
    /// ABOVE SDR white (anchor = measured PU-Y of gray at 100 cd/m²).
    /// 0.0 on the SDR route (index-stable). E2 PARTITION CAVEAT: these
    /// bins overlap each other AND the existing `LUM_BRIGHT_ERR` bright
    /// bin — non-orthogonal by design; trainers must not treat the
    /// luminance bins as a partition of unity once append2 is on.
    pub const HL_BIN1: usize = 3;
    /// Highlight bin 2: anchor = measured PU-Y of gray at 1000 cd/m².
    pub const HL_BIN2: usize = 4;
}

// ============================================================================
// CSFW (f944+) constants — chunk-3 luminance-CSF tier-1: Y-only
// luminance-weighted GLOBAL_* pooling lanes
// (`docs/CSF_CHUNK3_DESIGN_2026-07-28.md`; coordinator descope 2026-07-28:
// tier-1 ships the 12 achromatic lanes f944..f955 only — the chroma tiers
// keep the f956..f979 claim). The φ quadratics are DERIVED AND FROZEN, not
// fitted: castleCSF's published achromatic-sustained luminance sensitivity
// (Eq. 21, constants 56.49/7.547/0.1445/5.583e−7/9.669e9) divided by each
// route's own encoding derivative (cube root on the SDR route; PU21
// `banding_glare` from `pu21.rs` on the HDR route), normalized at the
// route anchor (sRGB code 128 = 43.73 cd/m² SDR; 100 cd/m² HDR), quadratic
// least-squares in the ENCODED coordinate over codes 4–255 (SDR) /
// L ∈ [1, 4000] cd/m² (HDR — below ~0.5 cd/m² PU21's glare term and
// castleCSF's glare-free model disagree; the clamp owns that region).
// `csfw_phi_derivation_table` recomputes the derivation against the LIVE
// front-ends and brackets these constants — the `bandvis_delta_derivation_
// table` pattern that keeps them honest.
// ============================================================================

/// SDR-route achromatic weight quadratic `φ_Y(y) = c0 + c1·y + c2·y²`,
/// with `y` the LIVE reference Y-plane value
/// (`cbrt(rel + β) − cbrt(β) + 0.01`, β the opsin bias — measured through
/// `srgb_to_positive_xyb_planar_into`). LSQ over sRGB codes 4–253, this
/// refit: rms 0.092, max 0.431 (dark tail). Implementation-found
/// deviation from the design doc's §5.3 table: those values were fitted
/// in an idealized bias-free `cbrt(rel)` coordinate that the live plane
/// is NOT an affine map of (the opsin bias regularizes the doc's
/// dark-tail derivative collapse); §6's pre-composition rule resolves
/// in favor of the live coordinate, and `csfw_phi_derivation_table`
/// recomputes this exact fit and pins the constants to it.
pub(crate) const CSFW_PHI_Y_SDR: [f64; 3] = [1.77430, -5.81908, 4.04916];
/// HDR/PU-route achromatic weight quadratic, `y` the LIVE reference
/// PU-Y plane value (`pu21(mix)/PU_WHITE + 0.01`). LSQ over L ∈
/// [1, 4000] cd/m² log-uniform, this refit: rms 0.082, max 0.329 —
/// within 0.016 of the design §5.3 values (the doc's PU coordinate was
/// already live-accurate up to the +0.01 bias).
pub(crate) const CSFW_PHI_Y_PU: [f64; 3] = [0.78830, -1.10402, 0.30460];
/// Fitted amplitude for the achromatic luminance modulation — SEED 1.0
/// (design §5.3: `κ = 1` applies the derived castleCSF curve exactly;
/// `κ = 0` reproduces the unweighted features bit-for-bit). Stage-1/2
/// calibration (§9.2) adjusts this from measurement, never by hand.
pub(crate) const CSFW_KAPPA_Y: f64 = 1.0;
/// Per-band (per-scale) strength multipliers `λ_b`, achromatic only —
/// the published chromatic peak frequencies do NOT move with luminance
/// (castleCSF Eq. 24 discussion; design §3.3). `λ_2 ≡ 1` is the
/// identifiability anchor; all SEED 1.0. Pre-registered expectation for
/// the stage-1 fit: increasing toward finer scales (falsifier 2).
pub(crate) const CSFW_LAMBDA_B: [f64; 4] = [1.0, 1.0, 1.0, 1.0];
/// Weight clamp bounds (design §5.3): bracket every derived curve (SDR
/// span 2.96×, HDR chroma 2.5×) and own the sub-0.5 cd/m² glare
/// disagreement + the fit's dark-tail extrapolation. No lane can be
/// annihilated (§3.2: supra-threshold flattening — MAD's premise).
pub(crate) const CSFW_W_MIN: f64 = 0.25;
/// Upper weight clamp bound (see [`CSFW_W_MIN`]).
pub(crate) const CSFW_W_MAX: f64 = 4.0;

/// CSFW slots per scale (Y-only tier-1 block — no channel axis, same
/// documented layout deviation as append2; layout
/// `f944 + scale*CSFW_PER_SCALE + local`, 4 scales × 3 = 12 slots, full
/// vector = 956). The chroma tiers (X/B weighted twins, design §8.1)
/// keep the f956..f979 claim and are NOT emitted by this wave.
pub const CSFW_PER_SCALE: usize = 3;

/// Named local offsets within one scale's CSFW block (all Y-only).
///
/// Each lane is the luminance-CSF-weighted twin of the same scale's
/// unweighted Y-channel [`idx_append`] GLOBAL_* statistic: the pooling
/// mean `Σv/n` becomes `Σw·v/Σw` with the per-pixel REF-side weight
/// `w(y) = clamp(1 + κ_Y·λ_b·φ_Y(y), CSFW_W_MIN, CSFW_W_MAX)` evaluated
/// at the reference Y-plane value (the same per-pixel `ref_y` the HL
/// bins read). Same constants (`C_GDMEAN`/`C_GCONTRAST`), same clamps,
/// same bounds as the unweighted twins. E2-CLASS CAVEAT: the weighted
/// twins are correlated with their unweighted originals BY CONSTRUCTION
/// (`κ→0` makes them identical) — trainers must treat each (weighted,
/// unweighted) pair as non-orthogonal; LOO on a 956 bake adjudicates
/// slot-worth, exactly the BANDVIS discipline.
pub mod idx_csfw {
    /// Luminance-weighted global mean shift:
    /// `sat(|Σw·s − Σw·d| / Σw, C_GDMEAN)` — the weighted twin of
    /// [`super::idx_append::GLOBAL_DMEAN`] (Y). V3's worst cross-route
    /// diverging lane family (SROCC 0.49–0.85,
    /// `benchmarks/hdr_streaming_gates_2026-07-27.md`): a mean shift's
    /// visibility depends on WHERE in the tonal range it lands, and the
    /// cbrt/PU encodings weight that differently — the route-common
    /// physical weight is the fix (design §5.1).
    pub const W_GLOBAL_DMEAN: usize = 0;
    /// Luminance-weighted global contrast gain:
    /// `bounded_excess(gvar₂ʷ, gvar₁ʷ, C_GCONTRAST)` on the
    /// weighted-population variances `gvarᵢʷ = Σw·xᵢ²/Σw − (Σw·xᵢ/Σw)²`
    /// — weighted twin of [`super::idx_append::GLOBAL_CGAIN`] (Y).
    pub const W_GLOBAL_CGAIN: usize = 1;
    /// Reverse polarity of [`W_GLOBAL_CGAIN`] — weighted twin of
    /// [`super::idx_append::GLOBAL_CLOSS`] (Y).
    pub const W_GLOBAL_CLOSS: usize = 2;
}

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

/// The COARSEST row-group size any `fused_blur_h_ssim_*` kernel walks in.
///
/// The kernels transpose a group of rows into lanes and run one group at a
/// time plus a partial tail: **16 rows on `v4` and `v4x`**
/// (`blur.rs` `let row_groups = height / 16`), **8 on `v3`, `neon`, `wasm128`
/// and `scalar`** (`run_group(rg * 8, 8)`). Band boundaries in
/// [`fused_blur_h_ssim_banded`] are multiples of this coarsest size, which is
/// a multiple of the other, so on EVERY tier a band's grouping is a
/// sub-sequence of the whole-plane call's: same group size, same offset within
/// it, and only the final band can hold the partial group the whole-plane
/// call's tail would have run.
///
/// (`phase_a_blur_bands_are_bit_exact` additionally MEASURES agreement at band
/// sizes that are not multiples of either — the per-row independence
/// `box_blur_h_ring_matches_regathered_reference` pins is stronger than the
/// alignment argument. The alignment is what makes the claim hold without
/// depending on that, on tiers this box cannot run.)
pub(crate) const H_BLUR_ROW_GROUP: usize = 16;

/// Rows per phase-A H-blur band. A multiple of [`H_BLUR_ROW_GROUP`] — the
/// bit-exactness precondition — and the smallest such value, which is what
/// measurement wanted: at 16 rows a band's four output planes fit L2, and
/// phase-A BUSY time FELL 90.2 → 46.2 ms at 2304²/16T when the split landed,
/// so the bands are cheaper in total work as well as wider.
/// `benchmarks/fold_mt_scaling_2026-08-31.md`.
pub(crate) const H_BLUR_BAND_ROWS: usize = 16;
const _: () = assert!(H_BLUR_BAND_ROWS % H_BLUR_ROW_GROUP == 0);

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

/// Vectorized MSCN divisive normalizer — `resid / sqrt(var + c)`.
///
/// Deliberately IEEE `sqrt` + `div` (both correctly rounded on every vendor
/// and every tier), NOT `resid * (var + c).rsqrt()`. Under the pinned
/// magetypes 0.9.28 contract `rsqrt()` is the hardware estimate
/// (`vrsqrtps`, whose seed table is CPU-VENDOR-specific) plus one
/// Newton-Raphson step, which leaves a ~1e-8 rel vendor residue — the
/// AMD-vs-Intel divergence measured on exactly the `MSCN_DIFF_MEAN` /
/// `MSCN_DIFF_L2` append slots in the bf944 wave (imazen/zensim#56,
/// `benchmarks/backfill944_bigcodec_2026-08-02.md`); on NEON it is
/// `1/sqrt` then a multiply (double rounding), a third distinct result. The
/// scalar tail, the f64 reference in `attr_pass_b_main_px`, and the
/// attribution pass-B SIMD kernel all use this exact single-rounding form
/// already; this makes the production kernel match. Gated bit-exact against
/// the scalar IEEE result on every tier by
/// `mscn_norm_v_is_correctly_rounded_on_every_tier`.
#[inline(always)]
fn mscn_norm_v<T: F32x8Backend + Copy>(_token: T, resid: V8<T>, var: V8<T>, c: V8<T>) -> V8<T> {
    resid / (var + c).sqrt()
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
#[cfg_attr(not(test), allow(dead_code))] // test-referenced reference impl (raw_moment_reformulation_matches_terriberry)
struct OnlineMoments {
    n: f64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

#[cfg_attr(not(test), allow(dead_code))] // see struct note: exercised by the Terriberry parity test
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
    /// [`Folded720Append`](Self::Folded720Append) plus the f924+ append2
    /// block ([`idx_append2`]; 924 + 4×[`APPEND2_PER_SCALE`] = 944 slots).
    /// Additive-only and default-OFF — new-regime rows only (the HDR
    /// backfill wave); never mixed into 924-regime tables.
    Folded720Append2,
    /// [`Folded720Append2`](Self::Folded720Append2) plus the f944+ CSFW
    /// block ([`idx_csfw`]; 944 + 4×[`CSFW_PER_SCALE`] = 956 slots —
    /// chunk-3 tier-1, Y-only luminance-CSF-weighted GLOBAL_* twins).
    /// Additive-only and default-OFF — new-regime rows only (the HDR
    /// backfill wave); never mixed into 944- or 924-regime tables. The
    /// chroma tiers (f956..f979) are a later wave.
    Folded720Csfw,
}

/// Result of [`compute_v2_features_impl`]/[`crate::Zensim::compute_v2_features`].
#[derive(Debug, Clone)]
pub struct ZensimV2Result {
    features: Vec<f64>,
    n_scales: usize,
    regime: FeatureRegime,
    /// `V2NewFeatureToggles::v1_pools`: which of `f156..372` carry v1's
    /// live pool values (a distinct extraction regime per mode at the SAME
    /// layout width — see the toggle's doc).
    v1_pools: V1PoolsMode,
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
    /// Which of `f156..372` hold v1's LIVE pool values (the
    /// [`V2NewFeatureToggles::v1_pools`] regimes) rather than the folded
    /// walk's structural zeros. Same layout width either way — this is the
    /// regime-purity marker for that block.
    pub fn v1_pools(&self) -> V1PoolsMode {
        self.v1_pools
    }
    /// `v1_pools() != Off`.
    pub fn v1_pools_live(&self) -> bool {
        self.v1_pools != V1PoolsMode::Off
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
            FeatureRegime::Folded720Append2 => {
                let tail = self.n_scales * (3 * FEATURES_PER_CHANNEL_APPEND + APPEND2_PER_SCALE);
                let end = self.features.len() - tail;
                &self.features[end - v2_len..end]
            }
            FeatureRegime::Folded720Csfw => {
                let tail = self.n_scales
                    * (3 * FEATURES_PER_CHANNEL_APPEND + APPEND2_PER_SCALE + CSFW_PER_SCALE);
                let end = self.features.len() - tail;
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
            FeatureRegime::Folded720Append2 => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let start = self.features.len() - append_len - self.n_scales * APPEND2_PER_SCALE;
                Some(&self.features[start..start + append_len])
            }
            FeatureRegime::Folded720Csfw => {
                let append_len = self.n_scales * 3 * FEATURES_PER_CHANNEL_APPEND;
                let start = self.features.len()
                    - append_len
                    - self.n_scales * (APPEND2_PER_SCALE + CSFW_PER_SCALE);
                Some(&self.features[start..start + append_len])
            }
            _ => None,
        }
    }

    /// The f924+ append2 slots (`n_scales × APPEND2_PER_SCALE`), when the
    /// result carries them ([`FeatureRegime::Folded720Append2`] or
    /// [`FeatureRegime::Folded720Csfw`]).
    pub fn append2_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Append2 => {
                let len = self.n_scales * APPEND2_PER_SCALE;
                Some(&self.features[self.features.len() - len..])
            }
            FeatureRegime::Folded720Csfw => {
                let len = self.n_scales * APPEND2_PER_SCALE;
                let start = self.features.len() - len - self.n_scales * CSFW_PER_SCALE;
                Some(&self.features[start..start + len])
            }
            _ => None,
        }
    }

    /// The f944+ CSFW slots (`n_scales × CSFW_PER_SCALE`), when the
    /// result carries them ([`FeatureRegime::Folded720Csfw`]).
    pub fn csfw_features(&self) -> Option<&[f64]> {
        match self.regime {
            FeatureRegime::Folded720Csfw => {
                let len = self.n_scales * CSFW_PER_SCALE;
                Some(&self.features[self.features.len() - len..])
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

/// How a declared-HDR source's pixel values map to absolute display light
/// (cd/m²) on the folded/append STREAMING HDR route (HDR_PLAN chunk 2).
/// The declaration is explicit at the entry — the route never guesses a
/// transfer from pixel bytes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HdrEncoding {
    /// Pixels are already absolute linear light in cd/m² (the
    /// [`crate::Zensim::compute_pu_linear`] contract; e.g. decoded EXR).
    Linear,
    /// Pixels are PQ (SMPTE ST 2084) code values in `[0, 1]`, decoded per
    /// channel through the display model (`min(EOTF_PQ, peak) + black +
    /// reflection`). `peak_nits = 10000.0` decodes at spec peak with no
    /// display clamp; `1000.0` matches pycvvdp's `standard_hdr_pq`.
    Pq {
        /// Display peak luminance, cd/m².
        peak_nits: f32,
    },
    /// Pixels are HLG (ITU-R BT.2100) signal values in `[0, 1]`, decoded
    /// through the reference OOTF (`F_D = peak·Y_s^(γ−1)·E_s`) with the
    /// BT.2100 system gamma for `peak_nits`/`ambient_lux`.
    Hlg {
        /// Display peak luminance, cd/m².
        peak_nits: f32,
        /// Ambient illuminance, lux (5.0 = reference environment).
        ambient_lux: f32,
    },
}

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
    /// Emit the f924+ append2 block ([`idx_append2`]: BANDVIS gain/loss,
    /// the free luminance conditioner, HDR-gated highlight bins —
    /// `APPEND2_PER_SCALE` Y-only slots per scale). Default OFF: with
    /// this false every existing path, layout, and byte — SDR and HDR
    /// routes — is unchanged. Requires `append_block` (asserted): the
    /// block sits at f924+ and reuses append accumulators. Additive-only;
    /// joins the NEXT extraction regime wave (the HDR backfill).
    pub append2_block: bool,
    /// Emit the f944+ CSFW block ([`idx_csfw`]: chunk-3 tier-1 —
    /// luminance-CSF-weighted GLOBAL_* pooling twins, `CSFW_PER_SCALE`
    /// Y-only slots per scale, both routes with route-local derived φ
    /// constants). Default OFF: with this false the CSFW pass never runs
    /// and every existing path, layout, and byte — SDR and HDR routes —
    /// is unchanged. Requires `append2_block` (asserted): the block sits
    /// at f944+ after append2. Additive-only; joins the NEXT extraction
    /// regime wave (the HDR backfill).
    pub csfw_block: bool,
    /// BANDVIS dst-activity plane (the recorded V3(b)/(c) cross-fire fix
    /// from `benchmarks/append2_bandvis_gates_2026-07-27.md` REMAINDERS
    /// #3, implemented + adjudicated 2026-08-02 for the SOTA-944 P1.5
    /// sequencing decision): compute a distorted-side activity plane
    /// (`box_blur(|dst − mu2|)` — the exact dst twin of the existing
    /// ref-side `activity` chain, Y channel only, ≈+5%-class cost) and
    /// re-weight the BANDVIS **GAIN** polarity by the dst's own local
    /// flatness as a per-pixel POOLING weight:
    /// `gain_px = excess(band_dst, band_src)·(1 − sat(act_dst, C_ACTIVITY))`.
    /// The weight sits OUTSIDE the FR ratio — the adjudication measured
    /// that a flatness mask INSIDE `bounded_excess` is ratio-cancelled
    /// (scale-invariance) and suppresses real banding MORE than the
    /// dither it targets. **LOSS stays bit-identical to the OFF math**:
    /// the dst/src-side weights both measured direction-inverting on the
    /// deband credit, and LOSS is the LYB-validated workhorse polarity.
    /// Identity pairs stay exactly 0 either way.
    ///
    /// ADJUDICATION VERDICT (2026-08-02): both pre-registered masking
    /// arms FAILED their suppression gates at the resonant scale (banding
    /// contours ARE local activity — the plane cannot separate sparse
    /// contours from dense texture); production extraction (bigcodec +
    /// every P1 backfill) runs with this OFF. The toggle is retained as
    /// the P3/LOO research surface: the shipped GAIN combine is the
    /// strongest candidate measured (geometry cross-fire 0.33×, deband
    /// margin 2.2× OFF's). Default OFF: no dst-activity plane is
    /// computed and every path/byte — both routes, all regimes — is
    /// unchanged (const-split kernel; the OFF instantiations emit
    /// today's exact operation sequences). When ON, ONLY the four
    /// per-scale `idx_append2::BANDVIS_GAIN` slots change; every other
    /// slot INCLUDING `BANDVIS_LOSS` stays bit-identical. Requires
    /// `append2_block` (asserted).
    /// Record: `benchmarks/bandvis_dst_activity_2026-08-02.md`.
    pub append2_dst_activity: bool,
    /// Emit v1's peak / masked / IW pool blocks (`f156..372`) LIVE inside
    /// the folded walk instead of the structural zeros (2026-08-30, the
    /// carrier lane: `benchmarks/balance_campaign_2026-08-28.md` "carriers
    /// named + costed" → "un-zero the native slots under a regime flag").
    /// Per v1-aligned band the fold hook replays v1's extended strip
    /// section — see [`fold_v1_basic_bands`] — so at
    /// `pyramid_plane_stride(w) == w` the block is BIT-IDENTICAL to v1's 372
    /// extraction (`folded720_v1_pools_match_v1_path`). Default OFF: rows
    /// with the block live are their OWN extraction regime — never
    /// column-mix them with zeroed-block folded rows
    /// (`ZensimV2Result::v1_pools`; extractor modes `foldapp2carriers` /
    /// `foldapp2pools`). [`V1PoolsMode::Carriers`] emits ONLY the ten
    /// carrier slots the `fused944native` tables carry (the peaks are free;
    /// the masked/IW art-L4 slots need the activity map + the fused edge
    /// kernel at scales 0-1 only); [`V1PoolsMode::Full`] emits all 216.
    /// MEASURED (zenbench paired, `benches/fold_pools_bench.rs`): the full
    /// block is NOT free — +25-32% @576², +34-40% @1152² over the zeroed
    /// fold; see the campaign ledger for the carriers-only cost.
    pub v1_pools: V1PoolsMode,
    /// **TEST/BENCH INSTRUMENTATION — NOT A PRODUCT MODE** (`#[doc(hidden)]`
    /// 2026-08-30 by user decision: 944 with all pools live is
    /// the ONLY product mode; there is no 372-only product path and no public
    /// or CLI surface offers one). Retained solely as the control arm for
    /// `folded_v1_only_matches_full_walk` and for pricing what the v2-era
    /// blocks cost inside the one product walk.
    ///
    /// It stays `pub` only because the struct is constructed by external
    /// crates with `..Default::default()`, and functional record update
    /// requires every field to be visible — making it `pub(crate)` breaks
    /// every out-of-crate constructor, zenmetrics included. `#[doc(hidden)]`
    /// takes it off the documented surface, which is as far as this can go
    /// without a builder-style redesign of the struct (listed for approval).
    ///
    /// BLOCK-SKIPPING (2026-08-30): compute ONLY v1's blocks — `f0..156`
    /// basic plus, with [`Self::v1_pools`], `f156..372` pools — and skip
    /// every v2-era block (dense-348, gradient, append, append2/BANDVIS,
    /// CSFW). Skips their upstream work too: phase A drops the four
    /// `box_blur_v_from_copy` sweeps that produce the V-blurred
    /// `mu1/mu2/ssq/s12` strips AND the v2 activity chain, because nothing
    /// v1 needs reads them (`fold_v1_basic_bands` takes the H-blurred
    /// planes and computes its own activity internally).
    ///
    /// PURE COMPUTE-SKIPPING — the slots that ARE emitted are bit-identical
    /// to the same request with this off (gated by
    /// `folded_v1_only_matches_full_walk`). Semantics are untouched; this
    /// is not the padded-width question.
    ///
    /// The emitted vector keeps the requesting regime's WIDTH, with
    /// `f372..` left at the structural 0.0 — the final 372-width plumbing
    /// waits on the pad-semantics decision (campaign §8), so that this
    /// lever can be measured and shipped independently of it.
    #[doc(hidden)]
    pub v1_only: bool,
}

/// Which of v1's pool slots (`f156..372`) the folded walk emits live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum V1PoolsMode {
    /// Structural zeros (the production folded regimes).
    #[default]
    Off,
    /// The ten carrier slots of the `fused944native` regime — `art_l8` at
    /// (s1,c0) (s1,c2) (s2,c0) (s3,c2) = f178/190/196/226,
    /// `masked_art_4th` at s0 c0-2 = f231/237/243, `iw_art_4th` at (s0,c0)
    /// (s1,c0) (s1,c2) = f303/321/333 — every other pool slot stays 0.
    Carriers,
    /// **The peak block only** (`f156..228`) — masked and IW stay 0.
    ///
    /// Costs the SAME as [`Self::Off`]: the peak accumulators (`ssim_d8`,
    /// `edge_art8`, `edge_det8` and the three running maxima) are produced
    /// UNCONDITIONALLY by `fused::fused_vblur_features_ssim` on every path
    /// and merged unconditionally by `V1BasicSums::accumulate`, so `Off` has
    /// been paying for them all along and simply declined to emit them. What
    /// this mode skips relative to [`Self::Full`] is the masked/IW pass
    /// group — the band activity chain (`abs_diff_into` +
    /// `box_blur_1pass_into`), the fused kernel's `store_mu`/`store_sigma`
    /// side-outputs, and the three `*_inline_both` kernels — which is the
    /// whole measurable cost of the pool block.
    ///
    /// The band scratch is still handed to the band so the memory-traffic
    /// self-blur shape (`FoldHSource::SelfBlur`) stays available; it owns
    /// only the four H planes there, never the pool planes.
    Peaks,
    /// All 216 slots: peaks (72) + masked (72) + IW (72), v1-exact.
    Full,
}

impl V1PoolsMode {
    /// The ten carrier slots (`Carriers`), in v1's 372 layout.
    pub const CARRIER_SLOTS: [usize; 10] = [178, 190, 196, 226, 231, 237, 243, 303, 321, 333];
}
impl Default for V2NewFeatureToggles {
    fn default() -> Self {
        Self {
            gradient_features: true,
            transducer_bank: true,
            blockiness: true,
            transducers_luma_only: false,
            append_block: false,
            append2_block: false,
            csfw_block: false,
            append2_dst_activity: false,
            v1_pools: V1PoolsMode::Off,
            v1_only: false,
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
/// numerical risk (and zero risk to v1's golden gate, since v1 never
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
    /// Streamed-walk σ-split plane: this kernel strip's `blur(src²)` wide
    /// buffer, filled per strip by the streaming walk (square + H + V
    /// inside the wide window — the P2 tiling) after the activity chain
    /// frees `abs_src`/`activity_tmp` for reuse as its temps. Unused by
    /// the materialized walk (which reads cached/replayed full planes).
    bs2: Vec<f32>,
    /// Distorted-side activity plane (`box_blur(|dst − mu2|)`) for the
    /// BANDVIS dst self-mask ([`V2NewFeatureToggles::append2_dst_activity`],
    /// Y channel only). LAZILY grown — allocated empty so the default-OFF
    /// heap profile (12 MP heaptrack 221.04 MB, append2 gates V5) is
    /// bit-for-bit unchanged; sized on first use by `stream_phase_a`.
    activity_dst: Vec<f32>,
}
/// Which groups of [`ScratchV2Strip`]'s planes a walk will actually write.
///
/// The set is a property of the REQUEST, and the fold-footprint lane measured
/// what ignoring it costs: a `v1_only` + [`V1PoolsMode::Full`] score — the
/// fold-backed scoring walk — runs self-blur bands, so phase A never executes
/// and the ONLY planes it writes are `src_wide` / `dst_wide`. The other twelve
/// were still allocated at `3 × 148 × width × 4` bytes apiece: **21,312·width
/// bytes untouched**, 24.6 MB at 1152² and 49.1 MB at 2304².
///
/// Under stock glibc those pages are demand-zero (`vec![0.0; n]` →
/// `alloc_zeroed` → `mmap`) and never fault, so they cost address space rather
/// than RSS — but that is an ALLOCATOR POLICY, not a property of the program.
/// MEASURED: with `MALLOC_ARENA_MAX=1` the same binary's `score_fold` peak RSS
/// at 1152²/1T goes 61,952 KiB → 71,768 KiB, because the allocation now comes
/// from the main heap and `calloc` has to memset it. Not asking for the planes
/// is the only way to not depend on that.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct StripPlaneNeeds {
    /// `mu1_h` / `mu2_h` / `ssq_h` / `s12_h` — phase A's fused-H outputs.
    /// False only when every v1 band blurs its own rows ([`FoldHSource::SelfBlur`]).
    h: bool,
    /// `mu1` / `mu2` / `ssq` / `s12` / `abs_src` / `activity_tmp` / `activity`
    /// / `bs2` — everything below `run_blur_pass_inner`'s `want_v2` early
    /// return, plus the σ-split and dst-activity buffers that reuse its temps.
    v2: bool,
}

impl StripPlaneNeeds {
    /// Every plane — what the materialized walks and the reference-moment
    /// cache ask for, and the safe default for any new caller.
    const ALL: Self = Self { h: true, v2: true };
    const NONE: Self = Self {
        h: false,
        v2: false,
    };
    fn union(self, o: Self) -> Self {
        Self {
            h: self.h || o.h,
            v2: self.v2 || o.v2,
        }
    }
}

impl ScratchV2Strip {
    fn new(max_n: usize) -> Self {
        Self::new_for(max_n, StripPlaneNeeds::ALL)
    }

    fn new_for(max_n: usize, needs: StripPlaneNeeds) -> Self {
        let hn = if needs.h { max_n } else { 0 };
        let vn = if needs.v2 { max_n } else { 0 };
        Self {
            src_wide: vec![0.0f32; max_n],
            dst_wide: vec![0.0f32; max_n],
            mu1_h: vec![0.0f32; hn],
            mu2_h: vec![0.0f32; hn],
            ssq_h: vec![0.0f32; hn],
            s12_h: vec![0.0f32; hn],
            mu1: vec![0.0f32; vn],
            mu2: vec![0.0f32; vn],
            ssq: vec![0.0f32; vn],
            s12: vec![0.0f32; vn],
            abs_src: vec![0.0f32; vn],
            activity_tmp: vec![0.0f32; vn],
            activity: vec![0.0f32; vn],
            bs2: vec![0.0f32; vn],
            activity_dst: Vec::new(),
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
    /// The UNION of every plane set asked for so far — grow-only, exactly like
    /// `sized_for`, so a driver that alternates a v1-only score with a 944
    /// extraction converges to [`StripPlaneNeeds::ALL`] and never thrashes.
    sized_needs: StripPlaneNeeds,
    /// Recycled rolling-plane buffers for the STREAMING walk's
    /// [`crate::feature_v2_stream::StripPlaneProducer`] (drained at
    /// construction, refilled by `recycle` at walk end) — steady-state
    /// batch extraction performs zero producer allocation.
    stream_pool: Vec<Vec<f32>>,
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
            sized_needs: StripPlaneNeeds::NONE,
            stream_pool: Vec::new(),
        }
    }

    /// Grow (never shrink) every buffer to hold `strip_max_n` elements.
    fn ensure(&mut self, strip_max_n: usize) {
        self.ensure_for(strip_max_n, StripPlaneNeeds::ALL);
    }

    /// [`Self::ensure`] restricted to the plane groups the caller will write
    /// (see [`StripPlaneNeeds`]). Both the size and the set are grow-only, so
    /// this can only ever allocate LESS than `ensure` — never differently.
    fn ensure_for(&mut self, strip_max_n: usize, needs: StripPlaneNeeds) {
        let want = self.sized_needs.union(needs);
        if strip_max_n > self.sized_for || want != self.sized_needs {
            let n = strip_max_n.max(self.sized_for);
            self.strips = [
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
                ScratchV2Strip::new_for(n, want),
            ];
            self.sized_for = n;
            self.sized_needs = want;
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
        // materialized walk: always wants the v2 planes
        true,
        // the materialized walk parallelises at the channel/scale level
        // above this call; a nested band fan-out here is not its axis.
        false,
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
        ..
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
        // materialized walk: always wants the v2 planes
        true,
        // the materialized walk parallelises at the channel/scale level
        // above this call; a nested band fan-out here is not its axis.
        false,
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
    }
}

/// Build the full per-scale/per-channel moment cache for a prepared
/// reference (see [`V2PreparedReference::moments`]).
fn fill_ref_moments(scales: &[([Vec<f32>; 3], usize, usize)]) -> Vec<[V2RefMoments; 3]> {
    let (_, w0, h0) = &scales[0];
    #[allow(clippy::absurd_extreme_comparisons, clippy::unnecessary_min_or_max)]
    let bypass_rows = (*h0).min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = w0 * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    let mut scratch = ScratchV2Strip::new(strip_max_n);
    scales
        .iter()
        .map(|(planes, width, height)| {
            std::array::from_fn(|ch| {
                compute_ref_moments_channel(&planes[ch], *width, *height, &mut scratch)
            })
        })
        .collect()
}

/// Shared body for [`run_blur_pass`]/[`run_blur_pass_strip`] — the actual
/// fused H-blur + 4x V-blur + activity sequence, over explicit buffers.
#[allow(clippy::too_many_arguments)]
/// Row-band-parallel wrapper for [`crate::blur::fused_blur_h_ssim`].
///
/// **Bit-exact by construction, and the construction is the whole argument.**
/// A horizontal box blur is an independent running-sum recurrence per row, and
/// the kernels walk rows in groups of [`H_BLUR_ROW_GROUP`]
/// (`fused_blur_h_ssim_*_inner`'s `run_group(rg * 8, 8)` + one partial tail).
/// Bands here start at multiples of that group size, so every band's internal
/// grouping is a **sub-sequence of the whole-plane call's** — each row lands in
/// a group of the same size, at the same offset within it, on the same SIMD
/// tier. Only the LAST band can hold a partial group, which is exactly the
/// partial group the whole-plane call's tail would have run.
/// `phase_a_blur_bands_are_bit_exact` pins that over every band size and
/// geometry; `box_blur_h_ring_matches_regathered_reference` independently pins
/// the per-row claim the construction rests on.
///
/// Predecessor context (`extraction_perf_and_buffered_removal_2026-08-30.md`
/// §11.1): this lever was implemented, measured NEUTRAL and reverted **in the
/// 944-full walk**, where phase A is a small share behind `dense_block_kernel`.
/// In the `v1_only` SCORING request that backs the fold engine there is no
/// dense kernel at all and phase A is 36 % of the walk at 16 threads with an
/// occupancy of 0.157 — a different measurement, not a re-run of that one.
#[allow(clippy::too_many_arguments)]
fn fused_blur_h_ssim_banded(
    src: &[f32],
    dst: &[f32],
    mu1_h: &mut [f32],
    mu2_h: &mut [f32],
    ssq_h: &mut [f32],
    s12_h: &mut [f32],
    width: usize,
    height_local: usize,
    #[allow(unused_variables)] parallel: bool,
) {
    #[cfg(feature = "threads")]
    if parallel && height_local > H_BLUR_BAND_ROWS && width > 0 {
        use rayon::prelude::*;
        let n = H_BLUR_BAND_ROWS * width;
        mu1_h
            .par_chunks_mut(n)
            .zip(mu2_h.par_chunks_mut(n))
            .zip(ssq_h.par_chunks_mut(n))
            .zip(s12_h.par_chunks_mut(n))
            .enumerate()
            .for_each(|(b, (((m1, m2), sq), s12))| {
                let __t = crate::fold_timing::start();
                let y0 = b * H_BLUR_BAND_ROWS;
                let rows = m1.len() / width;
                let lo = y0 * width;
                let hi = lo + rows * width;
                crate::blur::fused_blur_h_ssim(
                    &src[lo..hi],
                    &dst[lo..hi],
                    m1,
                    m2,
                    sq,
                    s12,
                    width,
                    rows,
                    BLUR_RADIUS,
                );
                crate::fold_timing::stop(__t, crate::fold_timing::Phase::BlurBandBusy, 0);
            });
        return;
    }
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
}

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
    want_v2: bool,
    parallel: bool,
) {
    let n = width * height_local;
    let mu1_h = &mut mu1_h[..n];
    let mu2_h = &mut mu2_h[..n];
    let ssq_h = &mut ssq_h[..n];
    let s12_h = &mut s12_h[..n];
    let __t_h = crate::fold_timing::start();
    fused_blur_h_ssim_banded(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height_local,
        parallel,
    );
    crate::fold_timing::stop(__t_h, crate::fold_timing::Phase::BlurHWall, 0);

    // BLOCK-SKIPPING: everything below this line feeds v2-era kernels ONLY.
    // `fold_v1_basic_bands` reads the H-blurred planes above and computes its
    // own activity internally, so a v1-only request can stop here — four
    // whole-window `box_blur_v_from_copy` sweeps and the activity chain
    // (abs-diff + a full 1-pass blur) are skipped, not merely unread.
    if !want_v2 {
        return;
    }

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
    /// append2 BANDVIS sums — accumulated ONLY by the `BANDVIS = true`
    /// kernel instantiation (Y channel with `append2_block` on); zero and
    /// untouched everywhere else, so the existing lanes' operations are
    /// bit-identical when append2 is off.
    sum_bv_gain: f64,
    sum_bv_loss: f64,
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
        self.sum_bv_gain += other.sum_bv_gain;
        self.sum_bv_loss += other.sum_bv_loss;
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
/// `BV_DSTACT` (BANDVIS dst-activity combine, `append2_dst_activity`):
/// when true, `act_dst` carries the distorted-side activity plane (same
/// shape as `activity` — `width*height`, no halo, own-row reads only) and
/// the BANDVIS GAIN polarity becomes the pure-band FR excess pooled under
/// the dst's own flatness weight `1 − sat(act_dst, C_ACTIVITY)` (LOSS
/// keeps the toggle-off math bit-exactly — see the adjudication record in
/// `benchmarks/bandvis_dst_activity_2026-08-02.md` for why both
/// in-ratio masking arms were rejected). Only meaningful with
/// `BANDVIS = true`; with `BV_DSTACT = false` the parameter is dead
/// (`&[]`) and the emitted operation sequence is the pre-fix one, bit
/// for bit.
#[inline]
fn gradient_block_kernel_generic<
    T: F32x8Backend + Copy,
    const BANDVIS: bool,
    const BV_DSTACT: bool,
>(
    token: T,
    src_h: &[f32],
    dst_h: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
) -> GradientAccum {
    debug_assert_eq!(src_h.len(), width * (height + 2));
    debug_assert_eq!(dst_h.len(), width * (height + 2));
    debug_assert_eq!(activity.len(), width * height);
    if BV_DSTACT {
        debug_assert!(BANDVIS, "BV_DSTACT is a BANDVIS refinement");
        debug_assert_eq!(act_dst.len(), width * height);
    }

    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let c_gms = V8::<T>::splat(token, C_GMS as f32);
    let c_ring_err = V8::<T>::splat(token, C_RING_ERR as f32);
    let c_activity = V8::<T>::splat(token, C_ACTIVITY as f32);
    let c_ring_edge = V8::<T>::splat(token, C_RING_EDGE as f32);
    let c_band_dst = V8::<T>::splat(token, C_BAND_DST as f32);
    let c_band_src = V8::<T>::splat(token, C_BAND_SRC as f32);
    // append2 BANDVIS constants (dead splats when `BANDVIS == false`).
    let bv_lo = V8::<T>::splat(token, bv_delta_lo);
    let bv_hi = V8::<T>::splat(token, bv_delta_hi);
    let c_bv = V8::<T>::splat(token, C_BV as f32);

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

        if BANDVIS {
            // Soft CURVATURE band-pass × flatness mask, FR excess pair
            // (see `idx_append2::BANDVIS_GAIN`). Second differences, not
            // first: a linear gradient of ANY steepness has |∇²| = 0, so
            // smooth ramps never enter the band (measured to be the
            // load-bearing property — a first-difference band could not
            // separate sub-step smooth gradients from steps; both
            // polarities mis-fired on ramp fixtures). A plateau step
            // reports the FULL step in |∇²| at its flanking pixels, so
            // the one-code-step δ derivation carries verbatim. Scalar
            // mirrors the SIMD formula exactly.
            let d2x_src = sxl + sxr - 2.0 * s;
            let d2y_src = syu + syd - 2.0 * s;
            let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
            let d2x_dst = dxl + dxr - 2.0 * dd;
            let d2y_dst = dyu + dyd - 2.0 * dd;
            let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
            let flat = 1.0 - act_b;
            let band = |g: f64| -> f64 {
                saturate(g, bv_delta_lo as f64) * (1.0 - saturate(g, bv_delta_hi as f64))
            };
            if BV_DSTACT {
                // `append2_dst_activity` SHIPPED combine (adjudication:
                // `benchmarks/bandvis_dst_activity_2026-08-02.md`).
                // GAIN = arm-2 visibility-weighted POOLING: the FR excess
                // on the PURE band terms, weighted by the DST's own
                // flatness OUTSIDE the ratio — the only place a flatness
                // mask survives `bounded_excess`'s scale-invariance
                // (arm 1, flat multiplied INSIDE the pair, measured
                // ratio-cancelled: it suppressed real banding MORE than
                // dither). Measured: lattice cross-fire 0.33×, deband
                // margin 2.2× the OFF margin.
                // LOSS = the OFF math BIT-EXACTLY (identical expressions,
                // identical order): the arm-2 weight measured
                // direction-INVERTING on the deband credit (the banded
                // src's own contours zero the very pixels whose removal
                // should be credited), and LOSS is the LYB-validated
                // workhorse — it must not move.
                let flat_d = 1.0 - saturate(act_dst[y * width + x] as f64, C_ACTIVITY);
                let b_src = band(curv_src) * flat;
                let b_dst = band(curv_dst) * flat;
                let (_, loss) = bounded_excess_pair(b_dst, b_src, C_BV);
                let (g0, _) = bounded_excess_pair(band(curv_dst), band(curv_src), C_BV);
                acc.sum_bv_gain += g0 * flat_d;
                acc.sum_bv_loss += loss;
            } else {
                let b_src = band(curv_src) * flat;
                let b_dst = band(curv_dst) * flat;
                let (gain, loss) = bounded_excess_pair(b_dst, b_src, C_BV);
                acc.sum_bv_gain += gain;
                acc.sum_bv_loss += loss;
            }
        }
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
            let (mut r_bv_gain, mut r_bv_loss) = (zero, zero);

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

                if BANDVIS {
                    // Curvature band (see the scalar sibling's rationale).
                    let two = one + one;
                    let d2x_s = sxl + sxr - two * s;
                    let d2y_s = syu + syd - two * s;
                    let curv_s = (d2x_s * d2x_s + d2y_s * d2y_s).sqrt();
                    let d2x_d = dxl + dxr - two * dd;
                    let d2y_d = dyu + dyd - two * dd;
                    let curv_d = (d2x_d * d2x_d + d2y_d * d2y_d).sqrt();
                    let flat = one - act_b;
                    let band_s =
                        saturate_v(token, curv_s, bv_lo) * (one - saturate_v(token, curv_s, bv_hi));
                    let band_d =
                        saturate_v(token, curv_d, bv_lo) * (one - saturate_v(token, curv_d, bv_hi));
                    if BV_DSTACT {
                        // SHIPPED combine (see the scalar sibling): GAIN
                        // = pure-band FR excess × dst flatness (pooling
                        // weight); LOSS = the OFF math bit-exactly.
                        let actd = ld_at!(act_dst, act_row + x);
                        let flat_d = one - saturate_v(token, actd, c_activity);
                        let b_src = band_s * flat;
                        let b_dst = band_d * flat;
                        let (_, loss) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                        let (g0, _) = bounded_excess_pair_v(token, band_d, band_s, c_bv);
                        r_bv_gain += g0 * flat_d;
                        r_bv_loss += loss;
                    } else {
                        let b_src = band_s * flat;
                        let b_dst = band_d * flat;
                        let (gain, loss) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                        r_bv_gain += gain;
                        r_bv_loss += loss;
                    }
                }

                x += 8;
            }

            acc.sum_gms += r_gms.reduce_add() as f64;
            acc.sum_gms2 += r_gms2.reduce_add() as f64;
            acc.sum_ringing += r_ring.reduce_add() as f64;
            acc.sum_banding += r_band.reduce_add() as f64;
            acc.sum_grad_src += r_gsrc.reduce_add() as f64;
            acc.sum_grad_dst += r_gdst.reduce_add() as f64;
            if BANDVIS {
                acc.sum_bv_gain += r_bv_gain.reduce_add() as f64;
                acc.sum_bv_loss += r_bv_loss.reduce_add() as f64;
            }

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
    gradient_block_kernel_generic::<_, false, false>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        0.0,
        0.0,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, false>(
        token,
        src,
        dst,
        activity,
        &[],
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn gradient_block_kernel_entry_bandvis_dstact(
    token: Token,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    act_dst: &[f32],
    width: usize,
    height: usize,
    bv_delta_lo: f32,
    bv_delta_hi: f32,
) -> GradientAccum {
    gradient_block_kernel_generic::<_, true, true>(
        token,
        src,
        dst,
        activity,
        act_dst,
        width,
        height,
        bv_delta_lo,
        bv_delta_hi,
    )
}

/// `bandvis`: `Some((δ_lo, δ_hi))` accumulates the append2 BANDVIS pair
/// (Y channel with `append2_block` on — the CROSS-pattern const split, so
/// chroma/off paths pay nothing and change no byte). `bv_act_dst`:
/// `Some(dst-activity strip)` switches the BANDVIS dst band term to the
/// dst self-mask (`append2_dst_activity` — a third const-split
/// instantiation; `None` runs the pre-fix bytes exactly). Ignored without
/// `bandvis`.
fn gradient_block_kernel(
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    bandvis: Option<(f32, f32)>,
    bv_act_dst: Option<&[f32]>,
) -> GradientAccum {
    match (bandvis, bv_act_dst) {
        (None, _) => incant!(
            gradient_block_kernel_entry(src, dst, activity, width, height),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), None) => incant!(
            gradient_block_kernel_entry_bandvis(src, dst, activity, width, height, lo, hi),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
        (Some((lo, hi)), Some(act_dst)) => incant!(
            gradient_block_kernel_entry_bandvis_dstact(
                src, dst, activity, act_dst, width, height, lo, hi
            ),
            [v4x, v4, v3, neon, wasm128, scalar]
        ),
    }
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
    /// append2 highlight bins (HDR route + `append2_block` only — the
    /// `HL = true` kernel instantiation; untouched zeros otherwise).
    ws_hl1: WeightedSum,
    ws_hl2: WeightedSum,
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
        self.ws_hl1.accumulate(&other.ws_hl1);
        self.ws_hl2.accumulate(&other.ws_hl2);
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
fn append_block_kernel_generic<T: F32x8Backend + Copy, const CROSS: bool, const HL: bool>(
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
        let (mut p_hl1w, mut p_hl1v, mut p_hl2w, mut p_hl2v) = (zero, zero, zero, zero);

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

            if HL {
                // append2 highlight bins (`idx_append2::HL_BIN1/2`):
                // `w = sat(max(ry − anchor, 0), C_HL)` pooling mse_i.
                let hl1_anchor = V8::<T>::splat(token, HL1_Y_ANCHOR);
                let hl2_anchor = V8::<T>::splat(token, HL2_Y_ANCHOR);
                let c_hl = V8::<T>::splat(token, C_HL as f32);
                let w1 = saturate_v(token, (ry - hl1_anchor).max(zero), c_hl);
                let w2 = saturate_v(token, (ry - hl2_anchor).max(zero), c_hl);
                p_hl1w += w1;
                p_hl1v += w1 * mse_i;
                p_hl2w += w2;
                p_hl2v += w2 * mse_i;
            }

            let var1 = (b2 - m1 * m1).max(zero);
            let var2 = ((ld!(ssq) - b2) - m2 * m2).max(zero);
            // #56: exact sqrt+div, not rsqrt — see `mscn_norm_v`.
            let n1 = mscn_norm_v(token, s - m1, var1, c_mvar);
            let n2 = mscn_norm_v(token, dd - m2, var2, c_mvar);
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
        if HL {
            acc.ws_hl1.num += p_hl1v.reduce_add() as f64;
            acc.ws_hl1.den += p_hl1w.reduce_add() as f64;
            acc.ws_hl2.num += p_hl2v.reduce_add() as f64;
            acc.ws_hl2.den += p_hl2w.reduce_add() as f64;
        }

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
            if HL {
                let w1 = saturate((ry - HL1_Y_ANCHOR as f64).max(0.0), C_HL);
                let w2 = saturate((ry - HL2_Y_ANCHOR as f64).max(0.0), C_HL);
                acc.ws_hl1.add(w1, mse_i);
                acc.ws_hl2.add(w2, mse_i);
            }

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
    append_block_kernel_generic::<_, true, false>(
        token, src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height,
    )
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn append_block_kernel_entry_cross_hl(
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
    append_block_kernel_generic::<_, true, true>(
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
    append_block_kernel_generic::<_, false, false>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        bs2,
        activity,
        ref_y,
        &[],
        &[],
        width,
        height,
    )
}

/// Runtime dispatch wrapper for the append kernel (same `incant!` shape as
/// [`dense_block_kernel`]). `cross` carries the X/B activity strips for the
/// Y channel; `None` compiles the cross-channel chain out entirely.
/// `hl`: append2 highlight bins on (Y channel, HDR route, `append2_block`)
/// — requires `cross` (Y-only by construction; asserted).
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
    hl: bool,
    width: usize,
    height: usize,
) -> AppendAccum {
    match cross {
        Some((act_x, act_b)) if hl => {
            incant!(
                append_block_kernel_entry_cross_hl(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
        Some((act_x, act_b)) => {
            incant!(
                append_block_kernel_entry_cross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, act_x, act_b, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
        None => {
            debug_assert!(!hl, "HL bins are Y-only (cross) by construction");
            incant!(
                append_block_kernel_entry_nocross(
                    src, dst, mu1, mu2, ssq, bs2, activity, ref_y, width, height
                ),
                [v4x, v4, v3, neon, wasm128, scalar]
            )
        }
    }
}

// ============================================================================
// CSFW (f944+) kernel — chunk-3 tier-1 second pass over the Y strip rows.
//
// A SEPARATE const-gated pass, not a growth of the append kernel: the append
// kernel already carries ~19 row-lane accumulators against the 16-register
// AVX2/NEON budget with accepted L1 spills — five more lanes land on the
// wrong side of that cliff (design §4.2). A second pass over strip-resident
// planes re-reads L2, not DRAM (§A.15's locality), so the marginal cost is
// close to the 10 ops/px of its own arithmetic. With `csfw_block` off this
// pass is never invoked and every existing kernel is untouched machine code
// — the byte-stability guarantee is structural.
// ============================================================================

/// Per-row-reduced f64 accumulators for the CSFW weighted-pool pass:
/// `Σw, Σw·s, Σw·d, Σw·s², Σw·d²` (design §4.1 — pure sums of products,
/// strip-foldable exactly like [`AppendAccum`]).
#[derive(Default, Clone, Copy)]
struct CsfwAccum {
    sum_w: f64,
    sum_ws: f64,
    sum_wd: f64,
    sum_ws2: f64,
    sum_wd2: f64,
}

impl CsfwAccum {
    /// Strip-partial fold, same reasoning as [`AppendAccum::accumulate`].
    #[inline]
    fn accumulate(&mut self, other: &CsfwAccum) {
        self.sum_w += other.sum_w;
        self.sum_ws += other.sum_ws;
        self.sum_wd += other.sum_wd;
        self.sum_ws2 += other.sum_ws2;
        self.sum_wd2 += other.sum_wd2;
    }
}

/// Per-walk CSFW parameters, derived once from the route: the effective
/// per-scale weight quadratic `w(y) = clamp(b0 + b1·y + b2·y², w_min,
/// w_max)` with `b = [1 + g·c0, g·c1, g·c2]`, `g = κ_Y·λ_b` folded at
/// walk setup (design §4.2: `κ_c·λ_b` folds to one constant per band —
/// no runtime multiply of the two) and `c` the route's derived φ
/// quadratic ([`CSFW_PHI_Y_SDR`]/[`CSFW_PHI_Y_PU`]).
#[derive(Clone, Copy)]
struct CsfwParams {
    eff: [[f64; 3]; crate::NUM_SCALES],
}

impl CsfwParams {
    fn for_phi(phi: [f64; 3]) -> Self {
        let eff = std::array::from_fn(|scale| {
            let g = CSFW_KAPPA_Y * CSFW_LAMBDA_B[scale];
            [1.0 + g * phi[0], g * phi[1], g * phi[2]]
        });
        Self { eff }
    }
}

/// CSFW SIMD kernel body — generic over any `F32x8Backend` token, same
/// row-lane-then-f64 reduction structure as [`append_block_kernel_generic`].
/// `ref_y` is the reference Y strip rows (for the Y channel these are the
/// same values as `src`; the argument stays separate so the weight is
/// explicitly a function of the REFERENCE plane only — design §4.1 — and
/// so a chroma tier can reuse the kernel unchanged).
#[inline(always)]
fn csfw_block_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
) -> CsfwAccum {
    let b0 = V8::<T>::splat(token, eff[0] as f32);
    let b1 = V8::<T>::splat(token, eff[1] as f32);
    let b2 = V8::<T>::splat(token, eff[2] as f32);
    let w_min = V8::<T>::splat(token, CSFW_W_MIN as f32);
    let w_max = V8::<T>::splat(token, CSFW_W_MAX as f32);

    let mut acc = CsfwAccum::default();
    let width8 = width - (width % 8);

    for y in 0..height {
        let row = y * width;
        let (mut r_w, mut r_ws, mut r_wd, mut r_ws2, mut r_wd2) = (
            V8::<T>::zero(token),
            V8::<T>::zero(token),
            V8::<T>::zero(token),
            V8::<T>::zero(token),
            V8::<T>::zero(token),
        );

        let mut x = 0usize;
        while x < width8 {
            let i = row + x;
            let s = V8::<T>::from_array(token, src[i..i + 8].try_into().unwrap());
            let dd = V8::<T>::from_array(token, dst[i..i + 8].try_into().unwrap());
            let ry = V8::<T>::from_array(token, ref_y[i..i + 8].try_into().unwrap());

            // w = clamp(b0 + y·(b1 + y·b2), w_min, w_max) — Horner, 2 FMA.
            let w = ry.mul_add(ry.mul_add(b2, b1), b0).max(w_min).min(w_max);
            let ws = w * s;
            let wd = w * dd;
            r_w += w;
            r_ws += ws;
            r_wd += wd;
            r_ws2 += ws * s;
            r_wd2 += wd * dd;

            x += 8;
        }

        acc.sum_w += r_w.reduce_add() as f64;
        acc.sum_ws += r_ws.reduce_add() as f64;
        acc.sum_wd += r_wd.reduce_add() as f64;
        acc.sum_ws2 += r_ws2.reduce_add() as f64;
        acc.sum_wd2 += r_wd2.reduce_add() as f64;

        // Scalar tail — same formulas in f64, the house tail idiom.
        for x in width8..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let ry = ref_y[i] as f64;
            let w = (eff[0] + ry * (eff[1] + ry * eff[2])).clamp(CSFW_W_MIN, CSFW_W_MAX);
            let ws = w * s;
            let wd = w * dd;
            acc.sum_w += w;
            acc.sum_ws += ws;
            acc.sum_wd += wd;
            acc.sum_ws2 += ws * s;
            acc.sum_wd2 += wd * dd;
        }
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn csfw_block_kernel_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
) -> CsfwAccum {
    csfw_block_kernel_generic(token, src, dst, ref_y, eff, width, height)
}

/// Runtime dispatch wrapper for the CSFW kernel (same `incant!` shape as
/// [`append_block_kernel`]).
fn csfw_block_kernel(
    src: &[f32],
    dst: &[f32],
    ref_y: &[f32],
    eff: [f64; 3],
    width: usize,
    height: usize,
) -> CsfwAccum {
    incant!(
        csfw_block_kernel_entry(src, dst, ref_y, eff, width, height),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Finalize one scale's CSFW block into its 3 output slots — the weighted
/// twins of [`finish_append`]'s GLOBAL_* trio, same constants, same
/// clamps. `Σw ≥ w_min·n > 0` structurally; the [`WeightedSum`]-class
/// 1e-12 denominator floor is kept as defense in depth.
fn finish_csfw(csfw: &CsfwAccum, out: &mut [f64]) {
    debug_assert_eq!(out.len(), CSFW_PER_SCALE);
    if csfw.sum_w < 1e-12 {
        out[idx_csfw::W_GLOBAL_DMEAN] = 0.0;
        out[idx_csfw::W_GLOBAL_CGAIN] = 0.0;
        out[idx_csfw::W_GLOBAL_CLOSS] = 0.0;
        return;
    }
    out[idx_csfw::W_GLOBAL_DMEAN] =
        saturate((csfw.sum_ws - csfw.sum_wd).abs() / csfw.sum_w, C_GDMEAN);
    let wmean_s = csfw.sum_ws / csfw.sum_w;
    let wmean_d = csfw.sum_wd / csfw.sum_w;
    let gvar1_w = (csfw.sum_ws2 / csfw.sum_w - wmean_s * wmean_s).max(0.0);
    let gvar2_w = (csfw.sum_wd2 / csfw.sum_w - wmean_d * wmean_d).max(0.0);
    let (g_cgain, g_closs) = bounded_excess_pair(gvar2_w, gvar1_w, C_GCONTRAST);
    out[idx_csfw::W_GLOBAL_CGAIN] = g_cgain.clamp(0.0, 1.0);
    out[idx_csfw::W_GLOBAL_CLOSS] = g_closs.clamp(0.0, 1.0);
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

/// `out[i] = input[i]²` — the streamed walk's σ-split square (reads the
/// wide window in place, writes into a free scratch buffer). Plain scalar
/// loop; LLVM auto-vectorizes the independent multiply.
fn square_into(input: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(input.iter()) {
        *o = v * v;
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
    // --- v1 pool blocks (`V2NewFeatureToggles::v1_pools`): the peak
    //     accumulators the fused kernel returns anyway, plus the masked /
    //     IW sums v1's extended strip section derives per band
    //     (`streaming::ScaleAccumulators` field-for-field). ---
    ssim_d8: f64,
    edge_art8: f64,
    edge_det8: f64,
    ssim_max: f32,
    edge_art_max: f32,
    edge_det_max: f32,
    masked_ssim_d: f64,
    masked_ssim_d4: f64,
    masked_ssim_d2: f64,
    masked_art4: f64,
    masked_det4: f64,
    masked_mse: f64,
    iw_ssim_d: f64,
    iw_ssim_d4: f64,
    iw_ssim_d2: f64,
    iw_art4: f64,
    iw_det4: f64,
    iw_mse: f64,
}

impl V1BasicSums {
    /// Add another band's sums into this one. Used by both paths of
    /// [`fold_v1_basic_bands`]; the parallel path merges IN BAND ORDER so the
    /// f64 addition sequence matches the serial one exactly.
    fn merge(&mut self, o: &V1BasicSums) {
        self.ssim_d += o.ssim_d;
        self.ssim_d4 += o.ssim_d4;
        self.ssim_d2 += o.ssim_d2;
        self.edge_art += o.edge_art;
        self.edge_art4 += o.edge_art4;
        self.edge_art2 += o.edge_art2;
        self.edge_det += o.edge_det;
        self.edge_det4 += o.edge_det4;
        self.edge_det2 += o.edge_det2;
        self.mse += o.mse;
        self.hf_sq_src += o.hf_sq_src;
        self.hf_sq_dst += o.hf_sq_dst;
        self.hf_abs_src += o.hf_abs_src;
        self.hf_abs_dst += o.hf_abs_dst;
        self.ssim_d8 += o.ssim_d8;
        self.edge_art8 += o.edge_art8;
        self.edge_det8 += o.edge_det8;
        self.ssim_max = self.ssim_max.max(o.ssim_max);
        self.edge_art_max = self.edge_art_max.max(o.edge_art_max);
        self.edge_det_max = self.edge_det_max.max(o.edge_det_max);
        self.masked_ssim_d += o.masked_ssim_d;
        self.masked_ssim_d4 += o.masked_ssim_d4;
        self.masked_ssim_d2 += o.masked_ssim_d2;
        self.masked_art4 += o.masked_art4;
        self.masked_det4 += o.masked_det4;
        self.masked_mse += o.masked_mse;
        self.iw_ssim_d += o.iw_ssim_d;
        self.iw_ssim_d4 += o.iw_ssim_d4;
        self.iw_ssim_d2 += o.iw_ssim_d2;
        self.iw_art4 += o.iw_art4;
        self.iw_det4 += o.iw_det4;
        self.iw_mse += o.iw_mse;
    }

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
        // Peaks: v1 merges them the same way (`streaming.rs`
        // `accum.ssim_d8[c] += strip_acc.ssim_d8; accum.ssim_max[c] =
        // accum.ssim_max[c].max(strip_acc.ssim_max)`, …) — free to carry.
        self.ssim_d8 += s.ssim_d8;
        self.edge_art8 += s.edge_art8;
        self.edge_det8 += s.edge_det8;
        self.ssim_max = self.ssim_max.max(s.ssim_max);
        self.edge_art_max = self.edge_art_max.max(s.edge_art_max);
        self.edge_det_max = self.edge_det_max.max(s.edge_det_max);
    }

    /// Finalize the v1 pool blocks — peaks (6/ch), masked (6/ch), IW
    /// (6/ch) — into their v1 slot order, replicating
    /// `streaming::ScaleAccumulators::finalize` + `metric.rs`'s pass-2/3/4
    /// pushes (`.abs()` on the masked/IW ssim + art/det L4 slots, none on
    /// the peaks / mse).
    fn finalize_pools_into(&self, n: usize, peaks: &mut [f64], masked: &mut [f64], iw: &mut [f64]) {
        debug_assert_eq!(peaks.len(), 6);
        debug_assert_eq!(masked.len(), 6);
        debug_assert_eq!(iw.len(), 6);
        let one_over_n = 1.0 / n as f64;
        peaks[0] = f64::from(self.ssim_max);
        peaks[1] = f64::from(self.edge_art_max);
        peaks[2] = f64::from(self.edge_det_max);
        peaks[3] = (self.ssim_d8 * one_over_n).max(0.0).powf(0.125);
        peaks[4] = (self.edge_art8 * one_over_n).max(0.0).powf(0.125);
        peaks[5] = (self.edge_det8 * one_over_n).max(0.0).powf(0.125);
        masked[0] = (self.masked_ssim_d * one_over_n).abs();
        masked[1] = (self.masked_ssim_d4 * one_over_n).max(0.0).powf(0.25).abs();
        masked[2] = (self.masked_ssim_d2 * one_over_n).max(0.0).sqrt().abs();
        masked[3] = (self.masked_art4 * one_over_n).max(0.0).powf(0.25).abs();
        masked[4] = (self.masked_det4 * one_over_n).max(0.0).powf(0.25).abs();
        masked[5] = self.masked_mse * one_over_n;
        iw[0] = (self.iw_ssim_d * one_over_n).abs();
        iw[1] = (self.iw_ssim_d4 * one_over_n).max(0.0).powf(0.25).abs();
        iw[2] = (self.iw_ssim_d2 * one_over_n).max(0.0).sqrt().abs();
        iw[3] = (self.iw_art4 * one_over_n).max(0.0).powf(0.25).abs();
        iw[4] = (self.iw_det4 * one_over_n).max(0.0).powf(0.25).abs();
        iw[5] = self.iw_mse * one_over_n;
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
/// Band slots per strip — the fixed fan-out width of the band-parallel path.
const V1_BANDS_PER_STRIP: usize = STRIP_ROWS / V1_BAND_ROWS;
const V1_BAND_OVERLAP: usize = 5;

/// v1's masking / IW strengths (`metric::config_from_params`:
/// `extended_masking_strength: 4.0`, `iw_strength: 4.0`) — the fold's pool
/// replay uses the same constants so the block reproduces v1's numbers.
const V1_MASK_K: f32 = 4.0;
const V1_IW_K: f32 = 4.0;

/// What a v1 band does with its [`FoldPoolScratch`] — the band-level
/// resolution of [`V1PoolsMode`].
///
/// [`Self::HOnly`] exists because the pool block is NOT three independent
/// jobs. Peaks ride the fused kernel's unconditional L8/max tier, while
/// masked and IW share ONE activity chain, ONE sigma store and three
/// `*_inline_both` kernels that compute both strengths in a single sweep. So
/// the only compute boundary inside `f156..372` is *peaks* vs
/// *masked-and-IW*, and this enum is that boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BandPoolWork {
    /// No pool arithmetic at all. The scratch is present only so a
    /// [`FoldHSource::SelfBlur`] band can own its four H planes.
    HOnly,
    /// Activity chain + `edge_diff_channel_inline_both` (the carrier slots'
    /// `art_4th`); no sigma store, no MSE, no masked/IW SSIM.
    Carriers,
    /// Every pool slot: activity chain, sigma store, and all three
    /// `*_inline_both` kernels.
    Full,
}

/// Band-local planes for the v1 pool replay (`V2NewFeatureToggles::v1_pools`):
/// sized for one v1 band buffer (`V1_BAND_ROWS + 2 * V1_BAND_OVERLAP` rows ×
/// width), grown on first use per channel accumulator and reused across
/// bands, strips and scales (the widest scale sizes it once).
#[derive(Debug, Clone, Default)]
struct FoldPoolScratch {
    /// V-blurred `mu1` / `mu2` (the fused kernel's `store_mu` side-output).
    mu1_v: Vec<f32>,
    mu2_v: Vec<f32>,
    /// `|src − H_blur(src)|` (v1's `bufs.mask` role).
    act_raw: Vec<f32>,
    /// One-pass-blurred activity (v1's `bufs.mul_buf` role).
    act: Vec<f32>,
    /// V-blurred `ssq_h` / `s12_h` (v1's `bufs.sigma1_sq` / `bufs.sigma12`);
    /// `ssq_v` doubles as the activity blur's temp before it is filled.
    ssq_v: Vec<f32>,
    s12_v: Vec<f32>,
    /// BAND-LOCAL H-blurred planes, for the self-blur band shape
    /// ([`FoldHSource::SelfBlur`]). Only allocated when that shape runs.
    h: [Vec<f32>; 4],
}

impl FoldPoolScratch {
    fn ensure(&mut self, n: usize) {
        for b in [
            &mut self.mu1_v,
            &mut self.mu2_v,
            &mut self.act_raw,
            &mut self.act,
            &mut self.ssq_v,
            &mut self.s12_v,
        ] {
            if b.len() < n {
                b.resize(n, 0.0);
            }
        }
    }

    /// Grow the band-local H planes to `n` elements. Separate from
    /// [`Self::ensure`] so the precomputed-H shape never allocates them.
    fn ensure_h(&mut self, n: usize) {
        for b in self.h.iter_mut() {
            if b.len() < n {
                b.resize(n, 0.0);
            }
        }
    }
}

/// Where a v1 band's four H-blurred planes come from.
///
/// **This is the fold's memory-traffic lever, and it is a measured one.**
/// The 16-independent-process throughput test at 2304² (see
/// `benchmarks/fold_mt_scaling_2026-08-31.md`) puts the fold's ceiling at
/// **4.2×** on a box where the buffered walk reaches **10.9×** — from the same
/// serial speed. The whole difference is this: buffered blurs a band's rows
/// into band-PRIVATE buffers and consumes them in the SAME task, while the
/// fold blurred the whole 148-row strip window in one set of tasks and read it
/// back in another, so four planes round-tripped through L3/DRAM per strip.
///
/// `SelfBlur` gives the fold buffered's shape: each band H-blurs exactly the
/// `[b0 − overlap, b1 + overlap)` rows it is about to consume, into its own
/// scratch. It costs redundant blur at the band seams — 42 rows blurred per 32
/// consumed against 148 per 128, i.e. **+40 % blur compute** — and buys a
/// self-contained task.
///
/// Bit-exact either way: the H blur is an independent per-row recurrence
/// (`phase_a_blur_bands_are_bit_exact`), so a band's rows carry the same bits
/// whichever call produced them, and the band kernel below reads the same rows
/// at the same offsets. `fold_self_blur_matches_precomputed_h` is the gate.
#[derive(Clone, Copy)]
enum FoldHSource<'a> {
    /// The four planes, over the strip's whole wide window (row 0 of each
    /// plane is wide-window row 0).
    Precomputed([&'a [f32]; 4]),
    /// Each band blurs its own rows from the raw windows into
    /// `FoldPoolScratch::h`.
    SelfBlur,
}

/// One v1-aligned band of the fold hook. Extracted from `fold_v1_basic_bands`
/// verbatim so the band-parallel and serial paths run IDENTICAL code — the
/// only difference between them is which `V1BasicSums` the band accumulates
/// into, and in what order those are merged.
#[allow(clippy::too_many_arguments)]
fn fold_v1_one_band(
    b0: usize,
    width: usize,
    rows_end: usize,
    strip_y0: usize,
    halo_offset: usize,
    height: usize,
    h_src: FoldHSource<'_>,
    raw: [&[f32]; 2],
    sums: &mut V1BasicSums,
    mut pools: Option<(&mut FoldPoolScratch, BandPoolWork)>,
) -> usize {
    let [src, dst] = raw;
    let rows = 0..rows_end;
    let _ = &rows;
    let b1 = (b0 + V1_BAND_ROWS).min(rows_end);
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
    if let Some((ps, work)) = pools.as_mut().map(|(p, f)| (&mut **p, *f)) {
        let band_n = h_local * width;
        // Size to the WIDEST band this plane can ever hold, not to this band's
        // own height (fold-footprint lane). `ensure` grows through
        // `Vec::resize`, which reserves `max(2*cap, need)`: the first band of
        // a strip is TOP-CLAMPED to `V1_BAND_ROWS + V1_BAND_OVERLAP` rows, so
        // slot 0 was sized 37·width and then DOUBLED to 74·width on the first
        // interior band — 76 % more than it can use, on 10 planes × 3
        // channels. Measured (heaptrack, 1152²): 27.65 MB against the 23.22 MB
        // the slots actually hold, i.e. 3840·width bytes of pure growth slack.
        // Asking for the maximum up front makes every slot exactly 42·width.
        //
        // The pool planes are grown BELOW the `HOnly` arm: peaks-only bands
        // never touch them (feature-cost lane), so allocating them here would
        // hand a skipping request the footprint it is skipping.
        let band_cap_n = (V1_BAND_ROWS + 2 * V1_BAND_OVERLAP).min(height.max(1)) * width;
        debug_assert!(
            band_n <= band_cap_n,
            "band {band_n} exceeds the {band_cap_n} capacity bound"
        );
        let self_blur = matches!(h_src, FoldHSource::SelfBlur);
        if self_blur {
            ps.ensure_h(band_cap_n);
        }
        if work == BandPoolWork::HOnly {
            // PEAKS mode. Identical arithmetic to the `pools == None` arm
            // below — every store flag off, no activity, no masked/IW kernel
            // — run on whichever H planes this band is using. The peak sums
            // the caller wants are the kernel's unconditional L8/max tier,
            // so this costs exactly what `V1PoolsMode::Off` costs and the
            // emitted peak slots are bit-identical to `Full`'s.
            if self_blur {
                let [h0, h1, h2, h3] = &mut ps.h;
                crate::blur::fused_blur_h_ssim(
                    &src[span.clone()],
                    &dst[span.clone()],
                    &mut h0[..band_n],
                    &mut h1[..band_n],
                    &mut h2[..band_n],
                    &mut h3[..band_n],
                    width,
                    h_local,
                    BLUR_RADIUS,
                );
            }
            let (mu1_h, mu2_h, ssq_h, s12_h, span_h) = if self_blur {
                (
                    &ps.h[0][..band_n],
                    &ps.h[1][..band_n],
                    &ps.h[2][..band_n],
                    &ps.h[3][..band_n],
                    0..band_n,
                )
            } else {
                let FoldHSource::Precomputed([a, b, c, d]) = h_src else {
                    unreachable!("SelfBlur handled above")
                };
                (a, b, c, d, span.clone())
            };
            sums.accumulate(&crate::fused::fused_vblur_features_ssim(
                &mu1_h[span_h.clone()],
                &mu2_h[span_h.clone()],
                &ssq_h[span_h.clone()],
                &s12_h[span_h],
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
                &mut [],
                &mut [],
                false,
            ));
            return b1;
        }
        let full = work == BandPoolWork::Full;
        ps.ensure(band_cap_n);
        // ONE destructure, so the band-local H planes can be READ while the
        // pool planes stay mutable — disjoint fields of the same `&mut`.
        let FoldPoolScratch {
            mu1_v,
            mu2_v,
            act_raw,
            act,
            ssq_v,
            s12_v,
            h,
        } = ps;
        let [h0, h1, h2, h3] = h;
        if self_blur {
            // The band's OWN H blur, over exactly the rows it consumes.
            // Bit-identical to reading those rows out of a whole-window call
            // (`phase_a_blur_bands_are_bit_exact`); the point is that these
            // four planes never leave this task.
            crate::blur::fused_blur_h_ssim(
                &src[span.clone()],
                &dst[span.clone()],
                &mut h0[..band_n],
                &mut h1[..band_n],
                &mut h2[..band_n],
                &mut h3[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
            );
        }
        let (mu1_h, mu2_h, ssq_h, s12_h, span_h) = match h_src {
            FoldHSource::Precomputed([a, b, c, d]) => (a, b, c, d, span.clone()),
            FoldHSource::SelfBlur => (
                &h0[..band_n],
                &h1[..band_n],
                &h2[..band_n],
                &h3[..band_n],
                0..band_n,
            ),
        };
        // ORDER: activity FIRST, fused kernel SECOND. The activity blur
        // borrows `ssq_v` as its scratch temp, and the fused kernel then
        // overwrites `ssq_v`'s INNER rows with the real V-blurred sigma
        // (`store_sigma`) — so the two uses are disjoint in time and the
        // temp costs no extra buffer. Reordering is sum-neutral: the
        // `sums` fields written here (masked/IW) are disjoint from the
        // ones `accumulate` writes (basic).
        //
        // v1 computes `|src − H_blur(src)|` with its own H kernel
        // (`box_blur_h_into_abs_diff`); the fold already holds the
        // H-blurred src as `mu1_h` (the shared fused H-pass plane), so the
        // activity is one abs-diff pass — the pool parity gate proves the
        // two H kernels agree bit-for-bit on these planes.
        crate::simd_ops::abs_diff_into(
            &src[span.clone()],
            &mu1_h[span_h.clone()],
            &mut act_raw[..band_n],
        );
        crate::blur::box_blur_1pass_into(
            &act_raw[..band_n],
            &mut act[..band_n],
            &mut ssq_v[..band_n],
            width,
            h_local,
            BLUR_RADIUS,
        );
        // `store_sigma` replaces the two `box_blur_v_from_copy(ssq_h →
        // ssq_v)` / `(s12_h → s12_v)` band sweeps the `Full` arm used to
        // run after this call: the fused kernel already carries the same
        // running V-blur sums in registers and divides by the same
        // `1.0 / diam`, so the stored planes are BIT-IDENTICAL to what
        // that second sweep produced (and it writes only the inner rows —
        // the only rows the masked/IW SSIM kernel reads). `Carriers`
        // needs no sigma, so it stores none and `ssq_v` simply stays the
        // activity temp.
        sums.accumulate(&crate::fused::fused_vblur_features_ssim(
            &mu1_h[span_h.clone()],
            &mu2_h[span_h.clone()],
            &ssq_h[span_h.clone()],
            &s12_h[span_h.clone()],
            &src[span.clone()],
            &dst[span.clone()],
            width,
            h_local,
            inner_start,
            inner_h,
            BLUR_RADIUS,
            &mut mu1_v[..band_n],
            &mut mu2_v[..band_n],
            true,
            &mut empty_sd,
            false,
            &mut ssq_v[..band_n],
            &mut s12_v[..band_n],
            full,
        ));
        let inner = inner_start * width..(inner_start + inner_h) * width;
        let inner_src = &src[span.start + inner.start..span.start + inner.end];
        let inner_dst = &dst[span.start + inner.start..span.start + inner.end];
        let inner_mu1 = &mu1_v[inner.clone()];
        let inner_mu2 = &mu2_v[inner.clone()];
        let act_inner = &act[inner.clone()];
        if full {
            // The masked/IW SSIM + MSE slots — the sigma planes are
            // already in `ssq_v` / `s12_v` (the fused kernel's
            // `store_sigma` side-output above), so this arm no longer runs
            // its own two V-blur band sweeps.
            let (mse_m, mse_i) = crate::simd_ops::build_inline_mse(
                act_inner, V1_MASK_K, V1_IW_K, inner_src, inner_dst,
            );
            sums.masked_mse += mse_m;
            sums.iw_mse += mse_i;
            let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) =
                crate::simd_ops::ssim_channel_inline_both(
                    inner_mu1,
                    inner_mu2,
                    &ssq_v[inner.clone()],
                    &s12_v[inner.clone()],
                    act_inner,
                    V1_MASK_K,
                    V1_IW_K,
                );
            sums.masked_ssim_d += sd_m;
            sums.masked_ssim_d4 += sd4_m;
            sums.masked_ssim_d2 += sd2_m;
            sums.iw_ssim_d += sd_i;
            sums.iw_ssim_d4 += sd4_i;
            sums.iw_ssim_d2 += sd2_i;
        }
        let ((art4_m, det4_m), (art4_i, det4_i)) = crate::simd_ops::edge_diff_channel_inline_both(
            inner_src, inner_dst, inner_mu1, inner_mu2, act_inner, V1_MASK_K, V1_IW_K,
        );
        sums.masked_art4 += art4_m;
        sums.masked_det4 += det4_m;
        sums.iw_art4 += art4_i;
        sums.iw_det4 += det4_i;
    } else {
        // No pool scratch => no band-local H buffer either, so this arm only
        // serves `FoldHSource::Precomputed`. `fold_v1_basic_bands` refuses the
        // combination up front rather than reaching here.
        let FoldHSource::Precomputed([mu1_h, mu2_h, ssq_h, s12_h]) = h_src else {
            unreachable!("SelfBlur requires pool scratch (checked in fold_v1_basic_bands)")
        };
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
            &mut [],
            &mut [],
            false,
        ));
    }
    b1
}

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
/// With `pools == None` the store flags are off, so the kernel's
/// `mu1/mu2/sd` side-outputs are never written — empty slices are safe
/// (every write in every tier is `if store_*`-gated). With `Some`
/// (`V2NewFeatureToggles::v1_pools`) each band ALSO replays v1's extended
/// strip section (`streaming::process_strip_channel`'s `need_activity`
/// block) on the band buffer: the kernel stores its V-blurred mu1/mu2 for
/// the inner rows, the ref-side activity is `|src − H_blur(src)|`
/// one-pass-blurred over the SAME band buffer (mirror-clamped at its
/// edges exactly like v1's strip buffer), the sigma planes are the shared
/// H-planes V-blurred over the band, and the three fused masked+IW
/// kernels (`build_inline_mse`, `ssim_channel_inline_both`,
/// `edge_diff_channel_inline_both`) run on the inner rows at v1's
/// `k = 4` / `k_iw = 4`. Band extents, buffer contents and reduction
/// order are v1's, so the pooled sums are bit-identical to v1's whenever
/// the plane values are.
///
/// `raw = [src, dst]` and, for [`FoldHSource::Precomputed`], the four
/// H-planes, all in the SAME buffer-local coordinate system (`local = row −
/// strip_y0 + halo_offset`): the strip path passes the scratch H-planes +
/// halo buffers; the whole-plane path passes the full-plane H-planes + the
/// real src/dst planes with `strip_y0 = halo_offset = 0`.
/// [`FoldHSource::SelfBlur`] passes no H-planes — each band produces its own
/// from `raw`, which is the memory-traffic shape (see that type's doc).
/// SelfBlur requires `pools` (the band-local H buffer lives in the band's
/// [`FoldPoolScratch`]) and is refused otherwise.
///
/// RAII closer for the [`crate::fold_timing::Phase::FoldWall`] span, so the
/// early `return` in the parallel arm still records it. Diagnostic only.
struct FoldTimingGuard(Option<std::time::Instant>);
impl Drop for FoldTimingGuard {
    fn drop(&mut self) {
        crate::fold_timing::stop(self.0.take(), crate::fold_timing::Phase::FoldWall, 0);
    }
}

#[allow(clippy::too_many_arguments)]
fn fold_v1_basic_bands(
    width: usize,
    rows: core::ops::Range<usize>,
    strip_y0: usize,
    halo_offset: usize,
    height: usize,
    h_src: FoldHSource<'_>,
    raw: [&[f32]; 2],
    sums: &mut V1BasicSums,
    pools: Option<(&mut [FoldPoolScratch], BandPoolWork)>,
    parallel: bool,
) {
    assert!(
        !matches!(h_src, FoldHSource::SelfBlur) || pools.is_some(),
        "FoldHSource::SelfBlur needs the band pool scratch for its H buffer"
    );
    debug_assert_eq!(rows.start % V1_BAND_ROWS, 0, "strips are 32-row aligned");
    let __t_fold = crate::fold_timing::start();
    let __fold_guard = FoldTimingGuard(__t_fold);
    // Band starts, in order. Bands are INDEPENDENT: each reads a clamped
    // window of the shared planes and produces its own sums, so the only
    // cross-band coupling is the accumulation itself.
    let mut starts = Vec::new();
    let mut b = rows.start;
    while b < rows.end {
        starts.push(b);
        b = (b + V1_BAND_ROWS).min(rows.end);
    }

    // BIT-EXACTNESS OF THE PARALLEL PATH. Each band accumulates into its own
    // zero-initialised `V1BasicSums`, and the merge below runs SEQUENTIALLY IN
    // BAND ORDER — so the sequence of f64 additions is `((0 + b0) + b1) + ...`,
    // exactly what the in-place serial loop performed. (An unordered or tree
    // reduction would NOT be: f64 addition is not associative. The max fields
    // are order-free either way.) Both paths below use the same
    // local-then-merge shape so serial and parallel are identical to each
    // other by construction, not merely by argument.
    #[cfg(feature = "threads")]
    if parallel && starts.len() > 1 {
        use rayon::prelude::*;
        // Each band gets its OWN persistent scratch slot, so nothing is
        // allocated in the hot path. `map_init(FoldPoolScratch::default)` was
        // tried first and measured a NET LOSS: it re-allocates ~580 KB per
        // worker per strip per channel.
        let (slots, work) = match pools {
            Some((s, f)) => (Some(s), f),
            None => (None, BandPoolWork::HOnly),
        };
        let locals: Vec<V1BasicSums> = match slots {
            // CHUNKED over the slots the channel actually keeps
            // (`band_slots_for`). With at least as many slots as bands —
            // every pool of 4+ threads — `per` is 1 and this is exactly the
            // old one-band-per-slot zip. With fewer, a slot's bands run
            // sequentially inside one task, which costs parallelism the pool
            // could not have supplied anyway and saves the slots outright.
            //
            // ORDER, which is the whole bit-exactness argument: `par_chunks`
            // and `par_iter_mut` are both INDEXED, so `collect` yields chunks
            // in band order and each chunk yields its bands in band order —
            // flattening reproduces the serial band sequence exactly, and the
            // merge below is the same left-to-right f64 fold either way.
            Some(slots) => {
                let per = starts.len().div_ceil(slots.len().max(1)).max(1);
                let chunked: Vec<Vec<V1BasicSums>> = starts
                    .par_chunks(per)
                    .zip(slots.par_iter_mut())
                    .map(|(chunk, ps)| {
                        let mut out = Vec::with_capacity(chunk.len());
                        for &b0 in chunk {
                            let __t = crate::fold_timing::start();
                            let mut local = V1BasicSums::default();
                            fold_v1_one_band(
                                b0,
                                width,
                                rows.end,
                                strip_y0,
                                halo_offset,
                                height,
                                h_src,
                                raw,
                                &mut local,
                                Some((&mut *ps, work)),
                            );
                            crate::fold_timing::stop(__t, crate::fold_timing::Phase::BandBusy, 0);
                            out.push(local);
                        }
                        out
                    })
                    .collect();
                debug_assert_eq!(
                    chunked.iter().map(Vec::len).sum::<usize>(),
                    starts.len(),
                    "chunked band fan-out must cover every band exactly once"
                );
                chunked.into_iter().flatten().collect()
            }
            None => starts
                .par_iter()
                .map(|&b0| {
                    let __t = crate::fold_timing::start();
                    let mut local = V1BasicSums::default();
                    fold_v1_one_band(
                        b0,
                        width,
                        rows.end,
                        strip_y0,
                        halo_offset,
                        height,
                        h_src,
                        raw,
                        &mut local,
                        None,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::BandBusy, 0);
                    local
                })
                .collect(),
        };
        for l in &locals {
            sums.merge(l);
        }
        return;
    }
    #[cfg(not(feature = "threads"))]
    let _ = parallel;

    let mut pools = pools;
    for (i, &b0) in starts.iter().enumerate() {
        let mut local = V1BasicSums::default();
        fold_v1_one_band(
            b0,
            width,
            rows.end,
            strip_y0,
            halo_offset,
            height,
            h_src,
            raw,
            &mut local,
            pools
                .as_mut()
                .map(|(p, f)| (&mut p[i.min(p.len() - 1)], *f)),
        );
        sums.merge(&local);
    }
}

/// One channel-scale of the plain-v2 (V2Bounded) materialized walk: the
/// strip loop over halo-gathered wide buffers. (The folded/append regimes
/// route through the STREAMING walk since the C5 switchover — this
/// function serves only `compute_v2_features*`.)
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

    // Phase-6 (§A.16 lever B): small images skip the strip machinery
    // entirely — see `STRIP_BYPASS_HEIGHT`'s doc and `compute_channel_
    // scale_v2_whole`. Currently DISABLED (`STRIP_BYPASS_HEIGHT=0`, see its
    // doc) so this comparison is a permanent `false` today — written
    // generically (not as a dead-code-eliding special case) so a future
    // session re-enabling the lever only needs to change the constant.
    #[allow(clippy::absurd_extreme_comparisons)]
    if height <= STRIP_BYPASS_HEIGHT {
        return compute_channel_scale_v2_whole(src, dst, width, height, toggles, scratch, out);
    }

    let max_wide_h = STRIP_ROWS + 2 * HALO_P;
    assert!(
        width * max_wide_h <= scratch.mu1.len(),
        "scratch buffers must be sized for the largest strip+halo"
    );

    let mut dense = DenseAccum::default();
    let mut grad = GradientAccum::default();

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
            run_blur_pass_strip_cached_ref(width, wide_h, scratch);
        } else {
            run_blur_pass_strip(width, wide_h, scratch);
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
            let strip_grad =
                gradient_block_kernel(src_g, dst_g, activity_strip, width, strip_h, None, None);
            grad.accumulate(&strip_grad);
        }

        y0 += strip_h;
    }

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    finish_channel_scale(&dense, &grad, sum_blockiness, n, out)
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
fn compute_channel_scale_v2_whole(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    scratch: &mut ScratchV2Strip,
    out: &mut [f64],
) -> (f64, f64) {
    let n = width * height;
    debug_assert!(
        n <= scratch.mu1.len(),
        "scratch buffers must be sized for at least STRIP_BYPASS_HEIGHT rows \
         (see the scratch allocation in compute_v2_features_impl_with_toggles)"
    );

    run_blur_pass(src, dst, width, height, scratch);

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
        gradient_block_kernel(&src_g, &dst_g, activity, width, height, None, None)
    } else {
        GradientAccum::default()
    };

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

    finish_channel_scale(&dense, &grad, sum_blockiness, n, out)
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
    prepare_v2_reference_impl_inner(source, max_pixels, parallel, cache_moments)
}

fn prepare_v2_reference_impl_inner(
    source: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    cache_moments: bool,
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
    let moments = cache_moments.then(|| fill_ref_moments(&scales));
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
        prepared, distorted, max_pixels, parallel, toggles, scratch,
    )
}

/// Folded-720 pair entry (see [`FeatureRegime::Folded720`] for the
/// layout) — routes to the STREAMING walk (C5 switchover, 2026-07-26,
/// user-approved: the materialized folded walk and its reference cache
/// are deleted; `benchmarks/streaming_foldapp_gates_2026-07-26.md`).
pub(crate) fn compute_folded720_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    let mut scratch = V2Scratch::new();
    compute_folded720_streaming_impl(
        source,
        distorted,
        max_pixels,
        parallel,
        toggles,
        &mut scratch,
    )
}

/// Folded-720-plus-append pair entry (see
/// [`FeatureRegime::Folded720Append`]): the folded 720 layout with the
/// f720+ append block (204 slots) emitted after it. Forces
/// `toggles.append_block = true`. Routes to the STREAMING walk (C5).
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

// ============================================================================
// STREAMING folded-720[+append] walk (C2 of
// docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md).
//
// Consumes `feature_v2_stream::StripPlaneProducer`'s kernel strips instead
// of materialized full-image pyramids: per emitted strip, phase A fills
// each channel's wide windows (`gather_strip_halo` semantics from the
// rolling planes) + runs the SAME blur pass the materialized walk runs on
// its moments-free path + the P2 strip-tiled bs2 chain; phase B runs the
// SAME dense/gradient/append/fold kernels and folds their per-strip
// accumulators into per-(scale, channel) totals in kernel-strip order —
// the exact f64 op sequence of the materialized walk. Finalize replays
// the materialized walk's per-scale epilogue (finish + fold finalize +
// edge-width chain) on the merged accumulators. Output is BITWISE equal
// to the materialized pair path (test-gated below); no full-image f32
// plane exists on either side at any point.
// ============================================================================

/// append2 per-walk parameters, derived once from the route + toggles:
/// BANDVIS δ half-points (route-specific EMPIRICAL constants) and whether
/// the HDR highlight bins are live.
#[derive(Clone, Copy)]
struct Append2Params {
    bv_delta_lo: f32,
    bv_delta_hi: f32,
    hl_bins: bool,
    /// BANDVIS dst self-mask live ([`V2NewFeatureToggles::append2_dst_activity`]):
    /// phase A computes the Y-channel dst-activity plane, phase B routes it
    /// into the BANDVIS-dstact kernel instantiation. Route-independent.
    dst_activity: bool,
}

/// Per-channel accumulator state for the streaming walk: per-scale totals
/// for every family the walk pools, merged strictly in kernel-strip order.
struct StreamChannelAccums {
    dense: Vec<DenseAccum>,
    grad: Vec<GradientAccum>,
    app: Vec<AppendAccum>,
    v1: Vec<V1BasicSums>,
    /// Row-ordered blockiness partials `(sum_v, sum_h)` (P1 canonical).
    block: Vec<(f64, f64)>,
    /// CSFW weighted-pool partials (Y channel + `csfw_block` only —
    /// untouched zeros otherwise).
    csfw: Vec<CsfwAccum>,
    /// Band-local planes for the v1 pool replay (`v1_pools`); empty
    /// (never allocated) when the toggle is off.
    /// One scratch per band SLOT within a strip (`STRIP_ROWS /
    /// V1_BAND_ROWS`), so the band-parallel path reuses buffers instead of
    /// allocating ~580 KB per worker per strip — which measured as a net
    /// LOSS before this existed.
    pool_scratch: Vec<FoldPoolScratch>,
}

impl StreamChannelAccums {
    /// `n_band_slots` is the number of band scratches this channel keeps —
    /// see [`band_slots_for`]. It is a CAPACITY, not a tiling: bands are
    /// chunked across whatever slots exist (`fold_v1_basic_bands`), so any
    /// count in `1..=V1_BANDS_PER_STRIP` computes the same bytes.
    fn new(n_scales: usize, n_band_slots: usize) -> Self {
        Self {
            dense: vec![DenseAccum::default(); n_scales],
            grad: vec![GradientAccum::default(); n_scales],
            app: vec![AppendAccum::default(); n_scales],
            v1: vec![V1BasicSums::default(); n_scales],
            block: vec![(0.0, 0.0); n_scales],
            csfw: vec![CsfwAccum::default(); n_scales],
            pool_scratch: (0..n_band_slots.clamp(1, V1_BANDS_PER_STRIP))
                .map(|_| FoldPoolScratch::default())
                .collect(),
        }
    }
}

/// How many [`FoldPoolScratch`] slots one channel needs — the fold-footprint
/// lane's answer to "why is the fold heavier than buffered below ~2.5 MP".
///
/// The band scratch is the fold's single largest term (measured: 33 % of the
/// walk's peak heap at every size, `benchmarks/fold_footprint_2026-08-31.md`),
/// and it was sized by the FAN-OUT SHAPE — `V1_BANDS_PER_STRIP` slots per
/// channel × 3 channels = 12 band buffers — regardless of how many could ever
/// be live. Buffered's equivalent (`streaming`'s `ScaleBuffers`, 7 planes over
/// the same 42-row band) is sized by the WORKER COUNT, via `map_init`: one per
/// rayon worker. At one thread that is 12 buffers against buffered's 1, on
/// identical work, and it is most of the gap.
///
/// A band cannot run without a thread to run it on, so `min(bands, threads)`
/// is the count that can ever be simultaneously live inside one channel task
/// (rayon nests: a channel's band `par_iter` can be stolen by every thread in
/// the pool). Above that the extra slots are memory nothing can reach.
///
/// Byte-neutral: which scratch a band borrows cannot reach a feature, because
/// every band fully overwrites the region it reads and merges into its own
/// zero-initialised `V1BasicSums`, which are then reduced in BAND ORDER either
/// way (`fold_v1_basic_bands`). `fold_engine_parity` sweeps rayon pools
/// 1/2/3/8/16, which is now also a sweep of this function's output (1/2/3/4/4).
fn band_slots_for(band_parallel: bool) -> usize {
    #[cfg(feature = "threads")]
    let n = if band_parallel {
        V1_BANDS_PER_STRIP.min(rayon::current_num_threads().max(1))
    } else {
        1
    };
    #[cfg(not(feature = "threads"))]
    let n = {
        let _ = band_parallel;
        1usize
    };
    n
}

/// True when the append kernel runs for this (channel, scale) — mirrors
/// the materialized walk's [`APPEND_SKIP_B_SCALE0`] dispatch.
#[inline]
fn append_cell_active(append_on: bool, ch: usize, scale: usize) -> bool {
    append_on && !(APPEND_SKIP_B_SCALE0 && ch == 2 && scale == 0)
}

/// Retained per-(scale, channel) state from ONE streaming folded walk —
/// the extractor-side hooks of the fused folded-944 score+attribution
/// compare (campaign appendix N). Holds full-image copies of the pyramid
/// core rows and the phase-A V-blur planes — BITWISE the values the walk
/// computed (the C1 producer gate holds strip windows byte-equal to the
/// materialized pyramids, and phase A is the same blur chain
/// `attr_blur_cache_channel` replicates) — plus the EXACT per-cell pooled
/// accumulators and mean-gradients captured at finalize. Buffers are
/// reused across calls (`ensure`); every retained row a fused compare
/// later reads is written each walk, and the only conditionally-computed
/// plane (`bs2` where the append kernel is inactive — the
/// `APPEND_SKIP_B_SCALE0` cell) is zero-filled explicitly so session
/// reuse is deterministic. Memory: 8 planes × 3 ch × Σ scale sizes
/// (~42 MB at 576²) — the same class the standalone attribution's
/// materialized pyramids + plane sets pay.
pub(crate) struct FoldRetention {
    dims: Vec<(usize, usize)>,
    /// `[scale][ch]` source-side pyramid planes (core rows).
    pyr_src: Vec<[Vec<f32>; 3]>,
    /// `[scale][ch]` distorted-side pyramid planes (core rows).
    pyr_dst: Vec<[Vec<f32>; 3]>,
    /// `[scale][ch]` phase-A V-blur planes (mu1/mu2/ssq/s12/act/bs2).
    planes: Vec<[AttrChPlanes; 3]>,
    /// `[scale][ch]` exact pooled cells (walk accumulators + blockiness).
    cells: Vec<[AttrCellSums; 3]>,
    /// `[scale][ch]` (mean grad src, mean grad dst) from finalize.
    mg: Vec<[(f64, f64); 3]>,
}

impl Default for FoldRetention {
    fn default() -> Self {
        Self::new()
    }
}

impl FoldRetention {
    pub(crate) fn new() -> Self {
        Self {
            dims: Vec::new(),
            pyr_src: Vec::new(),
            pyr_dst: Vec::new(),
            planes: Vec::new(),
            cells: Vec::new(),
            mg: Vec::new(),
        }
    }

    /// Size every buffer for the walk's scale dims (no-op when unchanged;
    /// cells/mg/planes are fully rewritten by each retaining walk).
    fn ensure(&mut self, dims: &[(usize, usize)]) {
        if self.dims.as_slice() == dims {
            return;
        }
        self.dims = dims.to_vec();
        let mk = |n: usize| -> [Vec<f32>; 3] { std::array::from_fn(|_| vec![0.0f32; n]) };
        self.pyr_src = dims.iter().map(|&(w, h)| mk(w * h)).collect();
        self.pyr_dst = dims.iter().map(|&(w, h)| mk(w * h)).collect();
        self.planes = dims
            .iter()
            .map(|&(w, h)| std::array::from_fn(|_| AttrChPlanes::new(w * h)))
            .collect();
        self.cells = vec![[AttrCellSums::default(); 3]; dims.len()];
        self.mg = vec![[(0.0, 0.0); 3]; dims.len()];
    }

    /// Copy one (strip, channel)'s core rows out of the walk: pyramid
    /// windows + phase-A planes. Pure copies — the walk's own accumulation
    /// is untouched (G-N1 rests on this).
    fn copy_strip(
        &mut self,
        info: &crate::feature_v2_stream::StripInfo,
        ch: usize,
        scr: &ScratchV2Strip,
        src_win: &[f32],
        dst_win: &[f32],
        want_bs2: bool,
    ) {
        let w = info.plane_w;
        let off = HALO_P * w;
        let n = w * info.strip_h;
        let out = info.y0 * w;
        let scale = info.scale;
        self.pyr_src[scale][ch][out..out + n].copy_from_slice(&src_win[off..off + n]);
        self.pyr_dst[scale][ch][out..out + n].copy_from_slice(&dst_win[off..off + n]);
        let p = &mut self.planes[scale][ch];
        p.mu1[out..out + n].copy_from_slice(&scr.mu1[off..off + n]);
        p.mu2[out..out + n].copy_from_slice(&scr.mu2[off..off + n]);
        p.ssq[out..out + n].copy_from_slice(&scr.ssq[off..off + n]);
        p.s12[out..out + n].copy_from_slice(&scr.s12[off..off + n]);
        p.act[out..out + n].copy_from_slice(&scr.activity[off..off + n]);
        if want_bs2 {
            p.bs2[out..out + n].copy_from_slice(&scr.bs2[off..off + n]);
        } else {
            // Never computed for this cell (`APPEND_SKIP_B_SCALE0`) — its
            // pass-B terms carry zero coefficients, but zero the plane so
            // reuse across pairs is deterministic.
            p.bs2[out..out + n].fill(0.0);
        }
    }
}

/// One (strip, channel)'s wide input windows: zero-copy slices of the
/// producer's rolling planes for interior strips, or the scratch
/// `src_wide`/`dst_wide` buffers filled via `fill_wide` when the strip
/// touches the true top/bottom (reflection must be materialized). Both
/// forms hold IDENTICAL bytes for identical rows (C1's producer gate), so
/// every downstream kernel is bitwise-indifferent to which one it reads —
/// the zero-copy path just skips two `n_wide` memcpys per channel-strip.
/// Re-derive one (strip, channel)'s wide windows for a SHARED-borrow
/// consumer (phase B): zero-copy producer slices for interior strips, or
/// the scratch buffers phase A filled for edge strips. Identical bytes
/// either way (C1's producer gate).
fn stream_windows_shared<'w, S: ImageSource, D: ImageSource>(
    producer: &'w crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    scr: &'w ScratchV2Strip,
) -> (&'w [f32], &'w [f32]) {
    use crate::feature_v2_stream::Side;
    let n_wide = info.plane_w * info.wide_h();
    match (
        producer.wide_window(Side::Source, ch, info),
        producer.wide_window(Side::Distorted, ch, info),
    ) {
        (Some(sw), Some(dw)) => (sw, dw),
        _ => (&scr.src_wide[..n_wide], &scr.dst_wide[..n_wide]),
    }
}

/// The wide-window GATHER half of [`stream_phase_a`], for the self-blur band
/// shape that skips the rest of phase A.
///
/// Interior strips take the producer's rolling-plane slices directly and this
/// is a no-op; only a strip touching the true top or bottom needs the
/// reflect-padded copy into the scratch, which is exactly what
/// [`stream_windows_shared`] then reads back. Byte-identical to what phase A
/// does — it is the same two calls, lifted.
fn stream_gather_windows<S: ImageSource, D: ImageSource>(
    producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    scr: &mut ScratchV2Strip,
) {
    use crate::feature_v2_stream::Side;
    let n_wide = info.plane_w * info.wide_h();
    if producer.wide_window(Side::Source, ch, info).is_none()
        || producer.wide_window(Side::Distorted, ch, info).is_none()
    {
        producer.fill_wide(Side::Source, ch, info, &mut scr.src_wide[..n_wide]);
        producer.fill_wide(Side::Distorted, ch, info, &mut scr.dst_wide[..n_wide]);
    }
}

/// Phase A of one (strip, channel): resolve the wide windows (zero-copy
/// producer slices for interior strips; `fill_wide` into the scratch
/// buffers when the strip touches the true top/bottom), then the
/// materialized walk's moments-free blur pass (fused H + 4x V +
/// activity), plus the P2 strip-tiled bs2 σ-split chain when the append
/// cell is active (`abs_src`/`activity_tmp` are dead after the activity
/// chain and serve as its square/H temps).
fn stream_phase_a<S: ImageSource, D: ImageSource>(
    producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
    want_bs2: bool,
    want_act_dst: bool,
    want_v2: bool,
    // `parallel`: row-band-parallelise the H-blur
    // (`fused_blur_h_ssim_banded`). The channel fan-out above is only 3-way,
    // so on a >3-thread pool phase A is the walk's occupancy floor; bands are
    // its second axis.
    parallel: bool,
    scr: &mut ScratchV2Strip,
) {
    use crate::feature_v2_stream::Side;
    let width = info.plane_w;
    let wide_h = info.wide_h();
    let n_wide = width * wide_h;
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
        bs2,
        activity_dst,
    } = scr;
    let (src_win, dst_win): (&[f32], &[f32]) = match (
        producer.wide_window(Side::Source, ch, info),
        producer.wide_window(Side::Distorted, ch, info),
    ) {
        (Some(sw), Some(dw)) => (sw, dw),
        _ => {
            producer.fill_wide(Side::Source, ch, info, &mut src_wide[..n_wide]);
            producer.fill_wide(Side::Distorted, ch, info, &mut dst_wide[..n_wide]);
            (&src_wide[..n_wide], &dst_wide[..n_wide])
        }
    };
    run_blur_pass_inner(
        src_win,
        dst_win,
        width,
        wide_h,
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
        want_v2,
        parallel,
    );
    // BANDVIS dst self-mask (`append2_dst_activity`, Y channel only): the
    // exact dst twin of the ref activity chain — `box_blur(|dst − mu2|)`
    // over the same wide window, mu2 already produced by the fused blur
    // pass above. `abs_src`/`activity_tmp` are dead after the ref chain
    // and serve as its temps (the same reuse the bs2 fill below makes —
    // which runs after and overwrites them again, so ordering here is
    // load-bearing: act-dst BEFORE bs2).
    if want_act_dst {
        if activity_dst.len() < n_wide {
            activity_dst.resize(n_wide, 0.0);
        }
        crate::simd_ops::abs_diff_into(dst_win, &mu2[..n_wide], &mut abs_src[..n_wide]);
        crate::blur::box_blur_1pass_into(
            &abs_src[..n_wide],
            &mut activity_dst[..n_wide],
            &mut activity_tmp[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
    }
    if want_bs2 {
        square_into(src_win, &mut abs_src[..n_wide]);
        crate::blur::box_blur_h(
            &abs_src[..n_wide],
            &mut activity_tmp[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
        crate::blur::box_blur_v_from_copy(
            &activity_tmp[..n_wide],
            &mut bs2[..n_wide],
            width,
            wide_h,
            BLUR_RADIUS,
        );
    }
}

/// Phase B of one (strip, channel): the materialized walk's kernel set
/// over the phase-A planes, folded into `acc` in kernel-strip order.
///
/// `refy_strip` is the reference Y plane's strip rows (from the
/// producer's rolling plane — valid until the next `next_strip` call);
/// `cross` is `Some((act_x, act_b))` strip slices for the Y channel's
/// cross-masked transducer (X/B's phase-A activity, still live in their
/// scratches — computed once per strip, replacing the materialized pair
/// path's whole-plane replay planes).
///
/// LOCALITY CONTRACT: call this IMMEDIATELY after the same channel's
/// [`stream_phase_a`] — the fold band replay re-reads the H planes and
/// the kernels re-read the V planes while they are cache-hot (the same
/// reason the materialized walk runs its fold hook "before the dense
/// kernel purely for locality"). Interleaving another channel's phase A
/// in between evicts ~9 MB of wide buffers and was measured at
/// +50 ms/pair on aic3-100 (the C3 two-phase draft).
#[allow(clippy::too_many_arguments)]
fn stream_phase_b(
    scr: &ScratchV2Strip,
    src_win: &[f32],
    dst_win: &[f32],
    info: &crate::feature_v2_stream::StripInfo,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    v2_blocks: bool,
    band_parallel: bool,
    // `self_blur`: each v1 band produces its OWN H planes instead of reading
    // the strip-wide ones phase A wrote. Only legal when phase A did NOT run
    // (`!v2_blocks`) and `v1_pools` is `Full` (SelfBlur needs the band pool
    // scratch) — `foldapp_streaming_walk` owns that decision.
    self_blur: bool,
    append: bool,
    refy_strip: &[f32],
    cross: Option<(&[f32], &[f32])>,
    append2: Option<Append2Params>,
    csfw: Option<CsfwParams>,
    acc: &mut StreamChannelAccums,
) {
    let width = info.plane_w;
    let strip_h = info.strip_h;
    let y0 = info.y0;
    let scale = info.scale;
    let n_wide = width * info.wide_h();
    let off = HALO_P * width;
    let strip_n = width * strip_h;

    let src_strip = &src_win[off..off + strip_n];
    let dst_strip = &dst_win[off..off + strip_n];

    // BLOCK-SKIPPING: with `v1_only` the phase-A V planes and activity were
    // never produced, so nothing below may touch them. The v1 fold hook
    // reads only the H planes + the raw windows and is run unconditionally
    // below.
    if v2_blocks {
        let mu1_strip = &scr.mu1[off..off + strip_n];
        let mu2_strip = &scr.mu2[off..off + strip_n];
        let ssq_strip = &scr.ssq[off..off + strip_n];
        let s12_strip = &scr.s12[off..off + strip_n];
        let act_strip = &scr.activity[off..off + strip_n];

        let d = dense_block_kernel(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            s12_strip,
            act_strip,
            width,
            strip_h,
            toggles.transducer_bank,
        );
        acc.dense[scale].accumulate(&d);

        if toggles.gradient_features {
            let g_off = (HALO_P - 1) * width;
            let g_n = width * (strip_h + 2);
            // BANDVIS accumulates only on (Y, append2 on) — the const-split
            // instantiation; every other path runs the byte-identical
            // BANDVIS=false kernel. The dst self-mask plane routes in only
            // when `append2_dst_activity` is live (third instantiation).
            let bandvis = append2
                .filter(|_| cross.is_some())
                .map(|p| (p.bv_delta_lo, p.bv_delta_hi));
            let bv_act_dst = append2
                .filter(|p| cross.is_some() && p.dst_activity)
                .map(|_| &scr.activity_dst[off..off + strip_n]);
            let g = gradient_block_kernel(
                &src_win[g_off..g_off + g_n],
                &dst_win[g_off..g_off + g_n],
                act_strip,
                width,
                strip_h,
                bandvis,
                bv_act_dst,
            );
            acc.grad[scale].accumulate(&g);
        }
    }

    if fold_v1 {
        fold_v1_basic_bands(
            width,
            y0..y0 + strip_h,
            y0,
            HALO_P,
            info.plane_h,
            if self_blur {
                FoldHSource::SelfBlur
            } else {
                FoldHSource::Precomputed([
                    &scr.mu1_h[..n_wide],
                    &scr.mu2_h[..n_wide],
                    &scr.ssq_h[..n_wide],
                    &scr.s12_h[..n_wide],
                ])
            },
            [&src_win[..n_wide], &dst_win[..n_wide]],
            &mut acc.v1[scale],
            match toggles.v1_pools {
                V1PoolsMode::Off => None,
                // Peaks: the scratch travels only so a self-blur band owns
                // its H planes — no pool arithmetic runs (`HOnly`).
                V1PoolsMode::Peaks => Some((&mut acc.pool_scratch[..], BandPoolWork::HOnly)),
                // The carrier slots need the activity + edge kernel at
                // scales 0-1 only (masked_art_4th s0, iw_art_4th s0-s1);
                // the peaks come from the kernel at every scale for free.
                V1PoolsMode::Carriers if scale <= 1 => {
                    Some((&mut acc.pool_scratch[..], BandPoolWork::Carriers))
                }
                V1PoolsMode::Carriers => None,
                V1PoolsMode::Full => Some((&mut acc.pool_scratch[..], BandPoolWork::Full)),
            },
            band_parallel,
        );
    }

    if append {
        debug_assert!(v2_blocks, "append_block requires the v2 planes");
        debug_assert_eq!(refy_strip.len(), strip_n);
        // Re-bind the phase-A V planes: they are scoped to the `v2_blocks`
        // arm above, and `append` can only be set when that arm ran.
        let mu1_strip = &scr.mu1[off..off + strip_n];
        let mu2_strip = &scr.mu2[off..off + strip_n];
        let ssq_strip = &scr.ssq[off..off + strip_n];
        let act_strip = &scr.activity[off..off + strip_n];
        let hl = append2.map(|p| p.hl_bins).unwrap_or(false) && cross.is_some();
        let a = append_block_kernel(
            src_strip,
            dst_strip,
            mu1_strip,
            mu2_strip,
            ssq_strip,
            &scr.bs2[off..off + strip_n],
            act_strip,
            refy_strip,
            cross,
            hl,
            width,
            strip_h,
        );
        acc.app[scale].accumulate(&a);

        // CSFW pass accumulates only on (Y, csfw_block on) — a separate
        // kernel invocation over the strip-resident rows (design §4.2);
        // every other path never dispatches it. `cross.is_some()` is the
        // Y-channel discriminant, same as the BANDVIS gate above.
        if let Some(cp) = csfw.filter(|_| cross.is_some()) {
            let c = csfw_block_kernel(
                src_strip,
                dst_strip,
                refy_strip,
                cp.eff[scale],
                width,
                strip_h,
            );
            acc.csfw[scale].accumulate(&c);
        }
    }

    if v2_blocks && toggles.blockiness {
        let (sum_v, sum_h) = &mut acc.block[scale];
        blockiness_sparse_strip_wide(
            &src_win[..n_wide],
            &dst_win[..n_wide],
            width,
            y0,
            strip_h,
            sum_v,
            sum_h,
        );
    }
}

/// [`blockiness_sparse_rows`] over a strip's WIDE buffer: buffer row
/// `HALO_P + k` is plane row `y0 + k`. The horizontal family at plane row
/// `y0` (a lattice row whenever `y0 > 0` — kernel strips are 128-aligned)
/// reads its upper neighbor from the halo row `HALO_P − 1`, which is the
/// REAL plane row `y0 − 1` for every strip that can fire it (`y > 0`
/// never fires on the top strip's first row).
///
/// `sum_v`/`sum_h` are the CALLER's running totals, accumulated into
/// directly — NOT per-strip partials folded afterwards — so the global
/// f64 op sequence continues seamlessly across kernel strips and equals
/// [`blockiness_sparse_rows`]'s canonical whole-range form bit-for-bit
/// (a per-strip partial would reassociate: `(Σ strip0) + (Σ strip1)` ≠
/// one running sum — measured as a 1-ULP BLOCKINESS divergence when this
/// function briefly returned partials).
fn blockiness_sparse_strip_wide(
    src_wide: &[f32],
    dst_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    sum_v: &mut f64,
    sum_h: &mut f64,
) {
    for k in 0..strip_h {
        let y = y0 + k;
        let row = (HALO_P + k) * width;
        let mut x = BLOCK_LATTICE;
        while x < width {
            let i = row + x;
            let step_dst = (dst_wide[i] as f64 - dst_wide[i - 1] as f64).abs();
            let step_src = (src_wide[i] as f64 - src_wide[i - 1] as f64).abs();
            *sum_v += bounded_excess(step_dst, step_src, C_BLOCK);
            x += BLOCK_LATTICE;
        }
        if y.is_multiple_of(BLOCK_LATTICE) && y > 0 {
            for x in 0..width {
                let i = row + x;
                let i_up = i - width;
                let step_dst = (dst_wide[i] as f64 - dst_wide[i_up] as f64).abs();
                let step_src = (src_wide[i] as f64 - src_wide[i_up] as f64).abs();
                *sum_h += bounded_excess(step_dst, step_src, C_BLOCK);
            }
        }
    }
}

/// Streaming folded-720[+append] pair entry: validation + sub-64
/// reflect-pad exactly like the materialized pair entry
/// ([`compute_folded720_impl_with_toggles`] → prepare → with-ref inner),
/// then the streaming walk. Output is bitwise equal to the materialized
/// path (`streamed_foldapp_bitwise_vs_materialized`).
pub(crate) fn compute_folded720_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    // HDR routing (HDR_PLAN chunk 2) in the exact position the
    // unconditional `reject_hdr_input` calls held: a pair of
    // HDR-declared sources whose pixels are ABSOLUTE LINEAR cd/m²
    // (`LinearF32Rgba` + `is_hdr()`, the `compute_pu_linear` contract)
    // routes to the PU front-end with `HdrEncoding::Linear`. Every other
    // HDR-flagged shape (code-value formats without a declared transfer,
    // translucent alpha, mixed SDR/HDR pairs) still gets
    // `HdrInputRequiresPuPath` — the reject remains for every path the
    // HDR validation did not cover. sRGB sources take the identical
    // SDR path as before this routing existed (byte-stability gate).
    if source.is_hdr() || distorted.is_hdr() {
        fn linear_ok(
            is_hdr: bool,
            fmt: crate::source::PixelFormat,
            alpha: crate::source::AlphaMode,
        ) -> bool {
            is_hdr
                && fmt == crate::source::PixelFormat::LinearF32Rgba
                && matches!(alpha, crate::source::AlphaMode::Opaque)
        }
        if linear_ok(source.is_hdr(), source.pixel_format(), source.alpha_mode())
            && linear_ok(
                distorted.is_hdr(),
                distorted.pixel_format(),
                distorted.alpha_mode(),
            )
        {
            return compute_folded720_hdr_streaming_impl(
                source,
                distorted,
                HdrEncoding::Linear,
                max_pixels,
                parallel,
                toggles,
                scratch,
            );
        }
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras::default(),
        ));
    }
    Ok(foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras::default(),
    ))
}

/// **The fold-backed v1-372 extraction** (fold-engine lane, 2026-08-30):
/// the streaming walk asked for exactly what a v1 score needs and nothing
/// else — `v1_only` (skip every v2-era block *and* its upstream V-blur /
/// activity work; pure compute-skipping, gated by
/// `folded_v1_only_matches_full_walk`) plus [`V1PoolsMode::Full`] (emit
/// `f156..372` live). Returns `(features, mean_offset)` where
/// `features[0..372]` is **bit-identical to v1's 372-feature extraction** at
/// every width under option C — `v1_372_bit_exact_to_fold_at_every_width` is
/// the gate — and `mean_offset` reproduces
/// `streaming::compute_xyb_mean_offset` bit-exactly (see [`MeanOffsetRows`]).
///
/// The returned vector is the requesting regime's WIDTH (720 here) with
/// `f372..` at the structural `0.0`; the caller truncates to its config's
/// v1 width, which is a prefix (`[0,156)` basic · `[156,228)` peaks ·
/// `[228,300)` masked · `[300,372)` IW).
///
/// SDR only. A declared-HDR pair returns
/// [`ZensimError::HdrInputRequiresPuPath`] — the PU entries keep the
/// buffered walk until the fold's PU front-end grows a matching
/// mean-offset path.
pub(crate) fn compute_folded_v1_372_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    // `None` = `V1PoolsMode::Full` (compute everything), the unconditional
    // pre-2026-08-31 behaviour. `Some(Peaks)` is per-profile weight-skipping,
    // resolved by `fold_engine::score_pool_mode` from the bake's layer 0.
    pool_mode: Option<V1PoolsMode>,
) -> Result<(Vec<f64>, [f64; 3]), ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    if source.is_hdr() || distorted.is_hdr() {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let toggles = V2NewFeatureToggles {
        v1_pools: pool_mode.unwrap_or(V1PoolsMode::Full),
        v1_only: true,
        ..V2NewFeatureToggles::default()
    };
    // Sub-64 reflect-pad BEFORE the walk, exactly as
    // `metric::compute_with_config_inner` does — so both engines' features
    // AND mean_offset are taken over the same padded plane. The two arms are
    // written out rather than closed over because the padded and unpadded
    // sides are different `ImageSource` types.
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        let mut mo = MeanOffsetRows::new(padded_src.width(), padded_src.height());
        let res = foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                mean_offset: Some(&mut mo),
                ..Default::default()
            },
        );
        return Ok((res.into_features(), mo.finish()));
    }
    let mut mo = MeanOffsetRows::new(source.width(), source.height());
    let res = foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            mean_offset: Some(&mut mo),
            ..Default::default()
        },
    );
    Ok((res.into_features(), mo.finish()))
}

/// Whether a [`crate::streaming::PrecomputedReference`] can feed the fold's
/// source side for a walk over `(w, h)` COMPUTE dims — the ref-cached form's
/// admission test, and **the single owner of that predicate**.
///
/// Requires the cache to hold exactly [`crate::NUM_SCALES`] levels whose
/// dimensions match the fold's own floor-halving recurrence and whose planes
/// are tightly packed at each level's width. `PrecomputedReference` stops its
/// pyramid at `w < 8 || h < 8`, so a cache with fewer levels (or built at
/// different dims) is refused and the caller re-derives from the image
/// instead of scoring a mismatched pyramid.
pub(crate) fn cached_ref_feed_usable(
    scales: &[crate::streaming::XybPyramidLevel],
    w: usize,
    h: usize,
) -> bool {
    if scales.len() != crate::NUM_SCALES {
        return false;
    }
    let (mut sw, mut sh) = (w, h);
    for (planes, cw, ch) in scales.iter() {
        if *cw != sw || *ch != sh {
            return false;
        }
        if planes.iter().any(|p| p.len() != sw * sh) {
            return false;
        }
        sw /= 2;
        sh /= 2;
    }
    true
}

/// **The ref-cached fold-backed v1-372 extraction** (fold-engine lane stage
/// 3): [`compute_folded_v1_372_streaming_impl`] with the source side fed from
/// an already-built XYB pyramid, so N distorted candidates amortise one
/// reference's decode + sRGB→XYB conversion + 3-level downscale chain — the
/// shape `Zensim::precompute_reference` / `compute_with_ref` has always had on
/// the buffered path.
///
/// `distorted` must already be at the cache's COMPUTE dims (the caller
/// reflect-pads a sub-64 distorted, exactly as `Zensim::compute_with_ref`
/// does). Returns `None` when the cache cannot feed the fold
/// ([`cached_ref_feed_usable`]) — the caller then falls back rather than
/// scoring a different pyramid.
///
/// Bit-identical to N independent [`compute_folded_v1_372_streaming_impl`]
/// calls; `fold_ref_cache_matches_independent_computes` is the gate.
pub(crate) fn compute_folded_v1_372_with_ref_impl(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    parallel: bool,
    scratch: &mut V2Scratch,
    pool_mode: Option<V1PoolsMode>,
) -> Option<(Vec<f64>, [f64; 3])> {
    let (cw, ch) = (precomputed.scales[0].1, precomputed.scales[0].2);
    if distorted.width() != cw || distorted.height() != ch {
        return None;
    }
    if !cached_ref_feed_usable(&precomputed.scales, cw, ch) {
        return None;
    }
    let toggles = V2NewFeatureToggles {
        v1_pools: pool_mode.unwrap_or(V1PoolsMode::Full),
        v1_only: true,
        ..V2NewFeatureToggles::default()
    };
    let mut mo = MeanOffsetRows::new(cw, ch);
    // `source` is unused by the producer on the cached feed (every source-side
    // row is copied from the cache); it is still the type parameter, so pass
    // the distorted image and let the feed override it. The producer's
    // `ref_planes` branch is taken for EVERY side-0 fill, so no pixel of this
    // argument is ever read for side 0.
    let res = foldapp_streaming_walk(
        distorted,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            mean_offset: Some(&mut mo),
            ref_planes: Some(&precomputed.scales),
            ..Default::default()
        },
    );
    Some((res.into_features(), mo.finish()))
}

/// The declared-HDR folded/append streaming entry (HDR_PLAN chunk 2):
/// pixel values → absolute cd/m² per `encoding` → the UPIQ-validated
/// PU-XYB front-end → the UNCHANGED streaming 924 walk (same layout,
/// same formulas, same constants — `FeatureRegime` reports as the same
/// folded regimes; HDR-ness is a property of the extraction run, tracked
/// by the caller exactly like the fold-vs-ext regime rule).
///
/// Accepted sources: `LinearF32Rgba` (f32; cd/m² for `Linear`, code
/// values for `Pq`/`Hlg`) or `Srgb16Rgba` (u16 code values, `Pq`/`Hlg`
/// only) — both with `AlphaMode::Opaque` (the alpha noise-background
/// compositor is `[0,1]`-relative and NOT validated on absolute-light
/// pixels). `ColorPrimaries` are taken as-is (no gamut mapping — the
/// `compute_pu_linear` contract). Everything else:
/// `HdrInputRequiresPuPath`.
pub(crate) fn compute_folded720_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let shape_ok = |fmt: crate::source::PixelFormat, alpha: crate::source::AlphaMode| {
        let fmt_ok = match encoding {
            HdrEncoding::Linear => fmt == crate::source::PixelFormat::LinearF32Rgba,
            HdrEncoding::Pq { .. } | HdrEncoding::Hlg { .. } => {
                fmt == crate::source::PixelFormat::LinearF32Rgba
                    || fmt == crate::source::PixelFormat::Srgb16Rgba
            }
        };
        fmt_ok && matches!(alpha, crate::source::AlphaMode::Opaque)
    };
    if !shape_ok(source.pixel_format(), source.alpha_mode())
        || !shape_ok(distorted.pixel_format(), distorted.alpha_mode())
    {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let front_end = crate::feature_v2_stream::FrontEnd::Hdr(encoding);
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            front_end,
            scratch,
            FoldWalkExtras::default(),
        ));
    }
    Ok(foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        front_end,
        scratch,
        FoldWalkExtras::default(),
    ))
}

/// Folded-720+append+append2 pair entry (944; [`FeatureRegime::
/// Folded720Append2`]): forces `append_block` + `append2_block`.
pub(crate) fn compute_folded720_append2_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

/// Declared-HDR 944 entry: [`compute_folded720_hdr_streaming_impl`] with
/// append + append2 forced on (HL bins live; PU-domain BANDVIS δs).
pub(crate) fn compute_folded720_append2_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch,
    )
}

/// Streaming folded-944 extraction WITH attribution retention — the fused
/// compare's extractor-side hooks (campaign appendix N). Identical
/// validation + sub-64 reflect-pad to [`compute_folded720_streaming_impl`],
/// `append_block` + `append2_block` forced on, accumulation UNTOUCHED (the
/// hooks only copy), so the output features are BITWISE the canonical 944
/// extraction (G-N1 gate). SDR route only for now: HDR-declared inputs get
/// [`ZensimError::HdrInputRequiresPuPath`] (the 944 set is SDR-by-design;
/// an HDR fused route is registered future work).
///
/// Gated on `custom-profiles`: the fused entry ([`crate::Fused944Session`])
/// lives in the `attribution` module, which only exists under that feature —
/// same both-features rule as the `Fused944Session` re-export (af4417f8).
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_folded944_streaming_with_retention(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    retention: &mut FoldRetention,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    if source.is_hdr() || distorted.is_hdr() {
        return Err(ZensimError::HdrInputRequiresPuPath);
    }
    let toggles = V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        ..V2NewFeatureToggles::default()
    };
    if source.width() < crate::metric::MIN_PYRAMID_DIM
        || source.height() < crate::metric::MIN_PYRAMID_DIM
    {
        let padded_src = crate::metric::reflect_pad_to_min(source);
        let padded_dst = crate::metric::reflect_pad_to_min(distorted);
        return Ok(foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                retention: Some(retention),
                ..Default::default()
            },
        ));
    }
    Ok(foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            retention: Some(retention),
            ..Default::default()
        },
    ))
}

/// Reusable f32 pass-B scratch for the fused retention path (appendix P
/// lever 1): plane-sized buffers kept across compares in the
/// [`crate::Fused944Session`] so per-iteration callers pay no allocation /
/// page-fault cost.
///
/// `custom-profiles`-gated with the rest of the fused-retention cluster: its
/// only consumer is the `attribution` module (af4417f8 rule).
#[cfg(feature = "custom-profiles")]
#[derive(Default)]
pub(crate) struct PassBScratchF32 {
    pub(crate) canvas: Vec<f32>,
    pub(crate) scale_density: Vec<f32>,
    pub(crate) win_plane: Vec<f32>,
    pub(crate) spread_tmp: Vec<f32>,
    pub(crate) spread_out: Vec<f32>,
}

/// The v2/append/append2 attribution density built from walk RETENTION
/// (appendix N): coefficients derived from the EXACT streaming
/// accumulators + finalize mean-gradients, pass B over the retained
/// planes — sharing the coefficient derivation ([`derive_v2app_coeffs`] /
/// [`attr_ew_coeff`]) verbatim with the standalone
/// [`compute_v2_append_attribution`], with the f32 combine kernels
/// (appendix P lever 1: `attr_pass_b_rows_f32` — the standalone keeps the
/// strict f64 pass B). The retained planes are bitwise the standalone's
/// `AttrChPlanes` (producer gate + same blur chain), so the densities
/// differ ONLY by the coefficient inputs (exact accums vs the standalone's
/// 1e-9-parity pass-A replication) and the f32 combine — measured inside
/// the C3a tolerance class (G-N2/G-P2 gate). Returns the TRIMMED f32
/// density at `orig_w × orig_h` (the fused caller adds it into the f32
/// basic canvas; features stay owned by the real extraction).
///
/// `custom-profiles`-gated: the f32 pass B below reaches into
/// `crate::attribution` for its upsample kernel, and that module is
/// configured out without the feature — a `feature-regime-v2`-only build
/// failed with E0433 here until this gate (the 87c5e9ef note; same
/// both-features rule as the `Fused944Session` re-export, af4417f8).
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution_from_retention(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    orig_w: usize,
    orig_h: usize,
    scratch: &mut PassBScratchF32,
) -> Vec<f32> {
    let (w0, h0) = ret.dims[0];
    let n0 = w0 * h0;
    scratch.canvas.clear();
    scratch.canvas.resize(n0, 0.0);
    let mut canvas = std::mem::take(&mut scratch.canvas);
    retention_pass_b_all_scales(
        ret,
        s_v2,
        s_append,
        s_append2,
        parallel,
        scratch,
        &mut crate::attribution::AttrSinkF32::Canvas(&mut canvas),
    );
    // Trim the (possibly reflect-padded sub-64) canvas to the original.
    let out = if orig_w == w0 && orig_h == h0 {
        canvas.clone()
    } else {
        let mut out = Vec::with_capacity(orig_w * orig_h);
        for y in 0..orig_h.min(h0) {
            out.extend_from_slice(&canvas[y * w0..y * w0 + orig_w.min(w0)]);
        }
        out
    };
    scratch.canvas = canvas;
    out
}

/// Level-2 sibling of
/// [`compute_v2_append_attribution_from_retention`]: pass B folds each
/// scale straight into the caller's [`BinAccum`](crate::attribution::BinAccum)
/// — the session's full-resolution pass-B canvas stays untouched (and
/// unallocated on binned-only sessions), and no trimmed clone is made.
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution_from_retention_into_bins(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    scratch: &mut PassBScratchF32,
    accum: &mut crate::attribution::BinAccum,
) {
    retention_pass_b_all_scales(
        ret,
        s_v2,
        s_append,
        s_append2,
        parallel,
        scratch,
        &mut crate::attribution::AttrSinkF32::Bins(accum),
    );
}

/// Shared per-scale pass-B loop over the retention for both sinks.
#[cfg(feature = "custom-profiles")]
fn retention_pass_b_all_scales(
    ret: &FoldRetention,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    parallel: bool,
    scratch: &mut PassBScratchF32,
    sink: &mut crate::attribution::AttrSinkF32<'_>,
) {
    let n_scales = ret.dims.len();
    let (w0, h0) = ret.dims[0];
    let want_append = s_append.is_some();
    let n0 = w0 * h0;
    scratch.scale_density.resize(n0, 0.0);
    scratch.win_plane.resize(n0, 0.0);
    for scale in 0..n_scales {
        attr_pass_b_for_scale_f32(
            scale,
            [
                &ret.pyr_src[scale][0],
                &ret.pyr_src[scale][1],
                &ret.pyr_src[scale][2],
            ],
            [
                &ret.pyr_dst[scale][0],
                &ret.pyr_dst[scale][1],
                &ret.pyr_dst[scale][2],
            ],
            &ret.planes[scale],
            &ret.cells,
            &ret.mg,
            &ret.dims,
            n_scales,
            s_v2,
            s_append,
            s_append2,
            want_append,
            parallel,
            w0,
            h0,
            &mut scratch.scale_density,
            &mut scratch.win_plane,
            sink,
            &mut scratch.spread_tmp,
            &mut scratch.spread_out,
        );
    }
}

/// Folded-720+append+append2+CSFW pair entry (956; [`FeatureRegime::
/// Folded720Csfw`]): forces `append_block` + `append2_block` + `csfw_block`.
pub(crate) fn compute_folded720_csfw_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    toggles.csfw_block = true;
    compute_folded720_impl_with_toggles(source, distorted, max_pixels, parallel, toggles)
}

/// Declared-HDR 956 entry: [`compute_folded720_hdr_streaming_impl`] with
/// append + append2 + CSFW forced on (PU-route φ constants).
pub(crate) fn compute_folded720_csfw_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    toggles.append2_block = true;
    toggles.csfw_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch,
    )
}

/// [`compute_folded720_hdr_streaming_impl`] with the append block forced
/// on (the 924 shape).
pub(crate) fn compute_folded720_append_hdr_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    encoding: HdrEncoding,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_hdr_streaming_impl(
        source, distorted, encoding, max_pixels, parallel, toggles, scratch,
    )
}

/// [`compute_folded720_streaming_impl`] with the append block forced on
/// (the [`FeatureRegime::Folded720Append`] entry shape).
pub(crate) fn compute_folded720_append_streaming_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    mut toggles: V2NewFeatureToggles,
    scratch: &mut V2Scratch,
) -> Result<ZensimV2Result, ZensimError> {
    toggles.append_block = true;
    compute_folded720_streaming_impl(source, distorted, max_pixels, parallel, toggles, scratch)
}

/// The streaming walk body (inputs already validated + ≥ 64px).
/// `retention`: the fused compare's extractor-side hooks (appendix N) —
/// `Some` copies per-strip planes + finalize accumulators into the
/// [`FoldRetention`]; `None` (every pre-existing caller) is the untouched
/// walk. The hooks only COPY: accumulation, kernels, and finalize are
/// identical either way, so retained and plain walks emit bitwise-equal
/// features (G-N1).
/// Per-scale-0-row `Σ_x (src[x] − dst[x])` sums, in f64, one triple per row —
/// the decomposition that lets the streaming fold reproduce the buffered
/// path's `ZensimResult::mean_offset` **bit-exactly** rather than to a
/// tolerance.
///
/// `streaming::compute_xyb_mean_offset` sums 64-row chunks, each chunk
/// summing per-row f64 sums of `(s[i] − d[i]) as f64` — note the subtraction
/// happens in **f32** and only the difference is widened. Two properties make
/// it reproducible from a strip walk that visits scale-0 rows in a different
/// grouping:
///
/// 1. It is already **thread-invariant**: the rayon arm is an
///    order-preserving `into_par_iter().map().collect()` and the final
///    reduction walks that `Vec` serially in row order, so parallelism never
///    reassociates it.
/// 2. It **decomposes per row**: chunk `acc` is a left-to-right f64 sum of
///    per-row sums, so recording the per-row sums as the strips go past and
///    performing [`Self::finish`]'s identical chunk reduction afterwards
///    lands on the same bits.
///
/// Cost is one `Vec<[f64; 3]>` of `height` entries (55 KB at 2304², against
/// the walk's measured 163 MB peak RSS) plus one f32 subtract + f64 add per
/// scale-0 pixel per channel, taken while the producer's rolling plane rows
/// are already cache-hot.
/// Rows per task in [`MeanOffsetRows::add_strip`]'s fan-out. Purely a
/// schedule parameter — the values are per-row and assignment-only, so no
/// choice here can move a byte (`mean_offset_row_bands_are_bit_exact`).
#[cfg(feature = "threads")]
const MEAN_OFFSET_BAND_ROWS: usize = 16;

pub(crate) struct MeanOffsetRows {
    rows: Vec<[f64; 3]>,
    width: usize,
    height: usize,
}

impl MeanOffsetRows {
    pub(crate) fn new(width: usize, height: usize) -> Self {
        Self {
            rows: vec![[0.0f64; 3]; height],
            width,
            height,
        }
    }

    /// Record one scale-0 strip's rows for all three channels. `src`/`dst` are
    /// the producer's rolling-plane rows covering `[y0, y0 + n_rows)` at
    /// `plane_w` stride; `plane_w == width` at scale 0 under option C.
    ///
    /// **Row-parallel, and BIT-EXACT by construction** (fold-MT lane). Every
    /// `rows[y][ch]` is ASSIGNED — not accumulated into — by a pure function of
    /// that one row of `src`/`dst`, so each element is written exactly once
    /// with a value that does not depend on any other row, any other channel,
    /// or the order in which they run. The only ordered arithmetic is the
    /// left-to-right `f64` sum WITHIN a row, and that is never split.
    /// [`Self::finish`]'s 64-row-chunk reduction then reads `rows` in its own
    /// fixed order regardless.
    ///
    /// Why it earns a fan-out: this side channel is a full pass over the
    /// scale-0 planes for all three channels (127 MB read per walk at 2304²)
    /// and it ran SERIALLY between `next_strip` and the channel fan-out — it
    /// was the whole of the 7.1 ms the phase profile could not account for,
    /// i.e. 13 % of a 16-thread compare at degree 1.
    fn add_strip(
        &mut self,
        y0: usize,
        n_rows: usize,
        plane_w: usize,
        src: [&[f32]; 3],
        dst: [&[f32]; 3],
        #[allow(unused_variables)] parallel: bool,
    ) {
        let y1 = (y0 + n_rows).min(self.height);
        if y1 <= y0 {
            return;
        }
        let width = self.width;
        let rows = &mut self.rows[y0..y1];
        let one = |k_base: usize, out: &mut [[f64; 3]]| {
            for (i, o) in out.iter_mut().enumerate() {
                let k = k_base + i;
                for ch in 0..3 {
                    let s = &src[ch][k * plane_w..k * plane_w + width];
                    let d = &dst[ch][k * plane_w..k * plane_w + width];
                    let mut row_sum = 0.0f64;
                    for i in 0..width {
                        row_sum += (s[i] - d[i]) as f64;
                    }
                    o[ch] = row_sum;
                }
            }
        };
        #[cfg(feature = "threads")]
        if parallel && rows.len() > MEAN_OFFSET_BAND_ROWS {
            use rayon::prelude::*;
            rows.par_chunks_mut(MEAN_OFFSET_BAND_ROWS)
                .enumerate()
                .for_each(|(b, out)| one(b * MEAN_OFFSET_BAND_ROWS, out));
            return;
        }
        one(0, rows);
    }

    /// The exact reduction `streaming::compute_xyb_mean_offset` performs:
    /// 64-row chunks in ascending order, per channel a left-to-right sum of
    /// the chunk's row sums, then a left-to-right sum over chunks, then one
    /// divide by `width · height`. Reproduced loop-for-loop on purpose — this
    /// is the function whose *shape* is the guarantee.
    pub(crate) fn finish(&self) -> [f64; 3] {
        let n = (self.width * self.height) as f64;
        let chunk_rows = 64usize;
        let mut chunks: Vec<[f64; 3]> = Vec::with_capacity(self.height.div_ceil(chunk_rows));
        let mut row_start = 0usize;
        while row_start < self.height {
            let row_end = (row_start + chunk_rows).min(self.height);
            let mut diff = [0.0f64; 3];
            for (c, d) in diff.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for y in row_start..row_end {
                    acc += self.rows[y][c];
                }
                *d = acc;
            }
            chunks.push(diff);
            row_start += chunk_rows;
        }
        let mut offset = [0.0f64; 3];
        for chunk_diff in &chunks {
            for (o, &d) in offset.iter_mut().zip(chunk_diff.iter()) {
                *o += d;
            }
        }
        for o in &mut offset {
            *o /= n;
        }
        offset
    }
}

/// The fold walk's optional SIDE CHANNELS, bundled so the walk keeps one
/// options parameter instead of growing a positional tail.
///
/// All three are `None` on the plain extraction path, which is then
/// byte-for-byte the walk as it was before any of them existed.
#[derive(Default)]
pub(crate) struct FoldWalkExtras<'a> {
    /// Appendix-N retention hooks (the fused-944 attribution session).
    pub(crate) retention: Option<&'a mut FoldRetention>,
    /// Per-scale-0-row `Σ_x (src − dst)` sums, for the fold-backed engine's
    /// bit-exact `mean_offset` (see [`MeanOffsetRows`]).
    pub(crate) mean_offset: Option<&'a mut MeanOffsetRows>,
    /// A pre-built source-side XYB pyramid the producer copies from instead
    /// of decoding + converting + downscaling (the ref-cached fold form).
    pub(crate) ref_planes: Option<&'a [crate::streaming::XybPyramidLevel]>,
}

fn foldapp_streaming_walk<S: ImageSource, D: ImageSource>(
    source: &S,
    distorted: &D,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    front_end: crate::feature_v2_stream::FrontEnd,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> ZensimV2Result {
    let FoldWalkExtras {
        mut retention,
        mut mean_offset,
        ref_planes,
    } = extras;
    use crate::feature_v2_stream::StripPlaneProducer;
    let fold_v1 = true;
    // BLOCK-SKIPPING: a v1-only request computes NOTHING v2-era. Every
    // v2 toggle is forced off here rather than asserted, so the caller can
    // set `v1_only` on top of any existing toggle set and get the v1 blocks
    // of that request — which is what makes it a drop-in A/B in the bench.
    let v2_blocks = !toggles.v1_only;
    // BAND PARALLELISM: the channel fan-out is only 3-way, and the fold was
    // MEASURED to saturate at exactly 3 threads (2.27x) and REGRESS beyond
    // it. Bands inside a channel are independent, so they are the next axis;
    // rayon nests fine inside the channel fan-out.
    let band_parallel = parallel;
    // LAYOUT vs COMPUTE. The emitted vector keeps the WIDTH and REGIME the
    // caller asked for — a v1-only 944 request is still a 944 row, with
    // `f372..` at the structural 0.0 — while the `*_on` flags below drive
    // what is actually COMPUTED. Deriving the regime from the compute flags
    // instead would silently hand back a 720-wide vector for a 944 request.
    let layout_append = toggles.append_block;
    let layout_append2 = toggles.append2_block;
    let layout_csfw = toggles.csfw_block;
    let append_on = toggles.append_block && v2_blocks;
    let append2_on = toggles.append2_block && v2_blocks;
    assert!(
        !layout_append2 || layout_append,
        "append2_block requires append_block (f924+ sits after the append block)"
    );
    assert!(
        !toggles.append2_dst_activity || append2_on,
        "append2_dst_activity requires append2_block (it refines the BANDVIS lanes)"
    );
    let append2 = append2_on.then(|| {
        use crate::feature_v2_stream::FrontEnd;
        match front_end {
            FrontEnd::Sdr => Append2Params {
                bv_delta_lo: BV_DELTA_LO_SDR,
                bv_delta_hi: BV_DELTA_HI_SDR,
                hl_bins: false,
                dst_activity: toggles.append2_dst_activity,
            },
            FrontEnd::Hdr(_) => Append2Params {
                bv_delta_lo: BV_DELTA_LO_PU,
                bv_delta_hi: BV_DELTA_HI_PU,
                hl_bins: true,
                dst_activity: toggles.append2_dst_activity,
            },
        }
    });
    let csfw_on = toggles.csfw_block && v2_blocks;
    assert!(
        !layout_csfw || layout_append2,
        "csfw_block requires append2_block (f944+ sits after the append2 block)"
    );
    // Route-local derived φ: the SAME weighting mechanism on both routes,
    // pre-composed with each route's own encoding (design §6 — runtime
    // never inverts an encoding; the route-dependence lives entirely in
    // which constant set is selected, the `BV_DELTA_*` pattern).
    let csfw = csfw_on.then(|| {
        use crate::feature_v2_stream::FrontEnd;
        match front_end {
            FrontEnd::Sdr => CsfwParams::for_phi(CSFW_PHI_Y_SDR),
            FrontEnd::Hdr(_) => CsfwParams::for_phi(CSFW_PHI_Y_PU),
        }
    });
    let n_scales = crate::NUM_SCALES;
    let (w0, h0) = (source.width(), source.height());

    // Scale dims: floor-halving, matching both the producer and the
    // materialized walk's `downscale_2x_inplace` returns.
    let mut dims = Vec::with_capacity(n_scales);
    {
        let (mut w, mut h) = (w0, h0);
        for _ in 0..n_scales {
            dims.push((w, h));
            w /= 2;
            h /= 2;
        }
    }

    let strip_max_n = w0 * (STRIP_ROWS + 2 * HALO_P);
    // HOISTED so the strip scratch can be sized to the planes this walk will
    // actually write (fold-footprint lane). Both flags are strip-independent —
    // they were computed inside the loop purely because that is where they are
    // read — and the loop now reads these, so there is one definition of each.
    #[cfg(feature = "threads")]
    let fuse_channels = parallel && !append_on && retention.is_none();
    #[cfg(not(feature = "threads"))]
    let fuse_channels = false;
    let self_blur =
        fuse_channels && !v2_blocks && fold_v1 && matches!(toggles.v1_pools, V1PoolsMode::Full);
    scratch.ensure_for(
        strip_max_n,
        StripPlaneNeeds {
            // Phase A is skipped WHOLE under self-blur bands, so its four
            // fused-H outputs are never written.
            h: !self_blur,
            v2: v2_blocks,
        },
    );
    let V2Scratch {
        strips: scratch_strips,
        stream_pool,
        ..
    } = scratch;

    let mut accums: [StreamChannelAccums; 3] =
        std::array::from_fn(|_| StreamChannelAccums::new(n_scales, band_slots_for(band_parallel)));
    if let Some(ret) = retention.as_deref_mut() {
        ret.ensure(&dims);
    }
    // BANDVIS dst self-mask live: phase A additionally produces the Y
    // channel's dst-activity plane (and ONLY the Y channel's — the block
    // is Y-only, so X/B never pay the chain).
    let act_dst_on = append2.map(|p| p.dst_activity).unwrap_or(false);
    let mut producer = StripPlaneProducer::new_with_ref_feed(
        source,
        distorted,
        parallel,
        stream_pool,
        front_end,
        ref_planes,
    );

    let __t_walk = crate::fold_timing::start();
    loop {
        let __t_prod = crate::fold_timing::start();
        let next = producer.next_strip();
        let Some(info) = next else {
            crate::fold_timing::stop(__t_prod, crate::fold_timing::Phase::Producer, 0);
            break;
        };
        let scale = info.scale;
        crate::fold_timing::stop(__t_prod, crate::fold_timing::Phase::Producer, scale);

        // mean_offset side-channel (fold-engine lane): the scale-0 strips
        // tile [0, h0) exactly once in ascending order, so each row's
        // Σ_x (src − dst) is recorded exactly once here. Emitted for EVERY
        // strip layout — the walk's toggles never change which strips the
        // producer emits, only what is computed from them.
        if scale == 0
            && let Some(mo) = mean_offset.as_deref_mut()
        {
            let __t_mo = crate::fold_timing::start();
            let y1 = info.y0 + info.strip_h;
            let src: [&[f32]; 3] = core::array::from_fn(|ch| {
                producer.rows(crate::feature_v2_stream::Side::Source, ch, 0, info.y0, y1)
            });
            let dst: [&[f32]; 3] = core::array::from_fn(|ch| {
                producer.rows(
                    crate::feature_v2_stream::Side::Distorted,
                    ch,
                    0,
                    info.y0,
                    y1,
                )
            });
            mo.add_strip(info.y0, info.strip_h, info.plane_w, src, dst, parallel);
            crate::fold_timing::stop(__t_mo, crate::fold_timing::Phase::MeanOffset, scale);
        }

        // ref_y strip rows straight from the producer's rolling plane
        // (valid until the next `next_strip` call).
        let need_refy = append_on;
        let refy = if need_refy {
            producer.rows(
                crate::feature_v2_stream::Side::Source,
                1,
                scale,
                info.y0,
                info.y0 + info.strip_h,
            )
        } else {
            &[][..]
        };

        // FUSED PER-CHANNEL FAN-OUT (fold-MT lane). When no channel reads
        // another channel's phase-A output and nothing runs BETWEEN the two
        // phases, the split fan-out below buys nothing and costs the walk's
        // locality contract: it holds THREE channels' ~9 MB phase-A buffers
        // live across a barrier, then re-reads them from wherever they landed.
        // The serial arm already fuses A->B per channel for exactly that
        // reason (measured +50 ms/pair on aic3-100 when interleaved); this is
        // the same shape under threads.
        //
        // The two preconditions, checked rather than assumed:
        // * `!append_on` => `cross` is `None` on every channel and `refy` is
        //   empty, so Y needs nothing from X/B — the ONLY cross-channel edge
        //   in the walk.
        // * `retention.is_none()` => nothing runs between the phases (the
        //   retention hook is a `&mut` shared across channels and must stay
        //   in the split arm).
        // A `v1_only` SCORING request satisfies both by construction; the
        // 944-full product extraction satisfies neither and keeps the split.
        //
        // Byte-neutral by construction: identical kernels on identical inputs
        // into per-channel accumulators that are already disjoint. Only which
        // thread runs what, and when, changes.
        // `fuse_channels` and `self_blur` are HOISTED above the strip loop —
        // they are strip-independent, and the strip scratch is now SIZED from
        // them (`StripPlaneNeeds`), so there is one definition of each.
        //
        // SELF-BLUR BANDS. With `v1_only` phase A's ONLY output is the four
        // H-blurred planes over the strip's whole 148-row wide window, and the
        // v1 bands are its only consumer. Writing them there and reading them
        // back in a different task is four planes through L3/DRAM per strip —
        // which the 16-independent-process throughput test measures as the
        // fold's ceiling (4.2x against buffered's 10.9x from the same serial
        // speed; `benchmarks/fold_mt_scaling_2026-08-31.md`). Letting each
        // band blur exactly the rows it consumes into its own scratch drops
        // phase A entirely and makes every band a self-contained task — the
        // shape `streaming::process_channel_strip` has always had.
        //
        // `V1PoolsMode::Full` or `Peaks` is required because the band-local
        // H buffer lives in the band's `FoldPoolScratch`, and one of those is
        // what every fold-backed SCORE asks for. (`Peaks` hands the band the
        // scratch for the H planes alone — see `BandPoolWork::HOnly`.)
        #[cfg(feature = "threads")]
        let self_blur = fuse_channels
            && !v2_blocks
            && fold_v1
            && matches!(toggles.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks);
        #[cfg(feature = "threads")]
        if fuse_channels {
            use rayon::prelude::*;
            let __t_f = crate::fold_timing::start();
            scratch_strips
                .par_iter_mut()
                .zip(accums.par_iter_mut())
                .enumerate()
                .for_each(|(ch, (scr, acc))| {
                    let __t = crate::fold_timing::start();
                    if self_blur {
                        // Phase A is skipped whole; only the wide-window
                        // gather it also performs is still needed.
                        stream_gather_windows(&producer, &info, ch, scr);
                    } else {
                        stream_phase_a(&producer, &info, ch, false, false, v2_blocks, true, scr);
                    }
                    let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                    stream_phase_b(
                        scr,
                        src_win,
                        dst_win,
                        &info,
                        toggles,
                        fold_v1,
                        v2_blocks,
                        band_parallel,
                        self_blur,
                        false,
                        &[][..],
                        None,
                        append2,
                        csfw,
                        acc,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::PhaseBBusy, scale);
                });
            crate::fold_timing::stop(__t_f, crate::fold_timing::Phase::PhaseBWall, scale);
            continue;
        }

        #[cfg(feature = "threads")]
        let ran_parallel = if parallel {
            use rayon::prelude::*;
            // Parallel fan-out keeps the two-phase shape (A for all
            // channels, then B) — Y's cross inputs need X/B's phase A
            // done. Values are identical to the serial order (pure
            // kernels on identical inputs); only cache locality differs,
            // and parallel wall is not the 1-thread gate path.
            let __t_a = crate::fold_timing::start();
            scratch_strips
                .par_iter_mut()
                .enumerate()
                .for_each(|(ch, scr)| {
                    let __t = crate::fold_timing::start();
                    stream_phase_a(
                        &producer,
                        &info,
                        ch,
                        append_cell_active(append_on, ch, scale),
                        ch == 1 && act_dst_on && append_cell_active(append_on, 1, scale),
                        v2_blocks,
                        true,
                        scr,
                    );
                    crate::fold_timing::stop(__t, crate::fold_timing::Phase::PhaseABusy, scale);
                });
            crate::fold_timing::stop(__t_a, crate::fold_timing::Phase::PhaseAWall, scale);
            let __t_between = crate::fold_timing::start();
            let scratches: &[ScratchV2Strip; 3] = scratch_strips;
            // Retention hooks (appendix N): serial copies between the two
            // fan-outs — pure reads of the phase-A scratches + windows.
            if let Some(ret) = retention.as_deref_mut() {
                for (ch, scr) in scratches.iter().enumerate() {
                    let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                    ret.copy_strip(
                        &info,
                        ch,
                        scr,
                        src_win,
                        dst_win,
                        append_cell_active(append_on, ch, scale),
                    );
                }
            }
            crate::fold_timing::stop(__t_between, crate::fold_timing::Phase::Between, scale);
            let __t_b = crate::fold_timing::start();
            accums.par_iter_mut().enumerate().for_each(|(ch, acc)| {
                let __t = crate::fold_timing::start();
                let (src_win, dst_win) =
                    stream_windows_shared(&producer, &info, ch, &scratches[ch]);
                let cross = if ch == 1 && append_cell_active(append_on, ch, scale) {
                    let off = HALO_P * info.plane_w;
                    let strip_n = info.plane_w * info.strip_h;
                    Some((
                        &scratches[0].activity[off..off + strip_n],
                        &scratches[2].activity[off..off + strip_n],
                    ))
                } else {
                    None
                };
                stream_phase_b(
                    &scratches[ch],
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    append_cell_active(append_on, ch, scale),
                    refy,
                    cross,
                    append2,
                    csfw,
                    acc,
                );
                crate::fold_timing::stop(__t, crate::fold_timing::Phase::PhaseBBusy, scale);
            });
            crate::fold_timing::stop(__t_b, crate::fold_timing::Phase::PhaseBWall, scale);
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let ran_parallel = false;

        if !ran_parallel {
            // Serial: FUSED phase A+B per channel (the locality contract
            // in `stream_phase_b`'s doc — H/V planes stay cache-hot into
            // the band replay + kernels), and ONE shared scratch set for
            // all three channels — each stage overwrites the same ~9 MB
            // in place, so the working set stays L2/L3-resident across
            // channels AND strips (three per-channel sets measured
            // +15 ms/pair on aic3-100: each channel's buffers were
            // evicted by the other two between its consecutive strips).
            // Order X(0), B(2), then Y(1); Y's cross transducer reads
            // X/B's activity STRIP rows, stashed into the (otherwise
            // idle) second/third scratch sets' activity buffers — a
            // 2×strip_n copy, bitwise-identical values.
            let off = HALO_P * info.plane_w;
            let strip_n = info.plane_w * info.strip_h;
            let (s0, s_rest) = scratch_strips.split_at_mut(1);
            let scr = &mut s0[0];
            let (stash_x_buf, stash_b_buf) = s_rest.split_at_mut(1);
            let y_active = append_cell_active(append_on, 1, scale);
            for ch in [0usize, 2] {
                let active = append_cell_active(append_on, ch, scale);
                stream_phase_a(&producer, &info, ch, active, false, v2_blocks, false, scr);
                if y_active {
                    let stash = if ch == 0 {
                        &mut stash_x_buf[0].activity
                    } else {
                        &mut stash_b_buf[0].activity
                    };
                    stash[..strip_n].copy_from_slice(&scr.activity[off..off + strip_n]);
                }
                let (src_win, dst_win) = stream_windows_shared(&producer, &info, ch, scr);
                if let Some(ret) = retention.as_deref_mut() {
                    ret.copy_strip(&info, ch, scr, src_win, dst_win, active);
                }
                stream_phase_b(
                    scr,
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    active,
                    refy,
                    None,
                    append2,
                    csfw,
                    &mut accums[ch],
                );
            }
            {
                stream_phase_a(
                    &producer,
                    &info,
                    1,
                    y_active,
                    y_active && act_dst_on,
                    v2_blocks,
                    false,
                    scr,
                );
                let cross = if y_active {
                    Some((
                        &stash_x_buf[0].activity[..strip_n],
                        &stash_b_buf[0].activity[..strip_n],
                    ))
                } else {
                    None
                };
                let (src_win, dst_win) = stream_windows_shared(&producer, &info, 1, scr);
                if let Some(ret) = retention.as_deref_mut() {
                    ret.copy_strip(&info, 1, scr, src_win, dst_win, y_active);
                }
                stream_phase_b(
                    scr,
                    src_win,
                    dst_win,
                    &info,
                    toggles,
                    fold_v1,
                    v2_blocks,
                    band_parallel,
                    false,
                    y_active,
                    refy,
                    cross,
                    append2,
                    csfw,
                    &mut accums[1],
                );
            }
        }
    }

    if let Some(t) = __t_walk {
        crate::fold_timing::walk_done(t.elapsed().as_nanos() as u64);
    }
    producer.recycle(stream_pool);

    // --- Finalize: the materialized walk's per-scale epilogue, replayed
    //     on the merged accumulators (identical f64 ops on identical
    //     values ⇒ identical bits). ---
    let v1_total = if fold_v1 { n_scales * 3 * 31 } else { 0 };
    let v12_total = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let append_total = if layout_append {
        n_scales * 3 * FEATURES_PER_CHANNEL_APPEND
    } else {
        0
    };
    let append2_total = if layout_append2 {
        n_scales * APPEND2_PER_SCALE
    } else {
        0
    };
    let csfw_total = if layout_csfw {
        n_scales * CSFW_PER_SCALE
    } else {
        0
    };
    let mut features = vec![0.0f64; v12_total + append_total + append2_total + csfw_total];
    let (features_v12, features_tail) = features.split_at_mut(v12_total);
    let (features_app, features_tail2) = features_tail.split_at_mut(append_total);
    let (features_app2, features_csfw) = features_tail2.split_at_mut(append2_total);
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    #[allow(clippy::needless_range_loop)] // scale derives 3+ offsets across distinct arrays
    for scale in 0..n_scales {
        let (width, height) = dims[scale];
        let n = width * height;
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let app_scale_base = scale * 3 * FEATURES_PER_CHANNEL_APPEND;
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];

        for (ch, acc) in accums.iter().enumerate() {
            let out = &mut features_v12
                [v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL..]
                [..FEATURES_PER_CHANNEL_V2_TOTAL];
            let sum_blockiness = if toggles.blockiness {
                acc.block[scale].0 + acc.block[scale].1
            } else {
                0.0
            };
            grads[ch] =
                finish_channel_scale(&acc.dense[scale], &acc.grad[scale], sum_blockiness, n, out);
            if append_cell_active(append_on, ch, scale) {
                let out_app = &mut features_app
                    [app_scale_base + ch * FEATURES_PER_CHANNEL_APPEND..]
                    [..FEATURES_PER_CHANNEL_APPEND];
                finish_append(
                    &acc.dense[scale],
                    &acc.app[scale],
                    &acc.grad[scale],
                    n,
                    ch == 1,
                    toggles,
                    out_app,
                );
            }
            apply_transducer_luma_gate(out, ch, toggles);
        }

        // Retention hooks (appendix N): the EXACT pooled cells + the
        // mean-gradients this finalize just derived — the fused compare's
        // coefficient inputs (the standalone derives them from its own
        // 1e-9-parity pass-A replication instead).
        if let Some(ret) = retention.as_deref_mut() {
            for (ch, acc) in accums.iter().enumerate() {
                ret.cells[scale][ch] = AttrCellSums {
                    dense: acc.dense[scale],
                    grad: acc.grad[scale],
                    app: acc.app[scale],
                    blockiness: if toggles.blockiness {
                        acc.block[scale].0 + acc.block[scale].1
                    } else {
                        0.0
                    },
                    n,
                };
                ret.mg[scale][ch] = grads[ch];
            }
        }

        if fold_v1 {
            for (ch, acc) in accums.iter().enumerate() {
                let base = scale * 39 + ch * 13;
                acc.v1[scale].finalize_into(n, &mut features_v12[base..base + 13]);
                if toggles.v1_pools != V1PoolsMode::Off {
                    // v1's block-major 372 layout: [basic 156][peaks 72]
                    // [masked 72][iw 72], each pool block scale-major then
                    // channel-major, 6 per (scale, ch) (`metric.rs` passes
                    // 2/3/4). `n_scales`-generic: 13/6/6/6 per (scale, ch).
                    let cell = (scale * 3 + ch) * 6;
                    let peaks0 = n_scales * 39 + cell;
                    let masked0 = n_scales * 57 + cell;
                    let iw0 = n_scales * 75 + cell;
                    let mut peaks = [0.0f64; 6];
                    let mut masked = [0.0f64; 6];
                    let mut iw = [0.0f64; 6];
                    acc.v1[scale].finalize_pools_into(n, &mut peaks, &mut masked, &mut iw);
                    if toggles.v1_pools == V1PoolsMode::Full {
                        features_v12[peaks0..peaks0 + 6].copy_from_slice(&peaks);
                        features_v12[masked0..masked0 + 6].copy_from_slice(&masked);
                        features_v12[iw0..iw0 + 6].copy_from_slice(&iw);
                    } else if toggles.v1_pools == V1PoolsMode::Peaks {
                        // Peaks only; the masked/IW sums were never
                        // accumulated, so their slots stay the structural 0.
                        features_v12[peaks0..peaks0 + 6].copy_from_slice(&peaks);
                    } else {
                        // Carriers: only the ten `fused944native` slots go
                        // live (peak slot 4 = art_l8, masked/iw slot 3 =
                        // art_4th); everything else stays the structural 0.
                        for &slot in V1PoolsMode::CARRIER_SLOTS.iter() {
                            if slot == peaks0 + 4 {
                                features_v12[slot] = peaks[4];
                            } else if slot == masked0 + 3 {
                                features_v12[slot] = masked[3];
                            } else if slot == iw0 + 3 {
                                features_v12[slot] = iw[3];
                            }
                        }
                    }
                }
            }
        }

        for ch in 0..3 {
            let (gsrc, gdst) = grads[ch];
            if let Some((prev_gsrc, prev_gdst)) = prev_grad[ch] {
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
                let this_base = v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[this_base + idx::EDGE_WIDTH_CHANGE] =
                    features_v12[prev_base + idx::EDGE_WIDTH_CHANGE];
            }
        }
    }

    // --- append2 finalize (Y-only, per scale): BANDVIS means from the Y
    //     gradient accumulators, the free luminance conditioner from the
    //     Y append accumulator's Σs, and the HDR highlight bins. ---
    if append2_on {
        #[allow(clippy::needless_range_loop)] // scale derives offsets across distinct arrays
        for scale in 0..n_scales {
            let (width, height) = dims[scale];
            let n_f = (width * height) as f64;
            let base = scale * APPEND2_PER_SCALE;
            let gy = &accums[1].grad[scale];
            let ay = &accums[1].app[scale];
            let out2 = &mut features_app2[base..base + APPEND2_PER_SCALE];
            if toggles.gradient_features {
                out2[idx_append2::BANDVIS_GAIN] = (gy.sum_bv_gain / n_f).clamp(0.0, 1.0);
                out2[idx_append2::BANDVIS_LOSS] = (gy.sum_bv_loss / n_f).clamp(0.0, 1.0);
            }
            // Reference-only conditioner: sat(mean ref-Y, C_LUM_T) — the
            // same `t` mapping the luminance bins use per pixel.
            out2[idx_append2::LUMA_MEAN_REF] = saturate(ay.sum_s / n_f, C_LUM_T).clamp(0.0, 1.0);
            // HL bins: WeightedSum.finish() is 0 when Σw ≈ 0 (SDR route
            // or no highlight mass).
            out2[idx_append2::HL_BIN1] = ay.ws_hl1.finish().clamp(0.0, 1.0);
            out2[idx_append2::HL_BIN2] = ay.ws_hl2.finish().clamp(0.0, 1.0);
        }
    }

    // --- CSFW finalize (Y-only, per scale): the weighted GLOBAL_* twins
    //     from the Y CSFW accumulators (design §4.1). ---
    if csfw_on {
        for scale in 0..n_scales {
            let base = scale * CSFW_PER_SCALE;
            finish_csfw(
                &accums[1].csfw[scale],
                &mut features_csfw[base..base + CSFW_PER_SCALE],
            );
        }
    }

    ZensimV2Result {
        features,
        n_scales,
        v1_pools: if fold_v1 {
            toggles.v1_pools
        } else {
            V1PoolsMode::Off
        },
        regime: match (layout_append, layout_append2, layout_csfw) {
            (true, true, true) => FeatureRegime::Folded720Csfw,
            (true, true, false) => FeatureRegime::Folded720Append2,
            (true, false, _) => FeatureRegime::Folded720Append,
            (false, _, _) => FeatureRegime::Folded720,
        },
    }
}

fn compute_v2_features_with_ref_impl_inner(
    prepared: &V2PreparedReference,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
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
    let mut features = vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL];
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
    // small-image bypass) needs `width*height` rows of scratch — bump the
    // allocation to cover whichever is larger (`STRIP_BYPASS_HEIGHT=0`
    // today makes this `.min()` a permanent no-op).
    #[allow(clippy::unnecessary_min_or_max)]
    let bypass_rows = height.min(STRIP_BYPASS_HEIGHT);
    let strip_max_n = width * (STRIP_ROWS + 2 * HALO_P).max(bypass_rows);
    scratch.ensure(strip_max_n);
    let scratch = &mut scratch.strips;

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
        // `parallel` flag are active.
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];
        let out_region = &mut features[scale_base..][..3 * FEATURES_PER_CHANNEL_V2_TOTAL];

        #[cfg(feature = "threads")]
        let ran_parallel = if parallel {
            use rayon::prelude::*;
            let results: Vec<(f64, f64)> = out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .collect::<Vec<_>>()
                .into_par_iter()
                .zip(scratch.par_iter_mut())
                .enumerate()
                .map(|(ch, (out, scr))| {
                    let g = compute_channel_scale_v2(
                        &src_planes[ch],
                        &dst_planes[ch],
                        width,
                        height,
                        toggles,
                        moments_for(ch),
                        scr,
                        out,
                    );
                    apply_transducer_luma_gate(out, ch, toggles);
                    g
                })
                .collect();
            grads = [results[0], results[1], results[2]];
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let ran_parallel = false;

        if !ran_parallel {
            for (ch, out) in out_region
                .chunks_mut(FEATURES_PER_CHANNEL_V2_TOTAL)
                .enumerate()
            {
                grads[ch] = compute_channel_scale_v2(
                    &src_planes[ch],
                    &dst_planes[ch],
                    width,
                    height,
                    toggles,
                    moments_for(ch),
                    &mut scratch[ch],
                    out,
                );
                apply_transducer_luma_gate(out, ch, toggles);
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
                let prev_base = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features[prev_base + idx::EDGE_WIDTH_CHANGE] =
                    1.0 - bounded_sim(decay_src, decay_dst, C_EDGEWIDTH);
            }
            prev_grad[ch] = Some((gsrc, gdst));

            if scale == n_scales - 1 && n_scales >= 2 {
                // Coarsest scale has no next scale to compare against --
                // duplicate the previous (second-coarsest) scale's value
                // (documented approximation; see module/spec doc).
                let this_base = scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                let prev_base = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features[this_base + idx::EDGE_WIDTH_CHANGE] =
                    features[prev_base + idx::EDGE_WIDTH_CHANGE];
            }
        }

        if scale + 1 < n_scales {
            // BUG TRAP: all 3 channels must downscale from the SAME
            // (width, height) — only update the outer width/height AFTER
            // the per-channel loop (see the test-suite-caught corruption
            // this comment originally documented).
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
        v1_pools: V1PoolsMode::Off,
        regime: FeatureRegime::V2Bounded,
    })
}

// ============================================================================
// Attribution densities for the v2 (f372-719) + append (f720-923) blocks
// (task #67 C2a — the coverage levers the C1 basic density measured missing).
//
// Same contract as `crate::attribution`: per-pixel density whose rectangle
// sum over a block approximates `Σ_k s_k·Δf_k(block)` — the first-order
// score change from refining the block to reference — with TRUE integrands
// and ABSOLUTE normalization. `s` slices are the RAW model gradient
// `∂score/∂f_k`; the sign fold happens via the `density = Σ_k s_k·(δf_k)_i`
// rule ((δf_k)_i = the first-order feature change from refining pixel i's
// content, e.g. `−v_i/N` for a mean-pooled error signal).
//
// Integrand classes (all derived from the production kernels in this file):
//  - mean pools (v2: ssim_mean/art/det/mse/hf×3/pjnd×3/gms/ringing/banding;
//    append: xmask/lumt/mscn×2/contrast×2/tex): (δf)_i = −v_i/N. Exact.
//  - reference-weighted pools (masked/iw, append luminance bins):
//    (δf)_i = −w_i·v_i/Σw (w reference-side). Exact.
//  - self-weighted soft-peak (w_i = sat(v_i, C_PEAK), f = Σw·v/Σw):
//    (δf)_i = w(v_i)·(f − v_i)/Σw — SIGNED; full-plane sum is exactly 0.
//  - deviation pools (v2 SSIM_DEV2/DEV4; append GMS/ART/DET_DEV2):
//    central-moment chain rule, SIGNED (zeroing a mean-valued pixel RAISES
//    the deviation). DEV2: (δf)_i = (2μv−v²)/(2f·N); DEV4 via raw moments.
//  - global slots (append GLOBAL_DMEAN/CGAIN/CLOSS): exact chain rule on
//    whole-plane means/variances, SIGNED.
//  - blockiness: lattice-step terms, split 50/50 across the step pair.
//  - EDGE_WIDTH_CHANGE: exact two-scale chain rule on the adjacent-scale
//    mean-gradient ratios (incl. the last-scale copy's weight).
//  - reference-only (PJND_FRAGILITY, GRAD_SRC_MEAN): (δf)_i = 0 exactly.
//  - structural zeros (X/B transducer slots, the (B, scale 0) append cell):
//    features are constant 0.0 in the regime ⇒ Δf ≡ 0 ⇒ integrand 0,
//    regardless of the probed gradient (mirrors the harness's f156-371
//    structural-zero handling).
//
// The finalize clamps (`clamp01`/`clamp02`) are treated as inert (features
// live strictly inside their bounds on real content; at a saturated clamp
// the true gradient is 0 and the density would over-attribute — accepted
// and documented as an approximation).
//
// Pass A replays the MATERIALIZED strip walk (STRIP_ROWS + HALO_P halos,
// `run_blur_pass_strip`, the σ-split `bs2` chain from `stream_phase_a`) and
// then runs the PRODUCTION kernels over the cached planes, so every pooled
// scalar feeding a coefficient is production-arithmetic (parity-gated by
// `v2_append_attr_features_match_production`). Pass B re-derives the
// per-pixel signals in f64 from the same planes — the density's plane sums
// therefore match production features to per-pixel f32-vs-f64 recompute
// noise (~1e-7 rel), not 1e-9; the pooled-scalar parity is the strict gate.
// ============================================================================

// Coarse per-section timers for ZENSIM_ATTR_PERF=1 (seconds, thread-local).
thread_local! {
    static PERF_BLUR: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static PERF_KERN: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
    static PERF_PASSB: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

/// Output of [`compute_v2_append_attribution`].
#[cfg(feature = "custom-profiles")]
pub(crate) struct V2AppendAttribution {
    /// Full-resolution density (f64, `width × height`, trimmed to the
    /// original image), in score units per pixel.
    pub density: Vec<f64>,
    pub width: usize,
    pub height: usize,
    /// Pass-A finalized v2 features (`n_scales·3·29`, scale-major) — the
    /// production-kernel pooled values the coefficients were derived from.
    /// Read by the 1e-9 production-parity gate
    /// (`v2_append_attr_features_match_production`); retained on the struct
    /// as the audit surface for any consumer of the density.
    #[allow(dead_code)]
    pub v2_features: Vec<f64>,
    /// Pass-A finalized append features (`n_scales·3·17`), empty when the
    /// append block was not requested. Same audit role as `v2_features`.
    #[allow(dead_code)]
    pub append_features: Vec<f64>,
}

/// Per-(scale, channel) pass-A pooled state.
///
/// Stays compiled in every configuration (it is a [`FoldRetention`] field
/// type, and the walk's retention hooks write it), but its fields are only
/// READ by the `custom-profiles`-gated attribution-density cluster — hence
/// the targeted dead-code allowance.
#[cfg_attr(not(feature = "custom-profiles"), allow(dead_code))]
#[derive(Default, Clone, Copy)]
struct AttrCellSums {
    dense: DenseAccum,
    grad: GradientAccum,
    app: AppendAccum,
    blockiness: f64,
    n: usize,
}

/// Per-scale per-channel cached planes (scale-plane sized slices of
/// scale-0-sized buffers).
struct AttrChPlanes {
    mu1: Vec<f32>,
    mu2: Vec<f32>,
    ssq: Vec<f32>,
    s12: Vec<f32>,
    act: Vec<f32>,
    bs2: Vec<f32>,
}

impl AttrChPlanes {
    fn new(n: usize) -> Self {
        Self {
            mu1: vec![0.0; n],
            mu2: vec![0.0; n],
            ssq: vec![0.0; n],
            s12: vec![0.0; n],
            act: vec![0.0; n],
            bs2: vec![0.0; n],
        }
    }
}

/// Blur + cache one (scale, channel): materialized strip walk
/// (`gather_strip_halo` + `run_blur_pass_strip` + the `stream_phase_a`
/// σ-split chain for `bs2`), core rows copied into `planes`.
#[cfg(feature = "custom-profiles")]
fn attr_blur_cache_channel(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    want_bs2: bool,
    scratch: &mut ScratchV2Strip,
    planes: &mut AttrChPlanes,
) {
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
        gather_strip_halo(
            dst,
            width,
            height,
            y0,
            wide_h,
            HALO_P,
            &mut scratch.dst_wide[..n_wide],
        );
        run_blur_pass_strip(width, wide_h, scratch);
        if want_bs2 {
            // Same chain as `stream_phase_a`: square(src) → H-box → V-box.
            // `abs_src`/`activity_tmp` are free after the activity chain.
            square_into(&scratch.src_wide[..n_wide], &mut scratch.abs_src[..n_wide]);
            crate::blur::box_blur_h(
                &scratch.abs_src[..n_wide],
                &mut scratch.activity_tmp[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
            crate::blur::box_blur_v_from_copy(
                &scratch.activity_tmp[..n_wide],
                &mut scratch.bs2[..n_wide],
                width,
                wide_h,
                BLUR_RADIUS,
            );
        }
        let off = HALO_P * width;
        let strip_n = width * strip_h;
        let out = y0 * width;
        planes.mu1[out..out + strip_n].copy_from_slice(&scratch.mu1[off..off + strip_n]);
        planes.mu2[out..out + strip_n].copy_from_slice(&scratch.mu2[off..off + strip_n]);
        planes.ssq[out..out + strip_n].copy_from_slice(&scratch.ssq[off..off + strip_n]);
        planes.s12[out..out + strip_n].copy_from_slice(&scratch.s12[off..off + strip_n]);
        planes.act[out..out + strip_n].copy_from_slice(&scratch.activity[off..off + strip_n]);
        if want_bs2 {
            planes.bs2[out..out + strip_n].copy_from_slice(&scratch.bs2[off..off + strip_n]);
        }
        y0 += strip_h;
    }
}

/// Pass A kernels for one (scale, channel) over cached planes: production
/// dense/gradient/append kernels in kernel-strip order + canonical
/// blockiness. `cross` = `(act_x, act_b)` full planes for the Y channel.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_a_kernels(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    want_append: bool,
    append_active: bool,
    grad_halo: &mut Vec<f32>,
    grad_halo_d: &mut Vec<f32>,
) -> AttrCellSums {
    let mut cell = AttrCellSums {
        n: width * height,
        ..Default::default()
    };
    let mut y0 = 0usize;
    while y0 < height {
        let strip_h = STRIP_ROWS.min(height - y0);
        let base = y0 * width;
        let strip_n = width * strip_h;
        let src_s = &src[base..base + strip_n];
        let dst_s = &dst[base..base + strip_n];
        let d = dense_block_kernel(
            src_s,
            dst_s,
            &planes.mu1[base..base + strip_n],
            &planes.mu2[base..base + strip_n],
            &planes.ssq[base..base + strip_n],
            &planes.s12[base..base + strip_n],
            &planes.act[base..base + strip_n],
            width,
            strip_h,
            true, // transducer_bank (V2NewFeatureToggles::default())
        );
        cell.dense.accumulate(&d);
        // Gradient needs a 1-row halo; gather from the full planes (real
        // rows except the reflect_101 mirror at the true image edges —
        // identical values to the production wide-window slices).
        let g_n = width * (strip_h + 2);
        grad_halo.resize(g_n, 0.0);
        grad_halo_d.resize(g_n, 0.0);
        gather_strip_halo(src, width, height, y0, strip_h + 2, 1, grad_halo);
        gather_strip_halo(dst, width, height, y0, strip_h + 2, 1, grad_halo_d);
        let g = gradient_block_kernel(
            grad_halo,
            grad_halo_d,
            &planes.act[base..base + strip_n],
            width,
            strip_h,
            None,
            None,
        );
        cell.grad.accumulate(&g);
        if want_append && append_active {
            let a = append_block_kernel(
                src_s,
                dst_s,
                &planes.mu1[base..base + strip_n],
                &planes.mu2[base..base + strip_n],
                &planes.ssq[base..base + strip_n],
                &planes.bs2[base..base + strip_n],
                &planes.act[base..base + strip_n],
                &ref_y[base..base + strip_n],
                cross.map(|(x, b)| (&x[base..base + strip_n], &b[base..base + strip_n])),
                false, // hl (append2) — not part of the 924 regime
                width,
                strip_h,
            );
            cell.app.accumulate(&a);
        }
        y0 += strip_h;
    }
    cell.blockiness = blockiness_sparse(src, dst, width, height);
    cell
}

/// All deferred pass-B coefficients for one (scale, channel), derived from
/// the raw gradients and the pass-A pooled sums. Field semantics: each is
/// the multiplier on the named per-pixel term in the pass-B combine (the
/// `s_k` sign fold and `1/N` already applied).
#[cfg(feature = "custom-profiles")]
#[derive(Default, Clone, Copy)]
struct V2AppCoeffs {
    // v2 mean-pool slots: coefficient × per-pixel value.
    c_ssim: f64,
    c_art: f64,
    c_det: f64,
    c_mse: f64,
    c_hf_gain: f64,
    c_hf_loss: f64,
    c_hf_mag: f64,
    c_pjnd: f64,
    c_pjnd_lo: f64,
    c_pjnd_hi: f64,
    c_gms: f64,
    c_ringing: f64,
    c_banding: f64,
    c_blockiness: f64,
    // v2 weighted pools: coefficient × (w · v); w recomputed per pixel.
    c_mask_ssim: f64,
    c_mask_art: f64,
    c_mask_det: f64,
    c_mask_mse: f64,
    c_iw_ssim: f64,
    c_iw_art: f64,
    c_iw_det: f64,
    c_iw_mse: f64,
    // soft-peak: s_k/W and the pooled f per member.
    sp_ssim: (f64, f64),
    sp_art: (f64, f64),
    sp_det: (f64, f64),
    // ssim dev pools: μ, raw2, raw3, f2, f4 and the two s_k.
    dev_mu: f64,
    dev_raw2: f64,
    dev_raw3: f64,
    dev_f2: f64,
    dev_f4: f64,
    s_dev2: f64,
    s_dev4: f64,
    // append mean slots.
    c_xmask: f64,
    c_lumt: f64,
    c_mscn: f64,
    c_mscn2: f64,
    c_cgain: f64,
    c_closs: f64,
    c_tex: f64,
    // append luminance bins: coefficient × (w · mse_i) per bin.
    c_dark: f64,
    c_mid: f64,
    c_bright: f64,
    // append dev pools (gms/art/det): (s_k, μ, f) per member.
    gd_gms: (f64, f64, f64),
    gd_art: (f64, f64, f64),
    gd_det: (f64, f64, f64),
    // append globals.
    g_dmean: f64, // coefficient × (s − d)
    g_var: f64,   // coefficient × ((s²−d²) − 2·gmean_d·(s−d))
    gmean_d: f64,
    // edge-width: coefficient × (|∇d| − |∇s|).
    c_edgew: f64,
    // append2 BANDVIS pair (f924+, Y-only): coefficient × the per-pixel
    // FR-excess indicator. Mean pools like the v2 hf pair — `−s_k/N`.
    c_bv_gain: f64,
    c_bv_loss: f64,
    inv_n: f64,
}

#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn derive_v2app_coeffs(
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    scale: usize,
    ch: usize,
    cell: &AttrCellSums,
    append_active: bool,
    cross: bool,
    edgew_coeff: f64,
) -> V2AppCoeffs {
    let n_f = cell.n as f64;
    let inv_n = 1.0 / n_f;
    let v2b = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
    let gv2 = |slot: usize| s_v2.get(v2b + slot).copied().unwrap_or(0.0);
    let apb = scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
    let gap = |slot: usize| -> f64 {
        s_append
            .and_then(|s| s.get(apb + slot))
            .copied()
            .unwrap_or(0.0)
    };
    let dense = &cell.dense;
    let app = &cell.app;
    let grad = &cell.grad;

    let mut c = V2AppCoeffs {
        inv_n,
        // Mean pools: density += s_k · (−v_i/N).
        c_ssim: -gv2(idx::SSIM_MEAN) * inv_n,
        c_art: -gv2(idx::ART) * inv_n,
        c_det: -gv2(idx::DET) * inv_n,
        c_mse: -gv2(idx::MSE) * inv_n,
        c_hf_gain: -gv2(idx::HF_GAIN) * inv_n,
        c_hf_loss: -gv2(idx::HF_LOSS) * inv_n,
        c_hf_mag: -gv2(idx::HF_MAG_LOSS) * inv_n,
        c_pjnd: -gv2(idx::PJND_TRANSDUCER) * inv_n,
        c_pjnd_lo: -gv2(idx::PJND_TRANSDUCER_LOW_K) * inv_n,
        c_pjnd_hi: -gv2(idx::PJND_TRANSDUCER_HIGH_K) * inv_n,
        c_gms: -gv2(idx::GMS) * inv_n,
        c_ringing: -gv2(idx::RINGING) * inv_n,
        c_banding: -gv2(idx::BANDING) * inv_n,
        c_blockiness: -gv2(idx::BLOCKINESS) * inv_n,
        c_edgew: edgew_coeff,
        ..Default::default()
    };

    // ── append2 (f924-943; `idx_append2`), Y-only, no channel axis ────────
    // Layout `f924 + scale*APPEND2_PER_SCALE + local`. Registered
    // per-slot decomposability (campaign appendix E.1 / the
    // `slot_decomposability` TSV) — the block splits, it is not uniform:
    //
    //  * BANDVIS_GAIN / BANDVIS_LOSS — class **E**. Plain mean over the
    //    plane of a per-pixel `bounded_excess_pair` indicator, i.e. the
    //    exact pooling form of the v2 HF_GAIN/HF_LOSS/HF_MAG_LOSS slots:
    //    `(δf)_i = −v_i/N`. Spatialized below.
    //  * LUMA_MEAN_REF — class **N by definition**: `sat(mean(ref Y))` is
    //    REFERENCE-ONLY, so `∂f/∂(distorted) ≡ 0` and a zero density is the
    //    exactly-correct answer (same footing as v2 `PJND_FRAGILITY` and
    //    append `GRAD_SRC_MEAN`, which also carry no term here).
    //  * HL_BIN1 / HL_BIN2 — class **N (structural zero) on this route**.
    //    Their form (`Σw·mse_i/Σw`, reference-side w) IS class-E, identical
    //    to the append luminance bins — but they are computed only under the
    //    HDR const-generic `HL`, and this attribution path is structurally
    //    SDR (`attr_pass_a_kernels` passes `hl = false`; both sides come
    //    through the SDR `prepare_v2_reference_impl`). On the SDR route
    //    `Σw ≡ 0` ⇒ `WeightedSum::finish() ≡ 0` ⇒ the feature is identically
    //    0.0 ⇒ `Δf ≡ 0` for any probed gradient — the same footing as the
    //    X/B transducer slots and the `APPEND_SKIP_B_SCALE0` cell.
    //
    // So exactly 2 of the 5 local slots carry a term, and the other 3 are
    // deliberately, explicitly zero rather than silently unreached.
    if ch == APPEND2_CHANNEL {
        let ap2b = scale * APPEND2_PER_SCALE;
        let gap2 = |slot: usize| -> f64 {
            s_append2
                .and_then(|s| s.get(ap2b + slot))
                .copied()
                .unwrap_or(0.0)
        };
        c.c_bv_gain = -gap2(idx_append2::BANDVIS_GAIN) * inv_n;
        c.c_bv_loss = -gap2(idx_append2::BANDVIS_LOSS) * inv_n;
        // idx_append2::LUMA_MEAN_REF / HL_BIN1 / HL_BIN2: no term, by the
        // classification above. Deliberate omission, not an oversight.
    }

    // Weighted pools: density += s_k · (−w_i·v_i/Σw). Σw is shared per
    // family (same weight for all 4 members).
    let mask_den = dense.ws_mask_ssim.den;
    if mask_den > 0.0 {
        c.c_mask_ssim = -gv2(idx::MASKED_SSIM) / mask_den;
        c.c_mask_art = -gv2(idx::MASKED_ART) / mask_den;
        c.c_mask_det = -gv2(idx::MASKED_DET) / mask_den;
        c.c_mask_mse = -gv2(idx::MASKED_MSE) / mask_den;
    }
    let iw_den = dense.ws_iw_ssim.den;
    if iw_den > 0.0 {
        c.c_iw_ssim = -gv2(idx::IW_SSIM) / iw_den;
        c.c_iw_art = -gv2(idx::IW_ART) / iw_den;
        c.c_iw_det = -gv2(idx::IW_DET) / iw_den;
        c.c_iw_mse = -gv2(idx::IW_MSE) / iw_den;
    }
    // Soft-peak (self-weighted): density += s_k · w(v_i)·(f − v_i)/W.
    let sp = |ws: &WeightedSum, sk: f64| -> (f64, f64) {
        if ws.den > 0.0 {
            (sk / ws.den, ws.num / ws.den)
        } else {
            (0.0, 0.0)
        }
    };
    c.sp_ssim = sp(&dense.ws_peak_ssim, gv2(idx::SSIM_SOFT_PEAK));
    c.sp_art = sp(&dense.ws_peak_art, gv2(idx::ART_SOFT_PEAK));
    c.sp_det = sp(&dense.ws_peak_det, gv2(idx::DET_SOFT_PEAK));
    // SSIM deviation pools (central moments of the d map).
    let mu = dense.sum_d * inv_n;
    let raw2 = dense.sum_d2 * inv_n;
    let raw3 = dense.sum_d3 * inv_n;
    let raw4 = dense.sum_d4 * inv_n;
    let m2 = (raw2 - mu * mu).max(0.0);
    let m4 = (raw4 - 4.0 * mu * raw3 + 6.0 * mu * mu * raw2 - 3.0 * mu.powi(4)).max(0.0);
    c.dev_mu = mu;
    c.dev_raw2 = raw2;
    c.dev_raw3 = raw3;
    c.dev_f2 = m2.sqrt();
    c.dev_f4 = m4.powf(0.25);
    c.s_dev2 = gv2(idx::SSIM_DEV2);
    c.s_dev4 = gv2(idx::SSIM_DEV4);

    if append_active {
        if cross {
            c.c_xmask = -gap(idx_append::XMASK_TRANSDUCER) * inv_n;
            c.c_lumt = -gap(idx_append::LUM_TRANSDUCER) * inv_n;
        }
        c.c_mscn = -gap(idx_append::MSCN_DIFF_MEAN) * inv_n;
        c.c_mscn2 = -gap(idx_append::MSCN_DIFF_L2) * inv_n;
        c.c_cgain = -gap(idx_append::CONTRAST_GAIN) * inv_n;
        c.c_closs = -gap(idx_append::CONTRAST_LOSS) * inv_n;
        c.c_tex = -gap(idx_append::TEXTURE_DISSIM) * inv_n;
        if app.ws_dark.den > 0.0 {
            c.c_dark = -gap(idx_append::LUM_DARK_ERR) / app.ws_dark.den;
        }
        let mid_den = n_f - app.ws_dark.den - app.ws_bright.den;
        if mid_den > 0.0 {
            c.c_mid = -gap(idx_append::LUM_MID_ERR) / mid_den;
        }
        if app.ws_bright.den > 0.0 {
            c.c_bright = -gap(idx_append::LUM_BRIGHT_ERR) / app.ws_bright.den;
        }
        // Append deviation pools: (s_k, μ, f).
        let gd = |sk: f64, sum: f64, sum2: f64| -> (f64, f64, f64) {
            let m = sum * inv_n;
            let f = ((sum2 * inv_n) - m * m).max(0.0).sqrt();
            (sk, m, f)
        };
        c.gd_gms = gd(gap(idx_append::GMS_DEV2), grad.sum_gms, grad.sum_gms2);
        c.gd_art = gd(gap(idx_append::ART_DEV2), dense.sum_art, app.sum_art2);
        c.gd_det = gd(gap(idx_append::DET_DEV2), dense.sum_det, app.sum_det2);
        // Globals. GLOBAL_DMEAN: f = sat(|Δ|, C), Δ = (Σs−Σd)/N;
        // (δf)_i = −sat'(|Δ|)·sign(Δ)·(s_i−d_i)/N.
        let delta = (app.sum_s - app.sum_d) * inv_n;
        let sat_p = C_GDMEAN / ((delta.abs() + C_GDMEAN) * (delta.abs() + C_GDMEAN));
        c.g_dmean = gap(idx_append::GLOBAL_DMEAN) * (-sat_p * delta.signum() * inv_n);
        // GLOBAL_CGAIN/CLOSS: chain rule on gvar2 (gvar1 is reference-side).
        let gmean_s = app.sum_s * inv_n;
        let gmean_d = app.sum_d * inv_n;
        let gvar1 = (app.sum_s2 * inv_n - gmean_s * gmean_s).max(0.0);
        let gvar2 = (app.sum_d2 * inv_n - gmean_d * gmean_d).max(0.0);
        let denom = gvar1 + gvar2 + C_GCONTRAST;
        // δgvar2 per refined pixel = ((s²−d²) − 2·gmean_d·(s−d))/N.
        let df_dg2 = if gvar2 > gvar1 {
            gap(idx_append::GLOBAL_CGAIN) * ((2.0 * gvar1 + C_GCONTRAST) / (denom * denom))
        } else if gvar1 > gvar2 {
            gap(idx_append::GLOBAL_CLOSS) * (-(2.0 * gvar1 + C_GCONTRAST) / (denom * denom))
        } else {
            0.0
        };
        c.g_var = df_dg2 * inv_n;
        c.gmean_d = gmean_d;
        // GRAD_SRC_MEAN (slot 16): reference-only ⇒ integrand 0 (no term).
    }
    c
}

/// Pass B for one (scale, channel): f64 per-pixel combine over the cached
/// planes. Three sweeps: the main per-pixel loop, the gradient-family loop
/// (needs ±1-row neighbors), and the sparse blockiness lattice.
///
/// C2b bleed allocation (MEASURED variant: window-only): mass is routed by
/// SUPPORT class into two planes — `id_plane` (everything pixel-supported
/// plus, per the 8-cell A/B below, the residual-form signals) and
/// `win_plane` (the pure blur-window signals: the ssim `d` map, the
/// var-based contrast/texture signals, and the ssim-derived pools — whose
/// per-pixel values are functions of K-blurred planes only). The caller
/// spreads `win_plane` ONCE per scale with the sum-preserving box spread.
/// The 50/50 residual split (art/det/hf/mscn half-spread) was measured and
/// REGRESSED all 8 gate cells (−0.01..−0.08); the pure `I − K` adjoint for
/// residual signals allocates zero net mass and is structurally wrong for
/// removal semantics — both recorded in the C2b benchmark section.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_channel(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: &V2AppCoeffs,
    parallel: bool,
    id_plane: &mut [f64],
    win_plane: &mut [f64],
) {
    // Row-banded parallel main+gradient sweeps (C2b Part 2): each band
    // writes only its own rows of id/win (disjoint), reads shared planes
    // (gradient reads ±1 row of src/dst — immutable). Blockiness stays
    // serial (it writes the row ABOVE at lattice rows). Only the rayon arm
    // bands, so the const is `threads`-gated with it.
    #[cfg(feature = "threads")]
    const BAND: usize = 64;
    #[cfg(feature = "threads")]
    if parallel && height > BAND {
        use rayon::prelude::*;
        id_plane[..width * height]
            .par_chunks_mut(BAND * width)
            .zip(win_plane[..width * height].par_chunks_mut(BAND * width))
            .enumerate()
            .for_each(|(band, (id_rows, win_rows))| {
                let y0 = band * BAND;
                let y1 = (y0 + BAND).min(height);
                attr_pass_b_rows(
                    src, dst, width, height, planes, cross, ref_y, co, y0, y1, id_rows, win_rows,
                );
            });
        attr_pass_b_blockiness(src, dst, width, height, co, id_plane);
        return;
    }
    let _ = parallel;
    attr_pass_b_rows(
        src, dst, width, height, planes, cross, ref_y, co, 0, height, id_plane, win_plane,
    );
    attr_pass_b_blockiness(src, dst, width, height, co, id_plane);
}

/// Rows `[y0, y1)` of the main + gradient sweeps; `id_rows`/`win_rows` are
/// the band-local output slices (row `y` writes at `(y − y0) * width`).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_rows(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: &V2AppCoeffs,
    y0: usize,
    y1: usize,
    id_plane: &mut [f64],
    win_plane: &mut [f64],
) {
    let out_off = y0 * width;
    let cross_on = cross.is_some();
    for y in y0..y1 {
        let row = y * width;
        for x in 0..width {
            let i = row + x;
            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = planes.mu1[i] as f64;
            let m2 = planes.mu2[i] as f64;
            let sq = planes.ssq[i] as f64;
            let s12v = planes.s12[i] as f64;
            let act = planes.act[i] as f64;

            // Class accumulators: pixel-supported (a_id), window-supported
            // (a_win), and residual-form (a_res; routed to id per the
            // measured A/B — see the fn doc).
            let mut a_id = 0.0f64;
            let mut a_win = 0.0f64;
            let mut a_res = 0.0f64;

            // Dense family (same formulas as `dense_block_kernel`'s tail).
            let d = ssim_d_local(m1, m2, s12v, sq);
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
            let (art_i, det_i) = if diff_dst > diff_src {
                (edge_dissim, 0.0)
            } else if diff_dst < diff_src {
                (0.0, edge_dissim)
            } else {
                (0.0, 0.0)
            };
            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = saturate(raw_diff * raw_diff, C_MSE);
            a_win += co.c_ssim * d;
            a_res += co.c_art * art_i + co.c_det * det_i;
            a_id += co.c_mse * mse_i;

            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let (hf_gain_i, hf_loss_i) =
                bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF);
            a_res += co.c_hf_gain * hf_gain_i + co.c_hf_loss * hf_loss_i;
            a_res += co.c_hf_mag * bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            a_id += co.c_pjnd * pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP);
            a_id += co.c_pjnd_lo * pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP);
            a_id +=
                co.c_pjnd_hi * pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP);

            // Weighted pools (masked / iw): −s_k·w_i·v_i/Σw.
            let sat_act = saturate(act, C_ACTIVITY);
            let mask_w = 1.0 - sat_act;
            let iw_w = sat_act + IW_WEIGHT_FLOOR;
            // Weights are reference-side; each member routes by its VALUE's
            // support class (ssim → window, art/det → residual, mse → pixel).
            a_win += mask_w * (co.c_mask_ssim * d) + iw_w * (co.c_iw_ssim * d);
            a_res += mask_w * (co.c_mask_art * art_i + co.c_mask_det * det_i)
                + iw_w * (co.c_iw_art * art_i + co.c_iw_det * det_i);
            a_id += mask_w * (co.c_mask_mse * mse_i) + iw_w * (co.c_iw_mse * mse_i);

            // Soft-peak: s_k·w(v)·(f − v)/W  (sp = (s_k/W, f)).
            a_win += co.sp_ssim.0 * saturate(d, C_PEAK) * (co.sp_ssim.1 - d);
            a_res += co.sp_art.0 * saturate(art_i, C_PEAK) * (co.sp_art.1 - art_i);
            a_res += co.sp_det.0 * saturate(det_i, C_PEAK) * (co.sp_det.1 - det_i);

            // SSIM deviation pools (v = d).
            if co.s_dev2 != 0.0 && co.dev_f2 > 1e-12 {
                a_win += co.s_dev2 * (2.0 * co.dev_mu * d - d * d)
                    / (2.0 * co.dev_f2 * (1.0 / co.inv_n));
            }
            if co.s_dev4 != 0.0 && co.dev_f4 > 1e-12 {
                let mu = co.dev_mu;
                let dm4 = -d.powi(4) + 4.0 * d * co.dev_raw3 + 4.0 * mu * d.powi(3)
                    - 12.0 * mu * d * co.dev_raw2
                    - 6.0 * mu * mu * d * d
                    + 12.0 * mu.powi(3) * d;
                a_win += co.s_dev4 * co.inv_n * dm4 / (4.0 * co.dev_f4.powi(3));
            }

            // Append block.
            let b2 = planes.bs2[i] as f64;
            let var1 = (b2 - m1 * m1).max(0.0);
            let var2 = ((sq - b2) - m2 * m2).max(0.0);
            let n1 = (s - m1) / (var1 + C_MSCN_VAR).sqrt();
            let n2 = (dd - m2) / (var2 + C_MSCN_VAR).sqrt();
            let dn = n1 - n2;
            a_res += co.c_mscn * saturate(dn.abs(), C_MSCN_ABS);
            a_res += co.c_mscn2 * saturate(dn * dn, C_MSCN_SQ);
            let (cg, cl) = bounded_excess_pair(var2, var1, C_CONTRAST);
            a_win += co.c_cgain * cg + co.c_closs * cl;
            a_win += co.c_tex * (1.0 - bounded_sim(var1, var2, C_CONTRAST));

            if cross_on {
                let ry = ref_y[i] as f64;
                let t = saturate(ry, C_LUM_T);
                a_id += co.c_lumt
                    * pjnd_transducer(
                        raw_abs_err,
                        act + t * (K_LUM_ADAPT / K_PJND_MASK),
                        K_PJND_MASK,
                        C_PJND_CLAMP,
                    );
                if let Some((ax, ab)) = cross {
                    let act_c = ax[i] as f64 + ab[i] as f64;
                    a_id += co.c_xmask
                        * pjnd_transducer(
                            raw_abs_err,
                            act + act_c * (K_XCH / K_PJND_MASK),
                            K_PJND_MASK,
                            C_PJND_CLAMP,
                        );
                }
            }
            // Luminance bins (t from the reference Y plane, all channels).
            {
                let ry = ref_y[i] as f64;
                let t = saturate(ry, C_LUM_T);
                let one_mt = 1.0 - t;
                let wd = one_mt * one_mt;
                let wb = t * t;
                let wm = 1.0 - wd - wb;
                a_id += (co.c_dark * wd + co.c_mid * wm + co.c_bright * wb) * mse_i;
            }
            // Append dev pools for art/det (gms handled in the gradient loop).
            let gdp = |g: (f64, f64, f64), v: f64, inv_n: f64| -> f64 {
                let (sk, m, f) = g;
                if sk != 0.0 && f > 1e-12 {
                    sk * (2.0 * m * v - v * v) * inv_n / (2.0 * f)
                } else {
                    0.0
                }
            };
            a_res += gdp(co.gd_art, art_i, co.inv_n);
            a_res += gdp(co.gd_det, det_i, co.inv_n);
            // Globals (pixel class — raw whole-plane sums).
            a_id += co.g_dmean * (s - dd);
            a_id += co.g_var * ((s * s - dd * dd) - 2.0 * co.gmean_d * (s - dd));

            id_plane[i - out_off] += a_id + a_res;
            win_plane[i - out_off] += a_win;
        }
    }

    // Gradient family: gms / ringing / banding / gms-dev / edge-width /
    // append2 BANDVIS.
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let need_grad = co.c_gms != 0.0
        || co.c_ringing != 0.0
        || co.c_banding != 0.0
        || co.gd_gms.0 != 0.0
        || co.c_edgew != 0.0
        || need_bandvis;
    if need_grad {
        for y in y0..y1 {
            let row = y * width;
            let y_up = reflect_101(y as isize - 1, height);
            let y_dn = reflect_101(y as isize + 1, height);
            for x in 0..width {
                let i = row + x;
                let xl = x.saturating_sub(1);
                let xr = (x + 1).min(width - 1);
                let sxl = src[row + xl] as f64;
                let sxr = src[row + xr] as f64;
                let syu = src[y_up * width + x] as f64;
                let syd = src[y_dn * width + x] as f64;
                let dxl = dst[row + xl] as f64;
                let dxr = dst[row + xr] as f64;
                let dyu = dst[y_up * width + x] as f64;
                let dyd = dst[y_dn * width + x] as f64;
                let gx_s = sxr - sxl;
                let gy_s = syd - syu;
                let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
                let gx_d = dxr - dxl;
                let gy_d = dyd - dyu;
                let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
                let g = 1.0 - bounded_sim(gmag_s, gmag_d, C_GMS);
                let mut acc = co.c_gms * g;
                if co.gd_gms.0 != 0.0 && co.gd_gms.2 > 1e-12 {
                    acc += co.gd_gms.0 * (2.0 * co.gd_gms.1 * g - g * g) * co.inv_n
                        / (2.0 * co.gd_gms.2);
                }
                if co.c_ringing != 0.0 {
                    let s = src[i] as f64;
                    let dd = dst[i] as f64;
                    let act = planes.act[i] as f64;
                    let err_b = saturate((s - dd).abs(), C_RING_ERR);
                    let act_b = saturate(act, C_ACTIVITY);
                    let edge_r = saturate(gmag_s, C_RING_EDGE);
                    acc += co.c_ringing * (err_b * act_b * (1.0 - edge_r));
                }
                if co.c_banding != 0.0 {
                    let edge_excess = bounded_excess(gmag_d, gmag_s, C_BAND_DST);
                    let src_smooth = 1.0 - saturate(gmag_s, C_BAND_SRC);
                    acc += co.c_banding * (edge_excess * src_smooth);
                }
                if need_bandvis {
                    // append2 BANDVIS (`idx_append2::BANDVIS_GAIN/LOSS`) —
                    // the OFF-toggle production form (`append2_dst_activity`
                    // is default-OFF and adjudicated OFF for every
                    // production extraction, and this path fixes toggles to
                    // `V2NewFeatureToggles::default()`). Second differences
                    // reuse the four neighbour loads the gradient family
                    // already performs, so the terms are near-free and the
                    // neighbour convention (x: clamp, y: reflect_101 via the
                    // production halo rows) matches the kernel exactly.
                    let s_c = src[i] as f64;
                    let d_c = dst[i] as f64;
                    let d2x_src = sxl + sxr - 2.0 * s_c;
                    let d2y_src = syu + syd - 2.0 * s_c;
                    let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
                    let d2x_dst = dxl + dxr - 2.0 * d_c;
                    let d2y_dst = dyu + dyd - 2.0 * d_c;
                    let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
                    let flat = 1.0 - saturate(planes.act[i] as f64, C_ACTIVITY);
                    let band = |g: f64| -> f64 {
                        saturate(g, BV_DELTA_LO_SDR as f64)
                            * (1.0 - saturate(g, BV_DELTA_HI_SDR as f64))
                    };
                    let b_src = band(curv_src) * flat;
                    let b_dst = band(curv_dst) * flat;
                    let (gain_i, loss_i) = bounded_excess_pair(b_dst, b_src, C_BV);
                    acc += co.c_bv_gain * gain_i + co.c_bv_loss * loss_i;
                }
                // Edge-width: coefficient × (|∇d| − |∇s|) — the per-pixel
                // change of this scale's mean distorted gradient under
                // refinement is −(|∇d|−|∇s|)/N (folded into c_edgew).
                acc += co.c_edgew * (gmag_d - gmag_s);
                id_plane[i - out_off] += acc;
            }
        }
    }
}

/// Blockiness lattice terms: split 50/50 across the step pair (a step
/// dies only when BOTH pixels are refined; halving allocates the mass
/// consistently when a partition boundary splits the pair). Serial — the
/// horizontal family writes the row above.
#[cfg(feature = "custom-profiles")]
fn attr_pass_b_blockiness(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    co: &V2AppCoeffs,
    id_plane: &mut [f64],
) {
    if co.c_blockiness != 0.0 {
        for y in 0..height {
            let row = y * width;
            let mut x = BLOCK_LATTICE;
            while x < width {
                let i = row + x;
                let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
                let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
                let v = co.c_blockiness * bounded_excess(step_dst, step_src, C_BLOCK);
                id_plane[i] += 0.5 * v;
                id_plane[i - 1] += 0.5 * v;
                x += BLOCK_LATTICE;
            }
            if y % BLOCK_LATTICE == 0 && y > 0 {
                for x in 0..width {
                    let i = row + x;
                    let i_up = i - width;
                    let step_dst = (dst[i] as f64 - dst[i_up] as f64).abs();
                    let step_src = (src[i] as f64 - src[i_up] as f64).abs();
                    let v = co.c_blockiness * bounded_excess(step_dst, step_src, C_BLOCK);
                    id_plane[i] += 0.5 * v;
                    id_plane[i_up] += 0.5 * v;
                }
            }
        }
    }
}

/// f32 coefficient pack for the FUSED pass-B kernel (appendix P lever 1):
/// [`V2AppCoeffs`] with the dev-pool / gdp guards PRE-FOLDED into
/// polynomial coefficients so the pixel loop is branch-free, and the
/// cross/lum-transducer terms zeroed for non-Y channels (the f64 kernel's
/// `cross_on` branch, folded into coefficients). Serves the fused
/// folded-944 entry ONLY — the standalone f64 pass-B and its strict tests
/// are untouched (the C3a precedent).
#[cfg(feature = "custom-profiles")]
#[derive(Clone, Copy)]
struct V2AppCoeffsF32 {
    c_ssim: f32,
    c_art: f32,
    c_det: f32,
    c_mse: f32,
    c_hf_gain: f32,
    c_hf_loss: f32,
    c_hf_mag: f32,
    c_pjnd: f32,
    c_pjnd_lo: f32,
    c_pjnd_hi: f32,
    c_gms: f32,
    c_ringing: f32,
    c_banding: f32,
    c_blockiness: f32,
    c_mask_ssim: f32,
    c_mask_art: f32,
    c_mask_det: f32,
    c_mask_mse: f32,
    c_iw_ssim: f32,
    c_iw_art: f32,
    c_iw_det: f32,
    c_iw_mse: f32,
    sp_ssim: (f32, f32),
    sp_art: (f32, f32),
    sp_det: (f32, f32),
    /// ssim-`d` dev-pool polynomial (dev2 + dev4 folded, guards applied at
    /// fold time): `pd1·d + pd2·d² + pd3·d³ + pd4·d⁴` added to the window
    /// class.
    pd1: f32,
    pd2: f32,
    pd3: f32,
    pd4: f32,
    c_xmask: f32,
    c_lumt: f32,
    c_mscn: f32,
    c_mscn2: f32,
    c_cgain: f32,
    c_closs: f32,
    c_tex: f32,
    c_dark: f32,
    c_mid: f32,
    c_bright: f32,
    /// gdp pools as lin/quad pairs: `lin·v + quad·v²` (guards folded).
    ga_lin: f32,
    ga_quad: f32,
    gd_lin: f32,
    gd_quad: f32,
    gg_lin: f32,
    gg_quad: f32,
    g_dmean: f32,
    g_var: f32,
    gmean_d: f32,
    c_edgew: f32,
    c_bv_gain: f32,
    c_bv_loss: f32,
}

/// Fold a derived [`V2AppCoeffs`] into the f32 pack. `cross_on` = the f64
/// kernel's `cross.is_some()` (Y channel): when false, the lum-transducer
/// and cross-mask terms are zeroed exactly as the f64 branch skips them.
#[cfg(feature = "custom-profiles")]
fn v2app_coeffs_fold_f32(co: &V2AppCoeffs, cross_on: bool) -> V2AppCoeffsF32 {
    // dev2: s·(2μd − d²) / (2·f2·N)  →  lin/quad in d.
    let (mut pd1, mut pd2, mut pd3, mut pd4) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    if co.s_dev2 != 0.0 && co.dev_f2 > 1e-12 {
        let k = co.s_dev2 * co.inv_n / (2.0 * co.dev_f2);
        pd1 += k * 2.0 * co.dev_mu;
        pd2 += -k;
    }
    // dev4: k4·dm4 with dm4 = −d⁴ + 4d·raw3 + 4μd³ − 12μd·raw2 − 6μ²d² + 12μ³d.
    if co.s_dev4 != 0.0 && co.dev_f4 > 1e-12 {
        let k4 = co.s_dev4 * co.inv_n / (4.0 * co.dev_f4.powi(3));
        let mu = co.dev_mu;
        pd1 += k4 * (4.0 * co.dev_raw3 - 12.0 * mu * co.dev_raw2 + 12.0 * mu.powi(3));
        pd2 += k4 * (-6.0) * mu * mu;
        pd3 += k4 * 4.0 * mu;
        pd4 += -k4;
    }
    let gdp_lq = |g: (f64, f64, f64)| -> (f32, f32) {
        let (sk, m, f) = g;
        if sk != 0.0 && f > 1e-12 {
            let k = sk * co.inv_n / (2.0 * f);
            ((k * 2.0 * m) as f32, (-k) as f32)
        } else {
            (0.0, 0.0)
        }
    };
    let (ga_lin, ga_quad) = gdp_lq(co.gd_art);
    let (gd_lin, gd_quad) = gdp_lq(co.gd_det);
    let (gg_lin, gg_quad) = gdp_lq(co.gd_gms);
    V2AppCoeffsF32 {
        c_ssim: co.c_ssim as f32,
        c_art: co.c_art as f32,
        c_det: co.c_det as f32,
        c_mse: co.c_mse as f32,
        c_hf_gain: co.c_hf_gain as f32,
        c_hf_loss: co.c_hf_loss as f32,
        c_hf_mag: co.c_hf_mag as f32,
        c_pjnd: co.c_pjnd as f32,
        c_pjnd_lo: co.c_pjnd_lo as f32,
        c_pjnd_hi: co.c_pjnd_hi as f32,
        c_gms: co.c_gms as f32,
        c_ringing: co.c_ringing as f32,
        c_banding: co.c_banding as f32,
        c_blockiness: co.c_blockiness as f32,
        c_mask_ssim: co.c_mask_ssim as f32,
        c_mask_art: co.c_mask_art as f32,
        c_mask_det: co.c_mask_det as f32,
        c_mask_mse: co.c_mask_mse as f32,
        c_iw_ssim: co.c_iw_ssim as f32,
        c_iw_art: co.c_iw_art as f32,
        c_iw_det: co.c_iw_det as f32,
        c_iw_mse: co.c_iw_mse as f32,
        sp_ssim: (co.sp_ssim.0 as f32, co.sp_ssim.1 as f32),
        sp_art: (co.sp_art.0 as f32, co.sp_art.1 as f32),
        sp_det: (co.sp_det.0 as f32, co.sp_det.1 as f32),
        pd1: pd1 as f32,
        pd2: pd2 as f32,
        pd3: pd3 as f32,
        pd4: pd4 as f32,
        c_xmask: if cross_on { co.c_xmask as f32 } else { 0.0 },
        c_lumt: if cross_on { co.c_lumt as f32 } else { 0.0 },
        c_mscn: co.c_mscn as f32,
        c_mscn2: co.c_mscn2 as f32,
        c_cgain: co.c_cgain as f32,
        c_closs: co.c_closs as f32,
        c_tex: co.c_tex as f32,
        c_dark: co.c_dark as f32,
        c_mid: co.c_mid as f32,
        c_bright: co.c_bright as f32,
        ga_lin,
        ga_quad,
        gd_lin,
        gd_quad,
        gg_lin,
        gg_quad,
        g_dmean: co.g_dmean as f32,
        g_var: co.g_var as f32,
        gmean_d: co.gmean_d as f32,
        c_edgew: co.c_edgew as f32,
        c_bv_gain: co.c_bv_gain as f32,
        c_bv_loss: co.c_bv_loss as f32,
    }
}

/// Scalar per-pixel MAIN-sweep integrand of the f32 fused pass-B — shared
/// by the SIMD kernel's row tail and the sub-8-wide fallback, so there is
/// exactly ONE formula source. Returns `(id_add, win_add)` (`id` carries
/// the pixel + residual classes, `win` the window class).
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_px(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    sq: f32,
    s12v: f32,
    act: f32,
    b2v: f32,
    axv: f32,
    abv: f32,
    ry: f32,
    co: &V2AppCoeffsF32,
) -> (f32, f32) {
    const C1: f32 = C1_V2 as f32;
    const C2: f32 = C2_V2 as f32;
    const CE: f32 = C_EDGE as f32;
    const CM: f32 = C_MSE as f32;
    const CH: f32 = C_HF as f32;
    const CPC: f32 = C_PJND_CLAMP as f32;
    const KP: f32 = K_PJND_MASK as f32;
    const KPL: f32 = K_PJND_MASK_LOW as f32;
    const KPH: f32 = K_PJND_MASK_HIGH as f32;
    const CA: f32 = C_ACTIVITY as f32;
    const IWF: f32 = IW_WEIGHT_FLOOR as f32;
    const CPK: f32 = C_PEAK as f32;
    const CMV: f32 = C_MSCN_VAR as f32;
    const CMA: f32 = C_MSCN_ABS as f32;
    const CMS: f32 = C_MSCN_SQ as f32;
    const CC: f32 = C_CONTRAST as f32;
    const CLT: f32 = C_LUM_T as f32;
    const KLA_OVER_KP: f32 = (K_LUM_ADAPT / K_PJND_MASK) as f32;
    const KX_OVER_KP: f32 = (K_XCH / K_PJND_MASK) as f32;
    let sat = |x: f32, c: f32| -> f32 {
        let x = x.max(0.0);
        x / (x + c)
    };
    let pjnd = |x: f32, act: f32, k: f32| -> f32 { x / (x + CPC * (1.0 + k * act)) };
    let mut a_id = 0.0f32;
    let mut a_win = 0.0f32;
    let mut a_res = 0.0f32;
    // Dense family.
    let na = 2.0 * m1 * m2 + C1;
    let nb = m1 * m1 + m2 * m2 + C1;
    let cov = s12v - m1 * m2;
    let nc = 2.0 * cov + C2;
    let nd = sq - m1 * m1 - m2 * m2 + C2;
    let d = (1.0 - (na * nc) / (nb * nd)).max(0.0);
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim =
        1.0 - (2.0 * diff_src * diff_dst + CE) / (diff_src * diff_src + diff_dst * diff_dst + CE);
    let art_i = if diff_dst > diff_src {
        edge_dissim
    } else {
        0.0
    };
    let det_i = if diff_dst < diff_src {
        edge_dissim
    } else {
        0.0
    };
    let raw_diff = s - dd;
    let raw_abs_err = raw_diff.abs();
    let mse_i = sat(raw_diff * raw_diff, CM);
    a_win += co.c_ssim * d;
    a_res += co.c_art * art_i + co.c_det * det_i;
    a_id += co.c_mse * mse_i;
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    let (hf_gain_i, hf_loss_i) = {
        let a2 = hf_dst * hf_dst;
        let b2 = hf_src * hf_src;
        let r = 1.0 / (a2 + b2 + CH);
        ((a2 - b2).max(0.0) * r, (b2 - a2).max(0.0) * r)
    };
    a_res += co.c_hf_gain * hf_gain_i + co.c_hf_loss * hf_loss_i;
    a_res += co.c_hf_mag * {
        let a = hf_src.abs();
        let b = hf_dst.abs();
        (a - b).max(0.0) / (a + b + CH)
    };
    a_id += co.c_pjnd * pjnd(raw_abs_err, act, KP);
    a_id += co.c_pjnd_lo * pjnd(raw_abs_err, act, KPL);
    a_id += co.c_pjnd_hi * pjnd(raw_abs_err, act, KPH);
    // Weighted pools.
    let sat_act = sat(act, CA);
    let mask_w = 1.0 - sat_act;
    let iw_w = sat_act + IWF;
    a_win += mask_w * (co.c_mask_ssim * d) + iw_w * (co.c_iw_ssim * d);
    a_res += mask_w * (co.c_mask_art * art_i + co.c_mask_det * det_i)
        + iw_w * (co.c_iw_art * art_i + co.c_iw_det * det_i);
    a_id += mask_w * (co.c_mask_mse * mse_i) + iw_w * (co.c_iw_mse * mse_i);
    // Soft-peak.
    a_win += co.sp_ssim.0 * sat(d, CPK) * (co.sp_ssim.1 - d);
    a_res += co.sp_art.0 * sat(art_i, CPK) * (co.sp_art.1 - art_i);
    a_res += co.sp_det.0 * sat(det_i, CPK) * (co.sp_det.1 - det_i);
    // Dev polynomial: pd1·d + pd2·d² + pd3·d³ + pd4·d⁴.
    let dsq = d * d;
    a_win += (co.pd1 + co.pd3 * dsq) * d + (co.pd2 + co.pd4 * dsq) * dsq;
    // Append block.
    let var1 = (b2v - m1 * m1).max(0.0);
    let var2 = ((sq - b2v) - m2 * m2).max(0.0);
    let n1 = (s - m1) / (var1 + CMV).sqrt();
    let n2 = (dd - m2) / (var2 + CMV).sqrt();
    let dn = n1 - n2;
    a_res += co.c_mscn * sat(dn.abs(), CMA);
    a_res += co.c_mscn2 * sat(dn * dn, CMS);
    let (cg, cl) = {
        let r = 1.0 / (var2 + var1 + CC);
        ((var2 - var1).max(0.0) * r, (var1 - var2).max(0.0) * r)
    };
    a_win += co.c_cgain * cg + co.c_closs * cl;
    a_win += co.c_tex * (1.0 - (2.0 * var1 * var2 + CC) / (var1 * var1 + var2 * var2 + CC));
    // Luminance transducer + cross-mask (coefficients zero off-Y).
    let t = sat(ry, CLT);
    if co.c_lumt != 0.0 {
        a_id += co.c_lumt * pjnd(raw_abs_err, act + t * KLA_OVER_KP, KP);
    }
    if co.c_xmask != 0.0 {
        let act_c = axv + abv;
        a_id += co.c_xmask * pjnd(raw_abs_err, act + act_c * KX_OVER_KP, KP);
    }
    // Luminance bins.
    let one_mt = 1.0 - t;
    let wd = one_mt * one_mt;
    let wb = t * t;
    let wm = 1.0 - wd - wb;
    a_id += (co.c_dark * wd + co.c_mid * wm + co.c_bright * wb) * mse_i;
    // gdp pools (art/det).
    a_res += (co.ga_lin + co.ga_quad * art_i) * art_i;
    a_res += (co.gd_lin + co.gd_quad * det_i) * det_i;
    // Globals (factored g_var form).
    a_id += co.g_dmean * raw_diff;
    a_id += co.g_var * raw_diff * ((s + dd) - 2.0 * co.gmean_d);
    (a_id + a_res, a_win)
}

/// SIMD main sweep of the f32 fused pass-B — generic over any
/// [`F32x8Backend`] token, 8 pixels per step with the scalar
/// [`attr_pass_b_main_px`] tail. Same formulas lane-for-lane (selects via
/// [`GenericF32x8::blend`]); the auto-vectorizer refused the 55-constant
/// monolith (measured scalar `divss` codegen — appendix P lever 1), so
/// this is the explicit magetypes port, the same §A.14 pattern as
/// `dense_block_kernel_generic`.
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let sp = |v: f32| V8::<T>::splat(token, v);
    let c1 = sp(C1_V2 as f32);
    let c2 = sp(C2_V2 as f32);
    let c_edge = sp(C_EDGE as f32);
    let c_mse = sp(C_MSE as f32);
    let c_hf = sp(C_HF as f32);
    let c_pjnd_clamp = sp(C_PJND_CLAMP as f32);
    let k_mid = sp(K_PJND_MASK as f32);
    let k_lo = sp(K_PJND_MASK_LOW as f32);
    let k_hi = sp(K_PJND_MASK_HIGH as f32);
    let c_activity = sp(C_ACTIVITY as f32);
    let iw_floor = sp(IW_WEIGHT_FLOOR as f32);
    let c_peak = sp(C_PEAK as f32);
    let c_mscn_var = sp(C_MSCN_VAR as f32);
    let c_mscn_abs = sp(C_MSCN_ABS as f32);
    let c_mscn_sq = sp(C_MSCN_SQ as f32);
    let c_contrast = sp(C_CONTRAST as f32);
    let c_lum_t = sp(C_LUM_T as f32);
    let kla = sp((K_LUM_ADAPT / K_PJND_MASK) as f32);
    let kx = sp((K_XCH / K_PJND_MASK) as f32);
    let sat_v = |x: V8<T>, c: V8<T>| -> V8<T> {
        let x = x.max(zero);
        x / (x + c)
    };
    let has_lumt = co.c_lumt != 0.0;
    let has_xmask = co.c_xmask != 0.0;
    let out_off = y0 * width;
    let width8 = width - (width % 8);
    for y in y0..y1 {
        let row = y * width;
        let orow = row - out_off;
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
            let sq = ld!(ssq);
            let act = ld!(act_p);
            let mut a_id = zero;
            let mut a_win = zero;
            let mut a_res = zero;
            // Dense family.
            let d = ssim_d_local_v(token, m1, m2, ld!(s12), sq, c1, c2);
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = one - bounded_sim_v(token, diff_src, diff_dst, c_edge);
            let art_i = V8::<T>::blend(diff_dst.simd_gt(diff_src), edge_dissim, zero);
            let det_i = V8::<T>::blend(diff_dst.simd_lt(diff_src), edge_dissim, zero);
            let raw_diff = s - dd;
            let raw_abs_err = raw_diff.abs();
            let mse_i = sat_v(raw_diff * raw_diff, c_mse);
            a_win += sp(co.c_ssim) * d;
            a_res += sp(co.c_art) * art_i + sp(co.c_det) * det_i;
            a_id += sp(co.c_mse) * mse_i;
            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let (hf_gain_i, hf_loss_i) =
                bounded_excess_pair_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
            a_res += sp(co.c_hf_gain) * hf_gain_i + sp(co.c_hf_loss) * hf_loss_i;
            a_res += sp(co.c_hf_mag) * bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
            a_id += sp(co.c_pjnd) * pjnd_transducer_v(token, raw_abs_err, act, k_mid, c_pjnd_clamp);
            a_id +=
                sp(co.c_pjnd_lo) * pjnd_transducer_v(token, raw_abs_err, act, k_lo, c_pjnd_clamp);
            a_id +=
                sp(co.c_pjnd_hi) * pjnd_transducer_v(token, raw_abs_err, act, k_hi, c_pjnd_clamp);
            // Weighted pools.
            let sat_act = sat_v(act, c_activity);
            let mask_w = one - sat_act;
            let iw_w = sat_act + iw_floor;
            a_win += mask_w * (sp(co.c_mask_ssim) * d) + iw_w * (sp(co.c_iw_ssim) * d);
            a_res += mask_w * (sp(co.c_mask_art) * art_i + sp(co.c_mask_det) * det_i)
                + iw_w * (sp(co.c_iw_art) * art_i + sp(co.c_iw_det) * det_i);
            a_id += mask_w * (sp(co.c_mask_mse) * mse_i) + iw_w * (sp(co.c_iw_mse) * mse_i);
            // Soft-peak.
            a_win += sp(co.sp_ssim.0) * sat_v(d, c_peak) * (sp(co.sp_ssim.1) - d);
            a_res += sp(co.sp_art.0) * sat_v(art_i, c_peak) * (sp(co.sp_art.1) - art_i);
            a_res += sp(co.sp_det.0) * sat_v(det_i, c_peak) * (sp(co.sp_det.1) - det_i);
            // Dev polynomial.
            let dsq = d * d;
            a_win += (sp(co.pd1) + sp(co.pd3) * dsq) * d + (sp(co.pd2) + sp(co.pd4) * dsq) * dsq;
            // Append block.
            let b2v = ld!(bs2);
            let var1 = (b2v - m1 * m1).max(zero);
            let var2 = ((sq - b2v) - m2 * m2).max(zero);
            let n1 = (s - m1) / (var1 + c_mscn_var).sqrt();
            let n2 = (dd - m2) / (var2 + c_mscn_var).sqrt();
            let dn = n1 - n2;
            a_res += sp(co.c_mscn) * sat_v(dn.abs(), c_mscn_abs);
            a_res += sp(co.c_mscn2) * sat_v(dn * dn, c_mscn_sq);
            let (cg, cl) = bounded_excess_pair_v(token, var2, var1, c_contrast);
            a_win += sp(co.c_cgain) * cg + sp(co.c_closs) * cl;
            a_win += sp(co.c_tex) * (one - bounded_sim_v(token, var1, var2, c_contrast));
            // Luminance transducer + cross-mask (Y only; uniform branches).
            let ry = ld!(ref_y);
            let t = sat_v(ry, c_lum_t);
            if has_lumt {
                a_id += sp(co.c_lumt)
                    * pjnd_transducer_v(token, raw_abs_err, act + t * kla, k_mid, c_pjnd_clamp);
            }
            if has_xmask {
                let act_c = ld!(ax) + ld!(ab);
                a_id += sp(co.c_xmask)
                    * pjnd_transducer_v(token, raw_abs_err, act + act_c * kx, k_mid, c_pjnd_clamp);
            }
            // Luminance bins.
            let one_mt = one - t;
            let wd = one_mt * one_mt;
            let wb = t * t;
            let wm = one - wd - wb;
            a_id += (sp(co.c_dark) * wd + sp(co.c_mid) * wm + sp(co.c_bright) * wb) * mse_i;
            // gdp pools.
            a_res += (sp(co.ga_lin) + sp(co.ga_quad) * art_i) * art_i;
            a_res += (sp(co.gd_lin) + sp(co.gd_quad) * det_i) * det_i;
            // Globals (factored g_var form).
            a_id += sp(co.g_dmean) * raw_diff;
            a_id += sp(co.g_var) * raw_diff * ((s + dd) - sp(2.0 * co.gmean_d));
            let o = orow + x;
            let id_new =
                V8::<T>::from_array(token, id_plane[o..o + 8].try_into().unwrap()) + a_id + a_res;
            id_plane[o..o + 8].copy_from_slice(&id_new.to_array());
            let win_new =
                V8::<T>::from_array(token, win_plane[o..o + 8].try_into().unwrap()) + a_win;
            win_plane[o..o + 8].copy_from_slice(&win_new.to_array());
            x += 8;
        }
        // Scalar tail — the shared per-pixel formula.
        for x in width8..width {
            let i = row + x;
            let (id_add, win_add) = attr_pass_b_main_px(
                src[i], dst[i], mu1[i], mu2[i], ssq[i], s12[i], act_p[i], bs2[i], ax[i], ab[i],
                ref_y[i], co,
            );
            id_plane[orow + x] += id_add;
            win_plane[orow + x] += win_add;
        }
    }
}

/// Tiered magetypes entry for the pass-B main sweep.
#[cfg(feature = "custom-profiles")]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_main_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    attr_pass_b_main_kernel_generic(
        token, src, dst, mu1, mu2, ssq, s12, act_p, bs2, ax, ab, ref_y, co, width, y0, y1,
        id_plane, win_plane,
    );
}

/// f32 twin of [`attr_pass_b_rows`] (appendix P lever 1): the SIMD main
/// sweep ([`attr_pass_b_main_entry`], magetypes-tiered) plus the gradient
/// family sweep ([`attr_pass_b_grad_rows_f32`], `#[autoversion]` — its
/// smaller body auto-vectorizes). Same integrand formulas as the f64
/// kernel; the dev/gdp pools are pre-folded polynomials and the `g_var`
/// global uses the factored `(s−d)·((s+d) − 2·gmean_d)` form
/// (algebraically identical, f32-stable). Cross activity slices are
/// dummies (== `ref_y`) for non-Y channels — their coefficients are
/// zeroed at fold time. Precision class: the fused entry's C3a tolerance
/// (G-P2), NOT the standalone's strict identities.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_rows_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    act_p: &[f32],
    bs2: &[f32],
    ax: &[f32],
    ab: &[f32],
    ref_y: &[f32],
    co: V2AppCoeffsF32,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    incant!(
        attr_pass_b_main_entry(
            src, dst, mu1, mu2, ssq, s12, act_p, bs2, ax, ab, ref_y, &co, width, y0, y1, id_plane,
            win_plane
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
    attr_pass_b_grad_rows_f32(src, dst, width, height, act_p, co, y0, y1, id_plane);
}

/// Scalar per-pixel gradient-family integrand of the f32 fused pass-B —
/// shared by the SIMD kernel's edge/tail pixels (ONE formula source).
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_px(
    s_c: f32,
    d_c: f32,
    act_i: f32,
    sxl: f32,
    sxr: f32,
    syu: f32,
    syd: f32,
    dxl: f32,
    dxr: f32,
    dyu: f32,
    dyd: f32,
    co: &V2AppCoeffsF32,
    need_bandvis: bool,
) -> f32 {
    const CA: f32 = C_ACTIVITY as f32;
    const CG: f32 = C_GMS as f32;
    const CRE: f32 = C_RING_ERR as f32;
    const CRG: f32 = C_RING_EDGE as f32;
    const CBD: f32 = C_BAND_DST as f32;
    const CBS: f32 = C_BAND_SRC as f32;
    const CBV: f32 = C_BV as f32;
    let sat = |x: f32, c: f32| -> f32 {
        let x = x.max(0.0);
        x / (x + c)
    };
    let gx_s = sxr - sxl;
    let gy_s = syd - syu;
    let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
    let gx_d = dxr - dxl;
    let gy_d = dyd - dyu;
    let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
    let g = 1.0 - (2.0 * gmag_s * gmag_d + CG) / (gmag_s * gmag_s + gmag_d * gmag_d + CG);
    let mut acc = co.c_gms * g;
    acc += (co.gg_lin + co.gg_quad * g) * g;
    if co.c_ringing != 0.0 {
        let err_b = sat((s_c - d_c).abs(), CRE);
        let act_b = sat(act_i, CA);
        let edge_r = sat(gmag_s, CRG);
        acc += co.c_ringing * (err_b * act_b * (1.0 - edge_r));
    }
    if co.c_banding != 0.0 {
        let edge_excess = (gmag_d - gmag_s).max(0.0) / (gmag_d + gmag_s + CBD);
        let src_smooth = 1.0 - sat(gmag_s, CBS);
        acc += co.c_banding * (edge_excess * src_smooth);
    }
    if need_bandvis {
        let d2x_src = sxl + sxr - 2.0 * s_c;
        let d2y_src = syu + syd - 2.0 * s_c;
        let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
        let d2x_dst = dxl + dxr - 2.0 * d_c;
        let d2y_dst = dyu + dyd - 2.0 * d_c;
        let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
        let flat = 1.0 - sat(act_i, CA);
        let band = |g: f32| -> f32 { sat(g, BV_DELTA_LO_SDR) * (1.0 - sat(g, BV_DELTA_HI_SDR)) };
        let b_src = band(curv_src) * flat;
        let b_dst = band(curv_dst) * flat;
        let (gain_i, loss_i) = {
            let r = 1.0 / (b_dst + b_src + CBV);
            ((b_dst - b_src).max(0.0) * r, (b_src - b_dst).max(0.0) * r)
        };
        acc += co.c_bv_gain * gain_i + co.c_bv_loss * loss_i;
    }
    acc + co.c_edgew * (gmag_d - gmag_s)
}

/// SIMD gradient-family sweep — generic over any [`F32x8Backend`] token;
/// interior pixels 8-wide with unaligned ±1 neighbor loads, x-edges and
/// the sub-8 tail through the scalar [`attr_pass_b_grad_px`].
#[cfg(feature = "custom-profiles")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    act_p: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    height: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let has_ringing = co.c_ringing != 0.0;
    let has_banding = co.c_banding != 0.0;
    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let two = V8::<T>::splat(token, 2.0);
    let sp = |v: f32| V8::<T>::splat(token, v);
    let c_gms = sp(C_GMS as f32);
    let c_activity = sp(C_ACTIVITY as f32);
    let c_ring_err = sp(C_RING_ERR as f32);
    let c_ring_edge = sp(C_RING_EDGE as f32);
    let c_band_dst = sp(C_BAND_DST as f32);
    let c_band_src = sp(C_BAND_SRC as f32);
    let c_bv = sp(C_BV as f32);
    let bv_lo = sp(BV_DELTA_LO_SDR);
    let bv_hi = sp(BV_DELTA_HI_SDR);
    let sat_v = |x: V8<T>, c: V8<T>| -> V8<T> {
        let x = x.max(zero);
        x / (x + c)
    };
    let out_off = y0 * width;
    for y in y0..y1 {
        let row = y * width;
        let ru = reflect_101(y as isize - 1, height) * width;
        let rd = reflect_101(y as isize + 1, height) * width;
        if width == 1 {
            let i = row;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                src[i],
                src[i],
                src[ru],
                src[rd],
                dst[i],
                dst[i],
                dst[ru],
                dst[rd],
                co,
                need_bandvis,
            );
            continue;
        }
        // x = 0 (xl clamps to 0).
        {
            let i = row;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                src[i],
                src[i + 1],
                src[ru],
                src[rd],
                dst[i],
                dst[i + 1],
                dst[ru],
                dst[rd],
                co,
                need_bandvis,
            );
        }
        // Interior [1, width−1): 8-wide chunks + scalar remainder.
        let interior_end = width - 1;
        let mut x = 1usize;
        while x + 8 <= interior_end {
            let i = row + x;
            macro_rules! ldo {
                ($plane:expr, $off:expr) => {
                    V8::<T>::from_array(token, $plane[$off..$off + 8].try_into().unwrap())
                };
            }
            let s_c = ldo!(src, i);
            let d_c = ldo!(dst, i);
            let sxl = ldo!(src, i - 1);
            let sxr = ldo!(src, i + 1);
            let syu = ldo!(src, ru + x);
            let syd = ldo!(src, rd + x);
            let dxl = ldo!(dst, i - 1);
            let dxr = ldo!(dst, i + 1);
            let dyu = ldo!(dst, ru + x);
            let dyd = ldo!(dst, rd + x);
            let gx_s = sxr - sxl;
            let gy_s = syd - syu;
            let gmag_s = (gx_s * gx_s + gy_s * gy_s).sqrt();
            let gx_d = dxr - dxl;
            let gy_d = dyd - dyu;
            let gmag_d = (gx_d * gx_d + gy_d * gy_d).sqrt();
            let g = one - bounded_sim_v(token, gmag_s, gmag_d, c_gms);
            let mut acc = sp(co.c_gms) * g;
            acc += (sp(co.gg_lin) + sp(co.gg_quad) * g) * g;
            if has_ringing {
                let err_b = sat_v((s_c - d_c).abs(), c_ring_err);
                let act_b = sat_v(ldo!(act_p, i), c_activity);
                let edge_r = sat_v(gmag_s, c_ring_edge);
                acc += sp(co.c_ringing) * (err_b * act_b * (one - edge_r));
            }
            if has_banding {
                let edge_excess = bounded_excess_v(token, gmag_d, gmag_s, c_band_dst);
                let src_smooth = one - sat_v(gmag_s, c_band_src);
                acc += sp(co.c_banding) * (edge_excess * src_smooth);
            }
            if need_bandvis {
                let d2x_src = sxl + sxr - two * s_c;
                let d2y_src = syu + syd - two * s_c;
                let curv_src = (d2x_src * d2x_src + d2y_src * d2y_src).sqrt();
                let d2x_dst = dxl + dxr - two * d_c;
                let d2y_dst = dyu + dyd - two * d_c;
                let curv_dst = (d2x_dst * d2x_dst + d2y_dst * d2y_dst).sqrt();
                let flat = one - sat_v(ldo!(act_p, i), c_activity);
                let b_src = sat_v(curv_src, bv_lo) * (one - sat_v(curv_src, bv_hi)) * flat;
                let b_dst = sat_v(curv_dst, bv_lo) * (one - sat_v(curv_dst, bv_hi)) * flat;
                let (gain_i, loss_i) = bounded_excess_pair_v(token, b_dst, b_src, c_bv);
                acc += sp(co.c_bv_gain) * gain_i + sp(co.c_bv_loss) * loss_i;
            }
            acc += sp(co.c_edgew) * (gmag_d - gmag_s);
            let o = i - out_off;
            let id_new = V8::<T>::from_array(token, id_plane[o..o + 8].try_into().unwrap()) + acc;
            id_plane[o..o + 8].copy_from_slice(&id_new.to_array());
            x += 8;
        }
        while x < interior_end {
            let i = row + x;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                src[i - 1],
                src[i + 1],
                src[ru + x],
                src[rd + x],
                dst[i - 1],
                dst[i + 1],
                dst[ru + x],
                dst[rd + x],
                co,
                need_bandvis,
            );
            x += 1;
        }
        // x = width−1 (xr clamps to width−1).
        {
            let x = width - 1;
            let i = row + x;
            id_plane[i - out_off] += attr_pass_b_grad_px(
                src[i],
                dst[i],
                act_p[i],
                src[i - 1],
                src[i],
                src[ru + x],
                src[rd + x],
                dst[i - 1],
                dst[i],
                dst[ru + x],
                dst[rd + x],
                co,
                need_bandvis,
            );
        }
    }
}

/// Tiered magetypes entry for the pass-B gradient sweep.
#[cfg(feature = "custom-profiles")]
#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    act_p: &[f32],
    co: &V2AppCoeffsF32,
    width: usize,
    height: usize,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    attr_pass_b_grad_kernel_generic(token, src, dst, act_p, co, width, height, y0, y1, id_plane);
}

/// Gradient-family sweep of the f32 fused pass-B (gms / gms-dev / ringing
/// / banding / BANDVIS / edge-width) — magetypes-tiered dispatch; skipped
/// entirely when every gradient-family coefficient is zero.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_grad_rows_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    act_p: &[f32],
    co: V2AppCoeffsF32,
    y0: usize,
    y1: usize,
    id_plane: &mut [f32],
) {
    let need_bandvis = co.c_bv_gain != 0.0 || co.c_bv_loss != 0.0;
    let need_grad = co.c_gms != 0.0
        || co.c_ringing != 0.0
        || co.c_banding != 0.0
        || co.gg_lin != 0.0
        || co.gg_quad != 0.0
        || co.c_edgew != 0.0
        || need_bandvis;
    if !need_grad {
        return;
    }
    incant!(
        attr_pass_b_grad_entry(src, dst, act_p, &co, width, height, y0, y1, id_plane),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// f32 twin of [`attr_pass_b_blockiness`] (serial — the horizontal family
/// writes the row above).
#[cfg(feature = "custom-profiles")]
fn attr_pass_b_blockiness_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    c_blockiness: f32,
    id_plane: &mut [f32],
) {
    const CB: f32 = C_BLOCK as f32;
    if c_blockiness != 0.0 {
        for y in 0..height {
            let row = y * width;
            let mut x = BLOCK_LATTICE;
            while x < width {
                let i = row + x;
                let step_dst = (dst[i] - dst[i - 1]).abs();
                let step_src = (src[i] - src[i - 1]).abs();
                let v =
                    c_blockiness * ((step_dst - step_src).max(0.0) / (step_dst + step_src + CB));
                id_plane[i] += 0.5 * v;
                id_plane[i - 1] += 0.5 * v;
                x += BLOCK_LATTICE;
            }
            if y % BLOCK_LATTICE == 0 && y > 0 {
                for x in 0..width {
                    let i = row + x;
                    let i_up = i - width;
                    let step_dst = (dst[i] - dst[i_up]).abs();
                    let step_src = (src[i] - src[i_up]).abs();
                    let v = c_blockiness
                        * ((step_dst - step_src).max(0.0) / (step_dst + step_src + CB));
                    id_plane[i] += 0.5 * v;
                    id_plane[i_up] += 0.5 * v;
                }
            }
        }
    }
}

/// f32 twin of [`attr_pass_b_channel`]: row-banded parallel main+gradient
/// sweeps into f32 id/win planes; blockiness serial.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_channel_f32(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    planes: &AttrChPlanes,
    cross: Option<(&[f32], &[f32])>,
    ref_y: &[f32],
    co: V2AppCoeffsF32,
    parallel: bool,
    id_plane: &mut [f32],
    win_plane: &mut [f32],
) {
    // Band size for the rayon arm below; `threads`-gated with it.
    #[cfg(feature = "threads")]
    const BAND: usize = 64;
    let (ax, ab) = cross.unwrap_or((ref_y, ref_y));
    let run = |y0: usize, y1: usize, id_rows: &mut [f32], win_rows: &mut [f32]| {
        attr_pass_b_rows_f32(
            src,
            dst,
            width,
            height,
            &planes.mu1,
            &planes.mu2,
            &planes.ssq,
            &planes.s12,
            &planes.act,
            &planes.bs2,
            ax,
            ab,
            ref_y,
            co,
            y0,
            y1,
            id_rows,
            win_rows,
        );
    };
    #[cfg(feature = "threads")]
    if parallel && height > BAND {
        use rayon::prelude::*;
        id_plane[..width * height]
            .par_chunks_mut(BAND * width)
            .zip(win_plane[..width * height].par_chunks_mut(BAND * width))
            .enumerate()
            .for_each(|(band, (id_rows, win_rows))| {
                let y0 = band * BAND;
                let y1 = (y0 + BAND).min(height);
                run(y0, y1, id_rows, win_rows);
            });
        attr_pass_b_blockiness_f32(src, dst, width, height, co.c_blockiness, id_plane);
        return;
    }
    let _ = parallel;
    run(0, height, id_plane, win_plane);
    attr_pass_b_blockiness_f32(src, dst, width, height, co.c_blockiness, id_plane);
}

/// Edge-width pass-B coefficient for one (scale `u`, `ch`) — a free fn
/// (extracted 2026-08-05, appendix N) so the fused folded-944 path, which
/// derives its inputs from walk RETENTION instead of the attr pass-A
/// replication, shares the exact formula. Contributions from E(t) at
/// t = u (denominator role of the decay ratio) and t = u−1 (numerator
/// role); E(t) = 1 − bounded_sim(a, b); the LAST scale's slot is a copy
/// of E(n−2) so its gradient weight folds into E(n−2)'s. δmgd per
/// refined pixel = −(|∇d|−|∇s|)_i/N on BOTH terms (sign FD-gate-verified
/// in C2a). `dims[u]` are scale `u`'s plane dims.
#[cfg(feature = "custom-profiles")]
fn attr_ew_coeff(
    s_v2: &[f64],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    u: usize,
    ch: usize,
) -> f64 {
    if n_scales < 2 {
        return 0.0;
    }
    let mut coeff = 0.0f64;
    let n_u = (dims[u].0 * dims[u].1) as f64;
    let s_at = |t: usize| -> f64 {
        let v2b = t * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
        let mut sk = s_v2
            .get(v2b + idx::EDGE_WIDTH_CHANGE)
            .copied()
            .unwrap_or(0.0);
        if t == n_scales - 2 {
            let lb = (n_scales - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            sk += s_v2
                .get(lb + idx::EDGE_WIDTH_CHANGE)
                .copied()
                .unwrap_or(0.0);
        }
        sk
    };
    // (−∂bs/∂b, pgd, gd) for E(t).
    let de_db_at = |t: usize| -> (f64, f64, f64) {
        let (pgs, pgd) = mg[t][ch];
        let (gs, gd) = mg[t + 1][ch];
        let a = gs / (pgs + C_GRAD_DECAY);
        let b = gd / (pgd + C_GRAD_DECAY);
        let den = a * a + b * b + C_EDGEWIDTH;
        let dbs_db = (2.0 * a * den - (2.0 * a * b + C_EDGEWIDTH) * 2.0 * b) / (den * den);
        (-dbs_db, pgd, gd)
    };
    // E(u): scale u is the DENOMINATOR (mgd_t) of b.
    if u + 1 < n_scales {
        let sk = s_at(u);
        if sk != 0.0 {
            let (de_db, pgd, gd) = de_db_at(u);
            coeff +=
                sk * de_db * (-gd / ((pgd + C_GRAD_DECAY) * (pgd + C_GRAD_DECAY))) * (-1.0 / n_u);
        }
    }
    // E(u−1): scale u is the NUMERATOR (mgd_{t+1}) of b.
    if u >= 1 {
        let sk = s_at(u - 1);
        if sk != 0.0 {
            let (de_db, pgd, _gd) = de_db_at(u - 1);
            coeff += sk * de_db * (1.0 / (pgd + C_GRAD_DECAY)) * (-1.0 / n_u);
        }
    }
    coeff
}

/// Pass B for one scale (combine → spread window mass → upsample into the
/// full-res canvas) — a free fn (extracted 2026-08-05, appendix N) shared
/// verbatim by the standalone attribution pipeline and the fused
/// folded-944 retention path. `rplanes`/`dplanes` are the scale's pyramid
/// planes, `dims` the per-scale plane dims, `(w0, h0)` the scale-0
/// (canvas) dims.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_for_scale(
    scale: usize,
    rplanes: [&[f32]; 3],
    dplanes: [&[f32]; 3],
    planes: &[AttrChPlanes; 3],
    cells: &[[AttrCellSums; 3]],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    want_append: bool,
    parallel: bool,
    w0: usize,
    h0: usize,
    scale_density: &mut [f64],
    win_plane: &mut [f64],
    sink: &mut crate::attribution::AttrSinkF64<'_>,
    spread_tmp: &mut Vec<f64>,
) {
    let tpb = std::time::Instant::now();
    let (ws, hs) = dims[scale];
    let n = ws * hs;
    scale_density[..n].fill(0.0);
    win_plane[..n].fill(0.0);
    for ch in 0..3 {
        let append_active = append_cell_active(want_append, ch, scale);
        let cross: Option<(&[f32], &[f32])> = if ch == 1 {
            Some((&planes[0].act, &planes[2].act))
        } else {
            None
        };
        let co = derive_v2app_coeffs(
            s_v2,
            s_append,
            s_append2,
            scale,
            ch,
            &cells[scale][ch],
            append_active,
            ch == 1,
            attr_ew_coeff(s_v2, mg, dims, n_scales, scale, ch),
        );
        attr_pass_b_channel(
            rplanes[ch],
            dplanes[ch],
            ws,
            hs,
            &planes[ch],
            cross,
            rplanes[1],
            &co,
            parallel,
            &mut scale_density[..n],
            &mut win_plane[..n],
        );
    }
    crate::blur::box_spread_sum_preserving(&mut win_plane[..n], ws, hs, BLUR_RADIUS, spread_tmp);
    for (d, s) in scale_density[..n].iter_mut().zip(win_plane[..n].iter()) {
        *d += *s;
    }
    PERF_PASSB.with(|c| c.set(c.get() + tpb.elapsed().as_secs_f64()));
    // Sum-preserving footprint upsample (the v2 pyramid floor-halves,
    // so footprints are full 2^s × 2^s blocks; ÷4^s is exact).
    let factor = 1usize << scale;
    match sink {
        crate::attribution::AttrSinkF64::Canvas(canvas) => {
            if factor == 1 {
                for (c, &v) in canvas.iter_mut().zip(scale_density[..n].iter()) {
                    *c += v;
                }
            } else {
                let inv_area = 1.0 / ((factor * factor) as f64);
                for sy in 0..hs {
                    let y0 = sy * factor;
                    let y1 = (y0 + factor).min(h0);
                    for sx in 0..ws {
                        let v = scale_density[sy * ws + sx] * inv_area;
                        let x0 = sx * factor;
                        let x1 = (x0 + factor).min(w0);
                        for row in canvas[y0 * w0..].chunks_mut(w0).take(y1.saturating_sub(y0)) {
                            for slot in &mut row[x0..x1] {
                                *slot += v;
                            }
                        }
                    }
                }
            }
        }
        crate::attribution::AttrSinkF64::Bins(accum) => {
            accum.add_scale_plane_f64(&scale_density[..n], ws, hs, factor);
        }
    }
}

/// f32 twin of [`attr_pass_b_for_scale`] (appendix P lever 1) — the FUSED
/// entry's per-scale pass B: f64 coefficient derivation (identical inputs),
/// folded to the f32 pack, f32 combine kernels, window spread fused with
/// the window→identity merge ([`crate::blur::box_spread_merge_f32`]), f32
/// sum-preserving footprint upsample into the f32 canvas.
///
/// `custom-profiles`-gated: calls `crate::attribution::
/// upsample_add_sum_preserving_f32`, which does not exist without that
/// feature (the fused-entry cluster; af4417f8 rule).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
fn attr_pass_b_for_scale_f32(
    scale: usize,
    rplanes: [&[f32]; 3],
    dplanes: [&[f32]; 3],
    planes: &[AttrChPlanes; 3],
    cells: &[[AttrCellSums; 3]],
    mg: &[[(f64, f64); 3]],
    dims: &[(usize, usize)],
    n_scales: usize,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    want_append: bool,
    parallel: bool,
    w0: usize,
    h0: usize,
    scale_density: &mut [f32],
    win_plane: &mut [f32],
    sink: &mut crate::attribution::AttrSinkF32<'_>,
    spread_tmp: &mut Vec<f32>,
    spread_out: &mut Vec<f32>,
) {
    let tpb = std::time::Instant::now();
    let (ws, hs) = dims[scale];
    let n = ws * hs;
    scale_density[..n].fill(0.0);
    win_plane[..n].fill(0.0);
    for ch in 0..3 {
        let append_active = append_cell_active(want_append, ch, scale);
        let cross: Option<(&[f32], &[f32])> = if ch == 1 {
            Some((&planes[0].act, &planes[2].act))
        } else {
            None
        };
        let co = derive_v2app_coeffs(
            s_v2,
            s_append,
            s_append2,
            scale,
            ch,
            &cells[scale][ch],
            append_active,
            ch == 1,
            attr_ew_coeff(s_v2, mg, dims, n_scales, scale, ch),
        );
        let co32 = v2app_coeffs_fold_f32(&co, cross.is_some());
        attr_pass_b_channel_f32(
            rplanes[ch],
            dplanes[ch],
            ws,
            hs,
            &planes[ch],
            cross,
            rplanes[1],
            co32,
            parallel,
            &mut scale_density[..n],
            &mut win_plane[..n],
        );
    }
    // Window spread fused with the window→identity merge (value-exact vs
    // spread-then-add; parallel is bitwise-invariant per its gate).
    crate::blur::box_spread_merge_f32(
        &mut win_plane[..n],
        &mut scale_density[..n],
        ws,
        hs,
        BLUR_RADIUS,
        spread_tmp,
        spread_out,
        parallel && n >= crate::blur::SPREAD_PARALLEL_MIN_N,
    );
    PERF_PASSB.with(|c| c.set(c.get() + tpb.elapsed().as_secs_f64()));
    match sink {
        crate::attribution::AttrSinkF32::Canvas(canvas) => {
            crate::attribution::upsample_add_sum_preserving_f32(
                &scale_density[..n],
                ws,
                hs,
                canvas,
                w0,
                h0,
                1usize << scale,
            );
        }
        crate::attribution::AttrSinkF32::Bins(accum) => {
            accum.add_scale_plane_f32(&scale_density[..n], ws, hs, 1usize << scale);
        }
    }
}

/// Build the v2 (+ optional append) attribution density for a pair.
///
/// `s_v2` = raw `∂score/∂f` for the v2 block (`f372..720` layout,
/// `scale*87 + ch*29 + slot`, up to 348 entries); `s_append` likewise for
/// the append block (`f720..924`, `scale*51 + ch*17 + slot`, 204 entries);
/// `s_append2` for the append2 block (`f924..944`, `scale*5 + slot`, 20
/// entries, Y-only — see `derive_v2app_coeffs`'s append2 section for which
/// of the 5 local slots carry a term and why the other three are exactly
/// zero). Toggles are fixed to `V2NewFeatureToggles::default()` — the
/// 924/944-regime canon. See the section comment above for integrand
/// classes and the documented approximations.
///
/// The whole attribution-density cluster (this entry, its pass A/B helpers,
/// and the f32 fused twins above) is `custom-profiles`-gated: every non-test
/// consumer lives in `crate::attribution`, which only exists under that
/// feature — the same both-features rule af4417f8 applied to the
/// `Fused944Session` re-export. Without the gates, a
/// `feature-regime-v2`-only build failed to COMPILE (E0433 on
/// `crate::attribution` in the f32 pass B; flagged in 87c5e9ef) and, once
/// that was gated, dead-coded the remaining 27 items of the cluster.
#[cfg(feature = "custom-profiles")]
pub(crate) fn compute_v2_append_attribution(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
) -> Result<V2AppendAttribution, ZensimError> {
    let (density, v2_features, append_features) = compute_v2_append_attribution_impl(
        reference, distorted, s_v2, s_append, s_append2, max_pixels, parallel, None,
    )?;
    Ok(V2AppendAttribution {
        density: density.expect("canvas arm always yields a density"),
        width: reference.width(),
        height: reference.height(),
        v2_features,
        append_features,
    })
}

/// Level-2 sibling: the per-scale v2/append mass folds straight into the
/// caller's [`BinAccum`](crate::attribution::BinAccum) — no full-resolution
/// v2 density plane, no trim copy. Returns `(v2_features, append_features)`
/// (the pass-A audit surface).
#[cfg(feature = "custom-profiles")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_v2_append_attribution_into_bins(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
    accum: &mut crate::attribution::BinAccum,
) -> Result<(Vec<f64>, Vec<f64>), ZensimError> {
    let (_, v2_features, append_features) = compute_v2_append_attribution_impl(
        reference,
        distorted,
        s_v2,
        s_append,
        s_append2,
        max_pixels,
        parallel,
        Some(accum),
    )?;
    Ok((v2_features, append_features))
}

/// Shared core of the two entries above: `bins == None` reproduces the
/// pre-Level-2 canvas path byte-identically (alloc → accumulate → trim);
/// `Some(accum)` folds per scale into the bins and allocates no canvas.
#[cfg(feature = "custom-profiles")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn compute_v2_append_attribution_impl(
    reference: &impl ImageSource,
    distorted: &impl ImageSource,
    s_v2: &[f64],
    s_append: Option<&[f64]>,
    s_append2: Option<&[f64]>,
    max_pixels: Option<usize>,
    parallel: bool,
    bins: Option<&mut crate::attribution::BinAccum>,
) -> Result<(Option<Vec<f64>>, Vec<f64>, Vec<f64>), ZensimError> {
    // ZENSIM_ATTR_PERF=1: coarse section timing to stderr (perf lever triage).
    let perf_log = std::env::var("ZENSIM_ATTR_PERF").as_deref() == Ok("1");
    let t0 = std::time::Instant::now();
    let rprep = prepare_v2_reference_impl(reference, max_pixels, parallel, false)?;
    let dprep = prepare_v2_reference_impl(distorted, max_pixels, parallel, false)?;
    let t_prep = t0.elapsed();
    let n_scales = rprep.scales.len().min(dprep.scales.len());
    let (w0, h0) = (rprep.scales[0].1, rprep.scales[0].2);
    let want_append = s_append.is_some();
    let toggles = V2NewFeatureToggles::default();

    // Single-sweep pipeline (C2b Part 2): pass B(s) needs the edge-width
    // coefficients, which need mean gradients up to scale s+1 — so the
    // schedule is A(0), A(1), B(0), A(2), B(1), ..., B(n−1), with TWO
    // ping-ponged plane sets so scale s's caches survive A(s+1). This
    // removes the former second full blur+cache sweep entirely.
    let mut scratches: [ScratchV2Strip; 3] = [
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
        ScratchV2Strip::new(w0 * (STRIP_ROWS + 2 * HALO_P)),
    ];
    let mut planes_pp: [[AttrChPlanes; 3]; 2] = [
        [
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
        ],
        [
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
            AttrChPlanes::new(w0 * h0),
        ],
    ];
    let mut grad_halo: Vec<f32> = Vec::new();
    let mut grad_halo_d: Vec<f32> = Vec::new();

    let mut cells: Vec<[AttrCellSums; 3]> = vec![[AttrCellSums::default(); 3]; n_scales];
    let mut v2_features = vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL];
    let mut append_features = if want_append {
        vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_APPEND]
    } else {
        Vec::new()
    };
    // (mean grad src, mean grad dst) per (scale, ch) for edge-width.
    let mut mg = vec![[(0.0f64, 0.0f64); 3]; n_scales];
    let want_canvas = bins.is_none();
    let mut canvas = if want_canvas {
        vec![0.0f64; w0 * h0]
    } else {
        Vec::new()
    };
    let mut sink = match bins {
        Some(accum) => crate::attribution::AttrSinkF64::Bins(accum),
        None => crate::attribution::AttrSinkF64::Canvas(&mut canvas),
    };
    let mut scale_density = vec![0.0f64; w0 * h0];
    let mut win_plane = vec![0.0f64; w0 * h0];
    let mut spread_tmp: Vec<f64> = Vec::new();

    // ── Pass A for one scale into the given plane set. ──
    #[allow(clippy::too_many_arguments)]
    fn pass_a_scale(
        rprep: &V2PreparedReference,
        dprep: &V2PreparedReference,
        scale: usize,
        want_append: bool,
        parallel: bool,
        toggles: V2NewFeatureToggles,
        planes: &mut [AttrChPlanes; 3],
        scratches: &mut [ScratchV2Strip; 3],
        cells: &mut [[AttrCellSums; 3]],
        mg: &mut [[(f64, f64); 3]],
        v2_features: &mut [f64],
        append_features: &mut [f64],
        grad_halo: &mut Vec<f32>,
        grad_halo_d: &mut Vec<f32>,
    ) {
        let (ref rplanes, ws, hs) = rprep.scales[scale];
        let dplanes = &dprep.scales[scale].0;
        let tb = std::time::Instant::now();
        // Channel-parallel blur+cache (C2b Part 2): the three channels are
        // independent given per-channel scratch.
        #[cfg(feature = "threads")]
        let did_parallel = if parallel {
            let [p0, p1, p2] = planes;
            let [s0, s1, s2] = scratches;
            rayon::join(
                || attr_blur_cache_channel(&rplanes[0], &dplanes[0], ws, hs, want_append, s0, p0),
                || {
                    rayon::join(
                        || {
                            attr_blur_cache_channel(
                                &rplanes[1],
                                &dplanes[1],
                                ws,
                                hs,
                                want_append,
                                s1,
                                p1,
                            )
                        },
                        || {
                            attr_blur_cache_channel(
                                &rplanes[2],
                                &dplanes[2],
                                ws,
                                hs,
                                want_append,
                                s2,
                                p2,
                            )
                        },
                    )
                },
            );
            true
        } else {
            false
        };
        #[cfg(not(feature = "threads"))]
        let did_parallel = false;
        // `parallel` is only consulted by the rayon arm above; consume it on
        // no-`threads` builds (same idiom as `attr_pass_b_channel`).
        #[cfg(not(feature = "threads"))]
        let _ = parallel;
        if !did_parallel {
            for ch in 0..3 {
                attr_blur_cache_channel(
                    &rplanes[ch],
                    &dplanes[ch],
                    ws,
                    hs,
                    want_append,
                    &mut scratches[0],
                    &mut planes[ch],
                );
            }
        }
        PERF_BLUR.with(|c| c.set(c.get() + tb.elapsed().as_secs_f64()));
        let tk = std::time::Instant::now();
        for ch in 0..3 {
            let append_active = append_cell_active(want_append, ch, scale);
            let cross: Option<(&[f32], &[f32])> = if ch == 1 {
                Some((&planes[0].act, &planes[2].act))
            } else {
                None
            };
            let cell = attr_pass_a_kernels(
                &rplanes[ch],
                &dplanes[ch],
                ws,
                hs,
                &planes[ch],
                cross,
                &rplanes[1],
                want_append,
                append_active,
                grad_halo,
                grad_halo_d,
            );
            let v2b =
                scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
            let (gs, gd) = finish_channel_scale(
                &cell.dense,
                &cell.grad,
                cell.blockiness,
                cell.n,
                &mut v2_features[v2b..v2b + FEATURES_PER_CHANNEL_V2_TOTAL],
            );
            mg[scale][ch] = (gs, gd);
            if want_append {
                let apb =
                    scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
                if append_active {
                    finish_append(
                        &cell.dense,
                        &cell.app,
                        &cell.grad,
                        cell.n,
                        ch == 1,
                        toggles,
                        &mut append_features[apb..apb + FEATURES_PER_CHANNEL_APPEND],
                    );
                }
            }
            cells[scale][ch] = cell;
        }
        PERF_KERN.with(|c| c.set(c.get() + tk.elapsed().as_secs_f64()));
    }

    // ── Pass B (combine → spread → upsample): the shared free fn
    //    `attr_pass_b_for_scale` (+ `attr_ew_coeff` inside it) — extracted
    //    2026-08-05 (appendix N) so the fused folded-944 retention path
    //    shares them verbatim. Behavior here is unchanged. ──
    let dims: Vec<(usize, usize)> = rprep.scales.iter().map(|s| (s.1, s.2)).collect();
    let pass_b_scale = |scale: usize,
                        planes: &[AttrChPlanes; 3],
                        cells: &[[AttrCellSums; 3]],
                        mg: &[[(f64, f64); 3]],
                        scale_density: &mut Vec<f64>,
                        win_plane: &mut Vec<f64>,
                        sink: &mut crate::attribution::AttrSinkF64<'_>,
                        spread_tmp: &mut Vec<f64>| {
        let (ref rplanes, _, _) = rprep.scales[scale];
        let dplanes = &dprep.scales[scale].0;
        attr_pass_b_for_scale(
            scale,
            [&rplanes[0], &rplanes[1], &rplanes[2]],
            [&dplanes[0], &dplanes[1], &dplanes[2]],
            planes,
            cells,
            mg,
            &dims,
            n_scales,
            s_v2,
            s_append,
            s_append2,
            want_append,
            parallel,
            w0,
            h0,
            scale_density,
            win_plane,
            sink,
            spread_tmp,
        );
    };

    // ── Pipeline: A(s); once A(s) exists, fill the cross-scale edge-width
    //    FEATURES for scale s−1 (production replication incl. the
    //    last-scale copy), then B(s−1); finish with B(n−1). ──
    for scale in 0..n_scales {
        {
            let [set_a, set_b] = &mut planes_pp;
            let pset = if scale % 2 == 0 { set_a } else { set_b };
            pass_a_scale(
                &rprep,
                &dprep,
                scale,
                want_append,
                parallel,
                toggles,
                pset,
                &mut scratches,
                &mut cells,
                &mut mg,
                &mut v2_features,
                &mut append_features,
                &mut grad_halo,
                &mut grad_halo_d,
            );
        }
        if scale >= 1 {
            #[allow(clippy::needless_range_loop)] // ch indexes two scales of mg + feature bases
            for ch in 0..3 {
                let (pgs, pgd) = mg[scale - 1][ch];
                let (gs, gd) = mg[scale][ch];
                let decay_src = gs / (pgs + C_GRAD_DECAY);
                let decay_dst = gd / (pgd + C_GRAD_DECAY);
                let prev_b = (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                v2_features[prev_b + idx::EDGE_WIDTH_CHANGE] =
                    1.0 - bounded_sim(decay_src, decay_dst, C_EDGEWIDTH);
                if scale == n_scales - 1 && n_scales >= 2 {
                    let this_b = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                        + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                    v2_features[this_b + idx::EDGE_WIDTH_CHANGE] =
                        v2_features[prev_b + idx::EDGE_WIDTH_CHANGE];
                }
            }
            let prev = scale - 1;
            pass_b_scale(
                prev,
                &planes_pp[prev % 2],
                &cells,
                &mg,
                &mut scale_density,
                &mut win_plane,
                &mut sink,
                &mut spread_tmp,
            );
        }
    }
    {
        let last = n_scales - 1;
        pass_b_scale(
            last,
            &planes_pp[last % 2],
            &cells,
            &mg,
            &mut scale_density,
            &mut win_plane,
            &mut sink,
            &mut spread_tmp,
        );
    }
    if perf_log {
        eprintln!(
            "ATTRPERF v2app: prep {:.1} ms | pipeline {:.1} ms (A: blur+cache {:.1}, kernels {:.1}; B: combine {:.1})",
            t_prep.as_secs_f64() * 1e3,
            t0.elapsed().as_secs_f64() * 1e3 - t_prep.as_secs_f64() * 1e3,
            PERF_BLUR.with(|c| c.get()) * 1e3,
            PERF_KERN.with(|c| c.get()) * 1e3,
            PERF_PASSB.with(|c| c.get()) * 1e3,
        );
        PERF_BLUR.with(|c| c.set(0.0));
        PERF_KERN.with(|c| c.set(0.0));
        PERF_PASSB.with(|c| c.set(0.0));
    }

    // (The sink's borrow of `canvas` ends with its last use above.)
    // Trim the (possibly reflect-padded sub-64) canvas to the original
    // (canvas arm only; the bins arm clipped at fold time).
    let density = if want_canvas {
        let (ow, oh) = (reference.width(), reference.height());
        Some(if ow == w0 && oh == h0 {
            canvas
        } else {
            let mut out = Vec::with_capacity(ow * oh);
            for y in 0..oh.min(h0) {
                out.extend_from_slice(&canvas[y * w0..y * w0 + ow.min(w0)]);
            }
            out
        })
    } else {
        None
    };
    Ok((density, v2_features, append_features))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::source::RgbSlice;

    /// imazen/zensim#56 regression gate: the MSCN divisive normalizer must
    /// be the CORRECTLY-ROUNDED IEEE `resid / sqrt(var + c)` on every SIMD
    /// tier, lane for lane, bit for bit. That is what makes the
    /// `MSCN_DIFF_MEAN`/`MSCN_DIFF_L2` slots CPU-vendor-deterministic: IEEE
    /// sqrt and div are exactly rounded on every vendor, whereas the
    /// `rsqrt()` the kernel used before (`vrsqrtps` estimate + 1 NR step on
    /// x86; `1/sqrt` then multiply on NEON and scalar) is not — the bf944
    /// wave measured AMD≠Intel by ~1e-8 rel on exactly those 22 columns.
    ///
    /// Mutation-verified: substituting `resid * (var + c).rsqrt()` in
    /// `mscn_norm_v` fails this test on x86 (NR residue) AND on NEON /
    /// scalar (double rounding), so it cannot silently regress on any CI
    /// platform. The corpus deliberately includes the exact-zero variance
    /// floor (`var = 0` → `sqrt(C_MSCN_VAR)`), sub-floor, unit and large
    /// variances, and signed residuals spanning ±[1e-4, 4].
    fn check_mscn_norm_tier<T: F32x8Backend + Copy>(token: T, tier: &str) {
        let c = C_MSCN_VAR as f32;
        let cv = V8::<T>::splat(token, c);
        // Deterministic LCG so the corpus is identical on every platform.
        let mut state = 0x9E37_79B9u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut n_checked = 0usize;
        let mut n_mismatch = 0usize;
        let mut first_mismatch: Option<(f32, f32, f32, f32)> = None;
        for block in 0..512 {
            let mut resid = [0f32; 8];
            let mut var = [0f32; 8];
            for lane in 0..8 {
                let u = next();
                // Signed residual, log-spread magnitude 1e-4..4.
                let mag = 1e-4 * (4e4f32).powf(next());
                resid[lane] = if u < 0.5 { -mag } else { mag };
                // Variance: exact floor every 8th lane, else log-spread
                // 1e-9..1e2 (covers below the C_MSCN_VAR floor and far above).
                var[lane] = if (block + lane) % 8 == 0 {
                    0.0
                } else {
                    1e-9 * (1e11f32).powf(next())
                };
            }
            let got = mscn_norm_v(
                token,
                V8::<T>::from_array(token, resid),
                V8::<T>::from_array(token, var),
                cv,
            )
            .to_array();
            for lane in 0..8 {
                // Scalar Rust f32 sqrt and div are IEEE-754 correctly rounded.
                let want = resid[lane] / (var[lane] + c).sqrt();
                n_checked += 1;
                if got[lane].to_bits() != want.to_bits() {
                    n_mismatch += 1;
                    first_mismatch.get_or_insert((resid[lane], var[lane], got[lane], want));
                }
            }
        }
        assert_eq!(
            n_mismatch, 0,
            "tier {tier}: {n_mismatch}/{n_checked} lanes of mscn_norm_v are not the \
             correctly-rounded IEEE resid/sqrt(var+c); first: {first_mismatch:?} \
             (resid, var, got, want)"
        );
    }

    #[test]
    fn mscn_norm_v_is_correctly_rounded_on_every_tier() {
        use archmage::SimdToken as _;
        let mut tiers_run = 0usize;
        check_mscn_norm_tier(
            archmage::ScalarToken::summon().expect("scalar token is infallible"),
            "scalar",
        );
        tiers_run += 1;
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(t) = archmage::X64V3Token::summon() {
                check_mscn_norm_tier(t, "x86 v3");
                tiers_run += 1;
            }
            if let Some(t) = archmage::X64V4Token::summon() {
                check_mscn_norm_tier(t, "x86 v4");
                tiers_run += 1;
            }
            #[cfg(feature = "avx512")]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_mscn_norm_tier(t, "x86 v4x");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(t) = archmage::NeonToken::summon() {
                check_mscn_norm_tier(t, "neon");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(t) = archmage::Wasm128Token::summon() {
                check_mscn_norm_tier(t, "wasm128");
                tiers_run += 1;
            }
        }
        assert!(tiers_run >= 1, "no tier was exercised");
    }

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
    pub(crate) fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
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
    pub(crate) fn quantize_distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
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
    /// ERA-2 STRUCTURAL GATE: the properties the break actually rests on.
    ///
    /// **A correction the first version of this test earned.** It originally
    /// asserted "band-merge == serial fold, bit-exact" and FAILED at 127x93
    /// (7.854278564453125 vs 7.8542633056640625). The assertion was wrong, not
    /// the design: banding IS a different grouping — serial computes
    /// `((band0 + r32) + r33) + …` while banded computes `band0 + band1` — and
    /// f64/f32 addition is not associative. This is the same blocking
    /// non-associativity documented for the era-1 dense kernel; it does not go
    /// away because the shape is new.
    ///
    /// **So `ERA2_BAND_ROWS` is SEMANTICS, not a tuning knob.** Changing it
    /// changes output bytes and is an era decision. It is deliberately NOT
    /// derived from the thread count, image height, or anything else that
    /// varies at runtime — that is what makes thread-invariance structural.
    /// Anyone reaching for it to "tune the band size" must read this first.
    ///
    /// What is actually asserted:
    /// (a) the banded fold is INVARIANT to how bands are distributed across
    ///     workers — computing them in any order, or all at once, gives
    ///     bit-identical results, because the merge is sequential in band index;
    /// (b) lane `j` holds exactly the terms at `x ≡ j (mod 8)` in increasing
    ///     `x`, tail included — so the shape is width-independent, unlike
    ///     era-1's, whose tail landed in the f64 running total.
    #[test]
    fn era2_band_merge_and_tail_are_structural() {
        let term = |i: usize| -> f32 {
            let k = (i * 2654435761usize) % 65521;
            (k as f32) * (1.0 / 65536.0) - 0.5
        };

        // Fold one band's rows into its own lane set.
        let band_of = |plane: &[f32], width: usize, y0: usize, y1: usize| -> super::Lanes8 {
            let mut band = super::Lanes8::zero();
            for y in y0..y1 {
                let row = &plane[y * width..(y + 1) * width];
                let (chunks, tail) = row.as_chunks::<8>();
                for c in chunks {
                    band.add_chunk(c);
                }
                band.add_tail(tail);
            }
            band
        };

        for &(width, rows) in &[(576usize, 128usize), (127, 93), (61, 40), (8, 3), (5, 1)] {
            let plane: Vec<f32> = (0..width * rows).map(term).collect();

            let mut starts = Vec::new();
            let mut b = 0usize;
            while b < rows {
                starts.push(b);
                b = (b + super::ERA2_BAND_ROWS).min(rows);
            }

            // Reference: bands computed in index order, merged in index order.
            let mut reference = super::Lanes8::zero();
            for (i, &y0) in starts.iter().enumerate() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                let bandv = band_of(&plane, width, y0, y1);
                for (m, v) in reference.0.iter_mut().zip(bandv.0.iter()) {
                    *m += *v;
                }
            }

            // (a) Compute the bands in REVERSE order (as a worker pool would,
            // completing out of order), then merge in index order as the design
            // requires. Must be bit-identical.
            let mut partials: Vec<Option<super::Lanes8>> = vec![None; starts.len()];
            for (i, &y0) in starts.iter().enumerate().rev() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                partials[i] = Some(band_of(&plane, width, y0, y1));
            }
            let mut out_of_order = super::Lanes8::zero();
            for p in partials.iter().flatten() {
                for (m, v) in out_of_order.0.iter_mut().zip(p.0.iter()) {
                    *m += *v;
                }
            }
            assert_eq!(
                reference.reduce().to_bits(),
                out_of_order.reduce().to_bits(),
                "{width}x{rows}: computing bands out of order changed the result. \
                 era-2's thread-invariance rests on the MERGE being sequential in \
                 band index while the bands themselves are order-free."
            );

            // (b) lane j == the terms at x ≡ j (mod 8), tail included.
            let mut expect = [0.0f32; 8];
            for y in 0..rows {
                for x in 0..width {
                    expect[x % 8] += plane[y * width + x];
                }
            }
            let single = band_of(&plane, width, 0, rows);
            assert_eq!(
                single.0.map(f32::to_bits),
                expect.map(f32::to_bits),
                "{width}x{rows}: lane contents diverge from 'terms at x ≡ j (mod 8), \
                 in increasing x' — the tail fold is wrong."
            );
        }
    }

    /// The trap that motivates [`super::era2_reduce8`]: `reduce_add()` is
    /// TIER-DEPENDENT (§14.2), so it can never be part of era-2's semantics.
    ///
    /// This does not assert a divergence (this box runs one tier), it asserts
    /// the FIXED tree is what we think it is — an executable statement of the
    /// semantics, so a future refactor that silently swaps in `reduce_add()`
    /// fails here.
    #[test]
    fn era2_reduce_tree_is_fixed_and_explicit() {
        // Values chosen so the two known groupings genuinely differ: adjacent
        // pairing vs the x86_v3 (i, i+4) pairing.
        let a: [f32; 8] = [1.0, 1e-8, 1.0, 1e-8, -1.0, 1e-8, -1.0, 1e-8];
        let fixed = super::era2_reduce8(a);
        let adjacent = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64;
        let v3_style = (((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))) as f64;
        assert_eq!(
            fixed.to_bits(),
            adjacent.to_bits(),
            "era2_reduce8 must be the ADJACENT-pairwise tree, written out"
        );
        assert_ne!(
            adjacent.to_bits(),
            v3_style.to_bits(),
            "the two backend reduction shapes must be demonstrably different on \
             this fixture — if they agree, the fixture stopped exercising the \
             divergence and this test no longer documents anything"
        );
    }

    /// ERA-2 ORACLE GATE (`benchmarks/era2_perf_break_2026-08-31.md` §11).
    ///
    /// Judges the production kernels against an EXACT reference rather than
    /// against a previous implementation. Three things happen, in order:
    ///
    /// 1. **The oracle judges itself.** L1 (Neumaier-compensated f64) is
    ///    compared to L2 (Shewchuk exact expansion). If L1 drifts from L2
    ///    beyond compensated summation's own bound, the oracle is broken and
    ///    says so BEFORE it is used to judge anything else.
    /// 2. **Each production pool shape is measured against L2** — both the
    ///    `POOL_SIMD` lane form (what `v4x` ships today and what era-2 makes
    ///    universal) and the scalar-pool form (what every 16-register tier
    ///    ships today). Their deviations differ by construction, which is the
    ///    §10.4 accuracy-class change made visible instead of assumed.
    /// 3. **Every slot is checked against a bound derived from the SAME run** —
    ///    `Σ|xᵢ|` per slot comes out of the oracle, so the bound is computed
    ///    from measured magnitudes rather than guessed.
    ///
    /// Geometries deliberately straddle the classes that matter to the era-2
    /// reshape: tight vs non-tight width, `width % 8 == 0` vs not (the scalar
    /// tail), and heights that do and do not divide the band size.
    ///
    /// A deviation above its bound is a BUG — in the kernel or in the analysis
    /// — and the lane STOPS to find out which. It is never a tolerance to
    /// widen.
    #[test]
    fn era2_oracle_bounds_hold_for_every_pool_shape() {
        use oracle::{Exact, N_SLOTS, Neumaier, SLOT_NAMES, dense_accum_slots, slot_family};
        const U32: f64 = 5.960_464_477_539_063e-8; // 2^-24
        const U64: f64 = 1.110_223_024_625_157e-16; // 2^-53

        // (w, h, note)
        let geoms: &[(usize, usize, &str)] = &[
            (96, 64, "tight, w%8==0"),
            (200, 136, "non-tight, w%8==0"),
            (127, 93, "non-tight, w%8==7 (scalar tail)"),
            (61, 40, "sub-64, w%8==5 (tail + small)"),
        ];
        let mut worst_overall: (f64, String) = (0.0, String::new());
        // Per (variant, family): worst |dev| with the sum|x| and bound it came
        // with, so §10.5's table is filled from measurement, not guessed.
        let mut per: std::collections::BTreeMap<(&str, &str), (f64, f64, f64)> =
            std::collections::BTreeMap::new();

        for &(w, h, note) in geoms {
            let src_px = textured_image(w, h, 0xD1FF);
            let dst_px = quantize_distort(&src_px, w, h);
            let source = RgbSlice::new(&src_px, w, h);
            let distorted = RgbSlice::new(&dst_px, w, h);
            let src_planes = crate::streaming::convert_source_to_xyb(&source, w, false);
            let dst_planes = crate::streaming::convert_source_to_xyb(&distorted, w, false);
            let mut scratch = ScratchV2Strip::new(w * h);
            let n = w * h;

            for ch in 0..3 {
                run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
                let (mu1, mu2) = (&scratch.mu1[..n], &scratch.mu2[..n]);
                let (ssq, s12) = (&scratch.ssq[..n], &scratch.s12[..n]);
                let act = &scratch.activity[..n];

                let (l1, _) = oracle::dense_reference::<Neumaier>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );
                let (l2, sum_abs) = oracle::dense_reference::<Exact>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );

                // (1) The oracle judges itself. Compensated summation's error
                // is O(u) * Σ|x| regardless of n — that is the whole point of
                // the compensation — so a generous constant times u64 is a
                // true bound, not a fitted one.
                for i in 0..N_SLOTS {
                    let allow = 64.0 * U64 * sum_abs[i] + f64::MIN_POSITIVE;
                    assert!(
                        (l1[i] - l2[i]).abs() <= allow,
                        "{w}x{h} ch{ch} [{note}]: ORACLE SELF-CHECK FAILED at {} \
                         (L1 Neumaier {:e} vs L2 exact {:e}, allowed {:e}). The oracle is \
                         broken; do not trust any kernel judged by it.",
                        SLOT_NAMES[i],
                        l1[i],
                        l2[i],
                        allow
                    );
                }

                // (2)+(3) Each production pool shape vs L2, bound from Σ|x|.
                let lane_pool = dense_block_kernel(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );
                let scalar_pool = dense_block_kernel_pools_scalar(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );

                let era2 = dense_block_kernel_era2(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );
                for (variant, accum) in [
                    ("dispatched", &lane_pool),
                    ("pools_scalar", &scalar_pool),
                    ("era2", &era2),
                ] {
                    let got = dense_accum_slots(accum);
                    for i in 0..N_SLOTS {
                        let fam = slot_family(i);
                        // f32 lane accumulation: (chunks per lane) + (reduce
                        // tree depth) terms at u32. f64 aggregation: one add
                        // per row at u64. The scalar-pool variant accumulates
                        // its POOL slots per pixel in f64 instead, so those
                        // carry no u32 term.
                        // BOUND per design §12.2 + §12.5. Two components:
                        //
                        // (1) TERM-EVALUATION, absolute and cancellation-safe.
                        //     Every intermediate in these formulas is bounded
                        //     by 1 in magnitude (d, sal, the weights,
                        //     bounded_sim, saturate all live in [0,1]), so ONE
                        //     f32 rounding anywhere in a term contributes at
                        //     most u32·1 — ABSOLUTE, not relative to |term|.
                        //     That matters because `d = max(1-local,0)` with
                        //     `local ≈ 1` is a CANCELLING difference: d's
                        //     absolute error does not shrink as d does, and
                        //     the derived slots (d², d³, d⁴, and every pool
                        //     `num`) inherit it amplified by the derivative.
                        //     A bound written against the slot's own Σ|x|
                        //     understates those by two orders (Σ|d| is 282x
                        //     Σ|d²| here) — measured, and it is what the
                        //     oracle caught first.
                        //
                        // (2) ACCUMULATION, relative to Σ|x| (design §12.2).
                        //
                        // (1) dominates and is deliberately LOOSE — a proven
                        // upper bound, not a fitted one. The regression signal
                        // is the reported per-family COEFFICIENT below, which
                        // moves long before the bound is crossed.
                        const SAFETY: f64 = 2.0;
                        let k_eval = SAFETY
                            * match fam {
                                "core" => 13.0,
                                "hf" => 7.0,
                                "pjnd" => 5.0,
                                _ => 17.0, // pools: core formula + weight ops
                            };
                        let n_px = (w * h) as f64;
                        let chunks_per_lane = (w / 8) as f64;
                        let pools_in_f64 = variant == "pools_scalar" && fam == "pools";
                        // era-2 folds its tail into the lanes, so it has no
                        // per-pixel f64 tail term (era-1 does).
                        let era2_shape = variant == "era2";
                        let acc_u32 = if pools_in_f64 {
                            0.0
                        } else {
                            chunks_per_lane + 8.0
                        };
                        let acc_u64 = if pools_in_f64 {
                            n_px
                        } else {
                            h as f64
                                + 8.0
                                + if era2_shape {
                                    0.0
                                } else {
                                    (w % 8) as f64 * h as f64
                                }
                        };
                        let bound = k_eval * n_px * U32
                            + (acc_u32 * U32 + acc_u64 * U64) * sum_abs[i]
                            + 8.0 * U64 * l2[i].abs()
                            + f64::MIN_POSITIVE;
                        let dev = (got[i] - l2[i]).abs();
                        {
                            let e = per.entry((variant, fam)).or_insert((0.0, 0.0, 0.0));
                            if dev > e.0 {
                                *e = (dev, sum_abs[i], bound);
                            }
                        }
                        // Reported signal: dev as a fraction of its bound.
                        // A kernel change that degrades accuracy moves this
                        // long before it trips the assert.
                        let frac = dev / bound;
                        if frac > worst_overall.0 {
                            worst_overall = (
                                frac,
                                format!(
                                    "{} {}x{} ch{} [{}] {} ({}): dev {:e} = {:.1}% of bound {:e}",
                                    variant,
                                    w,
                                    h,
                                    ch,
                                    note,
                                    SLOT_NAMES[i],
                                    fam,
                                    dev,
                                    100.0 * frac,
                                    bound
                                ),
                            );
                        }
                        assert!(
                            dev <= bound,
                            "{w}x{h} ch{ch} [{note}] {variant}: slot {} ({}) deviates {:e} from \
                             the EXACT sum, above its proven bound {:e} (Σ|x| = {:e}). A bound \
                             violation is a BUG in the kernel or in the error analysis — find \
                             out which. Do NOT widen this.",
                            SLOT_NAMES[i],
                            fam,
                            dev,
                            bound,
                            sum_abs[i]
                        );
                    }
                }
            }
        }
        eprintln!(
            "ERA2-ORACLE worst deviation = {:.2}% of its proven bound\n  {}",
            100.0 * worst_overall.0,
            worst_overall.1
        );
        eprintln!("ERA2-ORACLE per-variant x family worst |dev| vs the EXACT sum:");
        eprintln!(
            "  {:<14} {:<7} {:>13} {:>13} {:>13} {:>8}",
            "variant", "family", "max|dev|", "sum|x|", "bound", "% bound"
        );
        for ((v, f), (dev, sa, bd)) in &per {
            eprintln!(
                "  {v:<14} {f:<7} {dev:>13.4e} {sa:>13.4e} {bd:>13.4e} {:>7.1}%",
                100.0 * dev / bd
            );
        }
    }

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
    /// `pyramid_plane_stride(w) == w` (multiples of 16 below 512, plus the
    /// non-bumped alignments above; see `blur::pyramid_plane_stride`).
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
    ///
    /// `training`-gated: the v1 side of the comparison is
    /// `compute_zensim_with_config`, a training-only export.
    #[cfg(feature = "training")]
    #[test]
    fn folded720_v1_basic_matches_v1_path() {
        // OPTION C (2026-08-30): the `expect_bit_exact` flag is GONE and
        // every geometry now asserts bit-exactness. 127 and 200 used to be
        // the "padded-width divergence class" — v1 pooled mirror-filled
        // phantom columns there and the fold did not. C removed the phantom
        // columns, so the class does not exist. Sanctioned re-pin (era-3,
        // docs/DATASET_HISTORY.md): a tightening, never a loosening.
        // ≥64 sizes only: `compute_zensim_with_config` (the fleet v1
        // extractor) does not reflect-pad sub-64 (see
        // `folded720_sub64_matches_padding_v1_entry` for that).
        for &(w, h) in &[
            (96usize, 64usize),
            (64, 300),
            (208, 144),
            (127, 93),
            (200, 150),
        ] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);

            // v1, the production extraction config (extended + IW = 372).
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
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
            // Every width is the bit-exact class under C. If this fires,
            // something structural regressed (band extents, H sharing, merge
            // order, or padding reintroduced) — investigate, never widen to
            // a tolerance.
            for i in 0..156 {
                assert_eq!(
                    ff[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: basic f{i} fold {:e} != v1 {:e}",
                    ff[i],
                    v1f[i]
                );
            }
        }
    }

    /// OPTION-C GATE (2026-08-30). **This test's meaning was deliberately
    /// INVERTED** when C shipped, and that is the point of the comment.
    ///
    /// It used to be `v1_padded_width_divergence_is_column_padding`, and it
    /// asserted that the buffered v1 path and the fold DISAGREE at every
    /// non-tight width — up to 81.6 % relative on a pool slot — because v1
    /// walked `simd_padded_width(width)` columns and pooled the mirror-filled
    /// extras while the fold walked the image. That divergence was the defect,
    /// not an invariant. C removes the phantom columns
    /// (`blur::pyramid_plane_stride` now returns the width), so the assertion
    /// **The bit-exactness gate for [`MeanOffsetRows::add_strip`]'s row
    /// fan-out** (fold-MT lane). Serial and parallel must agree on every
    /// `rows[y][ch]` AND on `finish()`, at heights that straddle the band
    /// size, at a width where the per-row `f64` sum has a long carry chain,
    /// and across a multi-strip walk (so the `y0` offsets differ).
    ///
    /// The property is assignment-only per (row, channel), so this cannot
    /// fail for an ordering reason — it can only fail if someone later splits
    /// the sum WITHIN a row, which is exactly the mistake to catch.
    #[test]
    fn mean_offset_row_bands_are_bit_exact() {
        for &(width, height, strip) in &[
            (64usize, 128usize, 128usize),
            (2304, 128, 128),
            (97, 40, 17),
            (7, 300, 128),
            (33, 33, 8),
        ] {
            let n = width * height;
            let src: Vec<f32> = (0..n)
                .map(|i| ((i * 2654435761usize) % 1013) as f32 / 1013.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| ((i * 40503usize + 3) % 1009) as f32 / 1009.0)
                .collect();
            let run = |parallel: bool| -> (Vec<[f64; 3]>, [f64; 3]) {
                let mut mo = MeanOffsetRows::new(width, height);
                let mut y0 = 0;
                while y0 < height {
                    let rows = strip.min(height - y0);
                    let lo = y0 * width;
                    let hi = lo + rows * width;
                    let s: [&[f32]; 3] = [&src[lo..hi], &src[lo..hi], &dst[lo..hi]];
                    let d: [&[f32]; 3] = [&dst[lo..hi], &src[lo..hi], &src[lo..hi]];
                    mo.add_strip(y0, rows, width, s, d, parallel);
                    y0 += rows;
                }
                (mo.rows.clone(), mo.finish())
            };
            let (rs, fs) = run(false);
            let (rp, fp) = run(true);
            for (y, (a, b)) in rs.iter().zip(rp.iter()).enumerate() {
                for ch in 0..3 {
                    assert_eq!(
                        a[ch].to_bits(),
                        b[ch].to_bits(),
                        "{width}x{height} strip {strip}: row {y} ch {ch} \
                         serial {} vs parallel {}",
                        a[ch],
                        b[ch]
                    );
                }
            }
            for ch in 0..3 {
                assert_eq!(
                    fs[ch].to_bits(),
                    fp[ch].to_bits(),
                    "{width}x{height} strip {strip}: finish()[{ch}]"
                );
            }
        }
    }

    /// **The bit-exactness gate for [`FoldHSource::SelfBlur`]** (fold-MT lane).
    ///
    /// A band that H-blurs its own `[b0 − overlap, b1 + overlap)` rows must
    /// produce byte-identical sums to one reading those rows out of a
    /// whole-window blur. The property it rests on is the H blur's per-row
    /// independence; this asserts the consequence directly, on every field of
    /// `V1BasicSums`, at geometries that exercise a top band (clamped `top`),
    /// a bottom band (clamped `bot`), a partial last band, and both pool
    /// modes' band counts.
    #[test]
    fn fold_self_blur_matches_precomputed_h() {
        for &(width, plane_h, strip_h) in &[
            (64usize, 128usize, 128usize),
            (97, 96, 96),
            (2304, 128, 128),
            (128, 200, 128),
            (65, 40, 40),
        ] {
            let wide_h = strip_h + 2 * HALO_P;
            let n = width * wide_h;
            let src: Vec<f32> = (0..n)
                .map(|i| ((i * 2654435761usize) % 977) as f32 / 977.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| ((i * 40503usize + 11) % 991) as f32 / 991.0)
                .collect();
            // The precomputed arm's four planes, over the WHOLE wide window.
            let mut h = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = h.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    wide_h,
                    BLUR_RADIUS,
                );
            }
            for &parallel in &[false, true] {
                let run = |h_src: FoldHSource<'_>| -> V1BasicSums {
                    let mut pool: Vec<FoldPoolScratch> = (0..V1_BANDS_PER_STRIP)
                        .map(|_| FoldPoolScratch::default())
                        .collect();
                    let mut sums = V1BasicSums::default();
                    fold_v1_basic_bands(
                        width,
                        0..strip_h.min(plane_h),
                        0,
                        HALO_P,
                        plane_h,
                        h_src,
                        [&src, &dst],
                        &mut sums,
                        Some((&mut pool[..], BandPoolWork::Full)),
                        parallel,
                    );
                    sums
                };
                let want = run(FoldHSource::Precomputed([&h[0], &h[1], &h[2], &h[3]]));
                let got = run(FoldHSource::SelfBlur);
                // `{:?}` on f64 is the shortest round-tripping form, so it
                // separates every pair of distinct finite values — the one
                // hole is NaN, whose payload it hides, so exclude NaN
                // explicitly rather than let it mask a difference.
                let wb = format!("{want:?}");
                let gb = format!("{got:?}");
                assert!(
                    !wb.contains("NaN"),
                    "{width}x{plane_h}: fixture produced NaN, which would make \
                     the Debug comparison below blind: {wb}"
                );
                assert_eq!(
                    wb, gb,
                    "{width}x{plane_h} strip {strip_h} par={parallel}: self-blur bands \
                     must equal precomputed-H bands"
                );
            }
        }
    }

    /// **The bit-exactness gate for [`fused_blur_h_ssim_banded`]** (fold-MT
    /// lane). The banded call must equal the whole-plane call BIT-FOR-BIT, at
    /// every band size that is a multiple of [`H_BLUR_ROW_GROUP`] and at
    /// heights that exercise a partial final group, a partial final band, and
    /// both at once.
    ///
    /// This is stronger than the shipped constant needs: it sweeps band sizes
    /// 8/12/16/24/32/64 so that re-tuning `H_BLUR_BAND_ROWS` on a future box
    /// cannot silently move a byte. Sizes 8, 12 and 24 are NOT multiples of
    /// [`H_BLUR_ROW_GROUP`] (16 — `v4`/`v4x` transpose 16 rows at a time;
    /// `v3`/`neon`/`wasm128`/`scalar` take 8), so they test the per-row
    /// independence itself rather than the alignment argument the shipped
    /// constant rests on. Recorded, not relied on: the shipped constant stays
    /// a multiple and the `const` assert beside it enforces that.
    #[test]
    fn phase_a_blur_bands_are_bit_exact() {
        for &(width, height) in &[
            (64usize, 148usize), // the scale-0 strip shape: 18 groups + a 4-row tail
            (2304, 40),          // wide, band-splittable
            (97, 148),           // odd width (scalar tail inside every row)
            (16, 7),             // shorter than one row group
            (33, 8),             // exactly one row group
            (127, 93),           // a golden-fixture height
            (5, 149),            // narrower than one SIMD lane
        ] {
            let n = width * height;
            let src: Vec<f32> = (0..n)
                .map(|i| ((i * 2654435761usize) % 1000) as f32 / 1000.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| ((i * 40503usize + 7) % 1000) as f32 / 1000.0)
                .collect();
            let mut want = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = want.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    height,
                    BLUR_RADIUS,
                );
            }
            for &bands in &[8usize, 12, 16, 24, 32, 64] {
                let mut got = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                // Replay the helper's banding by hand at an arbitrary band
                // size — the helper itself is pinned to one constant, and the
                // claim under test is about the BANDING, not that constant.
                let mut y0 = 0;
                while y0 < height {
                    let rows = bands.min(height - y0);
                    let lo = y0 * width;
                    let hi = lo + rows * width;
                    let (a, rest) = got.split_at_mut(1);
                    let (b, rest2) = rest.split_at_mut(1);
                    let (c, d) = rest2.split_at_mut(1);
                    crate::blur::fused_blur_h_ssim(
                        &src[lo..hi],
                        &dst[lo..hi],
                        &mut a[0][lo..hi],
                        &mut b[0][lo..hi],
                        &mut c[0][lo..hi],
                        &mut d[0][lo..hi],
                        width,
                        rows,
                        BLUR_RADIUS,
                    );
                    y0 += rows;
                }
                for (pi, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{width}x{height} bands={bands} plane {pi} idx {i}: \
                             banded {a} vs whole-plane {b}"
                        );
                    }
                }
            }
        }
    }

    /// flips: the two paths must now be BIT-IDENTICAL everywhere.
    ///
    /// This is a sanctioned re-pin — a named, deliberate semantics correction
    /// recorded in `docs/DATASET_HISTORY.md` (era-3) — NOT a tolerance
    /// widening. Nothing here is loosened: it went from "assert they differ,
    /// bounded" to "assert they are equal, exactly".
    ///
    /// The geometry set is kept verbatim from the characterisation, INCLUDING
    /// the three h = 93 cells that were the last residual under the option-A
    /// pre-pad workaround. They pass by equality here; the residual was an
    /// artifact of padding the input and does not exist under C.
    #[cfg(feature = "training")]
    #[test]
    fn v1_372_bit_exact_to_fold_at_every_width() {
        const CELLS: &[(usize, usize)] = &[
            // formerly "tight" (the only class that used to agree)
            (96, 64),
            (208, 144),
            (592, 80),
            (128, 93),
            // formerly divergent — even, non-tight
            (200, 150),
            (200, 151),
            (576, 96),
            (1152, 72),
            (100, 96),
            // formerly divergent — odd, non-tight
            (127, 64),
            (127, 96),
            (127, 128),
            (129, 96),
            (201, 96),
            (255, 96),
            (577, 80),
            // the h = 93 cells: the last residual under the pre-pad workaround
            (126, 93),
            (127, 93),
            (255, 93),
        ];
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        for &(w, h) in CELLS {
            assert_eq!(
                crate::blur::pyramid_plane_stride(w),
                w,
                "C: the pyramid stride must equal the width"
            );
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372, "{w}x{h}: v1 must emit 372");

            let mut scratch = V2Scratch::new();
            let folded = z
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(&src, w, h),
                    &RgbSlice::new(&dst, w, h),
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Full,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            let ff = folded.features();

            for i in 0..372 {
                assert_eq!(
                    ff[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: f{i} fold {:e} != buffered v1 {:e}. Under option C these \
                     paths compute the same statistic over the same pixels — a difference \
                     here means padding was reintroduced somewhere, or one walk changed. \
                     Do NOT widen this to a tolerance.",
                    ff[i],
                    v1f[i]
                );
            }
        }
    }

    /// C, pinned at its owner: no phantom columns at ANY width. A future
    /// stride/alignment experiment that reintroduces padding must fail here
    /// and read the reason, because padding the plane silently re-pollutes
    /// every v1 pool with columns that are not in the image (that was the
    /// era-2 defect, worth up to 81.6 % on a pool slot).
    #[test]
    fn pyramid_stride_has_no_phantom_columns() {
        for w in [
            1usize, 7, 8, 15, 16, 17, 63, 64, 96, 100, 127, 128, 200, 255, 256, 511, 512, 513, 576,
            592, 1024, 1152, 2304, 4096,
        ] {
            assert_eq!(
                crate::blur::pyramid_plane_stride(w),
                w,
                "pyramid_plane_stride({w}) must be {w} under option C"
            );
        }
    }

    /// **The per-profile weight-skipping gate** (feature-cost lane,
    /// 2026-08-31). [`V1PoolsMode::Peaks`] must be pure compute-skipping:
    /// every slot it DOES emit is bit-identical to what
    /// [`V1PoolsMode::Full`] emits, and the masked/IW block it skips is left
    /// at exactly `0.0` (never a partially-accumulated value, never NaN from
    /// finalising an accumulator nothing wrote).
    ///
    /// Run over the same 19 geometries as
    /// `v1_372_bit_exact_to_fold_at_every_width`, in both the `v1_only`
    /// (fold-backed SCORE) walk and the full 944 walk, serial AND parallel —
    /// the parallel arm matters because `Peaks` is the second mode allowed to
    /// take the band-local self-blur shape, so a band-boundary interaction is
    /// exactly the failure mode.
    #[cfg(feature = "training")]
    #[test]
    fn folded_peaks_mode_is_pure_compute_skipping() {
        const CELLS: &[(usize, usize)] = &[
            (96, 64),
            (208, 144),
            (592, 80),
            (128, 93),
            (200, 150),
            (200, 151),
            (576, 96),
            (1152, 72),
            (100, 96),
            (127, 64),
            (127, 96),
            (127, 128),
            (129, 96),
            (201, 96),
            (255, 96),
            (577, 80),
            (126, 93),
            (127, 93),
            (255, 93),
        ];
        for &v1_only in &[true, false] {
            for &parallel in &[false, true] {
                let z = crate::Zensim::new(crate::ZensimProfile::codec_target())
                    .with_parallel(parallel);
                let mut scratch = V2Scratch::new();
                for &(w, h) in CELLS {
                    let src = textured_image(w, h, 7);
                    let dst = quantize_distort(&src, w, h);
                    let sref = RgbSlice::new(&src, w, h);
                    let dref = RgbSlice::new(&dst, w, h);
                    let base = V2NewFeatureToggles {
                        v1_only,
                        append_block: !v1_only,
                        append2_block: !v1_only,
                        ..V2NewFeatureToggles::default()
                    };
                    let full = z
                        .compute_folded720_append_features_streaming(
                            &sref,
                            &dref,
                            V2NewFeatureToggles {
                                v1_pools: V1PoolsMode::Full,
                                ..base
                            },
                            &mut scratch,
                        )
                        .unwrap();
                    let peaks = z
                        .compute_folded720_append_features_streaming(
                            &sref,
                            &dref,
                            V2NewFeatureToggles {
                                v1_pools: V1PoolsMode::Peaks,
                                ..base
                            },
                            &mut scratch,
                        )
                        .unwrap();
                    assert_eq!(peaks.v1_pools(), V1PoolsMode::Peaks);
                    assert!(peaks.v1_pools_live());
                    assert_eq!(full.regime(), peaks.regime(), "{w}x{h}: regime moved");
                    let (ff, fp) = (full.features(), peaks.features());
                    assert_eq!(ff.len(), fp.len(), "{w}x{h}: width moved");
                    let tag = if v1_only { "v1_only" } else { "944" };
                    // Emitted slots: bit-identical.
                    for i in (0..228).chain(372..ff.len()) {
                        assert_eq!(
                            fp[i].to_bits(),
                            ff[i].to_bits(),
                            "{w}x{h} {tag} par={parallel}: slot {i} moved under Peaks ({:e} vs {:e})",
                            fp[i],
                            ff[i]
                        );
                    }
                    // Skipped block: exactly zero, and finite.
                    for i in 228..372 {
                        assert_eq!(
                            fp[i].to_bits(),
                            0.0f64.to_bits(),
                            "{w}x{h} {tag} par={parallel}: skipped slot {i} is {:e}, not +0.0",
                            fp[i]
                        );
                    }
                    // The peak block must actually carry values — a mode that
                    // emitted zeros there would pass the identity above only
                    // because Full's were zero too.
                    assert!(
                        fp[156..228].iter().any(|&v| v > 0.0),
                        "{w}x{h} {tag} par={parallel}: Peaks emitted an all-zero peak block"
                    );
                }
            }
        }
    }

    /// POOL PARITY GATE (2026-08-30, the carrier lane): with
    /// [`V2NewFeatureToggles::v1_pools`] the fold replays v1's extended
    /// strip section per v1-aligned band, so the peak / masked / IW blocks
    /// (`f156..372`) are **BIT-IDENTICAL** to the frozen v1 372 extraction
    /// at every width where the basic block is (`pyramid_plane_stride(w) ==
    /// w`); the padded-width class documents its divergence the same way
    /// (`folded720_v1_basic_matches_v1_path`), and
    /// `v1_padded_width_divergence_is_column_padding` shows that class is
    /// entirely v1's mirror-padded columns — pre-pad the input and 17 of 20
    /// geometries become bit-exact. Also gates that the toggle changes
    /// NOTHING else — basic + v2 slots bit-equal to the toggle-off fold —
    /// and that the toggle-off fold still zeroes the block.
    #[cfg(feature = "training")]
    #[test]
    fn folded720_v1_pools_match_v1_path() {
        // OPTION C: no divergence class any more — see
        // `folded720_v1_basic_matches_v1_path` for the same re-pin note.
        for &(w, h) in &[
            (96usize, 64usize),
            (64, 300),
            (208, 144),
            (127, 93),
            (200, 150),
        ] {
            let src = textured_image(w, h, 7);
            let dst = quantize_distort(&src, w, h);
            let cfg = crate::ZensimConfig {
                extended_features: true,
                compute_iw_features: true,
                allow_multithreading: false,
                ..Default::default()
            };
            let v1 = crate::compute_zensim_with_config(&src, &dst, w, h, cfg).unwrap();
            let v1f = v1.features();
            assert_eq!(v1f.len(), 372);
            let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);
            let mut scratch = V2Scratch::new();
            let off = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            let on = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Full,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            let carriers = z
                .compute_folded720_append_features_streaming(
                    &sref,
                    &dref,
                    V2NewFeatureToggles {
                        v1_pools: V1PoolsMode::Carriers,
                        ..V2NewFeatureToggles::default()
                    },
                    &mut scratch,
                )
                .unwrap();
            assert!(!off.v1_pools_live());
            assert_eq!(on.v1_pools(), V1PoolsMode::Full);
            assert_eq!(carriers.v1_pools(), V1PoolsMode::Carriers);
            assert_eq!(off.regime(), on.regime());
            let (fo, fp, fc) = (off.features(), on.features(), carriers.features());
            assert_eq!(fo.len(), fp.len());
            // Carriers: exactly the ten slots live, bit-equal to Full's
            // values (same arithmetic), every other slot unchanged from off.
            for i in 0..fo.len() {
                if V1PoolsMode::CARRIER_SLOTS.contains(&i) {
                    assert_eq!(
                        fc[i].to_bits(),
                        fp[i].to_bits(),
                        "{w}x{h}: carrier slot {i}"
                    );
                    assert!(
                        fc[i] > 0.0,
                        "{w}x{h}: carrier slot {i} is zero on a distorted pair"
                    );
                } else {
                    assert_eq!(
                        fc[i].to_bits(),
                        fo[i].to_bits(),
                        "{w}x{h}: non-carrier slot {i}"
                    );
                }
            }
            assert!(
                fo[156..372].iter().all(|&v| v == 0.0),
                "{w}x{h}: toggle-off fold must keep f156..372 at exactly 0.0"
            );
            for i in (0..156).chain(372..fo.len()) {
                assert_eq!(
                    fo[i].to_bits(),
                    fp[i].to_bits(),
                    "{w}x{h}: slot {i} changed by v1_pools ({:e} vs {:e})",
                    fo[i],
                    fp[i]
                );
            }
            // Every width is the bit-exact class under C.
            for i in 156..372 {
                assert_eq!(
                    fp[i].to_bits(),
                    v1f[i].to_bits(),
                    "{w}x{h}: pool slot {i} ({:e}) != v1 ({:e})",
                    fp[i],
                    v1f[i]
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
        // and pyramid_plane_stride(64) == 64 ⇒ bit-exact class.
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

    /// SUCCESSOR of `folded720_ref_paths_bit_identical` (C5: the
    /// prepared/moments entry forms are deleted with the reference
    /// cache): every REMAINING folded-720 entry form must agree bitwise
    /// on all 720 slots — the pair wrapper (`compute_folded720_features`,
    /// internal scratch), the explicit streaming batch form (caller
    /// scratch), and the parallel fan-out. Also keeps the old test's
    /// `view()`-exposes-the-v2-tail assertion.
    #[test]
    fn folded720_entry_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_features(&sref, &dref).unwrap();
        let mut scratch = V2Scratch::new();
        let a = z
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        assert_eq!(pair.features().len(), 720);
        for i in 0..720 {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair wrapper vs streaming batch form diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel streaming diverge at f{i}"
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

    /// SUCCESSOR of `append_ref_paths_bit_identical` (C5: the cached
    /// bs2/moments legs are deleted with the reference cache — the
    /// streamed walk computes bs2 + cross activities per kernel strip):
    /// every REMAINING append entry form must agree bitwise on ALL 924
    /// slots — pair wrapper, explicit streaming batch form, parallel
    /// fan-out, and a REUSED-scratch second run (the cross-pair leak
    /// gate the old cached legs also exercised).
    #[test]
    fn append_entry_paths_bit_identical() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let pair = z.compute_folded720_append_features(&sref, &dref).unwrap();
        assert_eq!(
            pair.features().len(),
            720 + 4 * 3 * FEATURES_PER_CHANNEL_APPEND
        );
        assert_eq!(pair.regime(), FeatureRegime::Folded720Append);
        assert_eq!(
            pair.append_features().expect("append regime").len(),
            4 * 3 * FEATURES_PER_CHANNEL_APPEND
        );

        let mut scratch = V2Scratch::new();
        let a = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        // Scratch reuse (second run on the same scratch).
        let a2 = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        for i in 0..pair.features().len() {
            assert_eq!(
                pair.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "pair wrapper vs streaming batch form diverge at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                a2.features()[i].to_bits(),
                "scratch reuse diverges at f{i}"
            );
            assert_eq!(
                pair.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel streaming diverge at f{i}"
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
                let base =
                    scale * 3 * FEATURES_PER_CHANNEL_APPEND + ch * FEATURES_PER_CHANNEL_APPEND;
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
        assert!(
            any_nonzero,
            "append block is entirely zero on a distorted pair"
        );
    }

    // ========================================================================
    // C2 streaming-walk parity gates
    // (docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md §3)
    // ========================================================================

    /// Originally the C2 gate (streamed vs the since-deleted materialized
    /// foldapp walk, bitwise). Post-C5 the pair entries ROUTE to the
    /// streaming walk, so this now gates the wrapper entries against the
    /// explicit-scratch streaming form across the full adversarial dim
    /// matrix (odd dims, sub-one-strip, exact/one-past strip multiples,
    /// 1-row strips, tall multi-production-chunk) — the geometry coverage
    /// the C2 gate provided lives on here.
    #[test]
    fn streamed_foldapp_bitwise_vs_materialized() {
        let cases = [
            (150usize, 170usize),
            (127, 93),
            (200, 150),
            (96, 517),
            // C3 adversarial rows: exact kernel-strip multiple (h = 256),
            // one past it (h = 257 — 1-row final strip), and h = 129
            // (1-row second strip at scale 0, sub-strip deeper scales).
            (96, 256),
            (96, 257),
            (72, 129),
        ];
        let z_parallel = false;
        for (w, h) in cases {
            let src = textured_image(w, h, 23);
            let dst = quantize_distort(&src, w, h);
            let sref = RgbSlice::new(&src, w, h);
            let dref = RgbSlice::new(&dst, w, h);

            // Foldapp (924).
            let mat = compute_folded720_append_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
            )
            .unwrap();
            let mut scratch = V2Scratch::new();
            let st = compute_folded720_append_streaming_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
            assert_eq!(mat.regime(), st.regime());
            assert_eq!(mat.features().len(), st.features().len());
            for i in 0..mat.features().len() {
                assert_eq!(
                    mat.features()[i].to_bits(),
                    st.features()[i].to_bits(),
                    "{w}x{h} foldapp: streamed diverges from materialized at f{i} \
                     ({} vs {})",
                    mat.features()[i],
                    st.features()[i]
                );
            }

            // Fold-only (720).
            let mat_f = compute_folded720_impl_with_toggles(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
            )
            .unwrap();
            let st_f = compute_folded720_streaming_impl(
                &sref,
                &dref,
                None,
                z_parallel,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
            assert_eq!(mat_f.features().len(), 720);
            for i in 0..720 {
                assert_eq!(
                    mat_f.features()[i].to_bits(),
                    st_f.features()[i].to_bits(),
                    "{w}x{h} fold: streamed diverges at f{i}"
                );
            }
        }
    }

    /// Sub-64 inputs reflect-pad before the streaming walk exactly like
    /// the materialized entry — outputs stay bitwise equal.
    #[test]
    fn streamed_sub64_matches_materialized() {
        let (w, h) = (40usize, 30usize);
        let src = textured_image(w, h, 9);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let mat = compute_folded720_append_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
        )
        .unwrap();
        let mut scratch = V2Scratch::new();
        let st = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        for i in 0..mat.features().len() {
            assert_eq!(
                mat.features()[i].to_bits(),
                st.features()[i].to_bits(),
                "sub-64 streamed diverges at f{i}"
            );
        }
    }

    /// Streaming walk: parallel channel fan-out is bitwise equal to the
    /// serial path (per-channel accumulators + deterministic strip order,
    /// so scheduling cannot change any sum).
    #[test]
    fn streamed_parallel_matches_serial() {
        let (w, h) = (200usize, 300usize);
        let src = textured_image(w, h, 31);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let mut scratch = V2Scratch::new();
        let serial = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            false,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        let parallel = compute_folded720_append_streaming_impl(
            &sref,
            &dref,
            None,
            true,
            V2NewFeatureToggles::default(),
            &mut scratch,
        )
        .unwrap();
        for i in 0..serial.features().len() {
            assert_eq!(
                serial.features()[i].to_bits(),
                parallel.features()[i].to_bits(),
                "streamed parallel/serial diverge at f{i}"
            );
        }
    }

    /// Scratch reuse across unrelated streamed pairs leaks nothing.
    #[test]
    fn streamed_scratch_reuse_matches_fresh() {
        let (w, h) = (150usize, 170usize);
        let src_a = textured_image(w, h, 3);
        let dst_a = quantize_distort(&src_a, w, h);
        let src_b = textured_image(w, h, 77);
        let dst_b = quantize_distort(&src_b, w, h);
        let toggles = V2NewFeatureToggles::default();

        let mut fresh = V2Scratch::new();
        let clean = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_b, w, h),
            &RgbSlice::new(&dst_b, w, h),
            None,
            false,
            toggles,
            &mut fresh,
        )
        .unwrap();

        let mut reused = V2Scratch::new();
        let _ = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_a, w, h),
            &RgbSlice::new(&dst_a, w, h),
            None,
            false,
            toggles,
            &mut reused,
        )
        .unwrap();
        let with_reuse = compute_folded720_append_streaming_impl(
            &RgbSlice::new(&src_b, w, h),
            &RgbSlice::new(&dst_b, w, h),
            None,
            false,
            toggles,
            &mut reused,
        )
        .unwrap();

        for i in 0..clean.features().len() {
            assert_eq!(
                clean.features()[i].to_bits(),
                with_reuse.features()[i].to_bits(),
                "streamed scratch reuse changed f{i}"
            );
        }
    }

    // ========================================================================
    // HDR route gates (HDR_PLAN chunk 2 — streaming PU front-end)
    // ========================================================================

    /// Minimal declared-HDR test source: absolute-linear cd/m² RGBA f32,
    /// `is_hdr() == true`, opaque.
    struct NitsImage {
        data: Vec<[f32; 4]>,
        w: usize,
        h: usize,
    }

    impl NitsImage {
        fn from_rgb_nits(rgb: &[[f32; 3]], w: usize, h: usize) -> Self {
            Self {
                data: rgb.iter().map(|p| [p[0], p[1], p[2], 1.0]).collect(),
                w,
                h,
            }
        }
    }

    impl crate::source::ImageSource for NitsImage {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> crate::source::PixelFormat {
            crate::source::PixelFormat::LinearF32Rgba
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
        }
        fn alpha_mode(&self) -> crate::source::AlphaMode {
            crate::source::AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            true
        }
    }

    /// PQ/HLG code-value test source (`Srgb16Rgba` container), HDR-flagged.
    struct Code16Image {
        data: Vec<[u16; 4]>,
        w: usize,
        h: usize,
    }

    impl Code16Image {
        fn from_unit_codes(codes: &[[f32; 3]], w: usize, h: usize) -> Self {
            Self {
                data: codes
                    .iter()
                    .map(|p| {
                        [
                            (p[0].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            (p[1].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            (p[2].clamp(0.0, 1.0) * 65535.0).round() as u16,
                            65535,
                        ]
                    })
                    .collect(),
                w,
                h,
            }
        }
    }

    impl crate::source::ImageSource for Code16Image {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> crate::source::PixelFormat {
            crate::source::PixelFormat::Srgb16Rgba
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
        }
        fn alpha_mode(&self) -> crate::source::AlphaMode {
            crate::source::AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            true
        }
    }

    /// Deterministic HDR test content spanning the given luminance range
    /// (geometric ramp + texture), in cd/m².
    fn nits_image(w: usize, h: usize, lo: f32, hi: f32, seed: u32) -> Vec<[f32; 3]> {
        let mut px = vec![[0.0f32; 3]; w * h];
        let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFFFF) as f32 / 65535.0
        };
        for y in 0..h {
            for x in 0..w {
                let t = (x + y * 3) as f32 / (w + h * 3) as f32;
                let base = lo * (hi / lo).powf(t);
                let n = 0.7 + 0.6 * next();
                px[y * w + x] = [base * n, base * (0.8 + 0.4 * next()), base * n * 0.9];
            }
        }
        px
    }

    fn distort_nits(src: &[[f32; 3]]) -> Vec<[f32; 3]> {
        src.iter()
            .map(|&[r, g, b]| {
                // Multiplicative + quantization-flavored distortion that
                // stays strictly positive at every luminance.
                [
                    (r * 1.06).max(0.001),
                    (g * 0.95 + 0.02 * r).max(0.001),
                    (b * 1.02).max(0.001),
                ]
            })
            .collect()
    }

    /// Unvalidated HDR shapes stay rejected: HDR-flagged 8-bit sRGB on the
    /// SDR entry, mixed SDR/HDR pairs, wrong format for `Linear` on the
    /// explicit entry.
    #[test]
    fn hdr_route_rejects_unvalidated_shapes() {
        let (w, h) = (96usize, 80usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // HDR-flagged source whose pixels are NOT absolute-linear f32:
        // still rejected on the plain entries (routing covers only the
        // validated Linear shape).
        struct HdrSrgb8 {
            data: Vec<[u8; 3]>,
            w: usize,
            h: usize,
        }
        impl crate::source::ImageSource for HdrSrgb8 {
            fn width(&self) -> usize {
                self.w
            }
            fn height(&self) -> usize {
                self.h
            }
            fn pixel_format(&self) -> crate::source::PixelFormat {
                crate::source::PixelFormat::Srgb8Rgb
            }
            fn row_bytes(&self, y: usize) -> &[u8] {
                bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
            }
            fn alpha_mode(&self) -> crate::source::AlphaMode {
                crate::source::AlphaMode::Opaque
            }
            fn is_hdr(&self) -> bool {
                true
            }
        }
        let bad = HdrSrgb8 {
            data: vec![[128, 128, 128]; w * h],
            w,
            h,
        };
        let e = z.compute_folded720_append_features(&bad, &bad).unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");

        // Mixed pair: SDR + HDR.
        let srgb = textured_image(w, h, 5);
        let sdr = RgbSlice::new(&srgb, w, h);
        let nits = nits_image(w, h, 1.0, 200.0, 7);
        let hdr = NitsImage::from_rgb_nits(&nits, w, h);
        let e = z.compute_folded720_append_features(&sdr, &hdr).unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");

        // Explicit entry, Linear encoding, code-value container: rejected.
        let codes = Code16Image::from_unit_codes(&nits_image(w, h, 0.0, 1.0, 9), w, h);
        let e = z
            .compute_folded720_append_features_hdr(
                &codes,
                &codes,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap_err();
        assert!(matches!(e, ZensimError::HdrInputRequiresPuPath), "{e:?}");
    }

    /// An HDR-declared absolute-linear pair routed through the PLAIN
    /// folded/append entries produces bit-identical output to the explicit
    /// HDR entry with `HdrEncoding::Linear` — the auto-route is the same
    /// walk.
    #[test]
    fn hdr_auto_route_matches_explicit_linear() {
        let (w, h) = (150usize, 170usize);
        let src = nits_image(w, h, 0.5, 800.0, 21);
        let dst = distort_nits(&src);
        let s_img = NitsImage::from_rgb_nits(&src, w, h);
        let d_img = NitsImage::from_rgb_nits(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let auto = z.compute_folded720_append_features(&s_img, &d_img).unwrap();
        let explicit = z
            .compute_folded720_append_features_hdr(
                &s_img,
                &d_img,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(auto.regime(), FeatureRegime::Folded720Append);
        for i in 0..auto.features().len() {
            assert_eq!(
                auto.features()[i].to_bits(),
                explicit.features()[i].to_bits(),
                "auto-route vs explicit diverge at f{i}"
            );
        }
    }

    /// Identity HDR pair: the error-driven append classes are EXACTLY 0 and
    /// the σ-derived ones sit in the identity ULP band — the same contract
    /// as `append_identity_pair_zeros`, on PU-domain planes.
    #[test]
    fn hdr_identity_pair_append_zeros() {
        let (w, h) = (150usize, 130usize);
        let src = nits_image(w, h, 0.05, 2000.0, 5);
        let img = NitsImage::from_rgb_nits(&src, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let r = z
            .compute_folded720_append_features_hdr(
                &img,
                &img,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
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
                        assert!((0.0..1.0).contains(&v), "grad_src_mean {v} s{scale} ch{ch}");
                    } else if EXACT_ZERO.contains(&local) {
                        assert_eq!(v, 0.0, "append {local} not zero on HDR identity: {v}");
                    } else {
                        // Identity ULP band, HDR-scaled: the σ-split noise
                        // (`(ssq − bs2) − mu2²` vs `bs2 − mu1²`) grows with
                        // the PLANE AMPLITUDE squared, and PU planes reach
                        // `pu21(10000)/PU_WHITE ≈ 2.5` vs the SDR path's
                        // ≤ ~1.0 — so the SDR test's 1e-4 band scales by
                        // ~6.5× here. Measured on this fixture: 4.3e-4 max
                        // (CONTRAST_GAIN); 2e-3 leaves ~4× headroom while
                        // still far below any real-signal value.
                        assert!(v.abs() < 2e-3, "append {local} above identity band: {v}");
                    }
                }
            }
        }
        // The whole 720 block must also be finite.
        assert!(r.features().iter().all(|v| v.is_finite()));
    }

    /// Extremes: content at the PU21 domain edges (0.005 and 10,000 cd/m²)
    /// through all three encodings — every one of the 924 slots finite and
    /// inside its documented bound class.
    #[test]
    fn hdr_extremes_bounded_all_encodings() {
        let (w, h) = (96usize, 96usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let check = |r: &ZensimV2Result, tag: &str| {
            assert_eq!(r.features().len(), 924, "{tag}");
            for (i, v) in r.features().iter().enumerate() {
                assert!(v.is_finite(), "{tag}: f{i} not finite: {v}");
                assert!(
                    (-1.0..=3.0).contains(v),
                    "{tag}: f{i} outside sanity range: {v}"
                );
            }
        };

        // Linear at the domain floor and ceiling (plus a full-range ramp).
        for (lo, hi, tag) in [
            (0.005f32, 0.05f32, "linear-black"),
            (5_000.0, 10_000.0, "linear-peak"),
            (0.005, 10_000.0, "linear-fullrange"),
        ] {
            let src = nits_image(w, h, lo, hi, 3);
            let dst = distort_nits(&src);
            let r = z
                .compute_folded720_append_features_hdr(
                    &NitsImage::from_rgb_nits(&src, w, h),
                    &NitsImage::from_rgb_nits(&dst, w, h),
                    HdrEncoding::Linear,
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            check(&r, tag);
        }

        // PQ code values 0.0..=1.0 (decodes to 0..10000 at spec peak;
        // clamps at 1000 for the display-limited variant).
        let codes = nits_image(w, h, 1e-4, 1.0, 13); // reuse generator as unit ramp
        let dstc: Vec<[f32; 3]> = codes
            .iter()
            .map(|&[r, g, b]| [(r * 0.97).min(1.0), (g * 1.02).min(1.0), b.min(1.0)])
            .collect();
        for (peak, tag) in [(10_000.0f32, "pq-spec-peak"), (1_000.0, "pq-1000")] {
            let r = z
                .compute_folded720_append_features_hdr(
                    &Code16Image::from_unit_codes(&codes, w, h),
                    &Code16Image::from_unit_codes(&dstc, w, h),
                    HdrEncoding::Pq { peak_nits: peak },
                    V2NewFeatureToggles::default(),
                    &mut scratch,
                )
                .unwrap();
            check(&r, tag);
        }

        // HLG signal values through the reference OOTF.
        let r = z
            .compute_folded720_append_features_hdr(
                &Code16Image::from_unit_codes(&codes, w, h),
                &Code16Image::from_unit_codes(&dstc, w, h),
                HdrEncoding::Hlg {
                    peak_nits: 1000.0,
                    ambient_lux: 5.0,
                },
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        check(&r, "hlg-1000");
    }

    /// BANDVIS δ-constant derivation (append2, 2026-07-27): measure the
    /// |∇| the span-2 central-difference kernel reports at one-code-step
    /// plateau edges, through the REAL front-ends, at all 4 walk scales.
    /// Prints the table (`--nocapture`) that
    /// `benchmarks/append2_bandvis_gates_2026-07-27.md` commits, and pins
    /// the shipped constants to brackets of the measurement.
    #[test]
    fn bandvis_delta_derivation_table() {
        let w = 512usize;
        let h = 128usize;

        // --- SDR: horizontal sRGB ramp posterized to 8-bit steps of k codes.
        // Adjacent plateaus differ by k/255 in sRGB code → Y-plane step
        // through srgb→XYB(cbrt).
        let sdr_step = |k: usize| -> f64 {
            let mut px = vec![[0u8; 3]; w * h];
            for y in 0..h {
                for x in 0..w {
                    // mid-gray plateaus: 118 or 118+k
                    let v = if x < w / 2 { 118u8 } else { (118 + k) as u8 };
                    px[y * w + x] = [v, v, v];
                }
            }
            let mut c0 = vec![0.0f32; w * h];
            let mut c1 = vec![0.0f32; w * h];
            let mut c2 = vec![0.0f32; w * h];
            crate::color::srgb_to_positive_xyb_planar_into(&px, &mut c0, &mut c1, &mut c2);
            // span-2 central difference at the edge = full plateau delta
            (c1[w / 2 + 1] as f64 - c1[w / 2 - 2] as f64).abs()
        };

        // --- HDR/PU: PQ 10-bit steps around given absolute luminances.
        let pu_step = |nits: f32, tenbit_steps: u32| -> f64 {
            // PQ inverse EOTF (code from nits) via bisection on pq_eotf.
            let code_of = |target: f32| -> f32 {
                let (mut lo, mut hi) = (0.0f32, 1.0f32);
                for _ in 0..50 {
                    let mid = 0.5 * (lo + hi);
                    if crate::transfer::pq_eotf(mid) < target {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                lo
            };
            let c = code_of(nits);
            let c2v = c + tenbit_steps as f32 / 1023.0;
            let l1 = crate::transfer::pq_eotf(c);
            let l2 = crate::transfer::pq_eotf(c2v);
            let px = [[l1, l1, l1], [l2, l2, l2]];
            let mut o0 = [0.0f32; 2];
            let mut o1 = [0.0f32; 2];
            let mut o2 = [0.0f32; 2];
            crate::color::linear_to_pu_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            (o1[1] as f64 - o1[0] as f64).abs()
        };

        println!("SDR (sRGB@118 gray, cbrt-Y step per k 8-bit codes):");
        for k in [1usize, 2, 4, 6] {
            println!("  k={k}: |ΔY| = {:.6}", sdr_step(k));
        }
        println!("HDR/PU (PQ 10-bit steps → PU-Y):");
        for nits in [1.0f32, 10.0, 100.0, 1000.0, 5000.0] {
            for st in [1u32, 4] {
                println!(
                    "  {nits} nits, {st} steps: |ΔPU-Y| = {:.6}",
                    pu_step(nits, st)
                );
            }
        }

        // Bracket assertions for the shipped constants (values chosen from
        // this measurement — see the gates doc):
        let one_sdr = sdr_step(1);
        assert!(
            BV_DELTA_LO_SDR as f64 > 0.3 * one_sdr && (BV_DELTA_LO_SDR as f64) < 0.7 * one_sdr,
            "δ_lo_sdr {} vs 1-step {}",
            BV_DELTA_LO_SDR,
            one_sdr
        );
        assert!(
            BV_DELTA_HI_SDR as f64 > 3.5 * one_sdr && (BV_DELTA_HI_SDR as f64) < 7.0 * one_sdr,
            "δ_hi_sdr {} vs 1-step {}",
            BV_DELTA_HI_SDR,
            one_sdr
        );
        // HL-bin anchors: the PU-route Y-plane value of a GRAY pixel at
        // 100 cd/m² (SDR white in the normalized domain) and 1000 cd/m².
        let y_of_nits = |nits: f32| -> f64 {
            let px = [[nits, nits, nits]];
            let mut o0 = [0.0f32; 1];
            let mut o1 = [0.0f32; 1];
            let mut o2 = [0.0f32; 1];
            crate::color::linear_to_pu_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            o1[0] as f64
        };
        println!("PU-route Y-plane gray values (HL anchors):");
        for nits in [80.0f32, 100.0, 203.0, 1000.0, 4000.0] {
            println!("  {nits} nits → y = {:.5}", y_of_nits(nits));
        }
        assert!(
            (HL1_Y_ANCHOR as f64 - y_of_nits(100.0)).abs() < 0.02,
            "HL1 anchor {} vs measured PU-Y(100 nits) {}",
            HL1_Y_ANCHOR,
            y_of_nits(100.0)
        );
        assert!(
            (HL2_Y_ANCHOR as f64 - y_of_nits(1000.0)).abs() < 0.02,
            "HL2 anchor {} vs measured PU-Y(1000 nits) {}",
            HL2_Y_ANCHOR,
            y_of_nits(1000.0)
        );

        let one_pu_mid = pu_step(100.0, 1);
        assert!(
            BV_DELTA_LO_PU as f64 > 0.3 * one_pu_mid && (BV_DELTA_LO_PU as f64) < 0.7 * one_pu_mid,
            "δ_lo_pu {} vs 1-step@100nits {}",
            BV_DELTA_LO_PU,
            one_pu_mid
        );
        assert!(
            BV_DELTA_HI_PU as f64 > 3.5 * one_pu_mid && (BV_DELTA_HI_PU as f64) < 7.0 * one_pu_mid,
            "δ_hi_pu {} vs 1-step@100nits {}",
            BV_DELTA_HI_PU,
            one_pu_mid
        );
    }

    // ========================================================================
    // append2 gates (BANDVIS + conditioner + HL bins)
    // ========================================================================

    /// Identity + layout + first-924 bit-stability + entry parity at 944.
    #[test]
    fn append2_layout_identity_and_first924_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let a924 = z.compute_folded720_append_features(&sref, &dref).unwrap();
        let a944 = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        assert_eq!(a944.regime(), FeatureRegime::Folded720Append2);
        assert_eq!(a944.features().len(), 944);
        assert_eq!(a944.append2_features().unwrap().len(), 20);
        assert_eq!(a944.append_features().unwrap().len(), 204);
        // Turning append2 on must not move a bit of the first 924.
        for i in 0..924 {
            assert_eq!(
                a924.features()[i].to_bits(),
                a944.features()[i].to_bits(),
                "append2 toggled on moved f{i}"
            );
        }
        // Parallel + scratch parity at 944.
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp.compute_folded720_append2_features(&sref, &dref).unwrap();
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel diverge at f{i}"
            );
        }
        // All 20 new slots bounded [0,1].
        for (i, v) in a944.append2_features().unwrap().iter().enumerate() {
            assert!((0.0..=1.0).contains(v) && v.is_finite(), "app2[{i}] = {v}");
        }

        // Identity pair: BANDVIS gain/loss EXACTLY 0; HL bins EXACTLY 0
        // on the SDR route; conditioner in (0,1).
        let idr = z.compute_folded720_append2_features(&sref, &sref).unwrap();
        let app2 = idr.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(app2[b + idx_append2::BANDVIS_GAIN], 0.0, "s{scale} gain");
            assert_eq!(app2[b + idx_append2::BANDVIS_LOSS], 0.0, "s{scale} loss");
            assert_eq!(app2[b + idx_append2::HL_BIN1], 0.0, "s{scale} hl1 (SDR)");
            assert_eq!(app2[b + idx_append2::HL_BIN2], 0.0, "s{scale} hl2 (SDR)");
            let lm = app2[b + idx_append2::LUMA_MEAN_REF];
            assert!(lm > 0.0 && lm < 1.0, "s{scale} luma_mean {lm}");
        }
    }

    // --- Shared BANDVIS fixtures (gates doc V3). Extracted 2026-08-02 so
    // the P1.5 dst-activity adjudication tests measure the IDENTICAL
    // pixel content as the OFF-math characterization pins (formulas
    // verbatim from the original inline builders; the behavior test's
    // value pins + the gates-doc printed numbers gate the motion).
    // `benchmarks/bandvis_dst_activity_2026-08-02.md`. ---

    /// Smooth diagonal gradient ramp (sRGB codes 32..224).
    fn bv_ramp(w: usize, h: usize) -> Vec<[u8; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let v = v as u8;
                [v, v, v]
            })
            .collect()
    }

    /// Mid-tread posterize to `bits` (the gates-doc ladder operator).
    fn bv_posterize(px: &[[u8; 3]], bits: u32) -> Vec<[u8; 3]> {
        let mask = !((1u16 << (8 - bits)) - 1) as u8;
        let half = ((1u16 << (8 - bits)) / 2) as u8;
        px.iter()
            .map(|&[r, g, b]| [(r & mask) | half, (g & mask) | half, (b & mask) | half])
            .collect()
    }

    /// 4-bit ordered-Bayer (4×4) dither of the diagonal ramp.
    fn bv_bayer_dither(w: usize, h: usize) -> Vec<[u8; 3]> {
        let bayer = [
            [0u8, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ];
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let t = (bayer[y % 4][x % 4] as f32 + 0.5) / 16.0;
                let step = 16.0; // 4-bit quantization step
                let q = ((v / step + t - 0.5).floor() * step + step / 2.0).clamp(0.0, 255.0) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// 4-bit hash-noise (TPDF-class) dither of the diagonal ramp.
    fn bv_noise_dither(w: usize, h: usize) -> Vec<[u8; 3]> {
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let v = 32.0 + 192.0 * (x + y) as f32 / (w + h - 2) as f32;
                let mut hsh = (x as u32)
                    .wrapping_mul(0x9E37_79B9)
                    .wrapping_add((y as u32).wrapping_mul(0x85EB_CA6B));
                hsh ^= hsh >> 15;
                hsh = hsh.wrapping_mul(0x2C1B_3C6D);
                hsh ^= hsh >> 13;
                let t = ((hsh & 0xFFFF) as f32 + 0.5) / 65536.0;
                let step = 16.0;
                let q = ((v / step + t - 0.5).floor() * step + step / 2.0).clamp(0.0, 255.0) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// ±16-code hash-textured version of `ramp` (the b2 source fixture).
    fn bv_textured_ramp(ramp: &[[u8; 3]]) -> Vec<[u8; 3]> {
        ramp.iter()
            .enumerate()
            .map(|(i, &[v, _, _])| {
                let mut hsh = (i as u32).wrapping_mul(0x27D4_EB2F);
                hsh ^= hsh >> 15;
                let n = ((hsh & 0x1F) as i16) - 16; // ±16 codes texture
                let t = (v as i16 + n).clamp(0, 255) as u8;
                [t, t, t]
            })
            .collect()
    }

    /// ±6-code 8-px DC-lattice overlay on `ramp` (the V3(c) fixture).
    fn bv_blocky(ramp: &[[u8; 3]], w: usize) -> Vec<[u8; 3]> {
        ramp.iter()
            .enumerate()
            .map(|(i, &[v, _, _])| {
                let (x, y) = (i % w, i / w);
                let dc = if ((x / 8) + (y / 8)) % 2 == 0 { 6 } else { -6 };
                let q = (v as i16 + dc).clamp(0, 255) as u8;
                [q, q, q]
            })
            .collect()
    }

    /// V3(a,b,c,e) behavioral: posterize-ladder monotonicity, dither
    /// masking, blocky-not-banded separation, debanding credit.
    #[test]
    fn append2_bandvis_behavior() {
        let (w, h) = (256usize, 256usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        // Smooth diagonal gradient ramp (sRGB codes 32..224).
        let ramp: Vec<[u8; 3]> = bv_ramp(w, h);
        let posterize = bv_posterize;
        let gain_at = |dst: &[[u8; 3]]| -> Vec<f64> {
            let r = z
                .compute_folded720_append2_features(
                    &RgbSlice::new(&ramp, w, h),
                    &RgbSlice::new(dst, w, h),
                )
                .unwrap();
            let app2 = r.append2_features().unwrap().to_vec();
            (0..4)
                .map(|s| app2[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                .collect()
        };

        // (a) Ladder at the COARSE scales the detector serves (see
        // `idx_append2::BANDVIS_GAIN`'s measured-mechanics note): within
        // the visibility band (6→5→4 bit = 4/8/16-code steps), fewer
        // levels ⇒ strictly more banding at scales 2 and 3; 3-bit
        // (32-code steps) attenuates vs 4-bit — the CAMBI-faithful upper
        // contrast cutoff (giant steps are edges, not banding).
        let g6 = gain_at(&posterize(&ramp, 6));
        let g5 = gain_at(&posterize(&ramp, 5));
        let g4 = gain_at(&posterize(&ramp, 4));
        let g3 = gain_at(&posterize(&ramp, 3));
        println!(
            "ladder: 6bit {g6:?}\n        5bit {g5:?}\n        4bit {g4:?}\n        3bit {g3:?}"
        );
        // MEASURED STRUCTURE (design-true invariants; full matrix in the
        // gates doc): the response over the posterize ladder is UNIMODAL —
        // it rises through the visibility band (step sizes approaching the
        // band optimum √(δ_lo·δ_hi) ≈ 1.6 8-bit codes × edge density) and
        // rolls off past the contrast cap (steps ≫ δ_hi are edges, not
        // banding — CAMBI's own cap). The per-scale slots form the
        // resonance curve consumers read.
        let g7 = gain_at(&posterize(&ramp, 7));
        let peak = |g: &Vec<f64>| {
            g.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, &v)| (i, v))
                .unwrap()
        };
        let (s7, p7) = peak(&g7);
        let (s6, p6) = peak(&g6);
        let (s5, p5) = peak(&g5);
        let (s4, p4) = peak(&g4);
        let (s3b, p3) = peak(&g3);
        println!(
            "resonance: 7b s{s7}@{p7:.4} 6b s{s6}@{p6:.4} 5b s{s5}@{p5:.4} 4b s{s4}@{p4:.4} 3b s{s3b}@{p3:.4}"
        );
        // 1. Every posterize rung fires clearly (curvature SNR: the
        //    weakest rung is still >0.05 on this fixture).
        for (tag, p) in [("7b", p7), ("6b", p6), ("5b", p5), ("4b", p4), ("3b", p3)] {
            assert!(p > 0.05, "{tag} rung must fire: {p}");
        }
        // 2. Unimodal: the ladder's interior peak exceeds both endpoints.
        let pmax = [p7, p6, p5, p4, p3].into_iter().fold(0.0f64, f64::max);
        assert!(
            pmax > p7 && pmax > p3,
            "ladder should peak in the interior: {p7} .. {pmax} .. {p3}"
        );
        // 3. Cap rolloff: monotone decline past the band (5b → 4b → 3b).
        assert!(
            p5 > p4 && p4 > p3,
            "cap rolloff not monotone: {p5} {p4} {p3}"
        );

        // (b) Dither masking: ordered-dither the SAME 4-bit posterize —
        // gain must drop substantially.
        let dithered: Vec<[u8; 3]> = bv_bayer_dither(w, h);
        // (b1) NOISE dither (the industry deband practice — TPDF-class
        // random dither): the quantization residual decorrelates and
        // averages out sub-band at coarse scales ⇒ GAIN drops
        // substantially vs plain posterize.
        let noise_dithered: Vec<[u8; 3]> = bv_noise_dither(w, h);
        let gn = gain_at(&noise_dithered);
        let ratio_noise = gn[3] / g4[3].max(1e-12);
        println!(
            "noise-dither masking [scale3]: undithered {:.5} dithered {:.5} ratio {:.3}",
            g4[3], gn[3], ratio_noise
        );
        // V3(b) VERDICT — MISS, recorded (gates doc): dst-side dither
        // (noise AND ordered) FIRES rather than masking. Structural: any
        // ~1-code quantization residual has dense mid-band curvature, and
        // the flatness mask is REF-side (the specified design); local
        // dst-texture masking needs a dst-activity plane (~+5% CPU,
        // outside this wave's ≤+2% budget — REMAINDERS). Pin the measured
        // behavior so a silent change trips a test:
        assert!(
            ratio_noise > 1.0,
            "characterization pin: dst noise-dither currently FIRES (ratio {ratio_noise}); \
             if this improved, the gates doc + REMAINDERS need updating"
        );

        // (b2) Source-texture masking (the activity mask's ref-side
        // mechanism): the SAME posterization applied to a noise-textured
        // source fires much less than on the clean ramp.
        let textured_src: Vec<[u8; 3]> = bv_textured_ramp(&ramp);
        let r_tex = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&textured_src, w, h),
                &RgbSlice::new(&bv_posterize(&textured_src, 4), w, h),
            )
            .unwrap();
        let a2t = r_tex.append2_features().unwrap();
        let g_tex3 = a2t[3 * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN];
        println!(
            "src-texture masking [scale3]: clean-src {:.5} textured-src {:.5} ratio {:.3}",
            g4[3],
            g_tex3,
            g_tex3 / g4[3].max(1e-12)
        );
        assert!(
            g_tex3 < 0.5 * g4[3],
            "source texture should mask banding: {g_tex3} vs {}",
            g4[3]
        );

        // RECORDED characteristic (not asserted): ordered 4×4 Bayer at
        // this 16-code step size aliases a ~2-code low-frequency pattern
        // into the coarse scales — in-band curvature that GAIN reports
        // (the pattern IS visible at this amplitude; noise dither above
        // is the practice BANDVIS should and does credit). Numbers for
        // the gates doc:
        let gd = gain_at(&dithered);
        println!(
            "ordered-Bayer dither [per scale]: {:?} (vs undithered s3 {:.5})",
            gd, g4[3]
        );

        // (c) Blocky-but-not-banded: 8px-lattice DC shifts fire blockiness,
        // not bandvis.
        let blocky: Vec<[u8; 3]> = bv_blocky(&ramp, w);
        let rb = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&ramp, w, h),
                &RgbSlice::new(&blocky, w, h),
            )
            .unwrap();
        let app2b = rb.append2_features().unwrap();
        let gain_blocky_max = (0..4)
            .map(|s| app2b[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .fold(0.0f64, f64::max);
        let blockiness = rb.view().blockiness(0, 1);
        println!(
            "blocky fixture: max bandvis_gain {gain_blocky_max:.5} blockiness(Y,s0) {blockiness:.5}"
        );
        // V3(c) VERDICT — MISS, recorded (gates doc + REMAINDERS): a
        // dense DC-shift lattice at its resonant scale fires BANDVIS
        // strongly (the flatness mask is REF-side; the dst's own blocky
        // texture does not self-mask — same root cause as V3(b)).
        // BLOCKINESS remains the lattice-SPECIFIC discriminator the head
        // pairs it with. Pinned invariants: blockiness fires decisively,
        // and the bandvis cross-response is nonzero (characterization pin
        // — if either moves, re-verdict the gates doc).
        assert!(
            blockiness > 0.1,
            "blocky fixture must fire blockiness: {blockiness}"
        );
        assert!(
            gain_blocky_max > 0.1,
            "characterization pin: dense DC lattices currently cross-fire bandvis \
             ({gain_blocky_max}); if this improved, update the gates doc"
        );

        // (e) Debanding credit: banded SOURCE, ideally-debanded dst (the
        // smooth ramp itself) — LOSS fires at the banding's resonant
        // scale, GAIN doesn't.
        let r = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&posterize(&ramp, 4), w, h),
                &RgbSlice::new(&ramp, w, h),
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        let b3 = 3 * APPEND2_PER_SCALE;
        let (gain3, loss3) = (
            app2[b3 + idx_append2::BANDVIS_GAIN],
            app2[b3 + idx_append2::BANDVIS_LOSS],
        );
        println!("debanding (ideal) [scale3]: gain {gain3:.5} loss {loss3:.5}");
        // V3(e): the LOSS credit direction holds (loss > gain). A stronger
        // loss ≫ gain is unattainable in u8 fixtures: an 8-bit "smooth"
        // ramp is itself a 1-code micro-staircase whose post-downscale
        // curvature residual (~0.3 code) sits at the band's lower edge —
        // the gain term correctly reports it (gates doc).
        assert!(
            loss3 > gain3,
            "debanding must credit LOSS over GAIN at scale 3: g {gain3} l {loss3}"
        );
        assert!(loss3 > 0.1, "debanding LOSS should fire strongly: {loss3}");
        // RECORDED characteristic (not asserted): a dithered "deband" at
        // this 4-bit magnitude leaves ~2-code residual pattern after
        // downscale — in-band dense texture that GAIN reads as new
        // stepping at deep scales (arguably correct: ordered dither at
        // this amplitude is visible mottle). Numbers for the gates doc:
        let rd = z
            .compute_folded720_append2_features(
                &RgbSlice::new(&posterize(&ramp, 4), w, h),
                &RgbSlice::new(&dithered, w, h),
            )
            .unwrap();
        let a2 = rd.append2_features().unwrap();
        println!(
            "deband-by-dither [scale3]: gain {:.5} loss {:.5}",
            a2[b3 + idx_append2::BANDVIS_GAIN],
            a2[b3 + idx_append2::BANDVIS_LOSS]
        );
    }

    /// P1.5 dst-activity toggle — structural gates (pre-registered F6/F7/
    /// F8 in `benchmarks/bandvis_dst_activity_2026-08-02.md`, tightened
    /// to the SHIPPED combine): toggle ON moves ONLY the four BANDVIS
    /// GAIN slots of the 944 vector — every other slot INCLUDING
    /// BANDVIS_LOSS is bit-identical (the shipped combine keeps LOSS =
    /// the OFF math exactly) — identity stays exactly 0, serial ≡
    /// parallel, and the toggle is live (GAIN lanes DO move on a
    /// quantized pair).
    #[test]
    fn append2_dst_activity_lanes_only_identity_and_parallel() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let t_off = V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            ..Default::default()
        };
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..t_off
        };
        let off = z
            .compute_folded720_append_features_streaming(&sref, &dref, t_off, &mut scratch)
            .unwrap();
        let on = z
            .compute_folded720_append_features_streaming(&sref, &dref, t_on, &mut scratch)
            .unwrap();
        assert_eq!(off.features().len(), 944);
        assert_eq!(on.features().len(), 944);
        assert_eq!(on.regime(), FeatureRegime::Folded720Append2);
        // F7 (tightened): only the BANDVIS GAIN lanes may differ — LOSS
        // is bit-stable by the shipped combine's construction.
        let mut lanes_moved = 0usize;
        for i in 0..944 {
            let gain_lane = i >= 924 && (i - 924) % APPEND2_PER_SCALE == idx_append2::BANDVIS_GAIN;
            if gain_lane {
                lanes_moved += (off.features()[i].to_bits() != on.features()[i].to_bits()) as usize;
            } else {
                assert_eq!(
                    off.features()[i].to_bits(),
                    on.features()[i].to_bits(),
                    "append2_dst_activity moved a non-GAIN slot f{i}"
                );
            }
        }
        assert!(
            lanes_moved > 0,
            "toggle must be LIVE on a quantized pair (no GAIN lane moved)"
        );
        // F6: identity pair stays exactly 0 with the toggle ON (activity
        // twins are bitwise-identical on identical planes ⇒ FR pair 0).
        let idr = z
            .compute_folded720_append_features_streaming(&sref, &sref, t_on, &mut scratch)
            .unwrap();
        let app2 = idr.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(app2[b + idx_append2::BANDVIS_GAIN], 0.0, "s{scale} gain");
            assert_eq!(app2[b + idx_append2::BANDVIS_LOSS], 0.0, "s{scale} loss");
        }
        // F8: serial ≡ parallel at 944 with the toggle ON.
        #[cfg(feature = "threads")]
        {
            let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
            let mut scratch_p = V2Scratch::new();
            let on_p = zp
                .compute_folded720_append_features_streaming(&sref, &dref, t_on, &mut scratch_p)
                .unwrap();
            for i in 0..944 {
                assert_eq!(
                    on.features()[i].to_bits(),
                    on_p.features()[i].to_bits(),
                    "serial vs parallel diverge at f{i} (dst-activity ON)"
                );
            }
        }
    }

    /// P1.5 dst-activity adjudication matrix (F1–F5 of
    /// `benchmarks/bandvis_dst_activity_2026-08-02.md`): OFF and ON arms
    /// over the IDENTICAL gates-doc fixtures (shared `bv_*` builders).
    /// Prints the full per-scale table the doc records. Asserted here:
    /// the design-contract invariants (real banding keeps firing with the
    /// self-mask ON; identity/direction gates) — the adjudication gates
    /// (F2 ratio, F3 reduction, …) are recorded in the doc with the
    /// measured outcome, and the adjudicated pins were added below after
    /// the doc's results section landed.
    #[test]
    fn append2_dst_activity_behavior_matrix() {
        let (w, h) = (256usize, 256usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let t_off = V2NewFeatureToggles {
            append_block: true,
            append2_block: true,
            ..Default::default()
        };
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..t_off
        };
        let mut gl = |src: &[[u8; 3]], dst: &[[u8; 3]], on: bool| -> (Vec<f64>, Vec<f64>) {
            let r = z
                .compute_folded720_append_features_streaming(
                    &RgbSlice::new(src, w, h),
                    &RgbSlice::new(dst, w, h),
                    if on { t_on } else { t_off },
                    &mut scratch,
                )
                .unwrap();
            let a = r.append2_features().unwrap();
            (
                (0..4)
                    .map(|s| a[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                    .collect(),
                (0..4)
                    .map(|s| a[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_LOSS])
                    .collect(),
            )
        };

        let ramp = bv_ramp(w, h);
        let p = |bits| bv_posterize(&ramp, bits);

        // F1: the posterize ladder, both arms (asserts AFTER the full
        // matrix prints, so a single miss still shows every number).
        let mut peaks_on = Vec::new();
        for bits in [7u32, 6, 5, 4, 3] {
            let (g_off, _) = gl(&ramp, &p(bits), false);
            let (g_on, _) = gl(&ramp, &p(bits), true);
            println!("F1 ladder {bits}b OFF {g_off:?}\n              ON  {g_on:?}");
            let pk = g_on.iter().cloned().fold(0.0f64, f64::max);
            peaks_on.push(pk);
        }

        // F2: noise-dither ratio @s3, both arms.
        let nd = bv_noise_dither(w, h);
        let (g4_off, _) = gl(&ramp, &p(4), false);
        let (g4_on, _) = gl(&ramp, &p(4), true);
        let (gn_off, ln_off) = gl(&ramp, &nd, false);
        let (gn_on, ln_on) = gl(&ramp, &nd, true);
        let ratio_off = gn_off[3] / g4_off[3].max(1e-12);
        let ratio_on = gn_on[3] / g4_on[3].max(1e-12);
        println!(
            "F2 noise-dither s3: OFF undith {:.5} dith {:.5} ratio {:.3} | ON undith {:.5} dith {:.5} ratio {:.3}",
            g4_off[3], gn_off[3], ratio_off, g4_on[3], gn_on[3], ratio_on
        );
        println!(
            "F2 noise-dither LOSS s0..s3: OFF {ln_off:?} ON {ln_on:?} (dither-as-partial-deband credit direction)"
        );

        // F2b: ordered-Bayer per scale, both arms.
        let bd = bv_bayer_dither(w, h);
        let (gb_off, _) = gl(&ramp, &bd, false);
        let (gb_on, _) = gl(&ramp, &bd, true);
        println!("F2b bayer GAIN OFF {gb_off:?}\n               ON  {gb_on:?}");

        // F3: DC lattice, both arms.
        let blk = bv_blocky(&ramp, w);
        let (gk_off, _) = gl(&ramp, &blk, false);
        let (gk_on, _) = gl(&ramp, &blk, true);
        let mx_off = gk_off.iter().cloned().fold(0.0f64, f64::max);
        let mx_on = gk_on.iter().cloned().fold(0.0f64, f64::max);
        println!(
            "F3 lattice GAIN OFF {gk_off:?} max {mx_off:.4}\n                ON  {gk_on:?} max {mx_on:.4}"
        );

        // F4: source-texture masking, both arms.
        let tex = bv_textured_ramp(&ramp);
        let texp = bv_posterize(&tex, 4);
        let (gt_off, _) = gl(&tex, &texp, false);
        let (gt_on, _) = gl(&tex, &texp, true);
        println!(
            "F4 src-texture s3: OFF {:.5} (ratio {:.3}) ON {:.5} (ratio {:.3})",
            gt_off[3],
            gt_off[3] / g4_off[3].max(1e-12),
            gt_on[3],
            gt_on[3] / g4_on[3].max(1e-12)
        );
        assert!(
            gt_on[3] < 0.5 * g4_on[3],
            "F4: source texture must keep masking with dst self-mask ON: {} vs {}",
            gt_on[3],
            g4_on[3]
        );

        // F5: deband credit, both arms.
        let (gd_off, ld_off) = gl(&p(4), &ramp, false);
        let (gd_on, ld_on) = gl(&p(4), &ramp, true);
        println!(
            "F5 deband s3: OFF gain {:.5} loss {:.5} | ON gain {:.5} loss {:.5}",
            gd_off[3], ld_off[3], gd_on[3], ld_on[3]
        );

        // --- Adjudicated pins (shipped combine; full registered-gate
        // PASS/MISS ledger in the doc) ---
        // F1: in-band rungs keep firing; the 3b cap-tail rung measured
        // 0.0347 — UNDER the registered 0.05 bar (recorded MISS: 32-code
        // steps carry strong self-activity AND sit in the "edges not
        // banding" cap regime) — pinned at its measured level so a silent
        // further collapse trips.
        for (i, bits) in [7u32, 6, 5, 4].iter().enumerate() {
            assert!(
                peaks_on[i] > 0.05,
                "F1: {bits}b rung must keep firing with the dst weight ON: {}",
                peaks_on[i]
            );
        }
        assert!(
            peaks_on[4] > 0.02,
            "F1: 3b cap-tail rung collapsed below its measured level: {}",
            peaks_on[4]
        );
        let pmax = peaks_on.iter().cloned().fold(0.0f64, f64::max);
        assert!(
            pmax > peaks_on[3] && pmax > peaks_on[4],
            "F1: peak must sit in the visibility band, not the cap tail (ON): {peaks_on:?}"
        );
        assert!(
            peaks_on[2] > peaks_on[3] && peaks_on[3] > peaks_on[4],
            "F1: cap rolloff not monotone (ON): {peaks_on:?}"
        );
        // Shipped-combine construction: LOSS is BIT-identical to OFF on
        // every fixture (the toggle re-weights GAIN only).
        for s in 0..4 {
            assert_eq!(
                ln_on[s].to_bits(),
                ln_off[s].to_bits(),
                "LOSS moved under the toggle (dither fixture, s{s})"
            );
            assert_eq!(
                ld_on[s].to_bits(),
                ld_off[s].to_bits(),
                "LOSS moved under the toggle (deband fixture, s{s})"
            );
        }
        // F3 (adjudicated PASS): the pooling weight suppresses the
        // lattice cross-fire decisively.
        assert!(
            mx_on < 0.5 * mx_off,
            "F3: lattice cross-fire must stay < 0.5x OFF: {mx_on} vs {mx_off}"
        );
        // F5 (shipped combine): LOSS keeps the OFF value, GAIN takes the
        // arm-2 weight — direction restored with a margin ABOVE the OFF
        // margin.
        assert!(
            ld_on[3] > gd_on[3],
            "F5: debanding must credit LOSS over GAIN at s3 (ON): g {} l {}",
            gd_on[3],
            ld_on[3]
        );
        assert!(
            ld_on[3] > 0.1,
            "F5: debanding LOSS should fire strongly (ON): {}",
            ld_on[3]
        );
        assert!(
            (ld_on[3] - gd_on[3]) > (ld_off[3] - gd_off[3]),
            "F5: the shipped combine's deband margin must exceed the OFF margin"
        );
    }

    /// V4-class: HL bins fire on >SDR-white highlight error, exactly 0 on
    /// SDR-range HDR content; BANDVIS runs the PU constants on the HDR
    /// route (posterized 10-bit-ish ramp fires).
    #[test]
    fn append2_hdr_hl_bins_and_pu_bandvis() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // Highlight-error pair: ramp 50..3000 nits, error only >300 nits.
        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 50.0 * (3000.0f32 / 50.0).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let dst: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, g, b]| {
                if r > 300.0 {
                    [r * 1.15, g * 1.15, b * 1.15]
                } else {
                    [r, g, b]
                }
            })
            .collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&dst, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        println!(
            "HL highlight-error: hl1 {:.5} hl2 {:.5} dark-ctl mse {:.5}",
            app2[idx_append2::HL_BIN1],
            app2[idx_append2::HL_BIN2],
            r.append_features().unwrap()[idx_append::LUM_DARK_ERR]
        );
        assert!(app2[idx_append2::HL_BIN1] > 1e-3, "HL1 should fire");
        assert!(app2[idx_append2::HL_BIN2] > 1e-3, "HL2 should fire");
        // Slot-forensics F-4 coverage (benchmarks/extractor_slot_forensics_
        // 2026-08-05.md): the 8 HL slots (f927/932/937/942 + f928/933/938/943)
        // are the ONLY route-gated members of the audit's 39 never-populated
        // set — prove every one of them populates on the HDR route, not just
        // scale 0 (the highlight region survives every downscale, so all 4
        // per-scale kernels must accumulate mass).
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert!(
                app2[b + idx_append2::HL_BIN1] > 1e-3,
                "HL1 must fire at scale {scale} on the HDR route (f{})",
                924 + b + idx_append2::HL_BIN1
            );
            assert!(
                app2[b + idx_append2::HL_BIN2] > 1e-3,
                "HL2 must fire at scale {scale} on the HDR route (f{})",
                924 + b + idx_append2::HL_BIN2
            );
        }

        // SDR-range HDR content (≤80 nits): bins exactly 0.
        let sdr_ramp: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                let v = (r / 3000.0 * 80.0).max(0.05);
                [v, v, v]
            })
            .collect();
        let sdr_dst: Vec<[f32; 3]> = sdr_ramp.iter().map(|&[r, g, b]| [r * 1.1, g, b]).collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&sdr_ramp, w, h),
                &NitsImage::from_rgb_nits(&sdr_dst, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2 = r.append2_features().unwrap();
        for scale in 0..4 {
            let b = scale * APPEND2_PER_SCALE;
            assert_eq!(
                app2[b + idx_append2::HL_BIN1],
                0.0,
                "hl1 on ≤80-nit content"
            );
            assert_eq!(
                app2[b + idx_append2::HL_BIN2],
                0.0,
                "hl2 on ≤80-nit content"
            );
        }

        // PU-route BANDVIS: posterize the PU ramp (quantize nits through
        // a PQ-code-like grid) — gain fires with the PU δ constants.
        let poster: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                // quantize log-luminance to 24 plateaus
                let lg = (r / 50.0).ln() / (3000.0f32 / 50.0).ln();
                let q = (lg * 24.0).floor() / 24.0;
                let v = 50.0 * (3000.0f32 / 50.0).powf(q);
                [v, v, v]
            })
            .collect();
        let r = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let app2r = r.append2_features().unwrap();
        let g_coarse = (2..4)
            .map(|s| app2r[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .fold(0.0f64, f64::max);
        println!(
            "PU-route posterize gain per scale: {:?}",
            (0..4)
                .map(|s| app2r[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
                .collect::<Vec<_>>()
        );
        assert!(
            g_coarse > 1e-4,
            "PU-domain BANDVIS should fire at coarse scales on posterized HDR ramp"
        );
    }

    /// P1.5 dst-activity toggle on the DECLARED-HDR route (pre-registered
    /// F9): toggle ON moves only the BANDVIS lanes (HL bins, conditioner,
    /// first-924 all bit-stable), and PU-domain BANDVIS still fires on
    /// the posterized HDR ramp with the self-mask ON.
    #[test]
    fn append2_dst_activity_hdr_route_lanes_and_fires() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        // The V4 PU fixture: log-luminance ramp 50..3000 nits, posterized
        // to 24 plateaus (identical construction to the test above).
        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 50.0 * (3000.0f32 / 50.0).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let poster: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, _, _]| {
                let lg = (r / 50.0).ln() / (3000.0f32 / 50.0).ln();
                let q = (lg * 24.0).floor() / 24.0;
                let v = 50.0 * (3000.0f32 / 50.0).powf(q);
                [v, v, v]
            })
            .collect();
        let t_on = V2NewFeatureToggles {
            append2_dst_activity: true,
            ..V2NewFeatureToggles::default()
        };
        let off = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let on = z
            .compute_folded720_append2_features_hdr(
                &NitsImage::from_rgb_nits(&ramp, w, h),
                &NitsImage::from_rgb_nits(&poster, w, h),
                HdrEncoding::Linear,
                t_on,
                &mut scratch,
            )
            .unwrap();
        let mut lanes_moved = 0usize;
        for i in 0..944 {
            let gain_lane = i >= 924 && (i - 924) % APPEND2_PER_SCALE == idx_append2::BANDVIS_GAIN;
            if gain_lane {
                lanes_moved += (off.features()[i].to_bits() != on.features()[i].to_bits()) as usize;
            } else {
                assert_eq!(
                    off.features()[i].to_bits(),
                    on.features()[i].to_bits(),
                    "HDR route: append2_dst_activity moved a non-GAIN slot f{i}"
                );
            }
        }
        assert!(lanes_moved > 0, "HDR route: toggle must be live");
        let app2 = on.append2_features().unwrap();
        let g_on: Vec<f64> = (0..4)
            .map(|s| app2[s * APPEND2_PER_SCALE + idx_append2::BANDVIS_GAIN])
            .collect();
        println!("F9 PU-route posterize GAIN (dst-activity ON): {g_on:?}");
        let g_coarse_on = g_on[2..4].iter().cloned().fold(0.0f64, f64::max);
        assert!(
            g_coarse_on > 1e-4,
            "F9: PU-domain BANDVIS must still fire with the dst self-mask ON"
        );
    }

    // ========================================================================
    // CSFW gates (chunk-3 tier-1: luminance-weighted GLOBAL_* twins)
    // ========================================================================

    /// Identity + layout + first-944 bit-stability + SDR entry parity at
    /// 956 (the append2 gate shape, one block up).
    #[test]
    fn csfw_layout_identity_and_first944_bit_stable() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        let a944 = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        let a956 = z.compute_folded720_csfw_features(&sref, &dref).unwrap();
        assert_eq!(a956.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(a956.features().len(), 956);
        assert_eq!(a956.csfw_features().unwrap().len(), 12);
        assert_eq!(a956.append2_features().unwrap().len(), 20);
        assert_eq!(a956.append_features().unwrap().len(), 204);
        // The windowed accessors must agree with the 944 result's views.
        assert_eq!(
            a956.append2_features().unwrap(),
            a944.append2_features().unwrap()
        );
        assert_eq!(
            a956.append_features().unwrap(),
            a944.append_features().unwrap()
        );
        // Turning CSFW on must not move a bit of the first 944.
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                a956.features()[i].to_bits(),
                "csfw toggled on moved f{i}"
            );
        }
        // Parallel + scratch parity at 956.
        let zp = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(true);
        let b = zp.compute_folded720_csfw_features(&sref, &dref).unwrap();
        for i in 0..956 {
            assert_eq!(
                a956.features()[i].to_bits(),
                b.features()[i].to_bits(),
                "serial vs parallel diverge at f{i}"
            );
        }
        // SDR entry parity: the pair entry vs BOTH toggle-carrying
        // streaming batch forms — byte-identical 956.
        let mut scratch = V2Scratch::new();
        let toggles = V2NewFeatureToggles {
            append2_block: true,
            csfw_block: true,
            ..V2NewFeatureToggles::default()
        };
        let via_append_stream = z
            .compute_folded720_append_features_streaming(&sref, &dref, toggles, &mut scratch)
            .unwrap();
        let via_fold_stream = z
            .compute_folded720_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles {
                    append_block: true,
                    ..toggles
                },
                &mut scratch,
            )
            .unwrap();
        assert_eq!(via_append_stream.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(via_fold_stream.regime(), FeatureRegime::Folded720Csfw);
        for i in 0..956 {
            assert_eq!(
                a956.features()[i].to_bits(),
                via_append_stream.features()[i].to_bits(),
                "pair vs append-streaming entry diverge at f{i}"
            );
            assert_eq!(
                a956.features()[i].to_bits(),
                via_fold_stream.features()[i].to_bits(),
                "pair vs fold-streaming entry diverge at f{i}"
            );
        }
        // All 12 new slots bounded [0,1] + finite.
        for (i, v) in a956.csfw_features().unwrap().iter().enumerate() {
            assert!((0.0..=1.0).contains(v) && v.is_finite(), "csfw[{i}] = {v}");
        }
        // The weighted twins must actually differ from the unweighted
        // GLOBAL_* on a real distortion (the weight is not a no-op): at
        // least one scale's W_GLOBAL_DMEAN differs when the unweighted
        // twin is nonzero.
        let app = a956.append_features().unwrap();
        let csfw = a956.csfw_features().unwrap();
        let mut any_diff = false;
        for scale in 0..4 {
            let u = app[scale * (3 * FEATURES_PER_CHANNEL_APPEND)
                + FEATURES_PER_CHANNEL_APPEND
                + idx_append::GLOBAL_DMEAN];
            let wv = csfw[scale * CSFW_PER_SCALE + idx_csfw::W_GLOBAL_DMEAN];
            if u > 1e-9 && (wv - u).abs() > 1e-12 {
                any_diff = true;
            }
        }
        assert!(
            any_diff,
            "weighted GLOBAL_DMEAN identical to unweighted on a distorted pair — weight inert?"
        );

        // Identity pair: every CSFW slot EXACTLY 0 (v ≡ 0 ⇒ weighted
        // pools of identical planes cancel exactly, independent of w).
        let idr = z.compute_folded720_csfw_features(&sref, &sref).unwrap();
        let csfw_id = idr.csfw_features().unwrap();
        for (i, v) in csfw_id.iter().enumerate() {
            assert_eq!(*v, 0.0, "identity csfw[{i}] = {v}");
        }
    }

    /// The doc-predicted luminance direction (design §5.1/§5.2): the SDR
    /// achromatic weight peaks in the darks (~×1.8 near sRGB code 32)
    /// and falls below 1 in the highlights (~×0.62 at code 255), so a
    /// mean shift confined to DARK content must weigh MORE in
    /// `W_GLOBAL_DMEAN` than its unweighted twin, and a bright-confined
    /// shift LESS. Same direction on the HDR route (castleCSF-over-PU21
    /// residual: w 1.95 at 1 cd/m² → 0.63 at 4000).
    #[test]
    fn csfw_luminance_direction_dark_up_bright_down() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);

        // SDR: left half dark (code 20), right half bright (code 220).
        let half = |dark: u8, bright: u8| -> Vec<[u8; 3]> {
            (0..w * h)
                .map(|i| {
                    let v = if i % w < w / 2 { dark } else { bright };
                    [v, v, v]
                })
                .collect()
        };
        let src = half(20, 220);
        let dark_shift = half(26, 220); // +6 codes, dark half only
        let bright_shift = half(20, 226); // +6 codes, bright half only
        let lane_pair = |dst: &Vec<[u8; 3]>| -> (f64, f64) {
            let r = z
                .compute_folded720_csfw_features(
                    &RgbSlice::new(&src, w, h),
                    &RgbSlice::new(dst, w, h),
                )
                .unwrap();
            let u = r.append_features().unwrap()
                [FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN];
            let wv = r.csfw_features().unwrap()[idx_csfw::W_GLOBAL_DMEAN];
            (wv, u)
        };
        let (wd, ud) = lane_pair(&dark_shift);
        let (wb, ub) = lane_pair(&bright_shift);
        println!(
            "SDR dark-shift:   weighted {wd:.6} unweighted {ud:.6} (ratio {:.3})",
            wd / ud
        );
        println!(
            "SDR bright-shift: weighted {wb:.6} unweighted {ub:.6} (ratio {:.3})",
            wb / ub
        );
        assert!(
            wd > ud * 1.05,
            "dark-confined mean shift must up-weigh: {wd} vs {ud}"
        );
        assert!(
            wb < ub * 0.95,
            "bright-confined mean shift must down-weigh: {wb} vs {ub}"
        );

        // HDR route: left half 2 cd/m², right half 400 cd/m².
        let nits_half = |dark: f32, bright: f32| -> Vec<[f32; 3]> {
            (0..w * h)
                .map(|i| {
                    let v = if i % w < w / 2 { dark } else { bright };
                    [v, v, v]
                })
                .collect()
        };
        let mut scratch = V2Scratch::new();
        let hsrc = nits_half(2.0, 400.0);
        let hdark = nits_half(2.4, 400.0); // +20% dark half
        let hbright = nits_half(2.0, 480.0); // +20% bright half
        let hdr_lane_pair = |dst: &Vec<[f32; 3]>, scratch: &mut V2Scratch| -> (f64, f64) {
            let r = z
                .compute_folded720_csfw_features_hdr(
                    &NitsImage::from_rgb_nits(&hsrc, w, h),
                    &NitsImage::from_rgb_nits(dst, w, h),
                    HdrEncoding::Linear,
                    V2NewFeatureToggles::default(),
                    scratch,
                )
                .unwrap();
            let u = r.append_features().unwrap()
                [FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN];
            let wv = r.csfw_features().unwrap()[idx_csfw::W_GLOBAL_DMEAN];
            (wv, u)
        };
        let (hwd, hud) = hdr_lane_pair(&hdark, &mut scratch);
        let (hwb, hub) = hdr_lane_pair(&hbright, &mut scratch);
        println!(
            "HDR dark-shift:   weighted {hwd:.6} unweighted {hud:.6} (ratio {:.3})",
            hwd / hud
        );
        println!(
            "HDR bright-shift: weighted {hwb:.6} unweighted {hub:.6} (ratio {:.3})",
            hwb / hub
        );
        assert!(
            hwd > hud * 1.05,
            "HDR dark-confined shift must up-weigh: {hwd} vs {hud}"
        );
        assert!(
            hwb < hub * 0.95,
            "HDR bright-confined shift must down-weigh: {hwb} vs {hub}"
        );
    }

    /// HDR-route CSFW: 956 shape, HDR entry parity, identity zeros, and
    /// bounded firing on a real gain-error nits pair.
    #[test]
    fn csfw_hdr_route_entry_parity_and_smoke() {
        let (w, h) = (128usize, 128usize);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();

        let ramp: Vec<[f32; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let t = (x + y) as f32 / (w + h - 2) as f32;
                let nits = 0.5 * (2000.0f32 / 0.5).powf(t);
                [nits, nits, nits]
            })
            .collect();
        let dst: Vec<[f32; 3]> = ramp
            .iter()
            .map(|&[r, g, b]| [r * 1.12, g * 1.12, b * 1.12])
            .collect();
        let sref = NitsImage::from_rgb_nits(&ramp, w, h);
        let dref = NitsImage::from_rgb_nits(&dst, w, h);

        let a = z
            .compute_folded720_csfw_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        assert_eq!(a.regime(), FeatureRegime::Folded720Csfw);
        assert_eq!(a.features().len(), 956);
        // First-944 bit-stability on the HDR route.
        let a944 = z
            .compute_folded720_append2_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        for i in 0..944 {
            assert_eq!(
                a944.features()[i].to_bits(),
                a.features()[i].to_bits(),
                "HDR csfw toggled on moved f{i}"
            );
        }
        // HDR entry parity: the csfw entry vs the append2/append HDR
        // entries carrying the toggle — byte-identical 956.
        let via_a2 = z
            .compute_folded720_append2_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    csfw_block: true,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap();
        let via_app = z
            .compute_folded720_append_features_hdr(
                &sref,
                &dref,
                HdrEncoding::Linear,
                V2NewFeatureToggles {
                    append2_block: true,
                    csfw_block: true,
                    ..V2NewFeatureToggles::default()
                },
                &mut scratch,
            )
            .unwrap();
        for i in 0..956 {
            assert_eq!(
                a.features()[i].to_bits(),
                via_a2.features()[i].to_bits(),
                "HDR csfw vs append2-entry toggles diverge at f{i}"
            );
            assert_eq!(
                a.features()[i].to_bits(),
                via_app.features()[i].to_bits(),
                "HDR csfw vs append-entry toggles diverge at f{i}"
            );
        }
        // Bounded + fires: a global 12% gain must move the weighted
        // global lanes.
        let csfw = a.csfw_features().unwrap();
        for (i, v) in csfw.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(v) && v.is_finite(),
                "hdr csfw[{i}] = {v}"
            );
        }
        let fired = (0..4).any(|s| csfw[s * CSFW_PER_SCALE + idx_csfw::W_GLOBAL_DMEAN] > 1e-4);
        assert!(
            fired,
            "W_GLOBAL_DMEAN should fire on a 12% gain pair: {csfw:?}"
        );

        // HDR identity: exactly 0 on every CSFW slot.
        let idr = z
            .compute_folded720_csfw_features_hdr(
                &sref,
                &sref,
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        for (i, v) in idr.csfw_features().unwrap().iter().enumerate() {
            assert_eq!(*v, 0.0, "HDR identity csfw[{i}] = {v}");
        }
    }

    /// CSFW φ-constant derivation (design §13, the
    /// `bandvis_delta_derivation_table` pattern): recompute the derived
    /// weight `w(L) = S_Ach(L) / (L · dV/dL)` from castleCSF Eq. 21 and
    /// the LIVE front-end encodings (numeric dV/dL through
    /// `srgb_to_positive_xyb_planar_into` / `linear_to_pu_xyb_planar_into`
    /// gray probes), normalize at each route's anchor, and bracket the
    /// shipped quadratics `1 + φ(y_live)` against it. Prints the table
    /// (`--nocapture`) committed in `benchmarks/csf_tier1_gates_2026-07-28.md`.
    #[test]
    fn csfw_phi_derivation_table() {
        // castleCSF Appendix Table 5, achromatic sustained (Eq. 21):
        // S(Y) = k1·(1 + k2/Y)^(−k3) · [1 − (1 + k4/Y)^(−k5)].
        let s_ach = |l: f64| -> f64 {
            let (k1, k2, k3, k4, k5) = (56.49, 7.547, 0.1445, 5.583e-7, 9.669e9);
            // (1 + k4/Y)^(−k5) with k4/Y ≪ 1: exp(−k5·ln1p(k4/Y)) — the
            // numerically-stable form of the high-luminance roll-off
            // (≈ exp(−5398/Y)).
            let roll = 1.0 - (-k5 * (k4 / l).ln_1p()).exp();
            k1 * (1.0 + k2 / l).powf(-k3) * roll
        };

        // --- SDR route: live Y-plane of gray at sRGB code c, and the
        //     standard_4k display model (Y_peak 200, Y_black 0.2,
        //     Y_refl 0.39788736).
        let y_sdr = |c: u8| -> f64 {
            let px = [[c, c, c]];
            let (mut o0, mut o1, mut o2) = ([0.0f32; 1], [0.0f32; 1], [0.0f32; 1]);
            crate::color::srgb_to_positive_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            o1[0] as f64
        };
        let srgb_eotf = |c: f64| -> f64 {
            let v = c / 255.0;
            if v <= 0.040_449_936 {
                v / 12.92
            } else {
                ((v + 0.055) / 1.055).powf(2.4)
            }
        };
        let l_of_code = |c: f64| -> f64 { (200.0 - 0.2) * srgb_eotf(c) + 0.2 + 0.397_887_36 };

        // dy/dL through the live front-end, centered difference over ±2
        // codes (the plane is C¹ in code).
        let anchor_code = 128u8;
        let w_derived_sdr = |c: u8| -> f64 {
            let (cm, cp) = (c - 2, c + 2);
            let dy = y_sdr(cp) - y_sdr(cm);
            let dl = l_of_code(cp as f64) - l_of_code(cm as f64);
            let l = l_of_code(c as f64);
            s_ach(l) / (l * (dy / dl))
        };
        let norm_sdr = w_derived_sdr(anchor_code);
        println!("SDR route (castleCSF ÷ live cbrt front-end, norm @ code 128):");
        println!("  code    L[cd/m²]  y_live   w_derived  w_shipped  |err|");
        let mut sdr_errs = Vec::new();
        for c in [8u8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 248] {
            let l = l_of_code(c as f64);
            let y = y_sdr(c);
            let wd = w_derived_sdr(c) / norm_sdr;
            let ws = (1.0
                + CSFW_KAPPA_Y
                    * (CSFW_PHI_Y_SDR[0] + y * (CSFW_PHI_Y_SDR[1] + y * CSFW_PHI_Y_SDR[2])))
                .clamp(CSFW_W_MIN, CSFW_W_MAX);
            let err = (ws - wd).abs();
            sdr_errs.push((c, err));
            println!("  {c:4}  {l:9.3}  {y:.4}   {wd:.4}     {ws:.4}     {err:.4}");
        }

        // --- HDR/PU route: live PU-Y plane of gray at L cd/m², norm @ 100.
        let y_pu = |nits: f64| -> f64 {
            let px = [[nits as f32, nits as f32, nits as f32]];
            let (mut o0, mut o1, mut o2) = ([0.0f32; 1], [0.0f32; 1], [0.0f32; 1]);
            crate::color::linear_to_pu_xyb_planar_into(&px, &mut o0, &mut o1, &mut o2);
            o1[0] as f64
        };
        let w_derived_pu = |nits: f64| -> f64 {
            let (lm, lp) = (nits * 0.99, nits * 1.01);
            let dy = y_pu(lp) - y_pu(lm);
            let dl = lp - lm;
            s_ach(nits) / (nits * (dy / dl))
        };
        let norm_pu = w_derived_pu(100.0);
        println!("HDR/PU route (castleCSF ÷ live PU front-end, norm @ 100 cd/m²):");
        println!("  L[cd/m²]  y_live   w_derived  w_shipped  |err|");
        let mut pu_errs = Vec::new();
        for nits in [
            1.0f64, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 1000.0, 2000.0, 4000.0,
        ] {
            let y = y_pu(nits);
            let wd = w_derived_pu(nits) / norm_pu;
            let ws = (1.0
                + CSFW_KAPPA_Y
                    * (CSFW_PHI_Y_PU[0] + y * (CSFW_PHI_Y_PU[1] + y * CSFW_PHI_Y_PU[2])))
                .clamp(CSFW_W_MIN, CSFW_W_MAX);
            let err = (ws - wd).abs();
            pu_errs.push((nits, err));
            println!("  {nits:8.1}  {y:.4}   {wd:.4}     {ws:.4}     {err:.4}");
        }

        // --- Live-coordinate LSQ refit (the shipped constants' actual
        //     derivation): quadratic least squares of `w_derived − 1` on
        //     `[1, y_live, y_live²]`, exactly the design §5.2 recipe but
        //     composed with the LIVE front-end encoding rather than the
        //     doc's idealized `cbrt(rel)` / `PU21/PU_WHITE` coordinates.
        //     (Found during implementation: the SDR Y plane is
        //     `cbrt(rel + β) − cbrt(β) + 0.01` with the opsin bias β —
        //     NOT an affine map of `cbrt(rel)` — so the doc's §5.3
        //     table values are mis-anchored when evaluated at the live
        //     plane value; the doc's §6 pre-composition rule + §13
        //     live-bracket requirement resolve in favor of the live
        //     coordinate. Recorded in the gates doc.)
        let fit_quadratic = |samples: &[(f64, f64)]| -> [f64; 3] {
            // Normal equations for min Σ (c0 + c1·y + c2·y² − t)².
            let mut a = [[0.0f64; 3]; 3];
            let mut b = [0.0f64; 3];
            for &(y, t) in samples {
                let basis = [1.0, y, y * y];
                for i in 0..3 {
                    for j in 0..3 {
                        a[i][j] += basis[i] * basis[j];
                    }
                    b[i] += basis[i] * t;
                }
            }
            // Gaussian elimination, partial pivot (3×3).
            let mut m = [
                [a[0][0], a[0][1], a[0][2], b[0]],
                [a[1][0], a[1][1], a[1][2], b[1]],
                [a[2][0], a[2][1], a[2][2], b[2]],
            ];
            for col in 0..3 {
                let piv = (col..3)
                    .max_by(|&i, &j| m[i][col].abs().partial_cmp(&m[j][col].abs()).unwrap())
                    .unwrap();
                m.swap(col, piv);
                let piv_row = m[col];
                #[allow(clippy::needless_range_loop)] // row != col guard needs the index
                for row in 0..3 {
                    if row != col {
                        let f = m[row][col] / piv_row[col];
                        for (k, mk) in m[row].iter_mut().enumerate().skip(col) {
                            *mk -= f * piv_row[k];
                        }
                    }
                }
            }
            [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]]
        };

        let sdr_samples: Vec<(f64, f64)> = (4u8..=253)
            .map(|c| (y_sdr(c), w_derived_sdr(c) / norm_sdr - 1.0))
            .collect();
        let sdr_fit = fit_quadratic(&sdr_samples);
        let (sdr_rms, sdr_max) = {
            let (mut s2, mut mx) = (0.0f64, 0.0f64);
            for &(y, t) in &sdr_samples {
                let e = (sdr_fit[0] + y * (sdr_fit[1] + y * sdr_fit[2]) - t).abs();
                s2 += e * e;
                mx = mx.max(e);
            }
            ((s2 / sdr_samples.len() as f64).sqrt(), mx)
        };
        println!(
            "SDR live-coordinate refit: c0 {:+.5} c1 {:+.5} c2 {:+.5}  (rms {:.4} max {:.4})",
            sdr_fit[0], sdr_fit[1], sdr_fit[2], sdr_rms, sdr_max
        );

        let pu_samples: Vec<(f64, f64)> = (0..200)
            .map(|i| {
                let l = (4000.0f64 / 1.0).powf(i as f64 / 199.0);
                (y_pu(l), w_derived_pu(l) / norm_pu - 1.0)
            })
            .collect();
        let pu_fit = fit_quadratic(&pu_samples);
        let (pu_rms, pu_max) = {
            let (mut s2, mut mx) = (0.0f64, 0.0f64);
            for &(y, t) in &pu_samples {
                let e = (pu_fit[0] + y * (pu_fit[1] + y * pu_fit[2]) - t).abs();
                s2 += e * e;
                mx = mx.max(e);
            }
            ((s2 / pu_samples.len() as f64).sqrt(), mx)
        };
        println!(
            "PU  live-coordinate refit: c0 {:+.5} c1 {:+.5} c2 {:+.5}  (rms {:.4} max {:.4})",
            pu_fit[0], pu_fit[1], pu_fit[2], pu_rms, pu_max
        );

        // The SHIPPED constants must BE the live-coordinate refit (the
        // one true derivation loop: castleCSF Eq. 21 → live encoding →
        // LSQ → const). Curve-level agreement over each route's fit
        // range, |Δw| < 0.01 everywhere.
        for &(y, _) in &sdr_samples {
            let ship = CSFW_PHI_Y_SDR[0] + y * (CSFW_PHI_Y_SDR[1] + y * CSFW_PHI_Y_SDR[2]);
            let refit = sdr_fit[0] + y * (sdr_fit[1] + y * sdr_fit[2]);
            assert!(
                (ship - refit).abs() < 0.01,
                "shipped SDR φ drifted from the live refit at y={y}: {ship} vs {refit}"
            );
        }
        for &(y, _) in &pu_samples {
            let ship = CSFW_PHI_Y_PU[0] + y * (CSFW_PHI_Y_PU[1] + y * CSFW_PHI_Y_PU[2]);
            let refit = pu_fit[0] + y * (pu_fit[1] + y * pu_fit[2]);
            assert!(
                (ship - refit).abs() < 0.01,
                "shipped PU φ drifted from the live refit at y={y}: {ship} vs {refit}"
            );
        }
        // And the fitted curve agrees with the castleCSF-derived one
        // within the doc's honest fit class (§5.2: quadratic residuals —
        // brackets set from this run's printed rms/max).
        assert!(
            sdr_rms < 0.25 && sdr_max < 0.9,
            "SDR refit residual blew up: rms {sdr_rms} max {sdr_max}"
        );
        assert!(
            pu_rms < 0.15 && pu_max < 0.45,
            "PU refit residual blew up: rms {pu_rms} max {pu_max}"
        );
        // The derived curves stay inside the clamp everywhere in the fit
        // range — the clamp is a guard band, not an active regularizer.
        for c in 4u8..=252 {
            let wd = w_derived_sdr(c) / norm_sdr;
            assert!(
                wd > CSFW_W_MIN && wd < CSFW_W_MAX,
                "derived SDR w at code {c} outside clamp: {wd}"
            );
        }
        // Seeds: λ_2 is the identifiability anchor and κ seeds at the
        // derived-curve strength.
        assert_eq!(CSFW_LAMBDA_B[2], 1.0);
        assert_eq!(CSFW_KAPPA_Y, 1.0);
    }

    // ── task #67 C2a: v2+append attribution density ─────────────────────

    /// Pass A of the attribution builder must reproduce the production
    /// v2 + append features: same kernels over replicated planes, so the
    /// pooled scalars every coefficient derives from are
    /// production-arithmetic. Compared against the canonical folded-append
    /// STREAMING extractor (what the 924 bakes were trained on).
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn v2_append_attr_features_match_production() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let prod = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let pf = prod.features();
        assert_eq!(pf.len(), 924);

        let s_zero_v2 = vec![0.0f64; 348];
        let s_zero_app = vec![0.0f64; 204];
        let attr = compute_v2_append_attribution(
            &sref,
            &dref,
            &s_zero_v2,
            Some(&s_zero_app),
            None,
            None,
            false,
        )
        .unwrap();
        assert_eq!(attr.v2_features.len(), 348);
        assert_eq!(attr.append_features.len(), 204);
        let mut worst = 0.0f64;
        for k in 0..348 {
            let (a, b) = (attr.v2_features[k], pf[372 + k]);
            let rel = (a - b).abs() / b.abs().max(1e-12);
            worst = worst.max(rel);
            assert!(
                rel <= 1e-9,
                "v2 slot {k}: attr {a} vs production {b} (rel {rel:.3e})"
            );
        }
        for k in 0..204 {
            let (a, b) = (attr.append_features[k], pf[720 + k]);
            let rel = (a - b).abs() / b.abs().max(1e-12);
            worst = worst.max(rel);
            assert!(
                rel <= 1e-9,
                "append slot {k}: attr {a} vs production {b} (rel {rel:.3e})"
            );
        }
        let _ = worst;
    }

    /// Sum identities of the density (full-image rectangle sum) for the
    /// pool classes, against the PRODUCTION feature values: mean pools
    /// (Σ density = −s_k·f_k), reference-weighted pools (same identity),
    /// and the self-weighted soft-peak (Σ density = 0 exactly, since
    /// Σ w(v)(f−v) ≡ 0). Tolerances are 1e-5-class: pass B re-derives the
    /// per-pixel signals in f64 while the kernels pooled f32-lane values —
    /// a documented recompute-precision gap, NOT a pooling mismatch (that
    /// is gated at 1e-9 by `v2_append_attr_features_match_production`).
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn v2_append_attr_sum_identities() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let prod = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let pf = prod.features();

        let run = |sv2: Vec<f64>, sap: Option<Vec<f64>>| -> f64 {
            let attr = compute_v2_append_attribution(
                &sref,
                &dref,
                &sv2,
                sap.as_deref(),
                None,
                None,
                false,
            )
            .unwrap();
            attr.density.iter().sum()
        };

        // Mean pools: v2 ART (scale 0, ch Y, slot 3) and append
        // MSCN_DIFF_MEAN (scale 1, ch X, slot 5) + XMASK (scale 0, Y).
        for (v2_idx, app_idx, feat_idx) in [
            (Some(29 + idx::ART), None, 372 + 29 + idx::ART),
            (
                None,
                Some(3 * FEATURES_PER_CHANNEL_APPEND + idx_append::MSCN_DIFF_MEAN),
                720 + 3 * FEATURES_PER_CHANNEL_APPEND + idx_append::MSCN_DIFF_MEAN,
            ),
            (
                None,
                Some(FEATURES_PER_CHANNEL_APPEND + idx_append::XMASK_TRANSDUCER),
                720 + FEATURES_PER_CHANNEL_APPEND + idx_append::XMASK_TRANSDUCER,
            ),
            // Reference-weighted pools: v2 MASKED_SSIM (scale 0, Y) and
            // append LUM_DARK_ERR (scale 0, Y).
            (
                Some(29 + idx::MASKED_SSIM),
                None,
                372 + 29 + idx::MASKED_SSIM,
            ),
            (
                None,
                Some(FEATURES_PER_CHANNEL_APPEND + idx_append::LUM_DARK_ERR),
                720 + FEATURES_PER_CHANNEL_APPEND + idx_append::LUM_DARK_ERR,
            ),
        ] {
            let mut sv2 = vec![0.0f64; 348];
            let mut sap = vec![0.0f64; 204];
            if let Some(k) = v2_idx {
                sv2[k] = -1.0;
            }
            if let Some(k) = app_idx {
                sap[k] = -1.0;
            }
            let sum = run(sv2, Some(sap));
            let expect = pf[feat_idx];
            assert!(
                expect > 0.0,
                "fixture must exercise feature {feat_idx} (got {expect})"
            );
            // 1e-5: the kernels pool f32-lane values (the dense kernel's
            // POOL_SIMD masked/iw block is f32-lane-accumulated on v4x —
            // documented 5e-4 reassociation class in the module docs) while
            // pass B re-derives in f64; measured gap ~2e-6 on this fixture
            // (MASKED_SSIM). Pooling parity itself is gated at 1e-9 by
            // `v2_append_attr_features_match_production`.
            assert!(
                (sum - expect).abs() <= 1e-5 * expect.max(1e-9),
                "slot f{feat_idx}: density sum {sum} vs feature {expect}"
            );
        }

        // Soft-peak: full-image sum is identically 0 (Σ w(v)(f−v) = 0).
        let mut sv2 = vec![0.0f64; 348];
        sv2[29 + idx::SSIM_SOFT_PEAK] = -1.0;
        let attr =
            compute_v2_append_attribution(&sref, &dref, &sv2, None, None, None, false).unwrap();
        let total: f64 = attr.density.iter().sum();
        let mass: f64 = attr.density.iter().map(|v| v.abs()).sum();
        // 1e-4·mass: the identity Σw(v)·v = f·Σw(v) is exact only when `f`
        // is computed from the same per-pixel arithmetic; here `f` comes
        // from the f32-lane kernel pools while the density recomputes in
        // f64 (measured residual ~2e-5·mass). A sign/formula error would
        // land at ~mass scale — 4 orders above this bound.
        assert!(
            total.abs() <= 1e-4 * mass.max(1e-12),
            "soft-peak density must sum to 0: total {total}, |mass| {mass}"
        );
        assert!(mass > 0.0, "soft-peak density must be non-trivial");
    }

    // ── append2 (f924-943) attribution gates — campaign appendix E ────────
    //
    // The block was silently dropped by `compute_attribution_density_full`'s
    // `s[720..min(len, 924)]` slice until 2026-08-04. These three gates pin
    // the registered per-slot determination: BANDVIS_GAIN/LOSS are class-E
    // mean pools (spatialized, sum-identity + FD-direction verified), and
    // LUMA_MEAN_REF / HL_BIN1 / HL_BIN2 are exactly zero BY DEFINITION
    // (reference-only; HDR-gated on an SDR-only route) rather than by an
    // unreached slice bound.

    /// Plane-sum identity for the append2 BANDVIS pair against the
    /// PRODUCTION 944-regime features, plus the exactly-zero gate for the
    /// three non-decomposable slots. Same 1e-5 class as
    /// `v2_append_attr_sum_identities` (pass B re-derives in f64 over
    /// f32-lane-pooled kernels).
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn append2_attr_sum_identities_and_zero_slots() {
        let (w, h) = (150usize, 170usize);
        let src = textured_image(w, h, 23);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let prod = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        let app2 = prod.append2_features().unwrap().to_vec();
        let n_scales = prod.n_scales();

        let run = |sap2: Vec<f64>| -> f64 {
            let attr =
                compute_v2_append_attribution(&sref, &dref, &[], None, Some(&sap2), None, false)
                    .unwrap();
            attr.density.iter().sum()
        };

        // BANDVIS is a COARSE-scale detector (its own doc: expect ~0 at
        // scale 0/1). Gate every scale where the production feature is
        // non-trivial, and require that at least one such scale exists —
        // so a fixture that stopped exercising BANDVIS fails loudly rather
        // than vacuously passing.
        let mut checked = 0usize;
        for scale in 0..n_scales {
            for local in [idx_append2::BANDVIS_GAIN, idx_append2::BANDVIS_LOSS] {
                let k = scale * APPEND2_PER_SCALE + local;
                let expect = app2[k];
                if expect <= 1e-6 {
                    continue;
                }
                let mut sap2 = vec![0.0f64; n_scales * APPEND2_PER_SCALE];
                sap2[k] = -1.0;
                let sum = run(sap2);
                assert!(
                    (sum - expect).abs() <= 1e-5 * expect.max(1e-9),
                    "append2 slot (scale {scale}, local {local}): density sum {sum} vs \
                     production feature {expect}"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "fixture must exercise at least one BANDVIS slot (production app2 = {app2:?})"
        );

        // The three registered class-N slots: the machinery must emit
        // EXACTLY zero density for any gradient on them, at every scale.
        for scale in 0..n_scales {
            for local in [
                idx_append2::LUMA_MEAN_REF,
                idx_append2::HL_BIN1,
                idx_append2::HL_BIN2,
            ] {
                let mut sap2 = vec![0.0f64; n_scales * APPEND2_PER_SCALE];
                sap2[scale * APPEND2_PER_SCALE + local] = -1.0;
                let attr = compute_v2_append_attribution(
                    &sref,
                    &dref,
                    &[],
                    None,
                    Some(&sap2),
                    None,
                    false,
                )
                .unwrap();
                let mass: f64 = attr.density.iter().map(|v| v.abs()).sum();
                assert_eq!(
                    mass, 0.0,
                    "append2 slot (scale {scale}, local {local}) is class N — the density must \
                     be identically 0, got |mass| {mass}"
                );
            }
        }
    }

    /// FD DIRECTION gate for the two new append2 integrands (the C2a
    /// precedent — an FD direction test is what caught the edge-width
    /// sign bug before it landed). Refining a block toward the reference
    /// must move the production feature by the amount the density's
    /// rectangle sum predicts, in SIGN and order of magnitude.
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn append2_attr_bandvis_fd_direction() {
        let (w, h) = (160usize, 160usize);
        let src = textured_image(w, h, 11);
        // A posterizing distortion is what BANDVIS exists to see.
        let mut dst = src.clone();
        for p in dst.iter_mut() {
            for c in p.iter_mut() {
                *c = (*c / 24) * 24;
            }
        }
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let base = z.compute_folded720_append2_features(&sref, &dref).unwrap();
        let a0 = base.append2_features().unwrap().to_vec();
        let n_scales = base.n_scales();

        // Refine the LEFT HALF to the reference. A half-plane is the
        // minimum-seam region for its area: one interior seam of `h` pixels
        // (the other three sides are image edges, which reflect-pad rather
        // than abut unrefined content). That matters because the integrand
        // is exact only for pixels whose 3-tap curvature stencil stays
        // inside the refined set, and the seam-contaminated fraction grows
        // as `2^scale` up the pyramid — a compact block leaves 17 % of its
        // scale-2 pixels seam-adjacent (measured: it flips the LOSS sign
        // there), a half-plane leaves 5 %. See `benchmarks/
        // attribution_append2_e1_2026-08-04.md` for the measured comparison.
        let (bx, by, bw, bh) = (0usize, 0usize, w / 2, h);
        let mut refined = dst.clone();
        for y in by..by + bh {
            for x in bx..bx + bw {
                refined[y * w + x] = src[y * w + x];
            }
        }
        let rref = RgbSlice::new(&refined, w, h);
        let after = z.compute_folded720_append2_features(&sref, &rref).unwrap();
        let a1 = after.append2_features().unwrap().to_vec();

        let mut checked = 0usize;
        for scale in 0..n_scales {
            for local in [idx_append2::BANDVIS_GAIN, idx_append2::BANDVIS_LOSS] {
                let k = scale * APPEND2_PER_SCALE + local;
                let true_delta = a1[k] - a0[k];
                // Only gate slots the refinement actually moves; a slot that
                // does not move carries no direction to check.
                if true_delta.abs() <= 1e-4 {
                    continue;
                }
                let mut sap2 = vec![0.0f64; n_scales * APPEND2_PER_SCALE];
                sap2[k] = 1.0; // raw ∂score/∂f = +1 ⇒ density = (δf)_i
                let attr = compute_v2_append_attribution(
                    &sref,
                    &dref,
                    &[],
                    None,
                    Some(&sap2),
                    None,
                    false,
                )
                .unwrap();
                let pred: f64 = (by..by + bh)
                    .flat_map(|y| (bx..bx + bw).map(move |x| (y, x)))
                    .map(|(y, x)| attr.density[y * attr.width + x])
                    .sum();
                // EXACTNESS (valid at every scale, no finite-removal caveat):
                // the whole-plane density sum IS the production feature. This
                // is the class-E claim itself; measured agreement is 8-9
                // significant digits on this fixture.
                let whole: f64 = attr.density.iter().sum();
                assert!(
                    (whole + a0[k]).abs() <= 1e-5 * a0[k].abs().max(1e-9),
                    "append2 plane-sum identity (scale {scale}, local {local}): Σdensity \
                     {whole} vs −feature {}",
                    -a0[k]
                );
                assert!(
                    pred.signum() == true_delta.signum(),
                    "append2 FD direction (scale {scale}, local {local}): predicted {pred} vs \
                     true Δf {true_delta} — SIGN MISMATCH"
                );
                // Order of magnitude. This is a FIRST-ORDER integrand asked
                // to predict a HALF-PLANE finite removal, and the seam +
                // pyramid coupling grow with scale, so the honest bound is a
                // factor-4 band (same shape as the C2a FD gates). Measured
                // pred/true on this fixture, 2026-08-04:
                //   scale 0  gain 1.02  loss 1.02
                //   scale 1  gain 1.14  loss 1.12
                //   scale 2  gain 1.50  loss 1.26
                //   scale 3  gain 3.78  loss 2.38   ← the finite-removal floor
                // The monotone drift up the pyramid is the documented
                // approximation, not a formula error: a formula error shows
                // up as a SIGN flip or an order-of-magnitude miss, and the
                // exact plane-sum identity above holds at every scale.
                let ratio = pred.abs() / true_delta.abs();
                assert!(
                    (0.25..=4.0).contains(&ratio),
                    "append2 FD magnitude (scale {scale}, local {local}): predicted {pred} vs \
                     true Δf {true_delta} (ratio {ratio})"
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "refinement must move at least one BANDVIS slot (a0 {a0:?} a1 {a1:?})"
        );
    }

    /// Global-slot integrand sanity: refining a block moves GLOBAL_DMEAN's
    /// density block-sum onto the TRUE feature delta (first-order; the
    /// global slots have no blur bleed, so the agreement is tight).
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn v2_append_attr_global_dmean_matches_true_delta() {
        let (w, h) = (128usize, 128usize);
        let src = textured_image(w, h, 7);
        // Distortion with a global mean shift (so GLOBAL_DMEAN is active).
        let mut dst = quantize_distort(&src, w, h);
        for p in dst.iter_mut() {
            p[0] = p[0].saturating_add(6);
            p[1] = p[1].saturating_add(6);
        }
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let base = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        // Refine a 16×16 block (copy reference pixels).
        let (bx0, by0, bs) = (48usize, 64usize, 16usize);
        let mut refined = dst.clone();
        for y in by0..by0 + bs {
            for x in bx0..bx0 + bs {
                refined[y * w + x] = src[y * w + x];
            }
        }
        let rref = RgbSlice::new(&refined, w, h);
        let ref_feats = z
            .compute_folded720_append_features_streaming(
                &sref,
                &rref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        // GLOBAL_DMEAN, scale 0, ch Y.
        let k = 720 + FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN;
        let true_delta = ref_feats.features()[k] - base.features()[k];
        assert!(true_delta != 0.0, "fixture must move GLOBAL_DMEAN");

        let mut sap = vec![0.0f64; 204];
        sap[FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_DMEAN] = 1.0; // s_k = +1
        let sv2 = vec![0.0f64; 348];
        let attr = compute_v2_append_attribution(&sref, &dref, &sv2, Some(&sap), None, None, false)
            .unwrap();
        // density block sum ≈ s_k · Δf = true_delta (s_k = 1).
        let mut block_sum = 0.0f64;
        for y in by0..by0 + bs {
            for x in bx0..bx0 + bs {
                block_sum += attr.density[y * w + x];
            }
        }
        let rel = (block_sum - true_delta).abs() / true_delta.abs();
        assert!(
            rel <= 0.05,
            "GLOBAL_DMEAN first-order: density block sum {block_sum} vs true Δf {true_delta} (rel {rel:.4})"
        );
    }

    /// Finite-difference direction check for every SIGNED / chain-rule
    /// integrand family: refine one block, compare each slot's density
    /// block-sum against the TRUE production feature delta. These are
    /// first-order estimates of blurred, clamped pool functionals under a
    /// FINITE removal — magnitude agreement is loose by nature (blur bleed
    /// alone moves mass across the block border) — but the SIGN and the
    /// gross magnitude must agree, or a chain-rule sign error exists.
    // Exercises the custom-profiles-gated attribution-density cluster.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn v2_append_attr_signed_integrand_directions() {
        let (w, h) = (128usize, 128usize);
        let src = textured_image(w, h, 11);
        let dst = quantize_distort(&src, w, h);
        let sref = RgbSlice::new(&src, w, h);
        let dref = RgbSlice::new(&dst, w, h);
        let z = crate::Zensim::new(crate::ZensimProfile::codec_target()).with_parallel(false);
        let mut scratch = V2Scratch::new();
        let base = z
            .compute_folded720_append_features_streaming(
                &sref,
                &dref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();
        let (bx0, by0, bs) = (32usize, 48usize, 32usize);
        let mut refined = dst.clone();
        for y in by0..by0 + bs {
            for x in bx0..bx0 + bs {
                refined[y * w + x] = src[y * w + x];
            }
        }
        let rref = RgbSlice::new(&refined, w, h);
        let rfeats = z
            .compute_folded720_append_features_streaming(
                &sref,
                &rref,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap();

        // (label, v2 slice index or append slice index)
        let v2_cases: &[(&str, usize)] = &[
            ("SSIM_DEV2 s0 Y", 29 + idx::SSIM_DEV2),
            ("SSIM_SOFT_PEAK s0 Y", 29 + idx::SSIM_SOFT_PEAK),
            ("EDGE_WIDTH s0 Y", 29 + idx::EDGE_WIDTH_CHANGE),
            ("BLOCKINESS s0 Y", 29 + idx::BLOCKINESS),
        ];
        let app_cases: &[(&str, usize)] = &[
            (
                "GMS_DEV2 s0 Y",
                FEATURES_PER_CHANNEL_APPEND + idx_append::GMS_DEV2,
            ),
            (
                "GLOBAL_CGAIN s0 Y",
                FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_CGAIN,
            ),
            (
                "GLOBAL_CLOSS s0 Y",
                FEATURES_PER_CHANNEL_APPEND + idx_append::GLOBAL_CLOSS,
            ),
        ];
        let mut checked = 0usize;
        let mut check = |label: &str, sv2: Vec<f64>, sap: Option<Vec<f64>>, feat_idx: usize| {
            let true_delta = rfeats.features()[feat_idx] - base.features()[feat_idx];
            // Slots the fixture barely moves can't direction-check.
            if true_delta.abs() < 1e-7 {
                return;
            }
            let attr = compute_v2_append_attribution(
                &sref,
                &dref,
                &sv2,
                sap.as_deref(),
                None,
                None,
                false,
            )
            .unwrap();
            let mut block_sum = 0.0f64;
            for y in by0..by0 + bs {
                for x in bx0..bx0 + bs {
                    block_sum += attr.density[y * w + x];
                }
            }
            // s_k = +1 ⇒ block sum ≈ Δf. Sign must match; magnitude within
            // a factor of ~3 either way (blur bleed + curvature).
            assert!(
                block_sum.signum() == true_delta.signum(),
                "{label}: sign mismatch — density {block_sum} vs true Δf {true_delta}"
            );
            let ratio = block_sum / true_delta;
            assert!(
                (0.3..=3.0).contains(&ratio),
                "{label}: magnitude off — density {block_sum} vs true Δf {true_delta} (ratio {ratio:.3})"
            );
            checked += 1;
        };
        for &(label, k) in v2_cases {
            let mut sv2 = vec![0.0f64; 348];
            sv2[k] = 1.0;
            check(label, sv2, None, 372 + k);
        }
        for &(label, k) in app_cases {
            let mut sap = vec![0.0f64; 204];
            sap[k] = 1.0;
            check(label, vec![0.0f64; 348], Some(sap), 720 + k);
        }
        assert!(
            checked >= 3,
            "fixture too tame — only {checked} signed slots direction-checked"
        );
    }
}

// ============================================================================
// THE SCALAR ORACLE (era-2, 2026-08-31) — `benchmarks/era2_perf_break_2026-08-31.md` §11
//
// TEST-ONLY. Never in the product build. Two levels, because "correct" must
// not itself be an approximation:
//
//   L1 `Neumaier` — a readable scalar reference in unambiguous left-to-right
//      order with Kahan-Babuska-NEUMAIER compensation (chosen over plain
//      Kahan, which loses the correction when the running total is smaller in
//      magnitude than the addend — a real case here, since these sums start
//      at 0 and the first terms can dominate).
//   L2 `Exact`   — Shewchuk non-overlapping expansion via TwoSum. The
//      expansion holds the EXACT sum of the f64 terms; `value()` rounds it
//      once by summing components smallest-first.
//
// SCOPE, stated precisely: the oracle is exact with respect to SUMMATION, and
// the per-pixel formulas are evaluated in f64. That is the right reference,
// because what era-2 changes IS the summation order and the precision of the
// accumulation — not the formulas. Where a production tier evaluates the
// per-pixel terms in f32 lanes (the POOL_SIMD pool path), its deviation from
// this oracle legitimately includes that term-precision difference, and §10.4
// of the design says so.
//
// STANDING ROLE: this is the regression instrument for ALL future perf work on
// these kernels, not an era-2 one-off. Measure a new kernel against the oracle
// BEFORE measuring it against a bench.
#[cfg(any(test, feature = "oracle"))]
// The oracle is a RULER, not semantics: its consumers are `cfg(test)` gates and
// the `oracle`-feature harnesses. Building the lib with `oracle` on but without
// `cfg(test)` therefore leaves most of it unreferenced, which is correct rather
// than a smell.
#[allow(dead_code)]
pub(crate) mod oracle {
    use super::*;

    /// Number of accumulator slots: 13 plain sums + 11 `WeightedSum` × (num, den).
    pub(crate) const N_SLOTS: usize = 13 + 11 * 2;

    /// Slot names, index-aligned with [`Slots`], grouped by FAMILY so the gate
    /// can report per-family worst cases (design §11.2).
    pub(crate) const SLOT_NAMES: [&str; N_SLOTS] = [
        "sum_d",
        "sum_d2",
        "sum_d3",
        "sum_d4",
        "sum_art",
        "sum_det",
        "sum_mse",
        "sum_hf_gain",
        "sum_hf_loss",
        "sum_hf_mag_loss",
        "sum_pjnd",
        "sum_pjnd_lo",
        "sum_pjnd_hi",
        "ws_peak_ssim.num",
        "ws_peak_ssim.den",
        "ws_peak_art.num",
        "ws_peak_art.den",
        "ws_peak_det.num",
        "ws_peak_det.den",
        "ws_mask_ssim.num",
        "ws_mask_ssim.den",
        "ws_mask_art.num",
        "ws_mask_art.den",
        "ws_mask_det.num",
        "ws_mask_det.den",
        "ws_mask_mse.num",
        "ws_mask_mse.den",
        "ws_iw_ssim.num",
        "ws_iw_ssim.den",
        "ws_iw_art.num",
        "ws_iw_art.den",
        "ws_iw_det.num",
        "ws_iw_det.den",
        "ws_iw_mse.num",
        "ws_iw_mse.den",
    ];

    /// The family each slot belongs to, for per-family bound reporting.
    pub(crate) fn slot_family(i: usize) -> &'static str {
        match i {
            0..=6 => "core",
            7..=9 => "hf",
            10..=12 => "pjnd",
            _ => "pools",
        }
    }

    /// A compensated or exact accumulator.
    pub(crate) trait Acc: Default + Clone {
        fn add(&mut self, x: f64);
        fn value(&self) -> f64;
    }

    /// L1: Kahan-Babuska-Neumaier compensated summation.
    #[derive(Clone, Default, Debug)]
    pub(crate) struct Neumaier {
        sum: f64,
        c: f64,
    }
    impl Acc for Neumaier {
        #[inline]
        fn add(&mut self, x: f64) {
            let t = self.sum + x;
            // The Neumaier branch: pick the operand that is larger in
            // magnitude as the one whose bits survive, so the correction is
            // captured in BOTH orderings (plain Kahan drops it in one).
            if self.sum.abs() >= x.abs() {
                self.c += (self.sum - t) + x;
            } else {
                self.c += (x - t) + self.sum;
            }
            self.sum = t;
        }
        #[inline]
        fn value(&self) -> f64 {
            self.sum + self.c
        }
    }

    /// `a + b` exactly, as (rounded sum, exact error). Knuth's TwoSum — no
    /// assumption about the relative magnitudes of `a` and `b`.
    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        let err = (a - (s - bb)) + (b - bb);
        (s, err)
    }

    /// L2: Shewchuk non-overlapping expansion. The component list holds the
    /// EXACT sum; nothing is discarded.
    #[derive(Clone, Default, Debug)]
    pub(crate) struct Exact {
        parts: Vec<f64>,
    }
    impl Acc for Exact {
        fn add(&mut self, x: f64) {
            if x == 0.0 {
                return;
            }
            let mut carry = x;
            let mut keep = 0usize;
            for j in 0..self.parts.len() {
                let (hi, lo) = two_sum(carry, self.parts[j]);
                if lo != 0.0 {
                    self.parts[keep] = lo;
                    keep += 1;
                }
                carry = hi;
            }
            self.parts.truncate(keep);
            if carry != 0.0 {
                self.parts.push(carry);
            }
        }
        fn value(&self) -> f64 {
            // Components are non-overlapping and increasing in magnitude;
            // summing smallest-first rounds the exact total once.
            let mut s = 0.0;
            for &p in &self.parts {
                s += p;
            }
            s
        }
    }

    /// One accumulator per slot.
    #[derive(Clone)]
    pub(crate) struct Slots<A: Acc> {
        pub acc: Vec<A>,
    }
    impl<A: Acc> Default for Slots<A> {
        fn default() -> Self {
            Self {
                acc: vec![A::default(); N_SLOTS],
            }
        }
    }
    impl<A: Acc> Slots<A> {
        #[inline]
        fn add(&mut self, i: usize, x: f64) {
            self.acc[i].add(x);
        }
        pub(crate) fn values(&self) -> [f64; N_SLOTS] {
            let mut out = [0.0f64; N_SLOTS];
            for (o, a) in out.iter_mut().zip(self.acc.iter()) {
                *o = a.value();
            }
            out
        }
    }

    /// Flatten a production [`DenseAccum`] into the same slot order.
    pub(super) fn dense_accum_slots(a: &DenseAccum) -> [f64; N_SLOTS] {
        [
            a.sum_d,
            a.sum_d2,
            a.sum_d3,
            a.sum_d4,
            a.sum_art,
            a.sum_det,
            a.sum_mse,
            a.sum_hf_gain,
            a.sum_hf_loss,
            a.sum_hf_mag_loss,
            a.sum_pjnd,
            a.sum_pjnd_lo,
            a.sum_pjnd_hi,
            a.ws_peak_ssim.num,
            a.ws_peak_ssim.den,
            a.ws_peak_art.num,
            a.ws_peak_art.den,
            a.ws_peak_det.num,
            a.ws_peak_det.den,
            a.ws_mask_ssim.num,
            a.ws_mask_ssim.den,
            a.ws_mask_art.num,
            a.ws_mask_art.den,
            a.ws_mask_det.num,
            a.ws_mask_det.den,
            a.ws_mask_mse.num,
            a.ws_mask_mse.den,
            a.ws_iw_ssim.num,
            a.ws_iw_ssim.den,
            a.ws_iw_art.num,
            a.ws_iw_art.den,
            a.ws_iw_det.num,
            a.ws_iw_det.den,
            a.ws_iw_mse.num,
            a.ws_iw_mse.den,
        ]
    }

    /// The reference implementation of the dense pooled-feature math.
    ///
    /// Plain scalar loops, row-major, unambiguous left-to-right order, every
    /// per-pixel term evaluated in f64 with the SAME formulas the production
    /// scalar tail uses (`ssim_d_local`, `bounded_sim`, `saturate`,
    /// `bounded_excess_pair`, `bounded_excess`, `pjnd_transducer`) — this is a
    /// reference for the ACCUMULATION, so the formulas are shared rather than
    /// re-derived, and any formula change is caught by the production tests
    /// rather than silently diverging here.
    ///
    /// Also returns `sum_abs`, the per-slot `Σ|xᵢ|`, which is what the error
    /// bounds in design §10.5 are proportional to — so the gate can compute
    /// each slot's bound from the same run that measures its deviation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn dense_reference<A: Acc>(
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
    ) -> ([f64; N_SLOTS], [f64; N_SLOTS]) {
        let mut s: Slots<A> = Slots::default();
        let mut sum_abs = [0.0f64; N_SLOTS];
        let push = |s: &mut Slots<A>, sum_abs: &mut [f64; N_SLOTS], i: usize, v: f64| {
            s.add(i, v);
            sum_abs[i] += v.abs();
        };

        for y in 0..height {
            for x in 0..width {
                let i = y * width + x;
                let sv = src[i] as f64;
                let dv = dst[i] as f64;
                let m1 = mu1[i] as f64;
                let m2 = mu2[i] as f64;
                let act = activity[i] as f64;

                let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64);
                push(&mut s, &mut sum_abs, 0, d);
                push(&mut s, &mut sum_abs, 1, d * d);
                push(&mut s, &mut sum_abs, 2, d * d * d);
                push(&mut s, &mut sum_abs, 3, d * d * d * d);

                let diff_src = (sv - m1).abs();
                let diff_dst = (dv - m2).abs();
                let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
                let (mut art_i, mut det_i) = (0.0, 0.0);
                if diff_dst > diff_src {
                    art_i = edge_dissim;
                } else if diff_dst < diff_src {
                    det_i = edge_dissim;
                }
                push(&mut s, &mut sum_abs, 4, art_i);
                push(&mut s, &mut sum_abs, 5, det_i);

                let raw_sq_err = (sv - dv) * (sv - dv);
                let mse_i = saturate(raw_sq_err, C_MSE);
                push(&mut s, &mut sum_abs, 6, mse_i);

                let hf_src = sv - m1;
                let hf_dst = dv - m2;
                let (hf_gain_i, hf_loss_i) =
                    bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF);
                push(&mut s, &mut sum_abs, 7, hf_gain_i);
                push(&mut s, &mut sum_abs, 8, hf_loss_i);
                push(
                    &mut s,
                    &mut sum_abs,
                    9,
                    bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF),
                );

                let raw_abs_err = (sv - dv).abs();
                push(
                    &mut s,
                    &mut sum_abs,
                    10,
                    pjnd_transducer(raw_abs_err, act, K_PJND_MASK, C_PJND_CLAMP),
                );
                if transducer_bank {
                    push(
                        &mut s,
                        &mut sum_abs,
                        11,
                        pjnd_transducer(raw_abs_err, act, K_PJND_MASK_LOW, C_PJND_CLAMP),
                    );
                    push(
                        &mut s,
                        &mut sum_abs,
                        12,
                        pjnd_transducer(raw_abs_err, act, K_PJND_MASK_HIGH, C_PJND_CLAMP),
                    );
                }

                // Weighted pools, in the production order.
                let sat_act = saturate(act, C_ACTIVITY);
                let mask_w = 1.0 - sat_act;
                let iw_w = sat_act + IW_WEIGHT_FLOOR;
                let sal_ssim = saturate(d, C_PEAK);
                let sal_art = saturate(art_i, C_PEAK);
                let sal_det = saturate(det_i, C_PEAK);
                for (base, w, v) in [
                    (13, sal_ssim, d),
                    (15, sal_art, art_i),
                    (17, sal_det, det_i),
                    (19, mask_w, d),
                    (21, mask_w, art_i),
                    (23, mask_w, det_i),
                    (25, mask_w, mse_i),
                    (27, iw_w, d),
                    (29, iw_w, art_i),
                    (31, iw_w, det_i),
                    (33, iw_w, mse_i),
                ] {
                    push(&mut s, &mut sum_abs, base, w * v);
                    push(&mut s, &mut sum_abs, base + 1, w);
                }
            }
        }
        (s.values(), sum_abs)
    }
}

// ============================================================================
// ERA-2 DENSE KERNEL (`benchmarks/era2_perf_break_2026-08-31.md` §2, §13.2, §14)
//
// NOT YET THE DEFAULT. Built alongside era-1 so it can be gated and measured
// without moving a byte; the flip is a separate, sequenced commit (design §8 —
// it lands after the fold-engine lane pins its parity stages).
//
// THE SHAPE, and every part of it is semantics rather than implementation:
//   * 8 f32 virtual lanes per accumulator, term at pixel x -> lane[x mod 8],
//     entered via `as_chunks::<8>` — NEVER modulo indexing, which was MEASURED
//     11x slower because it defeats vectorisation entirely (§13);
//   * the scalar tail folds into the SAME lanes, so the shape is
//     width-independent (era-1's tail landed in the f64 running total, which is
//     what made its grouping depend on `width % 8`);
//   * a FIXED reduction tree — `era2_reduce8`, never `reduce_add()`, which is
//     TIER-DEPENDENT (§14.2: x86_v3 pairs lane i with i+4 first, wasm/scalar
//     pair adjacent lanes — different groupings of the same eight addends);
//   * an f64 BAND layer: rows fold into a band partial, band partials merge in
//     band order. Band layout is a pure function of (height, ERA2_BAND_ROWS),
//     never of thread count.
//
// PASS STRUCTURE IS A PERF KNOB, NOT SEMANTICS. Splitting the row into several
// passes changes neither which terms an accumulator sees nor their order — each
// accumulator still receives its terms in increasing x — so any pass split
// yields IDENTICAL bytes. That frees the split to be tuned per tier for
// register pressure without touching the era.

/// era-2 band height. A pure function of geometry; deliberately NOT derived
/// from the thread count, which is what makes thread-invariance structural.
// NOTE ON `dead_code`: these primitives are consumed by the era-2 dense kernel,
// which lands in the SEQUENCED flip commit (design §8 — after the fold-engine
// lane pins its parity stages). Until then they are exercised only by the
// era-2 structural gates, so a non-test build sees them as unused. The
// allow comes off with the kernel.
#[allow(dead_code)]
pub(crate) const ERA2_BAND_ROWS: usize = 32;

/// The era-2 horizontal reduction — **part of the semantics**.
///
/// Pairwise rather than sequential (tighter error, equally fixed), and written
/// out rather than delegated: `GenericF32x8::reduce_add` resolves to a
/// per-backend order (§14.2), so calling it would make the reduction tree an
/// unspecified, tier-dependent operation — precisely what the era-2 identity
/// theorem forbids.
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn era2_reduce8(a: [f32; 8]) -> f64 {
    (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64
}

/// One era-2 accumulator: 8 f32 virtual lanes.
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub(crate) struct Lanes8(pub [f32; 8]);
#[allow(dead_code)]
impl Lanes8 {
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 8])
    }
    /// Add one full 8-wide chunk, lane-wise.
    #[inline(always)]
    fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    /// Fold a partial tail into the SAME lanes: tail element k -> lane k.
    #[inline(always)]
    fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    #[inline(always)]
    fn reduce(self) -> f64 {
        era2_reduce8(self.0)
    }
}

/// era-2 per-pixel term evaluation, f32 — the SIMD-shaped semantics
/// (user directive: the SIMD shape is canonical and the scalar path matches
/// it, never the reverse). Mirrors the `_v` helpers operating on one lane.
#[inline(always)]
fn e2_ssim_d(mu1: f32, mu2: f32, s12: f32, ssq: f32) -> f32 {
    let (c1, c2) = (C1_V2 as f32, C2_V2 as f32);
    let a = 2.0 * mu1 * mu2 + c1;
    let b = mu1 * mu1 + mu2 * mu2 + c1;
    let cov = s12 - mu1 * mu2;
    let c = 2.0 * cov + c2;
    let d = ssq - mu1 * mu1 - mu2 * mu2 + c2;
    let local = (a * c) / (b * d);
    (1.0 - local).max(0.0)
}
#[inline(always)]
fn e2_bounded_sim(a: f32, b: f32, c: f32) -> f32 {
    (2.0 * a * b + c) / (a * a + b * b + c)
}
#[inline(always)]
fn e2_saturate(x: f32, c: f32) -> f32 {
    let x = x.max(0.0);
    x / (x + c)
}
#[inline(always)]
fn e2_bounded_excess_pair(a: f32, b: f32, c: f32) -> (f32, f32) {
    let r = 1.0 / (a + b + c);
    ((a - b).max(0.0) * r, (b - a).max(0.0) * r)
}
#[inline(always)]
fn e2_bounded_excess(a: f32, b: f32, c: f32) -> f32 {
    (a - b).max(0.0) / (a + b + c)
}
#[inline(always)]
fn e2_pjnd(raw_abs_err: f32, act: f32, k: f32, c: f32) -> f32 {
    raw_abs_err / (raw_abs_err + c * (1.0 + k * act))
}

/// era-2 core terms for one pixel: the values every downstream family reads.
#[inline(always)]
fn e2_terms(s: f32, dd: f32, m1: f32, m2: f32, s12: f32, ssq: f32) -> (f32, f32, f32, f32) {
    let d = e2_ssim_d(m1, m2, s12, ssq);
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim = 1.0 - e2_bounded_sim(diff_src, diff_dst, C_EDGE as f32);
    let (mut art_i, mut det_i) = (0.0f32, 0.0f32);
    if diff_dst > diff_src {
        art_i = edge_dissim;
    } else if diff_dst < diff_src {
        det_i = edge_dissim;
    }
    let raw_sq_err = (s - dd) * (s - dd);
    let mse_i = e2_saturate(raw_sq_err, C_MSE as f32);
    (d, art_i, det_i, mse_i)
}

/// **THE ERA-2 DENSE KERNEL.**
///
/// One code path, no tier-specific bodies. Written over `[f32; 8]` fixed
/// arrays so cross-tier bit-identity is not a proof obligation but a
/// tautology — there is only one implementation — while LLVM still
/// auto-vectorises the fixed-size chunks (the workspace's fixed-size-array
/// pattern). If a tier ever needs a hand-written body for speed, it must be
/// proven identical to this one; the oracle gate is where that happens.
///
/// SHAPE (all of it semantics, see `benchmarks/era2_perf_break_2026-08-31.md`):
/// 8 f32 virtual lanes entered via `as_chunks::<8>` with the tail folded into
/// the same lanes; `era2_reduce8`'s fixed tree, never `reduce_add()`; per-row
/// reduction into an f64 band partial (bounding the f32 lane depth exactly as
/// era-1 did, so accuracy is unchanged); band partials merged in band index
/// order, with `ERA2_BAND_ROWS` a pure function of geometry.
///
/// PASS SPLIT (perf only — §14.4 proves any split gives identical bytes):
/// **two passes per row.** Pass A evaluates the terms once, accumulates the 13
/// core families, and writes the four reusable per-pixel values to a row
/// scratch; pass B reads that scratch for the 22 pool accumulators. This keeps
/// each pass's live accumulator set inside a 16-register tier's budget without
/// re-deriving the division-heavy terms, and it is the split to revisit first
/// if the tier numbers ask for it.
/// Cross-vendor / cross-tier harness access: all 35 dense accumulator slots
/// for both eras, so a static binary can print per-slot hashes on two boxes.
/// `benchmarks/era2_perf_break_2026-08-31.md` §15.
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn harness_dense_slots(
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
    era2: bool,
) -> [f64; 35] {
    let a = if era2 {
        dense_block_kernel_era2(
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
    } else {
        dense_block_kernel(
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
    };
    oracle::dense_accum_slots(&a)
}

/// Which SIMD tier the dispatcher actually selected on this host — reported by
/// the harness so a cross-box difference is never mistaken for a vendor effect
/// when it is a TIER effect (they are confounded on real hardware).
#[cfg(any(test, feature = "oracle"))]
pub fn harness_active_tier() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken as _;
        if archmage::X64V4xToken::summon().is_some() {
            return "v4x (AVX-512)";
        }
        if archmage::X64V4Token::summon().is_some() {
            return "v4 (AVX2)";
        }
        if archmage::X64V3Token::summon().is_some() {
            return "v3 (SSE4.2)";
        }
    }
    "scalar/other"
}

/// Bench/gate access to the era-1 dispatched kernel (see `dense_block_kernel`).
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn bench_dense_era1(
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
) -> f64 {
    let a = dense_block_kernel(
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
    );
    a.sum_d + a.ws_iw_mse.num
}

/// Bench/gate access to the era-2 kernel.
#[cfg(any(test, feature = "oracle"))]
#[allow(clippy::too_many_arguments)]
pub fn bench_dense_era2(
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
) -> f64 {
    let a = dense_block_kernel_era2(
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
    );
    a.sum_d + a.ws_iw_mse.num
}

/// Tier dispatch for the era-2 kernel. The BODY is one source shared by every
/// tier (see `dense_block_kernel_era2_inner`); `#[magetypes]` exists here only
/// to put that body inside each tier's `target_feature` region.
///
/// **This is what the first measurement was missing.** Plain Rust compiles to
/// baseline x86-64 (SSE2) no matter what the host supports, which measured
/// **4.4x slower** than era-1's `#[arcane]`-wrapped kernel — not a shape
/// problem, an ISA problem. Cross-tier bit-identity survives because the
/// source is identical and Rust does not contract `a*b+c` into an FMA without
/// fast-math (which is off); it is VERIFIED, not assumed, by
/// `era2_vendor_probe` across the AMD/Intel pair.
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_era2(
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
        dense_block_kernel_era2_inner(
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

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments, dead_code)]
fn dense_block_kernel_era2_inner(
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
    // The token is unused: this body is plain fixed-size-array arithmetic and
    // magetypes is here purely for the per-tier `target_feature` region.
    let _ = token;
    let mut acc = DenseAccum::default();
    // Row scratch for the pass-A -> pass-B handoff (d, art, det, mse).
    let mut sc_d = vec![0.0f32; width];
    let mut sc_art = vec![0.0f32; width];
    let mut sc_det = vec![0.0f32; width];
    let mut sc_mse = vec![0.0f32; width];

    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        // f64 band partials — one per accumulator slot.
        let mut band = [0.0f64; 35];

        for y in b0..b1 {
            let row = y * width;
            // ---- PASS A: terms + the 13 core families ----
            //
            // FIXED-SIZE CHUNKS. The first version of this loop used
            // `let n = (width - x).min(8)` with a runtime bound, and it
            // MEASURED 9-10x SLOWER than era-1 because LLVM cannot prove
            // `n == 8` and so refuses to vectorise. `as_chunks::<8>` hands the
            // body a `&[f32; 8]` whose length is a compile-time constant —
            // the workspace's fixed-size-array pattern, and the difference
            // between a kernel and a scalar loop.
            let mut l_d = Lanes8::zero();
            let mut l_d2 = Lanes8::zero();
            let mut l_d3 = Lanes8::zero();
            let mut l_d4 = Lanes8::zero();
            let mut l_art = Lanes8::zero();
            let mut l_det = Lanes8::zero();
            let mut l_mse = Lanes8::zero();
            let mut l_hfg = Lanes8::zero();
            let mut l_hfl = Lanes8::zero();
            let mut l_hfm = Lanes8::zero();
            let mut l_pj = Lanes8::zero();
            let mut l_pjl = Lanes8::zero();
            let mut l_pjh = Lanes8::zero();

            let rs = &src[row..row + width];
            let rd = &dst[row..row + width];
            let r1 = &mu1[row..row + width];
            let r2 = &mu2[row..row + width];
            let rq = &ssq[row..row + width];
            let r12 = &s12[row..row + width];
            let ra = &activity[row..row + width];

            let (cs, ts) = rs.as_chunks::<8>();
            let (cd, td) = rd.as_chunks::<8>();
            let (c1, t1) = r1.as_chunks::<8>();
            let (c2, t2) = r2.as_chunks::<8>();
            let (cq, tq) = rq.as_chunks::<8>();
            let (c12, t12) = r12.as_chunks::<8>();
            let (ca, ta) = ra.as_chunks::<8>();

            // One closure, used for the full chunks AND the tail, so the two
            // paths cannot drift. `n` is 8 for chunks and `width % 8` for the
            // tail; the chunk call sites pass fixed-size arrays so the bound
            // folds away there.
            let core_step = |n: usize,
                             s8: &[f32; 8],
                             d8: &[f32; 8],
                             m1: &[f32; 8],
                             m2: &[f32; 8],
                             q8: &[f32; 8],
                             p8: &[f32; 8],
                             a8: &[f32; 8],
                             out_x: usize,
                             l_d: &mut Lanes8,
                             l_d2: &mut Lanes8,
                             l_d3: &mut Lanes8,
                             l_d4: &mut Lanes8,
                             l_art: &mut Lanes8,
                             l_det: &mut Lanes8,
                             l_mse: &mut Lanes8,
                             l_hfg: &mut Lanes8,
                             l_hfl: &mut Lanes8,
                             l_hfm: &mut Lanes8,
                             l_pj: &mut Lanes8,
                             l_pjl: &mut Lanes8,
                             l_pjh: &mut Lanes8,
                             sc_d: &mut [f32],
                             sc_art: &mut [f32],
                             sc_det: &mut [f32],
                             sc_mse: &mut [f32]| {
                let mut t_d = [0.0f32; 8];
                let mut t_a = [0.0f32; 8];
                let mut t_t = [0.0f32; 8];
                let mut t_m = [0.0f32; 8];
                let mut t_hg = [0.0f32; 8];
                let mut t_hl = [0.0f32; 8];
                let mut t_hm = [0.0f32; 8];
                let mut t_pj = [0.0f32; 8];
                let mut t_pl = [0.0f32; 8];
                let mut t_ph = [0.0f32; 8];
                for k in 0..8 {
                    let (sv, dv) = (s8[k], d8[k]);
                    let (a1, a2) = (m1[k], m2[k]);
                    let (dd, art_i, det_i, mse_i) = e2_terms(sv, dv, a1, a2, p8[k], q8[k]);
                    t_d[k] = dd;
                    t_a[k] = art_i;
                    t_t[k] = det_i;
                    t_m[k] = mse_i;
                    let hf_src = sv - a1;
                    let hf_dst = dv - a2;
                    let (g, l) =
                        e2_bounded_excess_pair(hf_dst * hf_dst, hf_src * hf_src, C_HF as f32);
                    t_hg[k] = g;
                    t_hl[k] = l;
                    t_hm[k] = e2_bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF as f32);
                    let rae = (sv - dv).abs();
                    let act = a8[k];
                    t_pj[k] = e2_pjnd(rae, act, K_PJND_MASK as f32, C_PJND_CLAMP as f32);
                    t_pl[k] = e2_pjnd(rae, act, K_PJND_MASK_LOW as f32, C_PJND_CLAMP as f32);
                    t_ph[k] = e2_pjnd(rae, act, K_PJND_MASK_HIGH as f32, C_PJND_CLAMP as f32);
                }
                let mut t_d2 = [0.0f32; 8];
                let mut t_d3 = [0.0f32; 8];
                let mut t_d4 = [0.0f32; 8];
                for k in 0..8 {
                    let dd = t_d[k];
                    t_d2[k] = dd * dd;
                    t_d3[k] = dd * dd * dd;
                    t_d4[k] = dd * dd * dd * dd;
                }
                sc_d[out_x..out_x + n].copy_from_slice(&t_d[..n]);
                sc_art[out_x..out_x + n].copy_from_slice(&t_a[..n]);
                sc_det[out_x..out_x + n].copy_from_slice(&t_t[..n]);
                sc_mse[out_x..out_x + n].copy_from_slice(&t_m[..n]);
                l_d.add_tail(&t_d[..n]);
                l_d2.add_tail(&t_d2[..n]);
                l_d3.add_tail(&t_d3[..n]);
                l_d4.add_tail(&t_d4[..n]);
                l_art.add_tail(&t_a[..n]);
                l_det.add_tail(&t_t[..n]);
                l_mse.add_tail(&t_m[..n]);
                l_hfg.add_tail(&t_hg[..n]);
                l_hfl.add_tail(&t_hl[..n]);
                l_hfm.add_tail(&t_hm[..n]);
                l_pj.add_tail(&t_pj[..n]);
                if transducer_bank {
                    l_pjl.add_tail(&t_pl[..n]);
                    l_pjh.add_tail(&t_ph[..n]);
                }
            };

            for ci in 0..cs.len() {
                core_step(
                    8,
                    &cs[ci],
                    &cd[ci],
                    &c1[ci],
                    &c2[ci],
                    &cq[ci],
                    &c12[ci],
                    &ca[ci],
                    ci * 8,
                    &mut l_d,
                    &mut l_d2,
                    &mut l_d3,
                    &mut l_d4,
                    &mut l_art,
                    &mut l_det,
                    &mut l_mse,
                    &mut l_hfg,
                    &mut l_hfl,
                    &mut l_hfm,
                    &mut l_pj,
                    &mut l_pjl,
                    &mut l_pjh,
                    &mut sc_d,
                    &mut sc_art,
                    &mut sc_det,
                    &mut sc_mse,
                );
            }
            if !ts.is_empty() {
                // Pad the tail into fixed-size arrays so the SAME closure runs;
                // lanes beyond `n` are never accumulated or stored.
                let pad = |t: &[f32]| -> [f32; 8] {
                    let mut o = [0.0f32; 8];
                    o[..t.len()].copy_from_slice(t);
                    o
                };
                core_step(
                    ts.len(),
                    &pad(ts),
                    &pad(td),
                    &pad(t1),
                    &pad(t2),
                    &pad(tq),
                    &pad(t12),
                    &pad(ta),
                    cs.len() * 8,
                    &mut l_d,
                    &mut l_d2,
                    &mut l_d3,
                    &mut l_d4,
                    &mut l_art,
                    &mut l_det,
                    &mut l_mse,
                    &mut l_hfg,
                    &mut l_hfl,
                    &mut l_hfm,
                    &mut l_pj,
                    &mut l_pjl,
                    &mut l_pjh,
                    &mut sc_d,
                    &mut sc_art,
                    &mut sc_det,
                    &mut sc_mse,
                );
            }

            // ---- PASS B: the 11 weighted pools from the row scratch ----
            let (mut p_mw, mut p_iw) = (Lanes8::zero(), Lanes8::zero());
            let mut p_m = [Lanes8::zero(); 4];
            let mut p_i = [Lanes8::zero(); 4];
            let mut p_kn = [Lanes8::zero(); 3];
            let mut p_kd = [Lanes8::zero(); 3];
            {
                let mut pool_step = |n: usize,
                                     a8: &[f32; 8],
                                     d8: &[f32; 8],
                                     ar8: &[f32; 8],
                                     de8: &[f32; 8],
                                     ms8: &[f32; 8]| {
                    let mut mw = [0.0f32; 8];
                    let mut iw = [0.0f32; 8];
                    let mut mv = [[0.0f32; 8]; 4];
                    let mut iv = [[0.0f32; 8]; 4];
                    let mut kn = [[0.0f32; 8]; 3];
                    let mut kd = [[0.0f32; 8]; 3];
                    for k in 0..8 {
                        let sat = e2_saturate(a8[k], C_ACTIVITY as f32);
                        let m_w = 1.0 - sat;
                        let i_w = sat + IW_WEIGHT_FLOOR as f32;
                        mw[k] = m_w;
                        iw[k] = i_w;
                        let vals = [d8[k], ar8[k], de8[k], ms8[k]];
                        for j in 0..4 {
                            mv[j][k] = m_w * vals[j];
                            iv[j][k] = i_w * vals[j];
                        }
                        for j in 0..3 {
                            let sal = e2_saturate(vals[j], C_PEAK as f32);
                            kn[j][k] = sal * vals[j];
                            kd[j][k] = sal;
                        }
                    }
                    p_mw.add_tail(&mw[..n]);
                    p_iw.add_tail(&iw[..n]);
                    for j in 0..4 {
                        p_m[j].add_tail(&mv[j][..n]);
                        p_i[j].add_tail(&iv[j][..n]);
                    }
                    for j in 0..3 {
                        p_kn[j].add_tail(&kn[j][..n]);
                        p_kd[j].add_tail(&kd[j][..n]);
                    }
                };
                let (ka, _) = ra.as_chunks::<8>();
                let (kd_, _) = sc_d[..width].as_chunks::<8>();
                let (kr, _) = sc_art[..width].as_chunks::<8>();
                let (ke, _) = sc_det[..width].as_chunks::<8>();
                let (km, _) = sc_mse[..width].as_chunks::<8>();
                for ci in 0..ka.len() {
                    pool_step(8, &ka[ci], &kd_[ci], &kr[ci], &ke[ci], &km[ci]);
                }
                let rem = width % 8;
                if rem != 0 {
                    let base = width - rem;
                    let pad = |t: &[f32]| -> [f32; 8] {
                        let mut o = [0.0f32; 8];
                        o[..t.len()].copy_from_slice(t);
                        o
                    };
                    pool_step(
                        rem,
                        &pad(&ra[base..]),
                        &pad(&sc_d[base..width]),
                        &pad(&sc_art[base..width]),
                        &pad(&sc_det[base..width]),
                        &pad(&sc_mse[base..width]),
                    );
                }
            }

            // ---- per-row reduction into the f64 band partial ----
            let rowvals = [
                l_d, l_d2, l_d3, l_d4, l_art, l_det, l_mse, l_hfg, l_hfl, l_hfm, l_pj, l_pjl, l_pjh,
            ];
            for (b, v) in band[..13].iter_mut().zip(rowvals.iter()) {
                *b += v.reduce();
            }
            // peak num/den (ssim, art, det), then mask (4 num + shared den),
            // then iw (4 num + shared den) — the DenseAccum slot order.
            for j in 0..3 {
                band[13 + j * 2] += p_kn[j].reduce();
                band[14 + j * 2] += p_kd[j].reduce();
            }
            let mw_row = p_mw.reduce();
            let iw_row = p_iw.reduce();
            for j in 0..4 {
                band[19 + j * 2] += p_m[j].reduce();
                band[20 + j * 2] += mw_row;
                band[27 + j * 2] += p_i[j].reduce();
                band[28 + j * 2] += iw_row;
            }
        }

        // ---- band partials merge in BAND INDEX ORDER ----
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (j, w) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            j.num += band[w];
            j.den += band[w + 1];
        }
        b0 = b1;
    }
    acc
}
