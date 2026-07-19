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

/// Box blur radius at scale 0 — matches v1's `ZensimConfig::default()`.
const BLUR_RADIUS: usize = 5;
/// Oriented-blockiness lattice period (JPEG's 8x8 MCU grid).
const BLOCK_LATTICE: usize = 8;

// ============================================================================
// Bounded per-pixel formulas
// ============================================================================

/// Per-pixel SSIM-like dissimilarity, C1-bounded (D1 fix). Bounded [0, 2].
#[inline]
fn ssim_d_local(mu1: f64, mu2: f64, s12: f64, ssq: f64) -> f64 {
    let num_m = (2.0 * mu1 * mu2 + C1_V2) / (mu1 * mu1 + mu2 * mu2 + C1_V2);
    let cov = s12 - mu1 * mu2;
    let num_s = 2.0 * cov + C2_V2;
    let denom_s = ssq - mu1 * mu1 - mu2 * mu2 + C2_V2;
    let local = num_m * (num_s / denom_s);
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

/// Michelson-contrast-style saturating ratio `x/(x+c)`, bounded `[0, 1)`
/// for `x >= 0`.
#[inline]
fn saturate(x: f64, c: f64) -> f64 {
    let x = x.max(0.0);
    x / (x + c)
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

/// Vectorized [`ssim_d_local`] — identical algebra, f32 lanes.
#[inline]
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
    let num_m = (two * mu1 * mu2 + c1) / (mu1 * mu1 + mu2 * mu2 + c1);
    let cov = s12 - mu1 * mu2;
    let num_s = two * cov + c2;
    let denom_s = ssq - mu1 * mu1 - mu2 * mu2 + c2;
    let local = num_m * (num_s / denom_s);
    (one - local).max(zero)
}

/// Vectorized [`bounded_sim`] — `(2ab+c)/(a²+b²+c)`.
#[inline]
fn bounded_sim_v<T: F32x8Backend + Copy>(token: T, a: V8<T>, b: V8<T>, c: V8<T>) -> V8<T> {
    let two = V8::<T>::splat(token, 2.0);
    (two * a * b + c) / (a * a + b * b + c)
}

/// Vectorized [`bounded_excess`] — `max(0, a-b) / (a+b+c)`.
#[inline]
fn bounded_excess_v<T: F32x8Backend + Copy>(token: T, a: V8<T>, b: V8<T>, c: V8<T>) -> V8<T> {
    let zero = V8::<T>::zero(token);
    (a - b).max(zero) / (a + b + c)
}

/// Vectorized [`saturate`] — `max(x,0)/(max(x,0)+c)`.
#[inline]
fn saturate_v<T: F32x8Backend + Copy>(token: T, x: V8<T>, c: V8<T>) -> V8<T> {
    let zero = V8::<T>::zero(token);
    let x = x.max(zero);
    x / (x + c)
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
    pub fn view(&self) -> FeatureViewV2<'_> {
        FeatureViewV2::new(&self.features, self.n_scales)
            .expect("compute_v2_features always emits the v2-total layout for its own n_scales")
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
}
impl Default for V2NewFeatureToggles {
    fn default() -> Self {
        Self {
            gradient_features: true,
            transducer_bank: true,
            blockiness: true,
        }
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
struct ScratchV2 {
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
impl ScratchV2 {
    fn new(max_n: usize) -> Self {
        Self {
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
#[inline]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_generic<T: F32x8Backend + Copy>(
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
            r_hfg += bounded_excess_v(token, hf_dst_sq, hf_src_sq, c_hf);
            r_hfl += bounded_excess_v(token, hf_src_sq, hf_dst_sq, c_hf);
            r_hfm += bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);

            let raw_abs_err = raw_diff.abs();
            let t_mid = raw_abs_err / (one + k_mid * act);
            r_pjnd += saturate_v(token, t_mid, c_pjnd_clamp);
            if transducer_bank {
                let t_lo = raw_abs_err / (one + k_lo * act);
                let t_hi = raw_abs_err / (one + k_hi * act);
                r_pjnd_lo += saturate_v(token, t_lo, c_pjnd_clamp);
                r_pjnd_hi += saturate_v(token, t_hi, c_pjnd_clamp);
            }

            // §A.14 register-pressure fix (see the row-header comment
            // above): extract this chunk's d/art_i/det_i/mse_i/act lanes and
            // accumulate the 11 weighted-pool pairs scalar, per lane --
            // reuses the EXACT SAME formula as the scalar tail below via
            // `weighted_pool_accumulate_scalar`, not a re-derivation.
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
        // (weighted-pool accumulators already folded in scalar, per-lane,
        // inside the chunk loop above -- see the §A.14 fix note.)

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
            acc.sum_hf_gain += bounded_excess(hf_dst_sq, hf_src_sq, C_HF);
            acc.sum_hf_loss += bounded_excess(hf_src_sq, hf_dst_sq, C_HF);
            acc.sum_hf_mag_loss += bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            let raw_abs_err = (s - dd).abs();
            let t_mid = raw_abs_err / (1.0 + K_PJND_MASK * act);
            acc.sum_pjnd += saturate(t_mid, C_PJND_CLAMP);
            if transducer_bank {
                let t_lo = raw_abs_err / (1.0 + K_PJND_MASK_LOW * act);
                let t_hi = raw_abs_err / (1.0 + K_PJND_MASK_HIGH * act);
                acc.sum_pjnd_lo += saturate(t_lo, C_PJND_CLAMP);
                acc.sum_pjnd_hi += saturate(t_hi, C_PJND_CLAMP);
            }

            weighted_pool_accumulate_scalar(&mut acc, d, art_i, det_i, mse_i, act);
        }
    }

    acc
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
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
    dense_block_kernel_generic(
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
    sum_ringing: f64,
    sum_banding: f64,
    sum_grad_src: f64,
    sum_grad_dst: f64,
}

/// Gradient-block SIMD kernel body. Interior pixels (`1..width-1`,
/// `1..height-1`) are vectorized via SHIFTED unaligned loads for the
/// x-neighbors (`row[x-1..x+7]` / `row[x+1..x+9]`) and full-row-offset
/// loads for the y-neighbors (`row_u[x..x+8]` / `row_d[x..x+8]`) — the
/// same shifted-window-read pattern as `~/work/archmage/docs/site/content/
/// magetypes/examples/gaussian-blur.md`'s vertical pass. The single-pixel
/// border (`x==0`, `x==width-1`, `y==0`, `y==height-1`) uses the scalar
/// reflect-boundary formulas (`saturating_sub`/`.min(width-1)`) exactly as
/// before — a tiny fraction of pixels (2 columns + 2 rows out of
/// width*height), not worth the complexity of a SIMD boundary-clamp.
#[inline]
fn gradient_block_kernel_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
) -> GradientAccum {
    let zero = V8::<T>::zero(token);
    let one = V8::<T>::splat(token, 1.0);
    let c_gms = V8::<T>::splat(token, C_GMS as f32);
    let c_ring_err = V8::<T>::splat(token, C_RING_ERR as f32);
    let c_activity = V8::<T>::splat(token, C_ACTIVITY as f32);
    let c_ring_edge = V8::<T>::splat(token, C_RING_EDGE as f32);
    let c_band_dst = V8::<T>::splat(token, C_BAND_DST as f32);
    let c_band_src = V8::<T>::splat(token, C_BAND_SRC as f32);

    let mut acc = GradientAccum::default();

    // Scalar helper for one pixel — used for the whole first/last row
    // (y==0, y==height-1) and the first/last column of every interior
    // row, mirroring compute_channel_scale_v2's original reflect-boundary
    // neighbor logic exactly.
    let scalar_pixel = |x: usize, y: usize, acc: &mut GradientAccum| {
        let yu = y.saturating_sub(1);
        let yd = (y + 1).min(height - 1);
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(width - 1);
        let row = y * width;
        let row_u = yu * width;
        let row_d = yd * width;
        let i = row + x;
        let s = src[i] as f64;
        let dd = dst[i] as f64;
        let act = activity[i] as f64;
        let sxl = src[row + xl] as f64;
        let sxr = src[row + xr] as f64;
        let syu = src[row_u + x] as f64;
        let syd = src[row_d + x] as f64;
        let dxl = dst[row + xl] as f64;
        let dxr = dst[row + xr] as f64;
        let dyu = dst[row_u + x] as f64;
        let dyd = dst[row_d + x] as f64;

        let gx_src = sxr - sxl;
        let gy_src = syd - syu;
        let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
        let gx_dst = dxr - dxl;
        let gy_dst = dyd - dyu;
        let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

        acc.sum_grad_src += grad_src_mag;
        acc.sum_grad_dst += grad_dst_mag;
        acc.sum_gms += 1.0 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS);

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
        if y == 0 || y == height - 1 {
            for x in 0..width {
                scalar_pixel(x, y, &mut acc);
            }
            continue;
        }
        let row = y * width;
        let row_u = (y - 1) * width;
        let row_d = (y + 1) * width;

        scalar_pixel(0, y, &mut acc);
        if width > 2 {
            let interior_end = width - 1;
            let interior_w = interior_end - 1; // pixels [1, width-2]
            let chunk_end = 1 + interior_w - (interior_w % 8);

            let (mut r_gms, mut r_ring, mut r_band) = (zero, zero, zero);
            let (mut r_gsrc, mut r_gdst) = (zero, zero);

            let mut x = 1usize;
            while x < chunk_end {
                macro_rules! ld_at {
                    ($plane:expr, $off:expr) => {
                        V8::<T>::from_array(token, $plane[$off..$off + 8].try_into().unwrap())
                    };
                }
                let sxl = ld_at!(src, row + x - 1);
                let sxr = ld_at!(src, row + x + 1);
                let syu = ld_at!(src, row_u + x);
                let syd = ld_at!(src, row_d + x);
                let dxl = ld_at!(dst, row + x - 1);
                let dxr = ld_at!(dst, row + x + 1);
                let dyu = ld_at!(dst, row_u + x);
                let dyd = ld_at!(dst, row_d + x);
                let s = ld_at!(src, row + x);
                let dd = ld_at!(dst, row + x);
                let act = ld_at!(activity, row + x);

                let gx_src = sxr - sxl;
                let gy_src = syd - syu;
                let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                let gx_dst = dxr - dxl;
                let gy_dst = dyd - dyu;
                let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                r_gsrc += grad_src_mag;
                r_gdst += grad_dst_mag;
                r_gms += one - bounded_sim_v(token, grad_src_mag, grad_dst_mag, c_gms);

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
fn blockiness_sparse(src: &[f32], dst: &[f32], width: usize, height: usize) -> f64 {
    let mut sum = 0.0f64;
    // Vertical steps: for every column x that's a lattice boundary, walk
    // down every row y (all y, since the horizontal-step term at column x
    // fires for every row -- matches the original `x % LATTICE == 0 && x >
    // 0` condition, which held for all y).
    let mut x = BLOCK_LATTICE;
    while x < width {
        for y in 0..height {
            let i = y * width + x;
            let step_dst = (dst[i] as f64 - dst[i - 1] as f64).abs();
            let step_src = (src[i] as f64 - src[i - 1] as f64).abs();
            sum += bounded_excess(step_dst, step_src, C_BLOCK);
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
            sum += bounded_excess(step_dst, step_src, C_BLOCK);
        }
        y += BLOCK_LATTICE;
    }
    sum
}

/// Compute all [`FEATURES_PER_CHANNEL_V2_TOTAL`] v2 signals for one
/// channel at one scale (except [`idx::EDGE_WIDTH_CHANGE`], filled in by
/// the caller once the adjacent scale is known — see
/// [`compute_v2_features_impl`]). Writes into `out`. Returns
/// `(mean_grad_src, mean_grad_dst)` for the caller's cross-scale
/// edge-width computation (`(0,0)` when `toggles.gradient_features` is
/// off — the caller must not rely on edge-width in that case).
///
/// Phase-4 (§A.14): the per-pixel pass is now TWO magetypes-SIMD kernels
/// (`dense_block_kernel` always, `gradient_block_kernel` when
/// `toggles.gradient_features`) plus a sparse scalar blockiness pass when
/// `toggles.blockiness` — see the kernels' own doc comments above for the
/// full architecture. This function is now orchestration: call the
/// kernels, unpack their accumulators into `out`.
fn compute_channel_scale_v2(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
    scratch: &mut ScratchV2,
    out: &mut [f64],
) -> (f64, f64) {
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");
    assert_eq!(
        out.len(),
        FEATURES_PER_CHANNEL_V2_TOTAL,
        "out slice must hold exactly one channel-scale's v2 block"
    );
    assert!(
        n <= scratch.mu1.len(),
        "scratch buffers must be sized for the largest scale (scale 0)"
    );

    // --- Blur-pass arrays (inherently multi-pixel; matches v1's own
    //     architecture, which also carries mu1/mu2/s12/ssq/activity as
    //     full-image arrays ahead of its per-pixel kernel).
    //
    //     Perf pass (§A.13) MEASUREMENT NOTE: this box was under real,
    //     observed concurrent load from 10+ OTHER agent sessions while this
    //     was benchmarked (`uptime`/`ps aux` confirmed it mid-investigation)
    //     — an initial reading mis-flagged `fused_blur_h_ssim` (below) as a
    //     REGRESSION by comparing it against an *inferred* pre-perf-pass
    //     number (phase-2's 4.06x ratio x this session's v1 reading) rather
    //     than a directly-measured one. A same-day A/B against the
    //     alternative (separate `mul_into`/`sq_sum_into` + 4 simple
    //     `box_blur_1pass_into` calls, matching iteration-2's original
    //     structure) — 2 process launches per side, `v2_speed_baseline
    //     --group=1024x1024` — put fused_blur_h_ssim at 278-282ms and the
    //     separate-calls alternative at 310-331ms EVERY time, despite the
    //     shared v1 baseline itself drifting 66.6-70.5ms run to run. Fused
    //     wins by ~10-15% consistently regardless of ambient noise level, so
    //     it is the one kept. `fused_blur_h_ssim` is v1's OWN public fused
    //     H-blur primitive (H-blurred mu1/mu2/Σ(s²+d²)/Σ(s·d) in one SIMD
    //     pass, called directly by `streaming.rs::process_strip_channel` for
    //     the identical algebra) — reusing it here is not a v1 code change,
    //     zero risk to the v1 byte-identity golden gate. ---
    let mu1_h = &mut scratch.mu1_h[..n];
    let mu2_h = &mut scratch.mu2_h[..n];
    let ssq_h = &mut scratch.ssq_h[..n];
    let s12_h = &mut scratch.s12_h[..n];
    crate::blur::fused_blur_h_ssim(
        src,
        dst,
        mu1_h,
        mu2_h,
        ssq_h,
        s12_h,
        width,
        height,
        BLUR_RADIUS,
    );

    let mu1 = &mut scratch.mu1[..n];
    crate::blur::box_blur_v_from_copy(mu1_h, mu1, width, height, BLUR_RADIUS);
    let mu2 = &mut scratch.mu2[..n];
    crate::blur::box_blur_v_from_copy(mu2_h, mu2, width, height, BLUR_RADIUS);
    let ssq = &mut scratch.ssq[..n];
    crate::blur::box_blur_v_from_copy(ssq_h, ssq, width, height, BLUR_RADIUS);
    let s12 = &mut scratch.s12[..n];
    crate::blur::box_blur_v_from_copy(s12_h, s12, width, height, BLUR_RADIUS);

    let abs_src = &mut scratch.abs_src[..n];
    crate::simd_ops::abs_diff_into(src, mu1, abs_src);
    let activity = &mut scratch.activity[..n];
    crate::blur::box_blur_1pass_into(
        abs_src,
        activity,
        &mut scratch.activity_tmp[..n],
        width,
        height,
        BLUR_RADIUS,
    );
    // `mu1`/`mu2`/`ssq`/`s12`/`activity` are `&mut [f32]` from the writes
    // above; the per-pixel pass below only reads them (`mu1[i]` etc.),
    // which works directly through a `&mut` binding — no reborrow needed.

    // --- Phase-4 (§A.14): the dense block ALWAYS runs (mirrors the old
    //     unconditional per-pixel block); the gradient block and
    //     blockiness pass are gated exactly as before, just as separate
    //     kernel calls instead of inline per-pixel branches. ---
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
        gradient_block_kernel(src, dst, activity, width, height)
    } else {
        GradientAccum::default()
    };

    let sum_blockiness = if toggles.blockiness {
        blockiness_sparse(src, dst, width, height)
    } else {
        0.0
    };

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

    out[idx::SSIM_MEAN] = mean_d;
    out[idx::SSIM_DEV2] = dev2;
    out[idx::SSIM_DEV4] = dev4;
    out[idx::ART] = dense.sum_art / n_f;
    out[idx::DET] = dense.sum_det / n_f;
    out[idx::MSE] = dense.sum_mse / n_f;
    out[idx::HF_GAIN] = dense.sum_hf_gain / n_f;
    out[idx::HF_LOSS] = dense.sum_hf_loss / n_f;
    out[idx::HF_MAG_LOSS] = dense.sum_hf_mag_loss / n_f;
    out[idx::SSIM_SOFT_PEAK] = dense.ws_peak_ssim.finish();
    out[idx::ART_SOFT_PEAK] = dense.ws_peak_art.finish();
    out[idx::DET_SOFT_PEAK] = dense.ws_peak_det.finish();
    out[idx::MASKED_SSIM] = dense.ws_mask_ssim.finish();
    out[idx::MASKED_ART] = dense.ws_mask_art.finish();
    out[idx::MASKED_DET] = dense.ws_mask_det.finish();
    out[idx::MASKED_MSE] = dense.ws_mask_mse.finish();
    out[idx::IW_SSIM] = dense.ws_iw_ssim.finish();
    out[idx::IW_ART] = dense.ws_iw_art.finish();
    out[idx::IW_DET] = dense.ws_iw_det.finish();
    out[idx::IW_MSE] = dense.ws_iw_mse.finish();
    out[idx::PJND_TRANSDUCER] = dense.sum_pjnd / n_f;
    out[idx::PJND_FRAGILITY] = 1.0 - saturate(grad.sum_grad_src / n_f, C_PJND_GRAD);
    out[idx::GMS] = grad.sum_gms / n_f;
    out[idx::PJND_TRANSDUCER_LOW_K] = dense.sum_pjnd_lo / n_f;
    out[idx::PJND_TRANSDUCER_HIGH_K] = dense.sum_pjnd_hi / n_f;
    out[idx::BLOCKINESS] = sum_blockiness / n_f;
    out[idx::RINGING] = grad.sum_ringing / n_f;
    out[idx::BANDING] = grad.sum_banding / n_f;
    out[idx::EDGE_WIDTH_CHANGE] = 0.0; // filled in by the caller (needs the adjacent scale)

    (grad.sum_grad_src / n_f, grad.sum_grad_dst / n_f)
}

// ============================================================================
// Top-level entry point
// ============================================================================

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
pub(crate) fn compute_v2_features_impl_with_toggles(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    toggles: V2NewFeatureToggles,
) -> Result<ZensimV2Result, ZensimError> {
    crate::metric::validate_pair(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;

    let padded_src = crate::metric::reflect_pad_to_min(source);
    let padded_dst = crate::metric::reflect_pad_to_min(distorted);
    let mut width = padded_src.width();
    let mut height = padded_src.height();

    let mut src_planes = crate::streaming::convert_source_to_xyb(&padded_src, width, parallel);
    let mut dst_planes = crate::streaming::convert_source_to_xyb(&padded_dst, width, parallel);

    let n_scales = crate::NUM_SCALES;
    let mut features = vec![0.0f64; n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL];
    // Per-channel (mean_grad_src, mean_grad_dst) from the previous
    // (finer) scale, for the edge-width-change cross-scale comparison.
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    // Perf pass (§A.13): one scratch buffer set PER CHANNEL, sized once for
    // the largest (scale-0) `n`, reused across all `n_scales` calls for
    // that channel. 3 separate sets (rather than 1 reused across channels
    // too) so the 3 channels within a scale can run independently — see
    // the `threads` parallel branch below, which needs disjoint `&mut`
    // scratch per closure.
    let n0 = width * height;
    let mut scratch: [ScratchV2; 3] = [ScratchV2::new(n0), ScratchV2::new(n0), ScratchV2::new(n0)];

    for scale in 0..n_scales {
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;

        // Each channel's compute is fully independent WITHIN a scale (only
        // `prev_grad[ch]` from the SAME channel's earlier scale is read,
        // below, after all 3 channels finish, and each channel owns its
        // own scratch set) -- so the 3-channel fan-out can run in
        // parallel when both the `threads` feature and the caller's
        // `parallel` flag are active. Both branches write into the same
        // `features[scale_base..]` chunks; only the iteration strategy
        // differs.
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
                    compute_channel_scale_v2(
                        &src_planes[ch],
                        &dst_planes[ch],
                        width,
                        height,
                        toggles,
                        scr,
                        out,
                    )
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
                    &mut scratch[ch],
                    out,
                );
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
            // the per-channel loop. Updating them mid-loop (as an earlier
            // draft of this function did) downscales channels 1/2 from
            // channel 0's ALREADY-HALVED dimensions, corrupting the
            // pyramid (caught by this file's own test suite: "src plane
            // length must be width*height" panics on every test that
            // exercises scale > 0).
            let mut new_wh = (width, height);
            for ch in 0..3 {
                new_wh = crate::blur::downscale_2x_inplace(&mut src_planes[ch], width, height);
                crate::blur::downscale_2x_inplace(&mut dst_planes[ch], width, height);
            }
            width = new_wh.0;
            height = new_wh.1;
        }
    }

    Ok(ZensimV2Result {
        features,
        n_scales,
        regime: FeatureRegime::V2Bounded,
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
        let h = 152; // deliberately non-power-of-2, exercises reflect padding too
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
}
