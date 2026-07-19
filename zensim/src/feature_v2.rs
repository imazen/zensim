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

/// Terriberry (2007) single-pass online moments: tracks `n`, `mean`, and
/// the 2nd/3rd/4th central-moment sums `M2/M3/M4 = Σ(x-mean)^k`. `M3` is
/// tracked only because `M4`'s incremental update formula depends on its
/// PRE-update value — not used as an output here. This is what makes
/// `ssim_dev2`/`ssim_dev4` (the D8 GMSD-style deviation moments) genuinely
/// single-pass and O(1)-space, unlike iteration 1's store-then-revisit
/// design.
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

/// Compute all [`FEATURES_PER_CHANNEL_V2_TOTAL`] v2 signals for one
/// channel at one scale (except [`idx::EDGE_WIDTH_CHANGE`], filled in by
/// the caller once the adjacent scale is known — see
/// [`compute_v2_features_impl`]). Writes into `out`. Returns
/// `(mean_grad_src, mean_grad_dst)` for the caller's cross-scale
/// edge-width computation (`(0,0)` when `toggles.gradient_features` is
/// off — the caller must not rely on edge-width in that case).
///
/// One `for y { for x { ... } }` pass: every basic/peak/masked/IW/PJND/new
/// signal is an O(1) running accumulator (Terriberry moments for the SSIM
/// deviation terms, running weighted-sum pairs for masked/IW/soft-peak,
/// plain running sums for everything else) — see the module doc.
fn compute_channel_scale_v2(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    toggles: V2NewFeatureToggles,
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

    // --- Blur-pass arrays (inherently multi-pixel; matches v1's own
    //     architecture, which also carries mu1/mu2/s12/ssq/activity as
    //     full-image arrays ahead of its per-pixel kernel). ---
    let mut tmp = vec![0.0f32; n];
    let mut mu1 = vec![0.0f32; n];
    let mut mu2 = vec![0.0f32; n];
    crate::blur::box_blur_1pass_into(src, &mut mu1, &mut tmp, width, height, BLUR_RADIUS);
    crate::blur::box_blur_1pass_into(dst, &mut mu2, &mut tmp, width, height, BLUR_RADIUS);

    let mut prod = vec![0.0f32; n];
    crate::simd_ops::mul_into(src, dst, &mut prod);
    let mut s12 = vec![0.0f32; n];
    crate::blur::box_blur_1pass_into(&prod, &mut s12, &mut tmp, width, height, BLUR_RADIUS);

    let mut ssq_in = vec![0.0f32; n];
    crate::simd_ops::sq_sum_into(src, dst, &mut ssq_in);
    let mut ssq = vec![0.0f32; n];
    crate::blur::box_blur_1pass_into(&ssq_in, &mut ssq, &mut tmp, width, height, BLUR_RADIUS);

    let mut abs_src = vec![0.0f32; n];
    crate::simd_ops::abs_diff_into(src, &mu1, &mut abs_src);
    let mut activity = vec![0.0f32; n];
    crate::blur::box_blur_1pass_into(
        &abs_src,
        &mut activity,
        &mut tmp,
        width,
        height,
        BLUR_RADIUS,
    );
    drop(tmp);
    drop(prod);
    drop(ssq_in);
    drop(abs_src);

    // --- O(1) accumulators (this is the whole point of iteration 2: NO
    //     per-pixel Vec<f32> maps for the signal family below). ---
    let mut ssim_moments = OnlineMoments::default();
    let (mut sum_art, mut sum_det, mut sum_mse) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sum_hf_gain, mut sum_hf_loss, mut sum_hf_mag_loss) = (0.0f64, 0.0f64, 0.0f64);
    let (mut sum_pjnd, mut sum_pjnd_lo, mut sum_pjnd_hi) = (0.0f64, 0.0f64, 0.0f64);
    let mut sum_gms = 0.0f64;
    let mut sum_ringing = 0.0f64;
    let mut sum_banding = 0.0f64;
    let mut sum_blockiness = 0.0f64;
    let mut sum_grad_src = 0.0f64;
    let mut sum_grad_dst = 0.0f64;

    let mut ws_peak_ssim = WeightedSum::default();
    let mut ws_peak_art = WeightedSum::default();
    let mut ws_peak_det = WeightedSum::default();
    let mut ws_mask_ssim = WeightedSum::default();
    let mut ws_mask_art = WeightedSum::default();
    let mut ws_mask_det = WeightedSum::default();
    let mut ws_mask_mse = WeightedSum::default();
    let mut ws_iw_ssim = WeightedSum::default();
    let mut ws_iw_art = WeightedSum::default();
    let mut ws_iw_det = WeightedSum::default();
    let mut ws_iw_mse = WeightedSum::default();

    for y in 0..height {
        let yu = y.saturating_sub(1);
        let yd = (y + 1).min(height - 1);
        let row = y * width;
        let row_u = yu * width;
        let row_d = yd * width;
        for x in 0..width {
            let i = row + x;
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(width - 1);

            let s = src[i] as f64;
            let dd = dst[i] as f64;
            let m1 = mu1[i] as f64;
            let m2 = mu2[i] as f64;
            let act = activity[i] as f64;

            // --- SSIM dissimilarity (D1 fix) + Terriberry moments (D8 fix) ---
            let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64);
            ssim_moments.update(d);

            // --- Edge artifact/detail (D3 fix) ---
            let diff_src = (s - m1).abs();
            let diff_dst = (dd - m2).abs();
            let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
            let (mut art_i, mut det_i) = (0.0, 0.0);
            if diff_dst > diff_src {
                art_i = edge_dissim;
            } else if diff_dst < diff_src {
                det_i = edge_dissim;
            }
            sum_art += art_i;
            sum_det += det_i;

            // --- Bounded MSE (D6 partial fix) ---
            let raw_sq_err = (s - dd) * (s - dd);
            let mse_i = saturate(raw_sq_err, C_MSE);
            sum_mse += mse_i;

            // --- HF gain/loss/mag-loss (D2 fix, made genuinely per-pixel) ---
            let hf_src = s - m1;
            let hf_dst = dd - m2;
            let hf_src_sq = hf_src * hf_src;
            let hf_dst_sq = hf_dst * hf_dst;
            sum_hf_gain += bounded_excess(hf_dst_sq, hf_src_sq, C_HF);
            sum_hf_loss += bounded_excess(hf_src_sq, hf_dst_sq, C_HF);
            sum_hf_mag_loss += bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF);

            // --- PJND transducer bank (candidate: masking-transducer
            //     bank — 2 extra k values beyond the core K_PJND_MASK,
            //     which is iteration 1, always on). ---
            let raw_abs_err = (s - dd).abs();
            let t_mid = raw_abs_err / (1.0 + K_PJND_MASK * act);
            sum_pjnd += saturate(t_mid, C_PJND_CLAMP);
            if toggles.transducer_bank {
                let t_lo = raw_abs_err / (1.0 + K_PJND_MASK_LOW * act);
                let t_hi = raw_abs_err / (1.0 + K_PJND_MASK_HIGH * act);
                sum_pjnd_lo += saturate(t_lo, C_PJND_CLAMP);
                sum_pjnd_hi += saturate(t_hi, C_PJND_CLAMP);
            }

            // --- Neighbor loads (raw pixel values; shared by the gradient
            //     group AND blockiness, kept unconditional since they're
            //     cheap array reads with no sqrt/div — the EXPENSIVE part
            //     gated below is the sqrt-based magnitude + its dependent
            //     accumulators). ---
            if toggles.gradient_features || toggles.blockiness {
                let sxl = src[row + xl] as f64;
                let sxr = src[row + xr] as f64;
                let syu = src[row_u + x] as f64;
                let syd = src[row_d + x] as f64;
                let dxl = dst[row + xl] as f64;
                let dxr = dst[row + xr] as f64;
                let dyu = dst[row_u + x] as f64;
                let dyd = dst[row_d + x] as f64;

                // --- Gradients (raw pixel central differences; shared by
                //     GMS, chroma-edge-GMS [free, via channel axis],
                //     ringing, banding, edge-width). ---
                if toggles.gradient_features {
                    let gx_src = sxr - sxl;
                    let gy_src = syd - syu;
                    let grad_src_mag = (gx_src * gx_src + gy_src * gy_src).sqrt();
                    let gx_dst = dxr - dxl;
                    let gy_dst = dyd - dyu;
                    let grad_dst_mag = (gx_dst * gx_dst + gy_dst * gy_dst).sqrt();

                    sum_grad_src += grad_src_mag;
                    sum_grad_dst += grad_dst_mag;

                    // --- GMS (candidate 1 + 3) ---
                    sum_gms += 1.0 - bounded_sim(grad_src_mag, grad_dst_mag, C_GMS);

                    // --- Ringing (candidate: dilated-edge form). `activity`
                    //     (a BLURRED/spread local-energy signal) stands in
                    //     for `dilate(edge_r)` -- justified because
                    //     box-blur inherently spreads a sharp edge's
                    //     influence over `BLUR_RADIUS` pixels, giving a
                    //     "near a strong edge" halo without a separate
                    //     morphological dilation pass. `edge_r` itself uses
                    //     the SHARP (undilated) source gradient, matching
                    //     the A.10 form `err · dilate(edge_r) · (1−edge_r)`. ---
                    let err_b = saturate(raw_abs_err, C_RING_ERR);
                    let act_b = saturate(act, C_ACTIVITY);
                    let edge_r = saturate(grad_src_mag, C_RING_EDGE);
                    sum_ringing += err_b * act_b * (1.0 - edge_r);

                    // --- Banding (candidate: cheap fused approximation of
                    //     CAMBI's step-energy-in-low-gradient-regions
                    //     signature — see
                    //     docs/FEATURE_V2_SPEC_2026-07-18.md §A.11 for the
                    //     honest comparison against real CAMBI). MUST be a
                    //     genuine dst-vs-src COMPARISON (bounded_excess),
                    //     not dst's saturated edge alone — a real content
                    //     edge (present identically in both src and dst)
                    //     must NOT register as banding.
                    //     `bounded_excess(grad_dst, grad_src, c) == 0`
                    //     whenever `grad_dst <= grad_src` (identity-safe by
                    //     construction), restricted to originally-smooth
                    //     regions via `src_smooth_b`. ---
                    let edge_excess = bounded_excess(grad_dst_mag, grad_src_mag, C_BAND_DST);
                    let src_smooth_b = 1.0 - saturate(grad_src_mag, C_BAND_SRC);
                    sum_banding += edge_excess * src_smooth_b;
                }

                // --- Oriented blockiness (candidate). FR-ized: only EXCESS
                //     step energy vs the reference's own step at the same
                //     8-pixel-lattice position counts (a real content edge
                //     that happens to land on the lattice contributes ~0). ---
                if toggles.blockiness {
                    let mut block_i = 0.0f64;
                    if x % BLOCK_LATTICE == 0 && x > 0 {
                        let step_dst = (dst[i] as f64 - dxl).abs();
                        let step_src = (s - sxl).abs();
                        block_i += bounded_excess(step_dst, step_src, C_BLOCK);
                    }
                    if y % BLOCK_LATTICE == 0 && y > 0 {
                        let step_dst = (dst[i] as f64 - dyu).abs();
                        let step_src = (s - syu).abs();
                        block_i += bounded_excess(step_dst, step_src, C_BLOCK);
                    }
                    sum_blockiness += block_i;
                }
            }

            // --- Weighted pooling (masked / IW / soft-peak), inline —
            //     needs d/art_i/det_i/mse_i (just computed above) and
            //     activity (already available). ---
            let mask_w = 1.0 - saturate(act, C_ACTIVITY);
            let iw_w = saturate(act, C_ACTIVITY) + IW_WEIGHT_FLOOR;
            ws_mask_ssim.add(mask_w, d);
            ws_mask_art.add(mask_w, art_i);
            ws_mask_det.add(mask_w, det_i);
            ws_mask_mse.add(mask_w, mse_i);
            ws_iw_ssim.add(iw_w, d);
            ws_iw_art.add(iw_w, art_i);
            ws_iw_det.add(iw_w, det_i);
            ws_iw_mse.add(iw_w, mse_i);

            let sal_ssim = saturate(d, C_PEAK);
            let sal_art = saturate(art_i, C_PEAK);
            let sal_det = saturate(det_i, C_PEAK);
            ws_peak_ssim.add(sal_ssim, d);
            ws_peak_art.add(sal_art, art_i);
            ws_peak_det.add(sal_det, det_i);
        }
    }

    let n_f = n as f64;
    let (mean_d, dev2, dev4) = ssim_moments.finish();

    out[idx::SSIM_MEAN] = mean_d;
    out[idx::SSIM_DEV2] = dev2;
    out[idx::SSIM_DEV4] = dev4;
    out[idx::ART] = sum_art / n_f;
    out[idx::DET] = sum_det / n_f;
    out[idx::MSE] = sum_mse / n_f;
    out[idx::HF_GAIN] = sum_hf_gain / n_f;
    out[idx::HF_LOSS] = sum_hf_loss / n_f;
    out[idx::HF_MAG_LOSS] = sum_hf_mag_loss / n_f;
    out[idx::SSIM_SOFT_PEAK] = ws_peak_ssim.finish();
    out[idx::ART_SOFT_PEAK] = ws_peak_art.finish();
    out[idx::DET_SOFT_PEAK] = ws_peak_det.finish();
    out[idx::MASKED_SSIM] = ws_mask_ssim.finish();
    out[idx::MASKED_ART] = ws_mask_art.finish();
    out[idx::MASKED_DET] = ws_mask_det.finish();
    out[idx::MASKED_MSE] = ws_mask_mse.finish();
    out[idx::IW_SSIM] = ws_iw_ssim.finish();
    out[idx::IW_ART] = ws_iw_art.finish();
    out[idx::IW_DET] = ws_iw_det.finish();
    out[idx::IW_MSE] = ws_iw_mse.finish();
    out[idx::PJND_TRANSDUCER] = sum_pjnd / n_f;
    out[idx::PJND_FRAGILITY] = 1.0 - saturate(sum_grad_src / n_f, C_PJND_GRAD);
    out[idx::GMS] = sum_gms / n_f;
    out[idx::PJND_TRANSDUCER_LOW_K] = sum_pjnd_lo / n_f;
    out[idx::PJND_TRANSDUCER_HIGH_K] = sum_pjnd_hi / n_f;
    out[idx::BLOCKINESS] = sum_blockiness / n_f;
    out[idx::RINGING] = sum_ringing / n_f;
    out[idx::BANDING] = sum_banding / n_f;
    out[idx::EDGE_WIDTH_CHANGE] = 0.0; // filled in by the caller (needs the adjacent scale)

    (sum_grad_src / n_f, sum_grad_dst / n_f)
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

    for scale in 0..n_scales {
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        for ch in 0..3 {
            let out = &mut features[scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL..]
                [..FEATURES_PER_CHANNEL_V2_TOTAL];
            let (gsrc, gdst) = compute_channel_scale_v2(
                &src_planes[ch],
                &dst_planes[ch],
                width,
                height,
                toggles,
                out,
            );

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

    const TOL: f64 = 1e-6;

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
}
