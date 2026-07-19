//! V2 "bounded" feature extraction — opt-in, additive regime.
//!
//! Gated behind `feature-regime-v2` (default OFF). Full defect inventory
//! (D1-D9), design principles, per-feature formula/bound/citation table, and
//! validation plan: `docs/FEATURE_V2_SPEC_2026-07-18.md`.
//!
//! # What this is
//!
//! Iteration 1 ("the bounded core") of the v2 spec: a scalar
//! correctness-first reference implementation. It reuses this crate's
//! existing, already-tested primitives for the expensive/error-prone parts
//! (XYB color conversion via [`crate::streaming::convert_source_to_xyb`], box
//! blur via [`crate::blur::box_blur_1pass_into`], pyramid downscale via
//! [`crate::blur::downscale_2x_inplace`]) and reimplements ONLY the per-pixel
//! FORMULAS — which is exactly where the v1 defects (D1-D9) live. It is
//! **not** fused into the archmage/SIMD streaming hot path (`streaming.rs`);
//! that fusion is future work once a v2 formula set is validated. Throughput
//! is explicitly not a design goal of iteration 1.
//!
//! # What this is not
//!
//! v1's 372-feature extraction (`metric.rs`, `streaming.rs`, `simd_ops.rs`)
//! is completely untouched by this module. No v1 constant, formula, or
//! byte-layout changes. Every canonical parquet and shipped bake depends on
//! that freeze and it is preserved bit-for-bit whether `feature-regime-v2` is
//! on or off (see the `v1_suite_unaffected_by_v2_feature` marker test below —
//! the real guarantee is "the normal `cargo test` suite is identical either
//! way," exercised by CI running both feature combinations, not a single
//! test asserting it).
//!
//! # Design principles (every v2 feature satisfies all four)
//!
//! 1. **Bounded by construction** — a fixed finite range, via a saturating
//!    transform, never a raw ratio. See [`bounded_sim`] and [`saturate`].
//! 2. **Normalized consistently** — masked/IW/soft-peak pooling all go
//!    through the ONE canonical weighted-mean helper,
//!    [`crate::iw_pool::WeightedPool::mean`] (`Σw·v/Σw`, promoted out of
//!    `#[allow(dead_code)]` for this module — see the doc comment there).
//! 3. **Spatializable** — every feature is the mean of an EXPLICIT per-pixel
//!    map. This includes the GMSD-style deviation moments (D8 fix) and the
//!    soft-saliency peak weight (D4 fix), which replace v1 quantities that
//!    were not expressible as a per-pixel map at all (a raw 4th moment's
//!    true per-pixel gradient is non-uniform; a hard `max` is an order
//!    statistic, not a sum).
//! 4. **Sign-consistent** — every feature is error-oriented, higher = worse,
//!    EXCEPT `pjnd_fragility`, which is a reference-only masking-susceptibility
//!    prior (Bondžulić et al. 2022) rather than a src-vs-dst error term; its
//!    sign is still oriented so "higher = this region is more likely to show
//!    visible distortion" (documented at its computation site below).

use crate::error::ZensimError;
use crate::iw_pool::WeightedPool;
use crate::source::ImageSource;

// ============================================================================
// Layout constants
// ============================================================================

/// Bounded-basic block: SSIM mean + GMSD-style 2nd/4th deviation moments
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
/// Near-threshold (PJND) block: divisive-normalization transducer +
/// reference-only masking-susceptibility (spec §(c)).
pub const FEATURES_PER_CHANNEL_V2_PJND: usize = 2;
/// Total v2 signals per channel per scale (9+3+4+4+2 = 22).
pub const FEATURES_PER_CHANNEL_V2_TOTAL: usize = FEATURES_PER_CHANNEL_V2_BASIC
    + FEATURES_PER_CHANNEL_V2_PEAK
    + FEATURES_PER_CHANNEL_V2_MASKED
    + FEATURES_PER_CHANNEL_V2_IW
    + FEATURES_PER_CHANNEL_V2_PJND;

/// Named local offsets within one channel's v2 block — mirrors the
/// documented layout table `metric.rs` uses for the v1 basic-13 block.
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
}

// ============================================================================
// Bounding constants — each documented + cited (full derivations in
// docs/FEATURE_V2_SPEC_2026-07-18.md)
// ============================================================================

/// SSIM luminance stability constant, v2. `C1 = (K1*L)^2`, `K1=0.01` (Wang,
/// Bovik, Sheikh, Simoncelli 2004, IEEE TIP 13(4), the paper's default
/// constant), `L=1`. `L=1` is not a new assumption: it exactly reproduces
/// this crate's existing `simd_ops::C2 = 0.0009 = (0.03*1)^2` (`K2=0.03`,
/// the paper's other default constant) — so the XYB "unit dynamic range"
/// convention was already load-bearing in v1, this just makes it explicit
/// and reuses it for C1.
///
/// Restores the STANDARD SSIM luminance form
/// `num_m = (2·μ1·μ2 + C1) / (μ1² + μ2² + C1)` in place of v1's
/// `num_m = 1 - (μ1-μ2)²` (no C1 — see `simd_ops.rs:255`'s own comment
/// "There is no C1"), which is D1's root cause
/// (`benchmarks/ssim_moment_explosion_2026-07-16.md`: 5.8e6 explosion).
///
/// **Boundedness proof** (µ1,µ2 ≥ 0 is REQUIRED and holds here because
/// [`crate::streaming::convert_source_to_xyb`] exclusively calls the
/// *positive*-XYB conversion variants, `srgb_to_positive_xyb_planar_into` /
/// `linear_to_positive_xyb_planar_into` — verified at `streaming.rs`'s XYB
/// conversion call sites, 2026-07-18):
/// - `num_m ≤ 1` always (AM-GM: `(μ1-μ2)² ≥ 0 ⟹ μ1²+μ2² ≥ 2μ1μ2`, holds for
///   any real μ1, μ2, no sign assumption needed).
/// - `num_m ≥ 0` requires `μ1·μ2 ≥ -C1/2`; guaranteed when μ1, μ2 ≥ 0 (their
///   product is non-negative), which the positive-XYB convention provides.
/// - The contrast/structure ratio `num_s/denom_s ∈ [-1, 1]` (Cauchy-Schwarz:
///   `|cov| ≤ σ1·σ2 ≤ (σ1²+σ2²)/2`).
/// - So `d = max(0, 1 - num_m·(num_s/denom_s)) ∈ [0, 2]` — bounded by
///   construction, not by a post-hoc clamp.
pub const C1_V2: f64 = 0.0001;

/// Structure/contrast stability constant, v2 — same value and derivation as
/// v1's `simd_ops::C2`, kept as an independent constant so v2 never silently
/// inherits a future v1-only change to that constant.
pub const C2_V2: f64 = 0.0009;

/// GMSD (Xue, Zhang, Mou, Bovik 2013/2014, arXiv:1308.3052, Eq.4) / FSIM
/// (Zhang, Zhang, Mou, Zhang 2011, Eq.5-6) / DISTS (Ding, Ma, Wang,
/// Simoncelli 2020, arXiv:2004.07728) canonical bounded-similarity
/// stabilizer for the edge artifact/detail comparison. Same order of
/// magnitude as `C1_V2` (edge residuals are unit-XYB-scale, like SSIM's
/// local means).
pub const C_EDGE: f64 = 1e-4;

/// Saturating half-point for the bounded MSE replacement:
/// `mse=0.5` at squared error `C_MSE`, i.e. a per-pixel absolute XYB
/// difference of `sqrt(C_MSE) = 0.1` — roughly a 10% local-intensity shift
/// in the `L=1` XYB convention. `x/(x+c)` is the Michelson-contrast-style
/// saturating ratio family; see [`saturate`].
pub const C_MSE: f64 = 0.01;

/// Stabilizer for the HF gain/loss/mag-loss normalized-difference bounded
/// forms (same order of magnitude as `C_EDGE`; these are `(a-b)/(a+b+c)`-
/// shaped rather than GMSD's `(2ab+c)/(a²+b²+c)`, because gain/loss need a
/// SIGNED asymmetry — which pixel has more local energy — not a symmetric
/// similarity; the boundedness argument is the same denominator-dominates
/// argument).
pub const C_HF: f64 = 1e-4;

/// Saturating half-point for the soft-saliency peak weight
/// `sal(i) = x(i) / (x(i) + C_PEAK)`, which replaces v1's hard max/L8
/// pooling (D4 fix; FSIM's PC-as-weight pattern, self-weighted here).
pub const C_PEAK: f64 = 0.05;

/// Saturating half-point bounding the reference-activity signal before it is
/// used as a masked/IW pooling weight (D6 fix — bounds the weight
/// regardless of input dynamic range, unlike v1's unbounded `1 + k·a`, so
/// this holds for HDR/PU-linear headroom without a separate per-pipeline
/// audit).
pub const C_ACTIVITY: f64 = 0.01;

/// Weight floor for the IW (texture-emphasizing) pool so a perfectly flat
/// image doesn't produce an all-zero weight sum (mirrors `iw_pool.rs`'s
/// `weight_floor` concept).
pub const IW_WEIGHT_FLOOR: f64 = 1e-3;

/// ColorVideoVDP-style (Mantiuk, Hanji, Ashraf, Asano, Chapiro 2024,
/// SIGGRAPH 2024, arXiv:2401.11485, Eq.9-13) final soft-clamp half-point for
/// the PJND divisive-normalization transducer map: `D_hat = D/(D+C_PJND_CLAMP)`.
pub const C_PJND_CLAMP: f64 = 0.1;

/// Saturating half-point for the reference-only gradient-energy signal
/// (Bondžulić, Pavlović, Stojanović et al. 2022, PJND prediction for JPEG,
/// Vojnotehnički glasnik 70(2)).
pub const C_PJND_GRAD: f64 = 0.02;

/// Strength of the activity-based masking denominator in the PJND
/// transducer map (`err / (1 + k·activity)`), same order of magnitude as
/// v1's `k_mask`/`k_iw` (4.0) but now paired with a bounded final clamp
/// rather than left as a raw unbounded ratio.
pub const K_PJND_MASK: f64 = 4.0;

/// Box blur radius at scale 0 — matches v1's `ZensimConfig::default()`
/// (`metric.rs`). Not yet exposed as a config knob in iteration 1 (see
/// `docs/FEATURE_V2_SPEC_2026-07-18.md`'s as-built section for why).
const BLUR_RADIUS: usize = 5;

// ============================================================================
// Bounded per-pixel formulas
// ============================================================================

/// Per-pixel SSIM-like dissimilarity, C1-bounded (D1 fix). Bounded [0, 2] —
/// see the boundedness proof on [`C1_V2`].
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
/// Symmetric, `(0, 1]`, `=1` iff `a == b`. Requires `a, b >= 0`.
#[inline]
fn bounded_sim(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * a * b + c) / (a * a + b * b + c)
}

/// Bounded, SIGNED normalized-difference form: `max(0, a-b) / (a+b+c)`.
/// Used where gain/loss asymmetry (which of `a`, `b` is larger) is the
/// signal, not similarity. Bounded `[0, 1)`. Requires `a, b >= 0`.
#[inline]
fn bounded_excess(a: f64, b: f64, c: f64) -> f64 {
    (a - b).max(0.0) / (a + b + c)
}

/// Michelson-contrast-style saturating ratio `x/(x+c)`, bounded `[0, 1)` for
/// `x >= 0`. The one saturating-ratio family reused everywhere in v2 (design
/// principle 2 in the module doc / spec principle 2).
#[inline]
fn saturate(x: f64, c: f64) -> f64 {
    let x = x.max(0.0);
    x / (x + c)
}

// ============================================================================
// FeatureRegime + result + explicit-regime view
// ============================================================================

/// Explicit feature-extraction regime tag. Unlike v1's [`crate::FeatureView`]
/// (which infers its layout from `features.len()`, auto-detecting basic vs
/// basic+peaks vs basic+peaks+masked+iw), v2 results are tagged explicitly —
/// a v2 vector's length could in principle coincide with some v1 length
/// combination as the two regimes evolve independently, so a runtime length
/// *guess* would be fragile. The tag here is belt-and-suspenders: the
/// stronger disambiguator is that [`ZensimV2Result`] and [`FeatureViewV2`]
/// are distinct Rust types from v1's, so a caller cannot accidentally hand a
/// v1 vector to a v2 accessor (or vice versa) and have it silently "work."
/// See `docs/FEATURE_V2_SPEC_2026-07-18.md` §(d).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureRegime {
    /// The frozen, byte-stable v1 372-feature extraction (`metric.rs`).
    V1,
    /// The v2 "bounded" extraction this module implements.
    V2Bounded,
}

/// Result of [`compute_v2_features`]/[`crate::Zensim::compute_v2_features`]:
/// the flat v2 feature vector plus the metadata ([`FeatureRegime`],
/// scale count) needed to interpret its layout via [`FeatureViewV2`].
#[derive(Debug, Clone)]
pub struct ZensimV2Result {
    features: Vec<f64>,
    n_scales: usize,
    regime: FeatureRegime,
}

impl ZensimV2Result {
    /// Flat feature vector: `n_scales * 3 channels * FEATURES_PER_CHANNEL_V2_TOTAL`.
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// Consume and return the flat feature vector.
    pub fn into_features(self) -> Vec<f64> {
        self.features
    }

    /// Number of pyramid scales this result covers.
    pub fn n_scales(&self) -> usize {
        self.n_scales
    }

    /// The regime tag. Always [`FeatureRegime::V2Bounded`] for a result
    /// produced by this module — exposed so callers that thread results
    /// through generic code can assert on it rather than assume it.
    pub fn regime(&self) -> FeatureRegime {
        self.regime
    }

    /// Explicit-regime named view over this result's features.
    pub fn view(&self) -> FeatureViewV2<'_> {
        FeatureViewV2::new(&self.features, self.n_scales)
            .expect("compute_v2_features always emits the v2-total layout for its own n_scales")
    }
}

/// Named, explicit-regime view over a v2 feature vector.
///
/// Unlike v1's [`crate::FeatureView`], [`FeatureViewV2::new`] does not
/// *guess* the layout from length — there is only one v2 layout, so it
/// *validates* the exact expected length for the given `n_scales` and
/// returns `None` on any mismatch (fails loud, never silently mis-indexes).
#[derive(Debug, Clone, Copy)]
pub struct FeatureViewV2<'a> {
    features: &'a [f64],
    n_scales: usize,
}

impl<'a> FeatureViewV2<'a> {
    /// Construct a view. Returns `None` if `features.len()` does not equal
    /// the exact v2 total for `n_scales`.
    pub fn new(features: &'a [f64], n_scales: usize) -> Option<Self> {
        let expected = n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        if features.len() != expected {
            return None;
        }
        Some(Self { features, n_scales })
    }

    /// Number of scales in this view.
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
    /// Divisive-normalization near-threshold transducer (CVVDP-style; spec
    /// §(c) item 1). Error-oriented, higher = worse.
    pub fn pjnd_transducer(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_TRANSDUCER)
    }
    /// Reference-only masking-susceptibility prior (Bondžulić et al. 2022;
    /// spec §(c) item 2). NOT a src-vs-dst error term — see the module doc's
    /// design-principle-4 note and the computation site in
    /// [`compute_channel_scale_v2`].
    pub fn pjnd_fragility(&self, scale: usize, ch: usize) -> f64 {
        self.at(scale, ch, idx::PJND_FRAGILITY)
    }
}

// ============================================================================
// Per-channel-per-scale computation
// ============================================================================

/// Compute all [`FEATURES_PER_CHANNEL_V2_TOTAL`] v2 signals for one channel
/// at one scale. `src`/`dst` are XYB planes at this scale (tightly packed,
/// `width * height` elements each, non-negative — the "positive XYB"
/// convention; see [`C1_V2`]'s boundedness proof). Writes into `out`.
fn compute_channel_scale_v2(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    out: &mut [f64],
) {
    let n = width * height;
    assert_eq!(src.len(), n, "src plane length must be width*height");
    assert_eq!(dst.len(), n, "dst plane length must be width*height");
    assert_eq!(
        out.len(),
        FEATURES_PER_CHANNEL_V2_TOTAL,
        "out slice must hold exactly one channel-scale's v2 block"
    );

    // --- Blurred local means + the two blur-of-combination terms (same
    //     "one blur instead of two" trick v1 uses: blur(src*dst) and
    //     blur(src²+dst²) rather than separate blur(src²), blur(dst²)). ---
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

    // Reference-only local activity (same signal shape v1 uses for its
    // masked/IW weights: blurred |src - local mean|), bounded via
    // `saturate(.., C_ACTIVITY)` when consumed as a pooling weight (D6 fix).
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

    // --- Per-pixel bounded maps (single pass) ---
    let mut d_ssim = vec![0.0f32; n];
    let mut art = vec![0.0f32; n];
    let mut det = vec![0.0f32; n];
    let mut mse = vec![0.0f32; n];
    let mut hf_gain = vec![0.0f32; n];
    let mut hf_loss = vec![0.0f32; n];
    let mut hf_mag_loss = vec![0.0f32; n];
    let mut pjnd_trans = vec![0.0f32; n];

    let mut sum_d = 0.0f64;
    for i in 0..n {
        let m1 = mu1[i] as f64;
        let m2 = mu2[i] as f64;

        let d = ssim_d_local(m1, m2, s12[i] as f64, ssq[i] as f64);
        d_ssim[i] = d as f32;
        sum_d += d;

        // Edge artifact/detail: bounded-similarity of the local high-pass
        // magnitudes, split into gain (dst has MORE local structure, ringing/
        // blocking) vs loss (dst has LESS, blur/smoothing) halves — same
        // asymmetric-knob intent as v1, now via a bounded form (D3 fix).
        let s = src[i] as f64;
        let dd = dst[i] as f64;
        let diff_src = (s - m1).abs();
        let diff_dst = (dd - m2).abs();
        let edge_dissim = 1.0 - bounded_sim(diff_src, diff_dst, C_EDGE);
        if diff_dst > diff_src {
            art[i] = edge_dissim as f32;
        } else if diff_dst < diff_src {
            det[i] = edge_dissim as f32;
        }

        // Bounded MSE replacement (D6 partial fix — saturates regardless of
        // input dynamic range, so it needs no per-pipeline SDR/HDR audit).
        let raw_sq_err = (s - dd) * (s - dd);
        mse[i] = saturate(raw_sq_err, C_MSE) as f32;

        // HF gain/loss/mag-loss: v1 computed these as ratios of two POOLED
        // (whole-scale) energies (`hf_dst_L2/hf_src_L2`), which has no
        // per-pixel decomposition at all. v2 redefines them as genuinely
        // per-pixel comparisons of the local high-frequency residual
        // (D2 fix, and a spatializability upgrade beyond what D2 strictly
        // required — this is design principle 3 applied to a v1 quantity
        // that was never even attempting to be a per-pixel map).
        let hf_src = s - m1;
        let hf_dst = dd - m2;
        let hf_src_sq = hf_src * hf_src;
        let hf_dst_sq = hf_dst * hf_dst;
        hf_gain[i] = bounded_excess(hf_dst_sq, hf_src_sq, C_HF) as f32;
        hf_loss[i] = bounded_excess(hf_src_sq, hf_dst_sq, C_HF) as f32;
        hf_mag_loss[i] = bounded_excess(hf_src.abs(), hf_dst.abs(), C_HF) as f32;

        // PJND divisive-normalization transducer (ColorVideoVDP-style, spec
        // §(c) item 1): raw per-pixel error normalized by LOCAL masking
        // energy, THEN a final soft-clamp (matching CVVDP's own two-stage
        // form) — bounded, error-oriented.
        let raw_abs_err = (s - dd).abs();
        let t = raw_abs_err / (1.0 + K_PJND_MASK * activity[i] as f64);
        pjnd_trans[i] = saturate(t, C_PJND_CLAMP) as f32;
    }
    let mean_d = sum_d / n as f64;

    // GMSD-style deviation moments (D8 fix): the DEVIATION from the mean is
    // its own explicit per-pixel map, mean-pooled with the root applied last
    // — spatializable by construction, unlike a raw 4th moment.
    let mut sum_dev2 = 0.0f64;
    let mut sum_dev4 = 0.0f64;
    for &d in &d_ssim {
        let dev = d as f64 - mean_d;
        let dev2 = dev * dev;
        sum_dev2 += dev2;
        sum_dev4 += dev2 * dev2;
    }
    let ssim_dev2 = (sum_dev2 / n as f64).sqrt();
    let ssim_dev4 = (sum_dev4 / n as f64).powf(0.25);

    let mean_of = |v: &[f32]| -> f64 { v.iter().map(|&x| x as f64).sum::<f64>() / n as f64 };

    // --- Soft-saliency-weighted peak replacement (D4 fix): self-weighted
    //     saturating saliency (FSIM's PC-as-weight pattern, self-weighted
    //     here rather than by an external phase-congruency signal), pooled
    //     via the SAME canonical weighted-mean helper as masked/IW below. ---
    let sal_of = |v: &[f32]| -> Vec<f32> {
        v.iter()
            .map(|&x| saturate(x as f64, C_PEAK) as f32)
            .collect()
    };
    let ssim_soft_peak = WeightedPool::mean(&d_ssim, &sal_of(&d_ssim));
    let art_soft_peak = WeightedPool::mean(&art, &sal_of(&art));
    let det_soft_peak = WeightedPool::mean(&det, &sal_of(&det));

    // --- Masked (flat-emphasizing) / IW (texture-emphasizing) pooling
    //     (D5, D6 fix): ONE canonical `Σw·v/Σw` weighted mean
    //     (`WeightedPool::mean`, Wang & Li 2011 Eq.36 form exactly),
    //     applied to the SAME bounded per-pixel maps already computed
    //     above, weighted by the bounded activity signal. ---
    let mask_w: Vec<f32> = activity
        .iter()
        .map(|&a| (1.0 - saturate(a as f64, C_ACTIVITY)) as f32)
        .collect();
    let iw_w: Vec<f32> = activity
        .iter()
        .map(|&a| (saturate(a as f64, C_ACTIVITY) + IW_WEIGHT_FLOOR) as f32)
        .collect();

    let masked_ssim = WeightedPool::mean(&d_ssim, &mask_w);
    let masked_art = WeightedPool::mean(&art, &mask_w);
    let masked_det = WeightedPool::mean(&det, &mask_w);
    let masked_mse = WeightedPool::mean(&mse, &mask_w);

    let iw_ssim = WeightedPool::mean(&d_ssim, &iw_w);
    let iw_art = WeightedPool::mean(&art, &iw_w);
    let iw_det = WeightedPool::mean(&det, &iw_w);
    let iw_mse = WeightedPool::mean(&mse, &iw_w);

    // --- PJND fragility (reference-only; spec §(c) item 2): simple L1
    //     centered-difference gradient magnitude of the REFERENCE ALONE
    //     (Bondžulić et al. 2022 found this predicts first-JND at >92%
    //     correlation with no masking model at all), bounded, then
    //     INVERTED so "higher = this region is more susceptible to visible
    //     distortion" (design principle 4's documented exception — this is
    //     a masking-capacity prior, not a src-vs-dst error). ---
    let mut pjnd_frag = vec![0.0f32; n];
    for y in 0..height {
        let yu = y.saturating_sub(1);
        let yd = (y + 1).min(height - 1);
        for x in 0..width {
            let xl = x.saturating_sub(1);
            let xr = (x + 1).min(width - 1);
            let i = y * width + x;
            let gx = (src[y * width + xr] - src[y * width + xl]) as f64;
            let gy = (src[yd * width + x] - src[yu * width + x]) as f64;
            let g = gx.abs() + gy.abs();
            pjnd_frag[i] = (1.0 - saturate(g, C_PJND_GRAD)) as f32;
        }
    }

    out[idx::SSIM_MEAN] = mean_d;
    out[idx::SSIM_DEV2] = ssim_dev2;
    out[idx::SSIM_DEV4] = ssim_dev4;
    out[idx::ART] = mean_of(&art);
    out[idx::DET] = mean_of(&det);
    out[idx::MSE] = mean_of(&mse);
    out[idx::HF_GAIN] = mean_of(&hf_gain);
    out[idx::HF_LOSS] = mean_of(&hf_loss);
    out[idx::HF_MAG_LOSS] = mean_of(&hf_mag_loss);
    out[idx::SSIM_SOFT_PEAK] = ssim_soft_peak;
    out[idx::ART_SOFT_PEAK] = art_soft_peak;
    out[idx::DET_SOFT_PEAK] = det_soft_peak;
    out[idx::MASKED_SSIM] = masked_ssim;
    out[idx::MASKED_ART] = masked_art;
    out[idx::MASKED_DET] = masked_det;
    out[idx::MASKED_MSE] = masked_mse;
    out[idx::IW_SSIM] = iw_ssim;
    out[idx::IW_ART] = iw_art;
    out[idx::IW_DET] = iw_det;
    out[idx::IW_MSE] = iw_mse;
    out[idx::PJND_TRANSDUCER] = mean_of(&pjnd_trans);
    out[idx::PJND_FRAGILITY] = mean_of(&pjnd_frag);
}

// ============================================================================
// Top-level entry point
// ============================================================================

/// Compute the v2 "bounded" feature vector for a source/distorted pair.
///
/// Reuses [`crate::metric::validate_pair`] (dimension/HDR checks) and
/// [`crate::metric::reflect_pad_to_min`] (small-image padding) — identical
/// input handling to v1's entry points. Always uses [`crate::NUM_SCALES`]
/// (4) scales and `BLUR_RADIUS=5`, matching v1's defaults; iteration 1 does
/// not yet expose these as config knobs (see
/// `docs/FEATURE_V2_SPEC_2026-07-18.md`'s as-built section).
///
/// Called by [`crate::Zensim::compute_v2_features`] (the public entry
/// point); kept as a free function here so it can be unit-tested without a
/// `Zensim` instance.
pub(crate) fn compute_v2_features_impl(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
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

    for scale in 0..n_scales {
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        for ch in 0..3 {
            let out = &mut features[scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL..]
                [..FEATURES_PER_CHANNEL_V2_TOTAL];
            compute_channel_scale_v2(&src_planes[ch], &dst_planes[ch], width, height, out);
        }

        if scale + 1 < n_scales {
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

    /// (b) FeatureView-analog accessor test with the explicit regime tag —
    /// item 7d.
    #[test]
    fn regime_tag_and_view_accessors_match_raw_indexing() {
        let src = flat_image(64, 64, [128, 128, 128]);
        let mut dst = src.clone();
        // Perturb one pixel so the result isn't the trivial all-zero case.
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
                assert_eq!(
                    view.ssim_dev2(scale, ch),
                    result.features()[base + idx::SSIM_DEV2]
                );
                assert_eq!(
                    view.ssim_dev4(scale, ch),
                    result.features()[base + idx::SSIM_DEV4]
                );
                assert_eq!(view.art(scale, ch), result.features()[base + idx::ART]);
                assert_eq!(view.det(scale, ch), result.features()[base + idx::DET]);
                assert_eq!(view.mse(scale, ch), result.features()[base + idx::MSE]);
                assert_eq!(
                    view.hf_gain(scale, ch),
                    result.features()[base + idx::HF_GAIN]
                );
                assert_eq!(
                    view.hf_loss(scale, ch),
                    result.features()[base + idx::HF_LOSS]
                );
                assert_eq!(
                    view.hf_mag_loss(scale, ch),
                    result.features()[base + idx::HF_MAG_LOSS]
                );
                assert_eq!(
                    view.ssim_soft_peak(scale, ch),
                    result.features()[base + idx::SSIM_SOFT_PEAK]
                );
                assert_eq!(
                    view.masked_ssim(scale, ch),
                    result.features()[base + idx::MASKED_SSIM]
                );
                assert_eq!(
                    view.iw_ssim(scale, ch),
                    result.features()[base + idx::IW_SSIM]
                );
                assert_eq!(
                    view.pjnd_transducer(scale, ch),
                    result.features()[base + idx::PJND_TRANSDUCER]
                );
                assert_eq!(
                    view.pjnd_fragility(scale, ch),
                    result.features()[base + idx::PJND_FRAGILITY]
                );
            }
        }

        // FeatureViewV2::new fails loud on a length mismatch — no
        // length-sniffing / no silent mis-indexing.
        assert!(FeatureViewV2::new(result.features(), result.n_scales() + 1).is_none());
        assert!(
            FeatureViewV2::new(
                &result.features()[..result.features().len() - 1],
                result.n_scales()
            )
            .is_none()
        );
    }

    /// (c) Identity input => every v1-analog ERROR feature is exactly 0.
    /// `pjnd_fragility` is excluded — it's a reference-only property (see
    /// the module doc's design-principle-4 note), so it is legitimately
    /// non-zero even when src==dst.
    #[test]
    fn identity_input_zeroes_every_error_feature() {
        // A non-flat reference (checkerboard-ish) so this isn't a
        // degenerate all-flat case; identity must hold on real content too.
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
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                assert!(
                    view.ssim_mean(scale, ch).abs() < TOL,
                    "ssim_mean s{scale} c{ch}"
                );
                assert!(
                    view.ssim_dev2(scale, ch).abs() < TOL,
                    "ssim_dev2 s{scale} c{ch}"
                );
                assert!(
                    view.ssim_dev4(scale, ch).abs() < TOL,
                    "ssim_dev4 s{scale} c{ch}"
                );
                assert!(view.art(scale, ch).abs() < TOL, "art s{scale} c{ch}");
                assert!(view.det(scale, ch).abs() < TOL, "det s{scale} c{ch}");
                assert!(view.mse(scale, ch).abs() < TOL, "mse s{scale} c{ch}");
                assert!(
                    view.hf_gain(scale, ch).abs() < TOL,
                    "hf_gain s{scale} c{ch}"
                );
                assert!(
                    view.hf_loss(scale, ch).abs() < TOL,
                    "hf_loss s{scale} c{ch}"
                );
                assert!(
                    view.hf_mag_loss(scale, ch).abs() < TOL,
                    "hf_mag_loss s{scale} c{ch}"
                );
                assert!(
                    view.ssim_soft_peak(scale, ch).abs() < TOL,
                    "ssim_soft_peak s{scale} c{ch}"
                );
                assert!(
                    view.art_soft_peak(scale, ch).abs() < TOL,
                    "art_soft_peak s{scale} c{ch}"
                );
                assert!(
                    view.det_soft_peak(scale, ch).abs() < TOL,
                    "det_soft_peak s{scale} c{ch}"
                );
                assert!(
                    view.masked_ssim(scale, ch).abs() < TOL,
                    "masked_ssim s{scale} c{ch}"
                );
                assert!(
                    view.masked_art(scale, ch).abs() < TOL,
                    "masked_art s{scale} c{ch}"
                );
                assert!(
                    view.masked_det(scale, ch).abs() < TOL,
                    "masked_det s{scale} c{ch}"
                );
                assert!(
                    view.masked_mse(scale, ch).abs() < TOL,
                    "masked_mse s{scale} c{ch}"
                );
                assert!(
                    view.iw_ssim(scale, ch).abs() < TOL,
                    "iw_ssim s{scale} c{ch}"
                );
                assert!(view.iw_art(scale, ch).abs() < TOL, "iw_art s{scale} c{ch}");
                assert!(view.iw_det(scale, ch).abs() < TOL, "iw_det s{scale} c{ch}");
                assert!(view.iw_mse(scale, ch).abs() < TOL, "iw_mse s{scale} c{ch}");
                assert!(
                    view.pjnd_transducer(scale, ch).abs() < TOL,
                    "pjnd_transducer s{scale} c{ch}"
                );
                // NOT asserted zero: pjnd_fragility (reference-only).
            }
        }
    }

    /// (b) Adversarial fixture 1/3: flat source + noise — the exact D2
    /// pathology (v1's hf_gain hit ~1e7 here). v2's hf_gain/hf_loss must
    /// stay within their documented [0, 1) bound.
    #[test]
    fn bounded_range_flat_source_plus_noise_hf_gain() {
        let w = 64;
        let h = 64;
        let mut src = vec![[128u8, 128, 128]; w * h];
        let mut dst = src.clone();
        // Checkerboard "noise" injected only into the distorted image: the
        // source stays perfectly flat (hf_src ~ 0 everywhere), which is the
        // division-by-near-zero pathology.
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 2 == 0 {
                    dst[y * w + x] = [200, 60, 220];
                } else {
                    dst[y * w + x] = [50, 220, 40];
                }
            }
        }
        let _ = &mut src; // src stays flat
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                let g = view.hf_gain(scale, ch);
                let l = view.hf_loss(scale, ch);
                let m = view.hf_mag_loss(scale, ch);
                assert!(
                    (0.0..1.0).contains(&g),
                    "hf_gain OOB: {g} at s{scale} c{ch}"
                );
                assert!(
                    (0.0..1.0).contains(&l),
                    "hf_loss OOB: {l} at s{scale} c{ch}"
                );
                assert!(
                    (0.0..1.0).contains(&m),
                    "hf_mag_loss OOB: {m} at s{scale} c{ch}"
                );
                // The whole point: this is exactly the pattern that hit ~1e7
                // in v1. It must not even get CLOSE to exploding in v2.
                assert!(g < 10.0 && l < 10.0 && m < 10.0);
            }
        }
    }

    /// (b) Adversarial fixture 2/3: large chroma mean-shift — the D1
    /// `(mu1-mu2)^2` driver (measured 5.8e6 explosion in v1). v2's
    /// ssim_mean/dev2/dev4 must stay within their documented [0, 2] bound.
    #[test]
    fn bounded_range_large_chroma_mean_shift() {
        let w = 64;
        let h = 64;
        let src = flat_image(w, h, [0, 255, 0]); // flat green
        let dst = flat_image(w, h, [255, 0, 255]); // flat magenta — max hue swing
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                let m = view.ssim_mean(scale, ch);
                let d2 = view.ssim_dev2(scale, ch);
                let d4 = view.ssim_dev4(scale, ch);
                assert!(
                    (0.0..=2.0 + TOL).contains(&m),
                    "ssim_mean OOB: {m} at s{scale} c{ch}"
                );
                assert!(
                    (0.0..=2.0 + TOL).contains(&d2),
                    "ssim_dev2 OOB: {d2} at s{scale} c{ch}"
                );
                assert!(
                    (0.0..=2.0 + TOL).contains(&d4),
                    "ssim_dev4 OOB: {d4} at s{scale} c{ch}"
                );
            }
        }
    }

    /// (b) Adversarial fixture 3/3: hard edge on a flat chroma field — the
    /// exact `masked`/`IW` explosion pattern from
    /// `benchmarks/ssim_moment_explosion_2026-07-16.md` (flat region
    /// bordered by a hard chroma edge, 5.8e6 in `masked_ssim_4th`/
    /// `iw_ssim_4th`). v2's masked/iw blocks must stay within their
    /// documented [0, 2] bound (same range as `ssim_mean`, since
    /// `WeightedPool::mean` of a bounded map cannot exceed the map's bound).
    #[test]
    fn bounded_range_hard_edge_on_flat_masked_and_iw() {
        let w = 64;
        let h = 64;
        let mut src = vec![[0u8, 200, 0]; w * h]; // flat green field
        let mut dst = src.clone();
        // Hard vertical edge in the distorted image: right half shifts to a
        // very different hue — a codec-artifact-style hard edge on an
        // otherwise flat chroma region.
        for y in 0..h {
            for x in (w / 2)..w {
                dst[y * w + x] = [220, 0, 220];
            }
        }
        let _ = &mut src;
        let source = RgbSlice::new(&src, w, h);
        let distorted = RgbSlice::new(&dst, w, h);

        let result =
            compute_v2_features_impl(&source, &distorted, None, false).expect("compute ok");
        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                for (name, v) in [
                    ("masked_ssim", view.masked_ssim(scale, ch)),
                    ("iw_ssim", view.iw_ssim(scale, ch)),
                    ("masked_art", view.masked_art(scale, ch)),
                    ("iw_art", view.iw_art(scale, ch)),
                    ("masked_det", view.masked_det(scale, ch)),
                    ("iw_det", view.iw_det(scale, ch)),
                    ("masked_mse", view.masked_mse(scale, ch)),
                    ("iw_mse", view.iw_mse(scale, ch)),
                    ("ssim_soft_peak", view.ssim_soft_peak(scale, ch)),
                ] {
                    assert!(
                        (0.0..=2.0 + TOL).contains(&v),
                        "{name} OOB: {v} at s{scale} c{ch} (v1's analogous features hit 5.8e6 on this exact pattern)"
                    );
                }
            }
        }
    }

    /// Blanket bounded-range sweep across all adversarial fixtures + a
    /// random-noise pair, asserting every one of the 22 per-channel signals
    /// stays inside ITS documented bound. Cheap insurance beyond the
    /// targeted fixtures above.
    #[test]
    fn bounded_range_all_signals_random_content() {
        let w = 64;
        let h = 64;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        // Deterministic pseudo-random (LCG) — no external RNG dependency.
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
        let view = result.view();
        for scale in 0..result.n_scales() {
            for ch in 0..3 {
                let zero_one = [
                    view.art(scale, ch),
                    view.det(scale, ch),
                    view.mse(scale, ch),
                    view.hf_gain(scale, ch),
                    view.hf_loss(scale, ch),
                    view.hf_mag_loss(scale, ch),
                    view.pjnd_transducer(scale, ch),
                    view.pjnd_fragility(scale, ch),
                ];
                for v in zero_one {
                    assert!(
                        (0.0..1.0 + TOL).contains(&v),
                        "expected [0,1)-bounded signal got {v}"
                    );
                }
                let zero_two = [
                    view.ssim_mean(scale, ch),
                    view.ssim_dev2(scale, ch),
                    view.ssim_dev4(scale, ch),
                    view.ssim_soft_peak(scale, ch),
                    view.art_soft_peak(scale, ch),
                    view.det_soft_peak(scale, ch),
                    view.masked_ssim(scale, ch),
                    view.masked_art(scale, ch),
                    view.masked_det(scale, ch),
                    view.masked_mse(scale, ch),
                    view.iw_ssim(scale, ch),
                    view.iw_art(scale, ch),
                    view.iw_det(scale, ch),
                    view.iw_mse(scale, ch),
                ];
                for v in zero_two {
                    assert!(
                        v.is_finite() && (0.0..=2.0 + TOL).contains(&v),
                        "expected [0,2]-bounded signal got {v}"
                    );
                }
            }
        }
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

    /// Sub-64px input is reflect-padded, not rejected — matches v1 behavior.
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
        // At least one signal should register the injected difference.
        let view = result.view();
        let any_nonzero =
            (0..result.n_scales()).any(|s| (0..3).any(|c| view.ssim_mean(s, c) > TOL));
        assert!(
            any_nonzero,
            "single-pixel change should move ssim_mean somewhere"
        );
    }
}
