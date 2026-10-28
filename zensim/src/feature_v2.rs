//! V2 "bounded" feature extraction — additive regime.
//!
//! Gated behind `feature-regime-v2` (default ON since 2026-09-01 — the
//! Profile-D no-tax refactor, `benchmarks/profile_d_notax_2026-09-01.md`;
//! `--no-default-features` still removes this whole module). Full defect inventory
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

use crate::det_math::DetRoots;
use crate::error::ZensimError;
use crate::source::ImageSource;

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16 as GenericF32x16};

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
use crate::gmsbank_constants::{GMSBANK_B_C, GMSBANK_C, GMSBANK_CS_C, GMSBANK_X_C};
/// Saturating half-point for the reference-edge indicator used by ringing
/// (`edge_r` in the A.10 table's `err · dilate(edge_r) · (1−edge_r)` form).
pub const C_RING_EDGE: f64 = 0.02;
/// Saturating half-point bounding the raw per-pixel error before it enters
/// the ringing product (keeps ringing bounded regardless of input dynamic
/// range, same D6-style rationale as `C_ACTIVITY`).
pub const C_RING_ERR: f64 = 0.05;

/// Which per-pixel error the ringing family measures.
///
/// A measurement arm selected per process by `ZENSIM_RING_FORM`, modelled on
/// `ZENSIM_HF_GAIN` ([`crate::hf_gain_form::active_gain_form`]): no shipping
/// path sets it, and every stored table and bake is [`Self::Raw`].
///
/// imazen/zensimx#72 (`benchmarks/byconstruction_2026-10-22.md`): [`Self::Raw`]
/// saturates `|src − dst|`, which a global offset, contrast or saturation
/// change makes uniformly large off edges. Measured on 24 references, a
/// +8 luma shift was ringing's largest response, 2.4× a visible σ1.2 blur.
/// [`Self::HighPassMu`] measures the error after removing each image's SSIM
/// local mean (`mu1`, `mu2`, already computed for the dense kernel), so a
/// constant offset or a linear ramp contributes nothing away from the image
/// border, while noise and oscillation next to edges still do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum RingForm {
    /// `sat(|e|, C_RING_ERR) · sat(act, C_ACTIVITY) · (1 − sat(|∇src|, C_RING_EDGE))`,
    /// `e = src − dst`. The shipped form.
    #[default]
    Raw,
    /// The same product with `e = (src − mu1) − (dst − mu2)`.
    HighPassMu,
}

impl RingForm {}

#[cfg(test)]
mod ring_form_tests {
    use super::*;

    /// Deterministic xorshift values in `[lo, hi)`.
    struct Draw(u64);
    impl Draw {
        fn next(&mut self, lo: f64, hi: f64) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (lo + (hi - lo) * ((self.0 >> 11) as f64 / (1u64 << 53) as f64)) as f32
        }
    }

    /// 3×3 box mean with clamped borders: a stand-in local mean. The kernel
    /// takes whatever `mu` planes the walk supplies; these tests only need
    /// `mu(dst) = mu(src) + k` for `dst = src + k`.
    fn box3(p: &[f32], w: usize, h: usize) -> Vec<f32> {
        let mut out = vec![0.0; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0f32;
                for dy in [-1i64, 0, 1] {
                    for dx in [-1i64, 0, 1] {
                        let yy = (y as i64 + dy).clamp(0, h as i64 - 1) as usize;
                        let xx = (x as i64 + dx).clamp(0, w as i64 - 1) as usize;
                        s += p[yy * w + xx];
                    }
                }
                out[y * w + x] = s / 9.0;
            }
        }
        out
    }

    /// `(src_h, dst_h, act, mu1, mu2)`.
    type Planes = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

    /// `(src_h, dst_h, act, mu1, mu2)` for a `w×h` interior with one halo row
    /// above and below, `dst = f(src)`.
    fn planes(w: usize, h: usize, seed: u64, f: impl Fn(f32, &mut Draw) -> f32) -> Planes {
        let mut r = Draw(seed);
        let src_h: Vec<f32> = (0..(h + 2) * w).map(|_| r.next(0.2, 0.8)).collect();
        let dst_h: Vec<f32> = src_h.iter().map(|&v| f(v, &mut r)).collect();
        let act: Vec<f32> = (0..w * h).map(|_| r.next(0.0, 0.5)).collect();
        let mu1 = box3(&src_h[w..w + w * h], w, h);
        let mu2 = box3(&dst_h[w..w + w * h], w, h);
        (src_h, dst_h, act, mu1, mu2)
    }

    fn run(hp: bool, p: &Planes, w: usize, h: usize) -> GradientAccum {
        let (src_h, dst_h, act, mu1, mu2) = p;
        let m = crate::featcanon::Mode::Canon64;
        if hp {
            gradient_block_kernel_canon::<
                crate::featcanon::Lanes16F64,
                false,
                false,
                false,
                true,
                false,
            >(src_h, dst_h, act, &[], mu1, mu2, w, h, 0.0, 0.0, m)
        } else {
            gradient_block_kernel_canon::<
                crate::featcanon::Lanes16F64,
                false,
                false,
                false,
                false,
                false,
            >(src_h, dst_h, act, &[], mu1, mu2, w, h, 0.0, 0.0, m)
        }
    }

    #[test]
    fn a_global_offset_does_not_register_under_hpm_and_only_ringing_moves() {
        let (w, h) = (37usize, 11usize);
        let p = planes(w, h, 0x5249_4e47_4850_3403, |v, _| v + 0.03);
        let (raw, hp) = (run(false, &p, w, h), run(true, &p, w, h));
        assert!(raw.sum_ringing > 1.0, "{}", raw.sum_ringing);
        assert!(
            hp.sum_ringing < 1e-4 * raw.sum_ringing,
            "{} vs {}",
            hp.sum_ringing,
            raw.sum_ringing
        );
        for (a, b) in [
            (raw.sum_gms, hp.sum_gms),
            (raw.sum_gms2, hp.sum_gms2),
            (raw.sum_banding, hp.sum_banding),
            (raw.sum_grad_src, hp.sum_grad_src),
            (raw.sum_grad_dst, hp.sum_grad_dst),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    #[test]
    fn noise_still_registers_under_hpm() {
        let (w, h) = (37usize, 11usize);
        let p = planes(w, h, 0x5249_4e47_4850_3404, |v, r| v + r.next(-0.05, 0.05));
        let (raw, hp) = (run(false, &p, w, h), run(true, &p, w, h));
        assert!(
            hp.sum_ringing > 0.3 * raw.sum_ringing,
            "{} vs {}",
            hp.sum_ringing,
            raw.sum_ringing
        );
    }

    #[test]
    fn hpm_matches_an_independent_per_pixel_sum() {
        // Every pixel's term recomputed here in f64 from the definition; the
        // kernel's f32 terms pooled in f64 must agree to f32 precision.
        let (w, h) = (29usize, 7usize);
        let p = planes(w, h, 0x5249_4e47_4850_3405, |v, r| v + r.next(-0.08, 0.08));
        let (src_h, dst_h, act, mu1, mu2) = &p;
        let mut want = 0.0f64;
        for y in 0..h {
            for x in 0..w {
                let at = |pl: &[f32], yy: usize, xx: usize| pl[yy * w + xx] as f64;
                let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
                let gx = at(src_h, y + 1, xr) - at(src_h, y + 1, xl);
                let gy = at(src_h, y + 2, x) - at(src_h, y, x);
                let g = (gx * gx + gy * gy).sqrt();
                let i = y * w + x;
                let e =
                    (at(src_h, y + 1, x) - mu1[i] as f64) - (at(dst_h, y + 1, x) - mu2[i] as f64);
                want += saturate(e.abs(), C_RING_ERR)
                    * saturate(act[i] as f64, C_ACTIVITY)
                    * (1.0 - saturate(g, C_RING_EDGE));
            }
        }
        let got = run(true, &p, w, h).sum_ringing;
        assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
    }
}
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
    /// Global per-channel mean shift (A9): `sat(|mean(s)−mean(d)|,
    /// C_GDMEAN)` — the windowed features under-weigh global casts.
    pub const GLOBAL_DMEAN: usize = 13;
    /// Global contrast gain: `bounded_excess(gvar₂, gvar₁, C_GCONTRAST)`
    /// on whole-plane variances.
    pub const GLOBAL_CGAIN: usize = 14;
    /// Global contrast loss: reverse polarity of [`GLOBAL_CGAIN`].
    pub const GLOBAL_CLOSS: usize = 15;
}

/// Saturating half-point mapping the (cbrt-domain, ~[0,1]) reference Y
/// value to the bin parameter `t = y/(y+C_LUM_T)` used by the luminance
/// soft bins and the luminance transducer term. 0.35 puts `t=0.5` near
/// mid-gray in XYB's cube-root intensity scale.
pub const C_LUM_T: f64 = 0.35;
/// Saturating half-point for the global |Δmean| feature.
pub const C_GDMEAN: f64 = 0.02;
/// Stabilizer for the global variance gain/loss pair.
pub const C_GCONTRAST: f64 = 1e-4;

// ============================================================================
// append2 (f924+) constants — BANDVIS + luminance conditioner + HL bins.
// δ values are EMPIRICAL (test `bandvis_delta_derivation_table` prints the
// measurement; derivation committed in
// `benchmarks/append2_bandvis_gates_2026-07-27.md`). PROVISIONAL until the
// measurement lands — the derivation test brackets them against the live
// front-ends so a front-end change that invalidates them fails loudly.
// ============================================================================

/// Stabilizer for the BANDVIS FR `bounded_excess` pair (indicators are
/// [0,1] products — same class as `C_EDGE`).
pub(crate) const C_BV: f64 = 1e-4;

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
pub(crate) const APPEND2_CHANNEL: usize = 1;

/// Named local offsets within one scale's append2 block (all Y-only).
pub mod idx_append2 {
    /// Reference-only local-adaptation conditioner: `sat(mean(ref Y),
    /// C_LUM_T)` from the append kernel's existing `sum_s` accumulator
    /// (finalize-only — FREE). Correct-0 in any steering fold, like
    /// `PJND_FRAGILITY`/`GRAD_SRC_MEAN`.
    pub const LUMA_MEAN_REF: usize = 2;
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
}

// ============================================================================
// REV4 candidate bank (2026-09-23 lane — `benchmarks/rev4_featbank_impl_
// 2026-09-23.md`, spec `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §1.1).
// Four append-only families at f986+; every slot is difference-form and
// every flag is independent (the emit tail is absolute-positioned, so any
// subset stays index-stable). Layouts mirror `feature_defs::BLOCKS`:
// gridblk/ringbasis/tailhist are PerChannel (`(scale*3 + ch)*per_cell +
// local`), arttype is PerScale (`scale*ARTTYPE_PER_SCALE + local`).
// ============================================================================

/// mapdev slots per (scale, channel) cell (`feature_defs::MAPDEV_SIGNALS`).
pub(crate) const MAPDEV_PER_CELL: usize = 5;
/// z1max slots per (scale, channel) cell: 13 basic + 6 peaks.
pub(crate) const Z1MAX_PER_CELL: usize = 19;
/// gmsnative slots in total: X then B, 15 each, at scale 0.
pub(crate) const GMSNATIVE_WIDTH: usize = 30;

fn gmsbank_width(n_scales: usize) -> usize {
    crate::feature_defs::block_base(crate::feature_set_id::ComputeToken::Gmsbank, n_scales)
        .expect("registered C8")
        .1
        .width(n_scales)
}

/// gridblk slots per (scale, channel) cell: 6 signed log-magnitude bins
/// of the on-grid boundary step excess + on-grid mean + on/off ratio.
pub(crate) const GRIDBLK_PER_CELL: usize = 8;
/// ringbasis slots per (scale, channel) cell: 6 triangular
/// log-magnitude bins of the RINGING per-pixel term.
pub(crate) const RINGBASIS_PER_CELL: usize = 6;
/// tailhist slots per (scale, channel) cell: p95/p99/max for each of the
/// SSIM/ART/DET/MSE per-pixel maps (4 maps × 3).
pub(crate) const TAILHIST_PER_CELL: usize = 12;
/// arttype slots per scale (no channel axis — like append2/CSFW, each
/// signal names its channel): blur(Y), noise_x/y/b, bleed_x/b.
pub(crate) const ARTTYPE_PER_SCALE: usize = 6;
/// Tail histogram bin count per map (31 true-log interior edges
/// `10^(-6 + k*6/32)`, k = 1..31 → 32 partition cells).
pub(crate) const TAILHIST_BINS: usize = 32;

/// Named local offsets within one scale's arttype block.
pub(crate) mod idx_arttype {}

/// Triangular-hat bin count shared by gridblk and ringbasis.
pub(crate) const REV4_HATS: usize = 6;
/// Largest lattice period any rev4 cell can take (chroma at scale 0).
pub(crate) const GRIDBLK_MAX_PERIOD: usize = 16;

/// Binade LUT for the pinned C3 edges, used to bin in ~5 ops.
struct TailEdges {
    /// 31 ascending interior-edge bit patterns.
    edges: [u64; 31],
    /// `base[e]` = #edges strictly below binade `e`'s low end;
    /// `inner[e]` = #edges inside binade `e` (≤ 2: edge spacing
    /// 6/32 decade > binade width 0.301... i.e. at most `ceil(0.301/
    /// 0.1875) = 2` — asserted at build).
    base: Box<[u8; 1024]>,
    inner: Box<[u8; 1024]>,
}

impl TailEdges {

}

// ---------------------------------------------------------------------------
// Calibration diagnostics (registry-revision tooling, 2026-09-24).
// `ZENSIM_REV4_DIAG` set at process start enables per-cell fine-grained
// histogram dumps on stderr at finalize — used only for the C3 top-edge
// and C1 hat-centre calibration over TRAIN corpora. Default off; the
// `Option` check is one predictable branch per scatter/rescan pass.
// ---------------------------------------------------------------------------

/// 64 log bins per diagnostic histogram.
const DIAG_BINS: usize = 64;

fn rev4_diag_enabled() -> bool {
    false
}

// ---------------------------------------------------------------------------
// REV4 per-(scale, channel) accumulator cell
//
// Merge-free by construction: the streaming and materialized walks both run
// the SAME kernel strips in the SAME order and write into the cell directly
// (the `blockiness_sparse_strip_wide` running-total pattern — "a per-strip
// partial would reassociate"), so serial, parallel-channel and
// streaming-vs-materialized paths all produce identical f64 op sequences.
// ---------------------------------------------------------------------------

/// C3: 32-bin log histograms + exact maxima of the four dense per-pixel
/// maps (`d`, `art`, `det`, `mse` — in emitted order). Integer counts are
/// merge- and order-invariant; the max is order-free.
#[derive(Clone)]
struct TailAccum {
    hist: [[u32; TAILHIST_BINS]; 4],
    max: [f64; 4],
    /// Env-gated fine histogram for the top-edge calibration dump
    /// (`ZENSIM_REV4_DIAG`); `None` in every production/test path.
    diag: Option<Box<[[u32; DIAG_BINS]; 4]>>,
    /// `c3negfold`: bin through [`TailEdges::bin_folded`]. Set from the
    /// computation's revision (`Rev4` only); Rev1–Rev3 keep the legacy
    /// [`TailEdges::bin`] so the landed `rev4bank` Bin slots they emit stay
    /// byte-identical. The process revision equals the computation's
    /// (`ssim_form::refuse_rev4_mix`).
    fold: bool,
}

impl Default for TailAccum {
    fn default() -> Self {
        Self {
            hist: [[0; TAILHIST_BINS]; 4],
            max: [0.0; 4],
            fold: crate::ssim_form::active_revision() >= FormulaRevision::Rev5,
            diag: rev4_diag_enabled().then(|| Box::new([[0; DIAG_BINS]; 4])),
        }
    }
}

impl TailAccum {
}

/// C4-noise: flat-reference-region HF-gain mass for one cell.
/// featacc: `hfg` is a `SumVar` (f64-native terms) — `Seq` under production.
#[derive(Clone, Copy)]
struct FlatAccum {
    /// Σ `hf_gain_i` over pixels with `act ≤ C4_FLAT_ACT` (scan order).
    hfg: crate::featcanon::SumVar,
    /// Count of flat pixels.
    n: u64,
}

impl Default for FlatAccum {
    fn default() -> Self {
        Self {
            hfg: crate::featcanon::SumVar::f64_elems_meas(),
            n: 0,
        }
    }
}

/// C2: the 6 triangular `u`-domain bins of the per-pixel RINGING term —
/// `bins[k] = Σ_px ring·m_k(u(ring))` in scan order.
/// featacc: `SumVar` bins — `Seq` under production.
#[derive(Clone, Copy)]
struct RingAccum {
    bins: [crate::featcanon::SumVar; REV4_HATS],
}

impl Default for RingAccum {
    fn default() -> Self {
        Self {
            bins: std::array::from_fn(|_| crate::featcanon::SumVar::f64_elems_meas()),
        }
    }
}

/// C4-bleed: gradient magnitudes summed OUTSIDE the dilated dst-luma-edge
/// mask (scan order). featacc: `SumVar` — `Seq` under production.
#[derive(Clone, Copy)]
struct BleedAccum {
    out_src: crate::featcanon::SumVar,
    out_dst: crate::featcanon::SumVar,
}

impl Default for BleedAccum {
    fn default() -> Self {
        Self {
            out_src: crate::featcanon::SumVar::f64_elems_meas(),
            out_dst: crate::featcanon::SumVar::f64_elems_meas(),
        }
    }
}

/// C1: the stored `ẽ` boundary planes plus the per-phase profiles that
/// pick the winning phase at finalize (design note §C1 "Accumulation" —
/// the winning phase gates the emitted bins but is only known after the
/// whole plane is walked, so the kernel stores `ẽ` per boundary and the
/// rescan touches only on-grid positions: `2n/P` hat evaluations instead
/// of membership work on all `2n` boundaries).
///
/// `plane_v[y*w + x]` holds `ẽ` of the vertical boundary between columns
/// `x−1` and `x` (x ∈ 1..w); `plane_h[y*w + x]` holds the horizontal
/// boundary between rows `y−1` and `y` (y ∈ 1..h). Both are written by
/// `gridblk_strip_wide` in the SAME f32 arithmetic on every engine, so
/// the rescan reads bitwise-identical values in the streaming and
/// materialized walks. `abs_*`/`signed_*` are the per-phase `Σ|ẽ|`/`Σẽ`
/// profiles — f64 running totals fed by the kernel's lane-folded V8
/// accumulators (the lane map `(x−1, chunk parity) → phase` is fixed by
/// construction, so the accumulation order is identical on every tier,
/// engine and thread count — this family's determinism contract).
struct GridblkAccum {
    plane_v: Vec<f32>,
    plane_h: Vec<f32>,
    abs_v: [f64; GRIDBLK_MAX_PERIOD],
    signed_v: [f64; GRIDBLK_MAX_PERIOD],
    abs_h: [f64; GRIDBLK_MAX_PERIOD],
    signed_h: [f64; GRIDBLK_MAX_PERIOD],
    /// featacc `exact`-mode shadow planes: the same `ẽ` grid evaluated in
    /// f64. Empty in every other mode (and in product builds — two dead
    /// `Vec`s, 48 bytes of cell state). The rescan in
    /// [`finish_gridblk_cell`] reads whichever plane family is populated.
    plane_v64: Vec<f64>,
    plane_h64: Vec<f64>,
}

impl Default for GridblkAccum {
    fn default() -> Self {
        Self {
            plane_v: Vec::new(),
            plane_h: Vec::new(),
            abs_v: [0.0; GRIDBLK_MAX_PERIOD],
            signed_v: [0.0; GRIDBLK_MAX_PERIOD],
            abs_h: [0.0; GRIDBLK_MAX_PERIOD],
            signed_h: [0.0; GRIDBLK_MAX_PERIOD],
            plane_v64: Vec::new(),
            plane_h64: Vec::new(),
        }
    }
}

/// All rev4 state for one (scale, channel) cell. Untouched when every
/// rev4 compute flag is off.
#[derive(Default)]
struct Rev4CellAccum {
    grid: GridblkAccum,
    ring: RingAccum,
    tail: TailAccum,
    flat: FlatAccum,
    bleed: BleedAccum,
}

// ---------------------------------------------------------------------------
// REV4 emission: one PerChannel cell = gridblk(8) + ringbasis(6) +
// tailhist(12) = 26 slots in that order; arttype is emitted per scale by
// `finish_arttype_scale` (its blur term needs the cross-scale EWC value,
// which only exists at finalize).
// ---------------------------------------------------------------------------

/// Box blur radius at scale 0 — matches v1's `ZensimConfig::default()`.
pub(crate) const BLUR_RADIUS: usize = 5;
/// Oriented-blockiness lattice period (JPEG's 8x8 MCU grid).
pub(crate) const BLOCK_LATTICE: usize = 8;

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
const _: () = assert!(H_BLUR_BAND_ROWS.is_multiple_of(H_BLUR_ROW_GROUP));

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

// ============================================================================
// Bounded per-pixel formulas
// ============================================================================

/// GMSD/FSIM/DISTS canonical bounded-similarity form: `(2ab+c)/(a²+b²+c)`.
/// Bounded `(0, 1]`, `=1` iff `a == b`. Requires `a, b >= 0`.
#[inline]
pub(crate) fn bounded_sim(a: f64, b: f64, c: f64) -> f64 {
    (2.0 * a * b + c) / (a * a + b * b + c)
}

/// `1 − bounded_sim(a, b, c)`, the bounded dissimilarity `(a−b)²/(a²+b²+c)`.
/// Formed as `1 − ratio` it cancels when `a ≈ b` (the ratio rounds near 1);
/// with `stable` (revision 7, [`FormulaRevision::stable_forms`]) it forms the
/// exact quotient instead.
#[inline]
pub(crate) fn bounded_dissim(a: f64, b: f64, c: f64, stable: bool) -> f64 {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        1.0 - bounded_sim(a, b, c)
    }
}

/// [`bounded_dissim`] in f32, op for op `1 − (2ab + c)/(a² + b² + c)`, or the
/// exact quotient with `stable`.
#[inline(always)]
pub(crate) fn bounded_dissim32(a: f32, b: f32, c: f32, stable: bool) -> f32 {
    if stable {
        let d = a - b;
        d * d / (a * a + b * b + c)
    } else {
        1.0 - (2.0 * a * b + c) / (a * a + b * b + c)
    }
}

/// Bounded, SIGNED normalized-difference form: `max(0, a-b) / (a+b+c)`.
/// Bounded `[0, 1)`. Requires `a, b >= 0`.
#[inline]
pub(crate) fn bounded_excess(a: f64, b: f64, c: f64) -> f64 {
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

include!("feature_v2_kernels.rs");

#[cfg(test)]
mod global_contrast_rev2_tests {
    use super::*;

    /// **The F5 gate.** Replaces (does not re-run) the synthetic-image
    /// `free_extras_match_the_944_append_block` comparison, which could not
    /// see this defect because synthetic content does not produce the
    /// flat-but-not-constant regions where `mean²/gvar` is large.
    ///
    /// The property under test is the one that actually broke: given raw
    /// moments carrying only f32-grade precision — which is all either route
    /// can deliver, since both accumulate in f32 lanes — how far does the
    /// computed `GLOBAL_CGAIN`/`GLOBAL_CLOSS` land from the exact answer?
    ///
    /// MEASURED consequence on real pixels (60 source images, 1,120 paired
    /// tranche cells, `benchmarks/feature_rev2_2026-09-05.md`): free-vs-append
    /// disagreement past the 2e-5 bar falls from **8.30 % to 0.18 %** and the
    /// worst cell from 2.31e-4 to 2.49e-5.
    #[test]
    fn paired_form_survives_f32_grade_moments_where_the_raw_form_does_not() {
        // A flat-but-not-constant region: large mean, tiny variance, tiny
        // distortion — the configuration where `mean^2/gvar` is the amplifier.
        let n = 65536.0f64;
        let (ms, md) = (0.5f64, 0.5f64 + 3e-5);
        let (var_s, var_d) = (1.0e-6f64, 1.3e-6f64);

        let sum_s = ms * n;
        let sum_d = md * n;
        let sum_s2 = (var_s + ms * ms) * n;
        let sum_d2 = (var_d + md * md) * n;
        let sum_ds = (md - ms) * n;
        // sum((d-s)(d+s)) = sum(d^2) - sum(s^2), formed per pixel.
        let sum_dd = sum_d2 - sum_s2;

        // Exact answer, in full f64 from the variances themselves.
        let (want_gain, want_loss) = bounded_excess_pair(var_d, var_s, C_GCONTRAST);

        // f32-grade moments: round the four raw totals to f32, which is the
        // precision any f32 lane accumulation can deliver no matter how it is
        // ordered or compensated. The PAIRED inputs are per-pixel differences,
        // so they are small and survive the same rounding.
        let f32r = |x: f64| x as f32 as f64;
        let (raw_gain, raw_loss) = {
            let (_, g, l) = global_stats_from_raw_moments(
                f32r(sum_s),
                f32r(sum_d),
                f32r(sum_s2),
                f32r(sum_d2),
                f32r(sum_dd),
                f32r(sum_ds),
                n,
                false,
            );
            (g, l)
        };
        let (pair_gain, pair_loss) = {
            let (_, g, l) = global_stats_from_raw_moments(
                f32r(sum_s),
                f32r(sum_d),
                f32r(sum_s2),
                f32r(sum_d2),
                f32r(sum_dd),
                f32r(sum_ds),
                n,
                true,
            );
            (g, l)
        };

        let raw_err = (raw_gain - want_gain)
            .abs()
            .max((raw_loss - want_loss).abs());
        let pair_err = (pair_gain - want_gain)
            .abs()
            .max((pair_loss - want_loss).abs());
        // The bar is the audit's own 2e-5 route-parity bar, not a number
        // picked to make the test pass: the defect is exactly "the raw form
        // lands further from truth than the bar allows".
        const BAR: f64 = 2e-5;
        assert!(
            raw_err > BAR,
            "the raw form should cross the parity bar here (that IS the defect); got {raw_err}"
        );
        assert!(
            pair_err < BAR / 10.0 && pair_err < raw_err / 10.0,
            "paired form must land an order below the bar: paired {pair_err} vs raw {raw_err}"
        );
    }

    /// The reassociation is an ALGEBRAIC IDENTITY, so with exact inputs the
    /// two forms must agree. If they ever do not, the revision changed the
    /// definition rather than its conditioning.
    #[test]
    fn the_two_forms_are_the_same_function_in_exact_arithmetic() {
        for &(ms, md, var_s, var_d) in &[
            (0.5f64, 0.5f64, 0.01f64, 0.02f64),
            (0.2, 0.9, 0.3, 0.05),
            (0.0, 0.0, 0.0, 0.0),
            (1000.0, 1000.5, 4.0, 3.0),
        ] {
            let n = 4096.0;
            let sum_s = ms * n;
            let sum_d = md * n;
            let sum_s2 = (var_s + ms * ms) * n;
            let sum_d2 = (var_d + md * md) * n;
            let a = global_stats_from_raw_moments(
                sum_s,
                sum_d,
                sum_s2,
                sum_d2,
                sum_d2 - sum_s2,
                (md - ms) * n,
                n,
                false,
            );
            let b = global_stats_from_raw_moments(
                sum_s,
                sum_d,
                sum_s2,
                sum_d2,
                sum_d2 - sum_s2,
                (md - ms) * n,
                n,
                true,
            );
            let tol = 1e-9 * a.1.abs().max(a.2.abs()).max(1.0);
            assert_eq!(a.0, b.0, "GLOBAL_DMEAN must be untouched by the revision");
            assert!(
                (a.1 - b.1).abs() <= tol && (a.2 - b.2).abs() <= tol,
                "{a:?} vs {b:?}"
            );
        }
    }
}

/// The three luminance-binned bounded-error values, from the dark/bright
/// weighted sums plus `Σ mse_i` and `n` — the ONE implementation, shared by
/// [`finish_append`] (reading the append kernel's accumulators) and the
/// free-extras finalize (reading the BIT-SAME-DEFINITION sums off the v1
/// fused kernel). Returns `[dark, mid, bright]`, each already `clamp(0,2)`.
///
/// The mid bin is DERIVED, not accumulated: the Bernstein weights
/// `(1−t)²`, `2t(1−t)`, `t²` are a partition of unity, so
/// `Σw_mid = n − Σw_dark − Σw_bright` and
/// `Σw_mid·v = Σv − Σw_dark·v − Σw_bright·v`.
#[inline]
fn lum_bins_from_weighted_sums(
    ws_dark: WeightedSum,
    ws_bright: WeightedSum,
    sum_mse: f64,
    n_f: f64,
) -> [f64; 3] {
    let ws_mid = WeightedSum {
        num: sum_mse - ws_dark.num - ws_bright.num,
        den: n_f - ws_dark.den - ws_bright.den,
    };
    [
        ws_dark.finish().clamp(0.0, 2.0),
        ws_mid.finish().clamp(0.0, 2.0),
        ws_bright.finish().clamp(0.0, 2.0),
    ]
}

/// `out[i] = input[i]²` — the streamed walk's σ-split square (reads the
/// wide window in place, writes into a free scratch buffer). Plain scalar
/// loop; LLVM auto-vectorizes the independent multiply.
fn square_into(input: &[f32], out: &mut [f32]) {
    for (o, &v) in out.iter_mut().zip(input.iter()) {
        *o = v * v;
    }
}

/// Per-channel, per-scale raw sums for the folded v1 feature blocks.
/// The canonical fused kernel supplies basic and peak (max/L8) accumulators,
/// plus the weighted pools and free extras requested by the feature plan.
/// Finalizers below emit the corresponding slots with global scale
/// normalization. Peak slots 156..228 are supported and consumed by candidate
/// models; they are not deprecated or silently discarded.
#[derive(Debug, Clone, Copy)]
struct V1BasicSums {
    // featacc: every f64 sum slot is a `featcanon::SumVar` — production
    // resolves to `Seq` (the plain `+=` below), a measurement mode to the
    // matching lane/Neumaier form. Two constructors pick the ELEMENT axis:
    // `meas_f64` for slots fed f64 strip partials (`accumulate`), `meas_f32`
    // for slots fed f32-native terms (`accumulate_block` in z1max) — under
    // `c32` the former stays sequential (f64 terms) and the latter takes
    // f32 lanes.
    ssim_d: crate::featcanon::SumVar,
    ssim_d4: crate::featcanon::SumVar,
    ssim_d2: crate::featcanon::SumVar,
    edge_art: crate::featcanon::SumVar,
    edge_art4: crate::featcanon::SumVar,
    edge_art2: crate::featcanon::SumVar,
    edge_det: crate::featcanon::SumVar,
    edge_det4: crate::featcanon::SumVar,
    edge_det2: crate::featcanon::SumVar,
    mse: crate::featcanon::SumVar,
    hf_sq_src: crate::featcanon::SumVar,
    hf_sq_dst: crate::featcanon::SumVar,
    hf_abs_src: crate::featcanon::SumVar,
    hf_abs_dst: crate::featcanon::SumVar,
    // --- v1 pool blocks (`V2NewFeatureToggles::v1_pools`): the peak
    //     accumulators the fused kernel returns anyway, plus the masked /
    //     IW sums v1's extended strip section derives per band
    //     (`streaming::ScaleAccumulators` field-for-field). ---
    ssim_d8: crate::featcanon::SumVar,
    edge_art8: crate::featcanon::SumVar,
    edge_det8: crate::featcanon::SumVar,
    // Maxima are order-free; widened to f64 so the exact map eval can push
    // f64 block maxima without narrowing (production values were f32, and
    // f32→f64 is lossless, so production bits are unchanged).
    ssim_max: f64,
    edge_art_max: f64,
    edge_det_max: f64,
    masked_ssim_d: crate::featcanon::SumVar,
    masked_ssim_d4: crate::featcanon::SumVar,
    masked_ssim_d2: crate::featcanon::SumVar,
    masked_art4: crate::featcanon::SumVar,
    masked_det4: crate::featcanon::SumVar,
    masked_mse: crate::featcanon::SumVar,
    iw_ssim_d: crate::featcanon::SumVar,
    iw_ssim_d4: crate::featcanon::SumVar,
    iw_ssim_d2: crate::featcanon::SumVar,
    iw_art4: crate::featcanon::SumVar,
    iw_det4: crate::featcanon::SumVar,
    iw_mse: crate::featcanon::SumVar,
    // --- FREE raw moments ([`V1FreeExtras::RawMoments`]) ---
    // Σsrc, Σdst, Σsrc², Σdst² over this (scale, channel). Zero unless the
    // request asked for them; the fused kernel's own fields of the same
    // name are their only source.
    sum_s: crate::featcanon::SumVar,
    sum_d: crate::featcanon::SumVar,
    sum_s2: crate::featcanon::SumVar,
    sum_d2: crate::featcanon::SumVar,
    /// Revision 2's paired second moment `Σ(d−s)(d+s)`, carried alongside the
    /// raw pair so `global_stats_from_raw_moments` can pick per revision.
    sum_dd: crate::featcanon::SumVar,
    /// Revision 2's paired first difference `Σ(d−s)`.
    sum_ds: crate::featcanon::SumVar,
    // --- FREE bounded error ([`V1FreeExtras::RawMomentsPlusBoundedErr`]) ---
    // Σ sat((s−d)², C_MSE) over this (scale, channel), and the Y-only
    // dark/bright weighted sums of that same per-pixel value. Zero unless
    // the request asked for them.
    sum_msat: crate::featcanon::SumVar,
    lum_wd_num: crate::featcanon::SumVar,
    lum_wd_den: crate::featcanon::SumVar,
    lum_wb_num: crate::featcanon::SumVar,
    lum_wb_den: crate::featcanon::SumVar,
}

impl V1BasicSums {

    fn at_revision(revision: FormulaRevision) -> Self {
        use crate::featcanon::SumVar;
        Self {
            ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_sq_src: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_sq_dst: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_abs_src: SumVar::f64_elems(crate::featcanon::mode(revision)),
            hf_abs_dst: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_d8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_art8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            edge_det8: SumVar::f64_elems(crate::featcanon::mode(revision)),
            ssim_max: 0.0,
            edge_art_max: 0.0,
            edge_det_max: 0.0,
            masked_ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            masked_mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_ssim_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_art4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_det4: SumVar::f64_elems(crate::featcanon::mode(revision)),
            iw_mse: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_s: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_d: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_s2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_d2: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_dd: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_ds: SumVar::f64_elems(crate::featcanon::mode(revision)),
            sum_msat: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wd_num: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wd_den: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wb_num: SumVar::f64_elems(crate::featcanon::mode(revision)),
            lum_wb_den: SumVar::f64_elems(crate::featcanon::mode(revision)),
        }
    }

}

impl V1BasicSums {
    /// Add another band's sums into this one. Used by both paths of
    /// [`fold_v1_basic_bands`]; the parallel path merges IN BAND ORDER so the
    /// f64 addition sequence matches the serial one exactly.
    ///
    /// featacc: `merge_from` folds lane-aligned, so under a lane mode each
    /// lane's own sequence is preserved (the same in-band-order argument, per
    /// lane).
    fn merge(&mut self, o: &V1BasicSums) {
        self.ssim_d.merge_from(&o.ssim_d);
        self.ssim_d4.merge_from(&o.ssim_d4);
        self.ssim_d2.merge_from(&o.ssim_d2);
        self.edge_art.merge_from(&o.edge_art);
        self.edge_art4.merge_from(&o.edge_art4);
        self.edge_art2.merge_from(&o.edge_art2);
        self.edge_det.merge_from(&o.edge_det);
        self.edge_det4.merge_from(&o.edge_det4);
        self.edge_det2.merge_from(&o.edge_det2);
        self.mse.merge_from(&o.mse);
        self.hf_sq_src.merge_from(&o.hf_sq_src);
        self.hf_sq_dst.merge_from(&o.hf_sq_dst);
        self.hf_abs_src.merge_from(&o.hf_abs_src);
        self.hf_abs_dst.merge_from(&o.hf_abs_dst);
        self.ssim_d8.merge_from(&o.ssim_d8);
        self.edge_art8.merge_from(&o.edge_art8);
        self.edge_det8.merge_from(&o.edge_det8);
        self.ssim_max = self.ssim_max.max(o.ssim_max);
        self.edge_art_max = self.edge_art_max.max(o.edge_art_max);
        self.edge_det_max = self.edge_det_max.max(o.edge_det_max);
        self.masked_ssim_d.merge_from(&o.masked_ssim_d);
        self.masked_ssim_d4.merge_from(&o.masked_ssim_d4);
        self.masked_ssim_d2.merge_from(&o.masked_ssim_d2);
        self.masked_art4.merge_from(&o.masked_art4);
        self.masked_det4.merge_from(&o.masked_det4);
        self.masked_mse.merge_from(&o.masked_mse);
        self.iw_ssim_d.merge_from(&o.iw_ssim_d);
        self.iw_ssim_d4.merge_from(&o.iw_ssim_d4);
        self.iw_ssim_d2.merge_from(&o.iw_ssim_d2);
        self.iw_art4.merge_from(&o.iw_art4);
        self.iw_det4.merge_from(&o.iw_det4);
        self.iw_mse.merge_from(&o.iw_mse);
        self.sum_s.merge_from(&o.sum_s);
        self.sum_d.merge_from(&o.sum_d);
        self.sum_s2.merge_from(&o.sum_s2);
        self.sum_d2.merge_from(&o.sum_d2);
        self.sum_dd.merge_from(&o.sum_dd);
        self.sum_ds.merge_from(&o.sum_ds);
        self.sum_msat.merge_from(&o.sum_msat);
        self.lum_wd_num.merge_from(&o.lum_wd_num);
        self.lum_wd_den.merge_from(&o.lum_wd_den);
        self.lum_wb_num.merge_from(&o.lum_wb_num);
        self.lum_wb_den.merge_from(&o.lum_wb_den);
    }

    /// Merge one strip's channel partials. The term-level accumulation axis
    /// lives inside the kernel that produced `s`; this merge step folds each
    /// partial in production order (lane 0 under lane modes — the O(#strips)
    /// merge sequence is unchanged across candidates).
    fn accumulate(&mut self, s: &crate::fused::StripChannelAccum) {
        self.ssim_d.add64(0, s.ssim_d);
        self.ssim_d4.add64(0, s.ssim_d4);
        self.ssim_d2.add64(0, s.ssim_d2);
        self.edge_art.add64(0, s.edge_art);
        self.edge_art4.add64(0, s.edge_art4);
        self.edge_art2.add64(0, s.edge_art2);
        self.edge_det.add64(0, s.edge_det);
        self.edge_det4.add64(0, s.edge_det4);
        self.edge_det2.add64(0, s.edge_det2);
        self.mse.add64(0, s.mse);
        self.hf_sq_src.add64(0, s.hf_sq_src);
        self.hf_sq_dst.add64(0, s.hf_sq_dst);
        self.hf_abs_src.add64(0, s.hf_abs_src);
        self.hf_abs_dst.add64(0, s.hf_abs_dst);
        // Peaks: v1 merges them the same way (`streaming.rs`
        // `accum.ssim_d8[c] += strip_acc.ssim_d8; accum.ssim_max[c] =
        // accum.ssim_max[c].max(strip_acc.ssim_max)`, …) — free to carry.
        self.ssim_d8.add64(0, s.ssim_d8);
        self.edge_art8.add64(0, s.edge_art8);
        self.edge_det8.add64(0, s.edge_det8);
        self.ssim_max = self.ssim_max.max(s.ssim_max as f64);
        self.edge_art_max = self.edge_art_max.max(s.edge_art_max as f64);
        self.edge_det_max = self.edge_det_max.max(s.edge_det_max as f64);
        // FREE raw moments: plain sums, so an unconditional merge is exact
        // whether the kernel filled them (+0.0 otherwise) — same shape as
        // the peaks above.
        self.sum_s.add64(0, s.sum_s);
        self.sum_d.add64(0, s.sum_d);
        self.sum_s2.add64(0, s.sum_s2);
        self.sum_d2.add64(0, s.sum_d2);
        self.sum_dd.add64(0, s.sum_dd);
        self.sum_ds.add64(0, s.sum_ds);
        // FREE bounded error: plain sums, merged the same unconditional way
        // as the raw moments above (+0.0 when the kernel did not fill them).
        self.sum_msat.add64(0, s.sum_msat);
        self.lum_wd_num.add64(0, s.lum_wd_num);
        self.lum_wd_den.add64(0, s.lum_wd_den);
        self.lum_wb_num.add64(0, s.lum_wb_num);
        self.lum_wb_den.add64(0, s.lum_wb_den);
    }

    /// Finalize the v1 pool blocks — peaks (6/ch), masked (6/ch), IW
    /// (6/ch) — into their v1 slot order, replicating
    /// `streaming::ScaleAccumulators::finalize` + `metric.rs`'s pass-2/3/4
    /// pushes (`.abs()` on the masked/IW ssim + art/det L4 slots, none on
    /// the peaks / mse).
    fn finalize_pools_into(
        &self,
        n: usize,
        peaks: &mut [f64],
        masked: &mut [f64],
        iw: &mut [f64],
        revision: crate::feature_defs::FormulaRevision,
    ) {
        debug_assert_eq!(peaks.len(), 6);
        debug_assert_eq!(masked.len(), 6);
        debug_assert_eq!(iw.len(), 6);
        let one_over_n = 1.0 / n as f64;
        // Hoisted out of the nine root calls below: it reads a `OnceLock`,
        // which LLVM cannot hoist for you. Same discipline as `gain_form`.
        let root_form = crate::det_math::RootForm::at_revision(Some(revision));
        peaks[0] = self.ssim_max;
        peaks[1] = self.edge_art_max;
        peaks[2] = self.edge_det_max;
        peaks[3] = (self.ssim_d8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        peaks[4] = (self.edge_art8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        peaks[5] = (self.edge_det8.fin() * one_over_n)
            .max(0.0)
            .eighth_root(root_form);
        masked[0] = (self.masked_ssim_d.fin() * one_over_n).abs();
        masked[1] = (self.masked_ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[2] = (self.masked_ssim_d2.fin() * one_over_n)
            .max(0.0)
            .sqrt()
            .abs();
        masked[3] = (self.masked_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[4] = (self.masked_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        masked[5] = self.masked_mse.fin() * one_over_n;
        iw[0] = (self.iw_ssim_d.fin() * one_over_n).abs();
        iw[1] = (self.iw_ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[2] = (self.iw_ssim_d2.fin() * one_over_n).max(0.0).sqrt().abs();
        iw[3] = (self.iw_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[4] = (self.iw_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        iw[5] = self.iw_mse.fin() * one_over_n;
    }

    /// Finalize the FREE v2-era slots ([`V1FreeExtras::RawMoments`]) this
    /// (scale, channel)'s raw moments can produce: the three `GLOBAL_*`
    /// append values (via the shared [`global_stats_from_raw_moments`]) and
    /// append2's `LUMA_MEAN_REF` (Y-channel only, the same
    /// `saturate(mean ref, C_LUM_T).clamp(0,1)` the append2 finalize applies
    /// to the append kernel's `sum_s`).
    fn finalize_free(
        &self,
        n: usize,
        rev: crate::feature_defs::FormulaRevision,
    ) -> ([f64; 3], f64) {
        let n_f = n as f64;
        let (dmean, cgain, closs) = global_stats_from_raw_moments(
            self.sum_s.fin(),
            self.sum_d.fin(),
            self.sum_s2.fin(),
            self.sum_d2.fin(),
            self.sum_dd.fin(),
            self.sum_ds.fin(),
            n_f,
            rev.paired_global_contrast(),
        );
        (
            [dmean, cgain.clamp(0.0, 1.0), closs.clamp(0.0, 1.0)],
            saturate(self.sum_s.fin() / n_f, C_LUM_T).clamp(0.0, 1.0),
        )
    }

    /// Finalize the class-C bounded-error slots
    /// ([`V1FreeExtras::RawMomentsPlusBoundedErr`]) from the fused kernel's
    /// `Σ mse_i` + luminance-weighted sums: the v2-348 `MSE` value and the
    /// dark / mid / bright luminance-binned means.
    ///
    /// Both routes go through [`lum_bins_from_weighted_sums`] and the same
    /// `clamp01(Σ mse_i / n)` the v2 finalize applies, so the parity gate is
    /// on the ACCUMULATION rather than on two hand-copied formulas — the
    /// discipline `global_stats_from_raw_moments` already established.
    fn finalize_free_bounded(&self, n: usize) -> (f64, [f64; 3]) {
        let n_f = n as f64;
        (
            (self.sum_msat.fin() / n_f).clamp(0.0, 1.0),
            lum_bins_from_weighted_sums(
                WeightedSum {
                    num: self.lum_wd_num.fin(),
                    den: self.lum_wd_den.fin(),
                },
                WeightedSum {
                    num: self.lum_wb_num.fin(),
                    den: self.lum_wb_den.fin(),
                },
                self.sum_msat.fin(),
                n_f,
            ),
        )
    }

    /// Finalize into one channel's 13 basic features, replicating v1's
    /// pooling + assembly EXACTLY (`streaming::ChannelAccum::finalize` +
    /// `metric.rs`'s pass-1 feature push, `.abs()` included): mean, L4
    /// (`(Σx⁴/n)^¼`), L2 (`(Σx²/n)^½`) for ssim/art/det; mse mean; the
    /// three HF ratio features with their `1e-10` guards. `n` is the
    /// scale's full pixel count (`w_s * h_s`) — identical to v1's per-scale
    /// accumulator `n` since both walks cover every row exactly once.
    fn finalize_into(
        &self,
        n: usize,
        out: &mut [f64],
        revision: crate::feature_defs::FormulaRevision,
    ) {
        debug_assert_eq!(out.len(), 13);
        let one_over_n = 1.0 / n as f64;
        let root_form = crate::det_math::RootForm::at_revision(Some(revision));
        out[0] = (self.ssim_d.fin() * one_over_n).abs();
        out[1] = (self.ssim_d4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[2] = (self.ssim_d2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[3] = (self.edge_art.fin() * one_over_n).abs();
        out[4] = (self.edge_art4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[5] = (self.edge_art2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[6] = (self.edge_det.fin() * one_over_n).abs();
        out[7] = (self.edge_det4.fin() * one_over_n)
            .max(0.0)
            .quarter_root(root_form)
            .abs();
        out[8] = (self.edge_det2.fin() * one_over_n).max(0.0).sqrt().abs();
        out[9] = self.mse.fin() * one_over_n;
        // Same owner as the buffered walk (`crate::hf_gain_form`), which is
        // what keeps the fold's v1 pools bit-identical to v1's under EVERY
        // arm rather than only under the shipped one.
        let gain_form = crate::hf_gain_form::HfGainForm::at_revision(Some(revision));
        let var_src = self.hf_sq_src.fin() * one_over_n;
        let var_dst = self.hf_sq_dst.fin() * one_over_n;
        out[10] = crate::hf_gain_form::hf_energy_loss(var_src, var_dst);
        out[12] = crate::hf_gain_form::hf_energy_gain(gain_form, var_src, var_dst);
        let mad_src = self.hf_abs_src.fin() * one_over_n;
        let mad_dst = self.hf_abs_dst.fin() * one_over_n;
        out[11] = crate::hf_gain_form::hf_mag_loss(mad_src, mad_dst);
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

#[cfg(test)]
thread_local! {
    // Observe actual weighted-pool execution, not planner declarations.
    static POOL_TEST_WIDTHS: std::cell::RefCell<Option<Vec<usize>>> = const { std::cell::RefCell::new(None) };
    // Scales whose strips the streaming walk skipped (declared plans reading nothing there).
    static SKIPPED_SCALES: std::cell::RefCell<u8> = const { std::cell::RefCell::new(0) };
}

/// Band-local planes for the v1 pool replay (`V2NewFeatureToggles::v1_pools`):
/// sized for one v1 band buffer (`V1_BAND_ROWS + 2 * V1_BAND_OVERLAP` rows ×
/// width), grown on first use per channel accumulator and reused across
/// bands, strips and scales (the widest scale sizes it once).
#[derive(Debug, Clone, Default)]
struct FoldPoolScratch {
    /// V-blurred `mu1` / `mu2` (the fused kernel's `store_mu` side-output).
    stable_sd: Vec<f32>,
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
    /// Rev5 consumers share the already V-blurred planes.
    Vertical([&'a [f32]; 4]),
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
    // `free`: which FREE extra accumulators the fused kernel should carry —
    // the raw moments (Σs, Σd, Σs², Σd²) that finalize the append block's
    // GLOBAL_* trio and append2's LUMA_MEAN_REF, and/or the class-C
    // bounded-error family (v2 `MSE` + the Y luminance bins). None of them
    // needs a new plane, load or pass ([`V1FreeExtras`],
    // [`crate::fused::FreeExtrasWork`]).
    free: crate::fused::FreeExtrasWork,
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
    if let FoldHSource::Vertical([a, b, c, d]) = h_src {
        debug_assert!(free.revision() >= FormulaRevision::Rev5);
        // Radius zero selects the same canonical per-element/pool owner,
        // reading the shared vertical values without another blur sweep.
        sums.accumulate(&crate::fused::fused_vblur_features_ssim(
            &a[span.clone()],
            &b[span.clone()],
            &c[span.clone()],
            &d[span.clone()],
            &src[span.clone()],
            &dst[span],
            width,
            h_local,
            inner_start,
            inner_h,
            0,
            &mut [],
            &mut [],
            false,
            &mut [],
            false,
            &mut [],
            &mut [],
            false,
            free,
            crate::fused::ExtPoolsWork::default(),
            &[],
        ));
        return b1;
    }
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
                crate::blur::fused_blur_h_ssim_at_revision(
                    &src[span.clone()],
                    &dst[span.clone()],
                    &mut h0[..band_n],
                    &mut h1[..band_n],
                    &mut h2[..band_n],
                    &mut h3[..band_n],
                    width,
                    h_local,
                    BLUR_RADIUS,
                    free.revision(),
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
                free,
                crate::fused::ExtPoolsWork::default(),
                &[],
            ));
            return b1;
        }
        #[cfg(test)]
        POOL_TEST_WIDTHS.with_borrow_mut(|widths| {
            if let Some(widths) = widths {
                widths.push(width);
            }
        });
        let full = work == BandPoolWork::Full;
        let stable = free.revision() >= crate::feature_defs::FormulaRevision::Rev5;
        if stable && full {
            ps.stable_sd.resize(band_cap_n, 0.0);
        }
        ps.ensure(band_cap_n);
        // ONE destructure, so the band-local H planes can be READ while the
        // pool planes stay mutable — disjoint fields of the same `&mut`.
        let FoldPoolScratch {
            stable_sd,
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
            crate::blur::fused_blur_h_ssim_at_revision(
                &src[span.clone()],
                &dst[span.clone()],
                &mut h0[..band_n],
                &mut h1[..band_n],
                &mut h2[..band_n],
                &mut h3[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
                free.revision(),
            );
        }
        let (mu1_h, mu2_h, ssq_h, s12_h, span_h) = match h_src {
            FoldHSource::Vertical(_) => unreachable!("shared vertical planes handled above"),
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
        // Revision 3: the SAME fused extension the streaming path runs
        // (`streaming.rs`, `fused_ext`) — the activity's V blur and every
        // masked/IW pool ride inside the SSIM V sweep, over the same bands
        // in the same order, which is what keeps the rev-3 fold/streaming
        // pool parity gate bit-exact. Here `act` then holds the H-BLURRED
        // activity the sweep V-blurs in-register.
        let fused_ext = stable;
        if fused_ext {
            crate::blur::box_blur_h(
                &act_raw[..band_n],
                &mut act[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
            );
        } else {
            crate::blur::box_blur_1pass_into(
                &act_raw[..band_n],
                &mut act[..band_n],
                &mut ssq_v[..band_n],
                width,
                h_local,
                BLUR_RADIUS,
            );
        }
        let ext = crate::fused::ExtPoolsWork {
            on: fused_ext,
            mask: true,
            iw: true,
            k_mask: V1_MASK_K,
            k_iw: V1_IW_K,
        };
        // `store_sigma` replaces the two `box_blur_v_from_copy(ssq_h →
        // ssq_v)` / `(s12_h → s12_v)` band sweeps the `Full` arm used to
        // run after this call: the fused kernel already carries the same
        // running V-blur sums in registers and divides by the same
        // `1.0 / diam`, so the stored planes are BIT-IDENTICAL to what
        // that second sweep produced (and it writes only the inner rows —
        // the only rows the masked/IW SSIM kernel reads). `Carriers`
        // needs no sigma, so it stores none and `ssq_v` simply stays the
        // activity temp.
        let acc = crate::fused::fused_vblur_features_ssim(
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
            if stable && full {
                &mut stable_sd[..band_n]
            } else {
                &mut empty_sd
            },
            stable && full,
            &mut ssq_v[..band_n],
            &mut s12_v[..band_n],
            full,
            free,
            ext,
            if fused_ext { &act[..band_n] } else { &[] },
        );
        sums.accumulate(&acc);
        if fused_ext {
            if full {
                sums.masked_mse.add64(0, acc.masked_mse);
                sums.iw_mse.add64(0, acc.iw_mse);
                sums.masked_ssim_d.add64(0, acc.masked_ssim_d);
                sums.masked_ssim_d4.add64(0, acc.masked_ssim_d4);
                sums.masked_ssim_d2.add64(0, acc.masked_ssim_d2);
                sums.iw_ssim_d.add64(0, acc.iw_ssim_d);
                sums.iw_ssim_d4.add64(0, acc.iw_ssim_d4);
                sums.iw_ssim_d2.add64(0, acc.iw_ssim_d2);
            }
            sums.masked_art4.add64(0, acc.masked_art4);
            sums.masked_det4.add64(0, acc.masked_det4);
            sums.iw_art4.add64(0, acc.iw_art4);
            sums.iw_det4.add64(0, acc.iw_det4);
            return b1;
        }
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
            sums.masked_mse.add64(0, mse_m);
            sums.iw_mse.add64(0, mse_i);
            let ((sd_m, sd4_m, sd2_m), (sd_i, sd4_i, sd2_i)) = if stable {
                // `stable_sd` is sized under `stable && full` above and read
                // here under `stable` — equivalent only because this arm IS
                // the `full` one. Pinned rather than assumed: moving this
                // pooling out of `if full` would otherwise read a zeroed
                // buffer and quietly emit zero masked/IW SSIM.
                debug_assert!(full && stable_sd.len() >= inner.end);
                crate::simd_ops::ssim_signal_inline_both(
                    &stable_sd[inner.clone()],
                    act_inner,
                    V1_MASK_K,
                    V1_IW_K,
                )
            } else {
                crate::simd_ops::ssim_channel_inline_both(
                    inner_mu1,
                    inner_mu2,
                    &ssq_v[inner.clone()],
                    &s12_v[inner.clone()],
                    act_inner,
                    V1_MASK_K,
                    V1_IW_K,
                )
            };
            sums.masked_ssim_d.add64(0, sd_m);
            sums.masked_ssim_d4.add64(0, sd4_m);
            sums.masked_ssim_d2.add64(0, sd2_m);
            sums.iw_ssim_d.add64(0, sd_i);
            sums.iw_ssim_d4.add64(0, sd4_i);
            sums.iw_ssim_d2.add64(0, sd2_i);
        }
        let ((art4_m, det4_m), (art4_i, det4_i)) = crate::simd_ops::edge_diff_channel_inline_both(
            inner_src, inner_dst, inner_mu1, inner_mu2, act_inner, V1_MASK_K, V1_IW_K,
        );
        sums.masked_art4.add64(0, art4_m);
        sums.masked_det4.add64(0, det4_m);
        sums.iw_art4.add64(0, art4_i);
        sums.iw_det4.add64(0, det4_i);
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
            free,
            crate::fused::ExtPoolsWork::default(),
            &[],
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
    // `free`: see [`fold_v1_one_band`] — forwarded verbatim to every band on
    // both the serial and the band-parallel path, so the two stay identical
    // by construction.
    free: crate::fused::FreeExtrasWork,
) {
    assert!(
        !matches!(h_src, FoldHSource::SelfBlur) || pools.is_some(),
        "FoldHSource::SelfBlur needs the band pool scratch for its H buffer"
    );
    debug_assert_eq!(rows.start % V1_BAND_ROWS, 0, "strips are 32-row aligned");
    // Band starts, in order. Bands are INDEPENDENT: each reads a clamped
    // window of the shared planes and produces its own sums, so the only
    // cross-band coupling is the accumulation itself.
    //
    // Held in a FIXED array rather than a `Vec`. This function runs once per
    // (strip, channel, scale), and a `Vec::new()` grown by `push` costs one
    // allocation plus up to two reallocs every time — measured at **232
    // allocations per 1152² fast-class compare** against a reused `V2Scratch`
    // (`zensim/tests/fastclass_alloc_steady_state.rs`,
    // `benchmarks/kernel_fastclass_2026-09-05.md` §3.2), i.e. ~4 per
    // (strip, channel), which is exactly this. The bound is structural, not a
    // guess: a strip is at most `STRIP_ROWS` rows and a band is
    // `V1_BAND_ROWS`, so there are at most `V1_BANDS_PER_STRIP` starts. The
    // assert states that rather than trusting it, so a future change to either
    // constant fails loudly here instead of silently truncating a band — which
    // would drop its rows from the sums and move feature bytes.
    let mut starts_buf = [0usize; V1_BANDS_PER_STRIP];
    let mut n_starts = 0usize;
    let mut b = rows.start;
    while b < rows.end {
        assert!(
            n_starts < V1_BANDS_PER_STRIP,
            "band starts overflow: rows {}..{} span {} rows, more than \
             V1_BANDS_PER_STRIP ({V1_BANDS_PER_STRIP}) bands of {V1_BAND_ROWS}",
            rows.start,
            rows.end,
            rows.end - rows.start
        );
        starts_buf[n_starts] = b;
        n_starts += 1;
        b = (b + V1_BAND_ROWS).min(rows.end);
    }
    let starts: &[usize] = &starts_buf[..n_starts];

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
                            let mut local = V1BasicSums::at_revision(free.revision());
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
                                free,
                            );
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
                    let mut local = V1BasicSums::at_revision(free.revision());
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
                        free,
                    );
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
        let mut local = V1BasicSums::at_revision(free.revision());
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
            free,
        );
        sums.merge(&local);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_channel_scale_at_revision(
    dense: &DenseAccum,
    grad: &GradientAccum,
    sum_blockiness: f64,
    n: usize,
    out: &mut [f64],
    omit_pjnd_fragility: bool,
    revision: FormulaRevision,
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
    let (m2, m4) = if dense.central.n > 0.0 {
        (dense.central.m2 / n_f, dense.central.m4 / n_f)
    } else {
        let raw2 = dense.sum_d2 / n_f;
        let raw3 = dense.sum_d3 / n_f;
        let raw4 = dense.sum_d4 / n_f;
        (
            (raw2 - mean_d * mean_d).max(0.0),
            (raw4 - 4.0 * mean_d * raw3 + 6.0 * mean_d * mean_d * raw2 - 3.0 * mean_d.powi(4))
                .max(0.0),
        )
    };
    let dev2 = m2.sqrt();
    let dev4 = m4.quarter_root(crate::det_math::RootForm::at_revision(Some(revision)));

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
    if !omit_pjnd_fragility {
        out[idx::PJND_FRAGILITY] = 1.0 - saturate(grad.sum_grad_src / n_f, C_PJND_GRAD);
    }
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

impl V2PreparedReference {}

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
    /// featacc: [`SumVar`] slots — `Seq` unless a measurement mode is set
    /// (f64-native elements: `c32` resolves to `Seq`, i.e. production
    /// accumulation, per the f64-element rule).
    block: Vec<(crate::featcanon::SumVar, crate::featcanon::SumVar)>,
    /// CSFW weighted-pool partials (Y channel + `csfw_block` only —
    /// untouched zeros otherwise).
    csfw: Vec<CsfwAccum>,
    /// REV4 per-scale cells (`gridblk`/`ringbasis`/`tailhist` plus this
    /// channel's share of `arttype`) — written by the dense/gradient hook
    /// bundles and the gridblk strip kernel, all inside `stream_phase_b`.
    /// Untouched zeros when every rev4 compute flag is off.
    rev4: Vec<Rev4CellAccum>,
    /// Band-local planes for the v1 pool replay (`v1_pools`); empty
    /// (never allocated) when the toggle is off.
    /// One scratch per band SLOT within a strip (`STRIP_ROWS /
    /// V1_BAND_ROWS`), so the band-parallel path reuses buffers instead of
    /// allocating ~580 KB per worker per strip — which measured as a net
    /// LOSS before this existed.
    pool_scratch: Vec<FoldPoolScratch>,
}

/// A bounded, reusable owned halo window and its independent v2 strip result.
/// Slots contain no full-image planes. Workers never touch running totals.
#[cfg(feature = "threads")]
struct Rev5StripJob {
    info: crate::feature_v2_stream::StripInfo,
    ch: usize,
    scratch: ScratchV2Strip,
    sized_for: usize,
    result: StreamChannelAccums,
    bands: [V1BasicSums; V1_BANDS_PER_STRIP],
}

// Private policy, chosen by measurement (benchmarks/rev5perf5_budget_design_2026-10-09.md):
// 256 MiB costs 0.7-21.9 % against uncapped at >=4096-wide inputs on 16-32 threads
// (128 MiB: 15.5-46.0 %), at 0.48-0.72x uncapped peak RSS. This bounds owned queue
// payload, not allocator overhead or the whole score.
#[cfg(feature = "threads")]
const REV5_JOB_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[cfg(feature = "threads")]
fn rev5_job_bytes(max_n: usize) -> Option<usize> {
    use core::mem::size_of;
    // Eleven float planes, the slot itself, and every fixed-length result
    // vector allocated by StreamChannelAccums::new(NUM_SCALES, 1). Queued
    // jobs never allocate the pool scratch's float planes.
    let cell = size_of::<DenseAccum>()
        + size_of::<GradientAccum>()
        + size_of::<AppendAccum>()
        + size_of::<V1BasicSums>()
        + size_of::<(crate::featcanon::SumVar, crate::featcanon::SumVar)>()
        + size_of::<CsfwAccum>()
        + size_of::<Rev4CellAccum>();
    let fixed = size_of::<Rev5StripJob>()
        + crate::NUM_SCALES * cell
        + size_of::<FoldPoolScratch>()
        + crate::NUM_SCALES * size_of::<Vec<(DenseAccum, GradientAccum)>>()
        + 3 * size_of::<(DenseAccum, GradientAccum)>();
    max_n
        .checked_mul(11)?
        .checked_mul(size_of::<f32>())?
        .checked_add(fixed)
}

// Below this many slots the walk keeps the original route. That route runs a
// strip's three channels in parallel (`fuse_channels`) and each channel's v1
// bands in parallel (`band_parallel`); a queued job runs its bands serially.
// Which is faster at a given slot count is measured, not assumed: see
// benchmarks/rev5perf5_budget_design_2026-10-09.md. Both routes produce
// identical bits; the floor changes scheduling only.
#[cfg(feature = "threads")]
const REV5_MIN_JOB_SLOTS: usize = 3;

#[cfg(feature = "threads")]
fn rev5_job_limit(max_n: usize, threads: usize, budget: usize) -> usize {
    let limit = rev5_job_bytes(max_n).map_or(0, |bytes| threads.min(16).min(budget / bytes));
    if limit < REV5_MIN_JOB_SLOTS { 0 } else { limit }
}

#[cfg(feature = "threads")]
fn rev5_prepare_jobs(jobs: &mut Vec<Rev5StripJob>, limit: usize, max_n: usize) {
    // Drop excess slots BEFORE growing parent scratch or new job planes.
    // A narrower subsequent pair cannot keep oversized cached planes outside
    // the current request's byte accounting.
    jobs.truncate(limit);
    for job in jobs.iter_mut() {
        if job.sized_for > max_n {
            job.scratch = ScratchV2Strip::new(0);
            job.sized_for = 0;
        }
    }
    if jobs.capacity() != limit {
        let old = core::mem::take(jobs);
        *jobs = Vec::with_capacity(limit);
        jobs.extend(old);
    }
}

#[cfg(feature = "threads")]
impl Rev5StripJob {
    fn new(info: crate::feature_v2_stream::StripInfo, ch: usize) -> Self {
        Self {
            info,
            ch,
            scratch: ScratchV2Strip::new(0),
            sized_for: 0,
            result: StreamChannelAccums::at_revision(crate::NUM_SCALES, 1, FormulaRevision::Rev5),
            bands: core::array::from_fn(|_| V1BasicSums::at_revision(FormulaRevision::Rev5)),
        }
    }

    /// Take the strip and size the scratch for its window.
    fn prepare(&mut self, info: crate::feature_v2_stream::StripInfo, ch: usize) {
        self.info = info;
        self.ch = ch;
        let n = info.plane_w * info.wide_h();
        if n > self.sized_for {
            // Release the previous allocation before constructing its larger
            // replacement, so both plane sets never coexist inside the cap.
            self.scratch = ScratchV2Strip::new(0);
            self.sized_for = 0;
            self.scratch = rev5_job_scratch(n);
            self.sized_for = n;
        }
    }

    fn gather<S: ImageSource, D: ImageSource>(
        &mut self,
        producer: &crate::feature_v2_stream::StripPlaneProducer<'_, S, D>,
        info: crate::feature_v2_stream::StripInfo,
        ch: usize,
        borrow_source: bool,
    ) {
        use crate::feature_v2_stream::Side;
        self.prepare(info, ch);
        let n = info.plane_w * info.wide_h();
        for (side, out) in [
            (Side::Source, &mut self.scratch.src_wide),
            (Side::Distorted, &mut self.scratch.dst_wide),
        ] {
            if side == Side::Source && borrow_source {
                continue;
            }
            if let Some(window) = producer.wide_window(side, ch, &info) {
                out[..n].copy_from_slice(window);
            } else {
                producer.fill_wide(side, ch, &info, &mut out[..n]);
            }
        }
    }
}

/// Rev5's basic hook reads vertical planes, so horizontal moments are dead
/// after vertical blur. Reuse two of those planes as activity-chain temps;
/// no append bs2 or separate activity temps are retained in queued jobs.
#[cfg(feature = "threads")]
fn rev5_job_scratch(n: usize) -> ScratchV2Strip {
    let mut scratch = ScratchV2Strip::new_for(n, StripPlaneNeeds::NONE);
    for plane in [
        &mut scratch.mu1_h,
        &mut scratch.mu2_h,
        &mut scratch.ssq_h,
        &mut scratch.s12_h,
        &mut scratch.mu1,
        &mut scratch.mu2,
        &mut scratch.ssq,
        &mut scratch.s12,
        &mut scratch.activity,
    ] {
        plane.resize(n, 0.0);
    }
    scratch
}

#[cfg(feature = "threads")]
fn rev5_cached_window<'a>(
    reference: Option<&'a [crate::streaming::XybPyramidLevel]>,
    info: &crate::feature_v2_stream::StripInfo,
    ch: usize,
) -> Option<&'a [f32]> {
    if !info.interior() {
        return None;
    }
    let (planes, width, height) = &reference?[info.scale];
    debug_assert_eq!((*width, *height), (info.plane_w, info.plane_h));
    Some(&planes[ch][(info.y0 - HALO_P) * width..(info.y0 + info.strip_h + HALO_P) * width])
}

#[cfg(feature = "threads")]
fn rev5_job_blur(
    info: &crate::feature_v2_stream::StripInfo,
    scratch: &mut ScratchV2Strip,
    revision: FormulaRevision,
    source: Option<&[f32]>,
) {
    let n = info.plane_w * info.wide_h();
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
        activity,
        ..
    } = scratch;
    let src = source.unwrap_or(&src_wide[..n]);
    // The moments in one pass where both blurs run their per-output trees:
    // the horizontal planes then live a column panel at a time in the first
    // rows of `mu1_h`..`s12_h` instead of filling them. Only pools of eight or
    // more threads take it, where the jobs' planes contend for the cache
    // (paired, B at revision 7, 1507×2048 / 1024²: eight threads −3.6% /
    // +0.1%, sixteen on eight cores −22.0% / −16.4%, 32 −10.8% / −3.1%); four
    // threads ran +2.4% / −3.0% and one thread 1–2% slower on the panels'
    // shorter row runs, so smaller pools and the serial walk keep two passes.
    let fused = rayon::current_num_threads() >= 8
        && crate::blur::fused_blur_hv_ssim_at_revision(
            src,
            &dst_wide[..n],
            &mut mu1[..n],
            &mut mu2[..n],
            &mut ssq[..n],
            &mut s12[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
            [
                &mut mu1_h[..n],
                &mut mu2_h[..n],
                &mut ssq_h[..n],
                &mut s12_h[..n],
            ],
        );
    if !fused {
        fused_blur_h_ssim_banded(
            src,
            &dst_wide[..n],
            &mut mu1_h[..n],
            &mut mu2_h[..n],
            &mut ssq_h[..n],
            &mut s12_h[..n],
            info.plane_w,
            info.wide_h(),
            false,
            revision,
        );
    }
    if !fused {
        for (horizontal, vertical) in [
            (&mu1_h[..n], &mut mu1[..n]),
            (&mu2_h[..n], &mut mu2[..n]),
            (&ssq_h[..n], &mut ssq[..n]),
            (&s12_h[..n], &mut s12[..n]),
        ] {
            crate::blur::box_blur_v_from_copy_at_revision(
                horizontal,
                vertical,
                info.plane_w,
                info.wide_h(),
                BLUR_RADIUS,
                revision,
            );
        }
    }
    // The activity plane in one pass where its blurs run their per-output
    // trees, `mu1_h` and `mu2_h` holding a panel's rows. Pools of sixteen or
    // more threads only (paired against the fused moments alone, B at
    // revision 7, 1507×2048 / 1024²: sixteen on eight cores −13.5% / −4.9%,
    // 32 −8.1% / −7.9%, but eight +1.4% / +1.0%).
    if !(fused
        && rayon::current_num_threads() >= 16
        && crate::blur::activity_at_revision(
            src,
            &mu1[..n],
            &mut activity[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
            &mut mu1_h[..n],
            &mut mu2_h[..n],
        ))
    {
        crate::simd_ops::abs_diff_into(src, &mu1[..n], &mut mu1_h[..n]);
        crate::blur::box_blur_1pass_at_revision(
            &mu1_h[..n],
            &mut activity[..n],
            &mut mu2_h[..n],
            info.plane_w,
            info.wide_h(),
            BLUR_RADIUS,
            revision,
        );
    }
}

/// One queued strip job's work: phase A into its own planes, phase B into its
/// own partials (and its retained rows when `output` is given). Runs on any
/// thread; [`rev5_merge_job`] adds the partials to the walk's totals.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn rev5_execute_job(
    job: &mut Rev5StripJob,
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    let info = &job.info;
    let scale = info.scale;
    let local = compute.at_scale(scale);
    let toggles = V2NewFeatureToggles {
        v1_pools: local.v1_pools,
        v1_only: false,
        gradient_features: local.gradient,
        // This running sum must be accumulated in original pixel order.
        blockiness: false,
        ..toggles
    };
    job.result.dense[scale] = DenseAccum::default();
    job.result.grad[scale] = GradientAccum::default();
    let cached_source = rev5_cached_window(reference, info, job.ch);
    // Revision 7: the strip's blockiness lane partials, here in the
    // parallel job (both branches below); the merge after the batch adds
    // them in strip order.
    if local.blockiness && local.formula_revision.row_lanes() {
        let n = info.plane_w * info.wide_h();
        let src = cached_source.unwrap_or(&job.scratch.src_wide[..n]);
        let mode = crate::featcanon::mode(local.formula_revision);
        let (v, h) = &mut job.result.block[scale];
        *v = crate::featcanon::SumVar::f64_elems(mode);
        *h = crate::featcanon::SumVar::f64_elems(mode);
        blockiness_sparse_strip_wide(
            src,
            &job.scratch.dst_wide[..n],
            info.plane_w,
            info.y0,
            info.strip_h,
            local.formula_revision,
            v,
            h,
        );
    }
    rev5_job_blur(
        info,
        &mut job.scratch,
        compute.formula_revision,
        cached_source,
    );
    let n = info.plane_w * info.wide_h();
    let src = cached_source.unwrap_or(&job.scratch.src_wide[..n]);
    stream_phase_b(
        &job.scratch,
        src,
        &job.scratch.dst_wide[..n],
        info,
        toggles,
        false,
        true,
        false,
        false,
        local.free_work(job.ch, scale),
        &mut job.result,
    );
    if fold_v1 {
        let raw = [src, &job.scratch.dst_wide[..n]];
        let vertical = FoldHSource::Vertical([
            &job.scratch.mu1[..n],
            &job.scratch.mu2[..n],
            &job.scratch.ssq[..n],
            &job.scratch.s12[..n],
        ]);
        let mut b0 = info.y0;
        for band in &mut job.bands[..info.strip_h.div_ceil(V1_BAND_ROWS)] {
            *band = V1BasicSums::at_revision(FormulaRevision::Rev5);
            b0 = fold_v1_one_band(
                b0,
                info.plane_w,
                info.y0 + info.strip_h,
                info.y0,
                HALO_P,
                info.plane_h,
                vertical,
                raw,
                band,
                None,
                local.free_work(job.ch, scale),
            );
        }
    }
}

/// Add one executed strip job's partials to the walk's totals. Jobs merge
/// once each, in the producer's order; do not reorder or tree-reduce them.
#[cfg(feature = "threads")]
fn rev5_merge_job(
    job: &Rev5StripJob,
    compute: &ComputeSet,
    fold_v1: bool,
    totals: &mut [StreamChannelAccums; 3],
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    let info = &job.info;
    let scale = info.scale;
    let local = compute.at_scale(scale);
    let total = &mut totals[job.ch];
    total.dense[scale].accumulate(&job.result.dense[scale]);
    total.grad[scale].accumulate(&job.result.grad[scale]);
    if fold_v1 {
        // Preserve each canonical 32-row band addition, never merge a
        // pre-reduced strip sum into the running basic accumulator.
        for band in &job.bands[..info.strip_h.div_ceil(V1_BAND_ROWS)] {
            total.v1[scale].merge(band);
        }
    }
    if local.blockiness {
        let (sum_v, sum_h) = &mut total.block[scale];
        if local.formula_revision.row_lanes() {
            let (v, h) = &job.result.block[scale];
            sum_v.merge_from(v);
            sum_h.merge_from(h);
        } else {
            // Revisions 4–6 keep one running sum: the strip's windows, here.
            let n = info.plane_w * info.wide_h();
            let src =
                rev5_cached_window(reference, info, job.ch).unwrap_or(&job.scratch.src_wide[..n]);
            let dst = &job.scratch.dst_wide[..n];
            blockiness_sparse_strip_wide(
                src,
                dst,
                info.plane_w,
                info.y0,
                info.strip_h,
                local.formula_revision,
                sum_v,
                sum_h,
            );
        }
    }
}

/// Queued strip jobs without batch barriers, for walks that retain nothing
/// and read no cached reference: each gathered job is spawned on the rayon
/// pool at once, finished jobs come back over a channel and merge in the
/// producer's order ([`rev5_merge_job`], the batch runner's merge), and the
/// producer waits only when `limit` jobs are in flight. The merge order is the
/// batch runner's, so every value has the batch runner's bits; the producer
/// gathers and converts while the pool computes.
#[cfg(feature = "threads")]
struct Rev5Pipeline {
    compute: ComputeSet,
    fold_v1: bool,
    limit: usize,
    free: Vec<Rev5StripJob>,
    in_flight: usize,
    next_seq: usize,
    merge_seq: usize,
    done: std::collections::BTreeMap<usize, Rev5StripJob>,
    tx: std::sync::mpsc::Sender<(usize, std::thread::Result<Rev5StripJob>)>,
    rx: std::sync::mpsc::Receiver<(usize, std::thread::Result<Rev5StripJob>)>,
}

#[cfg(feature = "threads")]
impl Rev5Pipeline {
    fn new(compute: ComputeSet, fold_v1: bool, limit: usize, free: Vec<Rev5StripJob>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel();
        Self {
            compute,
            fold_v1,
            limit,
            free,
            in_flight: 0,
            next_seq: 0,
            merge_seq: 0,
            done: std::collections::BTreeMap::new(),
            tx,
            rx,
        }
    }

    /// A job slot for the next strip, merging finished jobs (oldest first)
    /// until fewer than `limit` are in flight.
    fn slot(
        &mut self,
        info: crate::feature_v2_stream::StripInfo,
        ch: usize,
        totals: &mut [StreamChannelAccums; 3],
    ) -> Rev5StripJob {
        while self.in_flight >= self.limit {
            self.wait(totals);
        }
        self.free
            .pop()
            .unwrap_or_else(|| Rev5StripJob::new(info, ch))
    }

    /// Spawn a gathered job.
    fn submit(&mut self, mut job: Rev5StripJob, toggles: V2NewFeatureToggles) {
        let (seq, tx, compute, fold_v1) =
            (self.next_seq, self.tx.clone(), self.compute, self.fold_v1);
        self.next_seq += 1;
        self.in_flight += 1;
        rayon::spawn(move || {
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                rev5_execute_job(&mut job, &compute, toggles, fold_v1, None);
            }));
            // The walk owns the receiver until every spawned job reports.
            let _ = tx.send((seq, run.map(|()| job)));
        });
    }

    /// Take at least one finished job and merge every job that is next in
    /// order. A rayon worker waiting here runs pending pool work instead of
    /// blocking, so a walk called from inside the pool cannot starve its own
    /// jobs.
    fn wait(&mut self, totals: &mut [StreamChannelAccums; 3]) {
        let mut got = None;
        while got.is_none() {
            match self.rx.try_recv() {
                Ok(m) => got = Some(m),
                Err(std::sync::mpsc::TryRecvError::Empty) => match rayon::yield_now() {
                    Some(rayon::Yield::Executed) => {}
                    Some(rayon::Yield::Idle) => std::thread::yield_now(),
                    None => got = Some(self.rx.recv().expect("strip job channel")),
                },
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    unreachable!("the pipeline holds a sender")
                }
            }
        }
        let mut next = got;
        while let Some((seq, run)) = next.take() {
            let job = run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
            self.done.insert(seq, job);
            next = self.rx.try_recv().ok();
        }
        while let Some(job) = self.done.remove(&self.merge_seq) {
            rev5_merge_job(&job, &self.compute, self.fold_v1, totals, None);
            self.merge_seq += 1;
            self.in_flight -= 1;
            self.free.push(job);
        }
    }

    /// Finish and merge every job in flight; returns the job slots.
    fn drain(&mut self, totals: &mut [StreamChannelAccums; 3]) {
        while self.in_flight > 0 {
            self.wait(totals);
        }
    }
}

/// A finished task of [`rev5_block_walk`]: a produced chunk (side, chunk
/// index) or a strip job (sequence number).
#[cfg(feature = "threads")]
enum BlockWalkMsg {
    Chunk(
        usize,
        usize,
        std::thread::Result<crate::feature_v2_blocks::ChunkPlanes>,
    ),
    // Boxed: a job slot is about 21 KB, a chunk message about 120 bytes.
    Job(usize, std::thread::Result<Box<Rev5StripJob>>),
}

/// The plain revision-5+ walk with parallel row production
/// ([`crate::feature_v2_blocks`]): chunk tasks convert rows and downscale them
/// into immutable blocks, strip jobs copy their own windows out of the blocks,
/// and this thread only schedules, adds the `globaff` rows and merges finished
/// jobs. Jobs are submitted, and merged, in the streaming walk's order: strips
/// in ascending order within each scale and channels in order within a strip,
/// so every accumulator receives the same partials in the same order and every
/// value keeps its bits. Production runs at most a few chunks ahead of the
/// rows the next strip needs.
#[cfg(feature = "threads")]
#[allow(clippy::too_many_arguments)]
fn rev5_block_walk<S: ImageSource, D: ImageSource, const ALL_CHANNELS: bool>(
    source: &S,
    distorted: &D,
    dims: &[(usize, usize)],
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    limit: usize,
    omit_scale0_xb: bool,
    slots: &mut Vec<Rev5StripJob>,
    pool: &std::sync::Arc<crate::feature_v2_blocks::BlockPool>,
    totals: &mut [StreamChannelAccums; 3],
    globaff: &mut [[crate::globaff::GlobAffAccum; 3]],
    stop: Option<&dyn enough::Stop>,
) {
    use crate::feature_v2_blocks::{CHUNK_ROWS, RowBlock, gather_window, produce_chunk};
    use crate::feature_v2_stream::StripInfo;
    use std::collections::{BTreeMap, VecDeque};
    use std::sync::Arc;

    let n_scales = dims.len();
    let h0 = dims[0].1;
    let n_chunks = h0.div_ceil(CHUNK_ROWS);
    let threads = rayon::current_num_threads().max(2);
    // Production may run this many scale-0 rows past the rows the next
    // strip needs, with at most this many chunk tasks (one per side) queued.
    // At most twelve strip jobs hold scratch at once: each holds about
    // 9.8 MB at 1507 columns, and sixteen at once overflowed a 96 MiB L3 at
    // sixteen threads on eight cores (paired, B at revision 7, 1507×2048:
    // twelve −13.2% and eight −20.8% against sixteen; at 32 threads on two
    // CCDs twelve −2.5% and eight +8.8%). Eight chunks of look-ahead instead
    // of one per thread: −3.6% at sixteen threads, −7.6% at 32 (1024²).
    let ahead = threads.min(8) * CHUNK_ROWS;
    let prod_limit = 2 * threads.min(8);
    let limit = limit.min(12);
    // The streaming producer's conversion revision.
    let revision = toggles.formula_revision;
    let globaff_on = !globaff.is_empty();
    let compute = *compute;

    rayon::in_place_scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<BlockWalkMsg>();
        // Retained blocks per side, channel and scale, in row order; rows
        // `[.., hi[scale])` are produced on both sides.
        let mut planes: [[Vec<VecDeque<Arc<RowBlock>>>; 3]; 2] = core::array::from_fn(|_| {
            core::array::from_fn(|_| (0..n_scales).map(|_| VecDeque::new()).collect())
        });
        let mut hi = vec![0usize; n_scales];
        let mut arrived: BTreeMap<usize, [Option<crate::feature_v2_blocks::ChunkPlanes>; 2]> =
            BTreeMap::new();
        let mut appended = 0usize;
        let mut next_chunk = 0usize;
        let mut prod_in_flight = 0usize;
        let mut next_ks = vec![0usize; n_scales];
        // The next channel to submit within each scale's current strip.
        let mut next_ch = vec![0usize; n_scales];
        let mut free = core::mem::take(slots);
        // Strip scratch released by finished jobs whose merge does not read
        // it (every job but revision-5/6 blockiness, which the merge sums
        // from the job's windows): `limit` bounds the scratch in use, so a
        // finished job never holds its planes while it waits to merge.
        let mut free_scratch: Vec<(ScratchV2Strip, usize)> = Vec::new();
        let (mut in_scratch, mut pending) = (0usize, 0usize);
        let (mut next_seq, mut merge_seq) = (0usize, 0usize);
        let mut done: BTreeMap<usize, Rev5StripJob> = BTreeMap::new();
        let mut stopped = false;
        let keeps_scratch = |scale: usize| {
            let local = compute.at_scale(scale);
            local.blockiness && !local.formula_revision.row_lanes()
        };

        let strip_at = |s: usize, ks: usize| {
            let (plane_w, plane_h) = dims[s];
            let y0 = ks * STRIP_ROWS;
            (y0 < plane_h).then(|| StripInfo {
                scale: s,
                y0,
                strip_h: STRIP_ROWS.min(plane_h - y0),
                plane_w,
                plane_h,
            })
        };
        // The scale-0 rows that make a strip's window complete.
        let need0 = |info: &StripInfo| {
            ((info.y0 + info.strip_h + HALO_P).min(info.plane_h) << info.scale).min(h0)
        };

        loop {
            if !stopped && stop.is_some_and(|token| token.check().is_err()) {
                stopped = true;
            }

            // Production, a bounded distance past the next strip's rows.
            if !stopped {
                let want0 = (0..n_scales)
                    .filter_map(|s| strip_at(s, next_ks[s]).map(|info| need0(&info)))
                    .min()
                    .unwrap_or(h0);
                while next_chunk < n_chunks
                    && prod_in_flight + 2 <= prod_limit
                    && next_chunk * CHUNK_ROWS < want0 + ahead
                {
                    let k = next_chunk;
                    for side in 0..2 {
                        let tx = tx.clone();
                        scope.spawn(move |_| {
                            let run =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    if side == 0 {
                                        produce_chunk(
                                            source,
                                            k,
                                            dims,
                                            omit_scale0_xb,
                                            revision,
                                            pool,
                                        )
                                    } else {
                                        produce_chunk(
                                            distorted,
                                            k,
                                            dims,
                                            omit_scale0_xb,
                                            revision,
                                            pool,
                                        )
                                    }
                                }));
                            // The walk holds the receiver until every task reports.
                            let _ = tx.send(BlockWalkMsg::Chunk(side, k, run));
                        });
                    }
                    next_chunk += 1;
                    prod_in_flight += 2;
                }
            }

            // Every ready strip, shallowest scale first, while job slots last.
            'emit: for s in 0..n_scales {
                if stopped {
                    break;
                }
                while let Some(info) = strip_at(s, next_ks[s]) {
                    if hi[s] < (info.y0 + info.strip_h + HALO_P).min(info.plane_h) {
                        break;
                    }
                    if next_ch[s] == 0 {
                        // `globaff`: every strip of every scale, all three
                        // channels, in strip order, before the dead-scale skip.
                        if globaff_on {
                            let y1 = info.y0 + info.strip_h;
                            for (ch, acc) in globaff[s].iter_mut().enumerate() {
                                let [src, dst] = [&planes[0][ch][s], &planes[1][ch][s]];
                                for (bs, bd) in src.iter().zip(dst) {
                                    let (a, b) = (bs.r0().max(info.y0), bs.end().min(y1));
                                    if a < b {
                                        acc.add_rows(bs.rows(a, b), bd.rows(a, b), info.plane_w);
                                    }
                                }
                            }
                        }
                        if compute.scale_dead(s) {
                            #[cfg(test)]
                            SKIPPED_SCALES.with_borrow_mut(|v| *v |= 1 << s);
                            next_ch[s] = 3;
                        }
                    }
                    let local = compute.at_scale(s);
                    let strip_toggles = V2NewFeatureToggles {
                        v1_pools: local.v1_pools,
                        v1_only: !local.v2_blocks,
                        gradient_features: local.gradient,
                        blockiness: local.blockiness,
                        ..toggles
                    };
                    while next_ch[s] < 3 {
                        let ch = next_ch[s];
                        if !ALL_CHANNELS && !compute.channel_active(s, ch) {
                            next_ch[s] += 1;
                            continue;
                        }
                        if in_scratch >= limit {
                            break 'emit;
                        }
                        let lo = info.y0.saturating_sub(HALO_P);
                        let top = (info.y0 + info.strip_h + HALO_P).min(info.plane_h);
                        let window = |side: usize| -> Vec<Arc<RowBlock>> {
                            planes[side][ch][s]
                                .iter()
                                .filter(|b| b.end() > lo && b.r0() < top)
                                .cloned()
                                .collect()
                        };
                        let (src_blocks, dst_blocks) = (window(0), window(1));
                        let mut job = free.pop().unwrap_or_else(|| Rev5StripJob::new(info, ch));
                        if job.sized_for == 0
                            && let Some((scratch, sized_for)) = free_scratch.pop()
                        {
                            job.scratch = scratch;
                            job.sized_for = sized_for;
                        }
                        job.prepare(info, ch);
                        let (seq, tx) = (next_seq, tx.clone());
                        scope.spawn(move |_| {
                            let run =
                                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let n = info.plane_w * info.wide_h();
                                    gather_window(
                                        &src_blocks,
                                        &info,
                                        &mut job.scratch.src_wide[..n],
                                    );
                                    gather_window(
                                        &dst_blocks,
                                        &info,
                                        &mut job.scratch.dst_wide[..n],
                                    );
                                    drop((src_blocks, dst_blocks));
                                    rev5_execute_job(
                                        &mut job,
                                        &compute,
                                        strip_toggles,
                                        fold_v1,
                                        None,
                                    );
                                }));
                            let _ = tx.send(BlockWalkMsg::Job(seq, run.map(|()| Box::new(job))));
                        });
                        next_seq += 1;
                        in_scratch += 1;
                        pending += 1;
                        next_ch[s] += 1;
                    }
                    next_ks[s] += 1;
                    next_ch[s] = 0;
                    // Retire the blocks no later strip of this scale reads.
                    let floor = strip_at(s, next_ks[s])
                        .map_or(usize::MAX, |next| next.y0.saturating_sub(HALO_P));
                    for side in &mut planes {
                        for ch in side.iter_mut() {
                            while ch[s].front().is_some_and(|b| b.end() <= floor) {
                                ch[s].pop_front();
                            }
                        }
                    }
                }
            }

            let emitted = (0..n_scales).all(|s| strip_at(s, next_ks[s]).is_none());
            if (emitted || stopped) && pending == 0 && prod_in_flight == 0 {
                break;
            }

            // Wait for at least one finished task, then take every other
            // finished one. A rayon worker waiting here runs pending pool work.
            let mut msg = None;
            while msg.is_none() {
                match rx.try_recv() {
                    Ok(m) => msg = Some(m),
                    Err(std::sync::mpsc::TryRecvError::Empty) => match rayon::yield_now() {
                        Some(rayon::Yield::Executed) => {}
                        Some(rayon::Yield::Idle) => std::thread::yield_now(),
                        None => msg = Some(rx.recv().expect("block walk channel")),
                    },
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        unreachable!("the walk holds a sender")
                    }
                }
            }
            while let Some(m) = msg.take() {
                match m {
                    BlockWalkMsg::Chunk(side, k, run) => {
                        prod_in_flight -= 1;
                        let chunk =
                            run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
                        arrived.entry(k).or_default()[side] = Some(chunk);
                        // Append whole chunks (both sides) in order.
                        while let Some(both) = arrived.get(&appended)
                            && both.iter().all(Option::is_some)
                        {
                            let both = arrived.remove(&appended).expect("present");
                            for (side, chunk) in both.into_iter().enumerate() {
                                for (ch, scales) in chunk.expect("present").into_iter().enumerate()
                                {
                                    for (s, block) in scales.into_iter().enumerate() {
                                        if let Some(block) = block {
                                            if ch == 1 {
                                                hi[s] = hi[s].max(block.end());
                                            }
                                            planes[side][ch][s].push_back(block);
                                        }
                                    }
                                }
                            }
                            appended += 1;
                        }
                    }
                    BlockWalkMsg::Job(seq, run) => {
                        let mut job =
                            *run.unwrap_or_else(|payload| std::panic::resume_unwind(payload));
                        if !keeps_scratch(job.info.scale) {
                            let scratch =
                                core::mem::replace(&mut job.scratch, ScratchV2Strip::new(0));
                            free_scratch.push((scratch, core::mem::take(&mut job.sized_for)));
                            in_scratch -= 1;
                        }
                        done.insert(seq, job);
                        while let Some(job) = done.remove(&merge_seq) {
                            rev5_merge_job(&job, &compute, fold_v1, totals, None);
                            if keeps_scratch(job.info.scale) {
                                in_scratch -= 1;
                            }
                            merge_seq += 1;
                            pending -= 1;
                            free.push(job);
                        }
                    }
                }
                msg = rx.try_recv().ok();
            }
        }
        // Reattach the released scratch so the next walk's first `limit`
        // slots keep their planes.
        free.sort_by_key(|job| core::cmp::Reverse(job.sized_for));
        for job in free.iter_mut().filter(|job| job.sized_for == 0) {
            let Some((scratch, sized_for)) = free_scratch.pop() else {
                break;
            };
            job.scratch = scratch;
            job.sized_for = sized_for;
        }
        free.sort_by_key(|job| core::cmp::Reverse(job.sized_for));
        *slots = free;
    });
}

#[cfg(feature = "threads")]
fn rev5_run_jobs(
    jobs: &mut [Rev5StripJob],
    compute: &ComputeSet,
    toggles: V2NewFeatureToggles,
    fold_v1: bool,
    totals: &mut [StreamChannelAccums; 3],
    reference: Option<&[crate::streaming::XybPyramidLevel]>,
) {
    use rayon::prelude::*;
    jobs.par_iter_mut()
        .for_each(|job| rev5_execute_job(job, compute, toggles, fold_v1, reference));
    // Every cell's kernel partial is merged once, in the producer's order.
    // Do not replace this loop with a parallel/tree reduction.
    for job in jobs.iter() {
        rev5_merge_job(job, compute, fold_v1, totals, reference);
    }
}

impl StreamChannelAccums {
    /// `n_band_slots` is the number of band scratches this channel keeps —
    /// see [`band_slots_for`]. It is a CAPACITY, not a tiling: bands are
    /// chunked across whatever slots exist (`fold_v1_basic_bands`), so any
    /// count in `1..=V1_BANDS_PER_STRIP` computes the same bytes.
    fn at_revision(n_scales: usize, n_band_slots: usize, revision: FormulaRevision) -> Self {
        Self {
            dense: vec![DenseAccum::default(); n_scales],
            grad: vec![GradientAccum::default(); n_scales],
            app: vec![AppendAccum::default(); n_scales],
            v1: vec![V1BasicSums::at_revision(revision); n_scales],
            block: vec![
                (
                    crate::featcanon::SumVar::f64_elems(crate::featcanon::mode(revision)),
                    crate::featcanon::SumVar::f64_elems(crate::featcanon::mode(revision))
                );
                n_scales
            ],
            csfw: vec![CsfwAccum::default(); n_scales],
            rev4: (0..n_scales).map(|_| Rev4CellAccum::default()).collect(),
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
///
/// Its only call site is inside a `#[cfg(feature = "threads")]` block, so a
/// `threads`-off build (e.g. the `feature-regime-v2`-alone CI permutation)
/// has no caller for it; `cfg_attr` scopes the allow to exactly that case
/// rather than silencing dead-code detection when `threads` IS on.
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
    revision: FormulaRevision,
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
        info.scale,
        revision,
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
    // The resolved per-channel work order. Do not reconstruct it from public
    // toggles: that loses private subset restrictions and runs dead reductions.
    free: crate::fused::FreeExtrasWork,
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

        let d = dense_block_kernel_at_revision(
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
            toggles.formula_revision,
        );
        acc.dense[scale].accumulate(&d);
        if toggles.gradient_features {
            let g_off = (HALO_P - 1) * width;
            let g_n = width * (strip_h + 2);
            let g = gradient_block_kernel_at_revision(
                &src_win[g_off..g_off + g_n],
                &dst_win[g_off..g_off + g_n],
                act_strip,
                Some((mu1_strip, mu2_strip)),
                width,
                strip_h,
                toggles.formula_revision,
            );
            acc.grad[scale].accumulate(&g);
        }
    }

    if fold_v1 {
        stream_fold_v1(
            scr,
            src_win,
            dst_win,
            info,
            toggles,
            v2_blocks,
            band_parallel,
            self_blur,
            free,
            acc,
        );
    }

    if v2_blocks && toggles.blockiness {
        let (sum_v, sum_h) = &mut acc.block[scale];
        blockiness_sparse_strip_wide(
            &src_win[..n_wide],
            &dst_win[..n_wide],
            width,
            y0,
            strip_h,
            toggles.formula_revision,
            sum_v,
            sum_h,
        );
    }
}

/// Replay each basic band directly into the running total. Keeping this
/// ordered merge separate lets strip tasks evaluate v2 in parallel without
/// reassociating any basic band or sparse blockiness sum.
#[allow(clippy::too_many_arguments)]
fn stream_fold_v1(
    scr: &ScratchV2Strip,
    src_win: &[f32],
    dst_win: &[f32],
    info: &crate::feature_v2_stream::StripInfo,
    toggles: V2NewFeatureToggles,
    v2_blocks: bool,
    band_parallel: bool,
    self_blur: bool,
    free: crate::fused::FreeExtrasWork,
    acc: &mut StreamChannelAccums,
) {
    let width = info.plane_w;
    let y0 = info.y0;
    let strip_h = info.strip_h;
    let scale = info.scale;
    let n_wide = width * info.wide_h();
    fold_v1_basic_bands(
        width,
        y0..y0 + strip_h,
        y0,
        HALO_P,
        info.plane_h,
        if free.revision() >= FormulaRevision::Rev5 && v2_blocks {
            FoldHSource::Vertical([
                &scr.mu1[..n_wide],
                &scr.mu2[..n_wide],
                &scr.ssq[..n_wide],
                &scr.s12[..n_wide],
            ])
        } else if self_blur {
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
        // FREE EXTRAS: the fused kernel carries an extra accumulator only
        // when a block that is OFF will finalize it (`ComputeSet::
        // free_work`). With the owning blocks on, their own kernels own
        // those slots and every flag is false, so the full 944 walk is
        // untouched.
        free,
    );
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
#[allow(clippy::too_many_arguments)]
fn blockiness_sparse_strip_wide(
    src_wide: &[f32],
    dst_wide: &[f32],
    width: usize,
    y0: usize,
    strip_h: usize,
    revision: FormulaRevision,
    sum_v: &mut crate::featcanon::SumVar,
    sum_h: &mut crate::featcanon::SumVar,
) {
    use crate::featcanon::{LanesF64, SumVar};
    let spread = revision.row_lanes();
    // Revision 7: the strip's own lane partials, added to the running lanes
    // once, so a strip's terms can be computed apart from the others (in
    // parallel) and merged in strip order with the same bits.
    let partial = spread && matches!((&*sum_v, &*sum_h), (SumVar::L64(_), SumVar::L64(_)));
    let (mut pv, mut ph) = (SumVar::L64(LanesF64::zero()), SumVar::L64(LanesF64::zero()));
    let (v, h) = if partial {
        (&mut pv, &mut ph)
    } else {
        (&mut *sum_v, &mut *sum_h)
    };
    for k in 0..strip_h {
        let y = y0 + k;
        blockiness_row(
            src_wide,
            dst_wide,
            (HALO_P + k) * width,
            width,
            y.is_multiple_of(BLOCK_LATTICE) && y > 0,
            spread,
            v,
            h,
        );
    }
    if partial {
        sum_v.merge_from(&pv);
        sum_h.merge_from(&ph);
    }
}

// ============================================================================
// REV4 C1 `gridblk` strip kernel (design note §C1).
//
// `ẽ = (|Δdst| − |Δsrc|) / ((act_a + act_b)/2 + C_ACTIVITY)` is computed
// for EVERY vertical boundary (x ∈ 1..w) and horizontal boundary
// (y ∈ 1..h) — not the sparse lattice-only subset `blockiness` visits,
// because the winning phase is data-dependent and the profile needs all
// `2n` boundaries. The winning phase is only known after the whole plane
// is walked, so the kernel STORES `ẽ` into the cell's `plane_v`/`plane_h`
// f32 buffers and `finish_gridblk_cell` rescans the on-grid positions
// only (~2n/P hat evaluations — the store-then-rescan shape).
//
// Wide-window contract (same as `blockiness_sparse_strip_wide`, extended
// by the activity plane): `*_wide` are `width * (strip_h + 2*HALO_P)`,
// buffer row `HALO_P + k` = plane row `y0 + k`. `act_wide` covers the
// identical wide window including halo rows (phase A fills all of
// `scratch.activity`), so the H boundary at plane row `y0` reads the REAL
// plane row `y0 − 1` from the halo, and the scalar/V8 paths index the
// same offsets.
//
// PROFILE DETERMINISM: the emitted values are defined by THIS kernel —
// per-phase `Σ|ẽ|`/`Σẽ` accumulate as: (a) two V8 lane pairs for the
// chunked V span (chunks start at x = 1 + 8c, so lane k of parity-j
// chunks covers positions ≡ (1 + 8j + k) mod period — the fold happens
// once per strip into the f64 phase totals); (b) per-row V8 accumulators
// for H (a row's boundaries share phase `y mod period`), lane-summed in
// fixed f32 order then added to f64; (c) scalar tails fold into the f64
// phase totals in row order. Every engine, tier and thread count replays
// the identical sequence, so the feature is bit-deterministic while its
// values carry f32-lane accumulation precision — the documented
// exception to the row-ordered-f64 convention, chosen because a
// per-boundary scalar scatter cannot meet the family's runtime budget.
// ============================================================================

// ============================================================================
// REV4 C4 `arttype` bleed mask builder (design note §C4).
//
// The mask marks every pixel within Chebyshev distance 1 of a dst-Y
// edge — where "edge" is the gradient kernel's own test
// (`(gx² + gy²).sqrt() >= C_RING_EDGE`, x-neighbours column-clamped,
// y-neighbours the caller-supplied rows). Built once per strip before
// the channel fan-out so the serial path (X/B processed before Y) has it
// ready for the chroma gradient hooks.
//
// `dh` holds the dst-Y plane rows `reflect_101(y0 − 2 + r, plane_h)` for
// `r ∈ 0..strip_h + 4`, fetched by the caller (producer `rows` in the
// streaming walk, `gather_strip_halo` in the materialized path). The
// edge formula is evaluated in V8 f32 uniformly at every position —
// the deterministic definition; scalar tails use the identical f32 op
// sequence. `mul_add` is deliberately NOT used (its FMA fusion is
// tier-dependent — the same reason the production kernels spell
// `a*b + c` out).
// ============================================================================

/// Per-strip C4 mask scratch (sized by the caller once per walk).
#[derive(Default)]
struct BleedMaskWork {
    /// Dst-Y rows for `reflect_101(y0 − 2 + r)`, r ∈ 0..strip_h+4.
    dh: Vec<f32>,
    /// One raw edge-flag row (width f32, {0,1}).
    e_row: Vec<f32>,
    /// Horizontally-dilated edge rows j ∈ 0..strip_h+2 (E at plane row
    /// `y0 − 1 + j`, zero when that row is out of plane).
    ehd: Vec<f32>,
    /// The finished mask, `width * strip_h`, {0.0, 1.0}.
    mask: Vec<f32>,
}

impl BleedMaskWork {
}

pub(crate) fn compute_folded_v1_372_streaming_stoppable(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    max_pixels: Option<usize>,
    parallel: bool,
    scratch: &mut V2Scratch,
    // The serving PLAN, resolved by `fold_engine::score_plan` from the
    // profile's bakes. `None` = the unconditional pre-2026-08-31 behaviour
    // (`V1PoolsMode::Full`, v1-only, v1 layout) — spelled out below as a
    // literal rather than reconstructed from a plan, so the default path is
    // the SAME code, not merely an equivalent one.
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Result<(Vec<f64>, [f64; 3]), ZensimError> {
    crate::metric::validate_pair_dims(source, distorted)?;
    crate::metric::check_within_max_pixels(source.width(), source.height(), max_pixels)?;
    let toggles = match plan {
        // A plan asks for its own (compute, layout) pair — which is how a
        // bake declaring a layout wider than 372 becomes servable at all.
        Some(p) => p.toggles(),
        None => V2NewFeatureToggles {
            v1_pools: V1PoolsMode::Full,
            v1_only: true,
            ..V2NewFeatureToggles::default()
        },
    };
    if let Some(p) = plan
        && let Some(sampling) = p.compute.sampling
    {
        let dims = sampling.dims(source.width(), source.height());
        let mut mo = MeanOffsetRows::at_revision(dims[0].0, dims[0].1, toggles.formula_revision);
        let res = foldapp_streaming_walk(
            source,
            distorted,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                stop,
                compute: Some(p.compute),
                mean_offset: Some(&mut mo),
                ..Default::default()
            },
        );
        return Ok((res.into_features(), mo.finish()));
    }
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
        let mut mo = MeanOffsetRows::at_revision(
            padded_src.width(),
            padded_src.height(),
            toggles.formula_revision,
        );
        let res = foldapp_streaming_walk(
            &padded_src,
            &padded_dst,
            parallel,
            toggles,
            crate::feature_v2_stream::FrontEnd::Sdr,
            scratch,
            FoldWalkExtras {
                stop,
                compute: plan.map(|p| p.compute),
                mean_offset: Some(&mut mo),
                ..Default::default()
            },
        );
        return Ok((res.into_features(), mo.finish()));
    }
    let mut mo =
        MeanOffsetRows::at_revision(source.width(), source.height(), toggles.formula_revision);
    let res = foldapp_streaming_walk(
        source,
        distorted,
        parallel,
        toggles,
        crate::feature_v2_stream::FrontEnd::Sdr,
        scratch,
        FoldWalkExtras {
            stop,
            compute: plan.map(|p| p.compute),
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

pub(crate) fn compute_folded_v1_372_with_ref_stoppable(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    parallel: bool,
    scratch: &mut V2Scratch,
    pool_mode: Option<V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Option<(Vec<f64>, [f64; 3])> {
    if !ref_feed_admits(precomputed, distorted, plan) {
        return None;
    }
    let (cw, ch) = (precomputed.scales[0].1, precomputed.scales[0].2);
    let toggles = plan.map_or(
        V2NewFeatureToggles {
            v1_pools: pool_mode.unwrap_or(V1PoolsMode::Full),
            v1_only: true,
            ..V2NewFeatureToggles::default()
        },
        crate::feature_plan::Plan::toggles,
    );
    let mut mo = MeanOffsetRows::at_revision(cw, ch, toggles.formula_revision);
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
            stop,
            mean_offset: Some(&mut mo),
            ref_planes: Some(&precomputed.scales),
            compute: plan.map(|p| p.compute),
            ..Default::default()
        },
    );
    Some((res.into_features(), mo.finish()))
}

/// Whether `precomputed` can feed the source side of a fold walk over
/// `distorted` under `plan` — the ref-cached form's admission test, shared by
/// [`compute_folded_v1_372_with_ref_impl`] and the retaining
/// [`compute_folded_with_ref_retained`].
///
/// Refuses a sampled cache or plan, a pre-Rev5 v2 plan, HDR input, a
/// distorted image not at the cache's compute dims, a cache that fails
/// [`cached_ref_feed_usable`], and a plan with a restored-cut side pass
/// (`mapdev` / `z1max`): those rebuild both pyramids from the source pixels,
/// which a ref-fed walk does not have (the walk asserts on that combination).
pub(crate) fn ref_feed_admits(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    plan: Option<&crate::feature_plan::Plan>,
) -> bool {
    if precomputed.sampling.is_some()
        || plan.is_some_and(|p| {
            p.compute.sampling.is_some()
                || (p.compute.v2_blocks && p.compute.formula_revision < FormulaRevision::Rev5)
        })
    {
        return false;
    }
    let Some(&(_, cw, ch)) = precomputed.scales.first() else {
        return false;
    };
    distorted.width() == cw
        && distorted.height() == ch
        && cached_ref_feed_usable(&precomputed.scales, cw, ch)
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
    fn at_revision(width: usize, height: usize, revision: FormulaRevision) -> Self {
        // Rev5 exposes no global XYB offset: the supported model inputs do
        // not consume it. Keep a zero metadata value without an image pass.
        let height = if revision >= FormulaRevision::Rev5 {
            0
        } else {
            height
        };
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
        if self.height == 0 {
            return [0.0; 3];
        }
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
    // A cancelled partial walk is discarded by the named scorer's post-check.
    pub(crate) stop: Option<&'a dyn enough::Stop>,
    /// Resolved model/research plan; raw public extraction keeps all channels.
    pub(crate) compute: Option<ComputeSet>,
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
    let compute = extras
        .compute
        .unwrap_or_else(|| ComputeSet::from_toggles(toggles));
    if compute.full_res_xb && compute.coarse_y_only_scales == 0 {
        foldapp_streaming_walk_impl::<S, D, true>(
            source, distorted, parallel, toggles, front_end, scratch, extras,
        )
    } else {
        assert!(compute.allows_full_res_y_subset());
        foldapp_streaming_walk_impl::<S, D, false>(
            source, distorted, parallel, toggles, front_end, scratch, extras,
        )
    }
}

fn foldapp_streaming_walk_impl<S: ImageSource, D: ImageSource, const ALL_CHANNELS: bool>(
    source: &S,
    distorted: &D,
    parallel: bool,
    toggles: V2NewFeatureToggles,
    front_end: crate::feature_v2_stream::FrontEnd,
    scratch: &mut V2Scratch,
    extras: FoldWalkExtras<'_>,
) -> ZensimV2Result {
    let FoldWalkExtras {
        stop,
        mut mean_offset,
        ref_planes,
        compute,
    } = extras;
    use crate::feature_v2_stream::StripPlaneProducer;
    // ITEM D: one derivation of WHAT this request computes. Every local
    // below reads from it instead of re-deriving from `toggles`.
    let mut compute = compute.unwrap_or_else(|| ComputeSet::from_toggles(toggles));
    // rev5: plan-supplied sets are already scoped (`from_toggles` and
    // `Plan::normalized` both apply `rev5_scope`); a hand-built set is
    // clamped here so the walk can never run a family Rev5 does not serve.
    compute.rev5_scope();
    let fold_v1 = compute.v1_basic;
    // BLOCK-SKIPPING: a v1-only request computes NOTHING v2-era. Every
    // v2 toggle is forced off in `ComputeSet::from_toggles` rather than
    // asserted, so the caller can set `v1_only` on top of any existing
    // toggle set and get the v1 blocks of that request — which is what
    // makes it a drop-in A/B in the bench.
    let v2_blocks = compute.v2_blocks;
    // BAND PARALLELISM: the channel fan-out is only 3-way, and the fold was
    // MEASURED to saturate at exactly 3 threads (2.27x) and REGRESS beyond
    // it. Bands inside a channel are independent, so they are the next axis;
    // rayon nests fine inside the channel fan-out.
    let band_parallel = parallel;
    assert!(
        true,
        "append2_dst_activity requires append2_block (it refines the BANDVIS lanes)"
    );
    assert!(
        true,
        "csfw_block requires append2_block (f944+ sits after the append2 block)"
    );
    assert!(true, "gmsbank requires the full f1322 prefix layout");
    assert!(
        true,
        "mapdev requires the C8 layout (f1502 sits after f1501)"
    );
    assert!(
        true,
        "z1max requires the mapdev layout (f1562 sits after f1561)"
    );
    assert!(
        true,
        "gmsnative requires the z1max layout (f1790 sits after f1789)"
    );
    assert!(
        true,
        "globaff requires the gmsnative layout (f1867 sits after f1866)"
    );
    assert!(
        true,
        "globaff is computed only at formula revision 6, into its own layout"
    );
    assert!(
        true,
        "globaff is defined on the walk's own pyramid, not a sampled one"
    );
    // Route-local derived φ: the SAME weighting mechanism on both routes,
    // pre-composed with each route's own encoding (design §6 — runtime
    // never inverts an encoding; the route-dependence lives entirely in
    // which constant set is selected, the `BV_DELTA_*` pattern).
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

    if let Some(sampling) = compute.sampling {
        dims = sampling.dims(w0, h0);
    }
    let strip_max_n = dims[0].0 * (STRIP_ROWS + 2 * HALO_P);
    // HOISTED so the strip scratch can be sized to the planes this walk will
    // actually write (fold-footprint lane). Both flags are strip-independent —
    // they were computed inside the loop purely because that is where they are
    // read — and the loop now reads these, so there is one definition of each.
    #[cfg(feature = "threads")]
    let fuse_channels = parallel;
    #[cfg(not(feature = "threads"))]
    let fuse_channels = false;
    let self_blur = fuse_channels && compute.self_blur_eligible();
    // Phase A is skipped WHOLE under self-blur bands, so its four fused-H
    // outputs are never written — `plane_needs` is where that is decided.
    // Bounded strip queue: independent v2 cells retain their original 128-row
    // kernels and strip partials. Basic bands and blockiness still merge in
    // producer order. No image-sized pyramid or thread-dependent reduction.
    #[cfg(feature = "threads")]
    let job_budget = REV5_JOB_BUDGET_BYTES;
    #[cfg(all(test, feature = "threads"))]
    let job_budget = scratch.rev5_job_budget.unwrap_or(job_budget);
    #[cfg(feature = "threads")]
    let batch_limit = if parallel
        && compute.formula_revision >= FormulaRevision::Rev5
        && compute.v2_blocks
        && rayon::current_num_threads() >= 2
        && h0 > STRIP_ROWS
    {
        debug_assert!(true, "queued merges cover only Rev5's supported families");
        // If this entire walk fits the existing bounded queue, dispatch it
        // together instead of creating a full batch followed by a tiny tail.
        // The byte budget and canonical strip/merge order are unchanged.
        let jobs: usize = dims
            .iter()
            .enumerate()
            .map(|(scale, &(_, h))| {
                if compute.at_scale(scale).v2_blocks {
                    let channels = (0..3)
                        .filter(|&ch| ALL_CHANNELS || compute.channel_active(scale, ch))
                        .count();
                    h.div_ceil(STRIP_ROWS) * channels
                } else {
                    0
                }
            })
            .sum();
        let capacity = rev5_job_limit(strip_max_n, 16, job_budget);
        if jobs >= REV5_MIN_JOB_SLOTS && jobs <= capacity {
            jobs
        } else {
            // One job per thread, at least four (measured on a 1507×2048
            // photo, B at revision 7: two threads 21.2 → 18.7 ms with batches
            // of four; at eight threads batches of sixteen were slower, 7.8 →
            // 10.1 ms, their scratch no longer fitting the cache). The
            // pipeline uses the same number as its in-flight limit; at eight
            // threads limits of four and six were slower than eight.
            rev5_job_limit(strip_max_n, rayon::current_num_threads().max(4), job_budget)
        }
    } else {
        0
    };
    #[cfg(feature = "threads")]
    rev5_prepare_jobs(&mut scratch.rev5_jobs, batch_limit, strip_max_n);
    let needs = compute.plane_needs(self_blur);
    #[cfg(feature = "threads")]
    let needs = if batch_limit > 0 {
        // Only basic-only scale cells can reach the old phase A. They need
        // horizontal planes; queued v2 cells own all their vertical planes.
        StripPlaneNeeds {
            h: needs.h && compute.v2_scales != ComputeSet::ALL_SCALES,
            v2: false,
        }
    } else {
        needs
    };
    scratch.ensure_for(strip_max_n, needs);
    let V2Scratch {
        strips: scratch_strips,
        stream_pool,
        #[cfg(feature = "threads")]
        rev5_jobs,
        #[cfg(feature = "threads")]
        block_pool,
        ..
    } = scratch;

    // Finite queries need the omitted raw scale-0 distorted channels too.
    // Keep them during the prepared walk instead of converting the image again.
    // This changes storage only; channel_active still controls every feature.
    let mut accums: [StreamChannelAccums; 3] = std::array::from_fn(|_| {
        StreamChannelAccums::at_revision(
            n_scales,
            band_slots_for(band_parallel),
            compute.formula_revision,
        )
    });
    // BANDVIS dst self-mask live: phase A additionally produces the Y
    // channel's dst-activity plane (and ONLY the Y channel's — the block
    // is Y-only, so X/B never pay the chain).
    let mut producer = StripPlaneProducer::new_with_ref_feed(
        source,
        distorted,
        parallel,
        stream_pool,
        front_end,
        ref_planes,
        compute.sampling,
        toggles.formula_revision,
        // `globaff` reads every channel's rows at every scale, so the
        // scale-zero chroma rows stay in the rolling planes.
        !compute.full_res_xb,
    );
    let mut globaff: Vec<[crate::globaff::GlobAffAccum; 3]> = { Vec::new() };

    #[cfg(feature = "threads")]
    let mut queued = 0;
    // Plain SDR walks whose every live scale runs the v2 blocks produce their
    // rows in parallel chunk tasks, and their strip jobs copy their own
    // windows (`rev5_block_walk`); the streaming producer above is unused.
    #[cfg(feature = "threads")]
    let block_walk = {
        let omit_xb = !compute.full_res_xb
            && !crate::feature_v2_stream::linear_pyramid();
        batch_limit > 0
            && ref_planes.is_none()
            && compute.sampling.is_none()
            && matches!(front_end, crate::feature_v2_stream::FrontEnd::Sdr)
            && !(ALL_CHANNELS && omit_xb)
            && (0..n_scales).all(|s| compute.scale_dead(s) || compute.at_scale(s).v2_blocks)
    };
    // Other plain walks (nothing retained, no cached reference) run their
    // strip jobs through the pipeline instead of batches: same merge order,
    // no barrier between batches, production overlapped with the jobs.
    #[cfg(feature = "threads")]
    let mut pipeline =
        (batch_limit > 0 && ref_planes.is_none() && !block_walk)
            .then(|| Rev5Pipeline::new(compute, fold_v1, batch_limit, core::mem::take(rev5_jobs)));

    #[cfg(feature = "threads")]
    if block_walk {
        rev5_block_walk::<S, D, ALL_CHANNELS>(
            source,
            distorted,
            &dims,
            &compute,
            toggles,
            fold_v1,
            batch_limit,
            !compute.full_res_xb
                && !crate::feature_v2_stream::linear_pyramid(),
            rev5_jobs,
            block_pool,
            &mut accums,
            &mut globaff,
            stop,
        );
    }
    loop {
        #[cfg(feature = "threads")]
        if block_walk {
            break;
        }
        if stop.is_some_and(|token| token.check().is_err()) {
            break;
        }
        let next = producer.next_strip();
        let Some(info) = next else {
            break;
        };
        let scale = info.scale;
        // `globaff`: every strip of every scale, all three channels, in
        // strip order — before the dead-scale skip, which knows nothing of it.
        let local = compute.at_scale(scale);
        let v2_blocks = local.v2_blocks;
        let toggles = V2NewFeatureToggles {
            v1_pools: local.v1_pools,
            v1_only: !v2_blocks,
            gradient_features: local.gradient,
            blockiness: local.blockiness,
            ..toggles
        };
        // A declared-ID plan that reads no v1-era slot and runs no v2-era work at this scale skips its strips whole
        // (the producer has already carried the pyramid through them). Only the pre-Rev5 scale-0 side channel keeps
        // them.
        if compute.scale_dead(scale)
            && !(scale == 0
                && compute.formula_revision < FormulaRevision::Rev5
                && mean_offset.is_some())
        {
            #[cfg(test)]
            SKIPPED_SCALES.with_borrow_mut(|v| *v |= 1 << scale);
            continue;
        }

        #[cfg(feature = "threads")]
        if let Some(pipe) = pipeline.as_mut() {
            // A basic-only cell keeps its existing path; finish every queued
            // cell first so per-channel/scale ordering is never disturbed.
            if !v2_blocks {
                pipe.drain(&mut accums);
            } else {
                for ch in 0..3 {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        continue;
                    }
                    let mut job = pipe.slot(info, ch, &mut accums);
                    job.gather(&producer, info, ch, false);
                    pipe.submit(job, toggles);
                }
                continue;
            }
        }
        #[cfg(feature = "threads")]
        if batch_limit > 0 && pipeline.is_none() {
            // A basic-only cell keeps its existing path; finish older queued
            // cells first so per-channel/scale ordering is never disturbed.
            if !v2_blocks && queued > 0 {
                rev5_run_jobs(
                    &mut rev5_jobs[..queued],
                    &compute,
                    toggles,
                    fold_v1,
                    &mut accums,
                    ref_planes.filter(|_| compute.sampling.is_none()),
                );
                queued = 0;
            }
            if v2_blocks {
                for ch in 0..3 {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        continue;
                    }
                    if queued == rev5_jobs.len() {
                        rev5_jobs.push(Rev5StripJob::new(info, ch));
                    }
                    let job = &mut rev5_jobs[queued];
                    job.gather(
                        &producer,
                        info,
                        ch,
                        compute.sampling.is_none()
                            && rev5_cached_window(ref_planes, &info, ch).is_some(),
                    );
                    queued += 1;
                    if queued == batch_limit {
                        rev5_run_jobs(
                            &mut rev5_jobs[..queued],
                            &compute,
                            toggles,
                            fold_v1,
                            &mut accums,
                            ref_planes.filter(|_| compute.sampling.is_none()),
                        );
                        queued = 0;
                    }
                }
                continue;
            }
        }

        // mean_offset side-channel (fold-engine lane): the scale-0 strips
        // tile [0, h0) exactly once in ascending order, so each row's
        // Σ_x (src − dst) is recorded exactly once here. Emitted for EVERY
        // strip layout — the walk's toggles never change which strips the
        // producer emits, only what is computed from them.
        if scale == 0
            && compute.formula_revision < FormulaRevision::Rev5
            && let Some(mo) = mean_offset.as_deref_mut()
        {
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
        }

        // FUSED PER-CHANNEL FAN-OUT (fold-MT lane). No channel reads another
        // channel's phase-A output, so each worker finishes its own phase B.
        // A split fan-out costs the walk's
        // locality contract: it holds THREE channels' ~9 MB phase-A buffers
        // live across a barrier, then re-reads them from wherever they landed.
        // The serial arm already fuses A->B per channel for exactly that
        // reason (measured +50 ms/pair on aic3-100 when interleaved); this is
        // the same shape under threads.
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
        // ONE owner: `ComputeSet::self_blur_eligible`, which is also what
        // sized the strip scratch above. Re-deriving it here is what let the
        // two drift apart on `Peaks` (see that function's doc); the local
        // `fuse_channels` conjunct stays because it is a runtime parallelism
        // fact the compute set does not model.
        #[cfg(feature = "threads")]
        let self_blur = fuse_channels && local.self_blur_eligible();

        #[cfg(feature = "threads")]
        if fuse_channels {
            use rayon::prelude::*;
            scratch_strips
                .par_iter_mut()
                .zip(accums.par_iter_mut())
                .enumerate()
                .for_each(|(ch, (scr, acc))| {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return;
                    }
                    if self_blur {
                        // Phase A is skipped whole; only the wide-window
                        // gather it also performs is still needed.
                        stream_gather_windows(&producer, &info, ch, scr);
                    } else {
                        stream_phase_a(
                            &producer,
                            &info,
                            ch,
                            false,
                            false,
                            v2_blocks,
                            true,
                            scr,
                            compute.formula_revision,
                        );
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
                        local.free_work(ch, scale),
                        acc,
                    );
                });
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
            scratch_strips
                .par_iter_mut()
                .enumerate()
                .for_each(|(ch, scr)| {
                    if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                        return;
                    }
                    stream_phase_a(
                        &producer,
                        &info,
                        ch,
                        false,
                        false,
                        v2_blocks,
                        true,
                        scr,
                        compute.formula_revision,
                    );
                });
            let scratches: &[ScratchV2Strip; 3] = scratch_strips;
            // Retention hooks (appendix N): serial copies between the two
            // fan-outs — pure reads of the phase-A scratches + windows.
            accums.par_iter_mut().enumerate().for_each(|(ch, acc)| {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    return;
                }
                let (src_win, dst_win) =
                    stream_windows_shared(&producer, &info, ch, &scratches[ch]);
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
                    local.free_work(ch, scale),
                    acc,
                );
            });
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
            for ch in [0usize, 2] {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    continue;
                }
                stream_phase_a(
                    &producer,
                    &info,
                    ch,
                    false,
                    false,
                    v2_blocks,
                    false,
                    scr,
                    compute.formula_revision,
                );
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
                    false,
                    local.free_work(ch, scale),
                    &mut accums[ch],
                );
            }
            {
                stream_phase_a(
                    &producer,
                    &info,
                    1,
                    false,
                    false,
                    v2_blocks,
                    false,
                    scr,
                    compute.formula_revision,
                );
                let (src_win, dst_win) = stream_windows_shared(&producer, &info, 1, scr);
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
                    local.free_work(1, scale),
                    &mut accums[1],
                );
            }
        }
    }

    #[cfg(feature = "threads")]
    if queued > 0 {
        rev5_run_jobs(
            &mut rev5_jobs[..queued],
            &compute,
            toggles,
            fold_v1,
            &mut accums,
            ref_planes.filter(|_| compute.sampling.is_none()),
        );
    }
    #[cfg(feature = "threads")]
    if let Some(mut pipe) = pipeline.take() {
        pipe.drain(&mut accums);
        *rev5_jobs = core::mem::take(&mut pipe.free);
    }

    producer.recycle(stream_pool);

    // --- Finalize: the materialized walk's per-scale epilogue, replayed
    //     on the merged accumulators (identical f64 ops on identical
    //     values ⇒ identical bits). ---
    let v1_total = if fold_v1 { n_scales * 3 * 31 } else { 0 };
    let v12_total = v1_total + n_scales * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
    let mut features = vec![0.0f64; v12_total];
    let features_v12 = &mut features[..];
    let mut prev_grad: [Option<(f64, f64)>; 3] = [None; 3];

    #[allow(clippy::needless_range_loop)] // scale derives 3+ offsets across distinct arrays
    for scale in 0..n_scales {
        let local = compute.at_scale(scale);
        let v2_blocks = local.v2_blocks;
        let toggles = V2NewFeatureToggles {
            v1_pools: local.v1_pools,
            v1_only: !v2_blocks,
            gradient_features: local.gradient,
            blockiness: local.blockiness,
            ..toggles
        };
        let (width, height) = dims[scale];
        let n = width * height;
        let scale_base = scale * 3 * FEATURES_PER_CHANNEL_V2_TOTAL;
        let app_scale_base = scale * 3 * FEATURES_PER_CHANNEL_APPEND;
        let mut grads: [(f64, f64); 3] = [(0.0, 0.0); 3];

        for (ch, acc) in accums.iter().enumerate() {
            if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                continue;
            }
            let out = &mut features_v12
                [v1_total + scale_base + ch * FEATURES_PER_CHANNEL_V2_TOTAL..]
                [..FEATURES_PER_CHANNEL_V2_TOTAL];
            let sum_blockiness = if toggles.blockiness {
                acc.block[scale].0.fin() + acc.block[scale].1.fin()
            } else {
                0.0
            };
            grads[ch] = finish_channel_scale_at_revision(
                &acc.dense[scale],
                &acc.grad[scale],
                sum_blockiness,
                n,
                out,
                compute.omit_pjnd_fragility,
                compute.formula_revision,
            );
            apply_transducer_luma_gate(out, ch, toggles);
        }

        // Retention hooks (appendix N): the EXACT pooled cells + the
        // mean-gradients this finalize just derived — the fused compare's
        // coefficient inputs (the standalone derives them from its own
        // 1e-9-parity pass-A replication instead).

        if fold_v1 {
            for (ch, acc) in accums.iter().enumerate() {
                if !ALL_CHANNELS && !compute.channel_active(scale, ch) {
                    continue;
                }
                let base = scale * 39 + ch * 13;
                acc.v1[scale].finalize_into(
                    n,
                    &mut features_v12[base..base + 13],
                    compute.formula_revision,
                );
                // FREE EXTRAS ([`V1FreeExtras::RawMoments`]): the v2-era
                // slots this walk can finalize from the fused kernel's raw
                // moments, written ONLY where the owning block is off — so
                // this can never race the append/append2 finalize below.
                // Index-stable: every other slot of those blocks keeps its
                // structural 0.0.
                // `!= Off`, not `== RawMoments`: every variant is a
                // SUPERSET of the raw moments, so an exact-equality test
                // here silently zeroes these 40 slots the moment a wider
                // variant is requested (caught by
                // `class_c_extras_are_pure_addition_to_the_free_walk`).
                if toggles.free_extras != V1FreeExtras::Off {
                    let (globals, luma_mean) =
                        acc.v1[scale].finalize_free(n, toggles.formula_revision);
                    // `GLOBAL_*` live in the append block. The 944 walk skips
                    // the (B, scale 0) append cell entirely
                    // (`APPEND_SKIP_B_SCALE0`), so this must skip it too or
                    // the emitted row would not be the 944 row.
                    // `LUMA_MEAN_REF` is append2's, Y-only.
                }
                // CLASS-C FREE EXTRAS: the bounded-error slots, written under
                // the same "only where the owning block is off" rule. The v2
                // finalize above already ran and left `MSE` at its structural
                // `clamp01(0/n) = 0.0` when `v2_blocks` is off, so this
                // overwrite is the value's first and only real write.
                if toggles.free_extras == V1FreeExtras::RawMomentsPlusBoundedErr {
                    let (mse, lum) = acc.v1[scale].finalize_free_bounded(n);
                    if !v2_blocks {
                        features_v12[v1_total
                            + scale_base
                            + ch * FEATURES_PER_CHANNEL_V2_TOTAL
                            + idx::MSE] = mse;
                    }
                    // The luminance bins are append's, and Y-only for the
                    // same register-carry reason the kernel is.
                }
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
                    acc.v1[scale].finalize_pools_into(
                        n,
                        &mut peaks,
                        &mut masked,
                        &mut iw,
                        compute.formula_revision,
                    );
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
            if let Some((prev_gsrc, prev_gdst)) = prev_grad[ch]
                && compute.at_scale(scale).gradient
                && compute.at_scale(scale - 1).gradient
            {
                let decay_src = gsrc / (prev_gsrc + C_GRAD_DECAY);
                let decay_dst = gdst / (prev_gdst + C_GRAD_DECAY);
                let prev_base = v1_total
                    + (scale - 1) * 3 * FEATURES_PER_CHANNEL_V2_TOTAL
                    + ch * FEATURES_PER_CHANNEL_V2_TOTAL;
                features_v12[prev_base + idx::EDGE_WIDTH_CHANGE] = bounded_dissim(
                    decay_src,
                    decay_dst,
                    C_EDGEWIDTH,
                    compute.formula_revision.stable_forms(),
                );
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

    // --- CSFW finalize (Y-only, per scale): the weighted GLOBAL_* twins
    //     from the Y CSFW accumulators (design §4.1). ---

    // --- REV4 finalize (per scale): C1 cells rescan their stored `ẽ`
    //     planes; C2/C3 fold their accumulators; C4's blur reads the
    //     already-emitted EWC_Y × HF_LOSS_Y — final only after the scale
    //     loop's cross-scale EWC fill, which is why this sits after it.

    ZensimV2Result { features }
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
//  - blockiness: lattice-step terms with one-sided shares per step end plus a per-step pair term
//    applied by rectangle queries at cut steps (exact for footprint-aligned rectangle repairs; the
//    sampling-geometry path keeps the symmetric 50/50 split).
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

/// Per-(scale, channel) pass-A pooled state.
///
/// Stays compiled in every configuration (it is a [`FoldRetention`] field
/// type, and the walk's retention hooks write it), but its fields are only
/// READ by the `custom-profiles`-gated attribution-density cluster — hence
/// the targeted dead-code allowance.
#[derive(Default, Clone, Copy)]
pub(crate) struct AttrCellSums {
    pub(crate) dense: DenseAccum,
    pub(crate) grad: GradientAccum,
    app: AppendAccum,
    pub(crate) blockiness: f64,
    pub(crate) n: usize,
}

/// Per-scale per-channel cached planes (scale-plane sized slices of
/// scale-0-sized buffers).
pub(crate) struct AttrChPlanes {
    pub(crate) mu1: Vec<f32>,
    pub(crate) mu2: Vec<f32>,
    pub(crate) ssq: Vec<f32>,
    pub(crate) s12: Vec<f32>,
    pub(crate) act: Vec<f32>,
    bs2: Vec<f32>,
}

impl AttrChPlanes {
}

// ============================================================================
// Tests
// ============================================================================

include!("feature_v2_tests.rs");

include!("feature_v2_oracle.rs");
