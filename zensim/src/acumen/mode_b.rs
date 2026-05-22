//! Mode B-lite preprocessor — per-pixel achromatic CSF weighting.
//!
//! The single primitive proven to help on Path B / Gate A: pre-multiply
//! the input RGB by a per-pixel scalar weight derived from each pixel's
//! local-adapted luminance and the reference's mean luminance, BEFORE
//! the pyramid is built. The downstream feature extractor sees
//! spatially-CSF-weighted content, which lets the trained MLP encode
//! local-adaptation-aware contrast.
//!
//! This is the *lite* approximation of the full per-band per-pixel
//! Mode B (one scalar weight per pixel, evaluated at a single
//! pyramid band's spatial frequency rather than per-band per-channel).
//! The full kernel-level Mode B lives in `zensim-gpu`'s pipeline.
//!
//! **Algorithm**:
//! 1. Per-pixel BT.709 linear luminance from sRGB-encoded RGB.
//! 2. 3-pass separable sliding-window box blur (O(W·H) per pass) at
//!    blur radius σ (≈ Gaussian σ) for local adaptation.
//! 3. Per-pixel CSF lookup at L_adapt(x,y) and `ρ = ppd / 2^(band+1)`.
//! 4. Normalize by CSF at image-mean L (output ≈ unit scale).
//! 5. Multiply input RGB by per-pixel scalar weight, clamp,
//!    8-bit truncate.
//!
//! Used as `extract_acumen_features --acumen-arch mode_b` /
//! `mode_b_per_band` and (when wired) the production extraction
//! pipeline.

use crate::acumen::castle_csf::{CastleCsfLut, Channel};
use crate::acumen::viewing::ViewingCondition;

/// Hyperparameters for [`apply_mode_b_premultiply`]. The defaults
/// match the sweep winner config (σ=8, band=3, clamp=[0.1, 4.0]).
#[derive(Clone, Copy, Debug)]
pub struct ModeBConfig {
    /// Local-adaptation blur radius in pixels (≈ Gaussian σ).
    pub blur_sigma: usize,
    /// Pyramid band index used for the CSF lookup frequency.
    /// `ρ = viewing.ppd / 2^(band_idx + 1)`. Defaults to 3 (lowest
    /// frequency band in a 4-scale pyramid). band=2 is also viable.
    pub band_idx: u32,
    /// Lower bound for the per-pixel weight (numerical safety).
    pub clamp_lo: f32,
    /// Upper bound for the per-pixel weight (numerical safety).
    pub clamp_hi: f32,
}

impl Default for ModeBConfig {
    fn default() -> Self {
        Self {
            blur_sigma: 8,
            band_idx: 3,
            clamp_lo: 0.1,
            clamp_hi: 4.0,
        }
    }
}

/// Apply Mode B-lite preprocessing to an sRGB-packed-u8 image.
///
/// Returns a new `Vec<u8>` of the same length as input (3*W*H bytes).
/// The pyramid + feature pipeline can then consume this output
/// in place of the raw input.
///
/// **Cost**: ~45 ms / 1 MP single-threaded on a modern x86. Per-pair
/// preprocessing applies to both ref + dist (≈ 90 ms/pair @ 1080p).
/// Cache the ref-side output across distortion variants of the same
/// reference to amortize.
pub fn apply_mode_b_premultiply(
    lut: &CastleCsfLut,
    viewing: ViewingCondition,
    rgb: &[u8],
    w: u32,
    h: u32,
    cfg: ModeBConfig,
) -> Vec<u8> {
    let n = (w as usize) * (h as usize);
    debug_assert_eq!(rgb.len(), n * 3);

    // Step 1: per-pixel linear luminance.
    let mut lum = vec![0.0_f32; n];
    for i in 0..n {
        let r = srgb_u8_to_linear(rgb[3 * i]);
        let g = srgb_u8_to_linear(rgb[3 * i + 1]);
        let b = srgb_u8_to_linear(rgb[3 * i + 2]);
        lum[i] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    }

    // Step 2: 3-pass box blur for local-adaptation envelope.
    let blurred = box_blur_3pass(&lum, w as usize, h as usize, cfg.blur_sigma);

    // Step 3+4: per-pixel CSF lookup + multiply, normalize by mean-L CSF.
    let peak_nits = viewing.peak_luminance_nits;
    let rho = viewing.ppd / (2u32.pow(cfg.band_idx + 1) as f32);
    let log_rho = rho.log10();
    let mean_l = (lum.iter().sum::<f32>() / n as f32) * peak_nits;
    let norm_l = viewing.adapted_luminance_nits(mean_l).max(1e-3);
    let csf_at_mean = lut.sensitivity(log_rho, norm_l.log10(), Channel::Achromatic);

    let mut out = Vec::with_capacity(n * 3);
    for i in 0..n {
        let l_nits = (blurred[i] * peak_nits).max(1e-3);
        let l_adapt = viewing.adapted_luminance_nits(l_nits).max(1e-3);
        let csf_here = lut.sensitivity(log_rho, l_adapt.log10(), Channel::Achromatic);
        let w_scalar = (csf_here / csf_at_mean).clamp(cfg.clamp_lo, cfg.clamp_hi);
        for ch in 0..3 {
            let v = rgb[3 * i + ch] as f32 * w_scalar;
            out.push(v.clamp(0.0, 255.0) as u8);
        }
    }
    out
}

#[inline]
fn srgb_u8_to_linear(v: u8) -> f32 {
    let u = v as f32 / 255.0;
    if u <= 0.040_45 {
        u / 12.92
    } else {
        ((u + 0.055) / 1.055).powf(2.4)
    }
}

/// 3-pass sliding-window box blur (≈ Gaussian σ). Separable, O(W·H)
/// per pass = O(6·W·H) total. Approximates a Gaussian of standard
/// deviation `r` for r ≥ ~4 pixels.
fn box_blur_3pass(input: &[f32], w: usize, h: usize, sigma: usize) -> Vec<f32> {
    let r = sigma;
    let mut buf = input.to_vec();
    let mut tmp = vec![0.0_f32; input.len()];
    for _ in 0..3 {
        box_blur_h(&buf, &mut tmp, w, h, r);
        std::mem::swap(&mut buf, &mut tmp);
        box_blur_v(&buf, &mut tmp, w, h, r);
        std::mem::swap(&mut buf, &mut tmp);
    }
    buf
}

fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    for y in 0..h {
        let row = y * w;
        let mut sum: f32 = 0.0;
        let init_hi = (r + 1).min(w);
        for x in 0..init_hi {
            sum += src[row + x];
        }
        let mut count = init_hi;
        for x in 0..w {
            dst[row + x] = sum / count as f32;
            let add_x = x + r + 1;
            if add_x < w {
                sum += src[row + add_x];
                count += 1;
            }
            if x >= r {
                sum -= src[row + x - r];
                count -= 1;
            }
        }
    }
}

fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    for x in 0..w {
        let mut sum: f32 = 0.0;
        let init_hi = (r + 1).min(h);
        for y in 0..init_hi {
            sum += src[y * w + x];
        }
        let mut count = init_hi;
        for y in 0..h {
            dst[y * w + x] = sum / count as f32;
            let add_y = y + r + 1;
            if add_y < h {
                sum += src[add_y * w + x];
                count += 1;
            }
            if y >= r {
                sum -= src[(y - r) * w + x];
                count -= 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identity check: a uniform-gray image should produce a weight
    /// of exactly 1.0 (since adapted L = mean L), so the output is
    /// bit-equivalent to the input (within u8 quantization).
    #[test]
    fn uniform_gray_is_identity() {
        let lut_bytes: &[u8] = include_bytes!("../../data/castle_csf_v0_5_4_cvvdp.lut");
        let lut = CastleCsfLut::from_bytes(lut_bytes).unwrap();
        let viewing = ViewingCondition::LAB_REFERENCE;
        let w = 32u32;
        let h = 32u32;
        let rgb = vec![128u8; (w * h * 3) as usize];
        let out = apply_mode_b_premultiply(&lut, viewing, &rgb, w, h, ModeBConfig::default());
        // Every pixel should be ~128 ± 1 (u8 quantization)
        for &v in &out {
            assert!(v >= 127 && v <= 129, "uniform-gray output should be ~128, got {v}");
        }
    }
}
