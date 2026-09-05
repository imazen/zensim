//! Color space conversion: linear RGB / sRGB → XYB
//!
//! XYB is the perceptual color space used by both ssimulacra2 and butteraugli.
//! X ≈ red-green opponent, Y ≈ luminance, B ≈ blue channel.
//! The cube root (LMS cone response) is the key nonlinearity.
//!
//! Two entry points for XYB conversion:
//! - [`srgb_to_positive_xyb_planar_into`]: sRGB u8 input (LUT-based linearization + SIMD)
//! - [`linear_to_positive_xyb_planar_into`]: linear f32 input (SIMD, skips LUT)
//!
//! RGBA compositing helpers produce linear f32 RGB output for all input formats,
//! ensuring identical XYB values regardless of input pixel format.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

// Opsin absorbance matrix (from jpegli/ssimulacra2).
// `pub(crate)` so sibling feature modules (e.g. `xyb_lms_features`)
// can lift them without duplicating constants — single source of truth.
pub(crate) const K_M02: f32 = 0.078;
pub(crate) const K_M00: f32 = 0.30;
pub(crate) const K_M01: f32 = 1.0 - K_M02 - K_M00;
pub(crate) const K_M12: f32 = 0.078;
pub(crate) const K_M10: f32 = 0.23;
pub(crate) const K_M11: f32 = 1.0 - K_M12 - K_M10;
pub(crate) const K_M20: f32 = 0.243_422_69;
pub(crate) const K_M21: f32 = 0.204_767_45;
pub(crate) const K_M22: f32 = 1.0 - K_M20 - K_M21;
pub(crate) const K_B0: f32 = 0.003_793_073_4;

// ─── Gamut conversion matrices (linear light, row-major) ─────────────────
//
// Convert from wide-gamut linear RGB to sRGB linear RGB.
// Computed as: sRGB_from_XYZ × XYZ_from_<source>.
// All share the D65 whitepoint, so no chromatic adaptation is needed.

/// Display P3 linear → sRGB linear (3×3, row-major).
///
/// P3 and sRGB share D65 whitepoint; P3 has wider red/green primaries.
/// Matrix = M_srgb_from_xyz × M_xyz_from_p3.
#[rustfmt::skip]
const P3_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.224_940_2, -0.224_940_2,  0.0        ],
    [-0.042_056_955, 1.042_056_9,  0.0        ],
    [-0.019_637_555, -0.078_636_04, 1.098_273_6],
];

/// BT.2020 linear → sRGB linear (3×3, row-major).
///
/// BT.2020 covers a much wider gamut than sRGB. Out-of-gamut colors
/// (negative sRGB values) are clamped to [0, 1].
/// Matrix = M_srgb_from_xyz × M_xyz_from_bt2020.
#[rustfmt::skip]
const BT2020_TO_SRGB: [[f32; 3]; 3] = [
    [ 1.660_491, -0.587_641_1, -0.072_849_9],
    [-0.124_550_5,  1.132_899_9, -0.008_349_4],
    [-0.018_151_0, -0.100_578_6,  1.118_729_6],
];

use crate::source::{ColorPrimaries, GamutMapping};

/// Apply a gamut conversion matrix to a linear RGB pixel.
///
/// Converts from the source color primaries to sRGB linear light.
/// For [`ColorPrimaries::Srgb`] this is a no-op. Out-of-gamut handling
/// follows `mapping` (issue #17): [`GamutMapping::Clip`] clamps to
/// \[0, 1\] (post-display-clamp semantics, the default);
/// [`GamutMapping::Preserve`] lets negative / >1 components flow into
/// XYB (the opsin stage's `max(0)` on the post-mix sum keeps the
/// cube-root domain valid), making codec gamut clipping detectable.
#[inline]
pub(crate) fn apply_gamut_matrix(
    rgb: &mut [f32; 3],
    primaries: ColorPrimaries,
    mapping: GamutMapping,
) {
    #[allow(unreachable_patterns)]
    let m = match primaries {
        ColorPrimaries::Srgb => return,
        ColorPrimaries::DisplayP3 => &P3_TO_SRGB,
        ColorPrimaries::Bt2020 => &BT2020_TO_SRGB,
        _ => return, // future variants: pass through unchanged
    };
    let [r, g, b] = *rgb;
    let out = [
        m[0][0] * r + m[0][1] * g + m[0][2] * b,
        m[1][0] * r + m[1][1] * g + m[1][2] * b,
        m[2][0] * r + m[2][1] * g + m[2][2] * b,
    ];
    #[allow(unreachable_patterns)]
    match mapping {
        GamutMapping::Clip => {
            rgb[0] = out[0].clamp(0.0, 1.0);
            rgb[1] = out[1].clamp(0.0, 1.0);
            rgb[2] = out[2].clamp(0.0, 1.0);
        }
        GamutMapping::Preserve => *rgb = out,
        // Future variants behave as the documented default (Clip).
        _ => {
            rgb[0] = out[0].clamp(0.0, 1.0);
            rgb[1] = out[1].clamp(0.0, 1.0);
            rgb[2] = out[2].clamp(0.0, 1.0);
        }
    }
}

/// Convert sRGB u8 to linear f32 via lookup table.
#[inline(always)]
pub(crate) fn srgb_u8_to_linear(v: u8) -> f32 {
    linear_srgb::default::srgb_u8_to_linear(v)
}

/// Convert sRGB u16 (0-65535) to linear f32.
#[inline]
pub(crate) fn srgb_u16_to_linear(v: u16) -> f32 {
    linear_srgb::default::srgb_u16_to_linear(v)
}

/// Fast cube root: bit manipulation + 2 Newton-Raphson iterations in f32.
/// Accurate to ~20 bits (sufficient for image quality metrics).
#[inline(always)]
pub(crate) fn cbrtf_fast(x: f32) -> f32 {
    // cbrt(0) = 0; the Halley iteration below would produce NaN for x=0
    // because t*numerator underflows below f32 min subnormal, yielding 0/0
    // in the second iteration.
    if x == 0.0 {
        return 0.0;
    }
    let mut t = cbrtf_initial(x);
    // Halley's method in f32 (each step roughly triples correct bits: 5→15→45)
    // Use mul_add to match SIMD Halley iterations: x.mul_add(2, r) and r.mul_add(2, x)
    // Division-first form t*(num/den) avoids underflow in t*num for tiny x.
    let mut r = t * t * t;
    t *= x.mul_add(2.0, r) / r.mul_add(2.0, x);
    r = t * t * t;
    t *= x.mul_add(2.0, r) / r.mul_add(2.0, x);
    t
}

/// Cube root initial estimate via bit manipulation (~5 bits accuracy).
/// Cheap integer-only operation; use as seed for Halley's refinement.
#[inline(always)]
fn cbrtf_initial(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    let ui_out = (ui & 0x8000_0000) | hx;
    f32::from_bits(ui_out)
}

/// Convert interleaved sRGB u8 to planar positive XYB, writing into pre-allocated buffers.
/// Each output slice must be at least `pixels.len()` long.
///
/// Used by the streaming pipeline in `streaming.rs` — avoids per-strip
/// allocations by reusing caller-owned XYB plane buffers.
pub fn srgb_to_positive_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        srgb_to_positive_xyb_planar_inner(pixels, x_out, y_out, b_out),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// Convert interleaved sRGB u8 to planar XYB (without positive shift).
#[allow(dead_code)]
pub fn srgb_to_xyb_planar(pixels: &[[u8; 3]]) -> [Vec<f32>; 3] {
    let n = pixels.len();
    let mut x_plane = vec![0.0f32; n];
    let mut y_plane = vec![0.0f32; n];
    let mut b_plane = vec![0.0f32; n];

    incant!(
        srgb_to_xyb_planar_inner(pixels, &mut x_plane, &mut y_plane, &mut b_plane),
        [v3, neon, wasm128, scalar]
    );

    [x_plane, y_plane, b_plane]
}

/// Convert interleaved sRGB u8 to planar XYB, writing into pre-allocated buffers.
#[allow(dead_code)]
pub fn srgb_to_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
) {
    incant!(
        srgb_to_xyb_planar_inner(pixels, x_plane, y_plane, b_plane),
        [v3, neon, wasm128, scalar]
    );
}

// Note: the standalone `make_positive_xyb` post-process function was
// removed in 0.3.0. The streaming pipeline now fuses the conversion +
// positive shift in `srgb_to_positive_xyb_planar_into` and
// `linear_to_positive_xyb_planar_into`, so the separate shift had no
// remaining callers.

// --- SIMD implementations ---

/// AVX-512 fused sRGB → XYB + make_positive: 16 pixels at a time.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v4(
    token: archmage::X64V4Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat(token, K_M00);
    let m01 = f32x16::splat(token, K_M01);
    let m02 = f32x16::splat(token, K_M02);
    let m10 = f32x16::splat(token, K_M10);
    let m11 = f32x16::splat(token, K_M11);
    let m12 = f32x16::splat(token, K_M12);
    let m20 = f32x16::splat(token, K_M20);
    let m21 = f32x16::splat(token, K_M21);
    let m22 = f32x16::splat(token, K_M22);
    let bias = f32x16::splat(token, K_B0);
    let zero = f32x16::zero(token);
    let ab = f32x16::splat(token, absorbance_bias);
    let half = f32x16::splat(token, 0.5);
    let fourteen = f32x16::splat(token, 14.0);
    let x_bias = f32x16::splat(token, 0.42);
    let y_bias = f32x16::splat(token, 0.01);
    let b_bias = f32x16::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 16 bounds checks per 16-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x16::from_array(token, r_arr);
        let g = f32x16::from_array(token, g_arr);
        let b = f32x16::from_array(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        // SIMD cube root (~15 bits, single Halley iteration; safe because
        // mixed >= K_B0 ≈ 0.0038 — no zeros, denormals, NaN, or infinities).
        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // Remainder with AVX2 (f32x8)
    let v3 = token.v3();
    let absorbance_bias_neg = absorbance_bias;
    let ab8 = f32x8::splat(v3, absorbance_bias);
    let half8 = f32x8::splat(v3, 0.5);
    let zero8 = f32x8::zero(v3);
    let m00_8 = f32x8::splat(v3, K_M00);
    let m01_8 = f32x8::splat(v3, K_M01);
    let m02_8 = f32x8::splat(v3, K_M02);
    let m10_8 = f32x8::splat(v3, K_M10);
    let m11_8 = f32x8::splat(v3, K_M11);
    let m12_8 = f32x8::splat(v3, K_M12);
    let m20_8 = f32x8::splat(v3, K_M20);
    let m21_8 = f32x8::splat(v3, K_M21);
    let m22_8 = f32x8::splat(v3, K_M22);
    let bias8 = f32x8::splat(v3, K_B0);
    let fourteen8 = f32x8::splat(v3, 14.0);
    let x_bias8 = f32x8::splat(v3, 0.42);
    let y_bias8 = f32x8::splat(v3, 0.01);
    let b_bias8 = f32x8::splat(v3, 0.55);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        let r = f32x8::from_array(v3, r_arr);
        let g = f32x8::from_array(v3, g_arr);
        let b = f32x8::from_array(v3, b_arr);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v4x(
    token: archmage::X64V4xToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat(token, K_M00);
    let m01 = f32x16::splat(token, K_M01);
    let m02 = f32x16::splat(token, K_M02);
    let m10 = f32x16::splat(token, K_M10);
    let m11 = f32x16::splat(token, K_M11);
    let m12 = f32x16::splat(token, K_M12);
    let m20 = f32x16::splat(token, K_M20);
    let m21 = f32x16::splat(token, K_M21);
    let m22 = f32x16::splat(token, K_M22);
    let bias = f32x16::splat(token, K_B0);
    let zero = f32x16::zero(token);
    let ab = f32x16::splat(token, absorbance_bias);
    let half = f32x16::splat(token, 0.5);
    let fourteen = f32x16::splat(token, 14.0);
    let x_bias = f32x16::splat(token, 0.42);
    let y_bias = f32x16::splat(token, 0.01);
    let b_bias = f32x16::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 16 bounds checks per 16-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x16::from_array(token, r_arr);
        let g = f32x16::from_array(token, g_arr);
        let b = f32x16::from_array(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        // SIMD cube root (~15 bits, single Halley iteration; safe because
        // mixed >= K_B0 ≈ 0.0038 — no zeros, denormals, NaN, or infinities).
        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // Remainder with AVX2 (f32x8)
    let v3 = token.v3();
    let absorbance_bias_neg = absorbance_bias;
    let ab8 = f32x8::splat(v3, absorbance_bias);
    let half8 = f32x8::splat(v3, 0.5);
    let zero8 = f32x8::zero(v3);
    let m00_8 = f32x8::splat(v3, K_M00);
    let m01_8 = f32x8::splat(v3, K_M01);
    let m02_8 = f32x8::splat(v3, K_M02);
    let m10_8 = f32x8::splat(v3, K_M10);
    let m11_8 = f32x8::splat(v3, K_M11);
    let m12_8 = f32x8::splat(v3, K_M12);
    let m20_8 = f32x8::splat(v3, K_M20);
    let m21_8 = f32x8::splat(v3, K_M21);
    let m22_8 = f32x8::splat(v3, K_M22);
    let bias8 = f32x8::splat(v3, K_B0);
    let fourteen8 = f32x8::splat(v3, 14.0);
    let x_bias8 = f32x8::splat(v3, 0.42);
    let y_bias8 = f32x8::splat(v3, 0.01);
    let b_bias8 = f32x8::splat(v3, 0.55);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }
        let r = f32x8::from_array(v3, r_arr);
        let g = f32x8::from_array(v3, g_arr);
        let b = f32x8::from_array(v3, b_arr);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Fused sRGB → XYB + make_positive in one pass with vectorized Halley iterations.
#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_positive_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);
    // Positive-shift constants
    let fourteen = f32x8::splat(token, 14.0);
    let x_bias = f32x8::splat(token, 0.42);
    let y_bias = f32x8::splat(token, 0.01);
    let b_bias = f32x8::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Fused make_positive: X*14+0.42, Y+0.01, (B-Y)+0.55
        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Generic fused sRGB → XYB + make_positive with vectorized Halley iterations.
#[magetypes(neon, wasm128, scalar)]
fn srgb_to_positive_xyb_planar_inner(
    token: Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);
    let fourteen = f32x8::splat(token, 14.0);
    let x_bias_v = f32x8::splat(token, 0.42);
    let y_bias_v = f32x8::splat(token, 0.01);
    let b_bias_v = f32x8::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Indexing the `pixels` SLICE inside the loop
        // left 8 bounds checks per 8-pixel iteration in the
        // emitted code. Same loads, same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Fused make_positive: X*14+0.42, Y+0.01, (B-Y)+0.55
        let x_pos = x.mul_add(fourteen, x_bias_v);
        let y_pos = y + y_bias_v;
        let b_pos = (t2 - y) + b_bias_v;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn srgb_to_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 pixels, linearize via LUT, transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        // Opsin absorbance matrix multiply with FMA
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let c0 = mixed0.cbrt_midp() + ab;
        let c1 = mixed1.cbrt_midp() + ab;
        let c2 = mixed2.cbrt_midp();

        // XYB transform: X = 0.5*(c0-c1), Y = 0.5*(c0+c1), B = c2
        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Store directly to planar output
        let x_arr = x.to_array();
        let y_arr = y.to_array();
        let b_arr_out = c2.to_array();
        x_out[base..base + 8].copy_from_slice(&x_arr);
        y_out[base..base + 8].copy_from_slice(&y_arr);
        b_out[base..base + 8].copy_from_slice(&b_arr_out);
    }

    // Scalar remainder
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mut mixed0 = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)));
        let mut mixed1 = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)));
        let mut mixed2 = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)));

        mixed0 = mixed0.max(0.0);
        mixed1 = mixed1.max(0.0);
        mixed2 = mixed2.max(0.0);

        let bias_neg = -cbrtf_fast(K_B0);
        mixed0 = cbrtf_fast(mixed0) + bias_neg;
        mixed1 = cbrtf_fast(mixed1) + bias_neg;
        mixed2 = cbrtf_fast(mixed2);

        x_out[i] = 0.5 * (mixed0 - mixed1);
        y_out[i] = 0.5 * (mixed0 + mixed1);
        b_out[i] = mixed2;
    }
}

/// Generic sRGB → XYB (without positive shift) with scalar cube root.
#[magetypes(neon, wasm128, scalar)]
fn srgb_to_xyb_planar_inner(
    token: Token,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        // Load 8 pixels, linearize via LUT, transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[u8; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = srgb_u8_to_linear(p[0]);
            g_arr[i] = srgb_u8_to_linear(p[1]);
            b_arr[i] = srgb_u8_to_linear(p[2]);
        }

        let r = f32x8::from_array(token, r_arr);
        let g = f32x8::from_array(token, g_arr);
        let b = f32x8::from_array(token, b_arr);

        // Opsin absorbance matrix multiply with FMA
        let mixed0 = m00.mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)));
        let mixed1 = m10.mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)));
        let mixed2 = m20.mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)));

        let mixed0 = mixed0.max(zero);
        let mixed1 = mixed1.max(zero);
        let mixed2 = mixed2.max(zero);

        let c0 = mixed0.cbrt_midp() + ab;
        let c1 = mixed1.cbrt_midp() + ab;
        let c2 = mixed2.cbrt_midp();

        // XYB transform: X = 0.5*(c0-c1), Y = 0.5*(c0+c1), B = c2
        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        // Store directly to planar output
        x_out[base..base + 8].copy_from_slice(&x.to_array());
        y_out[base..base + 8].copy_from_slice(&y.to_array());
        b_out[base..base + 8].copy_from_slice(&c2.to_array());
    }

    // Scalar remainder
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = srgb_u8_to_linear(p[0]);
        let g = srgb_u8_to_linear(p[1]);
        let b = srgb_u8_to_linear(p[2]);

        let mut mixed0 = K_M00.mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)));
        let mut mixed1 = K_M10.mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)));
        let mut mixed2 = K_M20.mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)));

        mixed0 = mixed0.max(0.0);
        mixed1 = mixed1.max(0.0);
        mixed2 = mixed2.max(0.0);

        mixed0 = cbrtf_fast(mixed0) + absorbance_bias;
        mixed1 = cbrtf_fast(mixed1) + absorbance_bias;
        mixed2 = cbrtf_fast(mixed2);

        x_out[i] = 0.5 * (mixed0 - mixed1);
        y_out[i] = 0.5 * (mixed0 + mixed1);
        b_out[i] = mixed2;
    }
}

// ---------------------------------------------------------------------------
// Linear RGB → positive XYB conversion
// ---------------------------------------------------------------------------

/// Convert interleaved linear f32 RGB to planar positive XYB.
///
/// Input: `&[[f32; 3]]` — linear-light RGB values (typically in [0.0, 1.0]).
/// Output: 3 planes (X, Y, B) each of length `pixels.len()`, already positive-shifted.
///
/// This is the same opsin matrix + cube root + positive shift as the sRGB u8 path,
/// but skips the sRGB LUT linearization step. Results are identical for the same
/// linear RGB values (within floating-point precision).
pub fn linear_to_positive_xyb_planar_into(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        linear_to_positive_xyb_planar_inner(pixels, x_out, y_out, b_out),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// AVX-512 path for linear f32 → positive XYB (16 pixels at a time).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v4(
    token: archmage::X64V4Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat(token, K_M00);
    let m01 = f32x16::splat(token, K_M01);
    let m02 = f32x16::splat(token, K_M02);
    let m10 = f32x16::splat(token, K_M10);
    let m11 = f32x16::splat(token, K_M11);
    let m12 = f32x16::splat(token, K_M12);
    let m20 = f32x16::splat(token, K_M20);
    let m21 = f32x16::splat(token, K_M21);
    let m22 = f32x16::splat(token, K_M22);
    let bias = f32x16::splat(token, K_B0);
    let zero = f32x16::zero(token);
    let ab = f32x16::splat(token, absorbance_bias);
    let half = f32x16::splat(token, 0.5);
    let fourteen = f32x16::splat(token, 14.0);
    let x_bias = f32x16::splat(token, 0.42);
    let y_bias = f32x16::splat(token, 0.01);
    let b_bias = f32x16::splat(token, 0.55);

    let one = f32x16::splat(token, 1.0);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut [0, 1]
        let r = f32x16::from_array(token, r_arr).max(zero).min(one);
        let g = f32x16::from_array(token, g_arr).max(zero).min(one);
        let b = f32x16::from_array(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // AVX2 remainder
    let v3 = token.v3();
    let ab8 = f32x8::splat(v3, absorbance_bias);
    let half8 = f32x8::splat(v3, 0.5);
    let zero8 = f32x8::zero(v3);
    let m00_8 = f32x8::splat(v3, K_M00);
    let m01_8 = f32x8::splat(v3, K_M01);
    let m02_8 = f32x8::splat(v3, K_M02);
    let m10_8 = f32x8::splat(v3, K_M10);
    let m11_8 = f32x8::splat(v3, K_M11);
    let m12_8 = f32x8::splat(v3, K_M12);
    let m20_8 = f32x8::splat(v3, K_M20);
    let m21_8 = f32x8::splat(v3, K_M21);
    let m22_8 = f32x8::splat(v3, K_M22);
    let bias8 = f32x8::splat(v3, K_B0);
    let fourteen8 = f32x8::splat(v3, 14.0);
    let x_bias8 = f32x8::splat(v3, 0.42);
    let y_bias8 = f32x8::splat(v3, 0.01);
    let b_bias8 = f32x8::splat(v3, 0.55);

    let one8 = f32x8::splat(v3, 1.0);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        // Clamp to display gamut [0, 1]
        let r = f32x8::from_array(v3, r_arr).max(zero8).min(one8);
        let g = f32x8::from_array(v3, g_arr).max(zero8).min(one8);
        let b = f32x8::from_array(v3, b_arr).max(zero8).min(one8);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v4x(
    token: archmage::X64V4xToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x16::splat(token, K_M00);
    let m01 = f32x16::splat(token, K_M01);
    let m02 = f32x16::splat(token, K_M02);
    let m10 = f32x16::splat(token, K_M10);
    let m11 = f32x16::splat(token, K_M11);
    let m12 = f32x16::splat(token, K_M12);
    let m20 = f32x16::splat(token, K_M20);
    let m21 = f32x16::splat(token, K_M21);
    let m22 = f32x16::splat(token, K_M22);
    let bias = f32x16::splat(token, K_B0);
    let zero = f32x16::zero(token);
    let ab = f32x16::splat(token, absorbance_bias);
    let half = f32x16::splat(token, 0.5);
    let fourteen = f32x16::splat(token, 14.0);
    let x_bias = f32x16::splat(token, 0.42);
    let y_bias = f32x16::splat(token, 0.01);
    let b_bias = f32x16::splat(token, 0.55);

    let one = f32x16::splat(token, 1.0);

    let n = pixels.len();
    let chunks = n / 16;

    for chunk in 0..chunks {
        let base = chunk * 16;

        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 16] = pixels[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut [0, 1]
        let r = f32x16::from_array(token, r_arr).max(zero).min(one);
        let g = f32x16::from_array(token, g_arr).max(zero).min(one);
        let b = f32x16::from_array(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 16].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 16].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 16].copy_from_slice(&b_pos.to_array());
    }

    // AVX2 remainder
    let v3 = token.v3();
    let ab8 = f32x8::splat(v3, absorbance_bias);
    let half8 = f32x8::splat(v3, 0.5);
    let zero8 = f32x8::zero(v3);
    let m00_8 = f32x8::splat(v3, K_M00);
    let m01_8 = f32x8::splat(v3, K_M01);
    let m02_8 = f32x8::splat(v3, K_M02);
    let m10_8 = f32x8::splat(v3, K_M10);
    let m11_8 = f32x8::splat(v3, K_M11);
    let m12_8 = f32x8::splat(v3, K_M12);
    let m20_8 = f32x8::splat(v3, K_M20);
    let m21_8 = f32x8::splat(v3, K_M21);
    let m22_8 = f32x8::splat(v3, K_M22);
    let bias8 = f32x8::splat(v3, K_B0);
    let fourteen8 = f32x8::splat(v3, 14.0);
    let x_bias8 = f32x8::splat(v3, 0.42);
    let y_bias8 = f32x8::splat(v3, 0.01);
    let b_bias8 = f32x8::splat(v3, 0.55);

    let one8 = f32x8::splat(v3, 1.0);

    let rem_start = chunks * 16;
    let rem_chunks = (n - rem_start) / 8;
    for chunk in 0..rem_chunks {
        let base = rem_start + chunk * 8;
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        // Clamp to display gamut [0, 1]
        let r = f32x8::from_array(v3, r_arr).max(zero8).min(one8);
        let g = f32x8::from_array(v3, g_arr).max(zero8).min(one8);
        let b = f32x8::from_array(v3, b_arr).max(zero8).min(one8);

        let mixed0 = m00_8
            .mul_add(r, m01_8.mul_add(g, m02_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed1 = m10_8
            .mul_add(r, m11_8.mul_add(g, m12_8.mul_add(b, bias8)))
            .max(zero8);
        let mixed2 = m20_8
            .mul_add(r, m21_8.mul_add(g, m22_8.mul_add(b, bias8)))
            .max(zero8);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab8;
        let c1 = t1 + ab8;
        let x = half8 * (c0 - c1);
        let y = half8 * (c0 + c1);
        let x_pos = x.mul_add(fourteen8, x_bias8);
        let y_pos = y + y_bias8;
        let b_pos = (t2 - y) + b_bias8;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (rem_start + rem_chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// AVX2 path for linear f32 → positive XYB (8 pixels at a time).
#[cfg(target_arch = "x86_64")]
#[arcane]
fn linear_to_positive_xyb_planar_inner_v3(
    token: archmage::X64V3Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);
    let fourteen = f32x8::splat(token, 14.0);
    let x_bias = f32x8::splat(token, 0.42);
    let y_bias = f32x8::splat(token, 0.01);
    let b_bias = f32x8::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut: out-of-range values from lossy reconstruction
        // aren't visible on a real display, so measuring them would overcount error.
        let r = f32x8::from_array(token, r_arr).max(zero).min(one);
        let g = f32x8::from_array(token, g_arr).max(zero).min(one);
        let b = f32x8::from_array(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias);
        let y_pos = y + y_bias;
        let b_pos = (t2 - y) + b_bias;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    let absorbance_bias_neg = absorbance_bias;
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias_neg;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias_neg;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Generic linear f32 to positive XYB with gamut clamping and Halley iterations.
#[magetypes(neon, wasm128, scalar)]
fn linear_to_positive_xyb_planar_inner(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;

    let absorbance_bias = -cbrtf_fast(K_B0);

    let m00 = f32x8::splat(token, K_M00);
    let m01 = f32x8::splat(token, K_M01);
    let m02 = f32x8::splat(token, K_M02);
    let m10 = f32x8::splat(token, K_M10);
    let m11 = f32x8::splat(token, K_M11);
    let m12 = f32x8::splat(token, K_M12);
    let m20 = f32x8::splat(token, K_M20);
    let m21 = f32x8::splat(token, K_M21);
    let m22 = f32x8::splat(token, K_M22);
    let bias = f32x8::splat(token, K_B0);
    let zero = f32x8::zero(token);
    let one = f32x8::splat(token, 1.0);
    let ab = f32x8::splat(token, absorbance_bias);
    let half = f32x8::splat(token, 0.5);
    let fourteen = f32x8::splat(token, 14.0);
    let x_bias_v = f32x8::splat(token, 0.42);
    let y_bias_v = f32x8::splat(token, 0.01);
    let b_bias_v = f32x8::splat(token, 0.55);

    let n = pixels.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];
        // FIXED-SIZE ARRAY PATTERN (CLAUDE.md "Performance
        // Optimization"): ONE range check at the boundary, zero
        // interior. Same loads in the same order — bit-exact.
        let px: &[[f32; 3]; 8] = pixels[base..base + 8]
            .try_into()
            .expect("8 pixels per chunk");
        for i in 0..8 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        // Clamp to display gamut: out-of-range values from lossy reconstruction
        // aren't visible on a real display, so measuring them would overcount error.
        let r = f32x8::from_array(token, r_arr).max(zero).min(one);
        let g = f32x8::from_array(token, g_arr).max(zero).min(one);
        let b = f32x8::from_array(token, b_arr).max(zero).min(one);

        let mixed0 = m00
            .mul_add(r, m01.mul_add(g, m02.mul_add(b, bias)))
            .max(zero);
        let mixed1 = m10
            .mul_add(r, m11.mul_add(g, m12.mul_add(b, bias)))
            .max(zero);
        let mixed2 = m20
            .mul_add(r, m21.mul_add(g, m22.mul_add(b, bias)))
            .max(zero);

        let t0 = mixed0.cbrt_midp();
        let t1 = mixed1.cbrt_midp();
        let t2 = mixed2.cbrt_midp();

        let c0 = t0 + ab;
        let c1 = t1 + ab;

        let x = half * (c0 - c1);
        let y = half * (c0 + c1);

        let x_pos = x.mul_add(fourteen, x_bias_v);
        let y_pos = y + y_bias_v;
        let b_pos = (t2 - y) + b_bias_v;

        x_out[base..base + 8].copy_from_slice(&x_pos.to_array());
        y_out[base..base + 8].copy_from_slice(&y_pos.to_array());
        b_out[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    // Scalar remainder
    for i in (chunks * 8)..n {
        let p = pixels[i];
        let r = p[0].clamp(0.0, 1.0);
        let g = p[1].clamp(0.0, 1.0);
        let b = p[2].clamp(0.0, 1.0);

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// Unclamped scalar sibling of [`linear_to_positive_xyb_planar_into`],
/// used ONLY by the [`GamutMapping::Preserve`](crate::source::GamutMapping)
/// conversion path (issue #17).
///
/// Input components may be negative or > 1 (out-of-sRGB-gamut linear light
/// after a wide-gamut primaries matrix) — the clamped kernels would clip
/// them back to the gamut boundary, re-masking exactly the difference
/// `Preserve` exists to expose. The opsin mix's `max(0)` keeps the
/// cube-root domain valid for arbitrary finite input; inputs must be
/// finite (u8-linearized wide-gamut rows always are; `LinearF32Rgba`
/// callers own their float hygiene in this opt-in mode).
///
/// Per-pixel math mirrors the clamped kernels' scalar remainder exactly
/// (same `mul_add` chains, same [`cbrtf_fast`], same biases), so for
/// in-`[0,1]` inputs the outputs are bit-identical to the scalar path —
/// locked by `unclamped_matches_clamped_scalar_for_in_gamut` below.
pub(crate) fn linear_to_positive_xyb_planar_into_unclamped(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);
    for (i, p) in pixels.iter().enumerate() {
        let [r, g, b] = *p;

        let mixed0 = K_M00
            .mul_add(r, K_M01.mul_add(g, K_M02.mul_add(b, K_B0)))
            .max(0.0);
        let mixed1 = K_M10
            .mul_add(r, K_M11.mul_add(g, K_M12.mul_add(b, K_B0)))
            .max(0.0);
        let mixed2 = K_M20
            .mul_add(r, K_M21.mul_add(g, K_M22.mul_add(b, K_B0)))
            .max(0.0);

        let c0 = cbrtf_fast(mixed0) + absorbance_bias;
        let c1 = cbrtf_fast(mixed1) + absorbance_bias;
        let c2 = cbrtf_fast(mixed2);

        let x = 0.5 * (c0 - c1);
        let y = 0.5 * (c0 + c1);

        x_out[i] = x.mul_add(14.0, 0.42);
        y_out[i] = y + 0.01;
        b_out[i] = (c2 - y) + 0.55;
    }
}

/// HDR PU-XYB conversion (scalar): **absolute-luminance** linear RGB (cd/m²,
/// NOT clamped to `[0,1]`) → PU-encoded XYB-like planes.
///
/// The HDR-path analog of [`linear_to_positive_xyb_planar_into`]: identical
/// opsin mix + opponent structure, but each opsin-mixed channel goes through
/// **PU21** (normalized so 100 cd/m² → ~1.0, matching the cube-root white
/// point) instead of the cube root. The cube-root-domain absorbance centering
/// (`-cbrt(K_B0)`) is dropped — in PU space `K_B0` clamps below `PU21_L_MIN`,
/// so its contribution is ~0. See `docs/HDR_PLAN.md` §2b for the design
/// rationale and the validation gate. Scalar only (HDR is a minority path;
/// SIMD is a later optimization).
/// PU21(100 cd/m²) ≈ 256.3 — normalizes a 100-nit reference white to ~1.0,
/// the range the cube-root XYB white point sits in, so the downstream
/// opponent biases + feature kernels stay in calibration.
const PU_WHITE: f32 = 256.3;
/// Opponent X amplification in PU space. The cube-root path uses 14× to make
/// the tiny red-green cube-root difference visible; PU-space opsin
/// differences are already large, and HDR validation favors
/// luminance-dominant weighting (see docs/HDR_PLAN.md §2b).
const PU_X_SCALE: f32 = 4.0;

/// Scalar per-pixel PU-XYB (also the SIMD kernels' tail path and the
/// parity-test reference).
#[inline]
fn pu_xyb_pixel(p: [f32; 3]) -> (f32, f32, f32) {
    use crate::pu21::pu21_encode;
    let mixed0 = K_M00
        .mul_add(p[0], K_M01.mul_add(p[1], K_M02.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed1 = K_M10
        .mul_add(p[0], K_M11.mul_add(p[1], K_M12.mul_add(p[2], K_B0)))
        .max(0.0);
    let mixed2 = K_M20
        .mul_add(p[0], K_M21.mul_add(p[1], K_M22.mul_add(p[2], K_B0)))
        .max(0.0);
    let c0 = pu21_encode(mixed0) / PU_WHITE;
    let c1 = pu21_encode(mixed1) / PU_WHITE;
    let c2 = pu21_encode(mixed2) / PU_WHITE;
    let x = 0.5 * (c0 - c1);
    let y = 0.5 * (c0 + c1);
    (x.mul_add(PU_X_SCALE, 0.42), y + 0.01, (c2 - y) + 0.55)
}

/// Absolute-luminance linear RGB → positive PU-XYB planes, SIMD-dispatched
/// (8 px/iter via the generic magetypes tiers; `x^p = exp2_midp_precise(p·log2_midp_precise(x))`
/// — the midp_precise transcendentals hold the scalar↔SIMD divergence to
/// ≤ 2e-3 per channel; `simd_matches_scalar_within_band` pins it).
pub(crate) fn linear_to_pu_xyb_planar_into(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        pu_xyb_rows_inner(pixels, x_out, y_out, b_out),
        [v3, neon, wasm128, scalar]
    );
}

#[magetypes(v3, neon, wasm128, scalar)]
fn pu_xyb_rows_inner(
    token: Token,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    // PU21 banding_glare parameters — single source: `pu21::P`.
    const P0: f32 = crate::pu21::P[0];
    const P1: f32 = crate::pu21::P[1];
    const P2: f32 = crate::pu21::P[2];
    const P3: f32 = crate::pu21::P[3];
    const P4: f32 = crate::pu21::P[4];
    const P5: f32 = crate::pu21::P[5];
    const P6: f32 = crate::pu21::P[6];
    let l_min = f32x8::splat(token, crate::pu21::PU21_L_MIN);
    let l_max = f32x8::splat(token, crate::pu21::PU21_L_MAX);
    let zero = f32x8::splat(token, 0.0);
    let one = f32x8::splat(token, 1.0);
    let p0 = f32x8::splat(token, P0);
    let p1 = f32x8::splat(token, P1);
    let p2 = f32x8::splat(token, P2);
    let p3 = f32x8::splat(token, P3);
    let p4 = f32x8::splat(token, P4);
    let p5 = f32x8::splat(token, P5);
    let p6 = f32x8::splat(token, P6);
    let inv_white = f32x8::splat(token, 1.0 / PU_WHITE);
    let pu = |v: f32x8| -> f32x8 {
        let y = v.max(l_min).min(l_max);
        let yp = (p3 * y.log2_midp_precise()).exp2_midp_precise();
        let inner = (p0 + p1 * yp) / (one + p2 * yp);
        (p6 * ((p4 * inner.log2_midp_precise()).exp2_midp_precise() - p5)).max(zero)
    };

    let n = pixels.len();
    let chunks = n / 8;
    let mut r = [0.0f32; 8];
    let mut g = [0.0f32; 8];
    let mut b = [0.0f32; 8];
    for c in 0..chunks {
        let base = c * 8;
        for (j, px) in pixels[base..base + 8].iter().enumerate() {
            r[j] = px[0];
            g[j] = px[1];
            b[j] = px[2];
        }
        let vr = f32x8::from_array(token, r);
        let vg = f32x8::from_array(token, g);
        let vb = f32x8::from_array(token, b);
        let kb0 = f32x8::splat(token, K_B0);
        let m0 = (f32x8::splat(token, K_M00) * vr
            + f32x8::splat(token, K_M01) * vg
            + f32x8::splat(token, K_M02) * vb
            + kb0)
            .max(zero);
        let m1 = (f32x8::splat(token, K_M10) * vr
            + f32x8::splat(token, K_M11) * vg
            + f32x8::splat(token, K_M12) * vb
            + kb0)
            .max(zero);
        let m2 = (f32x8::splat(token, K_M20) * vr
            + f32x8::splat(token, K_M21) * vg
            + f32x8::splat(token, K_M22) * vb
            + kb0)
            .max(zero);
        let c0 = pu(m0) * inv_white;
        let c1 = pu(m1) * inv_white;
        let c2 = pu(m2) * inv_white;
        let half = f32x8::splat(token, 0.5);
        let x = half * (c0 - c1);
        let y = half * (c0 + c1);
        (x * f32x8::splat(token, PU_X_SCALE) + f32x8::splat(token, 0.42))
            .store((&mut x_out[base..base + 8]).try_into().unwrap());
        (y + f32x8::splat(token, 0.01)).store((&mut y_out[base..base + 8]).try_into().unwrap());
        ((c2 - y) + f32x8::splat(token, 0.55))
            .store((&mut b_out[base..base + 8]).try_into().unwrap());
    }
    for i in (chunks * 8)..n {
        let (x, y, bb) = pu_xyb_pixel(pixels[i]);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = bb;
    }
}

// ---------------------------------------------------------------------------
// RGBA/BGRA compositing helpers — all produce linear f32 RGB output
// ---------------------------------------------------------------------------

/// Precomputed deterministic noise table in linear light. 4096 values in
/// \[0.2, 0.8\], indexed by `(x ^ hash(y)) % 4096`.
///
/// Generated from integer-only hashing (wrapping multiply + xorshift) for
/// bit-identical results across all platforms. The final normalization
/// (u16 → f32 multiply + add) is exact in f32.
///
/// 4096 entries (16 KiB) fits in L1 cache and produces visually uniform
/// noise with no banding or diagonal artifacts.
const ALPHA_BG_TABLE: [f32; 4096] = {
    let mut table = [0.0f32; 4096];
    let mut i = 0u32;
    while i < 4096 {
        let mut h = i.wrapping_mul(2654435761);
        h ^= h >> 16;
        h = h.wrapping_mul(0x45d9f3b);
        h ^= h >> 16;
        table[i as usize] = 0.2 + (h & 0xFFFF) as f32 * (0.6 / 65535.0);
        i += 1;
    }
    table
};

/// Deterministic noise background value in linear light for the given pixel.
///
/// XORs the x coordinate with a hashed y to break row correlation, then
/// indexes into [`ALPHA_BG_TABLE`]. Cost: one wrapping multiply, one XOR,
/// one table lookup (always in L1).
#[inline(always)]
fn alpha_background_linear(x: usize, y: usize) -> f32 {
    let yh = (y as u32).wrapping_mul(2654435761);
    ALPHA_BG_TABLE[((x as u32) ^ yh) as usize & 0xFFF]
}

/// Composite sRGB u8 RGBA over a deterministic noise background, producing
/// linear f32 RGB.
///
/// Linearizes both foreground and background, then alpha-blends in linear space.
/// The noise background avoids the structured-pattern amplification that a
/// checkerboard causes in the multi-scale SSIM metric.
///
/// Uses straight alpha: `out = src * a + bg * (1-a)`.
pub(crate) fn composite_srgb8_rgba_to_linear(row: &[[u8; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[r, g, b, a]) in row.iter().enumerate() {
        if a == 255 {
            out[x] = [
                srgb_u8_to_linear(r),
                srgb_u8_to_linear(g),
                srgb_u8_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u8_to_linear(r);
            let gl = srgb_u8_to_linear(g);
            let bl = srgb_u8_to_linear(b);
            out[x] = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

/// Composite sRGB u8 BGRA over a deterministic noise background, producing
/// linear f32 RGB.
///
/// Swizzles B↔R during linearization. Alpha blending in linear space.
pub(crate) fn composite_srgb8_bgra_to_linear(row: &[[u8; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[b, g, r, a]) in row.iter().enumerate() {
        if a == 255 {
            out[x] = [
                srgb_u8_to_linear(r),
                srgb_u8_to_linear(g),
                srgb_u8_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u8_to_linear(r);
            let gl = srgb_u8_to_linear(g);
            let bl = srgb_u8_to_linear(b);
            out[x] = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

/// Composite linear f32 RGBA over a noise background, producing linear f32 RGB.
pub(crate) fn composite_linear_f32_rgba(row: &[[f32; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[r, g, b, a]) in row.iter().enumerate() {
        if a >= 1.0 {
            out[x] = [r, g, b];
        } else if a <= 0.0 {
            let bg = alpha_background_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let inv = 1.0 - a;
            let bg = alpha_background_linear(x, y);
            out[x] = [
                r.mul_add(a, bg * inv),
                g.mul_add(a, bg * inv),
                b.mul_add(a, bg * inv),
            ];
        }
    }
}

/// Composite sRGB u16 RGBA over a noise background, producing linear f32 RGB.
///
/// u16 values 0-65535 are linearized via `srgb_u16_to_linear()`, then alpha-blended
/// in linear space against the noise background.
pub(crate) fn composite_srgb16_rgba_to_linear(
    row: &[u8],
    width: usize,
    y: usize,
    out: &mut [[f32; 3]],
) {
    for (x, out_pixel) in out.iter_mut().enumerate().take(width) {
        let off = x * 8; // 4 channels × 2 bytes
        let r = u16::from_ne_bytes([row[off], row[off + 1]]);
        let g = u16::from_ne_bytes([row[off + 2], row[off + 3]]);
        let b = u16::from_ne_bytes([row[off + 4], row[off + 5]]);
        let a = u16::from_ne_bytes([row[off + 6], row[off + 7]]);
        if a == 65535 {
            *out_pixel = [
                srgb_u16_to_linear(r),
                srgb_u16_to_linear(g),
                srgb_u16_to_linear(b),
            ];
        } else if a == 0 {
            let bg = alpha_background_linear(x, y);
            *out_pixel = [bg, bg, bg];
        } else {
            let alpha = a as f32 / 65535.0;
            let inv = 1.0 - alpha;
            let bg = alpha_background_linear(x, y);
            let rl = srgb_u16_to_linear(r);
            let gl = srgb_u16_to_linear(g);
            let bl = srgb_u16_to_linear(b);
            *out_pixel = [
                rl.mul_add(alpha, bg * inv),
                gl.mul_add(alpha, bg * inv),
                bl.mul_add(alpha, bg * inv),
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ColorPrimaries;

    /// Gated-mirror parity (issue #17): the unclamped scalar converter
    /// must be BIT-identical to the clamped entry's scalar remainder for
    /// in-gamut input. n = 7 (< the 8-wide SIMD chunk) forces the clamped
    /// entry onto its scalar remainder, where the clamp is an arithmetic
    /// no-op on in-[0,1] values — any divergence means the mirror drifted.
    #[test]
    fn unclamped_matches_clamped_scalar_for_in_gamut() {
        let pixels: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.25, 0.5, 0.75],
            [0.999, 0.001, 0.5],
            [0.1, 0.9, 0.3],
            [0.66, 0.33, 0.0],
            [0.5, 0.5, 0.5],
        ];
        let n = pixels.len();
        let (mut xc, mut yc, mut bc) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        let (mut xu, mut yu, mut bu) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        linear_to_positive_xyb_planar_into(&pixels, &mut xc, &mut yc, &mut bc);
        linear_to_positive_xyb_planar_into_unclamped(&pixels, &mut xu, &mut yu, &mut bu);
        for i in 0..n {
            assert_eq!(xc[i].to_bits(), xu[i].to_bits(), "X differs at px {i}");
            assert_eq!(yc[i].to_bits(), yu[i].to_bits(), "Y differs at px {i}");
            assert_eq!(bc[i].to_bits(), bu[i].to_bits(), "B differs at px {i}");
        }
    }

    /// The unclamped converter handles out-of-gamut input without NaN/inf:
    /// the opsin mix's `max(0)` keeps the cube-root domain valid.
    #[test]
    fn unclamped_out_of_gamut_stays_finite() {
        let pixels: Vec<[f32; 3]> = vec![
            [1.66, -0.125, -0.018], // BT.2020 red through the matrix
            [-0.51, 1.04, -0.31],
            [2.0, -1.0, 3.0],
        ];
        let n = pixels.len();
        let (mut x, mut y, mut b) = (vec![0.0f32; n], vec![0.0f32; n], vec![0.0f32; n]);
        linear_to_positive_xyb_planar_into_unclamped(&pixels, &mut x, &mut y, &mut b);
        for i in 0..n {
            assert!(x[i].is_finite() && y[i].is_finite() && b[i].is_finite());
        }
    }

    /// Verify P3→sRGB matrix: sRGB white (1,1,1) should stay (1,1,1).
    #[test]
    fn p3_to_srgb_preserves_white() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "P3 white channel {c}: expected 1.0, got {val}",
            );
        }
    }

    /// Verify BT.2020→sRGB matrix: sRGB white (1,1,1) should stay (1,1,1).
    #[test]
    fn bt2020_to_srgb_preserves_white() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "BT.2020 white channel {c}: expected 1.0, got {val}",
            );
        }
    }

    /// P3 red primary (1,0,0) in linear P3 → sRGB linear should clamp:
    /// R > 1.0 → clamped to 1.0, G/B negative → clamped to 0.0.
    #[test]
    fn p3_red_clamps_to_srgb_gamut() {
        let mut rgb = [1.0f32, 0.0, 0.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        assert_eq!(rgb[0], 1.0, "R should be clamped to 1.0");
        assert_eq!(rgb[1], 0.0, "G should be clamped to 0.0");
        assert_eq!(rgb[2], 0.0, "B should be clamped to 0.0");
    }

    /// Srgb primaries should be a no-op.
    #[test]
    fn srgb_is_noop() {
        let mut rgb = [0.5f32, 0.3, 0.8];
        let original = rgb;
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Srgb, GamutMapping::Clip);
        assert_eq!(rgb, original);
    }

    /// P3 grey (0.5, 0.5, 0.5) should stay approximately (0.5, 0.5, 0.5)
    /// since the matrices share D65 whitepoint.
    #[test]
    fn p3_grey_stays_grey() {
        let mut rgb = [0.5f32, 0.5, 0.5];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3, GamutMapping::Clip);
        for (c, &val) in rgb.iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 1e-3,
                "P3 grey channel {c}: expected ~0.5, got {val}",
            );
        }
    }

    /// BT.2020 red (1,0,0) should clamp more aggressively than P3 red.
    #[test]
    fn bt2020_red_clamps_to_srgb_gamut() {
        let mut rgb = [1.0f32, 0.0, 0.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020, GamutMapping::Clip);
        assert_eq!(rgb[0], 1.0, "R should be clamped to 1.0");
        assert_eq!(rgb[1], 0.0, "G should be clamped to 0.0");
        assert_eq!(rgb[2], 0.0, "B should be clamped to 0.0");
    }

    /// Verify matrix rows sum to ~1.0 (whitepoint preservation).
    #[test]
    fn matrix_rows_sum_to_one() {
        for (name, m) in [("P3", P3_TO_SRGB), ("BT.2020", BT2020_TO_SRGB)] {
            for (row_idx, row) in m.iter().enumerate() {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-4,
                    "{name} row {row_idx} sum: {sum} (expected ~1.0)"
                );
            }
        }
    }
}

/// Bench-only hooks (`benches/pu21_bench.rs`) — not public API.
#[doc(hidden)]
pub fn bench_pu_xyb_dispatch(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    linear_to_pu_xyb_planar_into(pixels, x_out, y_out, b_out);
}

/// Bench-only scalar reference — see [`bench_pu_xyb_dispatch`].
#[doc(hidden)]
pub fn bench_pu_xyb_scalar(
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    for (i, p) in pixels.iter().enumerate() {
        let (x, y, b) = pu_xyb_pixel(*p);
        x_out[i] = x;
        y_out[i] = y;
        b_out[i] = b;
    }
}

#[cfg(test)]
mod pu_simd_parity_tests {
    use super::*;

    /// SIMD (incant-dispatched) vs scalar per-pixel PU-XYB: the lowp
    /// transcendentals must stay within a tight band of the scalar powf
    /// path across the full luminance range (incl. HDR extremes), so the
    /// dispatched conversion cannot shift scores between machines.
    #[test]
    fn simd_matches_scalar_within_band() {
        // 1031 px (not a multiple of 8 — exercises the tail) spanning
        // 0.001..12000 nits log-spaced with chroma variation.
        let n = 1031;
        let pixels: Vec<[f32; 3]> = (0..n)
            .map(|i| {
                let t = i as f32 / (n - 1) as f32;
                let y = 0.001 * (12_000.0f32 / 0.001).powf(t);
                [y * 1.2, y, y * 0.7]
            })
            .collect();
        let mut xs = vec![0.0f32; n];
        let mut ys = vec![0.0f32; n];
        let mut bs = vec![0.0f32; n];
        linear_to_pu_xyb_planar_into(&pixels, &mut xs, &mut ys, &mut bs);
        let mut max_d = 0.0f32;
        for (i, p) in pixels.iter().enumerate() {
            let (x, y, b) = pu_xyb_pixel(*p);
            for (got, want) in [(xs[i], x), (ys[i], y), (bs[i], b)] {
                let d = (got - want).abs();
                if d > max_d {
                    max_d = d;
                }
            }
        }
        // Outputs live in ~[0, 2.5]; 2e-3 abs keeps score impact negligible
        // while leaving room for the midp_precise transcendental error budget
        // compounding through two pow chains.
        assert!(max_d <= 2e-3, "SIMD vs scalar max |delta| = {max_d}");
    }
}
