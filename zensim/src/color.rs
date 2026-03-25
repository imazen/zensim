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
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

// Opsin absorbance matrix (from jpegli/ssimulacra2)
const K_M02: f32 = 0.078;
const K_M00: f32 = 0.30;
const K_M01: f32 = 1.0 - K_M02 - K_M00;
const K_M12: f32 = 0.078;
const K_M10: f32 = 0.23;
const K_M11: f32 = 1.0 - K_M12 - K_M10;
const K_M20: f32 = 0.243_422_69;
const K_M21: f32 = 0.204_767_45;
const K_M22: f32 = 1.0 - K_M20 - K_M21;
const K_B0: f32 = 0.003_793_073_4;

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

use crate::source::ColorPrimaries;

/// Apply a gamut conversion matrix to a linear RGB pixel.
///
/// Converts from the source color primaries to sRGB linear light.
/// For [`ColorPrimaries::Srgb`] this is a no-op. Results are clamped
/// to \[0, 1\] — out-of-gamut colors are clipped (acceptable for SDR).
#[inline]
pub(crate) fn apply_gamut_matrix(rgb: &mut [f32; 3], primaries: ColorPrimaries) {
    #[allow(unreachable_patterns)]
    let m = match primaries {
        ColorPrimaries::Srgb => return,
        ColorPrimaries::DisplayP3 => &P3_TO_SRGB,
        ColorPrimaries::Bt2020 => &BT2020_TO_SRGB,
        _ => return, // future variants: pass through unchanged
    };
    let [r, g, b] = *rgb;
    rgb[0] = (m[0][0] * r + m[0][1] * g + m[0][2] * b).clamp(0.0, 1.0);
    rgb[1] = (m[1][0] * r + m[1][1] * g + m[1][2] * b).clamp(0.0, 1.0);
    rgb[2] = (m[2][0] * r + m[2][1] * g + m[2][2] * b).clamp(0.0, 1.0);
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
#[allow(dead_code)] // For future streaming optimization (avoids per-strip allocations)
pub fn srgb_to_positive_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    incant!(
        srgb_to_positive_xyb_planar_inner(pixels, x_out, y_out, b_out),
        [v4, v3, scalar]
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
        [v3, scalar]
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
        [v3, scalar]
    );
}

/// Make XYB values positive (required for SSIM-like comparisons).
/// X: multiply by 14, add 0.42
/// Y: add 0.01
/// B: B = B - Y + 0.55
#[inline(always)]
#[allow(dead_code)]
pub(crate) fn make_positive_xyb(x: &mut [f32], y: &mut [f32], b: &mut [f32]) {
    incant!(make_positive_xyb_inner(x, y, b), [v3, scalar]);
}

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
    let two = f32x16::splat(token, 2.0);
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
        for i in 0..16 {
            let p = pixels[base + i];
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

        // Scalar initial estimates (integer bit manipulation)
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..16 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        // Halley's iterations in SIMD (3 channels interleaved for ILP)
        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x16::from_array(token, est0);
        let mut t1 = f32x16::from_array(token, est1);
        let mut t2 = f32x16::from_array(token, est2);

        // Iteration 1
        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

        // Iteration 2
        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

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
    let two8 = f32x8::splat(v3, 2.0);
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
        for i in 0..8 {
            let p = pixels[base + i];
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

        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..8 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x8::from_array(v3, est0);
        let mut t1 = f32x8::from_array(v3, est1);
        let mut t2 = f32x8::from_array(v3, est2);

        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two8, r0)) / (x0 + r0.mul_add(two8, zero8));
        t1 *= (x1.mul_add(two8, r1)) / (x1 + r1.mul_add(two8, zero8));
        t2 *= (x2.mul_add(two8, r2)) / (x2 + r2.mul_add(two8, zero8));

        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two8, r0)) / (x0 + r0.mul_add(two8, zero8));
        t1 *= (x1.mul_add(two8, r1)) / (x1 + r1.mul_add(two8, zero8));
        t2 *= (x2.mul_add(two8, r2)) / (x2 + r2.mul_add(two8, zero8));

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
    let two = f32x8::splat(token, 2.0);
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
        for i in 0..8 {
            let p = pixels[base + i];
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

        // Scalar initial estimates (integer bit manipulation)
        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..8 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        // Halley's iterations in SIMD (3 channels interleaved for ILP)
        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x8::from_array(token, est0);
        let mut t1 = f32x8::from_array(token, est1);
        let mut t2 = f32x8::from_array(token, est2);

        // Iteration 1
        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

        // Iteration 2
        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

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

fn srgb_to_positive_xyb_planar_inner_scalar(
    _token: archmage::ScalarToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    for (i, p) in pixels.iter().enumerate() {
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
        for i in 0..8 {
            let p = pixels[base + i];
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

        // Cube root (scalar — hard to vectorize due to bit manipulation)
        let mut m0 = mixed0.to_array();
        let mut m1 = mixed1.to_array();
        let mut m2 = mixed2.to_array();
        for i in 0..8 {
            m0[i] = cbrtf_fast(m0[i]);
            m1[i] = cbrtf_fast(m1[i]);
            m2[i] = cbrtf_fast(m2[i]);
        }

        let c0 = f32x8::from_array(token, m0) + ab;
        let c1 = f32x8::from_array(token, m1) + ab;
        let c2 = f32x8::from_array(token, m2);

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

fn srgb_to_xyb_planar_inner_scalar(
    _token: archmage::ScalarToken,
    pixels: &[[u8; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    for (i, p) in pixels.iter().enumerate() {
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

#[cfg(target_arch = "x86_64")]
#[arcane]
fn make_positive_xyb_inner_v3(
    token: archmage::X64V3Token,
    x: &mut [f32],
    y: &mut [f32],
    b: &mut [f32],
) {
    let fourteen = f32x8::splat(token, 14.0);
    let x_bias = f32x8::splat(token, 0.42);
    let y_bias = f32x8::splat(token, 0.01);
    let b_bias = f32x8::splat(token, 0.55);

    let n = x.len();
    let chunks = n / 8;

    for chunk in 0..chunks {
        let base = chunk * 8;

        let xv = f32x8::from_array(token, x[base..][..8].try_into().unwrap());
        let yv = f32x8::from_array(token, y[base..][..8].try_into().unwrap());
        let bv = f32x8::from_array(token, b[base..][..8].try_into().unwrap());

        let x_pos = xv.mul_add(fourteen, x_bias);
        let y_pos = yv + y_bias;
        let b_pos = (bv - yv) + b_bias;

        x[base..base + 8].copy_from_slice(&x_pos.to_array());
        y[base..base + 8].copy_from_slice(&y_pos.to_array());
        b[base..base + 8].copy_from_slice(&b_pos.to_array());
    }

    for i in (chunks * 8)..n {
        let xv = x[i];
        let yv = y[i];
        let bv = b[i];
        x[i] = xv.mul_add(14.0, 0.42);
        y[i] = yv + 0.01;
        b[i] = (bv - yv) + 0.55;
    }
}

fn make_positive_xyb_inner_scalar(
    _token: archmage::ScalarToken,
    x: &mut [f32],
    y: &mut [f32],
    b: &mut [f32],
) {
    for i in 0..x.len() {
        let xv = x[i];
        let yv = y[i];
        let bv = b[i];
        x[i] = xv.mul_add(14.0, 0.42);
        y[i] = yv + 0.01;
        b[i] = (bv - yv) + 0.55;
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
        [v4, v3, scalar]
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
    let two = f32x16::splat(token, 2.0);
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
        for i in 0..16 {
            let p = pixels[base + i];
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

        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..16 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x16::from_array(token, est0);
        let mut t1 = f32x16::from_array(token, est1);
        let mut t2 = f32x16::from_array(token, est2);

        // Halley iteration 1
        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

        // Halley iteration 2
        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

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
    let two8 = f32x8::splat(v3, 2.0);
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
        for i in 0..8 {
            let p = pixels[base + i];
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

        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..8 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x8::from_array(v3, est0);
        let mut t1 = f32x8::from_array(v3, est1);
        let mut t2 = f32x8::from_array(v3, est2);

        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two8, r0)) / (x0 + r0.mul_add(two8, zero8));
        t1 *= (x1.mul_add(two8, r1)) / (x1 + r1.mul_add(two8, zero8));
        t2 *= (x2.mul_add(two8, r2)) / (x2 + r2.mul_add(two8, zero8));

        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two8, r0)) / (x0 + r0.mul_add(two8, zero8));
        t1 *= (x1.mul_add(two8, r1)) / (x1 + r1.mul_add(two8, zero8));
        t2 *= (x2.mul_add(two8, r2)) / (x2 + r2.mul_add(two8, zero8));

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
    let two = f32x8::splat(token, 2.0);
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
        for i in 0..8 {
            let p = pixels[base + i];
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

        let mut est0 = mixed0.to_array();
        let mut est1 = mixed1.to_array();
        let mut est2 = mixed2.to_array();
        for i in 0..8 {
            est0[i] = cbrtf_initial(est0[i]);
            est1[i] = cbrtf_initial(est1[i]);
            est2[i] = cbrtf_initial(est2[i]);
        }

        let x0 = mixed0;
        let x1 = mixed1;
        let x2 = mixed2;
        let mut t0 = f32x8::from_array(token, est0);
        let mut t1 = f32x8::from_array(token, est1);
        let mut t2 = f32x8::from_array(token, est2);

        let mut r0 = t0 * t0 * t0;
        let mut r1 = t1 * t1 * t1;
        let mut r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

        r0 = t0 * t0 * t0;
        r1 = t1 * t1 * t1;
        r2 = t2 * t2 * t2;
        t0 *= (x0.mul_add(two, r0)) / (x0 + r0.mul_add(two, zero));
        t1 *= (x1.mul_add(two, r1)) / (x1 + r1.mul_add(two, zero));
        t2 *= (x2.mul_add(two, r2)) / (x2 + r2.mul_add(two, zero));

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

/// Scalar fallback for linear f32 → positive XYB.
fn linear_to_positive_xyb_planar_inner_scalar(
    _token: archmage::ScalarToken,
    pixels: &[[f32; 3]],
    x_out: &mut [f32],
    y_out: &mut [f32],
    b_out: &mut [f32],
) {
    let absorbance_bias = -cbrtf_fast(K_B0);

    for (i, p) in pixels.iter().enumerate() {
        // Clamp to display gamut: out-of-range values from lossy reconstruction
        // aren't visible on a real display, so measuring them would overcount error.
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

// ---------------------------------------------------------------------------
// RGBA/BGRA compositing helpers — all produce linear f32 RGB output
// ---------------------------------------------------------------------------

/// Checkerboard background value in linear light for the given pixel position.
#[inline(always)]
fn checkerboard_linear(x: usize, y: usize) -> f32 {
    if ((x >> 3) ^ (y >> 3)) & 1 == 0 {
        0.0
    } else {
        1.0
    }
}

/// Composite sRGB u8 RGBA over a checkerboard, producing linear f32 RGB.
///
/// Linearizes both foreground and background, then alpha-blends in linear space.
/// This ensures consistent XYB values regardless of whether the input was
/// sRGB u8 or linear f32.
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
            let bg = checkerboard_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = checkerboard_linear(x, y);
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

/// Composite sRGB u8 BGRA over a checkerboard, producing linear f32 RGB.
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
            let bg = checkerboard_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let alpha = a as f32 * (1.0 / 255.0);
            let inv = 1.0 - alpha;
            let bg = checkerboard_linear(x, y);
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

/// Composite linear f32 RGBA over a checkerboard, producing linear f32 RGB.
pub(crate) fn composite_linear_f32_rgba(row: &[[f32; 4]], y: usize, out: &mut [[f32; 3]]) {
    for (x, &[r, g, b, a]) in row.iter().enumerate() {
        if a >= 1.0 {
            out[x] = [r, g, b];
        } else if a <= 0.0 {
            let bg = checkerboard_linear(x, y);
            out[x] = [bg, bg, bg];
        } else {
            let inv = 1.0 - a;
            let bg = checkerboard_linear(x, y);
            out[x] = [
                r.mul_add(a, bg * inv),
                g.mul_add(a, bg * inv),
                b.mul_add(a, bg * inv),
            ];
        }
    }
}

/// Composite sRGB u16 RGBA over a checkerboard, producing linear f32 RGB.
///
/// u16 values 0-65535 are linearized via `srgb_u16_to_linear()`, then alpha-blended
/// in linear space against the checkerboard background.
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
            let bg = checkerboard_linear(x, y);
            *out_pixel = [bg, bg, bg];
        } else {
            let alpha = a as f32 / 65535.0;
            let inv = 1.0 - alpha;
            let bg = checkerboard_linear(x, y);
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

    /// Verify P3→sRGB matrix: sRGB white (1,1,1) should stay (1,1,1).
    #[test]
    fn p3_to_srgb_preserves_white() {
        let mut rgb = [1.0f32, 1.0, 1.0];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3);
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
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020);
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
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3);
        assert_eq!(rgb[0], 1.0, "R should be clamped to 1.0");
        assert_eq!(rgb[1], 0.0, "G should be clamped to 0.0");
        assert_eq!(rgb[2], 0.0, "B should be clamped to 0.0");
    }

    /// Srgb primaries should be a no-op.
    #[test]
    fn srgb_is_noop() {
        let mut rgb = [0.5f32, 0.3, 0.8];
        let original = rgb;
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Srgb);
        assert_eq!(rgb, original);
    }

    /// P3 grey (0.5, 0.5, 0.5) should stay approximately (0.5, 0.5, 0.5)
    /// since the matrices share D65 whitepoint.
    #[test]
    fn p3_grey_stays_grey() {
        let mut rgb = [0.5f32, 0.5, 0.5];
        apply_gamut_matrix(&mut rgb, ColorPrimaries::DisplayP3);
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
        apply_gamut_matrix(&mut rgb, ColorPrimaries::Bt2020);
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
