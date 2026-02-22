//! Color space conversion: sRGB → linear RGB → XYB
//!
//! XYB is the perceptual color space used by both ssimulacra2 and butteraugli.
//! X ≈ red-green opponent, Y ≈ luminance, B ≈ blue channel.
//! The cube root (LMS cone response) is the key nonlinearity.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

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

/// sRGB u8 → linear f32 lookup table (256 entries)
fn srgb_lut() -> &'static [f32; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut lut = [0.0f32; 256];
        for i in 0..256u16 {
            let s = i as f64 / 255.0;
            let linear = if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            };
            lut[i as usize] = linear as f32;
        }
        lut
    })
}

/// Convert sRGB u8 to linear f32 via lookup table.
#[inline(always)]
pub fn srgb_u8_to_linear(v: u8) -> f32 {
    srgb_lut()[v as usize]
}

/// Fast cube root: bit manipulation + 2 Newton-Raphson iterations.
/// ~4x faster than libm cbrtf, accurate to ~1 ULP.
#[inline(always)]
pub(crate) fn cbrtf_fast(x: f32) -> f32 {
    const B1: u32 = 709_958_130;
    let mut ui = x.to_bits();
    let hx = (ui & 0x7FFF_FFFF) / 3 + B1;
    ui = (ui & 0x8000_0000) | hx;
    let mut t = f64::from(f32::from_bits(ui));
    let xf64 = f64::from(x);
    let mut r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    r = t * t * t;
    t = t * (xf64 + xf64 + r) / (xf64 + r + r);
    t as f32
}

/// Convert interleaved sRGB u8 to planar XYB.
///
/// Input: `&[[u8; 3]]` (sRGB pixels)
/// Output: 3 planes (X, Y, B) each of length `pixels.len()`
pub fn srgb_to_xyb_planar(pixels: &[[u8; 3]]) -> [Vec<f32>; 3] {
    let n = pixels.len();
    let mut x_plane = vec![0.0f32; n];
    let mut y_plane = vec![0.0f32; n];
    let mut b_plane = vec![0.0f32; n];

    incant!(
        srgb_to_xyb_planar_inner(pixels, &mut x_plane, &mut y_plane, &mut b_plane),
        [v3]
    );

    [x_plane, y_plane, b_plane]
}

/// Convert interleaved sRGB u8 to planar XYB, writing into pre-allocated buffers.
pub fn srgb_to_xyb_planar_into(
    pixels: &[[u8; 3]],
    x_plane: &mut [f32],
    y_plane: &mut [f32],
    b_plane: &mut [f32],
) {
    incant!(
        srgb_to_xyb_planar_inner(pixels, x_plane, y_plane, b_plane),
        [v3]
    );
}

/// Make XYB values positive (required for SSIM-like comparisons).
/// X: multiply by 14, add 0.42
/// Y: add 0.01
/// B: B = B - Y + 0.55
#[inline(always)]
pub(crate) fn make_positive_xyb(x: &mut [f32], y: &mut [f32], b: &mut [f32]) {
    incant!(make_positive_xyb_inner(x, y, b), [v3]);
}

// --- SIMD implementations ---

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

        let mut mixed0 = K_M00 * r + K_M01 * g + K_M02 * b + K_B0;
        let mut mixed1 = K_M10 * r + K_M11 * g + K_M12 * b + K_B0;
        let mut mixed2 = K_M20 * r + K_M21 * g + K_M22 * b + K_B0;

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

        let mut mixed0 = K_M00 * r + K_M01 * g + K_M02 * b + K_B0;
        let mut mixed1 = K_M10 * r + K_M11 * g + K_M12 * b + K_B0;
        let mut mixed2 = K_M20 * r + K_M21 * g + K_M22 * b + K_B0;

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
        x[i] = xv * 14.0 + 0.42;
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
        x[i] = xv * 14.0 + 0.42;
        y[i] = yv + 0.01;
        b[i] = (bv - yv) + 0.55;
    }
}
