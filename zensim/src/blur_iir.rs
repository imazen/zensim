//! IIR (recursive) Gaussian blur — Charalampidis 2016.
//!
//! Drop-in replacement for `box_blur_1pass_into` when the `iir-blur` feature is
//! enabled. The box-blur radius `r` is mapped to a Gaussian σ = √(r·(r+1)/3),
//! which matches the variance of a single box-blur pass.
//!
//! O(N) per pixel, independent of σ.
//!
//! Boundary handling is **zero-padding** (mathematically required by the
//! Charalampidis formulation — the DC pole is on the edge of marginal
//! stability and clamp-to-edge diverges). Tiny synthetic images therefore
//! disagree with the box-blur path on the borders; on real photographs the
//! deviation is small (~1-3% on butteraugli's GB82 corpus).

#![cfg(feature = "iir-blur")]
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use archmage::{autoversion, incant, magetypes};
use core::f64::consts::PI;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
use magetypes::simd::v4::f32x16 as MtF32x16;

/// IIR weights for one Gaussian sigma. Three parallel 2-pole sections (k=1,3,5),
/// constants derived in f64 for numerical stability and stored as f32.
#[derive(Clone, Copy, Debug)]
pub struct IirCoeffs {
    pub radius: i32,
    pub mul_in: [f32; 3],
    pub mul_prev: [f32; 3],
}

impl IirCoeffs {
    pub fn for_sigma(sigma: f32) -> Self {
        let sigma = sigma as f64;
        let radius = 3.2795_f64.mul_add(sigma, 0.2546).round();
        let pi_div_2r = PI / (2.0 * radius);
        let omega = [pi_div_2r, 3.0 * pi_div_2r, 5.0 * pi_div_2r];

        let p = [
            1.0 / (0.5 * omega[0]).tan(),
            -1.0 / (0.5 * omega[1]).tan(),
            1.0 / (0.5 * omega[2]).tan(),
        ];
        let r = [
            p[0] * p[0] / omega[0].sin(),
            -p[1] * p[1] / omega[1].sin(),
            p[2] * p[2] / omega[2].sin(),
        ];
        let neg_half_sigma2 = -0.5 * sigma * sigma;
        let recip_radius = 1.0 / radius;
        let rho = [
            (neg_half_sigma2 * omega[0] * omega[0]).exp() * recip_radius,
            (neg_half_sigma2 * omega[1] * omega[1]).exp() * recip_radius,
            (neg_half_sigma2 * omega[2] * omega[2]).exp() * recip_radius,
        ];
        let d_13 = p[0].mul_add(r[1], -r[0] * p[1]);
        let d_35 = p[1].mul_add(r[2], -r[1] * p[2]);
        let d_51 = p[2].mul_add(r[0], -r[2] * p[0]);
        let recip_d13 = 1.0 / d_13;
        let zeta_15 = d_35 * recip_d13;
        let zeta_35 = d_51 * recip_d13;

        let g0 = 1.0;
        let g1 = radius.mul_add(radius, -sigma * sigma);
        let g2 = zeta_15.mul_add(rho[0], zeta_35 * rho[1]) + rho[2];
        let beta = solve_3x3(
            [
                [p[0], p[1], p[2]],
                [r[0], r[1], r[2]],
                [zeta_15, zeta_35, 1.0],
            ],
            [g0, g1, g2],
        );

        let mul_in = [
            (-beta[0] * (omega[0] * (radius + 1.0)).cos()) as f32,
            (-beta[1] * (omega[1] * (radius + 1.0)).cos()) as f32,
            (-beta[2] * (omega[2] * (radius + 1.0)).cos()) as f32,
        ];
        let mul_prev = [
            (-2.0 * omega[0].cos()) as f32,
            (-2.0 * omega[1].cos()) as f32,
            (-2.0 * omega[2].cos()) as f32,
        ];

        Self {
            radius: radius as i32,
            mul_in,
            mul_prev,
        }
    }

    /// Pick coefficients for a Gaussian that matches the variance of a single
    /// box-blur pass of `radius` (window 2r+1). σ = √(r·(r+1)/3).
    pub fn for_box_radius(radius: usize) -> Self {
        let r = radius as f64;
        let sigma = ((r * (r + 1.0)) / 3.0).sqrt() as f32;
        Self::for_sigma(sigma.max(1e-3))
    }
}

fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let inv_det = 1.0 / det;
    let x0 = b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]);
    let x1 = a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]);
    let x2 = a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    [x0 * inv_det, x1 * inv_det, x2 * inv_det]
}

#[allow(unused_imports)]
#[autoversion]
fn horizontal_pass(input: &[f32], output: &mut [f32], width: usize, coeffs: &IirCoeffs) {
    for (in_row, out_row) in input
        .chunks_exact(width)
        .zip(output.chunks_exact_mut(width))
    {
        horizontal_row(in_row, out_row, coeffs);
    }
}

#[inline(always)]
fn horizontal_row(input: &[f32], output: &mut [f32], coeffs: &IirCoeffs) {
    let width = input.len() as isize;
    let big_n = coeffs.radius as isize;

    let mi1 = coeffs.mul_in[0];
    let mi3 = coeffs.mul_in[1];
    let mi5 = coeffs.mul_in[2];
    let mp1 = coeffs.mul_prev[0];
    let mp3 = coeffs.mul_prev[1];
    let mp5 = coeffs.mul_prev[2];

    let mut prev_1 = 0f32;
    let mut prev_3 = 0f32;
    let mut prev_5 = 0f32;
    let mut prev2_1 = 0f32;
    let mut prev2_3 = 0f32;
    let mut prev2_5 = 0f32;

    let mut n = -big_n + 1;
    while n < width {
        let left = n - big_n - 1;
        let right = n + big_n - 1;
        let left_val = if left >= 0 && left < width {
            input[left as usize]
        } else {
            0f32
        };
        let right_val = if right >= 0 && right < width {
            input[right as usize]
        } else {
            0f32
        };
        let sum = left_val + right_val;

        let out_1 = sum.mul_add(mi1, -mp1.mul_add(prev_1, prev2_1));
        let out_3 = sum.mul_add(mi3, -mp3.mul_add(prev_3, prev2_3));
        let out_5 = sum.mul_add(mi5, -mp5.mul_add(prev_5, prev2_5));

        prev2_1 = prev_1;
        prev2_3 = prev_3;
        prev2_5 = prev_5;
        prev_1 = out_1;
        prev_3 = out_3;
        prev_5 = out_5;

        if n >= 0 {
            output[n as usize] = out_1 + out_3 + out_5;
        }
        n += 1;
    }
}

fn vertical_pass(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    incant!(
        vertical_pass_inner(input, output, width, height, coeffs),
        [v4, v3, neon, wasm128, scalar]
    )
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
fn vertical_pass_inner_v4(
    token: archmage::X64V4Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    const LANES: usize = 16;
    let big_n = coeffs.radius as isize;
    let height_i = height as isize;
    let groups = width / LANES;

    let mi1 = MtF32x16::splat(token, coeffs.mul_in[0]);
    let mi3 = MtF32x16::splat(token, coeffs.mul_in[1]);
    let mi5 = MtF32x16::splat(token, coeffs.mul_in[2]);
    let mp1 = MtF32x16::splat(token, coeffs.mul_prev[0]);
    let mp3 = MtF32x16::splat(token, coeffs.mul_prev[1]);
    let mp5 = MtF32x16::splat(token, coeffs.mul_prev[2]);
    let zeroes = MtF32x16::zero(token);

    for g in 0..groups {
        let col = g * LANES;
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_v = if top >= 0 && top < height_i {
                MtF32x16::from_array(
                    token,
                    input[top as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                MtF32x16::from_array(
                    token,
                    input[bottom as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let sum = top_v + bot_v;

            let acc1 = prev_1.mul_add(mp1, prev2_1);
            let acc3 = prev_3.mul_add(mp3, prev2_3);
            let acc5 = prev_5.mul_add(mp5, prev2_5);
            let out1 = sum.mul_add(mi1, -acc1);
            let out3 = sum.mul_add(mi3, -acc3);
            let out5 = sum.mul_add(mi5, -acc5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = out1 + out3 + out5;
                let dst = n as usize * width + col;
                output[dst..dst + LANES].copy_from_slice(&result.to_array());
            }
            n += 1;
        }
    }

    let scalar_start = groups * LANES;
    if scalar_start < width {
        vertical_pass_scalar_columns(input, output, width, height, scalar_start, coeffs);
    }
}

#[magetypes(v3, neon, wasm128, scalar)]
fn vertical_pass_inner(
    token: Token,
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    coeffs: &IirCoeffs,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    let big_n = coeffs.radius as isize;
    let height_i = height as isize;
    let groups = width / LANES;

    let mi1 = f32x8::splat(token, coeffs.mul_in[0]);
    let mi3 = f32x8::splat(token, coeffs.mul_in[1]);
    let mi5 = f32x8::splat(token, coeffs.mul_in[2]);
    let mp1 = f32x8::splat(token, coeffs.mul_prev[0]);
    let mp3 = f32x8::splat(token, coeffs.mul_prev[1]);
    let mp5 = f32x8::splat(token, coeffs.mul_prev[2]);
    let zeroes = f32x8::zero(token);

    for g in 0..groups {
        let col = g * LANES;
        let mut prev_1 = zeroes;
        let mut prev_3 = zeroes;
        let mut prev_5 = zeroes;
        let mut prev2_1 = zeroes;
        let mut prev2_3 = zeroes;
        let mut prev2_5 = zeroes;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;

            let top_v = if top >= 0 && top < height_i {
                f32x8::from_array(
                    token,
                    input[top as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                f32x8::from_array(
                    token,
                    input[bottom as usize * width + col..][..LANES]
                        .try_into()
                        .unwrap(),
                )
            } else {
                zeroes
            };
            let sum = top_v + bot_v;

            let acc1 = prev_1.mul_add(mp1, prev2_1);
            let acc3 = prev_3.mul_add(mp3, prev2_3);
            let acc5 = prev_5.mul_add(mp5, prev2_5);
            let out1 = sum.mul_add(mi1, -acc1);
            let out3 = sum.mul_add(mi3, -acc3);
            let out5 = sum.mul_add(mi5, -acc5);

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out1;
            prev_3 = out3;
            prev_5 = out5;

            if n >= 0 {
                let result = out1 + out3 + out5;
                let dst = n as usize * width + col;
                output[dst..dst + LANES].copy_from_slice(&result.to_array());
            }
            n += 1;
        }
    }

    let scalar_start = groups * LANES;
    if scalar_start < width {
        vertical_pass_scalar_columns(input, output, width, height, scalar_start, coeffs);
    }
}

#[allow(unused_imports)]
#[autoversion]
fn vertical_pass_scalar_columns(
    input: &[f32],
    output: &mut [f32],
    width: usize,
    height: usize,
    start_x: usize,
    coeffs: &IirCoeffs,
) {
    let big_n = coeffs.radius as isize;
    let height_i = height as isize;

    let mi1 = coeffs.mul_in[0];
    let mi3 = coeffs.mul_in[1];
    let mi5 = coeffs.mul_in[2];
    let mp1 = coeffs.mul_prev[0];
    let mp3 = coeffs.mul_prev[1];
    let mp5 = coeffs.mul_prev[2];

    for x in start_x..width {
        let mut prev_1 = 0f32;
        let mut prev_3 = 0f32;
        let mut prev_5 = 0f32;
        let mut prev2_1 = 0f32;
        let mut prev2_3 = 0f32;
        let mut prev2_5 = 0f32;

        let mut n = -big_n + 1;
        while n < height_i {
            let top = n - big_n - 1;
            let bottom = n + big_n - 1;
            let top_v = if top >= 0 && top < height_i {
                input[top as usize * width + x]
            } else {
                0f32
            };
            let bot_v = if bottom >= 0 && bottom < height_i {
                input[bottom as usize * width + x]
            } else {
                0f32
            };
            let sum = top_v + bot_v;

            let out_1 = sum.mul_add(mi1, -mp1.mul_add(prev_1, prev2_1));
            let out_3 = sum.mul_add(mi3, -mp3.mul_add(prev_3, prev2_3));
            let out_5 = sum.mul_add(mi5, -mp5.mul_add(prev_5, prev2_5));

            prev2_1 = prev_1;
            prev2_3 = prev_3;
            prev2_5 = prev_5;
            prev_1 = out_1;
            prev_3 = out_3;
            prev_5 = out_5;

            if n >= 0 {
                output[n as usize * width + x] = out_1 + out_3 + out_5;
            }
            n += 1;
        }
    }
}

/// Drop-in replacement for `box_blur_1pass_into` using an IIR Gaussian whose
/// variance matches a single (2r+1)-tap box blur.
///
/// `temp` is used as a one-row-pass scratch buffer; the H pass writes into it,
/// the V pass reads from it.
pub fn iir_blur_1pass_into(
    input: &[f32],
    output: &mut [f32],
    temp: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
) {
    let coeffs = IirCoeffs::for_box_radius(radius);
    horizontal_pass(input, temp, width, &coeffs);
    vertical_pass(temp, output, width, height, &coeffs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dc_constant_preserved() {
        let w = 64;
        let h = 64;
        let n = w * h;
        let input = vec![0.5f32; n];
        let mut temp = vec![0f32; n];
        let mut output = vec![0f32; n];
        iir_blur_1pass_into(&input, &mut output, &mut temp, w, h, 3);
        let center = output[(h / 2) * w + (w / 2)];
        assert!(
            (center - 0.5).abs() < 5e-3,
            "center should be ~0.5, got {center}",
        );
    }

    #[test]
    fn impulse_sum_unity() {
        let w = 128;
        let h = 128;
        let n = w * h;
        let mut input = vec![0f32; n];
        input[(h / 2) * w + (w / 2)] = 1.0;
        let mut temp = vec![0f32; n];
        let mut output = vec![0f32; n];
        iir_blur_1pass_into(&input, &mut output, &mut temp, w, h, 5);
        let sum: f32 = output.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.05,
            "2D impulse sum {sum}, expected ~1.0",
        );
    }

    #[test]
    fn coeffs_for_box_radius_3() {
        // radius=3 → σ = √(12/3) = 2.0
        let coeffs = IirCoeffs::for_box_radius(3);
        let coeffs_ref = IirCoeffs::for_sigma(2.0);
        assert_eq!(coeffs.radius, coeffs_ref.radius);
    }
}
