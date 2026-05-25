//! f32 SIMD-accelerated encoder kernels.
//!
//! Mirrors `simd_encoder.rs` but operates on f32 slices, doubling the
//! SIMD lane count: f32x16 on AVX-512 (vs f64x8), f32x8 on AVX2 (vs
//! f64x4). Expected ~2× throughput on compute-bound inner loops.
//!
//! The f32 precision is sufficient for the encoder forward/backward:
//! input features are standardized (mean=0, std=1) and the bake output
//! is already f32. The only precision-sensitive operation is the Adam
//! optimizer (momentum/variance accumulators), which can stay f64 if
//! needed via mixed-precision training.

#![allow(clippy::needless_range_loop)]

use archmage::{incant, magetypes};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16 as GenericF32x16};

/// f32 encoder forward: same semantics as the f64 version.
#[inline]
pub fn encoder_forward_f32(
    x: &[f32],
    w1: &[f32],
    b1: &[f32],
    n_features: usize,
    n_hidden: usize,
    leaky_alpha: f32,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(b1.len(), n_hidden);
    debug_assert_eq!(w1.len(), n_features * n_hidden);

    let mut h_pre = b1.to_vec();
    incant!(accumulate_rows_f32(x, w1, &mut h_pre, n_features, n_hidden), [+v4]);
    let h = incant!(apply_leaky_relu_f32(&h_pre, leaky_alpha), [+v4]);
    (h_pre, h)
}

/// f32 dot product with bias.
#[inline]
pub fn dot_bias_f32(h: &[f32], w: &[f32], bias: f32) -> f32 {
    debug_assert_eq!(h.len(), w.len());
    bias + incant!(dot_product_f32(h, w), [+v4])
}

// =============================================================================
// AVX-512 path — f32x16 (16 lanes per instruction)
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_f32_v4(
    token: archmage::X64V4Token,
    x: &[f32], w1: &[f32], h_pre: &mut [f32],
    n_features: usize, n_hidden: usize,
) {
    let chunks = n_hidden / 16;
    let tail = chunks * 16;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 { continue; }
        let s_v = GenericF32x16::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 16;
            let h_block: &mut [f32; 16] = (&mut h_pre[off..off + 16]).try_into().unwrap();
            let w_block: &[f32; 16] = (&row[off..off + 16]).try_into().unwrap();
            let h_v = GenericF32x16::load(token, h_block);
            let w_v = GenericF32x16::load(token, w_block);
            s_v.mul_add(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden {
            h_pre[j] += s * row[j];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_f32_v4(token: archmage::X64V4Token, h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 16;
    let tail = chunks * 16;
    let leaky_v = GenericF32x16::splat(token, leaky_alpha);
    let zero_v = GenericF32x16::zero(token);
    for c in 0..chunks {
        let off = c * 16;
        let in_block: &[f32; 16] = (&h_pre[off..off + 16]).try_into().unwrap();
        let out_block: &mut [f32; 16] = (&mut h[off..off + 16]).try_into().unwrap();
        let v = GenericF32x16::load(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        GenericF32x16::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_f32_v4(token: archmage::X64V4Token, h: &[f32], w: &[f32]) -> f32 {
    let chunks = h.len() / 16;
    let tail = chunks * 16;
    let mut acc = GenericF32x16::zero(token);
    for c in 0..chunks {
        let off = c * 16;
        let h_block: &[f32; 16] = (&h[off..off + 16]).try_into().unwrap();
        let w_block: &[f32; 16] = (&w[off..off + 16]).try_into().unwrap();
        let h_v = GenericF32x16::load(token, h_block);
        let w_v = GenericF32x16::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    let mut lanes = [0.0f32; 16];
    acc.store(&mut lanes);
    let mut lane_sum = 0.0f32;
    for &l in &lanes { lane_sum += l; }
    let mut tail_sum = 0.0f32;
    for j in tail..h.len() { tail_sum += h[j] * w[j]; }
    lane_sum + tail_sum
}

// =============================================================================
// AVX2 path — f32x8
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn accumulate_rows_f32_v3(
    token: archmage::X64V3Token,
    x: &[f32], w1: &[f32], h_pre: &mut [f32],
    n_features: usize, n_hidden: usize,
) {
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 { continue; }
        let s_v = GenericF32x8::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 8;
            let h_block: &mut [f32; 8] = (&mut h_pre[off..off + 8]).try_into().unwrap();
            let w_block: &[f32; 8] = (&row[off..off + 8]).try_into().unwrap();
            let h_v = GenericF32x8::load(token, h_block);
            let w_v = GenericF32x8::load(token, w_block);
            s_v.mul_add(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden { h_pre[j] += s * row[j]; }
    }
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn apply_leaky_relu_f32_v3(token: archmage::X64V3Token, h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 8;
    let tail = chunks * 8;
    let leaky_v = GenericF32x8::splat(token, leaky_alpha);
    let zero_v = GenericF32x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let in_block: &[f32; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f32; 8] = (&mut h[off..off + 8]).try_into().unwrap();
        let v = GenericF32x8::load(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        GenericF32x8::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn dot_product_f32_v3(token: archmage::X64V3Token, h: &[f32], w: &[f32]) -> f32 {
    let chunks = h.len() / 8;
    let tail = chunks * 8;
    let mut acc = GenericF32x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let h_block: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = GenericF32x8::load(token, h_block);
        let w_v = GenericF32x8::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0f32;
    for j in tail..h.len() { tail_sum += h[j] * w[j]; }
    lane_sum + tail_sum
}

// =============================================================================
// NEON / WASM / scalar fallback — f32x8 generic
// =============================================================================

#[magetypes(neon, wasm128, scalar)]
fn accumulate_rows_f32(
    token: Token,
    x: &[f32], w1: &[f32], h_pre: &mut [f32],
    n_features: usize, n_hidden: usize,
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let chunks = n_hidden / 8;
    let tail = chunks * 8;
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 { continue; }
        let s_v = f32x8::splat(token, s);
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for c in 0..chunks {
            let off = c * 8;
            let h_block: &mut [f32; 8] = (&mut h_pre[off..off + 8]).try_into().unwrap();
            let w_block: &[f32; 8] = (&row[off..off + 8]).try_into().unwrap();
            let h_v = f32x8::load(token, h_block);
            let w_v = f32x8::load(token, w_block);
            s_v.mul_add(w_v, h_v).store(h_block);
        }
        for j in tail..n_hidden { h_pre[j] += s * row[j]; }
    }
}

#[magetypes(neon, wasm128, scalar)]
fn apply_leaky_relu_f32(token: Token, h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let mut h = vec![0.0f32; h_pre.len()];
    let chunks = h_pre.len() / 8;
    let tail = chunks * 8;
    let leaky_v = f32x8::splat(token, leaky_alpha);
    let zero_v = f32x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let in_block: &[f32; 8] = (&h_pre[off..off + 8]).try_into().unwrap();
        let out_block: &mut [f32; 8] = (&mut h[off..off + 8]).try_into().unwrap();
        let v = f32x8::load(token, in_block);
        let scaled = v * leaky_v;
        let neg_mask = v.simd_lt(zero_v);
        f32x8::blend(neg_mask, scaled, v).store(out_block);
    }
    for j in tail..h_pre.len() {
        let v = h_pre[j];
        h[j] = if v >= 0.0 { v } else { leaky_alpha * v };
    }
    h
}

#[magetypes(neon, wasm128, scalar)]
fn dot_product_f32(token: Token, h: &[f32], w: &[f32]) -> f32 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let chunks = h.len() / 8;
    let tail = chunks * 8;
    let mut acc = f32x8::zero(token);
    for c in 0..chunks {
        let off = c * 8;
        let h_block: &[f32; 8] = (&h[off..off + 8]).try_into().unwrap();
        let w_block: &[f32; 8] = (&w[off..off + 8]).try_into().unwrap();
        let h_v = f32x8::load(token, h_block);
        let w_v = f32x8::load(token, w_block);
        acc = h_v.mul_add(w_v, acc);
    }
    let mut lanes = [0.0f32; 8];
    acc.store(&mut lanes);
    let lane_sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    let mut tail_sum = 0.0f32;
    for j in tail..h.len() { tail_sum += h[j] * w[j]; }
    lane_sum + tail_sum
}

/// f32 encoder backprop: accumulate gw1/gb1 from dl_dh_pre.
#[inline]
pub fn encoder_backprop_layer1_f32(
    x: &[f32],
    dl_dh_pre: &[f32],
    gw1: &mut [f32],
    gb1: &mut [f32],
    n_features: usize,
    n_hidden: usize,
) {
    debug_assert_eq!(x.len(), n_features);
    debug_assert_eq!(dl_dh_pre.len(), n_hidden);
    debug_assert_eq!(gw1.len(), n_features * n_hidden);
    debug_assert_eq!(gb1.len(), n_hidden);

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

/// f32 LeakyReLU backward.
#[inline]
pub fn leaky_relu_backward_f32(dl_dh: &[f32], h_pre: &[f32], leaky_alpha: f32) -> Vec<f32> {
    debug_assert_eq!(dl_dh.len(), h_pre.len());
    dl_dh
        .iter()
        .zip(h_pre.iter())
        .map(|(&dh, &hp)| if hp >= 0.0 { dh } else { leaky_alpha * dh })
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    struct Xs32(u64);
    impl Xs32 {
        fn new(seed: u64) -> Self { Self(seed | 1) }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x; x
        }
        fn next_unit(&mut self) -> f32 {
            (self.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
        }
    }

    fn random_vec_f32(rng: &mut Xs32, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.next_unit()).collect()
    }

    fn random_sparse_f32(rng: &mut Xs32, n: usize, zero_frac: f64) -> Vec<f32> {
        let thresh = (2.0 * zero_frac - 1.0) as f32;
        (0..n).map(|_| if rng.next_unit() < thresh { 0.0 } else { rng.next_unit() }).collect()
    }

    #[test]
    fn encoder_forward_f32_production_shape() {
        let n_features = 372;
        let n_hidden = 128;
        let alpha = 0.01f32;
        let mut rng = Xs32::new(0xCAFE);
        let x = random_sparse_f32(&mut rng, n_features, 0.3);
        let w1 = random_vec_f32(&mut rng, n_features * n_hidden);
        let b1 = random_vec_f32(&mut rng, n_hidden);
        let (h_pre, h) = encoder_forward_f32(&x, &w1, &b1, n_features, n_hidden, alpha);
        assert_eq!(h_pre.len(), n_hidden);
        assert_eq!(h.len(), n_hidden);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn dot_bias_f32_matches_scalar() {
        let n = 128;
        let mut rng = Xs32::new(0xBEEF);
        let h = random_vec_f32(&mut rng, n);
        let w = random_vec_f32(&mut rng, n);
        let bias = rng.next_unit();
        let scalar = bias + h.iter().zip(w.iter()).map(|(&a, &b)| a * b).sum::<f32>();
        let simd = dot_bias_f32(&h, &w, bias);
        let rel = (scalar - simd).abs() / scalar.abs().max(1e-6);
        assert!(rel < 1e-5, "scalar={scalar} simd={simd} rel={rel:e}");
    }

    #[test]
    #[ignore = "performance microbench"]
    fn encoder_f32_speedup_vs_f64() {
        use std::time::Instant;
        use crate::simd_encoder;
        let n_features = 372;
        let n_hidden = 128;
        let n_iters = 5000;
        let mut rng = Xs32::new(0x1234);
        let x32 = random_sparse_f32(&mut rng, n_features, 0.3);
        let w32 = random_vec_f32(&mut rng, n_features * n_hidden);
        let b32 = random_vec_f32(&mut rng, n_hidden);
        let x64: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        let w64: Vec<f64> = w32.iter().map(|&v| v as f64).collect();
        let b64: Vec<f64> = b32.iter().map(|&v| v as f64).collect();

        for _ in 0..200 { std::hint::black_box(encoder_forward_f32(&x32, &w32, &b32, n_features, n_hidden, 0.01)); }
        for _ in 0..200 { std::hint::black_box(simd_encoder::encoder_forward(&x64, &w64, &b64, n_features, n_hidden, 0.01)); }

        let t0 = Instant::now();
        for _ in 0..n_iters {
            std::hint::black_box(simd_encoder::encoder_forward(&x64, &w64, &b64, n_features, n_hidden, 0.01));
        }
        let f64_ns = t0.elapsed().as_nanos() as f64 / n_iters as f64;

        let t1 = Instant::now();
        for _ in 0..n_iters {
            std::hint::black_box(encoder_forward_f32(&x32, &w32, &b32, n_features, n_hidden, 0.01));
        }
        let f32_ns = t1.elapsed().as_nanos() as f64 / n_iters as f64;

        eprintln!("encoder_forward @ 372×128:\n  f64: {:.2} µs\n  f32: {:.2} µs\n  speedup: {:.2}×",
            f64_ns / 1e3, f32_ns / 1e3, f64_ns / f32_ns);
    }
}
