//! SIMD-accelerated element-wise operations for SSIM computation.

use crate::ssim_form::{ssim_dissim_raw_scalar, ssim_dissim8, ssim_dissim16};
#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::f32x16;

/// Element-wise multiply: out[i] = a[i] * b[i]
pub fn mul_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(mul_into_inner(a, b, out), [v4, v3, neon, wasm128, scalar]);
}

/// Element-wise: out[i] = a[i]*a[i] + b[i]*b[i] (sum of squares)
pub fn sq_sum_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(
        sq_sum_into_inner(a, b, out),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// Compute sum of squared differences: sum((a[i] - b[i])²)
pub fn sq_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    // aarch64: use the SCALAR variant. On AArch64, NEON is baseline, so LLVM
    // autovectorises the scalar body anyway — the hand-written NEON variant is
    // competing with the autovectoriser, not with scalar code, and it LOSES.
    // Measured on Apple M4 Pro at 1024x1024 (zensim-bench/benches/stage_isolation.rs,
    // within-group A/B): NEON 0.351ms vs scalar 0.117ms — the NEON variant is 3x SLOWER.
    // Verified BIT-IDENTICAL to the NEON variant before switching (differing
    // lanes = 0), so this is a pure speed change with no effect on any score.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if archmage::NeonToken::summon().is_some() {
            return sq_diff_sum_inner_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                a,
                b,
            );
        }
    }
    incant!(sq_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Compute sum of absolute differences: sum(|a[i] - b[i]|)
pub fn abs_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    // aarch64: use the SCALAR variant. On AArch64, NEON is baseline, so LLVM
    // autovectorises the scalar body anyway — the hand-written NEON variant is
    // competing with the autovectoriser, not with scalar code, and it LOSES.
    // Measured on Apple M4 Pro at 1024x1024 (zensim-bench/benches/stage_isolation.rs,
    // within-group A/B): NEON 0.351ms vs scalar 0.087ms — the NEON variant is 4x SLOWER.
    // Verified BIT-IDENTICAL to the NEON variant before switching (differing
    // lanes = 0), so this is a pure speed change with no effect on any score.
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if archmage::NeonToken::summon().is_some() {
            return abs_diff_sum_inner_scalar(
                archmage::ScalarToken::summon().expect("infallible"),
                a,
                b,
            );
        }
    }
    incant!(abs_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Element-wise absolute difference: out[i] = |a[i] - b[i]|
///
/// **Deprecated for the streaming hot path (Phase 2 Lever 3, 2026-05-22)**
/// — the activity path uses [`crate::blur::box_blur_h_into_abs_diff`]
/// which fuses the blur with the abs-diff. Kept as a reference
/// implementation for the SIMD dispatch trio (v4/v3/scalar).
pub(crate) fn abs_diff_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(
        abs_diff_into_inner(a, b, out),
        [v4, v3, neon, wasm128, scalar]
    );
}

/// Like ssim_channel but also computes 8th-power pool and max.
/// Returns (sum_d, sum_d4, sum_d2, sum_d8, max_d).
/// d8 = d4*d4 (one extra multiply per pixel). L8 = (sum_d8/N)^(1/8).
pub fn ssim_channel_extended(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    incant!(
        ssim_channel_extended_inner(mu1, mu2, sum_sq, sigma12),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Like edge_diff_channel but also computes 8th-power pool and max for artifact/detail.
/// Returns (art_mean, art_4th, det_mean, det_4th, art_2nd, det_2nd, art_8th, det_8th, max_art, max_det).
pub fn edge_diff_channel_extended(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    incant!(
        edge_diff_extended_inner(img1, img2, mu1, mu2),
        [v4, v3, neon, wasm128, scalar]
    )
}

// --- SIMD implementations ---

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_sum_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&va.mul_add(va, vb * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&va.mul_add(va, vb * vb).to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = a[i].mul_add(a[i], b[i] * b[i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_sum_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&va.mul_add(va, vb * vb).to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = a[i].mul_add(a[i], b[i] * b[i]);
    }
}

#[magetypes(neon, wasm128, scalar)]
fn sq_sum_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        va.mul_add(va, vb * vb).store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = a.mul_add(a, b * b);
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_diff_sum_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 16;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        let d = a[i] - b[i];
        sum += (d * d) as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_diff_sum_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 8;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for i in (chunks * 8)..n {
        let d = a[i] - b[i];
        sum += (d * d) as f64;
    }
    sum
}

#[magetypes(neon, wasm128, scalar)]
fn sq_diff_sum_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let mut sum = 0.0f64;
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    for (&a, &b) in a_tail.iter().zip(b.iter().skip(a_chunks.len() * 8)) {
        let d = a - b;
        sum += (d * d) as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_sum_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 16;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        let d = (a[i] - b[i]).abs() as f64;
        sum += d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_sum_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32]) -> f64 {
    let n = a.len();
    let chunks = n / 8;
    let mut sum = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for i in (chunks * 8)..n {
        let d = (a[i] - b[i]).abs() as f64;
        sum += d;
    }
    sum
}

#[magetypes(neon, wasm128, scalar)]
fn abs_diff_sum_inner(token: Token, a: &[f32], b: &[f32]) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let mut sum = 0.0f64;
    for (ac, bc) in a_chunks.iter().zip(b_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    for (&a, &b) in a_tail.iter().zip(b.iter().skip(a_chunks.len() * 8)) {
        let d = (a - b).abs() as f64;
        sum += d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mul_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va * vb).to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn mul_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va * vb).to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = a[i] * b[i];
    }
}

#[magetypes(neon, wasm128, scalar)]
fn mul_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        (va * vb).store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = a * b;
    }
}

// === abs_diff_into ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array_t(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array_t(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va - vb).abs().to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array_t(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(v3, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va - vb).abs().to_array());
    }
    for i in (chunks * 16 + chunks8 * 8)..n {
        out[i] = (a[i] - b[i]).abs();
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn abs_diff_into_inner_v3(token: archmage::X64V3Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 8;
    for c in 0..chunks {
        let base = c * 8;
        let va = f32x8::from_array_t(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array_t(token, b[base..][..8].try_into().unwrap());
        out[base..base + 8].copy_from_slice(&(va - vb).abs().to_array());
    }
    for i in (chunks * 8)..n {
        out[i] = (a[i] - b[i]).abs();
    }
}

#[magetypes(neon, wasm128, scalar)]
fn abs_diff_into_inner(token: Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let (a_chunks, a_tail) = a.as_chunks::<8>();
    let (b_chunks, _) = b.as_chunks::<8>();
    let (out_chunks, out_tail) = out.as_chunks_mut::<8>();
    for ((ac, bc), oc) in a_chunks.iter().zip(b_chunks).zip(out_chunks) {
        let va = f32x8::from_array_t(token, *ac);
        let vb = f32x8::from_array_t(token, *bc);
        (va - vb).abs().store(oc);
    }
    for ((&a, &b), o) in a_tail
        .iter()
        .zip(b.iter().skip(a_chunks.len() * 8))
        .zip(out_tail)
    {
        *o = (a - b).abs();
    }
}

// === ssim_channel_masked: returns (sum_d, sum_d4, sum_d2) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_extended_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x16::zero_t(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array_t(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array_t(token, s12[base..][..16].try_into().unwrap());

        let d = (ssim_dissim16(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    for i in (chunks * 16)..n {
        let d = (ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_extended_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    let zero = f32x8::zero_t(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array_t(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array_t(token, s12[base..][..8].try_into().unwrap());

        let d = (ssim_dissim8(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    for i in (chunks * 8)..n {
        let d = (ssim_dissim_raw_scalar(form, mu1[i], mu2[i], sum_sq[i], s12[i])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_extended_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let form = crate::ssim_form::active_luma_form();
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let zero = f32x8::zero_t(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for (((m1c, m2c), ssqc), s12c) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
    {
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);
        let ssq = f32x8::from_array_t(token, *ssqc);
        let s12v = f32x8::from_array_t(token, *s12c);

        let d = (ssim_dissim8(token, form, m1, m2, ssq, s12v)).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;
        let d8 = d4 * d4;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d8 += d8.reduce_add() as f64;
        max_d_vec = max_d_vec.max(d);
    }

    let mut max_d = max_d_vec.reduce_max();
    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let d = (ssim_dissim_raw_scalar(form, m1v, mu2[j], sum_sq[j], s12[j])).max(0.0f32);
        let d2 = d * d;
        let d4 = d2 * d2;
        sum_d += d as f64;
        sum_d4 += d4 as f64;
        sum_d2 += d2 as f64;
        sum_d8 += (d4 * d4) as f64;
        max_d = max_d.max(d);
    }

    (sum_d, sum_d4, sum_d2, sum_d8, max_d)
}

// === edge_diff_channel_extended: returns (art, art4, det, det4, art2, det2, art8, det8, max_art, max_det) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_extended_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    let one = f32x16::splat_t(token, 1.0);
    let zero = f32x16::zero_t(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array_t(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array_t(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array_t(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array_t(token, mu2[base..][..16].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_extended_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array_t(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array_t(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array_t(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array_t(token, mu2[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_extended_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64, f64, f64, f64, f64, f32, f32) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat_t(token, 1.0);
    let zero = f32x8::zero_t(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_art8 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;
    let mut sum_det8 = 0.0f64;
    let mut max_art_vec = zero;
    let mut max_det_vec = zero;

    for (((i1c, i2c), m1c), m2c) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
    {
        let i1 = f32x8::from_array_t(token, *i1c);
        let i2 = f32x8::from_array_t(token, *i2c);
        let m1 = f32x8::from_array_t(token, *m1c);
        let m2 = f32x8::from_array_t(token, *m2c);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let a8 = a4 * a4;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        let dl8 = dl4 * dl4;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art8 += a8.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det8 += dl8.reduce_add() as f64;
        max_art_vec = max_art_vec.max(artifact);
        max_det_vec = max_det_vec.max(detail_lost);
    }

    let mut max_art = max_art_vec.reduce_max();
    let mut max_det = max_det_vec.reduce_max();

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d1 = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;
        sum_art += artifact as f64;
        sum_art4 += a4 as f64;
        sum_art2 += a2 as f64;
        sum_art8 += (a4 * a4) as f64;
        sum_det += detail_lost as f64;
        sum_det4 += dl4 as f64;
        sum_det2 += dl2 as f64;
        sum_det8 += (dl4 * dl4) as f64;
        max_art = max_art.max(artifact);
        max_det = max_det.max(detail_lost);
    }

    (
        sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2, sum_art8, sum_det8, max_art,
        max_det,
    )
}
