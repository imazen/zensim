//! SIMD-accelerated element-wise operations for SSIM computation.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
use archmage::magetypes;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[cfg(target_arch = "x86_64")]
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
    incant!(sq_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Compute sum of absolute differences: sum(|a[i] - b[i]|)
pub fn abs_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    incant!(abs_diff_sum_inner(a, b), [v4, v3, neon, wasm128, scalar])
}

/// Element-wise absolute difference: out[i] = |a[i] - b[i]|
pub fn abs_diff_into(a: &[f32], b: &[f32], out: &mut [f32]) {
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

/// Like ssim_channel but weights each pixel distance by mask[i] before accumulation.
/// Returns (sum_d, sum_d4, sum_d2) — mean, 4th-power, and 2nd-power pools.
pub fn ssim_channel_masked(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    incant!(
        ssim_channel_masked_inner(mu1, mu2, sum_sq, sigma12, mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused 2-mask variant of [`ssim_channel_masked`]. Computes the
/// unweighted SSIM distance once per pixel and accumulates with two
/// different masks into two separate `(sum_d, sum_d4, sum_d2)`
/// tuples in a single pass.
///
/// Used when both `extended_features` (masked) and `compute_iw_features`
/// (IW) are enabled — saves ~one full denominator + division load per
/// pixel by sharing the SSIM map across two pools.
///
/// Returns `((sum_d_a, sum_d4_a, sum_d2_a), (sum_d_b, sum_d4_b, sum_d2_b))`
/// where `_a` corresponds to `mask_a` and `_b` to `mask_b`.
#[inline]
pub fn ssim_channel_masked_2(
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    sigma12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    incant!(
        ssim_channel_masked_2_inner(mu1, mu2, sum_sq, sigma12, mask_a, mask_b),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Like edge_diff_channel but weights each pixel distance by mask[i].
/// Returns (art_mean, art_4th, det_mean, det_4th, art_2nd, det_2nd).
pub fn edge_diff_channel_masked(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    incant!(
        edge_diff_masked_inner(img1, img2, mu1, mu2, mask),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused 2-mask variant of [`edge_diff_channel_masked`]. Computes the
/// raw edge response (`(1 + |img2 - mu2|) / (1 + |img1 - mu1|) - 1`)
/// once per pixel and accumulates with two different masks in a single
/// pass. Used in the masked + IW combined path.
///
/// Returns `((art4_a, det4_a), (art4_b, det4_b))` — only the 4th-power
/// pools are produced because the streaming caller only consumes
/// `art4`/`det4` from the masked/IW edge path (the mean and 2nd-pool
/// edge features come from the unmasked extended pass).
pub fn edge_diff_channel_masked_2_art4_det4(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    incant!(
        edge_diff_masked_2_art4_det4_inner(img1, img2, mu1, mu2, mask_a, mask_b),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Fused weight-build + MSE-pool for the masked + IW combined path.
/// Given the blurred-activity buffer, builds **both** the masked weight
/// (`1 / (1 + k_mask * a)`) and the IW weight (`1 + k_iw * a`) into
/// `mask_out` / `iw_out`, and accumulates `Σ (src-dst)² · mask` for
/// **both** masks in the same pass.
///
/// Replaces 4 separate scalar loops in `process_strip`:
/// 1. masked weight construction
/// 2. iw weight construction
/// 3. masked MSE accumulator
/// 4. iw MSE accumulator
///
/// Returns `(masked_mse_sum, iw_mse_sum)`.
pub fn build_weights_and_mse(
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    incant!(
        build_weights_and_mse_inner(activity, k_mask, k_iw, src, dst, mask_out, iw_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Build only the masked weight (`1 / (1 + k_mask * a)`) and accumulate
/// `Σ (src-dst)² · mask`. Used when only `extended_features` is on.
pub fn build_mask_weight_and_mse(
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    incant!(
        build_mask_weight_and_mse_inner(activity, k_mask, src, dst, mask_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

/// Build only the IW weight (`1 + k_iw * a`) and accumulate
/// `Σ (src-dst)² · w_iw`. Used when only `compute_iw_features` is on.
pub fn build_iw_weight_and_mse(
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    incant!(
        build_iw_weight_and_mse_inner(activity, k_iw, src, dst, iw_out),
        [v4, v3, neon, wasm128, scalar]
    )
}

// --- SIMD implementations ---

/// SSIM stability constant for the structure/contrast term.
/// Same value as ssimulacra2. There is no C1 — the luminance term
/// uses `1 - (mu1-mu2)²` without a denominator (see metric.rs docs).
const C2: f32 = 0.0009;

#[cfg(target_arch = "x86_64")]
#[arcane]
fn sq_sum_into_inner_v4(token: archmage::X64V4Token, a: &[f32], b: &[f32], out: &mut [f32]) {
    let n = a.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        let va = f32x16::from_array(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&va.mul_add(va, vb * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(v3, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(token, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, *ac);
        let vb = f32x8::from_array(token, *bc);
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
        let va = f32x16::from_array(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array(token, b[base..][..16].try_into().unwrap());
        let d = va - vb;
        sum += (d * d).reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(v3, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(token, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, *ac);
        let vb = f32x8::from_array(token, *bc);
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
        let va = f32x16::from_array(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array(token, b[base..][..16].try_into().unwrap());
        let d = (va - vb).abs();
        sum += d.reduce_add() as f64;
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(v3, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(token, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, *ac);
        let vb = f32x8::from_array(token, *bc);
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
        let va = f32x16::from_array(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va * vb).to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(v3, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(token, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, *ac);
        let vb = f32x8::from_array(token, *bc);
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
        let va = f32x16::from_array(token, a[base..][..16].try_into().unwrap());
        let vb = f32x16::from_array(token, b[base..][..16].try_into().unwrap());
        out[base..base + 16].copy_from_slice(&(va - vb).abs().to_array());
    }
    let v3 = token.v3();
    let chunks8 = (n - chunks * 16) / 8;
    for c in 0..chunks8 {
        let base = chunks * 16 + c * 8;
        let va = f32x8::from_array(v3, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(v3, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, a[base..][..8].try_into().unwrap());
        let vb = f32x8::from_array(token, b[base..][..8].try_into().unwrap());
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
        let va = f32x8::from_array(token, *ac);
        let vb = f32x8::from_array(token, *bc);
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
fn ssim_channel_masked_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let c2v = f32x16::splat(token, C2);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let zero = f32x16::zero(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array(token, s12[base..][..16].try_into().unwrap());
        let mv = f32x16::from_array(token, mask[base..][..16].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = ((one - (num_m * num_s) / denom_s) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d = ((1.0f32 - (num_m * num_s) / denom_s) * mask[i]).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array(token, s12[base..][..8].try_into().unwrap());
        let mv = f32x8::from_array(token, mask[base..][..8].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = ((one - (num_m * num_s) / denom_s) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d = ((1.0f32 - (num_m * num_s) / denom_s) * mask[i]).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_masked_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();
    let (mask_chunks, _) = mask.as_chunks::<8>();

    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for ((((m1c, m2c), ssqc), s12c), mc) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(mask_chunks)
    {
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);
        let ssq = f32x8::from_array(token, *ssqc);
        let s12v = f32x8::from_array(token, *s12c);
        let mv = f32x8::from_array(token, *mc);

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = ((one - (num_m * num_s) / denom_s) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let mu_diff = m1v - mu2[j];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1v).mul_add(mu2[j], s12[j]), C2);
        let denom_s = (-mu2[j]).mul_add(mu2[j], (-m1v).mul_add(m1v, sum_sq[j])) + C2;
        let d = ((1.0f32 - (num_m * num_s) / denom_s) * mask[j]).max(0.0f32);
        let d2 = d * d;
        sum_d += d as f64;
        sum_d2 += d2 as f64;
        sum_d4 += (d2 * d2) as f64;
    }

    (sum_d, sum_d4, sum_d2)
}

// === ssim_channel_masked_2: fused 2-mask variant ===
// Computes unweighted SSIM distance once, applies two masks, accumulates
// into two separate (sum_d, sum_d4, sum_d2) tuples in one pass.

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_2_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let c2v = f32x16::splat(token, C2);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let zero = f32x16::zero(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array(token, s12[base..][..16].try_into().unwrap());
        let mva = f32x16::from_array(token, mask_a[base..][..16].try_into().unwrap());
        let mvb = f32x16::from_array(token, mask_b[base..][..16].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        // Shared unweighted distance ∈ [0, 2]
        let d_raw = one - (num_m * num_s) / denom_s;

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d_raw = 1.0f32 - (num_m * num_s) / denom_s;
        let da = (d_raw * mask_a[i]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[i]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_masked_2_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array(token, s12[base..][..8].try_into().unwrap());
        let mva = f32x8::from_array(token, mask_a[base..][..8].try_into().unwrap());
        let mvb = f32x8::from_array(token, mask_b[base..][..8].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d_raw = one - (num_m * num_s) / denom_s;

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d_raw = 1.0f32 - (num_m * num_s) / denom_s;
        let da = (d_raw * mask_a[i]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[i]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

#[magetypes(neon, wasm128, scalar)]
fn ssim_channel_masked_2_inner(
    token: Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64, f64), (f64, f64, f64)) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let (mu1_chunks, mu1_tail) = mu1.as_chunks::<8>();
    let (mu2_chunks, _) = mu2.as_chunks::<8>();
    let (ssq_chunks, _) = sum_sq.as_chunks::<8>();
    let (s12_chunks, _) = s12.as_chunks::<8>();
    let (mask_a_chunks, _) = mask_a.as_chunks::<8>();
    let (mask_b_chunks, _) = mask_b.as_chunks::<8>();

    let mut sum_da = 0.0f64;
    let mut sum_d4a = 0.0f64;
    let mut sum_d2a = 0.0f64;
    let mut sum_db = 0.0f64;
    let mut sum_d4b = 0.0f64;
    let mut sum_d2b = 0.0f64;

    for (((((m1c, m2c), ssqc), s12c), mac), mbc) in mu1_chunks
        .iter()
        .zip(mu2_chunks)
        .zip(ssq_chunks)
        .zip(s12_chunks)
        .zip(mask_a_chunks)
        .zip(mask_b_chunks)
    {
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);
        let ssq = f32x8::from_array(token, *ssqc);
        let s12v = f32x8::from_array(token, *s12c);
        let mva = f32x8::from_array(token, *mac);
        let mvb = f32x8::from_array(token, *mbc);

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d_raw = one - (num_m * num_s) / denom_s;

        let da = (d_raw * mva).max(zero);
        let d2a = da * da;
        let d4a = d2a * d2a;
        let db = (d_raw * mvb).max(zero);
        let d2b = db * db;
        let d4b = d2b * d2b;

        sum_da += da.reduce_add() as f64;
        sum_d2a += d2a.reduce_add() as f64;
        sum_d4a += d4a.reduce_add() as f64;
        sum_db += db.reduce_add() as f64;
        sum_d2b += d2b.reduce_add() as f64;
        sum_d4b += d4b.reduce_add() as f64;
    }

    let off = mu1_chunks.len() * 8;
    for (i, &m1v) in mu1_tail.iter().enumerate() {
        let j = off + i;
        let mu_diff = m1v - mu2[j];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1v).mul_add(mu2[j], s12[j]), C2);
        let denom_s = (-mu2[j]).mul_add(mu2[j], (-m1v).mul_add(m1v, sum_sq[j])) + C2;
        let d_raw = 1.0f32 - (num_m * num_s) / denom_s;
        let da = (d_raw * mask_a[j]).max(0.0f32);
        let d2a = da * da;
        let db = (d_raw * mask_b[j]).max(0.0f32);
        let d2b = db * db;
        sum_da += da as f64;
        sum_d2a += d2a as f64;
        sum_d4a += (d2a * d2a) as f64;
        sum_db += db as f64;
        sum_d2b += d2b as f64;
        sum_d4b += (d2b * d2b) as f64;
    }

    ((sum_da, sum_d4a, sum_d2a), (sum_db, sum_d4b, sum_d2b))
}

// === edge_diff_channel_masked: returns (art, art4, det, det4, art2, det2) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let mv = f32x16::from_array(token, mask[base..][..16].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[i];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let mv = f32x8::from_array(token, mask[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[i];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_masked_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();
    let (mask_chunks, _) = mask.as_chunks::<8>();

    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for ((((i1c, i2c), m1c), m2c), mc) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(mask_chunks)
    {
        let i1 = f32x8::from_array(token, *i1c);
        let i2 = f32x8::from_array(token, *i2c);
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);
        let mv = f32x8::from_array(token, *mc);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d1 = ((one + diff2) / (one + diff1) - one) * mv;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;

        sum_art += artifact.reduce_add() as f64;
        sum_art2 += a2.reduce_add() as f64;
        sum_art4 += (a2 * a2).reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det2 += dl2.reduce_add() as f64;
        sum_det4 += (dl2 * dl2).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d1 = ((1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32) * mask[j];

        let artifact = d1.max(0.0f32);
        let detail_lost = (-d1).max(0.0f32);
        let a2 = artifact * artifact;
        let dl2 = detail_lost * detail_lost;
        sum_art += artifact as f64;
        sum_art2 += a2 as f64;
        sum_art4 += (a2 * a2) as f64;
        sum_det += detail_lost as f64;
        sum_det2 += dl2 as f64;
        sum_det4 += (dl2 * dl2) as f64;
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

// === edge_diff_channel_masked_2_art4_det4: fused 2-mask variant ===
// Computes the unweighted edge response once, applies two masks, returns
// only the 4th-power pools (the only ones consumed by the streaming caller
// on the masked + IW paths). Returns ((art4_a, det4_a), (art4_b, det4_b)).

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_2_art4_det4_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let i1 = f32x16::from_array(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let mva = f32x16::from_array(token, mask_a[base..][..16].try_into().unwrap());
        let mvb = f32x16::from_array(token, mask_b[base..][..16].try_into().unwrap());

        // Shared unweighted edge response ∈ [..., ...]
        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[i];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[i];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_masked_2_art4_det4_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let mva = f32x8::from_array(token, mask_a[base..][..8].try_into().unwrap());
        let mvb = f32x8::from_array(token, mask_b[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[i];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[i];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

#[magetypes(neon, wasm128, scalar)]
fn edge_diff_masked_2_art4_det4_inner(
    token: Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask_a: &[f32],
    mask_b: &[f32],
) -> ((f64, f64), (f64, f64)) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let (i1_chunks, i1_tail) = img1.as_chunks::<8>();
    let (i2_chunks, _) = img2.as_chunks::<8>();
    let (m1_chunks, _) = mu1.as_chunks::<8>();
    let (m2_chunks, _) = mu2.as_chunks::<8>();
    let (ma_chunks, _) = mask_a.as_chunks::<8>();
    let (mb_chunks, _) = mask_b.as_chunks::<8>();

    let mut sum_art4a = 0.0f64;
    let mut sum_det4a = 0.0f64;
    let mut sum_art4b = 0.0f64;
    let mut sum_det4b = 0.0f64;

    for (((((i1c, i2c), m1c), m2c), mac), mbc) in i1_chunks
        .iter()
        .zip(i2_chunks)
        .zip(m1_chunks)
        .zip(m2_chunks)
        .zip(ma_chunks)
        .zip(mb_chunks)
    {
        let i1 = f32x8::from_array(token, *i1c);
        let i2 = f32x8::from_array(token, *i2c);
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);
        let mva = f32x8::from_array(token, *mac);
        let mvb = f32x8::from_array(token, *mbc);

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();
        let d_raw = (one + diff2) / (one + diff1) - one;

        let da = d_raw * mva;
        let artifact_a = da.max(zero);
        let detail_a = (-da).max(zero);
        let a2a = artifact_a * artifact_a;
        let dl2a = detail_a * detail_a;
        sum_art4a += (a2a * a2a).reduce_add() as f64;
        sum_det4a += (dl2a * dl2a).reduce_add() as f64;

        let db = d_raw * mvb;
        let artifact_b = db.max(zero);
        let detail_b = (-db).max(zero);
        let a2b = artifact_b * artifact_b;
        let dl2b = detail_b * detail_b;
        sum_art4b += (a2b * a2b).reduce_add() as f64;
        sum_det4b += (dl2b * dl2b).reduce_add() as f64;
    }

    let off = i1_chunks.len() * 8;
    for (i, _) in i1_tail.iter().enumerate() {
        let j = off + i;
        let diff1 = (img1[j] - mu1[j]).abs();
        let diff2 = (img2[j] - mu2[j]).abs();
        let d_raw = (1.0f32 + diff2) / (1.0f32 + diff1) - 1.0f32;

        let da = d_raw * mask_a[j];
        let aa = da.max(0.0f32);
        let dla = (-da).max(0.0f32);
        let a2a = aa * aa;
        let dl2a = dla * dla;
        sum_art4a += (a2a * a2a) as f64;
        sum_det4a += (dl2a * dl2a) as f64;

        let db = d_raw * mask_b[j];
        let ab = db.max(0.0f32);
        let dlb = (-db).max(0.0f32);
        let a2b = ab * ab;
        let dl2b = dlb * dlb;
        sum_art4b += (a2b * a2b) as f64;
        sum_det4b += (dl2b * dl2b) as f64;
    }

    ((sum_art4a, sum_det4a), (sum_art4b, sum_det4b))
}

// === build_weights_and_mse: fused weight-build + MSE ===
// In one SIMD pass, given the blurred activity buffer:
// - writes mask_out[i] = 1 / (1 + k_mask * activity[i])
// - writes iw_out[i]   = 1 + k_iw   * activity[i]
// - accumulates Σ (src-dst)² · mask  and  Σ (src-dst)² · iw_weight
// Returns (masked_mse_sum, iw_mse_sum).

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_weights_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    let one = f32x16::splat(token, 1.0);
    let kmv = f32x16::splat(token, k_mask);
    let kiv = f32x16::splat(token, k_iw);

    let n = activity.len();
    let chunks = n / 16;
    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());

        // mask = 1 / (1 + k_mask * a)  → use rcp via division (denominator never zero
        //   because k_mask ≥ 0, a ≥ 0, +1 floor)
        let mask = one / kmv.mul_add(av, one);
        // iw_weight = 1 + k_iw * a
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask_out[base..base + 16].copy_from_slice(&mask.to_array());
        iw_out[base..base + 16].copy_from_slice(&iw.to_array());
    }

    for i in (chunks * 16)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        mask_out[i] = mask;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_weights_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    let one = f32x8::splat(token, 1.0);
    let kmv = f32x8::splat(token, k_mask);
    let kiv = f32x8::splat(token, k_iw);

    let n = activity.len();
    let chunks = n / 8;
    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());

        let mask = one / kmv.mul_add(av, one);
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask_out[base..base + 8].copy_from_slice(&mask.to_array());
        iw_out[base..base + 8].copy_from_slice(&iw.to_array());
    }

    for i in (chunks * 8)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        mask_out[i] = mask;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

#[magetypes(neon, wasm128, scalar)]
fn build_weights_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
    iw_out: &mut [f32],
) -> (f64, f64) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat(token, 1.0);
    let kmv = f32x8::splat(token, k_mask);
    let kiv = f32x8::splat(token, k_iw);

    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (mo_chunks, mo_tail) = mask_out.as_chunks_mut::<8>();
    let (io_chunks, io_tail) = iw_out.as_chunks_mut::<8>();

    let mut mse_mask = 0.0f64;
    let mut mse_iw = 0.0f64;

    for ((((ac, sc), dc), mo), io) in a_chunks
        .iter()
        .zip(s_chunks)
        .zip(d_chunks)
        .zip(mo_chunks)
        .zip(io_chunks)
    {
        let av = f32x8::from_array(token, *ac);
        let sv = f32x8::from_array(token, *sc);
        let dv = f32x8::from_array(token, *dc);

        let mask = one / kmv.mul_add(av, one);
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse_mask += (d2 * mask).reduce_add() as f64;
        mse_iw += (d2 * iw).reduce_add() as f64;
        mask.store(mo);
        iw.store(io);
    }

    let off = a_chunks.len() * 8;
    for ((i, &a), (mo, io)) in a_tail
        .iter()
        .enumerate()
        .zip(mo_tail.iter_mut().zip(io_tail.iter_mut()))
    {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        let iw = 1.0f32 + k_iw * a;
        *mo = mask;
        *io = iw;
        let d = src[j] - dst[j];
        let d2 = d * d;
        mse_mask += (d2 * mask) as f64;
        mse_iw += (d2 * iw) as f64;
    }

    (mse_mask, mse_iw)
}

// === build_mask_weight_and_mse: single-mask variant ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_mask_weight_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    let one = f32x16::splat(token, 1.0);
    let kmv = f32x16::splat(token, k_mask);
    let n = activity.len();
    let chunks = n / 16;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());
        let mask = one / kmv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask_out[base..base + 16].copy_from_slice(&mask.to_array());
    }
    for i in (chunks * 16)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        mask_out[i] = mask;
        let d = src[i] - dst[i];
        mse += (d * d * mask) as f64;
    }
    mse
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_mask_weight_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    let one = f32x8::splat(token, 1.0);
    let kmv = f32x8::splat(token, k_mask);
    let n = activity.len();
    let chunks = n / 8;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());
        let mask = one / kmv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask_out[base..base + 8].copy_from_slice(&mask.to_array());
    }
    for i in (chunks * 8)..n {
        let a = activity[i];
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        mask_out[i] = mask;
        let d = src[i] - dst[i];
        mse += (d * d * mask) as f64;
    }
    mse
}

#[magetypes(neon, wasm128, scalar)]
fn build_mask_weight_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_mask: f32,
    src: &[f32],
    dst: &[f32],
    mask_out: &mut [f32],
) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat(token, 1.0);
    let kmv = f32x8::splat(token, k_mask);
    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (mo_chunks, mo_tail) = mask_out.as_chunks_mut::<8>();
    let mut mse = 0.0f64;
    for (((ac, sc), dc), mo) in a_chunks.iter().zip(s_chunks).zip(d_chunks).zip(mo_chunks) {
        let av = f32x8::from_array(token, *ac);
        let sv = f32x8::from_array(token, *sc);
        let dv = f32x8::from_array(token, *dc);
        let mask = one / kmv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * mask).reduce_add() as f64;
        mask.store(mo);
    }
    let off = a_chunks.len() * 8;
    for ((i, &a), mo) in a_tail.iter().enumerate().zip(mo_tail.iter_mut()) {
        let j = off + i;
        let mask = 1.0f32 / (1.0f32 + k_mask * a);
        *mo = mask;
        let d = src[j] - dst[j];
        mse += (d * d * mask) as f64;
    }
    mse
}

// === build_iw_weight_and_mse: single-IW variant ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_iw_weight_and_mse_inner_v4(
    token: archmage::X64V4Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    let one = f32x16::splat(token, 1.0);
    let kiv = f32x16::splat(token, k_iw);
    let n = activity.len();
    let chunks = n / 16;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 16;
        let av = f32x16::from_array(token, activity[base..][..16].try_into().unwrap());
        let sv = f32x16::from_array(token, src[base..][..16].try_into().unwrap());
        let dv = f32x16::from_array(token, dst[base..][..16].try_into().unwrap());
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw_out[base..base + 16].copy_from_slice(&iw.to_array());
    }
    for i in (chunks * 16)..n {
        let a = activity[i];
        let iw = 1.0f32 + k_iw * a;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        mse += (d * d * iw) as f64;
    }
    mse
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn build_iw_weight_and_mse_inner_v3(
    token: archmage::X64V3Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    let one = f32x8::splat(token, 1.0);
    let kiv = f32x8::splat(token, k_iw);
    let n = activity.len();
    let chunks = n / 8;
    let mut mse = 0.0f64;
    for c in 0..chunks {
        let base = c * 8;
        let av = f32x8::from_array(token, activity[base..][..8].try_into().unwrap());
        let sv = f32x8::from_array(token, src[base..][..8].try_into().unwrap());
        let dv = f32x8::from_array(token, dst[base..][..8].try_into().unwrap());
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw_out[base..base + 8].copy_from_slice(&iw.to_array());
    }
    for i in (chunks * 8)..n {
        let a = activity[i];
        let iw = 1.0f32 + k_iw * a;
        iw_out[i] = iw;
        let d = src[i] - dst[i];
        mse += (d * d * iw) as f64;
    }
    mse
}

#[magetypes(neon, wasm128, scalar)]
fn build_iw_weight_and_mse_inner(
    token: Token,
    activity: &[f32],
    k_iw: f32,
    src: &[f32],
    dst: &[f32],
    iw_out: &mut [f32],
) -> f64 {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let one = f32x8::splat(token, 1.0);
    let kiv = f32x8::splat(token, k_iw);
    let (a_chunks, a_tail) = activity.as_chunks::<8>();
    let (s_chunks, _) = src.as_chunks::<8>();
    let (d_chunks, _) = dst.as_chunks::<8>();
    let (io_chunks, io_tail) = iw_out.as_chunks_mut::<8>();
    let mut mse = 0.0f64;
    for (((ac, sc), dc), io) in a_chunks.iter().zip(s_chunks).zip(d_chunks).zip(io_chunks) {
        let av = f32x8::from_array(token, *ac);
        let sv = f32x8::from_array(token, *sc);
        let dv = f32x8::from_array(token, *dc);
        let iw = kiv.mul_add(av, one);
        let diff = sv - dv;
        let d2 = diff * diff;
        mse += (d2 * iw).reduce_add() as f64;
        iw.store(io);
    }
    let off = a_chunks.len() * 8;
    for ((i, &a), io) in a_tail.iter().enumerate().zip(io_tail.iter_mut()) {
        let j = off + i;
        let iw = 1.0f32 + k_iw * a;
        *io = iw;
        let d = src[j] - dst[j];
        mse += (d * d * iw) as f64;
    }
    mse
}

// === ssim_channel_extended: returns (sum_d, sum_d4, sum_d2, sum_d8, max_d) ===

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_extended_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64, f64, f64, f32) {
    let c2v = f32x16::splat(token, C2);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let zero = f32x16::zero(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array(token, s12[base..][..16].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = (one - (num_m * num_s) / denom_s).max(zero);
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
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
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
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;
    let mut sum_d8 = 0.0f64;
    let mut max_d_vec = zero;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array(token, s12[base..][..8].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = (one - (num_m * num_s) / denom_s).max(zero);
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
        let mu_diff = mu1[i] - mu2[i];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-mu1[i]).mul_add(mu2[i], s12[i]), C2);
        let denom_s = (-mu2[i]).mul_add(mu2[i], (-mu1[i]).mul_add(mu1[i], sum_sq[i])) + C2;
        let d = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
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
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

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
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);
        let ssq = f32x8::from_array(token, *ssqc);
        let s12v = f32x8::from_array(token, *s12c);

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add((-m1).mul_add(m2, s12v), c2v);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + c2v;
        let d = (one - (num_m * num_s) / denom_s).max(zero);
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
        let mu_diff = m1v - mu2[j];
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1v).mul_add(mu2[j], s12[j]), C2);
        let denom_s = (-mu2[j]).mul_add(mu2[j], (-m1v).mul_add(m1v, sum_sq[j])) + C2;
        let d = (1.0f32 - (num_m * num_s) / denom_s).max(0.0f32);
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
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

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
        let i1 = f32x16::from_array(token, img1[base..][..16].try_into().unwrap());
        let i2 = f32x16::from_array(token, img2[base..][..16].try_into().unwrap());
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());

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
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

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
        let i1 = f32x8::from_array(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());

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
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

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
        let i1 = f32x8::from_array(token, *i1c);
        let i2 = f32x8::from_array(token, *i2c);
        let m1 = f32x8::from_array(token, *m1c);
        let m2 = f32x8::from_array(token, *m2c);

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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_test_data(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut mu1 = Vec::with_capacity(n);
        let mut mu2 = Vec::with_capacity(n);
        let mut sum_sq = Vec::with_capacity(n);
        let mut s12 = Vec::with_capacity(n);
        let mut mask_a = Vec::with_capacity(n);
        let mut mask_b = Vec::with_capacity(n);
        for i in 0..n {
            let x = (i as f32) / (n as f32);
            mu1.push(0.5 + 0.1 * x);
            mu2.push(0.5 + 0.1 * x + 0.01 * (x - 0.5));
            sum_sq.push(0.25 + 0.02 * x);
            s12.push(0.24 + 0.02 * x);
            mask_a.push(0.5 + 0.5 * x); // smooth mask in [0.5, 1.0]
            mask_b.push(1.0 + 2.0 * x); // IW-style weight in [1.0, 3.0]
        }
        (mu1, mu2, sum_sq, s12, mask_a, mask_b)
    }

    #[test]
    fn ssim_channel_masked_2_matches_two_single_calls() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let (mu1, mu2, sum_sq, s12, mask_a, mask_b) = mk_test_data(n);
            let (da_ref, d4a_ref, d2a_ref) =
                ssim_channel_masked(&mu1, &mu2, &sum_sq, &s12, &mask_a);
            let (db_ref, d4b_ref, d2b_ref) =
                ssim_channel_masked(&mu1, &mu2, &sum_sq, &s12, &mask_b);
            let ((da, d4a, d2a), (db, d4b, d2b)) =
                ssim_channel_masked_2(&mu1, &mu2, &sum_sq, &s12, &mask_a, &mask_b);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            assert!(
                rel(da, da_ref) < 1e-5,
                "n={}: da {} vs ref {}",
                n,
                da,
                da_ref
            );
            assert!(
                rel(d4a, d4a_ref) < 1e-5,
                "n={}: d4a {} vs ref {}",
                n,
                d4a,
                d4a_ref
            );
            assert!(
                rel(d2a, d2a_ref) < 1e-5,
                "n={}: d2a {} vs ref {}",
                n,
                d2a,
                d2a_ref
            );
            assert!(
                rel(db, db_ref) < 1e-5,
                "n={}: db {} vs ref {}",
                n,
                db,
                db_ref
            );
            assert!(
                rel(d4b, d4b_ref) < 1e-5,
                "n={}: d4b {} vs ref {}",
                n,
                d4b,
                d4b_ref
            );
            assert!(
                rel(d2b, d2b_ref) < 1e-5,
                "n={}: d2b {} vs ref {}",
                n,
                d2b,
                d2b_ref
            );
        }
    }

    #[test]
    fn edge_diff_channel_masked_2_matches_two_single_calls() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut img1 = Vec::with_capacity(n);
            let mut img2 = Vec::with_capacity(n);
            let mut mu1 = Vec::with_capacity(n);
            let mut mu2 = Vec::with_capacity(n);
            let mut mask_a = Vec::with_capacity(n);
            let mut mask_b = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                img1.push(0.5 + 0.05 * (x * 7.0).sin());
                img2.push(0.5 + 0.05 * (x * 7.0).sin() + 0.01 * x);
                mu1.push(0.5 + 0.1 * x);
                mu2.push(0.5 + 0.1 * x + 0.01 * (x - 0.5));
                mask_a.push(0.5 + 0.5 * x);
                mask_b.push(1.0 + 2.0 * x);
            }
            let (_, art4_a_ref, _, det4_a_ref, _, _) =
                edge_diff_channel_masked(&img1, &img2, &mu1, &mu2, &mask_a);
            let (_, art4_b_ref, _, det4_b_ref, _, _) =
                edge_diff_channel_masked(&img1, &img2, &mu1, &mu2, &mask_b);
            let ((art4_a, det4_a), (art4_b, det4_b)) =
                edge_diff_channel_masked_2_art4_det4(&img1, &img2, &mu1, &mu2, &mask_a, &mask_b);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            assert!(
                rel(art4_a, art4_a_ref) < 1e-5,
                "n={}: art4_a {} vs ref {}",
                n,
                art4_a,
                art4_a_ref
            );
            assert!(
                rel(det4_a, det4_a_ref) < 1e-5,
                "n={}: det4_a {} vs ref {}",
                n,
                det4_a,
                det4_a_ref
            );
            assert!(
                rel(art4_b, art4_b_ref) < 1e-5,
                "n={}: art4_b {} vs ref {}",
                n,
                art4_b,
                art4_b_ref
            );
            assert!(
                rel(det4_b, det4_b_ref) < 1e-5,
                "n={}: det4_b {} vs ref {}",
                n,
                det4_b,
                det4_b_ref
            );
        }
    }

    #[test]
    fn build_weights_and_mse_matches_scalar() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut activity = Vec::with_capacity(n);
            let mut src = Vec::with_capacity(n);
            let mut dst = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                activity.push(x * 0.5);
                src.push(0.5 + 0.05 * x);
                dst.push(0.5 + 0.05 * x + 0.01 * (x - 0.5));
            }
            let k_mask = 4.0f32;
            let k_iw = 4.0f32;
            // Scalar reference
            let mut mask_ref = vec![0f32; n];
            let mut iw_ref = vec![0f32; n];
            let mut mse_mask_ref = 0.0f64;
            let mut mse_iw_ref = 0.0f64;
            for i in 0..n {
                let a = activity[i];
                mask_ref[i] = 1.0 / (1.0 + k_mask * a);
                iw_ref[i] = 1.0 + k_iw * a;
                let d = src[i] - dst[i];
                mse_mask_ref += (d * d * mask_ref[i]) as f64;
                mse_iw_ref += (d * d * iw_ref[i]) as f64;
            }
            // SIMD path
            let mut mask_out = vec![0f32; n];
            let mut iw_out = vec![0f32; n];
            let (mse_mask, mse_iw) = build_weights_and_mse(
                &activity,
                k_mask,
                k_iw,
                &src,
                &dst,
                &mut mask_out,
                &mut iw_out,
            );
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            for i in 0..n {
                assert!(
                    (mask_out[i] - mask_ref[i]).abs() < 1e-6,
                    "n={}: mask[{}] {} vs ref {}",
                    n,
                    i,
                    mask_out[i],
                    mask_ref[i]
                );
                assert!(
                    (iw_out[i] - iw_ref[i]).abs() < 1e-6,
                    "n={}: iw[{}] {} vs ref {}",
                    n,
                    i,
                    iw_out[i],
                    iw_ref[i]
                );
            }
            assert!(
                rel(mse_mask, mse_mask_ref) < 1e-5,
                "n={}: mse_mask {} vs ref {}",
                n,
                mse_mask,
                mse_mask_ref
            );
            assert!(
                rel(mse_iw, mse_iw_ref) < 1e-5,
                "n={}: mse_iw {} vs ref {}",
                n,
                mse_iw,
                mse_iw_ref
            );
        }
    }

    #[test]
    fn build_iw_weight_and_mse_matches_scalar() {
        for &n in &[16usize, 17, 32, 100, 256, 1024] {
            let mut activity = Vec::with_capacity(n);
            let mut src = Vec::with_capacity(n);
            let mut dst = Vec::with_capacity(n);
            for i in 0..n {
                let x = (i as f32) / (n as f32);
                activity.push(x * 0.5);
                src.push(0.5 + 0.05 * x);
                dst.push(0.5 + 0.05 * x + 0.01 * (x - 0.5));
            }
            let k_iw = 4.0f32;
            // Scalar reference
            let mut iw_ref = vec![0f32; n];
            let mut mse_iw_ref = 0.0f64;
            for i in 0..n {
                let a = activity[i];
                iw_ref[i] = 1.0 + k_iw * a;
                let d = src[i] - dst[i];
                mse_iw_ref += (d * d * iw_ref[i]) as f64;
            }
            let mut iw_out = vec![0f32; n];
            let mse_iw = build_iw_weight_and_mse(&activity, k_iw, &src, &dst, &mut iw_out);
            let rel = |a: f64, b: f64| (a - b).abs() / a.abs().max(1e-9);
            for i in 0..n {
                assert!(
                    (iw_out[i] - iw_ref[i]).abs() < 1e-6,
                    "n={}: iw[{}] {} vs ref {}",
                    n,
                    i,
                    iw_out[i],
                    iw_ref[i]
                );
            }
            assert!(
                rel(mse_iw, mse_iw_ref) < 1e-5,
                "n={}: mse_iw {} vs ref {}",
                n,
                mse_iw,
                mse_iw_ref
            );
        }
    }
}
