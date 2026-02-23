//! SIMD-accelerated element-wise operations for SSIM computation.

#[cfg(target_arch = "x86_64")]
use archmage::arcane;
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x16;

/// Element-wise multiply: out[i] = a[i] * b[i]
pub fn mul_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(mul_into_inner(a, b, out), [v4, v3]);
}

/// Element-wise: out[i] = a[i]*a[i] + b[i]*b[i] (sum of squares)
pub fn sq_sum_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(sq_sum_into_inner(a, b, out), [v4, v3]);
}

/// Compute SSIM distance map and return (sum_d, sum_d4) for a single channel.
/// d = max(0, 1 - (num_m * num_s) / denom_s)
/// where:
///   num_m = 1 - (mu1 - mu2)^2
///   num_s = 2*sigma12 - 2*mu1*mu2 + C2
///   denom_s = sum_sq - mu1^2 - mu2^2 + C2  (sum_sq = blur(src^2 + dst^2))
pub fn ssim_channel(mu1: &[f32], mu2: &[f32], sum_sq: &[f32], sigma12: &[f32]) -> (f64, f64) {
    incant!(ssim_channel_inner(mu1, mu2, sum_sq, sigma12), [v4, v3])
}

/// Compute sum of squared differences: sum((a[i] - b[i])²)
pub fn sq_diff_sum(a: &[f32], b: &[f32]) -> f64 {
    incant!(sq_diff_sum_inner(a, b), [v4, v3])
}

/// Element-wise absolute difference: out[i] = |a[i] - b[i]|
pub fn abs_diff_into(a: &[f32], b: &[f32], out: &mut [f32]) {
    incant!(abs_diff_into_inner(a, b, out), [v4, v3]);
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
        [v4, v3]
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
    incant!(edge_diff_masked_inner(img1, img2, mu1, mu2, mask), [v4, v3])
}

/// Compute edge difference features for a single channel.
/// Returns (artifact_mean, artifact_4th, detail_lost_mean, detail_lost_4th).
pub fn edge_diff_channel(
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64) {
    incant!(edge_diff_inner(img1, img2, mu1, mu2), [v4, v3])
}

// --- SIMD implementations ---

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
        out[i] = a[i] * a[i] + b[i] * b[i];
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
        out[i] = a[i] * a[i] + b[i] * b[i];
    }
}

fn sq_sum_into_inner_scalar(_token: archmage::ScalarToken, a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * a[i] + b[i] * b[i];
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
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
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
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
    }
    sum
}

fn sq_diff_sum_inner_scalar(_token: archmage::ScalarToken, a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..a.len() {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
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

fn mul_into_inner_scalar(_token: archmage::ScalarToken, a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_inner_v4(
    token: archmage::X64V4Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64) {
    let c2v = f32x16::splat(token, C2);
    let one = f32x16::splat(token, 1.0);
    let two = f32x16::splat(token, 2.0);
    let zero = f32x16::zero(token);

    let n = mu1.len();
    let chunks = n / 16;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;

    for c in 0..chunks {
        let base = c * 16;
        let m1 = f32x16::from_array(token, mu1[base..][..16].try_into().unwrap());
        let m2 = f32x16::from_array(token, mu2[base..][..16].try_into().unwrap());
        let ssq = f32x16::from_array(token, sum_sq[base..][..16].try_into().unwrap());
        let s12v = f32x16::from_array(token, s12[base..][..16].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add(s12v - m1 * m2, c2v);
        let denom_s = ssq - m1 * m1 - m2 * m2 + c2v;
        let d = (one - (num_m * num_s) / denom_s).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = (1.0 - (num_m * num_s) / denom_s).max(0.0);
        sum_d += d;
        sum_d4 += d * d * d * d;
    }

    (sum_d, sum_d4)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn ssim_channel_inner_v3(
    token: archmage::X64V3Token,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64) {
    let c2v = f32x8::splat(token, C2);
    let one = f32x8::splat(token, 1.0);
    let two = f32x8::splat(token, 2.0);
    let zero = f32x8::zero(token);

    let n = mu1.len();
    let chunks = n / 8;
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());
        let ssq = f32x8::from_array(token, sum_sq[base..][..8].try_into().unwrap());
        let s12v = f32x8::from_array(token, s12[base..][..8].try_into().unwrap());

        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, one);
        let num_s = two.mul_add(s12v - m1 * m2, c2v);
        let denom_s = ssq - m1 * m1 - m2 * m2 + c2v;
        let d = (one - (num_m * num_s) / denom_s).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = (1.0 - (num_m * num_s) / denom_s).max(0.0);
        sum_d += d;
        sum_d4 += d * d * d * d;
    }

    (sum_d, sum_d4)
}

fn ssim_channel_inner_scalar(
    _token: archmage::ScalarToken,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
) -> (f64, f64) {
    let n = mu1.len();
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;

    for i in 0..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = (1.0 - (num_m * num_s) / denom_s).max(0.0);
        sum_d += d;
        sum_d4 += d * d * d * d;
    }

    (sum_d, sum_d4)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_inner_v4(
    token: archmage::X64V4Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64) {
    let one = f32x16::splat(token, 1.0);
    let zero = f32x16::zero(token);

    let n = img1.len();
    let chunks = n / 16;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;

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
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0;

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det4 += det.powi(4);
    }

    (sum_art, sum_art4, sum_det, sum_det4)
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn edge_diff_inner_v3(
    token: archmage::X64V3Token,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64) {
    let one = f32x8::splat(token, 1.0);
    let zero = f32x8::zero(token);

    let n = img1.len();
    let chunks = n / 8;
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;

    for c in 0..chunks {
        let base = c * 8;
        let i1 = f32x8::from_array(token, img1[base..][..8].try_into().unwrap());
        let i2 = f32x8::from_array(token, img2[base..][..8].try_into().unwrap());
        let m1 = f32x8::from_array(token, mu1[base..][..8].try_into().unwrap());
        let m2 = f32x8::from_array(token, mu2[base..][..8].try_into().unwrap());

        let diff1 = (i1 - m1).abs();
        let diff2 = (i2 - m2).abs();

        // d1 = (1 + diff2) / (1 + diff1) - 1
        let d1 = (one + diff2) / (one + diff1) - one;

        let artifact = d1.max(zero);
        let detail_lost = (-d1).max(zero);

        let a2 = artifact * artifact;
        let a4 = a2 * a2;
        let dl2 = detail_lost * detail_lost;
        let dl4 = dl2 * dl2;

        sum_art += artifact.reduce_add() as f64;
        sum_art4 += a4.reduce_add() as f64;
        sum_det += detail_lost.reduce_add() as f64;
        sum_det4 += dl4.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0;

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det4 += det.powi(4);
    }

    (sum_art, sum_art4, sum_det, sum_det4)
}

fn edge_diff_inner_scalar(
    _token: archmage::ScalarToken,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
) -> (f64, f64, f64, f64) {
    let n = img1.len();
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;

    for i in 0..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0;

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det4 += det.powi(4);
    }

    (sum_art, sum_art4, sum_det, sum_det4)
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

fn abs_diff_into_inner_scalar(
    _token: archmage::ScalarToken,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    for i in 0..a.len() {
        out[i] = (a[i] - b[i]).abs();
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
        let num_s = two.mul_add(s12v - m1 * m2, c2v);
        let denom_s = ssq - m1 * m1 - m2 * m2 + c2v;
        let d = ((one - (num_m * num_s) / denom_s) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 16)..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = ((1.0 - (num_m * num_s) / denom_s) * f64::from(mask[i])).max(0.0);
        sum_d += d;
        sum_d2 += d * d;
        sum_d4 += d * d * d * d;
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
        let num_s = two.mul_add(s12v - m1 * m2, c2v);
        let denom_s = ssq - m1 * m1 - m2 * m2 + c2v;
        let d = ((one - (num_m * num_s) / denom_s) * mv).max(zero);
        let d2 = d * d;
        let d4 = d2 * d2;

        sum_d += d.reduce_add() as f64;
        sum_d2 += d2.reduce_add() as f64;
        sum_d4 += d4.reduce_add() as f64;
    }

    for i in (chunks * 8)..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = ((1.0 - (num_m * num_s) / denom_s) * f64::from(mask[i])).max(0.0);
        sum_d += d;
        sum_d2 += d * d;
        sum_d4 += d * d * d * d;
    }

    (sum_d, sum_d4, sum_d2)
}

fn ssim_channel_masked_inner_scalar(
    _token: archmage::ScalarToken,
    mu1: &[f32],
    mu2: &[f32],
    sum_sq: &[f32],
    s12: &[f32],
    mask: &[f32],
) -> (f64, f64, f64) {
    let n = mu1.len();
    let mut sum_d = 0.0f64;
    let mut sum_d4 = 0.0f64;
    let mut sum_d2 = 0.0f64;

    for i in 0..n {
        let md = f64::from(mu1[i] - mu2[i]);
        let num_m = 1.0 - md * md;
        let num_s = 2.0 * f64::from(s12[i] - mu1[i] * mu2[i]) + f64::from(C2);
        let denom_s =
            f64::from(sum_sq[i]) - f64::from(mu1[i] * mu1[i]) - f64::from(mu2[i] * mu2[i])
                + f64::from(C2);
        let d = ((1.0 - (num_m * num_s) / denom_s) * f64::from(mask[i])).max(0.0);
        sum_d += d;
        sum_d2 += d * d;
        sum_d4 += d * d * d * d;
    }

    (sum_d, sum_d4, sum_d2)
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
        let d1 = (f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0) * f64::from(mask[i]);

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art2 += art * art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det2 += det * det;
        sum_det4 += det.powi(4);
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
        let d1 = (f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0) * f64::from(mask[i]);

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art2 += art * art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det2 += det * det;
        sum_det4 += det.powi(4);
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}

fn edge_diff_masked_inner_scalar(
    _token: archmage::ScalarToken,
    img1: &[f32],
    img2: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    mask: &[f32],
) -> (f64, f64, f64, f64, f64, f64) {
    let n = img1.len();
    let mut sum_art = 0.0f64;
    let mut sum_art4 = 0.0f64;
    let mut sum_art2 = 0.0f64;
    let mut sum_det = 0.0f64;
    let mut sum_det4 = 0.0f64;
    let mut sum_det2 = 0.0f64;

    for i in 0..n {
        let diff1 = (img1[i] - mu1[i]).abs();
        let diff2 = (img2[i] - mu2[i]).abs();
        let d1 = (f64::from((1.0 + diff2) / (1.0 + diff1)) - 1.0) * f64::from(mask[i]);

        let art = d1.max(0.0);
        let det = (-d1).max(0.0);
        sum_art += art;
        sum_art2 += art * art;
        sum_art4 += art.powi(4);
        sum_det += det;
        sum_det2 += det * det;
        sum_det4 += det.powi(4);
    }

    (sum_art, sum_art4, sum_det, sum_det4, sum_art2, sum_det2)
}
