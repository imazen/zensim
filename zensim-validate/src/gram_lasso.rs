//! Gram-standardization + lasso coordinate descent — the Rust owner of the
//! `scripts/v_next/linear_projections_2026-07-03.py` fit math
//! (`MixGram.__init__` + `MixGram.lasso`), ported op-for-op in f64 so a
//! same-input run is BIT-EXACT against the Python fit (task #68, the
//! Rust-native BHdr chain; parity-gated by `bake_dial_refit fit-lasso
//! --parity-fit`).
//!
//! Also home to the two numeric-formatting shims that chain needs:
//! * [`f64_to_f16_bits`] — single-rounding IEEE f64→f16 (ties-to-even),
//!   matching `np.float64.astype(np.float16)`. A `f64 as f32` → f32→f16
//!   two-step DOUBLE-ROUNDS (≈2⁻¹³ per-value flip odds — ~4% across 372
//!   weights), so the direct conversion is load-bearing for byte parity.
//! * [`py_repr_f64`] — CPython `repr(float)` formatting (shortest digits,
//!   scientific iff decimal point ≤ −4 or > 16, 2-digit signed exponent),
//!   needed because the `zentrain.feature_transform_params` metadata TEXT is
//!   part of the shipped bake bytes and Rust's `{}` never uses exponent
//!   notation (`1e-05` would become `0.00001`).

/// Standardized Gram system: `G = Σ z zᵀ`, `c = Σ z (y − ȳ)` over the
/// weighted data mass `W`, plus the standardization (`mu`, `sd`) and target
/// mean (`ybar`) needed to run + bake the fitted head.
pub struct StandardizedGram {
    pub n_feat: usize,
    /// Row-major `n_feat × n_feat` standardized Gram.
    pub g: Vec<f64>,
    pub c: Vec<f64>,
    /// Total weighted row mass `W`.
    pub w_total: f64,
    pub mu: Vec<f64>,
    pub sd: Vec<f64>,
    pub ybar: f64,
}

/// One group's RAW moments for [`standardize_gram_multi`]: exactly what a
/// frozen gram npz stores (`s_mat = S = Σ x xᵀ` row-major, `s_vec = Σ x`,
/// `q = Σ x·y`, `y1 = Σ y`, `n_rows = n`) plus its mix weight.
pub struct GramGroup<'a> {
    pub weight: f64,
    pub s_mat: &'a [f64],
    pub s_vec: &'a [f64],
    pub q: &'a [f64],
    pub y1: f64,
    pub n_rows: f64,
}

/// Mirror of `MixGram.__init__` for a single `(group, weight, target)` —
/// the shipped-BHdr case (`hdrmix = [("hdr_v3mix", 1.0, "human_score")]`).
///
/// Inputs are the RAW per-group moments exactly as stored in the frozen
/// gram npz. With `weight == 1.0` the weighting (`S += w·S_z`, …) is exact
/// pass-through, so the standardized system is bit-identical to numpy's.
/// Delegates to [`standardize_gram_multi`] with one group — the multi
/// accumulation reduces to exactly the old single-group expressions
/// (`acc = w·v`, one rounding), so this stays bit-exact vs the Python fit
/// (guarded by the `--parity-fit` gate and the unit tests below).
pub fn standardize_gram(
    n_feat: usize,
    weight: f64,
    s_mat: &[f64],
    s_vec: &[f64],
    q: &[f64],
    y1: f64,
    n_rows: f64,
) -> Result<StandardizedGram, String> {
    standardize_gram_multi(
        n_feat,
        &[GramGroup {
            weight,
            s_mat,
            s_vec,
            q,
            y1,
            n_rows,
        }],
    )
}

/// Multi-group `MixGram.__init__`: accumulate `S += w·S_z ; s += w·s_z ;
/// q += w·q_z ; Y1 += w·Y1_z ; W += w·n` over the groups IN ORDER (each
/// `·` and `+` rounds once, matching numpy's `acc += w * arr` loop), then
/// standardize the combined system. Every expression keeps the Python's
/// operation ORDER (no FMA — Rust guarantees no contraction).
pub fn standardize_gram_multi(
    n_feat: usize,
    groups: &[GramGroup<'_>],
) -> Result<StandardizedGram, String> {
    if groups.is_empty() {
        return Err("standardize_gram_multi: no groups".into());
    }
    for (gi, g) in groups.iter().enumerate() {
        if g.s_mat.len() != n_feat * n_feat || g.s_vec.len() != n_feat || g.q.len() != n_feat {
            return Err(format!(
                "standardize_gram_multi: group {gi} shape mismatch (n_feat={n_feat}, S={}, s={}, q={})",
                g.s_mat.len(),
                g.s_vec.len(),
                g.q.len()
            ));
        }
    }
    // First group: acc = w·v (single rounding — identical to the old
    // single-group path). Subsequent groups: acc += w·v (mul then add,
    // two roundings, matching `S += w * S_z`).
    let g0 = &groups[0];
    let mut s_mat: Vec<f64> = g0.s_mat.iter().map(|v| g0.weight * v).collect();
    let mut s_vec: Vec<f64> = g0.s_vec.iter().map(|v| g0.weight * v).collect();
    let mut q: Vec<f64> = g0.q.iter().map(|v| g0.weight * v).collect();
    let mut y1 = g0.weight * g0.y1;
    let mut w_total = g0.weight * g0.n_rows;
    for g in &groups[1..] {
        for (acc, v) in s_mat.iter_mut().zip(g.s_mat) {
            *acc += g.weight * v;
        }
        for (acc, v) in s_vec.iter_mut().zip(g.s_vec) {
            *acc += g.weight * v;
        }
        for (acc, v) in q.iter_mut().zip(g.q) {
            *acc += g.weight * v;
        }
        y1 += g.weight * g.y1;
        w_total += g.weight * g.n_rows;
    }

    // mu = s / W
    let mu: Vec<f64> = s_vec.iter().map(|v| v / w_total).collect();
    // var = np.maximum(S.diagonal() / W - mu * mu, 0.0)
    // sd  = sqrt(var); sd[sd < 1e-9] = 1.0
    let mut sd = vec![0.0f64; n_feat];
    for i in 0..n_feat {
        let var = s_mat[i * n_feat + i] / w_total - mu[i] * mu[i];
        let var = if var < 0.0 { 0.0 } else { var };
        let s = var.sqrt();
        sd[i] = if s < 1e-9 { 1.0 } else { s };
    }
    // Sc = S - W * outer(mu, mu);  G = Sc / outer(sd, sd)
    let mut g = vec![0.0f64; n_feat * n_feat];
    for i in 0..n_feat {
        for j in 0..n_feat {
            let sc = s_mat[i * n_feat + j] - w_total * (mu[i] * mu[j]);
            g[i * n_feat + j] = sc / (sd[i] * sd[j]);
        }
    }
    // ybar = Y1 / W;  c = (q - mu * Y1) / sd
    let ybar = y1 / w_total;
    let c: Vec<f64> = (0..n_feat).map(|i| (q[i] - mu[i] * y1) / sd[i]).collect();
    Ok(StandardizedGram {
        n_feat,
        g,
        c,
        w_total,
        mu,
        sd,
        ybar,
    })
}

/// `np.sign` semantics: ±1 for nonzero, the (signed) zero itself for zero.
/// Only the zero case differs from `f64::signum` (which returns ±1.0 for
/// ±0.0); in the lasso update the zero branch always multiplies a 0.0
/// soft-threshold term, so this matters only for `-0.0` bookkeeping.
fn np_sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        x
    }
}

/// Mirror of `MixGram.lasso`: cyclic coordinate descent on `(G/W, c/W)`
/// with fixed sweep order `j = 0..n`, `lam` on the mean-loss scale,
/// soft-threshold update, and max-|Δw| sweep convergence (`< tol` breaks).
///
/// Bit-exactness notes (each mirrors one numpy expression):
/// * `Gn = G / W`, `cn = c / W`: elementwise scalar divide, rounded once.
/// * `d = Gn.diagonal().copy(); d[d < 1e-12] = 1e-12`.
/// * `rho = cn[j] - Gw[j] + d[j] * w[j]` — left-to-right, `*` first.
/// * `nw = sign(rho) * max(|rho| - lam, 0.0) / d[j]` — `(sign·max)/d`.
/// * `Gw += Gn[:, j] * (nw - w[j])` — per-element mul then add (two
///   roundings, no FMA), reading COLUMN `j`.
pub fn lasso_cd(sg: &StandardizedGram, lam: f64, n_sweeps: usize, tol: f64) -> Vec<f64> {
    lasso_cd_slice(sg, lam, n_sweeps, tol, None)
}

/// [`lasso_cd`] restricted to a coordinate SLICE — the ADD156-class
/// `w[out-of-slice] = 0` constraint (SOTA-944 §3a: additive spatializable
/// subsets are gram-column selections). `slice = None` sweeps `0..n` with
/// the EXACT float-op sequence of the unrestricted solver (the full-index
/// walk is the same loop; BHdr byte-parity depends on this).
pub fn lasso_cd_slice(
    sg: &StandardizedGram,
    lam: f64,
    n_sweeps: usize,
    tol: f64,
    slice: Option<&[usize]>,
) -> Vec<f64> {
    let n = sg.n_feat;
    // Column-major copy of Gn = G / W so the update reads a contiguous
    // column. (The frozen grams are bitwise symmetric, but we still index
    // (i, j) exactly as numpy does — no symmetry assumption.)
    let mut gn_cols = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            gn_cols[j * n + i] = sg.g[i * n + j] / sg.w_total;
        }
    }
    let cn: Vec<f64> = sg.c.iter().map(|v| v / sg.w_total).collect();
    let mut d: Vec<f64> = (0..n).map(|j| gn_cols[j * n + j]).collect();
    for v in &mut d {
        if *v < 1e-12 {
            *v = 1e-12;
        }
    }
    let all: Vec<usize>;
    let idx: &[usize] = match slice {
        Some(s) => s,
        None => {
            all = (0..n).collect();
            &all
        }
    };
    let mut w = vec![0.0f64; n];
    let mut gw = vec![0.0f64; n];
    for _ in 0..n_sweeps {
        let mut delta = 0.0f64;
        for &j in idx {
            let rho = cn[j] - gw[j] + d[j] * w[j];
            let nw = np_sign(rho) * (rho.abs() - lam).max(0.0) / d[j];
            if nw != w[j] {
                let diff = nw - w[j];
                let col = &gn_cols[j * n..(j + 1) * n];
                for (gwi, gnij) in gw.iter_mut().zip(col) {
                    *gwi += gnij * diff;
                }
                let ad = diff.abs();
                if ad > delta {
                    delta = ad;
                }
                w[j] = nw;
            }
        }
        if delta < tol {
            break;
        }
    }
    w
}

/// Box-constrained cyclic coordinate descent on the SAME standardized
/// system as [`lasso_cd`] — the SOTA-944 owner extension for the BVLS
/// (bounded-variable least-squares) head class (`benchmarks/
/// sota944_campaign_2026-08-03.md` §3e/§4). Solves
/// `min ½ wᵀ(G/W)w − (c/W)ᵀw  s.t.  lo ≤ w ≤ hi` (bounds in z-space; a
/// raw-space sign constraint maps to the same-signed z-space bound because
/// `w_raw = w_z / sd` with `sd > 0`).
///
/// Identical `Gn`/`cn`/`d` precompute and column-major update walk as
/// `lasso_cd`; the per-coordinate update is the exact box projection
/// `w_j ← clamp(ρ_j / d_j, lo_j, hi_j)`, which for a PSD system converges
/// to the box-QP optimum. Deterministic (fixed sweep order, no seed).
/// scipy `lsq_linear(method="bvls")` parity is NOT claimed — this is a
/// fresh-fit owner, gated by its own unit tests (unconstrained ≡ λ=0
/// lasso; active-bound fixture solved by hand).
pub fn box_cd(
    sg: &StandardizedGram,
    lo: &[f64],
    hi: &[f64],
    n_sweeps: usize,
    tol: f64,
) -> Vec<f64> {
    box_cd_slice(sg, lo, hi, n_sweeps, tol, None)
}

/// [`box_cd`] restricted to a coordinate slice (see [`lasso_cd_slice`]).
pub fn box_cd_slice(
    sg: &StandardizedGram,
    lo: &[f64],
    hi: &[f64],
    n_sweeps: usize,
    tol: f64,
    slice: Option<&[usize]>,
) -> Vec<f64> {
    let n = sg.n_feat;
    assert_eq!(lo.len(), n, "box_cd: lo bound length");
    assert_eq!(hi.len(), n, "box_cd: hi bound length");
    let mut gn_cols = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            gn_cols[j * n + i] = sg.g[i * n + j] / sg.w_total;
        }
    }
    let cn: Vec<f64> = sg.c.iter().map(|v| v / sg.w_total).collect();
    let mut d: Vec<f64> = (0..n).map(|j| gn_cols[j * n + j]).collect();
    for v in &mut d {
        if *v < 1e-12 {
            *v = 1e-12;
        }
    }
    let all: Vec<usize>;
    let idx: &[usize] = match slice {
        Some(s) => s,
        None => {
            all = (0..n).collect();
            &all
        }
    };
    let mut w = vec![0.0f64; n];
    let mut gw = vec![0.0f64; n];
    for _ in 0..n_sweeps {
        let mut delta = 0.0f64;
        for &j in idx {
            let rho = cn[j] - gw[j] + d[j] * w[j];
            let nw = (rho / d[j]).clamp(lo[j], hi[j]);
            if nw != w[j] {
                let diff = nw - w[j];
                let col = &gn_cols[j * n..(j + 1) * n];
                for (gwi, gnij) in gw.iter_mut().zip(col) {
                    *gwi += gnij * diff;
                }
                let ad = diff.abs();
                if ad > delta {
                    delta = ad;
                }
                w[j] = nw;
            }
        }
        if delta < tol {
            break;
        }
    }
    w
}

/// Single-rounding IEEE 754 f64 → binary16 conversion (round-to-nearest,
/// ties-to-even), bit-identical to numpy's `npy_double_to_half`
/// (`numpy/core/src/npymath/halffloat.c`, BSD-3; algorithm ported —
/// including its one-branch round-half-even trick and the significand
/// carry that overflows into the exponent).
pub fn f64_to_f16_bits(x: f64) -> u16 {
    let d = x.to_bits();
    let h_sgn = ((d & 0x8000_0000_0000_0000) >> 48) as u16;
    let d_exp = d & 0x7ff0_0000_0000_0000;

    // Exponent overflow / NaN → signed inf / NaN.
    if d_exp >= 0x40f0_0000_0000_0000 {
        if d_exp == 0x7ff0_0000_0000_0000 {
            let d_sig = d & 0x000f_ffff_ffff_ffff;
            if d_sig != 0 {
                // NaN: keep the top significand bits, force quiet-ness.
                let mut ret = 0x7c00u16 + (d_sig >> 42) as u16;
                if ret == 0x7c00 {
                    ret += 1;
                }
                return h_sgn + ret;
            }
            return h_sgn + 0x7c00; // ±inf
        }
        return h_sgn + 0x7c00; // overflow → ±inf
    }

    // Exponent underflow → subnormal half or signed zero.
    if d_exp <= 0x3f00_0000_0000_0000 {
        // |x| < 2^-24 / 2 → signed zero (2^-25 itself ties to even = 0).
        if d_exp < 0x3e60_0000_0000_0000 {
            return h_sgn;
        }
        // Build the subnormal significand.
        let exp_shift = (d_exp >> 52) - 998;
        let mut d_sig = 0x0010_0000_0000_0000u64 + (d & 0x000f_ffff_ffff_ffff);
        d_sig <<= exp_shift;
        // Round-half-even in one branch (see the normal path below).
        if (d_sig & 0x003f_ffff_ffff_ffff) != 0x0010_0000_0000_0000 {
            d_sig += 0x0010_0000_0000_0000;
        }
        let h_sig = (d_sig >> 53) as u16;
        return h_sgn + h_sig;
    }

    // Normal range. Biased half exponent field in place:
    let h_exp = ((d_exp - 0x3f00_0000_0000_0000) >> 42) as u16;
    let mut d_sig = d & 0x000f_ffff_ffff_ffff;
    // Round-half-even: adding 2^41 rounds up when the dropped bits exceed
    // half, or equal half with an odd kept LSB; the single excluded pattern
    // (kept LSB 0, round bit 1, rest 0) is the tie that truncates to even.
    if (d_sig & 0x0000_07ff_ffff_ffff) != 0x0000_0200_0000_0000 {
        d_sig += 0x0000_0200_0000_0000;
    }
    let h_sig = (d_sig >> 42) as u16;
    // A significand carry (0x3ff → 0x400) propagates into the exponent,
    // which is exactly right (including overflow to inf at h_exp 30).
    h_sgn + h_exp + h_sig
}

/// f16 bits → the exact f64 value (via the crate-canonical f16→f32 decode;
/// both widenings are exact). Matches `np.float16.astype(np.float64)`.
pub fn f16_bits_to_f64(bits: u16) -> f64 {
    zenpredict::f16_bits_to_f32(bits) as f64
}

/// CPython `repr(float)` (py3 `str` == `repr`): shortest round-trip digits,
/// fixed-point iff the decimal point lands in (−4, 16], otherwise
/// `d.ddde±XX` scientific with a ≥2-digit signed exponent, and a trailing
/// `.0` on integral fixed-point values.
///
/// Rust's `{:e}` supplies the identical shortest digit string (both sides
/// produce the unique shortest correctly-rounded decimal); this function
/// only re-arranges presentation.
pub fn py_repr_f64(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    let neg = x < 0.0;
    let sci = format!("{:e}", x.abs()); // e.g. "1.07035e-6", "2e0"
    let (mant, exp_str) = sci.split_once('e').expect("{:e} always has an exponent");
    let exp: i32 = exp_str.parse().expect("{:e} exponent is an integer");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let decpt = exp + 1; // value = 0.<digits> * 10^decpt
    let body = if decpt <= -4 || decpt > 16 {
        let e10 = decpt - 1;
        let mut s = String::with_capacity(digits.len() + 6);
        s.push(digits.as_bytes()[0] as char);
        if digits.len() > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        s.push('e');
        s.push(if e10 < 0 { '-' } else { '+' });
        s.push_str(&format!("{:02}", e10.abs()));
        s
    } else if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt as usize >= digits.len() {
        format!("{}{}.0", digits, "0".repeat(decpt as usize - digits.len()))
    } else {
        format!(
            "{}.{}",
            &digits[..decpt as usize],
            &digits[decpt as usize..]
        )
    };
    if neg { format!("-{body}") } else { body }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------- f16

    /// Every finite f16 value must round-trip f16 → f64 → f16 exactly.
    #[test]
    fn f16_round_trip_all_finite() {
        for bits in 0u16..=0xffff {
            let exp = bits & 0x7c00;
            if exp == 0x7c00 {
                continue; // inf/nan handled separately
            }
            let back = f64_to_f16_bits(f16_bits_to_f64(bits));
            assert_eq!(back, bits, "bits {bits:#06x} failed round trip");
        }
    }

    /// Midpoints between every pair of adjacent finite f16 values must tie
    /// to even, and one-ulp(f64) nudges off the midpoint must go to the
    /// nearer neighbor.
    #[test]
    fn f16_ties_to_even_and_nudges() {
        for bits in 0u16..0x7bff {
            // positive ascending pairs (a, a+1); skip into inf at 0x7c00
            let a = f16_bits_to_f64(bits);
            let b = f16_bits_to_f64(bits + 1);
            let mid = (a + b) / 2.0; // exactly representable in f64
            let even = if bits & 1 == 0 { bits } else { bits + 1 };
            assert_eq!(f64_to_f16_bits(mid), even, "tie at {bits:#06x}");
            let lo = f64::from_bits(mid.to_bits() - 1);
            let hi = f64::from_bits(mid.to_bits() + 1);
            assert_eq!(f64_to_f16_bits(lo), bits, "below-mid at {bits:#06x}");
            assert_eq!(f64_to_f16_bits(hi), bits + 1, "above-mid at {bits:#06x}");
        }
    }

    /// The double-rounding trap: x just below an f16 midpoint whose f32
    /// rounding lands ON the midpoint. f64→f32→f16 goes UP (tie to even);
    /// the correct single rounding goes DOWN.
    #[test]
    fn f16_no_double_rounding() {
        let a = f16_bits_to_f64(0x3c01); // 1.0009765625 (odd)
        let b = f16_bits_to_f64(0x3c02); // even
        let mid = (a + b) / 2.0;
        let x = f64::from_bits(mid.to_bits() - 1); // just below the tie
        // the two-step conversion demonstrates the trap… (f32→f64 widening
        // is exact, so f64_to_f16_bits doubles as the f32→f16 rounder here)
        let via_f32 = x as f32; // rounds UP to the midpoint (exact in f32)
        assert_eq!(via_f32 as f64, mid);
        let two_step = f64_to_f16_bits(via_f32 as f64);
        assert_eq!(two_step, 0x3c02, "double rounding goes up (tie to even)");
        // …and the direct conversion avoids it.
        assert_eq!(f64_to_f16_bits(x), 0x3c01, "single rounding stays down");
    }

    #[test]
    fn f16_specials_and_overflow() {
        assert_eq!(f64_to_f16_bits(0.0), 0x0000);
        assert_eq!(f64_to_f16_bits(-0.0), 0x8000);
        assert_eq!(f64_to_f16_bits(f64::INFINITY), 0x7c00);
        assert_eq!(f64_to_f16_bits(f64::NEG_INFINITY), 0xfc00);
        assert_eq!(f64_to_f16_bits(65504.0), 0x7bff); // max finite
        assert_eq!(f64_to_f16_bits(65519.999), 0x7bff); // below the tie
        assert_eq!(f64_to_f16_bits(65520.0), 0x7c00); // tie → inf (even)
        assert_eq!(f64_to_f16_bits(1e300), 0x7c00);
        assert_eq!(f64_to_f16_bits(2f64.powi(-25)), 0x0000); // tie → 0 (even)
        let min_sub = 2f64.powi(-24);
        assert_eq!(f64_to_f16_bits(min_sub), 0x0001);
        assert_eq!(f64_to_f16_bits(min_sub * 0.75), 0x0001); // rounds up
        assert!(f64_to_f16_bits(f64::NAN) & 0x7c00 == 0x7c00);
        assert!(f64_to_f16_bits(f64::NAN) & 0x03ff != 0);
    }

    // ------------------------------------------------------------ py repr

    /// Reference outputs generated with CPython 3:
    /// `repr(float(v))` for each case (2026-07-29).
    #[test]
    fn py_repr_matches_cpython() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (2.0, "2.0"),
            (0.5, "0.5"),
            (100.0, "100.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e-4, "0.0001"),
            (0.000101883, "0.000101883"),
            (1e-5, "1e-05"),
            (1.07035e-06, "1.07035e-06"),
            (2.6458e-12, "2.6458e-12"),
            (-23.1646, "-23.1646"),
            (1.19209e-07, "1.19209e-07"),
            (0.144186, "0.144186"),
            (123456789012345.6, "123456789012345.6"),
            (65504.0, "65504.0"),
            (5e-324, "5e-324"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.3333333333333333"),
            (9999999999999998.0, "9999999999999998.0"),
            (1e100, "1e+100"),
            (-4.940656458412465e-100, "-4.940656458412465e-100"),
            (1234.5678, "1234.5678"),
        ];
        for (x, expect) in cases {
            assert_eq!(py_repr_f64(*x), *expect, "value {x:?}");
        }
    }

    /// Round-trip: the emitted string must parse back to the same bits.
    #[test]
    fn py_repr_round_trips() {
        let vals = [
            1.07035e-06,
            0.000101883,
            65519.999,
            -3.77245,
            2.38419e-07,
            f64::MIN_POSITIVE,
            1.0000000000000002,
        ];
        for v in vals {
            let s = py_repr_f64(v);
            let back: f64 = s.parse().expect("parse back");
            assert_eq!(back.to_bits(), v.to_bits(), "{v:?} -> {s} -> {back:?}");
        }
    }

    // -------------------------------------------------------------- lasso

    /// Build raw moments from an explicit tiny design matrix the same way
    /// `cmd_gram` accumulates them (S = XᵀX, s = Σx, q = Xᵀy, Y1 = Σy).
    fn moments(x: &[[f64; 3]], y: &[f64]) -> ([f64; 9], [f64; 3], [f64; 3], f64, f64) {
        let mut s_mat = [0.0f64; 9];
        let mut s_vec = [0.0f64; 3];
        let mut q = [0.0f64; 3];
        let mut y1 = 0.0f64;
        for (row, &yv) in x.iter().zip(y) {
            for i in 0..3 {
                for j in 0..3 {
                    s_mat[i * 3 + j] += row[i] * row[j];
                }
                s_vec[i] += row[i];
                q[i] += row[i] * yv;
            }
            y1 += yv;
        }
        (s_mat, s_vec, q, y1, x.len() as f64)
    }

    /// On an orthogonal standardized design, lasso has the closed-form
    /// soft-threshold solution: w_j = sign(r_j)·max(|r_j| − λ, 0) where
    /// r_j = corr(z_j, y). Feature 2 is pure noise with tiny correlation, so
    /// a moderate λ must zero it while shrinking the others by exactly λ/1.
    #[test]
    fn lasso_soft_thresholds_orthogonal_design() {
        // 8 rows, 3 features: f0/f1 = ±1 orthogonal patterns, f2 tiny noise.
        let x = [
            [1.0, 1.0, 0.001],
            [1.0, -1.0, -0.001],
            [-1.0, 1.0, 0.001],
            [-1.0, -1.0, -0.001],
            [1.0, 1.0, -0.001],
            [1.0, -1.0, 0.001],
            [-1.0, 1.0, -0.001],
            [-1.0, -1.0, 0.001],
        ];
        // y = 0.8·f0 + 0.3·f1 (+ nothing from f2)
        let y: Vec<f64> = x.iter().map(|r| 0.8 * r[0] + 0.3 * r[1]).collect();
        let (s_mat, s_vec, q, y1, n) = moments(&x, &y);
        let sg = standardize_gram(3, 1.0, &s_mat, &s_vec, &q, y1, n).expect("standardize");
        assert!((sg.ybar - 0.0).abs() < 1e-15);

        let lam = 0.05;
        let w = lasso_cd(&sg, lam, 200, 1e-10);
        // standardized f0/f1 are exactly the ±1 columns (sd = 1), so
        // cn_j = corr = 0.8 / 0.3; soft-threshold shrinks by exactly lam.
        assert!((w[0] - (0.8 - lam)).abs() < 1e-12, "w0 = {}", w[0]);
        assert!((w[1] - (0.3 - lam)).abs() < 1e-12, "w1 = {}", w[1]);
        assert_eq!(w[2], 0.0, "noise feature must be zeroed exactly");
    }

    /// λ = 0 on a well-conditioned system converges to the OLS solution of
    /// the standardized normal equations (verify via residual, not op-order).
    #[test]
    fn lasso_lam0_solves_normal_equations() {
        let x = [
            [0.9, 0.1, -0.3],
            [0.2, -0.7, 0.5],
            [-0.4, 0.6, 0.8],
            [0.1, 0.2, -0.9],
            [-0.8, -0.5, 0.2],
            [0.5, 0.9, 0.4],
            [0.3, -0.2, -0.6],
            [-0.6, 0.4, 0.1],
        ];
        let y = [0.7, -0.2, 0.9, -0.5, -0.6, 1.1, -0.1, 0.3];
        let (s_mat, s_vec, q, y1, n) = moments(&x, &y);
        let sg = standardize_gram(3, 1.0, &s_mat, &s_vec, &q, y1, n).expect("standardize");
        let w = lasso_cd(&sg, 0.0, 10_000, 1e-14);
        // residual of G w = c
        for i in 0..3 {
            let gw: f64 = (0..3).map(|j| sg.g[i * 3 + j] * w[j]).sum();
            assert!(
                (gw - sg.c[i]).abs() < 1e-9,
                "normal-eq residual row {i}: {gw} vs {}",
                sg.c[i]
            );
        }
    }

    /// Unbounded box_cd must reach the same optimum as λ=0 lasso_cd (both
    /// solve the unconstrained standardized least squares).
    #[test]
    fn box_cd_unbounded_matches_lam0_lasso() {
        let x = [
            [0.9, 0.1, -0.3],
            [0.2, -0.7, 0.5],
            [-0.4, 0.6, 0.8],
            [0.1, 0.2, -0.9],
            [-0.8, -0.5, 0.2],
            [0.5, 0.9, 0.4],
            [0.3, -0.2, -0.6],
            [-0.6, 0.4, 0.1],
        ];
        let y = [0.7, -0.2, 0.9, -0.5, -0.6, 1.1, -0.1, 0.3];
        let (s_mat, s_vec, q, y1, n) = moments(&x, &y);
        let sg = standardize_gram(3, 1.0, &s_mat, &s_vec, &q, y1, n).expect("standardize");
        let w_l = lasso_cd(&sg, 0.0, 10_000, 1e-14);
        let w_b = box_cd(
            &sg,
            &[f64::NEG_INFINITY; 3],
            &[f64::INFINITY; 3],
            10_000,
            1e-14,
        );
        for j in 0..3 {
            assert!(
                (w_l[j] - w_b[j]).abs() < 1e-9,
                "box vs lasso λ=0 differ at {j}: {} vs {}",
                w_b[j],
                w_l[j]
            );
        }
    }

    /// Active lower bound: on an orthogonal design where the unconstrained
    /// solution has w1 < 0, the [0, ∞) box must pin w1 to EXACTLY 0 and
    /// leave the orthogonal w0 at its unconstrained value (orthogonality ⇒
    /// no compensation).
    #[test]
    fn box_cd_pins_negative_coordinate_at_zero() {
        let x = [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
        ];
        // y = 0.8·f0 − 0.4·f1; f2 constant (sd floor path).
        let y: Vec<f64> = x.iter().map(|r| 0.8 * r[0] - 0.4 * r[1]).collect();
        let (s_mat, s_vec, q, y1, n) = moments(&x, &y);
        let sg = standardize_gram(3, 1.0, &s_mat, &s_vec, &q, y1, n).expect("standardize");
        let lo = [0.0, 0.0, 0.0];
        let hi = [f64::INFINITY; 3];
        let w = box_cd(&sg, &lo, &hi, 1_000, 1e-14);
        assert!((w[0] - 0.8).abs() < 1e-12, "w0 = {}", w[0]);
        assert_eq!(w[1], 0.0, "negative-direction weight must sit ON the bound");
        assert_eq!(w[2], 0.0, "constant feature stays zero");
    }

    /// Slice restriction: sweeping only {0} must leave every other weight
    /// EXACTLY zero and give f0 its single-feature solution (orthogonal
    /// design ⇒ same as the full solve's w0).
    #[test]
    fn lasso_slice_restricts_coordinates() {
        let x = [
            [1.0, 1.0, 0.5],
            [1.0, -1.0, -0.5],
            [-1.0, 1.0, 0.5],
            [-1.0, -1.0, -0.5],
            [1.0, 1.0, -0.5],
            [1.0, -1.0, 0.5],
            [-1.0, 1.0, -0.5],
            [-1.0, -1.0, 0.5],
        ];
        let y: Vec<f64> = x.iter().map(|r| 0.8 * r[0] + 0.3 * r[1]).collect();
        let (s_mat, s_vec, q, y1, n) = moments(&x, &y);
        let sg = standardize_gram(3, 1.0, &s_mat, &s_vec, &q, y1, n).expect("standardize");
        let w = lasso_cd_slice(&sg, 0.0, 1000, 1e-14, Some(&[0usize]));
        assert!((w[0] - 0.8).abs() < 1e-10, "w0 = {}", w[0]);
        assert_eq!(w[1], 0.0);
        assert_eq!(w[2], 0.0);
        // None slice must equal the plain solver bit-for-bit.
        let wa = lasso_cd(&sg, 0.05, 200, 1e-10);
        let wb = lasso_cd_slice(&sg, 0.05, 200, 1e-10, None);
        assert_eq!(
            wa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            wb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// Multi-group accumulation == the standardized system of the pooled
    /// weighted data. Two groups with weights (1.0, 0.5): the combined
    /// moments must equal moments computed over the concatenation with the
    /// second group's rows carrying weight 0.5 — verified analytically via
    /// the closed forms (mu = Σwx/Σwn etc.), and the single-group path must
    /// be BIT-identical to the delegating `standardize_gram`.
    #[test]
    fn multi_group_matches_pooled_weighted_moments() {
        let xa = [[1.0, 2.0, -0.5], [0.5, -1.0, 2.0], [2.0, 0.0, 1.0]];
        let ya = [1.0, 2.0, 3.0];
        let xb = [[3.0, 1.0, 0.0], [-2.0, 0.5, 1.5]];
        let yb = [4.0, -1.0];
        let (sa, va, qa, y1a, na) = moments(&xa, &ya);
        let mut sb = [0.0f64; 9];
        let mut vb = [0.0f64; 3];
        let mut qb = [0.0f64; 3];
        let mut y1b = 0.0f64;
        for (row, &yv) in xb.iter().zip(&yb) {
            for i in 0..3 {
                for j in 0..3 {
                    sb[i * 3 + j] += row[i] * row[j];
                }
                vb[i] += row[i];
                qb[i] += row[i] * yv;
            }
            y1b += yv;
        }
        let wa = 1.0;
        let wb = 0.5;
        let sg = standardize_gram_multi(
            3,
            &[
                GramGroup {
                    weight: wa,
                    s_mat: &sa,
                    s_vec: &va,
                    q: &qa,
                    y1: y1a,
                    n_rows: na,
                },
                GramGroup {
                    weight: wb,
                    s_mat: &sb,
                    s_vec: &vb,
                    q: &qb,
                    y1: y1b,
                    n_rows: xb.len() as f64,
                },
            ],
        )
        .expect("multi");
        // Closed-form pooled checks (analytic, tolerance-based).
        let w_total = wa * na + wb * xb.len() as f64;
        assert!((sg.w_total - w_total).abs() < 1e-12);
        for i in 0..3 {
            let sum_i: f64 = xa.iter().map(|r| r[i]).sum::<f64>() * wa
                + xb.iter().map(|r| r[i]).sum::<f64>() * wb;
            assert!((sg.mu[i] - sum_i / w_total).abs() < 1e-12, "mu[{i}]");
        }
        let ybar = (wa * y1a + wb * y1b) / w_total;
        assert!((sg.ybar - ybar).abs() < 1e-12);

        // Single group through multi == the delegating standardize_gram,
        // bit for bit.
        let a = standardize_gram(3, 0.75, &sa, &va, &qa, y1a, na).expect("single");
        let b = standardize_gram_multi(
            3,
            &[GramGroup {
                weight: 0.75,
                s_mat: &sa,
                s_vec: &va,
                q: &qa,
                y1: y1a,
                n_rows: na,
            }],
        )
        .expect("multi single");
        assert_eq!(a.ybar.to_bits(), b.ybar.to_bits());
        assert_eq!(a.w_total.to_bits(), b.w_total.to_bits());
        for k in 0..9 {
            assert_eq!(a.g[k].to_bits(), b.g[k].to_bits(), "g[{k}]");
        }
        for k in 0..3 {
            assert_eq!(a.c[k].to_bits(), b.c[k].to_bits(), "c[{k}]");
            assert_eq!(a.mu[k].to_bits(), b.mu[k].to_bits(), "mu[{k}]");
            assert_eq!(a.sd[k].to_bits(), b.sd[k].to_bits(), "sd[{k}]");
        }
    }

    /// The np.sign port: ±0 pass through, nonzero → ±1.
    #[test]
    fn np_sign_semantics() {
        assert_eq!(np_sign(3.5), 1.0);
        assert_eq!(np_sign(-2.0), -1.0);
        assert_eq!(np_sign(0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(np_sign(-0.0).to_bits(), (-0.0f64).to_bits());
    }
}
