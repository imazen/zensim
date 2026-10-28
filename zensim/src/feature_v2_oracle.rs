
// ============================================================================
// ERA-2 DENSE KERNEL (`benchmarks/era2_perf_break_2026-08-31.md` §2, §13.2, §14)
//
// NOT YET THE DEFAULT. Built alongside era-1 so it can be gated and measured
// without moving a byte; the flip is a separate, sequenced commit (design §8 —
// it lands after the fold-engine lane pins its parity stages).
//
// THE SHAPE, and every part of it is semantics rather than implementation:
//   * 8 f32 virtual lanes per accumulator, term at pixel x -> lane[x mod 8],
//     entered via `as_chunks::<8>` — NEVER modulo indexing, which was MEASURED
//     11x slower because it defeats vectorisation entirely (§13);
//   * the scalar tail folds into the SAME lanes, so the shape is
//     width-independent (era-1's tail landed in the f64 running total, which is
//     what made its grouping depend on `width % 8`);
//   * a FIXED reduction tree — `era2_reduce8`, never `reduce_add()`, which is
//     TIER-DEPENDENT (§14.2: x86_v3 pairs lane i with i+4 first, wasm/scalar
//     pair adjacent lanes — different groupings of the same eight addends);
//   * an f64 BAND layer: rows fold into a band partial, band partials merge in
//     band order. Band layout is a pure function of (height, ERA2_BAND_ROWS),
//     never of thread count.
//
// PASS STRUCTURE IS A PERF KNOB, NOT SEMANTICS. Splitting the row into several
// passes changes neither which terms an accumulator sees nor their order — each
// accumulator still receives its terms in increasing x — so any pass split
// yields IDENTICAL bytes. That frees the split to be tuned per tier for
// register pressure without touching the era.

/// era-2 band height. A pure function of geometry; deliberately NOT derived
/// from the thread count, which is what makes thread-invariance structural.
// NOTE ON `dead_code`: these primitives are consumed by the era-2 dense kernel,
// which lands in the SEQUENCED flip commit (design §8 — after the fold-engine
// lane pins its parity stages). Until then they are exercised only by the
// era-2 structural gates, so a non-test build sees them as unused. The
// allow comes off with the kernel.
pub(crate) const ERA2_BAND_ROWS: usize = 32;

#[allow(unused_imports)]
pub(crate) use crate::featcanon::era2_reduce8;

/// Tier dispatch for the era-2 kernel. The BODY is one source shared by every
/// tier (see `dense_block_kernel_era2_inner`); `#[magetypes]` exists here only
/// to put that body inside each tier's `target_feature` region.
///
/// **This is what the first measurement was missing.** Plain Rust compiles to
/// baseline x86-64 (SSE2) no matter what the host supports, which measured
/// **4.4x slower** than era-1's `#[arcane]`-wrapped kernel — not a shape
/// problem, an ISA problem. Cross-tier bit-identity survives because the
/// source is identical and Rust does not contract `a*b+c` into an FMA without
/// fast-math (which is off); it is VERIFIED, not assumed, by
/// `era2_vendor_probe` across the AMD/Intel pair.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era2(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    incant!(
        dense_block_kernel_era2_entry(
            src,
            dst,
            mu1,
            mu2,
            ssq,
            s12,
            activity,
            width,
            height,
            transducer_bank
        ),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// Two disjoint-tier entries emitting the same base name (the era-1
// `dense_block_kernel_entry` pattern), kept because §14.4's byte-neutrality
// claim is asserted against BOTH instantiations by
// `era2_fused_and_two_pass_are_bit_identical`.
//
// BOTH ARE CURRENTLY `false` (two-pass), because the fused hypothesis was
// MEASURED AND FALSIFIED. The reasoning was that v4x's 32 SIMD registers would
// hold 13 core + 16 pool accumulators, so it could skip the scratch
// round-trip. Measured at 576x128: fused **445.5 us** against two-pass
// **226.1 us** — 2x WORSE, taking era-2 from 2.12x to 3.98x of era-1. 29
// `Lanes8` accumulators is 232 live f32 values; 32 registers do not hold that,
// and it spills. This is the same wall §A.14 hit when it scalarised the pools
// in the first place, and why POOL_SIMD was capped at 16 accumulators rather
// than 22. The split is not a 16-register concession — it is the right
// structure on every tier.
#[magetypes(+v4x, -v4, -v3, -neon, -wasm128, -scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era2_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    dense_block_kernel_era2_generic::<_>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

#[magetypes(v4, v3, neon, wasm128, scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era2_entry(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    dense_block_kernel_era2_generic::<_>(
        token,
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
    )
}

/// Accumulate the first `n` lanes of a term array into a `V8` accumulator,
/// preserving lane identity and add order.
///
/// Used ONLY for the row tail. Zero-padding the tail into a full `V8` add is
/// NOT equivalent: the pool weights are `mask_w = 1 - saturate(act)`, and
/// `saturate(0) = 0`, so a padded lane contributes `mask_w = 1.0` (and
/// `iw_w = IW_WEIGHT_FLOOR`) to the weight denominators. Those are not
/// accumulator-neutral, so the tail is masked rather than padded.
#[inline(always)]
fn v8_add_first<T: F32x8Backend + Copy>(token: T, acc: V8<T>, t: &[f32; 8], n: usize) -> V8<T> {
    let mut a = acc.to_array();
    for k in 0..n {
        a[k] += t[k];
    }
    V8::<T>::from_array_t(token, a)
}

/// `#[inline(always)]`, not `#[inline]` — this body exists ONLY to fuse into
/// its entry's `target_feature` region. era-1 measured a **5.3x** whole-
/// extraction regression when the equivalent hint stopped being honoured and
/// the vector ops compiled into calls outside the feature region; the same
/// trap applies here verbatim.
///
/// V8 REWRITE (era-2 stage B): accumulators are `V8<T>` magetypes vectors, not
/// `[f32; 8]` arrays. `V8` **is** the 8 virtual lanes, so every semantic is
/// unchanged — lane `j` still takes the terms at `x ≡ j (mod 8)` in increasing
/// `x`, the reduction is still the explicit `era2_reduce8` tree over
/// `to_array()` and never `reduce_add()` (which is tier-dependent, §14.2), the
/// tail still folds into lane `k`, and band partials still merge in band order.
/// Term evaluation moves to the `_v` helpers, i.e. **the same functions era-1's
/// SIMD path uses**, so the per-pixel math is era-1's exactly.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_era2_generic<T: F32x8Backend + Copy>(
    token: T,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
) -> DenseAccum {
    // Only revision 1 reaches this kernel; revisions 5 and 7 use the canonical kernels.
 // the split is chosen by tier; see the entries above
    let zero = V8::<T>::zero_t(token);
    let one = V8::<T>::splat_t(token, 1.0);
    let c1 = V8::<T>::splat_t(token, C1_V2 as f32);
    let c2 = V8::<T>::splat_t(token, C2_V2 as f32);
    let c_edge = V8::<T>::splat_t(token, C_EDGE as f32);
    let c_mse = V8::<T>::splat_t(token, C_MSE as f32);
    let c_hf = V8::<T>::splat_t(token, C_HF as f32);
    let c_pj = V8::<T>::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V8::<T>::splat_t(token, K_PJND_MASK as f32);
    let k_lo = V8::<T>::splat_t(token, K_PJND_MASK_LOW as f32);
    let k_hi = V8::<T>::splat_t(token, K_PJND_MASK_HIGH as f32);
    let c_act = V8::<T>::splat_t(token, C_ACTIVITY as f32);
    let c_peak = V8::<T>::splat_t(token, C_PEAK as f32);
    let iw_floor = V8::<T>::splat_t(token, IW_WEIGHT_FLOOR as f32);

    let mut acc = DenseAccum::default();
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];

        for y in b0..b1 {
            let row = y * width;
            // 13 core + 16 pool accumulators, all V8.
            let mut r = [zero; 13];
            let mut p_mw = zero;
            let mut p_iw = zero;
            let mut p_m = [zero; 4];
            let mut p_i = [zero; 4];
            let mut p_kn = [zero; 3];
            let mut p_kd = [zero; 3];

            // One term-evaluation closure, shared by the full-chunk and tail
            // paths so they cannot drift. Returns every term as a V8.
            macro_rules! terms {
                ($s:expr, $dd:expr, $m1:expr, $m2:expr, $q:expr, $p:expr, $act:expr) => {{
                    let (s, dd, m1, m2, q, p, act) = ($s, $dd, $m1, $m2, $q, $p, $act);
                    let d = ssim_d_local_v(token, m1, m2, p, q, c1, c2, false, false);
                    let d2 = d * d;
                    let diff_src = (s - m1).abs();
                    let diff_dst = (dd - m2).abs();
                    let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, false);
                    let gt = diff_dst.simd_gt(diff_src);
                    let lt = diff_dst.simd_lt(diff_src);
                    let art_i = V8::<T>::blend(gt, edge_dissim, zero);
                    let det_i = V8::<T>::blend(lt, edge_dissim, zero);
                    let raw_diff = s - dd;
                    let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
                    let hf_src = s - m1;
                    let hf_dst = dd - m2;
                    let (hfg, hfl) =
                        bounded_excess_pair_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
                    let hfm = bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
                    let rae = raw_diff.abs();
                    let pj = pjnd_transducer_v(token, rae, act, k_mid, c_pj);
                    let (pjl, pjh) = if transducer_bank {
                        (
                            pjnd_transducer_v(token, rae, act, k_lo, c_pj),
                            pjnd_transducer_v(token, rae, act, k_hi, c_pj),
                        )
                    } else {
                        (zero, zero)
                    };
                    // d³ and d⁴ share `d2`, matching era-1's SIMD path exactly
                    // (`d2 * d`, `d2 * d2`) rather than re-multiplying from `d`.
                    (
                        [
                            d,
                            d2,
                            d2 * d,
                            d2 * d2,
                            art_i,
                            det_i,
                            mse_i,
                            hfg,
                            hfl,
                            hfm,
                            pj,
                            pjl,
                            pjh,
                        ],
                        art_i,
                        det_i,
                        mse_i,
                        d,
                        act,
                    )
                }};
            }

            // Pool contributions from already-evaluated terms.
            macro_rules! pools {
                ($d:expr, $art:expr, $det:expr, $mse:expr, $act:expr) => {{
                    let (d, art_i, det_i, mse_i, act) = ($d, $art, $det, $mse, $act);
                    let sat = saturate_v(token, act, c_act);
                    let m_w = one - sat;
                    let i_w = sat + iw_floor;
                    let vals = [d, art_i, det_i, mse_i];
                    let mut mv = [zero; 4];
                    let mut iv = [zero; 4];
                    for j in 0..4 {
                        mv[j] = m_w * vals[j];
                        iv[j] = i_w * vals[j];
                    }
                    let mut kn = [zero; 3];
                    let mut kd = [zero; 3];
                    for j in 0..3 {
                        let sal = saturate_v(token, vals[j], c_peak);
                        kn[j] = sal * vals[j];
                        kd[j] = sal;
                    }
                    (m_w, i_w, mv, iv, kn, kd)
                }};
            }

            let n8 = width - width % 8;
            let mut x = 0usize;
            macro_rules! ld {
                ($plane:expr, $off:expr) => {
                    V8::<T>::from_array_t(token, $plane[$off..$off + 8].try_into().unwrap())
                };
            }
            while x < n8 {
                let i = row + x;
                let (t, art_i, det_i, mse_i, d, act) = terms!(
                    ld!(src, i),
                    ld!(dst, i),
                    ld!(mu1, i),
                    ld!(mu2, i),
                    ld!(ssq, i),
                    ld!(s12, i),
                    ld!(activity, i)
                );
                for j in 0..13 {
                    r[j] += t[j];
                }
                let (m_w, i_w, mv, iv, kn, kd) = pools!(d, art_i, det_i, mse_i, act);
                p_mw += m_w;
                p_iw += i_w;
                for j in 0..4 {
                    p_m[j] += mv[j];
                    p_i[j] += iv[j];
                }
                for j in 0..3 {
                    p_kn[j] += kn[j];
                    p_kd[j] += kd[j];
                }
                x += 8;
            }
            // TAIL: masked, never zero-padded (see `v8_add_first`).
            let rem = width - n8;
            if rem != 0 {
                let pad = |pl: &[f32]| -> V8<T> {
                    let mut o = [0.0f32; 8];
                    o[..rem].copy_from_slice(&pl[row + n8..row + width]);
                    V8::<T>::from_array_t(token, o)
                };
                let (t, art_i, det_i, mse_i, d, act) = terms!(
                    pad(src),
                    pad(dst),
                    pad(mu1),
                    pad(mu2),
                    pad(ssq),
                    pad(s12),
                    pad(activity)
                );
                for j in 0..13 {
                    r[j] = v8_add_first(token, r[j], &t[j].to_array(), rem);
                }
                let (m_w, i_w, mv, iv, kn, kd) = pools!(d, art_i, det_i, mse_i, act);
                p_mw = v8_add_first(token, p_mw, &m_w.to_array(), rem);
                p_iw = v8_add_first(token, p_iw, &i_w.to_array(), rem);
                for j in 0..4 {
                    p_m[j] = v8_add_first(token, p_m[j], &mv[j].to_array(), rem);
                    p_i[j] = v8_add_first(token, p_i[j], &iv[j].to_array(), rem);
                }
                for j in 0..3 {
                    p_kn[j] = v8_add_first(token, p_kn[j], &kn[j].to_array(), rem);
                    p_kd[j] = v8_add_first(token, p_kd[j], &kd[j].to_array(), rem);
                }
            }

            // Per-row reduction into the f64 band partial, via the FIXED tree.
            for j in 0..13 {
                band[j] += era2_reduce8(r[j].to_array());
            }
            for j in 0..3 {
                band[13 + j * 2] += era2_reduce8(p_kn[j].to_array());
                band[14 + j * 2] += era2_reduce8(p_kd[j].to_array());
            }
            let mw_row = era2_reduce8(p_mw.to_array());
            let iw_row = era2_reduce8(p_iw.to_array());
            for j in 0..4 {
                band[19 + j * 2] += era2_reduce8(p_m[j].to_array());
                band[20 + j * 2] += mw_row;
                band[27 + j * 2] += era2_reduce8(p_i[j].to_array());
                band[28 + j * 2] += iw_row;
            }
        }

        // Band partials merge in BAND INDEX ORDER.
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

// ============================================================================
// featacc dense candidates — measurement-only bodies
// ============================================================================
//
// `dense_block_kernel_era2_generic` IS the shipped canonical arithmetic for
// the dense block: f32 SIMD lanes (`x mod 8`) reduced per row through the
// fixed `era2_reduce8` tree into f64 band partials. The mirrors below put
// the accumulation on the `Pool` axis and the element eval on its own axis
// so the candidates can be measured instead of asserted:
//
// * [`dense_block_kernel_canon`]<P>: per-pixel SCALAR f32 element eval in
//   the SIMD body's exact op order — magetypes `*`/`+`/`/` are unfused on
//   every backend, so the scalar replay is element-bitwise with era-2 on
//   every tier — then `P::add(x & 7, term)` into per-row pools, `fin()` per
//   row, `band[j] +=`, strip `+=` — the era-2 accumulation structure with
//   only the pool arithmetic swapped. `P = LanesF32` replays era-2 bit for
//   bit; `LanesF64`/`Neum64` are candidates (e)/(f).
// * [`dense_block_kernel_exact`]: f64 element eval via the existing f64
//   scalar helpers (`ssim_d_local`, `bounded_sim`, …), Neumaier pools —
//   the oracle arm.

/// The 13 core + 16 pool terms of one dense pixel, f32 scalar — mirrors
/// `dense_block_kernel_era2_generic`'s `terms!`/`pools!` op-for-op.
#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub(crate) fn dense_terms32(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    q: f32,
    p: f32,
    act: f32,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
) -> (
    [f32; 13],
    f32,
    f32,
    f32,
    f32,
    f32,
    [f32; 4],
    [f32; 4],
    [f32; 3],
    [f32; 3],
) {
    let c1 = C1_V2 as f32;
    let c2 = C2_V2 as f32;
    // `ssim_d_local_v` — identical op order (p = s12, q = ssq).
    let d = {
        let a = 2.0f32 * m1 * m2 + c1;
        let b = m1 * m1 + m2 * m2 + c1;
        let dl = q - m1 * m1 - m2 * m2 + c2;
        if direct {
            let mu_diff = m1 - m2;
            let err_var = (p - mu_diff * mu_diff).max(0.0);
            let lum = if stable { mu_diff * mu_diff } else { b - a };
            ((dl * lum + a * err_var) / (b * dl)).max(0.0)
        } else {
            let cov = p - m1 * m2;
            let c = 2.0f32 * cov + c2;
            let local = (a * c) / (b * dl);
            (1.0f32 - local).max(0.0)
        }
    };
    let d2 = d * d;
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    // `bounded_dissim_v` — `1 − (2ab+c)/(a²+b²+c)`, or revision 7's exact
    // `(a−b)²/(a²+b²+c)`, plain ops in the same order.
    let edge_dissim = crate::feature_v2::bounded_dissim32(diff_src, diff_dst, C_EDGE as f32, stable);
    // `simd_gt`/`blend` — scalar select, same values.
    let (art_i, det_i) = if diff_dst > diff_src {
        (edge_dissim, 0.0f32)
    } else if diff_dst < diff_src {
        (0.0f32, edge_dissim)
    } else {
        (0.0f32, 0.0f32)
    };
    let raw_diff = s - dd;
    // `saturate_v(raw_diff², c_mse)` — `.max(0)` then `x/(x+c)`.
    let raw_sq = raw_diff * raw_diff;
    let mse_i = raw_sq.max(0.0) / (raw_sq.max(0.0) + C_MSE as f32);
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    // `bounded_excess_pair_v` — ONE denom, TWO divisions (no recip-mul).
    let (hfg, hfl) = {
        let a = hf_dst * hf_dst;
        let b = hf_src * hf_src;
        let denom = a + b + C_HF as f32;
        ((a - b).max(0.0) / denom, (b - a).max(0.0) / denom)
    };
    let hfm = (hf_src.abs() - hf_dst.abs()).max(0.0) / (hf_src.abs() + hf_dst.abs() + C_HF as f32);
    let rae = raw_diff.abs();
    // `pjnd_transducer_v` — `rae / (rae + c*(1 + k*act))`.
    let pj = rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK as f32 * act));
    let (pjl, pjh) = if transducer_bank {
        (
            rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK_LOW as f32 * act)),
            rae / (rae + C_PJND_CLAMP as f32 * (1.0f32 + K_PJND_MASK_HIGH as f32 * act)),
        )
    } else {
        (0.0, 0.0)
    };
    let t = [
        d,
        d2,
        d2 * d,
        d2 * d2,
        art_i,
        det_i,
        mse_i,
        hfg,
        hfl,
        hfm,
        pj,
        pjl,
        pjh,
    ];
    // `pools!` — mask/IW weights + soft-peak saliencies, f32.
    let sat = act.max(0.0) / (act.max(0.0) + C_ACTIVITY as f32);
    let m_w = 1.0f32 - sat;
    let i_w = sat + IW_WEIGHT_FLOOR as f32;
    let vals = [d, art_i, det_i, mse_i];
    let mut mv = [0.0f32; 4];
    let mut iv = [0.0f32; 4];
    for j in 0..4 {
        mv[j] = m_w * vals[j];
        iv[j] = i_w * vals[j];
    }
    let mut kn = [0.0f32; 3];
    let mut kd = [0.0f32; 3];
    for j in 0..3 {
        let sal = vals[j].max(0.0) / (vals[j].max(0.0) + C_PEAK as f32);
        kn[j] = sal * vals[j];
        kd[j] = sal;
    }
    (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd)
}

/// `Pool`-generic scalar dense kernel — see the module note above.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon<P: crate::featcanon::Pool>(
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    use crate::featcanon::{Lanes16F32, ROW_LANES_FOLD};
    let direct = revision >= FormulaRevision::Rev5;
    let stable_moments = revision >= FormulaRevision::Rev5;
    let forms7 = revision.stable_forms();
    // Revision 7's row lanes (`crate::featcanon::Lanes16F32`): the columns of
    // the row's complete eight-column chunks add in f32 lanes, folded into the
    // f64 pools every `ROW_LANES_FOLD` columns and at the end of the chunks;
    // the f64 pools live for the band and finish once per band; the central
    // moments are two-pass per row.
    let rev7 = revision.row_lanes();
    let chunk_end = (width / 8) * 8;
    let mut dvals = if rev7 { vec![0.0f64; width] } else { Vec::new() };
    let mut acc = DenseAccum::default();
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];
        // Revision 7: the f64 lanes live for the whole band (each row's row
        // lanes fold into them), and each pool is finished once per band.
        let mut band_pools = [P::zero(); 29];
        for y in b0..b1 {
            let row = y * width;
            // `[r 0..13 | mw | iw | m 0..4 | i 0..4 | kn 0..3 | kd 0..3]`, the
            // vector body's layout.
            let mut row_pools = [P::zero(); 29];
            let pools = if rev7 {
                &mut band_pools
            } else {
                &mut row_pools
            };
            let (r, rest) = pools.split_at_mut(13);
            let (mw_iw, rest) = rest.split_at_mut(2);
            let (p_m, rest) = rest.split_at_mut(4);
            let (p_i, rest) = rest.split_at_mut(4);
            let (p_kn, p_kd) = rest.split_at_mut(3);
            let (p_mw, p_iw) = mw_iw.split_at_mut(1);
            let r: &mut [P; 13] = r.try_into().unwrap();
            let (p_mw, p_iw) = (&mut p_mw[0], &mut p_iw[0]);
            let (p_m, p_i): (&mut [P; 4], &mut [P; 4]) =
                (p_m.try_into().unwrap(), p_i.try_into().unwrap());
            let (p_kn, p_kd): (&mut [P; 3], &mut [P; 3]) =
                (p_kn.try_into().unwrap(), p_kd.try_into().unwrap());
            let mut local = [0.0f64; 16];
            let mut r32 = [Lanes16F32::default(); 13];
            let (mut p_mw32, mut p_iw32) = (Lanes16F32::default(), Lanes16F32::default());
            let (mut p_m32, mut p_i32) = ([Lanes16F32::default(); 4], [Lanes16F32::default(); 4]);
            let (mut p_kn32, mut p_kd32) = ([Lanes16F32::default(); 3], [Lanes16F32::default(); 3]);
            for x in 0..width {
                let i = row + x;
                let d = if rev7 && x < chunk_end {
                    let d = dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        &mut r32,
                        &mut p_mw32,
                        &mut p_iw32,
                        &mut p_m32,
                        &mut p_i32,
                        &mut p_kn32,
                        &mut p_kd32,
                    );
                    if (x + 1).is_multiple_of(ROW_LANES_FOLD) || x + 1 == chunk_end {
                        for (l32, l) in r32.iter_mut().zip(r.iter_mut()) {
                            l32.fold_into(l);
                        }
                        p_mw32.fold_into(p_mw);
                        p_iw32.fold_into(p_iw);
                        for (l32, l) in p_m32.iter_mut().zip(p_m.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_i32.iter_mut().zip(p_i.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_kn32.iter_mut().zip(p_kn.iter_mut()) {
                            l32.fold_into(l);
                        }
                        for (l32, l) in p_kd32.iter_mut().zip(p_kd.iter_mut()) {
                            l32.fold_into(l);
                        }
                    }
                    d
                } else {
                    dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        r,
                        p_mw,
                        p_iw,
                        p_m,
                        p_i,
                        p_kn,
                        p_kd,
                    )
                };
                if rev7 {
                    dvals[x] = d as f64;
                } else if stable_moments {
                    local[x & 15] = d as f64;
                    if x & 15 == 15 || x + 1 == width {
                        acc.central
                            .merge(OnlineMoments::block(&local[..(x & 15) + 1]));
                    }
                }
            }
            if rev7 {
                acc.central.merge(row_moments_canon(&dvals));
            } else {
                dense_band_add(&mut band, &row_pools, rev7);
            }
        }
        if rev7 {
            dense_band_add(&mut band, &band_pools, rev7);
        }
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

/// The `f32x8` sibling of [`dense_terms32`] — identical op order (the
/// scalar mirror was written to era-2's `terms!`/`pools!`; this calls the
/// same `*_v` helpers). All plain unfused ops — no `mul_add` anywhere —
/// so every lane IS one scalar element's rounding.
#[inline(always)]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn dense_terms32_v<V: Vf32>(
    token: V::Tok,
    s: V,
    dd: V,
    m1: V,
    m2: V,
    q: V,
    p: V,
    act: V,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
) -> (
    [V; 13],
    V,
    V,
    V,
    V,
    V,
    [V; 4],
    [V; 4],
    [V; 3],
    [V; 3],
    V,
) {
    let c1 = V::splat_t(token, C1_V2 as f32);
    let c2 = V::splat_t(token, C2_V2 as f32);
    let c_edge = V::splat_t(token, C_EDGE as f32);
    let c_mse = V::splat_t(token, C_MSE as f32);
    let c_hf = V::splat_t(token, C_HF as f32);
    let c_pj = V::splat_t(token, C_PJND_CLAMP as f32);
    let k_mid = V::splat_t(token, K_PJND_MASK as f32);
    let k_lo = V::splat_t(token, K_PJND_MASK_LOW as f32);
    let k_hi = V::splat_t(token, K_PJND_MASK_HIGH as f32);
    let c_act = V::splat_t(token, C_ACTIVITY as f32);
    let c_peak = V::splat_t(token, C_PEAK as f32);
    let iw_floor = V::splat_t(token, IW_WEIGHT_FLOOR as f32);
    let one = V::splat_t(token, 1.0);
    let zero = V::zero_t(token);
    let d = ssim_d_local_v(token, m1, m2, p, q, c1, c2, direct, stable);
    let d2 = d * d;
    let diff_src = (s - m1).abs();
    let diff_dst = (dd - m2).abs();
    let edge_dissim = bounded_dissim_v(token, diff_src, diff_dst, c_edge, stable);
    let gt = diff_dst.simd_gt(diff_src);
    let lt = diff_dst.simd_lt(diff_src);
    let art_i = V::blend(gt, edge_dissim, zero);
    let det_i = V::blend(lt, edge_dissim, zero);
    let raw_diff = s - dd;
    let mse_i = saturate_v(token, raw_diff * raw_diff, c_mse);
    let hf_src = s - m1;
    let hf_dst = dd - m2;
    let (hfg, hfl) = bounded_excess_split_v(token, hf_dst * hf_dst, hf_src * hf_src, c_hf);
    let hfm = bounded_excess_v(token, hf_src.abs(), hf_dst.abs(), c_hf);
    let rae = raw_diff.abs();
    let pj = pjnd_transducer_v(token, rae, act, k_mid, c_pj);
    let (pjl, pjh) = if transducer_bank {
        (
            pjnd_transducer_v(token, rae, act, k_lo, c_pj),
            pjnd_transducer_v(token, rae, act, k_hi, c_pj),
        )
    } else {
        (zero, zero)
    };
    let t = [
        d,
        d2,
        d2 * d,
        d2 * d2,
        art_i,
        det_i,
        mse_i,
        hfg,
        hfl,
        hfm,
        pj,
        pjl,
        pjh,
    ];
    let sat = saturate_v(token, act, c_act);
    let m_w = one - sat;
    let i_w = sat + iw_floor;
    let vals = [d, art_i, det_i, mse_i];
    let mut mv = [zero; 4];
    let mut iv = [zero; 4];
    for j in 0..4 {
        mv[j] = m_w * vals[j];
        iv[j] = i_w * vals[j];
    }
    // `art_i` and `det_i` are `edge_dissim` or `+0`, never both nonzero, and
    // `saturate(+0)` is `+0`: one division gives both soft-peak saturations.
    let sal_edge = saturate_v(token, edge_dissim, c_peak);
    let sal = [
        saturate_v(token, d, c_peak),
        V::blend(gt, sal_edge, zero),
        V::blend(lt, sal_edge, zero),
    ];
    let mut kn = [zero; 3];
    let mut kd = [zero; 3];
    for j in 0..3 {
        kn[j] = sal[j] * vals[j];
        kd[j] = sal[j];
    }
    (t, art_i, det_i, mse_i, m_w, i_w, mv, iv, kn, kd, act)
}

/// One column's contribution to the dense canonical pools — the scalar
/// element step shared by [`dense_block_kernel_canon`]'s `Pool`-generic
/// body and the canon64 vector body's `width % 8` tail.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dense_elem_canon<P: crate::featcanon::Pool>(
    s: f32,
    dd: f32,
    m1: f32,
    m2: f32,
    q: f32,
    p: f32,
    act: f32,
    x: usize,
    transducer_bank: bool,
    direct: bool,
    stable: bool,
    r: &mut [P; 13],
    p_mw: &mut P,
    p_iw: &mut P,
    p_m: &mut [P; 4],
    p_i: &mut [P; 4],
    p_kn: &mut [P; 3],
    p_kd: &mut [P; 3],
) -> f32 {
    let lane = x & (P::LANES - 1);
    let (t, _art_i, _det_i, _mse_i, m_w, i_w, mv, iv, kn, kd) =
        dense_terms32(s, dd, m1, m2, q, p, act, transducer_bank, direct, stable);
    for j in 0..13 {
        r[j].add(lane, t[j]);
    }
    p_mw.add(lane, m_w);
    p_iw.add(lane, i_w);
    for j in 0..4 {
        p_m[j].add(lane, mv[j]);
        p_i[j].add(lane, iv[j]);
    }
    for j in 0..3 {
        p_kn[j].add(lane, kn[j]);
        p_kd[j].add(lane, kd[j]);
    }
    t[0]
}

/// **canon64 vector body** — [`dense_block_kernel_canon`]'s `P = LanesF64`
/// arm with the term evaluation run eight columns at a time through
/// `f32x8` and the `width % 8` tail on [`dense_elem_canon`]. Every lane is
/// one column's scalar formula (all unfused ops; the `simd_gt`/`blend`
/// selection is the scalar `if/else` — `gt`/`lt` are both false on
/// equality, matching `(0, 0)`); `x & 7` chunk lanes are the canonical
/// lanes in order, so each `add_chunk` is the same eight `add(l, v)` calls
/// in the same per-pool order. `rev4_dense_pixel` stays scalar per
/// element (its `scatter` cannot vectorize). `_scalar`/`_wasm128` keep the
/// scalar canonical body.
// The AVX-512 builds keep the per-element pool scatter (`rev4_dense_pixel`,
// the `add(l, v)` lane pools) scalar — LLVM otherwise fuses the 29 pool
// updates into `vgatherdps`/`vscatterqpd` (4.3x slower than v3, measured
// 2026-10-03 on the separate-pool form). The chunk update below goes
// through `f64x8` repr ops (same `vwin_slide64x8!` pattern as fused.rs):
// each pool is one vector load/add/store, so there is no scalar per-lane
// RMW for LLVM to vectorize into gather/scatter.
/// Chunks of eight columns the sixteen-lane dense pools buffer between pool
/// updates (even, so a block starts on lanes 0..8; see the kernel below).
const DENSE_ILV: usize = 16;

/// The dense pools revision 7 accumulates: all but `d²`, `d³` and `d⁴`
/// (pools 1–3). From revision 5 on the SSIM deviation features read the
/// central moments, and nothing reads those three raw sums, so revision 7
/// leaves `sum_d2`, `sum_d3` and `sum_d4` at zero.
const DENSE_POOLS7: [usize; 26] = [
    0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
];

/// Add the 29 dense pools' finished sums to a band's 35 sums (`[r 0..13 | kn,kd
/// pairs | m,mw pairs | i,iw pairs]`; the mean-weight and importance-weight
/// sums are each the denominator of four weighted pools). Revision 7
/// (`rev7`) skips the raw `d²`, `d³`, `d⁴` pools ([`DENSE_POOLS7`]).
#[inline(always)]
fn dense_band_add<P: crate::featcanon::Pool>(band: &mut [f64; 35], pools: &[P; 29], rev7: bool) {
    for j in 0..13 {
        if rev7 && (1..4).contains(&j) {
            continue;
        }
        band[j] += pools[j].fin();
    }
    for j in 0..3 {
        band[13 + j * 2] += pools[23 + j].fin();
        band[14 + j * 2] += pools[26 + j].fin();
    }
    let mw = pools[13].fin();
    let iw = pools[14].fin();
    for j in 0..4 {
        band[19 + j * 2] += pools[15 + j].fin();
        band[20 + j * 2] += mw;
        band[27 + j * 2] += pools[19 + j].fin();
        band[28 + j * 2] += iw;
    }
}

#[magetypes(define(f32x8, f32x16, f64x8), v4x, v4, v3, -scalar)]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec<P: crate::featcanon::Pool64, const R7: bool>(
    token: Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    let direct = revision >= FormulaRevision::Rev5;
    let stable_moments = P::LANES == 16;
    let forms7 = revision.stable_forms();
    let mut acc = DenseAccum::default();
    // Provable-length plane slices: the chunk loop reads `i + 8 <= row +
    // full * 8` and the tail reads `row + x < height * width` — both literal
    // once the planes are cut to their exact spans, so no per-element
    // `slice_index_fail` path remains.
    let src = &src[..height * width];
    let dst = &dst[..height * width];
    let mu1 = &mu1[..height * width];
    let mu2 = &mu2[..height * width];
    let ssq = &ssq[..height * width];
    let s12 = &s12[..height * width];
    let activity = &activity[..height * width];
    // Blocked interchange for the sixteen-lane pools (DENSEPOOL,
    // `benchmarks/dense_pool_brief_2026-10-24.md`): each chunk's 29 term
    // vectors go to an L1-resident buffer, and every `DENSE_ILV` chunks each
    // pool loads its two lane halves once, adds the buffered chunks in order
    // and stores once. Lane `l` still adds column `x` (`x & 15 == l`) in
    // ascending `x` from the row's zero, so every sum is bit-identical; only
    // the pool memory traffic changes (a pool round trip per chunk was most
    // of this kernel's time). The eight-lane pools keep the per-chunk update.
    let mut ilv = [[0.0f32; 8]; 29 * DENSE_ILV];
    // The stable central moments of `d` (Rev5+): eight full 16-column
    // blocks at a time, one block per `f64x8` lane (`cols[k][b]` is value
    // `k` of block `b`). Each lane runs `OnlineMoments::block`'s scalar
    // sequence op for op (the same `-0.0`-seeded sum of `v − v₀`, mean and
    // `m2/m3/m4` adds), and the blocks merge in raster order, so the moments
    // are bit-identical; the serial per-block chain is what costs.
    let mut cols = [[0.0f64; 8]; 16];
    let central8 = |cols: &[[f64; 8]; 16], blocks: usize, central: &mut OnlineMoments| {
        if blocks == 0 {
            return;
        }
        let origin = f64x8::load_t(token, &cols[0]);
        let mut sum = f64x8::splat_t(token, -0.0);
        for col in cols {
            sum += f64x8::load_t(token, col) - origin;
        }
        let mean = origin + sum / f64x8::splat_t(token, 16.0);
        let zero = f64x8::zero_t(token);
        let (mut m2, mut m3, mut m4) = (zero, zero, zero);
        for col in cols {
            let d = f64x8::load_t(token, col) - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        let (mean, m2, m3, m4) = (mean.to_array(), m2.to_array(), m3.to_array(), m4.to_array());
        for b in 0..blocks {
            central.merge(OnlineMoments {
                n: 16.0,
                mean: mean[b],
                m2: m2[b],
                m3: m3[b],
                m4: m4[b],
            });
        }
    };
    // Revision 7 (`rowlanes`, `crate::featcanon::Lanes16F32`): the 29 pools
    // add each complete chunk's columns in f32 row lanes, folded into the f64
    // lanes every `ROW_LANES_FOLD` columns and before the tail (the f64 lanes
    // live for the band, finished once per band), and the central moments of
    // `d` are two-pass per row (`row_moments_canon`'s ops, lane-parallel)
    // with one merge per row. Revisions 5 and 6 keep their bits.
    // `R7` is `revision.row_lanes()` at compile time, so a revision-7 instance
    // carries none of the eight-lane and interleave buffers below.
    let rev7 = R7;
    debug_assert_eq!(rev7, revision.row_lanes());
    debug_assert!(!rev7 || stable_moments, "revision 7 runs on the sixteen-lane pools");
    let mut dvals = if rev7 { vec![0.0f64; width] } else { Vec::new() };
    let row_moments = |v: &[f64]| -> OnlineMoments {
        let n = v.len();
        let full16 = n / 16;
        let zero = f64x8::zero_t(token);
        let lanes = |lo: f64x8, hi: f64x8| -> [f64; 16] {
            let mut l = [0.0f64; 16];
            l[..8].copy_from_slice(&lo.to_array());
            l[8..].copy_from_slice(&hi.to_array());
            l
        };
        let (mut lo, mut hi) = (zero, zero);
        for b in 0..full16 {
            lo += f64x8::load_t(token, v[16 * b..16 * b + 8].try_into().unwrap());
            hi += f64x8::load_t(token, v[16 * b + 8..16 * b + 16].try_into().unwrap());
        }
        let mut s1 = lanes(lo, hi);
        for x in full16 * 16..n {
            s1[x & 15] += v[x];
        }
        let mean = crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s1)) / n as f64;
        let m = f64x8::splat_t(token, mean);
        let (mut a2, mut b2, mut a3, mut b3, mut a4, mut b4) = (zero, zero, zero, zero, zero, zero);
        for b in 0..full16 {
            let dl = f64x8::load_t(token, v[16 * b..16 * b + 8].try_into().unwrap()) - m;
            let dh = f64x8::load_t(token, v[16 * b + 8..16 * b + 16].try_into().unwrap()) - m;
            let (ql, qh) = (dl * dl, dh * dh);
            a2 += ql;
            b2 += qh;
            a3 += ql * dl;
            b3 += qh * dh;
            a4 += ql * ql;
            b4 += qh * qh;
        }
        let (mut s2, mut s3, mut s4) = (lanes(a2, b2), lanes(a3, b3), lanes(a4, b4));
        for x in full16 * 16..n {
            let d = v[x] - mean;
            let q = d * d;
            s2[x & 15] += q;
            s3[x & 15] += q * d;
            s4[x & 15] += q * q;
        }
        OnlineMoments {
            n: n as f64,
            mean,
            m2: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s2)),
            m3: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s3)),
            m4: crate::featcanon::Pool::fin(crate::featcanon::Lanes16F64(s4)),
        }
    };
    // Revision 7's f32 row lanes; every row's fold clears them.
    let mut pools32 = [crate::featcanon::Lanes16F32::default(); 29];
    let mut b0 = 0usize;
    while b0 < height {
        let b1 = (b0 + ERA2_BAND_ROWS).min(height);
        let mut band = [0.0f64; 35];
        // Revision 7: the f64 lanes live for the whole band (each row's row
        // lanes fold into them), and each pool is finished once per band.
        let mut band_pools = [P::zero(); 29];
        for y in b0..b1 {
            let row = y * width;
            // One flat accumulator row: `[r 0..13 | mw | iw | m 0..4 |
            // i 0..4 | kn 0..3 | kd 0..3]`. The chunk update goes through
            // `f64x8` repr ops — one vector load/add/store per pool — so
            // there is no scalar per-lane RMW for the AVX-512 build to
            // fuse into `vgatherqpd`/`vscatterqpd` (the 4.3x regression
            // the former v3-only dispatch worked around). The tail slices
            // give `dense_elem_canon` the same typed pool refs — zero cost.
            let mut row_pools = [P::zero(); 29];
            let pools = if rev7 { &mut band_pools } else { &mut row_pools };
            let mut local = [0.0f64; 16];
            let mut blocks = 0usize;
            let full = width / 8;
            let srow = &src[row..row + full * 8];
            let drow = &dst[row..row + full * 8];
            let m1row = &mu1[row..row + full * 8];
            let m2row = &mu2[row..row + full * 8];
            let qrow = &ssq[row..row + full * 8];
            let prow = &s12[row..row + full * 8];
            let arow = &activity[row..row + full * 8];
            macro_rules! chunk8_post {
                ($c:expr, $bufs:expr, $act8:expr) => {{
                    let c: usize = $c;
                    let i = c * 8;
                    let bufs: [[f32; 8]; 29] = $bufs;
                if rev7 {
                    let off = i & 15;
                    for j in DENSE_POOLS7 {
                        let lanes = pools32[j].chunk8(off);
                        (f32x8::load_t(token, lanes) + f32x8::load_t(token, &bufs[j])).store(lanes);
                    }
                    let dv: &mut [f64; 8] = (&mut dvals[i..i + 8]).try_into().unwrap();
                    for l in 0..8 {
                        dv[l] = bufs[0][l] as f64;
                    }
                    if (i + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else if stable_moments {
                    // Chunk `c` adds to lanes `(c & 1) * 8..`; a block starts
                    // at an even chunk (`DENSE_ILV` is even), so a buffered
                    // chunk's parity is its position's.
                    let k = c % DENSE_ILV;
                    for j in 0..29 {
                        ilv[j * DENSE_ILV + k] = bufs[j];
                    }
                    if k + 1 == DENSE_ILV || c + 1 == full {
                        for j in 0..29 {
                            let blk = &ilv[j * DENSE_ILV..j * DENSE_ILV + k + 1];
                            let mut lo = f64x8::load_t(token, pools[j].chunk8(0));
                            let mut hi = f64x8::load_t(token, pools[j].chunk8(8));
                            let (pairs, rest) = blk.as_chunks::<2>();
                            for [a, b] in pairs {
                                let a: [f64; 8] = std::array::from_fn(|l| a[l] as f64);
                                let b: [f64; 8] = std::array::from_fn(|l| b[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                                hi += f64x8::from_array_t(token, b);
                            }
                            if let [last] = rest {
                                let a: [f64; 8] = std::array::from_fn(|l| last[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                            }
                            lo.store(pools[j].chunk8(0));
                            hi.store(pools[j].chunk8(8));
                        }
                    }
                } else {
                    for j in 0..29 {
                        let a: [f64; 8] = std::array::from_fn(|l| bufs[j][l] as f64);
                        let lanes = pools[j].chunk8(i & (P::LANES - 1));
                        (f64x8::load_t(token, lanes) + f64x8::from_array_t(token, a))
                            .store(lanes);
                    }
                }
                if stable_moments && !rev7 {
                    let off = i & 15;
                    for l in 0..8 {
                        local[off + l] = bufs[0][l] as f64;
                        cols[off + l][blocks] = bufs[0][l] as f64;
                    }
                    if off == 8 {
                        blocks += 1;
                        if blocks == 8 {
                            central8(&cols, blocks, &mut acc.central);
                            blocks = 0;
                        }
                    } else if c + 1 == full && width.is_multiple_of(8) {
                        // The row ends on an eight-value block.
                        central8(&cols, blocks, &mut acc.central);
                        blocks = 0;
                        acc.central.merge(OnlineMoments::block(&local[..8]));
                    }
                }
                }};
            }
            let mut c = 0usize;
            while rev7 && steps_sixteen::<Token>() && c + 2 <= full {
                let i = c * 8;
                macro_rules! ld16 {
                    ($r:expr) => {
                        f32x16::load_t(token, $r[i..i + 16].try_into().unwrap())
                    };
                }
                let (t, _, _, _, m_w, i_w, mv, iv, kn, kd, act_v) = dense_terms32_v(
                    token,
                    ld16!(srow),
                    ld16!(drow),
                    ld16!(m1row),
                    ld16!(m2row),
                    ld16!(qrow),
                    ld16!(prow),
                    ld16!(arow),
                    transducer_bank,
                    direct,
                    forms7,
                );
                let vecs: [f32x16; 29] = [
                    t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12],
                    m_w, i_w, mv[0], mv[1], mv[2], mv[3], iv[0], iv[1], iv[2], iv[3], kn[0], kn[1],
                    kn[2], kd[0], kd[1], kd[2],
                ];
                if rev7 {
                    for j in DENSE_POOLS7 {
                        let lanes = &mut pools32[j].0;
                        (f32x16::load_t(token, lanes) + vecs[j]).store(lanes);
                    }
                    let d = vecs[0].to_array();
                    let dv: &mut [f64; 16] = (&mut dvals[i..i + 16]).try_into().unwrap();
                    for l in 0..16 {
                        dv[l] = d[l] as f64;
                    }
                    if (i + 16).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else {
                    let arrs: [[f32; 16]; 29] = vecs.map(|v| v.to_array());
                    let act16 = act_v.to_array();
                    for h in 0..2 {
                        let bufs: [[f32; 8]; 29] =
                            std::array::from_fn(|j| arrs[j][8 * h..8 * h + 8].try_into().unwrap());
                        let _act8: [f32; 8] = act16[8 * h..8 * h + 8].try_into().unwrap();
                        chunk8_post!(c + h, bufs, &act8);
                    }
                }
                c += 2;
            }
            // The remaining chunk (an odd count) and every chunk on v3: the
            // eight-wide step, unchanged.
            for c in c..full {
                let i = c * 8;
                let sv = f32x8::load_t(token, srow[i..i + 8].try_into().unwrap());
                let dv = f32x8::load_t(token, drow[i..i + 8].try_into().unwrap());
                let m1v = f32x8::load_t(token, m1row[i..i + 8].try_into().unwrap());
                let m2v = f32x8::load_t(token, m2row[i..i + 8].try_into().unwrap());
                let qv = f32x8::load_t(token, qrow[i..i + 8].try_into().unwrap());
                let pv = f32x8::load_t(token, prow[i..i + 8].try_into().unwrap());
                let actv = f32x8::load_t(token, arow[i..i + 8].try_into().unwrap());
                let (t, _art_i, _det_i, _mse_i, m_w, i_w, mv, iv, kn, kd, _act_v) = dense_terms32_v(
                    token,
                    sv,
                    dv,
                    m1v,
                    m2v,
                    qv,
                    pv,
                    actv,
                    transducer_bank,
                    direct,
                    forms7,
                );
                let bufs: [[f32; 8]; 29] = [
                    t[0].to_array(),
                    t[1].to_array(),
                    t[2].to_array(),
                    t[3].to_array(),
                    t[4].to_array(),
                    t[5].to_array(),
                    t[6].to_array(),
                    t[7].to_array(),
                    t[8].to_array(),
                    t[9].to_array(),
                    t[10].to_array(),
                    t[11].to_array(),
                    t[12].to_array(),
                    m_w.to_array(),
                    i_w.to_array(),
                    mv[0].to_array(),
                    mv[1].to_array(),
                    mv[2].to_array(),
                    mv[3].to_array(),
                    iv[0].to_array(),
                    iv[1].to_array(),
                    iv[2].to_array(),
                    iv[3].to_array(),
                    kn[0].to_array(),
                    kn[1].to_array(),
                    kn[2].to_array(),
                    kd[0].to_array(),
                    kd[1].to_array(),
                    kd[2].to_array(),
                ];
                if rev7 {
                    let off = i & 15;
                    for j in DENSE_POOLS7 {
                        let lanes = pools32[j].chunk8(off);
                        (f32x8::load_t(token, lanes) + f32x8::load_t(token, &bufs[j])).store(lanes);
                    }
                    let dv: &mut [f64; 8] = (&mut dvals[i..i + 8]).try_into().unwrap();
                    for l in 0..8 {
                        dv[l] = bufs[0][l] as f64;
                    }
                    if (i + 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                        for j in DENSE_POOLS7 {
                            pools32[j].fold_into(&mut pools[j]);
                        }
                    }
                } else if stable_moments {
                    // Chunk `c` adds to lanes `(c & 1) * 8..`; a block starts
                    // at an even chunk (`DENSE_ILV` is even), so a buffered
                    // chunk's parity is its position's.
                    let k = c % DENSE_ILV;
                    for j in 0..29 {
                        ilv[j * DENSE_ILV + k] = bufs[j];
                    }
                    if k + 1 == DENSE_ILV || c + 1 == full {
                        for j in 0..29 {
                            let blk = &ilv[j * DENSE_ILV..j * DENSE_ILV + k + 1];
                            let mut lo = f64x8::load_t(token, pools[j].chunk8(0));
                            let mut hi = f64x8::load_t(token, pools[j].chunk8(8));
                            let (pairs, rest) = blk.as_chunks::<2>();
                            for [a, b] in pairs {
                                let a: [f64; 8] = std::array::from_fn(|l| a[l] as f64);
                                let b: [f64; 8] = std::array::from_fn(|l| b[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                                hi += f64x8::from_array_t(token, b);
                            }
                            if let [last] = rest {
                                let a: [f64; 8] = std::array::from_fn(|l| last[l] as f64);
                                lo += f64x8::from_array_t(token, a);
                            }
                            lo.store(pools[j].chunk8(0));
                            hi.store(pools[j].chunk8(8));
                        }
                    }
                } else {
                    for j in 0..29 {
                        let a: [f64; 8] = std::array::from_fn(|l| bufs[j][l] as f64);
                        let lanes = pools[j].chunk8(i & (P::LANES - 1));
                        (f64x8::load_t(token, lanes) + f64x8::from_array_t(token, a))
                            .store(lanes);
                    }
                }
                if stable_moments && !rev7 {
                    let off = i & 15;
                    for l in 0..8 {
                        local[off + l] = bufs[0][l] as f64;
                        cols[off + l][blocks] = bufs[0][l] as f64;
                    }
                    if off == 8 {
                        blocks += 1;
                        if blocks == 8 {
                            central8(&cols, blocks, &mut acc.central);
                            blocks = 0;
                        }
                    } else if c + 1 == full && width.is_multiple_of(8) {
                        // The row ends on an eight-value block.
                        central8(&cols, blocks, &mut acc.central);
                        blocks = 0;
                        acc.central.merge(OnlineMoments::block(&local[..8]));
                    }
                }
            }
            // Full blocks before the tail's own blocks, in raster order.
            central8(&cols, blocks, &mut acc.central);
            // The end of the chunks, unless the last chunk already folded.
            if rev7 && !(full * 8).is_multiple_of(crate::featcanon::ROW_LANES_FOLD) {
                for j in DENSE_POOLS7 {
                    pools32[j].fold_into(&mut pools[j]);
                }
            }
            // Scalar tail — the canonical element step verbatim.
            {
                let (r, rest) = pools.split_at_mut(13);
                let (mw_iw, rest) = rest.split_at_mut(2);
                let (p_m, rest) = rest.split_at_mut(4);
                let (p_i, rest) = rest.split_at_mut(4);
                let (p_kn, p_kd) = rest.split_at_mut(3);
                let (mw, iw) = mw_iw.split_at_mut(1);
                for x in full * 8..width {
                    let i = row + x;
                    let d = dense_elem_canon(
                        src[i],
                        dst[i],
                        mu1[i],
                        mu2[i],
                        ssq[i],
                        s12[i],
                        activity[i],
                        x,
                        transducer_bank,
                        direct,
                        forms7,
                        r.try_into().unwrap(),
                        &mut mw[0],
                        &mut iw[0],
                        p_m.try_into().unwrap(),
                        p_i.try_into().unwrap(),
                        p_kn.try_into().unwrap(),
                        p_kd.try_into().unwrap(),
                    );
                    if rev7 {
                        dvals[x] = d as f64;
                    } else if stable_moments {
                        local[x & 15] = d as f64;
                        if x & 15 == 15 || x + 1 == width {
                            acc.central
                                .merge(OnlineMoments::block(&local[..(x & 15) + 1]));
                        }
                    }
                }
            }
            if rev7 {
                acc.central.merge(row_moments(&dvals[..width]));
            } else {
                dense_band_add(&mut band, pools, rev7);
            }
        }
        if rev7 {
            dense_band_add(&mut band, &band_pools, rev7);
        }
        let s = &mut acc;
        s.sum_d += band[0];
        s.sum_d2 += band[1];
        s.sum_d3 += band[2];
        s.sum_d4 += band[3];
        s.sum_art += band[4];
        s.sum_det += band[5];
        s.sum_mse += band[6];
        s.sum_hf_gain += band[7];
        s.sum_hf_loss += band[8];
        s.sum_hf_mag_loss += band[9];
        s.sum_pjnd += band[10];
        s.sum_pjnd_lo += band[11];
        s.sum_pjnd_hi += band[12];
        for (w, o) in [
            (&mut s.ws_peak_ssim, 13),
            (&mut s.ws_peak_art, 15),
            (&mut s.ws_peak_det, 17),
            (&mut s.ws_mask_ssim, 19),
            (&mut s.ws_mask_art, 21),
            (&mut s.ws_mask_det, 23),
            (&mut s.ws_mask_mse, 25),
            (&mut s.ws_iw_ssim, 27),
            (&mut s.ws_iw_art, 29),
            (&mut s.ws_iw_det, 31),
            (&mut s.ws_iw_mse, 33),
        ] {
            w.num += band[o];
            w.den += band[o + 1];
        }
        b0 = b1;
    }
    acc
}

/// Scalar-tier siblings of [`dense_block_kernel_canon64_vec`]: scalar and
/// wasm128 keep the `LanesF64` canonical body.
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec_scalar<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::ScalarToken,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        revision,
    )
}

/// Wasm128 sibling of [`dense_block_kernel_canon64_vec_scalar`].
#[cfg(target_arch = "wasm32")]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec_wasm128<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::Wasm128Token,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        revision,
    )
}

/// NEON runs the scalar canonical body (2026-10-03): NEON vector `max`/`min` propagate NaN where the canon's `f32::max`/
/// `f64::max` drop it (the REV4VEC CI aarch64 failure), so only the x86_64 fused tiers run a lane-parallel body.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
fn dense_block_kernel_canon64_vec_neon<P: crate::featcanon::Pool64, const R7: bool>(
    _token: archmage::NeonToken,
    src: &[f32],
    dst: &[f32],
    mu1: &[f32],
    mu2: &[f32],
    ssq: &[f32],
    s12: &[f32],
    activity: &[f32],
    width: usize,
    height: usize,
    transducer_bank: bool,
    revision: FormulaRevision,
) -> DenseAccum {
    dense_block_kernel_canon::<P>(
        src,
        dst,
        mu1,
        mu2,
        ssq,
        s12,
        activity,
        width,
        height,
        transducer_bank,
        revision,
    )
}
