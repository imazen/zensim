//! THE per-pixel SSIM dissimilarity — one owner for a form that was
//! hand-copied at 36 kernel sites.
//!
//! # Why this module exists
//!
//! Every SSIM-derived slot in the crate (`basic`'s `ssim_*`, the `masked`
//! block, the `IW` block — 3 signals x 2 blocks x 4 scales x 3 channels, plus
//! the basic block's own) is built from ONE per-pixel expression:
//!
//! ```text
//! num_m   = 1 - (mu1 - mu2)^2          // luminance
//! num_s   = 2*cov + C2                 // structure numerator
//! denom_s = var1 + var2 + C2           // structure denominator
//! d_raw   = 1 - num_m * num_s / denom_s
//! ```
//!
//! That expression was written out by hand **36 times** — 10 in
//! [`crate::fused`] and 26 in [`crate::simd_ops`], across four SIMD tiers and
//! their scalar tails, in vector and scalar spellings. `C2` itself had three
//! separate declarations. A form with 36 owners cannot be revised, which is
//! exactly the position the defect audit found the crate in: F4 (the one live
//! arithmetic defect) is a property of this expression, and there was nowhere
//! to fix it.
//!
//! Everything here is `#[inline(always)]` and generic over the backend trait,
//! following [`crate::fused`]'s `raw_moments_accumulate*` precedent. `nm` over
//! a release build shows zero `ssim_dissim_*` symbols surviving.
//!
//! # F4, and why the fix needs a form rather than a clamp
//!
//! `num_s/denom_s` is bounded in `[-1, 1]` by construction: `|2*cov| <= var1 +
//! var2` (Cauchy-Schwarz), so the structure term cannot run away. `num_m` is
//! `1 - D^2` for a mean difference `D`, which is **unbounded below**, so a
//! large local mean difference makes `d_raw` a large positive number. MEASURED
//! (`benchmarks/ssim_moment_explosion_2026-07-16.md`, 2,322,579 rows):
//! `f313 = iw_ssim_4th s0 ch2` reaches **5,814,302** against a photographic
//! p99.9 of **0.48**.
//!
//! The weights are NOT the amplifier, and the numbers on record prove it: the
//! `masked` weight is `1/(1 + k*a)`, bounded in `(0, 1]`, while the `IW`
//! weight is `1 + k*a` and unbounded — yet `f241` (masked, 5,797,029) and
//! `f313` (IW, 5,814,302) agree to **0.3 %**. A bounded weight cannot produce
//! 5.8e6 from a bounded `d_raw`, and an unbounded weight that mattered could
//! not land within 0.3 % of a bounded one. Both are ~1 there because the
//! pathology lives in flat regions where activity is ~0. The amplifier is
//! `num_m`.
//!
//! # Provenance: this is ssimulacra2's form, faithfully inherited
//!
//! `zensim/src/lib.rs` describes the no-`C1` luminance term as "ssimulacra2's
//! variant". VERIFIED against our own SSIMULACRA2 implementation rather than
//! assumed: `fast-ssim2`'s `simd_ops.rs`, `lib.rs` and `strip.rs` all compute
//! `num_m = mu_diff.mul_add(-mu_diff, 1.0)` with `C2 = 0.0009` and no `C1`.
//! So F4 is **inherited from the algorithm**, not a zensim coding slip, and
//! any bounded form here is a deliberate, measured DEVIATION from that
//! lineage — taken for a learned metric that must feed monotone linear heads,
//! and recorded as such.
//!
//! (`fast-ssim2` carries the same unbounded term. It is a different repo and
//! is NOT touched from here; the observation is reported, not acted on.)

use crate::feature_defs::FormulaRevision;
use archmage::autoversion;
use magetypes::simd::backends::{F32x8Backend, F32x16Backend};
use magetypes::simd::generic::{f32x8 as GenericF32x8, f32x16};

/// **Ablation mirror of [`stable_ssim_plane`], for separating the two halves
/// of the correction.**
///
/// The shipped kernel changed TWO things at once: it accumulates in f64, and
/// it forms the error variance DIRECTLY from a `(a-b)^2` moment instead of
/// recovering it from `var1 + var2 - 2*cov`. Which of those actually restores
/// locality is a question the shipped kernel cannot answer, because it does
/// both. This mirrors its algorithm exactly with the two knobs independent.
///
/// `f32_accum` rounds after every arithmetic operation, which models f32
/// EXACTLY rather than approximately: for `+ - * /` on f32-representable
/// operands, computing in f64 and rounding once to f32 is correctly rounded,
/// because f64's 53 bits exceed the 2p+2 = 50 needed at p = 24.
///
/// `direct_error` false switches the fourth moment to `a*b` and the final
/// expression to the legacy `1 - num_m*(2*cov + C2)/(var1 + var2 + C2)`, i.e.
/// the shipped formulation.
///
/// A test asserts `(f64, direct)` is BIT-IDENTICAL to [`stable_ssim_plane`],
/// so this is a faithful mirror and not a second implementation drifting on
/// its own.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ablation_plane(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    // Bit k set => moment k accumulates in f32. The four moments have very
    // different magnitudes — `a`, `b` and `a^2+b^2` are order 0.5 while
    // `(a-b)^2` is order 1e-6 on near-lossless content — so their f32 drift
    // differs by orders of magnitude and they need not share a width.
    f32_mask: u8,
    direct_error: bool,
    // Re-seed the recurrence exactly every `reset` positions (0 = never).
    // The "periodic stability reset": it does NOT make the sum a function of
    // its own window, but it bounds how far a change propagates — to one tile
    // instead of the whole row/column.
    reset: usize,
) -> Vec<f32> {
    let rk = |k: usize, x: f64| {
        if f32_mask & (1 << k) != 0 {
            x as f32 as f64
        } else {
            x
        }
    };
    // Non-moment arithmetic (the final expression) follows moment 0.
    let r = |x: f64| rk(0, x);
    let diameter = radius * 2 + 1;
    let rad = radius as isize;
    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };
    let at = |x: usize, y: usize| {
        let a = src[y * width + x] as f64;
        let b = dst[y * width + x] as f64;
        let fourth = if direct_error {
            let e = rk(3, a - b);
            rk(3, e * e)
        } else {
            rk(3, a * b)
        };
        [a, b, rk(2, rk(2, a * a) + rk(2, b * b)), fourth]
    };
    let horizontal = |y: usize, output: &mut [[f64; 4]]| {
        let mut sums = [0.0f64; 4];
        for dx in -rad..=rad {
            let m = at(mirror(dx, width), y);
            for k in 0..4 {
                sums[k] = rk(k, sums[k] + m[k]);
            }
        }
        for (x, cell) in output.iter_mut().enumerate() {
            if reset != 0 && x != 0 && x % reset == 0 {
                // Exact re-seed: recompute this window from its own samples.
                sums = [0.0; 4];
                for dx in -rad..=rad {
                    let m = at(mirror(x as isize + dx, width), y);
                    for k in 0..4 {
                        sums[k] = r(sums[k] + m[k]);
                    }
                }
            }
            *cell = sums;
            let add = at(mirror(x as isize + rad + 1, width), y);
            let rem = at(mirror(x as isize - rad, width), y);
            if add != rem {
                for k in 0..4 {
                    sums[k] = rk(k, rk(k, sums[k] + add[k]) - rem[k]);
                }
            }
        }
    };
    let mut rows = vec![[0.0f64; 4]; width * diameter];
    let mut vertical = vec![[0.0f64; 4]; width];
    for row in 0..diameter {
        let y = mirror(row as isize - rad, height);
        let ring = &mut rows[row * width..(row + 1) * width];
        horizontal(y, ring);
        for (sum, add) in vertical.iter_mut().zip(ring) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] + add[k]);
            }
        }
    }
    let mut out = vec![0.0f32; width * height];
    let inv_n = r(1.0 / (diameter as f64 * diameter as f64));
    let mut head = 0;
    for y in 0..height {
        for (value, m) in out[y * width..(y + 1) * width].iter_mut().zip(&vertical) {
            let a = r(m[0] * inv_n);
            let b = r(m[1] * inv_n);
            let mean_error2 = r(r(a - b) * r(a - b));
            let variance_sum = r(r(r(m[2] * inv_n) - r(a * a)) - r(b * b)).max(0.0);
            let luma_loss = match form {
                SsimLumaForm::Ssim2Legacy => mean_error2,
                SsimLumaForm::Clamp => mean_error2.min(1.0),
            };
            *value = if direct_error {
                let error_variance = r(r(m[3] * inv_n) - mean_error2).max(0.0);
                r(luma_loss
                    + r(r(1.0 - luma_loss) * r(error_variance / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
            } else {
                // The SHIPPED formulation: covariance recovered by subtraction.
                let cov = r(r(m[3] * inv_n) - r(a * b));
                r(1.0
                    - r(r(1.0 - luma_loss)
                        * r(r(r(2.0 * cov) + C2 as f64) / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
            };
        }
        if y + 1 == height {
            break;
        }
        if reset != 0 && (y + 1) % reset == 0 {
            // Exact vertical re-seed from the ring's own rows.
            for (x, sum) in vertical.iter_mut().enumerate() {
                *sum = [0.0; 4];
                for row in 0..diameter {
                    for k in 0..4 {
                        sum[k] = rk(k, sum[k] + rows[row * width + x][k]);
                    }
                }
            }
        }
        let ring = &mut rows[head * width..(head + 1) * width];
        for (sum, rem) in vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] - rem[k]);
            }
        }
        horizontal(mirror(y as isize + rad + 1, height), ring);
        for (sum, add) in vertical.iter_mut().zip(ring.iter()) {
            for k in 0..4 {
                sum[k] = rk(k, sum[k] + add[k]);
            }
        }
        head = (head + 1) % diameter;
    }
    out
}

/// **Tiled (van Herk / Gil-Werman) mirror of the moment pass.**
///
/// The sliding recurrence's locality failure is that the window sum at `x`
/// depends on every sample the running sum has passed over, not on the window.
/// Resetting the recurrence periodically bounds that, but does not remove it —
/// a change still perturbs the rest of its own tile.
///
/// This decomposition removes it outright. With tiles of exactly the window
/// diameter `D`, and per-tile `prefix` / `suffix` running sums, the window at
/// `x` is `suffix[x] + prefix[x + D - 1]`: the suffix reads only `x ..= tile
/// end`, the prefix only `next tile start ..= x + D - 1`, and both ranges lie
/// INSIDE the window. So the sum is a deterministic function of exactly the
/// window's own samples, in an order fixed by the window's offset against the
/// tile grid. Identical window contents therefore give a bit-identical sum at
/// ANY precision — locality stops being a numerical property and becomes a
/// structural one.
///
/// It is also cheaper where it counts: the serial dependency chain shrinks
/// from the whole row to `D`, and tiles are independent, so the prefix/suffix
/// passes vectorize across tiles instead of being latency-bound.
///
/// Same knobs as [`ablation_plane`], so the 2x2 can be re-run against this
/// structure.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ablation_plane_tiled(
    src: &[f32],
    dst: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    form: SsimLumaForm,
    f32_accum: bool,
    direct_error: bool,
) -> Vec<f32> {
    let r = |x: f64| if f32_accum { x as f32 as f64 } else { x };
    let d = radius * 2 + 1;
    let rad = radius as isize;
    let mirror = |i: isize, len: usize| {
        if i >= 0 && (i as usize) < len {
            i as usize
        } else {
            crate::metric::reflect_index(i.unsigned_abs(), len)
        }
    };
    let moment = |x: usize, y: usize| {
        let a = src[y * width + x] as f64;
        let b = dst[y * width + x] as f64;
        let fourth = if direct_error {
            let e = r(a - b);
            r(e * e)
        } else {
            r(a * b)
        };
        [a, b, r(r(a * a) + r(b * b)), fourth]
    };

    /// One van Herk pass over `n` outputs whose window is `padded[i ..= i+d-1]`.
    fn van_herk(padded: &[f64], n: usize, d: usize, r: &dyn Fn(f64) -> f64) -> Vec<f64> {
        let len = padded.len();
        let mut prefix = vec![0.0f64; len];
        let mut suffix = vec![0.0f64; len];
        let mut i = 0;
        while i < len {
            let end = (i + d).min(len);
            // prefix: left-to-right within the tile
            let mut acc = 0.0;
            for j in i..end {
                acc = r(acc + padded[j]);
                prefix[j] = acc;
            }
            // suffix: right-to-left within the tile
            acc = 0.0;
            for j in (i..end).rev() {
                acc = r(acc + padded[j]);
                suffix[j] = acc;
            }
            i = end;
        }
        (0..n)
            .map(|x| {
                let last = x + d - 1;
                if x % d == 0 {
                    suffix[x]
                } else {
                    r(suffix[x] + prefix[last])
                }
            })
            .collect()
    }

    // Horizontal: one padded row per moment, reflect-101 at the plane edge.
    let mut hrow = vec![0.0f64; width * height * 4];
    let mut padded = vec![0.0f64; width + 2 * radius];
    for y in 0..height {
        for k in 0..4 {
            for (i, cell) in padded.iter_mut().enumerate() {
                let x = mirror(i as isize - rad, width);
                *cell = moment(x, y)[k];
            }
            let out = van_herk(&padded, width, d, &r);
            for (x, v) in out.into_iter().enumerate() {
                hrow[(y * width + x) * 4 + k] = v;
            }
        }
    }

    // Vertical: same decomposition down each column of the row sums.
    let mut vsum = vec![0.0f64; width * height * 4];
    let mut col = vec![0.0f64; height + 2 * radius];
    for x in 0..width {
        for k in 0..4 {
            for (i, cell) in col.iter_mut().enumerate() {
                let y = mirror(i as isize - rad, height);
                *cell = hrow[(y * width + x) * 4 + k];
            }
            let out = van_herk(&col, height, d, &r);
            for (y, v) in out.into_iter().enumerate() {
                vsum[(y * width + x) * 4 + k] = v;
            }
        }
    }

    let inv_n = r(1.0 / (d as f64 * d as f64));
    let mut out = vec![0.0f32; width * height];
    for (i, value) in out.iter_mut().enumerate() {
        let m = &vsum[i * 4..i * 4 + 4];
        let a = r(m[0] * inv_n);
        let b = r(m[1] * inv_n);
        let mean_error2 = r(r(a - b) * r(a - b));
        let variance_sum = r(r(r(m[2] * inv_n) - r(a * a)) - r(b * b)).max(0.0);
        let luma_loss = match form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(1.0),
        };
        *value = if direct_error {
            let error_variance = r(r(m[3] * inv_n) - mean_error2).max(0.0);
            r(luma_loss + r(r(1.0 - luma_loss) * r(error_variance / r(variance_sum + C2 as f64))))
                .max(0.0) as f32
        } else {
            let cov = r(r(m[3] * inv_n) - r(a * b));
            r(1.0
                - r(r(1.0 - luma_loss)
                    * r(r(r(2.0 * cov) + C2 as f64) / r(variance_sum + C2 as f64))))
            .max(0.0) as f32
        };
    }
    out
}

/// Independent direct-window f64 reference, shared by the numerical
/// instrument and kernel tests. Centered moments deliberately avoid the
/// candidate's running-sum algorithm; raw algebra is a separate control.
/// Returns dissimilarities, maximum algebra disagreement, and direct local
/// means for the same instrument's edge-precision checks.
#[cfg(test)]
pub(crate) fn precision_reference(
    src: &[f32],
    dst: &[f32],
    w: usize,
    h: usize,
    radius: usize,
    form: SsimLumaForm,
) -> (Vec<f64>, f64, Vec<[f64; 2]>) {
    let mut result = Vec::with_capacity(w * h);
    let mut agreement = 0.0f64;
    let mut means = Vec::with_capacity(w * h);
    let mut samples = Vec::new();
    let radius = radius as isize;
    for y in 0..h {
        for x in 0..w {
            samples.clear();
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let xx = crate::metric::reflect_index((x as isize + dx).unsigned_abs(), w);
                    let yy = crate::metric::reflect_index((y as isize + dy).unsigned_abs(), h);
                    samples.push((src[yy * w + xx] as f64, dst[yy * w + xx] as f64));
                }
            }
            let count = samples.len() as f64;
            let m1 = samples.iter().map(|p| p.0).sum::<f64>() / count;
            let m2 = samples.iter().map(|p| p.1).sum::<f64>() / count;
            means.push([m1, m2]);
            let md = samples.iter().map(|p| p.0 - p.1).sum::<f64>() / count;
            let v1 = samples.iter().map(|p| (p.0 - m1).powi(2)).sum::<f64>() / count;
            let v2 = samples.iter().map(|p| (p.1 - m2).powi(2)).sum::<f64>() / count;
            let ve = samples
                .iter()
                .map(|p| (p.0 - p.1 - md).powi(2))
                .sum::<f64>()
                / count;
            let c2 = C2 as f64;
            let loss = match form {
                SsimLumaForm::Ssim2Legacy => md * md,
                SsimLumaForm::Clamp => (md * md).min(1.0),
            };
            let sd = loss + (1.0 - loss) * ve / (v1 + v2 + c2);
            let ssq = samples.iter().map(|p| p.0 * p.0 + p.1 * p.1).sum::<f64>() / count;
            let s12 = samples.iter().map(|p| p.0 * p.1).sum::<f64>() / count;
            let raw =
                1.0 - (1.0 - loss) * (2.0 * (s12 - m1 * m2) + c2) / (ssq - m1 * m1 - m2 * m2 + c2);
            agreement = agreement.max((sd - raw).abs());
            result.push(sd.max(0.0));
        }
    }
    (result, agreement, means)
}

/// SSIM structure/contrast stabiliser — ssimulacra2's value, and the ONE
/// declaration of it.
///
/// `0.0009 = 0.03^2 = (K2 * L)^2` at the textbook `K2 = 0.03`, `L = 1`. Before
/// this module it was declared three times (`fused.rs`, `simd_ops.rs`,
/// `feature_v2.rs`'s `C2_V2`), which is how [`C_SSIM_LUMA`] came to be missing
/// rather than merely omitted: there was no single place for it to be missing
/// FROM.
pub(crate) const C2: f32 = 0.0009;

/// Which luminance term the per-pixel dissimilarity uses.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic that a revision SELECTS. Three bounded arms exist because the
/// choice between them is a measurement (`docs/PLAN_FEATURE_REV2_2026-09-05.md`
/// R6), not an argument — see each variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum SsimLumaForm {
    /// `1 - D^2`. ssimulacra2's form, shipped, **unbounded below** (F4).
    #[default]
    Ssim2Legacy,
    /// `max(0, 1 - D^2)`. **Exact** for `D^2 <= 1` — the overwhelming majority
    /// of pixels — and flat above it, so it bounds without touching the
    /// photographic regime at all, at the cost of flattening rank among the
    /// worst pixels.
    Clamp,
}

impl SsimLumaForm {
    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::Ssim2Legacy,
            FormulaRevision::Rev5
            | FormulaRevision::Rev7 => Self::REV2_LUMA,
        }
    }

    /// The bounded arm revision 2 ships — **`Clamp`, DECIDED BY MEASUREMENT**
    /// (`benchmarks/f4_arm_decision_2026-09-05.md`, R6).
    ///
    /// Named once so the R6 probe and the kernels cannot disagree about which
    /// arm "rev2" means.
    ///
    /// # Why `Clamp` and not [`Self::SsimLumaC1`]
    ///
    /// `docs/PLAN_FEATURE_REV2_2026-09-05.md` §1.4 registered a prior — *"if
    /// the probe cannot separate them, arm A (`SsimLumaC1`) ships"* — because
    /// it reuses `feature_v2`'s own `bounded_sim` owner and preserves severity
    /// ORDER among pathological pixels. **The probe DID separate them**, so
    /// the prior never fired. Four arms were extracted from ONE binary over
    /// 217,756 rows (the seven human eval corpora + the full 196,086-row
    /// safesyn training leg), fitted through the shipped Profile-D recipe at
    /// three slices x two solvers, and graded on rank, dial and cell deltas:
    ///
    /// * **F4's pathology occurs on NONE of it.** `Clamp` — which differs from
    ///   the shipped form only where `(mu1-mu2)^2 > 1` — moves **0 cells**, and
    ///   no slot anywhere reaches `|f| > 2` against the 5,814,302 on record
    ///   (which belongs to the bigcodec sweep, a population with no local
    ///   pixels).
    /// * **`Clamp` is therefore bit-identical to revision 1 on every row R6
    ///   fits or scores** — features, Gram, solve, spline and ZNPR bytes, all
    ///   six bakes sha-for-sha — so its rank delta is exactly 0 with a
    ///   degenerate CI.
    /// * `SsimLumaC1` moves **29.4 M** healthy cells (worst |delta| 0.771) and
    ///   `Lorentz` **24.0 M** (worst 0.0901), against a pre-registered 1e-4
    ///   bar, to buy at most `+0.0025` CID22 in one of six variants. Both fail
    ///   the healthy-cell gate; neither wins a rank majority in more than one
    ///   variant.
    ///
    /// So `Clamp` is the unique arm that is ONLY a fix: it changes the metric
    /// exactly where the metric was unbounded and nowhere else, which is what
    /// makes a rev2 flip cheap for every table whose content resembles those
    /// corpora.
    ///
    /// **Its known cost, recorded rather than glossed:** above `D^2 = 1` every
    /// pixel gets the same `num_m = 0`, so tail ORDER is flat there — the exact
    /// property §1.4's prior was protecting. `d` stays bounded in `[0, 2]`
    /// regardless. If a future population makes that order load-bearing,
    /// [`Self::Lorentz`] is the registered successor (bounded, no Weber term,
    /// monotone in `D^2`, and 4-6 orders of magnitude closer to revision 1 on
    /// healthy content than `SsimLumaC1`) — not `SsimLumaC1`.
    pub(crate) const REV2_LUMA: Self = Self::Clamp;

    /// Whether `d_raw` is bounded below by 0 by construction, making the
    /// call sites' `.max(0)` floor redundant rather than load-bearing.
    ///
    /// True for every arm whose `num_m` is bounded in `[0, 1]`: with
    /// `num_s/denom_s` in `[-1, 1]`, `d_raw = 1 - num_m*(num_s/denom_s)` lands
    /// in `[0, 2]`.
    pub(crate) const fn bounds_dissim(self) -> bool {
        !matches!(self, Self::Ssim2Legacy)
    }
}

/// **The revision this build ships.**
///
/// Changing this line is the era flip. Everything else — the registry entries,
/// the gates, the recalculation — hangs off it, and
/// `docs/PLAN_FEATURE_REV2_2026-09-05.md` R1 is the control that proves the
/// machinery around it is inert while it still reads `Rev1`.
pub(crate) const SHIPPED_REVISION: FormulaRevision = FormulaRevision::Rev1;

/// The revision a computation without a declared revision runs at: always
/// [`SHIPPED_REVISION`]. A bake that declares its revision selects it per call.
pub(crate) const fn active_revision() -> FormulaRevision {
    SHIPPED_REVISION
}

/// Explicit research cross-revision replay keeps using the process kernels.
/// Normal model-bound requests never consult mutable thread/global overrides.
pub(crate) fn effective_revision(revision: FormulaRevision) -> FormulaRevision {
    revision
}

/// Refuse, EXPLICITLY, any route the active revision does not serve.
///
/// [`FormulaRevision::Rev3`] replaces the v1 SSIM signal with
/// [`stable_ssim_plane`], whose spatial support is exactly ONE reflect-101
/// box of `blur_radius`. `blur_passes != 1` selects
/// `streaming::process_strip_channel`'s separate blur+reduce fallback, which
/// no shipped profile reaches and whose `passes * radius` halo does not
/// describe a single box. Rev3 therefore does not serve it.
///
/// This is the ONLY correct third option. Serving it from the legacy moments
/// would silently mix two arithmetic eras inside one feature vector, and
/// panicking would put an abort on a public path; so every fallible entry
/// that turns a profile into a [`crate::metric::ZensimConfig`] calls this and
/// returns the error to the caller.
///
/// Callers that only inspect a config (pool-mode derivation, fold planning)
/// do not call it: they run no pixel kernel, and the entry that eventually
/// does has already refused.
///
/// REV4SERVE: Rev4 crossings are refused via [`refuse_rev4_mix`]: a
/// request at one revision served from a process pinned to another would
/// compute the request's label with the process's gates. A pure-Rev4
/// process (`ZENSIM_FORMULA_REV=4`) serves Rev4 computations — every leaf
/// on the served walk is canonical there.
pub(crate) fn check_route(config: &crate::metric::ZensimConfig) -> Result<(), crate::ZensimError> {
    refuse_rev4_mix(config.revision())?;
    if config.revision() >= FormulaRevision::Rev5 && config.blur_passes != 1 {
        return Err(crate::ZensimError::ModelForwardFailed {
            reason: "formula revision 3 serves blur_passes == 1 only (its stable SSIM moments are one reflect-101 box); use revision 1 or 2 for multi-pass blur profiles",
        });
    }
    Ok(())
}

/// The reason [`refuse_rev4_mix`] gives.
// the v2 walk's guard
pub(crate) const REV4_MIX: &str = "formula revisions 4 and 5 cannot be mixed with another revision in one process: other formula gates follow ZENSIM_FORMULA_REV, so the request and the process must carry the same revision when either is revision 4 or later";

/// **Refuse a computation whose revision disagrees with the process revision
/// when either one is Rev4** (featcanon D1).
///
/// Canonical arithmetic follows the computation's own revision (every
/// canonical leaf derives it from the revision it receives), but other
/// formula gates — the dense kernel's `direct` flag, `RootForm`, `PowForm`,
/// `HfGainForm` — still read the process switch. So a Rev4 request in a Rev3
/// process would compute Rev3 semantics, and a Rev3 request in a Rev4 process
/// would mix Rev4 gates into it. Both are refused rather than labelled with a
/// revision they did not compute. Mixing among Rev1–Rev3 is governed by the
/// pre-existing per-route checks and is unchanged.
// the v2 walk's guard
pub(crate) fn refuse_rev4_mix(requested: FormulaRevision) -> Result<(), crate::ZensimError> {
    let process = active_revision();
    if requested != process
        && (requested >= FormulaRevision::Rev5 || process >= FormulaRevision::Rev5)
    {
        return Err(crate::ZensimError::ModelLoadFailed { reason: REV4_MIX });
    }
    Ok(())
}

/// The luminance form the active revision selects.
///
/// Call this ONCE per kernel invocation, above the pixel loop — it reads a
/// `OnceLock`, which LLVM cannot hoist out of a loop for you.
#[inline]
pub(crate) fn active_luma_form() -> SsimLumaForm {
    SsimLumaForm::for_revision(active_revision())
}

/// Hoisted splats + the selected form, built once per kernel invocation.
///
/// Bundling them keeps the per-site call short and keeps [`SsimLumaForm`]
/// loop-invariant, so the `match` is unswitched out of the inner loop after
/// inlining rather than evaluated per pixel.
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats16<T: F32x16Backend + Copy> {
    form: SsimLumaForm,
    one: f32x16<T>,
    two: f32x16<T>,
    zero: f32x16<T>,
    c2: f32x16<T>,
}

impl<T: F32x16Backend + Copy> SsimSplats16<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: f32x16::splat_t(token, 1.0),
            two: f32x16::splat_t(token, 2.0),
            zero: f32x16::zero_t(token),
            c2: f32x16::splat_t(token, C2),
        }
    }

    /// `d_raw = 1 - num_m * num_s / denom_s`, BEFORE any weight multiply and
    /// before the call site's `.max(0)` floor — the two things that differ
    /// between the basic, masked and IW call sites.
    ///
    /// For [`SsimLumaForm::Ssim2Legacy`] the emitted operations are
    /// bit-identical to the hand-written form this replaced: same `mul_add`
    /// spellings, same order, same single divide.
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        s12: f32x16<T>,
    ) -> f32x16<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        // Every arm reduces to ONE divide by expressing the luminance term as
        // a (numerator, denominator) pair folded into the structure ratio.
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        // `den_m` is exactly 1.0 for the two arms that do not need it, and
        // `x * 1.0` is exact, so the legacy arm keeps its original rounding.
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    /// The DIRECT-ERROR form, for revision 3: the fourth moment is
    /// `Σ(a-b)²` (the H/V passes accumulate it in place of `Σab`), so the
    /// error variance is formed directly instead of recovered as
    /// `var1 + var2 - 2*cov`. Same algebra as the f64 reference `finalize`,
    /// in f32 with the crate's `mul_add` spellings. Bounded error, registered
    /// in `benchmarks/stable_ssim_kernel_2026-09-08.md`.
    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: f32x16<T>,
        m2: f32x16<T>,
        ssq: f32x16<T>,
        err: f32x16<T>,
    ) -> f32x16<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
    }
}

/// 8-lane sibling of [`SsimSplats16`].
#[derive(Clone, Copy)]
pub(crate) struct SsimSplats8<T: F32x8Backend + Copy> {
    form: SsimLumaForm,
    one: GenericF32x8<T>,
    two: GenericF32x8<T>,
    zero: GenericF32x8<T>,
    c2: GenericF32x8<T>,
}

impl<T: F32x8Backend + Copy> SsimSplats8<T> {
    #[inline(always)]
    pub(crate) fn new(token: T, form: SsimLumaForm) -> Self {
        Self {
            form,
            one: GenericF32x8::splat_t(token, 1.0),
            two: GenericF32x8::splat_t(token, 2.0),
            zero: GenericF32x8::zero_t(token),
            c2: GenericF32x8::splat_t(token, C2),
        }
    }

    /// See [`SsimSplats16::dissim`].
    #[inline(always)]
    pub(crate) fn dissim(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        s12: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let num_s = self.two.mul_add((-m1).mul_add(m2, s12), self.c2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + self.c2;
        let (num_m, den_m) = match self.form {
            SsimLumaForm::Ssim2Legacy => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one), self.one)
            }
            SsimLumaForm::Clamp => {
                let mu_diff = m1 - m2;
                (mu_diff.mul_add(-mu_diff, self.one).max(self.zero), self.one)
            }
        };
        self.one - (num_m * num_s) / (den_m * denom_s)
    }

    /// 8-lane sibling of [`SsimSplats16::direct`].
    #[inline(always)]
    pub(crate) fn direct(
        &self,
        m1: GenericF32x8<T>,
        m2: GenericF32x8<T>,
        ssq: GenericF32x8<T>,
        err: GenericF32x8<T>,
    ) -> GenericF32x8<T> {
        let mu_diff = m1 - m2;
        let mean_error2 = mu_diff * mu_diff;
        let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(self.zero);
        let error_variance = (err - mean_error2).max(self.zero);
        let luma_loss = match self.form {
            SsimLumaForm::Ssim2Legacy => mean_error2,
            SsimLumaForm::Clamp => mean_error2.min(self.one),
        };
        (self.one - luma_loss).mul_add(error_variance / (variance_sum + self.c2), luma_loss)
    }
}

/// Scalar sibling — the `width % LANES` tails every vector kernel carries.
///
/// Same operations, same order, same `mul_add` spellings as the vector arms,
/// so a tail row and a vector row agree to the extent f32 FMA allows.
#[inline(always)]
pub(crate) fn ssim_dissim_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    s12: f32,
) -> f32 {
    let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
    let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
    let (num_m, den_m) = match form {
        SsimLumaForm::Ssim2Legacy => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32), 1.0f32)
        }
        SsimLumaForm::Clamp => {
            let mu_diff = m1 - m2;
            (mu_diff.mul_add(-mu_diff, 1.0f32).max(0.0f32), 1.0f32)
        }
    };
    let d = 1.0f32 - (num_m * num_s) / (den_m * denom_s);
    // The boundedness claim, checked rather than asserted in prose: a bounded
    // luminance term puts `d` in `[0, 2]`, which is what makes the call sites'
    // `.max(0)` floor redundant instead of load-bearing. The tolerance is for
    // f32 rounding at the `num_s/denom_s = 1` boundary, not for slack in the
    // claim. Debug-only: this is the innermost scalar tail.
    debug_assert!(
        !form.bounds_dissim() || (-1e-5..=2.0 + 1e-5).contains(&d),
        "bounded SSIM form {form:?} produced d = {d} outside [0, 2]"
    );
    d
}

/// One-call wrapper over [`SsimSplats16`] for the 16-lane kernels.
///
/// The splats are loop-invariant constants, so building them inside an
/// `#[inline(always)]` call is free — LLVM hoists them out of the loop
/// exactly as it did when they were hand-declared above it. The struct stays
/// public to this crate for any future site that wants to hoist explicitly.
#[inline(always)]
pub(crate) fn ssim_dissim16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    s12: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).dissim(m1, m2, ssq, s12)
}

/// 8-lane sibling of [`ssim_dissim16`].
#[inline(always)]
pub(crate) fn ssim_dissim8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    s12: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).dissim(m1, m2, ssq, s12)
}

/// Scalar direct-error form — the tail sibling of [`SsimSplats16::direct`],
/// same spellings so vector and tail agree to their usual `mul_add` residual.
#[inline(always)]
pub(crate) fn ssim_direct_raw_scalar(
    form: SsimLumaForm,
    m1: f32,
    m2: f32,
    ssq: f32,
    err: f32,
) -> f32 {
    let mu_diff = m1 - m2;
    let mean_error2 = mu_diff * mu_diff;
    let variance_sum = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)).max(0.0f32);
    let error_variance = (err - mean_error2).max(0.0f32);
    let luma_loss = match form {
        SsimLumaForm::Ssim2Legacy => mean_error2,
        SsimLumaForm::Clamp => mean_error2.min(1.0f32),
    };
    (1.0f32 - luma_loss).mul_add(error_variance / (variance_sum + C2), luma_loss)
}

/// 16-lane direct-error form; see [`SsimSplats16::direct`].
#[inline(always)]
pub(crate) fn ssim_direct16<T: F32x16Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: f32x16<T>,
    m2: f32x16<T>,
    ssq: f32x16<T>,
    err: f32x16<T>,
) -> f32x16<T> {
    SsimSplats16::new(token, form).direct(m1, m2, ssq, err)
}

/// 8-lane direct-error form; see [`SsimSplats8::direct`].
#[inline(always)]
pub(crate) fn ssim_direct8<T: F32x8Backend + Copy>(
    token: T,
    form: SsimLumaForm,
    m1: GenericF32x8<T>,
    m2: GenericF32x8<T>,
    ssq: GenericF32x8<T>,
    err: GenericF32x8<T>,
) -> GenericF32x8<T> {
    SsimSplats8::new(token, form).direct(m1, m2, ssq, err)
}

#[cfg(test)]
mod tests {

    /// The three Rev3 retained-signal reducers over the same geometries the
    /// kernel cases use, snapshotted bitwise. `f64` sums are reduced per SIMD
    /// lane, so a tier with a different lane count reduces in a different
    /// order — these are recorded as bits, and any disagreement between tiers
    /// is therefore visible rather than absorbed by a tolerance.
    fn stable_pool_cases() -> Vec<[u64; 9]> {
        let mut out = Vec::new();
        for n in [1usize, 15, 16, 17, 31, 33, 64, 129, 512] {
            let signal: Vec<f32> = (0..n)
                .map(|i| ((i * 31 % 97) as f32 / 97.0) * 0.6)
                .collect();
            let activity: Vec<f32> = (0..n)
                .map(|i| ((i * 13 % 61) as f32 / 61.0) * 0.9)
                .collect();
            let ((m, m4, m2), (w, w4, w2)) =
                crate::simd_ops::ssim_signal_inline_both(&signal, &activity, 4.0, 4.0);
            let (om, om4, om2) = crate::simd_ops::ssim_signal_inline_mask(&signal, &activity, 4.0);
            let (oi, _, _) = crate::simd_ops::ssim_signal_iw_inline(&signal, &activity, 4.0);
            out.push([
                m.to_bits(),
                m4.to_bits(),
                m2.to_bits(),
                w.to_bits(),
                w4.to_bits(),
                w2.to_bits(),
                om.to_bits(),
                om4.to_bits() ^ om2.to_bits(),
                oi.to_bits(),
            ]);
        }
        out
    }

    use super::*;

    /// A sweep that reaches both the healthy regime and F4's pathology.
    ///
    /// `ssq` is `E[s^2] + E[d^2]`, so `denom_s = ssq - mu1^2 - mu2^2 + C2` is
    /// the variance sum; the cases keep it positive, as the walk does.
    fn cases() -> Vec<(f32, f32, f32, f32)> {
        let mut v = Vec::new();
        for &(m1, m2) in &[
            (0.0f32, 0.0f32),
            (0.3, 0.3),
            (0.3, 0.31), // photographic
            (0.05, 0.9),
            (50.0, 60.0), // the 2026-07-16 analytic rows
            (100.0, 400.0),
            (50.0, 2450.0),
            (2450.0, 50.0),
        ] {
            for &var in &[0.0f32, 1e-6, 1e-3, 0.25, 10.0] {
                for &cov_frac in &[-1.0f32, -0.5, 0.0, 0.5, 1.0] {
                    let ssq = m1 * m1 + m2 * m2 + 2.0 * var;
                    let s12 = m1 * m2 + cov_frac * var;
                    v.push((m1, m2, ssq, s12));
                }
            }
        }
        v
    }

    /// The exact expression the 36 hand-copied sites carried, transcribed
    /// verbatim — the control for the extraction.
    fn legacy_scalar(m1: f32, m2: f32, ssq: f32, s12: f32) -> f32 {
        let mu_diff = m1 - m2;
        let num_m = mu_diff.mul_add(-mu_diff, 1.0f32);
        let num_s = 2.0f32.mul_add((-m1).mul_add(m2, s12), C2);
        let denom_s = (-m2).mul_add(m2, (-m1).mul_add(m1, ssq)) + C2;
        1.0f32 - (num_m * num_s) / denom_s
    }

    /// **The extraction gate.** `Ssim2Legacy` must reproduce the replaced
    /// expression BIT-for-BIT, not merely closely: this is what lets the
    /// 36-site rewrite claim to be inert.
    #[test]
    fn legacy_arm_is_bit_identical_to_the_expression_it_replaced() {
        for (m1, m2, ssq, s12) in cases() {
            let got = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            let want = legacy_scalar(m1, m2, ssq, s12);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "legacy arm diverged at mu=({m1},{m2}) ssq={ssq} s12={s12}: {got} vs {want}"
            );
        }
    }

    /// **`Clamp` is exact where it matters.** It is defined to differ from the
    /// shipped form only where `(mu1-mu2)^2 > 1`, which is why it moved ZERO
    /// of 22,396 cells on the synthetic dump. Pinning that here means a future
    /// edit cannot quietly make it a general-purpose approximation.
    #[test]
    fn clamp_arm_is_bit_identical_to_legacy_below_the_knee() {
        for (m1, m2, ssq, s12) in cases() {
            if (m1 - m2) * (m1 - m2) > 1.0 {
                continue;
            }
            let c = ssim_dissim_raw_scalar(SsimLumaForm::Clamp, m1, m2, ssq, s12);
            let l = ssim_dissim_raw_scalar(SsimLumaForm::Ssim2Legacy, m1, m2, ssq, s12);
            assert_eq!(c.to_bits(), l.to_bits(), "clamp diverged below the knee");
        }
    }

    /// **G3.2** — pinning the shipped revision is the same as not pinning it.
    #[test]
    fn selecting_the_shipped_revision_is_a_no_op() {
        assert_eq!(
            SsimLumaForm::for_revision(SHIPPED_REVISION),
            SsimLumaForm::for_revision(active_revision()),
            "the active revision is not the shipped one (is ZENSIM_FORMULA_REV set?)"
        );
        assert!(
            !SsimLumaForm::Ssim2Legacy.bounds_dissim(),
            "the legacy arm must not claim to bound"
        );
    }
}
