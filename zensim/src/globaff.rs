//! The `globaff` family (f1867..1890, formula revision 6): a global affine-change
//! descriptor.
//!
//! Per pyramid scale and XYB channel, a least-squares fit `dst ≈ a·src + b`
//! over the whole plane, and two numbers:
//!
//! - `share`: the fraction of the mean squared error `E[(src − dst)²]` the fit
//!   explains, in `[0, 1]` (0 when the error is zero). An offset, a contrast
//!   change or a saturation change is explained almost entirely; noise and
//!   blur mostly are not.
//! - `explained`: that explained mean squared error,
//!   `E[(src − dst)²] − min_{a,b} E[(dst − a·src − b)²]`.
//!
//! The walk adds each emitted strip's rows, in strip order, into five sums per
//! (scale, channel): of the source `s`, the error `e = s − d` and of `s²`,
//! `e²` and `s·e`. Every sum uses 16 fixed virtual f64 lanes (lane = column
//! mod 16) closed by one fixed pairwise reduce, and nothing is fused, so the
//! result is the same bits on every SIMD tier and at every thread count.
//!
//! The moments are of the error, not of `d`: `mse = mean(e)² + var(e)` and
//! `explained = mean(e)² + min(var(e), cov(s, e)²/var(s))` are sums of
//! non-negative terms. The sidecar's raw form, `(ms − md)² + vss + vdd − 2·vsd`,
//! subtracts quantities of the size of the texture to get one of the size of
//! the error, and with one-pass sums over megapixels that loses every digit
//! of a small error (`large_mean_tiny_error_matches_compensated_two_pass`).
//!
//! Exploration record: `benchmarks/byconstruction_2026-10-22.md` ("Next
//! feature: a global affine-change descriptor"). The research sidecar
//! `zensim-bench/examples/byc_globaff.rs` used two-pass f64 moments over
//! whole planes; this owner's one-pass moments agree with it to f64 rounding
//! on the same planes (`one_pass_matches_two_pass_reference`).

use archmage::autoversion;

/// Signals per (scale, channel) cell: `share`, `explained`.
pub(crate) const GLOBAFF_PER_CELL: usize = 2;

/// The family's first feature id; its slots follow in
/// `2·(3·scale + channel) + {share, explained}` order.
#[allow(dead_code)]
pub(crate) const FIRST_ID: usize = 1867;

/// The family's slot count over the four scales.
#[allow(dead_code)]
pub(crate) const SLOTS: usize = crate::NUM_SCALES * 3 * GLOBAFF_PER_CELL;

/// Below this, a variance or a mean squared error counts as zero (the
/// sidecar's thresholds).
const TINY: f64 = 1e-12;

/// The five running sums of one (scale, channel) cell: `s`, `e = s − d`,
/// `s²`, `e²` and `s·e`.
#[derive(Clone, Copy, Debug, Default)]
#[repr(align(64))]
pub(crate) struct GlobAffAccum {
    s: [f64; 16],
    e: [f64; 16],
    ss: [f64; 16],
    ee: [f64; 16],
    se: [f64; 16],
    n: usize,
}

/// Add one row's lanes: `acc[x & 15] += f(src[x], dst[x])`, in column order.
/// The difference of two f32 values is exact in f64.
#[autoversion]
fn add_row(acc: &mut [[f64; 16]; 5], src: &[f32], dst: &[f32]) {
    let n = src.len().min(dst.len());
    let full = n - n % 16;
    let [as_, ae, ass, aee, ase] = acc;
    let mut x = 0;
    while x < full {
        let s: &[f32; 16] = src[x..x + 16].try_into().unwrap();
        let d: &[f32; 16] = dst[x..x + 16].try_into().unwrap();
        for k in 0..16 {
            let a = s[k] as f64;
            let e = a - d[k] as f64;
            as_[k] += a;
            ae[k] += e;
            ass[k] += a * a;
            aee[k] += e * e;
            ase[k] += a * e;
        }
        x += 16;
    }
    for k in 0..n - full {
        let a = src[full + k] as f64;
        let e = a - dst[full + k] as f64;
        as_[k] += a;
        ae[k] += e;
        ass[k] += a * a;
        aee[k] += e * e;
        ase[k] += a * e;
    }
}

/// The fixed pairwise reduce of 16 lanes.
fn reduce16(lanes: &[f64; 16]) -> f64 {
    let mut a = *lanes;
    for width in [8, 4, 2, 1] {
        for i in 0..width {
            a[i] = a[2 * i] + a[2 * i + 1];
        }
    }
    a[0]
}

impl GlobAffAccum {
    /// Add `rows` rows of `width` columns (row-major, stride `width`).
    pub(crate) fn add_rows(&mut self, src: &[f32], dst: &[f32], width: usize) {
        debug_assert_eq!(src.len(), dst.len());
        debug_assert!(width > 0 && src.len().is_multiple_of(width));
        let mut acc = [self.s, self.e, self.ss, self.ee, self.se];
        for (s, d) in src.chunks_exact(width).zip(dst.chunks_exact(width)) {
            add_row(&mut acc, s, d);
        }
        [self.s, self.e, self.ss, self.ee, self.se] = acc;
        self.n += src.len();
    }

    /// `(share, explained)` of the global affine fit.
    pub(crate) fn finish(&self) -> (f64, f64) {
        if self.n == 0 {
            return (0.0, 0.0);
        }
        let n = self.n as f64;
        let (ms, me) = (reduce16(&self.s) / n, reduce16(&self.e) / n);
        let vss = (reduce16(&self.ss) / n - ms * ms).max(0.0);
        let vee = (reduce16(&self.ee) / n - me * me).max(0.0);
        let cse = reduce16(&self.se) / n - ms * me;
        finish_error_moments(me, vss, vee, cse)
    }
}

/// `(share, explained)` from the error's moments: the sidecar's quantities,
/// with `resid = var(e) − cov(s, e)²/var(s)` (`d = s − e`) and the sidecar's
/// clamps and thresholds.
fn finish_error_moments(me: f64, vss: f64, vee: f64, cse: f64) -> (f64, f64) {
    let mse = me * me + vee;
    let affine = if vss > TINY {
        (cse * cse / vss).min(vee)
    } else {
        0.0
    };
    let explained = me * me + affine;
    let share = if mse > TINY {
        (explained / mse).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (share, explained)
}

/// `(share, explained)` from central moments of `s` and `d`: the sidecar's
/// formula, kept as the tests' independent reference.
#[cfg(test)]
fn finish_moments(ms: f64, md: f64, vss: f64, vdd: f64, vsd: f64) -> (f64, f64) {
    let mse = ((ms - md) * (ms - md) + vss + vdd - 2.0 * vsd).max(0.0);
    let resid = if vss > TINY {
        (vdd - vsd * vsd / vss).max(0.0)
    } else {
        vdd
    };
    let explained = (mse - resid).max(0.0);
    let share = if mse > TINY {
        (explained / mse).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (share, explained)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The family's id range is the registry's.
    #[test]
    fn first_id_and_slots_match_the_registry() {
        let ids: Vec<usize> = crate::feature_defs::family_slots(
            crate::feature_set_id::ComputeToken::GlobAff,
            crate::NUM_SCALES,
        )
        .iter_slots()
        .collect();
        assert_eq!(ids, (FIRST_ID..FIRST_ID + SLOTS).collect::<Vec<_>>());
    }

    /// The sidecar's two-pass f64 computation over one whole plane.
    fn two_pass(s: &[f32], d: &[f32]) -> (f64, f64) {
        let n = s.len() as f64;
        let ms = s.iter().map(|&v| v as f64).sum::<f64>() / n;
        let md = d.iter().map(|&v| v as f64).sum::<f64>() / n;
        let (mut vss, mut vdd, mut vsd) = (0.0f64, 0.0f64, 0.0f64);
        for (&a, &b) in s.iter().zip(d) {
            let (a, b) = (a as f64 - ms, b as f64 - md);
            vss += a * a;
            vdd += b * b;
            vsd += a * b;
        }
        finish_moments(ms, md, vss / n, vdd / n, vsd / n)
    }

    fn plane(w: usize, h: usize, seed: u32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(2654435761).wrapping_add(1);
        (0..w * h)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                0.3 + 0.2 * (x * 0.07).sin() * (y * 0.05).cos() + (state % 1000) as f32 * 1e-5
            })
            .collect()
    }

    fn accum(s: &[f32], d: &[f32], w: usize, strip: usize) -> GlobAffAccum {
        let mut a = GlobAffAccum::default();
        for (cs, cd) in s.chunks(w * strip).zip(d.chunks(w * strip)) {
            a.add_rows(cs, cd, w);
        }
        a
    }

    #[test]
    fn one_pass_matches_two_pass_reference() {
        let (w, h) = (203, 117);
        let s = plane(w, h, 1);
        for (name, d) in [
            ("offset", s.iter().map(|v| v + 0.03).collect::<Vec<_>>()),
            ("contrast", s.iter().map(|v| 0.85 * v + 0.04).collect()),
            ("noise", {
                let mut state = 0x1234_5679u32;
                s.iter()
                    .map(|a| {
                        state ^= state << 13;
                        state ^= state >> 17;
                        state ^= state << 5;
                        a + ((state % 2001) as f32 / 1000.0 - 1.0) * 0.02
                    })
                    .collect()
            }),
        ] {
            let (share, explained) = accum(&s, &d, w, 13).finish();
            let (rs, re) = two_pass(&s, &d);
            assert!((share - rs).abs() <= 1e-9, "{name}: share {share} vs {rs}");
            assert!(
                (explained - re).abs() <= 1e-9 * re.abs().max(1e-12),
                "{name}: explained {explained} vs {re}"
            );
        }
    }

    /// Neumaier-compensated sum.
    fn nsum(values: impl Iterator<Item = f64>) -> f64 {
        let (mut sum, mut comp) = (0.0f64, 0.0f64);
        for v in values {
            let t = sum + v;
            comp += if sum.abs() >= v.abs() {
                (sum - t) + v
            } else {
                (v - t) + sum
            };
            sum = t;
        }
        sum + comp
    }

    /// Two-pass central moments with compensated sums: a reference that does
    /// not cancel where one-pass raw moments do.
    fn two_pass_compensated(s: &[f32], d: &[f32]) -> (f64, f64) {
        let n = s.len() as f64;
        let ms = nsum(s.iter().map(|&v| v as f64)) / n;
        let md = nsum(d.iter().map(|&v| v as f64)) / n;
        let c = |f: &dyn Fn(f64, f64) -> f64| {
            nsum(
                s.iter()
                    .zip(d)
                    .map(|(&a, &b)| f(a as f64 - ms, b as f64 - md)),
            ) / n
        };
        finish_moments(ms, md, c(&|a, _| a * a), c(&|_, b| b * b), c(&|a, b| a * b))
    }

    /// Many pixels, a large mean and an error far smaller than the texture:
    /// where raw one-pass moments cancel (`E[s²] − E[s]²` and the
    /// `vss + vdd − 2·vsd` error). Tolerances (introduced here): share within
    /// 1e-6; explained within 1e-6 of itself or 1e-9 of the mean squared
    /// error, its scale (the explained part of zero-mean noise is itself
    /// about 1e-6 of the error).
    #[test]
    fn large_mean_tiny_error_matches_compensated_two_pass() {
        let (w, h) = (2048, 1024);
        let s: Vec<f32> = plane(w, h, 9).iter().map(|v| 0.5 + 0.25 * v).collect();
        let mut state = 0x9e37_79b9u32;
        let mut noise = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state % 2001) as f32 / 1000.0 - 1.0
        };
        for (name, d) in [
            (
                "tiny offset",
                s.iter().map(|v| v + 2e-5).collect::<Vec<_>>(),
            ),
            ("tiny noise", s.iter().map(|v| v + 1e-4 * noise()).collect()),
            (
                "tiny contrast",
                s.iter().map(|v| 1.0001 * v - 8e-5).collect(),
            ),
        ] {
            let (share, explained) = accum(&s, &d, w, 64).finish();
            let (rs, re) = two_pass_compensated(&s, &d);
            let mse = nsum(
                s.iter()
                    .zip(&d)
                    .map(|(&a, &b)| (a as f64 - b as f64).powi(2)),
            ) / s.len() as f64;
            assert!((share - rs).abs() <= 1e-6, "{name}: share {share} vs {rs}");
            assert!(
                (explained - re).abs() <= (1e-6 * re.abs()).max(1e-9 * mse),
                "{name}: explained {explained:e} vs {re:e} (mse {mse:e})"
            );
        }
    }

    #[test]
    fn global_edits_are_explained_and_noise_is_not() {
        let (w, h) = (96, 80);
        let s = plane(w, h, 3);
        let offset: Vec<f32> = s.iter().map(|v| v + 0.05).collect();
        let (share, explained) = accum(&s, &offset, w, 7).finish();
        assert!(share > 0.999, "offset share {share}");
        assert!(explained > 0.0);
        // Zero-mean noise independent of the image: almost nothing of it is affine.
        let mut state = 0x2545_f491u32;
        let noisy: Vec<f32> = s
            .iter()
            .map(|a| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                a + ((state % 2001) as f32 / 1000.0 - 1.0) * 0.05
            })
            .collect();
        let (share, _) = accum(&s, &noisy, w, 7).finish();
        assert!(share < 0.05, "noise share {share}");
    }

    #[test]
    fn identity_is_exactly_zero() {
        let (w, h) = (64, 64);
        let s = plane(w, h, 5);
        assert_eq!(accum(&s, &s, w, 16).finish(), (0.0, 0.0));
        assert_eq!(GlobAffAccum::default().finish(), (0.0, 0.0));
    }

    /// The lanes are a function of the column, so strip height (and therefore
    /// the walk's strip tiling and thread count) cannot reach a bit.
    #[test]
    fn strip_height_is_not_semantics() {
        let (w, h) = (77, 61);
        let s = plane(w, h, 21);
        let d: Vec<f32> = s.iter().map(|v| 0.9 * v + 0.01).collect();
        let a = accum(&s, &d, w, 1).finish();
        for strip in [2, 5, 16, 61] {
            assert_eq!(accum(&s, &d, w, strip).finish(), a, "strip {strip}");
        }
    }
}

