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

}
