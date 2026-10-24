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
//! (scale, channel). Every sum uses 16 fixed virtual f64 lanes (lane = column
//! mod 16) closed by one fixed pairwise reduce; products of two f32 values are
//! exact in f64 and nothing is fused, so the result is the same bits on every
//! SIMD tier and at every thread count.
//!
//! Exploration record: `benchmarks/byconstruction_2026-10-22.md` ("Next
//! feature: a global affine-change descriptor"). The research sidecar
//! `zensim-bench/examples/byc_globaff.rs` used two-pass f64 moments over
//! whole planes; this owner's one-pass moments agree with it to f64 rounding
//! on the same planes (`one_pass_matches_two_pass_reference`).

use archmage::autoversion;

/// Signals per (scale, channel) cell: `share`, `explained`.
pub(crate) const GLOBAFF_PER_CELL: usize = 2;

/// Below this, a variance or a mean squared error counts as zero (the
/// sidecar's thresholds).
const TINY: f64 = 1e-12;

/// The five running sums of one (scale, channel) cell.
#[derive(Clone, Copy, Debug, Default)]
#[repr(align(64))]
pub(crate) struct GlobAffAccum {
    s: [f64; 16],
    d: [f64; 16],
    ss: [f64; 16],
    dd: [f64; 16],
    sd: [f64; 16],
    n: usize,
}

/// Add one row's lanes: `acc[x & 15] += f(src[x], dst[x])`, in column order.
#[autoversion]
fn add_row(acc: &mut [[f64; 16]; 5], src: &[f32], dst: &[f32]) {
    let n = src.len().min(dst.len());
    let full = n - n % 16;
    let [as_, ad, ass, add, asd] = acc;
    let mut x = 0;
    while x < full {
        let s: &[f32; 16] = src[x..x + 16].try_into().unwrap();
        let d: &[f32; 16] = dst[x..x + 16].try_into().unwrap();
        for k in 0..16 {
            let (a, b) = (s[k] as f64, d[k] as f64);
            as_[k] += a;
            ad[k] += b;
            ass[k] += a * a;
            add[k] += b * b;
            asd[k] += a * b;
        }
        x += 16;
    }
    for k in 0..n - full {
        let (a, b) = (src[full + k] as f64, dst[full + k] as f64);
        as_[k] += a;
        ad[k] += b;
        ass[k] += a * a;
        add[k] += b * b;
        asd[k] += a * b;
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
        let mut acc = [self.s, self.d, self.ss, self.dd, self.sd];
        for (s, d) in src.chunks_exact(width).zip(dst.chunks_exact(width)) {
            add_row(&mut acc, s, d);
        }
        [self.s, self.d, self.ss, self.dd, self.sd] = acc;
        self.n += src.len();
    }

    /// `(share, explained)` of the global affine fit.
    pub(crate) fn finish(&self) -> (f64, f64) {
        if self.n == 0 {
            return (0.0, 0.0);
        }
        let n = self.n as f64;
        let (ms, md) = (reduce16(&self.s) / n, reduce16(&self.d) / n);
        let vss = (reduce16(&self.ss) / n - ms * ms).max(0.0);
        let vdd = (reduce16(&self.dd) / n - md * md).max(0.0);
        let vsd = reduce16(&self.sd) / n - ms * md;
        finish_moments(ms, md, vss, vdd, vsd)
    }
}

/// `(share, explained)` from central moments; the sidecar's formula.
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

/// The family through the served walk (formula revision 6).
#[cfg(all(test, feature = "custom-profiles", feature = "feature-regime-v2"))]
mod walk_tests {
    use super::finish_moments;
    use crate::RgbSlice;
    use crate::feature_defs::FormulaRevision;
    use crate::feature_layout::Layout;
    use crate::feature_plan::Plan;
    use crate::feature_set_id::{ComputeToken, SlotSet};
    use crate::streaming::{MultiScaleRef, PrecomputedReference};

    const BASE: usize = 1867;
    const WIDTH: usize = 1891;

    /// A textured source and a distorted copy: contrast and offset change on
    /// luma-ish values plus deterministic noise.
    fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut state = 0x9e37_79b9u32;
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let v = ((x * 7 + y * 13) % 200) as i32 + 20;
                let px = [v as u8, (v / 2 + 40) as u8, (255 - v) as u8];
                src.push(px);
                let n = (state % 9) as i32 - 4;
                let f = |c: u8| ((c as i32 * 9) / 10 + 12 + n).clamp(0, 255) as u8;
                dst.push([f(px[0]), f(px[1]), f(px[2])]);
            }
        }
        (src, dst)
    }

    fn walk(
        src: &[[u8; 3]],
        dst: &[[u8; 3]],
        w: usize,
        h: usize,
        rev: FormulaRevision,
        want: &SlotSet,
    ) -> Vec<f64> {
        let plan = Plan::derive_with_layout_at(want, Layout::identity(WIDTH), rev).expect("plan");
        let mut scratch = crate::feature_v2::V2Scratch::new();
        let extras = crate::feature_v2::FoldWalkExtras {
            compute: Some(plan.compute),
            ..Default::default()
        };
        crate::feature_v2::compute_folded720_streaming_extras(
            &RgbSlice::new(src, w, h),
            &RgbSlice::new(dst, w, h),
            None,
            false,
            plan.toggles(),
            &mut scratch,
            extras,
        )
        .expect("walk")
        .into_features()
    }

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

    /// The walk's descriptor equals a two-pass f64 computation over the same
    /// planes (the reference-cache pyramid, which the producer's planes equal
    /// byte for byte), at several sizes including odd ones and multi-strip.
    #[test]
    fn walk_matches_two_pass_reference_over_its_own_planes() {
        let want = crate::feature_defs::family_slots(ComputeToken::GlobAff, crate::NUM_SCALES);
        for (w, h) in [(64usize, 64usize), (97, 71), (203, 300)] {
            let (s, d) = pair(w, h);
            let f = walk(&s, &d, w, h, FormulaRevision::Rev6, &want);
            assert!(f.len() >= WIDTH, "{w}x{h}: walk emitted {}", f.len());
            let rs = PrecomputedReference::new(
                &RgbSlice::new(&s, w, h),
                crate::NUM_SCALES,
                false,
                FormulaRevision::Rev6,
            );
            let ds = PrecomputedReference::new(
                &RgbSlice::new(&d, w, h),
                crate::NUM_SCALES,
                false,
                FormulaRevision::Rev6,
            );
            for scale in 0..crate::NUM_SCALES {
                let (sp, sw, sh) = rs.scale(scale);
                let (dp, _, _) = ds.scale(scale);
                for c in 0..3 {
                    let n = sw * sh;
                    let (share, explained) = two_pass(&sp[c][..n], &dp[c][..n]);
                    let at = BASE + 2 * (3 * scale + c);
                    assert!(
                        (f[at] - share).abs() <= 1e-9,
                        "{w}x{h} s{scale} c{c}: share {} vs {share}",
                        f[at]
                    );
                    assert!(
                        (f[at + 1] - explained).abs() <= 1e-9 * explained.max(1e-9),
                        "{w}x{h} s{scale} c{c}: explained {} vs {explained}",
                        f[at + 1]
                    );
                }
            }
        }
    }

    /// Revision 6 against revision 5 on the same read set: only the twelve v2
    /// ringing slots move, and identity is exactly zero for the descriptor.
    #[test]
    fn rev6_moves_only_ringing_and_adds_the_descriptor() {
        let ns = crate::NUM_SCALES;
        let core = crate::feature_defs::family_slots(ComputeToken::Basic, ns)
            .union(&crate::feature_defs::family_slots(ComputeToken::V2, ns));
        let with_globaff = core.union(&crate::feature_defs::family_slots(
            ComputeToken::GlobAff,
            ns,
        ));
        let (w, h) = (131usize, 97usize);
        let (s, d) = pair(w, h);
        let f5 = walk(&s, &d, w, h, FormulaRevision::Rev5, &core);
        let f6 = walk(&s, &d, w, h, FormulaRevision::Rev6, &with_globaff);
        let ringing: Vec<usize> = (0..ns * 3)
            .map(|cell| {
                372 + cell * crate::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL
                    + crate::feature_v2::idx::RINGING
            })
            .collect();
        let mut moved = 0;
        for id in core.iter_slots() {
            if ringing.contains(&id) {
                moved += usize::from(f5[id].to_bits() != f6[id].to_bits());
            } else {
                assert_eq!(
                    f5[id].to_bits(),
                    f6[id].to_bits(),
                    "slot {id}: {} vs {}",
                    f5[id],
                    f6[id]
                );
            }
        }
        assert!(
            moved > 0,
            "the high-pass ringing form moved no ringing slot"
        );
        assert!((BASE..WIDTH).all(|id| f6[id].is_finite()));
        let same = walk(&s, &s, w, h, FormulaRevision::Rev6, &with_globaff);
        assert!(
            (BASE..WIDTH).all(|id| same[id] == 0.0),
            "identity descriptor is not zero"
        );
    }

    /// Below revision 6 a bake or plan that reads the family is refused.
    #[test]
    fn earlier_revisions_refuse_the_family() {
        let want = crate::feature_defs::family_slots(ComputeToken::GlobAff, crate::NUM_SCALES);
        assert!(
            Plan::derive_with_layout_at(&want, Layout::identity(WIDTH), FormulaRevision::Rev5)
                .is_err()
        );
    }
}
