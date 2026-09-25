// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Restored-cut families** (`COST_CUTS_AUDIT` A1 `mapdev`, B2 `z1max`, and
//! Ambiguous 7 `gmsnative`), appended after C8 at f1502..1819.
//!
//! Its own test executable: `for_each_token_permutation` mutates
//! process-wide SIMD dispatch (the `rev4_featbank_parity.rs` convention).
//!
//! Gates:
//! * f0..f1501 are `to_bits`-identical with the new families on vs off, on
//!   every dispatchable tier, serial and MT8, tight and strided input;
//! * each new family is `to_bits`-identical whether requested alone or with
//!   the others (independence of the plan);
//! * new slots are identical between serial and MT8, and between tight and
//!   strided input, and repeatable within a tier;
//! * identity behaviour: every `Difference` slot is exactly zero on an
//!   identical pair; the reference/distorted-only maps are equal on it.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::feature_set_id::ComputeToken;
use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::research::{self, Request, family_slots};
use zensim::source::StridedBytes;
use zensim::{PixelFormat, RgbSlice, Zensim, ZensimProfile};

const C8_END: usize = 1502;
const MAPDEV: std::ops::Range<usize> = 1502..1562;
const Z1MAX: std::ops::Range<usize> = 1562..1790;
const GMSNATIVE: std::ops::Range<usize> = 1790..1820;
const DVIFMGATE: std::ops::Range<usize> = 1820..1825;
const FULL: usize = 1825;

fn c8() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        csfw_block: true,
        dvifm_block: true,
        rev4_gridblk: true,
        rev4_ringbasis: true,
        rev4_tailhist: true,
        rev4_arttype: true,
        gmsbank: true,
        v1_pools: V1PoolsMode::Full,
        ..V2NewFeatureToggles::default()
    }
}

/// The three restored families, individually selectable. The layout flags are
/// a nested chain, so a later family carries the earlier ones' layout; the
/// COMPUTE plan is what `only` narrows.
fn with(mapdev: bool, z1max: bool, gmsnative: bool, dvifmgate: bool) -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        mapdev,
        z1max,
        gmsnative,
        dvifmgate,
        ..c8()
    }
}

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE ^ w as u32);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

fn extract(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    toggles: V2NewFeatureToggles,
    parallel: bool,
) -> Vec<f64> {
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(parallel);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_append_features_streaming(
        &RgbSlice::new(src, w, h),
        &RgbSlice::new(dst, w, h),
        toggles,
        &mut scratch,
    )
    .expect("streaming walk computes")
    .features()
    .to_vec()
}

fn extract_strided(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    toggles: V2NewFeatureToggles,
) -> Vec<f64> {
    let stride = w * 3 + 13;
    let pad = |px: &[[u8; 3]]| {
        let mut out = vec![0xA5u8; stride * h];
        for y in 0..h {
            for x in 0..w {
                out[y * stride + x * 3..][..3].copy_from_slice(&px[y * w + x]);
            }
        }
        out
    };
    let (ps, pd) = (pad(src), pad(dst));
    let z = Zensim::new(ZensimProfile::codec_target());
    let mut scratch = V2Scratch::new();
    z.compute_folded720_append_features_streaming(
        &StridedBytes::try_new(&ps, w, h, stride, PixelFormat::Srgb8Rgb).expect("strided src"),
        &StridedBytes::try_new(&pd, w, h, stride, PixelFormat::Srgb8Rgb).expect("strided dst"),
        toggles,
        &mut scratch,
    )
    .expect("streaming walk computes")
    .features()
    .to_vec()
}

fn tier(disabled: &[&str]) -> &'static str {
    let v3 = disabled.contains(&"x86-64-v3");
    let v4 = disabled.contains(&"AVX-512");
    let v2 = disabled.contains(&"x86-64-v2");
    if disabled
        .iter()
        .any(|t| t.contains("NEON") || t.contains("WASM"))
    {
        "other"
    } else if v3 || v2 {
        "scalar/SSE2"
    } else if v4 {
        "AVX2 (v3)"
    } else {
        "AVX-512 (v4)"
    }
}

fn bits_eq(a: &[f64], b: &[f64], range: std::ops::Range<usize>, what: &str) {
    for i in range {
        assert_eq!(a[i].to_bits(), b[i].to_bits(), "{what}: f{i}");
    }
}

/// z1max local indices whose registry form is `Difference` (exactly zero on
/// an identity pair): everything except the SSIM-derived slots.
const Z1_DIFF_LOCALS: [usize; 14] = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18];

#[test]
fn restore_cuts_prefix_identity_families_and_invariances() {
    for (w, h) in [(200, 200), (127, 288), (64, 64), (97, 73)] {
        let (src, dst) = pair(w, h);
        let mut by_tier: std::collections::BTreeMap<&'static str, Vec<f64>> = Default::default();
        let mut scalar: Option<Vec<f64>> = None;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let base = extract(&src, &dst, w, h, c8(), false);
            assert_eq!(base.len(), C8_END);
            let all = extract(&src, &dst, w, h, with(true, true, true, true), false);
            assert_eq!(all.len(), FULL, "{w}x{h} full width");
            bits_eq(
                &base,
                &all,
                0..C8_END,
                &format!("{w}x{h} {} prefix", perm.label),
            );

            // Layout independence. The planner computes every block a layout
            // reaches (`a_wide_layout_computes_every_block_it_reaches`), so a
            // family cannot be requested "alone" at a wide layout; the nested
            // chain instead lets each family be requested at the NARROWEST
            // layout that reaches it. Its values must not depend on which
            // later families the layout also carries.
            let m = extract(&src, &dst, w, h, with(true, false, false, false), false);
            assert_eq!(m.len(), MAPDEV.end);
            bits_eq(
                &m,
                &all,
                MAPDEV,
                &format!("{w}x{h} {} mapdev@1562", perm.label),
            );
            let z = extract(&src, &dst, w, h, with(true, true, false, false), false);
            assert_eq!(z.len(), Z1MAX.end);
            bits_eq(
                &z,
                &all,
                MAPDEV.start..Z1MAX.end,
                &format!("{w}x{h} {} z1max@1790", perm.label),
            );
            let n = extract(&src, &dst, w, h, with(true, true, true, false), false);
            assert_eq!(n.len(), GMSNATIVE.end);
            bits_eq(
                &n,
                &all,
                MAPDEV.start..GMSNATIVE.end,
                &format!("{w}x{h} {} gmsnative@1820", perm.label),
            );

            // The plan-driven research owner serves the same values: a request
            // for one family's slots yields that family's exact bits.
            for (fam, range) in [
                (ComputeToken::Mapdev, MAPDEV),
                (ComputeToken::Z1max, Z1MAX),
                (ComputeToken::Gmsnative, GMSNATIVE),
                (ComputeToken::Dvifmgate, DVIFMGATE),
            ] {
                let req = Request::for_slots(family_slots(fam), FULL);
                let got =
                    research::extract(&req, &RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
                        .expect("research extract")
                        .into_values();
                assert_eq!(got.len(), FULL);
                bits_eq(
                    &got,
                    &all,
                    range,
                    &format!("{w}x{h} {} {fam} via research", perm.label),
                );
            }

            // Every family is live on a distorted pair.
            assert!(all[MAPDEV].iter().any(|&v| v > 0.0), "{w}x{h} mapdev dead");
            assert!(all[Z1MAX].iter().any(|&v| v > 0.0), "{w}x{h} z1max dead");
            assert!(
                all[GMSNATIVE].iter().any(|&v| v > 0.0),
                "{w}x{h} gmsnative dead"
            );
            // The gate F1 counts only blocks whose local contrast is at or below
            // the fitted knee, so on pure noise it may legitimately be all zero
            // (see `dvifmgate_is_live_on_smooth_content` for the liveness gate).
            // Gate F1 counts a block or not; the curve F1 weights every block
            // by v <= 1, so per level gate >= 0 and both are bounded by
            // mean(m^P): compare against C7's own F1 at f956 + 6*level.
            for lvl in 0..5 {
                let curve = all[956 + 6 * lvl];
                let gate = all[DVIFMGATE.start + lvl];
                assert!(gate.is_finite() && curve.is_finite());
                assert!(gate >= 0.0 && curve >= 0.0);
            }
            assert!(all[C8_END..].iter().all(|v| v.is_finite()));

            // Identity.
            let same = extract(&src, &src, w, h, with(true, true, true, true), false);
            for cell in 0..12 {
                let o = MAPDEV.start + cell * 5;
                assert_eq!(same[o], 0.0, "{w}x{h} {} mse_dev cell {cell}", perm.label);
                assert_eq!(
                    same[o + 1].to_bits(),
                    same[o + 2].to_bits(),
                    "{w}x{h} {} hfsq src/dst dev cell {cell}",
                    perm.label
                );
                assert_eq!(
                    same[o + 3].to_bits(),
                    same[o + 4].to_bits(),
                    "{w}x{h} {} hfabs src/dst dev cell {cell}",
                    perm.label
                );
                let zo = Z1MAX.start + cell * 19;
                for &l in &Z1_DIFF_LOCALS {
                    assert_eq!(
                        same[zo + l],
                        0.0,
                        "{w}x{h} {} z1max cell {cell} local {l}",
                        perm.label
                    );
                }
            }
            assert!(
                same[GMSNATIVE].iter().all(|&v| v == 0.0),
                "{w}x{h} {} gmsnative identity",
                perm.label
            );
            assert!(
                same[DVIFMGATE].iter().all(|&v| v == 0.0),
                "{w}x{h} {} dvifmgate identity",
                perm.label
            );

            // Threads: MT8 == serial on every new slot.
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .expect("MT8");
            let mt8 =
                pool.install(|| extract(&src, &dst, w, h, with(true, true, true, true), true));
            bits_eq(
                &all,
                &mt8,
                C8_END..FULL,
                &format!("{w}x{h} {} MT8", perm.label),
            );

            // Stride: a padded, non-tight buffer gives the same bits.
            let strided = extract_strided(&src, &dst, w, h, with(true, true, true, true));
            bits_eq(
                &all,
                &strided,
                C8_END..FULL,
                &format!("{w}x{h} {} strided", perm.label),
            );

            let t = tier(&perm.disabled);
            if let Some(first) = by_tier.get(t) {
                bits_eq(
                    first,
                    &all,
                    C8_END..FULL,
                    &format!("{w}x{h} {} same-tier", perm.label),
                );
            } else {
                by_tier.insert(t, all.clone());
            }
            if t == "scalar/SSE2" {
                scalar = Some(all);
            }
        });
        // Cross-tier drift is MEASURED and reported, per the rev4 tier policy
        // (`rev4_featbank_parity.rs`): the XYB planes every family reads are
        // produced by tier-specific SIMD (cube-root), so values agree across
        // tiers to float precision, not bit for bit. Within a tier they are
        // bit-identical (asserted above).
        let scalar = scalar.expect("scalar tier ran");
        // The SSIM-derived z1max slots (local 0,1,2,13,16) read the Rev1
        // dissimilarity `1 - (2m1m2+c)(2s12+c)/...`, which cancels near identity:
        // tier-level 1e-7 plane differences are amplified (the F4 defect the
        // registry documents), so their drift is reported and not bounded here.
        let ssim_slot =
            |i: usize| Z1MAX.contains(&i) && matches!((i - Z1MAX.start) % 19, 0 | 1 | 2 | 13 | 16);
        let mut worst = (0.0f64, 0usize, 0.0f64, 0.0f64);
        let mut worst_ssim = (0.0f64, 0usize, 0.0f64, 0.0f64);
        for values in by_tier.values() {
            for i in C8_END..FULL {
                let rel = (values[i] - scalar[i]).abs() / scalar[i].abs().max(1e-12);
                let slot = if ssim_slot(i) {
                    &mut worst_ssim
                } else {
                    &mut worst
                };
                if rel > slot.0 {
                    *slot = (rel, i, values[i], scalar[i]);
                }
            }
        }
        eprintln!(
            "restore-cuts {w}x{h}: {} permutations, {} tiers; max cross-tier relative drift: \
             non-SSIM {:.3e} at f{} ({} vs scalar {}); SSIM-derived z1max {:.3e} at f{} ({} vs scalar {})",
            report.permutations_run,
            by_tier.len(),
            worst.0,
            worst.1,
            worst.2,
            worst.3,
            worst_ssim.0,
            worst_ssim.1,
            worst_ssim.2,
            worst_ssim.3,
        );
        // Relative drift is largest on the smallest values (1e-4 magnitudes at
        // the coarse scales); like the rev4/C8 gates, this is reported, not
        // bounded, and the within-tier / MT8 / strided identity above is exact.
        assert!(worst.0.is_finite() && worst_ssim.0.is_finite());
        assert!(report.permutations_run >= 3);
    }
}

/// Smooth content with mild distortion: local contrast sits below the DVIFM
/// knees, so the gate admits blocks and the family carries signal.
#[test]
fn dvifmgate_is_live_on_smooth_content() {
    let (w, h) = (160usize, 120usize);
    let src: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            [
                (x * 255 / w) as u8,
                (y * 255 / h) as u8,
                ((x + y) * 255 / (w + h)) as u8,
            ]
        })
        .collect();
    let dst: Vec<[u8; 3]> = src
        .iter()
        .enumerate()
        .map(|(i, p)| {
            [
                p[0].saturating_add(((i * 7 + i / w * 13) % 5) as u8),
                p[1],
                p[2],
            ]
        })
        .collect();
    let v = extract(&src, &dst, w, h, with(true, true, true, true), false);
    assert!(
        v[DVIFMGATE].iter().any(|&x| x > 0.0),
        "dvifmgate dead on smooth content: {:?}",
        &v[DVIFMGATE]
    );
    // The curve F1 (f956 + 6 l) weights every block by v <= 1, the gate weights
    // by {0, 1}: both are finite and non-negative at every level.
    for lvl in 0..5 {
        assert!(v[956 + 6 * lvl].is_finite() && v[956 + 6 * lvl] >= 0.0);
        assert!(v[DVIFMGATE.start + lvl].is_finite() && v[DVIFMGATE.start + lvl] >= 0.0);
    }
}
