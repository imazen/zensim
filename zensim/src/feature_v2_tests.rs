#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    
    use crate::source::RgbSlice;

    // Perf pass (§A.13): widened from 1e-6 to 5e-4 after switching mu1/mu2/
    // ssq/s12 to `crate::blur::fused_blur_h_ssim`'s single fused-multiply-add
    // sliding-window accumulation (`sum_sq = sv.mul_add(sv, dv.mul_add(dv,
    // sum_sq))`) in place of the old separate `sq_sum_into`/`mul_into`
    // elementwise passes (plain `sv*sv + dv*dv`, two roundings). FMA vs.
    // non-FMA reassociation is a legitimate, expected last-few-ULP source of
    // drift, not a logic change — but `ssim_d_local`'s `denom_s = ssq - mu1²
    // - mu2² + C2_V2` is a near-zero-minus-near-zero subtraction on identity
    // input (variance ≈ 0), stabilized only by the small `C2_V2 = 9e-4`
    // constant, so ULP-scale noise in `ssq` gets divided by ~9e-4 and
    // amplified ~1000x. This amplification is a PRE-EXISTING property of the
    // `ssim_d_local` formula (small-constant-stabilized ratio), not
    // something the perf pass introduced — only WHICH rounding path feeds it
    // changed. Observed on the identity fixture as TOL was raised: idx 0
    // (SSIM_MEAN) 2.52e-6, idx 1 (SSIM_DEV2) 1.47e-5, idx 9 (a
    // weighted-pool/SOFT_PEAK slot one denominator-division further from the
    // source) 1.20e-4 — each roughly one more small-constant division beyond
    // the last, consistent with amplified-not-new noise rather than a
    // distinct bug. 5e-4 leaves >4x headroom above the largest observed.
    // Within this file's own documented allowance (v2 has no downstream
    // consumers yet; v1's golden gate, which does NOT go through this
    // kernel, keeps its own separate zero-tolerance `==` comparison in
    // `tests/v1_golden_bytes.rs`).
    const TOL: f64 = 5e-4;

    /// Phase-4 (§A.14): proves the SIMD raw-moment reformulation
    /// (`compute_channel_scale_v2`'s `mean_d`/`m2`/`m4` block) is
    /// numerically equivalent to `OnlineMoments`'s Terriberry (2007)
    /// single-pass algorithm on the SAME data, not just "removed and
    /// hoped for the best". Uses `d` values in the actual documented
    /// range `[0, 2]` (not arbitrary floats) since that's what the real
    /// kernel feeds both paths. Also exercises `OnlineMoments` directly
    /// so it isn't dead code despite leaving the hot path.
    #[test]
    fn rev5_stable_block_moments_match_two_pass() {
        for base in [0.0, 0.5, 1.0, 1e4] {
            for n in [1usize, 16, 153, 1025] {
                let values: Vec<_> = (0..n)
                    .map(|i| base + ((i * 137 % 1009) as f64 - 500.0) * 1e-7)
                    .collect();
                let mean = values.iter().sum::<f64>() / n as f64;
                let m2 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
                let m4 = values.iter().map(|x| (x - mean).powi(4)).sum::<f64>();
                let mut actual = OnlineMoments::default();
                for block in values.chunks(16) {
                    actual.merge(OnlineMoments::block(block));
                }
                for (a, e) in [(actual.m2, m2), (actual.m4, m4)] {
                    assert!(
                        (a - e).abs() <= 1e-6 * e.abs().max(1e-30),
                        "base={base} n={n} actual={a} reference={e}"
                    );
                }
            }
        }
        let mut constant = OnlineMoments::default();
        for _ in 0..10 {
            constant.merge(OnlineMoments::block(&[0.7; 16]));
        }
        assert_eq!(constant.m2, 0.0);
        assert_eq!(constant.m4, 0.0);
    }

    // ------------------------------------------------------------------
    // Prepared-reference (pyramid reuse) path
    // ------------------------------------------------------------------

    /// Deterministic structured content: gradients + edges + pseudo-noise
    /// texture (LCG), so every feature family (SSIM, HF, gradient,
    /// blockiness, transducer) sees real signal.
    pub(crate) fn textured_image(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
        let mut state = seed | 1;
        let mut px = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = (state >> 24) as u8;
                let grad = ((x * 255) / w.max(1)) as u8;
                let edge = if (x / 9 + y / 7) % 2 == 0 { 200 } else { 40 };
                let r = grad.wrapping_add(noise / 4);
                let g = edge;
                let b = ((y * 255) / h.max(1)) as u8 ^ (noise / 8);
                px.push([r, g, b]);
            }
        }
        px
    }

    /// Blocky quantization distortion — deterministic, codec-flavored.
    pub(crate) fn quantize_distort(src: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut out = src.to_vec();
        for y in 0..h {
            for x in 0..w {
                let p = &mut out[y * w + x];
                for c in p.iter_mut() {
                    // 8x8-block DC-ish flattening + level quantization.
                    *c = (*c / 24) * 24 + ((x % 8 == 0 || y % 8 == 0) as u8) * 6;
                }
            }
        }
        out
    }

    /// ERA-2 STRUCTURAL GATE: the properties the break actually rests on.
    ///
    /// **A correction the first version of this test earned.** It originally
    /// asserted "band-merge == serial fold, bit-exact" and FAILED at 127x93
    /// (7.854278564453125 vs 7.8542633056640625). The assertion was wrong, not
    /// the design: banding IS a different grouping — serial computes
    /// `((band0 + r32) + r33) + …` while banded computes `band0 + band1` — and
    /// f64/f32 addition is not associative. This is the same blocking
    /// non-associativity documented for the era-1 dense kernel; it does not go
    /// away because the shape is new.
    ///
    /// **So `ERA2_BAND_ROWS` is SEMANTICS, not a tuning knob.** Changing it
    /// changes output bytes and is an era decision. It is deliberately NOT
    /// derived from the thread count, image height, or anything else that
    /// varies at runtime — that is what makes thread-invariance structural.
    /// Anyone reaching for it to "tune the band size" must read this first.
    ///
    /// What is actually asserted:
    /// (a) the banded fold is INVARIANT to how bands are distributed across
    ///     workers — computing them in any order, or all at once, gives
    ///     bit-identical results, because the merge is sequential in band index;
    /// (b) lane `j` holds exactly the terms at `x ≡ j (mod 8)` in increasing
    ///     `x`, tail included — so the shape is width-independent, unlike
    ///     era-1's, whose tail landed in the f64 running total.

    /// **The two `self_blur` predicates must be the SAME predicate.**
    ///
    /// There are two, and until 2026-09-05 they disagreed on
    /// [`V1PoolsMode::Peaks`]:
    ///
    /// * [`ComputeSet::self_blur_eligible`] — read at `stream_foldapp`'s
    ///   `plane_needs` call, which is what SIZES the strip scratch.
    /// * the `let self_blur = …` binding inside the strip loop's
    ///   `fuse_channels` arm, which is what DECIDES whether phase A runs.
    ///
    /// When the sizing predicate says `false` and the loop predicate says
    /// `true`, the scratch reserves phase A's four fused-H planes and the
    /// loop then skips phase A whole, so those planes are allocated,
    /// first-touched and never written. That is exactly the waste
    /// `StripPlaneNeeds` was introduced to remove
    /// (`benchmarks/fold_footprint_2026-08-31.md` §5), reappearing on the
    /// one mode the fast class actually ships:
    /// `fold_engine::pools_mode_for_need` returns `Peaks` for any bake that
    /// does not read masked/IW, which is every basic+peaks model.
    ///
    /// This test pins them together so the next edit to either one cannot
    /// silently re-open the gap. It asserts the LOOP's predicate, because
    /// the loop is the one that determines what is actually written — a
    /// sizing decision that disagrees with it is wrong by definition, in
    /// whichever direction it disagrees (under-sizing would be a
    /// correctness bug, over-sizing is the footprint bug found here).
    #[test]
    fn self_blur_sizing_predicate_matches_the_strip_loop_predicate() {
        for &pm in &[
            V1PoolsMode::Off,
            V1PoolsMode::Carriers,
            V1PoolsMode::Peaks,
            V1PoolsMode::Full,
        ] {
            for &v1_only in &[true, false] {
                let t = V2NewFeatureToggles {
                    v1_only,
                    v1_pools: pm,
                    ..Default::default()
                };
                let cs = ComputeSet::from_toggles(t);
                // Verbatim from the strip loop's `fuse_channels` arm, minus
                // `fuse_channels` itself (a runtime parallelism fact, which
                // both sites already AND in separately).
                let loop_predicate = !cs.v2_blocks
                    && cs.v1_basic
                    && matches!(t.v1_pools, V1PoolsMode::Full | V1PoolsMode::Peaks);
                assert_eq!(
                    cs.self_blur_eligible(),
                    loop_predicate,
                    "self_blur sizing vs loop predicate disagree at \
                     v1_pools={pm:?} v1_only={v1_only}: sizing says {}, loop says \
                     {loop_predicate}. A disagreement in this direction reserves \
                     phase A's four H planes and never writes them.",
                    cs.self_blur_eligible(),
                );
            }
        }
    }

    // ---------------------------------------------------------------- //
    // Feature-set IDs (`docs/FEATURE_SET_IDS.md`) — the COMPUTE half.
    // ---------------------------------------------------------------- //

    /// The trap that motivates [`super::era2_reduce8`]: `reduce_add()` is
    /// TIER-DEPENDENT (§14.2), so it can never be part of era-2's semantics.
    ///
    /// This does not assert a divergence (this box runs one tier), it asserts
    /// the FIXED tree is what we think it is — an executable statement of the
    /// semantics, so a future refactor that silently swaps in `reduce_add()`
    /// fails here.
    #[test]
    fn era2_reduce_tree_is_fixed_and_explicit() {
        // Values chosen so the two known groupings genuinely differ: adjacent
        // pairing vs the x86_v3 (i, i+4) pairing.
        let a: [f32; 8] = [1.0, 1e-8, 1.0, 1e-8, -1.0, 1e-8, -1.0, 1e-8];
        let fixed = super::era2_reduce8(a);
        let adjacent = (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64;
        let v3_style = (((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))) as f64;
        assert_eq!(
            fixed.to_bits(),
            adjacent.to_bits(),
            "era2_reduce8 must be the ADJACENT-pairwise tree, written out"
        );
        assert_ne!(
            adjacent.to_bits(),
            v3_style.to_bits(),
            "the two backend reduction shapes must be demonstrably different on \
             this fixture — if they agree, the fixture stopped exercising the \
             divergence and this test no longer documents anything"
        );
    }

    /// C, pinned at its owner: no phantom columns at ANY width. A future
    /// stride/alignment experiment that reintroduces padding must fail here
    /// and read the reason, because padding the plane silently re-pollutes
    /// every v1 pool with columns that are not in the image (that was the
    /// era-2 defect, worth up to 81.6 % on a pool slot).
    #[test]
    fn pyramid_stride_has_no_phantom_columns() {
        for w in [
            1usize, 7, 8, 15, 16, 17, 63, 64, 96, 100, 127, 128, 200, 255, 256, 511, 512, 513, 576,
            592, 1024, 1152, 2304, 4096,
        ] {
            assert_eq!(
                crate::blur::pyramid_plane_stride(w),
                w,
                "pyramid_plane_stride({w}) must be {w} under option C"
            );
        }
    }

    // =====================================================================
    // CLASS-C free slots (`V1FreeExtras::RawMomentsPlusBoundedErr`) —
    // `benchmarks/free_features_classC_2026-09-04.md`
    // =====================================================================

    /// The kernel's f32 copies of the two saturation constants must equal
    /// the f64 definitions the 944 walk uses. `fused.rs` is unconditional
    /// while this module is behind `feature-regime-v2`, so the constants
    /// cannot be imported; this is the gate that keeps the duplication
    /// honest (the same contract `fused::C_MSE_F32`'s doc states).
    #[test]
    fn class_c_kernel_constants_match() {
        // Stated in f32, which is the comparison that means something: the
        // 944 walk's own SIMD path splats `C_MSE as f32` / `C_LUM_T as f32`
        // (`append_block_kernel_generic`), so THAT is the value the class-C
        // kernel has to reproduce. Neither constant is representable in
        // f32, so an `as f64` round-trip equality would be false for a
        // correct implementation — it was, and this is the corrected claim.
        assert_eq!(crate::fused::C_MSE_F32, C_MSE as f32);
        assert_eq!(crate::fused::C_LUM_T_F32, C_LUM_T as f32);
    }

    /// **CLASS-C GATE 0 — the INTEGRAND, against an f64 scalar oracle.**
    ///
    /// The per-slot end-to-end oracle is gate 2 below (an independent
    /// kernel's value for the same slot). This one gates the arithmetic
    /// underneath it: the kernel's f32 per-pixel integrands
    /// (`sat((s−d)², C_MSE)` and the Bernstein weights of
    /// `t = sat(ref_Y, C_LUM_T)`) against the SAME f64 `saturate` the
    /// append kernel's own scalar tail calls, over a value range that
    /// spans the XYB planes' actual span including negatives (where the
    /// `max(0)` clamp is load-bearing) and the near-zero regime the
    /// saturation amplifies ~1/C.
    ///
    /// The bar is RELATIVE (1e-5) with a 1e-7 absolute floor: f32 carries
    /// ~1.2e-7 relative, and `sat` is contractive in relative error
    /// (`dsat/sat = (1−sat)·dx/x`), so a correct f32 evaluation lands two
    /// orders inside it. An algebra error does not.
    #[test]
    fn class_c_integrands_match_the_f64_scalar_oracle() {
        let vals: Vec<f32> = vec![
            -3.0, -1.0, -0.35, -1e-4, 0.0, 1e-7, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.35, 0.5, 1.0, 2.0,
            7.5,
        ];
        let mut worst_m = 0.0f64;
        let mut worst_w = 0.0f64;
        for &s in &vals {
            for &d in &vals {
                let mut acc_m = 0.0f32;
                let m = crate::fused::test_only_bounded_err_scalar(&mut acc_m, s - d);
                assert_eq!(
                    acc_m, m,
                    "the accumulator must hold exactly what it returned"
                );
                let want_m = saturate(((s - d) as f64) * ((s - d) as f64), C_MSE);
                let dm = (m as f64 - want_m).abs();
                worst_m = worst_m.max(dm / want_m.max(1e-7));
                assert!(
                    dm <= 1e-5 * want_m.abs().max(1e-2) + 1e-7,
                    "mse_i({s},{d}) = {m:e}, f64 oracle {want_m:e}"
                );

                let (mut wdn, mut wdd, mut wbn, mut wbd) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                crate::fused::test_only_lum_bins_scalar(
                    &mut wdn, &mut wdd, &mut wbn, &mut wbd, s, m,
                );
                let t_ref = saturate(s as f64, C_LUM_T);
                let (wd_ref, wb_ref) = ((1.0 - t_ref) * (1.0 - t_ref), t_ref * t_ref);
                for (got, want, name) in [
                    (wdd as f64, wd_ref, "w_dark"),
                    (wbd as f64, wb_ref, "w_bright"),
                    (wdn as f64, wd_ref * want_m, "w_dark*mse"),
                    (wbn as f64, wb_ref * want_m, "w_bright*mse"),
                ] {
                    let dw = (got - want).abs();
                    worst_w = worst_w.max(dw / want.abs().max(1e-7));
                    assert!(
                        dw <= 1e-5 * want.abs().max(1e-2) + 1e-7,
                        "{name}({s},{d}) = {got:e}, f64 oracle {want:e}"
                    );
                }
                // The partition of unity the mid bin is DERIVED from must
                // hold at every point, or `lum_bins_from_weighted_sums`'s
                // subtraction is not a mid-bin weight.
                let w_mid = 1.0 - wd_ref - wb_ref;
                assert!(
                    (w_mid - 2.0 * t_ref * (1.0 - t_ref)).abs() <= 1e-12,
                    "Bernstein partition of unity broken at s={s}"
                );
                assert!(w_mid >= -1e-12, "derived mid weight negative at s={s}");
            }
        }
        eprintln!(
            "class-C integrand vs f64 oracle: worst rel |Δ| mse {worst_m:e}, weights {worst_w:e}"
        );
    }

    /// **CLASS-C GATE 5 — append-only bookkeeping.**
    ///
    /// The class-C tranche adds NO new slot numbers: every one of its 24
    /// positions is an existing 944-layout slot, it is disjoint from the
    /// raw-moments free set, and the layout width is unchanged. A future
    /// lane that wants genuinely new features must EXTEND the layout past
    /// 944, never renumber — this test is what fails if someone forgets.
    #[test]
    fn class_c_slot_set_is_append_only_and_disjoint() {
        for n_scales in 1..=4 {
            let cc = class_c_slot_indices(n_scales);
            let rm = free_slot_indices(n_scales);
            let width = n_scales * 3 * (31 + FEATURES_PER_CHANNEL_V2_TOTAL)
                + n_scales * 3 * FEATURES_PER_CHANNEL_APPEND
                + n_scales * APPEND2_PER_SCALE;
            let set: std::collections::HashSet<usize> = cc.iter().copied().collect();
            assert_eq!(set.len(), cc.len(), "class-C slot list has duplicates");
            for &i in &cc {
                assert!(
                    i < width,
                    "class-C slot {i} is outside the {width}-wide layout"
                );
                assert!(
                    !rm.contains(&i),
                    "class-C slot {i} overlaps the raw-moments free set"
                );
            }
            // 3 MSE cells per scale (one per channel) + the Y luminance
            // trio per scale where the 944 walk computes that append cell.
            let want = n_scales * 3
                + (0..n_scales)
                    .filter(|&s| append_cell_active(true, APPEND2_CHANNEL, s))
                    .count()
                    * 3;
            assert_eq!(cc.len(), want, "class-C slot count at {n_scales} scales");
        }
        // The 4-scale set, spelled out, so a renumbering shows up as a diff
        // in review rather than as a silently different table.
        assert_eq!(
            class_c_slot_indices(4),
            vec![
                377, 406, 435, 739, 740, 741, // scale 0
                464, 493, 522, 790, 791, 792, // scale 1
                551, 580, 609, 841, 842, 843, // scale 2
                638, 667, 696, 892, 893, 894, // scale 3
            ]
        );
    }

    // ------------------------------------------------------------------
    // f720+ append block
    // ------------------------------------------------------------------

    // ========================================================================
    // C2 streaming-walk parity gates
    // (docs/STREAMING_FOLDAPP_C0_DESIGN_2026-07-26.md §3)
    // ========================================================================

    /// A declared-ID plan that reads no slot at scale 0 (no v1-era slot, no v2-era work) skips the scale-0 strips
    /// whole, and every slot it does read is bit-identical to the full plan's.
    #[test]
    fn declared_plan_skips_unread_scales_bit_identically() {
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let (w, h) = (160, 128);
        let mut st = 0x9e37_79b9u32;
        let mut rnd = move || {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            st
        };
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 7 + y * 3) % 256) as u8, ((x * x + y) % 256) as u8, ((y * 5 + x / 3) % 256) as u8]
            })
            .collect();
        let dst: Vec<[u8; 3]> = src
            .iter()
            .map(|p| p.map(|v| v.saturating_add((rnd() % 7) as u8)))
            .collect();
        let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        // basic + peaks + v2: the Rev5 families, so the test holds at every revision.
        let all: Vec<usize> = (0..228).chain(372..720).collect();
        let coarse: Vec<usize> = all
            .iter()
            .copied()
            .filter(|&id| crate::feature_defs::def_at(id, 4).is_some_and(|d| d.scale >= 1))
            .collect();
        let full = Plan::derive(&SlotSet::from_slots(all), 720).unwrap();
        let narrow = Plan::derive(&SlotSet::from_slots(coarse.iter().copied()), 720).unwrap();
        assert_eq!(narrow.compute.v1_scales, 0b1110);
        assert_eq!(narrow.compute.v2_scales & 1, 0);
        let stray: Vec<usize> = narrow
            .compute
            .populated_slots(4, 720)
            .iter_slots()
            .filter(|&id| crate::feature_defs::def_at(id, 4).is_none_or(|d| d.scale == 0))
            .collect();
        assert!(stray.is_empty(), "a skipped scale must not be reported as populated: {stray:?}");
        let run = |plan: &Plan| {
            SKIPPED_SCALES.with_borrow_mut(|v| *v = 0);
            let f = foldapp_streaming_walk(
                &sref,
                &dref,
                false,
                plan.toggles(),
                crate::feature_v2_stream::FrontEnd::Sdr,
                &mut V2Scratch::new(),
                FoldWalkExtras {
                    compute: Some(plan.compute),
                    ..FoldWalkExtras::default()
                },
            )
            .into_features();
            (f, SKIPPED_SCALES.with_borrow(|v| *v))
        };
        let (f_full, skipped_full) = run(&full);
        let (f_narrow, skipped_narrow) = run(&narrow);
        assert_eq!(skipped_full, 0);
        assert_eq!(skipped_narrow, 0b0001, "scale 0 must be skipped whole");
        for &id in &coarse {
            assert_eq!(f_full[id].to_bits(), f_narrow[id].to_bits(), "f{id} moved when scale 0 was skipped");
        }
    }

    // ========================================================================
    // HDR route gates (HDR_PLAN chunk 2 — streaming PU front-end)
    // ========================================================================

    // ========================================================================
    // append2 gates (BANDVIS + conditioner + HL bins)
    // ========================================================================

    // --- Shared BANDVIS fixtures (gates doc V3). Extracted 2026-08-02 so
    // the P1.5 dst-activity adjudication tests measure the IDENTICAL
    // pixel content as the OFF-math characterization pins (formulas
    // verbatim from the original inline builders; the behavior test's
    // value pins + the gates-doc printed numbers gate the motion).
    // `benchmarks/bandvis_dst_activity_2026-08-02.md`. ---

    // ========================================================================
    // CSFW gates (chunk-3 tier-1: luminance-weighted GLOBAL_* twins)
    // ========================================================================

    // ========================================================================
}
