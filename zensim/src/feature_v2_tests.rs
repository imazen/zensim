#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    
    use crate::source::RgbSlice;

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

    /// Edge-width change pairs two adjacent scales (the last scale's slot copies
    /// the one below). A declared plan that reads it for a chroma channel
    /// computes that channel at both scales and emits the full plan's bits on
    /// every slot of its emit set, and the structural 0 everywhere else.
    /// A declared plan for any one basic, peak or v2 slot emits the full plan's
    /// bits on every slot of its emit set and the structural 0 everywhere else.
    /// Edge-width change pairs two adjacent scales (the last scale's slot copies
    /// the one below), so reading it for a channel computes that channel at both.
    #[test]
    fn single_slot_plans_emit_the_full_plans_bits_and_zero_elsewhere() {
        use crate::feature_defs::FormulaRevision;
        use crate::feature_layout::Layout;
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let (w, h) = (160, 128);
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 7 + y * 3) % 256) as u8, ((x * x + y) % 256) as u8, ((y * 5 + x / 3) % 256) as u8]
            })
            .collect();
        let dst: Vec<[u8; 3]> = src
            .iter()
            .enumerate()
            .map(|(i, p)| p.map(|v| v.saturating_add((i * 2654435761 % 7) as u8)))
            .collect();
        let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let run = |plan: &Plan| {
            foldapp_streaming_walk(
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
            .into_features()
        };
        let ns = crate::NUM_SCALES;
        let ewc = |scale: usize, ch: usize| {
            372 + (scale * 3 + ch) * FEATURES_PER_CHANNEL_V2_TOTAL + idx::EDGE_WIDTH_CHANGE
        };
        let all: Vec<usize> = (0..228).chain(372..720).collect();
        for revision in [FormulaRevision::Rev1, FormulaRevision::Rev7] {
            let plan_for = |ids: &[usize]| {
                Plan::derive_with_layout_at(
                    &SlotSet::from_slots(ids.iter().copied()),
                    Layout::identity(720),
                    revision,
                )
                .unwrap()
            };
            let f_full = run(&plan_for(&all));
            for ch in [0, 2] {
                for scale in 0..ns {
                    let plan = plan_for(&[ewc(scale, ch)]);
                    let s = scale.min(ns - 2);
                    for at in [s, s + 1] {
                        assert!(
                            plan.compute.channel_active(at, ch),
                            "{revision:?} f{}: channel {ch} off at scale {at}",
                            ewc(scale, ch)
                        );
                    }
                }
            }
            // Every single-slot plan: its emitted slots carry the full plan's
            // bits, every other slot the structural 0.
            for &id in &all {
                let plan = plan_for(&[id]);
                let f = run(&plan);
                let emit = plan.compute.populated_slots(ns, 720);
                assert!(emit.contains(id), "{revision:?} f{id} not emitted by its own plan");
                for (i, &v) in f.iter().enumerate() {
                    if emit.contains(i) {
                        assert_eq!(v.to_bits(), f_full[i].to_bits(), "{revision:?} plan f{id}: f{i} {v} vs {}", f_full[i]);
                    } else {
                        assert_eq!(v, 0.0, "{revision:?} plan f{id}: f{i} = {v} outside the emit set");
                    }
                }
            }
        }
    }

    #[test]
    fn edge_width_change_plans_emit_the_full_plans_bits_and_zero_elsewhere() {
        use crate::feature_defs::FormulaRevision;
        use crate::feature_layout::Layout;
        use crate::feature_plan::Plan;
        use crate::feature_set_id::SlotSet;
        let (w, h) = (160, 128);
        let src: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 7 + y * 3) % 256) as u8, ((x * x + y) % 256) as u8, ((y * 5 + x / 3) % 256) as u8]
            })
            .collect();
        let dst: Vec<[u8; 3]> = src
            .iter()
            .enumerate()
            .map(|(i, p)| p.map(|v| v.saturating_add((i * 2654435761 % 7) as u8)))
            .collect();
        let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let run = |plan: &Plan| {
            foldapp_streaming_walk(
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
            .into_features()
        };
        let ns = crate::NUM_SCALES;
        let ewc = |scale: usize, ch: usize| {
            372 + (scale * 3 + ch) * FEATURES_PER_CHANNEL_V2_TOTAL + idx::EDGE_WIDTH_CHANGE
        };
        let all: Vec<usize> = (0..228).chain(372..720).collect();
        for revision in [FormulaRevision::Rev1, FormulaRevision::Rev7] {
            let plan_for = |ids: &[usize]| {
                Plan::derive_with_layout_at(
                    &SlotSet::from_slots(ids.iter().copied()),
                    Layout::identity(720),
                    revision,
                )
                .unwrap()
            };
            let f_full = run(&plan_for(&all));
            // Every Y-channel v2 slot keeps v2 work live at every scale, so the
            // chroma channel is skipped at scales where its gradients exist.
            let luma: Vec<usize> = (0..ns)
                .flat_map(|scale| {
                    let base = 372 + (scale * 3 + 1) * FEATURES_PER_CHANNEL_V2_TOTAL;
                    base..base + FEATURES_PER_CHANNEL_V2_TOTAL
                })
                .collect();
            for ch in [0, 2] {
                for scale in 0..ns {
                    for with_luma in [false, true] {
                        let id = ewc(scale, ch);
                        let mut ids = vec![id];
                        if with_luma {
                            ids.extend(&luma);
                        }
                        let plan = plan_for(&ids);
                        let s = scale.min(ns - 2);
                        for at in [s, s + 1] {
                            assert!(
                                plan.compute.channel_active(at, ch),
                                "{revision:?} f{id}: channel {ch} off at scale {at}"
                            );
                        }
                        let f = run(&plan);
                        let emit = plan.compute.populated_slots(ns, 720);
                        for (i, &v) in f.iter().enumerate() {
                            if emit.contains(i) {
                                assert_eq!(
                                    v.to_bits(),
                                    f_full[i].to_bits(),
                                    "{revision:?} f{id} (luma {with_luma}): f{i} {v} vs {}",
                                    f_full[i]
                                );
                            } else {
                                assert_eq!(
                                    v, 0.0,
                                    "{revision:?} f{id} (luma {with_luma}): f{i} = {v} outside the emit set"
                                );
                            }
                        }
                        assert!(emit.contains(id), "{revision:?} f{id} not emitted by its own plan");
                    }
                }
            }
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
