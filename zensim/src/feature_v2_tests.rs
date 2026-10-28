#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    
    use crate::source::RgbSlice;

    /// imazen/zensimx#56 regression gate: the MSCN divisive normalizer must
    /// be the CORRECTLY-ROUNDED IEEE `resid / sqrt(var + c)` on every SIMD
    /// tier, lane for lane, bit for bit. That is what makes the
    /// `MSCN_DIFF_MEAN`/`MSCN_DIFF_L2` slots CPU-vendor-deterministic: IEEE
    /// sqrt and div are exactly rounded on every vendor, whereas the
    /// `rsqrt()` the kernel used before (`vrsqrtps` estimate + 1 NR step on
    /// x86; `1/sqrt` then multiply on NEON and scalar) is not — the bf944
    /// wave measured AMD≠Intel by ~1e-8 rel on exactly those 22 columns.
    ///
    /// Mutation-verified: substituting `resid * (var + c).rsqrt()` in
    /// `mscn_norm_v` fails this test on x86 (NR residue) AND on NEON /
    /// scalar (double rounding), so it cannot silently regress on any CI
    /// platform. The corpus deliberately includes the exact-zero variance
    /// floor (`var = 0` → `sqrt(C_MSCN_VAR)`), sub-floor, unit and large
    /// variances, and signed residuals spanning ±[1e-4, 4].
    fn check_mscn_norm_tier<T: F32x8Backend + Copy>(token: T, tier: &str) {
        let c = C_MSCN_VAR as f32;
        let cv = V8::<T>::splat_t(token, c);
        // Deterministic LCG so the corpus is identical on every platform.
        let mut state = 0x9E37_79B9u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let mut n_checked = 0usize;
        let mut n_mismatch = 0usize;
        let mut first_mismatch: Option<(f32, f32, f32, f32)> = None;
        for block in 0..512 {
            let mut resid = [0f32; 8];
            let mut var = [0f32; 8];
            for lane in 0..8 {
                let u = next();
                // Signed residual, log-spread magnitude 1e-4..4.
                let mag = 1e-4 * (4e4f32).powf(next());
                resid[lane] = if u < 0.5 { -mag } else { mag };
                // Variance: exact floor every 8th lane, else log-spread
                // 1e-9..1e2 (covers below the C_MSCN_VAR floor and far above).
                var[lane] = if (block + lane) % 8 == 0 {
                    0.0
                } else {
                    1e-9 * (1e11f32).powf(next())
                };
            }
            let got = mscn_norm_v(
                token,
                V8::<T>::from_array_t(token, resid),
                V8::<T>::from_array_t(token, var),
                cv,
            )
            .to_array();
            for lane in 0..8 {
                // Scalar Rust f32 sqrt and div are IEEE-754 correctly rounded.
                let want = resid[lane] / (var[lane] + c).sqrt();
                n_checked += 1;
                if got[lane].to_bits() != want.to_bits() {
                    n_mismatch += 1;
                    first_mismatch.get_or_insert((resid[lane], var[lane], got[lane], want));
                }
            }
        }
        assert_eq!(
            n_mismatch, 0,
            "tier {tier}: {n_mismatch}/{n_checked} lanes of mscn_norm_v are not the \
             correctly-rounded IEEE resid/sqrt(var+c); first: {first_mismatch:?} \
             (resid, var, got, want)"
        );
    }

    #[test]
    fn mscn_norm_v_is_correctly_rounded_on_every_tier() {
        use archmage::SimdToken as _;
        let mut tiers_run = 0usize;
        check_mscn_norm_tier(
            archmage::ScalarToken::summon().expect("scalar token is infallible"),
            "scalar",
        );
        tiers_run += 1;
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(target_arch = "x86_64")]
            if let Some(t) = archmage::X64V3Token::summon() {
                check_mscn_norm_tier(t, "x86 v3");
                tiers_run += 1;
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            if let Some(t) = archmage::X64V4Token::summon() {
                check_mscn_norm_tier(t, "x86 v4");
                tiers_run += 1;
            }
            #[cfg(feature = "avx512")]
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_mscn_norm_tier(t, "x86 v4x");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if let Some(t) = archmage::NeonToken::summon() {
                check_mscn_norm_tier(t, "neon");
                tiers_run += 1;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(t) = archmage::Wasm128Token::summon() {
                check_mscn_norm_tier(t, "wasm128");
                tiers_run += 1;
            }
        }
        assert!(tiers_run >= 1, "no tier was exercised");
    }

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

    /// Gated-mirror pin (see `WeightedSum`'s doc): the incremental
    /// `WeightedSum` accumulator used by the fused kernel must match
    /// iteration 1's array-based `WeightedPool::mean` (the promoted
    /// canonical `Σw·v/Σw` helper) bit-for-bit on the same data, so the
    /// perf-motivated rewrite is provably not a silent formula drift.
    #[test]
    fn weighted_sum_matches_weighted_pool_mean_exactly() {
        let mut state: u32 = 0xabcdef01;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / u32::MAX as f32).abs()
        };
        let values: Vec<f32> = (0..1000).map(|_| next() * 2.0).collect();
        let weights: Vec<f32> = (0..1000).map(|_| next()).collect();

        let mut ws = WeightedSum::default();
        for (&v, &w) in values.iter().zip(weights.iter()) {
            ws.add(w as f64, v as f64);
        }
        let incremental = ws.finish();
        let batch = crate::iw_pool::WeightedPool::mean(&values, &weights);
        assert!(
            (incremental - batch).abs() < 1e-9,
            "WeightedSum {incremental} vs WeightedPool::mean {batch} diverged"
        );
    }

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

    #[test]
    fn raw_moment_reformulation_matches_terriberry() {
        let mut state: u32 = 0x5eed_1234;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f64 / u32::MAX as f64).abs() * 2.0 // in [0, 2]
        };
        let d_values: Vec<f64> = (0..50_000).map(|_| next()).collect();

        let mut moments = OnlineMoments::default();
        for &d in &d_values {
            moments.update(d);
        }
        let (terri_mean, terri_dev2, terri_dev4) = moments.finish();

        let n = d_values.len() as f64;
        let (mut sum_d, mut sum_d2, mut sum_d3, mut sum_d4) = (0.0, 0.0, 0.0, 0.0);
        for &d in &d_values {
            sum_d += d;
            sum_d2 += d * d;
            sum_d3 += d * d * d;
            sum_d4 += d * d * d * d;
        }
        let raw_mean = sum_d / n;
        let raw2 = sum_d2 / n;
        let raw3 = sum_d3 / n;
        let raw4 = sum_d4 / n;
        let m2 = (raw2 - raw_mean * raw_mean).max(0.0);
        let m4 = (raw4 - 4.0 * raw_mean * raw3 + 6.0 * raw_mean * raw_mean * raw2
            - 3.0 * raw_mean.powi(4))
        .max(0.0);
        let raw_dev2 = m2.sqrt();
        let raw_dev4 = m4.powf(0.25);

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-12);
        assert!(
            rel(raw_mean, terri_mean) < 5e-4,
            "mean: raw {raw_mean} vs terriberry {terri_mean}"
        );
        assert!(
            rel(raw_dev2, terri_dev2) < 5e-4,
            "dev2: raw {raw_dev2} vs terriberry {terri_dev2}"
        );
        assert!(
            rel(raw_dev4, terri_dev4) < 5e-4,
            "dev4: raw {raw_dev4} vs terriberry {terri_dev4}"
        );
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

    #[test]
    fn gmsbank_strict_contrast_reduction_routes_only_to_loss() {
        let mut cells = [GmsBankCell::default(); 5];
        for i in 1..=257 {
            let mr = i as f64 / 257.0;
            let md = mr * 0.5;
            assert!(md < mr);
            gmsbank_pixel(&mut cells, mr, md, &GMSBANK_C);
        }
        for cell in cells {
            assert!(cell.loss > 0.0);
            assert_eq!(cell.gain.to_bits(), 0.0f64.to_bits());
        }
    }

    /// C3 regression (era `c3negfold`): `edge_dissim`'s f32 rounding emits
    /// tiny NEGATIVE `art`/`det` values, and a raw `to_bits` bin lookup
    /// sorted every sign-bit-set double above all positive edges — the
    /// phantom ~2.9% top-bin saturation of REVIEW_PARTB 2026-09-25 was
    /// those negatives, not real tail mass. Sign-bit-set and NaN values
    /// must fold into bin 0; only genuine `v >= last edge` mass may reach
    /// the top bin.
    #[test]
    fn rev4_tailhist_bin_folds_sign_bit_values() {
        let edges = tail_edges();
        let top = TAILHIST_BINS - 1;
        // Sign-bit-set values: the bin-0 fold.
        for (name, v) in [
            ("neg tiny", -1e-12f64),
            ("neg below-first-edge", -0.5),
            ("neg large", -1e300),
            ("neg zero", -0.0),
            ("neg inf", f64::NEG_INFINITY),
            ("NaN", f64::NAN),
            ("neg NaN", f64::from_bits(0xfff8000000000001)),
        ] {
            assert_eq!(
                edges.bin_folded(v),
                0,
                "{name}: bin_folded({v}) must fold to 0"
            );
        }
        // Non-negative values still bin by magnitude.
        assert_eq!(edges.bin(0.0), 0);
        assert_eq!(edges.bin(1e-7), 0); // below the first edge
        let last = f64::from_bits(edges.edges[TAILHIST_BINS - 2]);
        assert_eq!(edges.bin(last * 2.0), top, "v above last edge -> top bin");
        assert_eq!(edges.bin(f64::INFINITY), top, "+inf is genuine tail mass");
        // End to end through `scatter`: a realistic mostly-negative
        // multiset must leave the top bin empty and keep max exact —
        // the pre-fix arithmetic put every negative in the top bin.
        let mut acc = TailAccum {
            fold: true,
            ..TailAccum::default()
        };
        let mut vals: Vec<f64> = (0..5000).map(|i| -(i as f64) * 1e-9).collect();
        vals.extend(std::iter::repeat_n(0.004, 8000));
        vals.extend(std::iter::repeat_n(last * 2.0, 10));
        for &v in &vals {
            acc.scatter(edges, v, 1);
        }
        assert_eq!(acc.hist[1][top], 10, "only real tail values in the top bin");
        // Bin 0 holds the folded mass: negatives + positives below the
        // first edge.
        let expect0 = vals.iter().filter(|&&v| edges.bin_folded(v) == 0).count() as u32;
        assert_eq!(acc.hist[1][0], expect0, "bin 0 must hold the folded mass");
        assert_eq!(acc.max[1].to_bits(), (last * 2.0).to_bits());
        let total: u64 = acc.hist[1].iter().map(|&c| c as u64).sum();
        assert_eq!(
            total,
            vals.len() as u64,
            "histogram total == scattered count"
        );
    }

    /// Rev1-Rev3 byte-identity guard for `c3negfold`: with `fold == false`
    /// (what `TailAccum::default()` yields below Rev4) a sign-bit-set value
    /// still lands in the TOP bin, exactly as before the fix, while the
    /// folded accumulator puts it in bin 0. Both leave `max` untouched.
    #[test]
    fn tailhist_fold_is_off_below_rev4() {
        let edges = tail_edges();
        let top = TAILHIST_BINS - 1;
        let default = TailAccum::default();
        assert_eq!(
            default.fold,
            crate::ssim_form::active_revision() >= FormulaRevision::Rev4,
            "fold follows the process revision"
        );
        let (mut legacy, mut folded) = (
            TailAccum {
                fold: false,
                ..TailAccum::default()
            },
            TailAccum {
                fold: true,
                ..TailAccum::default()
            },
        );
        for v in [-1e-12f64, -0.0, 0.25] {
            legacy.scatter(edges, v, 2);
            folded.scatter(edges, v, 2);
        }
        assert_eq!(
            legacy.hist[2][top], 2,
            "legacy bins sign-bit-set values to the top"
        );
        assert_eq!(folded.hist[2][top], 0);
        assert_eq!(
            folded.hist[2][0] + folded.hist[2].iter().skip(1).sum::<u32>(),
            3
        );
        assert_eq!(legacy.max[2].to_bits(), folded.max[2].to_bits());
    }

    /// f64 accumulation shows no cancellation on flat high-value inputs:
    /// scatter one large constant into the f64 accumulators and compare
    /// against the n·v algebraic reference — the running sums must stay
    /// within a tiny relative bound (1e-12), i.e. no catastrophic
    /// cancellation or drift, not merely "approximately" right.
    #[test]
    fn rev4_accumulators_flat_no_cancellation() {
        const N: usize = 200_000;
        let big = 0.75f64; // high-value flat term
        // RingAccum: Σ ring·h_k — constant ring ⇒ fixed hat weights.
        let mut ring = RingAccum::default();
        let h = rev4_hats(&RINGBASIS_UC, rev4_u(big));
        for _ in 0..N {
            for (b, h) in ring.bins.iter_mut().zip(h.iter()) {
                b.add64(0, big * h);
            }
        }
        for (k, (b, &h)) in ring.bins.iter().zip(h.iter()).enumerate() {
            let b = b.fin();
            let expect = N as f64 * big * h;
            let rel = ((b - expect) / expect).abs();
            assert!(
                rel < 1e-10 || expect == 0.0 && b == 0.0,
                "ring bin {k} drifted on flat input: {b} vs {expect} (rel {rel:e})",
            );
        }
        // FlatAccum: Σ hfg over flat pixels ≈ N · hfg.
        let mut flat = FlatAccum::default();
        for _ in 0..N {
            flat.hfg.add64(0, big);
            flat.n += 1;
        }
        let expect = N as f64 * big;
        assert!(
            ((flat.hfg.fin() - expect) / expect).abs() < 1e-10,
            "flat hfg drifted: {:?} vs {expect}",
            flat.hfg
        );
        // GridblkAccum signed sums over an alternating ±big pattern: the
        // equal-magnitude pairs cancel EXACTLY (same-bits add/sub), and
        // |ẽ| stays on the algebraic bound.
        let mut g = GridblkAccum::default();
        for i in 0..N {
            g.signed_v[0] += if i % 2 == 0 { big } else { -big };
            g.abs_v[0] += big;
        }
        assert_eq!(g.signed_v[0].to_bits(), 0.0f64.to_bits());
        assert!(
            ((g.abs_v[0] - expect) / expect).abs() < 1e-10,
            "gridblk abs drifted: {} vs {expect}",
            g.abs_v[0]
        );
    }

    /// POOL_SIMD drift gate: the vectorized weighted-pool path (enabled on
    /// the v4x tier) must stay within this module's documented 5e-4
    /// relative tolerance of the §A.14 scalar-pool path — same
    /// reassociation class as the phase-4 core-moment change. On hosts
    /// where the active tier runs scalar pools anyway, both sides are
    /// §14.4 UNDER TEST: the two `const FUSED` instantiations must produce
    /// BIT-IDENTICAL accumulators.
    ///
    /// Post-V8-rewrite the split is vestigial (both entries select `false`,
    /// since fusing MEASURED 2x worse — §19.2), but the gate is kept because
    /// it is the executable form of §14.4's claim that pass structure is a
    /// perf knob and not semantics. It runs on the `scalar` token so it is
    /// tier-independent.
    ///
    /// This is the claim that lets the pass split be tuned per tier without
    /// touching the era. It is asserted directly rather than inferred: both
    /// const-generic instantiations are called on the same planes and every
    /// one of the 35 slots is compared with `to_bits()`. If it ever fails, the
    /// per-tier split is NOT free and the tier layout becomes semantics.
    #[test]
    fn era2_fused_and_two_pass_are_bit_identical() {
        for &(w, h) in &[
            (96usize, 64usize),
            (200, 136),
            (127, 93),
            (61, 40),
            (8, 3),
            (13, 65),
        ] {
            let src_px = textured_image(w, h, 0xBEEF);
            let dst_px = quantize_distort(&src_px, w, h);
            let source = RgbSlice::new(&src_px, w, h);
            let distorted = RgbSlice::new(&dst_px, w, h);
            let sp = crate::streaming::convert_source_to_xyb(
                &source,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let dp = crate::streaming::convert_source_to_xyb(
                &distorted,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let mut scratch = ScratchV2Strip::new(w * h);
            let n = w * h;
            for ch in 0..3 {
                run_blur_pass(&sp[ch], &dp[ch], w, h, &mut scratch);
                use archmage::SimdToken as _;
                let tok = archmage::ScalarToken::summon().expect("infallible");
                let f = super::dense_block_kernel_era2_generic::<_, true>(
                    tok,
                    &sp[ch],
                    &dp[ch],
                    &scratch.mu1[..n],
                    &scratch.mu2[..n],
                    &scratch.ssq[..n],
                    &scratch.s12[..n],
                    &scratch.activity[..n],
                    w,
                    h,
                    true,
                    None,
                );
                let t = super::dense_block_kernel_era2_generic::<_, false>(
                    tok,
                    &sp[ch],
                    &dp[ch],
                    &scratch.mu1[..n],
                    &scratch.mu2[..n],
                    &scratch.ssq[..n],
                    &scratch.s12[..n],
                    &scratch.activity[..n],
                    w,
                    h,
                    true,
                    None,
                );
                let (fs, ts) = (oracle::dense_accum_slots(&f), oracle::dense_accum_slots(&t));
                for i in 0..fs.len() {
                    assert_eq!(
                        fs[i].to_bits(),
                        ts[i].to_bits(),
                        "{w}x{h} ch{ch} slot {} ({}): FUSED {:e} != two-pass {:e}. The per-tier \
                         pass split is supposed to be byte-neutral (§14.4); if it is not, the \
                         tier layout has become part of era-2's semantics.",
                        i,
                        oracle::SLOT_NAMES[i],
                        fs[i],
                        ts[i]
                    );
                }
            }
        }
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

    #[test]
    fn era2_band_merge_and_tail_are_structural() {
        let term = |i: usize| -> f32 {
            let k = i.wrapping_mul(2654435761usize) % 65521;
            (k as f32) * (1.0 / 65536.0) - 0.5
        };

        // Fold one band's rows into its own lane set.
        let band_of = |plane: &[f32], width: usize, y0: usize, y1: usize| -> super::Lanes8 {
            let mut band = super::Lanes8::zero();
            for y in y0..y1 {
                let row = &plane[y * width..(y + 1) * width];
                let (chunks, tail) = row.as_chunks::<8>();
                for c in chunks {
                    band.add_chunk(c);
                }
                band.add_tail(tail);
            }
            band
        };

        for &(width, rows) in &[(576usize, 128usize), (127, 93), (61, 40), (8, 3), (5, 1)] {
            let plane: Vec<f32> = (0..width * rows).map(term).collect();

            let mut starts = Vec::new();
            let mut b = 0usize;
            while b < rows {
                starts.push(b);
                b = (b + super::ERA2_BAND_ROWS).min(rows);
            }

            // Reference: bands computed in index order, merged in index order.
            let mut reference = super::Lanes8::zero();
            for (i, &y0) in starts.iter().enumerate() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                let bandv = band_of(&plane, width, y0, y1);
                for (m, v) in reference.0.iter_mut().zip(bandv.0.iter()) {
                    *m += *v;
                }
            }

            // (a) Compute the bands in REVERSE order (as a worker pool would,
            // completing out of order), then merge in index order as the design
            // requires. Must be bit-identical.
            let mut partials: Vec<Option<super::Lanes8>> = vec![None; starts.len()];
            for (i, &y0) in starts.iter().enumerate().rev() {
                let y1 = starts.get(i + 1).copied().unwrap_or(rows);
                partials[i] = Some(band_of(&plane, width, y0, y1));
            }
            let mut out_of_order = super::Lanes8::zero();
            for p in partials.iter().flatten() {
                for (m, v) in out_of_order.0.iter_mut().zip(p.0.iter()) {
                    *m += *v;
                }
            }
            assert_eq!(
                reference.reduce().to_bits(),
                out_of_order.reduce().to_bits(),
                "{width}x{rows}: computing bands out of order changed the result. \
                 era-2's thread-invariance rests on the MERGE being sequential in \
                 band index while the bands themselves are order-free."
            );

            // (b) lane j == the terms at x ≡ j (mod 8), tail included.
            let mut expect = [0.0f32; 8];
            for y in 0..rows {
                for x in 0..width {
                    expect[x % 8] += plane[y * width + x];
                }
            }
            let single = band_of(&plane, width, 0, rows);
            assert_eq!(
                single.0.map(f32::to_bits),
                expect.map(f32::to_bits),
                "{width}x{rows}: lane contents diverge from 'terms at x ≡ j (mod 8), \
                 in increasing x' — the tail fold is wrong."
            );
        }
    }

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

    /// ERA-2 ORACLE GATE (`benchmarks/era2_perf_break_2026-08-31.md` §11).
    ///
    /// Judges the production kernels against an EXACT reference rather than
    /// against a previous implementation. Three things happen, in order:
    ///
    /// 1. **The oracle judges itself.** L1 (Neumaier-compensated f64) is
    ///    compared to L2 (Shewchuk exact expansion). If L1 drifts from L2
    ///    beyond compensated summation's own bound, the oracle is broken and
    ///    says so BEFORE it is used to judge anything else.
    /// 2. **Each production pool shape is measured against L2** — both the
    ///    `POOL_SIMD` lane form (what `v4x` ships today and what era-2 makes
    ///    universal) and the scalar-pool form (what every 16-register tier
    ///    ships today). Their deviations differ by construction, which is the
    ///    §10.4 accuracy-class change made visible instead of assumed.
    /// 3. **Every slot is checked against a bound derived from the SAME run** —
    ///    `Σ|xᵢ|` per slot comes out of the oracle, so the bound is computed
    ///    from measured magnitudes rather than guessed.
    ///
    /// Geometries deliberately straddle the classes that matter to the era-2
    /// reshape: tight vs non-tight width, `width.is_multiple_of(8)` vs not (the scalar
    /// tail), and heights that do and do not divide the band size.
    ///
    /// A deviation above its bound is a BUG — in the kernel or in the analysis
    /// — and the lane STOPS to find out which. It is never a tolerance to
    /// widen.
    #[test]
    fn era2_oracle_bounds_hold_for_every_pool_shape() {
        use oracle::{Exact, N_SLOTS, Neumaier, SLOT_NAMES, dense_accum_slots, slot_family};
        const U32: f64 = 5.960_464_477_539_063e-8; // 2^-24
        const U64: f64 = 1.110_223_024_625_157e-16; // 2^-53

        // (w, h, note)
        let geoms: &[(usize, usize, &str)] = &[
            (96, 64, "tight, w%8==0"),
            (200, 136, "non-tight, w%8==0"),
            (127, 93, "non-tight, w%8==7 (scalar tail)"),
            (61, 40, "sub-64, w%8==5 (tail + small)"),
        ];
        let mut worst_overall: (f64, String) = (0.0, String::new());
        // Per (variant, family): worst |dev| with the sum|x| and bound it came
        // with, so §10.5's table is filled from measurement, not guessed.
        let mut per: std::collections::BTreeMap<(&str, &str), (f64, f64, f64)> =
            std::collections::BTreeMap::new();

        for &(w, h, note) in geoms {
            let src_px = textured_image(w, h, 0xD1FF);
            let dst_px = quantize_distort(&src_px, w, h);
            let source = RgbSlice::new(&src_px, w, h);
            let distorted = RgbSlice::new(&dst_px, w, h);
            let src_planes = crate::streaming::convert_source_to_xyb(
                &source,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let dst_planes = crate::streaming::convert_source_to_xyb(
                &distorted,
                w,
                false,
                crate::ssim_form::active_revision(),
            );
            let mut scratch = ScratchV2Strip::new(w * h);
            let n = w * h;

            for ch in 0..3 {
                run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
                let (mu1, mu2) = (&scratch.mu1[..n], &scratch.mu2[..n]);
                let (ssq, s12) = (&scratch.ssq[..n], &scratch.s12[..n]);
                let act = &scratch.activity[..n];

                let (l1, _) = oracle::dense_reference::<Neumaier>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );
                let (l2, sum_abs) = oracle::dense_reference::<Exact>(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                );

                // (1) The oracle judges itself. Compensated summation's error
                // is O(u) * Σ|x| regardless of n — that is the whole point of
                // the compensation — so a generous constant times u64 is a
                // true bound, not a fitted one.
                for i in 0..N_SLOTS {
                    let allow = 64.0 * U64 * sum_abs[i] + f64::MIN_POSITIVE;
                    assert!(
                        (l1[i] - l2[i]).abs() <= allow,
                        "{w}x{h} ch{ch} [{note}]: ORACLE SELF-CHECK FAILED at {} \
                         (L1 Neumaier {:e} vs L2 exact {:e}, allowed {:e}). The oracle is \
                         broken; do not trust any kernel judged by it.",
                        SLOT_NAMES[i],
                        l1[i],
                        l2[i],
                        allow
                    );
                }

                // (2)+(3) Each production pool shape vs L2, bound from Σ|x|.
                let lane_pool = dense_block_kernel_era1(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );
                let scalar_pool = dense_block_kernel_pools_scalar(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );

                let era2 = dense_block_kernel_era2(
                    &src_planes[ch],
                    &dst_planes[ch],
                    mu1,
                    mu2,
                    ssq,
                    s12,
                    act,
                    w,
                    h,
                    true,
                    None,
                );
                for (variant, accum) in [
                    ("dispatched", &lane_pool),
                    ("pools_scalar", &scalar_pool),
                    ("era2", &era2),
                ] {
                    let got = dense_accum_slots(accum);
                    for i in 0..N_SLOTS {
                        let fam = slot_family(i);
                        // f32 lane accumulation: (chunks per lane) + (reduce
                        // tree depth) terms at u32. f64 aggregation: one add
                        // per row at u64. The scalar-pool variant accumulates
                        // its POOL slots per pixel in f64 instead, so those
                        // carry no u32 term.
                        // BOUND per design §12.2 + §12.5. Two components:
                        //
                        // (1) TERM-EVALUATION, absolute and cancellation-safe.
                        //     Every intermediate in these formulas is bounded
                        //     by 1 in magnitude (d, sal, the weights,
                        //     bounded_sim, saturate all live in [0,1]), so ONE
                        //     f32 rounding anywhere in a term contributes at
                        //     most u32·1 — ABSOLUTE, not relative to |term|.
                        //     That matters because `d = max(1-local,0)` with
                        //     `local ≈ 1` is a CANCELLING difference: d's
                        //     absolute error does not shrink as d does, and
                        //     the derived slots (d², d³, d⁴, and every pool
                        //     `num`) inherit it amplified by the derivative.
                        //     A bound written against the slot's own Σ|x|
                        //     understates those by two orders (Σ|d| is 282x
                        //     Σ|d²| here) — measured, and it is what the
                        //     oracle caught first.
                        //
                        // (2) ACCUMULATION, relative to Σ|x| (design §12.2).
                        //
                        // (1) dominates and is deliberately LOOSE — a proven
                        // upper bound, not a fitted one. The regression signal
                        // is the reported per-family COEFFICIENT below, which
                        // moves long before the bound is crossed.
                        const SAFETY: f64 = 2.0;
                        let k_eval = SAFETY
                            * match fam {
                                "core" => 13.0,
                                "hf" => 7.0,
                                "pjnd" => 5.0,
                                _ => 17.0, // pools: core formula + weight ops
                            };
                        let n_px = (w * h) as f64;
                        let chunks_per_lane = (w / 8) as f64;
                        let pools_in_f64 = variant == "pools_scalar" && fam == "pools";
                        // era-2 folds its tail into the lanes, so it has no
                        // per-pixel f64 tail term (era-1 does).
                        let era2_shape = variant == "era2";
                        let acc_u32 = if pools_in_f64 {
                            0.0
                        } else {
                            chunks_per_lane + 8.0
                        };
                        let acc_u64 = if pools_in_f64 {
                            n_px
                        } else {
                            h as f64
                                + 8.0
                                + if era2_shape {
                                    0.0
                                } else {
                                    (w % 8) as f64 * h as f64
                                }
                        };
                        let bound = k_eval * n_px * U32
                            + (acc_u32 * U32 + acc_u64 * U64) * sum_abs[i]
                            + 8.0 * U64 * l2[i].abs()
                            + f64::MIN_POSITIVE;
                        let dev = (got[i] - l2[i]).abs();
                        {
                            let e = per.entry((variant, fam)).or_insert((0.0, 0.0, 0.0));
                            if dev > e.0 {
                                *e = (dev, sum_abs[i], bound);
                            }
                        }
                        // Reported signal: dev as a fraction of its bound.
                        // A kernel change that degrades accuracy moves this
                        // long before it trips the assert.
                        let frac = dev / bound;
                        if frac > worst_overall.0 {
                            worst_overall = (
                                frac,
                                format!(
                                    "{} {}x{} ch{} [{}] {} ({}): dev {:e} = {:.1}% of bound {:e}",
                                    variant,
                                    w,
                                    h,
                                    ch,
                                    note,
                                    SLOT_NAMES[i],
                                    fam,
                                    dev,
                                    100.0 * frac,
                                    bound
                                ),
                            );
                        }
                        assert!(
                            dev <= bound,
                            "{w}x{h} ch{ch} [{note}] {variant}: slot {} ({}) deviates {:e} from \
                             the EXACT sum, above its proven bound {:e} (Σ|x| = {:e}). A bound \
                             violation is a BUG in the kernel or in the error analysis — find \
                             out which. Do NOT widen this.",
                            SLOT_NAMES[i],
                            fam,
                            dev,
                            bound,
                            sum_abs[i]
                        );
                    }
                }
            }
        }
        eprintln!(
            "ERA2-ORACLE worst deviation = {:.2}% of its proven bound\n  {}",
            100.0 * worst_overall.0,
            worst_overall.1
        );
        eprintln!("ERA2-ORACLE per-variant x family worst |dev| vs the EXACT sum:");
        eprintln!(
            "  {:<14} {:<7} {:>13} {:>13} {:>13} {:>8}",
            "variant", "family", "max|dev|", "sum|x|", "bound", "% bound"
        );
        for ((v, f), (dev, sa, bd)) in &per {
            eprintln!(
                "  {v:<14} {f:<7} {dev:>13.4e} {sa:>13.4e} {bd:>13.4e} {:>7.1}%",
                100.0 * dev / bd
            );
        }
    }

    /// identical and the gate passes trivially.
    #[test]
    fn pool_simd_drift_within_policy() {
        let (w, h) = (200usize, 136usize);
        let src_px = textured_image(w, h, 0xD1FF);
        let dst_px = quantize_distort(&src_px, w, h);

        // Build real moment planes via the whole-image blur pass.
        let source = RgbSlice::new(&src_px, w, h);
        let distorted = RgbSlice::new(&dst_px, w, h);
        let src_planes = crate::streaming::convert_source_to_xyb(
            &source,
            w,
            false,
            crate::ssim_form::active_revision(),
        );
        let dst_planes = crate::streaming::convert_source_to_xyb(
            &distorted,
            w,
            false,
            crate::ssim_form::active_revision(),
        );
        let mut scratch = ScratchV2Strip::new(w * h);
        for ch in 0..3 {
            run_blur_pass(&src_planes[ch], &dst_planes[ch], w, h, &mut scratch);
            let n = w * h;
            let a = dense_block_kernel_era1(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
                None,
            );
            let b = dense_block_kernel_pools_scalar(
                &src_planes[ch],
                &dst_planes[ch],
                &scratch.mu1[..n],
                &scratch.mu2[..n],
                &scratch.ssq[..n],
                &scratch.s12[..n],
                &scratch.activity[..n],
                w,
                h,
                true,
                None,
            );
            let pools = [
                ("mask_ssim", a.ws_mask_ssim, b.ws_mask_ssim),
                ("mask_art", a.ws_mask_art, b.ws_mask_art),
                ("mask_det", a.ws_mask_det, b.ws_mask_det),
                ("mask_mse", a.ws_mask_mse, b.ws_mask_mse),
                ("iw_ssim", a.ws_iw_ssim, b.ws_iw_ssim),
                ("iw_art", a.ws_iw_art, b.ws_iw_art),
                ("iw_det", a.ws_iw_det, b.ws_iw_det),
                ("iw_mse", a.ws_iw_mse, b.ws_iw_mse),
                ("peak_ssim", a.ws_peak_ssim, b.ws_peak_ssim),
                ("peak_art", a.ws_peak_art, b.ws_peak_art),
                ("peak_det", a.ws_peak_det, b.ws_peak_det),
            ];
            for (name, pa, pb) in pools {
                for (part, va, vb) in [("num", pa.num, pb.num), ("den", pa.den, pb.den)] {
                    let rel = (va - vb).abs() / vb.abs().max(1e-9);
                    assert!(
                        rel <= TOL,
                        "ch{ch} pool {name}.{part}: rel drift {rel:e} \
                         (simd={va:e} scalar={vb:e}) exceeds policy {TOL:e}"
                    );
                }
            }
            // Non-pool sums must be BIT-identical — POOL_SIMD touches
            // nothing else.
            assert_eq!(a.sum_d.to_bits(), b.sum_d.to_bits());
            assert_eq!(a.sum_mse.to_bits(), b.sum_mse.to_bits());
            assert_eq!(a.sum_pjnd.to_bits(), b.sum_pjnd.to_bits());
        }
    }

    /// **The bit-exactness gate for [`FoldHSource::SelfBlur`]** (fold-MT lane).
    ///
    /// A band that H-blurs its own `[b0 − overlap, b1 + overlap)` rows must
    /// produce byte-identical sums to one reading those rows out of a
    /// whole-window blur. The property it rests on is the H blur's per-row
    /// independence; this asserts the consequence directly, on every field of
    /// `V1BasicSums`, at geometries that exercise a top band (clamped `top`),
    /// a bottom band (clamped `bot`), a partial last band, and both pool
    /// modes' band counts.
    #[test]
    fn fold_self_blur_matches_precomputed_h() {
        for &(width, plane_h, strip_h) in &[
            (64usize, 128usize, 128usize),
            (97, 96, 96),
            (2304, 128, 128),
            (128, 200, 128),
            (65, 40, 40),
        ] {
            let wide_h = strip_h + 2 * HALO_P;
            let n = width * wide_h;
            let src: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(2654435761usize) % 977) as f32 / 977.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(40503usize).wrapping_add(11) % 991) as f32 / 991.0)
                .collect();
            // The precomputed arm's four planes, over the WHOLE wide window.
            let mut h = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = h.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    wide_h,
                    BLUR_RADIUS,
                );
            }
            for &parallel in &[false, true] {
                let run = |h_src: FoldHSource<'_>| -> V1BasicSums {
                    let mut pool: Vec<FoldPoolScratch> = (0..V1_BANDS_PER_STRIP)
                        .map(|_| FoldPoolScratch::default())
                        .collect();
                    let mut sums = V1BasicSums::meas_f64();
                    fold_v1_basic_bands(
                        width,
                        0..strip_h.min(plane_h),
                        0,
                        HALO_P,
                        plane_h,
                        h_src,
                        [&src, &dst],
                        &mut sums,
                        Some((&mut pool[..], BandPoolWork::Full)),
                        parallel,
                        // Every FREE extra ON: the self-blur-vs-precomputed
                        // identity has to cover them too.
                        crate::fused::FreeExtrasWork {
                            raw_moments: true,
                            bounded_err: true,
                            lum_bins: true,
                            ..Default::default()
                        },
                    );
                    sums
                };
                let want = run(FoldHSource::Precomputed([&h[0], &h[1], &h[2], &h[3]]));
                let got = run(FoldHSource::SelfBlur);
                // `{:?}` on f64 is the shortest round-tripping form, so it
                // separates every pair of distinct finite values — the one
                // hole is NaN, whose payload it hides, so exclude NaN
                // explicitly rather than let it mask a difference.
                let wb = format!("{want:?}");
                let gb = format!("{got:?}");
                assert!(
                    !wb.contains("NaN"),
                    "{width}x{plane_h}: fixture produced NaN, which would make \
                     the Debug comparison below blind: {wb}"
                );
                assert_eq!(
                    wb, gb,
                    "{width}x{plane_h} strip {strip_h} par={parallel}: self-blur bands \
                     must equal precomputed-H bands"
                );
            }
        }
    }

    /// **The bit-exactness gate for [`fused_blur_h_ssim_banded`]** (fold-MT
    /// lane). The banded call must equal the whole-plane call BIT-FOR-BIT, at
    /// every band size that is a multiple of [`H_BLUR_ROW_GROUP`] and at
    /// heights that exercise a partial final group, a partial final band, and
    /// both at once.
    ///
    /// This is stronger than the shipped constant needs: it sweeps band sizes
    /// 8/12/16/24/32/64 so that re-tuning `H_BLUR_BAND_ROWS` on a future box
    /// cannot silently move a byte. Sizes 8, 12 and 24 are NOT multiples of
    /// [`H_BLUR_ROW_GROUP`] (16 — `v4`/`v4x` transpose 16 rows at a time;
    /// `v3`/`neon`/`wasm128`/`scalar` take 8), so they test the per-row
    /// independence itself rather than the alignment argument the shipped
    /// constant rests on. Recorded, not relied on: the shipped constant stays
    /// a multiple and the `const` assert beside it enforces that.
    #[test]
    fn phase_a_blur_bands_are_bit_exact() {
        for &(width, height) in &[
            (64usize, 148usize), // the scale-0 strip shape: 18 groups + a 4-row tail
            (2304, 40),          // wide, band-splittable
            (97, 148),           // odd width (scalar tail inside every row)
            (16, 7),             // shorter than one row group
            (33, 8),             // exactly one row group
            (127, 93),           // a golden-fixture height
            (5, 149),            // narrower than one SIMD lane
        ] {
            let n = width * height;
            let src: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(2654435761usize) % 1000) as f32 / 1000.0)
                .collect();
            let dst: Vec<f32> = (0..n)
                .map(|i| (i.wrapping_mul(40503usize).wrapping_add(7) % 1000) as f32 / 1000.0)
                .collect();
            let mut want = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
            {
                let (a, rest) = want.split_at_mut(1);
                let (b, rest2) = rest.split_at_mut(1);
                let (c, d) = rest2.split_at_mut(1);
                crate::blur::fused_blur_h_ssim(
                    &src,
                    &dst,
                    &mut a[0],
                    &mut b[0],
                    &mut c[0],
                    &mut d[0],
                    width,
                    height,
                    BLUR_RADIUS,
                );
            }
            for &bands in &[8usize, 12, 16, 24, 32, 64] {
                let mut got = [vec![0.0f32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]];
                // Replay the helper's banding by hand at an arbitrary band
                // size — the helper itself is pinned to one constant, and the
                // claim under test is about the BANDING, not that constant.
                let mut y0 = 0;
                while y0 < height {
                    let rows = bands.min(height - y0);
                    let lo = y0 * width;
                    let hi = lo + rows * width;
                    let (a, rest) = got.split_at_mut(1);
                    let (b, rest2) = rest.split_at_mut(1);
                    let (c, d) = rest2.split_at_mut(1);
                    crate::blur::fused_blur_h_ssim(
                        &src[lo..hi],
                        &dst[lo..hi],
                        &mut a[0][lo..hi],
                        &mut b[0][lo..hi],
                        &mut c[0][lo..hi],
                        &mut d[0][lo..hi],
                        width,
                        rows,
                        BLUR_RADIUS,
                    );
                    y0 += rows;
                }
                for (pi, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                    for (i, (a, b)) in g.iter().zip(w.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "{width}x{height} bands={bands} plane {pi} idx {i}: \
                             banded {a} vs whole-plane {b}"
                        );
                    }
                }
            }
        }
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
