//! Cross-tier reproducibility tests for zensim.
//!
//! Verifies that zensim produces identical (or bounded-ULP) results across
//! all SIMD dispatch tiers: scalar, SSE2, AVX2+FMA, AVX-512.
//!
//! Uses archmage's `for_each_token_permutation` to iterate over every unique
//! combination of disabled SIMD tokens on the current CPU.
//!
//! Run with: `cargo test -p zensim --test cross_tier -- --test-threads=1`

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::{ZensimResult, compute_zensim};

/// Generate deterministic test images: gradient source + small per-pixel distortion.
fn generate_test_images(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[128u8, 128, 128]; n];
    let mut dst = vec![[128u8, 128, 128]; n];
    for y in 0..h {
        for x in 0..w {
            let r = ((x * 255) / w) as u8;
            let g = ((y * 255) / h) as u8;
            let b = ((x + y) * 127 / (w + h)) as u8;
            src[y * w + x] = [r, g, b];
            dst[y * w + x] = [
                r.saturating_add(3),
                g.saturating_sub(2),
                b.saturating_add(1),
            ];
        }
    }
    (src, dst)
}

/// Max ULP distance between two f64 values.
/// Returns None for NaN or sign-mismatch (infinitely far apart).
fn ulp_distance(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    if a == b {
        return Some(0);
    }
    if a.is_sign_positive() != b.is_sign_positive() && a != 0.0 && b != 0.0 {
        return None;
    }
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    Some(ai.abs_diff(bi))
}

/// Classify a token permutation label into a tier name for grouping.
fn classify_tier(label: &str, disabled: &[&str]) -> &'static str {
    // Check what's disabled to determine effective ISA level
    let v3_disabled = disabled.contains(&"x86-64-v3");
    let v4_disabled = disabled.contains(&"AVX-512");
    let v2_disabled = disabled.contains(&"x86-64-v2");

    if v3_disabled || v2_disabled {
        "scalar/SSE2"
    } else if v4_disabled {
        "AVX2 (v3)"
    } else if label.contains("all enabled") || (!v4_disabled && !v3_disabled) {
        "AVX-512 (v4)"
    } else {
        "unknown"
    }
}

/// Collect results from all token permutations, group by effective tier,
/// and report divergence between tiers.
#[test]
fn score_reproducibility_across_tiers() {
    let w = 256;
    let h = 256;
    let (src, dst) = generate_test_images(w, h);

    let mut results: Vec<(String, Vec<String>, ZensimResult)> = Vec::new();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = compute_zensim(&src, &dst, w, h).expect("compute_zensim failed");
        let disabled: Vec<String> = perm.disabled.iter().map(|s| s.to_string()).collect();
        results.push((perm.label.clone(), disabled, result));
    });

    eprintln!("\n=== Cross-tier reproducibility test (256x256) ===");
    eprintln!("Permutations run: {}", report.permutations_run);
    for w in &report.warnings {
        eprintln!("  warning: {w}");
    }
    assert!(
        report.permutations_run >= 2,
        "Need at least 2 permutations (got {})",
        report.permutations_run,
    );

    // Group results by effective tier
    let mut tier_results: std::collections::BTreeMap<&'static str, Vec<(String, &ZensimResult)>> =
        std::collections::BTreeMap::new();
    for (label, disabled, result) in &results {
        let disabled_strs: Vec<&str> = disabled.iter().map(|s| s.as_str()).collect();
        let tier = classify_tier(label, &disabled_strs);
        tier_results
            .entry(tier)
            .or_default()
            .push((label.clone(), result));
    }

    eprintln!("\nResults by tier:");
    for (tier, entries) in &tier_results {
        for (label, r) in entries {
            eprintln!(
                "  [{tier:14}] {:50} score={:.10}  raw_dist={:.12}",
                label, r.score, r.raw_distance,
            );
        }
    }

    // Within-tier consistency: all permutations in the same tier must be bit-exact
    eprintln!("\n--- Within-tier consistency ---");
    for (tier, entries) in &tier_results {
        if entries.len() < 2 {
            continue;
        }
        let (ref_label, ref_r) = &entries[0];
        let mut max_score_ulp: u64 = 0;
        let mut max_feat_ulp: u64 = 0;
        for (_label, r) in &entries[1..] {
            if let Some(u) = ulp_distance(ref_r.score, r.score) {
                max_score_ulp = max_score_ulp.max(u);
            }
            for (rf, tf) in ref_r.features.iter().zip(r.features.iter()) {
                if let Some(u) = ulp_distance(*rf, *tf) {
                    max_feat_ulp = max_feat_ulp.max(u);
                }
            }
        }
        eprintln!(
            "  {tier:14}: {n} permutations, max score ULP={max_score_ulp}, max feat ULP={max_feat_ulp}  (ref: {ref_label})",
            n = entries.len(),
        );
        assert_eq!(
            max_score_ulp, 0,
            "Within-tier {tier}: score diverges by {max_score_ulp} ULP"
        );
        assert_eq!(
            max_feat_ulp, 0,
            "Within-tier {tier}: features diverge by {max_feat_ulp} ULP"
        );
    }

    // Cross-tier divergence: compare each tier against the highest tier
    eprintln!("\n--- Cross-tier divergence ---");
    let tier_order = ["AVX-512 (v4)", "AVX2 (v3)", "scalar/SSE2"];
    let mut tier_scores: Vec<(&str, f64, f64, &Vec<f64>)> = Vec::new();
    for tier in &tier_order {
        if let Some(entries) = tier_results.get(tier) {
            let (_, r) = &entries[0];
            tier_scores.push((tier, r.score, r.raw_distance, &r.features));
        }
    }

    let feature_names = [
        "ssim_mean",
        "ssim_4th",
        "edge_art_mean",
        "edge_art_4th",
        "edge_det_mean",
        "edge_det_4th",
        "mse",
        "var_loss",
        "tex_loss",
    ];

    if tier_scores.len() >= 2 {
        let (ref_tier, ref_score, ref_dist, ref_feats) = &tier_scores[0];
        for (tier, score, dist, feats) in &tier_scores[1..] {
            let score_ulp = ulp_distance(*ref_score, *score).unwrap_or(u64::MAX);
            let dist_ulp = ulp_distance(*ref_dist, *dist).unwrap_or(u64::MAX);
            let score_rel = (ref_score - score).abs() / ref_score.abs().max(1e-12);
            let dist_rel = (ref_dist - dist).abs() / ref_dist.abs().max(1e-12);

            eprintln!("\n  {ref_tier} vs {tier}:");
            eprintln!(
                "    score:    {ref_score:.10} vs {score:.10}  (ULP={score_ulp}, rel={score_rel:.2e})"
            );
            eprintln!(
                "    raw_dist: {ref_dist:.12} vs {dist:.12}  (ULP={dist_ulp}, rel={dist_rel:.2e})"
            );

            // Show features with largest divergence
            let mut feat_diffs: Vec<(usize, u64, f64, f64)> = Vec::new();
            for (i, (rf, tf)) in ref_feats.iter().zip(feats.iter()).enumerate() {
                if let Some(u) = ulp_distance(*rf, *tf) {
                    if u > 0 {
                        feat_diffs.push((i, u, *rf, *tf));
                    }
                }
            }
            feat_diffs.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!(
                "    features diverging: {}/{}",
                feat_diffs.len(),
                ref_feats.len()
            );
            eprintln!("    top 10 by ULP distance:");
            for &(i, u, rf, tf) in feat_diffs.iter().take(10) {
                let scale = i / 27;
                let within = i % 27;
                let ch = within / 9;
                let fi = within % 9;
                let rel = (rf - tf).abs() / rf.abs().max(1e-12);
                eprintln!(
                    "      feat[{i:3}] s{scale} c{ch} {:14} = {u:>16} ULP  rel={rel:.2e}  ({rf:.8e} vs {tf:.8e})",
                    feature_names[fi],
                );
            }
        }
    }

    // Final verdict
    eprintln!("\n--- Verdict ---");
    let tiers: Vec<&str> = tier_scores.iter().map(|(t, _, _, _)| *t).collect();
    if tier_scores.len() >= 2 {
        let (_, ref_score, _, _) = &tier_scores[0];
        for (tier, score, _, _) in &tier_scores[1..] {
            let rel = (ref_score - score).abs() / ref_score.abs().max(1e-12);
            let ulp = ulp_distance(*ref_score, *score).unwrap_or(u64::MAX);
            eprintln!(
                "  {} vs {}: score rel={:.2e}, ULP={}",
                tiers[0], tier, rel, ulp
            );
        }
    }
}

/// Same as above but for 512x512 to exercise more SIMD iterations.
#[test]
fn score_reproducibility_512x512() {
    let w = 512;
    let h = 512;
    let (src, dst) = generate_test_images(w, h);

    let mut results: Vec<(String, Vec<String>, ZensimResult)> = Vec::new();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = compute_zensim(&src, &dst, w, h).expect("compute_zensim failed");
        let disabled: Vec<String> = perm.disabled.iter().map(|s| s.to_string()).collect();
        results.push((perm.label.clone(), disabled, result));
    });

    assert!(report.permutations_run >= 2);

    // Group by tier
    let mut tier_results: std::collections::BTreeMap<&'static str, Vec<&ZensimResult>> =
        std::collections::BTreeMap::new();
    for (label, disabled, result) in &results {
        let disabled_strs: Vec<&str> = disabled.iter().map(|s| s.as_str()).collect();
        let tier = classify_tier(label, &disabled_strs);
        tier_results.entry(tier).or_default().push(result);
    }

    eprintln!("\n=== 512x512 cross-tier test ===");
    let tier_order = ["AVX-512 (v4)", "AVX2 (v3)", "scalar/SSE2"];
    for tier in &tier_order {
        if let Some(entries) = tier_results.get(tier) {
            eprintln!(
                "  {tier:14}: score={:.10}  ({} permutations)",
                entries[0].score,
                entries.len(),
            );
        }
    }

    // Within-tier: must be bit-exact
    for (tier, entries) in &tier_results {
        if entries.len() < 2 {
            continue;
        }
        for r in &entries[1..] {
            let u = ulp_distance(entries[0].score, r.score).unwrap_or(u64::MAX);
            assert_eq!(u, 0, "Within-tier {tier}: score diverges by {u} ULP");
        }
    }

    // Cross-tier: report
    if let (Some(v4), Some(v3)) = (
        tier_results.get("AVX-512 (v4)"),
        tier_results.get("AVX2 (v3)"),
    ) {
        let u = ulp_distance(v4[0].score, v3[0].score).unwrap_or(u64::MAX);
        let rel = (v4[0].score - v3[0].score).abs() / v4[0].score.abs().max(1e-12);
        eprintln!("  v4 vs v3: score ULP={u}, rel={rel:.2e}");
    }
    if let (Some(v4), Some(sc)) = (
        tier_results.get("AVX-512 (v4)"),
        tier_results.get("scalar/SSE2"),
    ) {
        let u = ulp_distance(v4[0].score, sc[0].score).unwrap_or(u64::MAX);
        let rel = (v4[0].score - sc[0].score).abs() / v4[0].score.abs().max(1e-12);
        eprintln!("  v4 vs scalar: score ULP={u}, rel={rel:.2e}");
    }
}
