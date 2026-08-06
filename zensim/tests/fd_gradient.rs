//! G-Y1 (campaign appendix Y, L-Y1): the batched FD-gradient entry
//! `score_features_fd_gradient_with_profile` is BITWISE-equal to the
//! sequential per-probe recipe (2·N calls of `score_features_with_profile`
//! with the jxl iteration-0 probe rule). The batched entry parses the bake
//! once and reuses one `Predictor`; every forward's arithmetic is the
//! shared canonical path, so equality is exact, not tolerance-class.

use zensim::ZensimProfile;

/// The sequential recipe, verbatim from the jxl loop's iteration-0 probe
/// (jxl-encoder `zensim_loop.rs`): eps = max(|b|·1e-3, 1e-5), central
/// difference, non-finite forward → component 0.0.
fn sequential_grad(profile: ZensimProfile, base: &[f64], w: u32, h: u32) -> Vec<f64> {
    let sf = |f: &[f64]| zensim::score_features_with_profile(profile, f, w, h).unwrap_or(f64::NAN);
    let mut grad = vec![0.0f64; base.len()];
    let mut probe = base.to_vec();
    for (k, gk) in grad.iter_mut().enumerate() {
        let eps = (base[k].abs() * 1e-3).max(1e-5);
        probe[k] = base[k] + eps;
        let up = sf(&probe);
        probe[k] = base[k] - eps;
        let dn = sf(&probe);
        probe[k] = base[k];
        *gk = if up.is_finite() && dn.is_finite() {
            (up - dn) / (2.0 * eps)
        } else {
            0.0
        };
    }
    grad
}

/// Deterministic, sign-varied base vector in a plausible feature range.
fn synthetic_base(n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            let m = ((k * 37 + 11) % 101) as f64 / 101.0; // [0, 1)
            let sign = if k % 5 == 3 { -1.0 } else { 1.0 };
            sign * m * (0.02 + 0.4 * ((k % 7) as f64 / 7.0))
        })
        .collect()
}

#[test]
fn fd_gradient_bitwise_matches_sequential() {
    // Latest profile = the shipped MLP-scored ship profile (single bake,
    // fast path). Caller width probed the same way the jxl mount does.
    let profile = ZensimProfile::B;
    let mut width_found = None;
    for n in [156usize, 228, 300, 372, 720, 924, 944] {
        let feats = vec![0.1f64; n];
        if zensim::score_features_with_profile(profile, &feats, 576, 576).is_ok() {
            width_found = Some(n);
            break;
        }
    }
    let n = width_found.expect("default profile accepts no probed feature width");
    let base = synthetic_base(n);

    let batched = zensim::score_features_fd_gradient_with_profile(profile, &base, 576, 576)
        .expect("batched FD gradient");
    let sequential = sequential_grad(profile, &base, 576, 576);

    assert_eq!(batched.len(), sequential.len());
    let mut n_nonzero = 0usize;
    for k in 0..base.len() {
        assert_eq!(
            batched[k].to_bits(),
            sequential[k].to_bits(),
            "gradient component {k} diverged: batched {} vs sequential {}",
            batched[k],
            sequential[k]
        );
        if batched[k] != 0.0 {
            n_nonzero += 1;
        }
    }
    // Sanity: the probe found real sensitivity (guards against a silent
    // all-NaN/all-zero degenerate pass).
    assert!(
        n_nonzero > 0,
        "gradient is identically zero — probe never engaged"
    );
}

#[test]
fn fd_gradient_prefix_width_matches_sequential() {
    // A base WIDER than the bake's caller width exercises the prefix
    // branch + the exact-zero tail shortcut. Bitwise equality must hold
    // (sequential computes 0.0 there via up == dn; batched skips).
    let profile = ZensimProfile::B;
    let mut width_found = None;
    for n in [156usize, 228, 300, 372, 720, 924, 944] {
        let feats = vec![0.1f64; n];
        if zensim::score_features_with_profile(profile, &feats, 576, 576).is_ok() {
            width_found = Some(n);
            break;
        }
    }
    let n = width_found.expect("default profile accepts no probed feature width");
    let base = synthetic_base(n + 16);

    let batched = zensim::score_features_fd_gradient_with_profile(profile, &base, 576, 576)
        .expect("batched FD gradient");
    let sequential = sequential_grad(profile, &base, 576, 576);
    for k in 0..base.len() {
        assert_eq!(
            batched[k].to_bits(),
            sequential[k].to_bits(),
            "gradient component {k} diverged (prefix case)"
        );
    }
    for (k, g) in batched.iter().enumerate().skip(n) {
        assert_eq!(*g, 0.0, "tail component {k} must be exactly zero");
    }
}
