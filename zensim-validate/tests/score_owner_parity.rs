//! **The gate that binds `zensim-validate`'s bake-evaluation scorer to
//! `zensim`'s own score arithmetic** (owner consolidation, 2026-09-06).
//!
//! # What went wrong, measured
//!
//! `bake_runtime.rs` and its `bake_compare.rs` fork carried their own copies of
//! the two mixing heads, the tanh output pin, the distance→score mapping and
//! the PCHIP output spline, documented as bit-exact with `zensim::metric`
//! **in prose, with nothing holding them there**. F19 (`zensim::det_math`)
//! then routed every transcendental on the product's score path through a
//! selectable [`PowForm`] so a score stops being a function of which libm the
//! binary linked against — and the mirror did not follow. `det_math`'s own
//! exposure table called that fork *"a BLOCKER on flipping
//! `SHIPPED_REVISION`"*.
//!
//! MEASURED 2026-09-06 BEFORE the fix, six shipped/board bakes ×
//! `cid22,kadid,tid,konjnd,aic3`: `bake_verdict --full-json` was
//! **byte-identical under `ZENSIM_POW_FORM=libm` and `=pure` on all six**.
//! The evaluation tooling was completely insensitive to the form the product
//! runtime obeys — the fork, observed rather than argued.
//!
//! # What this file gates
//!
//! 1. [`validate_scorer_follows_the_pow_form`] — the load-bearing one. It
//!    scores 10,000 rows through a REAL shipped bake (Profile A, whose
//!    `zentrain.per_sample_alpha_head` is the head this consolidation moved),
//!    digests every score's `to_bits()`, then **re-execs this same test binary
//!    with `ZENSIM_POW_FORM=pure`** and requires the digest to CHANGE.
//!    `active_pow_form()` is a process-wide `OnceLock`, so the subprocess is
//!    not a convenience — it is the only way to see two arms at all. A
//!    re-forked scorer calling `f64::powf` directly would reproduce the
//!    `libm` digest under both arms and fail here.
//! 2. [`post_dispatch_adapter_is_bit_identical_to_the_owner`] — the shape
//!    adapter in `bake_runtime` must add no arithmetic of its own: on 10,000
//!    real forward-pass hidden vectors, `score_from_network_output` must equal
//!    a hand-composed `zensim::score_math` head + pin + spline at the ACTIVE
//!    form, by `to_bits()`.
//! 3. [`spline_adapter_is_bit_identical_to_the_owner`] — same, for the PCHIP
//!    evaluator, over 10,001 points spanning both extrapolation tails and the
//!    whole interior.
//! 4. [`the_pchip_interior_is_capped_like_the_product_runtime`] — the second
//!    divergence this consolidation found and closed. The validate-side
//!    evaluator capped the upper EXTRAPOLATION (the 2026-07-04 audit) and left
//!    the INTERIOR uncapped, so a spline knot above 100 — which the wire
//!    format permits — would publish a score `Zensim::compute` reports as
//!    exactly 100. LATENT: 0 of the 49 bakes on disk declare such a knot.
//!
//! Nothing here needs a mounted corpus: the bake is `include_bytes!` and the
//! rows are generated from the bake's OWN scaler statistics, so they land in
//! the distribution its weights were fit on. There is no runtime skip.

use zenpredict::{Model, Predictor};
use zensim::det_math::{PowForm, active_pow_form};
use zensim::score_math;
use zensim_validate::bake_runtime::{self, CallerGather};
use zensim_validate::output_calibration_spline as ocs;

/// Profile A — the shipped bake that carries `zentrain.per_sample_alpha_head`,
/// i.e. the one whose score actually reaches the arithmetic under test.
const BAKE_A: &[u8] =
    include_bytes!("../../zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin");

/// Rows scored by every digest/parity assertion below.
const N_ROWS: usize = 10_000;

/// Env var the parent sets on the re-exec'd child; its value is the path the
/// child writes its digest to.
const CHILD_OUT: &str = "ZENSIM_SCORE_OWNER_PARITY_CHILD_OUT";

/// FNV-1a over every score's raw bits. A digest rather than a per-row compare
/// so the child can hand back one line.
fn digest(scores: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for s in scores {
        for b in s.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x1000_0000_01b3);
        }
    }
    h
}

/// `N_ROWS` deterministic feature rows in the bake's own caller layout.
///
/// Values are drawn around each input's `scaler_mean` at ±3 `scaler_scale`,
/// so the forward pass sees the range its weights were standardized for
/// instead of arbitrary numbers that would saturate the transforms.
fn rows(model: &Model, n: usize) -> Vec<Vec<f64>> {
    let width = model.caller_input_width();
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let inner = model.n_inputs();
    let mut s = 0x2026_0906_u64 | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| {
            (0..width)
                .map(|i| {
                    // The caller row is identity-laid-out and can be wider than
                    // the packed layer-0; index the stats defensively.
                    let j = i.min(inner.saturating_sub(1));
                    let m = *mean.get(j).unwrap_or(&0.0) as f64;
                    let sd = *scale.get(j).unwrap_or(&1.0) as f64;
                    m + (next() * 6.0 - 3.0) * if sd == 0.0 { 1.0 } else { sd }
                })
                .collect()
        })
        .collect()
}

/// Score `rows` through the validate runtime exactly as `bake_verdict` does.
fn score_all(model: &Model, feature_rows: &[Vec<f64>]) -> Vec<f64> {
    let psa = bake_runtime::extract_per_sample_alpha_head(model);
    let hyb = bake_runtime::extract_hybrid_head(model);
    let pin = bake_runtime::extract_tanh_output_head_scale(model);
    let spline = ocs::extract(model);
    let gather = CallerGather::for_model(model);
    let has_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = Predictor::new(model);
    let mut scratch = vec![0.0f32; model.caller_input_width()];
    feature_rows
        .iter()
        .map(|row| {
            bake_runtime::score_row(
                &mut predictor,
                has_transforms,
                psa.as_ref(),
                hyb.as_ref(),
                pin,
                spline.as_ref(),
                &gather,
                &mut scratch[..],
                row,
            )
        })
        .collect()
}

/// ★ The load-bearing gate: the validate scorer MOVES when the product's
/// `PowForm` moves.
///
/// A re-fork that calls `f64::powf`/`f64::exp` directly is form-invariant and
/// fails here — which is exactly the state this file was written to end.
#[test]
fn validate_scorer_follows_the_pow_form() {
    let model = Model::from_bytes(BAKE_A).expect("parse shipped Profile A bake");
    assert!(
        bake_runtime::extract_per_sample_alpha_head(&model).is_some(),
        "fixture no longer carries a per-sample-α head — this gate would be vacuous"
    );
    let feature_rows = rows(&model, N_ROWS);
    let scores = score_all(&model, &feature_rows);
    assert!(
        scores.iter().all(|s| s.is_finite()),
        "fixture rows produced a non-finite score; the generator is out of distribution"
    );
    let mine = digest(&scores);

    // Child half: write the digest and stop. Reached only via the re-exec
    // below, so there is no runtime skip in the normal path.
    if let Ok(out) = std::env::var(CHILD_OUT) {
        std::fs::write(&out, format!("{mine:016x}")).expect("child: write digest");
        return;
    }

    let out = std::env::temp_dir().join(format!(
        "zensim_score_owner_parity_{}_{}.digest",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos())
    ));
    let exe = std::env::current_exe().expect("current_exe");
    let status = std::process::Command::new(exe)
        .args([
            "--exact",
            "validate_scorer_follows_the_pow_form",
            "--nocapture",
        ])
        // The two accepted values are the same byte length on purpose —
        // `det_math` records an environment block's size shifting this
        // binary's layout by ~10 % at 2304².
        .env("ZENSIM_POW_FORM", "pure")
        .env(CHILD_OUT, &out)
        .status()
        .expect("re-exec the test binary under ZENSIM_POW_FORM=pure");
    assert!(status.success(), "child run failed: {status:?}");
    let theirs_hex = std::fs::read_to_string(&out).expect("child digest file");
    let _ = std::fs::remove_file(&out);
    let theirs = u64::from_str_radix(theirs_hex.trim(), 16).expect("child digest parse");

    assert_ne!(
        mine, theirs,
        "the validate scorer is INSENSITIVE to ZENSIM_POW_FORM over {N_ROWS} rows \
         (digest {mine:016x} under the default arm and {theirs:016x} under `pure`). \
         That is the pre-2026-09-06 fork: the score path is not reaching \
         `zensim::score_math`, so `bake_verdict` describes arithmetic the \
         product runtime does not run."
    );
}

/// The `bake_runtime` post-network dispatch is a SHAPE adapter and adds no
/// arithmetic of its own: bit-identical to `zensim::score_math` on every one
/// of `N_ROWS` real forward outputs, head + pin + spline.
///
/// # A MEASURED fact about this fixture, stated so nobody re-derives it
///
/// Profile A's per-sample-α head alone is **form-INVARIANT on all 10,000
/// rows** — the two `PowForm` arms agree bit-for-bit — even though
/// `|h|^6` disagrees on ~9.8 % of random doubles and `x^(1/6)` on ~14 %.
/// The reason is that A's hidden vector reaches ±2.6e4, so `alpha_logit`
/// saturates the ±20 clamp and `α` is 1.0 to f64 resolution; the entire
/// `y_pool` term — the only place the p-norm enters — is multiplied by
/// `(1 − α) ≈ 2e-9` and annihilated. What DOES move under the form on this
/// bake is the tanh pin's `exp`, which is why
/// [`validate_scorer_follows_the_pow_form`] digests the whole scored value
/// rather than the head. Both facts are asserted below so a fixture change
/// that silently removes the form sensitivity fails here.
#[test]
fn post_dispatch_adapter_is_bit_identical_to_the_owner() {
    let model = Model::from_bytes(BAKE_A).expect("parse shipped Profile A bake");
    let psa = bake_runtime::extract_per_sample_alpha_head(&model)
        .expect("fixture carries a per-sample-α head");
    let pin = bake_runtime::extract_tanh_output_head_scale(&model)
        .expect("fixture carries a tanh output pin");
    let spline = ocs::extract(&model).expect("fixture carries an output spline");
    let (w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm) = &psa;
    let params = score_math::PerSampleAlphaParams {
        w_alpha,
        b_alpha: *b_alpha,
        rank_w,
        rank_b: *rank_b,
        reducer_w: *reducer_w,
        reducer_b: *reducer_b,
        p_norm: *p_norm,
    };
    let feature_rows = rows(&model, N_ROWS);
    let gather = CallerGather::for_model(&model);
    let has_transforms = model.has_nontrivial_feature_transforms();
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; model.caller_input_width()];
    let form = active_pow_form();
    let other = match form {
        PowForm::LibmPowf => PowForm::PureRust,
        PowForm::PureRust => PowForm::LibmPowf,
    };
    let mut composed_arms_differ = 0usize;
    for row in &feature_rows {
        gather.fill(&mut scratch[..], row);
        let out = if has_transforms {
            predictor.predict_transformed(&scratch)
        } else {
            predictor.predict(&scratch)
        }
        .expect("forward");

        // Hand-compose the owner's pipeline, then require the adapter to
        // reproduce it bit-for-bit.
        let owner_head = score_math::per_sample_alpha_head(out, &params, form);
        let owner_pin = score_math::tanh_output_pin(owner_head, pin, form);
        let owner_full =
            score_math::pchip_eval_capped(owner_pin, &spline.xs, &spline.ys, &spline.derivs);
        let adapter = bake_runtime::score_from_network_output(
            out,
            Some(&psa),
            None,
            Some(pin),
            Some(&spline),
        );
        assert_eq!(
            adapter.to_bits(),
            owner_full.to_bits(),
            "bake_runtime added arithmetic of its own: {adapter} vs owner {owner_full}"
        );

        // …and the composed value must be form-SENSITIVE, or the subprocess
        // digest gate above would be measuring nothing on this fixture.
        let alt_head = score_math::per_sample_alpha_head(out, &params, other);
        let alt_pin = score_math::tanh_output_pin(alt_head, pin, other);
        let alt_full =
            score_math::pchip_eval_capped(alt_pin, &spline.xs, &spline.ys, &spline.derivs);
        if owner_full.to_bits() != alt_full.to_bits() {
            composed_arms_differ += 1;
        }
    }
    assert!(
        composed_arms_differ > 0,
        "the two PowForm arms agree on all {N_ROWS} scored rows — \
         `validate_scorer_follows_the_pow_form` would be vacuous on this fixture"
    );
}

/// The validate-side spline entry is a wire-format wrapper and adds no math:
/// bit-identical to the owner across both tails and the whole interior.
#[test]
fn spline_adapter_is_bit_identical_to_the_owner() {
    // A realistic dial spline: 8 knots, monotone, topping out at 100.
    let knots: [(f32, f32); 8] = [
        (4.5, 0.0),
        (7.2, 10.0),
        (15.3, 30.0),
        (28.1, 50.0),
        (38.7, 60.0),
        (55.0, 80.0),
        (68.4, 90.0),
        (88.9, 100.0),
    ];
    let mut payload = Vec::new();
    payload.extend_from_slice(&(knots.len() as u32).to_le_bytes());
    for (x, y) in &knots {
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
    }
    let spline = ocs::parse_payload(&payload).expect("parse dial spline");
    // Owner-derived knots must match the ones the parser stored.
    let owner_derivs = score_math::pchip_derivs(&spline.xs, &spline.ys);
    for (a, b) in spline.derivs.iter().zip(owner_derivs.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "deriv solve forked: {a} vs {b}");
    }
    // −20 .. +120 in 0.014 steps: both extrapolation tails plus every segment.
    for i in 0..=10_000 {
        let x = -20.0 + i as f64 * 0.014;
        let adapter = ocs::apply(x, &spline);
        let owner = score_math::pchip_eval_capped(x, &spline.xs, &spline.ys, &spline.derivs);
        assert_eq!(
            adapter.to_bits(),
            owner.to_bits(),
            "spline forked at x={x}: {adapter} vs owner {owner}"
        );
    }
}

/// The interior cap — the divergence this consolidation found and closed.
///
/// The validate-side evaluator capped its upper *extrapolation* at 100 (the
/// 2026-07-04 audit) but not the *interior* segment, while the product runtime
/// caps both. The reachable trigger is **a knot whose `y` exceeds 100**, which
/// the wire format permits and `parse_payload` does not reject. It is NOT
/// Hermite overshoot — the Fritsch-Carlson rule forbids that, and a first
/// draft of this test built an "overshoot" fixture and failed its own vacuity
/// guard proving it.
///
/// MEASURED over all 49 bakes on disk (`zensim/weights`, its `archive/`, and
/// `zensim-experimental/weights`): 0 declare a knot above 100, so the
/// divergence was latent and no published verdict moved. It is closed by
/// construction now; this test keeps it closed.
#[test]
fn the_pchip_interior_is_capped_like_the_product_runtime() {
    let knots: [(f32, f32); 3] = [(0.0, 0.0), (50.0, 80.0), (100.0, 130.0)];
    let mut payload = Vec::new();
    payload.extend_from_slice(&(knots.len() as u32).to_le_bytes());
    for (x, y) in &knots {
        payload.extend_from_slice(&x.to_le_bytes());
        payload.extend_from_slice(&y.to_le_bytes());
    }
    let spline = ocs::parse_payload(&payload).expect("the wire format accepts a knot above 100");
    let (xs, ys, d) = (&spline.xs, &spline.ys, &spline.derivs);

    // The OLD validate-side interior branch, verbatim and uncapped.
    let uncapped = |x: f64| {
        let (lo, hi) = (1usize, 2usize);
        let h = xs[hi] - xs[lo];
        let t = (x - xs[lo]) / h;
        let h00 = (1.0 + 2.0 * t) * (1.0 - t).powi(2);
        let h10 = t * (1.0 - t).powi(2);
        let h01 = t.powi(2) * (3.0 - 2.0 * t);
        let h11 = t.powi(2) * (t - 1.0);
        h00 * ys[lo] + h10 * h * d[lo] + h01 * ys[hi] + h11 * h * d[hi]
    };
    let worst = (1..1000)
        .map(|i| uncapped(50.0 + i as f64 * 0.05))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        worst > 100.0,
        "fixture no longer exceeds 100 in the interior (max {worst}); this gate would be vacuous"
    );
    for i in 1..1000 {
        let x = 50.0 + i as f64 * 0.05;
        let y = ocs::apply(x, &spline);
        assert!(
            y <= 100.0,
            "validate spline interior above 100 at x={x}: {y}"
        );
    }
}
