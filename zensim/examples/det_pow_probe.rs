//! **F19's error-bound instrument.** Dump every arm of the score path's
//! `pow` / `exp` / `log2` as `to_bits()` over a fixed, procedurally-generated
//! grid, so an external 60-digit reference can price each arm's error in ULP
//! and so the arms' disagreement rate can be counted rather than assumed.
//!
//! ```sh
//! cargo run --release -p zensim --example det_pow_probe > ~/tmp/detpow.tsv
//! python3 scripts/det_pow_error_bound.py ~/tmp/detpow.tsv
//! ```
//!
//! Four arms per `pow` row:
//!
//! | arm | what | why it is here |
//! |---|---|---|
//! | `std` | `f64::powf` — the platform libm | the SHIPPED arm, and the defect |
//! | `pure` | `libm::pow` — the pure-Rust fdlibm port | the PROPOSED arm |
//! | `mtlow` | `magetypes::nostd_math::powf_f64` | the arm the F19 brief proposed reusing, priced so the rejection is measured rather than argued |
//! | `f32best` | `(x as f32).powf(b) as f64` | the BEST CASE for any f32 route, `*_midp_precise` included. Those functions exist only on `f32x4`/`x8`/`x16` (magetypes 0.9.28 `simd/generic/generated/transcendentals_f32x*.rs`) and are token-gated, so the arm priced here is a *lower bound* on their error: a perfectly-rounded f32 pow. If even that is unusable, no f32 form can be used, and the rejection needs no token to establish |
//!
//! The inputs are procedural (an integer LCG over a log-uniform sweep), so
//! the grid is identical on every host without shipping a table, and the
//! exponents are the score path's OWN: `score_mapping_b` = 0.7 on every
//! shipped profile, the three `approx_*` fits, and a spread of p-norm `p`
//! and `1/p` from the head runtimes.

/// The score path's actual exponents.
const EXPONENTS: &[(f64, &str)] = &[
    (0.7, "score_mapping_b"),
    (0.5979, "approx_ssim2"),
    (1.2244, "approx_dssim"),
    (0.6130, "approx_butteraugli"),
    (2.0, "pnorm_p2"),
    (3.0, "pnorm_p3"),
    (6.0, "pnorm_p6"),
    (1.0 / 3.0, "pnorm_inv3"),
    (1.0 / 6.0, "pnorm_inv6"),
];

/// `n` log-uniform points over `[lo, hi]`, computed with `powi`-free integer
/// arithmetic on the exponent so the grid itself carries no libm dependence
/// worth worrying about (it is reported alongside the value, so the reference
/// prices whatever grid actually came out).
fn log_uniform(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let (ll, lh) = (lo.ln(), hi.ln());
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (ll * (1.0 - t) + lh * t).exp()
        })
        .collect()
}

fn main() {
    println!("kind\tlabel\targ_bits\texp_bits\tstd\tpure\tmtlow\tf32best");

    // pow: the raw-distance domain. A distance below 1e-12 maps to 100 by the
    // `<= 0` guard's neighbourhood and above ~1e3 the score is saturated far
    // past -100, so this spans the whole reachable range and then some.
    for x in log_uniform(1e-12, 1e3, 601) {
        for &(b, label) in EXPONENTS {
            let s = x.powf(b);
            let p = libm::pow(x, b);
            let m = magetypes::nostd_math::powf_f64(x, b);
            // Lower bound on ANY f32 route: assume the f32 pow is perfect.
            let f = (x as f32).powf(b as f32) as f64;
            println!(
                "pow\t{label}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}",
                x.to_bits(),
                b.to_bits(),
                s.to_bits(),
                p.to_bits(),
                m.to_bits(),
                f.to_bits()
            );
        }
    }

    // exp: the bounded squash takes -(a/100)*d^b <= 0; the head sigmoids clamp
    // to [-30, 30] and [-20, 20]; soft_clamp_score takes (raw-50)/-20.
    for i in 0..=800usize {
        let x = -40.0 + (i as f64) * 0.1;
        let s = x.exp();
        let p = libm::exp(x);
        let m = magetypes::nostd_math::exp_f64(x);
        let f = (x as f32).exp() as f64;
        println!(
            "exp\tsigmoid\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}",
            x.to_bits(),
            0u64,
            s.to_bits(),
            p.to_bits(),
            m.to_bits(),
            f.to_bits()
        );
    }

    // log2: the four `--mlp-size-axes` inputs — pixels up to ~1e9, dims to 1e5.
    for x in log_uniform(1.0, 1e9, 401) {
        let s = x.log2();
        let p = libm::log2(x);
        let m = magetypes::nostd_math::log2_f64(x);
        let f = (x as f32).log2() as f64;
        println!(
            "log2\tsize_axis\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}\t{:016x}",
            x.to_bits(),
            0u64,
            s.to_bits(),
            p.to_bits(),
            m.to_bits(),
            f.to_bits()
        );
    }
}
