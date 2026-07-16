//! `zensim-train-core`'s private stats MUST stay bit-identical to `zenstats`.
//!
//! **Why the duplication exists.** `zensim-train-core` is a deliberately
//! standalone, WASM-targeted, bit-exact trainer core (`publish = false`, "no
//! rayon-by-default, no std::fs" — see its Cargo.toml). `zenstats` is not
//! WASM-vetted, so the core carries its own ~50-line `pearson`/`ranks`/
//! `spearman` rather than take a cross-repo dependency. That is a defensible
//! boundary, not sloppiness.
//!
//! **Why this test exists.** `zensim-train-core/src/stats.rs` already declared
//! the rule — *"both impls must be kept in lock-step"* — and claimed it was
//! *"verified by `tests/test_zen_stats_rust_python_parity.py` in
//! `scripts/canonical_corpus/` once shipped"*. That file never shipped. So as
//! of 2026-07-15 the duplication was documented, declared safe, and the safety
//! had never once been checked. A rule with no enforcement is not a rule.
//!
//! **Why bit-identity and not approximate equality.** The trainer calls
//! `spearman` for per-epoch validation SROCC, which selects the best-epoch
//! checkpoint. A single ULP can flip which epoch wins and change the bytes of
//! every bake we ship. So "both compute Spearman" is not good enough; the
//! assertion is `to_bits()` equality.
//!
//! **The implementations are NOT textually the same**, which is exactly why
//! this needs measuring rather than reading:
//!   - `ranks()`: train-core sorts with `partial_cmp().unwrap_or(Equal)`,
//!     zenstats with `total_cmp()` — these differ on NaN.
//!   - `spearman()`: train-core computes `pearson(ranks(x), ranks(y))`, whose
//!     mean is `sum(ranks)/n`; zenstats uses the closed form `(n-1)/2`.
//!
//! They agree because a rank vector's sum is exactly `n(n-1)/2` and mid-rank
//! ties keep it a multiple of 0.5 — all exactly representable in f64 far below
//! 2^53 at our corpus sizes (largest canonical group: safesyn, 196,086 rows).
//! So the division is exact and the means coincide bit-for-bit. That argument
//! holds up to n ≈ 1.3e8; if a corpus ever approaches that, this test is where
//! it will surface.

use zenstats::panel as zs;

/// Deterministic LCG — no rand dep, and reproducible across platforms.
fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Sizes spanning every canonical corpus: aic3 (600), konjnd (1008), cid22
/// (4292), non-photo (10000), and safesyn (196,086) — the largest group any
/// recipe trains on.
const SIZES: &[usize] = &[2, 3, 5, 17, 50, 300, 600, 1008, 4292, 10000, 196_086];

#[test]
fn spearman_is_bit_identical_to_zenstats() {
    let mut seed = 12345u64;
    for &n in SIZES {
        for trial in 0..12 {
            let a: Vec<f64> = (0..n).map(|_| lcg(&mut seed)).collect();
            // Every third trial quantizes hard to force heavy ties — the case
            // where mid-ranks turn fractional and the two mean formulas could
            // plausibly part company.
            let b: Vec<f64> = (0..n)
                .map(|_| {
                    let v = lcg(&mut seed);
                    if trial % 3 == 0 { (v * 5.0).floor() } else { v }
                })
                .collect();

            let core = zensim_train_core::spearman(&a, &b);
            let canon = zs::spearman(&a, &b);
            assert_eq!(
                core.to_bits(),
                canon.to_bits(),
                "train-core and zenstats spearman diverged at n={n}, trial={trial}: \
                 {core} vs {canon} (delta {:e}). These must stay bit-identical — the \
                 trainer picks its best-epoch checkpoint on this value, so a ULP here \
                 changes every shipped bake's bytes. Fix the divergence; do NOT relax \
                 this to an approximate compare.",
                (core - canon).abs()
            );
        }
    }
}

#[test]
fn pearson_is_bit_identical_to_zenstats() {
    let mut seed = 999u64;
    for &n in SIZES {
        for _ in 0..12 {
            let a: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 100.0).collect();
            let b: Vec<f64> = (0..n).map(|_| lcg(&mut seed) * 100.0).collect();
            let core = zensim_train_core::pearson(&a, &b);
            let canon = zs::pearson(&a, &b);
            assert_eq!(
                core.to_bits(),
                canon.to_bits(),
                "train-core and zenstats pearson diverged at n={n}: {core} vs {canon}"
            );
        }
    }
}

#[test]
fn ranks_are_bit_identical_to_zenstats() {
    let mut seed = 4242u64;
    for &n in &[2usize, 3, 17, 300, 4292] {
        for trial in 0..12 {
            let v: Vec<f64> = (0..n)
                .map(|_| {
                    let x = lcg(&mut seed);
                    // ties on every other trial
                    if trial % 2 == 0 { (x * 4.0).floor() } else { x }
                })
                .collect();
            let core = zensim_train_core::ranks(&v);
            let canon = zs::ranks(&v);
            assert_eq!(core.len(), canon.len());
            for (i, (c, z)) in core.iter().zip(&canon).enumerate() {
                assert_eq!(
                    c.to_bits(),
                    z.to_bits(),
                    "train-core and zenstats ranks diverged at n={n}, trial={trial}, i={i}: \
                     {c} vs {z}"
                );
            }
        }
    }
}

/// Zero-variance input: both must return 0.0 rather than NaN from a 0/0. The
/// guards are written differently (`den < 1e-12` in both, but reached via
/// different code), so pin the shared contract.
#[test]
fn zero_variance_returns_zero_in_both() {
    let flat = vec![1.0f64; 64];
    let varied: Vec<f64> = (0..64).map(|i| i as f64).collect();
    for (a, b) in [(&flat, &varied), (&varied, &flat), (&flat, &flat)] {
        assert_eq!(
            zensim_train_core::pearson(a, b).to_bits(),
            zs::pearson(a, b).to_bits()
        );
        assert_eq!(
            zensim_train_core::spearman(a, b).to_bits(),
            zs::spearman(a, b).to_bits()
        );
    }
}
