//! Cross-vendor / cross-tier probe for the era-2 `reduce_add` hypothesis
//! (`benchmarks/era2_perf_break_2026-08-31.md` §15).
//!
//! Prints every dense accumulator slot, for era-1 and era-2, as raw f64 bits,
//! plus the SIMD tier the dispatcher actually chose on this host. Run the SAME
//! binary on two boxes and diff.
//!
//! era-1's horizontal reduction is `reduce_add()`, which resolves to a
//! per-backend order; era-2's is the written-out `era2_reduce8` tree. If era-1
//! diverges between boxes where era-2 does not, the hypothesis graduates from
//! plausible to measured ON THAT PAIR.
//!
//! `cargo run --release -p zensim --features oracle,training,custom-profiles,feature-regime-v2 --example era2_vendor_probe`
use zensim::feature_v2::{harness_active_tier, harness_dense_slots};

fn plane(w: usize, h: usize, salt: usize) -> Vec<f32> {
    (0..w * h)
        .map(|i| {
            let k = ((i + salt * 7919) * 2654435761usize) % 65521;
            (k as f32) * (1.0 / 65536.0) + 0.05
        })
        .collect()
}

fn main() {
    println!("# era-2 vendor/tier probe");
    println!("tier: {}", harness_active_tier());
    // Geometry classes that matter: tight, non-tight (the option-C class), and
    // one with a scalar tail (width % 8 != 0).
    for &(w, h) in &[(64usize, 64usize), (200, 150), (127, 93)] {
        let p: Vec<Vec<f32>> = (0..7).map(|s| plane(w, h, s)).collect();
        for era2 in [false, true] {
            let s = harness_dense_slots(
                &p[0], &p[1], &p[2], &p[3], &p[4], &p[5], &p[6], w, h, true, era2,
            );
            let era = if era2 { "era2" } else { "era1" };
            for (i, v) in s.iter().enumerate() {
                println!("{w}x{h} {era} slot{i:02} {:016x}", v.to_bits());
            }
        }
    }
}
