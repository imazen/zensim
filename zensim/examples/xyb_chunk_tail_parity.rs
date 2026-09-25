//! Diagnostic for the CLAUDE.md Known Bug "bulk sRGB→XYB gives different bits
//! for the same colour in the vector chunks and in the scalar remainder"
//! (2026-09-25).
//!
//! For every sRGB8 colour it converts nine copies with the conversion owner,
//! `srgb_to_positive_xyb_planar_into`. The first eight go through a vector
//! chunk and the ninth through the remainder. It prints, per SIMD tier this host
//! can dispatch to, how many colours give different bits, per XYB channel.
//! A consistent conversion prints 0.
//!
//! Tiers are walked with archmage's token permutations, so on x86_64 the scalar
//! tier is forced; on i686 (and other targets without runtime tokens) the walk is
//! the one native scalar dispatch. The scalar tier is fixed (zero-padded remainder,
//! user decision 2026-09-25) and the process exits 1 if it reports any difference.
//! The SIMD tiers' divergence stays open and only reports.
//!
//! ```sh
//! cargo run --release -p zensim --example xyb_chunk_tail_parity
//! ```
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::__bench_stages::srgb_to_positive_xyb_planar_into;

/// True when dispatch of the bulk XYB conversions lands on the scalar tier.
#[cfg(target_arch = "x86_64")]
fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::X64V3Token::summon().is_none()
}
#[cfg(target_arch = "aarch64")]
fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::NeonToken::summon().is_none()
}
#[cfg(target_arch = "wasm32")]
fn dispatches_scalar() -> bool {
    use archmage::SimdToken as _;
    archmage::Wasm128Token::summon().is_none()
}
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn dispatches_scalar() -> bool {
    true
}

fn main() {
    let mut scalar_differs = false;
    // The report only lists permutations and warnings; the per-tier lines below are the output.
    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let scalar = dispatches_scalar();
        let mut differing = [0u64; 3];
        let mut any_channel = 0u64;
        let mut max_ulp = 0u32;
        let (mut x, mut y, mut b) = ([0f32; 9], [0f32; 9], [0f32; 9]);
        for r in 0..=255u8 {
            for g in 0..=255u8 {
                for blue in 0..=255u8 {
                    let pixels = [[r, g, blue]; 9];
                    srgb_to_positive_xyb_planar_into(&pixels, &mut x, &mut y, &mut b);
                    let mut hit = false;
                    for (channel, plane) in [&x, &y, &b].into_iter().enumerate() {
                        // Positive XYB, so the bit distance is the ULP distance.
                        let ulp = plane[0].to_bits().abs_diff(plane[8].to_bits());
                        if ulp != 0 {
                            differing[channel] += 1;
                            max_ulp = max_ulp.max(ulp);
                            hit = true;
                        }
                    }
                    any_channel += u64::from(hit);
                }
            }
        }
        let total = 1u64 << 24;
        println!(
            "tier=\"{}\" dispatch_is_scalar={scalar} colours={total} differing={any_channel} ({:.3}%) \
             per-channel X/Y/B={differing:?} max_ulp={max_ulp}",
            perm.label,
            100.0 * any_channel as f64 / total as f64
        );
        scalar_differs |= scalar && any_channel != 0;
    });
    if scalar_differs {
        eprintln!(
            "FAIL: the scalar tier still gives different bits in the chunk and the remainder"
        );
        std::process::exit(1);
    }
}
