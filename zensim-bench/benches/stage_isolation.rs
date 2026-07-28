//! Per-stage NEON-vs-scalar attribution for the zensim extraction pipeline.
//!
//! The whole-pipeline `tier_isolation` bench established two things: NEON is
//! worth only ~1.26x here (butteraugli, a comparable multi-scale XYB+blur
//! metric, gets 3.4x on the same host), and the shortfall is inside the image
//! kernels rather than in scoring — `compute_extended_features` is 99% of
//! total time and carries the same ratio as `compute`.
//!
//! Attributing it further wants a profiler, and on macOS dtrace
//! (cargo-flamegraph) needs SIP disabled. This gets the same attribution
//! without one: run each hot stage under the identical SIMD/scalar A/B and see
//! which stages carry a healthy ratio and which do not.
//!
//! Run: `cd zensim-bench && cargo bench --bench stage_isolation`
//! Do NOT build with `-C target-cpu=native` (the tier cannot be disabled then).

use criterion::{Criterion, criterion_group, criterion_main};
use zensim::__bench_stages as st;

#[cfg(target_arch = "aarch64")]
type TierToken = archmage::NeonToken;
#[cfg(target_arch = "x86_64")]
type TierToken = archmage::X64V3Token;

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
const TIER_NAME: &str = if cfg!(target_arch = "aarch64") {
    "neon"
} else {
    "v3(avx2)"
};

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn set_simd(enabled: bool) -> bool {
    use archmage::SimdToken;
    TierToken::dangerously_disable_token_process_wide(!enabled).is_ok()
}

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
fn set_simd(_enabled: bool) -> bool {
    false
}

const W: usize = 1024;
const H: usize = 1024;
const N: usize = W * H;

fn plane(seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..N)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (s >> 8) as f32 / 16_777_216.0
        })
        .collect()
}

fn rgb8(seed: u32) -> Vec<[u8; 3]> {
    let mut s = seed.wrapping_mul(0x9e37_79b9) | 1;
    (0..N)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            [(s >> 24) as u8, (s >> 16) as u8, (s >> 8) as u8]
        })
        .collect()
}

/// Run one stage under both tiers in the SAME group, so the comparison is
/// within-group (cross-group comparison produced two false regressions in this
/// sweep — see the benchmarks/ records for zenjxl-decoder and linear-srgb).
fn ab(c: &mut Criterion, name: &str, mut f: impl FnMut()) {
    let mut group = c.benchmark_group(name);
    for (arm, simd) in [(TIER_NAME, true), ("scalar", false)] {
        group.bench_function(arm, |b| {
            set_simd(simd);
            b.iter(&mut f)
        });
    }
    set_simd(true);
    group.finish();
}

fn bench_stages(c: &mut Criterion) {
    if !set_simd(true) || !set_simd(false) {
        eprintln!("[stage_isolation] SIMD tier not toggleable here. Skipping.");
        return;
    }
    set_simd(true);
    eprintln!("[stage_isolation] comparing {TIER_NAME} vs forced scalar, {W}x{H}");

    let a = plane(1);
    let bb = plane(2);
    let px = rgb8(3);

    {
        let (mut out, mut tmp) = (vec![0f32; N], vec![0f32; N]);
        let a2 = a.clone();
        ab(c, "blur/box_blur_1pass", move || {
            st::box_blur_1pass_into(&a2, &mut out, &mut tmp, W, H, 3);
        });
    }
    {
        let (nw, nh) = (W / 2, H / 2);
        let mut dst = vec![0f32; nw * nh];
        let a2 = a.clone();
        ab(c, "blur/downscale_2x", move || {
            st::downscale_2x_into(&a2, W, &mut dst, nw, nh);
        });
    }
    {
        let (mut m1, mut m2) = (vec![0f32; N], vec![0f32; N]);
        let (mut ss, mut s12) = (vec![0f32; N], vec![0f32; N]);
        let (a2, b2) = (a.clone(), bb.clone());
        ab(c, "blur/fused_blur_h_ssim", move || {
            st::fused_blur_h_ssim(&a2, &b2, &mut m1, &mut m2, &mut ss, &mut s12, W, H, 3);
        });
    }
    {
        let (mut x, mut y, mut z) = (vec![0f32; N], vec![0f32; N], vec![0f32; N]);
        let p2 = px.clone();
        ab(c, "color/srgb_to_positive_xyb", move || {
            st::srgb_to_positive_xyb_planar_into(&p2, &mut x, &mut y, &mut z);
        });
    }
    {
        let mut out = vec![0f32; N];
        let (a2, b2) = (a.clone(), bb.clone());
        ab(c, "simd_ops/mul_into", move || {
            st::mul_into(&a2, &b2, &mut out);
        });
    }
    {
        let mut out = vec![0f32; N];
        let (a2, b2) = (a.clone(), bb.clone());
        ab(c, "simd_ops/sq_sum_into", move || {
            st::sq_sum_into(&a2, &b2, &mut out);
        });
    }
    {
        let (a2, b2) = (a.clone(), bb.clone());
        ab(c, "simd_ops/sq_diff_sum", move || {
            std::hint::black_box(st::sq_diff_sum(&a2, &b2));
        });
    }
    {
        let (a2, b2) = (a.clone(), bb.clone());
        ab(c, "simd_ops/abs_diff_sum", move || {
            std::hint::black_box(st::abs_diff_sum(&a2, &b2));
        });
    }
}

criterion_group!(benches, bench_stages);
criterion_main!(benches);
