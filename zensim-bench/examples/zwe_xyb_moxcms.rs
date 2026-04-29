//! Probe moxcms's "fast approx" 3D-LUT trilinear path on a u8→XYB-shaped
//! transform. Uses zenjpeg's embedded XYB ICC profile as the destination.
//!
//! The XYB definition this profile encodes differs from zensim's, but the
//! *shape* is identical — 3-channel u8 in, 3-channel out, going through
//! moxcms's lut16 / multidimensional path with the trilinear default.
//! Throughput tells us the ceiling we'd hit if we baked our own XYB into
//! a 3D CLUT and used moxcms (or its algorithm) to apply it.
//!
//! Run with:
//!   cargo run --release -p zensim-bench --example zwe_xyb_moxcms

use std::time::Instant;

use moxcms::{ColorProfile, Layout, TransformOptions};

static XYB_ICC_PROFILE: [u8; 720] = include!("xyb_icc_bytes.rs");

fn make_pixels(n: usize) -> Vec<u8> {
    (0..n * 3)
        .map(|i| ((i.wrapping_mul(31337) + 7) & 0xFF) as u8)
        .collect()
}

fn bench_size(label: &str, w: usize, h: usize, iters: usize) {
    let n = w * h;
    let src = make_pixels(n);
    let mut dst = vec![0u8; n * 3];

    let xyb_profile = match ColorProfile::new_from_slice(&XYB_ICC_PROFILE) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to parse XYB ICC profile: {e:?}");
            return;
        }
    };
    let srgb_profile = ColorProfile::new_srgb();

    let opts = TransformOptions::default();
    let transform = match srgb_profile.create_transform_8bit(
        Layout::Rgb,
        &xyb_profile,
        Layout::Rgb,
        opts,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to create transform for {label}: {e:?}");
            return;
        }
    };

    // Warmup
    for _ in 0..4 {
        transform.transform(&src, &mut dst).unwrap();
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        transform.transform(&src, &mut dst).unwrap();
        let dt = t0.elapsed().as_nanos() as u64;
        std::hint::black_box((&src, &dst));
        times.push(dt);
    }
    times.sort_unstable();
    let min = times[0];
    let p25 = times[iters / 4];
    let p50 = times[iters / 2];
    let ns_per_pixel_min = min as f64 / n as f64;
    let mp_per_s_min = (n as f64 * 1e3) / min as f64;
    let bytes_io = (n * 3 * 2) as f64;
    let bw = bytes_io / (min as f64 / 1e9) / (1024.0 * 1024.0 * 1024.0);
    println!(
        "{label:>11}  ({w}x{h}, {n}px, {iters}iter)  min={min}ns p25={p25} p50={p50}  ns/px={ns_per_pixel_min:.2}min  {mp_per_s_min:.0}MP/s  ~{bw:.2} GiB/s u8 I/O",
    );
}

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    println!("# zwe_xyb_moxcms: sRGB u8 → XYB u8 via zenjpeg's XYB ICC profile");
    println!("# (TransformOptions::default = trilinear + Q0.15 fixed-point + Low barycentric weights)");
    bench_size("256x256", 256, 256, iters);
    bench_size("512x512", 512, 512, iters);
    bench_size("1280x720", 1280, 720, iters);
    bench_size("1920x1080", 1920, 1080, iters);
    bench_size("3840x2160", 3840, 2160, iters / 2);
}
