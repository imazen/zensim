//! IW + extended-features memory cost measurement.
//!
//! Measures peak RSS during `Zensim::compute` for each of the four feature
//! configurations (basic, extended, IW-only, both). Reports the per-call
//! incremental memory cost vs basic. Run with:
//!
//! ```
//! cargo test --release --features training,threads -p zensim --test iw_memory_baseline \
//!     -- --nocapture --test-threads=1
//! ```
//!
//! Peak RSS read from `/proc/self/status` (VmHWM = high-watermark) — Linux only.

#![cfg(all(feature = "training", target_os = "linux"))]

use std::fs;
use zensim::{ZensimConfig, compute_zensim_with_config};

fn vm_hwm_kib() -> u64 {
    let s = fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kib: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .parse()
                .expect("parse VmHWM");
            return kib;
        }
    }
    panic!("VmHWM not in /proc/self/status");
}

fn reset_high_watermark() {
    // Linux 4.0+: writing "5" to /proc/self/clear_refs resets the
    // high-watermark counter without disturbing live mappings. If the
    // kernel doesn't support it, this is a no-op and the test reports
    // ABSOLUTE peak across all configs (still useful, just less precise).
    let _ = fs::write("/proc/self/clear_refs", "5");
}

fn make_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8; 3]; n];
    let mut dst = vec![[0u8; 3]; n];
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

fn measure_peak(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    extended: bool,
    iw: bool,
) -> u64 {
    let mut config = ZensimConfig::default();
    config.extended_features = extended;
    config.compute_iw_features = iw;
    config.compute_all_features = extended || iw;

    // Run once outside the measurement to fault in any shared globals
    // (lookup tables, archmage dispatch cache, etc).
    let _ = compute_zensim_with_config(src, dst, w, h, config);

    reset_high_watermark();
    let baseline = vm_hwm_kib();
    let _ = compute_zensim_with_config(src, dst, w, h, config);
    let peak = vm_hwm_kib();
    peak.saturating_sub(baseline)
}

fn run_geometry(name: &str, w: usize, h: usize) {
    let (src, dst) = make_pair(w, h);
    // Warm everything once
    let _ = compute_zensim_with_config(src.as_slice(), dst.as_slice(), w, h, ZensimConfig::default());

    let m_basic = measure_peak(&src, &dst, w, h, false, false);
    let m_ext   = measure_peak(&src, &dst, w, h, true,  false);
    let m_iw    = measure_peak(&src, &dst, w, h, false, true);
    let m_both  = measure_peak(&src, &dst, w, h, true,  true);

    eprintln!(
        "{name:>12}  basic +{b:>5} KiB  ext +{e:>5} KiB  iw +{i:>5} KiB  both +{bo:>5} KiB",
        b = m_basic, e = m_ext, i = m_iw, bo = m_both,
    );
}

#[test]
fn iw_memory_report() {
    eprintln!("\nIW + ext memory cost — Δ peak RSS per call (kB)");
    eprintln!("================================================================");
    run_geometry("256x256",     256, 256);
    run_geometry("512x512",     512, 512);
    run_geometry("1024x1024", 1024, 1024);
    run_geometry("2048x1024", 2048, 1024);
    run_geometry("4096x2048", 4096, 2048);
    eprintln!("================================================================");
}
