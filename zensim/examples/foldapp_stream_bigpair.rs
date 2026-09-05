// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! Large-synthetic-pair memory harness for the STREAMING folded+append
//! walk (G-RAM evidence for `benchmarks/streaming_foldapp_*`): generates
//! a deterministic W×H pair (the `tests/streaming_strips.rs` `make_pair`
//! pattern) and runs `compute_folded720_append_features_streaming` once.
//! Run under heaptrack to record peak heap at sizes where the
//! materialized path would allocate gigabytes:
//!
//! ```sh
//! cargo build --release -p zensim --features feature-regime-v2 \
//!   --example foldapp_stream_bigpair
//! heaptrack target/release/examples/foldapp_stream_bigpair 8000 10000
//! # ZENSIM_BIGPAIR_MODE=materialized compares the pre-C5 path (careful:
//! # allocates O(image) — multiple GB at 80 MP).
//! ```

use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// Deterministic procedural content — `tests/streaming_strips.rs`'s
/// `make_pair` shape (mode-5 hash variant), cheap to generate.
fn make_pair(w: usize, h: usize, seed: u32) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8, 0, 0]; n];
    let mut dst = vec![[0u8, 0, 0]; n];
    let h_fn = |a: u32, b: u32| -> u8 {
        let mut x = a
            .wrapping_mul(0x6C8E_9CF7)
            .wrapping_add(b.wrapping_mul(0x9E37_79B9))
            .wrapping_add(seed);
        x ^= x >> 16;
        x = x.wrapping_mul(0x85EB_CA6B);
        x ^= x >> 13;
        (x & 0xFF) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let r = h_fn(x as u32 / 4, y as u32 / 4);
            let g = h_fn(x as u32 / 2, y as u32 / 8);
            let b = h_fn(x as u32 / 8, y as u32 / 4);
            src[y * w + x] = [r, g, b];
            dst[y * w + x] = [
                r.saturating_add(2),
                g.saturating_sub(1),
                b.saturating_add(3),
            ];
        }
    }
    (src, dst)
}

fn main() {
    // `ZEN_S2_CAP_V3=1` disables `X64V4Token` process-wide, so the dispatch
    // ceiling becomes v3 (AVX2+FMA) — the SAME flag name and the same
    // refuse-rather-than-mislabel contract as
    // `zensim-bench/benches/ssim2_speed_bar.rs::cap_tier_v3`, so one
    // convention prices both tiers across both instruments. A cap cannot be
    // per-arm inside one process, so the honest shape is two sweeps with
    // this flipped; the env block stays byte-identical WITHIN a sweep,
    // which is what the ASLR protocol requires.
    #[cfg(target_arch = "x86_64")]
    if std::env::var("ZEN_S2_CAP_V3").as_deref() == Ok("1") {
        if let Err(e) = archmage::X64V4Token::dangerously_disable_token_process_wide(true) {
            eprintln!(
                "# ZEN_S2_CAP_V3=1 requested but could not disable X64V4Token: {e} \
                 — refusing to report a native-tier run mislabeled as capped."
            );
            std::process::exit(1);
        }
        eprintln!("# ZEN_S2_CAP_V3=1: X64V4Token disabled process-wide, ceiling=v3(AVX2)");
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    let w: usize = args.first().map(|s| s.parse().unwrap()).unwrap_or(8000);
    let h: usize = args.get(1).map(|s| s.parse().unwrap()).unwrap_or(10000);
    eprintln!(
        "generating {w}x{h} ({:.1} MP) pair...",
        (w * h) as f64 / 1e6
    );
    let t0 = std::time::Instant::now();
    let (src, dst) = make_pair(w, h, 1);
    eprintln!("generated in {:.2}s", t0.elapsed().as_secs_f64());

    // `ZENSIM_BIGPAIR_PARALLEL=1` runs the walk's own fan-out (channels +
    // H-blur bands); the default stays serial so `RAYON_NUM_THREADS=1`
    // 1T numbers are the harness's default shape.
    let parallel = std::env::var("ZENSIM_BIGPAIR_PARALLEL").as_deref() == Ok("1");
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(parallel);
    let s_img = RgbSlice::new(&src, w, h);
    let d_img = RgbSlice::new(&dst, w, h);
    let mut scratch = V2Scratch::new();
    let mode = std::env::var("ZENSIM_BIGPAIR_MODE").unwrap_or_else(|_| "streaming".into());
    // `ZENSIM_BIGPAIR_TOGGLES` selects the extraction SHAPE, so one binary
    // covers the three arms the perf work compares:
    //   `944full` — every pool live (`folded720append2pools`), the product mode
    //   `924`     — append2/csfw/pools off (the crate default)
    //   `372`     — `v1_only`: phase A never runs, each fold band self-blurs
    //   `156`/`15c`/`15f` — the basic-only class, its 944-layout control, and
    //             the free-superset walk (free_features_2026-09-01)
    let arm = std::env::var("ZENSIM_BIGPAIR_TOGGLES").unwrap_or_else(|_| "944full".into());
    let toggles = match arm.as_str() {
        "372" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Full,
            ..Default::default()
        },
        "924" => V2NewFeatureToggles::default(),
        // `944peaks` — the full 944 shape MINUS the masked/IW pool pass.
        // Peaks are free byproducts, so they stay.
        "944peaks" => V2NewFeatureToggles {
            append2_block: true,
            csfw_block: true,
            append2_dst_activity: true,
            v1_pools: V1PoolsMode::Peaks,
            ..Default::default()
        },
        // `156` — the basic-only (ADD156) class: v1 basic + free peaks, no
        // v2-era block at all.
        "156" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            ..Default::default()
        },
        // DIAGNOSTIC (added by the profile_d_notax lane, 2026-09-01):
        // `V1PoolsMode::Off` + `v1_only` — the exact combination `zensim-bench/
        // benches/ssim2_speed_bar.rs`'s `v1_basic` toggle constructs by hand
        // for the `add156_156basic` arm, bypassing the `pools_mode_for_need`
        // policy that never returns `Off` in production (`Off` and `Peaks`
        // cost the same to compute; `Off` just emits fewer live slots). Used
        // to confirm this specific combination does NOT hang core zensim
        // under threading — it doesn't (5.49-8.27 ms at 1152x1152/8T) — which
        // localized a `ssim2_speed_bar --bench` hang (add-only, no free arm)
        // to the zenbench-harness/Predictor-3-arm-group context instead of
        // this crate. See `benchmarks/profile_d_notax_2026-09-01.md` §3.3.
        // `15o` — the SAME toggles as `156off`, under a THREE-character name so
        // it can be interleaved against `156` without changing the environment
        // block's byte length (era-2 §22.5: the env block is a layout input, and
        // `TOGGLES=156off` vs `TOGGLES=156` is a 3-byte difference measuring a
        // different address space). `156off` is kept as a readable alias for
        // one-off runs where no A/B is being read.
        //
        // What the A/B prices: `pools_mode_for_need` NEVER returns
        // `V1PoolsMode::Off`, and the band-local self-blur shape
        // (`FoldHSource::SelfBlur`) is gated on `Full | Peaks` — so `Off` falls
        // back to phase A's four STRIP-wide H planes where `Peaks` blurs exactly
        // the rows each band consumes. `Off` and `Peaks` compute the same sums
        // and `Peaks` emits a superset; the only difference is footprint. This
        // arm is how that difference gets a number instead of an argument.
        "156off" | "15o" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Off,
            ..Default::default()
        },
        // --- the free-feature A/B (`benchmarks/free_features_2026-09-01.md`).
        // THREE characters like `156`, because the ASLR protocol (era-2 §22.5)
        // requires byte-identical environment blocks between interleaved arms:
        // the size of the env block is itself a layout input, and an arm
        // selected by a longer string is measuring a different address space.
        //
        // `15c` — the CONTROL: the 944 LAYOUT with the v1-only COMPUTE set, so
        // it differs from `15f` in nothing but the four accumulators. Its
        // measured delta against `15f` is the whole question; its delta
        // against `156` is what the wider output vector costs.
        "15c" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            append_block: true,
            append2_block: true,
            ..Default::default()
        },
        // `15f` — the FREE SUPERSET walk: the same compute set plus the four
        // raw moments, which finalize the 40 free v2-era slots.
        "15f" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            append_block: true,
            append2_block: true,
            free_extras: V1FreeExtras::RawMoments,
            ..Default::default()
        },
        // `15x` — the CLASS-C walk: `15f` plus the bounded-error tranche
        // (`benchmarks/free_features_classC_2026-09-04.md`). Three
        // characters, same reason as `15c`/`15f`. `15x` vs `15f` is the
        // class-C marginal cost; `15x` vs `15c` is the whole free set's.
        "15x" => V2NewFeatureToggles {
            v1_only: true,
            v1_pools: V1PoolsMode::Peaks,
            append_block: true,
            append2_block: true,
            free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
            ..Default::default()
        },
        // `944carriers` — pools reduced to the 10 named carrier slots.
        "944carriers" => V2NewFeatureToggles {
            append2_block: true,
            csfw_block: true,
            append2_dst_activity: true,
            v1_pools: V1PoolsMode::Carriers,
            ..Default::default()
        },
        _ => V2NewFeatureToggles {
            append2_block: true,
            csfw_block: true,
            append2_dst_activity: true,
            v1_pools: V1PoolsMode::Full,
            ..Default::default()
        },
    };
    // `ZENSIM_BIGPAIR_ITERS=N` repeats the timed walk N times (median
    // reported); the pair is generated once, outside the timed region.
    let iters: usize = std::env::var("ZENSIM_BIGPAIR_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    if iters > 1 {
        let mut ms: Vec<f64> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            let r = z
                .compute_folded720_append_features_streaming(&s_img, &d_img, toggles, &mut scratch)
                .expect("streaming foldapp");
            ms.push(t.elapsed().as_secs_f64() * 1e3);
            std::hint::black_box(r.features()[0]);
        }
        if let Ok(path) = std::env::var("ZENSIM_BIGPAIR_DUMP") {
            let r = z
                .compute_folded720_append_features_streaming(&s_img, &d_img, toggles, &mut scratch)
                .expect("streaming foldapp");
            let mut out = String::new();
            for (i, v) in r.features().iter().enumerate() {
                out.push_str(&format!("{i}\t{:.17e}\t{:016x}\n", v, v.to_bits()));
            }
            std::fs::write(&path, out).expect("dump");
            eprintln!("dumped {} features to {path}", r.features().len());
        }
        if let Ok(roll) = std::fs::read_to_string("/proc/self/smaps_rollup") {
            let g = |k: &str| -> String {
                roll.lines()
                    .find(|l| l.starts_with(k))
                    .map(|l| l.split_whitespace().nth(1).unwrap_or("?").to_string())
                    .unwrap_or_else(|| "?".into())
            };
            eprintln!(
                "smaps: Rss={} kB AnonHugePages={} kB",
                g("Rss:"),
                g("AnonHugePages:")
            );
        }
        ms.sort_by(f64::total_cmp);
        eprintln!(
            "arm={arm} {w}x{h} iters={iters} median {:.2} ms  min {:.2}  max {:.2}",
            ms[ms.len() / 2],
            ms[0],
            ms[ms.len() - 1]
        );
        return;
    }
    let t1 = std::time::Instant::now();
    let r = match mode.as_str() {
        "materialized" => z
            .compute_folded720_append_features(&s_img, &d_img)
            .expect("materialized foldapp"),
        _ => z
            .compute_folded720_append_features_streaming(&s_img, &d_img, toggles, &mut scratch)
            .expect("streaming foldapp"),
    };
    eprintln!(
        "{mode} foldapp: {:.2}s, {} features, f0={:.6} f923={:.6}",
        t1.elapsed().as_secs_f64(),
        r.features().len(),
        r.features()[0],
        r.features()[923],
    );
}
