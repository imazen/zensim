//! **The feature-invariant probe** — the measurement half of the
//! 2026-09-05 feature-defect audit (`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`).
//!
//! Answers the user's question ("do we have bugs in feature calculations?")
//! with numbers rather than opinions. Each mode measures ONE invariant that a
//! feature extractor is supposed to satisfy, over the engines (buffered v1
//! walk / streaming fold / 944 fold), the widths (tight vs non-tight vs
//! sub-64 vs odd), the thread counts, and the SIMD tiers.
//!
//! The cheap, always-true half of what this measures is asserted as a gate in
//! `zensim/tests/feature_invariants.rs`. This binary exists for the half that
//! is a *census* rather than a bar — "which slots are non-zero on an identical
//! pair", "which slots are non-monotone under a blur ladder" — where the
//! deliverable is the list, not a pass/fail.
//!
//! ```text
//! cargo run --release -p zensim --features training,feature-regime-v2,custom-profiles \
//!   --example feature_invariant_probe -- <mode> [out.tsv]
//! ```
//!
//! Modes: `identity` `determinism` `engineparity` `degenerate` `ladder`
//! `scale` `depth` `all`.
//!
//! Every mode writes a TSV whose first column is the mode, so the outputs
//! concatenate into one table.

use std::io::Write;

use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::fold_engine::ScoringEngine;
use zensim::{RgbSlice, Zensim, ZensimConfig, ZensimProfile, compute_zensim_with_config};

// ---------------------------------------------------------------------------
// generators (local — `tests/common` is not reachable from an example)
// ---------------------------------------------------------------------------

fn value_noise(w: usize, h: usize, seed: u32) -> Vec<[u8; 3]> {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    // Smooth-ish value noise: a coarse lattice bilinearly interpolated, plus a
    // fine speckle. Gives real gradient/edge/texture content at every scale so
    // the pools and the v2 blocks are genuinely exercised.
    let lat = 16usize;
    let (lw, lh) = (w / lat + 2, h / lat + 2);
    let grid: Vec<[f32; 3]> = (0..lw * lh)
        .map(|_| {
            [
                (next() % 256) as f32,
                (next() % 256) as f32,
                (next() % 256) as f32,
            ]
        })
        .collect();
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let (gx, gy) = (x / lat, y / lat);
            let (fx, fy) = ((x % lat) as f32 / lat as f32, (y % lat) as f32 / lat as f32);
            let mut px = [0u8; 3];
            for c in 0..3 {
                let a = grid[gy * lw + gx][c];
                let b = grid[gy * lw + gx + 1][c];
                let cc = grid[(gy + 1) * lw + gx][c];
                let d = grid[(gy + 1) * lw + gx + 1][c];
                let top = a + (b - a) * fx;
                let bot = cc + (d - cc) * fx;
                let v = top + (bot - top) * fy;
                let speckle = ((x * 7 + y * 13 + c * 29) % 17) as f32 - 8.0;
                px[c] = (v + speckle).clamp(0.0, 255.0) as u8;
            }
            out.push(px);
        }
    }
    out
}

/// Separable box blur repeated `passes` times — a controlled, strictly
/// increasing single-axis distortion ladder (more passes = strictly more
/// low-pass). Used as the monotonicity stimulus.
fn blur(src: &[[u8; 3]], w: usize, h: usize, radius: usize, passes: usize) -> Vec<[u8; 3]> {
    let mut cur: Vec<[f32; 3]> = src
        .iter()
        .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    for _ in 0..passes {
        let mut tmp = vec![[0.0f32; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0f32;
                for dx in -(radius as isize)..=(radius as isize) {
                    let xx = (x as isize + dx).clamp(0, w as isize - 1) as usize;
                    for c in 0..3 {
                        acc[c] += cur[y * w + xx][c];
                    }
                    n += 1.0;
                }
                for c in 0..3 {
                    tmp[y * w + x][c] = acc[c] / n;
                }
            }
        }
        for x in 0..w {
            for y in 0..h {
                let mut acc = [0.0f32; 3];
                let mut n = 0.0f32;
                for dy in -(radius as isize)..=(radius as isize) {
                    let yy = (y as isize + dy).clamp(0, h as isize - 1) as usize;
                    for c in 0..3 {
                        acc[c] += tmp[yy * w + x][c];
                    }
                    n += 1.0;
                }
                for c in 0..3 {
                    cur[y * w + x][c] = acc[c] / n;
                }
            }
        }
    }
    cur.iter()
        .map(|p| {
            [
                p[0].clamp(0.0, 255.0) as u8,
                p[1].clamp(0.0, 255.0) as u8,
                p[2].clamp(0.0, 255.0) as u8,
            ]
        })
        .collect()
}

/// Deterministic additive noise of a given amplitude — the second ladder axis.
fn add_noise(src: &[[u8; 3]], amp: i32, seed: u32) -> Vec<[u8; 3]> {
    let mut s = seed | 1;
    src.iter()
        .map(|p| {
            let mut out = [0u8; 3];
            for c in 0..3 {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let d = (s % (2 * amp as u32 + 1)) as i32 - amp;
                out[c] = (p[c] as i32 + d).clamp(0, 255) as u8;
            }
            out
        })
        .collect()
}

/// Quantize each channel to `step` — a blocky, strictly-coarsening ladder that
/// is monotone in a different way than blur (amplitude, not bandwidth).
fn quantize(src: &[[u8; 3]], step: u32) -> Vec<[u8; 3]> {
    src.iter()
        .map(|p| {
            let mut out = [0u8; 3];
            for c in 0..3 {
                let q = (p[c] as u32 / step) * step + step / 2;
                out[c] = q.min(255) as u8;
            }
            out
        })
        .collect()
}

fn box_downscale2(src: &[[u8; 3]], w: usize, h: usize) -> (Vec<[u8; 3]>, usize, usize) {
    let (dw, dh) = (w / 2, h / 2);
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        for x in 0..dw {
            let mut acc = [0u32; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = src[(y * 2 + dy) * w + (x * 2 + dx)];
                    for c in 0..3 {
                        acc[c] += p[c] as u32;
                    }
                }
            }
            out.push([(acc[0] / 4) as u8, (acc[1] / 4) as u8, (acc[2] / 4) as u8]);
        }
    }
    (out, dw, dh)
}

// ---------------------------------------------------------------------------
// feature extraction routes
// ---------------------------------------------------------------------------

/// The 944 fold, all pools live + append + append2 + the free/class-C tranche.
/// This is the ONLY route that genuinely COMPUTES a feature vector for an
/// identical pair — every `Zensim::compute*` and the v1-372 extractor entry
/// both short-circuit `source == distorted` and return a FABRICATED all-zero
/// payload (`metric.rs::identical_result`, `compute_zensim_with_config`).
fn fold944(
    r: &[[u8; 3]],
    d: &[[u8; 3]],
    w: usize,
    h: usize,
    pools: V1PoolsMode,
    free: V1FreeExtras,
    v1_only: bool,
) -> Vec<f64> {
    let toggles = V2NewFeatureToggles {
        v1_pools: pools,
        append_block: !v1_only,
        append2_block: !v1_only,
        v1_only,
        free_extras: free,
        ..Default::default()
    };
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_features_streaming(
        &RgbSlice::new(r, w, h),
        &RgbSlice::new(d, w, h),
        toggles,
        &mut scratch,
    )
    .expect("944 fold walk")
    .features()
    .to_vec()
}

/// The v1-372 config both v1-372 extractors use.
fn v1_config() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg.allow_multithreading = false;
    cfg
}

fn v1_372(r: &[[u8; 3]], d: &[[u8; 3]], w: usize, h: usize) -> Vec<f64> {
    compute_zensim_with_config(r, d, w, h, v1_config())
        .expect("v1-372")
        .features()
        .to_vec()
}

/// Geometry set: tight, non-tight, odd, sub-64, wider-than-H_TILE_WIDTH.
const CELLS: &[(usize, usize)] = &[
    (64, 64),   // tight, the golden fixture class
    (96, 64),   // tight
    (100, 96),  // non-tight, even
    (127, 93),  // non-tight, odd, the h=93 class
    (129, 96),  // non-tight, odd
    (200, 150), // the golden fixture class
    (255, 96),  // non-tight, odd
    (512, 96),  // the +16 padded class
    (576, 96),  // the +16 padded class
    (592, 80),  // tight above 512
    (48, 64),   // SUB-64 width — reflect-padded by the product entries
    (64, 48),   // SUB-64 height
    (43, 64),   // SUB-64, the R1b ragged class
    (1153, 72), // wider than H_TILE_WIDTH (1024), 129-column remainder
    (2049, 40), // two tile boundaries, 1-column remainder tile
];

fn out_writer(path: Option<&str>) -> Box<dyn Write> {
    match path {
        Some(p) => Box::new(std::fs::File::create(p).expect("create out")),
        None => Box::new(std::io::stdout()),
    }
}

// ---------------------------------------------------------------------------
// (a) IDENTITY — ref == dist must give the ZERO vector for every DIFFERENCE
//     feature. Census every slot that is not zero, at every width/layout.
// ---------------------------------------------------------------------------

fn mode_identity(out: &mut dyn Write) {
    writeln!(out, "mode\troute\tw\th\tslot\tvalue\tnonzero_rows_note").unwrap();
    // Per-route accumulators: slot -> (rows non-zero, max |v|).
    let routes: &[(&str, V1PoolsMode, V1FreeExtras, bool)] = &[
        (
            "fold944_full",
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        ),
        (
            "fold944_pools_off",
            V1PoolsMode::Off,
            V1FreeExtras::Off,
            false,
        ),
        (
            "fold_v1only_pools_full",
            V1PoolsMode::Full,
            V1FreeExtras::Off,
            true,
        ),
        (
            "fold_156free",
            V1PoolsMode::Peaks,
            V1FreeExtras::RawMoments,
            true,
        ),
    ];
    for &(name, pools, free, v1_only) in routes {
        let mut census: std::collections::BTreeMap<usize, (usize, f64)> = Default::default();
        for &(w, h) in CELLS {
            if w < 64 || h < 64 {
                // the fold has no reflect-pad front; the product entries do.
                continue;
            }
            let r = value_noise(w, h, 0xC0FFEE);
            let f = fold944(&r, &r, w, h, pools, free, v1_only);
            for (i, &v) in f.iter().enumerate() {
                if v != 0.0 {
                    let e = census.entry(i).or_insert((0, 0.0));
                    e.0 += 1;
                    e.1 = e.1.max(v.abs());
                }
            }
        }
        for (slot, (rows, maxabs)) in census {
            writeln!(out, "identity\t{name}\t-\t-\tf{slot}\t{maxabs:.6e}\t{rows}").unwrap();
        }
    }
    // And the FABRICATED payloads, for contrast.
    for &(w, h) in &[(200usize, 150usize), (127, 93)] {
        let r = value_noise(w, h, 0xC0FFEE);
        let f = v1_372(&r, &r, w, h);
        let nz = f.iter().filter(|&&v| v != 0.0).count();
        writeln!(
            out,
            "identity\tv1_372_shortcircuit\t{w}\t{h}\tnonzero_count\t{nz}\tfabricated"
        )
        .unwrap();
        let z = Zensim::new(ZensimProfile::codec_target());
        let res = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&r, w, h))
            .expect("compute");
        let nz2 = res.features().iter().filter(|&&v| v != 0.0).count();
        writeln!(
            out,
            "identity\tZensim_compute_shortcircuit\t{w}\t{h}\tnonzero_count\t{nz2}\tfabricated score={}",
            res.score()
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// (b) DETERMINISM — repeat runs in one process must be bit-identical.
// ---------------------------------------------------------------------------

fn mode_determinism(out: &mut dyn Write) {
    writeln!(out, "mode\troute\tw\th\treps\tfirst_diff_slot\tstatus").unwrap();
    for &(w, h) in CELLS {
        if w < 64 || h < 64 {
            continue;
        }
        let r = value_noise(w, h, 0xC0FFEE);
        let d = blur(&r, w, h, 1, 1);
        let mut base: Option<Vec<u64>> = None;
        let mut first_diff: Option<usize> = None;
        for _ in 0..5 {
            let bits: Vec<u64> = fold944(
                &r,
                &d,
                w,
                h,
                V1PoolsMode::Full,
                V1FreeExtras::RawMomentsPlusBoundedErr,
                false,
            )
            .iter()
            .map(|v| v.to_bits())
            .collect();
            match &base {
                None => base = Some(bits),
                Some(b) => {
                    if first_diff.is_none() {
                        first_diff = (0..b.len()).find(|&i| b[i] != bits[i]);
                    }
                }
            }
        }
        writeln!(
            out,
            "determinism\tfold944_full\t{w}\t{h}\t5\t{}\t{}",
            first_diff.map(|i| i.to_string()).unwrap_or("-".into()),
            if first_diff.is_none() {
                "BIT-IDENTICAL"
            } else {
                "DIVERGED"
            }
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// (c) ENGINE PARITY — the buffered v1-372 walk vs the fold's own v1 block, on
//     the same pixels, in the same process, at the same era.
// ---------------------------------------------------------------------------

fn mode_engine_parity(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tcomparison\tw\th\tslots\tcells_differing\tmax_abs_delta\tstatus"
    )
    .unwrap();
    for &(w, h) in CELLS {
        if w < 64 || h < 64 {
            continue;
        }
        let r = value_noise(w, h, 0xC0FFEE);
        let d = blur(&r, w, h, 1, 1);

        // buffered v1-372 vs the fold's v1_only f0..372
        let a = v1_372(&r, &d, w, h);
        let b = fold944(&r, &d, w, h, V1PoolsMode::Full, V1FreeExtras::Off, true);
        let mut ndiff = 0usize;
        let mut maxd = 0.0f64;
        for i in 0..372 {
            if a[i].to_bits() != b[i].to_bits() {
                ndiff += 1;
                maxd = maxd.max((a[i] - b[i]).abs());
            }
        }
        writeln!(
            out,
            "engineparity\tbuffered_v1_372__vs__fold_v1only\t{w}\t{h}\t372\t{ndiff}\t{maxd:.6e}\t{}",
            if ndiff == 0 { "BIT-EXACT" } else { "DIVERGED" }
        )
        .unwrap();

        // The FULL 944 walk's v1 block vs the same fold's v1_only block: does
        // turning the v2-era blocks on move v1's own slots?
        let c = fold944(
            &r,
            &d,
            w,
            h,
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        );
        let mut ndiff2 = 0usize;
        let mut maxd2 = 0.0f64;
        for i in 0..372 {
            if b[i].to_bits() != c[i].to_bits() {
                ndiff2 += 1;
                maxd2 = maxd2.max((b[i] - c[i]).abs());
            }
        }
        writeln!(
            out,
            "engineparity\tfold_v1only__vs__fold944_full\t{w}\t{h}\t372\t{ndiff2}\t{maxd2:.6e}\t{}",
            if ndiff2 == 0 { "BIT-EXACT" } else { "DIVERGED" }
        )
        .unwrap();

        // Buffered vs fold through the PRODUCT entry (score + full vector).
        let zb = Zensim::new(ZensimProfile::codec_target()).with_engine(ScoringEngine::Buffered);
        let zf = Zensim::new(ZensimProfile::codec_target()).with_engine(ScoringEngine::Fold);
        let rb = zb
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("buffered");
        let rf = zf
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("fold");
        let n = rb.features().len().min(rf.features().len());
        let mut ndiff3 = 0usize;
        let mut maxd3 = 0.0f64;
        for i in 0..n {
            if rb.features()[i].to_bits() != rf.features()[i].to_bits() {
                ndiff3 += 1;
                maxd3 = maxd3.max((rb.features()[i] - rf.features()[i]).abs());
            }
        }
        writeln!(
            out,
            "engineparity\tproduct_buffered__vs__product_fold\t{w}\t{h}\t{n}\t{ndiff3}\t{maxd3:.6e}\t{}",
            if ndiff3 == 0 && rb.score().to_bits() == rf.score().to_bits() {
                "BIT-EXACT"
            } else {
                "DIVERGED"
            }
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// (h) DEGENERATE INPUTS — no NaN, no Inf, on flat / black / white / extreme.
// ---------------------------------------------------------------------------

// The per-case closure type is exactly as complex as the case list needs
// (name + a boxed two-image generator); a `type` alias would just move the
// same signature one hop away.
#[allow(clippy::type_complexity)]
fn mode_degenerate(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tcase\tw\th\troute\tnan_slots\tinf_slots\tfirst_bad_slot\tmax_abs_finite"
    )
    .unwrap();
    let cases: Vec<(
        &str,
        Box<dyn Fn(usize, usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>)>,
    )> = vec![
        (
            "flat_gray_vs_flat_gray_off_by_one",
            Box::new(|w, h| (vec![[128, 128, 128]; w * h], vec![[129, 129, 129]; w * h])),
        ),
        (
            "all_black_vs_all_white",
            Box::new(|w, h| (vec![[0, 0, 0]; w * h], vec![[255, 255, 255]; w * h])),
        ),
        (
            "all_black_vs_one_pixel_lit",
            Box::new(|w, h| {
                let r = vec![[0u8, 0, 0]; w * h];
                let mut d = r.clone();
                d[0] = [255, 255, 255];
                (r, d)
            }),
        ),
        (
            "all_white_vs_one_pixel_dark",
            Box::new(|w, h| {
                let r = vec![[255u8, 255, 255]; w * h];
                let mut d = r.clone();
                d[w * h / 2] = [0, 0, 0];
                (r, d)
            }),
        ),
        (
            "flat_vs_noise",
            Box::new(|w, h| {
                let r = vec![[128u8, 128, 128]; w * h];
                let d = add_noise(&r, 60, 7);
                (r, d)
            }),
        ),
        (
            "saturated_channels",
            Box::new(|w, h| {
                let r: Vec<[u8; 3]> = (0..w * h)
                    .map(|i| [255, 0, if i % 2 == 0 { 255 } else { 0 }])
                    .collect();
                let d: Vec<[u8; 3]> = (0..w * h)
                    .map(|i| [0, 255, if i % 2 == 0 { 0 } else { 255 }])
                    .collect();
                (r, d)
            }),
        ),
    ];
    for (name, mk) in &cases {
        for &(w, h) in &[(64usize, 64usize), (127, 93), (200, 150), (1153, 72)] {
            let (r, d) = mk(w, h);
            for (route, pools, free, v1o) in [
                (
                    "fold944_full",
                    V1PoolsMode::Full,
                    V1FreeExtras::RawMomentsPlusBoundedErr,
                    false,
                ),
                ("fold_v1only", V1PoolsMode::Full, V1FreeExtras::Off, true),
            ] {
                let f = fold944(&r, &d, w, h, pools, free, v1o);
                let nan = f.iter().filter(|v| v.is_nan()).count();
                let inf = f.iter().filter(|v| v.is_infinite()).count();
                let first = f.iter().position(|v| !v.is_finite());
                let maxf = f
                    .iter()
                    .filter(|v| v.is_finite())
                    .fold(0.0f64, |a, &v| a.max(v.abs()));
                writeln!(
                    out,
                    "degenerate\t{name}\t{w}\t{h}\t{route}\t{nan}\t{inf}\t{}\t{maxf:.6e}",
                    first.map(|i| format!("f{i}")).unwrap_or("-".into())
                )
                .unwrap();
            }
            // The product entry too (it reflect-pads and scores).
            let z = Zensim::new(ZensimProfile::codec_target());
            let res = z
                .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
                .expect("compute");
            let nan = res.features().iter().filter(|v| v.is_nan()).count();
            let inf = res.features().iter().filter(|v| v.is_infinite()).count();
            writeln!(
                out,
                "degenerate\t{name}\t{w}\t{h}\tproduct_compute\t{nan}\t{inf}\t{}\tscore={:.6}",
                if res.score().is_finite() {
                    "-"
                } else {
                    "SCORE"
                },
                res.score()
            )
            .unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// (f) MONOTONICITY — per-slot response to a controlled single-axis ladder.
// ---------------------------------------------------------------------------

// Same shape as `mode_degenerate`'s allow: the boxed per-step closure list is
// the plain representation of "a named ladder of image transforms".
#[allow(clippy::type_complexity)]
fn mode_ladder(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tladder\tslot\tn_steps\tn_images\tstrict_mono_images\tviolations\tworst_backstep\tdirection\tmax_abs_range\tseries_img0"
    )
    .unwrap();
    let geoms = [(200usize, 150usize), (256, 192), (127, 93)];
    let seeds = [0xC0FFEEu32, 0xBEEF, 0x1234, 0xABCD];

    struct Ladder {
        name: &'static str,
        steps: Vec<Box<dyn Fn(&[[u8; 3]], usize, usize) -> Vec<[u8; 3]>>>,
    }
    let ladders = vec![
        Ladder {
            name: "boxblur_passes_1..6",
            steps: (1..=6)
                .map(|p| {
                    Box::new(move |s: &[[u8; 3]], w: usize, h: usize| blur(s, w, h, 1, p))
                        as Box<dyn Fn(&[[u8; 3]], usize, usize) -> Vec<[u8; 3]>>
                })
                .collect(),
        },
        Ladder {
            name: "noise_amp_4..48",
            steps: [4i32, 8, 16, 24, 32, 48]
                .into_iter()
                .map(|a| {
                    Box::new(move |s: &[[u8; 3]], _w: usize, _h: usize| add_noise(s, a, 0x5EED))
                        as Box<dyn Fn(&[[u8; 3]], usize, usize) -> Vec<[u8; 3]>>
                })
                .collect(),
        },
        Ladder {
            name: "quantize_step_4..64",
            steps: [4u32, 8, 16, 24, 40, 64]
                .into_iter()
                .map(|q| {
                    Box::new(move |s: &[[u8; 3]], _w: usize, _h: usize| quantize(s, q))
                        as Box<dyn Fn(&[[u8; 3]], usize, usize) -> Vec<[u8; 3]>>
                })
                .collect(),
        },
    ];

    for lad in &ladders {
        let n_steps = lad.steps.len();
        // absolute dynamic range per slot, and the first image's raw series —
        // without these a "violation" cannot be told from fp noise on a slot
        // that is essentially constant along the ladder.
        let mut max_range = vec![0.0f64; 944];
        let mut first_series: Vec<Vec<f64>> = vec![Vec::new(); 944];
        let (mut ctrl_total, mut ctrl_viol) = (0usize, 0usize);
        let mut ctrl_first = String::new();
        // per-slot: images strictly monotone (either direction), violation count,
        // worst backstep magnitude relative to the slot's own range.
        let width = 944;
        let mut mono_up = vec![0usize; width];
        let mut mono_dn = vec![0usize; width];
        let mut viol = vec![0usize; width];
        let mut worst = vec![0.0f64; width];
        let mut n_images = 0usize;
        for &(w, h) in &geoms {
            for &seed in &seeds {
                let r = value_noise(w, h, seed);
                // CONTROL: plain per-pixel MSE(ref, rung). A ladder whose own
                // MSE is not monotone cannot be used to call a FEATURE
                // non-monotone — the stimulus would be the defect, not the
                // feature. Emitted as pseudo-slot `CTRL_MSE` so it rides the
                // same violation counter as every real slot.
                let mses: Vec<f64> = lad
                    .steps
                    .iter()
                    .map(|f| {
                        let d = f(&r, w, h);
                        let mut acc = 0.0f64;
                        for (a, b) in r.iter().zip(d.iter()) {
                            for c in 0..3 {
                                let e = a[c] as f64 - b[c] as f64;
                                acc += e * e;
                            }
                        }
                        acc / (r.len() * 3) as f64
                    })
                    .collect();
                let mse_up = mses.windows(2).all(|p| p[1] >= p[0]);
                ctrl_total += 1;
                if !mse_up {
                    ctrl_viol += 1;
                    if ctrl_first.is_empty() {
                        ctrl_first = mses
                            .iter()
                            .map(|v| format!("{v:.4e}"))
                            .collect::<Vec<_>>()
                            .join(",");
                    }
                }
                let series: Vec<Vec<f64>> = lad
                    .steps
                    .iter()
                    .map(|f| {
                        let d = f(&r, w, h);
                        fold944(
                            &r,
                            &d,
                            w,
                            h,
                            V1PoolsMode::Full,
                            V1FreeExtras::RawMomentsPlusBoundedErr,
                            false,
                        )
                    })
                    .collect();
                n_images += 1;
                for s in 0..width {
                    let v: Vec<f64> = series.iter().map(|f| f[s]).collect();
                    let range = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                        - v.iter().cloned().fold(f64::INFINITY, f64::min);
                    if range > max_range[s] {
                        max_range[s] = range;
                    }
                    if first_series[s].is_empty() {
                        first_series[s] = v.clone();
                    }
                    let up = v.windows(2).all(|p| p[1] >= p[0]);
                    let dn = v.windows(2).all(|p| p[1] <= p[0]);
                    if up && !dn {
                        mono_up[s] += 1;
                    } else if dn && !up {
                        mono_dn[s] += 1;
                    } else if !up && !dn {
                        viol[s] += 1;
                        // the biggest step against the dominant direction,
                        // normalised by the slot's own range on this image.
                        let overall_up = v[v.len() - 1] >= v[0];
                        let mut back = 0.0f64;
                        for p in v.windows(2) {
                            let d = p[1] - p[0];
                            let against = if overall_up { -d } else { d };
                            if against > back {
                                back = against;
                            }
                        }
                        if range > 0.0 {
                            worst[s] = worst[s].max(back / range);
                        }
                    }
                }
            }
        }
        writeln!(
            out,
            "ladder\t{}\tCTRL_MSE\t{n_steps}\t{ctrl_total}\t{}\t{ctrl_viol}\t-\tup\t-\t{}",
            lad.name,
            ctrl_total - ctrl_viol,
            if ctrl_first.is_empty() {
                "monotone-on-every-image".to_string()
            } else {
                ctrl_first.clone()
            }
        )
        .unwrap();
        for s in 0..width {
            // Skip structurally-dead slots (constant zero everywhere).
            if mono_up[s] == 0 && mono_dn[s] == 0 && viol[s] == 0 {
                continue;
            }
            let dir = if mono_up[s] > mono_dn[s] {
                "up"
            } else {
                "down"
            };
            let series = first_series[s]
                .iter()
                .map(|v| format!("{v:.4e}"))
                .collect::<Vec<_>>()
                .join(",");
            writeln!(
                out,
                "ladder\t{}\tf{s}\t{n_steps}\t{n_images}\t{}\t{}\t{:.4}\t{dir}\t{:.6e}\t{}",
                lad.name,
                mono_up[s] + mono_dn[s],
                viol[s],
                worst[s],
                max_range[s],
                series
            )
            .unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// (g) SCALE CONSISTENCY — a per-scale family's response at scale k on the
//     original should track its scale k-1 response on a 2x-downscaled input.
// ---------------------------------------------------------------------------

fn mode_scale(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tw\th\tfamily_stride\tslot_full\tslot_half\tv_full_scale_k\tv_half_scale_km1\trel_delta"
    )
    .unwrap();
    // The v1 basic block is (scale, channel, feature) with 39 features/channel
    // at 372/4-scale => 156/4 = 39 per scale. Compare scale s on the full image
    // against scale s-1 on the halved image.
    let per_scale_v1 = 156 / 4;
    for &(w, h) in &[(512usize, 384usize), (256, 256)] {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = blur(&r, w, h, 1, 2);
        let (rh, hw, hh) = box_downscale2(&r, w, h);
        let (dh_, _, _) = box_downscale2(&d, w, h);
        let full = fold944(&r, &d, w, h, V1PoolsMode::Full, V1FreeExtras::Off, true);
        let half = fold944(
            &rh,
            &dh_,
            hw,
            hh,
            V1PoolsMode::Full,
            V1FreeExtras::Off,
            true,
        );
        for s in 1..4usize {
            for k in 0..per_scale_v1 {
                let a = full[s * per_scale_v1 + k];
                let b = half[(s - 1) * per_scale_v1 + k];
                let denom = a.abs().max(b.abs()).max(1e-12);
                writeln!(
                    out,
                    "scale\t{w}\t{h}\t{per_scale_v1}\tf{}\tf{}\t{a:.6e}\t{b:.6e}\t{:.4}",
                    s * per_scale_v1 + k,
                    (s - 1) * per_scale_v1 + k,
                    (a - b).abs() / denom
                )
                .unwrap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (h2) INPUT DEPTH — u8 vs u16 vs f32 of the same content.
// ---------------------------------------------------------------------------

fn mode_depth(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tw\th\tpair\tslots\tcells_differing\tmax_abs_delta"
    )
    .unwrap();
    for &(w, h) in &[(200usize, 150usize), (127, 93)] {
        let r = value_noise(w, h, 0xC0FFEE);
        let d = blur(&r, w, h, 1, 1);
        let z = Zensim::new(ZensimProfile::codec_target());
        let f8 = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h))
            .expect("u8")
            .features()
            .to_vec();
        // 16-bit RGBA of the same content, via the raw-bytes source.
        let to16 = |px: &[[u8; 3]]| -> Vec<u8> {
            let mut v = Vec::with_capacity(px.len() * 8);
            for p in px {
                for &c in p {
                    let x = ((c as u16) << 8) | c as u16; // 8->16 replicate
                    v.extend_from_slice(&x.to_le_bytes());
                }
                v.extend_from_slice(&0xFFFFu16.to_le_bytes());
            }
            v
        };
        let (r16, d16) = (to16(&r), to16(&d));
        let s16 = zensim::StridedBytes::with_alpha_mode(
            &r16,
            w,
            h,
            w * 8,
            zensim::PixelFormat::Srgb16Rgba,
            zensim::AlphaMode::Opaque,
        );
        let dd16 = zensim::StridedBytes::with_alpha_mode(
            &d16,
            w,
            h,
            w * 8,
            zensim::PixelFormat::Srgb16Rgba,
            zensim::AlphaMode::Opaque,
        );
        let f16v = z.compute(&s16, &dd16).expect("u16").features().to_vec();
        let n = f8.len().min(f16v.len());
        let mut nd = 0usize;
        let mut mx = 0.0f64;
        for i in 0..n {
            if f8[i] != f16v[i] {
                nd += 1;
                mx = mx.max((f8[i] - f16v[i]).abs());
            }
        }
        writeln!(
            out,
            "depth\t{w}\t{h}\tu8_vs_u16replicate\t{n}\t{nd}\t{mx:.6e}"
        )
        .unwrap();

        // f32 linear RGBA of the same content.
        let tof32 = |px: &[[u8; 3]]| -> Vec<u8> {
            let mut v = Vec::with_capacity(px.len() * 16);
            for p in px {
                for &c in p {
                    let u = c as f32 / 255.0;
                    let lin = if u <= 0.04045 {
                        u / 12.92
                    } else {
                        ((u + 0.055) / 1.055).powf(2.4)
                    };
                    v.extend_from_slice(&lin.to_le_bytes());
                }
                v.extend_from_slice(&1.0f32.to_le_bytes());
            }
            v
        };
        let (rf, df) = (tof32(&r), tof32(&d));
        let sf = zensim::StridedBytes::with_alpha_mode(
            &rf,
            w,
            h,
            w * 16,
            zensim::PixelFormat::LinearF32Rgba,
            zensim::AlphaMode::Opaque,
        );
        let dfk = zensim::StridedBytes::with_alpha_mode(
            &df,
            w,
            h,
            w * 16,
            zensim::PixelFormat::LinearF32Rgba,
            zensim::AlphaMode::Opaque,
        );
        let ffv = z.compute(&sf, &dfk).expect("f32").features().to_vec();
        let n2 = f8.len().min(ffv.len());
        let mut nd2 = 0usize;
        let mut mx2 = 0.0f64;
        for i in 0..n2 {
            if f8[i] != ffv[i] {
                nd2 += 1;
                mx2 = mx2.max((f8[i] - ffv[i]).abs());
            }
        }
        writeln!(
            out,
            "depth\t{w}\t{h}\tu8_vs_linearf32\t{n2}\t{nd2}\t{mx2:.6e}"
        )
        .unwrap();
    }
}

// ---------------------------------------------------------------------------
// (a2) THE IDENTITY CLIFF — `Zensim::compute` FABRICATES an all-zero vector for
//      `ref == dist` (metric.rs `identical_result`) instead of computing one.
//      Measure the step between that fabricated payload and what the SAME code
//      computes for a pair that differs by ONE byte in ONE channel.
// ---------------------------------------------------------------------------

fn mode_cliff(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tw\th\troute\tslot\tfabricated\tcomputed_identical\tcomputed_one_byte_diff\tcliff_vs_fabricated"
    )
    .unwrap();
    for &(w, h) in &[(200usize, 150usize), (512, 384)] {
        let r = value_noise(w, h, 0xC0FFEE);
        let mut d1 = r.clone();
        d1[w * h / 2][1] = d1[w * h / 2][1].wrapping_add(1); // one byte, one channel

        // The product entry: identical -> fabricated; one-byte -> computed.
        let z = Zensim::new(ZensimProfile::codec_target());
        let fab = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&r, w, h))
            .expect("identical");
        let one = z
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d1, w, h))
            .expect("one-byte");
        let mut worst: Vec<(f64, usize)> = (0..fab.features().len().min(one.features().len()))
            .map(|i| ((one.features()[i] - fab.features()[i]).abs(), i))
            .collect();
        worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(delta, i) in worst.iter().take(6) {
            writeln!(
                out,
                "cliff\t{w}\t{h}\tproduct_compute_372\tf{i}\t{:.6e}\t-\t{:.6e}\t{delta:.6e}",
                fab.features()[i],
                one.features()[i]
            )
            .unwrap();
        }
        writeln!(
            out,
            "cliff\t{w}\t{h}\tproduct_compute_372\tSCORE\t{:.9}\t-\t{:.9}\t{:.6e}",
            fab.score(),
            one.score(),
            (fab.score() - one.score()).abs()
        )
        .unwrap();

        // The 944 fold, which does NOT short-circuit: identical vs one-byte.
        let fi = fold944(
            &r,
            &r,
            w,
            h,
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        );
        let fo = fold944(
            &r,
            &d1,
            w,
            h,
            V1PoolsMode::Full,
            V1FreeExtras::RawMomentsPlusBoundedErr,
            false,
        );
        let mut worst2: Vec<(f64, usize)> = (0..944).map(|i| ((fo[i] - fi[i]).abs(), i)).collect();
        worst2.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(delta, i) in worst2.iter().take(6) {
            writeln!(
                out,
                "cliff\t{w}\t{h}\tfold944_full\tf{i}\t0.000000e0\t{:.6e}\t{:.6e}\t{:.6e}",
                fi[i], fo[i], delta
            )
            .unwrap();
        }
        // And the slots whose COMPUTED identity value is furthest from the
        // fabricated zero the product entry would have returned.
        let mut far: Vec<(f64, usize)> = (0..944).map(|i| (fi[i].abs(), i)).collect();
        far.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        for &(v, i) in far.iter().take(6) {
            writeln!(
                out,
                "cliff\t{w}\t{h}\tfold944_identity_vs_fabricated\tf{i}\t0.000000e0\t{:.6e}\t-\t{v:.6e}",
                fi[i]
            )
            .unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// (d) FULL-VECTOR DUMP — for cross-tier / cross-build diffing. Emits every
//     slot's `to_bits()` so a comparison is exact, not eyeballed.
// ---------------------------------------------------------------------------

fn mode_dump(out: &mut dyn Write) {
    writeln!(out, "mode\tw\th\troute\tslot\tbits\tvalue").unwrap();
    for &(w, h) in CELLS {
        if w < 64 || h < 64 {
            continue;
        }
        let r = value_noise(w, h, 0xC0FFEE);
        let d = blur(&r, w, h, 1, 1);
        for (route, pools, free, v1o) in [
            (
                "fold944_full",
                V1PoolsMode::Full,
                V1FreeExtras::RawMomentsPlusBoundedErr,
                false,
            ),
            ("fold_v1only", V1PoolsMode::Full, V1FreeExtras::Off, true),
        ] {
            let f = fold944(&r, &d, w, h, pools, free, v1o);
            for (i, v) in f.iter().enumerate() {
                writeln!(
                    out,
                    "dump\t{w}\t{h}\t{route}\tf{i}\t{:016x}\t{v:.17e}",
                    v.to_bits()
                )
                .unwrap();
            }
        }
        let v1 = v1_372(&r, &d, w, h);
        for (i, v) in v1.iter().enumerate() {
            writeln!(
                out,
                "dump\t{w}\t{h}\tbuffered_v1_372\tf{i}\t{:016x}\t{v:.17e}",
                v.to_bits()
            )
            .unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// (j) WIDTH MATRIX — which toggle shape yields which emitted width, and which
//     free-set slots are reachable in each.
// ---------------------------------------------------------------------------

fn mode_widths(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tv1_only\tappend\tappend2\tpools\tfree\twidth\tnonzero_above_720\tnonzero_total"
    )
    .unwrap();
    let (w, h) = (200usize, 150usize);
    let r = value_noise(w, h, 0xC0FFEE);
    let d = blur(&r, w, h, 1, 1);
    for v1_only in [true, false] {
        for append in [false, true] {
            for append2 in [false, true] {
                if append2 && !append {
                    continue; // append2_block requires append_block (asserted in the walk)
                }
                for (pn, pools) in [("Peaks", V1PoolsMode::Peaks), ("Full", V1PoolsMode::Full)] {
                    for (fname, free) in [
                        ("Off", V1FreeExtras::Off),
                        ("RawMoments", V1FreeExtras::RawMoments),
                        ("PlusBoundedErr", V1FreeExtras::RawMomentsPlusBoundedErr),
                    ] {
                        let toggles = V2NewFeatureToggles {
                            v1_pools: pools,
                            append_block: append,
                            append2_block: append2,
                            v1_only,
                            free_extras: free,
                            ..Default::default()
                        };
                        let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                        let mut scratch = V2Scratch::new();
                        let f = z
                            .compute_folded720_features_streaming(
                                &RgbSlice::new(&r, w, h),
                                &RgbSlice::new(&d, w, h),
                                toggles,
                                &mut scratch,
                            )
                            .expect("walk")
                            .features()
                            .to_vec();
                        let above = f.iter().skip(720).filter(|&&v| v != 0.0).count();
                        let tot = f.iter().filter(|&&v| v != 0.0).count();
                        writeln!(
                            out,
                            "widths\t{v1_only}\t{append}\t{append2}\t{pn}\t{fname}\t{}\t{above}\t{tot}",
                            f.len()
                        )
                        .unwrap();
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// (k) SERVABILITY — can `Zensim::compute` (THE production entry) serve each
//     shipped profile, on real pixels? Bake-level servability is measured by
//     `zensim/examples/serve_custom_bake.rs` (the fastclass2 campaign's probe);
//     this covers the PROFILE enum, which that probe cannot reach.
// ---------------------------------------------------------------------------

fn mode_profiles(out: &mut dyn Write) {
    writeln!(
        out,
        "mode\tprofile\tw\th\tstatus\tscore\tidentity_score\tn_features\tdetail"
    )
    .unwrap();
    let (w, h) = (200usize, 150usize);
    let r = value_noise(w, h, 0xC0FFEE);
    let d = blur(&r, w, h, 1, 1);
    let probe = |name: &str, p: ZensimProfile, out: &mut dyn Write| {
        let z = Zensim::new(p);
        let rs = RgbSlice::new(&r, w, h);
        let ds = RgbSlice::new(&d, w, h);
        let ident = z
            .compute(&rs, &rs)
            .map(|x| format!("{:.6}", x.score()))
            .unwrap_or_else(|e| format!("REFUSED({e:?})"));
        match z.compute(&rs, &ds) {
            Ok(res) => writeln!(
                out,
                "profiles\t{name}\t{w}\t{h}\tSERVED\t{:.6}\t{ident}\t{}\t-",
                res.score(),
                res.features().len()
            )
            .unwrap(),
            Err(e) => writeln!(
                out,
                "profiles\t{name}\t{w}\t{h}\tREFUSED\t-\t{ident}\t-\t{e:?}"
            )
            .unwrap(),
        }
    };
    probe("B", ZensimProfile::B, out);
    probe("BHdr", ZensimProfile::BHdr, out);
    probe("codec_target()", ZensimProfile::codec_target(), out);
    #[cfg(feature = "deprecated-profiles")]
    {
        #[allow(deprecated)]
        probe("A", ZensimProfile::A, out);
    }
    probe("PreviewV0_1", ZensimProfile::PreviewV0_1, out);
    probe("PreviewV0_2", ZensimProfile::PreviewV0_2, out);
    #[cfg(feature = "candidate-profiles")]
    {
        probe("C", ZensimProfile::C, out);
        probe("CHdr", ZensimProfile::CHdr, out);
        probe("D", ZensimProfile::D, out);
    }
    #[cfg(any(feature = "training", test))]
    probe("LegacyLinearV0_2", ZensimProfile::LegacyLinearV0_2, out);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("all");
    let path = args.get(2).map(|s| s.as_str());
    // Tier cap: run the WHOLE process on the v3/AVX2 ceiling so a second
    // invocation can be diffed against the native (v4x/AVX-512) one. Cargo
    // features are per-build, so per-arm capping inside one process is not
    // possible — the honest form is two runs (`ssim2_speed_bar` uses the same
    // shape). Refuses to continue if the cap was requested and not applied,
    // rather than silently reporting a native run as capped.
    #[cfg(target_arch = "x86_64")]
    if std::env::var("ZEN_FIP_CAP_V3").as_deref() == Ok("1") {
        match archmage::X64V4Token::dangerously_disable_token_process_wide(true) {
            Ok(()) => eprintln!("# ZEN_FIP_CAP_V3=1: X64V4Token disabled, ceiling = v3 (AVX2+FMA)"),
            Err(e) => {
                eprintln!("# ZEN_FIP_CAP_V3=1 requested but could not disable X64V4Token: {e}");
                std::process::exit(1);
            }
        }
    }
    let mut out = out_writer(path);
    let run = |m: &str, out: &mut dyn Write| match m {
        "identity" => mode_identity(out),
        "determinism" => mode_determinism(out),
        "engineparity" => mode_engine_parity(out),
        "degenerate" => mode_degenerate(out),
        "ladder" => mode_ladder(out),
        "scale" => mode_scale(out),
        "depth" => mode_depth(out),
        "cliff" => mode_cliff(out),
        "dump" => mode_dump(out),
        "widths" => mode_widths(out),
        "profiles" => mode_profiles(out),
        other => panic!("unknown mode {other}"),
    };
    if mode == "all" {
        for m in [
            "identity",
            "determinism",
            "engineparity",
            "degenerate",
            "scale",
            "depth",
            "cliff",
            "ladder",
        ] {
            eprintln!("== {m}");
            run(m, &mut out);
        }
    } else {
        run(mode, &mut out);
    }
}
