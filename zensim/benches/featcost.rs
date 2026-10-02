// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! FEATCOST: extraction cost each Rev4 feature family adds over the 228-column
//! core (basic f0..155 + peaks f156..227), measured as the PAIRED increment
//! `core+F - core` of whole `research::extract` walks, interleaved in-process
//! by zenbench. Measurement only: no feature arithmetic is touched.
//!
//! Run (see benchmarks/featcost_2026-10-01.meta for the exact command):
//!   `ZENSIM_FORMULA_REV=4 ZENSIM_MAX_TIER=v3 RAYON_NUM_THREADS=1 taskset -c 8-15 \
//!    ~/work/zen/scripts/run-heavy --mem 16G --jobs 4 -- \
//!    cargo bench -p zensim --bench featcost --features custom-profiles,training`
//!
//! Variants: `legacy:*` (v1 pools at width 372), `alone:F` (core + F's slots at the
//! layout width ending at F, i.e. what the planner runs for F INCLUDING its structural
//! prefix), `ladder:F` (core + every chain family up to F), `step:F` (derived
//! ladder:F - ladder:prev = F's marginal cost given its prefix).
//! Env: `FEATCOST_SIZES=64,256` narrows sizes; `FEATCOST_VARIANTS=a,b` narrows
//! variants (core is always kept); `FEATCOST_PROBE=1` prints each variant's
//! EFFECTIVE compute-token set (what the planner actually runs) and exits;
//! `FEATCOST_OUT=<prefix>` writes `<prefix>.raw.tsv` + `<prefix>.fit.tsv`.
//!
//! Inputs: two natural photos from the imazen-26 codec-corpus (nature +
//! people), centre-cropped square and Lanczos-resized to each size (never
//! upscaled past the source), with a JPEG q50 round-trip of the same crop as
//! the distorted image so each pair is non-identical.
#[path = "../examples/support/zen_io.rs"]
mod zen_io;

use std::fmt::Write as _;
use std::path::Path;
use zensim::RgbSlice;
use zensim::feature_set_id::ComputeToken as T;
use zensim::feature_set_id::SlotSet;
use zensim::research::{self, Request};

const CORPUS: &str = "/home/lilith/work/codec-corpus/imazen-26";
const PHOTOS: &[(&str, &str)] = &[
    (
        "nature",
        "1400-lilith-nature/1407_nature_rocky-coastline-ocean_20210608-132404-2_8160x6120.jpg",
    ),
    (
        "people",
        "2000-unsplash-people/2014_people_by-jing-chen-sld5c8su-ue-unsplash_6240x4160.jpg",
    ),
];
const SIZES: &[usize] = &[64, 256, 1024, 4096];

/// The walk's nested block chain, in slot order: a block at slot position p is
/// only reachable through a layout that reaches every block below it
/// (`LayoutBlocks::for_width`, feature_plan.rs; asserts in feature_v2.rs), so a
/// family's cost "alone" is its cost with its structural prefix.
const CHAIN: &[T] = &[
    T::V2,
    T::Append,
    T::Append2,
    T::Csfw,
    T::Dvifm,
    T::Gridblk,
    T::Ringbasis,
    T::Tailhist,
    T::Arttype,
    T::Gmsbank,
    T::Mapdev,
    T::Z1max,
    T::Gmsnative,
    T::Dvifmgate,
];
/// Pairs measured jointly (shared kernels) in addition to each alone.
const PAIRS: &[(T, T)] = &[
    (T::Dvifm, T::Dvifmgate),
    (T::Gmsbank, T::Gmsnative),
    (T::Mapdev, T::Z1max),
];

struct Variant {
    label: String,
    tokens: Vec<T>,
    width: usize,
}

fn name(t: T) -> String {
    t.to_string()
}

fn end_of(t: T) -> usize {
    research::family_slots(t).iter_slots().max().unwrap() + 1
}

fn variants() -> Vec<Variant> {
    let mut v = Vec::new();
    // legacy v1 pools (layout width 372: nothing above f371 is reached)
    for toks in [&[T::Masked][..], &[T::Iw], &[T::Masked, T::Iw]] {
        let l: Vec<String> = toks.iter().map(|t| name(*t)).collect();
        v.push(Variant {
            label: format!("legacy:{}", l.join("+")),
            tokens: toks.to_vec(),
            width: 372,
        });
    }
    // standalone-as-planned: core + F's slots at the layout width ending at F
    for &f in CHAIN {
        v.push(Variant {
            label: format!("alone:{}", name(f)),
            tokens: vec![f],
            width: end_of(f),
        });
    }
    for &(a, b) in PAIRS {
        v.push(Variant {
            label: format!("alone:{}+{}", name(a), name(b)),
            tokens: vec![a, b],
            width: end_of(a).max(end_of(b)),
        });
    }
    // ladder: core + every chain family up to and including F, at F's width
    for (i, &f) in CHAIN.iter().enumerate() {
        v.push(Variant {
            label: format!("ladder:{}", name(f)),
            tokens: CHAIN[..=i].to_vec(),
            width: end_of(f),
        });
    }
    v
}

/// Mirror of `zensim_validate::tier_cap::apply_from_env` (that crate depends on
/// zensim, so a zensim bench cannot call it): disable the AVX-512 tokens.
fn apply_tier_cap() {
    match std::env::var("ZENSIM_MAX_TIER").as_deref() {
        Err(_) => panic!("set ZENSIM_MAX_TIER=v3 (the fleet tier) for this bench"),
        Ok("v3") => {
            use archmage::{X64V4Token, X64V4xToken};
            X64V4xToken::dangerously_disable_token_process_wide(true).expect("cap v4x");
            X64V4Token::dangerously_disable_token_process_wide(true).expect("cap v4");
            eprintln!("tier cap: AVX-512 tokens disabled (ZENSIM_MAX_TIER=v3)");
        }
        Ok(o) => panic!("ZENSIM_MAX_TIER={o:?}: only v3"),
    }
}

fn core_slots() -> SlotSet {
    SlotSet::from_ranges([(0, 228)])
}

fn request(tokens: &[T], width: usize) -> Request {
    let mut want = core_slots();
    for &t in tokens {
        want = want.union(&research::family_slots(t));
    }
    Request::for_slots(want, width)
}

/// Square centre-crop to `min(w,h)` then Lanczos to `size` (resize is a no-op
/// when the crop already equals `size`).
fn prepared(photo: &str, size: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let (px, w, h) = zen_io::decode_rgb8(&Path::new(CORPUS).join(photo));
    let side = w.min(h);
    let (x0, y0) = ((w - side) / 2, (h - side) / 2);
    let mut crop = Vec::with_capacity(side * side);
    for y in 0..side {
        crop.extend_from_slice(&px[(y0 + y) * w + x0..(y0 + y) * w + x0 + side]);
    }
    assert!(size <= side, "{photo}: would upscale {side} -> {size}");
    let r = zen_io::resize_rgb8(&crop, side, side, size, size);
    let jpg = zen_io::encode_jpeg_q(&r, size, size, 50);
    let tmp = Path::new(&std::env::var("HOME").unwrap()).join("tmp/featcost");
    std::fs::create_dir_all(&tmp).unwrap();
    let f = tmp.join(format!("{size}_{}.jpg", jpg.len()));
    std::fs::write(&f, &jpg).unwrap();
    let (d, dw, dh) = zen_io::decode_rgb8(&f);
    assert_eq!((dw, dh), (size, size));
    assert_ne!(r, d);
    (r, d)
}

fn probe() {
    let (r, d) = prepared(PHOTOS[0].1, 256);
    let (s, t) = (RgbSlice::new(&r, 256, 256), RgbSlice::new(&d, 256, 256));
    let core = Variant {
        label: "core".into(),
        tokens: vec![],
        width: 228,
    };
    for v in std::iter::once(core).chain(variants()) {
        let e = research::extract(&request(&v.tokens, v.width), &s, &t).expect("extract");
        let id = e.feature_set_id().map(|i| i.to_string());
        println!(
            "{}\t{}\t{}",
            v.label,
            v.width,
            id.unwrap_or_else(|| "?".into())
        );
    }
}

fn ols(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let b = sxy / sxx;
    (my - b * mx, b)
}

fn main() {
    apply_tier_cap();
    assert_eq!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref(),
        Ok("4"),
        "set ZENSIM_FORMULA_REV=4 (canonical Rev4 arithmetic)"
    );
    if std::env::var("FEATCOST_PROBE").is_ok() {
        return probe();
    }
    let sizes: Vec<usize> = std::env::var("FEATCOST_SIZES")
        .map(|s| s.split(',').map(|v| v.parse().unwrap()).collect())
        .unwrap_or_else(|_| SIZES.to_vec());
    let only: Option<Vec<String>> = std::env::var("FEATCOST_VARIANTS")
        .ok()
        .map(|s| s.split(',').map(str::to_string).collect());
    let variants: Vec<Variant> = variants()
        .into_iter()
        .filter(|v| {
            only.as_ref()
                .is_none_or(|o| o.iter().any(|x| v.label.contains(x.as_str())))
        })
        .collect();

    // rows: (photo, size, variant, mean_ns, median_ns, n, diff_mean_ns, ci_lo, ci_hi)
    let mut raw = String::from(
        "photo\tsize\tvariant\tmean_ns\tmedian_ns\tn_rounds\tinc_mean_ns\tinc_ci_lo_ns\tinc_ci_hi_ns\n",
    );
    let mut table: Vec<(String, usize, String, f64, f64)> = Vec::new(); // photo,size,variant,mean,inc
    let photos: Vec<&(&str, &str)> = PHOTOS
        .iter()
        .filter(|(n, _)| {
            std::env::var("FEATCOST_PHOTOS").is_ok_and(|v| v.split(',').any(|x| x == *n))
                || std::env::var("FEATCOST_PHOTOS").is_err()
        })
        .collect();
    for &&(pname, ppath) in &photos {
        for &size in &sizes {
            let (r, d) = prepared(ppath, size);
            let (r, d): (&'static [[u8; 3]], &'static [[u8; 3]]) = (
                Box::leak(r.into_boxed_slice()),
                Box::leak(d.into_boxed_slice()),
            );
            let s = RgbSlice::new(r, size, size);
            let t = RgbSlice::new(d, size, size);
            let big = size >= 1024;
            let res = zenbench::run(|suite| {
                suite.compare(format!("featcost_{pname}_{size}"), |g| {
                    g.config()
                        .min_rounds(if size >= 4096 {
                            5
                        } else if big {
                            10
                        } else {
                            20
                        })
                        .max_rounds(if size >= 4096 {
                            8
                        } else if big {
                            20
                        } else {
                            80
                        })
                        .max_wall_time(std::time::Duration::from_secs(if size >= 4096 {
                            3600
                        } else if big {
                            600
                        } else {
                            90
                        }));
                    g.baseline("core");
                    let core = Variant {
                        label: "core".into(),
                        tokens: vec![],
                        width: 228,
                    };
                    for v in std::iter::once(&core).chain(variants.iter()) {
                        let req = request(&v.tokens, v.width);
                        g.bench(v.label.clone(), move |b| {
                            b.iter(|| {
                                zenbench::black_box(
                                    research::extract(&req, &s, &t).expect("extract"),
                                )
                                .values()
                                .len()
                            })
                        });
                    }
                });
            });
            let cmp = &res.comparisons[0];
            for bm in &cmp.benchmarks {
                let ana = cmp
                    .analyses
                    .iter()
                    .find(|(b, c, _)| b == "core" && *c == bm.name);
                let (inc, lo, hi) = ana
                    .map(|(_, _, a)| (a.diff.mean, a.ci_lower, a.ci_upper))
                    .unwrap_or((0.0, 0.0, 0.0));
                writeln!(
                    raw,
                    "{pname}\t{size}\t{}\t{:.0}\t{:.0}\t{}\t{inc:.0}\t{lo:.0}\t{hi:.0}",
                    bm.name, bm.summary.mean, bm.summary.median, bm.summary.n
                )
                .unwrap();
                table.push((pname.into(), size, bm.name.clone(), bm.summary.mean, inc));
            }
        }
    }

    // marginal ladder step of F over its predecessor rung (paired increments
    // over core subtracted rung by rung; no CI, derived from the rung means).
    let mut steps = Vec::new();
    for &&(pname, _) in &photos {
        for &size in &sizes {
            let inc = |l: &str| {
                table
                    .iter()
                    .find(|r| r.0 == pname && r.1 == size && r.2 == l)
                    .map(|r| r.4)
            };
            let mut prev = 0.0;
            for &f in CHAIN {
                if let Some(cur) = inc(&format!("ladder:{}", name(f))) {
                    steps.push((
                        pname.to_string(),
                        size,
                        format!("step:{}", name(f)),
                        0.0,
                        cur - prev,
                    ));
                    prev = cur;
                }
            }
        }
    }
    for r in &steps {
        writeln!(raw, "{}\t{}\t{}\t\t\t\t{:.0}\t\t", r.0, r.1, r.2, r.4).unwrap();
    }
    table.extend(steps);

    // time = alpha + beta * pixels, per (photo, variant); core fit on its own
    // absolute time, every other variant on its paired INCREMENT over core.
    let mut fit = String::from("photo\tvariant\tkind\talpha_ms\tbeta_ns_per_px\tn_sizes\tsizes\n");
    for &&(pname, _) in &photos {
        let mut names: Vec<String> = table
            .iter()
            .filter(|r| r.0 == pname)
            .map(|r| r.2.clone())
            .collect();
        names.dedup();
        names.sort();
        names.dedup();
        for v in names {
            let rows: Vec<_> = table.iter().filter(|r| r.0 == pname && r.2 == v).collect();
            if rows.len() < 2 {
                continue;
            }
            let xs: Vec<f64> = rows.iter().map(|r| (r.1 * r.1) as f64).collect();
            let (kind, ys): (&str, Vec<f64>) = if v == "core" {
                ("absolute", rows.iter().map(|r| r.3).collect())
            } else {
                ("increment", rows.iter().map(|r| r.4).collect())
            };
            let (a, b) = ols(&xs, &ys);
            let szs: Vec<String> = rows.iter().map(|r| r.1.to_string()).collect();
            writeln!(
                fit,
                "{pname}\t{v}\t{kind}\t{:.4}\t{:.4}\t{}\t{}",
                a / 1e6,
                b,
                rows.len(),
                szs.join(",")
            )
            .unwrap();
        }
    }
    print!("{raw}\n{fit}");
    if let Ok(prefix) = std::env::var("FEATCOST_OUT") {
        std::fs::write(format!("{prefix}.raw.tsv"), &raw).unwrap();
        std::fs::write(format!("{prefix}.fit.tsv"), &fit).unwrap();
    }
}
