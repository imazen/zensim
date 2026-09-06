//! **F5 route-parity measurement on REAL pixels** — the gate the shipped one
//! could not be (`docs/PLAN_FEATURE_REV2_2026-09-05.md` R3).
//!
//! The free raw-moment tranche is written by two different accumulations of
//! the same definition:
//!
//! * the **append** kernel (`feature_v2.rs`), which reduces its f32 lane
//!   accumulator to f64 **per row** and whose scalar tail goes straight to
//!   f64 per pixel; and
//! * the **free** walk (`fused.rs`), reached when a bake asks for the cheap
//!   `v1_only` plan, which under revision 1 accumulates in f32 lanes across a
//!   whole 32-row band before upgrading.
//!
//! `global_stats_from_raw_moments` then computes `Σs²/n − (Σs/n)²`, whose
//! relative error is the accumulated error amplified by `mean²/var` — which
//! is unbounded as a region flattens.
//!
//! **Why the shipped gate missed it.**
//! `free_extras_match_the_944_append_block` compares the two routes on
//! SYNTHETIC images, which do not produce flat-but-not-constant regions, so
//! the amplifier stays near 1. `fused.rs`'s own comment concluded the band
//! batching "costs negligible precision: worst |Δ| … 5.35e-6". On real pairs
//! the defect audit measures **2,607 of 28,601 cells (9.12 %) past 2e-5,
//! worst 3.63e-3**. This example is the real-pixel instrument, so the claim
//! is checked against the population that falsifies it.
//!
//! ```text
//! cargo run --release -p zensim \
//!   --features training,feature-regime-v2,custom-profiles,classification \
//!   --example f5_route_parity -- <dir-of-pngs> [n_images] [out.tsv]
//! ZENSIM_FORMULA_REV=2 …   # the same run under revision 2
//! ```

use std::io::Write;
use zensim::feature_v2::{V1FreeExtras, V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::{RgbSlice, Zensim, ZensimProfile};

/// The parity bar the audit used, and the one phase 2b must clear on 100 % of
/// cells rather than on a synthetic majority.
const BAR: f64 = 2e-5;

fn walk(r: &[[u8; 3]], d: &[[u8; 3]], w: usize, h: usize, v1_only: bool) -> Vec<f64> {
    let toggles = V2NewFeatureToggles {
        v1_pools: V1PoolsMode::Full,
        append_block: true,
        append2_block: true,
        v1_only,
        // The append kernel OWNS these slots when it runs; the free walk
        // writes them only on the `v1_only` plan. Asking for both on both
        // arms keeps the request identical and lets the plan decide.
        free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
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

/// A mild, deterministic degradation. The point of this instrument is REAL
/// SOURCE STATISTICS — `mean²/var` per plane is a property of the reference
/// content, which is what the synthetic gate lacked — so the distortion only
/// has to be plausible, not codec-accurate.
fn degrade(px: &[[u8; 3]], w: usize, h: usize) -> Vec<[u8; 3]> {
    let mut out = px.to_vec();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            for c in 0..3 {
                let a = px[y * w + x - 1][c] as u16
                    + px[y * w + x + 1][c] as u16
                    + px[(y - 1) * w + x][c] as u16
                    + px[(y + 1) * w + x][c] as u16;
                // blur, then a coarse quantization step — the shape a codec
                // leaves behind on flat regions.
                let v = (a / 4) as u8;
                out[y * w + x][c] = (v / 8) * 8;
            }
        }
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .expect("usage: f5_route_parity <dir> [n] [out.tsv]");
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(40);
    let mut out: Box<dyn Write> = match args.get(3) {
        Some(p) => Box::new(std::fs::File::create(p).expect("create out")),
        None => Box::new(std::io::stdout()),
    };

    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .expect("read dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "png"))
        .collect();
    paths.sort();
    paths.truncate(n);
    assert!(!paths.is_empty(), "no PNGs under {dir}");

    writeln!(
        out,
        "# revision: {}",
        std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "default".into())
    )
    .unwrap();
    writeln!(out, "image\tw\th\tslot\tappend\tfree\tabs_delta\trel_delta").unwrap();

    let (mut cells, mut over, mut worst_abs, mut worst_rel) = (0usize, 0usize, 0f64, 0f64);
    let mut slots_seen: std::collections::BTreeSet<usize> = Default::default();
    let mut images = 0usize;
    let (mut worst_slot, mut worst_img) = (0usize, String::new());
    for p in &paths {
        let Ok(img) = image::open(p) else { continue };
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        if w < 64 || h < 64 {
            continue;
        }
        let r: Vec<[u8; 3]> = rgb.pixels().map(|q| [q.0[0], q.0[1], q.0[2]]).collect();
        let d = degrade(&r, w, h);
        let app = walk(&r, &d, w, h, false);
        let fre = walk(&r, &d, w, h, true);
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        images += 1;

        // Compare only the slots BOTH routes write.
        //
        // The append arm computes the whole append block; the free arm's
        // `v1_only` plan computes only the free tranche and leaves the rest a
        // structural zero. Counting "computed vs structural zero" as a parity
        // failure measures the PLAN, not the accumulation — a first pass of
        // this instrument did exactly that and read 61 % over the bar with a
        // relative delta of 5.96e8, which is the signature of dividing by a
        // slot that was never meant to be written.
        for i in 720..923.min(app.len().min(fre.len())) {
            // THE RAW-MOMENT TRANCHE ONLY: append-block locals 13/14/15
            // (`GLOBAL_DMEAN`, `GLOBAL_CGAIN`, `GLOBAL_CLOSS`) over the
            // computed append cells. A first pass compared every slot both
            // arms happened to write, which pulled in ~8 slots outside the
            // tranche that the two PLANS legitimately compute by different
            // routes — measuring the plan, not the accumulation. The tell was
            // `worst_rel` reading 8.523328e2 identically across three
            // different numerical treatments: a gap that does not move when
            // the arithmetic changes is not an arithmetic gap.
            let local = (i - 720) % 17;
            if !(13..=15).contains(&local) {
                continue;
            }
            if app[i] == 0.0 || fre[i] == 0.0 {
                continue;
            }
            slots_seen.insert(i);
            cells += 1;
            let ad = (app[i] - fre[i]).abs();
            let rd = ad / app[i].abs().max(1e-12);
            if ad > BAR {
                over += 1;
            }
            {
                writeln!(
                    out,
                    "{name}\t{w}\t{h}\tf{i}\t{:.17e}\t{:.17e}\t{ad:.6e}\t{rd:.6e}",
                    app[i], fre[i]
                )
                .unwrap();
            }
            if ad > worst_abs {
                worst_abs = ad;
                worst_slot = i;
                worst_img = name.clone();
            }
            worst_rel = worst_rel.max(rd);
        }
    }
    let pct = 100.0 * over as f64 / cells.max(1) as f64;
    writeln!(out, "# SUMMARY images={images} slots={} cells={cells} over_{BAR:.0e}={over} ({pct:.4} %) worst_abs={worst_abs:.6e} at f{worst_slot} in {worst_img} worst_rel={worst_rel:.6e}", slots_seen.len()).unwrap();
    writeln!(out, "# SLOTS {slots_seen:?}").unwrap();
    eprintln!(
        "images={images} slots={} cells={cells} over={over} ({pct:.4} %) worst_abs={worst_abs:.6e} (f{worst_slot}, {worst_img}) worst_rel={worst_rel:.6e}",
        slots_seen.len()
    );
}
