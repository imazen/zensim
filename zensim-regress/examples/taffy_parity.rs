//! Backend-parity harness: render a corpus of representative layouts with
//! BOTH the native solver ([`layout::render_with_config`]) and the taffy
//! bridge ([`layout::taffy_backend::render_via_taffy`]), then quantify the
//! geometry divergence.
//!
//! Because both paths use the identical paint primitives / font, every
//! pixel difference is attributable to the layout solver alone.
//!
//! ```text
//! cargo run -p zensim-regress --features taffy-backend \
//!     --release --example taffy_parity -- /mnt/v/output/zensim/taffy-eval
//! ```
//!
//! Emits per-scene native/taffy/diff PNGs, a master montage, and a
//! `REPORT.md` with a divergence table + taffy coverage.

// Harness scaffolding: uniform `.into()` on scene bodies keeps the corpus
// table readable even where the expression is already a `Node`, and the
// `Vec::new()` + `push` shape mirrors how scenes accrete.
#![allow(clippy::useless_conversion, clippy::vec_init_then_push)]

use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;

use zensim_regress::Bitmap;
use zensim_regress::layout::taffy_backend;
use zensim_regress::layout::*;

fn solid(w: u32, h: u32, c: Color) -> Bitmap {
    Bitmap::from_pixel(w, h, c)
}

/// A labeled color swatch — exercises image + text together.
fn swatch(label: &str, w: u32, h: u32, c: Color) -> Node {
    image(solid(w, h, c)).label(label)
}

/// The corpus. Each scene is `(name, max_w, tree)`.
fn corpus() -> Vec<(&'static str, u32, Node)> {
    let mut v: Vec<(&'static str, u32, Node)> = Vec::new();

    v.push(("fixed_leaf", 200, fill(hex("#ff6f3c")).size(160, 80).into()));

    v.push((
        "padding",
        240,
        line("padded", WHITE).padding(20).background(hex("#243949")),
    ));

    v.push((
        "row_gap",
        400,
        row()
            .gap(8)
            .child(swatch("A", 60, 60, hex("#ef4444")))
            .child(swatch("B", 60, 60, hex("#10b981")))
            .child(swatch("C", 60, 60, hex("#3b82f6")))
            .into(),
    ));

    v.push((
        "column_center",
        300,
        column()
            .gap(8)
            .align_items(CrossAlign::Center)
            .child(swatch("title", 120, 40, hex("#3b82f6")))
            .child(swatch("subtitle", 80, 30, hex("#6b7280")))
            .into(),
    ));

    for (name, j) in [
        ("justify_between", MainAlign::SpaceBetween),
        ("justify_around", MainAlign::SpaceAround),
        ("justify_evenly", MainAlign::SpaceEvenly),
        ("justify_center", MainAlign::Center),
        ("justify_end", MainAlign::End),
    ] {
        v.push((
            name,
            360,
            row()
                .justify(j)
                .child(swatch("x", 50, 40, hex("#ef4444")))
                .child(swatch("y", 50, 40, hex("#10b981")))
                .child(swatch("z", 50, 40, hex("#3b82f6")))
                .size(360, 40)
                .into(),
        ));
    }

    v.push((
        "grow_1_2",
        400,
        row()
            .gap(6)
            .child(swatch("hug", 60, 40, hex("#6b7280")))
            .child(
                swatch("grow1", 10, 40, hex("#10b981"))
                    .grow(1)
                    .fill_height(),
            )
            .child(
                swatch("grow2", 10, 40, hex("#3b82f6"))
                    .grow(2)
                    .fill_height(),
            )
            .size(400, 40)
            .into(),
    ));

    v.push((
        "grid_uniform_3x2",
        360,
        grid()
            .cols(3)
            .equal_rows(2)
            .gap(4)
            .cell(0, 0, swatch("0,0", 60, 40, hex("#ef4444")))
            .cell(1, 0, swatch("1,0", 60, 40, hex("#10b981")))
            .cell(2, 0, swatch("2,0", 60, 40, hex("#3b82f6")))
            .cell(0, 1, swatch("0,1", 60, 40, hex("#f59e0b")))
            .cell(1, 1, swatch("1,1", 60, 40, hex("#a855f7")))
            .cell(2, 1, swatch("2,1", 60, 40, hex("#ec4899")))
            .size(360, 120)
            .into(),
    ));

    v.push((
        "grid_tracks_px_fr",
        360,
        grid()
            .columns([Track::Px(80), Track::Fr(1), Track::Fr(2)])
            .equal_rows(1)
            .gap(4)
            .cell(0, 0, swatch("80px", 60, 40, hex("#ef4444")))
            .cell(1, 0, swatch("1fr", 60, 40, hex("#10b981")))
            .cell(2, 0, swatch("2fr", 60, 40, hex("#3b82f6")))
            .size(360, 40)
            .into(),
    ));

    v.push((
        "grid_named_areas",
        360,
        grid()
            .areas(&[
                "header header header",
                "side   main   main",
                "side   foot   foot",
            ])
            .col_widths([Track::Px(100), Track::Fr(1), Track::Fr(1)])
            .row_heights([Track::Px(40), Track::Fr(1), Track::Px(30)])
            .gap(4)
            .place("header", swatch("HEADER", 60, 30, hex("#3b82f6")))
            .place("side", swatch("SIDE", 60, 30, hex("#10b981")))
            .place("main", swatch("MAIN", 60, 30, hex("#f59e0b")))
            .place("foot", swatch("FOOT", 60, 30, hex("#a855f7")))
            .size(360, 200)
            .into(),
    ));

    v.push((
        "aspect_16_9",
        300,
        row()
            .gap(8)
            .child(
                swatch("16:9", 10, 10, hex("#ef4444"))
                    .aspect_ratio(16, 9)
                    .fill_width(),
            )
            .size(300, 120)
            .into(),
    ));

    v.push((
        "percent_split",
        400,
        row()
            .gap(4)
            .child(
                swatch("25%", 10, 40, hex("#ef4444"))
                    .width_percent(0.25)
                    .fill_height(),
            )
            .child(swatch("rest", 10, 40, hex("#3b82f6")).grow(1).fill_height())
            .size(400, 40)
            .into(),
    ));

    v.push((
        "layers_overlay",
        240,
        layers()
            .child(fill(hex("#1f2937")))
            .child(
                line("on top", WHITE)
                    .align(HAlign::Right, VAlign::Bottom)
                    .padding(6),
            )
            .size(240, 100)
            .into(),
    ));

    // ── Text-free scenes: isolate the core solver from our fitted-text
    // /label measure convention. Pure fill boxes only.
    let box_ = |w: u32, h: u32, c: Color| -> Node { fill(c).size(w, h) };

    for (name, j) in [
        ("pure_justify_between", MainAlign::SpaceBetween),
        ("pure_justify_center", MainAlign::Center),
        ("pure_justify_evenly", MainAlign::SpaceEvenly),
    ] {
        v.push((
            name,
            360,
            row()
                .justify(j)
                .child(box_(50, 40, hex("#ef4444")))
                .child(box_(50, 40, hex("#10b981")))
                .child(box_(50, 40, hex("#3b82f6")))
                .size(360, 40)
                .into(),
        ));
    }

    v.push((
        "pure_grow_1_2",
        400,
        row()
            .gap(6)
            .child(box_(60, 40, hex("#6b7280")))
            .child(box_(10, 40, hex("#10b981")).grow(1).fill_height())
            .child(box_(10, 40, hex("#3b82f6")).grow(2).fill_height())
            .size(400, 40)
            .into(),
    ));

    v.push((
        "pure_grid_px_fr",
        360,
        grid()
            .columns([Track::Px(80), Track::Fr(1), Track::Fr(2)])
            .equal_rows(1)
            .gap(4)
            .cell(
                0,
                0,
                box_(60, 40, hex("#ef4444")).fill_width().fill_height(),
            )
            .cell(
                1,
                0,
                box_(60, 40, hex("#10b981")).fill_width().fill_height(),
            )
            .cell(
                2,
                0,
                box_(60, 40, hex("#3b82f6")).fill_width().fill_height(),
            )
            .size(360, 40)
            .into(),
    ));

    v.push((
        "pure_percent_split",
        400,
        row()
            .gap(4)
            .child(
                box_(10, 40, hex("#ef4444"))
                    .width_percent(0.25)
                    .fill_height(),
            )
            .child(box_(10, 40, hex("#3b82f6")).grow(1).fill_height())
            .size(400, 40)
            .into(),
    ));

    // A realistic 2×2 montage — the actual shipping shape.
    v.push((
        "montage_2x2",
        360,
        grid()
            .cols(2)
            .equal_rows(2)
            .gap(6)
            .padding(6)
            .cell(0, 0, swatch("EXPECTED", 120, 90, hex("#334155")))
            .cell(1, 0, swatch("ACTUAL", 120, 90, hex("#334155")))
            .cell(0, 1, swatch("PIXEL DIFF", 120, 90, hex("#7f1d1d")))
            .cell(1, 1, swatch("STRUCT DIFF", 120, 90, hex("#134e4a")))
            .background(hex("#0e0e10"))
            .into(),
    ));

    v
}

/// Divergence stats between two same-scene renders.
struct Diff {
    native_dim: (u32, u32),
    taffy_dim: (u32, u32),
    pct_differing: f64,
    mean_abs: f64,
    max_abs: u32,
}

fn compare(native: &Bitmap, taffy: &Bitmap) -> Diff {
    let (nw, nh) = native.dimensions();
    let (tw, th) = taffy.dimensions();
    let w = nw.min(tw);
    let h = nh.min(th);
    let mut differing = 0u64;
    let mut sum_abs = 0u64;
    let mut max_abs = 0u32;
    let mut count = 0u64;
    for y in 0..h {
        for x in 0..w {
            let a = native.get_pixel(x, y);
            let b = taffy.get_pixel(x, y);
            let mut pd = 0u32;
            for c in 0..4 {
                let d = (a[c] as i32 - b[c] as i32).unsigned_abs();
                pd = pd.max(d);
                sum_abs += d as u64;
            }
            if pd > 0 {
                differing += 1;
            }
            max_abs = max_abs.max(pd);
            count += 1;
        }
    }
    // Union area (penalize size mismatch as fully-differing extra pixels).
    let union = (nw.max(tw) as u64) * (nh.max(th) as u64);
    let overlap = count.max(1);
    let extra = union.saturating_sub(overlap);
    Diff {
        native_dim: (nw, nh),
        taffy_dim: (tw, th),
        pct_differing: 100.0 * (differing + extra) as f64 / union.max(1) as f64,
        mean_abs: sum_abs as f64 / (overlap * 4) as f64,
        max_abs,
    }
}

/// Stack three bitmaps horizontally with captions, via the native engine.
fn triptych(name: &str, native: &Bitmap, taffy: &Bitmap, diff: &Bitmap) -> Bitmap {
    let panel = |cap: &str, b: &Bitmap| -> Node {
        column()
            .gap(2)
            .align_items(CrossAlign::Center)
            .child(line(cap, WHITE))
            .child(image(b.clone()))
            .into()
    };
    let tree: Node = column()
        .gap(4)
        .child(line(name, hex("#ffd166")))
        .child(
            row()
                .gap(8)
                .align_items(CrossAlign::Start)
                .child(panel("NATIVE", native))
                .child(panel("TAFFY", taffy))
                .child(panel("DIFF x8", diff)),
        )
        .padding(6)
        .background(hex("#101014"));
    render(&tree, 2000)
}

fn main() {
    let out = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/v/output/zensim/taffy-eval"));
    std::fs::create_dir_all(&out).expect("mkdir out");

    let mut report = String::new();
    let _ = writeln!(report, "# taffy backend parity — zensim-regress layouts\n");
    let _ = writeln!(
        report,
        "Native solver vs taffy 0.12 bridge. Identical paint primitives, \
         so pixel deltas are pure layout-solver divergence.\n"
    );
    let _ = writeln!(
        report,
        "| scene | native dim | taffy dim | dims match | % pixels differ | mean |Δ| | max |Δ| |"
    );
    let _ = writeln!(report, "|---|---|---|:--:|--:|--:|--:|");

    let mut all_cov = CoverageSummary::default();
    let mut rows: Vec<Bitmap> = Vec::new();
    let mut exact = 0usize;
    let mut close = 0usize;
    let scenes = corpus();
    let n = scenes.len();

    for (name, max_w, tree) in scenes {
        let cfg = RenderConfig::new(max_w).with_bg(hex("#0e0e10"));
        let native = render_with_config(&tree, &cfg);
        let (taffy, cov) = taffy_backend::render_via_taffy(&tree, &cfg);
        all_cov.absorb(&cov);

        let d = compare(&native, &taffy);
        let dims_match = d.native_dim == d.taffy_dim;
        if dims_match && d.pct_differing == 0.0 {
            exact += 1;
        } else if dims_match && d.pct_differing < 2.0 {
            close += 1;
        }

        // Visual diff (amplified) when dims match; else a red placeholder.
        let diff_img = if dims_match {
            zensim_regress::diff_image::generate_diff_image(&native, &taffy, 8)
        } else {
            Bitmap::from_pixel(
                native.width().max(1),
                native.height().max(1),
                hex("#ff00ff"),
            )
        };

        let _ = native.save(out.join(format!("{name}_native.png")));
        let _ = taffy.save(out.join(format!("{name}_taffy.png")));
        rows.push(triptych(name, &native, &taffy, &diff_img));

        let _ = writeln!(
            report,
            "| `{}` | {}×{} | {}×{} | {} | {:.2}% | {:.2} | {} |",
            name,
            d.native_dim.0,
            d.native_dim.1,
            d.taffy_dim.0,
            d.taffy_dim.1,
            if dims_match { "yes" } else { "**NO**" },
            d.pct_differing,
            d.mean_abs,
            d.max_abs,
        );
    }

    let _ = writeln!(
        report,
        "\n**Summary:** {n} scenes — {exact} pixel-exact, {close} close (<2% pixels, dims match), \
         {} divergent.\n",
        n - exact - close
    );

    let _ = writeln!(report, "## Coverage\n");
    let _ = writeln!(report, "Mapped node kinds: {}\n", all_cov.mapped.join(", "));
    if all_cov.approximated.is_empty() {
        let _ = writeln!(report, "Approximations: none.\n");
    } else {
        let _ = writeln!(report, "Approximations:\n");
        for (k, why) in &all_cov.approximated {
            let _ = writeln!(report, "- `{k}`: {why}");
        }
    }

    // Master montage: stack all triptychs.
    let master: Node = {
        let mut c = column().gap(10);
        for r in &rows {
            c = c.child(image(r.clone()));
        }
        c.padding(10).background(hex("#08080a"))
    };
    let master_bmp = render(&master, 2100);
    let _ = master_bmp.save(out.join("montage.png"));
    std::fs::write(out.join("REPORT.md"), &report).expect("write report");

    println!("{report}");
    println!(
        "wrote {} scenes + montage.png + REPORT.md to {}",
        n,
        out.display()
    );
}

/// Accumulates coverage across scenes for the summary.
#[derive(Default)]
struct CoverageSummary {
    mapped: Vec<String>,
    approximated: Vec<(String, String)>,
}
impl CoverageSummary {
    fn absorb(&mut self, c: &taffy_backend::Coverage) {
        for m in &c.mapped {
            if !self.mapped.iter().any(|x| x == m) {
                self.mapped.push((*m).to_string());
            }
        }
        for (k, why) in &c.approximated {
            if !self.approximated.iter().any(|(a, _)| a == k) {
                self.approximated.push((k.to_string(), why.to_string()));
            }
        }
    }
}
