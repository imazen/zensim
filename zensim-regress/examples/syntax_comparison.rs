//! Three syntax styles producing the same 4-panel layout, for visual
//! comparison. Run with:
//!
//! ```text
//! cargo run -p zensim-regress --features _internal_api \
//!     --example syntax_comparison -- /tmp/syntax_compare
//! ```
//!
//! Each version writes the same PNG to a different file. Inspecting
//! the source side-by-side shows the syntactic feel of each option.

use std::env;
use std::path::PathBuf;

use zensim_regress::Bitmap;
use zensim_regress::layout::compose::*;
use zensim_regress::layout::*;
use zensim_regress::{column, layers, row};

fn solid(w: u32, h: u32, c: Color) -> Bitmap {
    Bitmap::from_pixel(w, h, c)
}

fn header_bar(title: &str) -> Node {
    layers()
        .child(fill(hex("#1f2733")))
        .child(
            line(title, WHITE)
                .align(HAlign::Left, VAlign::Center)
                .padding(8),
        )
        .height(SizeRule::Fixed(28))
}

// ── Style 1: existing fluent builder (status quo) ─────────────────────

fn fluent() -> Node {
    column()
        .gap(8)
        .child(header_bar("zensim — fluent builder"))
        .child(
            row()
                .gap(4)
                .child(image(solid(80, 80, hex("#ef4444"))).label("A"))
                .child(image(solid(80, 80, hex("#10b981"))).label("B")),
        )
        .child(
            row()
                .gap(4)
                .child(image(solid(80, 80, hex("#3b82f6"))).label("C"))
                .child(image(solid(80, 80, hex("#f59e0b"))).label("D")),
        )
        .padding(8)
        .background(hex("#0e0e10"))
}

// ── Style 2: macro_rules! array-children macros ──────────────────────

fn macros_style() -> Node {
    column![
        header_bar("zensim — array macros"),
        row![
            image(solid(80, 80, hex("#ef4444"))).label("A"),
            image(solid(80, 80, hex("#10b981"))).label("B"),
        ]
        .gap(4),
        row![
            image(solid(80, 80, hex("#3b82f6"))).label("C"),
            image(solid(80, 80, hex("#f59e0b"))).label("D"),
        ]
        .gap(4),
    ]
    .gap(8)
    .padding(8)
    .background(hex("#0e0e10"))
}

// ── Style 3: closure-builder ─────────────────────────────────────────

fn compose_style() -> Node {
    column_block(|c| {
        c.gap(8);
        c.push(header_bar("zensim — closure builder"));
        c.row(|r| {
            r.gap(4);
            r.image(solid(80, 80, hex("#ef4444")));
            r.image(solid(80, 80, hex("#10b981")));
        });
        c.row(|r| {
            r.gap(4);
            r.image(solid(80, 80, hex("#3b82f6")));
            r.image(solid(80, 80, hex("#f59e0b")));
        });
    })
    .padding(8)
    .background(hex("#0e0e10"))
}

// ── Closure-builder shines for refactoring fragments ──────────────────

fn compose_factored() -> Node {
    fn add_pair(r: &mut RowCtx, left: Bitmap, right: Bitmap) {
        r.gap(4);
        r.image(left);
        r.image(right);
    }
    column_block(|c| {
        c.gap(8);
        c.push(header_bar("zensim — factored fragments"));
        c.row(|r| {
            add_pair(
                r,
                solid(80, 80, hex("#ef4444")),
                solid(80, 80, hex("#10b981")),
            )
        });
        c.row(|r| {
            add_pair(
                r,
                solid(80, 80, hex("#3b82f6")),
                solid(80, 80, hex("#f59e0b")),
            )
        });
    })
    .padding(8)
    .background(hex("#0e0e10"))
}

// ── Layered "fancy" example showing all three at composing depth 2 ────

fn fluent_layered() -> Node {
    layers()
        .child(fill(hex("#1f2733")))
        .child(
            column()
                .gap(4)
                .child(line("EXPECTED", WHITE))
                .child(image(solid(120, 60, hex("#ef4444"))))
                .padding(8),
        )
        .into()
}

fn macros_layered() -> Node {
    layers![
        fill(hex("#1f2733")),
        column![
            line("EXPECTED", WHITE),
            image(solid(120, 60, hex("#ef4444"))),
        ]
        .gap(4)
        .padding(8),
    ]
    .into()
}

fn compose_layered() -> Node {
    layers_block(|l| {
        l.fill(hex("#1f2733"));
        l.column(|c| {
            c.gap(4);
            c.line("EXPECTED", WHITE);
            c.image(solid(120, 60, hex("#ef4444")));
        });
    })
}

fn main() {
    let dir: PathBuf = env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/syntax_compare".to_string())
        .into();
    std::fs::create_dir_all(&dir).expect("create output dir");

    for (name, n) in [
        ("01_fluent.png", fluent()),
        ("02_macros.png", macros_style()),
        ("03_compose.png", compose_style()),
        ("04_compose_factored.png", compose_factored()),
        ("05_fluent_layered.png", fluent_layered()),
        ("06_macros_layered.png", macros_layered()),
        ("07_compose_layered.png", compose_layered()),
    ] {
        let bm = render(&n, 400);
        bm.save(dir.join(name)).expect("save png");
    }
    println!("Wrote 7 scenes to {}", dir.display());
}
