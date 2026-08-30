// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.
//! era-2 design validation: does the f32-lane → f64-virtual-lane accumulation
//! shape COST anything? (`benchmarks/era2_perf_break_2026-08-31.md` §2)
//!
//! The design moves the accumulators from f32 SIMD lanes to 8 f64 virtual
//! lanes. That is an accuracy win by construction (§12.2) and it is what makes
//! cross-tier bit-identity possible (§2.2) — but f64 lanes are half as wide, so
//! the accumulation itself does more register work. **If that costs more than
//! band-parallelism buys, the premise needs revisiting**, so it is measured
//! BEFORE the reshape is built rather than discovered afterwards.
//!
//! This is a shape microbenchmark, not the real kernel: it isolates the
//! accumulate-into-lanes step over a representative row, with the term
//! evaluation stubbed to a cheap f32 expression so the accumulation dominates.
//! It answers "what does the lane shape cost", not "what does the kernel cost".
//!
//! Run: `cargo bench --bench era2_accum_shape -p zensim`
use std::hint::black_box;

const WIDTH: usize = 576;
const ROWS: usize = 128;

fn terms(w: usize) -> Vec<f32> {
    (0..w)
        .map(|i| ((i * 2654435761usize) % 65521) as f32 * (1.0 / 65536.0))
        .collect()
}

/// era-1 shape: 8 f32 lanes, zeroed per row, reduced to f64 once per row.
fn era1_f32_lanes(rows: &[Vec<f32>]) -> f64 {
    let mut acc = 0.0f64;
    for row in rows {
        let mut lanes = [0.0f32; 8];
        let n8 = row.len() - row.len() % 8;
        for c in row[..n8].as_chunks::<8>().0 {
            for (l, &v) in lanes.iter_mut().zip(c) {
                *l += v;
            }
        }
        let mut r = 0.0f32;
        for l in lanes {
            r += l;
        }
        acc += r as f64;
        for &v in &row[n8..] {
            acc += v as f64; // era-1's per-pixel f64 tail
        }
    }
    acc
}

/// era-2 shape: 8 f64 VIRTUAL lanes spanning the whole band, tail folded in,
/// one fixed reduction at the end.
fn era2_f64_virtual_lanes(rows: &[Vec<f32>]) -> f64 {
    let mut lanes = [0.0f64; 8];
    for row in rows {
        for (x, &v) in row.iter().enumerate() {
            lanes[x % 8] += v as f64;
        }
    }
    let mut acc = 0.0f64;
    for l in lanes {
        acc += l;
    }
    acc
}

/// era-2 shape, chunk-structured so LLVM can see the 8 independent lanes —
/// the form the real kernel would emit (`chunks_exact(8)` + a folded tail).
fn era2_f64_chunked(rows: &[Vec<f32>]) -> f64 {
    let mut lanes = [0.0f64; 8];
    for row in rows {
        let n8 = row.len() - row.len() % 8;
        for c in row[..n8].as_chunks::<8>().0 {
            for (l, &v) in lanes.iter_mut().zip(c) {
                *l += v as f64;
            }
        }
        for (k, &v) in row[n8..].iter().enumerate() {
            lanes[k] += v as f64;
        }
    }
    let mut acc = 0.0f64;
    for l in lanes {
        acc += l;
    }
    acc
}

fn main() {
    let rows: Vec<Vec<f32>> = (0..ROWS).map(|_| terms(WIDTH)).collect();
    let rows: &'static [Vec<f32>] = Box::leak(rows.into_boxed_slice());
    let result = zenbench::run(|suite| {
        suite.compare("era2_accum_shape_576x128", |group| {
            group
                .config()
                .max_rounds(200)
                .min_rounds(25)
                .max_wall_time(std::time::Duration::from_secs(240));
            group.bench("era1_f32_lanes", move |b| {
                b.iter(move || black_box(era1_f32_lanes(rows)))
            });
            group.bench("era2_f64_virtual", move |b| {
                b.iter(move || black_box(era2_f64_virtual_lanes(rows)))
            });
            group.bench("era2_f64_chunked", move |b| {
                b.iter(move || black_box(era2_f64_chunked(rows)))
            });
        });
    });
    let _ = result;
}
