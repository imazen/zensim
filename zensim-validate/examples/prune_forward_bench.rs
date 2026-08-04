//! zenbench A/B: MLP forward time, dead-column-pruned bake vs its
//! un-pruned twin.
//!
//! Pruning's storage win is small once the ZNPR payload is LZ4-compressed
//! (a run of zero `f16` weights compresses to nearly nothing). The win
//! that compression *cannot* give you is inference: layer 0 dominates a
//! `944 → 128 → 1` net, and the forward kernel's `if s == 0.0 { continue }`
//! fast path does NOT fire on a dead column — the standardized input
//! `(x − mean)/scale` is generically nonzero even when the whole weight
//! row is zero, so an un-pruned bake pays 944 SAXPY rows to accumulate
//! 277 rows of zeros.
//!
//! Both arms run `Predictor::predict_transformed` over the SAME feature
//! rows, interleaved round-robin by zenbench so thermal/turbo drift hits
//! both equally.
//!
//! Usage:
//!   cargo run --release -p zensim-validate --example prune_forward_bench -- \
//!       <unpruned.bin> <pruned.bin> [rows]
//!
//! Produce the pair with:
//!   bake_dial_refit pack --in X_dial.bin --out unpruned.bin --no-prune ...
//!   bake_dial_refit pack --in X_dial.bin --out pruned.bin ...

use std::env;

use zenpredict::{Model, Predictor};

/// Deterministic pseudo-random feature rows. Values are irrelevant to
/// the timing (the kernel is branch-free per weight) as long as the
/// standardized inputs are generically nonzero, which is the whole
/// point — a zero input would let the `s == 0.0` fast path skip the
/// row and hide the cost pruning removes.
fn make_rows(n_rows: usize, width: usize) -> Vec<Vec<f32>> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n_rows)
        .map(|_| {
            (0..width)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    // [-4, 4): a plausible standardized-feature range.
                    ((state >> 40) as f32 / 1024.0) - 4.0
                })
                .collect()
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!(
            "usage: prune_forward_bench <unpruned.bin> <pruned.bin> [rows]\n\
             both bakes must share a caller_input_width"
        );
        std::process::exit(2);
    }
    let n_rows: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(256);

    let a_bytes: &'static [u8] = Box::leak(
        std::fs::read(&args[1])
            .expect("read unpruned bake")
            .into_boxed_slice(),
    );
    let b_bytes: &'static [u8] = Box::leak(
        std::fs::read(&args[2])
            .expect("read pruned bake")
            .into_boxed_slice(),
    );
    let a_model: &'static Model = Box::leak(Box::new(
        Model::from_bytes(a_bytes).expect("parse unpruned"),
    ));
    let b_model: &'static Model =
        Box::leak(Box::new(Model::from_bytes(b_bytes).expect("parse pruned")));

    let width = a_model.caller_input_width();
    assert_eq!(
        width,
        b_model.caller_input_width(),
        "the two bakes must accept the same caller width — that is the whole contract"
    );
    eprintln!(
        "unpruned: caller_width {} layer0_in_dim {} ({} B file)\n\
         pruned:   caller_width {} layer0_in_dim {} ({} B file)\n\
         layer-0 rows removed: {} ({:.1}%)  rows={n_rows}",
        width,
        a_model.n_inputs(),
        a_bytes.len(),
        width,
        b_model.n_inputs(),
        b_bytes.len(),
        a_model.n_inputs() - b_model.n_inputs(),
        100.0 * (a_model.n_inputs() - b_model.n_inputs()) as f64 / a_model.n_inputs() as f64,
    );

    let rows: &'static [Vec<f32>] = Box::leak(make_rows(n_rows, width).into_boxed_slice());

    let result = zenbench::run(|suite| {
        suite.compare("predict_transformed", |group| {
            group.config().max_rounds(200);
            group.bench("unpruned", move |b| {
                let mut p = Predictor::new(a_model);
                b.iter(move || {
                    let mut acc = 0f32;
                    for row in rows {
                        acc += p.predict_transformed(row).expect("forward")[0];
                    }
                    zenbench::black_box(acc);
                })
            });
            group.bench("pruned", move |b| {
                let mut p = Predictor::new(b_model);
                b.iter(move || {
                    let mut acc = 0f32;
                    for row in rows {
                        acc += p.predict_transformed(row).expect("forward")[0];
                    }
                    zenbench::black_box(acc);
                })
            });
        });
    });
    let _ = result;
}
