//! **The cross-libc gate's instrument** — dump every v1-372 feature as
//! `to_bits()` over a fixed, procedurally-generated grid, so two builds of
//! the same commit against DIFFERENT libcs can be compared with `cmp`.
//!
//! ```sh
//! for T in x86_64-unknown-linux-gnu x86_64-unknown-linux-musl; do
//!   cargo build --release -p zensim --features training \
//!       --example libc_feature_dump --target "$T"
//!   ./target/$T/release/examples/libc_feature_dump > ~/tmp/dump.$T.tsv
//! done
//! cmp ~/tmp/dump.x86_64-unknown-linux-gnu.tsv \
//!     ~/tmp/dump.x86_64-unknown-linux-musl.tsv
//! ```
//!
//! `ZENSIM_ROOT_FORM=libm` reproduces the shipped (revision-1) FEATURE arm
//! and `=sqrt` the deterministic one; `ZENSIM_POW_FORM=libm|pure` does the
//! same for the SCORE (F19). Two knobs, not one, because F18's `sqrt`
//! derivation cannot reach `x^0.7` — so the SAME pair of binaries measures
//! the whole 2x2 and can show that fixing the features did not fix the
//! score.
//!
//! # Why the inputs are procedural
//!
//! Every pixel is generated in-process from an integer PRNG, so the two
//! builds are provably fed identical bytes without shipping a corpus into two
//! containers — a corpus that had to be decoded would put a *decoder* in the
//! comparison, which is exactly the confound this gate must not have.
//!
//! # The grid
//!
//! * the 20-cell parity geometry matrix, owned by
//!   `tests/common/parity_cells.rs` — the same list `fold_engine_parity` and
//!   `research_engine_parity` use, reached by `#[path]` rather than retyped
//! * a 200-cell distortion **ladder**: two geometries x 100 monotonically
//!   increasing quantisation steps, which is what exercises the pooled
//!   4th/8th roots across their whole magnitude range rather than at one
//!   operating point

use zensim::{ZensimConfig, compute_zensim_with_config};

#[path = "../tests/common/generators.rs"]
mod generators;
#[path = "../tests/common/parity_cells.rs"]
mod parity_cells;

fn v1_config() -> ZensimConfig {
    let mut cfg = ZensimConfig::default();
    cfg.compute_all_features = true;
    cfg.extended_features = true;
    cfg.compute_iw_features = true;
    cfg
}

/// A deterministic quantisation ladder step: round each channel to a
/// multiple of `step`. Integer-only, so it is bit-identical everywhere by
/// construction and cannot itself be the source of a difference.
fn quantise(src: &[[u8; 3]], step: u16) -> Vec<[u8; 3]> {
    let s = step.max(1);
    src.iter()
        .map(|p| {
            let q = |v: u8| -> u8 {
                let v = v as u16;
                let r = ((v + s / 2) / s) * s;
                r.min(255) as u8
            };
            [q(p[0]), q(p[1]), q(p[2])]
        })
        .collect()
}

fn emit(tag: &str, r: &[[u8; 3]], d: &[[u8; 3]], w: usize, h: usize) {
    let out = compute_zensim_with_config(r, d, w, h, v1_config())
        .unwrap_or_else(|e| panic!("{tag}: {e:?}"));
    let f = out.features();
    // The score is dumped too, and it is a SECOND, INDEPENDENT libc exposure:
    // `metric.rs`'s raw-distance -> score mapping calls `powf` with
    // non-power-of-two exponents, plus `exp` in the squash and both head
    // sigmoids and `log2` in the MLP size axes. That is F19, owned by
    // `det_math::PowForm` and selected by `ZENSIM_POW_FORM`. Dumping it in
    // the same file is what lets one gate measure both.
    println!("{tag}\tscore\t{:016x}", out.score().to_bits());
    for (i, v) in f.iter().enumerate() {
        println!("{tag}\tf{i}\t{:016x}", v.to_bits());
    }
}

fn main() {
    eprintln!(
        "root_form_env={:?} pow_form_env={:?}",
        std::env::var("ZENSIM_ROOT_FORM").ok(),
        std::env::var("ZENSIM_POW_FORM").ok()
    );

    for (ci, &(w, h)) in parity_cells::CELLS.iter().enumerate() {
        let r =
            generators::gen_value_noise(w, h, 0xC0FFEE ^ (ci as u32).wrapping_mul(2_654_435_761));
        let d = generators::distort_block_artifacts(&r, w, h);
        emit(&format!("cell{ci:02}_{w}x{h}"), &r, &d, w, h);
    }

    // 200 ladder cells: 2 geometries x 100 steps. 200x150 is the non-tight
    // width class option C fixed; 128x96 is tight. Both are far enough above
    // the 64px pyramid floor that all four scales are live.
    for &(w, h) in &[(200usize, 150usize), (128usize, 96usize)] {
        let r = generators::gen_mandelbrot(w, h);
        for step in 1u16..=100 {
            let d = quantise(&r, step);
            emit(&format!("ladder_{w}x{h}_q{step:03}"), &r, &d, w, h);
        }
    }
}
