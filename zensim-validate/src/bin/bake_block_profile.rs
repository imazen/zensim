//! bake_block_profile — static per-bake FEATURE-BLOCK usage fingerprint from
//! the bake bytes alone (no corpus, no scoring).
//!
//! For each feature family of the append-only numbering discipline —
//!   f0..f155    v1-basic
//!   f156..f371  the block ZEROED by the folded 924/944 regimes (slots
//!               preserved per the append-only discipline — never removed;
//!               carries real values in the v1-372 / 720 regimes)
//!   f372..f719  v2-348
//!   f720..f943  append-204
//! — count the encoder columns (layer-0 input columns; the linear weights for
//! a 1-layer bake) whose L2 norm over outputs is (a) exactly zero, (b)
//! near-zero (≤ 1e-6 × the layer's max column norm), (c) structurally used.
//!
//! This is the STRUCTURAL usage complement to the corpus-based contribution
//! measure (a sibling instrument): a column can be structurally nonzero yet
//! effectively unused when its input is always zero. MEASURED 2026-08-04:
//! the 944-regime trainers emit EXACT zeros for the whole f156..f371 block
//! (216/216 on `C_co3a_s1301`, `C_ensk2_s1303`), and prune part of the
//! append block (61/224 exact-zero) — so exact-zero counts are a real
//! discriminator, not init noise. Exact zeros also appear when a
//! trainer/packer explicitly zeroes (zerobias, lasso, BVLS).
//!
//! Model access goes through `zenpredict::Model` (the canonical loader —
//! handles v3.1 compression, dtype decode, canonical ordering); this bin
//! adds NO wire-format code of its own.
//!
//! Usage: bake_block_profile --bake <bake.bin> [--json]
//! Exit: 0 ok; 2 usage/load error.

use std::path::PathBuf;
use zenpredict::{Model, WeightStorage, f16_bits_to_f32};

/// (label, start, end) — end exclusive; clipped to the bake's in_dim.
const FAMILIES: &[(&str, usize, usize)] = &[
    ("f0_155", 0, 156),
    ("f156_371", 156, 372),
    ("f372_719", 372, 720),
    ("f720_943", 720, 944),
];
/// Near-zero threshold, relative to the layer's max column norm.
const NEAR_ZERO_REL: f64 = 1e-6;

fn usage() -> ! {
    eprintln!("usage: bake_block_profile --bake <bake.bin> [--json]");
    std::process::exit(2);
}

fn main() {
    let mut bake: Option<PathBuf> = None;
    let mut json = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--bake" => bake = args.next().map(PathBuf::from),
            "--json" => json = true,
            _ => usage(),
        }
    }
    let path = match bake {
        Some(p) => p,
        None => usage(),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake_block_profile: read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bake_block_profile: load {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    // Weights are row-major input-major: W[i * out_dim + o].
    let wget: Box<dyn Fn(usize, usize) -> f64> = match &layer.weights {
        WeightStorage::F32(w) => {
            let w = *w;
            Box::new(move |i, o| w[i * out_dim + o] as f64)
        }
        WeightStorage::F16(w) => {
            let w = *w;
            Box::new(move |i, o| f16_bits_to_f32(w[i * out_dim + o]) as f64)
        }
        WeightStorage::I8 { weights, scales } => {
            let (w, s) = (*weights, *scales);
            Box::new(move |i, o| w[i * out_dim + o] as f64 * s[o] as f64)
        }
    };
    let dtype = match &layer.weights {
        WeightStorage::F32(_) => "f32",
        WeightStorage::F16(_) => "f16",
        WeightStorage::I8 { .. } => "i8",
    };
    // Per-input-column L2 norm over outputs.
    let norms: Vec<f64> = (0..in_dim)
        .map(|i| (0..out_dim).map(|o| wget(i, o).powi(2)).sum::<f64>().sqrt())
        .collect();
    let max_norm = norms.iter().cloned().fold(0.0f64, f64::max);
    let near = NEAR_ZERO_REL * max_norm;

    let mut fams = Vec::new();
    for &(label, lo, hi) in FAMILIES {
        if lo >= in_dim {
            continue;
        }
        let hi = hi.min(in_dim);
        let cols = hi - lo;
        let sl = &norms[lo..hi];
        let exact_zero = sl.iter().filter(|&&n| n == 0.0).count();
        let near_zero = sl.iter().filter(|&&n| n > 0.0 && n <= near).count();
        let used = cols - exact_zero - near_zero;
        let fam_max = sl.iter().cloned().fold(0.0f64, f64::max);
        fams.push((label, cols, exact_zero, near_zero, used, fam_max));
    }
    let beyond = in_dim.saturating_sub(944);
    // "uses f156-371": structurally-used columns in the zeroed-block family.
    let uses_zeroed_block = fams
        .iter()
        .find(|(l, ..)| *l == "f156_371")
        .map(|&(_, _, _, _, used, _)| used > 0)
        .unwrap_or(false);

    if json {
        let fam_json: Vec<String> = fams
            .iter()
            .map(|(l, cols, ez, nz, used, mx)| {
                format!(
                    "\"{l}\":{{\"cols\":{cols},\"exact_zero\":{ez},\"near_zero\":{nz},\"used\":{used},\"max_col_norm\":{mx:.6e}}}"
                )
            })
            .collect();
        println!(
            "{{\"znpr_version\":{},\"n_inputs\":{},\"layer0_in_dim\":{in_dim},\"layer0_out_dim\":{out_dim},\"dtype\":\"{dtype}\",\"n_layers\":{},\"beyond_f943_cols\":{beyond},\"near_zero_rel\":{NEAR_ZERO_REL:e},\"uses_f156_371\":{uses_zeroed_block},\"families\":{{{}}}}}",
            model.version(),
            model.n_inputs(),
            model.n_layers(),
            fam_json.join(",")
        );
    } else {
        println!(
            "# {} — ZNPR v{}, {} layer(s), layer0 {in_dim}→{out_dim} ({dtype})",
            path.display(),
            model.version(),
            model.n_layers()
        );
        println!("family    cols  exact0  near0  used  max‖col‖");
        for (l, cols, ez, nz, used, mx) in &fams {
            println!("{l:<9} {cols:>4}  {ez:>6}  {nz:>5}  {used:>4}  {mx:.3e}");
        }
        if beyond > 0 {
            println!("(+{beyond} layer-0 columns beyond f943 — transform-expanded inputs)");
        }
        println!("uses_f156_371 (structural): {uses_zeroed_block}");
    }
}
