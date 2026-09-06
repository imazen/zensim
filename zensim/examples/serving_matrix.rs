// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **The FEATURE-MATRIX serving probe** — one deterministic TSV of what every
//! shipped profile compiled into THIS build actually scores.
//!
//! Its whole job is to be run under several cargo feature sets and diffed.
//! `scripts/serving_matrix.sh` is the gate that does that; this binary holds
//! no policy, only the measurement.
//!
//! ## Why it exists
//!
//! Since `cb2f412d` (cruft purge 2A) the shipped `A`, `B`, `BHdr` and `D`
//! bakes are DENSE: they declare the feature ids they read
//! (`zentrain.feature_ids`) and their layer-0 width is the size of that read
//! set, not `372`. Serving one correctly requires GATHERING those ids out of
//! the walk's identity-laid-out vector. When the gather is absent, the width
//! disagreement resolves POSITIONALLY (`prep_bake_input_f32`'s
//! `n_inputs < features.len()` prefix branch) — plausible numbers, wrong
//! features, no error. MEASURED on `zensim-wasm-tests` 2026-09-06: profile `A`
//! read 86.30 where the default build reads 93.15 on a single-LSB distortion.
//!
//! So the interesting failure is INVISIBLE from inside one build. The only
//! instrument that can see it is a cross-build diff, which is this file plus
//! the script.
//!
//! Output columns: `profile`, `cell`, `distortion`, `score_bits` (the f64's
//! `to_bits()`, so the diff is exact and not a formatting artifact), `score`.
//! A profile that refuses prints `REFUSED\t<reason>` in place of the two score
//! columns — a refusal is a legitimate outcome the gate accepts, a DIFFERENT
//! number is not.

use zensim::{RgbSlice, Zensim, ZensimProfile};

#[path = "../tests/common/mod.rs"]
mod common;

use common::distortions::truncate_lsb;
use common::generators::{distort_blur, gen_checkerboard, gen_value_noise};

/// The geometries. A subset of the parity matrix (`common::parity_cells`) —
/// the probe runs every profile over every cell over every distortion, so the
/// full 20-cell matrix would be a minute of wall time for no extra coverage of
/// the LAYOUT question, which is geometry-independent. These four span the
/// tight/non-tight width classes and the sub-64 reflect-pad path.
const CELLS: &[(usize, usize)] = &[(256, 256), (127, 93), (576, 96), (48, 40)];

/// The roster, mirroring `zensim::serving::shipped_profiles` — which an
/// example cannot call, because it is `pub(crate)` and exporting it would be a
/// public-API delta for a test fixture.
///
/// A second `#[cfg]`-dependent profile list is exactly the drift
/// `shipped_profiles`'s own doc warns about, so this one is not left
/// unguarded: `serving::tests::the_serving_matrix_example_carries_the_same_roster`
/// reads THIS FILE with `include_str!` and fails if the two bodies name
/// different `(gating feature, profile)` pairs. Keep the shape — an
/// unconditional `vec![…]` then `#[cfg(feature = "…")]` blocks — or that scan
/// stops seeing the roster (it asserts it found at least 8 pairs).
fn profiles() -> Vec<(&'static str, ZensimProfile)> {
    let mut v: Vec<(&'static str, ZensimProfile)> = vec![
        ("B", ZensimProfile::B),
        ("BHdr", ZensimProfile::BHdr),
        ("PreviewV0_1", ZensimProfile::PreviewV0_1),
        ("PreviewV0_2", ZensimProfile::PreviewV0_2),
    ];
    #[cfg(feature = "deprecated-profiles")]
    {
        #[allow(deprecated)]
        v.push(("A", ZensimProfile::A));
    }
    #[cfg(feature = "candidate-profiles")]
    {
        v.push(("C", ZensimProfile::C));
        v.push(("CHdr", ZensimProfile::CHdr));
        v.push(("D", ZensimProfile::D));
    }
    v
}

fn main() {
    println!("profile\tcell\tdistortion\tscore_bits\tscore");
    for (name, profile) in profiles() {
        let z = Zensim::new(profile);
        for &(w, h) in CELLS {
            let noise = gen_value_noise(w, h, 42);
            let checker = gen_checkerboard(w, h, 8);
            let cases = [
                ("truncate_lsb", &noise, truncate_lsb(&noise)),
                ("blur3", &noise, distort_blur(&noise, w, h, 3)),
                ("checker_lsb", &checker, truncate_lsb(&checker)),
            ];
            for (dname, src, dst) in &cases {
                let r = RgbSlice::new(src, w, h);
                let d = RgbSlice::new(dst, w, h);
                match z.compute(&r, &d) {
                    Ok(res) => {
                        let s = res.score();
                        println!("{name}\t{w}x{h}\t{dname}\t{:#018x}\t{s:.12}", s.to_bits());
                    }
                    Err(e) => println!("{name}\t{w}x{h}\t{dname}\tREFUSED\t{e}"),
                }
            }
        }
    }
}
