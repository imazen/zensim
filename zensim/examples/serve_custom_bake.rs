//! Serve an arbitrary ZNPR bake through the PRODUCTION scoring path
//! (`Zensim::compute`), so "is this candidate servable?" is a MEASUREMENT
//! rather than an inference from reading `profile.rs`.
//!
//! Written for `benchmarks/fastclass2_campaign_2026-09-05.md` gate G7. The
//! kernel lane (`benchmarks/kernel_fastclass_2026-09-05.md` §4 and commit
//! `8817f379`) established that `Zensim::compute` emits a **372-layout**
//! vector with `free_extras: Off`, so a 944-declared bake is refused and a
//! 156/228-slice bake at the v1-372 layout should serve. This example checks
//! the second half on real pixels instead of taking it on trust — the
//! `d_ship_flip` lane found the 944 refusal exactly this way.
//!
//! ```sh
//! cargo run --release --example serve_custom_bake \
//!   --features custom-profiles,candidate-profiles \
//!   -- <bake.bin> <ref.png> <dist.png>
//! ```
//!
//! Prints the bake's declared caller width, the served score, and the
//! reference's identity score (`ref` vs itself) — the C5 quantity — or the
//! exact error if the production path refuses the bake.

use std::sync::OnceLock;
use zensim::profile::ProfileParams;
use zensim::{RgbSlice, Zensim, ZensimProfile};

static BAKE: OnceLock<Vec<u8>> = OnceLock::new();
fn bake_bytes() -> &'static [u8] {
    BAKE.get().expect("bake set before use").as_slice()
}

fn load_rgb(path: &str) -> (Vec<[u8; 3]>, u32, u32) {
    let img = image::open(path)
        .unwrap_or_else(|e| panic!("open {path}: {e}"))
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let raw = img.into_raw();
    (raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), w, h)
}

fn main() {
    let mut a = std::env::args().skip(1);
    let bake_path = a.next().expect("usage: serve_custom_bake <bake.bin> <ref> <dist>");
    let ref_path = a.next().expect("ref image required");
    let dist_path = a.next().expect("dist image required");

    let bytes = std::fs::read(&bake_path).unwrap_or_else(|e| panic!("read {bake_path}: {e}"));
    println!("bake: {bake_path} ({} bytes)", bytes.len());
    match zenpredict::Model::from_bytes(&bytes) {
        Ok(m) => println!(
            "  declared: n_inputs={} caller_input_width={}",
            m.n_inputs(),
            m.caller_input_width()
        ),
        Err(e) => println!("  NOT a loadable ZNPR: {e:?}"),
    }
    BAKE.set(bytes).expect("set once");

    let params: &'static ProfileParams = Box::leak(Box::new(
        ProfileParams::builder()
            .mlp(bake_bytes)
            .extended_features(true)
            .compute_iw_features(true)
            .skip_score_mapping(true)
            .extrapolate_score(true)
            .build(),
    ));
    let profile = ZensimProfile::Custom {
        name: "fastclass2-servability",
        params,
    };
    let z = Zensim::new(profile);

    let (r, w, h) = load_rgb(&ref_path);
    let (d, dw, dh) = load_rgb(&dist_path);
    assert_eq!((w, h), (dw, dh), "ref and dist must share dimensions");

    let rs = RgbSlice::new(&r, w as usize, h as usize);
    let ds = RgbSlice::new(&d, w as usize, h as usize);
    // The whole point: this is the PRODUCTION entry point, not a training one.
    match z.compute(&rs, &ds) {
        Ok(res) => println!(
            "SERVED  score={:.6}  raw_distance={:.6}",
            res.score(), res.raw_distance()
        ),
        Err(e) => println!("REFUSED by Zensim::compute: {e:?}"),
    }
    match z.compute(&rs, &rs) {
        Ok(res) => println!("IDENTITY (ref vs ref) score={:.6}", res.score()),
        Err(e) => println!("IDENTITY REFUSED: {e:?}"),
    }
}
