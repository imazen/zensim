//! The slim crate's half of the repository-split parity gate.
//!
//! Prints one TSV line per (surface, profile, pair) in the format of
//! `split_parity_golden` in imazen/zensimx (the research crate this one was cut
//! from on 2026-10-28). Run both on the same machine and compare:
//!
//! ```text
//! cargo run --release -p zensim --features training --example split_parity_check > slim.tsv
//! ```
//!
//! The pairs come from `split_parity/pairs.rs`, a byte-identical copy of the research file.

include!("split_parity/pairs.rs");

use zensim::{DiffmapOptions, RgbSlice, Zensim, ZensimProfile, ZensimScratch};

fn fnv64(words: impl Iterator<Item = u64>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn f64s(v: &[f64]) -> u64 {
    fnv64(v.iter().map(|x| x.to_bits()))
}

fn f32s(v: &[f32]) -> u64 {
    fnv64(v.iter().map(|x| x.to_bits() as u64))
}

fn main() {
    let profiles = [
        ("PreviewV0_1", ZensimProfile::PreviewV0_1),
        ("PreviewV0_2", ZensimProfile::PreviewV0_2),
        ("A", ZensimProfile::A),
        ("E", ZensimProfile::E),
        ("ECodec", ZensimProfile::ECodec),
    ];
    let hex = |x: f64| format!("{:016x}", x.to_bits());
    let line = |surface: &str, profile: &str, pair: &str, value: String| println!("{surface}\t{profile}\t{pair}\t{value}");
    for pair in &split_parity_pairs() {
        let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
        let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
        for (pname, profile) in &profiles {
            let z = Zensim::new(*profile);
            let v = match z.compute(&src, &dst) {
                Ok(r) => format!("{} {} {:016x}", hex(r.score()), hex(r.raw_distance()), f64s(r.features())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute", pname, &pair.id, v);
            let pre = z.precompute_reference(&src);
            let v = match &pre {
                Ok(pre) => match z.compute_with_ref(pre, &dst) {
                    Ok(r) => format!("{} {}", hex(r.score()), hex(r.raw_distance())),
                    Err(e) => format!("ERR {e:?}"),
                },
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref", pname, &pair.id, v);
            let v = match &pre {
                Ok(pre) => {
                    let mut scratch = ZensimScratch::new();
                    match z.compute_with_ref_into(pre, &dst, &mut scratch) {
                        Ok(r) => format!("{} {}", hex(r.score()), hex(r.raw_distance())),
                        Err(e) => format!("ERR {e:?}"),
                    }
                }
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref_into", pname, &pair.id, v);
            let v = match z.compute_with_diffmap(&src, &dst, DiffmapOptions::default()) {
                Ok(d) => format!("{} {:016x}", hex(d.score()), f32s(d.diffmap())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_diffmap", pname, &pair.id, v);
            let v = match &pre {
                Ok(pre) => match z.compute_with_ref_and_diffmap(pre, &dst, DiffmapOptions::default()) {
                    Ok(d) => format!("{} {:016x}", hex(d.score()), f32s(d.diffmap())),
                    Err(e) => format!("ERR {e:?}"),
                },
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_with_ref_and_diffmap", pname, &pair.id, v);
            let v = match z.compute_all_features(&src, &dst) {
                Ok(r) => format!("{} {:016x}", hex(r.score()), f64s(r.features())),
                Err(e) => format!("ERR {e:?}"),
            };
            line("compute_all_features", pname, &pair.id, v);
        }
    }
}
