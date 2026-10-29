//! The slim crate's half of the repository-split parity gate.
//!
//! Prints one TSV line per (surface, profile, pair) in the format of
//! `split_parity_golden` in imazen/zensimx (the research crate this one was cut
//! from on 2026-10-28). Run both on the same machine and compare:
//!
//! ```text
//! cargo run --release -p zensim --features training,classification --example split_parity_check > slim.tsv
//! ```
//!
//! The pairs come from `split_parity/pairs.rs` and the extra input surfaces from
//! `split_parity/surfaces.rs`, byte-identical copies of the research files.

include!("split_parity/hdr.rs");
include!("split_parity/pairs.rs");
include!("split_parity/surfaces.rs");

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
    let line = |surface: &str, profile: &str, pair: &str, value: String| {
        println!("{surface}\t{profile}\t{pair}\t{value}")
    };
    let pairs = split_parity_pairs();
    for pair in &pairs {
        let src = RgbSlice::new(bytemuck::cast_slice(&pair.src), pair.width, pair.height);
        let dst = RgbSlice::new(bytemuck::cast_slice(&pair.dst), pair.width, pair.height);
        for (pname, profile) in &profiles {
            let z = Zensim::new(*profile);
            let v = match z.compute(&src, &dst) {
                Ok(r) => format!(
                    "{} {} {:016x}",
                    hex(r.score()),
                    hex(r.raw_distance()),
                    f64s(r.features())
                ),
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
                Ok(pre) => {
                    match z.compute_with_ref_and_diffmap(pre, &dst, DiffmapOptions::default()) {
                        Ok(d) => format!("{} {:016x}", hex(d.score()), f32s(d.diffmap())),
                        Err(e) => format!("ERR {e:?}"),
                    }
                }
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
    extra_surfaces(&profiles, &pairs, &line);
    hdr_surfaces(&line);
}

/// HDR cases through `Zensim::compute_hdr`, over the fixtures in
/// `split_parity/hdr.rs`. Emits the same `hdrfeat` rows as the golden side.
fn hdr_surfaces(line: &impl Fn(&str, &str, &str, String)) {
    use zensim::HdrEncoding;
    use zensim::source::{AlphaMode, ColorPrimaries, PixelFormat, StridedBytes};

    for c in hdr_cases() {
        let enc = match c.enc {
            0 => HdrEncoding::Linear,
            1 => HdrEncoding::Pq { peak_nits: 1000.0 },
            2 => HdrEncoding::Pq {
                peak_nits: 10_000.0,
            },
            _ => HdrEncoding::Hlg {
                peak_nits: 1000.0,
                ambient_lux: 5.0,
            },
        };
        let fmt = if c.srgb16 {
            PixelFormat::Srgb16Rgba
        } else {
            PixelFormat::LinearF32Rgba
        };
        let pri = if c.bt2020 {
            ColorPrimaries::Bt2020
        } else {
            ColorPrimaries::Srgb
        };
        let mk = |dst: bool| {
            let (bytes, stride) = gen_hdr_image(&c, dst);
            let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
            StridedBytes::try_with_alpha_mode(
                bytes,
                c.width,
                c.height,
                stride,
                fmt,
                AlphaMode::Opaque,
            )
            .expect("hdr fixture source")
            .with_color_primaries(pri)
        };
        let src = mk(false);
        let dst = mk(true);
        let id = hdr_case_id(&c);
        match Zensim::new(ZensimProfile::E)
            .with_parallel(c.parallel)
            .compute_hdr(&src, &dst, enc)
        {
            Ok(r) => {
                let v = r.features();
                line(
                    "hdrfeat",
                    c.enc_name,
                    &format!("{id}:sum"),
                    format!("{:016x}", f64s(v)),
                );
                for (i, f) in v.iter().enumerate() {
                    line(
                        "hdrfeat",
                        c.enc_name,
                        &format!("{id}:f{i}"),
                        format!("{:016x}", f.to_bits()),
                    );
                }
            }
            Err(e) => line(
                "hdrfeat",
                c.enc_name,
                &format!("{id}:sum"),
                format!("ERR {e:?}"),
            ),
        }
    }
}
