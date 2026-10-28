//! Contracts of the shipped profiles: identity, monotone damage, tier and thread
//! invariance of the revision-7 profiles, and golden scores.
//!
//! The golden scores were produced by the research crate this one was cut from
//! (imazen/zensimx, `split_parity_golden`, 2026-10-28) on the pairs in
//! `examples/split_parity/pairs.rs`; the slim crate reproduced all 6,930 values of
//! that run bit for bit before these were copied here.
//!
//! Run the tier test alone (`--test-threads=1`) when debugging: it disables SIMD
//! tiers process-wide while it runs.

include!("../examples/split_parity/pairs.rs");

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn pair(id: &str) -> SplitPair {
    split_parity_pairs()
        .into_iter()
        .find(|p| p.id == id)
        .unwrap_or_else(|| panic!("no pair {id}"))
}

fn score(profile: ZensimProfile, p: &SplitPair) -> f64 {
    let src = RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
    let dst = RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
    Zensim::new(profile).compute(&src, &dst).expect("compute").score()
}

fn by_name(name: &str) -> ZensimProfile {
    match name {
        "E" => ZensimProfile::E,
        "ECodec" => ZensimProfile::ECodec,
        "A" => ZensimProfile::A,
        "PreviewV0_2" => ZensimProfile::PreviewV0_2,
        _ => unreachable!(),
    }
}

/// (profile, pair, score bits) from the research reference run.
const GOLDEN: &[(&str, &str, u64)] = &[
    ("E", "64x64-photo-noise", 0x4052aad69a938ab2),
    ("ECodec", "64x64-photo-noise", 0x40523aeeb9e7c6a8),
    ("A", "64x64-photo-noise", 0x404df8e8f9af74e6),
    ("PreviewV0_2", "64x64-photo-noise", 0x4053c2809278bdab),
    ("E", "64x64-edges-blur", 0x401e2fe329bdc22c),
    ("ECodec", "64x64-edges-blur", 0x403892a4cc5e23a3),
    ("A", "64x64-edges-blur", 0xc001d4a1dbbf9ad3),
    ("PreviewV0_2", "64x64-edges-blur", 0x402fe8667b189400),
    ("E", "65x63-text-blocks", 0xc04bb7824bf08dbe),
    ("ECodec", "65x63-text-blocks", 0x40388e003c6fccd9),
    ("A", "65x63-text-blocks", 0x40351061fa16b761),
    ("PreviewV0_2", "65x63-text-blocks", 0xc042853a97107170),
    ("E", "128x96-photo-posterize", 0x405219d5ef2fb40e),
    ("ECodec", "128x96-photo-posterize", 0x4051084784f15682),
    ("A", "128x96-photo-posterize", 0x404ed25181330dd5),
    ("PreviewV0_2", "128x96-photo-posterize", 0x40537ea0198c0b60),
    ("E", "200x150-edges-shift", 0x4056be19206d8147),
    ("ECodec", "200x150-edges-shift", 0x40560fdf8baea128),
    ("A", "200x150-edges-shift", 0x40549b2c10c8cf58),
    ("PreviewV0_2", "200x150-edges-shift", 0x40553e6ff2624cbd),
    ("E", "200x150-photo-blur", 0x404a91a05f6040e3),
    ("ECodec", "200x150-photo-blur", 0x4047794a4cb413eb),
    ("A", "200x150-photo-blur", 0x405276e2c552885f),
    ("PreviewV0_2", "200x150-photo-blur", 0x40536d00e6303f17),
    ("E", "256x256-text-noise", 0x404807234bba5a54),
    ("ECodec", "256x256-text-noise", 0x40478cc06a650371),
    ("A", "256x256-text-noise", 0x404fce495f5f703d),
    ("PreviewV0_2", "256x256-text-noise", 0x4054b03976a363b0),
    ("E", "256x256-photo-pixels", 0x4058f426ec4d2c2d),
    ("ECodec", "256x256-photo-pixels", 0x4058f83453851038),
    ("A", "256x256-photo-pixels", 0x40580f0dd03070e0),
    ("PreviewV0_2", "256x256-photo-pixels", 0x4058dde8b46bdf4a),
    ("E", "333x251-edges-blocks", 0xc04a598d9a2688a1),
    ("ECodec", "333x251-edges-blocks", 0x40314e52d5f1ed6a),
    ("A", "333x251-edges-blocks", 0xc05224a3ea2fef19),
    ("PreviewV0_2", "333x251-edges-blocks", 0xc061ff02ae3ac7b4),
    ("E", "333x251-photo-identity", 0x4059000000000000),
    ("ECodec", "333x251-photo-identity", 0x4059000000000000),
    ("A", "333x251-photo-identity", 0x4059000000000000),
    ("PreviewV0_2", "333x251-photo-identity", 0x4059000000000000),
    ("E", "16x16-photo-noise", 0x4054c3e7e75a28dc),
    ("ECodec", "16x16-photo-noise", 0x40548bd9cea66699),
    ("A", "16x16-photo-noise", 0x40500aefe491ef04),
    ("PreviewV0_2", "16x16-photo-noise", 0x405520a00c68709c),
    ("E", "31x17-text-shift", 0x405349e63d8450b3),
    ("ECodec", "31x17-text-shift", 0x4051211124b16b76),
    ("A", "31x17-text-shift", 0x4056840035ddea76),
    ("PreviewV0_2", "31x17-text-shift", 0x40588bb8484bed5e),
    ("E", "512x384-photo-blocks", 0x404f49b428470fbb),
    ("ECodec", "512x384-photo-blocks", 0x404ee3dc61c36fcf),
    ("A", "512x384-photo-blocks", 0x404f83040febe95a),
    ("PreviewV0_2", "512x384-photo-blocks", 0x4052bdd01c513e9e),
    ("E", "1024x768-photo-noise", 0x404cdcd0363f2beb),
    ("ECodec", "1024x768-photo-noise", 0x4047dea51aa2d7b6),
    ("A", "1024x768-photo-noise", 0x40462562e87bc4cb),
    ("PreviewV0_2", "1024x768-photo-noise", 0x404e743b38b5e556),
];

/// The revision-7 profiles use one fixed arithmetic order, so on x86-64 their
/// scores must match the reference bit for bit. The revision-1 profiles and other
/// architectures may differ by FMA contraction, bounded here.
#[test]
fn golden_scores_match_the_research_reference() {
    let mut failures = Vec::new();
    for &(name, id, bits) in GOLDEN {
        let want = f64::from_bits(bits);
        let got = score(by_name(name), &pair(id));
        let exact = cfg!(target_arch = "x86_64") && (name == "E" || name == "ECodec");
        let ok = if exact {
            got.to_bits() == bits
        } else {
            (got - want).abs() <= 1e-6 * want.abs().max(1.0)
        };
        if !ok {
            failures.push(format!("{name} {id}: got {got:.17} want {want:.17}"));
        }
    }
    assert!(failures.is_empty(), "{} golden mismatches:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn identical_images_score_100() {
    for name in ["E", "ECodec", "A", "PreviewV0_2"] {
        for id in ["8x8-photo-identity", "65x63-edges-identity", "256x256-text-identity"] {
            assert_eq!(score(by_name(name), &pair(id)), 100.0, "{name} {id}");
        }
    }
}

/// Stronger damage of the same kind never scores higher (the revision-7 models are
/// monotone in every damage input by construction).
#[test]
fn more_noise_never_scores_higher() {
    let base = pair("256x256-photo-identity");
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let noise: Vec<i32> = (0..base.src.len())
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 3) as i32 - 1
        })
        .collect();
    for profile in [ZensimProfile::E, ZensimProfile::ECodec] {
        let mut last = f64::INFINITY;
        for amp in [0, 1, 2, 4, 8, 16, 32] {
            let dst: Vec<u8> = base
                .src
                .iter()
                .zip(&noise)
                .map(|(&v, &n)| (v as i32 + amp * n).clamp(0, 255) as u8)
                .collect();
            let p = SplitPair { id: String::new(), width: base.width, height: base.height, src: base.src.clone(), dst };
            let s = score(profile, &p);
            assert!(s <= last, "{profile}: amplitude {amp} scored {s} above {last}");
            last = s;
        }
    }
}

#[test]
fn rev7_scores_are_identical_across_simd_tiers() {
    let p = pair("333x251-photo-blur");
    for profile in [ZensimProfile::E, ZensimProfile::ECodec] {
        let mut scores = Vec::new();
        for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            scores.push((perm.label.clone(), score(profile, &p)));
        });
        let first = scores[0].1;
        for (label, s) in &scores {
            assert_eq!(s.to_bits(), first.to_bits(), "{profile}: tier {label} scored {s}, first tier {first}");
        }
    }
}

#[test]
fn rev7_scores_are_identical_across_thread_settings() {
    let p = pair("1024x768-edges-noise");
    let src = RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
    let dst = RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
    for profile in [ZensimProfile::E, ZensimProfile::ECodec] {
        let parallel = Zensim::new(profile).with_parallel(true).compute(&src, &dst).unwrap().score();
        let serial = Zensim::new(profile).with_parallel(false).compute(&src, &dst).unwrap().score();
        assert_eq!(parallel.to_bits(), serial.to_bits(), "{profile}");
    }
}

#[test]
fn latest_and_codec_target_select_the_rev7_profiles() {
    assert_eq!(ZensimProfile::latest(), ZensimProfile::E);
    assert_eq!(ZensimProfile::codec_target(), ZensimProfile::ECodec);
    assert_eq!(ZensimProfile::E.name(), "zensim-e");
    assert_eq!(ZensimProfile::ECodec.name(), "zensim-e-codec");
}
