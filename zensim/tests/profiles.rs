//! Contracts of the shipped profiles: identity, monotone damage, tier and thread
//! invariance of the revision-7 profiles, and golden scores.
//!
//! The golden scores were produced by the research crate this one was cut from
//! (imazen/zensimx, `split_parity_golden`, 2026-10-28) on the pairs in
//! `examples/split_parity/pairs.rs`; the slim crate reproduced all 6,930 values of
//! that run bit for bit before these were copied here. The adopted `A` (research
//! `F`) values come from the same example run on 2026-11-04 with the adopted bake as its general model
//! (its stripped copy scored identically on every surface).
//!
//! Run the tier test alone (`--test-threads=1`) when debugging: it disables SIMD
//! tiers process-wide while it runs. Every scoring test holds
//! `lock_token_testing()` so legacy goldens cannot observe a partial tier switch.

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
    Zensim::new(profile)
        .compute(&src, &dst)
        .expect("compute")
        .score()
}

fn by_name(name: &str) -> ZensimProfile {
    match name {
        "A" => ZensimProfile::A,
        "PreviewV0_1" => ZensimProfile::PreviewV0_1,
        "PreviewV0_2" => ZensimProfile::PreviewV0_2,
        _ => unreachable!(),
    }
}

/// (profile, pair, score bits) from the research reference run.
const GOLDEN: &[(&str, &str, u64)] = &[
    ("A", "64x64-photo-noise", 0x405267c1ff28136f),
    ("A", "64x64-edges-blur", 0x40335ff4af4a7e65),
    ("A", "65x63-text-blocks", 0xc0346f35499e665f),
    ("A", "128x96-photo-posterize", 0x40526de1012462ad),
    ("A", "200x150-edges-shift", 0x4056b61031aec2ec),
    ("A", "200x150-photo-blur", 0x404c3f1d5a752927),
    ("A", "256x256-text-noise", 0x40538a69e41bd571),
    ("A", "256x256-photo-pixels", 0x4058d5c1fb0d1fb8),
    ("A", "333x251-edges-blocks", 0xc04822574b9489e7),
    ("A", "333x251-photo-identity", 0x4059000000000000),
    ("A", "16x16-photo-noise", 0x405472ba34b9164c),
    ("A", "31x17-text-shift", 0x40562d2f88e71244),
    ("A", "512x384-photo-blocks", 0x404dcd8ed6bee9bc),
    ("A", "1024x768-photo-noise", 0x404a54e9256bcb0f),
    ("PreviewV0_2", "64x64-photo-noise", 0x4053c2809278bdab),
    ("PreviewV0_2", "64x64-edges-blur", 0x402fe8667b189400),
    ("PreviewV0_2", "65x63-text-blocks", 0xc042853a97107170),
    ("PreviewV0_2", "128x96-photo-posterize", 0x40537ea0198c0b60),
    ("PreviewV0_2", "200x150-edges-shift", 0x40553e6ff2624cbd),
    ("PreviewV0_2", "200x150-photo-blur", 0x40536d00e6303f17),
    ("PreviewV0_2", "256x256-text-noise", 0x4054b03976a363b0),
    ("PreviewV0_2", "256x256-photo-pixels", 0x4058dde8b46bdf4a),
    ("PreviewV0_2", "333x251-edges-blocks", 0xc061ff02ae3ac7b4),
    ("PreviewV0_2", "333x251-photo-identity", 0x4059000000000000),
    ("PreviewV0_2", "16x16-photo-noise", 0x405520a00c68709c),
    ("PreviewV0_2", "31x17-text-shift", 0x40588bb8484bed5e),
    ("PreviewV0_2", "512x384-photo-blocks", 0x4052bdd01c513e9e),
    ("PreviewV0_2", "1024x768-photo-noise", 0x404e743b38b5e556),
];

/// The revision-7 profiles use one fixed arithmetic order, so on x86-64 their
/// scores must match the reference bit for bit. The revision-1 profiles and other
/// architectures may differ by FMA contraction, bounded here.
#[test]
fn golden_scores_match_the_research_reference() {
    let _token = archmage::testing::lock_token_testing();
    let mut failures = Vec::new();
    for &(name, id, bits) in GOLDEN {
        let want = f64::from_bits(bits);
        let got = score(by_name(name), &pair(id));
        let exact = cfg!(target_arch = "x86_64") && name == "A";
        let ok = if exact {
            got.to_bits() == bits
        } else {
            (got - want).abs() <= 1e-6 * want.abs().max(1.0)
        };
        if !ok {
            failures.push(format!("{name} {id}: got {got:.17} want {want:.17}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} golden mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn identical_images_score_100() {
    let _token = archmage::testing::lock_token_testing();
    for name in ["A", "PreviewV0_1", "PreviewV0_2"] {
        for id in [
            "8x8-photo-identity",
            "65x63-edges-identity",
            "256x256-text-identity",
        ] {
            assert_eq!(score(by_name(name), &pair(id)), 100.0, "{name} {id}");
        }
    }
}

/// Stronger damage of the same kind never scores higher (the revision-7 models are
/// monotone in every damage input by construction).
#[test]
fn more_noise_never_scores_higher() {
    let _token = archmage::testing::lock_token_testing();
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
    {
        let profile = ZensimProfile::A;
        let mut last = f64::INFINITY;
        for amp in [0, 1, 2, 4, 8, 16, 32] {
            let dst: Vec<u8> = base
                .src
                .iter()
                .zip(&noise)
                .map(|(&v, &n)| (v as i32 + amp * n).clamp(0, 255) as u8)
                .collect();
            let p = SplitPair {
                id: String::new(),
                width: base.width,
                height: base.height,
                src: base.src.clone(),
                dst,
            };
            let s = score(profile, &p);
            assert!(
                s <= last,
                "{profile}: amplitude {amp} scored {s} above {last}"
            );
            last = s;
        }
    }
}

#[test]
fn rev7_scores_are_identical_across_simd_tiers() {
    let _token = archmage::testing::lock_token_testing();
    let p = pair("333x251-photo-blur");
    {
        let profile = ZensimProfile::A;
        let mut scores = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            scores.push((perm.label.clone(), score(profile, &p)));
        });
        assert!(!scores.is_empty(), "no tier ran: {report:?}");
        let first = scores[0].1;
        for (label, s) in &scores {
            assert_eq!(
                s.to_bits(),
                first.to_bits(),
                "{profile}: tier {label} scored {s}, first tier {first}"
            );
        }
    }
}

#[test]
fn rev7_scores_are_identical_across_thread_settings() {
    let _token = archmage::testing::lock_token_testing();
    let p = pair("1024x768-edges-noise");
    let src = RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
    let dst = RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
    {
        let profile = ZensimProfile::A;
        let parallel = Zensim::new(profile)
            .with_parallel(true)
            .compute(&src, &dst)
            .unwrap()
            .score();
        let serial = Zensim::new(profile)
            .with_parallel(false)
            .compute(&src, &dst)
            .unwrap()
            .score();
        assert_eq!(parallel.to_bits(), serial.to_bits(), "{profile}");
    }
}

/// `compute_all_features` fills every slot `compute` fills, with the same bits:
/// both extract at the profile's own formula revision.
#[cfg(feature = "training")]
#[test]
fn all_features_agree_with_compute_on_every_slot_compute_fills() {
    // The tier test disables SIMD tiers process-wide while it runs.
    let _token = archmage::testing::lock_token_testing();
    for id in [
        "31x17-photo-noise",
        "65x63-text-blocks",
        "200x150-edges-blur",
        "333x251-photo-posterize",
    ] {
        let p = pair(id);
        let src = RgbSlice::new(bytemuck::cast_slice(&p.src), p.width, p.height);
        let dst = RgbSlice::new(bytemuck::cast_slice(&p.dst), p.width, p.height);
        for profile in [ZensimProfile::A, ZensimProfile::PreviewV0_2] {
            let z = Zensim::new(profile);
            let served = z.compute(&src, &dst).unwrap();
            let all = z.compute_all_features(&src, &dst).unwrap();
            assert!(
                all.features().len() >= served.features().len(),
                "{profile} {id}"
            );
            for (i, (&s, &a)) in served.features().iter().zip(all.features()).enumerate() {
                if s != 0.0 {
                    assert_eq!(
                        s.to_bits(),
                        a.to_bits(),
                        "{profile} {id} f{i}: compute {s}, all {a}"
                    );
                }
            }
        }
    }
}

#[test]
fn latest_and_codec_target_select_the_rev7_profiles() {
    assert_eq!(ZensimProfile::latest(), ZensimProfile::A);
    assert_eq!(ZensimProfile::codec_target(), ZensimProfile::A);
    assert_eq!(ZensimProfile::A.name(), "zensim-a");
}

/// Canonical names select the adopted and earlier development models. The
/// legacy reference retains its existing cross-architecture rounding tolerance;
/// exact same-host rename parity is recorded in docs/PROFILES.md.
#[test]
fn canonical_names_select_the_shipped_models() {
    let _token = archmage::testing::lock_token_testing();
    for profile in [
        ZensimProfile::A,
        ZensimProfile::PreviewV0_1,
        ZensimProfile::PreviewV0_2,
    ] {
        assert_eq!(profile.to_string().parse(), Ok(profile));
    }
    assert_eq!("zensim-a".parse(), Ok(ZensimProfile::A));
    assert!("zensim-legacy-a".parse::<ZensimProfile>().is_err());
    assert!("zensim-f".parse::<ZensimProfile>().is_err());
    assert!("zensim-e".parse::<ZensimProfile>().is_err());
    assert!("zensim-e-codec".parse::<ZensimProfile>().is_err());
}
