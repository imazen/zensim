//! Owner-approved B rotation: the named/default surface must serve the exact
//! frozen bake, its declared extraction and its cached-reference path.
use sha2::{Digest, Sha256};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

const BAKE: &[u8] = include_bytes!("../weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin");

#[test]
fn default_b_is_the_frozen_diff_frag_artifact() {
    assert_eq!(
        Sha256::digest(BAKE)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        "89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e"
    );
    assert_eq!(ZensimProfile::codec_target(), ZensimProfile::B);
    assert_eq!(ZensimProfile::latest_preview(), ZensimProfile::B);
    assert_eq!(ZensimProfile::B.name(), "zensim-b");
    assert!(ZensimProfile::B.jnd_map().is_none());
}

#[test]
fn named_default_matches_bake_pixels_and_reference_cache() {
    let model = zenpredict::Model::from_bytes(BAKE).expect("frozen bake parses");
    assert_eq!(model.caller_input_width(), 820);
    assert_eq!(zensim::declared_feature_ids(&model).unwrap().len(), 420);
    let mut dynamic = BakeScorer::new(&model)
        .expect("Rev5 bake serves without an environment pin")
        .with_parallel(false);
    let named = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
    for (w, h) in [(1usize, 1usize), (31, 17), (64, 64), (97, 63), (257, 129)] {
        let mut src = Vec::with_capacity(w * h);
        let mut dst = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let a = ((x * 37 + y * 53 + x * y * 7) % 256) as u8;
                let p = [a, a.wrapping_mul(3), a.wrapping_add(81)];
                src.push(p);
                dst.push([p[0] & 0xf0, p[1].saturating_sub(11), p[2] & 0xfc]);
            }
        }
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let expected = dynamic.compute(&s, &d, None).expect("dynamic score");
        let actual = named.compute(&s, &d).expect("named score");
        assert_eq!(
            actual.score().to_bits(),
            expected.score().to_bits(),
            "{w}x{h}"
        );
        let reference = named.precompute_reference(&s).expect("reference cache");
        let cached = named
            .compute_with_ref(&reference, &d)
            .expect("cached score");
        assert_eq!(
            cached.score().to_bits(),
            actual.score().to_bits(),
            "cached {w}x{h}"
        );
        assert_eq!(named.compute(&s, &s).expect("identity").score(), 100.0);
    }
}

#[test]
fn b_revision_is_model_bound_across_process_defaults() {
    const CHILD: &str = "ZENSIM_B_BINDING_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut baseline = None;
        let mut baseline_values: Option<Vec<String>> = None;
        for revision in ["5", "1", "2", "3", "4"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "b_revision_is_model_bound_across_process_defaults",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("ZENSIM_FORMULA_REV", revision)
                .output()
                .expect("revision child");
            assert!(
                output.status.success(),
                "revision {revision}: {}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8(output.stdout).unwrap();
            let digest = stdout
                .lines()
                .find(|line| line.starts_with("B_DIGEST="))
                .expect("child digest")
                .to_owned();
            let values: Vec<String> = stdout
                .lines()
                .filter(|line| line.starts_with("B_VALUES="))
                .map(str::to_owned)
                .collect();
            if let Some(expected) = &baseline {
                if &digest != expected {
                    let differences: Vec<_> = values
                        .iter()
                        .zip(baseline_values.as_ref().unwrap())
                        .enumerate()
                        .flat_map(|(cell, (a, b))| {
                            a.split_whitespace()
                                .zip(b.split_whitespace())
                                .enumerate()
                                .filter(|(_, (a, b))| a != b)
                                .map(move |(id, (a, b))| {
                                    format!("cell {cell} field {id}: {a} != {b}")
                                })
                        })
                        .take(30)
                        .collect();
                    panic!("process revision {revision}: {differences:?}");
                }
            } else {
                baseline = Some(digest);
                baseline_values = Some(values);
            }
        }
        return;
    }
    let named = Zensim::new(ZensimProfile::B).with_parallel(false);
    let mut hash = Sha256::new();
    for (w, h) in [(1usize, 1usize), (31, 17), (97, 151), (257, 349)] {
        let src: Vec<_> = (0..w * h)
            .map(|i| {
                [
                    (i * 73 % 256) as u8,
                    (i * 17 % 256) as u8,
                    (i * 29 % 256) as u8,
                ]
            })
            .collect();
        let dst: Vec<_> = src
            .iter()
            .enumerate()
            .map(|(i, p)| [p[0] & 0xf8, p[1].wrapping_add((i % 21) as u8), p[2] & 0xe0])
            .collect();
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let result = named.compute(&s, &d).unwrap();
        println!(
            "B_VALUES={} {}",
            result.score().to_bits(),
            result
                .features()
                .iter()
                .map(|v| v.to_bits().to_string())
                .collect::<Vec<_>>()
                .join(" ")
        );
        hash.update(result.score().to_bits().to_le_bytes());
        for value in result.features() {
            hash.update(value.to_bits().to_le_bytes());
        }
        let mapped = named
            .compute_with_diffmap(
                &s,
                &d,
                zensim::DiffmapOptions {
                    include_edge_mse: true,
                    include_hf: true,
                    masking_strength: Some(4.0),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(result.score().to_bits(), mapped.score().to_bits());
        for value in mapped.diffmap() {
            hash.update(value.to_bits().to_le_bytes());
        }
        let reference = named.precompute_reference(&s).unwrap();
        let cached = named.compute_with_ref(&reference, &d).unwrap();
        assert_eq!(
            result.score().to_bits(),
            cached.score().to_bits(),
            "cached {w}x{h}"
        );
        let parallel = Zensim::new(ZensimProfile::B).compute(&s, &d).unwrap();
        assert_eq!(
            result.score().to_bits(),
            parallel.score().to_bits(),
            "parallel {w}x{h}"
        );
    }
    println!("B_DIGEST={:?}", hash.finalize());
}

#[test]
fn b_preserves_map_and_strip_scores_and_checks_reference_conversion() {
    const CHILD: &str = "ZENSIM_B_REFUSAL_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "b_preserves_map_and_strip_scores_and_checks_reference_conversion",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("ZENSIM_FORMULA_REV", "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    let src = vec![[63u8, 91, 127]; 96 * 96];
    let dst = vec![[60u8, 90, 120]; 96 * 96];
    let s = RgbSlice::new(&src, 96, 96);
    let d = RgbSlice::new(&dst, 96, 96);
    let b = Zensim::new(ZensimProfile::B).with_parallel(false);
    let legacy = Zensim::new(ZensimProfile::PreviewV0_2);
    let cache = legacy.precompute_reference(&s).unwrap();
    assert!(b.compute_with_ref(&cache, &d).is_err());
    let scalar = b.compute(&s, &d).unwrap().score().to_bits();
    assert_eq!(
        scalar,
        b.compute_streaming_strips_default(&s, &d)
            .unwrap()
            .score()
            .to_bits()
    );
    let reference = b.precompute_reference(&s).unwrap();
    assert_eq!(
        scalar,
        b.compute_with_ref_streaming_strips_default(&reference, &d)
            .unwrap()
            .score()
            .to_bits()
    );
    let mapped = b
        .compute_with_diffmap(&s, &d, zensim::DiffmapWeighting::Trained)
        .unwrap();
    assert_eq!(scalar, mapped.score().to_bits());
    assert_eq!(mapped.diffmap().len(), 96 * 96);
    assert!(mapped.diffmap().iter().all(|v| v.is_finite()));
    let error = b.compute_extended_features(&s, &d).unwrap_err();
    assert!(error.to_string().contains("basic + peaks + v2"), "{error}");
}

#[test]
fn b_planar_reference_uses_the_same_color_conversion() {
    let (w, h) = (97usize, 65usize);
    let pixels: Vec<[f32; 4]> = (0..w * h)
        .map(|i| {
            [
                (i % 233) as f32 / 233.0,
                (i * 7 % 239) as f32 / 239.0,
                (i * 11 % 251) as f32 / 251.0,
                1.0,
            ]
        })
        .collect();
    let distorted: Vec<[f32; 4]> = pixels
        .iter()
        .map(|p| [p[0] * 0.93, p[1] * 0.97, p[2] * 0.89, 1.0])
        .collect();
    let src = zensim::StridedBytes::try_new(
        bytemuck::cast_slice(&pixels),
        w,
        h,
        w * 16,
        zensim::PixelFormat::LinearF32Rgba,
    )
    .unwrap();
    let dst = zensim::StridedBytes::try_new(
        bytemuck::cast_slice(&distorted),
        w,
        h,
        w * 16,
        zensim::PixelFormat::LinearF32Rgba,
    )
    .unwrap();
    let planes: [Vec<f32>; 3] = core::array::from_fn(|ch| pixels.iter().map(|p| p[ch]).collect());
    let b = Zensim::new(ZensimProfile::B).with_parallel(false);
    let reference = b
        .precompute_reference_linear_planar([&planes[0], &planes[1], &planes[2]], w, h, w)
        .unwrap();
    let expected = b.compute(&src, &dst).unwrap().score().to_bits();
    assert_eq!(
        expected,
        b.compute_with_ref(&reference, &dst)
            .unwrap()
            .score()
            .to_bits()
    );
    let stride = w + 7;
    let dist_planes: [Vec<f32>; 3] = core::array::from_fn(|ch| {
        let mut plane = vec![f32::NAN; stride * h];
        for y in 0..h {
            for x in 0..w {
                plane[y * stride + x] = distorted[y * w + x][ch];
            }
        }
        plane
    });
    let mapped = b
        .compute_with_ref_and_diffmap_linear_planar(
            &reference,
            [&dist_planes[0], &dist_planes[1], &dist_planes[2]],
            w,
            h,
            stride,
            zensim::DiffmapOptions::default(),
        )
        .unwrap();
    assert_eq!(expected, mapped.score().to_bits());
    assert_eq!(mapped.diffmap().len(), w * h);
    assert!(mapped.diffmap().iter().all(|v| v.is_finite()));
}
