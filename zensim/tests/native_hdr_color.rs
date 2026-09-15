//! Public HDR color-basis gates with independent f64 colorimetry.
#![cfg(feature = "feature-regime-v2")]
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{
    AlphaMode, ColorPrimaries, ImageSource, PixelFormat, StridedBytes, Zensim, ZensimProfile,
};

fn matrix(p: ColorPrimaries) -> [[f64; 3]; 3] {
    match p {
        ColorPrimaries::Srgb => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        ColorPrimaries::DisplayP3 => [
            [1.2249401763, -0.2249401763, 0.],
            [-0.0420569547, 1.0420569547, 0.],
            [-0.0196375546, -0.0786360456, 1.0982736001],
        ],
        ColorPrimaries::Bt2020 => [
            [1.6604910021, -0.5876411388, -0.0728498633],
            [-0.1245504745, 1.1328998971, -0.0083494226],
            [-0.0181507634, -0.100578898, 1.1187296614],
        ],
        _ => unreachable!(),
    }
}
// Materialize the independently checked f32 matrix representation before
// comparing downstream features. Higher-order moments on tiny pyramids can
// amplify sub-ULP color differences; they are not a pixel-color error bound.
fn transform(pixels: &[[f32; 4]], primaries: ColorPrimaries) -> Vec<[f32; 4]> {
    let exact = matrix(primaries);
    let m: [[f32; 3]; 3] = match primaries {
        ColorPrimaries::Srgb => [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        ColorPrimaries::DisplayP3 => [
            [1.224_940_2, -0.224_940_2, 0.],
            [-0.042_056_955, 1.042_056_9, 0.],
            [-0.019_637_555, -0.078_636_04, 1.098_273_6],
        ],
        ColorPrimaries::Bt2020 => [
            [1.660_491, -0.587_641_1, -0.072_849_9],
            [-0.124_550_5, 1.132_899_9, -0.008_349_4],
            [-0.018_151, -0.100_578_6, 1.118_729_6],
        ],
        _ => unreachable!(),
    };
    pixels
        .iter()
        .map(|p| {
            let c: [f32; 3] = core::array::from_fn(|c| {
                let actual = m[c][0] * p[0] + m[c][1] * p[1] + m[c][2] * p[2];
                let expected: f64 = (0..3).map(|i| exact[c][i] * f64::from(p[i])).sum();
                assert!(
                    (f64::from(actual) - expected).abs()
                        < 2e-6
                            * p[..3]
                                .iter()
                                .map(|v| f64::from(v.abs()))
                                .fold(1.0, f64::max)
                );
                actual
            });
            [c[0], c[1], c[2], 1.]
        })
        .collect()
}

fn source(pixels: &[[f32; 4]], w: usize, h: usize, p: ColorPrimaries) -> StridedBytes<'_> {
    StridedBytes::with_alpha_mode(
        bytemuck::cast_slice(pixels),
        w,
        h,
        w * 16,
        PixelFormat::LinearF32Rgba,
        AlphaMode::Opaque,
    )
    .with_color_primaries(p)
}
fn features(a: &impl ImageSource, b: &impl ImageSource, encoding: HdrEncoding) -> Vec<f64> {
    Zensim::new(ZensimProfile::B)
        .with_parallel(false)
        .compute_folded720_append_features_hdr(
            a,
            b,
            encoding,
            V2NewFeatureToggles::default(),
            &mut V2Scratch::new(),
        )
        .unwrap()
        .features()
        .to_vec()
}
#[test]
fn hdr_public_features_match_independent_common_primaries_without_sdr_clipping() {
    for (w, h) in [(17, 9), (97, 65)] {
        let a: Vec<[f32; 4]> = (0..w * h)
            .map(|i| {
                let t = (i * 37 % 251) as f32;
                [
                    50. + t * 2.,
                    20. + (i * 19 % 191) as f32,
                    10. + (i * 13 % 173) as f32,
                    1.,
                ]
            })
            .collect();
        let b: Vec<_> = a
            .iter()
            .enumerate()
            .map(|(i, p)| [p[0] * 0.91, p[1] + (i % 7) as f32, p[2] * 1.08, 1.])
            .collect();
        for p in [ColorPrimaries::DisplayP3, ColorPrimaries::Bt2020] {
            let ar = transform(&a, p);
            let br = transform(&b, p);
            let actual = features(
                &source(&a, w, h, p),
                &source(&b, w, h, p),
                HdrEncoding::Linear,
            );
            let expected = features(
                &source(&ar, w, h, ColorPrimaries::Srgb),
                &source(&br, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            for (id, (x, y)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    x.is_finite() && (x - y).abs() < 2e-4 * y.abs().max(1.),
                    "{p:?} {w}x{h} f{id}: {x} vs {y}"
                );
            }
            #[cfg(feature = "custom-profiles")]
            {
                let recipe = serde_json::json!({
                    "schema_hash":1, "scaler_mean":[0.], "scaler_scale":[1.],
                    "metadata":[
                        {"key":"zentrain.feature_ids","type":"utf8","text":"9"},
                        {"key":"zentrain.formula_revision","type":"utf8","text":std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_|"1".into())}
                    ],
                    "layers":[{"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-10.],"biases":[100.]}]
                });
                let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap();
                let model = zenpredict::Model::from_bytes(&bytes).unwrap();
                let mut scorer = zensim::BakeScorer::new(&model)
                    .unwrap()
                    .with_parallel(false);
                let native = scorer
                    .compute_hdr(
                        &source(&a, w, h, p),
                        &source(&b, w, h, p),
                        HdrEncoding::Linear,
                        None,
                    )
                    .unwrap();
                let common = scorer
                    .compute_hdr(
                        &source(&ar, w, h, ColorPrimaries::Srgb),
                        &source(&br, w, h, ColorPrimaries::Srgb),
                        HdrEncoding::Linear,
                        None,
                    )
                    .unwrap();
                let cached = scorer
                    .score_features(&actual, w as u32, h as u32, None)
                    .unwrap();
                assert_eq!(native.to_bits(), common.to_bits());
                assert_eq!(native.to_bits(), cached.to_bits());
            }
            assert!(actual[9] > 0., "absolute-light error was clipped away");
            let mixed = features(
                &source(&a, w, h, p),
                &source(&a, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            assert!(
                mixed[9] > 1e-6,
                "identical codes in distinct primaries were treated as identical colors"
            );
        }
    }
}

#[test]
fn pq_and_hlg_native_u16_agree_with_independent_display_light() {
    let (w, h) = (97, 65);
    let a: Vec<[u16; 4]> = (0..w * h)
        .map(|i| {
            [
                10000 + (i * 37 % 35000) as u16,
                15000 + (i * 19 % 30000) as u16,
                12000 + (i * 11 % 28000) as u16,
                65535,
            ]
        })
        .collect();
    let b: Vec<_> = a
        .iter()
        .enumerate()
        .map(|(i, p)| [p[0] - 400, p[1] + (i % 500) as u16, p[2] - 200, 65535])
        .collect();
    let lift = 0.005 + 250. * 0.005 / std::f64::consts::PI;
    for prim in [
        ColorPrimaries::Srgb,
        ColorPrimaries::DisplayP3,
        ColorPrimaries::Bt2020,
    ] {
        for encoding in [
            HdrEncoding::Pq { peak_nits: 1000. },
            HdrEncoding::Hlg {
                peak_nits: 1000.,
                ambient_lux: 5.,
            },
        ] {
            let decode = |pixels: &[[u16; 4]]| -> Vec<[f32; 4]> {
                let v: Vec<[f32; 4]> = pixels
                    .iter()
                    .map(|p| {
                        let c: [f64; 3] = core::array::from_fn(|i| f64::from(p[i]) / 65535.);
                        let rgb = match encoding {
                            HdrEncoding::Pq { .. } => c.map(|v| {
                                let t = v.powf(32. / 2523.);
                                let l = 10000.
                                    * ((t - 3424. / 4096.).max(0.)
                                        / (2413. / 128. - 2392. / 128. * t))
                                        .powf(16384. / 2610.);
                                l.min(1000.) + lift
                            }),
                            HdrEncoding::Hlg { .. } => {
                                let c = c.map(|v| {
                                    if v <= 0.5 {
                                        v * v / 3.
                                    } else {
                                        ((v - 0.55991073) / 0.17883277).exp() / 12.
                                            + (1. - 4. * 0.17883277) / 12.
                                    }
                                });
                                let y = match prim {
                                    ColorPrimaries::Srgb => {
                                        [0.2126390059, 0.7151686788, 0.0721923154]
                                    }
                                    ColorPrimaries::DisplayP3 => {
                                        [0.2289745641, 0.6917385218, 0.0792869141]
                                    }
                                    _ => [0.2627, 0.6780, 0.0593],
                                };
                                let l = (0..3).map(|i| y[i] * c[i]).sum::<f64>();
                                c.map(|v| 1000. * l.powf(0.2) * v + lift)
                            }
                            _ => unreachable!(),
                        };
                        [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 1.]
                    })
                    .collect();
                transform(&v, prim)
            };
            let ar = decode(&a);
            let br = decode(&b);
            let native = |p: &[[u16; 4]]| {
                features(
                    &StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(&a),
                        w,
                        h,
                        w * 8,
                        PixelFormat::Srgb16Rgba,
                        AlphaMode::Opaque,
                    )
                    .with_color_primaries(prim),
                    &StridedBytes::with_alpha_mode(
                        bytemuck::cast_slice(p),
                        w,
                        h,
                        w * 8,
                        PixelFormat::Srgb16Rgba,
                        AlphaMode::Opaque,
                    )
                    .with_color_primaries(prim),
                    encoding,
                )
            };
            let actual = native(&b);
            let to_float = |pixels: &[[u16; 4]]| -> Vec<[f32; 4]> {
                pixels
                    .iter()
                    .map(|p| {
                        [
                            f32::from(p[0]) * (1.0 / 65535.0),
                            f32::from(p[1]) * (1.0 / 65535.0),
                            f32::from(p[2]) * (1.0 / 65535.0),
                            1.,
                        ]
                    })
                    .collect()
            };
            let (af, bf) = (to_float(&a), to_float(&b));
            let float_codes =
                features(&source(&af, w, h, prim), &source(&bf, w, h, prim), encoding);
            assert_eq!(
                actual, float_codes,
                "native u16 vs float code features {encoding:?} {prim:?}"
            );
            let expected = features(
                &source(&ar, w, h, ColorPrimaries::Srgb),
                &source(&br, w, h, ColorPrimaries::Srgb),
                HdrEncoding::Linear,
            );
            for (id, (x, y)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    x.is_finite() && (x - y).abs() < 1e-3 * y.abs().max(1.),
                    "{prim:?} {encoding:?} f{id}: {x} vs {y}"
                );
            }
        }
    }
}

#[test]
fn hdr_invalid_display_parameters_refuse_before_scoring() {
    let pixels = vec![[1., 2., 3., 1.]; 16 * 16];
    let src = source(&pixels, 16, 16, ColorPrimaries::Bt2020);
    for encoding in [
        HdrEncoding::Pq { peak_nits: 0. },
        HdrEncoding::Pq {
            peak_nits: f32::NAN,
        },
        HdrEncoding::Hlg {
            peak_nits: 1000.,
            ambient_lux: -1.,
        },
        HdrEncoding::Hlg {
            peak_nits: f32::INFINITY,
            ambient_lux: 5.,
        },
    ] {
        assert!(
            Zensim::new(ZensimProfile::B)
                .compute_folded720_append_features_hdr(
                    &src,
                    &src,
                    encoding,
                    V2NewFeatureToggles::default(),
                    &mut V2Scratch::new()
                )
                .is_err()
        );
    }
}

#[test]
fn public_wide_extraction_refuses_mixed_formula_revisions() {
    let p = vec![[20., 40., 10., 1.]; 17 * 9];
    let src = source(&p, 17, 9, ColorPrimaries::Bt2020);
    let current = zensim::feature_v2::active_formula_revision();
    let toggles = V2NewFeatureToggles {
        formula_revision: if current == zensim::feature_v2::FormulaRevision::Rev3 {
            zensim::feature_v2::FormulaRevision::Rev1
        } else {
            zensim::feature_v2::FormulaRevision::Rev3
        },
        ..V2NewFeatureToggles::default()
    };
    let z = Zensim::new(ZensimProfile::B);
    assert!(
        z.compute_folded720_append_features_hdr(
            &src,
            &src,
            HdrEncoding::Linear,
            toggles,
            &mut V2Scratch::new()
        )
        .is_err()
    );
}
