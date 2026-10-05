//! R5STEER2 diagnostic only. Calls the production pixel/feature forwards;
//! no alternate model, feature extractor or arithmetic implementation.
//! M2 has a finite-secant feature sensitivity, not an analytic pixel gradient.
//! Compare its probes at smaller steps and along actual pixel-edit directions.

use super::sha;
use serde::Deserialize;
use serde_json::{Value, json};
use zensim::{AlphaMode, BakeScorer, PixelFormat, RgbSlice, StridedBytes};

#[derive(Deserialize)]
struct Job {
    name: String,
    reference: String,
    distorted: String,
    models: Vec<String>,
    weights: Option<Vec<f64>>,
    block: usize,
    baseline_report: String,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn vector_error(a: &[f64], b: &[f64]) -> Value {
    let aa = dot(a, a);
    let bb = dot(b, b);
    let dd: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    json!({"relative_l2": dd.sqrt() / aa.sqrt().max(1e-30),
        "cosine": if aa == 0.0 && bb == 0.0 {1.0} else {dot(a,b) / (aa*bb).sqrt().max(1e-30)},
        "reference_l2": aa.sqrt(), "probe_l2": bb.sqrt()})
}

fn feature_fd(
    scorer: &mut BakeScorer<'_>,
    features: &[f64],
    dims: (usize, usize),
    factor: f64,
) -> Vec<f64> {
    let mut probe = features.to_vec();
    features
        .iter()
        .enumerate()
        .map(|(k, &v)| {
            // Independent probe loop, through the complete production forward.
            // factor=1e-3 exactly reproduces the owner's max(abs(f)*1e-3,1e-5).
            let eps = (v.abs() * factor).max(1e-5 * (factor / 1e-3));
            probe[k] = v + eps;
            let up = scorer
                .score_features(&probe, dims.0 as u32, dims.1 as u32, None)
                .unwrap();
            probe[k] = v - eps;
            let down = scorer
                .score_features(&probe, dims.0 as u32, dims.1 as u32, None)
                .unwrap();
            probe[k] = v;
            (up - down) / (2.0 * eps)
        })
        .collect()
}

fn source16(pixels: &[[u16; 4]], w: usize, h: usize) -> StridedBytes<'_> {
    StridedBytes::try_with_alpha_mode(
        bytemuck::cast_slice(pixels),
        w,
        h,
        w * 8,
        PixelFormat::Srgb16Rgba,
        AlphaMode::Opaque,
    )
    .unwrap()
}

pub(super) fn run(input: &str, output: &str) {
    assert!(
        !std::path::Path::new(output).exists(),
        "new output required"
    );
    let job_bytes = std::fs::read(input).unwrap();
    let job: Job = serde_json::from_slice(&job_bytes).unwrap();
    // Match this example's ordinary panel decoder; pixel hashes below bind it.
    let reference_image = image::open(&job.reference).unwrap().to_rgb8();
    let distorted_image = image::open(&job.distorted).unwrap().to_rgb8();
    let (w, h) = (
        reference_image.width() as usize,
        reference_image.height() as usize,
    );
    assert_eq!(reference_image.dimensions(), distorted_image.dimensions());
    let rpx: Vec<[u8; 3]> = reference_image.pixels().map(|p| p.0).collect();
    let dpx: Vec<[u8; 3]> = distorted_image.pixels().map(|p| p.0).collect();
    let blobs: Vec<_> = job
        .models
        .iter()
        .map(|p| std::fs::read(p).unwrap())
        .collect();
    let models: Vec<_> = blobs
        .iter()
        .map(|b| zenpredict::Model::from_bytes(b).unwrap())
        .collect();
    let mut pixel = BakeScorer::ensemble(&models, job.weights.as_deref())
        .unwrap()
        .with_parallel(false);
    let mut feature = BakeScorer::ensemble(&models, job.weights.as_deref())
        .unwrap()
        .with_parallel(false);
    let mut steering = BakeScorer::ensemble(&models, job.weights.as_deref())
        .unwrap()
        .with_parallel(false);
    let rs = RgbSlice::new(&rpx, w, h);
    let ds = RgbSlice::new(&dpx, w, h);
    let mut worker = steering.prepare_steering(&rs, 1).unwrap();
    let scored = worker.compute(&ds, None).unwrap();
    let base = pixel.compute(&rs, &ds, None).unwrap();
    assert_eq!(
        scored.result().features(),
        base.features(),
        "prepared/forward feature parity"
    );
    assert_eq!(scored.result().score().to_bits(), base.score().to_bits());
    let f = base.features().to_vec();
    let s = scored.sensitivities().to_vec();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(&job.baseline_report).unwrap()).unwrap();
    let recorded_score = reference["base_score"].as_f64().unwrap();
    // Allow decimal-parser rounding only when reading external JSON. The
    // prepared/pixel forwards above must remain bit-identical to each other.
    assert!(
        (recorded_score - base.score()).abs() <= 4.0 * f64::EPSILON * base.score().abs().max(1.0)
    );
    assert_eq!(
        reference["reference_pixels_sha256"],
        sha(bytemuck::cast_slice(&rpx))
    );
    assert_eq!(
        reference["distorted_pixels_sha256"],
        sha(bytemuck::cast_slice(&dpx))
    );
    let factors = [1e-2, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5];
    let gradients: Vec<_> = factors
        .iter()
        .map(|&factor| feature_fd(&mut feature, &f, (w, h), factor))
        .collect();
    assert_eq!(
        s, gradients[1],
        "same-step independent forward probes reproduce M2 sensitivities"
    );
    let mut gradient_rows: Vec<_> = factors
        .iter()
        .zip(&gradients)
        .map(|(&factor, g)| json!({"factor":factor,"error_vs_m2":vector_error(&s,g),"gradient":g}))
        .collect();

    let bx = w.div_ceil(job.block);
    let by = h.div_ceil(job.block);
    let mut full_deltas = Vec::new();
    let mut deltas = Vec::new();
    let mut bounds = Vec::new();
    let mut scratch = dpx.clone();
    for y in 0..by {
        for x in 0..bx {
            let rect = [
                x * job.block,
                y * job.block,
                ((x + 1) * job.block).min(w),
                ((y + 1) * job.block).min(h),
            ];
            for row in rect[1]..rect[3] {
                scratch[row * w + rect[0]..row * w + rect[2]]
                    .copy_from_slice(&rpx[row * w + rect[0]..row * w + rect[2]]);
            }
            let result = pixel
                .compute(&rs, &RgbSlice::new(&scratch, w, h), None)
                .unwrap();
            let df: Vec<_> = result
                .features()
                .iter()
                .zip(&f)
                .map(|(a, b)| a - b)
                .collect();
            deltas.push(result.score() - base.score());
            full_deltas.push(df);
            bounds.push(rect);
            for row in rect[1]..rect[3] {
                scratch[row * w + rect[0]..row * w + rect[2]]
                    .copy_from_slice(&dpx[row * w + rect[0]..row * w + rect[2]]);
            }
        }
    }
    let linear: Vec<_> = full_deltas.iter().map(|df| dot(&s, df)).collect();
    let m2 = super::spearman(&linear, &deltas);
    // serde_json's ordinary decimal parser can differ by one f64 ULP from
    // the serialized rank value. Pixel/features and model probes stay exact.
    assert!((m2 - reference["m2"].as_f64().unwrap()).abs() <= 4.0 * f64::EPSILON);
    for (row, g) in gradient_rows.iter_mut().zip(&gradients) {
        let pred: Vec<_> = full_deltas.iter().map(|df| dot(g, df)).collect();
        row["m2_full_repairs"] = json!(super::spearman(&pred, &deltas));
    }
    let worst = linear
        .iter()
        .zip(&deltas)
        .enumerate()
        .max_by(|(_, (a, b)), (_, (c, d))| (*a - *b).abs().total_cmp(&(*c - *d).abs()))
        .unwrap()
        .0;
    let best = deltas
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .unwrap()
        .0;
    let mut selected = vec![worst, best, bounds.len() / 2];
    selected.sort_unstable();
    selected.dedup();
    let mut probes = Vec::new();
    for &i in &selected {
        probes.push((format!("block-{i}"), full_deltas[i].clone()));
        let rect = bounds[i];
        let mut found = None;
        'search: for y in rect[1]..rect[3] {
            for x in rect[0]..rect[2] {
                for ch in 0..3 {
                    let idx = y * w + x;
                    if dpx[idx][ch] > 0 && dpx[idx][ch] < 255 && rpx[idx][ch] != dpx[idx][ch] {
                        found = Some((idx, ch));
                        break 'search;
                    }
                }
            }
        }
        if let Some((idx, ch)) = found {
            scratch[idx][ch] += 1;
            let up = pixel
                .compute(&rs, &RgbSlice::new(&scratch, w, h), None)
                .unwrap();
            scratch[idx][ch] -= 2;
            let down = pixel
                .compute(&rs, &RgbSlice::new(&scratch, w, h), None)
                .unwrap();
            scratch[idx] = dpx[idx];
            let df: Vec<_> = up
                .features()
                .iter()
                .zip(down.features())
                .map(|(a, b)| (a - b) / 2.0)
                .collect();
            probes.push((format!("pixel-{idx}-ch{ch}"), df));
        }
    }
    let mut directional = Vec::new();
    for eps in [1.0, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003] {
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut details = Vec::new();
        for (name, df) in &probes {
            let plus: Vec<_> = f.iter().zip(df).map(|(x, d)| x + eps * d).collect();
            let minus: Vec<_> = f.iter().zip(df).map(|(x, d)| x - eps * d).collect();
            let up = feature
                .score_features(&plus, w as u32, h as u32, None)
                .unwrap();
            let down = feature
                .score_features(&minus, w as u32, h as u32, None)
                .unwrap();
            let observed = (up - down) / (2.0 * eps);
            let predicted = dot(&s, df);
            a.push(predicted);
            b.push(observed);
            details.push(
                json!({"probe":name,"predicted":predicted,"central_fd":observed,
                "relative_error":(predicted-observed).abs()/observed.abs().max(1e-30),
                "score_span":up-down}),
            );
        }
        directional.push(json!({"eps":eps,"error":vector_error(&a,&b),"probes":details}));
    }
    let mut curvature = Vec::new();
    for t in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0] {
        let observed: Vec<_> = full_deltas
            .iter()
            .map(|df| {
                let probe: Vec<_> = f.iter().zip(df).map(|(x, d)| x + t * d).collect();
                feature
                    .score_features(&probe, w as u32, h as u32, None)
                    .unwrap()
                    - base.score()
            })
            .collect();
        let predicted: Vec<_> = linear.iter().map(|v| t * v).collect();
        if t == 1.0 {
            assert_eq!(
                observed, deltas,
                "pixel/feature forward parity for full repairs"
            );
        }
        curvature.push(
            json!({"fraction":t,"m2":super::spearman(&predicted,&observed),
            "error":vector_error(&predicted,&observed)}),
        );
    }

    // Sub-RGB8-LSB probes in the production SDR sRGB16 source path. Replicate
    // each byte to u16 (v*257), explicit opaque alpha and byte stride w*8.
    // Its base may differ from the RGB8 LUT conversion; measure that drift.
    let r16: Vec<[u16; 4]> = rpx
        .iter()
        .map(|p| {
            [
                u16::from(p[0]) * 257,
                u16::from(p[1]) * 257,
                u16::from(p[2]) * 257,
                65535,
            ]
        })
        .collect();
    let d16: Vec<[u16; 4]> = dpx
        .iter()
        .map(|p| {
            [
                u16::from(p[0]) * 257,
                u16::from(p[1]) * 257,
                u16::from(p[2]) * 257,
                65535,
            ]
        })
        .collect();
    let rs16 = source16(&r16, w, h);
    let base16 = pixel.compute(&rs16, &source16(&d16, w, h), None).unwrap();
    let s16 = feature
        .score_features_fd_gradient(base16.features(), w as u32, h as u32, None)
        .unwrap();
    let mut pixel_rows = Vec::new();
    for step in [1_u16, 4, 16, 64, 256] {
        let mut predicted = Vec::new();
        let mut observed = Vec::new();
        let mut details = Vec::new();
        for &i in &selected {
            for only_one in [false, true] {
                let rect = bounds[i];
                let mut up_px = d16.clone();
                let mut down_px = d16.clone();
                let mut changed = 0;
                'pixels: for y in rect[1]..rect[3] {
                    for x in rect[0]..rect[2] {
                        let idx = y * w + x;
                        for ch in 0..3 {
                            let v = d16[idx][ch];
                            if v >= step && v <= 65535 - step && r16[idx][ch] != v {
                                let sign = if r16[idx][ch] > v { 1_i32 } else { -1 };
                                up_px[idx][ch] = (i32::from(v) + sign * i32::from(step)) as u16;
                                down_px[idx][ch] = (i32::from(v) - sign * i32::from(step)) as u16;
                                changed += 1;
                                if only_one {
                                    break 'pixels;
                                }
                            }
                        }
                    }
                }
                if changed == 0 {
                    continue;
                }
                let up = pixel.compute(&rs16, &source16(&up_px, w, h), None).unwrap();
                let down = pixel
                    .compute(&rs16, &source16(&down_px, w, h), None)
                    .unwrap();
                let df: Vec<_> = up
                    .features()
                    .iter()
                    .zip(down.features())
                    .map(|(a, b)| a - b)
                    .collect();
                let eps = f64::from(step) / 65535.0;
                let actual = (up.score() - down.score()) / (2.0 * eps);
                let pred = dot(&s16, &df) / (2.0 * eps);
                predicted.push(pred);
                observed.push(actual);
                details.push(
                    json!({"block":i,"single_pixel":only_one,"changed_channels":changed,
                    "predicted":pred,"pixel_central_fd":actual,"score_span":up.score()-down.score(),
                    "relative_error":(pred-actual).abs()/actual.abs().max(1e-30)}),
                );
            }
        }
        pixel_rows.push(
            json!({"code_step":step,"normalized_eps":f64::from(step)/65535.0,
            "error":vector_error(&predicted,&observed),"probes":details}),
        );
    }
    let report = json!({"schema":"r5steer2-gradient-v1","name":job.name,
        "revision":std::env::var("ZENSIM_FORMULA_REV").unwrap(),"job_sha256":sha(&job_bytes),
        "m2_sensitivity_kind":"production central feature secant; no analytic pixel gradient",
        "models":job.models.iter().zip(&blobs).map(|(p,b)| json!({"path":p,"sha256":sha(b)})).collect::<Vec<_>>(),
        "width":w,"height":h,"block":job.block,"base_score":base.score(),"m2":m2,
        "prepared_forward_features_bit_identical":true,"pixel_feature_forward_scores_bit_identical":true,
        "gradient_steps":gradient_rows,"selected_blocks":selected,"directional_feature_probes":directional,
        "finite_repair_curvature":curvature,"pixel16_probes":pixel_rows,
        "rgb16_base_score":base16.score(),"rgb16_minus_rgb8_score":base16.score()-base.score(),
        "rgb16_feature_error_vs_rgb8":vector_error(&f,base16.features())});
    std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    println!(
        "{}: M2 {m2:.12}; gradient and pixel checks complete",
        job.name
    );
}
