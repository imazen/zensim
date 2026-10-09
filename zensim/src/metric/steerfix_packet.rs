// Caller-selected engineering packet. Only the explicit instrument cfg compiles
// this module; ordinary all-feature tests never open these external pixels.
use super::*;
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

mod stats {
    include!("../../examples/support/coherence_stats.rs");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Objective {
    Served,
    PreFloor,
    SmoothFloor,
}

// The head/pin/winsor and feature transforms still execute in the native owner.
// Diagnostic calibration is detached only on this test-owned scorer, restored
// even on an error; every shipping scorer has Objective::Served.
pub(super) fn diagnostic_forward(
    scorer: &mut BakeScorer<'_>,
    features: &[f64],
    w: u32,
    h: u32,
    codec: Option<&str>,
) -> Result<f64, ZensimError> {
    assert!(scorer.members.is_empty() && scorer.disposition.is_none() && codec.is_none());
    let objective = scorer.diagnostic_objective;
    let metadata = Arc::clone(&scorer.metadata);
    scorer.diagnostic_objective = Objective::Served;
    let md = Arc::make_mut(&mut scorer.metadata);
    md.output_spline = None;
    md.per_codec_calibration = None;
    let raw = scorer.score_features(features, w, h, codec);
    scorer.metadata = metadata;
    scorer.diagnostic_objective = objective;
    let raw = raw?;
    Ok(match objective {
        Objective::Served => unreachable!(),
        Objective::PreFloor => raw,
        Objective::SmoothFloor => {
            let spline = scorer
                .metadata
                .output_spline
                .as_ref()
                .expect("smooth-floor needs spline");
            let span = spline.ys[spline.ys.len() - 1] - spline.ys[0];
            assert!(span > 0.0 && spline.derivs[0] > 0.0);
            let floor = spline.ys[0] - span;
            let linear = spline.ys[0] + spline.derivs[0] * (raw - spline.xs[0]);
            if raw <= spline.xs[0] && linear < floor {
                // C1 at the old floor crossing, strictly increasing beneath it;
                // span is the declared dial range, never fit to these cases.
                floor + span * ((linear - floor) / span).asinh()
            } else {
                crate::score_math::pchip_eval_capped(raw, &spline.xs, &spline.ys, &spline.derivs)
            }
        }
    })
}

#[derive(Deserialize)]
struct Job {
    cases: Vec<Case>,
    output: String,
    engine: bool,
}
#[derive(Deserialize)]
struct Case {
    key: String,
    reference: String,
    distorted: String,
    reference_sha256: String,
    distorted_sha256: String,
    reference_pixels_sha256: String,
    distorted_pixels_sha256: String,
    model: String,
    model_sha256: String,
    objective: Objective,
    block: usize,
}
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn image(path: &str, expected: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let bytes = std::fs::read(path).expect("admitted engineering pixel file");
    assert_eq!(sha(&bytes), expected);
    let img = image::load_from_memory(&bytes).unwrap().to_rgb8();
    (
        img.pixels().map(|p| p.0).collect(),
        img.width() as usize,
        img.height() as usize,
    )
}
fn owner(model: &crate::mlp::Model, objective: Objective) -> BakeScorer<'_> {
    let mut s = BakeScorer::new(model).unwrap().with_parallel(false);
    s.diagnostic_objective = objective;
    s
}

#[test]
fn engineering_packet() {
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
    let _token = archmage::testing::lock_token_testing();
    let path =
        std::env::var("STEERFIX_PACKET").expect("recipe selects an explicit engineering packet");
    let bytes = std::fs::read(&path).unwrap();
    let job: Job = serde_json::from_slice(&bytes).unwrap();
    assert!(
        !std::path::Path::new(&job.output).exists(),
        "fresh output required"
    );
    assert_eq!(
        job.engine,
        std::env::var("ZENSIM_NEIGHBOUR_EXACT").as_deref() == Ok("1")
    );
    let mut summary = Vec::new();
    for case in job.cases {
        let blob = std::fs::read(&case.model).unwrap();
        assert_eq!(sha(&blob), case.model_sha256);
        let model = crate::mlp::Model::from_bytes(&blob).unwrap();
        let (r, w, h) = image(&case.reference, &case.reference_sha256);
        let (d, dw, dh) = image(&case.distorted, &case.distorted_sha256);
        assert_eq!((w, h), (dw, dh));
        assert_eq!(sha(bytemuck::cast_slice(&r)), case.reference_pixels_sha256);
        assert_eq!(sha(bytemuck::cast_slice(&d)), case.distorted_pixels_sha256);
        let rs = crate::RgbSlice::new(&r, w, h);
        let ds = crate::RgbSlice::new(&d, w, h);
        let mut steering = owner(&model, case.objective);
        let mut worker = steering.prepare_steering(&rs, 1).unwrap();
        let mapped = worker.compute(&ds, None).unwrap();
        let mut pixel = owner(&model, case.objective);
        let base = pixel.compute(&rs, &ds, None).unwrap();
        assert_eq!(mapped.result().score().to_bits(), base.score().to_bits());
        assert_eq!(mapped.result().features(), base.features());
        let f = base.features();
        let s = mapped.sensitivities();
        // Independent feature probes on the same served/diagnostic forward.
        let mut feature = owner(&model, case.objective);
        let mut probe = f.to_vec();
        for (k, &v) in f.iter().enumerate() {
            let eps = (v.abs() * 1e-3).max(1e-5);
            probe[k] = v + eps;
            let up = feature
                .score_features(&probe, w as u32, h as u32, None)
                .unwrap();
            probe[k] = v - eps;
            let down = feature
                .score_features(&probe, w as u32, h as u32, None)
                .unwrap();
            probe[k] = v;
            assert_eq!(
                s[k],
                (up - down) / (2.0 * eps),
                "independent sensitivity f{k}"
            );
        }
        let mut served = owner(&model, Objective::Served);
        let served_base = served.compute(&rs, &ds, None).unwrap();
        let mut altered = d.clone();
        let mut blocks = Vec::new();
        let mut true_gain = Vec::new();
        let mut true_linear = Vec::new();
        let mut predicted = Vec::new();
        let mut served_bits = Vec::new();
        let mut bad = 0usize;
        let mut compared = 0usize;
        let mut max_abs = 0f64;
        let mut max_tol_ratio = 0f64;
        let mut coarse_actual = Vec::new();
        let mut coarse_local = Vec::new();
        for y0 in (0..h).step_by(case.block) {
            for x0 in (0..w).step_by(case.block) {
                let (x1, y1) = ((x0 + case.block).min(w), (y0 + case.block).min(h));
                for y in y0..y1 {
                    let range = y * w + x0..y * w + x1;
                    altered[range.clone()].copy_from_slice(&r[range]);
                }
                let input = crate::RgbSlice::new(&altered, w, h);
                let repaired = pixel.compute(&rs, &input, None).unwrap();
                let complete = served.compute(&rs, &input, None).unwrap();
                assert_eq!(repaired.features(), complete.features());
                let df: Vec<_> = repaired
                    .features()
                    .iter()
                    .zip(f)
                    .map(|(a, b)| a - b)
                    .collect();
                let linear: f64 = df.iter().zip(s).map(|(d, s)| d * s).sum();
                let gain = mapped.refinement_gain(x0, y0, x1, y1);
                let delta = repaired.score() - base.score();
                true_gain.push(delta);
                true_linear.push(linear);
                predicted.push(gain);
                served_bits.push(complete.score().to_bits());
                let mut ca = 0.0;
                let mut cl = 0.0;
                if job.engine {
                    let snap = mapped
                        .neighbour_exact
                        .as_ref()
                        .expect("snapshot must capture admitted case");
                    let deltas = snap
                        .deltas((x0, y0, x1, y1), &crate::local_refine::Candidate::Reference)
                        .unwrap();
                    for (id, local) in deltas {
                        let actual = df[id];
                        let error = (local - actual).abs();
                        // Existing local_refine golden contract, unmodified.
                        let tol = 4.0
                            * f64::from(f32::EPSILON)
                            * f[id].abs().max(repaired.features()[id].abs()).max(1e-20)
                            + 1e-6 * actual.abs();
                        compared += 1;
                        bad += usize::from(error > tol);
                        max_abs = max_abs.max(error);
                        max_tol_ratio = max_tol_ratio.max(error / tol.max(1e-30));
                        ca += s[id] * actual;
                        cl += s[id] * local;
                    }
                    coarse_actual.push(ca);
                    coarse_local.push(cl);
                }
                blocks.push(json!({"bounds":[x0,y0,x1,y1],"score_delta":delta,"linearized_gain":linear,"refinement_gain":gain,
                    "served_score_delta":complete.score()-served_base.score(),"coarse_full_linear":ca,"coarse_local_linear":cl}));
                for y in y0..y1 {
                    let range = y * w + x0..y * w + x1;
                    altered[range.clone()].copy_from_slice(&d[range]);
                }
            }
        }
        let m2 = stats::spearman(&true_linear, &true_gain);
        let m3f = stats::spearman(&predicted, &true_gain);
        let report = json!({"key":case.key,"objective":format!("{:?}",case.objective),"model":case.model_sha256,
            "base_score":base.score(),"served_base_bits":served_base.score().to_bits(),"served_repair_bits":served_bits,
            "m2":m2,"m3f":m3f,"pass":m2>=0.99 && m3f>=0.70,"engine":job.engine,"blocks":blocks,
            "engine_feature_comparisons":compared,"engine_feature_disagreements":bad,"engine_max_abs_error":max_abs,
            "engine_max_tolerance_ratio":max_tol_ratio,"coarse_gain_rank":if job.engine {Some(stats::spearman(&coarse_local,&coarse_actual))} else {None},
            "map_vs_true_linear_rank":stats::spearman(&predicted,&true_linear),"sensitivity_independent_probe_exact":true});
        eprintln!(
            "{} {:?} M2={m2:.12} M3f={m3f:.12} engine_bad={bad}/{compared}",
            case.key, case.objective
        );
        summary.push(report);
        // Progress is persisted after each completed case; failures cannot hide it.
        std::fs::write(
            format!("{}.partial", job.output),
            serde_json::to_vec(&summary).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        &job.output,
        serde_json::to_vec(&json!({"packet_sha256":sha(&bytes),"rows":summary})).unwrap(),
    )
    .unwrap();
}
