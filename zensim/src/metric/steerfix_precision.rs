//! Report-only E34 precision reference. Never compiled into serving code.
//! Deliberately separate from engineering_packet and its registered arithmetic.
use super::*;

#[derive(Clone, Copy)]
enum Input {
    Direct(usize),
    Product(usize, usize),
}

struct Reference {
    inputs: Vec<Input>,
    ids: Vec<usize>,
    means: Vec<f64>,
    scales: Vec<f64>,
    w1: Vec<f64>,
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
    spline: Option<Arc<OutputCalibrationSpline>>,
}

impl Reference {
    fn new(model: &crate::mlp::Model) -> Self {
        use zenpredict::{Activation, WeightStorage};
        let weights = |storage| match storage {
            WeightStorage::F32(w) => w.iter().map(|&x| f64::from(x)).collect(),
            WeightStorage::F16(w) => w
                .iter()
                .map(|&x| f64::from(zenpredict::f16_bits_to_f32(x)))
                .collect(),
            WeightStorage::I8 { .. } => panic!("unregistered weight storage"),
        };
        // Native loader validates layouts/derived grammar before the reference.
        let native = owner(model, Objective::Served);
        assert!(model.feature_transforms().is_none());
        let md = &native.metadata;
        assert!(
            md.per_sample_alpha.is_none()
                && md.hybrid_head.is_none()
                && md.minmax_head.is_none()
                && md.tanh_pin_scale.is_none()
                && md.per_codec_calibration.is_none()
        );
        assert_eq!(model.n_layers(), 2);
        let first = model.layer(0);
        let last = model.layer(1);
        assert_eq!(first.activation, Activation::Relu);
        assert_eq!(last.activation, Activation::Identity);
        assert_eq!(last.in_dim, first.out_dim);
        assert_eq!(last.out_dim, 1);
        let ids: Vec<_> = crate::feature_layout::declared_ids(model)
            .unwrap()
            .iter_slots()
            .collect();
        let inputs = if md.derived_inputs.is_some() {
            let metadata = model.metadata();
            let text = metadata.get_utf8("zensim.derived_inputs").unwrap();
            text.lines()
                .skip(1)
                .filter(|l| !l.trim().is_empty())
                .map(|line| {
                    let parts: Vec<_> = line.split_whitespace().collect();
                    match parts.as_slice() {
                        ["in", a] => Input::Direct(a.parse().unwrap()),
                        ["product", a, b] => Input::Product(a.parse().unwrap(), b.parse().unwrap()),
                        _ => panic!("unsupported derived input"),
                    }
                })
                .collect()
        } else {
            ids.iter().copied().map(Input::Direct).collect()
        };
        let result = Self {
            inputs,
            ids,
            means: model.scaler_mean().iter().map(|&x| f64::from(x)).collect(),
            scales: model
                .scaler_scale()
                .iter()
                .map(|&x| if x == 0.0 { 1.0 } else { f64::from(x) })
                .collect(),
            w1: weights(first.weights),
            b1: first.biases.iter().map(|&x| f64::from(x)).collect(),
            w2: weights(last.weights),
            b2: f64::from(last.biases[0]),
            spline: md.output_spline.clone(),
        };
        assert_eq!(result.inputs.len(), first.in_dim);
        result
    }

    fn hidden(&self, row: &[f64]) -> Vec<f64> {
        let mut hidden = self.b1.clone();
        for (i, input) in self.inputs.iter().enumerate() {
            let x = match *input {
                Input::Direct(a) => row[a],
                Input::Product(a, b) => row[a] * row[b],
            };
            let z = (x - self.means[i]) / self.scales[i];
            for (h, &w) in hidden
                .iter_mut()
                .zip(&self.w1[i * self.b1.len()..(i + 1) * self.b1.len()])
            {
                *h += z * w;
            }
        }
        hidden
    }

    fn raw(&self, row: &[f64]) -> f64 {
        self.hidden(row)
            .iter()
            .zip(&self.w2)
            .fold(self.b2, |y, (&h, &w)| y + h.max(0.0) * w)
    }

    fn score(&self, row: &[f64], raw: bool) -> f64 {
        let y = self.raw(row);
        if !raw && let Some(s) = &self.spline {
            crate::score_math::pchip_eval_capped(y, &s.xs, &s.ys, &s.derivs)
        } else {
            y
        }
    }

    fn exact(&self, row: &[f64], raw: bool) -> Vec<f64> {
        let hidden = self.hidden(row);
        let slope = if raw {
            1.0
        } else {
            self.spline
                .as_ref()
                .map_or(1.0, |s| spline_slope(self.raw(row), s))
        };
        let mut gradient = vec![0.0; row.len()];
        for (i, input) in self.inputs.iter().enumerate() {
            // Derivative at an exact ReLU kink uses the inactive (zero) side;
            // count such kinks in every measured case rather than hiding them.
            let g: f64 = hidden
                .iter()
                .zip(&self.w2)
                .enumerate()
                .filter(|(_, (h, _))| **h > 0.0)
                .map(|(j, (_, w))| w * self.w1[i * hidden.len() + j])
                .sum::<f64>()
                / self.scales[i]
                * slope;
            match *input {
                Input::Direct(a) => gradient[a] += g,
                Input::Product(a, b) => {
                    gradient[a] += g * row[b];
                    gradient[b] += g * row[a];
                }
            }
        }
        gradient
    }

    fn fd(&self, row: &[f64], raw: bool) -> Vec<f64> {
        let mut result = vec![0.0; row.len()];
        let mut probe = row.to_vec();
        for &i in &self.ids {
            let h = (row[i].abs() * 1e-3).max(1e-5);
            probe[i] = row[i] + h;
            let up = self.score(&probe, raw);
            probe[i] = row[i] - h;
            let down = self.score(&probe, raw);
            probe[i] = row[i];
            result[i] = (up - down) / (2.0 * h);
        }
        result
    }
}

fn spline_slope(x: f64, s: &OutputCalibrationSpline) -> f64 {
    let n = s.xs.len();
    if x <= s.xs[0] {
        let floor = 2.0 * s.ys[0] - s.ys[n - 1];
        return if s.ys[0] + s.derivs[0] * (x - s.xs[0]) > floor {
            s.derivs[0]
        } else {
            0.0
        };
    }
    if x >= s.xs[n - 1] {
        return if s.ys[n - 1] + s.derivs[n - 1] * (x - s.xs[n - 1]) < 100.0 {
            s.derivs[n - 1]
        } else {
            0.0
        };
    }
    if crate::score_math::pchip_eval_capped(x, &s.xs, &s.ys, &s.derivs) >= 100.0 {
        return 0.0;
    }
    let i = s.xs.partition_point(|v| *v <= x) - 1;
    let h = s.xs[i + 1] - s.xs[i];
    let t = (x - s.xs[i]) / h;
    ((6.0 * t * t - 6.0 * t) * s.ys[i]
        + (3.0 * t * t - 4.0 * t + 1.0) * h * s.derivs[i]
        + (-6.0 * t * t + 6.0 * t) * s.ys[i + 1]
        + (3.0 * t * t - 2.0 * t) * h * s.derivs[i + 1])
        / h
}

#[test]
fn reference_contracts() {
    // A two-unit network with one active and one inactive unit; product chain
    // rule and nontrivial scales. Closed-form y=100 - 3*(x0/2 + x0*x1/4).
    let r = Reference {
        inputs: vec![Input::Direct(0), Input::Product(0, 1)],
        ids: vec![0, 1],
        means: vec![0.0; 2],
        scales: vec![2.0, 4.0],
        w1: vec![1.0, -1.0, 1.0, -1.0],
        b1: vec![0.0; 2],
        w2: vec![-3.0, -7.0],
        b2: 100.0,
        spline: None,
    };
    let x = [2.0, 4.0];
    assert_eq!(r.raw(&x), 91.0);
    assert_eq!(r.exact(&x, true), [-4.5, -1.5]);
    for (a, b) in r.fd(&x, true).iter().zip(r.exact(&x, true)) {
        assert!((a - b).abs() < 1e-9);
    }
    // On x0 < 0, y=100+7*(x0/2+x0*x1/4), so dy/dx1=7*x0/4.
    assert_eq!(r.exact(&[-2.0, 4.0], true), [10.5, -3.5]);
    // A finite secant crossing a ReLU kink must differ from the active gradient.
    let kink = [1e-7, 4.0];
    assert!((r.fd(&kink, true)[0] - r.exact(&kink, true)[0]).abs() > 1.0);
    let xs = vec![0.0, 1.0, 3.0];
    let ys = vec![10.0, 30.0, 60.0];
    let s = OutputCalibrationSpline {
        derivs: crate::score_math::pchip_derivs(&xs, &ys),
        xs,
        ys,
    };
    for x in [-10.0, -1.0, 0.5, 2.0, 4.0, 10.0] {
        let h = 1e-5;
        let fd = (crate::score_math::pchip_eval_capped(x + h, &s.xs, &s.ys, &s.derivs)
            - crate::score_math::pchip_eval_capped(x - h, &s.xs, &s.ys, &s.derivs))
            / (2.0 * h);
        assert!((fd - spline_slope(x, &s)).abs() < 1e-7);
    }
}

#[derive(Deserialize)]
struct ModelInput {
    name: String,
    path: String,
    sha256: String,
    oracle: String,
    oracle_sha256: String,
}

#[derive(Deserialize)]
struct PrecisionJob {
    packet: String,
    packet_sha256: String,
    models: Vec<ModelInput>,
    output_dir: String,
    case_limit: usize,
}

#[test]
fn cached_tail_probe() {
    let job: PrecisionJob = serde_json::from_slice(
        &std::fs::read(std::env::var("E34_PRECISION_PACKET").unwrap()).unwrap(),
    )
    .unwrap();
    let out = std::path::Path::new(&job.output_dir);
    let mut worst = (0.0, serde_json::Value::Null, 0);
    for i in 0..135 {
        let rows: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(out.join(format!("case-{i:03}.json"))).unwrap())
                .unwrap();
        let row = &rows[0];
        assert_eq!(row["model"], "a");
        for (j, e) in row["raw_gradient_errors"][1]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let e = e.as_f64().unwrap();
            if e > worst.0 {
                worst = (e, row.clone(), j);
            }
        }
    }
    let bytes = std::fs::read(&job.models[0].path).unwrap();
    assert_eq!(sha(&bytes), job.models[0].sha256);
    let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
    let reference = Reference::new(&model);
    let bytes = std::fs::read(worst.1["feature_file"].as_str().unwrap()).unwrap();
    assert_eq!(sha(&bytes), worst.1["feature_sha256"].as_str().unwrap());
    let width = worst.1["feature_width"].as_u64().unwrap() as usize;
    let row: Vec<f64> = bytes[..width * 8]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    let slot = worst.1["gradient_ids"][worst.2].as_u64().unwrap() as usize;
    let h = (row[slot].abs() * 1e-3).max(1e-5);
    let mut probe = row.clone();
    probe[slot] += h;
    let up = reference.hidden(&probe);
    probe[slot] = row[slot] - h;
    let down = reference.hidden(&probe);
    let base = reference.hidden(&row);
    let crossings: Vec<_> = (0..base.len()).filter(|&i| (up[i] > 0.0) != (down[i] > 0.0))
        .map(|i| json!({"unit":i,"base":base[i],"up":up[i],"down":down[i],"output_weight":reference.w2[i]})).collect();
    assert!(!crossings.is_empty());
    let fd = reference.fd(&row, true)[slot];
    let exact = reference.exact(&row, true)[slot];
    assert_eq!((fd - exact).abs(), worst.0);
    let result = json!({"model":"a","key":worst.1["key"],"block":worst.1["block"],
        "slot":slot,"feature":row[slot],"h":h,"raw_fd64":fd,"raw_exact":exact,
        "absolute_error":worst.0,"crossing_units":crossings});
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(
            out.parent()
                .unwrap()
                .join(std::env::var("E34_PRECISION_TAIL_OUT").unwrap()),
        )
        .unwrap();
    file.write_all(serde_json::to_string_pretty(&result).unwrap().as_bytes())
        .unwrap();
    eprintln!("{result}");
}

#[test]
fn gate_precision_packet() {
    assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
    let _token = archmage::testing::lock_token_testing();
    let job_bytes =
        std::fs::read(std::env::var("E34_PRECISION_PACKET").expect("explicit report-only job"))
            .unwrap();
    let job: PrecisionJob = serde_json::from_slice(&job_bytes).unwrap();
    let packet_bytes = std::fs::read(&job.packet).unwrap();
    assert_eq!(sha(&packet_bytes), job.packet_sha256);
    let packet: Job = serde_json::from_slice(&packet_bytes).unwrap();
    assert_eq!(packet.cases.len(), 135);
    assert!(job.case_limit > 0 && job.case_limit <= 135);
    assert_eq!(
        job.models
            .iter()
            .map(|m| m.name.as_str())
            .collect::<Vec<_>>(),
        ["a", "c", "seed0", "l"]
    );
    let out = std::path::Path::new(&job.output_dir);
    std::fs::create_dir(out).expect("fresh output directory");
    std::fs::create_dir(out.join("features")).unwrap();
    let blobs: Vec<_> = job
        .models
        .iter()
        .map(|m| {
            let bytes = std::fs::read(&m.path).unwrap();
            assert_eq!(sha(&bytes), m.sha256);
            bytes
        })
        .collect();
    let models: Vec<_> = blobs
        .iter()
        .map(|b| crate::mlp::Model::from_bytes(b).unwrap())
        .collect();
    let refs: Vec<_> = models.iter().map(Reference::new).collect();
    let oracles: Vec<serde_json::Value> = job
        .models
        .iter()
        .map(|m| {
            let bytes = std::fs::read(&m.oracle).unwrap();
            assert_eq!(sha(&bytes), m.oracle_sha256);
            let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(value["rows"].as_array().unwrap().len(), 135);
            value
        })
        .collect();
    let mut extractor = owner(&models[1], Objective::Served);
    let mut native: Vec<_> = models.iter().map(|m| owner(m, Objective::Served)).collect();
    let mut result = Vec::new();
    for (case_index, case) in packet.cases.iter().take(job.case_limit).enumerate() {
        let (r, w, h) = image(&case.reference, &case.reference_sha256);
        let (d, dw, dh) = image(&case.distorted, &case.distorted_sha256);
        assert_eq!((w, h), (dw, dh));
        assert_eq!(sha(bytemuck::cast_slice(&r)), case.reference_pixels_sha256);
        assert_eq!(sha(bytemuck::cast_slice(&d)), case.distorted_pixels_sha256);
        let rs = crate::RgbSlice::new(&r, w, h);
        let ds = crate::RgbSlice::new(&d, w, h);
        let base = extractor.compute(&rs, &ds, None).unwrap();
        let f = base.features();
        let mut available = vec![false; f.len()];
        extractor.mark_feature_reads(&mut available);
        let mut base32 = Vec::new();
        let mut base64 = Vec::new();
        let mut gradients = Vec::new();
        let mut raw_modes = Vec::new();
        for (m, (n, reference)) in native.iter_mut().zip(&refs).enumerate() {
            let mut needed = vec![false; f.len()];
            n.mark_feature_reads(&mut needed);
            assert!(
                needed
                    .iter()
                    .zip(&available)
                    .all(|(&need, &have)| !need || have)
            );
            let old = &oracles[m]["rows"][case_index];
            assert_eq!(old["key"].as_str().unwrap(), case.key);
            assert_eq!(old["model"].as_str().unwrap(), job.models[m].sha256);
            assert_eq!(
                old["blocks"][0]["bounds"][2].as_u64().unwrap() as usize,
                case.block
            );
            let score = n.score_features(f, w as u32, h as u32, None).unwrap();
            assert_eq!(
                score.to_bits(),
                old["served_base_bits"].as_u64().unwrap(),
                "base replay {} {}",
                job.models[m].name,
                case.key
            );
            let raw = old["sensitivity_objective"].as_str().unwrap() == "PreFloor";
            let s = n
                .prepared_steering_gradient(f, w as u32, h as u32, None)
                .unwrap();
            let mut direct = owner(
                &models[m],
                if raw {
                    Objective::PreFloor
                } else {
                    Objective::Served
                },
            );
            assert_eq!(
                s,
                direct
                    .score_features_fd_gradient(f, w as u32, h as u32, None)
                    .unwrap()
            );
            gradients.push([s, reference.fd(f, raw), reference.exact(f, raw)]);
            raw_modes.push(raw);
            base32.push(score);
            base64.push(reference.score(f, false));
        }
        let mut gains32: Vec<Vec<f64>> = vec![Vec::new(); models.len()];
        let mut gains64 = gains32.clone();
        let mut linear: Vec<[Vec<f64>; 3]> = (0..models.len())
            .map(|_| [Vec::new(), Vec::new(), Vec::new()])
            .collect();
        let mut altered = d.clone();
        let mut block_index = 0;
        use std::io::Write;
        let feature_path = out.join("features").join(format!("{case_index:03}.f64le"));
        let mut feature_file = std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&feature_path)
                .unwrap(),
        );
        for &x in f {
            feature_file.write_all(&x.to_le_bytes()).unwrap();
        }
        let mut max_score_error = vec![0.0_f64; models.len()];
        for y0 in (0..h).step_by(case.block) {
            for x0 in (0..w).step_by(case.block) {
                let (x1, y1) = ((x0 + case.block).min(w), (y0 + case.block).min(h));
                for y in y0..y1 {
                    let span = y * w + x0..y * w + x1;
                    altered[span.clone()].copy_from_slice(&r[span]);
                }
                let repaired = extractor
                    .compute(&rs, &crate::RgbSlice::new(&altered, w, h), None)
                    .unwrap();
                let row = repaired.features();
                assert_eq!(row.len(), f.len());
                for &x in row {
                    feature_file.write_all(&x.to_le_bytes()).unwrap();
                }
                for m in 0..models.len() {
                    let score = native[m]
                        .score_features(row, w as u32, h as u32, None)
                        .unwrap();
                    let old = &oracles[m]["rows"][case_index];
                    assert_eq!(
                        score.to_bits(),
                        old["served_repair_bits"][block_index].as_u64().unwrap(),
                        "repair replay {} {} {block_index}",
                        job.models[m].name,
                        case.key
                    );
                    gains32[m].push(score - base32[m]);
                    let y64 = refs[m].score(row, false);
                    gains64[m].push(y64 - base64[m]);
                    max_score_error[m] = max_score_error[m].max((y64 - score).abs());
                    for (method, g) in gradients[m].iter().enumerate() {
                        linear[m][method].push(
                            row.iter()
                                .zip(f)
                                .zip(g)
                                .map(|((&new, &old), &s)| (new - old) * s)
                                .sum(),
                        );
                    }
                }
                block_index += 1;
                for y in y0..y1 {
                    let span = y * w + x0..y * w + x1;
                    altered[span.clone()].copy_from_slice(&d[span]);
                }
            }
        }
        feature_file.flush().unwrap();
        drop(feature_file);
        let feature_sha = sha(&std::fs::read(&feature_path).unwrap());
        for m in 0..models.len() {
            let m2: Vec<_> = linear[m]
                .iter()
                .map(|v| stats::spearman(v, &gains32[m]))
                .collect();
            let m2_gain64: Vec<_> = linear[m]
                .iter()
                .map(|v| stats::spearman(v, &gains64[m]))
                .collect();
            let old = &oracles[m]["rows"][case_index];
            assert!(
                (m2[0] - old["m2"].as_f64().unwrap()).abs() < 1e-12,
                "registered M2 replay {} {}",
                job.models[m].name,
                case.key
            );
            let errors: Vec<Vec<_>> = gradients[m][..2]
                .iter()
                .map(|g| {
                    refs[m]
                        .ids
                        .iter()
                        .map(|&i| (g[i] - gradients[m][2][i]).abs())
                        .collect()
                })
                .collect();
            let raw_exact = refs[m].exact(f, true);
            if job.models[m].name == "l" {
                let coefficients = linear_coefficients(&models[m], f.len());
                assert_eq!(
                    raw_exact,
                    coefficients.iter().map(|c| -c).collect::<Vec<_>>()
                );
            }
            let raw_fd64 = refs[m].fd(f, true);
            let raw_fd32 = owner(&models[m], Objective::PreFloor)
                .score_features_fd_gradient(f, w as u32, h as u32, None)
                .unwrap();
            let raw_errors: Vec<Vec<_>> = [raw_fd32, raw_fd64]
                .iter()
                .map(|g| {
                    refs[m]
                        .ids
                        .iter()
                        .map(|&i| (g[i] - raw_exact[i]).abs())
                        .collect()
                })
                .collect();
            let hidden = refs[m].hidden(f);
            result.push(json!({"model":job.models[m].name,"model_sha256":job.models[m].sha256,"case_index":case_index,
                "key":case.key,"block":case.block,"blocks":block_index,"width":w,"height":h,
                "feature_width":f.len(),"feature_file":feature_path,"feature_sha256":feature_sha,
                "sensitivity_objective":if raw_modes[m] {"PreFloor"} else {"Served"},
                "m2":m2,"m2_f64_gains":m2_gain64,"registered_m3f":old["m3f"],
                "registered_pass":old["pass"],"gradient_ids":refs[m].ids,
                "gradient_errors":errors,"raw_gradient_errors":raw_errors,"gradients":gradients[m],
                "max_score_f64_vs_f32_error":max_score_error[m].max((base64[m]-base32[m]).abs()),
                "hidden_active":hidden.iter().filter(|&&v| v>0.0).count(),"hidden_exact_kinks":hidden.iter().filter(|&&v| v==0.0).count(),
                "linear_gains":linear[m],"registered_gains":gains32[m],"f64_gains":gains64[m]}));
            eprintln!(
                "{} {} b{} M2 f32/f64/exact {:?}",
                job.models[m].name, case.key, case.block, m2
            );
        }
        std::fs::write(
            out.join("progress.json"),
            serde_json::to_vec(&json!({"cases":case_index+1,"rows":result.len()})).unwrap(),
        )
        .unwrap();
        std::fs::write(
            out.join(format!("case-{case_index:03}.json")),
            serde_json::to_vec(&result[result.len() - 4..]).unwrap(),
        )
        .unwrap();
    }
    std::fs::write(out.join("complete.json"),serde_json::to_vec(&json!({"job_sha256":sha(&job_bytes),"packet_sha256":job.packet_sha256,"cases":job.case_limit,"model_cases":result.len(),"methods":["registered_f32_fd","f64_fd","analytic"],"registered_replay":"all base/repair score bits and M2 matched"})).unwrap()).unwrap();
}

// D12's independent f64 finite-difference reference uses the already captured
// label-free base feature rows. No pixel/label discovery and no gate selection.
#[test]
fn d12_exact_gradient_reference_packet() {
    let _guard = archmage::testing::lock_token_testing();
    let job: serde_json::Value = serde_json::from_slice(
        &std::fs::read(
            std::env::var("E34_D12_REFERENCE_JOB").expect("explicit D12 reference packet"),
        )
        .unwrap(),
    )
    .unwrap();
    let mut summaries = Vec::new();
    for m in job["models"].as_array().unwrap() {
        let bytes = std::fs::read(m["path"].as_str().unwrap()).unwrap();
        assert_eq!(sha(&bytes), m["sha256"].as_str().unwrap());
        let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
        let reference = Reference::new(&model);
        let mut analytic_checks = 0;
        let mut fd_checks = 0;
        let mut excluded = 0;
        let mut maximum = 0_f64;
        let mut fd_maximum = 0_f64;
        let mut tier_checks = 0;
        for (case_index, row) in job["rows"].as_array().unwrap().iter().enumerate() {
            let width = row["feature_width"].as_u64().unwrap() as usize;
            let whole = std::fs::read(row["feature_file"].as_str().unwrap()).unwrap();
            assert_eq!(sha(&whole), row["feature_sha256"].as_str().unwrap());
            let bytes = &whole[..8 * width];
            let (w, h) = (
                row["width"].as_u64().unwrap() as u32,
                row["height"].as_u64().unwrap() as u32,
            );
            let f: Vec<f64> = bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let mut scorer = owner(&model, Objective::Served);
            let score_bits = scorer.score_features(&f, w, h, None).unwrap().to_bits();
            let actual = scorer.score_features_fd_gradient(&f, w, h, None).unwrap();
            assert_eq!(
                score_bits,
                scorer.score_features(&f, w, h, None).unwrap().to_bits()
            );
            let expected = reference.exact(&f, false);
            for (&a, &b) in actual.iter().zip(&expected) {
                let error = (a - b).abs();
                maximum = maximum.max(error);
                assert!(
                    error <= 1e-10 * (1. + b.abs()),
                    "{} case {case_index} analytic {a} vs reference {b}",
                    m["name"]
                );
                analytic_checks += 1;
            }
            let base_hidden = reference.hidden(&f);
            let branches = |row: &[f64]| {
                let hidden: Vec<_> = reference.hidden(row).iter().map(|&x| x > 0.).collect();
                let y = reference.score(row, false);
                let floor = reference
                    .spline
                    .as_ref()
                    .map_or(f64::NEG_INFINITY, |s| 2. * s.ys[0] - s.ys[s.ys.len() - 1]);
                (hidden, y >= 100., y <= floor)
            };
            let base_branch = branches(&f);
            let mut probe = f.clone();
            for &id in &reference.ids {
                let x = f[id];
                let mut h = 1e-6 * (1. + x.abs());
                let mut finite = None;
                for _ in 0..20 {
                    probe[id] = x + h;
                    let upper_branch = branches(&probe);
                    let up = reference.score(&probe, false);
                    probe[id] = x - h;
                    let lower_branch = branches(&probe);
                    let down = reference.score(&probe, false);
                    probe[id] = x;
                    if upper_branch == base_branch && lower_branch == base_branch {
                        finite = Some((up - down) / (2. * h));
                        break;
                    }
                    h *= 0.5;
                }
                if let Some(fd) = finite {
                    let error = (actual[id] - fd).abs();
                    fd_maximum = fd_maximum.max(error);
                    assert!(
                        error <= 1e-7 * (1. + fd.abs()),
                        "{} case {case_index} f{id}: analytic {} vs f64 FD {fd}, h={h}",
                        m["name"],
                        actual[id]
                    );
                    fd_checks += 1;
                } else {
                    excluded += 1;
                }
            }
            assert_eq!(base_hidden, reference.hidden(&f));
            // The derivative has one fixed scalar order at every dispatch tier.
            let expected_bits: Vec<_> = actual.iter().map(|x| x.to_bits()).collect();
            let report = archmage::testing::for_each_token_permutation(
                archmage::testing::CompileTimePolicy::Warn,
                |_| {
                    let g = owner(&model, Objective::Served)
                        .score_features_fd_gradient(&f, w, h, None)
                        .unwrap();
                    assert_eq!(
                        g.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected_bits
                    );
                },
            );
            tier_checks += report.permutations_run;
        }
        assert!(fd_checks > 0);
        let summary = json!({"model":m["name"],"cases":job["rows"].as_array().unwrap().len(),"analytic_coordinates":analytic_checks,"f64_fd_coordinates":fd_checks,"crossing_coordinates_excluded":excluded,"tier_checks":tier_checks,"max_analytic_error":maximum,"max_f64_fd_error":fd_maximum,"analytic_tolerance":"1e-10*(1+abs(reference))","f64_fd_tolerance":"1e-7*(1+abs(fd))"});
        eprintln!("D12 reference {summary}");
        summaries.push(summary);
    }
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(job["output"].as_str().unwrap())
        .unwrap();
    use std::io::Write;
    writeln!(file,"{}",json!({"models":summaries,"scores_unchanged_by_gradient":true,"all_tier_gradient_bits_equal":true})).unwrap();
}
