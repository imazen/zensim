//! Bounds and train-only calibration for the existing demo/search owner.
use super::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use zensim_target::{SeedCurve, SeedEstimate};

const SCHEMA: &str = "reachable-target-v1";
const CONFIG: &str = "jpeg:ApproxJpegli0-100,420;webp:q0-100,method4;avif:q1-100,speed6;jxl:distance0.01-25,wrapper-default;png:lossless";
type TrainingCurves = BTreeMap<(usize, usize), Vec<Vec<(f32, f32)>>>;

#[derive(Serialize, Deserialize)]
struct SourceSet {
    corpus_commit: String,
    split_manifest_sha256: String,
    sources: Vec<Source>,
}
#[derive(Serialize, Deserialize)]
struct Source {
    path: PathBuf,
    sha256: String,
    origin: String,
    family: String,
    split: String,
    content_class: String,
}
#[derive(Serialize, Deserialize)]
struct Calibration {
    schema: String,
    binary_sha256: String,
    config: String,
    formula_revision: String,
    corpus_commit: String,
    split_manifest_sha256: String,
    training_sources: Vec<Source>,
    model_ids: Vec<String>,
    curves: Vec<Curve>,
}
#[derive(Serialize, Deserialize)]
struct Curve {
    codec: String,
    model: String,
    #[serde(flatten)]
    seed_curve: SeedCurve,
}
impl Curve {
    fn seed(&self, target: f32) -> Result<SeedEstimate> {
        self.seed_curve.estimate(target)
    }
}

struct Reconstruction {
    knob: f32,
    encoded: Vec<u8>,
    decoded: Vec<u8>,
    seconds: f64,
}

fn score(
    candidate: &mut Option<BakeScorer<'_>>,
    named: &Zensim,
    reference: &RgbSlice<'_>,
    decoded: &[u8],
    w: u32,
    h: u32,
    codec: CodecKind,
) -> Result<f32> {
    ensure!(
        decoded.len() == w as usize * h as usize * 3,
        "decoded length mismatch"
    );
    let d = RgbSlice::try_new(
        bytemuck::cast_slice::<u8, [u8; 3]>(decoded),
        w as usize,
        h as usize,
    )?;
    let s = match candidate {
        Some(c) => c.compute(reference, &d, Some(codec.extension()))?.score(),
        None => named.compute(reference, &d)?.score(),
    } as f32;
    ensure!(s.is_finite(), "nonfinite served score");
    Ok(s)
}

fn fit(codec: &str, model: &str, rows: &[Vec<(f32, f32)>], inverted: bool) -> Result<Curve> {
    Ok(Curve {
        codec: codec.into(),
        model: model.into(),
        seed_curve: SeedCurve::fit(rows, inverted)?,
    })
}

fn validate_split(sources: &SourceSet, calibration: Option<&Calibration>, fit: bool) -> Result<()> {
    ensure!(!sources.sources.is_empty(), "source manifest is empty");
    let mut hashes = BTreeSet::new();
    let mut origins = BTreeSet::new();
    for s in &sources.sources {
        ensure!(
            !s.origin.is_empty() && !s.family.is_empty(),
            "missing origin/family"
        );
        ensure!(hashes.insert(&s.sha256), "duplicate source bytes");
        ensure!(
            origins.insert(&s.origin),
            "one rendition per origin in this initial instrument"
        );
        ensure!(
            if fit {
                s.split == "train"
            } else {
                s.split == "validate"
            },
            "fit requires train; development evaluation requires validate (terminal test is reserved)"
        );
    }
    if let Some(c) = calibration {
        ensure!(
            c.corpus_commit == sources.corpus_commit
                && c.split_manifest_sha256 == sources.split_manifest_sha256,
            "calibration and evaluation corpus/split identities differ"
        );
        for t in &c.training_sources {
            ensure!(
                t.split == "train",
                "calibration contains non-training content"
            );
            ensure!(
                !sources
                    .sources
                    .iter()
                    .any(|s| s.origin == t.origin || s.family == t.family || s.sha256 == t.sha256),
                "calibration/evaluation source or family overlap"
            );
        }
    }
    Ok(())
}

fn witnessed_targets(scores: &[f32], fixed: &[f32], count: usize, tol: f32) -> Vec<f32> {
    let mut sorted = scores.to_vec();
    sorted.sort_by(f32::total_cmp);
    sorted.dedup();
    let mut targets: Vec<_> = fixed
        .iter()
        .copied()
        .filter(|t| scores.iter().any(|s| (s - t).abs() <= tol))
        .collect();
    for i in 0..count {
        targets.push(sorted[i * (sorted.len() - 1) / (count - 1).max(1)]);
    }
    targets.sort_by(f32::total_cmp);
    targets.dedup();
    targets
}

pub(super) fn run(args: &Args) -> Result<()> {
    ensure!(
        args.fit_calibration || args.calibration.is_some(),
        "choose --fit-calibration or --calibration with a source manifest"
    );
    ensure!(
        (3..=257).contains(&args.bound_steps),
        "bound-steps must be 3..257"
    );
    ensure!(
        args.witness_targets >= 2,
        "at least two witnessed targets required"
    );
    ensure!(
        !args.budgets.is_empty() && args.budgets.iter().all(|b| (1..=3).contains(b)),
        "this protocol measures budgets 1, 2, 3"
    );
    ensure!(
        args.tolerance.is_finite()
            && args.tolerance >= 0.0
            && args.targets.iter().all(|v| v.is_finite()),
        "invalid target/tolerance"
    );
    ensure!(
        std::env::var("ZENSIM_TARGET_SECANT")
            .unwrap_or_default()
            .is_empty(),
        "unset ZENSIM_TARGET_SECANT: policy arms are explicit"
    );
    let sources: SourceSet =
        serde_json::from_slice(&fs::read(args.source_manifest.as_ref().unwrap())?)?;
    let calibration: Option<Calibration> = args
        .calibration
        .as_ref()
        .map(|p| -> Result<_> { Ok(serde_json::from_slice(&fs::read(p)?)?) })
        .transpose()?;
    validate_split(&sources, calibration.as_ref(), args.fit_calibration)?;
    if args.fit_calibration {
        ensure!(sources.sources.len() >= 2, "need multiple train sources");
    }
    let codecs = args
        .codecs
        .iter()
        .map(|c| CodecKind::parse(c))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !codecs.is_empty() && !codecs.contains(&CodecKind::Png),
        "calibration requires lossy codec families"
    );
    ensure!(
        args.codecs.iter().collect::<BTreeSet<_>>().len() == args.codecs.len()
            && args.budgets.iter().collect::<BTreeSet<_>>().len() == args.budgets.len(),
        "duplicate codec or budget"
    );
    let bake_bytes = args
        .bake
        .iter()
        .map(fs::read)
        .collect::<std::io::Result<Vec<_>>>()?;
    let models = bake_bytes
        .iter()
        .map(|b| zenpredict::Model::from_bytes(b))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    for m in &models {
        BakeScorer::new(m)?;
    }
    let labels: Vec<String> = ["B".to_owned(), "D".to_owned()]
        .into_iter()
        .chain(bake_bytes.iter().map(|b| format!("bake:{}", sha(b))))
        .collect();
    ensure!(
        labels.iter().collect::<BTreeSet<_>>().len() == labels.len(),
        "duplicate model"
    );
    let binary_sha = sha(&fs::read(std::env::current_exe()?)?);
    let formula = std::env::var("ZENSIM_FORMULA_REV").unwrap_or_else(|_| "default".into());
    if let Some(c) = &calibration {
        ensure!(
            c.schema == SCHEMA
                && c.binary_sha256 == binary_sha
                && c.config == CONFIG
                && c.formula_revision == formula
                && c.model_ids == labels,
            "calibration scorer/binary/config/formula identity mismatch"
        );
        for codec in &args.codecs {
            for model in &labels {
                ensure!(
                    c.curves
                        .iter()
                        .filter(|v| &v.codec == codec && &v.model == model)
                        .count()
                        == 1,
                    "calibration missing or duplicates a codec/model curve"
                );
            }
        }
    }
    fs::create_dir(&args.out).context("choose a fresh output directory")?;
    fs::create_dir(args.out.join("bounds"))?;
    fs::write(
        args.out.join("INPUTS.json"),
        serde_json::to_vec_pretty(&json!({
            "schema":SCHEMA, "instrument":"zensim-target/demo_matrix", "binary_sha256":binary_sha,
            "config":CONFIG,"formula_revision":formula,"source_manifest":sources,
            "source_manifest_sha256":sha(&fs::read(args.source_manifest.as_ref().unwrap())?),
            "fit_calibration":args.fit_calibration,"calibration_sha256":args.calibration.as_ref().map(|p|fs::read(p).map(|b|sha(&b))).transpose()?,
            "models":labels,"codecs":args.codecs,"bound_steps":args.bound_steps,
            "requested_targets":args.targets,"witness_targets":args.witness_targets,
            "budgets":args.budgets,"tolerance":args.tolerance,"policies":["midpoint","train_curve"],
            "source_interpretation":"opaque sRGB RGB8; no ICC conversion",
            "claim":"bounded scalar-controller experiment; not native diffmap or product qualification",
            "calibration_fit":"median score per native knob; monotone running envelope; nearest nonflat segment inverse and slope"
        }))?,
    )?;
    let mut bounds_file = fs::File::create_new(args.out.join("bounds.jsonl"))?;
    let mut measurements = fs::File::create_new(args.out.join("measurements.jsonl"))?;
    let mut training = TrainingCurves::new();
    let mut expected_cells = 0;
    for (si, s) in sources.sources.iter().enumerate() {
        let (rgb, w, h, hash) = source(&s.path)?;
        ensure!(
            hash == s.sha256,
            "source bytes differ from manifest: {}",
            s.path.display()
        );
        let a = RgbSlice::try_new(
            bytemuck::cast_slice::<u8, [u8; 3]>(&rgb),
            w as usize,
            h as usize,
        )?;
        for (ci, &codec) in codecs.iter().enumerate() {
            let backend = zensim_target::codec::backend_for(codec);
            let (lo, hi) = backend.quality_range();
            let mut recon = Vec::new();
            for i in 0..args.bound_steps {
                let f = i as f32 / (args.bound_steps - 1) as f32;
                let knob = if i == 0 {
                    lo
                } else if i + 1 == args.bound_steps {
                    hi
                } else if codec == CodecKind::Jxl {
                    (lo.ln() + f * (hi.ln() - lo.ln())).exp()
                } else {
                    lo + f * (hi - lo)
                };
                let start = Instant::now();
                let (encoded, decoded) = backend.encode_decode(&rgb, w, h, knob)?;
                let seconds = start.elapsed().as_secs_f64();
                fs::write(
                    args.out
                        .join("bounds")
                        .join(format!("s{si}_c{ci}_q{i}.{}", codec.extension())),
                    &encoded,
                )?;
                recon.push(Reconstruction {
                    knob,
                    encoded,
                    decoded,
                    seconds,
                });
            }
            // Encode each ladder once; all scorer candidates see identical reconstructions.
            for (mi, label) in labels.iter().enumerate() {
                let profile = if mi == 1 {
                    ZensimProfile::D
                } else {
                    ZensimProfile::B
                };
                let named = Zensim::new(profile);
                let mut candidate = if mi >= 2 {
                    Some(BakeScorer::new(&models[mi - 2])?)
                } else {
                    None
                };
                let start = Instant::now();
                let scores: Vec<_> = recon
                    .iter()
                    .map(|r| score(&mut candidate, &named, &a, &r.decoded, w, h, codec))
                    .collect::<Result<_>>()?;
                let bound_score_seconds = start.elapsed().as_secs_f64();
                let min = scores.iter().copied().min_by(f32::total_cmp).unwrap();
                let max = scores.iter().copied().max_by(f32::total_cmp).unwrap();
                let targets =
                    witnessed_targets(&scores, &args.targets, args.witness_targets, args.tolerance);
                let bound_id = format!("s{si}_c{ci}_m{mi}");
                let inversions = scores
                    .windows(2)
                    .filter(|p| {
                        if backend.lower_quality_means_higher_score() {
                            p[1] > p[0] + 1e-5
                        } else {
                            p[1] < p[0] - 1e-5
                        }
                    })
                    .count();
                writeln!(
                    bounds_file,
                    "{}",
                    json!({"id":bound_id,"source":si,"origin":s.origin,
                        "family":s.family,"codec":args.codecs[ci],"model":label,"width":w,"height":h,
                        "attained_min":min,"attained_max":max,"scope":"sampled fixed configuration; not exhaustive codec extrema",
                        "inversions":inversions,"distinct_encodes":recon.iter().map(|r|sha(&r.encoded)).collect::<BTreeSet<_>>().len(),
                        "encode_decode_seconds":recon.iter().map(|r|r.seconds).sum::<f64>(),"score_seconds":bound_score_seconds,
                        "steering_targets":targets,
                        "requests":args.targets.iter().map(|t|json!({"target":t,"status":if scores.iter().any(|v|(v-t).abs()<=args.tolerance){"witnessed"}else if *t<min || *t>max {"outside_measured_envelope"}else{"unwitnessed_inside_envelope"}})).collect::<Vec<_>>(),
                        "probes":recon.iter().zip(&scores).map(|(r,score)|json!({"knob":r.knob,"score":score,"bytes":r.encoded.len(),"encoded_sha256":sha(&r.encoded),"decoded_sha256":sha(&r.decoded)})).collect::<Vec<_>>()
                    })
                )?;
                bounds_file.flush()?;
                if args.fit_calibration {
                    training.entry((ci, mi)).or_default().push(
                        recon
                            .iter()
                            .zip(&scores)
                            .map(|(r, s)| (r.knob, *s))
                            .collect(),
                    );
                    continue;
                }
                let curve = calibration
                    .as_ref()
                    .unwrap()
                    .curves
                    .iter()
                    .find(|c| c.codec == args.codecs[ci] && c.model == *label)
                    .unwrap();
                for (ti, &target) in targets.iter().enumerate() {
                    let seed = curve.seed(target)?;
                    for &budget in &args.budgets {
                        for policy in ["midpoint", "train_curve"] {
                            let spec = TargetSpec {
                                target,
                                tolerance: args.tolerance,
                                max_iterations: budget,
                                profile,
                                seed: if policy == "train_curve" {
                                    Some(seed)
                                } else {
                                    None
                                },
                            };
                            let start = Instant::now();
                            let r = match &mut candidate {
                                Some(c) => target_search_with_bake(&rgb, w, h, codec, spec, c),
                                None => target_search(&rgb, w, h, codec, spec),
                            }?;
                            let seconds = start.elapsed().as_secs_f64();
                            let verify_start = Instant::now();
                            let (encoded, decoded) =
                                backend.encode_decode(&rgb, w, h, r.final_knob)?;
                            ensure!(encoded == r.encoded, "final bitstream is not reproducible");
                            let achieved =
                                score(&mut candidate, &named, &a, &decoded, w, h, codec)?;
                            ensure!(
                                (achieved - r.achieved_score).abs() <= 1e-5,
                                "final score differs from loop"
                            );
                            let verification_seconds = verify_start.elapsed().as_secs_f64();
                            let ar = imgref::ImgRef::new(
                                bytemuck::cast_slice::<u8, rgb::RGB8>(&rgb),
                                w as usize,
                                h as usize,
                            );
                            let dr = imgref::ImgRef::new(
                                bytemuck::cast_slice::<u8, rgb::RGB8>(&decoded),
                                w as usize,
                                h as usize,
                            );
                            let ba = butteraugli::butteraugli(
                                ar,
                                dr,
                                &butteraugli::ButteraugliParams::default(),
                            )?;
                            let ssim = fast_ssim2::compute_ssimulacra2(
                                imgref::ImgRef::new(
                                    bytemuck::cast_slice::<u8, [u8; 3]>(&rgb),
                                    w as usize,
                                    h as usize,
                                ),
                                imgref::ImgRef::new(
                                    bytemuck::cast_slice::<u8, [u8; 3]>(&decoded),
                                    w as usize,
                                    h as usize,
                                ),
                            )?;
                            let name = format!(
                                "{bound_id}_t{ti}_k{budget}_{policy}.{}",
                                codec.extension()
                            );
                            fs::write(args.out.join(&name), &r.encoded)?;
                            writeln!(
                                measurements,
                                "{}",
                                json!({"schema":SCHEMA,"bound_id":bound_id,
                                    "source":si,"origin":s.origin,"family":s.family,"content_class":s.content_class,
                                    "codec":args.codecs[ci],"model":label,"target":target,"target_status":"witnessed",
                                    "policy":policy,"pass_budget":budget,"passes":r.iterations,"achieved":achieved,
                                    "error":achieved-target,"converged":r.converged,"tolerance":args.tolerance,
                                    "knob":r.final_knob,"bytes":r.encoded.len(),"loop_seconds":seconds,
                                    "verification_seconds":verification_seconds,"verification_extra_encodes":1,
                                    "process_peak_rss_kib":rss_kib(),"ssim2":ssim,"butteraugli_pnorm3":ba.pnorm_3,
                                    "encoded":name,"encoded_sha256":sha(&r.encoded),"decoded_sha256":sha(&decoded),
                                    "seed":spec.seed.map(|s|json!({"knob":s.knob,"score_per_knob":s.score_per_knob})),
                                    "probes":r.probes.iter().map(|p|json!({"knob":p.knob,"score":p.achieved_score,"bytes":p.byte_count})).collect::<Vec<_>>()
                                })
                            )?;
                            measurements.flush()?;
                            expected_cells += 1;
                        }
                    }
                }
                eprintln!(
                    "source {} {} {label}: bounds [{min:.3}, {max:.3}], {} witnessed targets complete",
                    s.origin,
                    args.codecs[ci],
                    targets.len()
                );
            }
            if args.fit_calibration {
                eprintln!(
                    "train source {} {} bound ladder complete",
                    s.origin, args.codecs[ci]
                );
            }
        }
    }
    if args.fit_calibration {
        let curves = training
            .iter()
            .map(|(&(ci, mi), rows)| {
                fit(
                    &args.codecs[ci],
                    &labels[mi],
                    rows,
                    zensim_target::codec::backend_for(codecs[ci])
                        .lower_quality_means_higher_score(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let c = Calibration {
            schema: SCHEMA.into(),
            binary_sha256: binary_sha,
            config: CONFIG.into(),
            formula_revision: formula,
            corpus_commit: sources.corpus_commit,
            split_manifest_sha256: sources.split_manifest_sha256,
            training_sources: sources.sources,
            model_ids: labels,
            curves,
        };
        fs::write(
            args.out.join("calibration.json"),
            serde_json::to_vec_pretty(&c)?,
        )?;
    }
    fs::write(
        args.out.join("COMPLETE"),
        serde_json::to_vec_pretty(&json!({
            "schema":SCHEMA,"measurements":expected_cells,"fit_calibration":args.fit_calibration
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witnessed_targets_exclude_gaps_and_preserve_negative_scores() {
        let scores = [-30., -30., 10., 50.];
        assert_eq!(
            witnessed_targets(&scores, &[-50., -10., 10.5, 99.], 3, 1.),
            vec![-30., 10., 10.5, 50.]
        );
    }

    #[test]
    fn inverse_seed_handles_distance_and_plateaus() {
        let curve = Curve {
            codec: "jxl".into(),
            model: "B".into(),
            seed_curve: SeedCurve::fit(
                &[vec![(0.01, 100.), (1., 80.), (2., 80.), (10., -20.)]],
                true,
            )
            .unwrap(),
        };
        let seed = curve.seed(-10.).unwrap();
        assert!((seed.knob - 9.2).abs() < 1e-5);
        assert_eq!(seed.score_per_knob, -12.5);
        assert!(curve.seed(1000.).unwrap().knob >= 0.01);
    }

    #[test]
    fn split_rejects_cross_family_leakage_and_train_eval() {
        let source = |id: &str, split: &str| Source {
            path: id.into(),
            sha256: id.into(),
            origin: id.into(),
            family: "shared-page".into(),
            split: split.into(),
            content_class: "scan".into(),
        };
        let s = SourceSet {
            corpus_commit: "c".into(),
            split_manifest_sha256: "s".into(),
            sources: vec![source("1003", "validate")],
        };
        let c = Calibration {
            schema: SCHEMA.into(),
            binary_sha256: "b".into(),
            config: CONFIG.into(),
            formula_revision: "1".into(),
            corpus_commit: "c".into(),
            split_manifest_sha256: "s".into(),
            training_sources: vec![source("1000", "train")],
            model_ids: vec![],
            curves: vec![],
        };
        assert!(validate_split(&s, Some(&c), false).is_err());
        assert!(validate_split(&s, None, true).is_err());
        assert!(validate_split(&s, None, false).is_ok());
    }
}
