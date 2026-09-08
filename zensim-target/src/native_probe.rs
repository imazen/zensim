//! Shared native-codec development experiment: train-only calibration, attained bounds and actual encode budgets.
use crate::{
    CodecKind, SeedCurve, TargetSpec, codec::CodecBackend, target_search_with_backend_and_bake,
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    time::Instant,
};
use zensim::{BakeScorer, RgbSlice};

const ARMS: [&str; 3] = ["scalar", "neutral", "active"];
const FIXED: [f32; 5] = [-10., 30., 70., 90., 99.];
const TOL: f32 = 1.;

#[derive(Clone, Serialize, Deserialize)]
struct Source {
    path: PathBuf,
    sha256: String,
    origin: String,
    family: String,
    split: String,
    content_class: String,
}
#[derive(Serialize, Deserialize)]
struct Sources {
    corpus_commit: String,
    split_manifest_sha256: String,
    sources: Vec<Source>,
}
#[derive(Serialize, Deserialize)]
struct Calibration {
    schema: String,
    config: String,
    driver_sha256: String,
    model_sha256: String,
    training: Sources,
    curves: BTreeMap<String, SeedCurve>,
}
#[derive(Clone, Serialize)]
struct Bound {
    origin: String,
    arm: String,
    knob: f32,
    score: f32,
    seed_score: f32,
    bytes: usize,
    encoded_sha256: String,
    decoded_sha256: String,
}

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn driver_sha() -> Result<String> {
    // Hashing an unstripped research binary must not dominate measured RSS.
    let mut file = fs::File::open(std::env::current_exe()?)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(hash.finalize().iter().map(|v| format!("{v:02x}")).collect())
}

fn fresh(path: &Path) -> Result<()> {
    ensure!(!path.exists(), "output already exists: {}", path.display());
    fs::create_dir_all(path)?;
    Ok(())
}
fn rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|l| l.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn source_set(path: &Path, split: &str) -> Result<Sources> {
    let value: Sources = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(!value.sources.is_empty(), "empty source manifest");
    let mut hashes = BTreeSet::new();
    let mut origins = BTreeSet::new();
    let mut families = BTreeSet::new();
    for s in &value.sources {
        ensure!(
            s.split == split
                && !s.origin.is_empty()
                && s.origin.bytes().all(|b| b.is_ascii_digit())
                && !s.family.is_empty(),
            "source role/identity mismatch"
        );
        ensure!(
            hashes.insert(&s.sha256) && origins.insert(&s.origin) && families.insert(&s.family),
            "duplicate source/family"
        );
        ensure!(
            sha(&fs::read(&s.path)?) == s.sha256,
            "source bytes changed: {}",
            s.origin
        );
    }
    Ok(value)
}
fn pixels(source: &Source) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(&source.path)?;
    ensure!(
        matches!(img.color(), image::ColorType::Rgb8 | image::ColorType::L8),
        "native instrument requires opaque 8-bit RGB/grayscale sources"
    );
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}
fn score(
    codec: CodecKind,
    scorer: &mut BakeScorer<'_>,
    rgb: &[u8],
    decoded: &[u8],
    w: u32,
    h: u32,
) -> Result<f32> {
    ensure!(
        rgb.len() == decoded.len() && rgb.len() == w as usize * h as usize * 3,
        "decoded dimensions/format mismatch"
    );
    let value = scorer
        .compute(
            &RgbSlice::new(rgb.as_chunks::<3>().0, w as usize, h as usize),
            &RgbSlice::new(decoded.as_chunks::<3>().0, w as usize, h as usize),
            Some(codec.extension()),
        )?
        .score() as f32;
    ensure!(value.is_finite(), "nonfinite served score");
    Ok(value)
}
/// One record per complete encode/decode. Pixel buffers are tightly packed sRGB8.
#[derive(Debug, Serialize)]
pub struct NativeProbeWork {
    pub knob: f32,
    pub encode_seconds: f64,
    pub decode_seconds: f64,
    pub internal_reconstructions: usize,
    pub native_pixel_comparisons: usize,
    pub map_evaluations: usize,
    /// Non-neutral maps actually used by the encoder, excluding unused final maps.
    pub consumed_maps: usize,
    pub native_loop_ms: f64,
    pub encoded_sha256: String,
    pub decoded_sha256: String,
}
/// An instrumented adapter performs exactly one full encode per call.
pub trait NativeProbeBackend: CodecBackend {
    fn take_work(&self) -> Vec<NativeProbeWork>;
}
/// Codec-owned wiring; reusable experiment and scoring live in this module.
pub trait NativeProbeCodec {
    fn codec(&self) -> CodecKind;
    fn configuration(&self) -> &str;
    fn quality_knots(&self) -> &[f32];
    fn bound_encodes(&self, arm: &str) -> usize;
    fn backend<'a>(
        &'a self,
        arm: &str,
        model: &'a zenpredict::Model,
        scratch: &Path,
        bake: &Path,
    ) -> Result<Box<dyn NativeProbeBackend + 'a>>;
    /// Decode emitted bytes to exactly width*height*3 tightly packed opaque sRGB8 bytes.
    fn decode(&self, encoded: &[u8], width: u32, height: u32) -> Result<Vec<u8>>;
}
fn validate_configuration(codec: &impl NativeProbeCodec) -> Result<()> {
    let knots = codec.quality_knots();
    ensure!(
        knots.len() >= 2
            && knots.iter().all(|k| k.is_finite())
            && knots.windows(2).all(|w| w[0] < w[1]),
        "invalid quality ladder"
    );
    ensure!(
        ARMS.iter()
            .all(|arm| (1..=3).contains(&codec.bound_encodes(arm))),
        "bound sequence must use 1..3 actual encodes"
    );
    Ok(())
}
fn validate_work(
    work: &[NativeProbeWork],
    knob: f32,
    encoded: &[u8],
    decoded: &[u8],
) -> Result<()> {
    ensure!(
        work.len() == 1,
        "one work record per complete encode required"
    );
    let w = &work[0];
    ensure!(
        w.knob == knob
            && w.encoded_sha256 == sha(encoded)
            && w.decoded_sha256 == sha(decoded)
            && [w.encode_seconds, w.decode_seconds, w.native_loop_ms]
                .iter()
                .all(|v| v.is_finite() && *v >= 0.),
        "work record does not describe this encode/decode"
    );
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn ladder(
    codec: &impl NativeProbeCodec,
    model: &zenpredict::Model,
    source: &Source,
    arm: &str,
    bake: &Path,
    out: &Path,
    scorer: &mut BakeScorer<'_>,
    sink: &mut fs::File,
) -> Result<Vec<Bound>> {
    let (rgb, w, h) = pixels(source)?;
    let mut rows = Vec::new();
    for (i, &knob) in codec.quality_knots().iter().enumerate() {
        let native = codec.backend(arm, model, out, bake)?;
        let mut work = Vec::new();
        let mut probes = Vec::new();
        let mut score_seconds = 0.;
        let mut last = None;
        for _ in 0..codec.bound_encodes(arm) {
            let (encoded, decoded) = native.encode_decode(&rgb, w, h, knob)?;
            let current = native.take_work();
            validate_work(&current, knob, &encoded, &decoded)?;
            let start = Instant::now();
            let value = score(codec.codec(), scorer, &rgb, &decoded, w, h)?;
            score_seconds += start.elapsed().as_secs_f64();
            probes.push(value);
            work.extend(current);
            last = Some((encoded, decoded, value));
        }
        let (encoded, decoded, value) = last.context("empty bound sequence")?;
        let row = Bound {
            origin: source.origin.clone(),
            arm: arm.into(),
            knob,
            score: value,
            seed_score: probes[0],
            bytes: encoded.len(),
            encoded_sha256: sha(&encoded),
            decoded_sha256: sha(&decoded),
        };
        let stem = format!("{}-{arm}-{i}", source.origin);
        fs::write(
            out.join(format!("{stem}.{}", codec.codec().extension())),
            encoded,
        )?;
        image::RgbImage::from_raw(w, h, decoded)
            .context("decoded image shape")?
            .save(out.join(format!("{stem}.png")))?;
        writeln!(
            sink,
            "{}",
            json!({"bound":row,"work":work,"bound_probe_scores":probes,"score_seconds":score_seconds,"process_peak_rss_kib":rss_kib()})
        )?;
        rows.push(row);
    }
    Ok(rows)
}

pub fn fit(codec: &impl NativeProbeCodec, root: &Path, bake: &Path, out: &Path) -> Result<()> {
    validate_configuration(codec)?;
    let training = source_set(&root.join("train_source_manifest.json"), "train")?;
    let bytes = fs::read(bake)?;
    let model = zenpredict::Model::from_bytes(&bytes)?;
    fresh(out)?;
    let mut scorer = BakeScorer::new(&model)?;
    let mut sink = fs::File::create(out.join("bounds.jsonl"))?;
    let mut curves = BTreeMap::new();
    for arm in ARMS {
        let mut training_rows = Vec::new();
        for source in &training.sources {
            let rows = ladder(
                codec,
                &model,
                source,
                arm,
                bake,
                out,
                &mut scorer,
                &mut sink,
            )?;
            training_rows.push(rows.iter().map(|r| (r.knob, r.seed_score)).collect());
        }
        let backend = codec.backend(arm, &model, out, bake)?;
        curves.insert(
            arm.into(),
            SeedCurve::fit(&training_rows, backend.lower_quality_means_higher_score())?,
        );
    }
    let calibration = Calibration {
        schema: "native-codec-target-v1".into(),
        config: codec.configuration().into(),
        driver_sha256: driver_sha()?,
        model_sha256: sha(&bytes),
        training,
        curves,
    };
    fs::write(
        out.join("calibration.json"),
        serde_json::to_vec_pretty(&calibration)?,
    )?;
    fs::write(
        out.join("COMPLETE"),
        "training ladders and calibration complete\n",
    )?;
    Ok(())
}

pub fn evaluate(
    codec: &impl NativeProbeCodec,
    root: &Path,
    calibration_path: &Path,
    bake: &Path,
    out: &Path,
) -> Result<()> {
    validate_configuration(codec)?;
    let validation = source_set(&root.join("source_manifest.json"), "validate")?;
    let calibration: Calibration = serde_json::from_slice(&fs::read(calibration_path)?)?;
    let bytes = fs::read(bake)?;
    ensure!(
        calibration.schema == "native-codec-target-v1"
            && calibration.config == codec.configuration()
            && calibration.driver_sha256 == driver_sha()?
            && calibration.model_sha256 == sha(&bytes),
        "calibration model/configuration/driver mismatch"
    );
    ensure!(
        calibration.training.corpus_commit == validation.corpus_commit
            && calibration.training.split_manifest_sha256 == validation.split_manifest_sha256,
        "corpus/split changed"
    );
    for train in &calibration.training.sources {
        ensure!(
            train.split == "train"
                && !validation.sources.iter().any(|v| v.origin == train.origin
                    || v.family == train.family
                    || v.sha256 == train.sha256),
            "training/validation family overlap"
        );
    }
    ensure!(
        calibration.curves.len() == ARMS.len(),
        "calibration arm count mismatch"
    );
    for arm in ARMS {
        calibration
            .curves
            .get(arm)
            .context("missing calibration arm")?
            .estimate(50.)?;
    }
    let model = zenpredict::Model::from_bytes(&bytes)?;
    let mut scorer = BakeScorer::new(&model)?;
    fresh(out)?;
    let mut sink = fs::File::create(out.join("bounds.jsonl"))?;
    let mut bounds = BTreeMap::new();
    // Every attained bound is established before any target controller runs.
    for source in &validation.sources {
        for arm in ARMS {
            bounds.insert(
                (source.origin.clone(), arm.to_string()),
                ladder(
                    codec,
                    &model,
                    source,
                    arm,
                    bake,
                    out,
                    &mut scorer,
                    &mut sink,
                )?,
            );
        }
    }
    sink.sync_all()?;
    fs::write(
        out.join("INPUTS.json"),
        serde_json::to_vec_pretty(
            &json!({"schema":"native-codec-target-v1","config":codec.configuration(),"codec":codec.codec().extension(),"quality_knots":codec.quality_knots(),"bound_encodes":ARMS.iter().map(|arm|(arm,codec.bound_encodes(arm))).collect::<BTreeMap<_,_>>(),"model_sha256":sha(&bytes),"driver_sha256":driver_sha()?,"calibration_sha256":sha(&fs::read(calibration_path)?),"sources":validation,"fixed_requests":FIXED,"tolerance":TOL,"budgets":[1,2,3],"policies":["midpoint","train_curve"],"arms":ARMS,"bounds_are_not_controller_inputs":true}),
        )?,
    )?;
    let mut measurements = fs::File::create(out.join("measurements.jsonl"))?;
    let mut coverage = Vec::new();
    let mut cases = 0;
    for source in &validation.sources {
        let (rgb, w, h) = pixels(source)?;
        let neutral = &bounds[&(source.origin.clone(), "neutral".into())];
        let mut scores: Vec<f32> = neutral.iter().map(|r| r.score).collect();
        scores.sort_by(f32::total_cmp);
        scores.dedup();
        let mut targets = FIXED.to_vec();
        for i in 0..5 {
            targets.push(scores[i * (scores.len() - 1) / 4]);
        }
        targets.sort_by(f32::total_cmp);
        targets.dedup();
        let feasible = |arm: &str, target: f32| {
            bounds[&(source.origin.clone(), arm.into())]
                .iter()
                .any(|r| (r.score - target).abs() <= TOL)
        };
        for &target in &targets {
            coverage.push(json!({"origin":source.origin,"target":target,"scalar_witnessed":feasible("scalar",target),"neutral_witnessed":feasible("neutral",target),"active_witnessed":feasible("active",target)}));
        }
        targets.retain(|&t| ARMS.iter().all(|arm| feasible(arm, t)));
        for arm in ARMS {
            for policy in ["midpoint", "train_curve"] {
                for budget in 1..=3 {
                    for &target in &targets {
                        let native = codec.backend(arm, &model, out, bake)?;
                        let seed = if policy == "train_curve" {
                            Some(calibration.curves[arm].estimate(target)?)
                        } else {
                            None
                        };
                        let start = Instant::now();
                        let result = target_search_with_backend_and_bake(
                            &rgb,
                            w,
                            h,
                            codec.codec(),
                            TargetSpec {
                                target,
                                tolerance: TOL,
                                max_iterations: budget,
                                seed,
                                ..TargetSpec::default()
                            },
                            &*native,
                            &mut scorer,
                        )?;
                        let search_seconds = start.elapsed().as_secs_f64();
                        let work = native.take_work();
                        ensure!(
                            work.len() == result.probes.len()
                                && result.iterations as usize == work.len(),
                            "complete-encode accounting mismatch"
                        );
                        let start = Instant::now();
                        let verified = codec.decode(&result.encoded, w, h)?;
                        let achieved = score(codec.codec(), &mut scorer, &rgb, &verified, w, h)?;
                        let verification_seconds = start.elapsed().as_secs_f64();
                        ensure!(
                            (achieved - result.achieved_score).abs() <= 1e-5,
                            "selected bitstream score disagrees with controller"
                        );
                        let file = format!("case-{cases}.{}", codec.codec().extension());
                        fs::write(out.join(&file), &result.encoded)?;
                        let decoded_file = format!("case-{cases}.png");
                        image::RgbImage::from_raw(w, h, verified)
                            .context("verification shape")?
                            .save(out.join(&decoded_file))?;
                        let probes:Vec<_>=result.probes.iter().map(|p|json!({"knob":p.knob,"score":p.achieved_score,"bytes":p.byte_count})).collect();
                        writeln!(
                            measurements,
                            "{}",
                            json!({"origin":source.origin,"family":source.family,"class":source.content_class,"arm":arm,"policy":policy,"budget":budget,"target":target,"achieved":achieved,"signed_error":achieved-target,"bytes":result.encoded.len(),"full_encodes":work.len(),"search_pixel_comparisons":result.probes.len(),"terminal_decodes":1,"terminal_pixel_comparisons":1,"search_seconds":search_seconds,"verification_seconds":verification_seconds,"total_seconds":search_seconds+verification_seconds,"process_peak_rss_kib":rss_kib(),"probes":probes,"native_work":work,"bitstream":file,"decoded":decoded_file,"encoded_sha256":sha(&result.encoded)})
                        )?;
                        cases += 1;
                    }
                }
            }
        }
    }
    fs::write(
        out.join("coverage.json"),
        serde_json::to_vec_pretty(&coverage)?,
    )?;
    fs::write(
        out.join("COMPLETE"),
        serde_json::to_vec_pretty(
            &json!({"cases":cases,"bounds":validation.sources.len()*ARMS.len()*codec.quality_knots().len()}),
        )?,
    )?;
    Ok(())
}
