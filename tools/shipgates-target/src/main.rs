//! Scalar-only adapter validation. The shared instrument owns calibration,
//! witnesses, search and accounting; codec-owned executables own encoding.
#![forbid(unsafe_code)]
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};
use zensim_target::{CodecKind, SeedCurve, TargetSpec, codec, target_search_with_backend_and_bake};
// Compile the existing owner as a private module so selecting a subset adds
// no public API to the unpublished library or to any shipped codec.
#[path = "../../../zensim-target/src/native_probe.rs"]
#[allow(dead_code)]
mod instrument;
use instrument::{NativeProbeBackend, NativeProbeCodec, NativeProbeWork};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
struct Adapter {
    codec: CodecKind,
    binary: PathBuf,
    oracle: PathBuf,
    root: PathBuf,
    source_by_pixels: BTreeMap<String, PathBuf>,
    sequence: Cell<usize>,
    knots: Vec<f32>,
    config: String,
}
impl Adapter {
    fn decode_native(&self, encoded: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
        let output = match self.codec {
            CodecKind::Jpeg => {
                let d = zenjpeg::decoder::Decoder::new().decode(encoded, enough::Unstoppable)?;
                ensure!((d.width(), d.height()) == (w, h), "JPEG dimensions");
                d.into_pixels_u8().context("JPEG RGB8")?
            }
            CodecKind::Webp => {
                let (pixels, dw, dh) = zenwebp::oneshot::decode_rgb(encoded)?;
                ensure!((dw, dh) == (w, h), "WebP dimensions");
                pixels
            }
            CodecKind::Avif => {
                let cfg = zenavif::DecoderConfig::new().prefer_8bit(true);
                let stop = almost_enough::StopToken::new(enough::Unstoppable);
                let d = zenavif::decode_with(encoded, &cfg, &stop)?;
                let p = d.as_slice();
                ensure!((d.width(), d.height()) == (w, h), "AVIF dimensions");
                ensure!(
                    p.descriptor()
                        .layout_compatible(zenpixels::PixelDescriptor::RGB8),
                    "AVIF RGB8"
                );
                let mut pixels = Vec::new();
                for row in 0..h as usize {
                    let start = row * p.stride();
                    pixels.extend_from_slice(&p.as_strided_bytes()[start..start + w as usize * 3]);
                }
                pixels
            }
            _ => anyhow::bail!("unsupported codec"),
        };
        ensure!(
            output.len() == w as usize * h as usize * 3,
            "decoded RGB shape"
        );
        Ok(output)
    }
}
impl NativeProbeCodec for Adapter {
    fn codec(&self) -> CodecKind {
        self.codec
    }
    fn configuration(&self) -> &str {
        &self.config
    }
    fn quality_knots(&self) -> &[f32] {
        &self.knots
    }
    fn bound_encodes(&self, _arm: &str) -> usize {
        1
    }
    fn backend<'a>(
        &'a self,
        arm: &str,
        _model: &'a zenpredict::Model,
        _scratch: &Path,
        _bake: &Path,
    ) -> Result<Box<dyn NativeProbeBackend + 'a>> {
        ensure!(
            arm == "scalar",
            "this adapter cannot qualify native spatial arms"
        );
        Ok(Box::new(Backend {
            adapter: self,
            work: RefCell::new(Vec::new()),
        }))
    }
    fn decode(&self, encoded: &[u8], w: u32, h: u32) -> Result<Vec<u8>> {
        self.decode_native(encoded, w, h)
    }
}
struct Backend<'a> {
    adapter: &'a Adapter,
    work: RefCell<Vec<NativeProbeWork>>,
}
impl NativeProbeBackend for Backend<'_> {
    fn take_work(&self) -> Vec<NativeProbeWork> {
        std::mem::take(&mut *self.work.borrow_mut())
    }
}
impl codec::CodecBackend for Backend<'_> {
    fn quality_range(&self) -> (f32, f32) {
        if self.adapter.codec == CodecKind::Avif {
            (1., 255.)
        } else {
            (0., 100.)
        }
    }
    fn lower_quality_means_higher_score(&self) -> bool {
        self.adapter.codec == CodecKind::Avif
    }
    fn encode_decode(&self, rgb: &[u8], w: u32, h: u32, knob: f32) -> Result<(Vec<u8>, Vec<u8>)> {
        let a = self.adapter;
        ensure!(rgb.len() == w as usize * h as usize * 3, "source RGB shape");
        let source = a
            .source_by_pixels
            .get(&sha(rgb))
            .context("unregistered source pixels")?;
        // Memory and storage admission are checked again before every actual encode.
        let available = fs::read_to_string("/proc/meminfo")?
            .lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .context("MemAvailable")?
            .split_whitespace()
            .nth(1)
            .context("available value")?
            .parse::<u64>()?;
        ensure!(available >= 8 * 1024 * 1024, "below8GiB memory reserve");
        let disk = Command::new("df")
            .args(["--output=avail", "-B1", "/home"])
            .output()?;
        ensure!(disk.status.success(), "disk admission command");
        let free: u64 = std::str::from_utf8(&disk.stdout)?
            .split_whitespace()
            .last()
            .context("disk bytes")?
            .parse()?;
        ensure!(
            free >= 40 * 1024 * 1024 * 1024,
            "below40GiB local disk reserve"
        );
        let seq = a.sequence.get();
        a.sequence.set(seq + 1);
        let dir = a.root.join(format!("probe-{seq:06}"));
        fs::create_dir(&dir)?;
        let probe = dir.join("native");
        let mut command = Command::new(&a.binary);
        for (name, _) in std::env::vars() {
            if name.starts_with("ZENSIM_")
                || name.starts_with("ZENJPEG_ZQ_")
                || name.starts_with("ZENWEBP_ZQ_")
                || name.starts_with("AVIF_")
            {
                command.env_remove(name);
            }
        }
        command
            .env("ZENSIM_FORMULA_REV", "5")
            .env("RAYON_NUM_THREADS", "1");
        match a.codec {
            CodecKind::Jpeg => {
                command
                    .args(["--image"])
                    .arg(source)
                    .args(["--quality", &knob.to_string(), "--out-dir"])
                    .arg(&probe);
            }
            CodecKind::Webp => {
                command
                    .arg("--image")
                    .arg(source)
                    .args([
                        "--global-quality",
                        &knob.to_string(),
                        "--method",
                        "6",
                        "--candidate-out",
                    ])
                    .arg(&probe);
            }
            CodecKind::Avif => {
                command
                    .arg("--image")
                    .arg(source)
                    .args(["--global-cq", &knob.to_string(), "--out-dir"])
                    .arg(&probe)
                    .env("AVIF_ZENSIM_BACKEND", "zenav1-svt")
                    .env("AVIF_ZENSIM_SPEED", "6")
                    .env("AVIF_RD_ARTIFACT_ROOT", dir.join("native-artifacts"));
            }
            _ => anyhow::bail!("unsupported codec"),
        }
        let explicit_environment: BTreeMap<_, _> = command
            .get_envs()
            .map(|(key, value)| {
                (
                    key.to_string_lossy().into_owned(),
                    value.map(|v| v.to_string_lossy().into_owned()),
                )
            })
            .collect();
        let argv: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();
        fs::write(
            dir.join("COMMAND.json"),
            serde_json::to_vec_pretty(
                &json!({"program":a.binary,"argv":argv,"explicit_environment":explicit_environment}),
            )?,
        )?;
        let start = Instant::now();
        let result = command.output()?;
        fs::write(dir.join("stdout.log"), &result.stdout)?;
        fs::write(dir.join("stderr.log"), &result.stderr)?;
        ensure!(
            result.status.success(),
            "native encoder failed: {}",
            dir.display()
        );
        let encode_seconds = start.elapsed().as_secs_f64();
        let manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(probe.join("result.json"))?)?;
        ensure!(
            manifest["full_encodes"] == 1,
            "one encode per native probe required"
        );
        let encoded = fs::read(probe.join(format!("selected.{}", a.codec.extension())))?;
        ensure!(
            manifest["encoded_sha256"] == sha(&encoded),
            "encoded manifest hash"
        );
        let start = Instant::now();
        let decoded = a.decode_native(&encoded, w, h)?;
        let decode_seconds = start.elapsed().as_secs_f64();
        let png = image::open(probe.join("selected.png"))?.to_rgb8();
        ensure!(
            png.dimensions() == (w, h) && png.as_raw() == &decoded,
            "terminal decoder differs from original native RGB"
        );
        for (kind, bytes, ext) in [
            ("encoded", encoded.as_slice(), a.codec.extension()),
            ("decoded", decoded.as_slice(), "rgb8"),
        ] {
            let path = a
                .root
                .join("artifacts")
                .join(kind)
                .join(format!("{}.{}", sha(bytes), ext));
            if path.exists() {
                ensure!(fs::read(&path)? == bytes, "CAS collision")
            } else {
                fs::write(&path, bytes)?;
            }
        }
        let encoded_path = a.root.join("artifacts/encoded").join(format!(
            "{}.{}",
            sha(&encoded),
            a.codec.extension()
        ));
        let mut oracle = Command::new(&a.oracle);
        match a.codec {
            CodecKind::Jpeg => {
                oracle
                    .args(["-rgb", "-outfile"])
                    .arg(dir.join("oracle.ppm"))
                    .arg(&encoded_path);
            }
            CodecKind::Webp => {
                oracle
                    .arg(&encoded_path)
                    .args(["-ppm", "-o"])
                    .arg(dir.join("oracle.ppm"));
            }
            CodecKind::Avif => {
                oracle
                    .args([
                        "-v",
                        "error",
                        "-xerror",
                        "-err_detect",
                        "explode",
                        "-threads",
                        "1",
                        "-c:v",
                        "libdav1d",
                        "-i",
                    ])
                    .arg(&encoded_path)
                    .args(["-f", "null", "-"]);
            }
            _ => unreachable!(),
        }
        let checked = oracle.output()?;
        fs::write(dir.join("oracle.stdout"), checked.stdout)?;
        fs::write(dir.join("oracle.stderr"), checked.stderr)?;
        ensure!(
            checked.status.success(),
            "independent decode oracle failed: {}",
            dir.display()
        );
        fs::write(
            a.root.join("PROGRESS.json"),
            serde_json::to_vec_pretty(
                &json!({"completed_probes":seq+1,"source":source,"knob":knob,"encoded_sha256":sha(&encoded),"bytes":encoded.len(),"first_party_rgb_and_independent_decode_verified":true}),
            )?,
        )?;
        let work = NativeProbeWork {
            knob,
            encode_seconds,
            decode_seconds,
            internal_reconstructions: 0,
            native_pixel_comparisons: 0,
            map_evaluations: 0,
            consumed_maps: 0,
            native_loop_ms: 0.,
            encoded_sha256: sha(&encoded),
            decoded_sha256: sha(&decoded),
        };
        fs::write(dir.join("WORK.json"), serde_json::to_vec_pretty(&work)?)?;
        self.work.borrow_mut().push(work);
        Ok((encoded, decoded))
    }
}
fn main() -> Result<()> {
    ensure!(
        std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok("5"),
        "explicit ZENSIM_FORMULA_REV=5 required"
    );
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        args.len() == 8,
        "usage: shipgates-target fit|eval codec native-binary inputs bake calibration-or-unused output oracle"
    );
    let action = args[0].to_str().context("action")?;
    let kind = CodecKind::parse(args[1].to_str().context("codec")?)?;
    ensure!(
        matches!(kind, CodecKind::Jpeg | CodecKind::Webp | CodecKind::Avif),
        "unsupported codec"
    );
    let binary = PathBuf::from(&args[2]).canonicalize()?;
    let inputs = PathBuf::from(&args[3]);
    let bake = PathBuf::from(&args[4]);
    let output = PathBuf::from(&args[6]);
    let retained = output.with_extension("retained");
    ensure!(!retained.exists(), "fresh retained output required");
    fs::create_dir_all(retained.join("artifacts/encoded"))?;
    fs::create_dir_all(retained.join("artifacts/decoded"))?;
    let mut source_by_pixels = BTreeMap::new();
    for name in ["train_source_manifest.json", "source_manifest.json"] {
        let manifest: serde_json::Value = serde_json::from_slice(&fs::read(inputs.join(name))?)?;
        for s in manifest["sources"].as_array().context("sources")? {
            let path = PathBuf::from(s["path"].as_str().context("source path")?);
            ensure!(sha(&fs::read(&path)?) == s["sha256"], "source file changed");
            let pixels = image::open(&path)?.to_rgb8();
            source_by_pixels.insert(sha(pixels.as_raw()), path);
        }
    }
    let encoder_sha = sha(&fs::read(&binary)?);
    let mut a = Adapter {
        codec: kind,
        binary,
        oracle: PathBuf::from(&args[7]).canonicalize()?,
        root: retained.clone(),
        source_by_pixels,
        sequence: Cell::new(0),
        knots: if kind == CodecKind::Avif {
            vec![
                1., 16., 32., 48., 64., 80., 96., 112., 128., 144., 160., 176., 192., 208., 224.,
                240., 255.,
            ]
        } else {
            (0..=20).map(|n| n as f32 * 5.).collect()
        },
        config: format!(
            "scalar-only;codec={};encoder={encoder_sha};formula5;opaque-sRGB8;full-native-decode;",
            kind.extension()
        ),
    };
    a.config += match kind {
        CodecKind::Jpeg => "444,q0-100",
        CodecKind::Webp => "method6,q0-100",
        CodecKind::Avif => {
            "zenav1-svt,s6,420,8bit,threads1,CQ1-255;backend-env=zenav1-svt;router-source925d832b"
        }
        _ => unreachable!(),
    };
    fs::write(
        retained.join("INPUTS.json"),
        serde_json::to_vec_pretty(
            &json!({"configuration":a.config,"encoder":a.binary,"encoder_sha256":encoder_sha,"maps":0,"scope":"scalar adapter validation; no active-map qualification"}),
        )?,
    )?;
    match action {
        "smoke" => {
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(inputs.join("train_source_manifest.json"))?)?;
            let path = PathBuf::from(
                manifest["sources"][0]["path"]
                    .as_str()
                    .context("TRAIN smoke source")?,
            );
            let rgb = image::open(path)?.to_rgb8();
            let native = Backend {
                adapter: &a,
                work: RefCell::new(Vec::new()),
            };
            let knob = if kind == CodecKind::Avif { 128. } else { 50. };
            let (encoded, decoded) = codec::CodecBackend::encode_decode(
                &native,
                rgb.as_raw(),
                rgb.width(),
                rgb.height(),
                knob,
            )?;
            fs::create_dir(&output)?;
            fs::write(
                output.join("COMPLETE.json"),
                serde_json::to_vec_pretty(
                    &json!({"encoded_sha256":sha(&encoded),"decoded_sha256":sha(&decoded),"work":native.take_work()}),
                )?,
            )?;
            Ok(())
        }
        "fit" => instrument::fit_arms(&a, &inputs, &bake, &output, &["scalar"]),
        "eval" => instrument::evaluate_arms(
            &a,
            &inputs,
            &PathBuf::from(&args[5]),
            &bake,
            &output,
            &["scalar"],
        ),
        _ => anyhow::bail!("unknown action"),
    }
}
