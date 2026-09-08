//! Actual encode/decode/score measurements with explicit, content-bound inputs.
use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};
use zensim_target::{CodecKind, TargetSpec, target_search, target_search_with_bake};

#[derive(Parser)]
struct Args {
    /// Opaque sRGB PNG source. Repeat for each reference; missing inputs fail.
    #[arg(long, required = true)]
    source: Vec<PathBuf>,
    /// Standalone candidate bake; complete embedded head/spline executes in Rust.
    #[arg(long)]
    bake: Vec<PathBuf>,
    #[arg(long, value_delimiter = ',', default_value = "jpeg,webp,avif")]
    codecs: Vec<String>,
    #[arg(
        long,
        value_delimiter = ',',
        allow_hyphen_values = true,
        default_value = "-10,30,70,90,99"
    )]
    targets: Vec<f32>,
    #[arg(long, default_value_t = 1.0)]
    tolerance: f32,
    #[arg(long, default_value_t = 8)]
    max_iterations: u32,
    /// Fresh output directory; existing results cannot be overwritten.
    #[arg(long)]
    out: PathBuf,
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn source(path: &Path) -> Result<(Vec<u8>, u32, u32, String)> {
    let bytes = fs::read(path).with_context(|| path.display().to_string())?;
    let dec = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::strict(),
        &enough::Unstoppable,
    )?;
    let pb = dec.pixels;
    let w = pb.width();
    let h = pb.height();
    let view = pb.as_slice();
    let data = view.as_strided_bytes();
    let desc = pb.descriptor();
    let channels = if desc.layout_compatible(zenpixels::PixelDescriptor::RGB8) {
        3
    } else if desc.layout_compatible(zenpixels::PixelDescriptor::RGBA8) {
        4
    } else {
        bail!("{}: expected RGB8/RGBA8 sRGB PNG", path.display());
    };
    let mut rgb = Vec::with_capacity(w as usize * h as usize * 3);
    for row in 0..h as usize {
        for p in data[row * view.stride()..row * view.stride() + w as usize * channels]
            .chunks_exact(channels)
        {
            ensure!(
                channels == 3 || p[3] == 255,
                "transparent source requires an explicit compositing policy"
            );
            rgb.extend_from_slice(&p[..3]);
        }
    }
    Ok((rgb, w, h, sha(&bytes)))
}
fn rss_kib() -> Option<u64> {
    fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find(|s| s.starts_with("VmHWM:"))?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}
fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        !args.targets.is_empty() && !args.codecs.is_empty(),
        "empty matrix"
    );
    let codecs = args
        .codecs
        .iter()
        .map(|c| CodecKind::parse(c))
        .collect::<Result<Vec<_>>>()?;
    let sources = args
        .source
        .iter()
        .map(|p| source(p))
        .collect::<Result<Vec<_>>>()?;
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
    fs::create_dir(&args.out).context("choose a fresh output directory")?;
    let binary = fs::read(std::env::current_exe()?)?;
    fs::write(
        args.out.join("INPUTS.json"),
        serde_json::to_vec_pretty(&json!({
            "instrument":"zensim-target/demo_matrix", "binary_sha256":sha(&binary),
            "sources": args.source.iter().zip(&sources).map(|(p,s)|json!({"path":p,"sha256":s.3,"width":s.1,"height":s.2})).collect::<Vec<_>>(),
            "bakes":args.bake.iter().zip(&bake_bytes).map(|(p,b)|json!({"path":p,"sha256":sha(b),"composition":"standalone embedded model; no external companions"})).collect::<Vec<_>>(),
            "profiles":["B","D"],"codecs":args.codecs,"targets":args.targets,
            "tolerance":args.tolerance,"max_iterations":args.max_iterations,
            "secant":std::env::var("ZENSIM_TARGET_SECANT").unwrap_or_default(),
            "judges":["fast-ssim2 default CPU","butteraugli default pnorm3","fixed zensim B"],
            "timing":"loop excludes independent judges and final reconstruction verification; scoring samples follow one warmup; VmHWM is process cumulative",
            "source_interpretation":"opaque sRGB RGB8; no ICC conversion"
        }))?,
    )?;
    let mut file = fs::File::create_new(args.out.join("measurements.jsonl"))?;
    for (si, (rgb, w, h, _)) in sources.iter().enumerate() {
        let a = RgbSlice::try_new(
            bytemuck::cast_slice::<u8, [u8; 3]>(rgb),
            *w as usize,
            *h as usize,
        )?;
        for (ci, &codec) in codecs.iter().enumerate() {
            for mi in 0..2 + models.len() {
                let profile = if mi == 1 {
                    ZensimProfile::D
                } else {
                    ZensimProfile::B
                };
                let mut candidate = if mi >= 2 {
                    Some(BakeScorer::new(&models[mi - 2])?)
                } else {
                    None
                };
                let label = if mi == 0 {
                    "B".to_owned()
                } else if mi == 1 {
                    "D".to_owned()
                } else {
                    args.bake[mi - 2]
                        .file_stem()
                        .unwrap()
                        .to_string_lossy()
                        .into_owned()
                };
                for (ti, &target) in args.targets.iter().enumerate() {
                    let spec = TargetSpec {
                        target,
                        tolerance: args.tolerance,
                        max_iterations: args.max_iterations,
                        profile,
                    };
                    let start = Instant::now();
                    let run = match candidate.as_mut() {
                        Some(c) => target_search_with_bake(rgb, *w, *h, codec, spec, c),
                        None => target_search(rgb, *w, *h, codec, spec),
                    };
                    let elapsed = start.elapsed().as_secs_f64();
                    let r = match run {
                        Ok(r) => r,
                        Err(e) => {
                            writeln!(
                                file,
                                "{}",
                                json!({"source":si,"codec":args.codecs[ci],"model":label,"target":target,"error":e.to_string(),"loop_seconds":elapsed})
                            )?;
                            file.flush()?;
                            return Err(e);
                        }
                    };
                    let (verify, decoded) = zensim_target::codec::backend_for(codec)
                        .encode_decode(rgb, *w, *h, r.final_knob)?;
                    ensure!(
                        verify == r.encoded,
                        "final reconstruction did not reproduce the returned bitstream"
                    );
                    let d = RgbSlice::try_new(
                        bytemuck::cast_slice::<u8, [u8; 3]>(&decoded),
                        *w as usize,
                        *h as usize,
                    )?;
                    let named = Zensim::new(profile);
                    let mut score = || -> Result<f64> {
                        Ok(match candidate.as_mut() {
                            Some(c) => c.compute(&a, &d, Some(codec.extension()))?.score(),
                            None => named.compute(&a, &d)?.score(),
                        })
                    };
                    ensure!(
                        (score()? as f32 - r.achieved_score).abs() <= 1e-5,
                        "loop/final scoring mismatch"
                    );
                    let mut times = Vec::new();
                    for _ in 0..5 {
                        let t = Instant::now();
                        std::hint::black_box(score()?);
                        times.push(t.elapsed().as_secs_f64());
                    }
                    let ar = imgref::ImgRef::new(
                        bytemuck::cast_slice::<u8, rgb::RGB8>(rgb),
                        *w as usize,
                        *h as usize,
                    );
                    let dr = imgref::ImgRef::new(
                        bytemuck::cast_slice::<u8, rgb::RGB8>(&decoded),
                        *w as usize,
                        *h as usize,
                    );
                    let ba = butteraugli::butteraugli(
                        ar,
                        dr,
                        &butteraugli::ButteraugliParams::default(),
                    )?;
                    let ssim = fast_ssim2::compute_ssimulacra2(
                        imgref::ImgRef::new(
                            bytemuck::cast_slice::<u8, [u8; 3]>(rgb),
                            *w as usize,
                            *h as usize,
                        ),
                        imgref::ImgRef::new(
                            bytemuck::cast_slice::<u8, [u8; 3]>(&decoded),
                            *w as usize,
                            *h as usize,
                        ),
                    )?;
                    let fixed_b = Zensim::new(ZensimProfile::B).compute(&a, &d)?.score();
                    let name = format!("s{si}_c{ci}_m{mi}_t{ti}.{}", codec.extension());
                    fs::write(args.out.join(&name), &r.encoded)?;
                    writeln!(
                        file,
                        "{}",
                        json!({"source":si,"codec":args.codecs[ci],"model":label,"target":target,
                            "achieved":r.achieved_score,"error":r.achieved_score-target,"converged":r.converged,
                            "knob":r.final_knob,"bytes":r.encoded.len(),"passes":r.iterations,"loop_seconds":elapsed,
                            "score_seconds":times,"process_peak_rss_kib":rss_kib(),"ssim2":ssim,"butteraugli_pnorm3":ba.pnorm_3,"fixed_b":fixed_b,
                            "encoded":name,"encoded_sha256":sha(&r.encoded),"decoded_sha256":sha(&decoded),
                            "probes":r.probes.iter().map(|p|json!({"knob":p.knob,"score":p.achieved_score,"bytes":p.byte_count})).collect::<Vec<_>>()
                        })
                    )?;
                    file.flush()?;
                    eprintln!(
                        "source {si} {} {label} target {target}: {} bytes, {} passes, converged {}",
                        args.codecs[ci],
                        r.encoded.len(),
                        r.iterations,
                        r.converged
                    );
                }
            }
        }
    }
    fs::write(args.out.join("COMPLETE"), "all requested cells measured\n")?;
    Ok(())
}
