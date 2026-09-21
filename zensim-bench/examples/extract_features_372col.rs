//! Extract 372-column zensim feature CSVs for arbitrary corpora.
//!
//! Used by the 2026-05-15 full-feature sweep that re-creates the training
//! ingest for V_20a IW / V_20b distortion-manifold bakes. The historical
//! 372-col CSV used to train V_20a was lost; this binary rebuilds it
//! deterministically against current source so downstream trainers can
//! consume a single canonical schema.
//!
//! Output schema (matches `/mnt/v/zen/zensim-training/2026-05-14-clean/`):
//!   ref_basename, human_score, f0, f1, ..., f371
//!
//! Features are computed via `compute_zensim_with_config` with
//! `extended_features = true` and `compute_iw_features = true` so the
//! emitted vector is 4 scales × 3 channels × 31 features per channel
//! = 372 columns (basic + peaks + masked + IW).
//!
//! ## Build (2026-09-04)
//!
//! The codecs are REQUIRED features, so no build exists in which this binary
//! runs but silently cannot read AVIF / JXL / BMP:
//!
//! ```text
//! cargo build --release -p zensim-bench --example extract_features_372col \
//!     --features training,zen-decode
//! ```
//!
//! A pair that cannot be decoded, whose dimensions disagree, or that is too
//! small for the pyramid is a HARD failure that aborts the run. Tolerating any
//! is the caller's explicit decision: `--allow-failures N` (default 0).
//!
//! For admitted `pairs-tsv` inputs, `--audit-jsonl PATH --audit-ssim2`
//! additionally records fast-ssim2 scores on the audit's decoded RGB8 buffers.
//! Pixel hashes bind both metrics to the same inputs. This flag does not alter
//! feature extraction or permit dropped pairs.
//!
//! `--input-contract sdr-native-clip-v1` retains native u16/linear-f32
//! samples for declared SDR primaries and converts arbitrary ICC through the
//! existing CMS. It requires explicit pairs-tsv and a fresh audit, refuses HDR
//! and unsupported/conflicting color interpretations, and records a separate
//! input era. The RGB8-only peer audit is refused for this contract.
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example extract_features_372col -- \
//!     --corpus konjnd \
//!     --path /mnt/v/datasets/KonJND-1k/KonJND-1k \
//!     --out  /mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_features_372col_2026-05-15.csv
//!
//!   cargo run --release -p zensim-bench --example extract_features_372col -- \
//!     --corpus aic3 \
//!     --path /mnt/v/dataset/aic3_ctc_epfl/decoded/info.csv \
//!     --out  /mnt/v/zen/zensim-training/2026-05-15-full-features/aic3_features_372col_2026-05-15.csv

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use zensim::{ZensimConfig, compute_zensim_with_config};

// THE decode owner (imazen-only: magic-byte detection via zencodec + zenjpeg /
// zenpng / zenwebp / zenavif / zenjxl). Shared verbatim with
// `verify_bitstream_decode`; never re-implement decoding here.
#[path = "shared/zen_decode.rs"]
mod zen_decode;

#[path = "shared/score_input.rs"]
mod score_input;
use score_input::{InputContract, ScoreInput};

#[path = "extract_features_372col/audit.rs"]
mod audit;

#[derive(Debug, Clone)]
struct Pair {
    reference: PathBuf,
    distorted: PathBuf,
    human_score: f64,
    ref_basename: String,
    /// Optional named extra target columns emitted alongside
    /// `human_score` (e.g., IW-SSIM on safesyn). Each entry becomes a
    /// CSV column between `human_score` and `f0`. The trainer's
    /// `--target-column NAME` flag (T1.1) selects which column is the
    /// regression target.
    extra_targets: Vec<(String, f64)>,
}

type FeatureRow = (String, f64, Vec<(String, f64)>, Vec<f64>);

fn main() {
    let mut args = std::env::args().skip(1);
    let mut corpus: Option<String> = None;
    let mut path: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut max_pairs: usize = usize::MAX;
    let mut allow_failures: usize = 0;
    let mut audit_out = None;
    let mut audit_bake = None;
    let mut audit_head = None;
    let mut audit_ensemble = None;
    let mut audit_weights = None;
    let mut audit_ssim2 = false;
    let mut sampling = None;
    let mut full_944 = false;
    let mut full_986 = false;
    let mut dvifm_spec = None;
    let mut dvifm_blocks = None;
    let mut dvifm_cap = 0usize;
    let mut dvifm_quant_f16 = false;
    let mut dvifm_hist = None;
    let mut input_contract = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--input-contract" => audit::take_value(&mut input_contract, args.next()),
            "--sampling" => sampling = Some(args.next().expect("--sampling value")),
            "--full-944" => full_944 = true,
            "--full-986" => full_986 = true,
            "--dvifm-spec" => dvifm_spec = Some(args.next().expect("--dvifm-spec value")),
            "--dvifm-block-stats" => {
                dvifm_blocks = Some(args.next().expect("--dvifm-block-stats value"))
            }
            "--dvifm-cap" => {
                dvifm_cap = args.next().expect("--dvifm-cap value").parse().unwrap()
            }
            "--dvifm-quant" => {
                dvifm_quant_f16 = match args.next().expect("--dvifm-quant value").as_str() {
                    "f16" => true,
                    "f32" => false,
                    other => panic!("--dvifm-quant must be f16|f32, got {other}"),
                }
            }
            "--dvifm-hist" => {
                dvifm_hist = Some(args.next().expect("--dvifm-hist value"))
            }
            "--corpus" => corpus = Some(args.next().unwrap()),
            "--path" => path = Some(args.next().unwrap().into()),
            "--out" => out = Some(args.next().unwrap().into()),
            "--max-pairs" => max_pairs = args.next().unwrap().parse().unwrap(),
            // How many pairs may fail extraction before the run aborts.
            // Default 0 — see the NO GRACEFUL SKIPS block below.
            "--allow-failures" => allow_failures = args.next().unwrap().parse().unwrap(),
            "--audit-jsonl" => audit::take_path(&mut audit_out, args.next()),
            "--audit-bake" => audit::take_path(&mut audit_bake, args.next()),
            "--audit-corruption-head" => audit::take_path(&mut audit_head, args.next()),
            "--audit-ensemble" => audit::take_value(&mut audit_ensemble, args.next()),
            "--audit-ensemble-weights" => audit::take_value(&mut audit_weights, args.next()),
            "--audit-ssim2" => audit_ssim2 = true,
            other => {
                eprintln!("unknown arg: {other}");
                std::process::exit(1);
            }
        }
    }
    let corpus = corpus.expect("--corpus REQUIRED (konjnd or aic3)");
    let path = path.expect("--path REQUIRED");
    let out = out.expect("--out REQUIRED");
    assert!(
        !(full_944 && full_986),
        "--full-944 and --full-986 are mutually exclusive"
    );
    assert!(
        (dvifm_spec.is_none() && dvifm_blocks.is_none() && dvifm_hist.is_none())
            || full_986,
        "--dvifm-spec/--dvifm-block-stats/--dvifm-hist require --full-986 (the research path)"
    );
    assert!(
        dvifm_hist.is_none() || dvifm_spec.is_some(),
        "--dvifm-hist needs --dvifm-spec (per-level g/edge constants)"
    );
    assert!(
        dvifm_hist.is_none() || dvifm_blocks.is_some(),
        "--dvifm-hist accompanies --dvifm-block-stats"
    );
    assert!(
        !full_986 || sampling.is_none(),
        "--full-986 does not take a sampling contract"
    );
    assert!(
        !full_944 || sampling.as_deref().is_none_or(|s| s.starts_with("v2:")),
        "--full-944 sampling requires a direct v2 contract"
    );
    assert!(
        full_944 || sampling.as_deref().is_none_or(|s| !s.starts_with("v2:")),
        "direct v2 sampling requires --full-944"
    );
    let input_contract = InputContract::parse(input_contract.as_deref().unwrap_or("legacy-rgb8"))
        .expect("input contract");
    if input_contract != InputContract::LegacyRgb8 {
        assert!(
            audit_out.is_some() && allow_failures == 0,
            "native input requires --audit-jsonl and zero failures"
        );
        assert!(
            matches!(corpus.as_str(), "pairs-tsv"),
            "native input requires explicit pairs-tsv"
        );
        assert!(
            !out.exists() && !audit_out.as_ref().unwrap().exists(),
            "native extraction requires fresh outputs"
        );
        for suffix in [".manifest.json", ".producer.bin"] {
            assert!(
                !PathBuf::from(format!("{}{suffix}", out.display())).exists(),
                "native extraction sidecar already exists"
            );
        }
    }
    // The w986 request goes through `research::extract` (the plan-driven
    // owner) — `--dvifm-spec`/`--dvifm-block-stats` exist only there. The
    // request is built once and shared by every pair.
    let research_req = full_986.then(|| {
        let spec = dvifm_spec.as_deref().map(|p| dvifm_spec_load(Path::new(p)));
        let spec_sha = dvifm_spec.as_deref().map(|p| sha256_hex_of(Path::new(p)));
        let mut req = zensim::research::Request::for_slots(
            zensim::feature_set_id::SlotSet::from_ranges([(0, 986)]),
            986,
        );
        if let Some(spec) = spec {
            req = req.with_dvifm_spec(spec);
        }
        if dvifm_blocks.is_some() {
            req = req.collect_dvifm_blocks(true);
        }
        (req, spec_sha)
    });
    let producer = if full_986 {
        None
    } else if full_944 {
        Some(diagnostic_producer(
            sampling.as_deref(),
            &out,
            input_contract,
        ))
    } else {
        sampling
            .as_deref()
            .map(|tag| diagnostic_producer(Some(tag), &out, input_contract))
    };
    // Per-pair block-record sink: streamed (a row's records run ~4 MB —
    // collecting them would not fit in memory). `index` keeps (row order,
    // byte offset, per-level record counts, input hashes) for the jsonl
    // written after the walk completes.
    let dvifm_sink = dvifm_blocks.as_ref().map(|p| {
        let file = std::fs::File::create(p).expect("create dvifm-block-stats file");
        let input_plane = research_req
            .as_ref()
            .and_then(|(r, _)| r.dvifm_spec())
            .map(|s| match s.input_plane {
                zensim::research::DvifmInputPlane::XybY => "xyb_y",
                zensim::research::DvifmInputPlane::YcbcrY => "ycbcr_y",
                zensim::research::DvifmInputPlane::YcbcrCb => "ycbcr_cb",
                zensim::research::DvifmInputPlane::YcbcrCr => "ycbcr_cr",
            })
            .unwrap_or("xyb_y");
        let hist = dvifm_hist.as_ref().map(|_| {
            let spec = research_req
                .as_ref()
                .and_then(|(r, _)| r.dvifm_spec())
                .expect("--dvifm-hist requires --dvifm-spec");
            DvifmHist {
                levels: spec
                    .levels
                    .iter()
                    .map(|lv| LevelHist {
                        counts: vec![0u32; HIST_BINS * HIST_BINS],
                        g: lv.g,
                        edge: lv.edge,
                        n_blocks: 0,
                        n_c_nonpos: 0,
                    })
                    .collect(),
            }
        });
        DvifmSink {
            file: std::sync::Mutex::new((std::io::BufWriter::new(file), 0)),
            index: std::sync::Mutex::new(Vec::new()),
            input_plane,
            cap: dvifm_cap,
            quant_f16: dvifm_quant_f16,
            hist: hist.map(std::sync::Mutex::new),
        }
    });
    let mut audit = audit::Config::load(
        audit_out,
        audit::CandidateInputs {
            bake: audit_bake,
            head: audit_head,
            ensemble: audit_ensemble,
            weights: audit_weights,
        },
        &out,
        allow_failures,
        sampling.as_deref(),
    )
    .expect("audit configuration");
    if audit_ssim2 {
        audit
            .as_mut()
            .expect("--audit-ssim2 requires --audit-jsonl")
            .enable_ssim2();
    }
    assert!(
        audit.is_none() || matches!(corpus.as_str(), "pairs" | "pairs-tsv"),
        "audit requires explicit pairs or pairs-tsv input"
    );

    if let Some(audit) = &audit {
        audit
            .validate_feature_width(if full_944 {
                944
            } else if full_986 {
                986
            } else {
                372
            })
            .expect("audit feature width");
    }

    let pairs: Vec<Pair> = match corpus.as_str() {
        "konjnd" => load_konjnd(&path, max_pairs),
        "konfig" => load_konfig(&path, max_pairs),
        "pairs-tsv" => load_pairs_tsv(&path, max_pairs),
        "konjnd_full" => load_konjnd_full(&path, max_pairs),
        "aic3" => load_aic3(&path, max_pairs),
        "aic4" => load_aic4(&path, max_pairs),
        "safesyn" => load_safesyn(&path, max_pairs),
        "qsweep" => load_qsweep_tsv(&path, max_pairs),
        "cid22_train" => load_cid22_train_tsv(&path, max_pairs),
        // Positional 3+-column TSV (ref, dist, target, [ref_basename],
        // [name=value...]). The header-driven variant is `pairs-tsv`.
        "pairs" => load_pairs_tsv_positional(&path, max_pairs),
        _ => {
            eprintln!(
                "--corpus must be one of: konjnd, konjnd_full, aic3, aic4, safesyn, qsweep, cid22_train, pairs (got {corpus:?})"
            );
            std::process::exit(2);
        }
    };

    let n_total = pairs.len();
    if audit.is_some() {
        audit::validate_pairs_input(&path, &pairs, max_pairs).expect("audit pair coverage");
    }
    eprintln!("Loaded {n_total} pairs from {corpus}");
    if n_total == 0 {
        eprintln!("no pairs loaded; exiting");
        std::process::exit(3);
    }

    let research_fsid = std::sync::Mutex::new(None::<String>);
    let started = std::time::Instant::now();
    let progress = AtomicUsize::new(0);
    let log_every = (n_total / 20).max(1);

    let scored: Vec<_> = pairs
        .par_iter()
        .enumerate()
        .map(|(row_index, kp)| {
            let p = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if p.is_multiple_of(log_every) {
                let elapsed = started.elapsed().as_secs_f64();
                let rate = p as f64 / elapsed;
                let eta = (n_total - p) as f64 / rate;
                eprintln!("  {corpus} {p}/{n_total} ({rate:.1}/s, ETA {eta:.0}s)");
            }
            let hashes = audit.as_ref().map(|_| audit::file_hashes(kp)).transpose()?;
            let (row, research_out) = extract_features(
                kp,
                producer.as_deref(),
                input_contract,
                research_req.as_ref().map(|(r, _)| r),
            )?;
            if let Some(out) = research_out {
                if let (Some(sink), Some(stats)) = (dvifm_sink.as_ref(), out.blocks) {
                    sink.write(row_index, kp, &stats)?;
                }
                if let Some(id) = out.feature_set_id {
                    let mut slot = research_fsid.lock().unwrap();
                    if slot.is_none() {
                        *slot = Some(id);
                    }
                }
            }
            let record = audit
                .as_ref()
                .map(|a| a.score(kp, &row.3, hashes.as_ref().unwrap(), input_contract))
                .transpose()?;
            Ok::<_, String>((row, record))
        })
        .collect();

    // NO GRACEFUL SKIPS. A pair that cannot be decoded or scored is a hard
    // failure by default; tolerating any is a decision the CALLER makes
    // explicitly with `--allow-failures N`, so it is visible in the invocation
    // chain instead of buried in the loop body. (This function used to
    // `.flatten()` an `Option` — a corpus could lose 30 % of its rows and the
    // only trace was a smaller row count in the "scored N/M" line.)
    let mut rows: Vec<FeatureRow> = Vec::new();
    let mut failures: Vec<String> = Vec::new();
    let mut audits = Vec::new();
    for r in scored {
        match r {
            Ok((row, record)) => {
                rows.push(row);
                if let Some(record) = record {
                    audits.push(record);
                }
            }
            Err(e) => failures.push(e),
        }
    }
    eprintln!(
        "scored {}/{} pairs in {:.1}s ({} failed)",
        rows.len(),
        n_total,
        started.elapsed().as_secs_f64(),
        failures.len()
    );
    if !failures.is_empty() {
        eprintln!(
            "{} of {n_total} pairs FAILED to extract (--allow-failures {allow_failures}):",
            failures.len()
        );
        for e in failures.iter().take(20) {
            eprintln!("  {e}");
        }
        if failures.len() > 20 {
            eprintln!("  … and {} more", failures.len() - 20);
        }
        if failures.len() > allow_failures {
            eprintln!(
                "ABORT: {} failures exceeds --allow-failures {allow_failures}. \
                 Fix the corpus or raise the budget deliberately; a partial \
                 extraction written silently is a data-integrity bug.",
                failures.len()
            );
            std::process::exit(4);
        }
    }

    let n_feat = rows.first().map(|r| r.3.len()).unwrap_or(0);
    let expected_width = if full_944 {
        944
    } else if full_986 {
        986
    } else {
        372
    };
    assert_eq!(n_feat, expected_width, "producer feature width");
    assert!(
        rows.iter()
            .all(|r| r.3.len() == expected_width && r.3.iter().all(|v| v.is_finite())),
        "invalid producer row"
    );

    // Header layout: ref_basename, human_score, <extra-target columns…>, f0..f<n-1>.
    // Extra target column names come from the first row's `extra_targets`; every row
    // is asserted to carry the same set in the same order (loader contract).
    let extra_names: Vec<String> = rows
        .first()
        .map(|r| r.2.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();

    // Write CSV
    use std::io::{BufWriter, Write};
    if let Some(parent) = out.parent()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent).expect("create output dir");
    }
    let f = std::fs::File::create(&out).expect("create output CSV");
    let mut w = BufWriter::with_capacity(1 << 20, f);
    write!(w, "ref_basename,human_score").unwrap();
    for name in &extra_names {
        write!(w, ",{name}").unwrap();
    }
    for i in 0..n_feat {
        write!(w, ",f{i}").unwrap();
    }
    writeln!(w).unwrap();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    for (ref_name, human, extras, feats) in &rows {
        write!(w, "{ref_name},{human}").unwrap();
        for (_, v) in extras {
            write!(w, ",{v}").unwrap();
        }
        for v in feats {
            write!(w, ",{v}").unwrap();
        }
        writeln!(w).unwrap();
    }
    w.flush().unwrap();
    if let Some(audit) = &audit {
        audit.write(&audits, n_total).expect("write complete audit");
    }
    eprintln!(
        "Wrote {} rows × {n_feat} features to {}",
        rows.len(),
        out.display()
    );
    if let Some((req, spec_sha)) = &research_req {
        // Write the block-record index (sorted by the pairs.tsv row order)
        // and the producer manifest once the walk is complete — the
        // manifest carries the feature-set id the extraction produced plus
        // the cache bytes' sha256.
        if let Some(sink) = dvifm_sink.as_ref() {
            let index_path = format!("{}.index.jsonl", dvifm_blocks.as_ref().unwrap());
            let mut index = std::mem::take(&mut *sink.index.lock().unwrap());
            index.sort_by_key(|e| e["row_index"].as_u64().unwrap());
            std::io::Write::flush(&mut sink.file.lock().unwrap().0).unwrap();
            let mut iw = std::io::BufWriter::new(
                std::fs::File::create(&index_path).expect("create block index"),
            );
            for e in &index {
                use std::io::Write as _;
                writeln!(iw, "{}", serde_json::to_string(e).unwrap()).unwrap();
            }
            std::io::Write::flush(&mut iw).unwrap();
            if let (Some(hist), Some(hp)) =
                (&sink.hist, dvifm_hist.as_ref())
            {
                let h = hist.lock().unwrap();
                // Layout: one JSON header line, then 5×256×256 u32 LE counts.
                let mut hw = std::io::BufWriter::new(
                    std::fs::File::create(hp).expect("create dvifm hist"),
                );
                let header = serde_json::json!({
                    "schema": "dvifm-hist-v1",
                    "input_plane": sink.input_plane,
                    "bins": HIST_BINS,
                    "ln_lo": HIST_LN_LO,
                    "ln_hi": HIST_LN_HI,
                    "bin0_semantics": "v<=0 or NaN (C: v=1 arm; m: term=0)",
                    "levels": h.levels.iter().map(|l| serde_json::json!({
                        "g": l.g, "edge": l.edge,
                        "n_blocks": l.n_blocks,
                        "n_c_nonpos": l.n_c_nonpos,
                    })).collect::<Vec<_>>(),
                    "spec": dvifm_spec,
                    "cap": sink.cap,
                    "quant": if sink.quant_f16 { "f16" } else { "f32" },
                });
                use std::io::Write as _;
                writeln!(hw, "{}", serde_json::to_string(&header).unwrap())
                    .unwrap();
                for l in &h.levels {
                    for &c in &l.counts {
                        hw.write_all(&c.to_le_bytes()).unwrap();
                    }
                }
                std::io::Write::flush(&mut hw).unwrap();
            }
        }
        write_research_manifest(
            &out,
            input_contract,
            req,
            research_fsid.lock().unwrap().as_deref(),
            spec_sha.as_deref(),
            dvifm_blocks.as_deref(),
        );
    }
}

/// The research path's per-pair side products (`--full-986`): the DVIFM
/// block records when `--dvifm-block-stats` was given, plus the
/// feature-set id string the extraction derived (same for every row —
/// captured once for the manifest).
struct ResearchOut {
    blocks: Option<zensim::research::DvifmBlockStats>,
    feature_set_id: Option<String>,
}

/// Extract one pair's canonical features: default 372, or the explicit producer.
///
/// Returns `Err` — never a silent skip — on every failure path. Decoding goes
/// through [`zen_decode`], the imazen-only decode owner: magic-byte format
/// detection plus zenjpeg / zenpng / zenwebp / zenavif / zenjxl. Before
/// 2026-09-04 this function called `image::open(..).ok()?`, which (a) has no
/// AVIF or JXL decoder in its default features, so 30.8 % of the safesyn
/// corpus was dropped without a word, and (b) decodes an XYB JPEG as an
/// ordinary JPEG, producing wrong pixels that still parse. See the module doc
/// of `shared/zen_decode.rs`.
fn extract_features(
    kp: &Pair,
    producer: Option<&[u8]>,
    contract: InputContract,
    research: Option<&zensim::research::Request>,
) -> Result<(FeatureRow, Option<ResearchOut>), String> {
    let src = ScoreInput::decode(&kp.reference, contract).map_err(|e| format!("reference: {e}"))?;
    let dst = ScoreInput::decode(&kp.distorted, contract).map_err(|e| format!("distorted: {e}"))?;
    if src.width != dst.width || src.height != dst.height {
        return Err(format!(
            "dimension mismatch: reference {}x{} ({}) vs distorted {}x{} ({})",
            src.width,
            src.height,
            kp.reference.display(),
            dst.width,
            dst.height,
            kp.distorted.display()
        ));
    }
    let w_us = src.width as usize;
    let h_us = src.height as usize;
    if w_us < 8 || h_us < 8 {
        return Err(format!(
            "image too small for the 4-scale pyramid: {w_us}x{h_us} ({})",
            kp.reference.display()
        ));
    }
    if let Some(req) = research {
        // The plan-driven owner: emits the w986 identity layout through the
        // SAME walk the producer path uses, and is the only surface that
        // carries the DVIFM constants override / block-record side output.
        let ext = zensim::research::extract(req, &src.source(), &dst.source())
            .map_err(|e| format!("research extract ({}): {e:?}", kp.distorted.display()))?;
        let out = ResearchOut {
            blocks: ext.dvifm_blocks().cloned(),
            feature_set_id: ext.feature_set_id().map(|id| id.to_string()),
        };
        return Ok((
            (
                kp.ref_basename.clone(),
                kp.human_score,
                kp.extra_targets.clone(),
                ext.values().to_vec(),
            ),
            Some(out),
        ));
    }
    if let Some(bytes) = producer {
        let model = zenpredict::Model::from_bytes(bytes).map_err(|e| e.to_string())?;
        let mut scorer = zensim::BakeScorer::new(&model).map_err(|e| e.to_string())?;
        let result = scorer
            .compute(&src.source(), &dst.source(), None)
            .map_err(|e| e.to_string())?;
        return Ok((
            (
                kp.ref_basename.clone(),
                kp.human_score,
                kp.extra_targets.clone(),
                result.features().to_vec(),
            ),
            None,
        ));
    }
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let result = if contract == InputContract::LegacyRgb8 {
        compute_zensim_with_config(
            src.bytes().as_chunks::<3>().0,
            dst.bytes().as_chunks::<3>().0,
            w_us,
            h_us,
            config,
        )
    } else {
        static PARAMS: std::sync::LazyLock<zensim::profile::ProfileParams> =
            std::sync::LazyLock::new(|| {
                zensim::profile::ProfileParams::builder()
                    .extended_features(true)
                    .compute_iw_features(true)
                    .build()
            });
        zensim::Zensim::new(zensim::ZensimProfile::Custom {
            params: &PARAMS,
            name: "native-full372-extractor",
        })
        .compute_all_features(&src.source(), &dst.source())
    }
    .map_err(|e| format!("compute_zensim ({}): {e:?}", kp.distorted.display()))?;
    let features: Vec<f64> = result.features().to_vec();
    Ok((
        (
            kp.ref_basename.clone(),
            kp.human_score,
            kp.extra_targets.clone(),
            features,
        ),
        None,
    ))
}

// ---------------------------------------------------------------------------
// `--dvifm-block-stats`: the training-only block-record side output.
// ---------------------------------------------------------------------------

/// `(C̃, m)` histogram axes. Both axes share one log domain: bin 0 is the
/// nonpositive/NaN arm (visibility v = 1 for C̃ ≤ 0; the m^P factor is 0 for
/// m = 0, so both arms carry exact semantics); bins 1..=255 cover
/// `exp(HIST_LN_LO)..exp(HIST_LN_HI)` uniformly in ln. `HIST_LN_LO = ln(1e-7)`,
/// `HIST_LN_HI = ln(16)` — block extrema/m differences live far inside this.
const HIST_BINS: usize = 256;
const HIST_LN_LO: f64 = -16.11809565095832;
const HIST_LN_HI: f64 = 2.772588722239781;

fn hist_bin(v: f64) -> usize {
    if !(v > 0.0) {
        return 0;
    }
    let t = (v.ln() - HIST_LN_LO) / (HIST_LN_HI - HIST_LN_LO);
    let k = 1 + (t * (HIST_BINS - 1) as f64).floor() as i64;
    k.clamp(1, (HIST_BINS - 1) as i64) as usize
}

/// One level's `(C̃, m)` accumulator plus the constants the histogram was
/// built at — C̃ depends on (g, edge), so they ride in the header.
struct LevelHist {
    counts: Vec<u32>, // HIST_BINS² row-major: [c_bin][m_bin]
    g: f64,
    edge: bool,
    n_blocks: u64,
    n_c_nonpos: u64,
}

/// Per-(plane,level) pooled histograms — the exact sufficient statistic for
/// any grid loss that is a sum of per-block terms `F(C̃_b, m_b)` evaluated at
/// bin centres.
struct DvifmHist {
    levels: Vec<LevelHist>,
}

/// Per-block C̃ for one side, replicating `dvifm::contrast_g_rec`: signed
/// power `phi_g(x) = sign(x)·|x|^g`; `edge` = min over the 4 corner
/// quadrants, else whole-block range.
fn dvifm_contrast(rec: &[f32], side: usize, g: f64, edge: bool) -> f64 {
    let (mx, mn) = if side == 0 {
        (&rec[2..6], &rec[6..10])
    } else {
        (&rec[10..14], &rec[14..18])
    };
    let phi = |x: f32| {
        let x = x as f64;
        x.signum() * x.abs().powf(g)
    };
    if edge {
        let mut c = f64::INFINITY;
        for q in 0..4 {
            c = c.min(phi(mx[q]) - phi(mn[q]));
        }
        c
    } else {
        let mut a = f64::NEG_INFINITY;
        let mut b = f64::INFINITY;
        for q in 0..4 {
            a = a.max(mx[q] as f64);
            b = b.min(mn[q] as f64);
        }
        a.signum() * a.abs().powf(g) - b.signum() * b.abs().powf(g)
    }
}

/// f32 → IEEE-754 binary16, round-to-nearest-even (the cache's quantisation).
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        return sign | if man == 0 { 0x7c00 } else { 0x7e00 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (1 - e + 13) as u32;
        let half = 1u32 << (shift - 1);
        let rounded = (m + half - 1 + ((m >> shift) & 1)) >> shift;
        return sign | rounded as u16;
    }
    let m16 = man + 0x0fff + ((man >> 13) & 1);
    if m16 & 0x0080_0000 != 0 {
        let e2 = e + 1;
        if e2 >= 31 {
            return sign | 0x7c00;
        }
        return sign | ((e2 as u16) << 10);
    }
    sign | ((e as u16) << 10) | ((m16 >> 13) as u16 & 0x3ff)
}

/// Streams each pair's records into the `.f32` blob from inside the
/// parallel walk — records never accumulate in memory. `index` keeps one
/// jsonl-ready entry per pair (row order, byte offset, per-level counts,
/// input hashes); `offset` is the running byte position.
///
/// `cap` bounds the kept records per row across the five levels
/// (deterministic stride per level, proportional allocation; 0 = keep all).
/// `quant_f16` stores each field as IEEE f16 (40 B/record at the v2
/// width, 36 B for v1 readers) instead of f32.
/// `hist`, when set, accumulates full-resolution `(C̃, m)` histograms over
/// ALL records — the cap never applies to the histograms.
struct DvifmSink {
    file: std::sync::Mutex<(std::io::BufWriter<std::fs::File>, u64)>,
    index: std::sync::Mutex<Vec<serde_json::Value>>,
    /// The spec's input-plane name, stamped into every index entry so a
    /// cache row is self-describing (Y′CbCr blocks are not XYB-Y blocks).
    input_plane: &'static str,
    cap: usize,
    quant_f16: bool,
    hist: Option<std::sync::Mutex<DvifmHist>>,
}

impl DvifmSink {
    fn write(
        &self,
        row_index: usize,
        kp: &Pair,
        stats: &zensim::research::DvifmBlockStats,
    ) -> Result<(), String> {
        use std::io::Write as _;
        let (ref_sha256, dist_sha256) = audit::file_hashes(kp)?;
        // Record width is derived, not assumed: `records[l]` is
        // `nby*nbx*REC_W` f32 (v2 = 20 with Weber means; v1 = 18).
        let recw = |l: usize| {
            let nb = (stats.grid[l].0 * stats.grid[l].1) as usize;
            if nb == 0 { 20 } else { stats.records[l].len() / nb }
        };
        // Histogram pass over the FULL record set (before any cap) — the
        // pooled (C̃, m) census is the exact statistic the C₀×β grid reads.
        if let Some(hist) = &self.hist {
            let mut h = hist.lock().unwrap();
            for (l, recs) in stats.records.iter().enumerate() {
                let lh = &mut h.levels[l];
                for rec in recs.chunks_exact(recw(l)) {
                    let cs = dvifm_contrast(rec, 0, lh.g, lh.edge);
                    let cd = dvifm_contrast(rec, 1, lh.g, lh.edge);
                    let ctilde = cs.min(cd);
                    let m = rec[0] as f64;
                    lh.counts[hist_bin(ctilde) * HIST_BINS + hist_bin(m)] += 1;
                    lh.n_blocks += 1;
                    if !(ctilde > 0.0) {
                        lh.n_c_nonpos += 1;
                    }
                }
            }
        }
        // Per-level stride caps: level l keeps ceil(n_l/stride_l) records
        // with stride_l = ceil(n_l/cap_l), cap_l ∝ n_l of the row total.
        let n_l: Vec<usize> = stats
            .records
            .iter()
            .enumerate()
            .map(|(l, r)| r.len() / recw(l))
            .collect();
        let n_total: usize = n_l.iter().sum();
        let caps: Vec<usize> = if self.cap > 0 && n_total > self.cap {
            n_l.iter()
                .map(|&n| {
                    if n == 0 {
                        0
                    } else {
                        (self.cap * n / n_total).max(1).min(n)
                    }
                })
                .collect()
        } else {
            n_l.clone()
        };
        let mut bytes = Vec::new();
        let mut level_records = [0u64; 5];
        let mut level_strides = [0u64; 5];
        for (l, recs) in stats.records.iter().enumerate() {
            let stride = if caps[l] >= n_l[l] || n_l[l] == 0 {
                1
            } else {
                n_l[l].div_ceil(caps[l])
            };
            level_strides[l] = stride as u64;
            let mut kept = 0u64;
            for (i, rec) in recs.chunks_exact(recw(l)).enumerate() {
                if stride > 1 && i % stride != 0 {
                    continue;
                }
                kept += 1;
                if self.quant_f16 {
                    for &v in rec {
                        bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
                    }
                } else {
                    for &v in rec {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            level_records[l] = kept;
            debug_assert!(kept as usize <= n_l[l]);
        }
        let mut guard = self.file.lock().unwrap();
        let (file, offset) = &mut *guard;
        let entry_offset = *offset;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        *offset += bytes.len() as u64;
        drop(guard);
        self.index.lock().unwrap().push(serde_json::json!({
            "row_index": row_index,
            "ref_basename": kp.ref_basename,
            "ref_path": kp.reference.display().to_string(),
            "dist_path": kp.distorted.display().to_string(),
            "ref_sha256": ref_sha256,
            "dist_sha256": dist_sha256,
            "grid": stats.grid,
            "offset": entry_offset,
            "level_records": level_records,
            "level_records_full": n_l,
            "level_strides": level_strides,
            "quant": if self.quant_f16 { "f16" } else { "f32" },
            "cap": self.cap,
            "input_plane": self.input_plane,
        }));
        Ok(())
    }
}

/// Parse a `--dvifm-spec` JSON file into a `research::DvifmSpec`.
///
/// Schema `dvifm-spec-v1`: `{"levels": [{"g","p","c0","beta","sharp","c_hi"
/// (or null = ∞),"f2_centers":[5],"band":"laplacian"|"local","edge"}×5]}`.
fn dvifm_spec_load(path: &Path) -> zensim::research::DvifmSpec {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("dvifm spec {}: {e}", path.display()));
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("dvifm spec JSON: {e}"));
    let num = |v: &serde_json::Value, k: &str| {
        v.get(k)
            .and_then(|x| x.as_f64())
            .unwrap_or_else(|| panic!("dvifm spec level missing `{k}`"))
    };
    let levels = v
        .get("levels")
        .and_then(|l| l.as_array())
        .expect("dvifm spec needs `levels`")
        .iter()
        .map(|lv| zensim::research::DvifmLevelSpec {
            g: num(lv, "g"),
            p: num(lv, "p"),
            c0: num(lv, "c0"),
            beta: num(lv, "beta"),
            sharp: num(lv, "sharp"),
            c_hi: lv
                .get("c_hi")
                .and_then(|x| x.as_f64())
                .unwrap_or(f64::INFINITY),
            f2_centers: {
                let c: Vec<f64> = lv
                    .get("f2_centers")
                    .and_then(|x| x.as_array())
                    .expect("f2_centers")
                    .iter()
                    .map(|x| x.as_f64().expect("f2_centers entry"))
                    .collect();
                c.try_into().expect("f2_centers needs 5 entries")
            },
            band: match lv.get("band").and_then(|b| b.as_str()) {
                Some("local") => zensim::research::DvifmBand::Local,
                _ => zensim::research::DvifmBand::Laplacian,
            },
            edge: lv.get("edge").and_then(|x| x.as_bool()).unwrap_or(true),
        })
        .collect();
    let input_plane = match v.get("input_plane").and_then(|x| x.as_str()) {
        None | Some("xyb_y") => zensim::research::DvifmInputPlane::XybY,
        Some("ycbcr_y") => zensim::research::DvifmInputPlane::YcbcrY,
        Some("ycbcr_cb") => zensim::research::DvifmInputPlane::YcbcrCb,
        Some("ycbcr_cr") => zensim::research::DvifmInputPlane::YcbcrCr,
        Some(other) => panic!("dvifm spec input_plane {other:?} unknown"),
    };
    zensim::research::DvifmSpec {
        levels,
        input_plane,
    }
}

fn sha256_hex_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).expect("hash dvifm spec");
    Sha256::digest(&bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}

/// The `.manifest.json` for the `--full-986` research path — same role as
/// the diagnostic producer's manifest: records the producer surface, the
/// feature-set identity, the formula revision, and (when present) the
/// DVIFM spec identity + block-record cache.
fn write_research_manifest(
    out: &Path,
    contract: InputContract,
    req: &zensim::research::Request,
    feature_set_id: Option<&str>,
    spec_sha256: Option<&str>,
    block_stats: Option<&str>,
) {
    let revision = std::env::var("ZENSIM_FORMULA_REV")
        .expect("diagnostic extraction requires explicit ZENSIM_FORMULA_REV");
    let emit = req
        .validate()
        .expect("w986 request must plan")
        .iter_slots()
        .collect::<Vec<usize>>();
    let mut manifest = serde_json::json!({
        "formula_revision": revision,
        "producer_surface": "zensim::research::extract",
        "layout": format!("w{}", req.layout_width()),
        "populated_feature_ids": emit,
        "feature_set_id": feature_set_id,
        "dvifm_spec_sha256": spec_sha256,
        "dvifm_block_stats": block_stats,
        "dvifm_block_record": {
            "schema": "dvifm-block-records-v2",
            "record_f32": 20,
            "fields": ["m","peak","cmax_s[4]","cmin_s[4]","cmax_d[4]","cmin_d[4]","mean_s","mean_d"],
            "order": "level-major, block-row-major over the full-block grid",
        },
    });
    if contract == InputContract::SdrNativeClipV1 {
        manifest["input_contract"] = serde_json::json!("sdr-native-clip-v1");
        manifest["input_era"] = serde_json::json!("native_sdr_clip_v1");
    }
    std::fs::write(
        format!("{}.manifest.json", out.display()),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
}

/// Generic (ref, dist) pairs from a TSV with header columns `ref_path`,
/// `dist_path`, optional `human_score` (default 0), optional extra numeric
/// columns emitted as extra targets. The universal escape hatch — any small
/// eval set becomes a feature parquet without a bespoke loader.
fn load_pairs_tsv(tsv: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::ReaderBuilder::new().delimiter(b'\t').from_path(tsv) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", tsv.display());
            return Vec::new();
        }
    };
    let headers = rdr.headers().expect("pairs tsv headers").clone();
    let idx = |n: &str| headers.iter().position(|h| h == n);
    let (ri, di) = (
        idx("ref_path").expect("ref_path col"),
        idx("dist_path").expect("dist_path col"),
    );
    let hi = idx("human_score");
    let extras: Vec<(usize, String)> = headers
        .iter()
        .enumerate()
        .filter(|(i, h)| *i != ri && *i != di && Some(*i) != hi && !h.is_empty())
        .filter(|(_, h)| *h != "img_num" || true)
        .map(|(i, h)| (i, h.to_string()))
        .collect();
    let mut pairs = Vec::new();
    for rec in rdr.records().flatten() {
        let reference = PathBuf::from(rec.get(ri).unwrap_or(""));
        let distorted = PathBuf::from(rec.get(di).unwrap_or(""));
        let human_score = hi
            .and_then(|i| rec.get(i))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        let extra_targets: Vec<(String, f64)> = extras
            .iter()
            .filter_map(|(i, n)| {
                rec.get(*i)
                    .and_then(|v| v.parse().ok())
                    .map(|x| (n.clone(), x))
            })
            .collect();
        let ref_basename = reference
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        pairs.push(Pair {
            reference,
            distorted,
            human_score,
            ref_basename,
            extra_targets,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// KonFiG-IQA (Men/Lin/Jenadeleh/Saupe 2021): 10 sources x 7 distortions x
/// 12 levels @ 0.25 JND (Part A) + motion blur x 30 levels @ 0.1 JND
/// (Part B). Levels are calibrated to uniform JND spacing BY DESIGN, so the
/// per-stimulus target comes from the level index directly — no Thurstonian
/// reconstruction needed for ingestion. Emits human_score = 1 - q/3.2 (a
/// [0,1] quality scale, rank-identical to the JND grid) + a native `q_jnd`
/// extra target column. Level 0 = pristine copy → identity anchor rows.
/// `path` = the KonFiG-IQA root (contains IMAGES/).
fn load_konfig(base: &Path, max: usize) -> Vec<Pair> {
    let images = base.join("IMAGES");
    let mut pairs = Vec::new();
    for (part, step) in [("PartA", 0.25_f64), ("PartB", 0.1_f64)] {
        let part_dir = images.join(part);
        let Ok(srcs) = std::fs::read_dir(&part_dir) else {
            continue;
        };
        for src in srcs.flatten() {
            let src_name = src.file_name().to_string_lossy().to_string();
            let reference = images
                .join("reference_images")
                .join(format!("{src_name}_0.png"));
            if !reference.is_file() {
                eprintln!("konfig: missing reference for {src_name}");
                continue;
            }
            let Ok(dists) = std::fs::read_dir(src.path()) else {
                continue;
            };
            for dist in dists.flatten() {
                let dist_name = dist.file_name().to_string_lossy().to_string();
                let Ok(files) = std::fs::read_dir(dist.path()) else {
                    continue;
                };
                for f in files.flatten() {
                    let fname = f.file_name().to_string_lossy().to_string();
                    let Some(level) = fname
                        .trim_end_matches(".png")
                        .rsplit('_')
                        .next()
                        .and_then(|t| t.parse::<u32>().ok())
                    else {
                        continue;
                    };
                    let q_jnd = step * level as f64;
                    pairs.push(Pair {
                        reference: reference.clone(),
                        distorted: f.path(),
                        human_score: 1.0 - q_jnd / 3.2,
                        ref_basename: format!("{src_name}_{dist_name}_{part}"),
                        extra_targets: vec![("q_jnd".into(), q_jnd)],
                    });
                    if pairs.len() >= max {
                        return pairs;
                    }
                }
            }
        }
    }
    pairs
}

fn load_konjnd(base: &Path, max: usize) -> Vec<Pair> {
    let csv_path = base.join("subjective_ratings.csv");
    let mut rdr = match csv::Reader::from_path(&csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let image_id = record.get(0).unwrap_or("");
        let comp = record.get(1).unwrap_or("");
        let mean_threshold: f64 = match record.get(3).unwrap_or("").parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let stem = image_id.trim_end_matches(".png");
        if stem.is_empty() {
            continue;
        }
        let level = mean_threshold.round().clamp(1.0, 100.0) as u32;
        let (subdir, ext) = match comp {
            "JPEG" => ("jpeg", "jpg"),
            "BPG" => ("bpg", "png"),
            _ => continue,
        };
        let dist_name = format!("{stem}_{comp}_{level:03}.{ext}");
        let ref_path = base.join("source_image").join(image_id);
        let dist_path = base.join(subdir).join(&dist_name);
        if !dist_path.exists() {
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: mean_threshold,
            ref_basename: image_id.to_string(),
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Full KonJND-1k loader: reads `konjnd_full_scored.csv` which lists
/// every (source × codec × quality) variant (~76k pairs) along with the
/// metric scores. We emit one row per pair, using `gpu_ssimulacra2 / 100`
/// as the `human_score` anchor — matching the convention used in the
/// existing `/mnt/v/zen/zensim-training/2026-05-14-clean/konjnd_aligned_features.csv`.
/// The score is NOT a real human MOS for these pairs; the canonical
/// 1008-source human-PJND anchors live in `subjective_ratings.csv`
/// (loaded by the `konjnd` corpus type).
fn load_konjnd_full(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let src_path = record.get(0).unwrap_or("");
        let dist_path = record.get(1).unwrap_or("");
        // gpu_ssimulacra2 / 100 → 0..1 score anchor
        let score_norm: f64 = match record.get(4).and_then(|s| s.parse::<f64>().ok()) {
            Some(v) => v / 100.0,
            None => continue,
        };
        if src_path.is_empty() || dist_path.is_empty() {
            continue;
        }
        let ref_pb = PathBuf::from(src_path);
        let basename = ref_pb
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        pairs.push(Pair {
            reference: ref_pb,
            distorted: PathBuf::from(dist_path),
            human_score: score_norm,
            ref_basename: basename,
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// AIC-3 CTC dataset loader. The info.csv columns are:
///   score.jnd, codec, img.number, img.name, quality, quality.selected, method
/// Reference image lives at `<dataset_root>/original/<img.name>.png` and
/// each distorted file at `<dataset_root>/decoded/<img.name>/<codec>_<img.name>_<quality>.png`.
/// `--path` should be the info.csv path; the dataset root is its grandparent.
fn load_aic3(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    // info.csv lives at <root>/decoded/info.csv → grandparent of csv = <root>
    let root: PathBuf = csv_path
        .parent()
        .and_then(|d| d.parent())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let original_dir = root.join("original");
    let decoded_dir = root.join("decoded");
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 5 {
            continue;
        }
        let score_jnd: f64 = match record.get(0).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let codec = record.get(1).unwrap_or("");
        let img_name = record.get(3).unwrap_or("");
        let quality = record.get(4).unwrap_or("");
        if codec.is_empty() || img_name.is_empty() || quality.is_empty() {
            continue;
        }
        let ref_path = original_dir.join(format!("{img_name}.png"));
        let dist_name = format!("{codec}_{img_name}_{quality}.png");
        let dist_path = decoded_dir.join(img_name).join(&dist_name);
        if !ref_path.exists() || !dist_path.exists() {
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: score_jnd,
            ref_basename: format!("{img_name}.png"),
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// AIC-4 sample dataset loader.
///
/// The AIC-4 dataset (Final Call for Proposals on Objective Quality
/// Assessment, JPEG WG1, 2025) provides reconstructed JND scores for
/// 5 source images × 6 codecs (AVIF, JPEG-1, JPEG-2000, JPEG-AI,
/// JPEG-XL, VVC) × 10 distortion levels = 300 distorted pairs.
///
/// The CSV is at `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv`:
///
/// ```text
/// img_num,codec,dlevel,img_source,img_distorted,distortion,CI_min,CI_max
/// 2,1,1,PTC_00002_0ref_00.png,PTC_00002_AVIF_01.png,0.12031473,0.09630131,0.14599248
/// ```
///
/// `img_source` and `img_distorted` point at the **PTC (cropped)** image
/// set used in the actual subjective study, NOT the full-resolution
/// images. The PTC images are 620×800 RGB 8-bit PNGs that live under
/// `<aic4_root>/PTC_images/<NNNNN>/<filename>` (zero-padded source id).
///
/// `--path` should point at the CSV. The image root is derived as
/// `<csv-parent>/../../dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset` —
/// but for portability we accept either:
///   1. `--path /mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv`
///      → looks for images under `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/PTC_images/`
///   2. `--path <aic4_root>/JPEG_AIC_reconstructed_jnd_scores.csv` with the CSV
///      copied into the dataset dir → looks for images under `<aic4_root>/PTC_images/`
///
/// `human_score = distortion` (signed JND units, AIC-3 CTC methodology
/// applied to a higher-fidelity codec set). Matches AIC-3's
/// `human_score = score.jnd` convention so downstream eval doesn't
/// need a per-corpus rescale.
fn load_aic4(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    // Probe for the dataset root: either alongside the CSV OR at the
    // canonical /mnt/v location.
    let aic4_root: PathBuf = {
        let canonical = PathBuf::from("/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset");
        let csv_sibling = csv_path
            .parent()
            .map(|d| d.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        if csv_sibling.join("PTC_images").is_dir() {
            csv_sibling
        } else if canonical.join("PTC_images").is_dir() {
            canonical
        } else {
            eprintln!(
                "load_aic4: could not locate PTC_images under {} or {}",
                csv_sibling.display(),
                canonical.display()
            );
            return Vec::new();
        }
    };
    let ptc_root = aic4_root.join("PTC_images");
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        if record.len() < 6 {
            continue;
        }
        let img_num_str = record.get(0).unwrap_or("");
        let img_source = record.get(3).unwrap_or("");
        let img_distorted = record.get(4).unwrap_or("");
        let distortion: f64 = match record.get(5).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let img_num: u32 = match img_num_str.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let img_dir = ptc_root.join(format!("{img_num:05}"));
        let ref_path = img_dir.join(img_source);
        let dist_path = img_dir.join(img_distorted);
        if !ref_path.exists() || !dist_path.exists() {
            continue;
        }
        pairs.push(Pair {
            reference: ref_path,
            distorted: dist_path,
            human_score: distortion,
            ref_basename: img_source.to_string(),
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Safesyn (safe-synthetic) loader with IW-SSIM target column.
///
/// Reads the enriched safesyn TSV at
/// `/mnt/v/zen/zensim-training/<date>/safesyn_with_iwssim.csv`
/// (produced by `scripts/v_next/merge_iwssim_into_safesyn.py`).
/// Schema: `source_path, decoded_path, codec, quality, width, height,
/// gpu_ssimulacra2, gpu_butteraugli, cpu_ssimulacra2, cpu_butteraugli,
/// size_bytes, run_id, dssim, iwssim`.
///
/// Emits one Pair per row with:
/// - `reference = source_path`
/// - `distorted = decoded_path`
/// - `ref_basename = source_path file stem`
/// - `human_score = cpu_ssimulacra2 / 100` (legacy ssim2 target in [0, 1])
/// - `extra_targets = [("iwssim", iwssim_value)]` — emitted in the
///   output CSV as a column between `human_score` and `f0`. The
///   trainer's `--target-column iwssim` flag (T1.1) selects this
///   column as the regression target instead of `human_score`.
///
/// This is the V_22-IW training-data input. The 196 086-pair safesyn
/// corpus is the only large-N corpus that carries an IW-SSIM target
/// (computed via `scripts/v_next/compute_iwssim_on_safesyn.py`).
fn load_safesyn(csv_path: &Path, max: usize) -> Vec<Pair> {
    let mut rdr = match csv::Reader::from_path(csv_path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("failed to open {}: {e}", csv_path.display());
            return Vec::new();
        }
    };
    // Header positions are stable (this file is generated, not
    // hand-edited), but look them up by name to be robust to future
    // column reorderings.
    let header = rdr
        .headers()
        .map(|h| h.iter().map(String::from).collect::<Vec<_>>())
        .unwrap_or_default();
    let pos = |name: &str| -> Option<usize> { header.iter().position(|c| c == name) };
    let src_col = match pos("source_path") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing source_path column", csv_path.display());
            return Vec::new();
        }
    };
    let dst_col = match pos("decoded_path") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing decoded_path column", csv_path.display());
            return Vec::new();
        }
    };
    let cpu_ssim2_col = match pos("cpu_ssimulacra2") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing cpu_ssimulacra2 column", csv_path.display());
            return Vec::new();
        }
    };
    // Half the safesyn rows have empty cpu_ssimulacra2 (zenavif / zenjxl
    // codec families were scored GPU-only). Both metrics are in
    // score_zensim units, so use cpu when present, fall back to gpu.
    let gpu_ssim2_col = match pos("gpu_ssimulacra2") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing gpu_ssimulacra2 column", csv_path.display());
            return Vec::new();
        }
    };
    let iwssim_col = match pos("iwssim") {
        Some(i) => i,
        None => {
            eprintln!("{}: missing iwssim column", csv_path.display());
            return Vec::new();
        }
    };
    let mut pairs = Vec::new();
    for record in rdr.records().flatten() {
        let src_path = record.get(src_col).unwrap_or("");
        let dst_path = record.get(dst_col).unwrap_or("");
        if src_path.is_empty() || dst_path.is_empty() {
            continue;
        }
        let ssim2_raw: f64 = match record
            .get(cpu_ssim2_col)
            .and_then(|s| s.parse::<f64>().ok())
            .or_else(|| {
                record
                    .get(gpu_ssim2_col)
                    .and_then(|s| s.parse::<f64>().ok())
            }) {
            Some(v) => v,
            None => continue,
        };
        let iwssim: f64 = match record.get(iwssim_col).and_then(|s| s.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let ref_pb = PathBuf::from(src_path);
        let basename = ref_pb
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();
        // Legacy convention: `human_score` is the ssim2 target divided
        // by 100 so it lands in [0, 1]. The trainer multiplies by 100
        // (default --target-scale) to recover score_zensim units.
        pairs.push(Pair {
            reference: ref_pb,
            distorted: PathBuf::from(dst_path),
            human_score: ssim2_raw / 100.0,
            ref_basename: basename,
            extra_targets: vec![("iwssim".to_string(), iwssim)],
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Generic q-sweep TSV loader for the `PreviewV0_5Tuner` evaluation
/// harness (2026-05-18). Reads a TSV with the columns:
///
/// ```text
///   ref_path  dist_path  image_id  codec  q
/// ```
///
/// where the first row is the header. `image_id` becomes
/// `ref_basename` (the field the trainer's downstream tooling
/// groups on); `q` is loaded into `human_score` (so the eval can
/// pivot by quality at scoring time without re-parsing); `codec`
/// becomes an extra target column. Monotonicity is measured
/// downstream by sorting per (`ref_basename`, `codec`) by `human_score`
/// (= q) and counting score(q+δ) ≤ score(q) inversions.
/// V11 CID22 training-only-subset loader. Reads the workspace TSV produced
/// by `scripts/canonical_corpus/v11_extract_cid22_train.py --build-pairs`
/// with the schema:
///
/// ```text
///   ref_path<TAB>dist_path<TAB>ref_basename<TAB>codec<TAB>q
/// ```
///
/// The `ref_basename` field is REWRITTEN as a composite
/// `"<ref_basename>|<codec>|<q>"` so each output CSV row carries its
/// full (ref, codec, q) join key without needing a numeric-hash sidecar.
/// `human_score` is set to NaN — the Python join step replaces it with
/// the ssim2_gpu score. `extra_targets` is empty.
///
/// Output CSV schema: `composite_key, human_score(NaN), f0..f371`.
/// Downstream Python split-by-'|' to recover the join key.
fn load_cid22_train_tsv(path: &Path, max: usize) -> Vec<Pair> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            return Vec::new();
        }
    };
    let r = BufReader::new(f);
    let mut lines = r.lines();
    let _header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Vec::new(),
    };
    let mut pairs = Vec::new();
    for line in lines {
        let line = line.expect("read corpus line");
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        let ref_path = cols[0];
        let dist_path = cols[1];
        let ref_basename = cols[2];
        let codec = cols[3];
        let q = cols[4];
        // Composite join key encoded into ref_basename for downstream Python
        // to split. The '|' delimiter is safe because CID22 basenames are
        // pure digits, codec names are alphanumeric, and q strings are
        // alphanumeric/underscore.
        let composite = format!("{ref_basename}|{codec}|{q}");
        pairs.push(Pair {
            reference: PathBuf::from(ref_path),
            distorted: PathBuf::from(dist_path),
            human_score: f64::NAN,
            ref_basename: composite,
            extra_targets: Vec::new(),
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// Generic (ref, dist, target[, extra…]) pairs TSV loader. The most
/// flexible mode — used to extract 372 features for ANY corpus that
/// can be expressed as image pairs, e.g. the structural-corruption
/// corpus (codec-corpus#7) for training a corruption-detection bake.
///
/// TSV format (header REQUIRED; column names free, positions fixed):
///   col 0: ref_path      (absolute or cwd-relative path to reference PNG)
///   col 1: dist_path     (path to distorted/corruption PNG)
///   col 2: target        (f64 regression target → `human_score` column)
///   col 3: ref_basename  (OPTIONAL; defaults to ref file stem)
///   col 4+: extra_target columns, each `name=value` → emitted between
///           human_score and f0 (the trainer's --target-column can pick them)
///
/// Example header + row:
///   ref_path<TAB>dist_path<TAB>target<TAB>ref_basename<TAB>butter_max=51.4
fn load_pairs_tsv_positional(path: &Path, max: usize) -> Vec<Pair> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            return Vec::new();
        }
    };
    let r = BufReader::new(f);
    let mut lines = r.lines();
    let _header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Vec::new(),
    };
    let mut pairs = Vec::new();
    for line in lines {
        let line = line.expect("read corpus line");
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 3 {
            continue;
        }
        let reference = PathBuf::from(cols[0]);
        let distorted = PathBuf::from(cols[1]);
        let human_score: f64 = match cols[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ref_basename = if cols.len() > 3 && !cols[3].is_empty() {
            cols[3].to_string()
        } else {
            reference
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        let mut extra_targets = Vec::new();
        for extra in cols.iter().skip(4) {
            if let Some((name, val)) = extra.split_once('=')
                && let Ok(v) = val.parse::<f64>()
            {
                extra_targets.push((name.to_string(), v));
            }
        }
        pairs.push(Pair {
            reference,
            distorted,
            human_score,
            ref_basename,
            extra_targets,
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

fn load_qsweep_tsv(path: &Path, max: usize) -> Vec<Pair> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let f = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("failed to open {}: {e}", path.display());
            return Vec::new();
        }
    };
    let r = BufReader::new(f);
    let mut lines = r.lines();
    let _header = match lines.next() {
        Some(Ok(h)) => h,
        _ => return Vec::new(),
    };
    let mut pairs = Vec::new();
    for line in lines {
        let line = line.expect("read corpus line");
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 5 {
            continue;
        }
        let ref_path = cols[0];
        let dist_path = cols[1];
        let image_id = cols[2];
        let codec = cols[3];
        let q: f64 = match cols[4].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        pairs.push(Pair {
            reference: PathBuf::from(ref_path),
            distorted: PathBuf::from(dist_path),
            human_score: q,
            ref_basename: image_id.to_string(),
            extra_targets: vec![(
                "codec".to_string(),
                codec
                    .bytes()
                    .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64))
                    as f64,
            )],
        });
        if pairs.len() >= max {
            break;
        }
    }
    pairs
}

/// A diagnostic all-live read-set bake makes this producer execute exactly the
/// same public pixel API as a fitted model. It is not a quality predictor.
fn diagnostic_producer(sampling: Option<&str>, out: &Path, contract: InputContract) -> Vec<u8> {
    let wide = sampling.is_none_or(|tag| tag.starts_with("v2:"));
    let keep_y = sampling.is_some_and(|tag| tag.starts_with("v1:y:"));
    let ids: Vec<usize> = (0..if wide { 944 } else { 228 })
        .filter(|&i| !keep_y || !matches!(i,0..=12|26..=38|156..=161|168..=173))
        .collect();
    let n = ids.len();
    let revision = std::env::var("ZENSIM_FORMULA_REV")
        .expect("diagnostic extraction requires explicit ZENSIM_FORMULA_REV");
    let mut spec = serde_json::json!({"schema_hash":1,"scaler_mean":vec![0.;n],"scaler_scale":vec![1.;n],
        "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":ids.iter().map(usize::to_string).collect::<Vec<_>>().join("\n")},
        {"key":"zentrain.formula_revision","type":"utf8","text":revision}],
        "layers":[{"in_dim":n,"out_dim":1,"activation":"identity","dtype":"f32","weights":vec![-0.1;n],"biases":[100.]}]});
    if let Some(tag) = sampling {
        spec["metadata"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(
            {"key":"zentrain.sampling","type":"utf8","text":tag}));
    }
    let bytes = zenpredict_bake::bake_from_json_str(&spec.to_string()).expect("producer bake");
    let model = zenpredict::Model::from_bytes(&bytes).expect("producer model");
    zensim::BakeScorer::new(&model).expect("servable sampling contract");
    let era = sampling.map_or_else(
        || format!("ceiling_rev{revision}"),
        |tag| {
            format!(
                "sampling_{}",
                tag.replace(':', "_").replace('/', "d").replace(',', "_")
            )
        },
    );
    let hash = zensim::feature_set_id::slots_hash8(ids.iter().copied());
    let family = if !wide {
        "basic+peaks@w372"
    } else {
        "basic+peaks+masked+iw+v2+append+append2@w944"
    };
    let identity = format!("{family}/{era}#{hash:08x}");
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut manifest = serde_json::json!({"sampling":sampling,"formula_revision":revision,"feature_set_id":identity,
        "populated_feature_ids":ids,"era":era,"producer_surface":"zensim::BakeScorer::compute"});
    if contract == InputContract::SdrNativeClipV1 {
        manifest["input_contract"] = serde_json::json!("sdr-native-clip-v1");
        manifest["input_era"] = serde_json::json!("native_sdr_clip_v1");
    }
    std::fs::write(
        format!("{}.manifest.json", out.display()),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    std::fs::write(format!("{}.producer.bin", out.display()), &bytes).unwrap();
    bytes
}
