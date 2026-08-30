//! Console stats demo — Mohammadi panel + dynamic range + per-codec
//! q→score mapping across the currently-shipped V0_5 zensim profiles.
//!
//! Renders to stdout as ANSI-coloured unicode tables that paste cleanly
//! into chat / markdown.
//!
//! Run:
//! ```
//! cargo run --release --bin preview_stats_demo -p zensim-validate
//! ```
//!
//! Adds task #181 (2026-05-20) — AIC-4 added to default eval set; this
//! demo is the at-a-glance verification harness.

use std::path::{Path, PathBuf};

use arrow::array::{
    Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray, UInt32Array,
};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Deterministic xorshift64* RNG so we can sample subsets reproducibly
/// without pulling in `rand` as a workspace dependency.
struct Xs64 {
    state: u64,
}

impl Xs64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            slice.swap(i, j);
        }
    }
}

use zenpredict::Model;
use zenpredict::Predictor;

use zensim_validate::output_calibration_spline::{self, OutputCalibrationSpline};
use zensim_validate::panel;
use zensim_validate::parquet_loader;

// Bake bytes — embedded the same way the zensim crate embeds them.
// Single source of truth so the demo runs against the exact bytes
// shipped to users.
const BAKE_BALANCED_V2: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_balanced_v2_2026-05-20.bin");
const BAKE_COMPRESSION_V2: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_compression_v2_2026-05-20.bin");
const BAKE_TUNER_V2: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_tuner_v6_2026-05-19.bin");
const BAKE_TUNER_V3: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_tuner_v9_2026-05-20.bin");
// EXP-CROSS-CODEC-V10 (2026-05-20): score-space reallocation
const BAKE_BALANCED_V3: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_balanced_v3_2026-05-20.bin");
const BAKE_COMPRESSION_V3: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_compression_v3_2026-05-20.bin");
const BAKE_TUNER_V4: &[u8] =
    include_bytes!("../../../zensim-experimental/weights/v_tuner_v10_2026-05-20.bin");

// ============================================================================
// Bake-side dispatch helpers — DEDUP-M (2026-05-26): moved to
// `zensim_validate::bake_runtime`. Bit-exact, f32 ±1e-6.
// ============================================================================

use zensim_validate::bake_runtime::{
    HybridHeadDispatch, PerSampleAlphaHeadDispatch, extract_hybrid_head,
    extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_row,
};

// ============================================================================
// Profile catalogue
// ============================================================================

struct ProfileEntry {
    label: &'static str,
    bake_bytes: &'static [u8],
}

const SHIPPING_PROFILES: &[ProfileEntry] = &[
    ProfileEntry {
        label: "PreviewV0_5TunerV4       (V10 anchor + PCHIP spline + extrapolate, default)",
        bake_bytes: BAKE_TUNER_V4,
    },
    ProfileEntry {
        label: "PreviewV0_5BalancedV3    (V_22-mix LARGE + iwssim + V10 spline + extrapolate)",
        bake_bytes: BAKE_BALANCED_V3,
    },
    ProfileEntry {
        label: "PreviewV0_5CompressionV3 (V_24-per-sample-α + V10 spline + extrapolate)",
        bake_bytes: BAKE_COMPRESSION_V3,
    },
    ProfileEntry {
        label: "PreviewV0_5TunerV3       (V9 anchor + PCHIP spline, JND=60/JOD=30)",
        bake_bytes: BAKE_TUNER_V3,
    },
    ProfileEntry {
        label: "PreviewV0_5BalancedV2    (V_22-mix LARGE + iwssim + V9 spline)",
        bake_bytes: BAKE_BALANCED_V2,
    },
    ProfileEntry {
        label: "PreviewV0_5CompressionV2 (V_24-per-sample-α + V9 spline)",
        bake_bytes: BAKE_COMPRESSION_V2,
    },
    ProfileEntry {
        label: "PreviewV0_5TunerV2       (legacy V6 anchor + tanh-pin)",
        bake_bytes: BAKE_TUNER_V2,
    },
];

// ============================================================================
// Corpus configuration
// ============================================================================

struct Corpus {
    #[allow(dead_code)]
    name: &'static str,
    display: &'static str,
    parquet: &'static str,
    enable_per_band: bool,
}

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "cid22",
        display: "CID22",
        parquet: "cid22_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "kadid",
        display: "KADIK10k",
        parquet: "kadid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "tid",
        display: "TID2013",
        parquet: "tid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "konjnd",
        display: "KonJND-1k",
        parquet: "konjnd_features_372col_2026-05-15.parquet",
        enable_per_band: false,
    },
    Corpus {
        name: "aic3",
        display: "AIC-3 CTC",
        parquet: "aic3_features_372col_2026-05-15.parquet",
        enable_per_band: false,
    },
    Corpus {
        name: "aic4",
        display: "AIC-4 sample",
        parquet: "aic4_features_372col_2026-05-20.parquet",
        enable_per_band: false,
    },
];

const FEATURES_ROOT: &str = zensim_validate::eval_roots::STORED_FEATURES_ROOT_2026_05_15;

// Butter parquets — (codec, file)
struct CodecParquet {
    codec: &'static str,
    path: &'static str,
}

const CODECS: &[CodecParquet] = &[
    CodecParquet {
        codec: "zenjpeg",
        path: "/mnt/v/zen/picker-training/2026-05-19/butter/zenjpeg.parquet",
    },
    CodecParquet {
        codec: "zenwebp",
        path: "/mnt/v/zen/picker-training/2026-05-19/butter/zenwebp.parquet",
    },
    CodecParquet {
        codec: "zenavif",
        path: "/mnt/v/zen/picker-training/2026-05-19/butter/zenavif.parquet",
    },
    CodecParquet {
        codec: "zenjxl",
        path: "/mnt/v/zen/picker-training/2026-05-19/butter/zenjxl.parquet",
    },
];

const Q_SWEEP: &[i32] = &[5, 15, 25, 35, 45, 55, 65, 75, 85, 95];
const N_DEMO_IMAGES: usize = 10;
const N_DYN_RANGE_SAMPLES: usize = 1000;

// ============================================================================
// Bake loading
// ============================================================================

struct LoadedBake {
    label: &'static str,
    model: Model,
    has_transforms: bool,
    per_sample_alpha_head: Option<PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<OutputCalibrationSpline>,
    n_inputs: usize,
}

fn load_bake_for_profile(entry: &ProfileEntry) -> Result<LoadedBake, String> {
    let model = Model::from_bytes(entry.bake_bytes)
        .map_err(|e| format!("Model::from_bytes for {}: {e:?}", entry.label))?;
    let has_transforms = model.has_nontrivial_feature_transforms();
    let per_sample_alpha_head = extract_per_sample_alpha_head(&model);
    let hybrid_head = extract_hybrid_head(&model);
    let tanh_pin_scale = extract_tanh_output_head_scale(&model);
    let output_spline = output_calibration_spline::extract(&model);
    let n_inputs = model.caller_input_width();
    Ok(LoadedBake {
        label: entry.label,
        model,
        has_transforms,
        per_sample_alpha_head,
        hybrid_head,
        tanh_pin_scale,
        output_spline,
        n_inputs,
    })
}

fn score_features(bake: &LoadedBake, scratch: &mut Vec<f32>, row: &[f64]) -> f64 {
    if scratch.len() != bake.n_inputs {
        scratch.resize(bake.n_inputs, 0.0);
    }
    let mut predictor = Predictor::new(&bake.model);
    score_row(
        &mut predictor,
        bake.has_transforms,
        bake.per_sample_alpha_head.as_ref(),
        bake.hybrid_head.as_ref(),
        bake.tanh_pin_scale,
        bake.output_spline.as_ref(),
        scratch,
        row,
    )
}

// ============================================================================
// Mohammadi panel
// ============================================================================

/// Pre-loaded corpus features so we don't reload parquets per profile.
struct LoadedCorpus<'a> {
    corpus: &'a Corpus,
    humans: Vec<f64>,
    feature_rows: Vec<Vec<f64>>,
}

fn load_corpus_once<'a>(
    corpus: &'a Corpus,
    features_root: &Path,
) -> Result<LoadedCorpus<'a>, String> {
    let path = PathBuf::from(features_root).join(corpus.parquet);
    let g = parquet_loader::load_parquet(&path, corpus.display, "human_score", 1.0)
        .map_err(|e| format!("load {} parquet: {e}", corpus.display))?;
    Ok(LoadedCorpus {
        corpus,
        humans: g.human_scores,
        feature_rows: g.feature_rows,
    })
}

fn render_corpus_panel(
    bake: &LoadedBake,
    loaded: &LoadedCorpus<'_>,
    out: &mut String,
) -> Result<(), String> {
    let corpus = loaded.corpus;
    let humans = &loaded.humans;
    let mut scratch = vec![0.0f32; bake.n_inputs];
    let scores: Vec<f64> = loaded
        .feature_rows
        .iter()
        .map(|row| score_features(bake, &mut scratch, row))
        .collect();
    let n = scores.len();
    let stats = aggregate_panel(&scores, humans);

    out.push_str(&format!(
        "│  {}: n={}  SROCC={:.4}  PLCC={:.4}  KROCC={:.4}  OR={:.4}  PWRC={:.4}  Z-RMSE={:.3}\n",
        corpus.display, n, stats.0, stats.1, stats.2, stats.3, stats.4, stats.5
    ));

    if corpus.enable_per_band {
        out.push_str("│  ┌─ 10-band SROCC ───────────────────────────────────────────────\n");
        let mut band_line = String::from("│  │");
        for band_idx in 0..10 {
            let lo = band_idx as f64 * 0.10;
            let hi = lo + 0.10;
            let idxs: Vec<usize> = humans
                .iter()
                .enumerate()
                .filter_map(|(i, &h)| {
                    if band_idx == 9 {
                        (h >= lo).then_some(i)
                    } else {
                        (h >= lo && h < hi).then_some(i)
                    }
                })
                .collect();
            if idxs.len() < 4 {
                band_line.push_str(&format!(" B{band_idx}: n/a({})", idxs.len()));
            } else {
                let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
                let s_b: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
                let r = panel::spearman(&s_b, &h_b).abs();
                let mark = if idxs.len() < 30 { "⚠" } else { " " };
                band_line.push_str(&format!(" B{band_idx}:{}{:.3}(n={})", mark, r, idxs.len()));
            }
        }
        out.push_str(&band_line);
        out.push('\n');
        out.push_str("│  └────────────────────────────────────────────────────────────────\n");
    }
    Ok(())
}

fn aggregate_panel(scores: &[f64], humans: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    // Match `panel::compute_panel`: PLCC / OR / PWRC are computed on
    // the 4-param-logistic-rescaled prediction (Mohammadi 2025 § IV-A
    // convention; absorbs polarity AND saturation).
    let srocc = panel::spearman(scores, humans).abs();
    let krocc = panel::kendall_tau(scores, humans).abs();
    let rescaled = panel::rescale_logistic(scores, humans);
    let plcc = panel::pearson(&rescaled, humans).abs();
    let pw = panel::pwrc_sa_st_auc(&rescaled, humans);
    let or_ = panel::outlier_ratio(&rescaled, humans);
    let z = panel::z_rmse(&rescaled, humans);
    (srocc, plcc, krocc, or_, pw, z)
}

// ============================================================================
// Dynamic range
// ============================================================================

fn dynamic_range(
    bake: &LoadedBake,
    cid22: &LoadedCorpus<'_>,
) -> Result<(f64, f64, f64, f64, f64), String> {
    // Score 1000 random rows from the CID22 val parquet (widest score
    // distribution per the corpus). Returns min/p5/p50/p95/max.
    let mut rng = Xs64::new(0xc1d22a1c4);
    let mut indices: Vec<usize> = (0..cid22.feature_rows.len()).collect();
    rng.shuffle(&mut indices);
    indices.truncate(N_DYN_RANGE_SAMPLES);
    let mut scratch = vec![0.0f32; bake.n_inputs];
    let mut scores: Vec<f64> = indices
        .iter()
        .map(|&i| score_features(bake, &mut scratch, &cid22.feature_rows[i]))
        .filter(|v| v.is_finite())
        .collect();
    if scores.is_empty() {
        return Err("no finite scores".into());
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = |p: f64| -> f64 {
        let idx = ((scores.len() - 1) as f64 * p).round() as usize;
        scores[idx]
    };
    Ok((
        scores[0],
        q(0.05),
        q(0.50),
        q(0.95),
        *scores.last().unwrap(),
    ))
}

// ============================================================================
// Per-codec q→score table
// ============================================================================

/// Load a butter parquet, projecting to (ref_basename, codec, q, f0..f371).
fn load_butter_parquet(path: &Path) -> Result<Vec<ButterRow>, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("open {path:?}: {e}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("{path:?}: parquet open: {e}"))?;
    let schema = builder.schema().clone();
    let arrow_fields = schema.fields();
    let n_arrow_cols = arrow_fields.len();

    let mut leaf_indices = vec![];
    let mut col_kinds: Vec<&'static str> = vec![];
    // ref_basename
    let ref_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or("missing ref_basename")?;
    leaf_indices.push(ref_idx);
    col_kinds.push("ref_basename");
    let codec_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "codec")
        .ok_or("missing codec")?;
    leaf_indices.push(codec_idx);
    col_kinds.push("codec");
    let q_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "q")
        .ok_or("missing q")?;
    leaf_indices.push(q_idx);
    col_kinds.push("q");

    let f0_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "f0")
        .ok_or("missing f0")?;
    let mut n_features = 0;
    while f0_idx + n_features < n_arrow_cols {
        let want = format!("f{}", n_features);
        if arrow_fields[f0_idx + n_features].name() != &want {
            break;
        }
        leaf_indices.push(f0_idx + n_features);
        col_kinds.push("feat");
        n_features += 1;
    }

    let parquet_schema = builder.parquet_schema().clone();
    let proj = ProjectionMask::leaves(&parquet_schema, leaf_indices.iter().copied());
    let reader = builder
        .with_projection(proj)
        .build()
        .map_err(|e| format!("{path:?}: build reader: {e}"))?;

    let mut rows: Vec<ButterRow> = Vec::new();
    for batch_res in reader {
        let batch = batch_res.map_err(|e| format!("{path:?}: batch read: {e}"))?;
        let n_rows = batch.num_rows();
        // The projected schema preserves column order — first 3 are
        // ref_basename, codec, q, then 372 features in order.
        let ref_col = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: ref_basename not utf8"))?;
        let codec_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| format!("{path:?}: codec not utf8"))?;
        let q_col = &batch.column(2);
        let q_vals: Vec<i32> = if let Some(arr) = q_col.as_any().downcast_ref::<Int32Array>() {
            (0..n_rows).map(|i| arr.value(i)).collect()
        } else if let Some(arr) = q_col.as_any().downcast_ref::<Int64Array>() {
            (0..n_rows).map(|i| arr.value(i) as i32).collect()
        } else if let Some(arr) = q_col.as_any().downcast_ref::<UInt32Array>() {
            (0..n_rows).map(|i| arr.value(i) as i32).collect()
        } else {
            return Err(format!("{path:?}: q column unexpected type"));
        };
        // Feature columns may be either f32 or f64 depending on the
        // parquet's writer (the butter parquets use f32; the canonical
        // training/val parquets use f64). Read both.
        enum FeatCol<'a> {
            F32(&'a Float32Array),
            F64(&'a Float64Array),
        }
        let feat_cols: Vec<FeatCol> = (3..batch.num_columns())
            .map(|c| {
                let arr = batch.column(c);
                if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                    FeatCol::F32(a)
                } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                    FeatCol::F64(a)
                } else {
                    panic!("feature column not f32/f64: {:?}", arr.data_type());
                }
            })
            .collect();
        for (i, &q) in q_vals.iter().enumerate() {
            let feats: Vec<f64> = feat_cols
                .iter()
                .map(|fc| match fc {
                    FeatCol::F32(a) => a.value(i) as f64,
                    FeatCol::F64(a) => a.value(i),
                })
                .collect();
            rows.push(ButterRow {
                ref_basename: ref_col.value(i).to_string(),
                codec: codec_col.value(i).to_string(),
                q,
                features: feats,
            });
        }
    }
    Ok(rows)
}

struct ButterRow {
    ref_basename: String,
    #[allow(dead_code)]
    codec: String,
    q: i32,
    features: Vec<f64>,
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f64 * p).round() as usize;
    values[idx]
}

fn build_qsweep_table(bakes: &[LoadedBake], out: &mut String) -> Result<(), String> {
    // For each codec parquet: pick the same N_DEMO_IMAGES ref_basenames
    // (intersected across all 4 codecs to ensure overlap), then for
    // each (profile × codec × q) compute median + p25 + p75 of the
    // scores across the picked images.
    let mut codec_rows: Vec<(String, Vec<ButterRow>)> = Vec::new();
    let mut codec_refs: Vec<std::collections::BTreeSet<String>> = Vec::new();
    for codec in CODECS {
        let rows = load_butter_parquet(Path::new(codec.path))?;
        let refs: std::collections::BTreeSet<String> =
            rows.iter().map(|r| r.ref_basename.clone()).collect();
        codec_refs.push(refs);
        codec_rows.push((codec.codec.to_string(), rows));
    }
    // Pick the deterministic intersection of refs across all codecs.
    let mut common = codec_refs[0].clone();
    for s in &codec_refs[1..] {
        common = &common & s;
    }
    let mut common_sorted: Vec<String> = common.into_iter().collect();
    common_sorted.sort();
    let mut rng = Xs64::new(0xc0dec2026);
    rng.shuffle(&mut common_sorted);
    common_sorted.truncate(N_DEMO_IMAGES);
    let pick: std::collections::BTreeSet<String> = common_sorted.iter().cloned().collect();

    out.push_str(&format!(
        "\nPer-codec q→score table — {} source images (deterministic 0xc0dec2026 sample of refs common to all 4 codec butter parquets); each cell shows median (p25..p75) across the {} images.\n",
        N_DEMO_IMAGES, N_DEMO_IMAGES
    ));

    // For each profile, per-codec table.
    for bake in bakes {
        out.push_str(
            "\n────────────────────────────────────────────────────────────────────────\n",
        );
        out.push_str(&format!("Profile: {}\n", bake.label));
        out.push_str("────────────────────────────────────────────────────────────────────────\n");
        out.push_str(&format!(
            "  q   │ {:>20} │ {:>20} │ {:>20} │ {:>20}\n",
            "zenjpeg", "zenwebp", "zenavif", "zenjxl"
        ));
        out.push_str("──────┼──────────────────────┼──────────────────────┼──────────────────────┼──────────────────────\n");
        for &q in Q_SWEEP {
            let mut cells: Vec<String> = Vec::with_capacity(4);
            for (_codec_name, rows) in &codec_rows {
                let mut scratch = vec![0.0f32; bake.n_inputs];
                let scores: Vec<f64> = rows
                    .iter()
                    .filter(|r| r.q == q && pick.contains(&r.ref_basename))
                    .map(|r| score_features(bake, &mut scratch, &r.features))
                    .filter(|v| v.is_finite())
                    .collect();
                if scores.is_empty() {
                    cells.push("           n/a".to_string());
                } else {
                    let mut s = scores.clone();
                    let p50 = percentile(&mut s, 0.50);
                    let p25 = percentile(&mut scores.clone(), 0.25);
                    let p75 = percentile(&mut scores.clone(), 0.75);
                    cells.push(format!("{:>5.1} ({:>5.1}..{:>5.1})", p50, p25, p75));
                }
            }
            out.push_str(&format!(
                " {:>4} │ {:>20} │ {:>20} │ {:>20} │ {:>20}\n",
                q, cells[0], cells[1], cells[2], cells[3]
            ));
        }
    }
    Ok(())
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let t0 = std::time::Instant::now();
    let features_root = PathBuf::from(FEATURES_ROOT);

    let bakes: Vec<LoadedBake> = SHIPPING_PROFILES
        .iter()
        .filter_map(|p| match load_bake_for_profile(p) {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("WARN: skipping {} — {e}", p.label);
                None
            }
        })
        .collect();

    println!("┌────────────────────────────────────────────────────────────────────────┐");
    println!("│ ZENSIM PROFILE STATS DEMO — task #181 (2026-05-20)                     │");
    println!("│ AIC-4 sample added as default eval set (300 pairs, 5 src × 6 codecs × │");
    println!("│ 10 dlevels). bake_verdict default now: cid22,kadid,tid,konjnd,aic3,aic4│");
    println!("└────────────────────────────────────────────────────────────────────────┘");
    println!();

    // ────────────────────── Per-profile panel section ──────────────────────
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" SECTION 1 — Full Mohammadi panel per (profile, corpus)");
    println!(" Stats: SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE (all polarity-aligned)");
    println!(" 10-band breakdown shown only for CID22/KADID/TID (corpora that use a");
    println!(" 0..1 normalized human-score scale partition-friendly).");
    println!("════════════════════════════════════════════════════════════════════════");

    // Pre-load each corpus once (parquet I/O is the bulk of the wall
    // time; reading 4 × identical parquets per profile is wasted).
    eprintln!(
        "preview_stats_demo: pre-loading {} corpora...",
        CORPORA.len()
    );
    let loaded_corpora: Vec<LoadedCorpus> = CORPORA
        .iter()
        .filter_map(|c| match load_corpus_once(c, &features_root) {
            Ok(lc) => {
                eprintln!("  loaded {} ({} pairs)", c.display, lc.feature_rows.len());
                Some(lc)
            }
            Err(e) => {
                eprintln!("  WARN: skipping {} — {e}", c.display);
                None
            }
        })
        .collect();
    eprintln!();

    for bake in &bakes {
        println!();
        println!("╔════════════════════════════════════════════════════════════════════════");
        println!("║ {}", bake.label);
        let head_kind = if bake.per_sample_alpha_head.is_some() {
            "per-sample-α"
        } else if bake.hybrid_head.is_some() {
            "hybrid-head"
        } else {
            "plain-MLP"
        };
        let tanh = if bake.tanh_pin_scale.is_some() {
            "tanh-pin "
        } else {
            ""
        };
        let spline = if bake.output_spline.is_some() {
            "PCHIP-spline"
        } else {
            ""
        };
        println!(
            "║ n_inputs={}  head={}  feature_transforms={}  {}{}",
            bake.n_inputs, head_kind, bake.has_transforms, tanh, spline
        );
        println!("╚════════════════════════════════════════════════════════════════════════");
        let mut panel_out = String::new();
        for loaded in &loaded_corpora {
            if let Err(e) = render_corpus_panel(bake, loaded, &mut panel_out) {
                panel_out.push_str(&format!("│  {}: ERROR {e}\n", loaded.corpus.display));
            }
        }
        print!("{}", panel_out);
    }

    // ────────────────────── Dynamic range section ──────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" SECTION 2 — Dynamic range per profile (1000 random CID22 val pairs)");
    println!(" Shows the production-runtime score min, p5, p50, p95, max.");
    println!("════════════════════════════════════════════════════════════════════════");
    println!();
    println!(
        "{:<58} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7} │ {:>7}",
        "profile", "min", "p5", "p50", "p95", "max", "range"
    );
    println!(
        "{:─<58}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}┼{:─>9}",
        "", "", "", "", "", "", ""
    );
    // Re-use the pre-loaded CID22 corpus for dynamic range computation.
    let cid22 = loaded_corpora.iter().find(|lc| lc.corpus.name == "cid22");
    for bake in &bakes {
        let result = match cid22 {
            Some(c) => dynamic_range(bake, c),
            None => Err("cid22 corpus missing".to_string()),
        };
        match result {
            Ok((mn, p5, p50, p95, mx)) => {
                println!(
                    "{:<58} │ {:>7.2} │ {:>7.2} │ {:>7.2} │ {:>7.2} │ {:>7.2} │ {:>7.2}",
                    bake.label,
                    mn,
                    p5,
                    p50,
                    p95,
                    mx,
                    p95 - p5
                );
            }
            Err(e) => println!("{:<58} │ ERROR {e}", bake.label),
        }
    }

    // ────────────────────── q-sweep section ──────────────────────
    println!();
    println!("════════════════════════════════════════════════════════════════════════");
    println!(" SECTION 3 — Per-codec q→score mapping (10 source images, q ∈ {{5..95 step 10}})");
    println!(" Codec butter parquets at /mnt/v/zen/picker-training/2026-05-19/butter/");
    println!(" Cell shows median (p25..p75) across the 10 images.");
    println!("════════════════════════════════════════════════════════════════════════");

    let mut qsweep_out = String::new();
    if let Err(e) = build_qsweep_table(&bakes, &mut qsweep_out) {
        println!("ERROR building q-sweep table: {e}");
    } else {
        print!("{}", qsweep_out);
    }

    println!();
    println!("────────────────────────────────────────────────────────────────────────");
    println!(
        "preview_stats_demo: complete in {:.2}s",
        t0.elapsed().as_secs_f64()
    );
}
