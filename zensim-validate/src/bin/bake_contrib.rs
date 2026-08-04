//! `bake_contrib` — exact per-input contribution accounting for a ZNPR bake.
//!
//! Answers "what is this bake tuning out?" mathematically: for each input k,
//! set the STANDARDIZED post-transform input `x̃_k = (t_k − mean_k)/scale_k`
//! to 0 (= the raw value at the scaler mean in transform space) and recompute
//! the score. The only `x̃_k`-dependence in the network is layer 0's
//! pre-activation, so the ablation is a rank-1 update
//! `z0' = z0 − x̃_k·W0[k,:]` followed by an exact re-forward of activation,
//! remaining layers, head dispatch, tanh pin and output spline — exact for
//! any depth (fp subtract-out error O(ulp·|z0|) ≈ 1e-6, one order below the
//! registered dead threshold).
//!
//! Registered method + thresholds: `benchmarks/sota944_campaign_2026-08-03.md`
//! "REGISTERED APPENDIX — bake_contrib" (§C.1). Per input the tool reports
//! mean|Δ|, p95|Δ|, std(Δ) (the rank-relevant measure — a constant Δ is a
//! pure offset carrying zero rank information), sign-consistency, per-corpus
//! mean|Δ|, Δ-SROCC for the top movers, and two analytic cross-checks
//! (`std(x̃_k)·‖W0[k,:]‖₂` and the |W|-chain-propagated variant). Dead ⟺
//! mean|Δ| < 1e-4 AND p95|Δ| < 1e-3 (score units); rank-dead ⟺ std(Δ) < 1e-4.
//!
//! Baseline scores are parity-gated against `bake_runtime::score_row`
//! (≤ 1e-6 per row) — the head/pin/spline tail is the SAME code
//! (`bake_runtime::score_from_network_output`), so the tool cannot fork the
//! scoring math. Min-max-head and expander-transform bakes are out of scope
//! (loud bail). I8 layer-0 bakes use per-weight dequant (inference scales
//! after accumulation) — the parity gate quantifies any drift; the shortlist
//! is f32/f16 where the decomposition is bit-exact.
//!
//! ## Usage
//!
//! ```sh
//! bake_contrib --bake H.bin \
//!   --corpus cid22:/path/ext_cid22val.parquet:human_score:100 \
//!   --corpus kadid:/path/ext_kadid.parquet:human_score:100 \
//!   [--rows 4000] [--packed H_packed.bin] [--top-movers 64] \
//!   [--out contrib.tsv] [--dump-scores scores.tsv] [--summary summary.md]
//! ```
//!
//! Corpus spec is `name:path:target_col:target_scale`; `target_col = -`
//! means no target (no Δ-SROCC for that corpus). `--rows N` caps a corpus by
//! deterministic stride decimation (registered for the imazen26 slices; the
//! cap applies to corpora whose name ends `@cap`, e.g. `imazen26@cap:...`).

use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use rayon::prelude::*;
use zenpredict::{Activation, Model, Predictor, WeightStorage, f16_bits_to_f32};
use zensim_validate::bake_runtime::{
    HybridHeadDispatch, PerSampleAlphaHeadDispatch, extract_hybrid_head, extract_minmax_head,
    extract_per_sample_alpha_head, extract_tanh_output_head_scale, score_from_network_output,
    score_row,
};
use zensim_validate::output_calibration_spline::{self, OutputCalibrationSpline};
use zensim_validate::panel::spearman;
use zensim_validate::parquet_loader::load_parquet;

/// Registered dead thresholds (§C.1): dead ⟺ mean|Δ| < DEAD_MEAN_ABS AND
/// p95|Δ| < DEAD_P95; rank-dead ⟺ std(Δ) < RANK_DEAD_STD. Score units.
const DEAD_MEAN_ABS: f64 = 1e-4;
const DEAD_P95: f64 = 1e-3;
const RANK_DEAD_STD: f64 = 1e-4;

/// Owned f32 copy of one layer (dequantized once, exactly for f32/f16).
struct OwnedLayer {
    in_dim: usize,
    out_dim: usize,
    activation: Activation,
    /// Row-major `in_dim × out_dim`: input i's row spans `w[i*out..(i+1)*out]`.
    w: Vec<f32>,
    b: Vec<f32>,
    /// True when the storage dtype was I8 (decomposed forward is then not
    /// bit-exact vs the inference kernel, which scales after accumulation).
    was_i8: bool,
}

fn dequant_layer(l: &zenpredict::LayerView<'_>) -> OwnedLayer {
    let (w, was_i8) = match &l.weights {
        WeightStorage::F32(w) => (w.to_vec(), false),
        WeightStorage::F16(w) => (w.iter().map(|&h| f16_bits_to_f32(h)).collect(), false),
        WeightStorage::I8 { weights, scales } => (
            weights
                .iter()
                .enumerate()
                .map(|(idx, &q)| (q as f32) * scales[idx % l.out_dim])
                .collect(),
            true,
        ),
    };
    OwnedLayer {
        in_dim: l.in_dim,
        out_dim: l.out_dim,
        activation: l.activation,
        w,
        b: l.biases.to_vec(),
        was_i8,
    }
}

fn apply_activation(buf: &mut [f32], act: Activation) {
    match act {
        Activation::Identity => {}
        Activation::Relu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v = 0.0;
                }
            }
        }
        Activation::LeakyRelu => {
            for v in buf.iter_mut() {
                if *v < 0.0 {
                    *v *= zenpredict::LEAKY_RELU_ALPHA;
                }
            }
        }
        // Activation is #[non_exhaustive]; new variants must be wired here
        // AND in zenpredict::inference before this tool can trust its math.
        _ => panic!("bake_contrib: unsupported activation {:?}", act),
    }
}

/// Transform + standardize one row into `x̃` (f32), mirroring
/// `score_row` (take/pad f32 copy) + `predict_transformed` (scalar
/// transforms) + `inference::forward` (standardize, scale==0 ⇒ 1.0).
fn transform_standardize(model: &Model, row: &[f64], xt: &mut [f32]) {
    let n = xt.len();
    let take = n.min(row.len());
    let transforms = model.feature_transforms();
    let params = model.feature_transform_params();
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    for i in 0..n {
        let raw = if i < take { row[i] as f32 } else { 0.0f32 };
        let t = match (transforms, params) {
            (Some(tf), Some(p)) => tf[i].apply_with_params(raw, &p[i]),
            (Some(tf), None) => tf[i].apply_with_params(raw, &[]),
            (None, _) => raw,
        };
        let s = scale[i];
        let safe = if s == 0.0 { 1.0 } else { s };
        xt[i] = (t - mean[i]) / safe;
    }
}

/// Layer-0 pre-activation `z0 = b0 + W0ᵀx̃`, bit-exact with
/// `inference::saxpy_matmul_*` (ascending-i accumulation, `mul_add`,
/// `x̃==0` skip).
fn layer0_preact(l0: &OwnedLayer, xt: &[f32], z0: &mut [f32]) {
    z0.copy_from_slice(&l0.b);
    let out = l0.out_dim;
    for (i, &x) in xt.iter().enumerate().take(l0.in_dim) {
        if x == 0.0 {
            continue;
        }
        let row = &l0.w[i * out..(i + 1) * out];
        for j in 0..out {
            z0[j] = x.mul_add(row[j], z0[j]);
        }
    }
}

/// Forward from a layer-0 pre-activation through activation + remaining
/// layers; returns the network output vector (into `out_buf`).
fn forward_from_z0(
    layers: &[OwnedLayer],
    z0: &[f32],
    h_buf: &mut Vec<f32>,
    out_buf: &mut Vec<f32>,
) {
    h_buf.clear();
    h_buf.extend_from_slice(z0);
    apply_activation(h_buf, layers[0].activation);
    for l in &layers[1..] {
        out_buf.clear();
        out_buf.extend_from_slice(&l.b);
        for (i, &x) in h_buf.iter().enumerate().take(l.in_dim) {
            if x == 0.0 {
                continue;
            }
            let row = &l.w[i * l.out_dim..(i + 1) * l.out_dim];
            for j in 0..l.out_dim {
                out_buf[j] = x.mul_add(row[j], out_buf[j]);
            }
        }
        apply_activation(out_buf, l.activation);
        std::mem::swap(h_buf, out_buf);
    }
    // Result lives in h_buf after the final swap.
    std::mem::swap(h_buf, out_buf);
}

struct Heads<'a> {
    psa: Option<&'a PerSampleAlphaHeadDispatch>,
    hybrid: Option<&'a HybridHeadDispatch>,
    pin: Option<f64>,
    spline: Option<&'a OutputCalibrationSpline>,
}

/// Per-row ablation result: baseline score + Δscore per input.
struct RowResult {
    baseline: f64,
    deltas: Vec<f32>,
    xt: Vec<f32>,
}

fn ablate_row(layers: &[OwnedLayer], heads: &Heads<'_>, model: &Model, row: &[f64]) -> RowResult {
    let n = layers[0].in_dim;
    let out = layers[0].out_dim;
    let mut xt = vec![0.0f32; n];
    transform_standardize(model, row, &mut xt);
    let mut z0 = vec![0.0f32; out];
    layer0_preact(&layers[0], &xt, &mut z0);
    let mut h = Vec::with_capacity(out.max(8));
    let mut o = Vec::with_capacity(out.max(8));
    forward_from_z0(layers, &z0, &mut h, &mut o);
    let baseline = score_from_network_output(&o, heads.psa, heads.hybrid, heads.pin, heads.spline);
    let mut deltas = vec![0.0f32; n];
    let mut z_abl = vec![0.0f32; out];
    for k in 0..n {
        let x = xt[k];
        if x == 0.0 {
            continue; // Δ ≡ 0 exactly: the forward skips x̃==0 terms.
        }
        let wrow = &layers[0].w[k * out..(k + 1) * out];
        for j in 0..out {
            z_abl[j] = z0[j] - x * wrow[j];
        }
        forward_from_z0(layers, &z_abl, &mut h, &mut o);
        let s = score_from_network_output(&o, heads.psa, heads.hybrid, heads.pin, heads.spline);
        deltas[k] = (s - baseline) as f32;
    }
    RowResult {
        baseline,
        deltas,
        xt,
    }
}

/// Feature-family label per the registered aggregation keys (§C.3).
fn family_of(idx: usize, n_inputs: usize) -> &'static str {
    match n_inputs {
        944 => match idx {
            0..=155 => "v1fold156",
            156..=371 => "zeros156-371",
            372..=719 => "v2-348",
            720..=923 => "append204",
            _ => "tail20",
        },
        720..=943 => match idx {
            0..=155 => "v1fold156",
            156..=371 => "zeros156-371",
            372..=719 => "v2-348",
            _ => "append204",
        },
        300..=719 => match idx {
            0..=155 => "basic",
            156..=227 => "peaks",
            228..=299 => "masked",
            _ => "iw",
        },
        156 => "v1basic",
        _ => "all",
    }
}

const FAMILY_ORDER: &[&str] = &[
    "v1fold156",
    "zeros156-371",
    "v2-348",
    "append204",
    "tail20",
    "basic",
    "peaks",
    "masked",
    "iw",
    "v1basic",
    "all",
];

struct CorpusSpec {
    name: String,
    path: PathBuf,
    target_col: Option<String>,
    target_scale: f64,
    /// Per-corpus stride-decimation cap (`name@N` spec syntax); falls back
    /// to the global `--rows` cap when absent.
    cap: Option<usize>,
}

struct CorpusData {
    name: String,
    /// Original parquet row index per kept row (stride decimation aware).
    row_idx: Vec<usize>,
    rows: Vec<Vec<f64>>,
    targets: Option<Vec<f64>>,
}

fn stride_cap<T: Clone>(v: &[T], cap: usize) -> Vec<T> {
    if v.len() <= cap {
        return v.to_vec();
    }
    let stride = v.len().div_ceil(cap);
    v.iter().step_by(stride).cloned().collect()
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let mut bake: Option<PathBuf> = None;
    let mut packed: Option<PathBuf> = None;
    let mut corpora: Vec<CorpusSpec> = Vec::new();
    let mut rows_cap: Option<usize> = None;
    let mut top_movers: usize = 64;
    let mut out_tsv: Option<PathBuf> = None;
    let mut dump_scores: Option<PathBuf> = None;
    let mut summary_path: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bake" => bake = Some(args.next().expect("--bake PATH").into()),
            "--packed" => packed = Some(args.next().expect("--packed PATH").into()),
            "--rows" => rows_cap = Some(args.next().expect("--rows N").parse().expect("usize")),
            "--top-movers" => {
                top_movers = args.next().expect("--top-movers K").parse().expect("usize");
            }
            "--out" => out_tsv = Some(args.next().expect("--out PATH").into()),
            "--dump-scores" => dump_scores = Some(args.next().expect("--dump-scores PATH").into()),
            "--summary" => summary_path = Some(args.next().expect("--summary PATH").into()),
            "--corpus" => {
                let spec = args.next().expect("--corpus name:path:target:scale");
                let parts: Vec<&str> = spec.split(':').collect();
                if parts.len() != 4 {
                    eprintln!("bad --corpus spec {spec:?} (want name:path:target_col:scale)");
                    return ExitCode::from(2);
                }
                let (name, cap) = match parts[0].split_once('@') {
                    Some((n, c)) => (n.to_string(), Some(c.parse().expect("name@N cap usize"))),
                    None => (parts[0].to_string(), None),
                };
                corpora.push(CorpusSpec {
                    name,
                    path: parts[1].into(),
                    target_col: if parts[2] == "-" {
                        None
                    } else {
                        Some(parts[2].to_string())
                    },
                    target_scale: parts[3].parse().expect("target_scale f64"),
                    cap,
                });
            }
            other => {
                eprintln!("unknown arg: {other}");
                return ExitCode::from(2);
            }
        }
    }
    let Some(bake_path) = bake else {
        eprintln!(
            "usage: bake_contrib --bake X.bin --corpus name:path:target:scale [...] \
             [--packed Y.bin] [--rows N] [--top-movers K] [--out TSV] \
             [--dump-scores TSV] [--summary MD]"
        );
        return ExitCode::from(2);
    };
    if corpora.is_empty() {
        eprintln!("at least one --corpus required");
        return ExitCode::from(2);
    }

    let bytes = match fs::read(&bake_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("read {}: {e}", bake_path.display());
            return ExitCode::from(1);
        }
    };
    let model = match Model::from_bytes(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("parse {}: {e:?}", bake_path.display());
            return ExitCode::from(1);
        }
    };
    if extract_minmax_head(&model).is_some() {
        eprintln!("min-max-head bakes are out of scope for bake_contrib (registered §C.1)");
        return ExitCode::from(2);
    }
    if model.has_expander_feature_transforms() {
        // Includes PRUNED bakes (FeatureTransform::Drop). Their contribution
        // report is the un-pruned parent's, minus the dropped columns — run
        // this against the pre-pack bake.
        eprintln!(
            "expander/drop-transform bakes are out of scope for bake_contrib \
             (registered §C.1); for a pruned bake, report on its pre-pack parent"
        );
        return ExitCode::from(2);
    }
    let n_inputs = model.n_inputs();
    let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
    if layers.is_empty() || layers[0].in_dim != n_inputs {
        eprintln!("bake has no layers / layer0 in_dim mismatch");
        return ExitCode::from(1);
    }
    let any_i8 = layers.iter().any(|l| l.was_i8);
    let psa = extract_per_sample_alpha_head(&model);
    let hybrid = extract_hybrid_head(&model);
    let pin = extract_tanh_output_head_scale(&model);
    let spline = output_calibration_spline::extract(&model);
    let heads = Heads {
        psa: psa.as_ref(),
        hybrid: hybrid.as_ref(),
        pin,
        spline: spline.as_ref(),
    };
    let has_transforms = model.has_nontrivial_feature_transforms();

    // ---- load corpora --------------------------------------------------
    let mut data: Vec<CorpusData> = Vec::new();
    for spec in &corpora {
        let target_col = spec.target_col.as_deref().unwrap_or("human_score");
        let g = match load_parquet(&spec.path, &spec.name, target_col, spec.target_scale) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("load {}: {e}", spec.path.display());
                return ExitCode::from(1);
            }
        };
        let idx: Vec<usize> = (0..g.feature_rows.len()).collect();
        let (row_idx, rows, targets) = match spec.cap.or(rows_cap) {
            Some(cap) if g.feature_rows.len() > cap => (
                stride_cap(&idx, cap),
                stride_cap(&g.feature_rows, cap),
                spec.target_col
                    .as_ref()
                    .map(|_| stride_cap(&g.human_scores, cap)),
            ),
            _ => (
                idx,
                g.feature_rows,
                spec.target_col.as_ref().map(|_| g.human_scores),
            ),
        };
        eprintln!(
            "corpus {}: {} rows ({} loaded), {} features",
            spec.name,
            rows.len(),
            row_idx.len(),
            g.n_features
        );
        data.push(CorpusData {
            name: spec.name.clone(),
            row_idx,
            rows,
            targets,
        });
    }

    // ---- ablation ------------------------------------------------------
    let n_corpora = data.len();
    let mut deltas: Vec<Vec<f32>> = Vec::with_capacity(n_inputs); // [input][global_row]
    let total_rows: usize = data.iter().map(|c| c.rows.len()).sum();
    for _ in 0..n_inputs {
        deltas.push(Vec::with_capacity(total_rows));
    }
    let mut baselines: Vec<Vec<f64>> = Vec::new(); // per corpus
    let mut xt_sum = vec![0.0f64; n_inputs];
    let mut xt_sumsq = vec![0.0f64; n_inputs];
    let mut xt_nonzero = vec![0usize; n_inputs];

    // Parity gate vs the canonical per-row runtime.
    let mut parity_max = 0.0f64;
    let mut parity_violations = 0usize;
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];

    for c in &data {
        let results: Vec<RowResult> = c
            .rows
            .par_iter()
            .map(|row| ablate_row(&layers, &heads, &model, row))
            .collect();
        let mut base = Vec::with_capacity(results.len());
        for (row, r) in c.rows.iter().zip(&results) {
            let canonical = score_row(
                &mut predictor,
                has_transforms,
                psa.as_ref(),
                hybrid.as_ref(),
                pin,
                spline.as_ref(),
                &mut scratch,
                row,
            );
            let diff = (canonical - r.baseline).abs();
            if diff > parity_max {
                parity_max = diff;
            }
            if diff > 1e-6 {
                parity_violations += 1;
            }
            base.push(r.baseline);
            for k in 0..n_inputs {
                deltas[k].push(r.deltas[k]);
                let x = r.xt[k] as f64;
                xt_sum[k] += x;
                xt_sumsq[k] += x * x;
                if r.xt[k] != 0.0 {
                    xt_nonzero[k] += 1;
                }
            }
        }
        baselines.push(base);
    }

    // ---- per-input aggregates -----------------------------------------
    let n_rows_f = total_rows as f64;
    let mut mean_abs = vec![0.0f64; n_inputs];
    let mut p95_abs = vec![0.0f64; n_inputs];
    let mut std_delta = vec![0.0f64; n_inputs];
    let mut sign_cons = vec![1.0f64; n_inputs];
    let mut n_nonzero_delta = vec![0usize; n_inputs];
    let mut per_corpus_mean_abs = vec![vec![0.0f64; n_corpora]; n_inputs];
    for k in 0..n_inputs {
        let d = &deltas[k];
        let mut sum = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut n_pos = 0usize;
        let mut n_neg = 0usize;
        for &v in d.iter() {
            let v = v as f64;
            sum += v;
            sum_abs += v.abs();
            sum_sq += v * v;
            if v > 0.0 {
                n_pos += 1;
            } else if v < 0.0 {
                n_neg += 1;
            }
        }
        mean_abs[k] = sum_abs / n_rows_f;
        let mean = sum / n_rows_f;
        std_delta[k] = (sum_sq / n_rows_f - mean * mean).max(0.0).sqrt();
        n_nonzero_delta[k] = n_pos + n_neg;
        if n_pos + n_neg > 0 {
            sign_cons[k] = n_pos.max(n_neg) as f64 / (n_pos + n_neg) as f64;
        }
        let mut absd: Vec<f32> = d.iter().map(|v| v.abs()).collect();
        absd.sort_by(|a, b| a.total_cmp(b));
        let p95_idx = ((0.95 * (absd.len() as f64 - 1.0)).round() as usize).min(absd.len() - 1);
        p95_abs[k] = absd[p95_idx] as f64;
        let mut off = 0usize;
        for (ci, c) in data.iter().enumerate() {
            let n = c.rows.len();
            let s: f64 = d[off..off + n].iter().map(|v| v.abs() as f64).sum();
            per_corpus_mean_abs[k][ci] = s / n as f64;
            off += n;
        }
    }
    let dead: Vec<bool> = (0..n_inputs)
        .map(|k| mean_abs[k] < DEAD_MEAN_ABS && p95_abs[k] < DEAD_P95)
        .collect();
    let rank_dead: Vec<bool> = (0..n_inputs)
        .map(|k| std_delta[k] < RANK_DEAD_STD)
        .collect();

    // ---- analytic cross-check ------------------------------------------
    // g = |W|-chain back-propagated ones from the output to hidden-0.
    let mut g = vec![1.0f64; layers.last().unwrap().out_dim];
    for l in layers[1..].iter().rev() {
        let mut g_prev = vec![0.0f64; l.in_dim];
        for (i, gp) in g_prev.iter_mut().enumerate() {
            let row = &l.w[i * l.out_dim..(i + 1) * l.out_dim];
            *gp = row
                .iter()
                .zip(&g)
                .map(|(&w, &gv)| (w.abs() as f64) * gv)
                .sum();
        }
        g = g_prev;
    }
    let mut analytic_simple = vec![0.0f64; n_inputs];
    let mut analytic_path = vec![0.0f64; n_inputs];
    for k in 0..n_inputs {
        let m = xt_sum[k] / n_rows_f;
        let var = (xt_sumsq[k] / n_rows_f - m * m).max(0.0);
        let sd = var.sqrt();
        let row = &layers[0].w[k * layers[0].out_dim..(k + 1) * layers[0].out_dim];
        let l2: f64 = row
            .iter()
            .map(|&w| (w as f64) * (w as f64))
            .sum::<f64>()
            .sqrt();
        let l2p: f64 = row
            .iter()
            .zip(&g)
            .map(|(&w, &gv)| {
                let t = (w as f64) * gv;
                t * t
            })
            .sum::<f64>()
            .sqrt();
        analytic_simple[k] = sd * l2;
        analytic_path[k] = sd * l2p;
    }
    let srocc_simple = spearman(&analytic_simple, &mean_abs).abs();
    let srocc_path = spearman(&analytic_path, &mean_abs).abs();

    // ---- Δ-SROCC for the top movers ------------------------------------
    let mut order: Vec<usize> = (0..n_inputs).collect();
    order.sort_by(|&a, &b| mean_abs[b].total_cmp(&mean_abs[a]));
    let top: Vec<usize> = order.iter().copied().take(top_movers).collect();
    // dsrocc[input][corpus] — only filled for top movers on target corpora.
    let mut dsrocc: Vec<Vec<Option<f64>>> = vec![vec![None; n_corpora]; n_inputs];
    let mut base_srocc = vec![None; n_corpora];
    {
        let mut off = 0usize;
        for (ci, c) in data.iter().enumerate() {
            let n = c.rows.len();
            if let Some(t) = &c.targets {
                let b0 = spearman(&baselines[ci], t).abs();
                base_srocc[ci] = Some(b0);
                for &k in &top {
                    let abl: Vec<f64> = baselines[ci]
                        .iter()
                        .zip(&deltas[k][off..off + n])
                        .map(|(&b, &d)| b + d as f64)
                        .collect();
                    dsrocc[k][ci] = Some(spearman(&abl, t).abs() - b0);
                }
            }
            off += n;
        }
    }

    // ---- packed-twin zero stats ----------------------------------------
    let mut packed_zero_frac: Option<Vec<f64>> = None;
    let mut packed_report = String::new();
    if let Some(pp) = &packed {
        match fs::read(pp) {
            Ok(pb) => match Model::from_bytes(&pb) {
                Ok(pm) if pm.n_inputs() == n_inputs => {
                    let pl0 = pm.layer(0);
                    let out = pl0.out_dim;
                    let mut frac = vec![0.0f64; n_inputs];
                    let mut all_zero = 0usize;
                    for (k, f) in frac.iter_mut().enumerate() {
                        let zeros = match &pl0.weights {
                            WeightStorage::F32(w) => w[k * out..(k + 1) * out]
                                .iter()
                                .filter(|&&v| v == 0.0)
                                .count(),
                            WeightStorage::F16(w) => w[k * out..(k + 1) * out]
                                .iter()
                                .filter(|&&h| h & 0x7fff == 0)
                                .count(),
                            WeightStorage::I8 { weights, .. } => weights[k * out..(k + 1) * out]
                                .iter()
                                .filter(|&&q| q == 0)
                                .count(),
                        };
                        *f = zeros as f64 / out as f64;
                        if zeros == out {
                            all_zero += 1;
                        }
                    }
                    let dtype_bytes = match &pl0.weights {
                        WeightStorage::F32(_) => 4,
                        WeightStorage::F16(_) => 2,
                        WeightStorage::I8 { .. } => 1,
                    };
                    let ablation_dead = dead.iter().filter(|&&d| d).count();
                    let both = (0..n_inputs).filter(|&k| dead[k] && frac[k] == 1.0).count();
                    let _ = writeln!(
                        packed_report,
                        "packed twin: {} ({} bytes). all-zero L0 columns: {all_zero}/{n_inputs}; \
                         ablation-dead ∩ packed-all-zero: {both} (ablation-dead: {ablation_dead}).\n\
                         prune arithmetic (registered, arithmetic only): removing an input column \
                         frees out_dim×{dtype_bytes}B (L0 row) + 8B (scaler mean+scale) ⇒ \
                         all-zero prune ≈ {} B; ablation-dead prune ≈ {} B (transform/bounds \
                         entries not counted; sparse_overrides implementation is future work).",
                        pp.display(),
                        pb.len(),
                        all_zero * (out * dtype_bytes + 8),
                        ablation_dead * (out * dtype_bytes + 8),
                    );
                    packed_zero_frac = Some(frac);
                }
                Ok(pm) => {
                    let _ = writeln!(
                        packed_report,
                        "packed twin SKIPPED: n_inputs {} != {}",
                        pm.n_inputs(),
                        n_inputs
                    );
                }
                Err(e) => {
                    let _ = writeln!(packed_report, "packed twin parse failed: {e:?}");
                }
            },
            Err(e) => {
                let _ = writeln!(packed_report, "packed twin read failed: {e}");
            }
        }
    }

    // ---- dump per-row baseline scores ----------------------------------
    if let Some(dp) = &dump_scores {
        let mut s = String::from("corpus\trow_idx\ttarget\tbaseline_score\n");
        for (ci, c) in data.iter().enumerate() {
            for (r, &ri) in c.row_idx.iter().enumerate() {
                let t = c
                    .targets
                    .as_ref()
                    .map(|t| format!("{:.9}", t[r]))
                    .unwrap_or_else(|| "-".into());
                let _ = writeln!(s, "{}\t{}\t{}\t{:.9}", c.name, ri, t, baselines[ci][r]);
            }
        }
        if let Err(e) = fs::write(dp, s) {
            eprintln!("write {}: {e}", dp.display());
            return ExitCode::from(1);
        }
    }

    // ---- TSV ------------------------------------------------------------
    if let Some(op) = &out_tsv {
        let mut s = String::from(
            "idx\tfamily\tfrac_rows_nonzero_xt\tmean_abs\tp95_abs\tstd\tsign_cons\tanalytic_simple\tanalytic_path\tpacked_zero_frac\tdead\trank_dead",
        );
        for c in &data {
            let _ = write!(s, "\tmean_abs_{}", c.name);
        }
        for c in &data {
            let _ = write!(s, "\tdsrocc_{}", c.name);
        }
        s.push('\n');
        for k in 0..n_inputs {
            let _ = write!(
                s,
                "{k}\t{}\t{:.4}\t{:.6e}\t{:.6e}\t{:.6e}\t{:.4}\t{:.6e}\t{:.6e}\t{}\t{}\t{}",
                family_of(k, n_inputs),
                xt_nonzero[k] as f64 / n_rows_f,
                mean_abs[k],
                p95_abs[k],
                std_delta[k],
                sign_cons[k],
                analytic_simple[k],
                analytic_path[k],
                packed_zero_frac
                    .as_ref()
                    .map(|f| format!("{:.4}", f[k]))
                    .unwrap_or_else(|| "-".into()),
                dead[k] as u8,
                rank_dead[k] as u8,
            );
            for v in &per_corpus_mean_abs[k][..n_corpora] {
                let _ = write!(s, "\t{v:.6e}");
            }
            for d in &dsrocc[k][..n_corpora] {
                match d {
                    Some(v) => {
                        let _ = write!(s, "\t{v:.6}");
                    }
                    None => s.push_str("\t-"),
                }
            }
            s.push('\n');
        }
        if let Err(e) = fs::write(op, s) {
            eprintln!("write {}: {e}", op.display());
            return ExitCode::from(1);
        }
    }

    // ---- summary --------------------------------------------------------
    let mut md = String::new();
    let _ = writeln!(md, "# bake_contrib — {}", bake_path.display());
    let _ = writeln!(
        md,
        "\nn_inputs={n_inputs}, layers={:?}, heads: psa={} hybrid={} pin={} spline={}, transforms={}, i8_dequant_caveat={}",
        layers
            .iter()
            .map(|l| format!("{}x{}:{:?}", l.in_dim, l.out_dim, l.activation))
            .collect::<Vec<_>>(),
        psa.is_some(),
        hybrid.is_some(),
        pin.is_some(),
        spline.is_some(),
        has_transforms,
        any_i8,
    );
    let _ = writeln!(
        md,
        "rows scored: {total_rows} across {n_corpora} corpora; parity vs score_row: max|diff|={parity_max:.3e}, violations(>1e-6)={parity_violations}"
    );
    let _ = writeln!(
        md,
        "analytic-vs-ablation SROCC: simple={srocc_simple:.4}, path={srocc_path:.4}"
    );
    // Structural-zero gate (944 regime).
    if n_inputs == 944 {
        let bad: Vec<usize> = (156..372).filter(|&k| mean_abs[k] != 0.0).collect();
        let _ = writeln!(
            md,
            "structural-zero gate f156-371: {} — {} nonzero of 216",
            if bad.is_empty() {
                "PASS (Δ ≡ 0 exactly)"
            } else {
                "FAIL"
            },
            bad.len()
        );
    }
    let _ = writeln!(
        md,
        "\ndead (mean|Δ|<{DEAD_MEAN_ABS:.0e} ∧ p95|Δ|<{DEAD_P95:.0e}): {} / {n_inputs}; rank-dead (std<{RANK_DEAD_STD:.0e}): {}",
        dead.iter().filter(|&&d| d).count(),
        rank_dead.iter().filter(|&&d| d).count()
    );
    let _ = writeln!(md, "\n## Family profile\n");
    let _ = writeln!(
        md,
        "| family | n | dead | rank-dead | Σmean|Δ| share | {} |",
        data.iter()
            .map(|c| format!("share {}", c.name))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let _ = writeln!(md, "|---|---:|---:|---:|---:|{}", "---:|".repeat(n_corpora));
    let total_mean_abs: f64 = mean_abs.iter().sum();
    let per_corpus_total: Vec<f64> = (0..n_corpora)
        .map(|ci| (0..n_inputs).map(|k| per_corpus_mean_abs[k][ci]).sum())
        .collect();
    for fam in FAMILY_ORDER {
        let idxs: Vec<usize> = (0..n_inputs)
            .filter(|&k| family_of(k, n_inputs) == *fam)
            .collect();
        if idxs.is_empty() {
            continue;
        }
        let fam_sum: f64 = idxs.iter().map(|&k| mean_abs[k]).sum();
        let fam_dead = idxs.iter().filter(|&&k| dead[k]).count();
        let fam_rdead = idxs.iter().filter(|&&k| rank_dead[k]).count();
        let per_c: Vec<String> = (0..n_corpora)
            .map(|ci| {
                let s: f64 = idxs.iter().map(|&k| per_corpus_mean_abs[k][ci]).sum();
                format!(
                    "{:.1}%",
                    100.0 * s / per_corpus_total[ci].max(f64::MIN_POSITIVE)
                )
            })
            .collect();
        let _ = writeln!(
            md,
            "| {fam} | {} | {fam_dead} | {fam_rdead} | {:.1}% | {} |",
            idxs.len(),
            100.0 * fam_sum / total_mean_abs.max(f64::MIN_POSITIVE),
            per_c.join(" | ")
        );
    }
    let _ = writeln!(md, "\n## Top movers (by overall mean|Δ|)\n");
    let _ = writeln!(
        md,
        "| idx | family | mean|Δ| | std | sign | {} |",
        data.iter()
            .map(|c| format!("ΔSROCC {}", c.name))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    let _ = writeln!(md, "|---:|---|---:|---:|---:|{}", "---:|".repeat(n_corpora));
    for &k in top.iter().take(16) {
        let per_c: Vec<String> = (0..n_corpora)
            .map(|ci| {
                dsrocc[k][ci]
                    .map(|v| format!("{v:+.4}"))
                    .unwrap_or_else(|| "-".into())
            })
            .collect();
        let _ = writeln!(
            md,
            "| {k} | {} | {:.4} | {:.4} | {:.2} | {} |",
            family_of(k, n_inputs),
            mean_abs[k],
            std_delta[k],
            sign_cons[k],
            per_c.join(" | ")
        );
    }
    let _ = writeln!(
        md,
        "\nbaseline |SROCC| per corpus: {}",
        data.iter()
            .enumerate()
            .map(|(ci, c)| format!(
                "{}={}",
                c.name,
                base_srocc[ci]
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or_else(|| "-".into())
            ))
            .collect::<Vec<_>>()
            .join(", ")
    );
    if !packed_report.is_empty() {
        let _ = writeln!(md, "\n## Packed twin\n\n{packed_report}");
    }
    println!("{md}");
    if let Some(sp) = &summary_path
        && let Err(e) = fs::write(sp, &md)
    {
        eprintln!("write {}: {e}", sp.display());
        return ExitCode::from(1);
    }

    ExitCode::SUCCESS
}

// ============================================================================
// Tests — the registered correctness gates (§C.1 b/c) that don't need a real
// 944 bake: a hand-built 3-input fixture with a known-dead input, the rank-1
// exactness bound, and ablation-vs-analytic agreement on the fixture.
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;
    use zenpredict::{Model, WeightDtype};
    use zenpredict_bake::{BakeLayer, BakeRequest, bake};

    /// 3-input fixture: input 1 has an all-zero W0 column ⇒ known-dead.
    /// Layer0 3→2 LeakyRelu, layer1 2→1 Identity.
    fn fixture_bytes() -> Vec<u8> {
        let w0 = [
            0.5f32, -0.3, /* input1: */ 0.0, 0.0, /* input2: */ 0.8, 0.1,
        ];
        let b0 = [0.1f32, -0.2];
        let w1 = [1.0f32, -0.7];
        let b1 = [0.05f32];
        let layers = [
            BakeLayer {
                in_dim: 3,
                out_dim: 2,
                activation: Activation::LeakyRelu,
                dtype: WeightDtype::F32,
                weights: &w0,
                biases: &b0,
            },
            BakeLayer {
                in_dim: 2,
                out_dim: 1,
                activation: Activation::Identity,
                dtype: WeightDtype::F32,
                weights: &w1,
                biases: &b1,
            },
        ];
        bake(&BakeRequest {
            schema_hash: 0,
            flags: 0,
            scaler_mean: &[0.2, 0.5, -0.1],
            scaler_scale: &[1.5, 2.0, 0.7],
            layers: &layers,
            feature_bounds: &[],
            metadata: &[],
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: false,
            hu_permutations: None,
        })
        .expect("fixture bake")
    }

    fn fixture_rows() -> Vec<Vec<f64>> {
        vec![
            vec![0.9, 1.0, -0.4],
            vec![-0.3, 0.2, 0.6],
            vec![0.2, 0.5, -0.1], // == scaler mean ⇒ x̃ ≡ 0 ⇒ every Δ ≡ 0
            vec![2.5, -1.0, 0.3],
            vec![0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn known_dead_input_is_exactly_dead_and_live_inputs_are_not() {
        let bytes = fixture_bytes();
        let model = Model::from_bytes(&bytes).unwrap();
        let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
        let heads = Heads {
            psa: None,
            hybrid: None,
            pin: None,
            spline: None,
        };
        for row in fixture_rows() {
            let r = ablate_row(&layers, &heads, &model, &row);
            // Input 1's W0 column is all-zero ⇒ removing it never moves z0.
            assert_eq!(r.deltas[1], 0.0, "known-dead input must have Δ == 0");
        }
        // Live inputs move the score on at least one non-mean row.
        let r = ablate_row(&layers, &heads, &model, &fixture_rows()[0]);
        assert!(r.deltas[0].abs() > 1e-4, "input0 should be live");
        assert!(r.deltas[2].abs() > 1e-4, "input2 should be live");
    }

    #[test]
    fn row_at_scaler_mean_has_all_zero_deltas() {
        // The structural-zero gate mechanism: x̃ == 0 ⇒ Δ ≡ 0 exactly.
        let bytes = fixture_bytes();
        let model = Model::from_bytes(&bytes).unwrap();
        let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
        let heads = Heads {
            psa: None,
            hybrid: None,
            pin: None,
            spline: None,
        };
        let r = ablate_row(&layers, &heads, &model, &fixture_rows()[2]);
        assert!(r.deltas.iter().all(|&d| d == 0.0));
    }

    #[test]
    fn rank1_update_matches_full_recompute_with_input_zeroed() {
        // Exactness gate: the rank-1 z0 update must equal a full re-forward
        // with x̃_k set to 0, to fp reassociation error (≤1e-5 score units).
        let bytes = fixture_bytes();
        let model = Model::from_bytes(&bytes).unwrap();
        let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
        let heads = Heads {
            psa: None,
            hybrid: None,
            pin: None,
            spline: None,
        };
        for row in fixture_rows() {
            let r = ablate_row(&layers, &heads, &model, &row);
            let mut xt = vec![0.0f32; 3];
            transform_standardize(&model, &row, &mut xt);
            for k in 0..3 {
                let mut xt_abl = xt.clone();
                xt_abl[k] = 0.0;
                let mut z0 = vec![0.0f32; 2];
                layer0_preact(&layers[0], &xt_abl, &mut z0);
                let (mut h, mut o) = (Vec::new(), Vec::new());
                forward_from_z0(&layers, &z0, &mut h, &mut o);
                let full = score_from_network_output(&o, None, None, None, None);
                let mut z0_base = vec![0.0f32; 2];
                layer0_preact(&layers[0], &xt, &mut z0_base);
                let (mut h2, mut o2) = (Vec::new(), Vec::new());
                forward_from_z0(&layers, &z0_base, &mut h2, &mut o2);
                let base = score_from_network_output(&o2, None, None, None, None);
                let rank1 = base + r.deltas[k] as f64;
                assert!(
                    (full - rank1).abs() <= 1e-5,
                    "rank-1 vs full recompute diverged: k={k} full={full} rank1={rank1}"
                );
            }
        }
    }

    #[test]
    fn baseline_matches_canonical_score_row() {
        // The parity gate as a unit test: decomposed baseline == score_row.
        let bytes = fixture_bytes();
        let model = Model::from_bytes(&bytes).unwrap();
        let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
        let heads = Heads {
            psa: None,
            hybrid: None,
            pin: None,
            spline: None,
        };
        let mut predictor = Predictor::new(&model);
        let mut scratch = vec![0.0f32; 3];
        for row in fixture_rows() {
            let r = ablate_row(&layers, &heads, &model, &row);
            let canonical = score_row(
                &mut predictor,
                model.has_nontrivial_feature_transforms(),
                None,
                None,
                None,
                None,
                &mut scratch,
                &row,
            );
            assert!(
                (canonical - r.baseline).abs() <= 1e-6,
                "parity: canonical={canonical} decomposed={}",
                r.baseline
            );
        }
    }

    #[test]
    fn analytic_agrees_with_ablation_on_the_fixture_dead_set() {
        // The dead input must be at the bottom of BOTH rankings; live inputs
        // must have positive analytic and positive ablation contribution.
        let bytes = fixture_bytes();
        let model = Model::from_bytes(&bytes).unwrap();
        let layers: Vec<OwnedLayer> = model.layers().map(|l| dequant_layer(&l)).collect();
        let heads = Heads {
            psa: None,
            hybrid: None,
            pin: None,
            spline: None,
        };
        let rows = fixture_rows();
        let mut mean_abs = [0.0f64; 3];
        let mut xt_sum = [0.0f64; 3];
        let mut xt_sumsq = [0.0f64; 3];
        for row in &rows {
            let r = ablate_row(&layers, &heads, &model, row);
            for k in 0..3 {
                mean_abs[k] += r.deltas[k].abs() as f64 / rows.len() as f64;
                let x = r.xt[k] as f64;
                xt_sum[k] += x;
                xt_sumsq[k] += x * x;
            }
        }
        let n = rows.len() as f64;
        for k in 0..3 {
            let m = xt_sum[k] / n;
            let sd = (xt_sumsq[k] / n - m * m).max(0.0).sqrt();
            let row = &layers[0].w[k * 2..(k + 1) * 2];
            let l2 = row
                .iter()
                .map(|&w| (w as f64) * (w as f64))
                .sum::<f64>()
                .sqrt();
            let analytic = sd * l2;
            if k == 1 {
                assert_eq!(analytic, 0.0, "dead input has zero analytic norm");
                assert_eq!(mean_abs[k], 0.0, "dead input has zero ablation");
            } else {
                assert!(analytic > 1e-4 && mean_abs[k] > 1e-4, "live input {k}");
            }
        }
    }
}
