//! `bake_compare` — canonical "A vs B" decisive-comparison tool
//! implementing § A.9 of `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`.
//!
//! Every ship/no-ship decision for a V_X bake should flow through
//! this binary. It emits:
//!
//! 1. The full Mohammadi 2025 panel (SROCC + PLCC + KROCC + OR +
//!    PWRC + Z-RMSE) per (corpus, band) for BOTH bakes — the same
//!    columns `bake_verdict` emits, but rendered side-by-side.
//!
//! 2. A per-(corpus, band) MRR table with columns
//!    `h_SROCC, p_SROCC, h_Z-RMSE, p_Z-RMSE, PWRC_diff, agreement_count,
//!    n_band, Decision, DecisiveScore`. `h_SROCC` and `h_Z-RMSE` are
//!    the Meng-Rosenthal-Rubin paired-correlation z-statistics
//!    against H0: "A and B are equally correlated with MOS".
//!
//! 3. Aggregate decisive verdict across all (band, corpus) decisions
//!    — counts of `ADecisivelyBeatsB / BDecisivelyBeatsA /
//!    PromisingNotDecisive / Tied / Noisy`. The "winner" is whichever
//!    bake has more decisive-band wins.
//!
//! 4. Optional JSON output (`--json results.json`) — structured form
//!    of (1) + (2) + (3) for downstream tooling (web site, dashboards).
//!
//! See `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` § A.9 lines 117-220 for
//! the formula and decisive rule this binary implements.
//!
//! Usage:
//! ```
//! bake_compare --a <bake_a.bin> --b <bake_b.bin> \
//!     [--corpora cid22,kadid,tid,konjnd,aic3] \
//!     [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features] \
//!     [--bands 10|4] \
//!     [--bootstrap-resamples 1000] \
//!     [--output report.md] \
//!     [--json results.json] \
//!     [--seed 42]
//! ```

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use zenpredict::{Model, Predictor};

use zensim_validate::panel::{
    Decision, DecisiveOutcome, PanelStats, compute_panel, decisive,
};
use zensim_validate::parquet_loader;

// ============================================================================
// Corpus registry — same shape as bake_verdict so that the two
// binaries can be diffed cleanly.
// ============================================================================

#[derive(Clone, Debug)]
struct Corpus {
    name: &'static str,
    display: &'static str,
    filename: &'static str,
    enable_per_band: bool,
}

const CORPORA: &[Corpus] = &[
    Corpus {
        name: "cid22",
        display: "CID22",
        filename: "cid22_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "kadid",
        display: "KADIK10k",
        filename: "kadid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "tid",
        display: "TID2013",
        filename: "tid_features_372col_2026-05-15.parquet",
        enable_per_band: true,
    },
    Corpus {
        name: "konjnd",
        display: "KonJND-1k (full)",
        filename: "konjnd_features_372col_2026-05-15.parquet",
        enable_per_band: false,
    },
    Corpus {
        name: "aic3",
        display: "AIC-3 CTC",
        filename: "aic3_features_372col_2026-05-15.parquet",
        enable_per_band: false,
    },
];

// ============================================================================
// CLI parsing
// ============================================================================

struct Args {
    a: PathBuf,
    b: PathBuf,
    corpora: Vec<&'static Corpus>,
    output: Option<PathBuf>,
    json: Option<PathBuf>,
    features_root: PathBuf,
    bands: u32,
    bootstrap_resamples: usize,
    seed: u64,
}

fn print_usage() {
    eprintln!(
        "bake_compare — canonical A vs B decisive comparison (§ A.9)\n\
\n\
USAGE:\n\
    bake_compare --a <bake_a.bin> --b <bake_b.bin>\n\
                 [--corpora cid22,kadid,tid,konjnd,aic3]\n\
                 [--bands 10|4]\n\
                 [--bootstrap-resamples 1000]\n\
                 [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features]\n\
                 [--output <path.md>]\n\
                 [--json <path.json>]\n\
                 [--seed 42]\n\
\n\
DEFAULTS:\n\
    --corpora             all 5 (cid22,kadid,tid,konjnd,aic3)\n\
    --bands               10 (B0..B9 width-10 grid)\n\
    --bootstrap-resamples 1000 (per § A.9 step 4)\n\
    --features-root       /mnt/v/zen/zensim-training/2026-05-15-full-features\n\
    --seed                42\n\
    --output              stdout\n\
\n\
EXAMPLES:\n\
    # Decide whether bake A is the next ship vs current ship B:\n\
    bake_compare --a v22_mix_LARGE_iwssim.bin --b v22_mix_konjnd.bin --output report.md\n\
\n\
    # Smoke test, faster:\n\
    bake_compare --a A.bin --b B.bin --corpora cid22 --bootstrap-resamples 200\n"
    );
}

fn parse_corpora_arg(arg: &str) -> Result<Vec<&'static Corpus>, String> {
    let mut out: Vec<&'static Corpus> = Vec::new();
    for name in arg.split(',') {
        let key = name.trim().to_lowercase();
        let found = CORPORA.iter().find(|c| c.name == key);
        match found {
            Some(c) => {
                if !out.iter().any(|existing| existing.name == c.name) {
                    out.push(c);
                }
            }
            None => {
                return Err(format!(
                    "unknown corpus {key:?} — known: {}",
                    CORPORA
                        .iter()
                        .map(|c| c.name)
                        .collect::<Vec<_>>()
                        .join(",")
                ));
            }
        }
    }
    Ok(out)
}

fn parse_args() -> Result<Args, String> {
    let mut a: Option<PathBuf> = None;
    let mut b: Option<PathBuf> = None;
    let mut corpora: Option<Vec<&'static Corpus>> = None;
    let mut output: Option<PathBuf> = None;
    let mut json: Option<PathBuf> = None;
    let mut features_root: PathBuf =
        PathBuf::from("/mnt/v/zen/zensim-training/2026-05-15-full-features");
    let mut bands: u32 = 10;
    let mut bootstrap_resamples: usize = 1000;
    let mut seed: u64 = 42;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--a" => {
                let v = args.next().ok_or("--a requires <path>")?;
                a = Some(PathBuf::from(v));
            }
            "--b" => {
                let v = args.next().ok_or("--b requires <path>")?;
                b = Some(PathBuf::from(v));
            }
            "--corpora" => {
                let v = args.next().ok_or("--corpora requires comma list")?;
                corpora = Some(parse_corpora_arg(&v)?);
            }
            "--output" => {
                let v = args.next().ok_or("--output requires <path>")?;
                output = Some(PathBuf::from(v));
            }
            "--json" => {
                let v = args.next().ok_or("--json requires <path>")?;
                json = Some(PathBuf::from(v));
            }
            "--features-root" => {
                let v = args.next().ok_or("--features-root requires <path>")?;
                features_root = PathBuf::from(v);
            }
            "--bands" => {
                let v = args.next().ok_or("--bands requires 10|4")?;
                bands = v
                    .parse()
                    .map_err(|_| "--bands must be 10 or 4".to_string())?;
                if bands != 10 && bands != 4 {
                    return Err("--bands must be 10 or 4".to_string());
                }
            }
            "--bootstrap-resamples" => {
                let v = args.next().ok_or("--bootstrap-resamples requires N")?;
                bootstrap_resamples = v
                    .parse()
                    .map_err(|_| "--bootstrap-resamples must be a positive integer".to_string())?;
            }
            "--seed" => {
                let v = args.next().ok_or("--seed requires u64")?;
                seed = v
                    .parse()
                    .map_err(|_| "--seed must be a u64".to_string())?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => {
                return Err(format!("unknown arg: {other}"));
            }
        }
    }
    let a = a.ok_or("--a is required (path to bake A)")?;
    let b = b.ok_or("--b is required (path to bake B)")?;
    let corpora = corpora.unwrap_or_else(|| CORPORA.iter().collect());
    Ok(Args {
        a,
        b,
        corpora,
        output,
        json,
        features_root,
        bands,
        bootstrap_resamples,
        seed,
    })
}

// ============================================================================
// Bake loading + scoring
// ============================================================================

struct LoadedBake {
    bytes: Vec<u8>,
    label: String,
}

fn load_bake(path: &Path, label: &str) -> Result<LoadedBake, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    // Sanity-parse: build a Model once so we surface header errors
    // up here instead of inside the per-corpus loop.
    let _ = Model::from_bytes(&bytes)
        .map_err(|e| format!("parse ZNPR bake at {}: {e:?}", path.display()))?;
    Ok(LoadedBake {
        bytes,
        label: label.to_string(),
    })
}

fn score_corpus(
    bake: &LoadedBake,
    feature_rows: &[Vec<f64>],
) -> Result<Vec<f64>, String> {
    // Construct a fresh Model + Predictor per corpus. Predictor's
    // scratch buffers are tied to the lifetime of Model, so we
    // cannot share across the threads — but a single-threaded
    // pass through MLP forward is already ~5 µs/row at h=128, so
    // wall time is bottlenecked on the bootstrap CI not the score
    // path.
    let model = Model::from_bytes(&bake.bytes)
        .map_err(|e| format!("parse bake {} during scoring: {e:?}", bake.label))?;
    let has_transforms = model.has_nontrivial_feature_transforms();
    let n_inputs = model.n_inputs();
    // EX-2 pool-head dispatch: extract the `zentrain.pool_head_reducer`
    // metadata so we can apply the [μ,σ,max,p_norm]→4→1 reducer to
    // the hidden-vector output instead of taking `out[0]`. Matches
    // the runtime dispatch in `zensim::metric::apply_mlp_scoring`.
    let pool_head_reducer: Option<([f32; 4], f32, f32)> = {
        let md = model.metadata();
        md.get("zentrain.pool_head_reducer").and_then(|entry| {
            let v = entry.value;
            if v.len() != 24 {
                None
            } else {
                let mut buf = [0f32; 6];
                for (i, chunk) in v.chunks_exact(4).take(6).enumerate() {
                    buf[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
                Some(([buf[0], buf[1], buf[2], buf[3]], buf[4], buf[5]))
            }
        })
    };
    // EX-2 follow-up: hybrid pool+rank head dispatch. Hybrid takes
    // priority — pool-head dispatch is skipped when hybrid metadata
    // is present.
    let hybrid_head: Option<(Vec<f32>, [f32; 4], f32, f32, f32, f32)> = {
        let md = model.metadata();
        md.get("zentrain.hybrid_head").and_then(|entry| {
            let n_hidden = model.n_outputs();
            let expected = (n_hidden + 8) * 4;
            if entry.value.len() != expected {
                None
            } else {
                let mut floats = Vec::with_capacity(n_hidden + 8);
                for chunk in entry.value.chunks_exact(4) {
                    floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
                let rank_w = floats[..n_hidden].to_vec();
                let reducer_w = [
                    floats[n_hidden + 2],
                    floats[n_hidden + 3],
                    floats[n_hidden + 4],
                    floats[n_hidden + 5],
                ];
                Some((
                    rank_w,
                    reducer_w,
                    floats[n_hidden],     // rank_b
                    floats[n_hidden + 1], // α_logit
                    floats[n_hidden + 6], // reducer_b
                    floats[n_hidden + 7], // p_norm
                ))
            }
        })
    };
    let pool_head_reducer = if hybrid_head.is_some() {
        None
    } else {
        pool_head_reducer
    };
    let mut predictor = Predictor::new(&model);
    let mut scratch = vec![0.0f32; n_inputs];
    let scores: Vec<f64> = feature_rows
        .iter()
        .map(|row| {
            let take = n_inputs.min(row.len());
            for i in 0..take {
                scratch[i] = row[i] as f32;
            }
            for f in scratch[take..].iter_mut() {
                *f = 0.0;
            }
            let result = if has_transforms {
                predictor.predict_transformed(&scratch)
            } else {
                predictor.predict(&scratch)
            };
            match result {
                Ok(out) => {
                    if let Some((rank_w, reducer_w, rank_b, alpha_logit, reducer_b, p_norm)) =
                        hybrid_head.as_ref()
                    {
                        let n = out.len() as f64;
                        if n <= 0.0 {
                            return f64::NAN;
                        }
                        let mut y_rank = *rank_b as f64;
                        let mut sum = 0.0f64;
                        let mut max_v = f64::NEG_INFINITY;
                        let mut sum_p = 0.0f64;
                        let p = *p_norm as f64;
                        for (j, &h) in out.iter().enumerate() {
                            let hf = h as f64;
                            y_rank += hf * rank_w[j] as f64;
                            sum += hf;
                            if hf > max_v {
                                max_v = hf;
                            }
                            sum_p += hf.abs().powf(p);
                        }
                        let mu = sum / n;
                        let mut var = 0.0f64;
                        for &h in out.iter() {
                            let d = h as f64 - mu;
                            var += d * d;
                        }
                        let sigma = (var / n).sqrt().max(0.0026);
                        let p_norm_stat = (sum_p / n).powf(1.0 / p);
                        let y_pool = mu * reducer_w[0] as f64
                            + sigma * reducer_w[1] as f64
                            + max_v * reducer_w[2] as f64
                            + p_norm_stat * reducer_w[3] as f64
                            + *reducer_b as f64;
                        let alpha = {
                            let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
                            1.0 / (1.0 + (-xc).exp())
                        };
                        return alpha * y_rank + (1.0 - alpha) * y_pool;
                    }
                    if let Some((rw, rb, p_norm)) = pool_head_reducer.as_ref() {
                        let n = out.len() as f64;
                        if n <= 0.0 {
                            return f64::NAN;
                        }
                        let mut sum = 0.0f64;
                        let mut max_v = f64::NEG_INFINITY;
                        let mut sum_p = 0.0f64;
                        let p = *p_norm as f64;
                        for &h in out.iter() {
                            let hf = h as f64;
                            sum += hf;
                            if hf > max_v {
                                max_v = hf;
                            }
                            sum_p += hf.abs().powf(p);
                        }
                        let mu = sum / n;
                        let mut var = 0.0f64;
                        for &h in out.iter() {
                            let d = h as f64 - mu;
                            var += d * d;
                        }
                        let sigma = (var / n).sqrt().max(0.0026);
                        let p_norm_stat = (sum_p / n).powf(1.0 / p);
                        mu * rw[0] as f64
                            + sigma * rw[1] as f64
                            + max_v * rw[2] as f64
                            + p_norm_stat * rw[3] as f64
                            + *rb as f64
                    } else {
                        out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
                    }
                }
                Err(_) => f64::NAN,
            }
        })
        .collect();
    Ok(scores)
}

// ============================================================================
// Per-band slicing
// ============================================================================

#[derive(Clone)]
struct BandSlice {
    label: String,
    range_label: String,
    indices: Vec<usize>,
}

fn make_bands_10(humans: &[f64]) -> Vec<BandSlice> {
    let mut out = Vec::with_capacity(10);
    for band_idx in 0..10 {
        let lo = band_idx as f64 * 0.10;
        let hi = lo + 0.10;
        let label = format!("B{band_idx}");
        let range_label = if band_idx == 9 {
            format!("[{lo:.2}, 1.00]")
        } else {
            format!("[{lo:.2}, {hi:.2})")
        };
        let indices: Vec<usize> = humans
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
        out.push(BandSlice {
            label,
            range_label,
            indices,
        });
    }
    out
}

/// Legacy CID22 4-band cut from the 2023 paper Table 5: B0 < 50,
/// B1 50-65, B2 65-90, B3 ≥ 90 — on the MCOS (0-100) scale.
/// `human_score` in the parquets is normalized to [0, 1] so we
/// divide the cutoffs by 100.
fn make_bands_4(humans: &[f64]) -> Vec<BandSlice> {
    let cuts: [(f64, f64, &str); 4] = [
        (0.0, 0.50, "B0_<50"),
        (0.50, 0.65, "B1_50-65"),
        (0.65, 0.90, "B2_65-90"),
        (0.90, 1.01, "B3_>=90"),
    ];
    cuts.iter()
        .map(|&(lo, hi, label)| {
            let range_label = format!("[{lo:.2}, {hi:.2})");
            let indices: Vec<usize> = humans
                .iter()
                .enumerate()
                .filter_map(|(i, &h)| (h >= lo && h < hi).then_some(i))
                .collect();
            BandSlice {
                label: label.to_string(),
                range_label,
                indices,
            }
        })
        .collect()
}

// ============================================================================
// Markdown emit helpers
// ============================================================================

fn panel_row(label: &str, p: &PanelStats) -> String {
    format!(
        "| {label} | {n} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |\n",
        p.srocc,
        p.plcc,
        p.krocc,
        p.or_ratio,
        p.pwrc,
        p.z_rmse,
        n = p.n
    )
}

fn decision_emoji(d: Decision) -> &'static str {
    match d {
        Decision::ADecisivelyBeatsB => "A>>B",
        Decision::BDecisivelyBeatsA => "B>>A",
        Decision::PromisingNotDecisive => "promising",
        Decision::Tied => "tied",
        Decision::Noisy => "noisy",
    }
}

// ============================================================================
// Per-corpus pipeline
// ============================================================================

struct CorpusResult {
    name: &'static str,
    display: &'static str,
    enable_per_band: bool,
    n_total: usize,
    aggregate: DecisiveOutcome,
    per_band: Vec<(BandSlice, DecisiveOutcome)>,
    body: String,
}

fn render_corpus(
    corpus: &Corpus,
    features_root: &Path,
    bake_a: &LoadedBake,
    bake_b: &LoadedBake,
    bands_mode: u32,
    bootstrap_resamples: usize,
    seed: u64,
) -> Result<CorpusResult, String> {
    let path = features_root.join(corpus.filename);
    let g = parquet_loader::load_parquet(&path, corpus.display, "human_score", 1.0)
        .map_err(|e| format!("load {} parquet: {e}", corpus.display))?;
    let humans = g.human_scores.clone();

    eprintln!(
        "  {}: scoring A on {} rows...",
        corpus.display,
        g.feature_rows.len()
    );
    let scores_a = score_corpus(bake_a, &g.feature_rows)?;
    eprintln!("  {}: scoring B...", corpus.display);
    let scores_b = score_corpus(bake_b, &g.feature_rows)?;
    eprintln!(
        "  {}: bootstrap CI ({} resamples) + decisive rule...",
        corpus.display, bootstrap_resamples
    );

    let aggregate = decisive(&scores_a, &scores_b, &humans, bootstrap_resamples, seed);

    let mut body = String::new();
    body.push_str(&format!("\n## {} (n={})\n\n", corpus.display, scores_a.len()));

    // Aggregate panel for both bakes.
    body.push_str("### Aggregate Mohammadi panel — A vs B\n\n");
    body.push_str("| Bake | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |\n");
    body.push_str("|---|--:|---:|---:|---:|---:|---:|---:|\n");
    body.push_str(&panel_row(&format!("A: {}", bake_a.label), &aggregate.panel_a));
    body.push_str(&panel_row(&format!("B: {}", bake_b.label), &aggregate.panel_b));
    body.push('\n');

    // Aggregate MRR / decisive row.
    body.push_str("### Aggregate MRR + decisive rule\n\n");
    body.push_str(
        "| n_band | r_AB | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |\n",
    );
    body.push_str(
        "|--:|---:|---:|---:|---:|---:|---:|--:|--:|---:|---|\n",
    );
    body.push_str(&format!(
        "| {} | {:.4} | {:.3} | {:.4} | {:.3} | {:.4} | {:+.4} | {} | {} | {:+.3} | {} |\n",
        aggregate.n_band,
        aggregate.r_ab,
        aggregate.h_srocc,
        aggregate.p_srocc,
        aggregate.h_z_rmse,
        aggregate.p_z_rmse,
        aggregate.pwrc_diff,
        aggregate.agreement_a,
        aggregate.agreement_b,
        aggregate.decisive_score,
        decision_emoji(aggregate.decision),
    ));
    body.push('\n');
    body.push_str(
        "_DecScore cutoff for decisive: |DecScore| > 7.84. \
        h_SROCC and h_Z-RMSE are Meng-Rosenthal-Rubin z-statistics; \
        |h| > 1.96 ⇒ p < 0.05 in the named bake's favor. agree_A / agree_B \
        count panel stats (of 6) whose bootstrap CI excludes 0 in that \
        bake's favor; the rule needs ≥4 in the winner's favor._\n",
    );

    // Per-band breakdown.
    let mut per_band: Vec<(BandSlice, DecisiveOutcome)> = Vec::new();
    if corpus.enable_per_band {
        let bands = if bands_mode == 4 {
            make_bands_4(&humans)
        } else {
            make_bands_10(&humans)
        };
        body.push_str(&format!(
            "\n### {} {}-band per-band panel + decisive rule\n\n",
            corpus.display, bands_mode
        ));
        body.push_str("**A's panel:**\n\n");
        body.push_str("| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |\n");
        body.push_str("|---|---|--:|---:|---:|---:|---:|---:|---:|\n");
        let mut per_band_a_panels: Vec<PanelStats> = Vec::with_capacity(bands.len());
        let mut per_band_b_panels: Vec<PanelStats> = Vec::with_capacity(bands.len());
        for band in &bands {
            if band.indices.len() < 4 {
                body.push_str(&format!(
                    "| {} | {} | {} | n/a | n/a | n/a | n/a | n/a | n/a |\n",
                    band.label,
                    band.range_label,
                    band.indices.len()
                ));
                per_band_a_panels.push(PanelStats::default());
                continue;
            }
            let h_b: Vec<f64> = band.indices.iter().map(|&i| humans[i]).collect();
            let s_a: Vec<f64> = band.indices.iter().map(|&i| scores_a[i]).collect();
            let p = compute_panel(&s_a, &h_b);
            per_band_a_panels.push(p);
            let noisy = if band.indices.len() < 30 { " ⚠" } else { "" };
            body.push_str(&format!(
                "| {}{noisy} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |\n",
                band.label,
                band.range_label,
                band.indices.len(),
                p.srocc,
                p.plcc,
                p.krocc,
                p.or_ratio,
                p.pwrc,
                p.z_rmse
            ));
        }

        body.push_str("\n**B's panel:**\n\n");
        body.push_str("| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |\n");
        body.push_str("|---|---|--:|---:|---:|---:|---:|---:|---:|\n");
        for band in &bands {
            if band.indices.len() < 4 {
                body.push_str(&format!(
                    "| {} | {} | {} | n/a | n/a | n/a | n/a | n/a | n/a |\n",
                    band.label,
                    band.range_label,
                    band.indices.len()
                ));
                per_band_b_panels.push(PanelStats::default());
                continue;
            }
            let h_b: Vec<f64> = band.indices.iter().map(|&i| humans[i]).collect();
            let s_b: Vec<f64> = band.indices.iter().map(|&i| scores_b[i]).collect();
            let p = compute_panel(&s_b, &h_b);
            per_band_b_panels.push(p);
            let noisy = if band.indices.len() < 30 { " ⚠" } else { "" };
            body.push_str(&format!(
                "| {}{noisy} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |\n",
                band.label,
                band.range_label,
                band.indices.len(),
                p.srocc,
                p.plcc,
                p.krocc,
                p.or_ratio,
                p.pwrc,
                p.z_rmse
            ));
        }

        // Decisive MRR row per band.
        body.push_str("\n**Per-band MRR + decisive rule (the ship-decision table):**\n\n");
        body.push_str(
            "| Band | n | h_SROCC | p_SROCC | h_Z-RMSE | p_Z-RMSE | PWRC_diff | agree_A | agree_B | DecScore | Decision |\n",
        );
        body.push_str(
            "|---|--:|---:|---:|---:|---:|---:|--:|--:|---:|---|\n",
        );
        for band in &bands {
            if band.indices.len() < 4 {
                body.push_str(&format!(
                    "| {} | {} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | noisy |\n",
                    band.label,
                    band.indices.len()
                ));
                continue;
            }
            let h_b: Vec<f64> = band.indices.iter().map(|&i| humans[i]).collect();
            let s_a: Vec<f64> = band.indices.iter().map(|&i| scores_a[i]).collect();
            let s_b: Vec<f64> = band.indices.iter().map(|&i| scores_b[i]).collect();
            let out =
                decisive(&s_a, &s_b, &h_b, bootstrap_resamples, seed ^ (band.label.len() as u64));
            let noisy = if band.indices.len() < 30 { " ⚠" } else { "" };
            body.push_str(&format!(
                "| {}{noisy} | {} | {:.3} | {:.4} | {:.3} | {:.4} | {:+.4} | {} | {} | {:+.3} | {} |\n",
                band.label,
                band.indices.len(),
                out.h_srocc,
                out.p_srocc,
                out.h_z_rmse,
                out.p_z_rmse,
                out.pwrc_diff,
                out.agreement_a,
                out.agreement_b,
                out.decisive_score,
                decision_emoji(out.decision),
            ));
            per_band.push((band.clone(), out));
        }
        body.push('\n');
        body.push_str(
            "_⚠ marks bands with n < 30 — decisive rule mandates n ≥ 30; bands below \
            that are emitted as `Noisy` and do NOT contribute to the aggregate verdict._\n",
        );
    } else {
        body.push('\n');
        body.push_str(&format!(
            "_Per-band breakdown skipped for {} — corpus uses a JND step grid (AIC-3) or \
            raw threshold scale (KonJND) that doesn't partition cleanly into the \
            CID22/KADID/TID-style 0..1 normalized bands. Aggregate decisive verdict above \
            is the load-bearing read on this corpus._\n",
            corpus.display
        ));
    }

    Ok(CorpusResult {
        name: corpus.name,
        display: corpus.display,
        enable_per_band: corpus.enable_per_band,
        n_total: scores_a.len(),
        aggregate,
        per_band,
        body,
    })
}

// ============================================================================
// JSON serialization (hand-written, no serde dep needed)
// ============================================================================

fn esc_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn finite_or_null(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "null".to_string()
    }
}

fn panel_to_json(p: &PanelStats) -> String {
    format!(
        "{{\"n\":{n},\"srocc\":{s},\"plcc\":{p_},\"krocc\":{k},\"or\":{o},\"pwrc\":{pw},\"z_rmse\":{z}}}",
        n = p.n,
        s = finite_or_null(p.srocc),
        p_ = finite_or_null(p.plcc),
        k = finite_or_null(p.krocc),
        o = finite_or_null(p.or_ratio),
        pw = finite_or_null(p.pwrc),
        z = finite_or_null(p.z_rmse),
    )
}

fn outcome_to_json(o: &DecisiveOutcome) -> String {
    let ci_pairs: Vec<String> = o
        .ci_delta
        .iter()
        .map(|(lo, hi)| {
            format!(
                "[{},{}]",
                finite_or_null(*lo),
                finite_or_null(*hi)
            )
        })
        .collect();
    format!(
        "{{\"n_band\":{n},\"panel_a\":{a},\"panel_b\":{b},\"r_ab\":{r},\"h_srocc\":{hs},\"p_srocc\":{ps},\"h_z_rmse\":{hz},\"p_z_rmse\":{pz},\"pwrc_diff\":{pd},\"ci_delta\":[{ci}],\"agreement_a\":{aa},\"agreement_b\":{ab},\"decisive_score\":{ds},\"decision\":{dec}}}",
        n = o.n_band,
        a = panel_to_json(&o.panel_a),
        b = panel_to_json(&o.panel_b),
        r = finite_or_null(o.r_ab),
        hs = finite_or_null(o.h_srocc),
        ps = finite_or_null(o.p_srocc),
        hz = finite_or_null(o.h_z_rmse),
        pz = finite_or_null(o.p_z_rmse),
        pd = finite_or_null(o.pwrc_diff),
        ci = ci_pairs.join(","),
        aa = o.agreement_a,
        ab = o.agreement_b,
        ds = finite_or_null(o.decisive_score),
        dec = esc_json_str(o.decision.as_str()),
    )
}

fn build_json_report(
    args: &Args,
    bake_a: &LoadedBake,
    bake_b: &LoadedBake,
    results: &[CorpusResult],
    counts: &AggregateCounts,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"a_bake\": {},\n", esc_json_str(&args.a.display().to_string())));
    out.push_str(&format!("  \"a_label\": {},\n", esc_json_str(&bake_a.label)));
    out.push_str(&format!("  \"b_bake\": {},\n", esc_json_str(&args.b.display().to_string())));
    out.push_str(&format!("  \"b_label\": {},\n", esc_json_str(&bake_b.label)));
    out.push_str(&format!("  \"bands_mode\": {},\n", args.bands));
    out.push_str(&format!(
        "  \"bootstrap_resamples\": {},\n  \"seed\": {},\n",
        args.bootstrap_resamples, args.seed
    ));
    out.push_str("  \"corpora\": [\n");
    for (i, r) in results.iter().enumerate() {
        out.push_str("    {\n");
        out.push_str(&format!("      \"name\": {},\n", esc_json_str(r.name)));
        out.push_str(&format!("      \"display\": {},\n", esc_json_str(r.display)));
        out.push_str(&format!("      \"n_total\": {},\n", r.n_total));
        out.push_str(&format!("      \"enable_per_band\": {},\n", r.enable_per_band));
        out.push_str(&format!("      \"aggregate\": {},\n", outcome_to_json(&r.aggregate)));
        out.push_str("      \"per_band\": [\n");
        for (j, (band, outcome)) in r.per_band.iter().enumerate() {
            out.push_str("        {\n");
            out.push_str(&format!("          \"label\": {},\n", esc_json_str(&band.label)));
            out.push_str(&format!("          \"range\": {},\n", esc_json_str(&band.range_label)));
            out.push_str(&format!("          \"outcome\": {}\n", outcome_to_json(outcome)));
            out.push_str(if j + 1 == r.per_band.len() {
                "        }\n"
            } else {
                "        },\n"
            });
        }
        out.push_str("      ]\n");
        out.push_str(if i + 1 == results.len() {
            "    }\n"
        } else {
            "    },\n"
        });
    }
    out.push_str("  ],\n");
    out.push_str(&format!(
        "  \"aggregate_counts\": {{\"a_decisively_beats_b\": {}, \"b_decisively_beats_a\": {}, \"promising_not_decisive\": {}, \"tied\": {}, \"noisy\": {}}},\n",
        counts.a_wins, counts.b_wins, counts.promising, counts.tied, counts.noisy
    ));
    out.push_str(&format!(
        "  \"overall_winner\": {}\n",
        esc_json_str(counts.overall_winner_label())
    ));
    out.push_str("}\n");
    out
}

// ============================================================================
// Aggregate counting
// ============================================================================

struct AggregateCounts {
    a_wins: usize,
    b_wins: usize,
    promising: usize,
    tied: usize,
    noisy: usize,
}

impl AggregateCounts {
    fn new() -> Self {
        Self {
            a_wins: 0,
            b_wins: 0,
            promising: 0,
            tied: 0,
            noisy: 0,
        }
    }

    fn record(&mut self, d: Decision) {
        match d {
            Decision::ADecisivelyBeatsB => self.a_wins += 1,
            Decision::BDecisivelyBeatsA => self.b_wins += 1,
            Decision::PromisingNotDecisive => self.promising += 1,
            Decision::Tied => self.tied += 1,
            Decision::Noisy => self.noisy += 1,
        }
    }

    fn overall_winner_label(&self) -> &'static str {
        if self.a_wins > self.b_wins {
            "A"
        } else if self.b_wins > self.a_wins {
            "B"
        } else if self.a_wins == 0 && self.b_wins == 0 {
            "no_decisive_evidence"
        } else {
            "tie"
        }
    }
}

// ============================================================================
// Main
// ============================================================================

fn bake_label_from_path(p: &Path) -> String {
    p.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("bake")
        .to_string()
}

fn main() -> ExitCode {
    let t0 = Instant::now();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bake_compare: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "bake_compare — A={}  B={}  corpora={}  bands={}  bootstrap={}",
        args.a.display(),
        args.b.display(),
        args.corpora
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(","),
        args.bands,
        args.bootstrap_resamples,
    );

    let bake_a = match load_bake(&args.a, &bake_label_from_path(&args.a)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake_compare: {e}");
            return ExitCode::from(1);
        }
    };
    let bake_b = match load_bake(&args.b, &bake_label_from_path(&args.b)) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake_compare: {e}");
            return ExitCode::from(1);
        }
    };

    let mut buf = String::new();
    buf.push_str("# bake_compare — decisive A vs B verdict (§ A.9)\n\n");
    buf.push_str(&format!("- **A**: `{}` (label: `{}`)\n", args.a.display(), bake_a.label));
    buf.push_str(&format!("- **B**: `{}` (label: `{}`)\n", args.b.display(), bake_b.label));
    buf.push_str(&format!("- Feature parquets: `{}`\n", args.features_root.display()));
    buf.push_str(&format!(
        "- Bands: `{}-band`  Bootstrap resamples: `{}`  Seed: `{}`\n",
        args.bands, args.bootstrap_resamples, args.seed
    ));
    buf.push_str("\n");
    buf.push_str(
        "Implements § A.9 of PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md. Decisive rule \
        (verbatim): `DECISIVE_FOR_BAND(A beats B) ⟺ n_band ≥ 30 ∧ |h_SROCC| > 1.96 ∧ \
        |h_Z-RMSE| > 1.96 ∧ PWRC_A > PWRC_B ∧ ≥4 of 6 panel stats favor A in 95% \
        bootstrap CI`. Otherwise: `PromisingNotDecisive`. n < 30: `Noisy`.\n\n",
    );

    let mut results: Vec<CorpusResult> = Vec::new();
    let mut counts = AggregateCounts::new();
    for corpus in &args.corpora {
        eprintln!("== {} ==", corpus.display);
        match render_corpus(
            corpus,
            &args.features_root,
            &bake_a,
            &bake_b,
            args.bands,
            args.bootstrap_resamples,
            args.seed,
        ) {
            Ok(r) => {
                // Aggregate counts: aggregate-level decision counts once
                // per corpus, AND each per-band decision (n ≥ 30 only).
                counts.record(r.aggregate.decision);
                for (_, outcome) in &r.per_band {
                    counts.record(outcome.decision);
                }
                results.push(r);
            }
            Err(e) => {
                eprintln!("bake_compare: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // Top-of-doc summary table — aggregate panel + decisive verdict
    // per corpus, one row each.
    buf.push_str("## Cross-corpus aggregate summary\n\n");
    buf.push_str(
        "| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Aggregate |\n",
    );
    buf.push_str(
        "|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n",
    );
    for r in &results {
        let a = &r.aggregate.panel_a;
        let b = &r.aggregate.panel_b;
        buf.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.3} | {:.3} | {:.4} | {:.4} | {:.3} | {:.3} | {:+.3} | {} |\n",
            r.display,
            r.n_total,
            a.srocc,
            b.srocc,
            a.z_rmse,
            b.z_rmse,
            a.pwrc,
            b.pwrc,
            r.aggregate.h_srocc,
            r.aggregate.h_z_rmse,
            r.aggregate.decisive_score,
            decision_emoji(r.aggregate.decision),
        ));
    }
    buf.push('\n');

    // Decisive-band totals across all (corpus, band) cells.
    buf.push_str("## Decisive-band totals across all (corpus × band) cells\n\n");
    buf.push_str(&format!(
        "- **ADecisivelyBeatsB**: {} cells\n\
- **BDecisivelyBeatsA**: {} cells\n\
- **PromisingNotDecisive**: {} cells\n\
- **Tied**: {} cells\n\
- **Noisy** (n < 30, no decision): {} cells\n\n",
        counts.a_wins, counts.b_wins, counts.promising, counts.tied, counts.noisy
    ));
    buf.push_str(&format!(
        "**Overall winner across decisive cells: `{}`** ({} A wins vs {} B wins)\n",
        counts.overall_winner_label(),
        counts.a_wins,
        counts.b_wins
    ));
    if counts.a_wins == 0 && counts.b_wins == 0 {
        buf.push_str("\n_No decisive cells in either direction — neither bake meets the \
        4-condition § A.9 rule on any (corpus × band) slice. The result is `promising` \
        / `tied` / `noisy` across the board. Treat as a ship-tie until a follow-up \
        comparison breaks the deadlock (e.g., on a held-out corpus, with a larger n)._\n");
    }

    // Per-corpus details.
    for r in &results {
        buf.push_str(&r.body);
    }

    let elapsed = t0.elapsed();
    buf.push_str(&format!(
        "\n---\nWall time: {:.2}s ({} pair rows scored × 2 bakes across {} corpora; \
        {} bootstrap resamples × bands).\n",
        elapsed.as_secs_f64(),
        results.iter().map(|r| r.n_total).sum::<usize>(),
        results.len(),
        args.bootstrap_resamples,
    ));

    // Write report.
    if let Some(out_path) = &args.output {
        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match File::create(out_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(buf.as_bytes()) {
                    eprintln!("bake_compare: failed to write {}: {e}", out_path.display());
                    return ExitCode::from(1);
                }
                eprintln!("wrote markdown report to {}", out_path.display());
            }
            Err(e) => {
                eprintln!("bake_compare: failed to create {}: {e}", out_path.display());
                return ExitCode::from(1);
            }
        }
    } else {
        print!("{buf}");
    }

    // Write JSON.
    if let Some(json_path) = &args.json {
        if let Some(parent) = json_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let json_str = build_json_report(&args, &bake_a, &bake_b, &results, &counts);
        match File::create(json_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(json_str.as_bytes()) {
                    eprintln!("bake_compare: failed to write {}: {e}", json_path.display());
                    return ExitCode::from(1);
                }
                eprintln!("wrote JSON report to {}", json_path.display());
            }
            Err(e) => {
                eprintln!("bake_compare: failed to create {}: {e}", json_path.display());
                return ExitCode::from(1);
            }
        }
    }

    eprintln!("bake_compare: complete in {:.2}s", elapsed.as_secs_f64());
    ExitCode::SUCCESS
}
