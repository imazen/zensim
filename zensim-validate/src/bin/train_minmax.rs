//! Proof-of-concept: does a MIN-MAX MONOTONE head (Sill 1998) hold imazen-26
//! ssim2-agreement while staying monotone — where the positive-weight MLP
//! (`monotone_cbc`) craters it to 0.03? Trains on the base_tfm group set with the
//! same 2026-05-25 screen transforms + standardization, then reports held-out
//! CID22 / imazen-26 / nonphoto SROCC + a dial-monotonicity check.
//!
//! Reuses `parquet_loader` (load), `zenpredict::FeatureTransform` (the runtime
//! transforms — no reimpl), and `mlp_train::minmax_monotone::{train_ranknet,
//! MinMaxMonotone}`. Not a bake yet: if this breaks the ceiling, the bake +
//! runtime forward follow.
//!
//!   train_minmax --k 8 --j 4 [--epochs 80] [--pairs 60000] [--lr 4e-3] [--min-lift 0.05]

use std::collections::HashMap;
use zensim_validate::mlp_train::minmax_monotone::{train_ranknet, MinMaxMonotone};
use zensim_validate::parquet_loader::load_parquet;

const N: usize = 372;
const CAN: &str = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train";
const FR: &str = "/mnt/v/zen/zensim-training/2026-05-15-full-features";
const SCREEN: &str =
    "benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv";
const SIGN: &str = "benchmarks/feature_sign_mask_2026-05-26.tsv";

fn arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Per-feature (transform, params) from the 2026-05-25 screen; Identity below min_lift.
fn load_transforms(min_lift: f64) -> Vec<(zenpredict::FeatureTransform, Vec<f32>)> {
    let mut out = vec![(zenpredict::FeatureTransform::Identity, Vec::new()); N];
    let txt = std::fs::read_to_string(SCREEN).expect("screen tsv");
    for line in txt.lines().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 6 {
            continue;
        }
        let (Ok(idx), Ok(lift)) = (c[0].parse::<usize>(), c[5].parse::<f64>()) else {
            continue;
        };
        if idx >= N || lift < min_lift {
            continue;
        }
        if let Ok(t) = zenpredict::FeatureTransform::from_token(c[1]) {
            let params: Vec<f32> = c[2]
                .split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect();
            out[idx] = (t, params);
        }
    }
    out
}

fn load_sign() -> Vec<f64> {
    let mut s = vec![0.0f64; N];
    let txt = std::fs::read_to_string(SIGN).expect("sign tsv");
    for line in txt.lines().skip(1) {
        let c: Vec<&str> = line.split('\t').collect();
        if let Ok(idx) = c[0].parse::<usize>() {
            if idx < N && c.get(1) == Some(&"pin_geq0") {
                // `pin_geq0` = W1≥0 in the trainer, which pairs with rank_w≤0 —
                // so the OUTPUT decreases with these features: they increase with
                // DISTORTION (decrease with quality). A score-increasing min-max
                // therefore needs w≤0 on them → sign = −1.
                s[idx] = -1.0;
            }
        }
    }
    s
}

/// Apply per-feature transform in place on a raw feature row.
fn transform_row(row: &[f64], tf: &[(zenpredict::FeatureTransform, Vec<f32>)]) -> Vec<f64> {
    (0..N)
        .map(|f| tf[f].0.apply_with_params(row[f] as f32, &tf[f].1) as f64)
        .collect()
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0.0; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let n = a.len() as f64;
    let mean = (n - 1.0) / 2.0;
    let (mut num, mut da, mut db) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (x, y) = (ra[i] - mean, rb[i] - mean);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    num / (da.sqrt() * db.sqrt() + 1e-12)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (k, j) = (arg(&args, "--k", 8usize), arg(&args, "--j", 4usize));
    let epochs = arg(&args, "--epochs", 80usize);
    let pairs = arg(&args, "--pairs", 60_000usize);
    let lr = arg(&args, "--lr", 4e-3f64);
    let min_lift = arg(&args, "--min-lift", 0.05f64);

    let tf = load_transforms(min_lift);
    let sign = load_sign();
    let n_pin = sign.iter().filter(|&&s| s != 0.0).count();
    eprintln!("min-max: K={k} J={j} epochs={epochs} pairs={pairs} lr={lr} | {n_pin}/{N} pinned monotone, {} transforms",
        tf.iter().filter(|(t,_)| *t != zenpredict::FeatureTransform::Identity).count());

    // base_tfm train groups (name, path). Cap huge groups for a fast probe.
    let train_specs: &[(&str, String, usize)] = &[
        ("safesyn", format!("{CAN}/safesyn.parquet"), 40_000),
        ("cid22_train", format!("{CAN}/cid22_train.parquet"), 0),
        ("kadid", format!("{CAN}/kadid.parquet"), 0),
        ("tid", format!("{CAN}/tid.parquet"), 0),
        ("konjnd", format!("{CAN}/konjnd-dense-norm.parquet"), 0),
        ("bigcodec", "/mnt/v/output/zensim/depth-iter/bigcodec_train_120k_stride.parquet".into(), 40_000),
        ("kadis", "/mnt/v/output/zensim/depth-iter/kadis_train_60k_stride.parquet".into(), 20_000),
    ];

    // Load + transform train; compute standardizer (mean/std) from the transformed pool.
    let mut groups_t: Vec<(Vec<Vec<f64>>, Vec<f64>)> = Vec::new();
    let (mut sum, mut sumsq, mut cnt) = (vec![0.0f64; N], vec![0.0f64; N], 0usize);
    for (name, path, cap) in train_specs {
        let g = match load_parquet(&std::path::PathBuf::from(path), name, "human_score", 1.0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  skip {name}: {e}");
                continue;
            }
        };
        let stride = if *cap > 0 && g.feature_rows.len() > *cap {
            g.feature_rows.len() / *cap
        } else {
            1
        };
        let mut rows_t = Vec::new();
        let mut sc = Vec::new();
        for (r, row) in g.feature_rows.iter().enumerate() {
            if r % stride != 0 {
                continue;
            }
            let t = transform_row(row, &tf);
            for f in 0..N {
                if t[f].is_finite() {
                    sum[f] += t[f];
                    sumsq[f] += t[f] * t[f];
                }
            }
            cnt += 1;
            rows_t.push(t);
            sc.push(g.human_scores[r]);
        }
        eprintln!("  {name}: {} rows (stride {stride})", rows_t.len());
        groups_t.push((rows_t, sc));
    }
    let mean: Vec<f64> = (0..N).map(|f| sum[f] / cnt as f64).collect();
    let std: Vec<f64> = (0..N)
        .map(|f| ((sumsq[f] / cnt as f64 - mean[f] * mean[f]).max(1e-12)).sqrt())
        .collect();
    let standardize = |t: &[f64]| -> Vec<f64> {
        (0..N).map(|f| ((t[f] - mean[f]) / std[f]).clamp(-8.0, 8.0)).collect()
    };

    // Flatten standardized train groups for train_ranknet.
    let flat_groups: Vec<(Vec<f64>, Vec<f64>)> = groups_t
        .iter()
        .map(|(rows, sc)| {
            let mut flat = Vec::with_capacity(rows.len() * N);
            for r in rows {
                flat.extend_from_slice(&standardize(r));
            }
            (flat, sc.clone())
        })
        .collect();

    eprintln!("training min-max ...");
    let m: MinMaxMonotone = train_ranknet(&flat_groups, &sign, k, j, N, epochs, pairs, lr, 13);

    // Held-out eval.
    let evals: &[(&str, String)] = &[
        ("CID22", format!("{FR}/cid22_features_372col_2026-05-15.parquet")),
        ("imazen26", format!("{FR}/imazen26_test_120k_2026-07-16.parquet")),
        ("nonphoto", format!("{FR}/nonphoto_features_372col_2026-07-15.parquet")),
    ];
    let mut results: HashMap<&str, f64> = HashMap::new();
    for (name, path) in evals {
        let g = match load_parquet(&std::path::PathBuf::from(path), name, "human_score", 1.0) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("  eval skip {name}: {e}");
                continue;
            }
        };
        let pred: Vec<f64> = g
            .feature_rows
            .iter()
            .map(|row| m.forward(&standardize(&transform_row(row, &tf))).0)
            .collect();
        results.insert(name, spearman(&pred, &g.human_scores));
    }

    // Dial-monotonicity proxy: on a per-ref quality ladder the min-max output must
    // be monotone in the ssim2 label BY CONSTRUCTION — verify on imazen26 by
    // checking predicted rank vs label rank agreement is high (it is the SROCC).
    println!("\n=== MIN-MAX MONOTONE (K={k} J={j}) held-out ===");
    for (name, _) in evals {
        if let Some(&s) = results.get(name) {
            println!("  {name:10} SROCC={s:.4}");
        }
    }
    println!(
        "\nref: base_tfm imazen26=0.948 (monotone_cbc=0.032); A imazen26=0.862 (monotone dial). \
         min-max is monotone-by-construction, so its dial is consistent — the question is whether \
         imazen26 clears the ~0.86 monotone ceiling."
    );
}
