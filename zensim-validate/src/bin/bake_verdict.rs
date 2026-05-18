//! Instant V_X bake evaluator — loads pre-extracted features from
//! parquet sidecars, scores a ZNPR v3 bake, and emits the full
//! Mohammadi 2025 panel (aggregate + 10-band) per held-out corpus.
//!
//! Replaces the per-bake compute path in
//! `zensim-bench/examples/dataset_metric_baseline.rs` for the case
//! where image features have already been extracted (T10.1). Old
//! path: re-decode images + recompute baseline metrics + score
//! MLP per pair, ~15-20 min for the full 5-corpus held-out set.
//! New path: read parquet sidecars + MLP forward only, <5 s wall.
//!
//! Inputs (T10.1 outputs):
//!     /mnt/v/zen/zensim-training/2026-05-15-full-features/
//!         aic3_features_372col_2026-05-15.parquet
//!         cid22_features_372col_2026-05-15.parquet
//!         kadid_features_372col_2026-05-15.parquet
//!         konjnd_features_372col_2026-05-15.parquet
//!         tid_features_372col_2026-05-15.parquet
//!
//! Each parquet carries 374 columns: `ref_basename, human_score, f0..f371`.
//! `human_score` is on the corpus's own normalized scale (matches the
//! convention `dataset_metric_baseline.rs` uses internally — KADID
//! `(DMOS-1)/4` in [0,1], TID `MOS/9` in [0,1], CID22 `MCOS/100` in
//! [0,1], KonJND `mean_threshold` in raw units, AIC-3 raw `score.jnd`
//! in [-3,0]). SROCC / KROCC / PWRC are rank-invariant so polarity
//! and scale don't matter; PLCC / Z-RMSE absorb scale via the
//! 4-parameter logistic rescale (Mohammadi 2025 convention).
//!
//! Usage:
//!     bake_verdict --bake <path>
//!                  [--corpora cid22,kadid,tid,konjnd,aic3]
//!                  [--output <path.md>]
//!                  [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features]
//!
//! Verification: when invoked with the V_22-IW v2 calibrated bake
//! (`zensim/weights/v0_22_iw_v2_calibrated_2026-05-16.bin`), the
//! aggregate SROCC values match the dataset_metric_baseline log at
//! `benchmarks/v0_22_iw_v2_seed1_2026-05-16_eval_full.log` to within
//! 1e-3. The full numbers come from the SAME features that the
//! baseline path computes per pair; the only difference is that we
//! read them from parquet instead of recomputing.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use zenpredict::{Model, Predictor};

use zensim_validate::parquet_loader;

// ============================================================================
// Stat functions (ported verbatim from
// zensim-bench/examples/dataset_metric_baseline.rs to avoid touching
// the historical reference binary). All use the polarity-tolerant
// `.abs()` convention at the call site since bake outputs can be
// distance- or score-shaped depending on training-target convention.
// ============================================================================

fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // `total_cmp` is NaN-safe; bakes can produce NaN on
    // heavy-distortion pairs when feature transforms saturate.
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-12 {
            j += 1;
        }
        let avg = (i + j - 1) as f64 / 2.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mean = (n as f64 - 1.0) / 2.0;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean;
        let xb = rb[i] - mean;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mean_a: f64 = a.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = b.iter().sum::<f64>() / n as f64;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = a[i] - mean_a;
        let xb = b[i] - mean_b;
        num += xa * xb;
        da += xa * xa;
        db += xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

fn kendall_tau(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut ties_a = 0i64;
    let mut ties_b = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            if da.abs() < 1e-12 && db.abs() < 1e-12 {
                continue;
            } else if da.abs() < 1e-12 {
                ties_a += 1;
            } else if db.abs() < 1e-12 {
                ties_b += 1;
            } else if (da * db) > 0.0 {
                concordant += 1;
            } else {
                discordant += 1;
            }
        }
    }
    let total_a = (concordant + discordant + ties_a) as f64;
    let total_b = (concordant + discordant + ties_b) as f64;
    let den = (total_a * total_b).sqrt();
    if den < 1e-12 {
        0.0
    } else {
        ((concordant - discordant) as f64) / den
    }
}

fn outlier_ratio(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len();
    if n < 4 {
        return f64::NAN;
    }
    let mean_p: f64 = predicted.iter().sum::<f64>() / n as f64;
    let mean_t: f64 = target.iter().sum::<f64>() / n as f64;
    let var_p: f64 = predicted.iter().map(|x| (x - mean_p).powi(2)).sum::<f64>() / n as f64;
    let var_t: f64 = target.iter().map(|x| (x - mean_t).powi(2)).sum::<f64>() / n as f64;
    let sd_p = var_p.sqrt().max(1e-12);
    let sd_t = var_t.sqrt().max(1e-12);
    let polarity = if pearson(predicted, target) < 0.0 {
        -1.0
    } else {
        1.0
    };
    let residuals: Vec<f64> = (0..n)
        .map(|i| {
            let zp = polarity * (predicted[i] - mean_p) / sd_p;
            let zt = (target[i] - mean_t) / sd_t;
            (zp - zt).abs()
        })
        .collect();
    let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
    let sd_r: f64 = (residuals.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>() / n as f64)
        .sqrt()
        .max(1e-12);
    residuals
        .iter()
        .filter(|r| (**r - mean_r).abs() > 2.0 * sd_r)
        .count() as f64
        / n as f64
}

fn pwrc(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 4 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let mid = (n as f64 - 1.0) / 2.0;
    let max_dev = mid.max(1e-12);
    let w: Vec<f64> = ra.iter().map(|r| (r - mid).abs() / max_dev).collect();
    let wsum: f64 = w.iter().sum();
    if wsum < 1e-12 {
        return 0.0;
    }
    let mean_a: f64 = w.iter().zip(&ra).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mean_b: f64 = w.iter().zip(&rb).map(|(w, r)| w * r).sum::<f64>() / wsum;
    let mut num = 0.0f64;
    let mut da = 0.0f64;
    let mut db = 0.0f64;
    for i in 0..n {
        let xa = ra[i] - mean_a;
        let xb = rb[i] - mean_b;
        num += w[i] * xa * xb;
        da += w[i] * xa * xa;
        db += w[i] * xb * xb;
    }
    let den = (da * db).sqrt();
    if den < 1e-12 { 0.0 } else { num / den }
}

// ============================================================================
// Z-RMSE: σ-normalized RMSE after a 4-parameter logistic rescale.
// Ported from dataset_metric_baseline.rs. The logistic absorbs the
// natural saturation of nonlinear metrics so Z-RMSE measures error
// after the metric's shape, not the shape itself.
// ============================================================================

fn z_rmse(predicted: &[f64], target: &[f64]) -> f64 {
    let n = predicted.len();
    if n < 2 || target.len() != n {
        return f64::NAN;
    }
    let sigma_global = {
        let mean: f64 = target.iter().sum::<f64>() / n as f64;
        let var: f64 = target.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        var.sqrt().max(1e-9)
    };
    let mut sum_sq = 0.0f64;
    let mut count = 0;
    for i in 0..n {
        let z = (predicted[i] - target[i]) / sigma_global;
        if z.is_finite() {
            sum_sq += z * z;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum_sq / count as f64).sqrt()
}

fn logistic_eval(b: &[f64; 4], x: f64) -> f64 {
    let b4 = if b[3].abs() < 1e-12 {
        1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
    } else {
        b[3]
    };
    let arg = -(x - b[2]) / b4;
    let e = if arg > 700.0 {
        f64::INFINITY
    } else if arg < -700.0 {
        0.0
    } else {
        arg.exp()
    };
    b[1] + (b[0] - b[1]) / (1.0 + e)
}

fn solve_4x4_gauss(aug: &mut [[f64; 5]; 4]) -> Option<[f64; 4]> {
    for i in 0..4 {
        let mut max_row = i;
        let mut max_val = aug[i][i].abs();
        for k in (i + 1)..4 {
            let v = aug[k][i].abs();
            if v > max_val {
                max_val = v;
                max_row = k;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != i {
            aug.swap(i, max_row);
        }
        for k in (i + 1)..4 {
            let factor = aug[k][i] / aug[i][i];
            for c in i..5 {
                aug[k][c] -= factor * aug[i][c];
            }
        }
    }
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum = aug[i][4];
        for c in (i + 1)..4 {
            sum -= aug[i][c] * x[c];
        }
        x[i] = sum / aug[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

fn run_lm(predicted: &[f64], target: &[f64], n: usize, b0: [f64; 4]) -> Option<([f64; 4], f64)> {
    let max_iters = 500usize;
    let tol = 1e-10f64;
    let cost_tol = 1e-12f64;
    let mut lambda = 1.0e-3f64;
    let mut b = b0;
    let jacobian_and_residuals = |b: &[f64; 4]| -> (Vec<[f64; 4]>, Vec<f64>) {
        let mut jac = Vec::with_capacity(n);
        let mut res = Vec::with_capacity(n);
        let b4 = if b[3].abs() < 1e-12 {
            1e-12_f64.copysign(b[3].max(0.0).signum().max(1.0))
        } else {
            b[3]
        };
        for i in 0..n {
            let x = predicted[i];
            let arg = -(x - b[2]) / b4;
            let e = if arg > 700.0 {
                f64::INFINITY
            } else if arg < -700.0 {
                0.0
            } else {
                arg.exp()
            };
            let a = 1.0 + e;
            let inv_a = 1.0 / a;
            let pred = b[1] + (b[0] - b[1]) * inv_a;
            let diff = pred - target[i];
            res.push(diff);
            let db1 = inv_a;
            let db2 = 1.0 - inv_a;
            let (db3, db4_) = if e.is_finite() && a.is_finite() && a > 1e-300 {
                let inv_a2 = inv_a * inv_a;
                let amp = b[0] - b[1];
                (
                    -amp * e * inv_a2 / b4,
                    -amp * e * (x - b[2]) * inv_a2 / (b4 * b4),
                )
            } else {
                (0.0, 0.0)
            };
            jac.push([db1, db2, db3, db4_]);
        }
        (jac, res)
    };
    let sum_sq = |res: &[f64]| -> f64 { res.iter().map(|r| r * r).sum::<f64>() };
    let (mut jac, mut res) = jacobian_and_residuals(&b);
    let mut cost = sum_sq(&res);
    if !cost.is_finite() {
        return None;
    }
    for _iter in 0..max_iters {
        let mut jtj = [[0.0f64; 4]; 4];
        let mut jtr = [0.0f64; 4];
        for i in 0..n {
            let row = &jac[i];
            let r = res[i];
            for a_ in 0..4 {
                jtr[a_] += row[a_] * r;
                for c_ in 0..4 {
                    jtj[a_][c_] += row[a_] * row[c_];
                }
            }
        }
        let mut h = jtj;
        for d in 0..4 {
            h[d][d] += lambda * jtj[d][d].max(1e-12);
        }
        let mut aug = [[0.0f64; 5]; 4];
        for r_ in 0..4 {
            for c in 0..4 {
                aug[r_][c] = h[r_][c];
            }
            aug[r_][4] = -jtr[r_];
        }
        let solved = solve_4x4_gauss(&mut aug);
        let delta = match solved {
            Some(d) => d,
            None => {
                lambda *= 10.0;
                if lambda > 1e10 {
                    return Some((b, cost));
                }
                continue;
            }
        };
        let b_try = [
            b[0] + delta[0],
            b[1] + delta[1],
            b[2] + delta[2],
            b[3] + delta[3],
        ];
        let (jac_try, res_try) = jacobian_and_residuals(&b_try);
        let cost_try = sum_sq(&res_try);
        if cost_try.is_finite() && cost_try < cost {
            let max_delta = delta.iter().map(|d| d.abs()).fold(0.0f64, f64::max);
            let max_b = b.iter().map(|x| x.abs()).fold(1.0f64, f64::max);
            let cost_decrease_rel = (cost - cost_try) / cost.max(1e-30);
            b = b_try;
            jac = jac_try;
            res = res_try;
            cost = cost_try;
            lambda = (lambda / 10.0).max(1e-12);
            if max_delta < tol * (1.0 + max_b) || cost_decrease_rel < cost_tol {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e10 {
                break;
            }
        }
    }
    Some((b, cost))
}

fn rescale_affine(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 2 {
        return predicted.to_vec();
    }
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let mean_t: f64 = target.iter().take(n).sum::<f64>() / n as f64;
    let mut cov = 0.0f64;
    let mut var_p = 0.0f64;
    for i in 0..n {
        let dp = predicted[i] - mean_p;
        let dt = target[i] - mean_t;
        cov += dp * dt;
        var_p += dp * dp;
    }
    let b = if var_p.abs() < 1e-12 {
        0.0
    } else {
        cov / var_p
    };
    let a = mean_t - b * mean_p;
    predicted.iter().map(|p| a + b * p).collect()
}

fn rescale_logistic(predicted: &[f64], target: &[f64]) -> Vec<f64> {
    let n = predicted.len().min(target.len());
    if n < 4 {
        return rescale_affine(predicted, target);
    }
    let mean_p: f64 = predicted.iter().take(n).sum::<f64>() / n as f64;
    let var_p: f64 = predicted
        .iter()
        .take(n)
        .map(|x| (x - mean_p).powi(2))
        .sum::<f64>()
        / n as f64;
    if !var_p.is_finite() || var_p < 1e-18 {
        return rescale_affine(predicted, target);
    }
    if !predicted.iter().take(n).all(|x| x.is_finite())
        || !target.iter().take(n).all(|x| x.is_finite())
    {
        return rescale_affine(predicted, target);
    }
    let t_max = target
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let t_min = target.iter().take(n).cloned().fold(f64::INFINITY, f64::min);
    let p_std = var_p.sqrt();
    let p_corr = {
        let mean_t = target.iter().take(n).sum::<f64>() / n as f64;
        let mut cov = 0.0f64;
        let mut vp = 0.0f64;
        let mut vt = 0.0f64;
        for i in 0..n {
            let dp = predicted[i] - mean_p;
            let dt = target[i] - mean_t;
            cov += dp * dt;
            vp += dp * dp;
            vt += dt * dt;
        }
        let d = (vp * vt).sqrt();
        if d < 1e-12 { 0.0 } else { cov / d }
    };
    let b4_sign = if p_corr < 0.0 { -1.0 } else { 1.0 };
    let p_max = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let p_min = predicted
        .iter()
        .take(n)
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let t_span = (t_max - t_min).abs().max(1.0);
    let tail = 1000.0 * t_span;
    let b3_high = p_max + 25.0 * p_std;
    let b3_low = p_min - 25.0 * p_std;
    let starts: [[f64; 4]; 13] = [
        [
            t_max,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 0.1 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p,
            (p_std * 10.0 * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p + p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        [
            t_max,
            t_min,
            mean_p - p_std,
            (p_std * b4_sign).copysign(b4_sign),
        ],
        [
            -tail,
            t_max,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            mean_p,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            mean_p,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            -tail,
            t_max,
            b3_high,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_max,
            -tail,
            b3_low,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            tail,
            t_min,
            b3_low,
            (p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
        [
            t_min,
            tail,
            b3_high,
            (-p_std * b4_sign).max(1e-3).copysign(b4_sign),
        ],
    ];
    let mut best_b: Option<[f64; 4]> = None;
    let mut best_cost = f64::INFINITY;
    for start in &starts {
        if let Some((b_fit, cost_fit)) = run_lm(predicted, target, n, *start) {
            if cost_fit < best_cost {
                best_cost = cost_fit;
                best_b = Some(b_fit);
            }
        }
    }
    let b: [f64; 4] = match best_b {
        Some(b) => b,
        None => return rescale_affine(predicted, target),
    };
    let any_bad = predicted
        .iter()
        .take(n)
        .any(|&x| !logistic_eval(&b, x).is_finite());
    if any_bad {
        return rescale_affine(predicted, target);
    }
    predicted.iter().map(|&x| logistic_eval(&b, x)).collect()
}

// ============================================================================
// Bake scoring helpers
// ============================================================================

/// Per-sample α head dispatch payload — parsed from the bake's
/// `zentrain.per_sample_alpha_head` metadata. Layout matches
/// `zensim-train-core::per_sample_alpha_head::bake_per_sample_alpha_head_v3`
/// (and zensim's runtime in `zensim::metric::forward_one_bake`).
type PerSampleAlphaHeadDispatch = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);

/// Read the `zentrain.per_sample_alpha_head` metadata payload, if any.
/// Returns `Some((W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm))`.
fn extract_per_sample_alpha_head(model: &Model) -> Option<PerSampleAlphaHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.per_sample_alpha_head")?;
    let n_hidden = model.n_outputs();
    let expected = (2 * n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm))
}

/// Score one row through the loaded MLP. Caller pre-allocates the
/// `f32_features` scratch buffer to avoid the per-row allocation
/// that would otherwise dominate runtime (~10 ms × 19k pairs ≈ 3
/// min if we reallocated every call).
///
/// When the bake carries `zentrain.per_sample_alpha_head` metadata
/// (V_24-per-sample-α architecture), the forward output is treated
/// as the hidden vector `h` (the bake's layer 2 is an identity
/// passthrough) and the runtime mixes a rank-head + pool-head pair
/// via a per-sample sigmoid gate. Bit-exact match with
/// `zensim::metric::forward_one_bake`'s dispatch.
fn score_row(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n_inputs = f32_features.len();
    let take = n_inputs.min(row.len());
    for i in 0..take {
        f32_features[i] = row[i] as f32;
    }
    // Pad with zeros if the parquet is wider than the bake (unlikely
    // — all T10.1 parquets are 372-wide and bakes are ≤ 372).
    for f in &mut f32_features[take..] {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    match result {
        Ok(out) => {
            // Per-sample-α head dispatch — out is the hidden vector h.
            if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) =
                per_sample_alpha_head
            {
                let n = out.len() as f64;
                if n <= 0.0 || out.len() != rank_w.len() || out.len() != w_alpha.len() {
                    return f64::NAN;
                }
                let mut y_rank = *rank_b as f64;
                let mut alpha_logit = *b_alpha as f64;
                let mut sum = 0.0f64;
                let mut max_v = f64::NEG_INFINITY;
                let mut sum_p = 0.0f64;
                let p = *p_norm as f64;
                for (j, &h) in out.iter().enumerate() {
                    let hf = h as f64;
                    y_rank += hf * rank_w[j] as f64;
                    alpha_logit += hf * w_alpha[j] as f64;
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
                    let xc = alpha_logit.clamp(-20.0, 20.0);
                    1.0 / (1.0 + (-xc).exp())
                };
                alpha * y_rank + (1.0 - alpha) * y_pool
            } else {
                out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
            }
        }
        Err(_) => f64::NAN,
    }
}

// ============================================================================
// Corpus registry
// ============================================================================

#[derive(Clone, Debug)]
struct Corpus {
    name: &'static str,
    /// Display name in tables (matches dataset_metric_baseline.rs
    /// for diff-friendliness across the two binaries).
    display: &'static str,
    /// Parquet path (slot under `<features_root>/`).
    filename: &'static str,
    /// Per-band partitioning enabled? AIC-3 has 600 pairs in a JND
    /// step grid; rank-based per-band stats collapse to 0 on shared
    /// scores (see dataset_metric_baseline.rs comment at L454-471).
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
        // KonJND `human_score` here is `mean_threshold` (raw,
        // unit unclear from extract_features_372col.rs but
        // appears to be a per-pair JND threshold in [22, 70]).
        // 10-band-on-[0,1] partitioning doesn't apply; skip.
        enable_per_band: false,
    },
    Corpus {
        name: "aic3",
        display: "AIC-3 CTC",
        filename: "aic3_features_372col_2026-05-15.parquet",
        // AIC-3 = JND step grid (see comment above + L454-471
        // of dataset_metric_baseline.rs); per-band aggregate
        // is misleading.
        enable_per_band: false,
    },
];

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

// ============================================================================
// CLI parsing
// ============================================================================

struct Args {
    bake: PathBuf,
    corpora: Vec<&'static Corpus>,
    output: Option<PathBuf>,
    features_root: PathBuf,
}

fn print_usage() {
    eprintln!(
        "bake_verdict — instant V_X bake eval from pre-extracted parquet features\n\
\n\
USAGE:\n\
    bake_verdict --bake <path>\n\
                 [--corpora cid22,kadid,tid,konjnd,aic3]\n\
                 [--output <path.md>]\n\
                 [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features]\n\
\n\
DEFAULTS:\n\
    --corpora       all 5 (cid22,kadid,tid,konjnd,aic3)\n\
    --output        stdout\n\
    --features-root /mnt/v/zen/zensim-training/2026-05-15-full-features\n"
    );
}

fn parse_args() -> Result<Args, String> {
    let mut bake: Option<PathBuf> = None;
    let mut corpora: Option<Vec<&'static Corpus>> = None;
    let mut output: Option<PathBuf> = None;
    let mut features_root: PathBuf =
        PathBuf::from("/mnt/v/zen/zensim-training/2026-05-15-full-features");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--bake" => {
                let v = args.next().ok_or("--bake requires <path>")?;
                bake = Some(PathBuf::from(v));
            }
            "--corpora" => {
                let v = args.next().ok_or("--corpora requires comma list")?;
                corpora = Some(parse_corpora_arg(&v)?);
            }
            "--output" => {
                let v = args.next().ok_or("--output requires <path>")?;
                output = Some(PathBuf::from(v));
            }
            "--features-root" => {
                let v = args.next().ok_or("--features-root requires <path>")?;
                features_root = PathBuf::from(v);
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
    let bake = bake.ok_or("--bake is required (path to ZNPR v3 bake)")?;
    let corpora = corpora.unwrap_or_else(|| CORPORA.iter().collect());
    Ok(Args {
        bake,
        corpora,
        output,
        features_root,
    })
}

// ============================================================================
// Per-corpus pipeline
// ============================================================================

struct CorpusResult {
    display: &'static str,
    n: usize,
    srocc: f64,
    plcc: f64,
    krocc: f64,
    or_ratio: f64,
    pwrc: f64,
    z_rmse: f64,
    body: String,
}

fn aggregate_panel(scores: &[f64], humans: &[f64]) -> (f64, f64, f64, f64, f64, f64) {
    let srocc = spearman(humans, scores).abs();
    let krocc = kendall_tau(humans, scores).abs();
    let pw = pwrc(humans, scores).abs();
    let or_ = outlier_ratio(scores, humans);
    let rescaled = rescale_logistic(scores, humans);
    let plcc = pearson(&rescaled, humans).abs();
    let z = z_rmse(&rescaled, humans);
    (srocc, plcc, krocc, or_, pw, z)
}

fn render_corpus(
    corpus: &Corpus,
    features_root: &Path,
    has_transforms: bool,
    n_inputs: usize,
    model: &Model,
) -> Result<CorpusResult, String> {
    let path = features_root.join(corpus.filename);
    let g = parquet_loader::load_parquet(&path, corpus.display, "human_score", 1.0)
        .map_err(|e| format!("load {} parquet: {e}", corpus.display))?;
    let humans = g.human_scores;
    let per_sample_alpha_head = extract_per_sample_alpha_head(model);
    let mut predictor = Predictor::new(model);

    // Score every row. f32 scratch buffer reused across all rows
    // to avoid the per-row allocation that would otherwise dominate
    // wall time on the bigger corpora (KADID has 10k rows × 372 f32s).
    let mut scratch = vec![0.0f32; n_inputs];
    let scores: Vec<f64> = g
        .feature_rows
        .iter()
        .map(|row| {
            score_row(
                &mut predictor,
                has_transforms,
                per_sample_alpha_head.as_ref(),
                &mut scratch,
                row,
            )
        })
        .collect();

    let n = scores.len();
    let (srocc, plcc, krocc, or_, pw, z) = aggregate_panel(&scores, &humans);

    let mut body = String::new();
    body.push_str(&format!("\n## {} (n={})\n\n", corpus.display, n));
    body.push_str("### Aggregate full Mohammadi panel (CLAUDE.md rigor mandate)\n\n");
    body.push_str("| Metric | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |\n");
    body.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    body.push_str(&format!(
        "| V_X bake | {srocc:.4} | {plcc:.4} | {krocc:.4} | {or_:.4} | {pw:.4} | {z:.3} |\n"
    ));
    body.push('\n');
    body.push_str(
        "_Z-RMSE column uses corpus-wide σ (per-stimulus σ unavailable from \
parquet sidecars). Rescale is 4-parameter logistic (Mohammadi 2025 convention), \
not affine — affine inflates Z-RMSE on nonlinear metrics by 30× because \
saturation regions dominate the residual._\n",
    );

    if corpus.enable_per_band {
        body.push('\n');
        body.push_str(&format!(
            "### {} 10-band full Mohammadi panel (PRIMARY release gate)\n\n",
            corpus.display
        ));
        body.push_str("| Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |\n");
        body.push_str("|---|---|--:|---:|---:|---:|---:|---:|---:|---:|\n");
        // Per-band cuts: every corpus that hits this branch
        // (CID22 / KADID / TID) has human_score normalized
        // into [0, 1] per the feature-extractor convention.
        // Width-10 grid on the 0-100 scale → width-0.10 on [0, 1].
        for band_idx in 0..10 {
            let lo = band_idx as f64 * 0.10;
            let hi = lo + 0.10;
            let label = format!("B{band_idx}");
            let range_label = if band_idx == 9 {
                format!("[{:.2}, 1.00]", lo)
            } else {
                format!("[{:.2}, {:.2})", lo, hi)
            };
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
                body.push_str(&format!(
                    "| {label} | {range_label} | {} | n/a | n/a | n/a | n/a | n/a | n/a | n/a |\n",
                    idxs.len()
                ));
                continue;
            }
            let h_b: Vec<f64> = idxs.iter().map(|&i| humans[i]).collect();
            let s_b: Vec<f64> = idxs.iter().map(|&i| scores[i]).collect();
            let (b_srocc, b_plcc, b_krocc, b_or, b_pwrc, b_z) = aggregate_panel(&s_b, &h_b);
            let rescaled = rescale_logistic(&s_b, &h_b);
            let mae: f64 = rescaled
                .iter()
                .zip(h_b.iter())
                .map(|(r, h)| (r - h).abs())
                .sum::<f64>()
                / idxs.len() as f64;
            let noisy = if idxs.len() < 30 { " ⚠" } else { "" };
            body.push_str(&format!(
                "| {label}{noisy} | {range_label} | {} | {b_srocc:.4} | {b_plcc:.4} | {b_krocc:.4} | {b_or:.4} | {b_pwrc:.4} | {b_z:.3} | {mae:.4} |\n",
                idxs.len()
            ));
        }
        body.push('\n');
        body.push_str(
            "_⚠ marks bands with n < 30 — point estimates are noisy (CI widths \
exceed ±0.3 SROCC at n<30; rankings between bakes are not statistically \
distinguishable). MAE / Z-RMSE computed after 4-parameter logistic rescale \
per Mohammadi 2025._\n",
        );
    } else {
        body.push('\n');
        body.push_str(&format!(
            "_Per-band breakdown skipped for {} — the corpus uses a JND step grid (AIC-3) \
or a raw threshold scale (KonJND) that doesn't partition cleanly into the \
CID22/KADID/TID-style 0..1 normalized bands. Aggregate panel is the load-bearing \
read on this corpus._\n",
            corpus.display
        ));
    }

    Ok(CorpusResult {
        display: corpus.display,
        n,
        srocc,
        plcc,
        krocc,
        or_ratio: or_,
        pwrc: pw,
        z_rmse: z,
        body,
    })
}

// ============================================================================
// Main
// ============================================================================

fn main() -> ExitCode {
    let t0 = Instant::now();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("bake_verdict: {e}");
            print_usage();
            return ExitCode::from(2);
        }
    };
    eprintln!(
        "bake_verdict — bake={}  features-root={}  corpora={}",
        args.bake.display(),
        args.features_root.display(),
        args.corpora
            .iter()
            .map(|c| c.name)
            .collect::<Vec<_>>()
            .join(",")
    );

    let bake_bytes = match std::fs::read(&args.bake) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("bake_verdict: failed to read bake {}: {e}", args.bake.display());
            return ExitCode::from(1);
        }
    };
    let model = match Model::from_bytes(&bake_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bake_verdict: failed to parse ZNPR bake: {e:?}");
            return ExitCode::from(1);
        }
    };
    let n_inputs = model.n_inputs();
    let has_transforms = model.has_nontrivial_feature_transforms();
    let has_per_sample_alpha = extract_per_sample_alpha_head(&model).is_some();
    eprintln!(
        "bake: n_inputs={n_inputs}  feature_transforms={}  per_sample_alpha_head={}",
        if has_transforms { "yes" } else { "no" },
        if has_per_sample_alpha { "yes" } else { "no" }
    );

    let mut buf = String::new();
    buf.push_str("# bake_verdict — instant V_X eval\n\n");
    buf.push_str(&format!("- Bake: `{}`\n", args.bake.display()));
    buf.push_str(&format!("- Feature parquets: `{}`\n", args.features_root.display()));
    buf.push_str(&format!("- Bake n_inputs: {n_inputs}\n"));
    buf.push_str(&format!(
        "- Feature transforms: {}\n",
        if has_transforms { "yes (uses predict_transformed)" } else { "no" }
    ));

    let mut results: Vec<CorpusResult> = Vec::new();
    for corpus in &args.corpora {
        match render_corpus(corpus, &args.features_root, has_transforms, n_inputs, &model) {
            Ok(r) => results.push(r),
            Err(e) => {
                eprintln!("bake_verdict: {e}");
                return ExitCode::from(1);
            }
        }
    }

    // One-row summary across all corpora at the top.
    buf.push_str("\n## Summary (one row per corpus)\n\n");
    buf.push_str("| Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE |\n");
    buf.push_str("|---|--:|---:|---:|---:|---:|---:|---:|\n");
    for r in &results {
        buf.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.3} |\n",
            r.display, r.n, r.srocc, r.plcc, r.krocc, r.or_ratio, r.pwrc, r.z_rmse
        ));
    }
    for r in &results {
        buf.push_str(&r.body);
    }

    let elapsed = t0.elapsed();
    buf.push_str(&format!(
        "\n---\nWall time: {:.2}s ({} pair rows scored across {} corpora).\n",
        elapsed.as_secs_f64(),
        results.iter().map(|r| r.n).sum::<usize>(),
        results.len()
    ));

    if let Some(out_path) = args.output {
        if let Some(parent) = out_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        match File::create(&out_path) {
            Ok(mut f) => {
                if let Err(e) = f.write_all(buf.as_bytes()) {
                    eprintln!("bake_verdict: failed to write {}: {e}", out_path.display());
                    return ExitCode::from(1);
                }
                eprintln!("wrote verdict to {}", out_path.display());
            }
            Err(e) => {
                eprintln!("bake_verdict: failed to create {}: {e}", out_path.display());
                return ExitCode::from(1);
            }
        }
    } else {
        print!("{buf}");
    }

    eprintln!(
        "bake_verdict: complete in {:.2}s",
        elapsed.as_secs_f64()
    );
    ExitCode::SUCCESS
}
