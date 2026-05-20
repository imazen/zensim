//! Per-codec picker MLP trainer.
//!
//! Reads a per-codec picker-training parquet
//! (`/mnt/v/zen/picker-training/2026-05-19/splits/<codec>_train.parquet`)
//! and trains a tiny `108_feat + 1_T = 109 → 64 → 32 → 1` MLP that maps
//! `(zenanalyze_features, target_T)` → `predicted_q` for that codec.
//!
//! Schema expected (per `/mnt/v/zen/picker-training/2026-05-19/splits/_summary.json`):
//!   `ref_basename | codec | q | achieved_zensim_tuner | encoded_bytes |
//!    width | height | feat_0 | feat_1 | ... (108 feat_* cols total)`
//!
//! Training tuples are built per (source × T):
//!   For each source, compute `q*(T) = argmin |achieved_zensim_tuner(q) − T|`
//!   over the 19 q rows {5,10,...,95}. T grid is {30, 35, ..., 90, 95} (14 targets).
//!
//! Output: a ZNPR v3 3-layer bake at the path passed via `--out`.
//!
//! Reuses [`zensim_validate::bake_two_layer_znpr_v3`]? — No: the picker
//! is 3-layer (input→hidden1→hidden2→output). We build the BakeRequest
//! directly with three [`BakeLayer`] entries.

use anyhow::{Context, Result, anyhow};
use arrow::array::{Array, Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use clap::Parser;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::PathBuf;

use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

/// Target T grid (zensim_tuner scores) for picker training.
/// Spans the meaningful range; T=63 is the PJND anchor.
const T_GRID: &[f64] = &[
    30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0,
];

/// Q sample grid (matches the 19 q rows per source in the parquet).
const Q_GRID: &[f64] = &[
    5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    85.0, 90.0, 95.0,
];

#[derive(Parser, Debug)]
#[command(name = "zensim_picker_train")]
struct Args {
    /// Per-codec training parquet (zenjpeg/zenwebp/zenavif/zenjxl).
    #[arg(long)]
    train: PathBuf,

    /// Per-codec validation parquet.
    #[arg(long)]
    val: PathBuf,

    /// Codec name (for logging, metadata).
    #[arg(long)]
    codec: String,

    /// Output ZNPR v3 bake path.
    #[arg(long)]
    out: PathBuf,

    /// Hidden layer 1 size.
    #[arg(long, default_value_t = 64)]
    hidden1: usize,

    /// Hidden layer 2 size.
    #[arg(long, default_value_t = 32)]
    hidden2: usize,

    /// Adam learning rate.
    #[arg(long, default_value_t = 1e-3)]
    lr: f64,

    /// L2 weight decay.
    #[arg(long, default_value_t = 1e-5)]
    l2: f64,

    /// LeakyReLU alpha.
    #[arg(long, default_value_t = 0.01)]
    leaky_alpha: f64,

    /// Minibatch size.
    #[arg(long, default_value_t = 32)]
    minibatch: usize,

    /// Max epochs.
    #[arg(long, default_value_t = 200)]
    epochs: usize,

    /// Early-stop patience (epochs without val improvement).
    #[arg(long, default_value_t = 30)]
    patience: usize,

    /// RNG seed.
    #[arg(long, default_value_t = 20260519)]
    seed: u64,
}

#[derive(Debug, Clone)]
struct SourceCurve {
    ref_basename: String,
    /// 108 zenanalyze features.
    features: Vec<f64>,
    /// 19 (q, achieved_zensim_tuner) pairs in q-ascending order.
    qz: Vec<(f64, f64)>,
}

/// Load one codec parquet, group by source basename, return per-source curves.
fn load_parquet(path: &PathBuf) -> Result<(Vec<SourceCurve>, Vec<String>)> {
    let file = File::open(path).with_context(|| format!("open {path:?}"))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("{path:?}: parquet open"))?;
    let schema = builder.schema().clone();
    let parquet_schema = builder.parquet_schema().clone();

    let arrow_fields = schema.fields();

    // Locate required cols.
    let ref_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "ref_basename")
        .ok_or_else(|| anyhow!("{path:?}: missing ref_basename"))?;
    let q_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "q")
        .ok_or_else(|| anyhow!("{path:?}: missing q"))?;
    let z_idx = arrow_fields
        .iter()
        .position(|f| f.name() == "achieved_zensim_tuner")
        .ok_or_else(|| anyhow!("{path:?}: missing achieved_zensim_tuner"))?;

    // Collect all feat_* columns in their order in the schema.
    let mut feat_indices: Vec<(usize, String)> = arrow_fields
        .iter()
        .enumerate()
        .filter_map(|(i, f)| {
            let n = f.name();
            if n.starts_with("feat_") {
                Some((i, n.clone()))
            } else {
                None
            }
        })
        .collect();
    // Keep feat order as it is in the parquet (already in numeric order from prep).
    let feat_names: Vec<String> = feat_indices.iter().map(|(_, n)| n.clone()).collect();
    feat_indices.sort_by_key(|(i, _)| *i);
    let n_features = feat_indices.len();
    if n_features == 0 {
        return Err(anyhow!("{path:?}: no feat_* columns"));
    }

    // Build projection mask.
    let mut wanted: Vec<usize> = vec![ref_idx, q_idx, z_idx];
    for (i, _) in &feat_indices {
        wanted.push(*i);
    }
    let mask = ProjectionMask::leaves(&parquet_schema, wanted.iter().copied());

    let mut sorted_wanted = wanted.clone();
    sorted_wanted.sort_unstable();

    let pos = |orig: usize| -> usize { sorted_wanted.iter().position(|&i| i == orig).unwrap() };
    let proj_ref = pos(ref_idx);
    let proj_q = pos(q_idx);
    let proj_z = pos(z_idx);
    let proj_feats: Vec<usize> = feat_indices.iter().map(|(i, _)| pos(*i)).collect();

    let reader = builder
        .with_projection(mask)
        .with_batch_size(16384)
        .build()
        .with_context(|| format!("{path:?}: build reader"))?;

    // Group by source.
    let mut groups: BTreeMap<String, SourceCurve> = BTreeMap::new();

    for batch_res in reader {
        let batch = batch_res.with_context(|| format!("{path:?}: read batch"))?;
        let n_rows = batch.num_rows();
        if n_rows == 0 {
            continue;
        }

        let ref_col = batch
            .column(proj_ref)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| anyhow!("ref_basename not String"))?;
        let q_col = batch.column(proj_q);
        let z_col = batch.column(proj_z);

        // Materialize cols to f64.
        let q_vec: Vec<f64> = match q_col.data_type() {
            DataType::Int64 => {
                let a = q_col.as_any().downcast_ref::<Int64Array>().unwrap();
                (0..n_rows).map(|i| a.value(i) as f64).collect()
            }
            DataType::Float64 => {
                let a = q_col.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..n_rows).map(|i| a.value(i)).collect()
            }
            other => return Err(anyhow!("q dtype: {other:?}")),
        };
        let z_vec: Vec<f64> = match z_col.data_type() {
            DataType::Float64 => {
                let a = z_col.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..n_rows).map(|i| a.value(i)).collect()
            }
            DataType::Float32 => {
                let a = z_col.as_any().downcast_ref::<Float32Array>().unwrap();
                (0..n_rows).map(|i| a.value(i) as f64).collect()
            }
            other => return Err(anyhow!("z dtype: {other:?}")),
        };

        // Per-col feat materialization.
        let mut feat_cols_f64: Vec<Vec<f64>> = Vec::with_capacity(n_features);
        for &pi in &proj_feats {
            let col = batch.column(pi);
            let v: Vec<f64> = match col.data_type() {
                DataType::Float32 => {
                    let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                DataType::Float64 => {
                    let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i)).collect()
                }
                DataType::Int64 => {
                    let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..n_rows).map(|i| a.value(i) as f64).collect()
                }
                other => return Err(anyhow!("feat dtype: {other:?}")),
            };
            feat_cols_f64.push(v);
        }

        for row in 0..n_rows {
            let key = ref_col.value(row).to_string();
            let q = q_vec[row];
            let z = z_vec[row];
            let entry = groups.entry(key.clone()).or_insert_with(|| {
                let mut feats = Vec::with_capacity(n_features);
                for col in &feat_cols_f64 {
                    feats.push(col[row]);
                }
                SourceCurve {
                    ref_basename: key,
                    features: feats,
                    qz: Vec::with_capacity(19),
                }
            });
            entry.qz.push((q, z));
        }
    }

    // Sort each curve by q ascending, drop sources with missing q points.
    let mut curves: Vec<SourceCurve> = groups
        .into_values()
        .filter_map(|mut c| {
            c.qz.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            if c.qz.len() < Q_GRID.len() {
                None
            } else {
                Some(c)
            }
        })
        .collect();

    // Stable ordering.
    curves.sort_by(|a, b| a.ref_basename.cmp(&b.ref_basename));

    Ok((curves, feat_names))
}

/// Given a per-source (q, z) curve and a target T, find q*(T) as the q on
/// the grid whose interpolated curve crosses T, using linear interpolation
/// between adjacent grid points. If T is below the curve's minimum, return
/// the first q with z > 0 (or q=5 if all zero); if above the max, return 95.
fn q_star_for_t(curve: &[(f64, f64)], t: f64) -> f64 {
    let n = curve.len();
    debug_assert!(n >= 2);
    // Find the lowest q index whose z >= T (curve is monotone-ish increasing in q,
    // but not strictly — we handle plateaus).
    // Strategy: find a bracket [i, i+1] where z[i] <= T <= z[i+1], linear interp.
    // If T < z[0], return q[0]. If T > z[n-1], return q[n-1].
    if t <= curve[0].1 {
        return curve[0].0;
    }
    if t >= curve[n - 1].1 {
        return curve[n - 1].0;
    }
    for i in 0..n - 1 {
        let (q0, z0) = curve[i];
        let (q1, z1) = curve[i + 1];
        if z0 <= t && t <= z1 {
            if (z1 - z0).abs() < 1e-9 {
                return q0;
            }
            let frac = (t - z0) / (z1 - z0);
            return q0 + frac * (q1 - q0);
        }
    }
    // Fallback (non-monotone curves): closest-q.
    let mut best_q = curve[0].0;
    let mut best_diff = (curve[0].1 - t).abs();
    for &(q, z) in curve {
        let d = (z - t).abs();
        if d < best_diff {
            best_diff = d;
            best_q = q;
        }
    }
    best_q
}

/// Build training tuples: (input_vec[n_features+1], q*).
fn build_tuples(curves: &[SourceCurve]) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n_features = curves[0].features.len();
    let mut x_rows: Vec<Vec<f64>> = Vec::with_capacity(curves.len() * T_GRID.len());
    let mut y_rows: Vec<f64> = Vec::with_capacity(curves.len() * T_GRID.len());
    for curve in curves {
        for &t in T_GRID {
            let q_star = q_star_for_t(&curve.qz, t);
            let mut x = Vec::with_capacity(n_features + 1);
            x.extend_from_slice(&curve.features);
            x.push(t);
            x_rows.push(x);
            y_rows.push(q_star);
        }
    }
    (x_rows, y_rows)
}

/// Three-layer MLP: in → h1 (LeakyReLU) → h2 (LeakyReLU) → 1 (Identity).
#[derive(Debug, Clone)]
struct Mlp3 {
    n_in: usize,
    n_h1: usize,
    n_h2: usize,
    /// Row-major [n_h1][n_in].
    w1: Vec<f64>,
    b1: Vec<f64>,
    /// Row-major [n_h2][n_h1].
    w2: Vec<f64>,
    b2: Vec<f64>,
    /// Row-major [1][n_h2].
    w3: Vec<f64>,
    b3: Vec<f64>,
}

impl Mlp3 {
    fn new(n_in: usize, n_h1: usize, n_h2: usize, rng: &mut Xoshiro) -> Self {
        // He init for LeakyReLU.
        let s1 = (2.0 / n_in as f64).sqrt();
        let s2 = (2.0 / n_h1 as f64).sqrt();
        let s3 = (2.0 / n_h2 as f64).sqrt();
        let mut w1 = vec![0.0; n_h1 * n_in];
        let mut w2 = vec![0.0; n_h2 * n_h1];
        let mut w3 = vec![0.0; 1 * n_h2];
        for w in w1.iter_mut() {
            *w = rng.gauss() * s1;
        }
        for w in w2.iter_mut() {
            *w = rng.gauss() * s2;
        }
        for w in w3.iter_mut() {
            *w = rng.gauss() * s3;
        }
        Self {
            n_in,
            n_h1,
            n_h2,
            w1,
            b1: vec![0.0; n_h1],
            w2,
            b2: vec![0.0; n_h2],
            w3,
            b3: vec![0.0],
        }
    }

    /// Forward pass returning y plus the activation buffers needed for backprop.
    #[allow(clippy::type_complexity)]
    fn forward(&self, x: &[f64], alpha: f64) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        debug_assert_eq!(x.len(), self.n_in);
        // h1_pre = b1 + w1 * x
        let mut h1_pre = self.b1.clone();
        for j in 0..self.n_h1 {
            let row = &self.w1[j * self.n_in..(j + 1) * self.n_in];
            let mut s = 0.0;
            for i in 0..self.n_in {
                s += row[i] * x[i];
            }
            h1_pre[j] += s;
        }
        let h1: Vec<f64> = h1_pre.iter().map(|&z| leaky(z, alpha)).collect();

        // h2_pre = b2 + w2 * h1
        let mut h2_pre = self.b2.clone();
        for j in 0..self.n_h2 {
            let row = &self.w2[j * self.n_h1..(j + 1) * self.n_h1];
            let mut s = 0.0;
            for i in 0..self.n_h1 {
                s += row[i] * h1[i];
            }
            h2_pre[j] += s;
        }
        let h2: Vec<f64> = h2_pre.iter().map(|&z| leaky(z, alpha)).collect();

        // y = b3 + w3 * h2
        let mut y = self.b3[0];
        for i in 0..self.n_h2 {
            y += self.w3[i] * h2[i];
        }
        (y, h1_pre, h1, h2_pre, h2)
    }
}

#[inline]
fn leaky(z: f64, alpha: f64) -> f64 {
    if z >= 0.0 { z } else { alpha * z }
}

#[inline]
fn leaky_deriv(z_pre: f64, alpha: f64) -> f64 {
    if z_pre >= 0.0 { 1.0 } else { alpha }
}

/// Adam state for an MLP.
#[derive(Debug)]
struct AdamMlp {
    m_w1: Vec<f64>,
    v_w1: Vec<f64>,
    m_b1: Vec<f64>,
    v_b1: Vec<f64>,
    m_w2: Vec<f64>,
    v_w2: Vec<f64>,
    m_b2: Vec<f64>,
    v_b2: Vec<f64>,
    m_w3: Vec<f64>,
    v_w3: Vec<f64>,
    m_b3: Vec<f64>,
    v_b3: Vec<f64>,
    t: usize,
    beta1: f64,
    beta2: f64,
    eps: f64,
    lr: f64,
    l2: f64,
}

impl AdamMlp {
    fn new(model: &Mlp3, lr: f64, l2: f64) -> Self {
        Self {
            m_w1: vec![0.0; model.w1.len()],
            v_w1: vec![0.0; model.w1.len()],
            m_b1: vec![0.0; model.b1.len()],
            v_b1: vec![0.0; model.b1.len()],
            m_w2: vec![0.0; model.w2.len()],
            v_w2: vec![0.0; model.w2.len()],
            m_b2: vec![0.0; model.b2.len()],
            v_b2: vec![0.0; model.b2.len()],
            m_w3: vec![0.0; model.w3.len()],
            v_w3: vec![0.0; model.w3.len()],
            m_b3: vec![0.0; model.b3.len()],
            v_b3: vec![0.0; model.b3.len()],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            lr,
            l2,
        }
    }

    fn step(
        &mut self,
        model: &mut Mlp3,
        g_w1: &mut [f64],
        g_b1: &mut [f64],
        g_w2: &mut [f64],
        g_b2: &mut [f64],
        g_w3: &mut [f64],
        g_b3: &mut [f64],
    ) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        update(
            &mut model.w1,
            g_w1,
            &mut self.m_w1,
            &mut self.v_w1,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            self.l2,
        );
        update(
            &mut model.b1,
            g_b1,
            &mut self.m_b1,
            &mut self.v_b1,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            0.0,
        );
        update(
            &mut model.w2,
            g_w2,
            &mut self.m_w2,
            &mut self.v_w2,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            self.l2,
        );
        update(
            &mut model.b2,
            g_b2,
            &mut self.m_b2,
            &mut self.v_b2,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            0.0,
        );
        update(
            &mut model.w3,
            g_w3,
            &mut self.m_w3,
            &mut self.v_w3,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            self.l2,
        );
        update(
            &mut model.b3,
            g_b3,
            &mut self.m_b3,
            &mut self.v_b3,
            self.beta1,
            self.beta2,
            self.eps,
            bc1,
            bc2,
            self.lr,
            0.0,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn update(
    w: &mut [f64],
    g: &mut [f64],
    m: &mut [f64],
    v: &mut [f64],
    beta1: f64,
    beta2: f64,
    eps: f64,
    bc1: f64,
    bc2: f64,
    lr: f64,
    l2: f64,
) {
    for i in 0..w.len() {
        let mut gi = g[i];
        if l2 > 0.0 {
            gi += l2 * w[i];
        }
        let m_new = beta1 * m[i] + (1.0 - beta1) * gi;
        let v_new = beta2 * v[i] + (1.0 - beta2) * gi * gi;
        m[i] = m_new;
        v[i] = v_new;
        let m_hat = m_new / bc1;
        let v_hat = v_new / bc2;
        w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        g[i] = 0.0;
    }
}

/// Xoshiro256** for reproducible init/shuffle.
#[derive(Debug)]
struct Xoshiro {
    s: [u64; 4],
}

impl Xoshiro {
    fn new(seed: u64) -> Self {
        // SplitMix64 to expand seed.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        let mut sm = || {
            z = z.wrapping_add(0x9E3779B97F4A7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
            x ^ (x >> 31)
        };
        Self {
            s: [sm(), sm(), sm(), sm()],
        }
    }

    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Standard normal via Box-Muller (single sample per call; pair-drop for simplicity).
    fn gauss(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            if u1 > 1e-300 {
                let u2 = self.next_f64();
                let mag = (-2.0 * u1.ln()).sqrt();
                return mag * (2.0 * std::f64::consts::PI * u2).cos();
            }
        }
    }

    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            v.swap(i, j);
        }
    }
}

/// Z-score normalize the features (and T column) by training mean+std.
fn compute_scaler(x_rows: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    let n_in = x_rows[0].len();
    let n = x_rows.len() as f64;
    let mut mean = vec![0.0; n_in];
    for row in x_rows {
        for i in 0..n_in {
            mean[i] += row[i];
        }
    }
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0; n_in];
    for row in x_rows {
        for i in 0..n_in {
            let d = row[i] - mean[i];
            var[i] += d * d;
        }
    }
    let std: Vec<f64> = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

fn standardize(x: &[f64], mean: &[f64], std: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(mean.iter())
        .zip(std.iter())
        .map(|((a, m), s)| (a - m) / s)
        .collect()
}

/// Compute mean |predicted_q − q*| over a (x_std, y) set.
fn eval_mae(model: &Mlp3, x_std_rows: &[Vec<f64>], y_rows: &[f64], alpha: f64) -> f64 {
    let n = y_rows.len() as f64;
    let mut s = 0.0;
    for i in 0..y_rows.len() {
        let (y_hat, _, _, _, _) = model.forward(&x_std_rows[i], alpha);
        s += (y_hat - y_rows[i]).abs();
    }
    s / n
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== zensim_picker_train ===");
    println!("codec: {}", args.codec);
    println!("train: {:?}", args.train);
    println!("val:   {:?}", args.val);
    println!("out:   {:?}", args.out);
    println!(
        "arch:  in → {} → {} → 1, leaky α={}",
        args.hidden1, args.hidden2, args.leaky_alpha
    );
    println!(
        "opt:   Adam lr={} l2={} mb={} epochs={} patience={}",
        args.lr, args.l2, args.minibatch, args.epochs, args.patience
    );
    println!("seed:  {}", args.seed);

    // 1. Load parquets.
    let (train_curves, feat_names) = load_parquet(&args.train)?;
    let (val_curves, _) = load_parquet(&args.val)?;
    println!(
        "train: {} sources × {} features",
        train_curves.len(),
        feat_names.len()
    );
    println!("val:   {} sources", val_curves.len());

    // 2. Build (x, y) tuples.
    let (x_train, y_train) = build_tuples(&train_curves);
    let (x_val, y_val) = build_tuples(&val_curves);
    println!("train tuples: {}", x_train.len());
    println!("val tuples:   {}", x_val.len());

    // 3. Scaler (fit on train).
    let (mean, std) = compute_scaler(&x_train);
    let x_train_std: Vec<Vec<f64>> = x_train
        .iter()
        .map(|x| standardize(x, &mean, &std))
        .collect();
    let x_val_std: Vec<Vec<f64>> = x_val.iter().map(|x| standardize(x, &mean, &std)).collect();

    let n_in = x_train_std[0].len();
    println!("n_in: {} (= {} feat + 1 T)", n_in, feat_names.len());

    // 4. Init model + optimizer.
    let mut rng = Xoshiro::new(args.seed);
    let mut model = Mlp3::new(n_in, args.hidden1, args.hidden2, &mut rng);
    let mut adam = AdamMlp::new(&model, args.lr, args.l2);

    // 5. Train loop.
    let mut best_val = f64::INFINITY;
    let mut best_model = model.clone();
    let mut stale = 0usize;
    let mb = args.minibatch;
    let mut indices: Vec<usize> = (0..x_train_std.len()).collect();

    // Pre-alloc gradient buffers.
    let mut g_w1 = vec![0.0; model.w1.len()];
    let mut g_b1 = vec![0.0; model.b1.len()];
    let mut g_w2 = vec![0.0; model.w2.len()];
    let mut g_b2 = vec![0.0; model.b2.len()];
    let mut g_w3 = vec![0.0; model.w3.len()];
    let mut g_b3 = vec![0.0; model.b3.len()];

    let n_h1 = args.hidden1;
    let n_h2 = args.hidden2;

    for epoch in 0..args.epochs {
        rng.shuffle(&mut indices);
        let mut epoch_loss = 0.0;
        let mut n_batches = 0usize;

        let mut cursor = 0;
        while cursor < indices.len() {
            let end = (cursor + mb).min(indices.len());
            let batch = &indices[cursor..end];
            let bs = batch.len() as f64;

            // Zero grads.
            g_w1.iter_mut().for_each(|x| *x = 0.0);
            g_b1.iter_mut().for_each(|x| *x = 0.0);
            g_w2.iter_mut().for_each(|x| *x = 0.0);
            g_b2.iter_mut().for_each(|x| *x = 0.0);
            g_w3.iter_mut().for_each(|x| *x = 0.0);
            g_b3.iter_mut().for_each(|x| *x = 0.0);

            let mut batch_loss = 0.0;

            for &idx in batch {
                let x = &x_train_std[idx];
                let target = y_train[idx];
                let (y, h1_pre, h1, h2_pre, h2) = model.forward(x, args.leaky_alpha);
                let err = y - target;
                batch_loss += 0.5 * err * err;

                // Backprop. Average over batch via 1/bs scaling at the end.
                // dL/dy = err
                let dl_dy = err / bs;
                // w3 grad: dL/dw3[i] += h2[i] * dl_dy
                for i in 0..n_h2 {
                    g_w3[i] += h2[i] * dl_dy;
                }
                g_b3[0] += dl_dy;
                // dL/dh2[i] = w3[i] * dl_dy
                // dL/dh2_pre[i] = dL/dh2[i] * leaky'(h2_pre[i])
                let mut dh2_pre = vec![0.0; n_h2];
                for i in 0..n_h2 {
                    let dh2_i = model.w3[i] * dl_dy;
                    dh2_pre[i] = dh2_i * leaky_deriv(h2_pre[i], args.leaky_alpha);
                }
                // w2 grad: g_w2[j*n_h1 + i] += h1[i] * dh2_pre[j]
                for j in 0..n_h2 {
                    g_b2[j] += dh2_pre[j];
                    let row = &mut g_w2[j * n_h1..(j + 1) * n_h1];
                    for i in 0..n_h1 {
                        row[i] += h1[i] * dh2_pre[j];
                    }
                }
                // dL/dh1[i] = sum_j w2[j*n_h1 + i] * dh2_pre[j]
                // dL/dh1_pre[i] = dL/dh1[i] * leaky'(h1_pre[i])
                let mut dh1_pre = vec![0.0; n_h1];
                for i in 0..n_h1 {
                    let mut s = 0.0;
                    for j in 0..n_h2 {
                        s += model.w2[j * n_h1 + i] * dh2_pre[j];
                    }
                    dh1_pre[i] = s * leaky_deriv(h1_pre[i], args.leaky_alpha);
                }
                // w1 grad: g_w1[j*n_in + i] += x[i] * dh1_pre[j]
                for j in 0..n_h1 {
                    g_b1[j] += dh1_pre[j];
                    let row = &mut g_w1[j * n_in..(j + 1) * n_in];
                    for i in 0..n_in {
                        row[i] += x[i] * dh1_pre[j];
                    }
                }
            }

            epoch_loss += batch_loss / bs;
            n_batches += 1;

            adam.step(
                &mut model, &mut g_w1, &mut g_b1, &mut g_w2, &mut g_b2, &mut g_w3, &mut g_b3,
            );

            cursor = end;
        }
        let avg_loss = epoch_loss / n_batches as f64;

        // Val MAE.
        let val_mae = eval_mae(&model, &x_val_std, &y_val, args.leaky_alpha);
        let train_mae = if epoch % 10 == 0 || epoch == args.epochs - 1 {
            eval_mae(&model, &x_train_std, &y_train, args.leaky_alpha)
        } else {
            f64::NAN
        };
        if val_mae < best_val {
            best_val = val_mae;
            best_model = model.clone();
            stale = 0;
        } else {
            stale += 1;
        }
        if epoch < 5 || epoch % 10 == 0 || stale == 0 || epoch == args.epochs - 1 {
            println!(
                "epoch {epoch:>3} | train_mse_q={avg_loss:8.3} | val_mae_q={val_mae:.3} | best={best_val:.3} | train_mae={train_mae:.3} | stale={stale}",
            );
        }
        if stale >= args.patience {
            println!("early stop at epoch {epoch} (stale={stale}, best_val_mae={best_val:.3})");
            break;
        }
    }
    println!("final best val MAE: {best_val:.4} q-units");

    // 6. Bake ZNPR v3 (3-layer).
    //
    // Layout transpose: training uses w[out_neuron][in_neuron] row-major
    // (i.e., row j is the j-th output's input weights); the zenpredict
    // runtime expects w[in_neuron][out_neuron] row-major (row i is the
    // i-th input's outgoing weights). Transpose at bake time.
    let scaler_mean_f32: Vec<f32> = mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = std.iter().map(|&v| v as f32).collect();

    fn transpose_f32(src: &[f64], rows: usize, cols: usize) -> Vec<f32> {
        // src is [rows][cols] row-major (j*cols + i for cell (j, i))
        // dst is [cols][rows] row-major (i*rows + j for cell (i, j))
        let mut out = vec![0.0f32; rows * cols];
        for j in 0..rows {
            for i in 0..cols {
                out[i * rows + j] = src[j * cols + i] as f32;
            }
        }
        out
    }

    let w1_f32: Vec<f32> = transpose_f32(&best_model.w1, n_h1, n_in);
    let b1_f32: Vec<f32> = best_model.b1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = transpose_f32(&best_model.w2, n_h2, n_h1);
    let b2_f32: Vec<f32> = best_model.b2.iter().map(|&v| v as f32).collect();
    let w3_f32: Vec<f32> = transpose_f32(&best_model.w3, 1, n_h2);
    let b3_f32: Vec<f32> = best_model.b3.iter().map(|&v| v as f32).collect();

    let layers = [
        BakeLayer {
            in_dim: n_in,
            out_dim: n_h1,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_h1,
            out_dim: n_h2,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w2_f32,
            biases: &b2_f32,
        },
        BakeLayer {
            in_dim: n_h2,
            out_dim: 1,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w3_f32,
            biases: &b3_f32,
        },
    ];

    // Metadata: provenance.
    let codec_md = args.codec.clone();
    let t_grid_str: String = T_GRID
        .iter()
        .map(|t| format!("{}", t))
        .collect::<Vec<_>>()
        .join(",");
    let q_grid_str: String = Q_GRID
        .iter()
        .map(|q| format!("{}", q))
        .collect::<Vec<_>>()
        .join(",");
    let mae_str = format!("{:.4}", best_val);
    let n_in_str = format!("{}", n_in);

    let metadata = [
        BakeMetadataEntry {
            key: "picker.codec",
            kind: zenpredict::MetadataType::Utf8,
            value: codec_md.as_bytes(),
        },
        BakeMetadataEntry {
            key: "picker.t_grid",
            kind: zenpredict::MetadataType::Utf8,
            value: t_grid_str.as_bytes(),
        },
        BakeMetadataEntry {
            key: "picker.q_grid",
            kind: zenpredict::MetadataType::Utf8,
            value: q_grid_str.as_bytes(),
        },
        BakeMetadataEntry {
            key: "picker.val_mae_q",
            kind: zenpredict::MetadataType::Utf8,
            value: mae_str.as_bytes(),
        },
        BakeMetadataEntry {
            key: "picker.n_in",
            kind: zenpredict::MetadataType::Utf8,
            value: n_in_str.as_bytes(),
        },
    ];

    let bake_bytes = bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &metadata,
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .map_err(|e| anyhow!("bake error: {e:?}"))?;

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&args.out, &bake_bytes).with_context(|| format!("write {:?}", args.out))?;
    println!(
        "wrote ZNPR v3 bake: {:?} ({} bytes)",
        args.out,
        bake_bytes.len()
    );

    Ok(())
}
