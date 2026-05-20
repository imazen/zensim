//! Backend dispatch + training loop orchestration.
//!
//! The actual training loop is generic over `R: Runtime` so the same
//! kernel code runs on every backend (CUDA / WGPU / CPU).

use cubecl::prelude::*;
use std::time::Instant;
use zensim_train_core::TrainingGroup;
use zensim_train_core::per_sample_alpha_head::PerSampleAlphaHeadModel;

use crate::kernels;
use crate::{GpuAnchorRows, GpuEquivPairs, GpuHparams, GpuRuntime, GpuTrainResult};

pub(crate) fn dispatch(
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    runtime: GpuRuntime,
    anchor: Option<&GpuAnchorRows<'_>>,
    equiv: Option<&GpuEquivPairs<'_>>,
) -> GpuTrainResult {
    match runtime {
        #[cfg(feature = "gpu-cuda")]
        GpuRuntime::Cuda => {
            let client =
                <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default());
            run::<cubecl::cuda::CudaRuntime>(client, groups, hp, n_features, anchor, equiv)
        }
        #[cfg(not(feature = "gpu-cuda"))]
        GpuRuntime::Cuda => panic!(
            "zensim-train-gpu: --gpu-runtime cuda requested but compiled without --features gpu-cuda"
        ),
        #[cfg(feature = "gpu-wgpu")]
        GpuRuntime::Wgpu => {
            let client =
                <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default());
            run::<cubecl::wgpu::WgpuRuntime>(client, groups, hp, n_features, anchor, equiv)
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        GpuRuntime::Wgpu => panic!(
            "zensim-train-gpu: --gpu-runtime wgpu requested but compiled without --features gpu-wgpu"
        ),
        #[cfg(feature = "gpu-cpu")]
        GpuRuntime::Cpu => {
            let client =
                <cubecl::cpu::CpuRuntime as Runtime>::client(&Default::default());
            run::<cubecl::cpu::CpuRuntime>(client, groups, hp, n_features, anchor, equiv)
        }
        #[cfg(not(feature = "gpu-cpu"))]
        GpuRuntime::Cpu => panic!(
            "zensim-train-gpu: --gpu-runtime cpu requested but compiled without --features gpu-cpu"
        ),
    }
}

/// SplitMix64 — matches `zensim-train-core::rng::SplitMix64`.
struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    fn next_u64(&mut self) -> u64 {
        let mut z = self.0;
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        z ^= z >> 30;
        z = z.wrapping_mul(0xBF58476D1CE4E5B9);
        z ^= z >> 27;
        z = z.wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        z
    }
    fn next_f64_01(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Pooled mean + pooled std (population) across all training-weighted
/// rows. Floor σ at 1e-6 to keep std (rare zero-variance feats) sane.
fn compute_scaler(
    groups: &[TrainingGroup<'_>],
    train_indices: &[usize],
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut total_rows: usize = 0;
    for &gi in train_indices {
        total_rows += groups[gi].features.len();
    }
    if total_rows == 0 {
        return (vec![0.0; n_features], vec![1.0; n_features]);
    }
    let mut sum = vec![0.0_f64; n_features];
    let mut sum_sq = vec![0.0_f64; n_features];
    for &gi in train_indices {
        for row in groups[gi].features {
            for (d, v) in row.iter().enumerate() {
                sum[d] += v;
                sum_sq[d] += v * v;
            }
        }
    }
    let n = total_rows as f64;
    let mean: Vec<f64> = sum.iter().map(|s| s / n).collect();
    let scale: Vec<f64> = (0..n_features)
        .map(|d| {
            let var = (sum_sq[d] / n) - mean[d] * mean[d];
            var.max(0.0).sqrt().max(1e-6)
        })
        .collect();
    (mean, scale)
}

struct StdData {
    group_rows: Vec<Vec<f32>>,
    group_scores: Vec<Vec<f32>>,
    weight_cdf: Vec<f64>,
}

/// Standardized anchor pool living on the host side. The kernel input
/// is a flat `n_rows × n_features` f32 row-major matrix.
struct StdAnchor {
    /// `n_rows × n_features` row-major, pre-standardized.
    rows: Vec<f32>,
    n_rows: usize,
    /// Per-row weight (f32).
    weights: Vec<f32>,
    /// Per-row target score (f32). When the caller supplies `None`,
    /// every row gets the same global target (we always materialize
    /// the constant vector for GPU upload).
    targets: Vec<f32>,
    /// CDF over row_weights for sampling.
    row_cdf: Vec<f64>,
    /// Sum of row weights (sanity gate).
    total_weight: f64,
}

/// Standardized equiv pool — A-side + B-side rows in two flat matrices.
struct StdEquiv {
    n_rows: usize,
    rows_a: Vec<f32>,
    rows_b: Vec<f32>,
    weights: Vec<f32>,
    /// Per-pair butter_diff. Empty vec when rank-preserve disabled.
    butter_diff: Vec<f32>,
    row_cdf: Vec<f64>,
    total_weight: f64,
}

fn standardize_aux_rows(
    features: &[&[f64]],
    weights: &[f64],
    targets: Option<&[f64]>,
    mean: &[f64],
    scale: &[f64],
    n_features: usize,
) -> StdAnchor {
    let n_rows = features.len();
    let mut rows = vec![0.0_f32; n_rows * n_features];
    for (r, f) in features.iter().enumerate() {
        let off = r * n_features;
        for d in 0..n_features {
            let v = (f[d] - mean[d]) / scale[d].max(1e-12);
            rows[off + d] = v as f32;
        }
    }
    let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
    let targets_f32: Vec<f32> = if let Some(ts) = targets {
        ts.iter().map(|&v| v as f32).collect()
    } else {
        vec![0.0_f32; n_rows]
    };
    let total: f64 = weights.iter().map(|&w| w.max(0.0)).sum();
    let mut cum = 0.0_f64;
    let row_cdf: Vec<f64> = weights
        .iter()
        .map(|&w| {
            cum += w.max(0.0);
            if total > 0.0 { cum / total } else { 0.0 }
        })
        .collect();
    StdAnchor {
        rows,
        n_rows,
        weights: weights_f32,
        targets: targets_f32,
        row_cdf,
        total_weight: total,
    }
}

fn standardize_aux_pairs(
    features_a: &[&[f64]],
    features_b: &[&[f64]],
    weights: &[f64],
    butter_diff: Option<&[f64]>,
    mean: &[f64],
    scale: &[f64],
    n_features: usize,
) -> StdEquiv {
    let n_rows = features_a.len();
    let mut rows_a = vec![0.0_f32; n_rows * n_features];
    let mut rows_b = vec![0.0_f32; n_rows * n_features];
    for r in 0..n_rows {
        let off = r * n_features;
        let fa = features_a[r];
        let fb = features_b[r];
        for d in 0..n_features {
            let scl = scale[d].max(1e-12);
            rows_a[off + d] = ((fa[d] - mean[d]) / scl) as f32;
            rows_b[off + d] = ((fb[d] - mean[d]) / scl) as f32;
        }
    }
    let weights_f32: Vec<f32> = weights.iter().map(|&w| w as f32).collect();
    let butter_f32: Vec<f32> = butter_diff
        .map(|bd| bd.iter().map(|&v| v as f32).collect())
        .unwrap_or_default();
    let total: f64 = weights.iter().map(|&w| w.max(0.0)).sum();
    let mut cum = 0.0_f64;
    let row_cdf: Vec<f64> = weights
        .iter()
        .map(|&w| {
            cum += w.max(0.0);
            if total > 0.0 { cum / total } else { 0.0 }
        })
        .collect();
    StdEquiv {
        n_rows,
        rows_a,
        rows_b,
        weights: weights_f32,
        butter_diff: butter_f32,
        row_cdf,
        total_weight: total,
    }
}

fn standardize_groups(
    groups: &[TrainingGroup<'_>],
    train_indices: &[usize],
    mean: &[f64],
    scale: &[f64],
    n_features: usize,
) -> StdData {
    let mut group_rows = Vec::with_capacity(train_indices.len());
    let mut group_scores = Vec::with_capacity(train_indices.len());
    let weights: Vec<f64> = train_indices
        .iter()
        .map(|&gi| groups[gi].train_weight)
        .collect();
    let total_w: f64 = weights.iter().sum();
    let weight_cdf: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w / total_w;
            Some(*acc)
        })
        .collect();

    for &gi in train_indices {
        let g = &groups[gi];
        let n_rows = g.features.len();
        let mut flat = vec![0.0_f32; n_rows * n_features];
        for (r, row) in g.features.iter().enumerate() {
            let off = r * n_features;
            for d in 0..n_features {
                let v = (row[d] - mean[d]) / scale[d];
                flat[off + d] = v as f32;
            }
        }
        group_rows.push(flat);
        group_scores.push(g.human_scores.iter().map(|&v| v as f32).collect());
    }
    StdData {
        group_rows,
        group_scores,
        weight_cdf,
    }
}

#[inline]
fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

fn run<R: Runtime>(
    client: ComputeClient<R>,
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    anchor: Option<&GpuAnchorRows<'_>>,
    equiv: Option<&GpuEquivPairs<'_>>,
) -> GpuTrainResult {
    let n_hidden = hp.n_hidden;
    let k_batch = hp.minibatch_k;
    let two_k = 2 * k_batch;
    let k_aux = hp.minibatch_k_aux.max(1);
    let probe_n = hp.dynamic_range_probe_n.max(2);
    assert!(n_hidden <= 1024, "n_hidden must be ≤ 1024 (CUDA cube limit); got {n_hidden}");
    assert!(k_batch >= 1, "minibatch_k must be ≥ 1; got {k_batch}");
    // Aux uses the same scratch buffers as the main pair step. The
    // worst-case aux batch is the cross-codec-eq pair (2 × k_aux rows)
    // and the σ-floor probe (probe_n rows). Both must fit in `two_k`.
    let max_aux_rows = (2 * k_aux).max(probe_n);
    assert!(
        max_aux_rows <= two_k,
        "aux batch ({max_aux_rows} rows) exceeds main scratch capacity ({two_k}); \
         reduce minibatch_k_aux/dynamic_range_probe_n or raise minibatch_k"
    );

    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_per_sample_alpha_head_gpu: need at least one group with train_weight > 0"
    );

    let (mean, scale) = compute_scaler(groups, &train_indices, n_features);
    let std_data = standardize_groups(groups, &train_indices, &mean, &scale, n_features);

    // ---- Phase 2 aux data: standardize anchor rows / equiv pairs ----
    let anchor_active = anchor.is_some() && hp.anchor_loss_weight > 0.0;
    let equiv_active = equiv.is_some() && hp.cross_codec_eq_weight > 0.0;
    let rank_preserve_active = equiv_active && hp.cross_codec_rank_preserve_weight > 0.0;
    let sigma_floor_active =
        equiv_active && hp.dynamic_range_floor_weight > 0.0 && probe_n >= 2;

    let std_anchor = if anchor_active {
        let a = anchor.unwrap();
        assert_eq!(
            a.features.len(),
            a.row_weights.len(),
            "anchor.features.len() != anchor.row_weights.len()"
        );
        assert_eq!(
            a.features.len(),
            a.target_scores.len(),
            "anchor.features.len() != anchor.target_scores.len()"
        );
        Some(standardize_aux_rows(
            a.features,
            a.row_weights,
            Some(a.target_scores),
            &mean,
            &scale,
            n_features,
        ))
    } else {
        None
    };

    let std_equiv = if equiv_active {
        let e = equiv.unwrap();
        assert_eq!(
            e.features_a.len(),
            e.features_b.len(),
            "equiv.features_a.len() != equiv.features_b.len()"
        );
        assert_eq!(
            e.features_a.len(),
            e.row_weights.len(),
            "equiv.features_a.len() != equiv.row_weights.len()"
        );
        let butter = if rank_preserve_active
            && !e.butter_diff.is_empty()
            && e.butter_diff.len() == e.features_a.len()
        {
            Some(e.butter_diff)
        } else {
            None
        };
        Some(standardize_aux_pairs(
            e.features_a,
            e.features_b,
            e.row_weights,
            butter,
            &mean,
            &scale,
            n_features,
        ))
    } else {
        None
    };

    let mut model = PerSampleAlphaHeadModel::new(n_features, n_hidden, hp.seed);
    model.scaler_mean = mean.clone();
    model.scaler_scale = scale.clone();

    // ---------- GPU buffers ----------
    let f32_bytes = |data: &[f32]| -> Vec<u8> { f32::as_bytes(data).to_vec() };
    let alloc_f32 = |data: &[f32]| -> cubecl::server::Handle {
        client.create_from_slice(f32::as_bytes(data))
    };
    let alloc_u32 = |data: &[u32]| -> cubecl::server::Handle {
        client.create_from_slice(u32::as_bytes(data))
    };
    let zeros = |n: usize| vec![0.0_f32; n];
    let zeros_u32 = |n: usize| vec![0u32; n];

    let w1_init: Vec<f32> = model.w1.iter().map(|&v| v as f32).collect();
    let b1_init: Vec<f32> = model.b1.iter().map(|&v| v as f32).collect();
    let rank_w_init: Vec<f32> = model.rank_w.iter().map(|&v| v as f32).collect();
    let reducer_w_init: Vec<f32> = model.reducer_w.iter().map(|&v| v as f32).collect();
    let w_alpha_init: Vec<f32> = model.w_alpha.iter().map(|&v| v as f32).collect();

    let w1_h = alloc_f32(&w1_init);
    let b1_h = alloc_f32(&b1_init);
    let rank_w_h = alloc_f32(&rank_w_init);
    let reducer_w_h = alloc_f32(&reducer_w_init);
    let w_alpha_h = alloc_f32(&w_alpha_init);
    let rank_b_h = alloc_f32(&[model.rank_b as f32]);
    let reducer_b_h = alloc_f32(&[model.reducer_b as f32]);
    let b_alpha_h = alloc_f32(&[model.b_alpha as f32]);

    // Adam state.
    let mw1 = alloc_f32(&zeros(n_features * n_hidden));
    let vw1 = alloc_f32(&zeros(n_features * n_hidden));
    let mb1 = alloc_f32(&zeros(n_hidden));
    let vb1 = alloc_f32(&zeros(n_hidden));
    let m_rank_w = alloc_f32(&zeros(n_hidden));
    let v_rank_w = alloc_f32(&zeros(n_hidden));
    let m_reducer_w = alloc_f32(&zeros(4));
    let v_reducer_w = alloc_f32(&zeros(4));
    let m_w_alpha = alloc_f32(&zeros(n_hidden));
    let v_w_alpha = alloc_f32(&zeros(n_hidden));
    let m_rank_b = alloc_f32(&zeros(1));
    let v_rank_b = alloc_f32(&zeros(1));
    let m_reducer_b = alloc_f32(&zeros(1));
    let v_reducer_b = alloc_f32(&zeros(1));
    let m_b_alpha = alloc_f32(&zeros(1));
    let v_b_alpha = alloc_f32(&zeros(1));

    // Gradient buffers. gw1 is non-atomic (1 thread per cell). All
    // other head/bias grads are atomic (multiple threads contribute).
    let gw1 = alloc_f32(&zeros(n_features * n_hidden));
    let g_rank_w = alloc_f32(&zeros(n_hidden));
    let g_reducer_w = alloc_f32(&zeros(4));
    let g_w_alpha = alloc_f32(&zeros(n_hidden));
    let gb1 = alloc_f32(&zeros(n_hidden));
    let g_rank_b = alloc_f32(&zeros(1));
    let g_reducer_b = alloc_f32(&zeros(1));
    let g_b_alpha = alloc_f32(&zeros(1));

    // Per-batch scratch (persistent — overwritten via new create each step).
    let h_pre_h = alloc_f32(&zeros(two_k * n_hidden));
    let h_h = alloc_f32(&zeros(two_k * n_hidden));
    let y_rank_h = alloc_f32(&zeros(two_k));
    let y_pool_h = alloc_f32(&zeros(two_k));
    let stats_h = alloc_f32(&zeros(two_k * 4));
    let max_idx_h = alloc_u32(&zeros_u32(two_k));
    let alpha_h = alloc_f32(&zeros(two_k));
    let y_pre_h = alloc_f32(&zeros(two_k));
    let y_score_h = alloc_f32(&zeros(two_k));
    let dl_dypre_h = alloc_f32(&zeros(two_k));
    let dh_pre_h = alloc_f32(&zeros(two_k * n_hidden));

    let mut rng = SplitMix64::new(hp.seed ^ 0x5A5A_5A5A_5A5A_5A5A);
    let mut adam_t: u64 = 0;
    let pairs_per_epoch = hp.pairs_per_epoch.max(k_batch);
    let steps_per_epoch = pairs_per_epoch.div_ceil(k_batch);

    let cube_dim_h = CubeDim::new_1d(n_hidden as u32);
    let cube_dim_256 = CubeDim::new_1d(256);

    let l2 = hp.l2_lambda as f32;
    let lr = hp.initial_lr as f32;
    let leaky_alpha = hp.leaky_alpha as f32;
    let tanh_scale = hp.tanh_output_head_scale as f32;
    let ranknet_w = hp.ranknet_weight as f32;
    let mse_w = hp.mse_weight as f32;
    let mono_w = hp.monotonicity_reg as f32;
    let mono_margin = hp.monotonicity_margin as f32;

    let start = Instant::now();
    let mut n_batches = 0usize;

    let mut x_host = vec![0.0_f32; two_k * n_features];
    let mut pair_hi_host = vec![0u32; k_batch];
    let mut pair_lo_host = vec![0u32; k_batch];
    let mut delta_target_host = vec![0.0_f32; k_batch];

    // ---- Phase 2 host scratch + persistent GPU handles ----
    let mut aux_x_host = vec![0.0_f32; (2 * k_aux).max(probe_n) * n_features];
    let mut aux_anchor_targets_host = vec![0.0_f32; k_aux];
    let mut aux_anchor_weights_host = vec![0.0_f32; k_aux];
    let mut aux_eq_weights_host = vec![0.0_f32; k_aux];
    let mut aux_eq_butter_host = vec![0.0_f32; k_aux];
    let anchor_w_f32 = hp.anchor_loss_weight as f32;
    let eq_w_f32 = hp.cross_codec_eq_weight as f32;
    let rp_w_f32 = hp.cross_codec_rank_preserve_weight as f32;
    let dr_w_f32 = hp.dynamic_range_floor_weight as f32;
    let dr_sigma_f32 = hp.dynamic_range_sigma_threshold as f32;
    // 4-element reduce output: [μ, σ_obs, grad_scale, loss]
    let sigma_reduce_h = alloc_f32(&zeros(4));

    for _epoch in 0..hp.n_epochs {
        for _step in 0..steps_per_epoch {
            // Host: sample K pairs.
            let mut active = 0usize;
            while active < k_batch {
                let r = rng.next_f64_01();
                let mut gi = 0usize;
                for (k, &c) in std_data.weight_cdf.iter().enumerate() {
                    if r < c {
                        gi = k;
                        break;
                    }
                    gi = k;
                }
                let rows = &std_data.group_rows[gi];
                let scores = &std_data.group_scores[gi];
                let n_rows = scores.len();
                if n_rows < 2 {
                    continue;
                }
                let ia = (rng.next_u64() as usize) % n_rows;
                let mut ib = (rng.next_u64() as usize) % n_rows;
                if ib == ia {
                    ib = (ib + 1) % n_rows;
                }
                let sa = scores[ia];
                let sb = scores[ib];
                if (sa - sb).abs() < 1e-9 {
                    continue;
                }
                let (ihi, ilo) = if sa > sb { (ia, ib) } else { (ib, ia) };
                let hi_off = active * n_features;
                let lo_off = (active + k_batch) * n_features;
                let src_hi_off = ihi * n_features;
                let src_lo_off = ilo * n_features;
                x_host[hi_off..hi_off + n_features]
                    .copy_from_slice(&rows[src_hi_off..src_hi_off + n_features]);
                x_host[lo_off..lo_off + n_features]
                    .copy_from_slice(&rows[src_lo_off..src_lo_off + n_features]);
                pair_hi_host[active] = active as u32;
                pair_lo_host[active] = (active + k_batch) as u32;
                delta_target_host[active] = scores[ihi] - scores[ilo];
                active += 1;
            }

            // Allocate per-batch handles (small/transient buffers).
            let x_batch_h = client.create_from_slice(&f32_bytes(&x_host));
            let pair_hi_h = client.create_from_slice(u32::as_bytes(&pair_hi_host));
            let pair_lo_h = client.create_from_slice(u32::as_bytes(&pair_lo_host));
            let delta_target_h = client.create_from_slice(&f32_bytes(&delta_target_host));

            // ---------- 1. Forward ----------
            unsafe {
                kernels::forward_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(two_k as u32, 1, 1),
                    cube_dim_h,
                    ArrayArg::from_raw_parts(x_batch_h.clone(), two_k * n_features),
                    ArrayArg::from_raw_parts(w1_h.clone(), n_features * n_hidden),
                    ArrayArg::from_raw_parts(b1_h.clone(), n_hidden),
                    ArrayArg::from_raw_parts(rank_w_h.clone(), n_hidden),
                    ArrayArg::from_raw_parts(reducer_w_h.clone(), 4),
                    ArrayArg::from_raw_parts(w_alpha_h.clone(), n_hidden),
                    ArrayArg::from_raw_parts(rank_b_h.clone(), 1),
                    ArrayArg::from_raw_parts(reducer_b_h.clone(), 1),
                    ArrayArg::from_raw_parts(b_alpha_h.clone(), 1),
                    ArrayArg::from_raw_parts(h_pre_h.clone(), two_k * n_hidden),
                    ArrayArg::from_raw_parts(h_h.clone(), two_k * n_hidden),
                    ArrayArg::from_raw_parts(y_rank_h.clone(), two_k),
                    ArrayArg::from_raw_parts(y_pool_h.clone(), two_k),
                    ArrayArg::from_raw_parts(stats_h.clone(), two_k * 4),
                    ArrayArg::from_raw_parts(max_idx_h.clone(), two_k),
                    ArrayArg::from_raw_parts(alpha_h.clone(), two_k),
                    ArrayArg::from_raw_parts(y_pre_h.clone(), two_k),
                    ArrayArg::from_raw_parts(y_score_h.clone(), two_k),
                    n_features as u32,
                    leaky_alpha,
                    tanh_scale,
                    n_hidden as u32,
                );
            }

            // Zero ∂L/∂y_pre buffer before per-pair writes.
            unsafe {
                kernels::zero_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(ceil_div(two_k as u32, 256), 1, 1),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(dl_dypre_h.clone(), two_k),
                );
            }

            // ---------- 2. Loss ----------
            unsafe {
                kernels::loss_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(ceil_div(k_batch as u32, 256), 1, 1),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(y_score_h.clone(), two_k),
                    ArrayArg::from_raw_parts(pair_hi_h.clone(), k_batch),
                    ArrayArg::from_raw_parts(pair_lo_h.clone(), k_batch),
                    ArrayArg::from_raw_parts(delta_target_h.clone(), k_batch),
                    ArrayArg::from_raw_parts(dl_dypre_h.clone(), two_k),
                    ranknet_w,
                    mse_w,
                    mono_w,
                    mono_margin,
                    tanh_scale,
                );
            }

            // ---------- 3. Zero head + b1 grads (atomic) ----------
            unsafe {
                // Re-interpret f32 buffer as Atomic<f32> via the same handle.
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(ceil_div(n_hidden as u32, 256).max(1), 1, 1),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(g_rank_w.clone(), n_hidden),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(4),
                    ArrayArg::from_raw_parts(g_reducer_w.clone(), 4),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(ceil_div(n_hidden as u32, 256).max(1), 1, 1),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(g_w_alpha.clone(), n_hidden),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(g_rank_b.clone(), 1),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(g_reducer_b.clone(), 1),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(1),
                    ArrayArg::from_raw_parts(g_b_alpha.clone(), 1),
                );
                kernels::zero_atomic_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(ceil_div(n_hidden as u32, 256).max(1), 1, 1),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(gb1.clone(), n_hidden),
                );
            }

            // ---------- 4. Backprop heads ----------
            unsafe {
                kernels::backprop_heads_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(two_k as u32, 1, 1),
                    cube_dim_h,
                    ArrayArg::from_raw_parts(h_h.clone(), two_k * n_hidden),
                    ArrayArg::from_raw_parts(h_pre_h.clone(), two_k * n_hidden),
                    ArrayArg::from_raw_parts(stats_h.clone(), two_k * 4),
                    ArrayArg::from_raw_parts(max_idx_h.clone(), two_k),
                    ArrayArg::from_raw_parts(y_rank_h.clone(), two_k),
                    ArrayArg::from_raw_parts(y_pool_h.clone(), two_k),
                    ArrayArg::from_raw_parts(alpha_h.clone(), two_k),
                    ArrayArg::from_raw_parts(dl_dypre_h.clone(), two_k),
                    ArrayArg::from_raw_parts(rank_w_h.clone(), n_hidden),
                    ArrayArg::from_raw_parts(reducer_w_h.clone(), 4),
                    ArrayArg::from_raw_parts(w_alpha_h.clone(), n_hidden),
                    ArrayArg::from_raw_parts(g_rank_w.clone(), n_hidden),
                    ArrayArg::from_raw_parts(g_reducer_w.clone(), 4),
                    ArrayArg::from_raw_parts(g_w_alpha.clone(), n_hidden),
                    ArrayArg::from_raw_parts(g_rank_b.clone(), 1),
                    ArrayArg::from_raw_parts(g_reducer_b.clone(), 1),
                    ArrayArg::from_raw_parts(g_b_alpha.clone(), 1),
                    ArrayArg::from_raw_parts(dh_pre_h.clone(), two_k * n_hidden),
                    leaky_alpha,
                    n_hidden as u32,
                );
            }

            // ---------- 5. Zero gw1 then backprop W1 ----------
            unsafe {
                kernels::zero_f32_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(
                        ceil_div((n_features * n_hidden) as u32, 256).max(1),
                        1,
                        1,
                    ),
                    cube_dim_256,
                    ArrayArg::from_raw_parts(gw1.clone(), n_features * n_hidden),
                );
                kernels::backprop_w1_kernel::launch::<R>(
                    &client,
                    CubeCount::Static(n_features as u32, 1, 1),
                    cube_dim_h,
                    ArrayArg::from_raw_parts(x_batch_h.clone(), two_k * n_features),
                    ArrayArg::from_raw_parts(dh_pre_h.clone(), two_k * n_hidden),
                    ArrayArg::from_raw_parts(gw1.clone(), n_features * n_hidden),
                    ArrayArg::from_raw_parts(gb1.clone(), n_hidden),
                    n_features as u32,
                    two_k as u32,
                    n_hidden as u32,
                );
            }

            // ---------- 5b. Phase 2 aux losses ----------
            //
            // Each aux kernel fires with probability `*_step_p` per
            // minibatch step. On a fire, K_aux samples are drawn and
            // run through forward + backprop; the gradient buffers
            // already populated by the main pair step ACCUMULATE the
            // aux gradients (we do NOT re-zero them). One Adam step
            // at the end of the minibatch consumes the combined
            // signal.
            //
            // Implementation detail: forward_kernel + backprop kernels
            // depend on `B = batch_rows`. Since the scratch buffers
            // are sized for `two_k` and aux uses ≤ `2 * k_aux`, we
            // can reuse them; we just have to launch with the right
            // CubeCount and pass the correct `batch_rows` to
            // backprop_w1.
            if anchor_active && rng.next_f64_01() < hp.anchor_step_p {
                if let Some(std_a) = std_anchor.as_ref() {
                    if std_a.n_rows > 0 && std_a.total_weight > 0.0 {
                        let ctx = AuxFireCtx::<R> {
                            client: &client,
                            cube_dim_h,
                            cube_dim_256,
                            n_features,
                            n_hidden,
                            two_k,
                            tanh_scale,
                            leaky_alpha,
                            w1_h: &w1_h,
                            b1_h: &b1_h,
                            rank_w_h: &rank_w_h,
                            reducer_w_h: &reducer_w_h,
                            w_alpha_h: &w_alpha_h,
                            rank_b_h: &rank_b_h,
                            reducer_b_h: &reducer_b_h,
                            b_alpha_h: &b_alpha_h,
                            h_pre_h: &h_pre_h,
                            h_h: &h_h,
                            y_rank_h: &y_rank_h,
                            y_pool_h: &y_pool_h,
                            stats_h: &stats_h,
                            max_idx_h: &max_idx_h,
                            alpha_h: &alpha_h,
                            y_pre_h: &y_pre_h,
                            y_score_h: &y_score_h,
                            dl_dypre_h: &dl_dypre_h,
                            dh_pre_h: &dh_pre_h,
                            gw1: &gw1,
                            gb1: &gb1,
                            g_rank_w: &g_rank_w,
                            g_reducer_w: &g_reducer_w,
                            g_w_alpha: &g_w_alpha,
                            g_rank_b: &g_rank_b,
                            g_reducer_b: &g_reducer_b,
                            g_b_alpha: &g_b_alpha,
                        };
                        fire_anchor_aux::<R>(
                            &ctx,
                            &mut rng,
                            std_a,
                            &mut aux_x_host,
                            &mut aux_anchor_targets_host,
                            &mut aux_anchor_weights_host,
                            k_aux,
                            anchor_w_f32,
                        );
                    }
                }
            }

            if equiv_active && rng.next_f64_01() < hp.cross_codec_eq_step_p {
                if let Some(std_e) = std_equiv.as_ref() {
                    if std_e.n_rows > 0 && std_e.total_weight > 0.0 {
                        let ctx = AuxFireCtx::<R> {
                            client: &client,
                            cube_dim_h,
                            cube_dim_256,
                            n_features,
                            n_hidden,
                            two_k,
                            tanh_scale,
                            leaky_alpha,
                            w1_h: &w1_h,
                            b1_h: &b1_h,
                            rank_w_h: &rank_w_h,
                            reducer_w_h: &reducer_w_h,
                            w_alpha_h: &w_alpha_h,
                            rank_b_h: &rank_b_h,
                            reducer_b_h: &reducer_b_h,
                            b_alpha_h: &b_alpha_h,
                            h_pre_h: &h_pre_h,
                            h_h: &h_h,
                            y_rank_h: &y_rank_h,
                            y_pool_h: &y_pool_h,
                            stats_h: &stats_h,
                            max_idx_h: &max_idx_h,
                            alpha_h: &alpha_h,
                            y_pre_h: &y_pre_h,
                            y_score_h: &y_score_h,
                            dl_dypre_h: &dl_dypre_h,
                            dh_pre_h: &dh_pre_h,
                            gw1: &gw1,
                            gb1: &gb1,
                            g_rank_w: &g_rank_w,
                            g_reducer_w: &g_reducer_w,
                            g_w_alpha: &g_w_alpha,
                            g_rank_b: &g_rank_b,
                            g_reducer_b: &g_reducer_b,
                            g_b_alpha: &g_b_alpha,
                        };
                        fire_equiv_aux::<R>(
                            &ctx,
                            &mut rng,
                            std_e,
                            &mut aux_x_host,
                            &mut aux_eq_weights_host,
                            &mut aux_eq_butter_host,
                            k_aux,
                            eq_w_f32,
                            rp_w_f32,
                        );
                    }
                }
            }

            if sigma_floor_active && rng.next_f64_01() < hp.dynamic_range_step_p {
                if let Some(std_e) = std_equiv.as_ref() {
                    if std_e.n_rows >= probe_n {
                        let ctx = AuxFireCtx::<R> {
                            client: &client,
                            cube_dim_h,
                            cube_dim_256,
                            n_features,
                            n_hidden,
                            two_k,
                            tanh_scale,
                            leaky_alpha,
                            w1_h: &w1_h,
                            b1_h: &b1_h,
                            rank_w_h: &rank_w_h,
                            reducer_w_h: &reducer_w_h,
                            w_alpha_h: &w_alpha_h,
                            rank_b_h: &rank_b_h,
                            reducer_b_h: &reducer_b_h,
                            b_alpha_h: &b_alpha_h,
                            h_pre_h: &h_pre_h,
                            h_h: &h_h,
                            y_rank_h: &y_rank_h,
                            y_pool_h: &y_pool_h,
                            stats_h: &stats_h,
                            max_idx_h: &max_idx_h,
                            alpha_h: &alpha_h,
                            y_pre_h: &y_pre_h,
                            y_score_h: &y_score_h,
                            dl_dypre_h: &dl_dypre_h,
                            dh_pre_h: &dh_pre_h,
                            gw1: &gw1,
                            gb1: &gb1,
                            g_rank_w: &g_rank_w,
                            g_reducer_w: &g_reducer_w,
                            g_w_alpha: &g_w_alpha,
                            g_rank_b: &g_rank_b,
                            g_reducer_b: &g_reducer_b,
                            g_b_alpha: &g_b_alpha,
                        };
                        fire_sigma_floor_aux::<R>(
                            &ctx,
                            &mut rng,
                            std_e,
                            &mut aux_x_host,
                            probe_n,
                            dr_w_f32,
                            dr_sigma_f32,
                            &sigma_reduce_h,
                        );
                    }
                }
            }

            // ---------- 6. L2 ----------
            if l2 > 0.0 {
                l2_add_plain::<R>(&client, &gw1, &w1_h, l2, n_features * n_hidden);
                l2_add_atomic::<R>(&client, &g_rank_w, &rank_w_h, l2, n_hidden);
                l2_add_atomic::<R>(&client, &g_reducer_w, &reducer_w_h, l2, 4);
                l2_add_atomic::<R>(&client, &g_w_alpha, &w_alpha_h, l2, n_hidden);
            }

            // ---------- 7. Adam step ----------
            adam_t += 1;
            let bc1 = 1.0_f32 - 0.9_f32.powi(adam_t as i32);
            let bc2 = 1.0_f32 - 0.999_f32.powi(adam_t as i32);

            adam_plain::<R>(
                &client,
                &w1_h,
                &gw1,
                &mw1,
                &vw1,
                lr,
                bc1,
                bc2,
                n_features * n_hidden,
            );
            adam_atomic::<R>(&client, &b1_h, &gb1, &mb1, &vb1, lr, bc1, bc2, n_hidden);
            adam_atomic::<R>(
                &client, &rank_w_h, &g_rank_w, &m_rank_w, &v_rank_w, lr, bc1, bc2, n_hidden,
            );
            adam_atomic::<R>(
                &client,
                &reducer_w_h,
                &g_reducer_w,
                &m_reducer_w,
                &v_reducer_w,
                lr,
                bc1,
                bc2,
                4,
            );
            adam_atomic::<R>(
                &client, &w_alpha_h, &g_w_alpha, &m_w_alpha, &v_w_alpha, lr, bc1, bc2, n_hidden,
            );
            adam_atomic::<R>(
                &client, &rank_b_h, &g_rank_b, &m_rank_b, &v_rank_b, lr, bc1, bc2, 1,
            );
            adam_atomic::<R>(
                &client,
                &reducer_b_h,
                &g_reducer_b,
                &m_reducer_b,
                &v_reducer_b,
                lr,
                bc1,
                bc2,
                1,
            );
            adam_atomic::<R>(
                &client, &b_alpha_h, &g_b_alpha, &m_b_alpha, &v_b_alpha, lr, bc1, bc2, 1,
            );

            n_batches += 1;
        }
    }

    // Flush remaining GPU work, then read back.
    let _ = client.flush();

    let read_f32 = |h: &cubecl::server::Handle, n: usize| -> Vec<f32> {
        let bytes = client.read_one(h.clone()).expect("GPU read_one failed");
        let mut out = vec![0.0_f32; n];
        let by = f32::from_bytes(&bytes);
        out.copy_from_slice(&by[..n]);
        out
    };

    let w1_out = read_f32(&w1_h, n_features * n_hidden);
    let b1_out = read_f32(&b1_h, n_hidden);
    let rank_w_out = read_f32(&rank_w_h, n_hidden);
    let reducer_w_out = read_f32(&reducer_w_h, 4);
    let w_alpha_out = read_f32(&w_alpha_h, n_hidden);
    let rank_b_out = read_f32(&rank_b_h, 1);
    let reducer_b_out = read_f32(&reducer_b_h, 1);
    let b_alpha_out = read_f32(&b_alpha_h, 1);

    model.w1 = w1_out.iter().map(|&v| v as f64).collect();
    model.b1 = b1_out.iter().map(|&v| v as f64).collect();
    model.rank_w = rank_w_out.iter().map(|&v| v as f64).collect();
    model.reducer_w = [
        reducer_w_out[0] as f64,
        reducer_w_out[1] as f64,
        reducer_w_out[2] as f64,
        reducer_w_out[3] as f64,
    ];
    model.w_alpha = w_alpha_out.iter().map(|&v| v as f64).collect();
    model.rank_b = rank_b_out[0] as f64;
    model.reducer_b = reducer_b_out[0] as f64;
    model.b_alpha = b_alpha_out[0] as f64;

    let elapsed = start.elapsed().as_secs_f64();
    GpuTrainResult {
        model,
        wall_seconds: elapsed,
        n_batches,
    }
}

// ---- helpers ----

fn l2_add_plain<R: Runtime>(
    client: &ComputeClient<R>,
    grad: &cubecl::server::Handle,
    weight: &cubecl::server::Handle,
    l2: f32,
    n: usize,
) {
    let cubes = ((n as u32) + 255) / 256;
    unsafe {
        kernels::l2_add_kernel::launch::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(grad.clone(), n),
            ArrayArg::from_raw_parts(weight.clone(), n),
            2.0_f32 * l2,
        );
    }
}

fn l2_add_atomic<R: Runtime>(
    client: &ComputeClient<R>,
    grad: &cubecl::server::Handle,
    weight: &cubecl::server::Handle,
    l2: f32,
    n: usize,
) {
    let cubes = ((n as u32) + 255) / 256;
    unsafe {
        kernels::l2_add_atomic_kernel::launch::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(grad.clone(), n),
            ArrayArg::from_raw_parts(weight.clone(), n),
            2.0_f32 * l2,
        );
    }
}

fn adam_plain<R: Runtime>(
    client: &ComputeClient<R>,
    w: &cubecl::server::Handle,
    g: &cubecl::server::Handle,
    m: &cubecl::server::Handle,
    v: &cubecl::server::Handle,
    lr: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) {
    let cubes = ((n as u32) + 255) / 256;
    unsafe {
        kernels::adam_step_kernel::launch::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(w.clone(), n),
            ArrayArg::from_raw_parts(g.clone(), n),
            ArrayArg::from_raw_parts(m.clone(), n),
            ArrayArg::from_raw_parts(v.clone(), n),
            lr,
            bc1,
            bc2,
        );
    }
}

fn adam_atomic<R: Runtime>(
    client: &ComputeClient<R>,
    w: &cubecl::server::Handle,
    g: &cubecl::server::Handle,
    m: &cubecl::server::Handle,
    v: &cubecl::server::Handle,
    lr: f32,
    bc1: f32,
    bc2: f32,
    n: usize,
) {
    let cubes = ((n as u32) + 255) / 256;
    unsafe {
        kernels::adam_step_atomic_grad_kernel::launch::<R>(
            client,
            CubeCount::Static(cubes.max(1), 1, 1),
            CubeDim::new_1d(256),
            ArrayArg::from_raw_parts(w.clone(), n),
            ArrayArg::from_raw_parts(g.clone(), n),
            ArrayArg::from_raw_parts(m.clone(), n),
            ArrayArg::from_raw_parts(v.clone(), n),
            lr,
            bc1,
            bc2,
        );
    }
}

// ============================================================================
// Phase 2 aux fire helpers (task #169, 2026-05-19)
// ============================================================================

/// Bundle of references threaded through every aux-fire helper. Keeps
/// the call sites in the main training loop readable.
struct AuxFireCtx<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    cube_dim_h: CubeDim,
    cube_dim_256: CubeDim,
    n_features: usize,
    n_hidden: usize,
    two_k: usize,
    tanh_scale: f32,
    leaky_alpha: f32,
    // Model parameters (read).
    w1_h: &'a cubecl::server::Handle,
    b1_h: &'a cubecl::server::Handle,
    rank_w_h: &'a cubecl::server::Handle,
    reducer_w_h: &'a cubecl::server::Handle,
    w_alpha_h: &'a cubecl::server::Handle,
    rank_b_h: &'a cubecl::server::Handle,
    reducer_b_h: &'a cubecl::server::Handle,
    b_alpha_h: &'a cubecl::server::Handle,
    // Per-batch scratch (reused — shared with main pair step).
    h_pre_h: &'a cubecl::server::Handle,
    h_h: &'a cubecl::server::Handle,
    y_rank_h: &'a cubecl::server::Handle,
    y_pool_h: &'a cubecl::server::Handle,
    stats_h: &'a cubecl::server::Handle,
    max_idx_h: &'a cubecl::server::Handle,
    alpha_h: &'a cubecl::server::Handle,
    y_pre_h: &'a cubecl::server::Handle,
    y_score_h: &'a cubecl::server::Handle,
    dl_dypre_h: &'a cubecl::server::Handle,
    dh_pre_h: &'a cubecl::server::Handle,
    // Gradient accumulators (write — main step already pre-zeroed
    // these; aux ADDS to them).
    gw1: &'a cubecl::server::Handle,
    gb1: &'a cubecl::server::Handle,
    g_rank_w: &'a cubecl::server::Handle,
    g_reducer_w: &'a cubecl::server::Handle,
    g_w_alpha: &'a cubecl::server::Handle,
    g_rank_b: &'a cubecl::server::Handle,
    g_reducer_b: &'a cubecl::server::Handle,
    g_b_alpha: &'a cubecl::server::Handle,
}

/// Sample one row index per the cumulative weight CDF.
fn sample_cdf(rng: &mut SplitMix64, cdf: &[f64]) -> usize {
    let u = rng.next_f64_01();
    let mut idx = cdf.partition_point(|&c| c < u);
    if idx >= cdf.len() {
        idx = cdf.len() - 1;
    }
    idx
}

/// Launch forward_kernel for `b_rows` rows reading from `x_batch_h`.
/// Reuses the persistent scratch handles bound in `ctx`. The y_score
/// output lives at `ctx.y_score_h[0..b_rows]` after the call.
fn launch_forward_aux<R: Runtime>(
    ctx: &AuxFireCtx<R>,
    x_batch_h: &cubecl::server::Handle,
    b_rows: usize,
) {
    unsafe {
        kernels::forward_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(b_rows as u32, 1, 1),
            ctx.cube_dim_h,
            ArrayArg::from_raw_parts(x_batch_h.clone(), b_rows * ctx.n_features),
            ArrayArg::from_raw_parts(ctx.w1_h.clone(), ctx.n_features * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.b1_h.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.rank_w_h.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.reducer_w_h.clone(), 4),
            ArrayArg::from_raw_parts(ctx.w_alpha_h.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.rank_b_h.clone(), 1),
            ArrayArg::from_raw_parts(ctx.reducer_b_h.clone(), 1),
            ArrayArg::from_raw_parts(ctx.b_alpha_h.clone(), 1),
            ArrayArg::from_raw_parts(ctx.h_pre_h.clone(), b_rows * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.h_h.clone(), b_rows * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.y_rank_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.y_pool_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.stats_h.clone(), b_rows * 4),
            ArrayArg::from_raw_parts(ctx.max_idx_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.alpha_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.y_pre_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.y_score_h.clone(), b_rows),
            ctx.n_features as u32,
            ctx.leaky_alpha,
            ctx.tanh_scale,
            ctx.n_hidden as u32,
        );
    }
}

/// Launch backprop_heads then backprop_w1 for `b_rows` rows. Both
/// kernels ADD to the gradient accumulators; aux fires after the
/// main pair step (which pre-zeroed gw1) and the main step also
/// pre-zeroed the atomic gradients via `zero_atomic_f32_kernel`, so
/// we just keep accumulating.
fn launch_backprop_aux<R: Runtime>(
    ctx: &AuxFireCtx<R>,
    x_batch_h: &cubecl::server::Handle,
    b_rows: usize,
) {
    unsafe {
        kernels::backprop_heads_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(b_rows as u32, 1, 1),
            ctx.cube_dim_h,
            ArrayArg::from_raw_parts(ctx.h_h.clone(), b_rows * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.h_pre_h.clone(), b_rows * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.stats_h.clone(), b_rows * 4),
            ArrayArg::from_raw_parts(ctx.max_idx_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.y_rank_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.y_pool_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.alpha_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), b_rows),
            ArrayArg::from_raw_parts(ctx.rank_w_h.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.reducer_w_h.clone(), 4),
            ArrayArg::from_raw_parts(ctx.w_alpha_h.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.g_rank_w.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.g_reducer_w.clone(), 4),
            ArrayArg::from_raw_parts(ctx.g_w_alpha.clone(), ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.g_rank_b.clone(), 1),
            ArrayArg::from_raw_parts(ctx.g_reducer_b.clone(), 1),
            ArrayArg::from_raw_parts(ctx.g_b_alpha.clone(), 1),
            ArrayArg::from_raw_parts(ctx.dh_pre_h.clone(), b_rows * ctx.n_hidden),
            ctx.leaky_alpha,
            ctx.n_hidden as u32,
        );
        kernels::backprop_w1_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ctx.n_features as u32, 1, 1),
            ctx.cube_dim_h,
            ArrayArg::from_raw_parts(x_batch_h.clone(), b_rows * ctx.n_features),
            ArrayArg::from_raw_parts(ctx.dh_pre_h.clone(), b_rows * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.gw1.clone(), ctx.n_features * ctx.n_hidden),
            ArrayArg::from_raw_parts(ctx.gb1.clone(), ctx.n_hidden),
            ctx.n_features as u32,
            b_rows as u32,
            ctx.n_hidden as u32,
        );
    }
}

/// Anchor MSE aux fire — K_aux rows.
#[allow(clippy::too_many_arguments)]
fn fire_anchor_aux<R: Runtime>(
    ctx: &AuxFireCtx<R>,
    rng: &mut SplitMix64,
    std_a: &StdAnchor,
    aux_x_host: &mut [f32],
    aux_targets_host: &mut [f32],
    aux_weights_host: &mut [f32],
    k_aux: usize,
    w_anchor: f32,
) {
    let nf = ctx.n_features;
    let k = k_aux.min(std_a.n_rows);
    if k == 0 {
        return;
    }
    // Host-side sample: K rows from anchor pool by row_cdf.
    for r in 0..k {
        let ai = sample_cdf(rng, &std_a.row_cdf);
        let src = &std_a.rows[ai * nf..(ai + 1) * nf];
        aux_x_host[r * nf..(r + 1) * nf].copy_from_slice(src);
        aux_targets_host[r] = std_a.targets[ai];
        aux_weights_host[r] = std_a.weights[ai];
    }
    let x_batch_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_x_host[..k * nf]));
    let targets_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_targets_host[..k]));
    let weights_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_weights_host[..k]));

    // Forward.
    launch_forward_aux::<R>(ctx, &x_batch_h, k);
    // dl_dypre[0..k] writeback. Zero the slot first (kernel writes
    // directly per row but it's good hygiene since the buffer is
    // shared with the main step).
    unsafe {
        kernels::zero_f32_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(k as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), k),
        );
        kernels::anchor_loss_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(k as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.y_score_h.clone(), k),
            ArrayArg::from_raw_parts(targets_h.clone(), k),
            ArrayArg::from_raw_parts(weights_h.clone(), k),
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), k),
            w_anchor,
            ctx.tanh_scale,
        );
    }
    // Backprop heads + W1 — ACCUMULATES into existing grad buffers.
    launch_backprop_aux::<R>(ctx, &x_batch_h, k);
    let _ = (ctx.two_k,); // sanity touch (silence unused warn on field)
}

/// Cross-codec equivalence aux fire — K_aux pairs (2*K_aux rows).
#[allow(clippy::too_many_arguments)]
fn fire_equiv_aux<R: Runtime>(
    ctx: &AuxFireCtx<R>,
    rng: &mut SplitMix64,
    std_e: &StdEquiv,
    aux_x_host: &mut [f32],
    aux_weights_host: &mut [f32],
    aux_butter_host: &mut [f32],
    k_aux: usize,
    w_eq: f32,
    w_rp: f32,
) {
    let nf = ctx.n_features;
    let k = k_aux.min(std_e.n_rows);
    if k == 0 {
        return;
    }
    let two_k_b = 2 * k;
    let rp_enabled = w_rp > 0.0 && !std_e.butter_diff.is_empty();
    // Host-side sample: K pairs. A-side rows 0..K, B-side rows K..2K.
    for r in 0..k {
        let ei = sample_cdf(rng, &std_e.row_cdf);
        let src_a = &std_e.rows_a[ei * nf..(ei + 1) * nf];
        let src_b = &std_e.rows_b[ei * nf..(ei + 1) * nf];
        aux_x_host[r * nf..(r + 1) * nf].copy_from_slice(src_a);
        let b_off = (r + k) * nf;
        aux_x_host[b_off..b_off + nf].copy_from_slice(src_b);
        aux_weights_host[r] = std_e.weights[ei];
        aux_butter_host[r] = if rp_enabled { std_e.butter_diff[ei] } else { 0.0 };
    }
    let x_batch_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_x_host[..two_k_b * nf]));
    let weights_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_weights_host[..k]));
    let butter_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_butter_host[..k]));

    // Forward 2K rows.
    launch_forward_aux::<R>(ctx, &x_batch_h, two_k_b);
    // Zero dl_dypre slot then run loss kernel (one thread per pair).
    unsafe {
        kernels::zero_f32_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(two_k_b as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), two_k_b),
        );
        kernels::cross_codec_eq_loss_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(k as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.y_score_h.clone(), two_k_b),
            ArrayArg::from_raw_parts(weights_h.clone(), k),
            ArrayArg::from_raw_parts(butter_h.clone(), k),
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), two_k_b),
            k as u32,
            w_eq,
            w_rp,
            ctx.tanh_scale,
        );
    }
    launch_backprop_aux::<R>(ctx, &x_batch_h, two_k_b);
}

/// σ-floor probe aux fire — N_probe rows from the equiv A-side pool.
#[allow(clippy::too_many_arguments)]
fn fire_sigma_floor_aux<R: Runtime>(
    ctx: &AuxFireCtx<R>,
    rng: &mut SplitMix64,
    std_e: &StdEquiv,
    aux_x_host: &mut [f32],
    probe_n: usize,
    w_dr: f32,
    sigma_threshold: f32,
    sigma_reduce_h: &cubecl::server::Handle,
) {
    let nf = ctx.n_features;
    let n_probe = probe_n.min(std_e.n_rows);
    if n_probe < 2 {
        return;
    }
    // Sample N_probe distinct(-ish — with replacement is fine) A-side
    // rows uniformly. CPU uses random unit per row.
    for r in 0..n_probe {
        let u = rng.next_f64_01();
        let pi = ((u * std_e.n_rows as f64) as usize).min(std_e.n_rows - 1);
        let src = &std_e.rows_a[pi * nf..(pi + 1) * nf];
        aux_x_host[r * nf..(r + 1) * nf].copy_from_slice(src);
    }
    let x_batch_h = ctx
        .client
        .create_from_slice(f32::as_bytes(&aux_x_host[..n_probe * nf]));

    launch_forward_aux::<R>(ctx, &x_batch_h, n_probe);

    // Reduction kernel writes (μ, σ_obs, grad_scale, loss) into the
    // 4-element reduce buffer. We always launch — the kernel handles
    // the no-violation case by writing grad_scale=0. The subsequent
    // sigma_floor_grad_kernel propagates that as a no-op via the
    // per-row multiply.
    unsafe {
        kernels::sigma_floor_reduce_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(ctx.y_score_h.clone(), n_probe),
            ArrayArg::from_raw_parts(sigma_reduce_h.clone(), 4),
            n_probe as u32,
            sigma_threshold,
            w_dr,
        );
        // Zero dl_dypre slot for the probe range.
        kernels::zero_f32_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(n_probe as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), n_probe),
        );
        kernels::sigma_floor_grad_kernel::launch::<R>(
            ctx.client,
            CubeCount::Static(ceil_div(n_probe as u32, 256).max(1), 1, 1),
            ctx.cube_dim_256,
            ArrayArg::from_raw_parts(ctx.y_score_h.clone(), n_probe),
            ArrayArg::from_raw_parts(sigma_reduce_h.clone(), 4),
            ArrayArg::from_raw_parts(ctx.dl_dypre_h.clone(), n_probe),
            ctx.tanh_scale,
        );
    }
    launch_backprop_aux::<R>(ctx, &x_batch_h, n_probe);
}
