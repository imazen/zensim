//! Backend dispatch + training loop orchestration.
//!
//! The actual training loop is generic over `R: Runtime` so the same
//! kernel code runs on every backend (CUDA / WGPU / CPU).

use cubecl::prelude::*;
use std::time::Instant;
use zensim_train_core::TrainingGroup;
use zensim_train_core::per_sample_alpha_head::PerSampleAlphaHeadModel;

use crate::kernels;
use crate::{GpuHparams, GpuRuntime, GpuTrainResult};

pub(crate) fn dispatch(
    groups: &[TrainingGroup<'_>],
    hp: &GpuHparams,
    n_features: usize,
    runtime: GpuRuntime,
) -> GpuTrainResult {
    match runtime {
        #[cfg(feature = "gpu-cuda")]
        GpuRuntime::Cuda => {
            let client =
                <cubecl::cuda::CudaRuntime as Runtime>::client(&Default::default());
            run::<cubecl::cuda::CudaRuntime>(client, groups, hp, n_features)
        }
        #[cfg(not(feature = "gpu-cuda"))]
        GpuRuntime::Cuda => panic!(
            "zensim-train-gpu: --gpu-runtime cuda requested but compiled without --features gpu-cuda"
        ),
        #[cfg(feature = "gpu-wgpu")]
        GpuRuntime::Wgpu => {
            let client =
                <cubecl::wgpu::WgpuRuntime as Runtime>::client(&Default::default());
            run::<cubecl::wgpu::WgpuRuntime>(client, groups, hp, n_features)
        }
        #[cfg(not(feature = "gpu-wgpu"))]
        GpuRuntime::Wgpu => panic!(
            "zensim-train-gpu: --gpu-runtime wgpu requested but compiled without --features gpu-wgpu"
        ),
        #[cfg(feature = "gpu-cpu")]
        GpuRuntime::Cpu => {
            let client =
                <cubecl::cpu::CpuRuntime as Runtime>::client(&Default::default());
            run::<cubecl::cpu::CpuRuntime>(client, groups, hp, n_features)
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
) -> GpuTrainResult {
    let n_hidden = hp.n_hidden;
    let k_batch = hp.minibatch_k;
    let two_k = 2 * k_batch;
    assert!(n_hidden <= 1024, "n_hidden must be ≤ 1024 (CUDA cube limit); got {n_hidden}");
    assert!(k_batch >= 1, "minibatch_k must be ≥ 1; got {k_batch}");

    let train_indices: Vec<usize> = (0..groups.len())
        .filter(|&i| groups[i].train_weight > 0.0)
        .collect();
    assert!(
        !train_indices.is_empty(),
        "train_per_sample_alpha_head_gpu: need at least one group with train_weight > 0"
    );

    let (mean, scale) = compute_scaler(groups, &train_indices, n_features);
    let std_data = standardize_groups(groups, &train_indices, &mean, &scale, n_features);

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
