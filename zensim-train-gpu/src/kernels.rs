//! CubeCL kernels for per-sample-α head training (Phase 1).
//!
//! All kernels use `f32` buffers. Gradients accumulate via
//! `Atomic<f32>::fetch_add` — supported on CUDA, dropped on WGPU Metal.
//!
//! ## Layout conventions
//!
//! With `K` pairs per minibatch and `B = 2·K` rows per forward pass:
//!
//! - `x_batch`: `[B × n_features]` row-major, pre-standardized.
//!   Layout: rows `0..K` are "hi", rows `K..2K` are "lo".
//! - `w1`: `[n_features × n_hidden]` row-major (matches CPU layout).
//! - `b1`: `[n_hidden]`.
//! - `h_pre`, `h`: `[B × n_hidden]`.
//! - `rank_w`, `w_alpha`: `[n_hidden]`.
//! - `reducer_w`: `[4]` (μ, σ, max, p6 coefficients).
//! - Per-row scalar outputs: `y_rank, y_pool, alpha, y_pre, y_score`
//!   each `[B]`.
//! - Per-pair targets: `delta_target` `[K]`.

use cubecl::prelude::*;

/// Pool-stat power norm — mirrors POOL_P_NORM = 6.0 in CPU.
pub const POOL_P_NORM: f32 = 6.0;
/// Floor used to suppress σ-divide blowup on near-constant h.
pub const POOL_STD_FLOOR: f32 = 1e-6;

/// Forward pass — one cube per row, CUBE_DIM_X = n_hidden.
///
/// Each thread j computes one h[b, j]; thread 0 does the per-row
/// reductions (μ, σ, max, p6) and head outputs after a `sync_cube()`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn forward_kernel(
    x_batch: &Array<f32>,
    w1: &Array<f32>,
    b1: &Array<f32>,
    rank_w: &Array<f32>,
    reducer_w: &Array<f32>,
    w_alpha: &Array<f32>,
    rank_b: &Array<f32>,
    reducer_b: &Array<f32>,
    b_alpha: &Array<f32>,
    h_pre_out: &mut Array<f32>,
    h_out: &mut Array<f32>,
    y_rank_out: &mut Array<f32>,
    y_pool_out: &mut Array<f32>,
    stats_out: &mut Array<f32>,
    max_idx_out: &mut Array<u32>,
    alpha_out: &mut Array<f32>,
    y_pre_out: &mut Array<f32>,
    y_score_out: &mut Array<f32>,
    n_features: u32,
    leaky_alpha: f32,
    tanh_scale: f32,
    #[comptime] n_hidden: u32,
) {
    let b = CUBE_POS_X;
    let j = UNIT_POS_X;
    if j >= n_hidden {
        terminate!();
    }

    let nh_us = n_hidden as usize;
    let nf_us = n_features as usize;
    let row_off = b as usize * nh_us + j as usize;
    let x_row_off = b as usize * nf_us;
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let clamp_pos = f32::new(20.0);
    let clamp_neg = f32::new(-20.0);

    // Step 1: per-thread accumulation for h_pre[b, j]
    let mut acc = b1[j as usize];
    let mut i_u = 0u32;
    while i_u < n_features {
        let xi = x_batch[x_row_off + i_u as usize];
        acc += xi * w1[i_u as usize * nh_us + j as usize];
        i_u += 1u32;
    }
    h_pre_out[row_off] = acc;
    let h_val = if acc >= zero { acc } else { leaky_alpha * acc };
    h_out[row_off] = h_val;

    sync_cube();

    if j == 0u32 {
        let row_h_off = b as usize * nh_us;
        let n_h_f = f32::cast_from(n_hidden);

        // y_rank
        let mut yr = rank_b[0];
        let mut k_u = 0u32;
        while k_u < n_hidden {
            yr += h_out[row_h_off + k_u as usize] * rank_w[k_u as usize];
            k_u += 1u32;
        }

        // Pool stats: μ, σ, max(h), p6
        let mut sum_h = zero;
        let mut max_h = h_out[row_h_off];
        let mut max_i = 0u32;
        let mut sum_h6 = zero;
        k_u = 0u32;
        while k_u < n_hidden {
            let hv = h_out[row_h_off + k_u as usize];
            sum_h += hv;
            if hv > max_h {
                max_h = hv;
                max_i = k_u;
            }
            let abs_h = if hv >= zero { hv } else { -hv };
            let h2 = abs_h * abs_h;
            sum_h6 += h2 * h2 * h2;
            k_u += 1u32;
        }
        let mu = sum_h / n_h_f;

        let mut sum_var = zero;
        k_u = 0u32;
        while k_u < n_hidden {
            let d = h_out[row_h_off + k_u as usize] - mu;
            sum_var += d * d;
            k_u += 1u32;
        }
        let var = sum_var / n_h_f;
        let var_safe = if var > zero { var } else { zero };
        let floor_sq = f32::new(POOL_STD_FLOOR * POOL_STD_FLOOR);
        let sigma = if var_safe > floor_sq {
            f32::sqrt(var_safe)
        } else {
            f32::new(POOL_STD_FLOOR)
        };

        let p6_inner = sum_h6 / n_h_f;
        let inv_p_norm = f32::new(1.0 / POOL_P_NORM);
        let p6 = if p6_inner > f32::new(1e-20) {
            f32::powf(p6_inner, inv_p_norm)
        } else {
            zero
        };

        let stats_off = b as usize * 4;
        stats_out[stats_off] = mu;
        stats_out[stats_off + 1] = sigma;
        stats_out[stats_off + 2] = max_h;
        stats_out[stats_off + 3] = p6;
        max_idx_out[b as usize] = max_i;

        let yp = mu * reducer_w[0]
            + sigma * reducer_w[1]
            + max_h * reducer_w[2]
            + p6 * reducer_w[3]
            + reducer_b[0];

        let mut al = b_alpha[0];
        k_u = 0u32;
        while k_u < n_hidden {
            al += h_out[row_h_off + k_u as usize] * w_alpha[k_u as usize];
            k_u += 1u32;
        }
        let al_c = if al > clamp_pos {
            clamp_pos
        } else if al < clamp_neg {
            clamp_neg
        } else {
            al
        };
        let alpha = one / (one + f32::exp(-al_c));
        let y_pre = alpha * yr + (one - alpha) * yp;

        let b_u = b as usize;
        y_rank_out[b_u] = yr;
        y_pool_out[b_u] = yp;
        alpha_out[b_u] = alpha;
        y_pre_out[b_u] = y_pre;

        if tanh_scale > zero {
            let z = y_pre / tanh_scale;
            let z_c = if z > clamp_pos {
                clamp_pos
            } else if z < clamp_neg {
                clamp_neg
            } else {
                z
            };
            let s = one / (one + f32::exp(-z_c));
            y_score_out[b_u] = f32::new(100.0) * s;
        } else {
            y_score_out[b_u] = y_pre;
        }
    }
}

/// Per-pair loss kernel — one thread per pair.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn loss_kernel(
    y_score: &Array<f32>,
    pair_hi: &Array<u32>,
    pair_lo: &Array<u32>,
    delta_target: &Array<f32>,
    dl_dypre_per_b: &mut Array<f32>,
    ranknet_w: f32,
    mse_w: f32,
    mono_w: f32,
    mono_margin: f32,
    tanh_scale: f32,
) {
    let k = ABSOLUTE_POS;
    let n_pairs = pair_hi.len();
    if k >= n_pairs {
        terminate!();
    }
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let two = f32::new(2.0);
    let clamp_pos = f32::new(20.0);
    let clamp_neg = f32::new(-20.0);

    let ihi = pair_hi[k] as usize;
    let ilo = pair_lo[k] as usize;
    let yhi_s = y_score[ihi];
    let ylo_s = y_score[ilo];
    let d_s = yhi_s - ylo_s;

    let d_c = if d_s > clamp_pos {
        clamp_pos
    } else if d_s < clamp_neg {
        clamp_neg
    } else {
        d_s
    };
    let neg_sig = one / (one + f32::exp(d_c));
    let rn_grad = -neg_sig * ranknet_w;

    let dt = delta_target[k];
    let mse_grad_hi = two * mse_w * (d_s - dt);

    let mono_active_pos = mono_margin - d_s;
    let mono_grad_hi = if mono_w > zero {
        if mono_active_pos > zero {
            -mono_w
        } else {
            zero
        }
    } else {
        zero
    };

    let dl_dyhi_s = rn_grad + mse_grad_hi + mono_grad_hi;
    let dl_dylo_s = -dl_dyhi_s;

    // Chain through tanh-pin Jacobian if active.
    let dl_dypre_hi = if tanh_scale > zero {
        let s_hi = yhi_s / f32::new(100.0);
        let jac_hi = (f32::new(100.0) / tanh_scale) * s_hi * (one - s_hi);
        dl_dyhi_s * jac_hi
    } else {
        dl_dyhi_s
    };
    let dl_dypre_lo = if tanh_scale > zero {
        let s_lo = ylo_s / f32::new(100.0);
        let jac_lo = (f32::new(100.0) / tanh_scale) * s_lo * (one - s_lo);
        dl_dylo_s * jac_lo
    } else {
        dl_dylo_s
    };

    dl_dypre_per_b[ihi] = dl_dypre_hi;
    dl_dypre_per_b[ilo] = dl_dypre_lo;
}

/// Head + pool backprop — one cube per batch row, threads = n_hidden.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn backprop_heads_kernel(
    h: &Array<f32>,
    h_pre: &Array<f32>,
    stats: &Array<f32>,
    max_idx: &Array<u32>,
    y_rank: &Array<f32>,
    y_pool: &Array<f32>,
    alpha: &Array<f32>,
    dl_dypre: &Array<f32>,
    rank_w: &Array<f32>,
    reducer_w: &Array<f32>,
    w_alpha: &Array<f32>,
    g_rank_w: &mut Array<Atomic<f32>>,
    g_reducer_w: &mut Array<Atomic<f32>>,
    g_w_alpha: &mut Array<Atomic<f32>>,
    g_rank_b: &mut Array<Atomic<f32>>,
    g_reducer_b: &mut Array<Atomic<f32>>,
    g_b_alpha: &mut Array<Atomic<f32>>,
    dh_pre_out: &mut Array<f32>,
    leaky_alpha: f32,
    #[comptime] n_hidden: u32,
) {
    let b = CUBE_POS_X;
    let j = UNIT_POS_X;
    if j >= n_hidden {
        terminate!();
    }
    let zero = f32::new(0.0);
    let one = f32::new(1.0);
    let nh_us = n_hidden as usize;
    let row_off = b as usize * nh_us + j as usize;
    let n_h_f = f32::cast_from(n_hidden);
    let b_u = b as usize;

    let dl_dy = dl_dypre[b_u];
    let a = alpha[b_u];
    let dl_dy_rank = dl_dy * a;
    let dl_dy_pool = dl_dy * (one - a);
    let dl_da = dl_dy * (y_rank[b_u] - y_pool[b_u]);
    let dl_dal = dl_da * a * (one - a);
    let hj = h[row_off];

    g_rank_w[j as usize].fetch_add(dl_dy_rank * hj);
    g_w_alpha[j as usize].fetch_add(dl_dal * hj);

    if j == 0u32 {
        g_rank_b[0].fetch_add(dl_dy_rank);
        g_reducer_b[0].fetch_add(dl_dy_pool);
        g_b_alpha[0].fetch_add(dl_dal);
        let stats_off = b_u * 4;
        g_reducer_w[0].fetch_add(dl_dy_pool * stats[stats_off]);
        g_reducer_w[1].fetch_add(dl_dy_pool * stats[stats_off + 1]);
        g_reducer_w[2].fetch_add(dl_dy_pool * stats[stats_off + 2]);
        g_reducer_w[3].fetch_add(dl_dy_pool * stats[stats_off + 3]);
    }

    let stats_off = b_u * 4;
    let mu = stats[stats_off];
    let sigma = stats[stats_off + 1];
    let p6 = stats[stats_off + 3];
    let mxi = max_idx[b_u];

    let dstat_mu = dl_dy_pool * reducer_w[0];
    let dstat_sigma = dl_dy_pool * reducer_w[1];
    let dstat_max = dl_dy_pool * reducer_w[2];
    let dstat_p6 = dl_dy_pool * reducer_w[3];

    let sigma_floor_check = f32::new(POOL_STD_FLOOR + 1e-7);
    let inv_sigma_n = if sigma > sigma_floor_check {
        one / (n_h_f * sigma)
    } else {
        zero
    };
    let p6_floor_thresh = f32::new(1e-12);
    let p6_floor = if p6 > p6_floor_thresh {
        p6
    } else {
        p6_floor_thresh
    };
    let inv_p6_pow5_n = one / (n_h_f * p6_floor * p6_floor * p6_floor * p6_floor * p6_floor);

    let abs_hj = if hj >= zero { hj } else { -hj };
    let sign_hj = if hj >= zero { one } else { -one };
    let abs_pow5 = abs_hj * abs_hj * abs_hj * abs_hj * abs_hj;

    let dh_max_contrib = if j == mxi { dstat_max } else { zero };

    let dh_j = dl_dy_rank * rank_w[j as usize]
        + dl_dal * w_alpha[j as usize]
        + dstat_mu / n_h_f
        + dstat_sigma * (hj - mu) * inv_sigma_n
        + dh_max_contrib
        + dstat_p6 * sign_hj * abs_pow5 * inv_p6_pow5_n;

    let hp_val = h_pre[row_off];
    let dh_pre_j = if hp_val >= zero {
        dh_j
    } else {
        leaky_alpha * dh_j
    };
    dh_pre_out[row_off] = dh_pre_j;
}

/// W1 + B1 backprop — outer product. Each thread (i, j) owns gw1[i, j].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn backprop_w1_kernel(
    x_batch: &Array<f32>,
    dh_pre: &Array<f32>,
    gw1: &mut Array<f32>,
    gb1: &mut Array<Atomic<f32>>,
    n_features: u32,
    batch_rows: u32,
    #[comptime] n_hidden: u32,
) {
    let i = CUBE_POS_X;
    let j = UNIT_POS_X;
    if i >= n_features {
        terminate!();
    }
    if j >= n_hidden {
        terminate!();
    }
    let nh_us = n_hidden as usize;
    let nf_us = n_features as usize;
    let zero = f32::new(0.0);

    let mut acc = zero;
    let mut acc_b = zero;
    let mut b_u = 0u32;
    while b_u < batch_rows {
        let dh_bj = dh_pre[b_u as usize * nh_us + j as usize];
        acc += x_batch[b_u as usize * nf_us + i as usize] * dh_bj;
        if i == 0u32 {
            acc_b += dh_bj;
        }
        b_u += 1u32;
    }
    gw1[i as usize * nh_us + j as usize] += acc;
    if i == 0u32 {
        gb1[j as usize].fetch_add(acc_b);
    }
}

/// Adam step — plain (non-atomic) gradient.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn adam_step_kernel(
    w: &mut Array<f32>,
    g: &mut Array<f32>,
    m: &mut Array<f32>,
    v: &mut Array<f32>,
    lr: f32,
    bc1: f32,
    bc2: f32,
) {
    let k = ABSOLUTE_POS;
    let n = w.len();
    if k >= n {
        terminate!();
    }
    let beta1 = f32::new(0.9);
    let beta2 = f32::new(0.999);
    let eps = f32::new(1e-8);
    let one = f32::new(1.0);
    let zero = f32::new(0.0);

    let g_k = g[k];
    let m_new = beta1 * m[k] + (one - beta1) * g_k;
    let v_new = beta2 * v[k] + (one - beta2) * g_k * g_k;
    m[k] = m_new;
    v[k] = v_new;
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;
    w[k] -= lr * m_hat / (f32::sqrt(v_hat) + eps);
    g[k] = zero;
}

/// Adam step — atomic gradient (loads via Atomic::load).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn adam_step_atomic_grad_kernel(
    w: &mut Array<f32>,
    g: &mut Array<Atomic<f32>>,
    m: &mut Array<f32>,
    v: &mut Array<f32>,
    lr: f32,
    bc1: f32,
    bc2: f32,
) {
    let k = ABSOLUTE_POS;
    let n = w.len();
    if k >= n {
        terminate!();
    }
    let beta1 = f32::new(0.9);
    let beta2 = f32::new(0.999);
    let eps = f32::new(1e-8);
    let one = f32::new(1.0);
    let zero = f32::new(0.0);

    let g_k = g[k].load();
    let m_new = beta1 * m[k] + (one - beta1) * g_k;
    let v_new = beta2 * v[k] + (one - beta2) * g_k * g_k;
    m[k] = m_new;
    v[k] = v_new;
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;
    w[k] -= lr * m_hat / (f32::sqrt(v_hat) + eps);
    g[k].store(zero);
}

/// Element-wise zero for plain f32 buffers.
#[cube(launch)]
pub fn zero_f32_kernel(buf: &mut Array<f32>) {
    let k = ABSOLUTE_POS;
    let n = buf.len();
    if k >= n {
        terminate!();
    }
    buf[k] = f32::new(0.0);
}

/// Element-wise zero for atomic-f32 buffers.
#[cube(launch)]
pub fn zero_atomic_f32_kernel(buf: &mut Array<Atomic<f32>>) {
    let k = ABSOLUTE_POS;
    let n = buf.len();
    if k >= n {
        terminate!();
    }
    buf[k].store(f32::new(0.0));
}

/// L2 regularizer: `grad[k] += 2·λ·weight[k]`.
#[cube(launch)]
pub fn l2_add_kernel(grad: &mut Array<f32>, weight: &Array<f32>, l2_two: f32) {
    let k = ABSOLUTE_POS;
    let n = grad.len();
    if k >= n {
        terminate!();
    }
    grad[k] += l2_two * weight[k];
}

/// L2 regularizer (atomic gradient).
#[cube(launch)]
pub fn l2_add_atomic_kernel(grad: &mut Array<Atomic<f32>>, weight: &Array<f32>, l2_two: f32) {
    let k = ABSOLUTE_POS;
    let n = grad.len();
    if k >= n {
        terminate!();
    }
    grad[k].fetch_add(l2_two * weight[k]);
}
