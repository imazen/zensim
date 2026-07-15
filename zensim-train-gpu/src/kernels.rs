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
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
    let clamp_pos = f32::new(20.0_f32);
    let clamp_neg = f32::new(-20.0_f32);

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
        let p6 = if p6_inner > f32::new(1e-20_f32) {
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
            y_score_out[b_u] = f32::new(100.0_f32) * s;
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
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
    let two = f32::new(2.0_f32);
    let clamp_pos = f32::new(20.0_f32);
    let clamp_neg = f32::new(-20.0_f32);

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
        let s_hi = yhi_s / f32::new(100.0_f32);
        let jac_hi = (f32::new(100.0_f32) / tanh_scale) * s_hi * (one - s_hi);
        dl_dyhi_s * jac_hi
    } else {
        dl_dyhi_s
    };
    let dl_dypre_lo = if tanh_scale > zero {
        let s_lo = ylo_s / f32::new(100.0_f32);
        let jac_lo = (f32::new(100.0_f32) / tanh_scale) * s_lo * (one - s_lo);
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
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
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
    let p6_floor_thresh = f32::new(1e-12_f32);
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
    let zero = f32::new(0.0_f32);

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
    let beta1 = f32::new(0.9_f32);
    let beta2 = f32::new(0.999_f32);
    let eps = f32::new(1e-8_f32);
    let one = f32::new(1.0_f32);
    let zero = f32::new(0.0_f32);

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
    let beta1 = f32::new(0.9_f32);
    let beta2 = f32::new(0.999_f32);
    let eps = f32::new(1e-8_f32);
    let one = f32::new(1.0_f32);
    let zero = f32::new(0.0_f32);

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
    buf[k] = f32::new(0.0_f32);
}

/// Element-wise zero for atomic-f32 buffers.
#[cube(launch)]
pub fn zero_atomic_f32_kernel(buf: &mut Array<Atomic<f32>>) {
    let k = ABSOLUTE_POS;
    let n = buf.len();
    if k >= n {
        terminate!();
    }
    buf[k].store(f32::new(0.0_f32));
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

// ============================================================================
// Phase 2 aux loss kernels (task #169, 2026-05-19)
// ============================================================================

/// Anchor loss kernel — one thread per row.
///
/// CPU equivalent (`zensim-validate::mlp_train::train_mlp_per_sample_alpha_head`
/// lines ~5680-5770): per row, with probability `anchor_step_p`, sample one
/// anchor feature vector + per-row target_score + per-row weight, forward
/// through the MLP, and apply a weighted MSE pull toward the target.
///
/// We K-batch this on GPU: `K_anchor` rows are forwarded in a single pass,
/// then this kernel writes `dl_dypre[k]` for each row. The same
/// `backprop_heads_kernel` + `backprop_w1_kernel` used by the main pair
/// step then chains gradients through the network. All aux gradients
/// accumulate into the same parameter-gradient buffers as the main pair
/// step; one Adam update absorbs the combined signal per minibatch.
///
/// Loss per row: `L = w · row_w · (y_score - target)^2`.
/// Gradient: `dL/dy_score = 2 · w · row_w · (y_score - target)`.
/// Chained through optional tanh-pin: `dL/dy_pre = dL/dy_score · dy/dy_pre`
/// where `dy/dy_pre = (100/scale) · σ · (1 - σ)` and `σ = y_score / 100`.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn anchor_loss_kernel(
    y_score: &Array<f32>,
    target_score: &Array<f32>,
    row_weight: &Array<f32>,
    dl_dypre_per_b: &mut Array<f32>,
    w_anchor: f32,
    tanh_scale: f32,
) {
    let k = ABSOLUTE_POS;
    let n = y_score.len();
    if k >= n {
        terminate!();
    }
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
    let two = f32::new(2.0_f32);

    let y_s = y_score[k];
    let tgt = target_score[k];
    let rw = row_weight[k];
    let err = y_s - tgt;
    let dl_dy_s = two * w_anchor * rw * err;

    let dl_dypre = if tanh_scale > zero {
        let s = y_s / f32::new(100.0_f32);
        let jac = (f32::new(100.0_f32) / tanh_scale) * s * (one - s);
        dl_dy_s * jac
    } else {
        dl_dy_s
    };
    dl_dypre_per_b[k] = dl_dypre;
}

/// Cross-codec equivalence loss kernel — one thread per pair (K).
///
/// Buffer layout: A-side rows are indices `0..K`, B-side rows are indices
/// `K..2K`. The forward pass has already populated `y_score[0..2K]`.
///
/// CPU equivalent (lines ~5780-5940): per pair, with probability
/// `cross_codec_eq_step_p`, forward both A and B, accumulate
///   `L_eq = w · row_w · (y_a - y_b)^2`
/// plus a butter-weighted rank-preserve term when |butter_diff| > 0:
///   `L_rp = (w_rp · |Δb|) · softplus(-s · (y_b - y_a))`,
///   `s = sign(butter_a - butter_b)`.
/// Sign convention: Δb > 0 ⇒ A is butter-worse than B ⇒ we want y_a < y_b.
/// `score_score` and `score_score_b` aren't separate inputs — the score
/// pair lives in `y_score[k]` (A) and `y_score[k + K]` (B).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cross_codec_eq_loss_kernel(
    y_score: &Array<f32>,
    row_weight: &Array<f32>,
    butter_diff: &Array<f32>,
    dl_dypre_per_b: &mut Array<f32>,
    k_pairs: u32,
    w_eq: f32,
    w_rp: f32,
    tanh_scale: f32,
) {
    let k = ABSOLUTE_POS;
    let kp_us = k_pairs as usize;
    if k >= kp_us {
        terminate!();
    }
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
    let two = f32::new(2.0_f32);
    let clamp_pos = f32::new(20.0_f32);
    let clamp_neg = f32::new(-20.0_f32);

    let k_u = k;
    let i_a = k_u;
    let i_b = k_u + kp_us;

    let ya = y_score[i_a];
    let yb = y_score[i_b];
    let rw = row_weight[k_u];
    let diff = ya - yb;
    // L_eq = w · row_w · diff²
    // dL/dy_a (score) = 2 · w · row_w · diff
    let scale_eq = two * w_eq * rw;
    let mut dl_dya_s = scale_eq * diff;
    let mut dl_dyb_s = -dl_dya_s;

    // Rank-preserve term. Disabled when w_rp == 0 OR butter_diff[k] == 0.
    if w_rp > zero {
        let db = butter_diff[k_u];
        // NaN guard: f32::is_finite isn't available in #[cube]; we rely
        // on the |db| > 1e-12 check below, which also rejects NaN
        // because NaN comparisons return false (so NaN > 1e-12 is false).
        let abs_db = if db >= zero { db } else { -db };
        if abs_db > f32::new(1e-12_f32) {
            let s = if db > zero { one } else { -one };
            let w_rp_eff = w_rp * abs_db;
            let u = s * (yb - ya);
            // sigmoid(u) numerically stable via clamping.
            let u_c = if u > clamp_pos {
                clamp_pos
            } else if u < clamp_neg {
                clamp_neg
            } else {
                u
            };
            let sig = one / (one + f32::exp(-u_c));
            // dL_rp/dy_b = -w · s · (1 - σ)
            // dL_rp/dy_a = +w · s · (1 - σ)
            let g = w_rp_eff * s * (one - sig);
            dl_dya_s += g;
            dl_dyb_s -= g;
        }
    }

    // Chain through tanh-pin Jacobian per side.
    let dl_dypre_a = if tanh_scale > zero {
        let s_a = ya / f32::new(100.0_f32);
        let jac_a = (f32::new(100.0_f32) / tanh_scale) * s_a * (one - s_a);
        dl_dya_s * jac_a
    } else {
        dl_dya_s
    };
    let dl_dypre_b = if tanh_scale > zero {
        let s_b = yb / f32::new(100.0_f32);
        let jac_b = (f32::new(100.0_f32) / tanh_scale) * s_b * (one - s_b);
        dl_dyb_s * jac_b
    } else {
        dl_dyb_s
    };

    dl_dypre_per_b[i_a] = dl_dypre_a;
    dl_dypre_per_b[i_b] = dl_dypre_b;
}

/// σ-floor probe reduction — one thread, sequential scan.
///
/// `y_score[0..n_probe]` are the forward outputs of one σ-floor probe
/// batch. This kernel computes mean μ and population σ, then emits a
/// per-batch `grad_scale` scalar derived from the CPU formula:
///
///   viol = sigma_threshold - σ
///   if viol > 0 AND σ > 1e-9:
///       grad_scale = -2 · w_dr · viol / (σ · n_probe)
///       loss      = w_dr · viol²
///   else:
///       grad_scale = 0
///       loss      = 0
///
/// Outputs:
/// - `out[0]` = μ
/// - `out[1]` = σ_obs (the observed σ; only meaningful when grad_scale ≠ 0)
/// - `out[2]` = grad_scale (0 when no violation)
/// - `out[3]` = loss (for diagnostics)
///
/// `n_probe` is small (default 40) so a single-thread reduction is fine —
/// avoids an extra round-trip to host.
#[cube(launch)]
pub fn sigma_floor_reduce_kernel(
    y_score: &Array<f32>,
    out: &mut Array<f32>,
    n_probe: u32,
    sigma_threshold: f32,
    w_dr: f32,
) {
    let tid = ABSOLUTE_POS;
    if tid > 0usize {
        terminate!();
    }
    let zero = f32::new(0.0_f32);
    let n_f = f32::cast_from(n_probe);

    let mut sum = zero;
    let mut i = 0u32;
    while i < n_probe {
        sum += y_score[i as usize];
        i += 1u32;
    }
    let mu = sum / n_f;

    let mut sumsq = zero;
    i = 0u32;
    while i < n_probe {
        let d = y_score[i as usize] - mu;
        sumsq += d * d;
        i += 1u32;
    }
    let var = sumsq / n_f;
    let var_safe = if var > zero { var } else { zero };
    let sigma_obs = f32::sqrt(var_safe);

    let viol = sigma_threshold - sigma_obs;
    let sigma_eps = f32::new(1e-9_f32);
    let (grad_scale, loss) = if viol > zero && sigma_obs > sigma_eps {
        let g = -f32::new(2.0_f32) * w_dr * viol / (sigma_obs * n_f);
        let l = w_dr * viol * viol;
        (g, l)
    } else {
        (zero, zero)
    };

    out[0] = mu;
    out[1] = sigma_obs;
    out[2] = grad_scale;
    out[3] = loss;
}

/// σ-floor per-row gradient kernel — one thread per probe row.
///
/// Reads `mu`, `grad_scale` from `reduce_out[0]` and `reduce_out[2]`. If
/// `grad_scale == 0` writes zero (no violation). Otherwise:
///
///   dL/dy_score[k] = grad_scale · (y_score[k] - μ)
///
/// chained through optional tanh-pin Jacobian.
///
/// CPU equivalent: lines ~6015-6060 (the per-probe-row loop computing
/// `dl_dy = grad_scale * (y_i - mu) * dy_dpre`).
#[cube(launch)]
pub fn sigma_floor_grad_kernel(
    y_score: &Array<f32>,
    reduce_out: &Array<f32>,
    dl_dypre_per_b: &mut Array<f32>,
    tanh_scale: f32,
) {
    let k = ABSOLUTE_POS;
    let n = y_score.len();
    if k >= n {
        terminate!();
    }
    let zero = f32::new(0.0_f32);
    let one = f32::new(1.0_f32);
    let mu = reduce_out[0];
    let grad_scale = reduce_out[2];

    let y_s = y_score[k];
    let dl_dy_s = grad_scale * (y_s - mu);
    let dl_dypre = if tanh_scale > zero {
        let s = y_s / f32::new(100.0_f32);
        let jac = (f32::new(100.0_f32) / tanh_scale) * s * (one - s);
        dl_dy_s * jac
    } else {
        dl_dy_s
    };
    dl_dypre_per_b[k] = dl_dypre;
}
