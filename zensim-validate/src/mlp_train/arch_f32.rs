//! F32-native arch_forward / arch_backward for the PSAH trainer hot path.
//!
//! The legacy `arch.rs` API takes f64 inputs and routes through the f64
//! wrappers in `zensim_train_core::simd_encoder`. Each call casts
//! `Vec<f64> → Vec<f32>` for x, w1, b1 (and w2_enc/b2_enc for 2-layer)
//! and casts the f32 outputs back to f64 — 5 fresh Vec allocations and
//! ~190 KB of cast work per pair-forward at the production
//! 372 × 128 × 64 shape. At 50 k pairs/epoch × 2 forwards × 2 layers
//! that's ~1 M short-lived Vecs and ~76 GB of memory bandwidth wasted
//! per epoch — swamping the f32 SIMD kernel itself.
//!
//! This module solves both:
//! 1. [`WeightScratchF32`] holds f32 copies of every weight buffer.
//!    The training loop calls [`WeightScratchF32::refresh`] once per
//!    Adam step (where the f64 weights actually change); the inner
//!    pair loop reads pre-cast f32 weights.
//! 2. [`arch_forward_f32`] / [`arch_backward_f32`] take `&[f32]`
//!    features (from a `Vec<Vec<f32>>` `std_features`) and route
//!    straight to the `simd_encoder` f32 SIMD primitives.
//!
//! Adam grad accumulators stay f64 (precision matters across many
//! pairs). The backward path uses thread-local f32 scratch buffers
//! for the SIMD kernel's outputs, then cast-adds into the f64 grad
//! accumulators in a single pass.

use std::cell::RefCell;
use zensim_train_core::simd_encoder::{
    dot_bias_f32, encoder_backprop_layer1_f32, encoder_forward_2layer_f32,
    encoder_forward_f32, leaky_relu_backward_f32, skip_backward_f32, skip_forward_f32,
};

use super::arch::ArchForward;

const POOL_STD_FLOOR: f32 = 1e-3;
const POOL_P_NORM: f32 = 6.0;

/// All-f32 forward intermediates needed by [`arch_backward_f32`].
///
/// Mirrors the f64 [`super::arch::ArchForward`] shape but holds f32
/// hidden vectors + stats so the SIMD kernels can read them directly.
#[derive(Default)]
pub struct ArchForwardF32 {
    pub y: f32,
    pub y_rank: f32,
    pub y_pool: f32,
    pub alpha: f32,
    #[allow(dead_code)]
    pub alpha_logit: f32,
    pub h_pre: Vec<f32>,
    pub h: Vec<f32>,
    pub stats: [f32; 4],
    pub max_idx: usize,
    pub h1_pre: Vec<f32>,
    pub h1: Vec<f32>,
}

impl ArchForwardF32 {
    /// Cast all f32 intermediates to f64, producing the legacy
    /// [`ArchForward`] shape. Used at the boundary with downstream
    /// code that still reads f64 (NiN buffer, aux-loss steps that
    /// haven't migrated yet, validation eval blocks). Takes `&self`
    /// so the caller can keep the f32 form for [`arch_backward_f32`].
    pub fn to_archforward(&self) -> ArchForward {
        ArchForward {
            y: self.y as f64,
            y_rank: self.y_rank as f64,
            y_pool: self.y_pool as f64,
            alpha: self.alpha as f64,
            alpha_logit: self.alpha_logit as f64,
            h_pre: cast_f32_slice_to_f64(&self.h_pre),
            h: cast_f32_slice_to_f64(&self.h),
            stats: [
                self.stats[0] as f64,
                self.stats[1] as f64,
                self.stats[2] as f64,
                self.stats[3] as f64,
            ],
            max_idx: self.max_idx,
            h1_pre: cast_f32_slice_to_f64(&self.h1_pre),
            h1: cast_f32_slice_to_f64(&self.h1),
        }
    }
}

/// F32 copies of every weight buffer used by [`arch_forward_f32`] /
/// [`arch_backward_f32`]. Owned by the training loop, refreshed via
/// [`refresh`](Self::refresh) immediately after each Adam step.
///
/// Capacity is set once at construction; `refresh` only does the cast
/// (no realloc). Scalars (`rank_b`, `reducer_b`, `b_alpha`, `b_skip`)
/// are stored inline.
pub struct WeightScratchF32 {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2_enc: Vec<f32>,
    pub b2_enc: Vec<f32>,
    pub w_skip: Vec<f32>,
    pub rank_w: Vec<f32>,
    pub reducer_w: [f32; 4],
    pub w_alpha: Vec<f32>,
    pub rank_b: f32,
    pub reducer_b: f32,
    pub b_alpha: f32,
    pub b_skip: f32,
}

impl WeightScratchF32 {
    pub fn new(
        n_features: usize,
        n_hidden1: usize,
        n_hidden_final: usize,
        use_2layer: bool,
        use_skip: bool,
    ) -> Self {
        Self {
            w1: vec![0.0; n_features * n_hidden1],
            b1: vec![0.0; n_hidden1],
            w2_enc: if use_2layer {
                vec![0.0; n_hidden1 * n_hidden_final]
            } else {
                Vec::new()
            },
            b2_enc: if use_2layer {
                vec![0.0; n_hidden_final]
            } else {
                Vec::new()
            },
            w_skip: if use_skip {
                vec![0.0; n_features]
            } else {
                Vec::new()
            },
            rank_w: vec![0.0; n_hidden_final],
            reducer_w: [0.0; 4],
            w_alpha: vec![0.0; n_hidden_final],
            rank_b: 0.0,
            reducer_b: 0.0,
            b_alpha: 0.0,
            b_skip: 0.0,
        }
    }

    /// Cast every f64 weight into the matching f32 buffer. Call after
    /// each Adam step (the only point where f64 weights change).
    #[allow(clippy::too_many_arguments)]
    pub fn refresh(
        &mut self,
        w1: &[f64],
        b1: &[f64],
        w2_enc: &[f64],
        b2_enc: &[f64],
        w_skip: &[f64],
        b_skip: f64,
        rank_w: &[f64],
        rank_b: f64,
        reducer_w: &[f64; 4],
        reducer_b: f64,
        w_alpha: &[f64],
        b_alpha: f64,
    ) {
        cast_f64_to_f32(w1, &mut self.w1);
        cast_f64_to_f32(b1, &mut self.b1);
        cast_f64_to_f32(w2_enc, &mut self.w2_enc);
        cast_f64_to_f32(b2_enc, &mut self.b2_enc);
        cast_f64_to_f32(w_skip, &mut self.w_skip);
        cast_f64_to_f32(rank_w, &mut self.rank_w);
        for k in 0..4 {
            self.reducer_w[k] = reducer_w[k] as f32;
        }
        cast_f64_to_f32(w_alpha, &mut self.w_alpha);
        self.rank_b = rank_b as f32;
        self.reducer_b = reducer_b as f32;
        self.b_alpha = b_alpha as f32;
        self.b_skip = b_skip as f32;
    }
}

fn cast_f32_slice_to_f64(src: &[f32]) -> Vec<f64> {
    src.iter().map(|&v| v as f64).collect()
}

fn cast_f64_to_f32(src: &[f64], dst: &mut Vec<f32>) {
    if dst.len() != src.len() {
        dst.resize(src.len(), 0.0);
    }
    for (i, &v) in src.iter().enumerate() {
        dst[i] = v as f32;
    }
}

#[inline]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn pool_stats_f32(h: &[f32]) -> ([f32; 4], usize) {
    let n = h.len() as f32;
    debug_assert!(n > 0.0, "pool_stats_f32 requires non-empty hidden vector");
    let mean: f32 = h.iter().sum::<f32>() / n;
    let var: f32 = h.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let std = var.sqrt().max(POOL_STD_FLOOR);
    let (max_idx, &max_val) = h
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))
        .expect("non-empty hidden vector");
    let sum_p: f32 = h.iter().map(|&v| v.abs().powf(POOL_P_NORM)).sum();
    let mean_p = sum_p / n;
    let p_norm = mean_p.powf(1.0 / POOL_P_NORM);
    ([mean, std, max_val, p_norm], max_idx)
}

/// Architecture-dispatched forward (1-layer or 2-layer, ± skip),
/// f32-native compute. Weights come pre-cast in `s` (refresh once per
/// Adam step); `x` is the std_features f64 row, cast into a
/// thread-local f32 scratch on entry (avoids per-pair malloc).
#[allow(clippy::too_many_arguments)]
pub fn arch_forward_f32(
    x: &[f64],
    s: &WeightScratchF32,
    n_features: usize,
    n_hidden1: usize,
    n_hidden_final: usize,
    leaky: f32,
    use_2layer: bool,
    use_skip: bool,
) -> ArchForwardF32 {
    with_x_f32(x, |x_f32| {
        let (h_pre, h, h1_pre, h1) = if use_2layer {
            let (h1p, h1v, h2p, h2v) = encoder_forward_2layer_f32(
                x_f32,
                &s.w1,
                &s.b1,
                &s.w2_enc,
                &s.b2_enc,
                n_features,
                n_hidden1,
                n_hidden_final,
                leaky,
            );
            (h2p, h2v, h1p, h1v)
        } else {
            let (hp, hv) =
                encoder_forward_f32(x_f32, &s.w1, &s.b1, n_features, n_hidden1, leaky);
            (hp, hv, Vec::new(), Vec::new())
        };

        let y_rank = dot_bias_f32(&h, &s.rank_w, s.rank_b);
        let (stats, max_idx) = pool_stats_f32(&h);
        let y_pool = stats[0] * s.reducer_w[0]
            + stats[1] * s.reducer_w[1]
            + stats[2] * s.reducer_w[2]
            + stats[3] * s.reducer_w[3]
            + s.reducer_b;
        let alpha_logit = dot_bias_f32(&h, &s.w_alpha, s.b_alpha);
        let alpha = sigmoid_f32(alpha_logit);
        let y_main = alpha * y_rank + (1.0 - alpha) * y_pool;

        let y = if use_skip {
            y_main + skip_forward_f32(x_f32, &s.w_skip, s.b_skip)
        } else {
            y_main
        };

        ArchForwardF32 {
            y,
            y_rank,
            y_pool,
            alpha,
            alpha_logit,
            h_pre,
            h,
            stats,
            max_idx,
            h1_pre,
            h1,
        }
    })
}

// Thread-local f32 scratch for backward grad accumulation + per-pair
// feature cast. SIMD kernels write into these; we then cast-add into
// the f64 Adam accumulators in a single linear pass. Reused across
// pair-steps per worker thread (Rayon workers each see their own).
thread_local! {
    static SCRATCH_X_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_GW1_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_GB1_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_GW2_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_GB2_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_DLDH1_F32: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

fn with_x_f32<R>(x: &[f64], body: impl FnOnce(&[f32]) -> R) -> R {
    SCRATCH_X_F32.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() != x.len() {
            buf.resize(x.len(), 0.0);
        }
        for (i, &v) in x.iter().enumerate() {
            buf[i] = v as f32;
        }
        body(&buf)
    })
}

fn with_scratch_grad<R>(
    n_grad_w1: usize,
    n_grad_b1: usize,
    n_grad_w2: usize,
    n_grad_b2: usize,
    n_dldh1: usize,
    body: impl FnOnce(&mut [f32], &mut [f32], &mut [f32], &mut [f32], &mut [f32]) -> R,
) -> R {
    SCRATCH_GW1_F32.with(|gw1| {
        SCRATCH_GB1_F32.with(|gb1| {
            SCRATCH_GW2_F32.with(|gw2| {
                SCRATCH_GB2_F32.with(|gb2| {
                    SCRATCH_DLDH1_F32.with(|dldh1| {
                        let mut gw1_b = gw1.borrow_mut();
                        let mut gb1_b = gb1.borrow_mut();
                        let mut gw2_b = gw2.borrow_mut();
                        let mut gb2_b = gb2.borrow_mut();
                        let mut dldh1_b = dldh1.borrow_mut();
                        ensure_zeroed(&mut gw1_b, n_grad_w1);
                        ensure_zeroed(&mut gb1_b, n_grad_b1);
                        ensure_zeroed(&mut gw2_b, n_grad_w2);
                        ensure_zeroed(&mut gb2_b, n_grad_b2);
                        ensure_zeroed(&mut dldh1_b, n_dldh1);
                        body(
                            &mut gw1_b,
                            &mut gb1_b,
                            &mut gw2_b,
                            &mut gb2_b,
                            &mut dldh1_b,
                        )
                    })
                })
            })
        })
    })
}

fn ensure_zeroed(buf: &mut Vec<f32>, n: usize) {
    if buf.len() < n {
        buf.resize(n, 0.0);
    }
    for v in buf[..n].iter_mut() {
        *v = 0.0;
    }
}

/// Cast-and-add: `dst[i] += src[i] as f64`. Single linear pass.
#[inline]
fn add_f32_to_f64(dst: &mut [f64], src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += s as f64;
    }
}

/// Architecture-dispatched backward (1-layer or 2-layer, ± skip),
/// f32-native compute. f64 Adam grad accumulators receive
/// `f32 → f64` cast-add at the end of the pass. `x` is cast into
/// the thread-local scratch on entry.
#[allow(clippy::too_many_arguments)]
pub fn arch_backward_f32(
    x: &[f64],
    fwd: &ArchForwardF32,
    dl_dy: f64,
    s: &WeightScratchF32,
    gw1_concat: &mut [f64],
    gb1_concat: &mut [f64],
    g_rank_w: &mut [f64],
    g_rank_b: &mut f64,
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    g_w_alpha: &mut [f64],
    g_b_alpha: &mut f64,
    n_features: usize,
    n_hidden1: usize,
    n_hidden_final: usize,
    leaky: f32,
    use_2layer: bool,
    use_skip: bool,
) {
    with_x_f32(x, |x_f32| arch_backward_f32_inner(
        x_f32, fwd, dl_dy, s,
        gw1_concat, gb1_concat,
        g_rank_w, g_rank_b, g_reducer_w, g_reducer_b, g_w_alpha, g_b_alpha,
        n_features, n_hidden1, n_hidden_final,
        leaky, use_2layer, use_skip,
    ))
}

#[allow(clippy::too_many_arguments)]
fn arch_backward_f32_inner(
    x: &[f32],
    fwd: &ArchForwardF32,
    dl_dy: f64,
    s: &WeightScratchF32,
    gw1_concat: &mut [f64],
    gb1_concat: &mut [f64],
    g_rank_w: &mut [f64],
    g_rank_b: &mut f64,
    g_reducer_w: &mut [f64; 4],
    g_reducer_b: &mut f64,
    g_w_alpha: &mut [f64],
    g_b_alpha: &mut f64,
    n_features: usize,
    n_hidden1: usize,
    n_hidden_final: usize,
    leaky: f32,
    use_2layer: bool,
    use_skip: bool,
) {
    // Heads backward (computed in f64 for grad-accumulator precision,
    // but reads f32 forward intermediates with on-the-fly promotion).
    let dl_dy_rank = dl_dy * fwd.alpha as f64;
    let dl_dy_pool = dl_dy * (1.0 - fwd.alpha as f64);
    let dl_dalpha = dl_dy * (fwd.y_rank as f64 - fwd.y_pool as f64);
    let dl_dalpha_logit = dl_dalpha * fwd.alpha as f64 * (1.0 - fwd.alpha as f64);

    for j in 0..n_hidden_final {
        g_rank_w[j] += dl_dy_rank * fwd.h[j] as f64;
    }
    *g_rank_b += dl_dy_rank;
    for k in 0..4 {
        g_reducer_w[k] += dl_dy_pool * fwd.stats[k] as f64;
    }
    *g_reducer_b += dl_dy_pool;
    for j in 0..n_hidden_final {
        g_w_alpha[j] += dl_dalpha_logit * fwd.h[j] as f64;
    }
    *g_b_alpha += dl_dalpha_logit;

    // dl_dh assembly (f32 to feed the f32 encoder backprop).
    let mut dl_dh_f32 = vec![0.0f32; n_hidden_final];
    let dl_dy_rank_f32 = dl_dy_rank as f32;
    let dl_dalpha_logit_f32 = dl_dalpha_logit as f32;
    for j in 0..n_hidden_final {
        dl_dh_f32[j] += dl_dy_rank_f32 * s.rank_w[j];
        dl_dh_f32[j] += dl_dalpha_logit_f32 * s.w_alpha[j];
    }
    let n_f = n_hidden_final as f32;
    let mu = fwd.stats[0];
    let sigma = fwd.stats[1];
    let p6 = fwd.stats[3];
    let inv_sigma_n = if sigma > POOL_STD_FLOOR + 1e-6 {
        1.0 / (n_f * sigma)
    } else {
        0.0
    };
    let p6_floor = p6.max(1e-6);
    let inv_p6_pow5_n = 1.0 / (n_f * p6_floor.powi(5));
    let dl_dstat_f32: [f32; 4] = [
        dl_dy_pool as f32 * s.reducer_w[0],
        dl_dy_pool as f32 * s.reducer_w[1],
        dl_dy_pool as f32 * s.reducer_w[2],
        dl_dy_pool as f32 * s.reducer_w[3],
    ];
    for j in 0..n_hidden_final {
        let hj = fwd.h[j];
        dl_dh_f32[j] += dl_dstat_f32[0] / n_f;
        dl_dh_f32[j] += dl_dstat_f32[1] * (hj - mu) * inv_sigma_n;
        if j == fwd.max_idx {
            dl_dh_f32[j] += dl_dstat_f32[2];
        }
        let abs_hj = hj.abs();
        let sign_hj = if hj >= 0.0 { 1.0 } else { -1.0 };
        let abs_pow5 = abs_hj * abs_hj * abs_hj * abs_hj * abs_hj;
        dl_dh_f32[j] += dl_dstat_f32[3] * sign_hj * abs_pow5 * inv_p6_pow5_n;
    }

    // Encoder backward — compute in f32 scratch, then cast-add to f64.
    let n_w1 = n_features * n_hidden1;
    let n_w2_enc = n_hidden1 * n_hidden_final;

    with_scratch_grad(
        n_w1,
        n_hidden1,
        n_w2_enc,
        n_hidden_final,
        n_hidden1,
        |gw1_f32, gb1_f32, gw2_f32, gb2_f32, dldh1_f32| {
            if use_2layer {
                // Layer 2.
                let dl_dh2_pre = leaky_relu_backward_f32(&dl_dh_f32, &fwd.h_pre, leaky);
                encoder_backprop_layer1_f32(
                    &fwd.h1,
                    &dl_dh2_pre,
                    gw2_f32,
                    gb2_f32,
                    n_hidden1,
                    n_hidden_final,
                );

                // Propagate to layer-1 (dl_dh1 = w2_enc · dl_dh2_pre).
                for j in 0..n_hidden1 {
                    let row = &s.w2_enc[j * n_hidden_final..(j + 1) * n_hidden_final];
                    let mut acc = 0.0f32;
                    for k in 0..n_hidden_final {
                        acc += dl_dh2_pre[k] * row[k];
                    }
                    dldh1_f32[j] = acc;
                }

                // Layer 1.
                let dl_dh1_pre = leaky_relu_backward_f32(dldh1_f32, &fwd.h1_pre, leaky);
                encoder_backprop_layer1_f32(
                    x,
                    &dl_dh1_pre,
                    gw1_f32,
                    gb1_f32,
                    n_features,
                    n_hidden1,
                );

                // Cast-add the f32 grads into the f64 accumulators.
                add_f32_to_f64(&mut gw1_concat[..n_w1], gw1_f32);
                add_f32_to_f64(&mut gb1_concat[..n_hidden1], gb1_f32);
                add_f32_to_f64(&mut gw1_concat[n_w1..n_w1 + n_w2_enc], gw2_f32);
                add_f32_to_f64(
                    &mut gb1_concat[n_hidden1..n_hidden1 + n_hidden_final],
                    gb2_f32,
                );
            } else {
                let dl_dh_pre = leaky_relu_backward_f32(&dl_dh_f32, &fwd.h_pre, leaky);
                encoder_backprop_layer1_f32(
                    x,
                    &dl_dh_pre,
                    gw1_f32,
                    gb1_f32,
                    n_features,
                    n_hidden1,
                );
                add_f32_to_f64(&mut gw1_concat[..n_w1], gw1_f32);
                add_f32_to_f64(&mut gb1_concat[..n_hidden1], gb1_f32);
            }
        },
    );

    if use_skip {
        let skip_offset_w = if use_2layer {
            n_features * n_hidden1 + n_hidden1 * n_hidden_final
        } else {
            n_features * n_hidden1
        };
        let skip_offset_b = if use_2layer {
            n_hidden1 + n_hidden_final
        } else {
            n_hidden1
        };
        let mut gw_skip_f32 = vec![0.0f32; n_features];
        let mut gb_skip_f32 = 0.0f32;
        skip_backward_f32(x, dl_dy as f32, &mut gw_skip_f32, &mut gb_skip_f32);
        add_f32_to_f64(
            &mut gw1_concat[skip_offset_w..skip_offset_w + n_features],
            &gw_skip_f32,
        );
        gb1_concat[skip_offset_b] += gb_skip_f32 as f64;
    }
}
