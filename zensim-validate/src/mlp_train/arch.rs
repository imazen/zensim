pub struct ArchForward {
    pub(crate) y: f64,
    pub(crate) y_rank: f64,
    pub(crate) y_pool: f64,
    pub(crate) alpha: f64,
    #[allow(dead_code)]
    pub(crate) alpha_logit: f64,
    pub(crate) h_pre: Vec<f64>,
    pub(crate) h: Vec<f64>,
    pub(crate) stats: [f64; 4],
    pub(crate) max_idx: usize,
    pub(crate) h1_pre: Vec<f64>,
    pub(crate) h1: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn arch_forward(
    x: &[f64],
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
    n_features: usize,
    n_hidden1: usize,
    n_hidden_final: usize,
    leaky: f64,
    use_2layer: bool,
    use_skip: bool,
) -> ArchForward {
    use zensim_train_core::per_sample_alpha_head as psah;
    use zensim_train_core::simd_encoder;

    let (h_pre, h, h1_pre, h1) = if use_2layer {
        let (h1p, h1v, h2p, h2v) = simd_encoder::encoder_forward_2layer(
            x, w1, b1, w2_enc, b2_enc, n_features, n_hidden1, n_hidden_final, leaky,
        );
        (h2p, h2v, h1p, h1v)
    } else {
        let (hp, hv) = simd_encoder::encoder_forward(x, w1, b1, n_features, n_hidden1, leaky);
        (hp, hv, Vec::new(), Vec::new())
    };

    let (y, y_rank, y_pool, alpha, alpha_logit, stats, max_idx) = psah::forward_heads(
        &h, rank_w, rank_b, reducer_w, reducer_b, w_alpha, b_alpha, n_hidden_final,
    );

    let y_final = if use_skip {
        y + simd_encoder::skip_forward(x, w_skip, b_skip)
    } else {
        y
    };

    ArchForward {
        y: y_final, y_rank, y_pool, alpha, alpha_logit,
        h_pre, h, stats, max_idx, h1_pre, h1,
    }
}

/// Architecture-dispatched backward for the per-sample α head. First
/// runs heads backprop to get dl/dh, then routes dl/dh through the
/// correct encoder backward (1-layer or 2-layer). If skip is active,
/// accumulates skip gradients.
///
/// Accumulates into the concatenated adam.gw1/gb1 slots:
///   [w1_grads | w2_enc_grads | w_skip_grads] for gw1
///   [b1_grads | b2_enc_grads | b_skip_grad ] for gb1
#[allow(clippy::too_many_arguments)]
pub fn arch_backward(
    x: &[f64],
    fwd: &ArchForward,
    dl_dy: f64,
    w1: &[f64],
    w2_enc: &[f64],
    rank_w: &[f64],
    reducer_w: &[f64; 4],
    w_alpha: &[f64],
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
    leaky: f64,
    use_2layer: bool,
    use_skip: bool,
) {
    use zensim_train_core::per_sample_alpha_head as psah;
    use zensim_train_core::simd_encoder;

    let dl_dh = psah::backprop_heads(
        &fwd.h, &fwd.stats, fwd.max_idx, fwd.y_rank, fwd.y_pool, fwd.alpha, dl_dy,
        rank_w, reducer_w, w_alpha,
        g_rank_w, g_rank_b, g_reducer_w, g_reducer_b,
        g_w_alpha, g_b_alpha, n_hidden_final, leaky,
    );

    if use_2layer {
        let n_w1 = n_features * n_hidden1;
        let n_w2_enc = n_hidden1 * n_hidden_final;

        // Layer 2: LeakyReLU backward on h2_pre, then gw2/gb2 update
        // using the cached h1 from the forward pass.
        let dl_dh2_pre = simd_encoder::leaky_relu_backward(&dl_dh, &fwd.h_pre, leaky);
        simd_encoder::encoder_backprop_layer1(
            &fwd.h1, &dl_dh2_pre,
            &mut gw1_concat[n_w1..n_w1 + n_w2_enc],
            &mut gb1_concat[n_hidden1..n_hidden1 + n_hidden_final],
            n_hidden1, n_hidden_final,
        );

        // Propagate through layer 2 to get dl/dh1.
        let mut dl_dh1 = vec![0.0f64; n_hidden1];
        for j in 0..n_hidden1 {
            let row = &w2_enc[j * n_hidden_final..(j + 1) * n_hidden_final];
            for k in 0..n_hidden_final {
                dl_dh1[j] += dl_dh2_pre[k] * row[k];
            }
        }

        // Layer 1: exact LeakyReLU backward using the cached h1_pre.
        let dl_dh1_pre = simd_encoder::leaky_relu_backward(&dl_dh1, &fwd.h1_pre, leaky);
        simd_encoder::encoder_backprop_layer1(
            x, &dl_dh1_pre,
            &mut gw1_concat[..n_w1], &mut gb1_concat[..n_hidden1],
            n_features, n_hidden1,
        );
    } else {
        let n_w1 = n_features * n_hidden1;
        let dl_dh_pre = simd_encoder::leaky_relu_backward(&dl_dh, &fwd.h_pre, leaky);
        simd_encoder::encoder_backprop_layer1(
            x, &dl_dh_pre,
            &mut gw1_concat[..n_w1], &mut gb1_concat[..n_hidden1],
            n_features, n_hidden1,
        );
    }

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
        let gw_skip = &mut gw1_concat[skip_offset_w..skip_offset_w + n_features];
        let gb_skip = &mut gb1_concat[skip_offset_b..skip_offset_b + 1];
        simd_encoder::skip_backward(x, dl_dy, gw_skip, &mut gb_skip[0]);
    }
}

