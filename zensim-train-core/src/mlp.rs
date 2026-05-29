//! Two-layer MLP primitives: standardizer, forward pass, backprop step,
//! batched group prediction.
//!
//! Bit-exact port from `zensim-validate/src/mlp_train.rs`. The shapes
//! are:
//! - `n_features → n_hidden` (first Linear)
//! - LeakyReLU with negative-slope `alpha`
//! - `n_hidden → 1` (second Linear)
//!
//! Layout convention: `w1` is row-major `(n_features × n_hidden)`,
//! `w2` is `(n_hidden × 1)` (so a flat slice of length `n_hidden`).

use crate::TrainingGroup;
use zenpredict::{Activation, WeightDtype};
use zenpredict_bake::{BakeLayer, BakeRequest, bake};

/// Bake a 2-layer MLP (LeakyReLU → Identity) into a ZNPR v3 byte stream.
///
/// Converts f64 weights to f32 once and feeds them to [`bake`].
/// Output is the same byte stream produced by
/// `zensim-validate::mlp_train::bake_two_layer_znpr_v3` — readable by
/// any `zenpredict::Predictor` loaded from those bytes.
///
/// ZNPR v2 production is prohibited per CLAUDE.md (2026-05-15).
#[allow(clippy::too_many_arguments)]
pub fn bake_two_layer_znpr_v3(
    scaler_mean: &[f64],
    scaler_scale: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_inputs: usize,
    n_hidden: usize,
    n_outputs: usize,
) -> Vec<u8> {
    let scaler_mean_f32: Vec<f32> = scaler_mean.iter().map(|&v| v as f32).collect();
    let scaler_scale_f32: Vec<f32> = scaler_scale.iter().map(|&v| v as f32).collect();
    let w1_f32: Vec<f32> = w1.iter().map(|&v| v as f32).collect();
    let b1_f32: Vec<f32> = b1.iter().map(|&v| v as f32).collect();
    let w2_f32: Vec<f32> = w2.iter().map(|&v| v as f32).collect();
    let b2_f32: Vec<f32> = b2.iter().map(|&v| v as f32).collect();
    let layers = [
        BakeLayer {
            in_dim: n_inputs,
            out_dim: n_hidden,
            activation: Activation::LeakyRelu,
            dtype: WeightDtype::F32,
            weights: &w1_f32,
            biases: &b1_f32,
        },
        BakeLayer {
            in_dim: n_hidden,
            out_dim: n_outputs,
            activation: Activation::Identity,
            dtype: WeightDtype::F32,
            weights: &w2_f32,
            biases: &b2_f32,
        },
    ];
    bake(&BakeRequest {
        schema_hash: 0,
        flags: 0,
        scaler_mean: &scaler_mean_f32,
        scaler_scale: &scaler_scale_f32,
        layers: &layers,
        feature_bounds: &[],
        metadata: &[],
        output_specs: &[],
        discrete_sets: &[],
        sparse_overrides: &[],
        feature_order: None,
        output_order: None,
        compressed: false,
        hu_permutations: None,
    })
    .expect("v3 bake of 2-layer MLP")
}

/// Compute per-feature `(mean, std)` across all `train_indices` groups.
/// Validation-only groups are excluded so the scaler never sees val data.
///
/// Floors `std` at 1e-8 to avoid divide-by-zero on degenerate features.
pub fn compute_scaler_from_groups(
    groups: &[TrainingGroup<'_>],
    train_indices: &[usize],
    n_features: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut count = 0u64;
    let mut mean = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features {
            for d in 0..n_features {
                mean[d] += f[d];
            }
            count += 1;
        }
    }
    let n = count.max(1) as f64;
    for m in &mut mean {
        *m /= n;
    }
    let mut var = vec![0.0f64; n_features];
    for &gi in train_indices {
        for f in groups[gi].features {
            for d in 0..n_features {
                let dx = f[d] - mean[d];
                var[d] += dx * dx;
            }
        }
    }
    let std = var.iter().map(|&v| (v / n).sqrt().max(1e-8)).collect();
    (mean, std)
}

/// Forward pass through `n_features → n_hidden (LeakyReLU α) → 1`.
///
/// Returns `(y, h_pre, h)` where `h_pre` is the pre-activation hidden
/// state and `h` is the post-LeakyReLU state. Backprop needs both:
/// `h_pre` selects the slope (1 vs α), `h` flows into the output grad.
///
/// Skip-zero optimization on the first matmul: feature columns with
/// `x[i] == 0.0` contribute nothing to `h_pre`, so we skip them. This
/// matters when standardized features are sparse (rare in our 228-dim
/// case but cheap to keep for bit-exactness with the existing trainer).
#[allow(clippy::too_many_arguments)]
pub fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) -> (f64, Vec<f64>, Vec<f64>) {
    let mut h_pre = b1.to_vec();
    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &w1[i * n_hidden..(i + 1) * n_hidden];
        for (acc, &w) in h_pre.iter_mut().zip(row.iter()) {
            *acc += s * w;
        }
    }
    let h: Vec<f64> = h_pre
        .iter()
        .map(|&v| if v >= 0.0 { v } else { alpha * v })
        .collect();
    let mut y = b2[0];
    for o in 0..n_hidden {
        y += h[o] * w2[o];
    }
    (y, h_pre, h)
}

/// Backpropagate one sample's loss-grad through the two-layer MLP.
///
/// `dl_dy` is `∂L/∂y` from upstream (RankNet or TV); this function
/// accumulates gradient contributions into `gw1`, `gb1`, `gw2`, `gb2`.
/// The Adam optimizer step (see `adam.rs`) then consumes those grads.
#[allow(clippy::too_many_arguments)]
pub fn backprop_step(
    x: &[f64],
    h_pre: &[f64],
    h: &[f64],
    dl_dy: f64,
    _w1: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    n_features: usize,
    n_hidden: usize,
    alpha: f64,
) {
    for o in 0..n_hidden {
        gw2[o] += dl_dy * h[o];
    }
    gb2[0] += dl_dy;

    let mut dl_dh_pre = vec![0.0f64; n_hidden];
    for o in 0..n_hidden {
        let dh = dl_dy * w2[o];
        dl_dh_pre[o] = if h_pre[o] >= 0.0 { dh } else { alpha * dh };
    }

    for i in 0..n_features {
        let s = x[i];
        if s == 0.0 {
            continue;
        }
        let row = &mut gw1[i * n_hidden..(i + 1) * n_hidden];
        for (g, &dh) in row.iter_mut().zip(dl_dh_pre.iter()) {
            *g += s * dh;
        }
    }
    for (g, &dh) in gb1.iter_mut().zip(dl_dh_pre.iter()) {
        *g += dh;
    }
}

/// Predict an entire group of standardized samples. Used in the
/// validation loop where we need every prediction in a group to
/// compute SROCC vs ground truth.
#[allow(clippy::too_many_arguments)]
pub fn predict_group(
    std_x: &[f64],
    n_pairs: usize,
    n_features: usize,
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    n_hidden: usize,
    alpha: f64,
) -> Vec<f64> {
    (0..n_pairs)
        .map(|i| {
            let xi = &std_x[i * n_features..(i + 1) * n_features];
            let (y, _, _) = forward(xi, w1, b1, w2, b2, n_features, n_hidden, alpha);
            y
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaler_uniform_features_zero_var() {
        // All rows identical → mean = the row, std floored at 1e-8.
        let scores = vec![80.0, 80.0];
        let rows: Vec<Vec<f64>> = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]];
        let refs: Vec<&[f64]> = rows.iter().map(|v| v.as_slice()).collect();
        let g = TrainingGroup {
            name: "uniform".into(),
            human_scores: &scores,
            features: &refs,
            metric_sigmas: None,
            train_weight: 1.0,
            validation_weight: 0.0,
        };
        let (m, s) = compute_scaler_from_groups(&[g], &[0], 3);
        assert!((m[0] - 1.0).abs() < 1e-12);
        assert!((m[1] - 2.0).abs() < 1e-12);
        assert!((m[2] - 3.0).abs() < 1e-12);
        // Floor: 1e-8.
        for v in s {
            assert!((v - 1e-8).abs() < 1e-15, "expected std floor 1e-8, got {v}");
        }
    }

    #[test]
    fn scaler_two_group_mean_std() {
        // Group A rows: [0,0], [2,2]; group B rows: [4,4], [6,6].
        // Combined: 4 samples, mean = [3,3], var = (9+1+1+9)/4 = 5,
        // std = sqrt(5) ≈ 2.2360679775.
        let sa = vec![70.0, 80.0];
        let sb = vec![60.0, 50.0];
        let ra: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![2.0, 2.0]];
        let rb: Vec<Vec<f64>> = vec![vec![4.0, 4.0], vec![6.0, 6.0]];
        let rar: Vec<&[f64]> = ra.iter().map(|v| v.as_slice()).collect();
        let rbr: Vec<&[f64]> = rb.iter().map(|v| v.as_slice()).collect();
        let groups = [
            TrainingGroup {
                name: "A".into(),
                human_scores: &sa,
                features: &rar,
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 0.0,
            },
            TrainingGroup {
                name: "B".into(),
                human_scores: &sb,
                features: &rbr,
                metric_sigmas: None,
                train_weight: 1.0,
                validation_weight: 0.0,
            },
        ];
        let (m, s) = compute_scaler_from_groups(&groups, &[0, 1], 2);
        assert!((m[0] - 3.0).abs() < 1e-12);
        assert!((m[1] - 3.0).abs() < 1e-12);
        let expect = 5.0f64.sqrt();
        assert!((s[0] - expect).abs() < 1e-12, "got {}", s[0]);
        assert!((s[1] - expect).abs() < 1e-12);
    }

    #[test]
    fn forward_zero_weights_returns_bias() {
        // All zero weights → y = b2[0] regardless of input.
        let x = vec![1.0, 2.0, 3.0];
        let w1 = vec![0.0; 3 * 4]; // 3 → 4
        let b1 = vec![0.0; 4];
        let w2 = vec![0.0; 4];
        let b2 = vec![7.0];
        let (y, _, _) = forward(&x, &w1, &b1, &w2, &b2, 3, 4, 0.01);
        assert!((y - 7.0).abs() < 1e-12);
    }

    #[test]
    fn forward_linear_identity_passthrough() {
        // n_features=1, n_hidden=1, w1=[2.0], b1=[0], w2=[3.0], b2=[1].
        // For x=4.0: h_pre = 8.0 (≥0), h = 8.0, y = 8*3 + 1 = 25.
        let x = vec![4.0];
        let (y, hp, h) = forward(&x, &[2.0], &[0.0], &[3.0], &[1.0], 1, 1, 0.01);
        assert!((y - 25.0).abs() < 1e-12);
        assert!((hp[0] - 8.0).abs() < 1e-12);
        assert!((h[0] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn forward_leaky_negative_branch() {
        // Same as above but x = -4.0 → h_pre = -8.0 < 0 → h = -0.08,
        // y = -0.08*3 + 1 = 0.76.
        let x = vec![-4.0];
        let (y, hp, h) = forward(&x, &[2.0], &[0.0], &[3.0], &[1.0], 1, 1, 0.01);
        assert!((hp[0] + 8.0).abs() < 1e-12);
        assert!((h[0] + 0.08).abs() < 1e-12);
        assert!((y - 0.76).abs() < 1e-12);
    }

    #[test]
    fn backprop_accumulates_no_double_zero() {
        // Sanity: with x=0 across features, no gradient flow into gw1.
        let x = vec![0.0, 0.0];
        let h_pre = vec![1.0, 1.0];
        let h = vec![1.0, 1.0];
        let w1 = vec![0.1; 4];
        let mut gw1 = vec![0.0; 4];
        let mut gb1 = vec![0.0; 2];
        let w2 = vec![1.0, 1.0];
        let mut gw2 = vec![0.0; 2];
        let mut gb2 = vec![0.0];
        backprop_step(
            &x, &h_pre, &h, 1.0, &w1, &mut gw1, &mut gb1, &w2, &mut gw2, &mut gb2, 2, 2, 0.01,
        );
        assert!(gw1.iter().all(|&g| g == 0.0));
        assert_eq!(gw2, vec![1.0, 1.0]);
        assert_eq!(gb2, vec![1.0]);
        assert_eq!(gb1, vec![1.0, 1.0]);
    }

    #[test]
    fn bake_two_layer_znpr_v3_header_and_size() {
        // 2 inputs, 3 hidden, 1 output → w1 6 floats, b1 3, w2 3, b2 1,
        // scaler 2+2 floats. We just check the bytes start with the
        // ZNPR magic and that version is v3 — exact byte layout is
        // governed by zenpredict_bake::bake and exercised by its own tests.
        let bytes = bake_two_layer_znpr_v3(
            &[0.5, -0.2],
            &[1.0, 1.0],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            &[0.0, 0.0, 0.0],
            &[1.0, 2.0, 3.0],
            &[0.5],
            2,
            3,
            1,
        );
        assert_eq!(&bytes[0..4], b"ZNPR", "expected ZNPR magic");
        // u16 version at offset 4 — must be v3 per CLAUDE.md (v2 prohibited).
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        assert_eq!(version, 3, "expected v3 — v2 production is banned");
    }

    #[test]
    fn predict_group_batched_matches_singletons() {
        let std_x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0]; // 2 samples × 2 features
        let w1 = vec![0.1, 0.2, 0.3, 0.4]; // 2 × 2
        let b1 = vec![0.5, 0.5];
        let w2 = vec![1.0, 1.0];
        let b2 = vec![0.1];
        let n_features = 2;
        let n_hidden = 2;
        let alpha = 0.01;
        let preds = predict_group(&std_x, 2, n_features, &w1, &b1, &w2, &b2, n_hidden, alpha);
        let (y0, _, _) = forward(
            &std_x[0..2],
            &w1,
            &b1,
            &w2,
            &b2,
            n_features,
            n_hidden,
            alpha,
        );
        let (y1, _, _) = forward(
            &std_x[2..4],
            &w1,
            &b1,
            &w2,
            &b2,
            n_features,
            n_hidden,
            alpha,
        );
        assert!((preds[0] - y0).abs() < 1e-15);
        assert!((preds[1] - y1).abs() < 1e-15);
    }
}
