//! Compose v1 ZNPK byte streams.
//!
//! Used to bake the V0_4 placeholder weights at runtime (via
//! `LazyLock`) and to write artifacts in `zensim-validate`. The
//! format is documented in [`super::model`].

/// Single-layer 228→1 linear MLP from a flat f64 weight vector.
///
/// `weights[i]` is multiplied by feature `i`; the per-input scaler
/// is identity (mean=0, scale=1) so the on-the-fly f64→f32 input
/// conversion in `MlpScorer::score_into` lines up with the bake.
///
/// Used to wrap `WEIGHTS_PREVIEW_V0_2` as the V0_4 placeholder. When
/// real V0_4 weights ship, this helper goes away — multi-layer bakes
/// will live in `zensim-validate`.
pub fn bake_linear_v1(weights: &[f64], schema_hash: u64) -> Vec<u8> {
    let n_inputs = weights.len();
    assert!(n_inputs > 0, "weights must be non-empty");
    let n_outputs = 1usize;
    let n_layers = 1usize;

    // Header (32 bytes).
    let mut buf = Vec::with_capacity(32 + 8 * n_inputs + 12 + 4 * n_inputs + 4);
    buf.extend_from_slice(b"ZNPK");
    buf.extend_from_slice(&1u16.to_le_bytes()); // version
    buf.extend_from_slice(&32u16.to_le_bytes()); // header_size
    buf.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    buf.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    buf.extend_from_slice(&(n_layers as u32).to_le_bytes());
    buf.extend_from_slice(&schema_hash.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // flags
    debug_assert_eq!(buf.len(), 32);

    // Scaler: identity (mean=0, scale=1).
    for _ in 0..n_inputs {
        buf.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for _ in 0..n_inputs {
        buf.extend_from_slice(&1.0f32.to_le_bytes());
    }

    // Single layer: in=n_inputs, out=1, identity activation, f32 weights.
    buf.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    buf.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    buf.push(0); // activation = Identity
    buf.push(0); // weight_dtype = F32
    buf.extend_from_slice(&[0, 0]); // reserved

    for &w in weights {
        buf.extend_from_slice(&(w as f32).to_le_bytes());
    }
    // Single bias = 0.
    buf.extend_from_slice(&0.0f32.to_le_bytes());

    buf
}
