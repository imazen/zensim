//! Synthetic-model roundtrip tests for the vendored MLP runtime.
//!
//! Builds a tiny MLP in memory, serializes it to the v1 binary format,
//! loads it via [`Model::from_bytes`], runs [`forward`], and verifies
//! the output matches a hand-computed reference. Covers all three
//! weight storage paths (f32, f16, i8) and key error cases.
//!
//! Vendored from `zenpicker::tests` v0.1.0 (originally
//! AGPL-3.0-only OR LicenseRef-Imazen-Commercial), re-licensed under
//! zensim's MIT OR Apache-2.0 by the copyright holder.

use super::error::MlpError;
use super::inference::forward;
use super::model::Model;

/// 8-aligned byte buffer for tests. Real callers wrap their
/// `include_bytes!` literal in `#[repr(C, align(8))] struct
/// Aligned([u8; N])` to guarantee alignment.
struct AlignedBuf {
    storage: Box<[u64]>,
    len: usize,
}

impl AlignedBuf {
    fn from_slice(src: &[u8]) -> Self {
        let n_u64 = src.len().div_ceil(8);
        let mut storage = vec![0u64; n_u64.max(1)].into_boxed_slice();
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        bytes[..src.len()].copy_from_slice(src);
        Self {
            storage,
            len: src.len(),
        }
    }

    fn as_bytes(&self) -> &[u8] {
        let bytes: &[u8] = bytemuck::cast_slice(&self.storage);
        &bytes[..self.len]
    }
}

/// Convert an f32 to its IEEE-754 half-precision bit pattern.
/// Round-to-nearest-even on mantissa truncation, with proper handling
/// of subnormal/inf/NaN.
fn f32_to_f16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp32 = ((bits >> 23) & 0xff) as i32;
    let mant32 = bits & 0x007f_ffff;
    if exp32 == 0xff {
        let m = if mant32 != 0 {
            ((mant32 >> 13) | 0x200) as u16
        } else {
            0
        };
        (sign << 15) | 0x7c00 | m
    } else if exp32 == 0 {
        sign << 15
    } else {
        let exp16 = exp32 - 127 + 15;
        if exp16 >= 0x1f {
            (sign << 15) | 0x7c00
        } else if exp16 <= 0 {
            if exp16 < -10 {
                sign << 15
            } else {
                let mant_with_implicit = mant32 | 0x0080_0000;
                let shift = 14 - exp16;
                let m = mant_with_implicit >> shift;
                let half = 1u32 << (shift - 1);
                let m = if (mant_with_implicit & half) != 0
                    && ((mant_with_implicit & (half - 1)) != 0 || (m & 1) != 0)
                {
                    m + 1
                } else {
                    m
                };
                (sign << 15) | (m as u16)
            }
        } else {
            let m = mant32 >> 13;
            let half = 1u32 << 12;
            let lower = mant32 & 0x1fff;
            let m = if lower > half || (lower == half && (m & 1) != 0) {
                m + 1
            } else {
                m
            };
            let mut e = exp16 as u32;
            let mut m = m;
            if m & 0x400 != 0 {
                m = 0;
                e += 1;
            }
            if e >= 0x1f {
                (sign << 15) | 0x7c00
            } else {
                (sign << 15) | ((e as u16) << 10) | (m as u16)
            }
        }
    }
}

/// Per-output i8 quantize a row-major f32 weight block of shape
/// (in_dim, out_dim). All-zero columns get `scale = 1.0` to avoid
/// div-by-zero.
fn quantize_i8_per_output(w: &[f32], in_dim: usize, out_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut scales = vec![0.0f32; out_dim];
    for o in 0..out_dim {
        let mut m = 0.0f32;
        for i in 0..in_dim {
            let a = w[i * out_dim + o].abs();
            if a > m {
                m = a;
            }
        }
        scales[o] = if m == 0.0 { 1.0 } else { m / 127.0 };
    }
    let mut q = vec![0i8; in_dim * out_dim];
    for i in 0..in_dim {
        for o in 0..out_dim {
            let v = w[i * out_dim + o] / scales[o];
            let r = v.round().clamp(-128.0, 127.0) as i32;
            q[i * out_dim + o] = r as i8;
        }
    }
    (q, scales)
}

/// Test layer-spec tuple: (in_dim, out_dim, activation_byte, weights, biases).
type LayerSpec<'a> = (usize, usize, u8, &'a [f32], &'a [f32]);

fn write_header(out: &mut Vec<u8>, n_inputs: usize, n_outputs: usize, n_layers: usize, schema_hash: u64) {
    out.extend_from_slice(b"ZNPK");
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(&(n_inputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_outputs as u32).to_le_bytes());
    out.extend_from_slice(&(n_layers as u32).to_le_bytes());
    out.extend_from_slice(&schema_hash.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    debug_assert_eq!(out.len(), 32);
}

fn write_v1_model_f32_with_scaler(
    out: &mut Vec<u8>,
    n_inputs: usize,
    scaler_mean: &[f32],
    scaler_scale: &[f32],
    layers: &[LayerSpec<'_>],
    schema_hash: u64,
) {
    debug_assert_eq!(scaler_mean.len(), n_inputs);
    debug_assert_eq!(scaler_scale.len(), n_inputs);
    let n_outputs = layers.last().unwrap().1;
    out.clear();
    write_header(out, n_inputs, n_outputs, layers.len(), schema_hash);
    for &m in scaler_mean {
        out.extend_from_slice(&m.to_le_bytes());
    }
    for &s in scaler_scale {
        out.extend_from_slice(&s.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(0); // weight_dtype = f32
        out.extend_from_slice(&[0, 0]);
        for &val in w {
            out.extend_from_slice(&val.to_le_bytes());
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

fn write_v1_model_f32(out: &mut Vec<u8>, n_inputs: usize, layers: &[LayerSpec<'_>], schema_hash: u64) {
    write_v1_model_f32_with_scaler(
        out,
        n_inputs,
        &vec![0.0; n_inputs],
        &vec![1.0; n_inputs],
        layers,
        schema_hash,
    );
}

fn write_v1_model_f16(out: &mut Vec<u8>, n_inputs: usize, layers: &[LayerSpec<'_>], schema_hash: u64) {
    let n_outputs = layers.last().unwrap().1;
    out.clear();
    write_header(out, n_inputs, n_outputs, layers.len(), schema_hash);
    for _ in 0..n_inputs {
        out.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for _ in 0..n_inputs {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(1); // weight_dtype = f16
        out.extend_from_slice(&[0, 0]);
        for &val in w {
            out.extend_from_slice(&f32_to_f16_bits(val).to_le_bytes());
        }
        if (in_d * out_d) % 2 == 1 {
            out.extend_from_slice(&[0, 0]);
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

fn write_v1_model_i8(out: &mut Vec<u8>, n_inputs: usize, layers: &[LayerSpec<'_>], schema_hash: u64) {
    let n_outputs = layers.last().unwrap().1;
    out.clear();
    write_header(out, n_inputs, n_outputs, layers.len(), schema_hash);
    for _ in 0..n_inputs {
        out.extend_from_slice(&0.0f32.to_le_bytes());
    }
    for _ in 0..n_inputs {
        out.extend_from_slice(&1.0f32.to_le_bytes());
    }
    for &(in_d, out_d, act, w, b) in layers {
        out.extend_from_slice(&(in_d as u32).to_le_bytes());
        out.extend_from_slice(&(out_d as u32).to_le_bytes());
        out.push(act);
        out.push(2); // weight_dtype = i8
        out.extend_from_slice(&[0, 0]);
        let (q, scales) = quantize_i8_per_output(w, in_d, out_d);
        for &v in &q {
            out.push(v as u8);
        }
        let pad = (4 - ((in_d * out_d) % 4)) % 4;
        for _ in 0..pad {
            out.push(0);
        }
        for &s in &scales {
            out.extend_from_slice(&s.to_le_bytes());
        }
        for &val in b {
            out.extend_from_slice(&val.to_le_bytes());
        }
    }
}

fn run(model: &Model<'_>, features: &[f32]) -> Vec<f32> {
    let n_out = model.n_outputs();
    let scratch_len = model.scratch_len();
    let mut a = vec![0f32; scratch_len];
    let mut b = vec![0f32; scratch_len];
    let mut out = vec![0f32; n_out];
    forward(model, features, &mut a, &mut b, &mut out).unwrap();
    out
}

#[test]
fn parse_minimal_one_layer_identity() {
    let mut buf = Vec::new();
    write_v1_model_f32(
        &mut buf,
        2,
        &[(2, 3, 0, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 1.0])],
        0xDEAD_BEEF_CAFE_F00D,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    assert_eq!(model.n_inputs(), 2);
    assert_eq!(model.n_outputs(), 3);
    assert_eq!(model.schema_hash(), 0xDEAD_BEEF_CAFE_F00D);

    let out = run(&model, &[3.0, 5.0]);
    assert_eq!(out, vec![3.0, 5.0, 1.0]);
}

#[test]
fn leaky_relu_scales_negatives() {
    let mut buf = Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 2, 2, &[-2.0, 1.0], &[0.0, 0.0])], 0);
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let out = run(&model, &[1.0]);
    assert!((out[0] - (-0.02)).abs() < 1e-6, "leaky relu negative leg: got {} expected -0.02", out[0]);
    assert!((out[1] - 1.0).abs() < 1e-6);
}

#[test]
fn relu_zeros_negatives() {
    let mut buf = Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 2, 1, &[-2.0, 1.0], &[0.0, 0.0])], 0);
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let out = run(&model, &[1.0]);
    assert_eq!(out, vec![0.0, 1.0]);
}

#[test]
fn two_layer_mlp() {
    let mut buf = Vec::new();
    write_v1_model_f32(
        &mut buf,
        2,
        &[
            (2, 4, 1, &[1.0, -1.0, 0.5, 0.0, 0.0, 1.0, 0.5, 1.0], &[0.0, 0.0, 0.0, 0.0]),
            (4, 3, 0, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0]),
        ],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let out = run(&model, &[2.0, 3.0]);
    assert!((out[0] - 5.0).abs() < 1e-5);
    assert!((out[1] - 4.0).abs() < 1e-5);
    assert!((out[2] - 5.5).abs() < 1e-5);
}

#[test]
fn scaler_divides_by_std_not_multiplies() {
    // Regression: bake stores sklearn's `StandardScaler.scale_` (= std).
    // Runtime must DIVIDE by it. A multiply would silently miscalibrate.
    // For x=14, mean=10, scale=4: correct standardized value = 1.0.
    let mut buf = Vec::new();
    write_v1_model_f32_with_scaler(
        &mut buf,
        1,
        &[10.0],
        &[4.0],
        &[(1, 1, 0, &[1.0], &[0.0])],
        0,
    );
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let out = run(&model, &[14.0]);
    let want = 1.0_f32;
    assert!(
        (out[0] - want).abs() < 1e-6,
        "scaler kernel produced {} for x=14 (mean=10, scale=4); expected {} = (x - mean) / scale",
        out[0],
        want,
    );
}

#[test]
fn f16_storage_matches_f32_within_quant_error() {
    // Same model, two encodings — outputs should match within f16 quant noise.
    let weights = [1.5_f32, -0.75, 0.25, -2.0, 0.125, 4.0];
    let biases = [0.5_f32, -1.0];
    let layer: LayerSpec = (3, 2, 0, &weights, &biases);

    let mut buf_f32 = Vec::new();
    write_v1_model_f32(&mut buf_f32, 3, &[layer], 0);
    let buf_f32 = AlignedBuf::from_slice(&buf_f32);
    let model_f32 = Model::from_bytes(buf_f32.as_bytes()).unwrap();

    let mut buf_f16 = Vec::new();
    write_v1_model_f16(&mut buf_f16, 3, &[layer], 0);
    let buf_f16 = AlignedBuf::from_slice(&buf_f16);
    let model_f16 = Model::from_bytes(buf_f16.as_bytes()).unwrap();

    let features = [0.5_f32, 1.0, -0.25];
    let out_f32 = run(&model_f32, &features);
    let out_f16 = run(&model_f16, &features);
    for (a, b) in out_f32.iter().zip(out_f16.iter()) {
        assert!((a - b).abs() < 1e-3, "f16 vs f32: {a} vs {b}");
    }
}

#[test]
fn i8_storage_matches_f32_within_quant_error() {
    // Same model, f32 and i8 encodings — outputs match within i8 quant
    // RMS noise (~1% relative). Small layer keeps absolute error tight.
    let weights = [1.5_f32, -0.75, 0.25, -2.0, 0.125, 4.0];
    let biases = [0.5_f32, -1.0];
    let layer: LayerSpec = (3, 2, 0, &weights, &biases);

    let mut buf_f32 = Vec::new();
    write_v1_model_f32(&mut buf_f32, 3, &[layer], 0);
    let buf_f32 = AlignedBuf::from_slice(&buf_f32);
    let model_f32 = Model::from_bytes(buf_f32.as_bytes()).unwrap();

    let mut buf_i8 = Vec::new();
    write_v1_model_i8(&mut buf_i8, 3, &[layer], 0);
    let buf_i8 = AlignedBuf::from_slice(&buf_i8);
    let model_i8 = Model::from_bytes(buf_i8.as_bytes()).unwrap();

    let features = [0.5_f32, 1.0, -0.25];
    let out_f32 = run(&model_f32, &features);
    let out_i8 = run(&model_i8, &features);
    for (a, b) in out_f32.iter().zip(out_i8.iter()) {
        // ~1% relative + small absolute slack for the per-output scaler granularity.
        let tol = a.abs() * 0.02 + 0.05;
        assert!((a - b).abs() < tol, "i8 vs f32: {a} vs {b} (tol {tol})");
    }
}

#[test]
fn bad_magic_rejected() {
    let mut buf = Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 1, 0, &[1.0], &[0.0])], 0);
    buf[0] = b'X';
    let aligned = AlignedBuf::from_slice(&buf);
    match Model::from_bytes(aligned.as_bytes()) {
        Err(MlpError::BadMagic { .. }) => {}
        other => panic!("expected BadMagic, got {other:?}"),
    }
}

#[test]
fn truncated_rejected() {
    let mut buf = Vec::new();
    write_v1_model_f32(&mut buf, 1, &[(1, 1, 0, &[1.0], &[0.0])], 0);
    buf.truncate(20); // chop mid-header
    let aligned = AlignedBuf::from_slice(&buf);
    match Model::from_bytes(aligned.as_bytes()) {
        Err(MlpError::Truncated { .. }) => {}
        other => panic!("expected Truncated, got {other:?}"),
    }
}

#[test]
fn feature_len_mismatch_rejected() {
    let mut buf = Vec::new();
    write_v1_model_f32(&mut buf, 3, &[(3, 1, 0, &[1.0, 1.0, 1.0], &[0.0])], 0);
    let aligned = AlignedBuf::from_slice(&buf);
    let model = Model::from_bytes(aligned.as_bytes()).unwrap();
    let mut a = vec![0f32; model.scratch_len()];
    let mut b = vec![0f32; model.scratch_len()];
    let mut out = vec![0f32; 1];
    let err = forward(&model, &[1.0, 2.0], &mut a, &mut b, &mut out).unwrap_err();
    matches!(err, MlpError::FeatureLenMismatch { expected: 3, got: 2 })
        .then_some(())
        .unwrap_or_else(|| panic!("expected FeatureLenMismatch, got {err:?}"));
}
