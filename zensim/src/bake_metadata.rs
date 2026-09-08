//! Validated ZNPR score metadata shared by runtime and model-author tooling.
//!
//! These are diagnostic wire-format views, not a second scoring surface.
//! Execute candidate models through `BakeScorer`; malformed present metadata
//! is an error, whereas an absent optional field is permitted.
use crate::ZensimError;

/// EXP-CROSS-CODEC-V9 (2026-05-20): post-network PCHIP spline calibration
/// metadata key. Payload is `[n_knots: u32 LE, n_knots × (x: f32 LE, y: f32 LE)]`,
/// i.e. `4 + 8·n_knots` bytes. Knots must be sorted strictly increasing by x.
/// When present, the runtime applies a monotone cubic Hermite (PCHIP)
/// interpolation to the post-tanh-pin score: `y_calibrated = pchip(y_pinned)`.
pub(crate) const OUTPUT_CALIBRATION_SPLINE_KEY: &str = "zentrain.output_calibration_spline";
pub(crate) const MINMAX_MONOTONE_HEAD_KEY: &str = "zentrain.minmax_monotone_head";
/// EXP-CROSS-CODEC-V11-E (2026-05-20): per-codec post-spline affine
/// calibration metadata key.
///
/// Payload layout (little-endian):
///   `[u32 n_codecs, n_codecs × (u32 name_len, name_len utf8 bytes, f32 alpha, f32 beta)]`
///
/// Applied AFTER the PCHIP spline (post all network forward + tanh-pin +
/// spline). For each entry, the runtime applies
/// `score_c = alpha_c + beta_c · spline(raw)` whenever the caller
/// supplies a matching codec name. Generic / unknown codec hint:
/// identity (alpha=0, beta=1).
///
/// The transform is monotone within codec (beta > 0 by construction),
/// so within-codec rank ordering is bit-exact preserved; only the
/// cross-codec systematic bias is adjusted toward consensus at JND
/// landmarks.
pub(crate) const PER_CODEC_CALIBRATION_KEY: &str = "zentrain.per_codec_calibration";
pub(crate) const HYBRID_HEAD_KEY: &str = "zentrain.hybrid_head";
pub(crate) const PER_SAMPLE_ALPHA_HEAD_KEY: &str = "zentrain.per_sample_alpha_head";
/// EXP-CROSS-CODEC-V4 (2026-05-19): tanh-pinned output head metadata
/// key. Payload is `[scale: f32 LE]` (4 bytes). When present, the
/// runtime wraps the per-sample-α head's raw output as
/// `y_score = 100 · σ(y_pre / scale)` — no post-hoc affine needed,
/// output is natively pinned to [0, 100].
pub(crate) const TANH_OUTPUT_HEAD_KEY: &str = "zentrain.tanh_output_head";

/// Parsed PCHIP spline payload: parallel `xs` and `ys` arrays plus the
/// precomputed monotone-Hermite slopes per knot (Fritsch–Carlson).
#[derive(Clone, Debug)]
pub struct OutputCalibrationSpline {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    /// Per-knot derivative (length == xs.len()).
    pub derivs: Vec<f64>,
}

/// Parse the `zentrain.output_calibration_spline` payload. Returns
/// `None` if the byte layout is wrong, knots are not strictly
/// increasing in x, or n_knots < 2.
pub fn parse_output_calibration_spline(payload: &[u8]) -> Option<OutputCalibrationSpline> {
    if payload.len() < 4 {
        return None;
    }
    let n = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    if n < 2 {
        return None;
    }
    let expected = 4usize.checked_add(8usize.checked_mul(n)?)?;
    if payload.len() != expected {
        return None;
    }
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let off = 4 + i * 8;
        let x = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as f64;
        let y = f32::from_le_bytes([
            payload[off + 4],
            payload[off + 5],
            payload[off + 6],
            payload[off + 7],
        ]) as f64;
        if !x.is_finite() || !y.is_finite() {
            return None;
        }
        xs.push(x);
        ys.push(y);
    }
    // Strictly increasing x. (All values were verified finite above, so
    // `<=` is an exact rewrite of the previous NaN-aware `!(a > b)`.)
    for i in 1..n {
        if xs[i] <= xs[i - 1] {
            return None;
        }
    }
    let derivs = crate::score_math::pchip_derivs(&xs, &ys);
    Some(OutputCalibrationSpline { xs, ys, derivs })
}

/// Parsed min-max monotone head: `k` outer-min groups × `j` inner-max pieces,
/// each a sign-constrained linear over `n` standardized features.
#[derive(Clone, Debug)]
pub struct MinMaxHeadMeta {
    pub k: usize,
    pub j: usize,
    pub n: usize,
    /// Row-major `[g][h][f]`, length `k·j·n`.
    pub w: Vec<f32>,
    /// Row-major `[g][h]`, length `k·j`.
    pub b: Vec<f32>,
}

/// Parse the `zentrain.minmax_monotone_head` payload. Returns `None` on any
/// layout error (short buffer, zero dims, wrong total length).
pub(crate) fn parse_minmax_head_meta(payload: &[u8]) -> Option<MinMaxHeadMeta> {
    if payload.len() < 12 {
        return None;
    }
    let rd_u32 = |o: usize| {
        u32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]]) as usize
    };
    let (k, j, n) = (rd_u32(0), rd_u32(4), rd_u32(8));
    if k == 0 || j == 0 || n == 0 {
        return None;
    }
    let n_w = k.checked_mul(j)?.checked_mul(n)?;
    let n_b = k.checked_mul(j)?;
    let expected = 12usize.checked_add(4usize.checked_mul(n_w.checked_add(n_b)?)?)?;
    if payload.len() != expected {
        return None;
    }
    let rd_f32 =
        |o: usize| f32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]]);
    let mut w = Vec::with_capacity(n_w);
    let mut off = 12;
    for _ in 0..n_w {
        w.push(rd_f32(off));
        off += 4;
    }
    let mut b = Vec::with_capacity(n_b);
    for _ in 0..n_b {
        b.push(rd_f32(off));
        off += 4;
    }
    if w.iter().chain(b.iter()).any(|v| !v.is_finite()) {
        return None;
    }
    Some(MinMaxHeadMeta { k, j, n, w, b })
}

/// One per-codec affine entry parsed from the metadata payload.
#[derive(Clone, Debug)]
pub(crate) struct PerCodecAffineEntry {
    /// Lowercase ASCII codec name (matched case-insensitively).
    pub name: String,
    /// `score = alpha + beta · raw`. Beta is positive by construction.
    pub alpha: f32,
    pub beta: f32,
}

/// Parsed per-codec calibration payload.
#[derive(Clone, Debug)]
pub(crate) struct PerCodecCalibration {
    pub entries: Vec<PerCodecAffineEntry>,
}

/// Parse the `zentrain.per_codec_calibration` payload. Returns
/// `None` if the payload is malformed (truncated header, ragged
/// entry, non-utf8 name, beta ≤ 0, non-finite alpha/beta).
pub(crate) fn parse_per_codec_calibration(payload: &[u8]) -> Option<PerCodecCalibration> {
    if payload.len() < 4 {
        return None;
    }
    let n_codecs = u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as usize;
    let mut off = 4usize;
    if n_codecs > (payload.len() - 4) / 12 {
        return None;
    }
    let mut entries: Vec<PerCodecAffineEntry> = Vec::with_capacity(n_codecs);
    for _ in 0..n_codecs {
        if off + 4 > payload.len() {
            return None;
        }
        let name_len = u32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]) as usize;
        off += 4;
        if off.checked_add(name_len)?.checked_add(8)? > payload.len() {
            return None;
        }
        let name_bytes = &payload[off..off + name_len];
        let name = std::str::from_utf8(name_bytes).ok()?.to_ascii_lowercase();
        off += name_len;
        let alpha = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        let beta = f32::from_le_bytes([
            payload[off],
            payload[off + 1],
            payload[off + 2],
            payload[off + 3],
        ]);
        off += 4;
        if !alpha.is_finite() || !beta.is_finite() || beta <= 0.0 {
            return None;
        }
        if name.is_empty() || entries.iter().any(|e| e.name == name) {
            return None;
        }
        entries.push(PerCodecAffineEntry { name, alpha, beta });
    }
    if off != payload.len() {
        return None;
    }
    Some(PerCodecCalibration { entries })
}

/// Parsed per-sample α head metadata payload.
#[derive(Clone, Debug)]
pub struct PerSampleAlphaMeta {
    pub w_alpha: Vec<f32>,
    pub b_alpha: f32,
    pub rank_w: Vec<f32>,
    pub rank_b: f32,
    pub reducer_w: [f32; 4],
    pub reducer_b: f32,
    pub p_norm: f32,
}

/// Parse the `zentrain.per_sample_alpha_head` payload. Returns
/// `None` if the payload length doesn't match `(2·n_hidden + 8)·4`.
pub(crate) fn parse_per_sample_alpha_meta(
    payload: &[u8],
    n_hidden: usize,
) -> Option<PerSampleAlphaMeta> {
    let expected = (2 * n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats: Vec<f32> = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in payload.as_chunks::<4>().0 {
        floats.push(f32::from_le_bytes(*chunk));
    }
    let w_alpha: Vec<f32> = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w: Vec<f32> = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    if floats.iter().any(|v| !v.is_finite()) || p_norm <= 0.0 {
        return None;
    }
    Some(PerSampleAlphaMeta {
        w_alpha,
        b_alpha,
        rank_w,
        rank_b,
        reducer_w,
        reducer_b,
        p_norm,
    })
}

/// Parsed hybrid-head metadata payload.
#[derive(Clone, Debug)]
pub struct HybridHeadMeta {
    pub rank_w: Vec<f32>,
    pub rank_b: f32,
    pub alpha_logit: f32,
    pub reducer_w: [f32; 4],
    pub reducer_b: f32,
    pub p_norm: f32,
}

/// Parse the `zentrain.hybrid_head` payload. Returns `None` if the
/// payload length doesn't match `(n_hidden + 8) · 4`.
pub(crate) fn parse_hybrid_head_meta(payload: &[u8], n_hidden: usize) -> Option<HybridHeadMeta> {
    let expected = (n_hidden + 8) * 4;
    if payload.len() != expected {
        return None;
    }
    let mut floats: Vec<f32> = Vec::with_capacity(n_hidden + 8);
    for chunk in payload.as_chunks::<4>().0 {
        floats.push(f32::from_le_bytes(*chunk));
    }
    let rank_w: Vec<f32> = floats[..n_hidden].to_vec();
    let rank_b = floats[n_hidden];
    let alpha_logit = floats[n_hidden + 1];
    let reducer_w = [
        floats[n_hidden + 2],
        floats[n_hidden + 3],
        floats[n_hidden + 4],
        floats[n_hidden + 5],
    ];
    let reducer_b = floats[n_hidden + 6];
    let p_norm = floats[n_hidden + 7];
    if floats.iter().any(|v| !v.is_finite()) || p_norm <= 0.0 {
        return None;
    }
    Some(HybridHeadMeta {
        rank_w,
        rank_b,
        alpha_logit,
        reducer_w,
        reducer_b,
        p_norm,
    })
}

/// Lazily-parsed bake metadata bundle. One instance per distinct
/// bake-bytes pointer, cached in [`bake_metadata_cache`] so the
/// per-sample-α, hybrid-head, tanh-pin, PCHIP-spline, and per-codec
/// calibration payloads only parse once.
///
/// Hot-loop motivation: encoder workloads call
/// `forward_one_bake_with_codec` once per distorted candidate against
/// a fixed `ProfileParams::mlp_bytes` slot. Re-parsing five metadata
/// blobs every call burned a constant ~µs that this cache reclaims.
///
/// Each field is `Option<Arc<T>>` so cloning the bundle (to release
/// the cache lock before forward dispatch) is cheap.
#[derive(Clone, Debug)]
pub struct ScoreMetadata {
    pub per_sample_alpha: Option<std::sync::Arc<PerSampleAlphaMeta>>,
    pub hybrid_head: Option<std::sync::Arc<HybridHeadMeta>>,
    pub minmax_head: Option<std::sync::Arc<MinMaxHeadMeta>>,
    pub tanh_pin_scale: Option<f64>,
    pub output_spline: Option<std::sync::Arc<OutputCalibrationSpline>>,
    pub(crate) per_codec_calibration: Option<std::sync::Arc<PerCodecCalibration>>,
}

pub fn parse_bake_metadata(model: &crate::mlp::Model) -> Result<ScoreMetadata, ZensimError> {
    use std::sync::Arc;
    fn optional<T>(
        model: &crate::mlp::Model,
        key: &'static str,
        parse: impl FnOnce(&[u8]) -> Option<T>,
    ) -> Result<Option<T>, ZensimError> {
        match model.metadata().get(key) {
            None => Ok(None),
            Some(entry) => parse(entry.value)
                .map(Some)
                .ok_or(ZensimError::ModelLoadFailed { reason: key }),
        }
    }
    let n = model.n_outputs();
    let per_sample_alpha = optional(model, PER_SAMPLE_ALPHA_HEAD_KEY, |v| {
        parse_per_sample_alpha_meta(v, n)
    })?
    .map(Arc::new);
    let hybrid_head =
        optional(model, HYBRID_HEAD_KEY, |v| parse_hybrid_head_meta(v, n))?.map(Arc::new);
    let minmax_head =
        optional(model, MINMAX_MONOTONE_HEAD_KEY, parse_minmax_head_meta)?.map(Arc::new);
    if usize::from(per_sample_alpha.is_some())
        + usize::from(hybrid_head.is_some())
        + usize::from(minmax_head.is_some())
        > 1
    {
        return Err(ZensimError::ModelLoadFailed {
            reason: "multiple incompatible output heads",
        });
    }
    let tanh_pin_scale = optional(model, TANH_OUTPUT_HEAD_KEY, parse_tanh_output_head_scale)?;
    let output_spline = optional(
        model,
        OUTPUT_CALIBRATION_SPLINE_KEY,
        parse_output_calibration_spline,
    )?
    .map(Arc::new);
    let per_codec_calibration = optional(
        model,
        PER_CODEC_CALIBRATION_KEY,
        parse_per_codec_calibration,
    )?
    .map(Arc::new);
    Ok(ScoreMetadata {
        per_sample_alpha,
        hybrid_head,
        minmax_head,
        tanh_pin_scale,
        output_spline,
        per_codec_calibration,
    })
}

/// Parse the `zentrain.tanh_output_head` payload — a single f32 LE
/// (4 bytes) encoding the sigmoid pin scale. Returns `None` if the
/// payload length is wrong or the scale is non-positive / non-finite.
pub(crate) fn parse_tanh_output_head_scale(payload: &[u8]) -> Option<f64> {
    if payload.len() != 4 {
        return None;
    }
    let scale = f32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]]) as f64;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}
