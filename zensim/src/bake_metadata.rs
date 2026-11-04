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

/// The output spline is shared by A; the retired v47 head is not served.
#[derive(Clone, Debug)]
pub(crate) struct ScoreMetadata {
    pub(crate) output_spline: Option<std::sync::Arc<OutputCalibrationSpline>>,
    pub(crate) derived_inputs: Option<std::sync::Arc<crate::derived_inputs::DerivedInputs>>,
}
pub(crate) fn parse_bake_metadata(model: &crate::mlp::Model) -> Result<ScoreMetadata, ZensimError> {
    let output_spline = match model.metadata().get(OUTPUT_CALIBRATION_SPLINE_KEY) {
        None => None,
        Some(entry) => Some(std::sync::Arc::new(
            parse_output_calibration_spline(entry.value).ok_or(ZensimError::ModelLoadFailed {
                reason: OUTPUT_CALIBRATION_SPLINE_KEY,
            })?,
        )),
    };
    let derived_inputs = crate::derived_inputs::from_model(model)
        .map_err(|reason| ZensimError::ModelLoadFailed { reason })?
        .map(std::sync::Arc::new);
    Ok(ScoreMetadata {
        output_spline,
        derived_inputs,
    })
}
