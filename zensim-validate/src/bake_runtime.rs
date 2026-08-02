//! Shared per-row bake scoring runtime — DEDUP-M.
//!
//! Until 2026-05-26 every binary that scored rows through a zenpredict
//! `Predictor` rolled its own `score_row` + extract-from-metadata helpers
//! (bake_verdict / qsweep_eval / preview_stats_demo / ensemble_score_rows /
//! score_pair_with_bake / predict_features_with_bake — six bins, ~90-95 %
//! shared logic). The canonical `zensim::metric::score_features_with_profile`
//! consumes a compiled-in `ZensimProfile`, NOT user-provided bake bytes, so
//! the bin paths can't call it directly — they need the same per-sample-α /
//! hybrid-head / tanh-pin / output-spline dispatch that
//! `zensim::metric::apply_mlp_scoring_with_codec` runs over a
//! `pub(crate)` surface.
//!
//! This module hosts the dispatch once and the six bins delegate to it.
//! The formulas are bit-exact to the pre-DEDUP-M per-bin copies and to
//! the canonical CPU flow:
//!
//! - `apply_per_sample_alpha_runtime` ↔ per-sample-α head
//! - `apply_hybrid_head_runtime` ↔ hybrid head
//! - `apply_tanh_output_pin` (`zensim::metric`) ↔ tanh output pin
//! - `apply_output_calibration_spline` (`zensim::metric`) ↔ output spline
//!
//! Acceptance gates (verified per DEDUP-M):
//! - Pre/post numerical output identical to f32 ±1e-6 on representative
//!   parquet rows (per-bin evidence files).
//! - All existing zensim-validate tests pass before+after.
//!
//! NO ALGORITHM CHANGE. The same formulas, the same floor (`0.0026`), the
//! same `clamp(-20.0, 20.0)` / `clamp(-30.0, 30.0)` bounds, the same
//! NaN-propagation. Bit-exact.
//!
//! # DEDUP-M2 (2026-05-26): HONEST-STOP — delegation to canonical zensim helper INFEASIBLE
//!
//! The DEDUP-M2 follow-on chunk attempted to remove this module's ~150 LOC of
//! per-row dispatch by promoting `zensim::metric::apply_mlp_scoring_with_codec`
//! to `pub` and having [`score_row`] delegate to it. **Analysis ruled out the
//! delegation; this module stays as the canonical bake-runtime path.** Reasons:
//!
//! 1. **Type-shape mismatch on inputs.** `apply_mlp_scoring_with_codec` takes
//!    `(&mut ZensimResult, &ProfileParams, w, h, codec_hint)`. `ZensimResult`
//!    carries a fully-computed feature vector from a real image-processing
//!    pipeline and exposes `pub(crate)` mutators (`set_mlp_score`,
//!    `mark_identical`). The bake-runtime path consumes a **single parquet row
//!    of f64 features** + a **long-lived `Predictor<'_>` + pre-allocated f32
//!    scratch** (re-used across ≥10k rows in hot loops like `bake_verdict`).
//!    Forcing parquet rows through `ZensimResult` would mean fabricating an
//!    instance per row + leaking `pub` on every mutator — not a localized API
//!    addition.
//!
//! 2. **Compile-time vs runtime bake bytes.** `ProfileParams::mlp_bytes` is
//!    `Option<fn() -> &'static [u8]>` — a function pointer returning a
//!    `&'static [u8]`. Every shipped profile loads its bake at compile time.
//!    The bake-runtime path takes runtime-loaded bake bytes from CLI args
//!    (e.g., `bake_verdict --bake /path/to/some.bin`), which cannot satisfy
//!    the `fn() -> &'static` signature without leaking the bytes into a
//!    static slot — an unbounded heap leak per CLI invocation.
//!
//! 3. **Scope mismatch on post-processing.** `apply_mlp_scoring_with_codec`
//!    runs the FULL canonical pipeline: ensemble classifier routing
//!    (`ensemble_classifier_bytes`), primary/B3 mix (`mlp_bytes_b3 +
//!    mlp_primary_mix`), `score_mapping_a/b` (`100 − A·d^B`),
//!    soft/hard/`extrapolate_score` clamping, per-codec post-spline affine
//!    calibration. [`score_row`] runs ONLY the per-sample-α / hybrid-head /
//!    tanh-pin / output-spline subset — the bake-eval tooling needs raw
//!    pre-clamp output for diagnostic dumps, and the ensemble/primary-mix
//!    knobs live one level up in `ensemble_score_rows`. Forcing the bake-eval
//!    sites through the full pipeline would lose the diagnostic-dump
//!    surface area + double-apply the post-spline calibration in
//!    `predict_features_with_bake`.
//!
//! 4. **Predictor reuse is structurally incompatible.** `forward_one_bake_with_codec`
//!    (called from inside `apply_mlp_scoring_with_codec`) constructs a fresh
//!    `Predictor::new(&model)` on every call. The bake-runtime path hot-loops
//!    over parquet rows with a single long-lived `Predictor` — the ~µs
//!    per-row Predictor construction would dominate wall time on 10k-row
//!    corpora (KADID, CID22) for the same forward math.
//!
//! **The two functions serve different purposes**: `apply_mlp_scoring_with_codec`
//! is the **encoder-side** full-pipeline ZensimResult mutator (one call per
//! distorted candidate against a compile-time profile); [`score_row`] is the
//! **eval-tooling** per-parquet-row helper (millions of calls against a
//! runtime-loaded bake). Both ARE bit-exact on the shared math (per-sample-α,
//! hybrid head, tanh-pin, output spline) — verified by the
//! `cid22_aggregate_srocc_matches_audit_reference` /
//! `cid22_first_row_matches_bake_verdict_reference` regression gates.
//!
//! **Future M3 candidates** (not in M2 scope):
//!
//! - **M3a — extract `forward_one_bake_with_codec` into a runtime-bytes API
//!   on `zensim::mlp` or new `zensim::scoring`**: would accept `&[u8]` bake
//!   bytes + an owned `Predictor` constructor knob (or a borrowed
//!   `&mut Predictor`). Then [`score_row`] could delegate the inner forward +
//!   metadata dispatch and keep its own scratch/Predictor reuse around it.
//!   Costs ~2-3 days: needs to factor out the bake-metadata cache (currently
//!   keyed on bake bytes pointer, which assumes `&'static [u8]`), the
//!   per-codec affine wiring, and the `Predictor::predict_transformed` vs
//!   `predict` choice. Bit-exact regression gates above MUST stay green.
//! - **M3b — propose a thin trait `BakeForwardOps`** that abstracts the
//!   shared math so `apply_mlp_scoring_with_codec` and [`score_row`] both
//!   delegate to one trait impl, leaving each function's
//!   bytes-loading/Predictor-reuse policy local. Lower-risk alternative to
//!   M3a; same regression gate.
//!
//! Until M3 lands, this module remains the canonical per-row bake-runtime
//! dispatch. No `apply_mlp_scoring_with_codec` promotion to `pub`; the
//! `pub(crate)` boundary continues to protect the encoder-side full pipeline
//! from being accidentally driven by parquet-row tooling.

use zenpredict::{Model, Predictor};

use crate::output_calibration_spline::{self, OutputCalibrationSpline};

/// Per-sample-α head dispatch payload — parsed from the bake's
/// `zentrain.per_sample_alpha_head` metadata. Layout matches
/// `zensim-train-core::per_sample_alpha_head::bake_per_sample_alpha_head_v3_with_tanh`.
///
/// `(W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm)`.
pub type PerSampleAlphaHeadDispatch = (Vec<f32>, f32, Vec<f32>, f32, [f32; 4], f32, f32);

/// Hybrid-head dispatch payload — parsed from the bake's
/// `zentrain.hybrid_head` metadata. Layout matches
/// `zensim-train-core::hybrid_head::bake_hybrid_head_v3`.
///
/// `(rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)`.
pub type HybridHeadDispatch = (Vec<f32>, f32, f32, [f32; 4], f32, f32);

/// Pool-stat sigma floor — bit-exact with `zensim::metric` (0.0026).
const POOL_STD_FLOOR: f64 = 0.0026;

/// Read the `zentrain.tanh_output_head` metadata payload, if any.
/// Returns the sigmoid pin scale (`f32 LE`, single value) or `None`
/// when the key is absent / payload malformed.
///
/// EXP-CROSS-CODEC-V4 (2026-05-19). When present, the final score
/// returned from [`score_row`] is wrapped as `100·σ(y_pre/scale)`,
/// matching `zensim::metric::apply_tanh_output_pin` bit-exactly.
pub fn extract_tanh_output_head_scale(model: &Model) -> Option<f64> {
    let md = model.metadata();
    let entry = md.get("zentrain.tanh_output_head")?;
    if entry.value.len() != 4 {
        return None;
    }
    let scale = f32::from_le_bytes([
        entry.value[0],
        entry.value[1],
        entry.value[2],
        entry.value[3],
    ]) as f64;
    if scale.is_finite() && scale > 0.0 {
        Some(scale)
    } else {
        None
    }
}

/// Read the `zentrain.per_sample_alpha_head` metadata payload, if any.
/// Returns `Some((W_α, b_α, rank_w, rank_b, reducer_w, reducer_b, p_norm))`.
pub fn extract_per_sample_alpha_head(model: &Model) -> Option<PerSampleAlphaHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.per_sample_alpha_head")?;
    let n_hidden = model.n_outputs();
    let expected = (2 * n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(2 * n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let w_alpha = floats[..n_hidden].to_vec();
    let b_alpha = floats[n_hidden];
    let rank_w = floats[n_hidden + 1..2 * n_hidden + 1].to_vec();
    let rank_b = floats[2 * n_hidden + 1];
    let reducer_w = [
        floats[2 * n_hidden + 2],
        floats[2 * n_hidden + 3],
        floats[2 * n_hidden + 4],
        floats[2 * n_hidden + 5],
    ];
    let reducer_b = floats[2 * n_hidden + 6];
    let p_norm = floats[2 * n_hidden + 7];
    Some((
        w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm,
    ))
}

/// Read the `zentrain.hybrid_head` metadata payload, if any.
/// Returns `Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))`.
pub fn extract_hybrid_head(model: &Model) -> Option<HybridHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.hybrid_head")?;
    let n_hidden = model.n_outputs();
    let expected = (n_hidden + 8) * 4;
    if entry.value.len() != expected {
        return None;
    }
    let mut floats = Vec::with_capacity(n_hidden + 8);
    for chunk in entry.value.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let rank_w = floats[..n_hidden].to_vec();
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
    Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm))
}

/// Min-max monotone head (Sill 1998) parsed from `zentrain.minmax_monotone_head`.
/// The score REPLACES the layer forward — `min_g max_h (w[g][h]·x + b[g][h])` over
/// the transform→scale→clamp±8 feature vector — so its runtime BYPASSES the
/// Predictor (see [`score_row_minmax`]). Bit-exact mirror of the min-max branch
/// in `zensim::metric` (the encoder-side path), same as the per-sample-α / hybrid
/// heads are mirrored here.
#[derive(Clone, Debug)]
pub struct MinMaxHeadDispatch {
    pub k: usize,
    pub j: usize,
    pub n: usize,
    /// Row-major `[g][h][f]`, length `k·j·n`.
    pub w: Vec<f32>,
    /// Row-major `[g][h]`, length `k·j`.
    pub b: Vec<f32>,
}

/// Read the `zentrain.minmax_monotone_head` payload, if any. Layout:
/// `[u32 k, u32 j, u32 n, w f32×(k·j·n), b f32×(k·j)]`.
pub fn extract_minmax_head(model: &Model) -> Option<MinMaxHeadDispatch> {
    let md = model.metadata();
    let entry = md.get("zentrain.minmax_monotone_head")?;
    let v = entry.value;
    if v.len() < 12 {
        return None;
    }
    let rd_u32 = |o: usize| u32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]) as usize;
    let (k, j, n) = (rd_u32(0), rd_u32(4), rd_u32(8));
    if k == 0 || j == 0 || n == 0 {
        return None;
    }
    let n_w = k.checked_mul(j)?.checked_mul(n)?;
    let n_b = k.checked_mul(j)?;
    let expected = 12usize.checked_add(4usize.checked_mul(n_w.checked_add(n_b)?)?)?;
    if v.len() != expected {
        return None;
    }
    let rd_f32 = |o: usize| f32::from_le_bytes([v[o], v[o + 1], v[o + 2], v[o + 3]]);
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
    Some(MinMaxHeadDispatch { k, j, n, w, b })
}

/// Score one row through a min-max monotone bake — the min-max REPLACES the
/// layer forward, so this reads the bake's scaler + feature transforms directly
/// and never touches a `Predictor`. Applies transform→scale→clamp±8→
/// `min_g max_h(w·x+b)`, then the shared tanh-pin + output spline. The ±8 clamp
/// mirrors the trainer's standardize clamp (`train_minmax`), so eval scoring is
/// bit-exact with the in-process held-out eval that qualified the bake. Returns
/// `f64::NAN` if the head's `n` disagrees with the bake's `n_inputs`.
pub fn score_row_minmax(
    model: &Model,
    mm: &MinMaxHeadDispatch,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
    row: &[f64],
) -> f64 {
    let n = model.n_inputs();
    if mm.n != n {
        return f64::NAN;
    }
    let transforms = model.feature_transforms();
    let params = model.feature_transform_params();
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let mut x = vec![0.0f64; n];
    for (i, slot) in x.iter_mut().enumerate() {
        let raw = *row.get(i).unwrap_or(&0.0) as f32;
        let t = match (transforms, params) {
            (Some(tf), Some(p)) => tf[i].apply_with_params(raw, &p[i]),
            (Some(tf), None) => tf[i].apply_with_params(raw, &[]),
            (None, _) => raw,
        } as f64;
        let s = scale[i] as f64;
        let safe = if s == 0.0 { 1.0 } else { s };
        *slot = ((t - mean[i] as f64) / safe).clamp(-8.0, 8.0);
    }
    let mut best_min = f64::INFINITY;
    for g in 0..mm.k {
        let mut best_max = f64::NEG_INFINITY;
        for h in 0..mm.j {
            let base = (g * mm.j + h) * n;
            let mut acc = mm.b[g * mm.j + h] as f64;
            for (wv, xv) in mm.w[base..base + n].iter().zip(&x[..n]) {
                acc += *wv as f64 * xv;
            }
            if acc > best_max {
                best_max = acc;
            }
        }
        if best_max < best_min {
            best_min = best_max;
        }
    }
    apply_post_dispatch(best_min, tanh_pin_scale, output_spline)
}

/// Score one row through a loaded MLP `Predictor`.
///
/// Per DEDUP-M, this is the single canonical implementation of the
/// per-row bake dispatch shared across `bake_verdict` / `qsweep_eval` /
/// `preview_stats_demo` / `ensemble_score_rows` / `score_pair_with_bake` /
/// `predict_features_with_bake`.
///
/// Caller pre-allocates `f32_features` to the bake's `n_inputs` so a
/// single buffer is reused across millions of rows (parquet-baked
/// evaluation runs at ≥10k rows/sec).
///
/// Order of operations (bit-exact with the per-bin copies + the canonical
/// `zensim::metric::apply_mlp_scoring_with_codec` flow):
///
/// 1. Copy `min(n_inputs, row.len())` features into the scratch buffer,
///    zero-pad any trailing slots.
/// 2. `predictor.predict_transformed` if `has_transforms`, else
///    `predictor.predict`.
/// 3. Per-sample-α head, hybrid head, or first-output fallback —
///    whichever metadata is supplied. NaN-propagate on size mismatch.
/// 4. Tanh output pin if `tanh_pin_scale` is `Some`.
/// 5. Output PCHIP spline if `output_spline` is `Some`.
///
/// Returns `f64::NAN` if `predictor.predict*` errors or the head
/// metadata size doesn't match `out.len()`.
#[allow(clippy::too_many_arguments)]
pub fn score_row(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    let n_inputs = f32_features.len();
    let take = n_inputs.min(row.len());
    for i in 0..take {
        f32_features[i] = row[i] as f32;
    }
    for f in &mut f32_features[take..] {
        *f = 0.0;
    }
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    let y_pre = match result {
        Ok(out) => apply_head_dispatch(out, per_sample_alpha_head, hybrid_head),
        Err(_) => f64::NAN,
    };
    apply_post_dispatch(y_pre, tanh_pin_scale, output_spline)
}

/// Variant of [`score_row`] that owns its scratch (for callers that don't
/// share buffers across rows). Allocates one `Vec<f32; n_inputs>` per
/// call. Used by `score_pair_with_bake` / `predict_features_with_bake`
/// where the per-call alloc is in the noise vs network forward time.
#[allow(clippy::too_many_arguments)]
pub fn score_with_bake_alloc(
    predictor: &mut Predictor<'_>,
    has_transforms: bool,
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
    n_inputs: usize,
    features: &[f64],
) -> f64 {
    let mut buf = vec![0.0f32; n_inputs];
    score_row(
        predictor,
        has_transforms,
        per_sample_alpha_head,
        hybrid_head,
        tanh_pin_scale,
        output_spline,
        &mut buf[..],
        features,
    )
}

/// Apply the per-sample-α head, hybrid head, or first-output fallback —
/// whichever metadata is supplied. `out` is the bake's forward-pass
/// output vector (the hidden vector `h` when either head metadata is
/// present, else the raw score).
fn apply_head_dispatch(
    out: &[f32],
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
) -> f64 {
    if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) =
        per_sample_alpha_head
    {
        let n = out.len() as f64;
        if n <= 0.0 || out.len() != rank_w.len() || out.len() != w_alpha.len() {
            return f64::NAN;
        }
        let mut y_rank = *rank_b as f64;
        let mut alpha_logit = *b_alpha as f64;
        let mut sum = 0.0_f64;
        let mut max_v = f64::NEG_INFINITY;
        let mut sum_p = 0.0_f64;
        let p = *p_norm as f64;
        for (j, &h) in out.iter().enumerate() {
            let hf = h as f64;
            y_rank += hf * rank_w[j] as f64;
            alpha_logit += hf * w_alpha[j] as f64;
            sum += hf;
            if hf > max_v {
                max_v = hf;
            }
            sum_p += hf.abs().powf(p);
        }
        let mu = sum / n;
        let mut var = 0.0_f64;
        for &h in out.iter() {
            let d = h as f64 - mu;
            var += d * d;
        }
        let sigma = (var / n).sqrt().max(POOL_STD_FLOOR);
        let p_norm_stat = (sum_p / n).powf(1.0 / p);
        let y_pool = mu * reducer_w[0] as f64
            + sigma * reducer_w[1] as f64
            + max_v * reducer_w[2] as f64
            + p_norm_stat * reducer_w[3] as f64
            + *reducer_b as f64;
        let alpha = {
            let xc = alpha_logit.clamp(-20.0, 20.0);
            1.0 / (1.0 + (-xc).exp())
        };
        alpha * y_rank + (1.0 - alpha) * y_pool
    } else if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) = hybrid_head {
        let n = out.len() as f64;
        if n <= 0.0 || out.len() != rank_w.len() {
            return f64::NAN;
        }
        let mut y_rank = *rank_b as f64;
        let mut sum = 0.0_f64;
        let mut max_v = f64::NEG_INFINITY;
        let mut sum_p = 0.0_f64;
        let p = *p_norm as f64;
        for (j, &h) in out.iter().enumerate() {
            let hf = h as f64;
            y_rank += hf * rank_w[j] as f64;
            sum += hf;
            if hf > max_v {
                max_v = hf;
            }
            sum_p += hf.abs().powf(p);
        }
        let mu = sum / n;
        let mut var = 0.0_f64;
        for &h in out.iter() {
            let d = h as f64 - mu;
            var += d * d;
        }
        let sigma = (var / n).sqrt().max(POOL_STD_FLOOR);
        let p_norm_stat = (sum_p / n).powf(1.0 / p);
        let y_pool = mu * reducer_w[0] as f64
            + sigma * reducer_w[1] as f64
            + max_v * reducer_w[2] as f64
            + p_norm_stat * reducer_w[3] as f64
            + *reducer_b as f64;
        let alpha = {
            let xc = (*alpha_logit as f64).clamp(-20.0, 20.0);
            1.0 / (1.0 + (-xc).exp())
        };
        alpha * y_rank + (1.0 - alpha) * y_pool
    } else {
        out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
    }
}

/// Apply the tanh pin + output spline post-network dispatch. Bit-exact
/// with `zensim::metric::apply_tanh_output_pin` and
/// `zensim::metric::apply_output_calibration_spline`.
fn apply_post_dispatch(
    y_pre: f64,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
) -> f64 {
    let y_after_pin = if let Some(scale) = tanh_pin_scale {
        if !y_pre.is_nan() {
            let xc = (y_pre / scale).clamp(-30.0, 30.0);
            let s = 1.0 / (1.0 + (-xc).exp());
            100.0 * s
        } else {
            y_pre
        }
    } else {
        y_pre
    };
    if let Some(spline) = output_spline
        && !y_after_pin.is_nan()
    {
        return output_calibration_spline::apply(y_after_pin, spline);
    }
    y_after_pin
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small `apply_head_dispatch` smoke test against a hand-rolled
    // per-sample-α payload. The math is exercised end-to-end by the
    // existing `tests/per_sample_alpha_runtime.rs` and
    // `tests/hybrid_head_runtime.rs` integration tests against canonical
    // bakes — those stay in place as the load-bearing regression gates.

    #[test]
    fn fallback_returns_first_output() {
        let out = vec![0.42_f32, 99.0];
        let y = apply_head_dispatch(&out, None, None);
        assert_eq!(y, 0.42_f32 as f64);
    }

    #[test]
    fn fallback_returns_nan_on_empty() {
        let out: Vec<f32> = vec![];
        let y = apply_head_dispatch(&out, None, None);
        assert!(y.is_nan());
    }

    #[test]
    fn per_sample_size_mismatch_returns_nan() {
        let out = vec![0.5_f32, 0.5];
        let psa: PerSampleAlphaHeadDispatch = (
            vec![1.0, 1.0, 1.0],
            0.0,
            vec![1.0, 1.0, 1.0],
            0.0,
            [1.0; 4],
            0.0,
            2.0,
        );
        let y = apply_head_dispatch(&out, Some(&psa), None);
        assert!(y.is_nan());
    }

    #[test]
    fn hybrid_size_mismatch_returns_nan() {
        let out = vec![0.5_f32, 0.5, 0.5];
        let hyb: HybridHeadDispatch = (vec![1.0, 1.0], 0.0, 0.0, [1.0; 4], 0.0, 2.0);
        let y = apply_head_dispatch(&out, None, Some(&hyb));
        assert!(y.is_nan());
    }

    #[test]
    fn tanh_pin_disabled_passthrough() {
        let y = apply_post_dispatch(42.0, None, None);
        assert_eq!(y, 42.0);
    }

    #[test]
    fn tanh_pin_enabled_clamps_to_0_100() {
        // sigmoid(0) = 0.5, so y_pre=0 with any scale yields 50.0.
        let y = apply_post_dispatch(0.0, Some(1.0), None);
        assert!((y - 50.0).abs() < 1e-9);
    }

    #[test]
    fn tanh_pin_propagates_nan() {
        let y = apply_post_dispatch(f64::NAN, Some(1.0), None);
        assert!(y.is_nan());
    }
}
