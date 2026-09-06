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
//! # ★ 2026-09-06 — the FLOAT MATH now has ONE owner; this module is a SHAPE adapter
//!
//! The paragraph this replaces (DEDUP-M2, 2026-05-26, "HONEST-STOP —
//! delegation to canonical zensim helper INFEASIBLE") is **correct about the
//! ENTRY POINTS and wrong as a conclusion about the ARITHMETIC**, and the
//! distinction cost a real bug. All four of its reasons stand and are why
//! [`score_row`] still exists: `apply_mlp_scoring_with_codec` takes a
//! `ZensimResult` built by an image pipeline, its `mlp_bytes` is a
//! `fn() -> &'static [u8]` a CLI path cannot satisfy without leaking, it runs
//! the full ensemble/mix/clamp/per-codec pipeline this tooling deliberately
//! stops short of, and it constructs a fresh `Predictor` per call where this
//! path hot-loops one across millions of rows. **None of that required
//! copying the float math**, and copying it is what broke:
//!
//! F19 (`zensim::det_math`) routed every transcendental on the score path
//! through a selectable [`zensim::det_math::PowForm`] so a score stops being a
//! function of which libm the binary linked against. `zensim::metric` follows
//! the form. This module's copy did not, and no test bound them — the
//! bit-exactness claim above was PROSE. MEASURED 2026-09-06 on six
//! shipped/board bakes: `bake_verdict --full-json` was byte-identical under
//! `ZENSIM_POW_FORM=libm` and `=pure`, i.e. the evaluation tooling was
//! completely insensitive to the form the product obeys. `det_math`'s own
//! exposure table called this fork *"a BLOCKER on flipping
//! `SHIPPED_REVISION`"*.
//!
//! So the arithmetic moved to [`zensim::score_math`] — `#[doc(hidden)] pub`,
//! borrowed parameter views so neither side adopts the other's storage — and
//! **both** `zensim::metric` and this module call it. What is left here is
//! exactly the part the DEDUP-M2 analysis identified as genuinely
//! validate-shaped: metadata parsing into public tuples, `Predictor` +
//! scratch reuse, the [`CallerGather`] policy, and the NaN short-circuits.
//! Gate: `zensim-validate/tests/score_owner_parity.rs` (a re-fork fails it) +
//! `tests/no_score_path_libm.rs` (a re-introduced `powf`/`exp` on these files
//! fails it).

use zenpredict::{Model, Predictor};
use zensim::det_math::active_pow_form;
use zensim::score_math;

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
    for chunk in entry.value.as_chunks::<4>().0.iter() {
        floats.push(f32::from_le_bytes(*chunk));
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
    for chunk in entry.value.as_chunks::<4>().0.iter() {
        floats.push(f32::from_le_bytes(*chunk));
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
    // INTERNAL layer-0 width on purpose (NOT `caller_input_width()`) — the
    // loop below indexes transforms by layer-0 position, so this path
    // requires the two to be equal and refuses the bake when they are not.
    // Campaign appendix E.9.
    let n = model.n_inputs();
    if mm.n != n {
        return f64::NAN;
    }
    // The loop below applies `transforms[i]` at layer-0 index `i`, which is
    // only the same index when the pipeline is 1:1. A variable-arity bake
    // (Sinusoidal expander, or a pruned bake carrying `drop`) breaks that
    // alignment — and the scalar `apply_with_params` panics on those
    // variants. Refuse rather than mis-index.
    if model.caller_input_width() != n {
        return f64::NAN;
    }
    let transforms = model.feature_transforms();
    let params = model.feature_transform_params();
    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    // The caller row is IDENTITY-laid-out; a bake that declares a DENSE read
    // set takes `row[ids[i]]` at layer-0 index `i`. Derived per call because
    // min-max bakes are rare and this path already allocates `x`; the hot
    // paths hoist a `CallerGather` instead.
    let gather = CallerGather::for_model(model);
    let mut x = vec![0.0f64; n];
    for (i, slot) in x.iter_mut().enumerate() {
        let src = match &gather {
            CallerGather::Positional => i,
            // `ids.len() == caller_input_width() == n` is guaranteed by
            // `declared_layout` plus the equality check above; `get` keeps a
            // future shape bug from panicking inside a scorer.
            CallerGather::ByFeatureId(ids) => ids.get(i).map_or(usize::MAX, |&v| usize::from(v)),
        };
        let raw = *row.get(src).unwrap_or(&0.0) as f32;
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

/// How a caller's feature row becomes the bake's input vector.
///
/// **The dense contract's consumer half** (`docs/PLAN_CRUFT_PURGE_2026-09-06.md`
/// increment B-2). A bake that declares `zentrain.feature_ids` reads its ids at
/// PACKED positions, so a positional copy would hand it whatever happened to sit
/// at those indices — plausible numbers, wrong features, no error. Build one per
/// bake (never per row) and thread it through [`score_row`].
///
/// [`CallerGather::Positional`] is byte-for-byte what this code did before the
/// type existed, and it is what EVERY bake shipped before 2026-09-06 resolves
/// to — so threading this through moves no number.
///
/// The `gather` parameter on [`score_row`] is deliberately **not** an `Option`
/// with a `None` default: a new scorer must decide, at the site where it loads
/// a bake, which of the two it means. That is the difference between "dense
/// bakes are refused" and "dense bakes are quietly mis-scored".
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CallerGather {
    /// Copy `min(n_inputs, row.len())` positions, zero the tail.
    Positional,
    /// Take `row[ids[j]]` into slot `j`; an id past the row's width becomes
    /// `0.0`, which is the same structural fill `Layout::gather` writes.
    ByFeatureId(Vec<u16>),
}

impl CallerGather {
    /// The gather this bake declares. Resolves through
    /// `zensim::declared_feature_ids` — the ONE owner of the declaration — so
    /// the evaluation tools and the `zensim` runtime cannot disagree about
    /// which ids a bake reads.
    pub fn for_model(model: &Model) -> CallerGather {
        match zensim::declared_feature_ids(model) {
            Some(ids) => CallerGather::ByFeatureId(ids),
            None => CallerGather::Positional,
        }
    }

    /// Is this bake dense? Used by callers that want to REPORT the fact.
    pub fn is_dense(&self) -> bool {
        matches!(self, CallerGather::ByFeatureId(_))
    }

    /// Can a grid or corpus of `n_features` columns feed this bake?
    ///
    /// **Positional keeps the EXACT rule every grid gate used before this type
    /// existed — `n_features == n_inputs`** — so no identity bake's grid is
    /// admitted or skipped differently. A DENSE bake reads ids out of a wider
    /// caller-laid-out row, so its requirement is that the row REACHES its
    /// highest declared id; a `==` test against its packed width would skip
    /// every real grid it can actually score, which is a silent coverage loss
    /// rather than a wrong number, and just as bad in a published verdict.
    pub fn accepts_row_width(&self, n_features: usize, n_inputs: usize) -> bool {
        match self {
            CallerGather::Positional => n_features == n_inputs,
            CallerGather::ByFeatureId(ids) => ids
                .iter()
                .max()
                .is_some_and(|&hi| n_features > usize::from(hi)),
        }
    }

    /// Can a PREFIX-tolerant source of `n_features` columns feed this bake?
    ///
    /// Two admission rules exist because two DIFFERENT historical rules exist,
    /// and collapsing them would move numbers in both directions.
    /// [`accepts_row_width`](Self::accepts_row_width) is the GRID rule: an
    /// exact `==`, because a dial/corruption grid that is not the bake's own
    /// width is a different instrument and admitting it silently would publish
    /// a panel about the wrong thing. This is the rule for a source whose
    /// contract has always been *"take the leading `n_inputs` columns"* — the
    /// kadis multi-metric per-pair table, which is 720 columns wide and feeds
    /// 372-input bakes by prefix. Using the grid rule there dropped the whole
    /// per-pair block from every identity bake's verdict (measured while
    /// wiring the dense gather, hence this split).
    ///
    /// The DENSE arm is the same for both: a declared bake needs the row to
    /// REACH its highest declared id, because it indexes by id and its own
    /// packed width says nothing about where those ids live.
    pub fn accepts_prefix_row_width(&self, n_features: usize, n_inputs: usize) -> bool {
        match self {
            CallerGather::Positional => n_features >= n_inputs,
            CallerGather::ByFeatureId(_) => self.accepts_row_width(n_features, n_inputs),
        }
    }

    /// Fill `dst` (already sized to the bake's caller width) from `row`.
    pub fn fill(&self, dst: &mut [f32], row: &[f64]) {
        match self {
            CallerGather::Positional => {
                let take = dst.len().min(row.len());
                for i in 0..take {
                    dst[i] = row[i] as f32;
                }
                for f in &mut dst[take..] {
                    *f = 0.0;
                }
            }
            CallerGather::ByFeatureId(ids) => {
                for (slot, &id) in dst.iter_mut().zip(ids.iter()) {
                    *slot = row.get(usize::from(id)).copied().unwrap_or(0.0) as f32;
                }
                // A dst longer than the declaration is a shape bug upstream;
                // zero rather than leave stale scratch, matching Positional.
                let tail = ids.len().min(dst.len());
                for f in &mut dst[tail..] {
                    *f = 0.0;
                }
            }
        }
    }
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
/// 1. Fill the scratch buffer from `row` — see [`CallerGather`]. Either a
///    positional copy of `min(n_inputs, row.len())` with the tail zeroed
///    (identity layouts: every bake that shipped before 2026-09-06), or a
///    GATHER of the ids the bake declares (dense layouts).
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
    gather: &CallerGather,
    f32_features: &mut [f32],
    row: &[f64],
) -> f64 {
    gather.fill(f32_features, row);
    let result = if has_transforms {
        predictor.predict_transformed(f32_features)
    } else {
        predictor.predict(f32_features)
    };
    match result {
        Ok(out) => score_from_network_output(
            out,
            per_sample_alpha_head,
            hybrid_head,
            tanh_pin_scale,
            output_spline,
        ),
        Err(_) => apply_post_dispatch(f64::NAN, tanh_pin_scale, output_spline),
    }
}

/// Apply the post-network dispatch — head (per-sample-α / hybrid /
/// first-output fallback), tanh pin, output spline — to a network
/// output vector. This is the tail of [`score_row`], factored out so
/// diagnostic tooling that recomputes the network forward itself
/// (e.g. `bake_contrib`'s exact mean-ablation, which edits layer-0
/// pre-activations) applies the IDENTICAL head/pin/spline math with
/// no possibility of a fork. Bit-exact: [`score_row`] delegates here.
pub fn score_from_network_output(
    out: &[f32],
    per_sample_alpha_head: Option<&PerSampleAlphaHeadDispatch>,
    hybrid_head: Option<&HybridHeadDispatch>,
    tanh_pin_scale: Option<f64>,
    output_spline: Option<&OutputCalibrationSpline>,
) -> f64 {
    let y_pre = apply_head_dispatch(out, per_sample_alpha_head, hybrid_head);
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
    // Derived per call because this entry OWNS its allocation anyway — the
    // hot loops use `score_row` with a hoisted `CallerGather`.
    let gather = CallerGather::for_model(predictor.model());
    score_row(
        predictor,
        has_transforms,
        per_sample_alpha_head,
        hybrid_head,
        tanh_pin_scale,
        output_spline,
        &gather,
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
    // F19 / owner consolidation (2026-09-06): the arithmetic is
    // `zensim::score_math`'s, reached through the SAME entry the product
    // runtime (`zensim::metric::apply_{per_sample_alpha,hybrid_head}_runtime`)
    // reaches. This function is now a SHAPE adapter — validate parses the
    // metadata into public tuples, `metric.rs` into private structs — and
    // carries no float math of its own.
    //
    // Read the form ONCE, per `det_math`'s own discipline.
    let form = active_pow_form();
    if let Some((w_alpha, b_alpha, rank_w, rank_b, reducer_w, reducer_b, p_norm)) =
        per_sample_alpha_head
    {
        score_math::per_sample_alpha_head(
            out,
            &score_math::PerSampleAlphaParams {
                w_alpha,
                b_alpha: *b_alpha,
                rank_w,
                rank_b: *rank_b,
                reducer_w: *reducer_w,
                reducer_b: *reducer_b,
                p_norm: *p_norm,
            },
            form,
        )
    } else if let Some((rank_w, rank_b, alpha_logit, reducer_w, reducer_b, p_norm)) = hybrid_head {
        score_math::hybrid_head(
            out,
            &score_math::HybridHeadParams {
                rank_w,
                rank_b: *rank_b,
                alpha_logit: *alpha_logit,
                reducer_w: *reducer_w,
                reducer_b: *reducer_b,
                p_norm: *p_norm,
            },
            form,
        )
    } else {
        out.first().copied().map(|v| v as f64).unwrap_or(f64::NAN)
    }
}

/// Apply a `--bake-post` policy to a raw bake score.
///
/// **THE one owner** of the four `--bake-post` modes. `qsweep_eval`,
/// `predict_features_with_bake` and `score_pair_with_bake` each carried a
/// BYTE-IDENTICAL copy of this function (verified line-for-line before the
/// merge); all three now call here.
///
/// | mode | result |
/// |---|---|
/// | `raw` | `raw`, untouched |
/// | `extrapolate` | `raw`, untouched — a separate name so a caller's no-clamp policy is explicit at the call site (EXP-CROSS-CODEC-V10) |
/// | `clamp` | `raw.clamp(0, 100)` |
/// | `mapped` / `mapped:A,B` | `(100 − A·max(raw,0)^B).clamp(0, 100)`, defaults `A=18, B=0.7` |
/// | anything else | `raw.clamp(0, 100)` |
///
/// The `mapped` arm is [`zensim::score_math::distance_to_score_mapped`] — the
/// product runtime's own mapping, routed through
/// [`zensim::det_math::PowForm`] — with this path's clamp on top. It is a
/// bit-exact rewrite of the `100.0 - a * d.powf(b)` the three copies carried:
/// at `d == 0` the owner short-circuits to exactly `100.0` and `0f64.powf(b)`
/// is `0.0`, so `100.0 - a*0.0` is the same `100.0`.
///
/// NaN in, NaN out, before any mode is consulted.
pub fn apply_post_mode(raw: f64, mode: &str) -> f64 {
    if raw.is_nan() {
        return f64::NAN;
    }
    match mode {
        "raw" => raw,
        "extrapolate" => raw,
        "clamp" => raw.clamp(0.0, 100.0),
        m if m.starts_with("mapped") => {
            // mapped or mapped:A,B
            let (a, b) = if let Some(rest) = m.strip_prefix("mapped:") {
                let mut it = rest.splitn(2, ',');
                let a: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(18.0);
                let b: f64 = it.next().and_then(|s| s.parse().ok()).unwrap_or(0.7);
                (a, b)
            } else {
                (18.0, 0.7)
            };
            let d = raw.max(0.0);
            score_math::distance_to_score_mapped(d, a, b, active_pow_form()).clamp(0.0, 100.0)
        }
        _ => raw.clamp(0.0, 100.0),
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
            // Owner: `zensim::score_math::tanh_output_pin`, the same call
            // `zensim::metric::apply_tanh_output_pin` makes. The NaN
            // short-circuit stays HERE because it is this path's shape choice,
            // not arithmetic: the owner propagates NaN by construction, so the
            // guard changes nothing numerically and is kept only so a NaN row
            // is visibly skipped rather than run through a squash.
            score_math::tanh_output_pin(y_pre, scale, active_pow_form())
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

    /// **B-2.1** — `Positional` is byte-for-byte the pre-2026-09-06 fill:
    /// copy `min(len, row.len())`, zero the tail. Written as an explicit
    /// reimplementation of the OLD code rather than a re-description, so the
    /// test would fail if the enum's arm ever drifted from it.
    #[test]
    fn positional_gather_reproduces_the_old_fill_exactly() {
        let legacy = |dst: &mut [f32], row: &[f64]| {
            let take = dst.len().min(row.len());
            for i in 0..take {
                dst[i] = row[i] as f32;
            }
            for f in &mut dst[take..] {
                *f = 0.0;
            }
        };
        let row: Vec<f64> = (0..12).map(|i| (i as f64) * 1.5 - 3.0).collect();
        for width in [1usize, 5, 12, 20] {
            let mut a = vec![7.5f32; width];
            let mut b = vec![7.5f32; width];
            CallerGather::Positional.fill(&mut a, &row);
            legacy(&mut b, &row);
            assert_eq!(
                a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "width {width}"
            );
        }
    }

    /// The gather takes the DECLARED ids, not the first N positions — the
    /// whole point. Includes an id past the row's width, which must become
    /// the structural `0.0` rather than panic.
    #[test]
    fn feature_id_gather_takes_the_declared_ids() {
        let row: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let g = CallerGather::ByFeatureId(vec![0, 3, 7, 9, 40]);
        let mut dst = vec![-1.0f32; 5];
        g.fill(&mut dst, &row);
        assert_eq!(dst, vec![0.0, 3.0, 7.0, 9.0, 0.0]);
        assert!(g.is_dense());
        assert!(!CallerGather::Positional.is_dense());
        // The negative control: a positional fill of the same width would
        // read f0..f4, which is a DIFFERENT vector.
        let mut pos = vec![-1.0f32; 5];
        CallerGather::Positional.fill(&mut pos, &row);
        assert_ne!(dst, pos);
    }

    /// **B-2.2 in table space** — a DENSE bake scored through the evaluation
    /// dispatch equals its WIDE twin scored on the same wide row.
    ///
    /// The two bakes are built to be the same function of the same three ids
    /// (one wide with zero rows at every other position, one dense declaring
    /// exactly those ids). Without the gather the dense one reads `f0,f1,f2`
    /// and the assertion fails — which the negative control below proves by
    /// forcing `Positional` on the dense bake.
    #[test]
    fn a_dense_bake_scores_like_its_wide_twin_through_score_row() {
        const IDS: [usize; 3] = [2, 5, 9];
        const W: usize = 12;
        let wide = bake_json(
            W,
            &(0..W)
                .map(|i| if IDS.contains(&i) { 1.0 } else { 0.0 })
                .collect::<Vec<f32>>(),
            None,
        );
        let dense = bake_json(IDS.len(), &vec![1.0f32; IDS.len()], Some(&IDS));
        let wm = Model::from_bytes(&wide).expect("wide bake");
        let dm = Model::from_bytes(&dense).expect("dense bake");
        assert_eq!(dm.caller_input_width(), IDS.len());
        let gw = CallerGather::for_model(&wm);
        let gd = CallerGather::for_model(&dm);
        assert!(!gw.is_dense(), "an undeclared bake must stay positional");
        assert!(gd.is_dense(), "the declaration must be READ");

        let row: Vec<f64> = (0..W).map(|i| (i as f64) * 0.75 + 0.125).collect();
        let mut pw = Predictor::new(&wm);
        let mut pd = Predictor::new(&dm);
        let mut bw = vec![0.0f32; wm.caller_input_width()];
        let mut bd = vec![0.0f32; dm.caller_input_width()];
        let sw = score_row(&mut pw, false, None, None, None, None, &gw, &mut bw, &row);
        let sd = score_row(&mut pd, false, None, None, None, None, &gd, &mut bd, &row);
        assert_eq!(sw.to_bits(), sd.to_bits(), "wide {sw} vs dense {sd}");

        // NEGATIVE CONTROL: the same dense bake, sliced positionally, reads
        // f0..f2 and must NOT agree.
        let mut bn = vec![0.0f32; dm.caller_input_width()];
        let sn = score_row(
            &mut pd,
            false,
            None,
            None,
            None,
            None,
            &CallerGather::Positional,
            &mut bn,
            &row,
        );
        assert_ne!(
            sw.to_bits(),
            sn.to_bits(),
            "a positional slice of a dense bake must differ, else the test proves nothing"
        );
    }

    /// `accepts_row_width` must keep the EXACT pre-existing grid rule for
    /// identity bakes (`==`), and admit a dense bake on any row that REACHES
    /// its highest declared id. Getting the dense side wrong is a silent
    /// COVERAGE loss in a published verdict, not a wrong number — measured:
    /// a dense shipped B skipped the whole corruption grid under the old `==`.
    #[test]
    fn grid_admission_keeps_the_old_rule_for_identity_and_reaches_for_dense() {
        let pos = CallerGather::Positional;
        assert!(pos.accepts_row_width(372, 372));
        assert!(!pos.accepts_row_width(944, 372), "the old rule is EQUALITY");
        assert!(!pos.accepts_row_width(300, 372));

        let dense = CallerGather::ByFeatureId(vec![3, 10, 369]);
        assert!(dense.accepts_row_width(372, 3), "reaches f369");
        assert!(dense.accepts_row_width(944, 3));
        // 370 columns are indices 0..=369, so f369 IS present — the boundary
        // is `n_features > hi`, and 369 columns (0..=368) is the short case.
        assert!(
            dense.accepts_row_width(370, 3),
            "370 columns reach f369 exactly"
        );
        assert!(!dense.accepts_row_width(369, 3), "369 columns stop at f368");
        assert!(
            !dense.accepts_row_width(3, 3),
            "its own packed width does NOT reach"
        );
    }

    /// The PREFIX rule is `>=` for identity bakes — the kadis per-pair source
    /// is 720 columns and has always fed 372-input bakes by prefix — while
    /// the GRID rule above stays `==`. The dense arm is the same reach test in
    /// both. Without this split, wiring the dense gather at the per-pair site
    /// silently dropped the block from every identity bake's verdict.
    #[test]
    fn prefix_admission_is_ge_for_identity_and_reach_for_dense() {
        let pos = CallerGather::Positional;
        assert!(pos.accepts_prefix_row_width(372, 372));
        assert!(
            pos.accepts_prefix_row_width(720, 372),
            "the kadis-720 source feeds a 372 bake by prefix"
        );
        assert!(!pos.accepts_prefix_row_width(300, 372), "too short");
        // The two rules genuinely disagree — if they ever agree everywhere,
        // one of them is dead and this split is unjustified.
        assert_ne!(
            pos.accepts_prefix_row_width(720, 372),
            pos.accepts_row_width(720, 372)
        );

        let dense = CallerGather::ByFeatureId(vec![3, 10, 369]);
        assert!(dense.accepts_prefix_row_width(720, 3));
        assert!(dense.accepts_prefix_row_width(370, 3));
        assert!(!dense.accepts_prefix_row_width(369, 3));
        assert!(
            !dense.accepts_prefix_row_width(3, 3),
            "a dense bake's own packed width never reaches its ids"
        );
    }

    /// A 1-layer sum-of-inputs bake, optionally declaring `feature_ids`.
    fn bake_json(n: usize, w: &[f32], ids: Option<&[usize]>) -> Vec<u8> {
        let arr = |v: &[f32]| {
            let body: Vec<String> = v.iter().map(|x| x.to_string()).collect();
            format!("[{}]", body.join(","))
        };
        let md = match ids {
            Some(ids) => {
                let list: Vec<String> = ids.iter().map(|i| i.to_string()).collect();
                format!(
                    r#"{{"key": "{}", "type": "utf8", "text": "{}"}}"#,
                    zensim::ZENTRAIN_FEATURE_IDS_KEY,
                    list.join("\\n")
                )
            }
            None => String::new(),
        };
        let json = format!(
            r#"{{"schema_hash": 1, "scaler_mean": {mean}, "scaler_scale": {scale},
                 "metadata": [{md}],
                 "layers": [{{"in_dim": {n}, "out_dim": 1, "activation": "identity",
                              "dtype": "f32", "weights": {w}, "biases": [0.0]}}]}}"#,
            mean = arr(&vec![0.0f32; n]),
            scale = arr(&vec![1.0f32; n]),
            w = arr(w),
        );
        zenpredict_bake::bake_from_json_str(&json).expect("synthetic bake")
    }
}
