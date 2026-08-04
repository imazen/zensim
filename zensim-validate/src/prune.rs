//! Dead-column pruning for ZNPR bakes — the analysis half of
//! `bake_dial_refit pack --prune`.
//!
//! A 944-input bake whose layer-0 weight rows are exactly zero on 277
//! of those inputs is *structurally* a 667-input model paying a
//! 944-input storage bill: 277 × `out_dim` weights plus 277 scaler
//! pairs that can never change a prediction. Pruning removes them and
//! declares [`FeatureTransform::Drop`] on those raw lines, so the
//! caller's feature vector width is unchanged and predictions are
//! unchanged.
//!
//! # The three classes of "dead" — only two are prunable
//!
//! This is the whole correctness story. `bake_contrib` measures which
//! inputs a bake is *effectively* ignoring **on a corpus**; that is a
//! superset of what is safe to remove, and the difference is a silent
//! behaviour change waiting to happen.
//!
//! | class | test | prunable | why |
//! |---|---|---|---|
//! | **1. weight-dead** | `W0[k, :]` is exactly `0.0` | **yes**, bit-identical | contributes `x̃ₖ · 0 = 0` for every input; removing it changes no bit |
//! | **2. transform-forced-constant** | the bake's own `feature_transform` on `k` maps *every* input to one constant `c` | **yes**, exact | the contribution is the constant `x̃(c) · W0[k, :]`, folded into `b0` |
//! | **3. inert on this corpus** | `bake_contrib` says mean\|Δ\| ≈ 0, but the weight is live and no transform forces it | **NO** | the training corpus merely never exercised it; another corpus will |
//!
//! Class 3 is exactly the case that looks identical to class 1 in a
//! corpus report and is *not* identical mathematically. This module
//! never prunes it: every decision here is derived from the bake's own
//! weights, transforms and scaler — **no corpus statistic is an input
//! to [`plan`]**, which is what makes class 3 structurally unreachable
//! rather than merely discouraged. Class-3 inputs fall out as
//! "retained" and are counted in [`PrunePlan::retained`].
//!
//! # Ordering inside `pack`
//!
//! Zerobias is what *creates* most weight-dead columns (a column of
//! sub-τ weights becomes a column of exact zeros), so the pipeline is
//! **zerobias → prune → dtype/quantize → spline refit**. The spline
//! still lands last, on the final packed net, preserving the
//! QUANTIZE-then-CALIBRATE invariant.

use std::collections::BTreeMap;

use zenpredict::{FeatureTransform, Model};

/// Why an input was dropped. See the module docs' class table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DropReason {
    /// Class 1 — `W0[k, :]` is exactly zero. Removal is bit-identical.
    WeightDead,
    /// Class 2 — the bake's own transform forces input `k` to
    /// `value` for every input, so its layer-0 contribution is the
    /// constant `standardized · W0[k, :]`, folded into `b0`.
    ForcedConstant {
        /// Post-transform constant.
        value: f32,
        /// `(value − scaler_mean[k]) / scaler_scale[k]` — what layer 0
        /// actually sees.
        standardized: f32,
    },
}

impl DropReason {
    pub fn label(&self) -> &'static str {
        match self {
            Self::WeightDead => "weight-dead",
            Self::ForcedConstant { .. } => "transform-forced-constant",
        }
    }
}

/// A pruning plan: which layer-0 inputs to keep, which to drop and why,
/// and the layer-0 bias delta the dropped constants fold into.
#[derive(Debug, Clone)]
pub struct PrunePlan {
    /// Caller-facing raw feature width. **Unchanged by pruning** —
    /// this is the contract that makes pruning invisible to callers.
    pub raw_width: usize,
    /// Layer-0 `in_dim` before pruning.
    pub n_inputs_before: usize,
    /// Layer-0 `in_dim` after pruning (`keep.len()`).
    pub n_inputs_after: usize,
    /// Layer-0 input indices retained, ascending.
    pub keep: Vec<usize>,
    /// `(layer-0 input index, raw index, reason)`, ascending.
    pub drop: Vec<(usize, usize, DropReason)>,
    /// Per-layer-0-output bias delta contributed by folded constants.
    /// Length `out_dim`; all zeros when no class-2 column was dropped.
    pub bias_delta: Vec<f32>,
    /// Raw inputs that are NOT dropped. Includes any input that is
    /// inert on some corpus but has a live weight and no forcing
    /// transform (class 3) — never prunable.
    pub retained: usize,
    /// Post-prune per-raw-input transforms (length `raw_width`).
    pub transforms: Vec<FeatureTransform>,
    /// Post-prune per-raw-input transform params (length `raw_width`).
    pub params: Vec<Vec<f32>>,
}

impl PrunePlan {
    /// True when nothing would change.
    pub fn is_noop(&self) -> bool {
        self.drop.is_empty()
    }

    /// Count of dropped inputs by class.
    pub fn class_counts(&self) -> (usize, usize) {
        let mut weight_dead = 0;
        let mut forced = 0;
        for (_, _, r) in &self.drop {
            match r {
                DropReason::WeightDead => weight_dead += 1,
                DropReason::ForcedConstant { .. } => forced += 1,
            }
        }
        (weight_dead, forced)
    }

    /// True when every dropped column is class 1 — in which case the
    /// pruned bake's predictions are **bit-identical**, not merely
    /// numerically equal.
    pub fn is_bit_identical(&self) -> bool {
        self.class_counts().1 == 0
    }

    /// Bytes of layer-0 weight storage removed at `dtype_bytes` per
    /// weight, plus the two `f32` scaler entries per dropped input.
    /// The *file* delta differs — the payload is LZ4-compressed and the
    /// transforms metadata changes length — so this is the decompressed
    /// footprint, not a file-size prediction.
    pub fn raw_bytes_saved(&self, out_dim: usize, dtype_bytes: usize) -> usize {
        self.drop.len() * (out_dim * dtype_bytes + 2 * 4)
    }

    /// Human-readable per-class table for the pack log.
    pub fn report(&self, out_dim: usize, dtype_bytes: usize) -> String {
        use std::fmt::Write as _;
        let (weight_dead, forced) = self.class_counts();
        let mut s = String::new();
        let _ = writeln!(
            s,
            "prune: layer-0 inputs {} -> {} (caller width unchanged at {})",
            self.n_inputs_before, self.n_inputs_after, self.raw_width
        );
        let _ = writeln!(
            s,
            "  class 1 weight-dead              {weight_dead:>5}  dropped (bit-identical)"
        );
        let _ = writeln!(
            s,
            "  class 2 transform-forced-const   {forced:>5}  dropped (folded into layer-0 bias)"
        );
        let _ = writeln!(
            s,
            "  class 3 inert-here / live weight {:>5}  RETAINED (never pruned — corpus-inert \
             is not mathematically dead)",
            self.retained
        );
        let _ = writeln!(
            s,
            "  layer-0 + scaler bytes freed     {:>5}  (decompressed; file delta differs — LZ4)",
            self.raw_bytes_saved(out_dim, dtype_bytes)
        );
        s
    }
}

/// Errors that make a bake out of scope for pruning. All are loud —
/// pruning silently doing nothing, or doing the wrong thing, is worse
/// than a refusal.
#[derive(Debug)]
pub enum PruneError {
    /// The bake declares a `Sinusoidal` expander. One raw input then
    /// maps to a *block* of layer-0 columns; deciding whether the block
    /// as a whole is dead is a different analysis. Out of scope.
    SinusoidalPresent { raw_index: usize },
    /// `feature_transforms` / `feature_transform_params` disagree with
    /// the layer-0 width — a malformed bake the loader should have
    /// rejected.
    TransformShapeMismatch { expected: usize, got: usize },
    /// Layer 0 is `i8`. Removing a *nonzero* row can change the
    /// per-output max-abs quantization scale and therefore every other
    /// weight's quantization, so class-2 folding is unsafe there.
    /// (Class-1 rows are all-zero and cannot hold the max, so they are
    /// still safe — this error only fires when a class-2 drop was
    /// requested on an i8 layer 0.)
    ForcedConstantOnI8Layer0,
}

impl std::fmt::Display for PruneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SinusoidalPresent { raw_index } => write!(
                f,
                "prune: bake declares a Sinusoidal expander at raw input {raw_index}; \
                 one raw input maps to a block of layer-0 columns and block-level \
                 deadness is a different analysis — pruning is out of scope for this bake"
            ),
            Self::TransformShapeMismatch { expected, got } => write!(
                f,
                "prune: feature transforms sum to {got} layer-0 inputs but the bake \
                 declares {expected} — malformed bake"
            ),
            Self::ForcedConstantOnI8Layer0 => write!(
                f,
                "prune: layer 0 is i8 and a transform-forced-constant column was \
                 selected; removing a nonzero row can change the per-output max-abs \
                 quantization scale, so this is not output-preserving. Re-run with \
                 --no-prune-constants (class-1 weight-dead pruning stays safe on i8)."
            ),
        }
    }
}

impl std::error::Error for PruneError {}

/// Probe values used to *confirm* a transform is constant. The
/// structural precondition (winsor family with crossed/equal bounds)
/// is checked first; these then verify it empirically against the real
/// runtime function, so a future change to any variant's math cannot
/// silently invalidate the analysis.
///
/// Deliberately excludes `NaN`: `clamp_inclusive` propagates NaN, so a
/// NaN feature is not mapped to the constant. A NaN feature already
/// means the caller handed the model garbage (the unpruned bake returns
/// NaN, i.e. no usable score); see [`plan`]'s docs.
const CONST_PROBES: [f32; 13] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    1e-30,
    -1e-30,
    1e6,
    -1e6,
    3.402_823_5e38,
    -3.402_823_5e38,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.123_456_79,
];

/// If `t(x, params)` is the same value for every input, return it.
///
/// Two independent gates, both required:
///
/// 1. **Structural** — only the winsor family can pin an output, and
///    only when its clamp bounds are finite and `lo >= hi` (the runtime
///    `clamp_inclusive` returns `lo` when `lo > hi`).
/// 2. **Empirical** — every [`CONST_PROBES`] value must produce
///    bit-identical output from the real `apply_with_params`.
///
/// **Gate 2 is the one that decides.** Gate 1 is a cheap pre-filter and
/// is necessary-but-not-sufficient: MEASURED 2026-08-04, two shapes that
/// satisfy it are not constant, because `WinsorP99` is a hand-rolled
/// `if x < lo {lo} else if x > hi {hi} else {x}` rather than the
/// `clamp_inclusive` the stacked variants use —
///
/// * `[0, 0]` passes `-0.0` straight through (`-0.0 < 0.0` and
///   `-0.0 > 0.0` are both false), so the output is `±0.0` depending on
///   the input sign;
/// * `[lo > hi]` yields `lo` for small `x` and `hi` for large `x` — two
///   values, where `clamp_inclusive` would have yielded only `lo`.
///
/// So the family is *not* uniform, and reasoning about params alone
/// would have mis-classified both. Gate 2 alone, conversely, could
/// false-positive on a transform that happens to saturate across the
/// probe set. Together: a cheap proof plus a check on the proof.
pub fn forced_constant(t: FeatureTransform, params: &[f32]) -> Option<f32> {
    // Gate 1: structural.
    let (lo, hi) = match t {
        FeatureTransform::WinsorP99
        | FeatureTransform::WinsorThenLog
        | FeatureTransform::WinsorThenLog1p
        | FeatureTransform::WinsorThenSignedCbrt
        | FeatureTransform::SignedCbrtThenWinsor => (*params.first()?, *params.get(1)?),
        // params = [eps, q_lo, q_hi]; the winsor is the OUTER op.
        FeatureTransform::ClipThenLog1pThenWinsor => (*params.get(1)?, *params.get(2)?),
        _ => return None,
    };
    if !lo.is_finite() || !hi.is_finite() || lo < hi {
        return None;
    }

    // Gate 2: empirical, against the real runtime function.
    let c = t.apply_with_params(CONST_PROBES[0], params);
    if !c.is_finite() {
        return None;
    }
    for &x in &CONST_PROBES[1..] {
        if t.apply_with_params(x, params).to_bits() != c.to_bits() {
            return None;
        }
    }
    Some(c)
}

/// Layer-0 weights as owned `f32`, row-major `in_dim × out_dim`.
/// Row `k` (input `k`) spans `w[k*out_dim .. (k+1)*out_dim]`.
pub struct Layer0View<'a> {
    pub in_dim: usize,
    pub out_dim: usize,
    pub weights: &'a [f32],
    pub biases: &'a [f32],
    /// True when the layer will be stored as `i8` — gates class 2.
    pub is_i8: bool,
}

/// Build a pruning plan from the bake's own weights, transforms and
/// scaler. **No corpus statistic is an input**, which is what keeps
/// class-3 columns structurally unreachable.
///
/// `prune_constants = false` restricts the plan to class 1, in which
/// case the resulting bake is bit-identical for *every* input including
/// NaN. With class 2 enabled a NaN feature on a forced-constant column
/// no longer propagates to the output (the unpruned bake returns NaN);
/// that only changes behaviour on input that was already garbage, but
/// it is the one respect in which class 2 is weaker than class 1.
pub fn plan(
    model: &Model,
    l0: &Layer0View<'_>,
    prune_constants: bool,
) -> Result<PrunePlan, PruneError> {
    let raw_width = model.caller_input_width();
    let transforms: Vec<FeatureTransform> = match model.feature_transforms() {
        Some(t) => t.to_vec(),
        None => vec![FeatureTransform::Identity; raw_width],
    };
    // Params are only meaningful alongside transforms — the runtime ignores
    // a stray params blob (`predict_transformed` returns early when
    // `feature_transforms()` is None), so mirror that rather than zipping a
    // possibly-mismatched length against a synthesized identity list.
    let params: Vec<Vec<f32>> = match (model.feature_transforms(), model.feature_transform_params())
    {
        (Some(_), Some(p)) => p.to_vec(),
        _ => vec![Vec::new(); raw_width],
    };

    // Map raw input -> layer-0 column. Only arity-1 transforms map 1:1;
    // arity-0 (an already-dropped input) maps to nothing. Sinusoidal is
    // refused above.
    let mut raw_of_l0: BTreeMap<usize, usize> = BTreeMap::new();
    let mut cursor = 0usize;
    for (raw, (&t, p)) in transforms.iter().zip(params.iter()).enumerate() {
        if matches!(t, FeatureTransform::Sinusoidal) {
            return Err(PruneError::SinusoidalPresent { raw_index: raw });
        }
        match t.output_arity(p) {
            0 => {}
            1 => {
                raw_of_l0.insert(cursor, raw);
                cursor += 1;
            }
            n => {
                // No other variable-arity variant exists today; if one
                // lands, refuse rather than guess.
                return Err(PruneError::TransformShapeMismatch {
                    expected: 1,
                    got: n,
                });
            }
        }
    }
    if cursor != l0.in_dim {
        return Err(PruneError::TransformShapeMismatch {
            expected: l0.in_dim,
            got: cursor,
        });
    }

    let mean = model.scaler_mean();
    let scale = model.scaler_scale();
    let mut keep = Vec::with_capacity(l0.in_dim);
    let mut drop = Vec::new();
    let mut bias_delta_f64 = vec![0f64; l0.out_dim];

    for k in 0..l0.in_dim {
        let raw = *raw_of_l0.get(&k).expect("cursor walk covered every k");
        let row = &l0.weights[k * l0.out_dim..(k + 1) * l0.out_dim];

        // Class 1: exactly-zero row. `-0.0 == 0.0` is true in IEEE, and
        // a -0.0 weight contributes `x·-0.0 = ∓0.0` which is additively
        // neutral, so both zeros qualify.
        if row.iter().all(|&w| w == 0.0) {
            drop.push((k, raw, DropReason::WeightDead));
            continue;
        }

        // Class 2: the bake's OWN transform pins this input.
        if prune_constants && let Some(c) = forced_constant(transforms[raw], &params[raw]) {
            if l0.is_i8 {
                return Err(PruneError::ForcedConstantOnI8Layer0);
            }
            let s = scale[k];
            let safe = if s == 0.0 { 1.0 } else { s };
            let xt = (c - mean[k]) / safe;
            if xt.is_finite() {
                for (o, delta) in bias_delta_f64.iter_mut().enumerate() {
                    *delta += xt as f64 * row[o] as f64;
                }
                drop.push((
                    k,
                    raw,
                    DropReason::ForcedConstant {
                        value: c,
                        standardized: xt,
                    },
                ));
                continue;
            }
        }

        keep.push(k);
    }

    // Post-prune transforms: dropped raw lines become `drop`.
    let mut out_transforms = transforms.clone();
    let mut out_params = params.clone();
    for (_, raw, _) in &drop {
        out_transforms[*raw] = FeatureTransform::Drop;
        out_params[*raw] = Vec::new();
    }

    let n_inputs_after = keep.len();
    Ok(PrunePlan {
        raw_width,
        n_inputs_before: l0.in_dim,
        n_inputs_after,
        keep,
        drop,
        bias_delta: bias_delta_f64.iter().map(|&d| d as f32).collect(),
        retained: n_inputs_after,
        transforms: out_transforms,
        params: out_params,
    })
}

/// Apply a plan to layer-0 weights: keep the planned rows in ascending
/// order. Returns the new row-major `keep.len() × out_dim` matrix.
pub fn prune_layer0_weights(plan: &PrunePlan, weights: &[f32], out_dim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(plan.keep.len() * out_dim);
    for &k in &plan.keep {
        out.extend_from_slice(&weights[k * out_dim..(k + 1) * out_dim]);
    }
    out
}

/// Apply a plan to layer-0 biases: add the folded constant
/// contributions. Returns a new bias vector of the same length.
pub fn prune_layer0_biases(plan: &PrunePlan, biases: &[f32]) -> Vec<f32> {
    biases
        .iter()
        .zip(plan.bias_delta.iter())
        .map(|(&b, &d)| if d == 0.0 { b } else { b + d })
        .collect()
}

/// Apply a plan to a per-layer-0-input array (scaler mean or scale).
pub fn prune_input_array(plan: &PrunePlan, values: &[f32]) -> Vec<f32> {
    plan.keep.iter().map(|&k| values[k]).collect()
}

/// Serialize the post-prune transforms + params to the two
/// line-aligned `zentrain.feature_transform*` metadata payloads.
pub fn transform_metadata(plan: &PrunePlan) -> (String, String) {
    let transforms = plan
        .transforms
        .iter()
        .map(|t| t.as_token())
        .collect::<Vec<_>>()
        .join("\n");
    let params = plan
        .params
        .iter()
        .map(|p| p.iter().map(fmt_param).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join("\n");
    (transforms, params)
}

/// Round-trip-exact `f32` formatting: Rust's `{}` for `f32` emits the
/// shortest string that parses back to the same bits, so re-baking a
/// bake's own params never perturbs them.
fn fmt_param(v: &f32) -> String {
    format!("{v}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winsor_with_equal_nonzero_bounds_is_constant() {
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[4.0, 4.0]),
            Some(4.0)
        );
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[-1.5, -1.5]),
            Some(-1.5)
        );
    }

    #[test]
    fn the_empirical_gate_rejects_winsor_cases_the_structural_gate_admits() {
        // MEASURED 2026-08-04 — these two both pass the structural gate
        // (`lo >= hi`, finite) and are NOT constant. They are why the
        // empirical probe is not decoration.
        //
        // `WinsorP99` is a hand-rolled `if x < lo {lo} else if x > hi {hi}
        // else {x}`, NOT the `clamp_inclusive` the stacked variants use, so:
        //
        //  * `[0, 0]` — `-0.0 < 0.0` is false and `-0.0 > 0.0` is false, so
        //    `-0.0` falls through the `else` and comes out as `-0.0`, whose
        //    bits differ from `+0.0`. Not bit-constant. This is the exact
        //    shape of the 24 `winsor_p99:[0,0]` columns in the sota944
        //    bakes — they are pruned as class 1 (weight-dead), which is the
        //    stronger guarantee anyway, so nothing is lost by being strict.
        //  * `[7, 2]` (lo > hi) — output is `lo` for small `x` and `hi` for
        //    large `x`. Two values, not one. (The *stacked* variants route
        //    through `clamp_inclusive`, which does return `lo` here — the
        //    family is not uniform, which is precisely why constancy is
        //    decided by probing the real function rather than by reasoning
        //    about the params.)
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[0.0, 0.0]),
            None
        );
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[7.0, 2.0]),
            None
        );
    }

    #[test]
    fn winsor_with_a_real_range_is_not_constant() {
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[0.0, 1.0]),
            None
        );
        assert_eq!(
            forced_constant(FeatureTransform::WinsorP99, &[-100.0, 100.0]),
            None
        );
    }

    #[test]
    fn stacked_winsor_variants_are_recognized() {
        // ln(clamp(x, e, e)) == 1
        let c = forced_constant(
            FeatureTransform::WinsorThenLog,
            &[core::f32::consts::E, core::f32::consts::E],
        )
        .expect("constant");
        assert!((c - 1.0).abs() < 1e-6, "ln(e) = 1, got {c}");
        // ClipThenLog1pThenWinsor params are [eps, q_lo, q_hi] — the
        // winsor bounds are 1 and 2, NOT 0 and 1.
        assert_eq!(
            forced_constant(FeatureTransform::ClipThenLog1pThenWinsor, &[0.1, 2.5, 2.5]),
            Some(2.5)
        );
        assert_eq!(
            forced_constant(FeatureTransform::ClipThenLog1pThenWinsor, &[0.1, 0.0, 5.0]),
            None
        );
    }

    #[test]
    fn non_winsor_variants_are_never_constant() {
        // Not even the saturating ones — soft_sign/soft_clip approach a
        // limit but never pin, and the probe set spans both signs.
        for t in [
            FeatureTransform::Identity,
            FeatureTransform::Log1p,
            FeatureTransform::SignedLog1p,
            FeatureTransform::SignedSqrt,
            FeatureTransform::SoftSign,
            FeatureTransform::SoftClip,
            FeatureTransform::SignedPow,
            FeatureTransform::QuantileBins,
            FeatureTransform::YeoJohnson,
        ] {
            assert_eq!(forced_constant(t, &[0.0, 0.0]), None, "{t:?}");
            assert_eq!(forced_constant(t, &[]), None, "{t:?}");
        }
    }
}
