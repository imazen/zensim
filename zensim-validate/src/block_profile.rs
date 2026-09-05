//! Static per-bake feature-block usage profiling — the analysis behind the
//! `bake_block_profile` binary (which is a thin CLI wrapper over
//! [`profile`]).
//!
//! # CALLER space, not internal space
//!
//! The feature families of the append-only numbering discipline —
//!
//! * `f0..f155`    v1-basic
//! * `f156..f371`  the block ZEROED by the folded 924/944 regimes (slots
//!   preserved per the append-only discipline — never removed; carries real
//!   values in the v1-372 / 720 regimes)
//! * `f372..f719`  v2-348
//! * `f720..f943`  append-204
//!
//! — are defined over the **caller feature numbering**: the vector a caller
//! hands to [`zenpredict::Predictor::predict_transformed`], whose width is
//! [`zenpredict::Model::caller_input_width`]. A bake's layer-0 columns are
//! **internal** indices, and the two diverge whenever the bake declares a
//! variable-arity [`zenpredict::FeatureTransform`]:
//!
//! * [`FeatureTransform::Drop`] (dead-column pruning, 2026-08-04): caller
//!   line `k` produces **zero** layer-0 columns, so a pruned 944-wide bake
//!   has e.g. 667 internal columns.
//! * [`FeatureTransform::Sinusoidal`]: caller line `k` expands to `2·N`
//!   layer-0 columns.
//!
//! Slicing the internal columns at caller-family boundaries is therefore
//! wrong on any such bake. On the first real pruned candidate
//! (`W10L9_s4003_packed`, 944 → 667) the internal-space slicing reported a
//! false `uses_f156_371: true`, a 295-wide "f372_719" and no `f720_943`
//! family at all, while the unpruned parent's ground truth is
//! f156-371 = 216/216 exact-zero (uses = false). This was the **fourth**
//! instance of the caller-width bug class (see the E.9 regression note in
//! `tests/prune_classes.rs` for instance #3, the coherence-harness regime
//! router) — hence the mapping lives here in library code with tests, not
//! inline in a binary.
//!
//! The mapping walks the dense `feature_transforms` vec exactly the way
//! [`zenpredict::Model::caller_input_width`] does: transform `k` consumes
//! [`FeatureTransform::output_arity`] internal columns, in caller order. A
//! caller line's column norm is the L2 norm over **all** weights of the
//! internal columns it produces — for scalar (arity-1) transforms and for
//! transform-free bakes this is bit-identical to the per-column norm the
//! pre-fix tool computed, a dropped line's norm is exactly `0.0` (an empty
//! sum), and an expanded line folds its `2·N` columns into one caller entry.
//!
//! # What the counts mean
//!
//! For each family we count the caller lines whose L2 norm over layer-0
//! output weights is (a) exactly zero, (b) near-zero (≤ [`NEAR_ZERO_REL`] ×
//! the max caller-line norm), (c) structurally used. A dropped caller line
//! counts as exactly zero — that is precisely the contract of pruning
//! (class-1 columns were exact zeros in the parent; class-2 constants can
//! never change a prediction), which is what makes a pruned bake's profile
//! equal its unpruned parent's.
//!
//! This is the STRUCTURAL usage complement to the corpus-based contribution
//! measure (`bake_contrib`): a column can be structurally nonzero yet
//! effectively unused when its input is always zero. MEASURED 2026-08-04:
//! the 944-regime trainers emit EXACT zeros for the whole f156..f371 block
//! (216/216 on `C_co3a_s1301`, `C_ensk2_s1303`), and zero part of the
//! append block (61/224 exact-zero) — so exact-zero counts are a real
//! discriminator, not init noise. Exact zeros also appear when a
//! trainer/packer explicitly zeroes (zerobias, lasso, BVLS).
//!
//! Model access goes through [`zenpredict::Model`] (the canonical loader —
//! handles v3.1 compression, dtype decode, canonical ordering); this module
//! adds NO wire-format code of its own.

use zenpredict::{FeatureTransform, Model, WeightStorage, f16_bits_to_f32};

/// (label, start, end) — end exclusive, in CALLER feature indices; clipped
/// to the bake's caller input width.
pub const FAMILIES: &[(&str, usize, usize)] = &[
    ("f0_155", 0, 156),
    ("f156_371", 156, 372),
    ("f372_719", 372, 720),
    ("f720_943", 720, 944),
];

/// The v1-372 COMPUTE families — the sub-division of `f0..f371` that maps to
/// what the extractor actually has to run, rather than to the append-only
/// numbering blocks of [`FAMILIES`].
///
/// v1's 372 layout is block-major (`metric::combine_scores` passes 1-4, and
/// `feature_v2`'s fold emit site): `[basic 156][peaks 72][masked 72][iw 72]`,
/// each block scale-major then channel-major. The four families are NOT four
/// independent compute jobs — see [`V1ComputeNeed`] for the shared-pass
/// structure that governs what skipping one actually saves.
pub const V1_FAMILIES: &[(&str, usize, usize)] = &[
    ("v1_basic", 0, 156),
    ("v1_peaks", 156, 228),
    ("v1_masked", 228, 300),
    ("v1_iw", 300, 372),
];

/// Near-zero threshold, relative to the max caller-line norm.
pub const NEAR_ZERO_REL: f64 = 1e-6;

/// Per-family usage counts. `cols` is the family's width in CALLER lines;
/// `exact_zero + near_zero + used == cols`.
#[derive(Debug, Clone, PartialEq)]
pub struct FamilyStats {
    pub label: &'static str,
    pub cols: usize,
    pub exact_zero: usize,
    pub near_zero: usize,
    pub used: usize,
    pub max_col_norm: f64,
}

impl FamilyStats {
    /// The structural counts alone — everything except `max_col_norm`,
    /// which is dtype-sensitive (an f16-packed twin's norms differ from
    /// its f32 parent's even though the structure is identical).
    pub fn counts(&self) -> (&'static str, usize, usize, usize, usize) {
        (
            self.label,
            self.cols,
            self.exact_zero,
            self.near_zero,
            self.used,
        )
    }
}

/// The full fingerprint. `n_inputs` / `layer0_in_dim` stay INTERNAL widths
/// (truthful about the stored net); `caller_input_width`, `n_dropped` and
/// every family stat are CALLER-space.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockProfile {
    pub znpr_version: u16,
    /// Internal (post-transform) width — `Model::n_inputs()`.
    pub n_inputs: usize,
    /// Caller feature-vector width — `Model::caller_input_width()`.
    pub caller_input_width: usize,
    /// Number of `Drop` transform lines (0 on an unpruned bake).
    pub n_dropped: usize,
    pub layer0_in_dim: usize,
    pub layer0_out_dim: usize,
    pub dtype: &'static str,
    pub n_layers: usize,
    /// Caller lines beyond f943 (a bake genuinely taking > 944 features).
    pub beyond_f943_cols: usize,
    /// Structurally-used caller lines in the zeroed-block family.
    pub uses_f156_371: bool,
    pub families: Vec<FamilyStats>,
    /// The [`V1_FAMILIES`] view of the same layer-0 norms — the COMPUTE
    /// families, as opposed to `families`' append-only numbering blocks.
    /// Empty when the bake is narrower than 157 caller lines.
    pub v1_families: Vec<FamilyStats>,
}

impl BlockProfile {
    /// The JSON document `promote_fulleval.py --set-block-profile` stores
    /// as `block_profile` (consumed by gauntlet.py + freeze_check — field
    /// names are load-bearing; additions are fine, renames are not).
    pub fn to_json(&self) -> String {
        let render = |fams: &[FamilyStats]| -> Vec<String> {
            fams.iter()
                .map(|f| {
                    format!(
                        "\"{}\":{{\"cols\":{},\"exact_zero\":{},\"near_zero\":{},\"used\":{},\"max_col_norm\":{:.6e}}}",
                        f.label, f.cols, f.exact_zero, f.near_zero, f.used, f.max_col_norm
                    )
                })
                .collect()
        };
        let fam_json = render(&self.families);
        let v1_json = render(&self.v1_families);
        format!(
            "{{\"znpr_version\":{},\"n_inputs\":{},\"caller_input_width\":{},\"n_dropped\":{},\"layer0_in_dim\":{},\"layer0_out_dim\":{},\"dtype\":\"{}\",\"n_layers\":{},\"beyond_f943_cols\":{},\"near_zero_rel\":{:e},\"uses_f156_371\":{},\"families\":{{{}}},\"v1_families\":{{{}}}}}",
            self.znpr_version,
            self.n_inputs,
            self.caller_input_width,
            self.n_dropped,
            self.layer0_in_dim,
            self.layer0_out_dim,
            self.dtype,
            self.n_layers,
            self.beyond_f943_cols,
            NEAR_ZERO_REL,
            self.uses_f156_371,
            fam_json.join(","),
            v1_json.join(",")
        )
    }

    /// Human-readable table. `source` is typically the bake path.
    pub fn render_text(&self, source: &str) -> String {
        use core::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(
            s,
            "# {source} — ZNPR v{}, {} layer(s), layer0 {}→{} ({})",
            self.znpr_version, self.n_layers, self.layer0_in_dim, self.layer0_out_dim, self.dtype
        );
        if self.n_dropped > 0 {
            let _ = writeln!(
                s,
                "# pruned: {} caller lines dropped (layer0 {} < caller width {}) — families below are CALLER-space",
                self.n_dropped, self.layer0_in_dim, self.caller_input_width
            );
        }
        s.push_str("family    cols  exact0  near0  used  max‖col‖\n");
        for f in &self.families {
            let _ = writeln!(
                s,
                "{:<9} {:>4}  {:>6}  {:>5}  {:>4}  {:.3e}",
                f.label, f.cols, f.exact_zero, f.near_zero, f.used, f.max_col_norm
            );
        }
        if self.beyond_f943_cols > 0 {
            let _ = writeln!(
                s,
                "(+{} caller features beyond f943)",
                self.beyond_f943_cols
            );
        }
        let _ = writeln!(s, "uses_f156_371 (structural): {}", self.uses_f156_371);
        if !self.v1_families.is_empty() {
            s.push_str("\n# v1-372 COMPUTE families (basic 156 | peaks 72 | masked 72 | IW 72)\n");
            s.push_str("family    cols  exact0  near0  used  max\u{2016}col\u{2016}\n");
            for f in &self.v1_families {
                let _ = writeln!(
                    s,
                    "{:<9} {:>4}  {:>6}  {:>5}  {:>4}  {:.3e}",
                    f.label, f.cols, f.exact_zero, f.near_zero, f.used, f.max_col_norm
                );
            }
        }
        s
    }
}

/// C2 wrong-regime guard (opus-review, campaign appendix W): does this bake
/// structurally use the f156-371 block that the FOLDED roots (the ext944 /
/// 924 extractions) feed as STRUCTURAL ZEROS?
///
/// A bake that uses that block scored at a folded root reads plausible-looking
/// garbage with no error — the class has two published instances: the
/// `ebothg_m504` board row (wrong-root read) and appendix U.R0 (shipped B
/// reads CID22 0.3862 at the 944 root against its true 0.8764 at 372).
/// Scorers targeting a folded root call this and REFUSE on `Some(..)` unless
/// the caller explicitly opts into the cross-regime read.
///
/// Returns `Ok(None)` when the read is safe (block unused — true for every
/// genuine folded-regime bake, which trained on those zeros, and for f0-155 /
/// ≤156-input bakes), `Ok(Some(reason))` when it is not, and `Err` only for a
/// malformed bake (same contract as [`profile`]).
pub fn folded_root_conflict(model: &Model) -> Result<Option<String>, String> {
    let p = profile(model)?;
    if !p.uses_f156_371 {
        return Ok(None);
    }
    let used = p
        .families
        .iter()
        .find(|f| f.label == "f156_371")
        .map(|f| f.used)
        .unwrap_or(0);
    Ok(Some(format!(
        "bake structurally uses {used} caller line(s) in f156-371, a block this folded \
         root feeds as STRUCTURAL ZEROS — the scored numbers would be plausible-looking \
         garbage (the ebothg_m504 / appendix-U.R0 wrong-regime class)"
    )))
}

/// Fold per-INTERNAL-column values back to CALLER lines via the bake's
/// declared transform arities — the same walk [`profile`] and
/// `Model::caller_input_width` perform. `fold` reduces one caller line's
/// internal-column slice (empty for a `Drop`ped line) to its caller value.
/// Returns `(per-caller values, n_dropped)`; errors on a malformed arity
/// table exactly as [`profile`] does.
fn fold_cols_to_caller(
    model: &Model,
    col_vals: &[f64],
    fold: impl Fn(&[f64]) -> f64,
) -> Result<(Vec<f64>, usize), String> {
    let in_dim = col_vals.len();
    match model.feature_transforms() {
        None => Ok((col_vals.to_vec(), 0)),
        Some(ts) => {
            let params = model.feature_transform_params();
            if let Some(p) = params
                && p.len() != ts.len()
            {
                return Err(format!(
                    "feature_transform_params has {} entries for {} transforms — malformed bake",
                    p.len(),
                    ts.len()
                ));
            }
            let mut out = Vec::with_capacity(ts.len());
            let mut dropped = 0usize;
            let mut cur = 0usize;
            for (k, t) in ts.iter().enumerate() {
                let pk: &[f32] = params.map(|p| p[k].as_slice()).unwrap_or(&[]);
                if *t == FeatureTransform::Drop {
                    dropped += 1;
                }
                let arity = t.output_arity(pk);
                let end = cur + arity;
                if end > in_dim {
                    return Err(format!(
                        "feature_transform arities overrun layer 0 (need ≥ {end} columns, have {in_dim}) — malformed bake"
                    ));
                }
                out.push(fold(&col_vals[cur..end]));
                cur = end;
            }
            if cur != in_dim {
                return Err(format!(
                    "feature_transform arities cover {cur} layer-0 columns but layer 0 has {in_dim} — malformed bake"
                ));
            }
            Ok((out, dropped))
        }
    }
}

/// C8 pack-default probe (opus-review, campaign appendix W): how many LIVE
/// layer-0 caller lines would a per-layer zerobias at `tau` kill OUTRIGHT
/// (every weight of the line `|w| < tau`)?
///
/// Returns `(killed, live)`. A high killed/live ratio means the flat zerobias
/// is miscalibrated for this bake: the tau was tuned for 100-500 KB dense
/// MLPs where sub-tau weights are noise, but on a sparse fit every surviving
/// coefficient is signal — measured damage: `--zerobias-bulk 0.005` cost
/// ADD156 −0.0069 CID22 (T.R11, 13 of 26 live lines killed) and wiped the
/// appendix-J group-lasso cell GL4_s2501 from 57 live rows to 3 (J.R3).
/// `bake_dial_refit pack` uses this to default zerobias to 0 on the sparse
/// class instead of repeating the incident.
pub fn zerobias_line_kill_fraction(model: &Model, tau: f64) -> Result<(usize, usize), String> {
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    let colmax = |get: &dyn Fn(usize, usize) -> f64| -> Vec<f64> {
        (0..in_dim)
            .map(|i| (0..out_dim).map(|o| get(i, o).abs()).fold(0.0f64, f64::max))
            .collect()
    };
    let col_max: Vec<f64> = match &layer.weights {
        WeightStorage::F32(w) => {
            let w = *w;
            colmax(&move |i, o| w[i * out_dim + o] as f64)
        }
        WeightStorage::F16(w) => {
            let w = *w;
            colmax(&move |i, o| f16_bits_to_f32(w[i * out_dim + o]) as f64)
        }
        WeightStorage::I8 { weights, scales } => {
            let (w, s) = (*weights, *scales);
            colmax(&move |i, o| w[i * out_dim + o] as f64 * s[o] as f64)
        }
    };
    let (line_max, _) = fold_cols_to_caller(model, &col_max, |sl| {
        sl.iter().cloned().fold(0.0f64, f64::max)
    })?;
    let live = line_max.iter().filter(|&&m| m > 0.0).count();
    let killed = line_max.iter().filter(|&&m| m > 0.0 && m < tau).count();
    Ok((killed, live))
}

/// Compute the caller-space block profile of a loaded model.
///
/// Errors (rather than mis-reporting) on a malformed bake whose declared
/// transform arities do not tile layer 0 — every valid bake satisfies
/// `Σ output_arity == layer0.in_dim` by construction.
/// Per-CALLER-LINE L2 weight norms over layer 0, the number of `Drop`ped
/// lines, and the layer-0 dtype — the shared kernel behind [`profile`] and
/// [`used_caller_lines`].
///
/// Extracted so the feature-set-id derivation
/// (`zensim_validate::feature_set`) reads the SAME norms the block profile
/// tabulates, rather than a second copy of the fold. Errors identically on a
/// malformed arity table.
pub fn caller_line_norms(model: &Model) -> Result<(Vec<f64>, usize, &'static str), String> {
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    // Weights are row-major input-major: W[i * out_dim + o].
    let wget: Box<dyn Fn(usize, usize) -> f64 + '_> = match &layer.weights {
        WeightStorage::F32(w) => {
            let w = *w;
            Box::new(move |i, o| w[i * out_dim + o] as f64)
        }
        WeightStorage::F16(w) => {
            let w = *w;
            Box::new(move |i, o| f16_bits_to_f32(w[i * out_dim + o]) as f64)
        }
        WeightStorage::I8 { weights, scales } => {
            let (w, s) = (*weights, *scales);
            Box::new(move |i, o| w[i * out_dim + o] as f64 * s[o] as f64)
        }
    };
    let dtype = match &layer.weights {
        WeightStorage::F32(_) => "f32",
        WeightStorage::F16(_) => "f16",
        WeightStorage::I8 { .. } => "i8",
    };
    // Per-INTERNAL-column sum of squared weights over outputs.
    let col_sq: Vec<f64> = (0..in_dim)
        .map(|i| (0..out_dim).map(|o| wget(i, o).powi(2)).sum::<f64>())
        .collect();
    // Fold internal columns back to CALLER lines. The dense transforms vec
    // (one entry per caller line) defines the mapping: transform k consumes
    // `output_arity` internal columns, in caller order — the same walk
    // `Model::caller_input_width` / the predict pipeline perform.
    let (caller_sq, n_dropped) = fold_cols_to_caller(model, &col_sq, |sl| {
        // A dropped line's slice is empty, and the empty f64 sum is
        // `-0.0` (the additive identity), which sqrt/max would
        // propagate into a "-0.000000e0" in the JSON — pin every
        // zero to +0.0 so a pruned family's stats byte-match its
        // unpruned parent's.
        let sq: f64 = sl.iter().sum();
        if sq == 0.0 { 0.0 } else { sq }
    })?;
    let caller_width = model.caller_input_width();
    if caller_sq.len() != caller_width {
        return Err(format!(
            "internal inconsistency: folded {} caller lines but caller_input_width() is {caller_width}",
            caller_sq.len()
        ));
    }
    Ok((
        caller_sq.iter().map(|s| s.sqrt()).collect(),
        n_dropped,
        dtype,
    ))
}

/// The CALLER lines a bake structurally READS — its consumer slot set.
///
/// "Structurally used" is [`profile`]'s own definition: a caller line whose
/// L2 norm over layer-0 output weights is neither exactly zero nor within
/// [`NEAR_ZERO_REL`] of the max line norm. A `Drop`ped (pruned) line is
/// exactly zero and therefore reads nothing — precisely pruning's contract —
/// so a packed bake's read-set equals its unpruned parent's.
pub fn used_caller_lines(model: &Model) -> Result<Vec<usize>, String> {
    let (norms, _, _) = caller_line_norms(model)?;
    let max_norm = norms.iter().cloned().fold(0.0f64, f64::max);
    let near = NEAR_ZERO_REL * max_norm;
    Ok((0..norms.len()).filter(|&i| norms[i] > near).collect())
}

pub fn profile(model: &Model) -> Result<BlockProfile, String> {
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    // ONE derivation of the caller-line norms, shared with `used_caller_lines`.
    let (norms, n_dropped, dtype) = caller_line_norms(model)?;
    let caller_width = model.caller_input_width();
    let max_norm = norms.iter().cloned().fold(0.0f64, f64::max);
    let near = NEAR_ZERO_REL * max_norm;

    let tabulate = |spec: &'static [(&'static str, usize, usize)]| -> Vec<FamilyStats> {
        let mut out = Vec::new();
        for &(label, lo, hi) in spec {
            if lo >= caller_width {
                continue;
            }
            let hi = hi.min(caller_width);
            let cols = hi - lo;
            let sl = &norms[lo..hi];
            let exact_zero = sl.iter().filter(|&&n| n == 0.0).count();
            let near_zero = sl.iter().filter(|&&n| n > 0.0 && n <= near).count();
            let used = cols - exact_zero - near_zero;
            let fam_max = sl.iter().cloned().fold(0.0f64, f64::max);
            out.push(FamilyStats {
                label,
                cols,
                exact_zero,
                near_zero,
                used,
                max_col_norm: fam_max,
            });
        }
        out
    };
    let families = tabulate(FAMILIES);
    let v1_families = tabulate(V1_FAMILIES);
    let uses_f156_371 = families
        .iter()
        .find(|f| f.label == "f156_371")
        .map(|f| f.used > 0)
        .unwrap_or(false);

    Ok(BlockProfile {
        znpr_version: model.version(),
        n_inputs: model.n_inputs(),
        caller_input_width: caller_width,
        n_dropped,
        layer0_in_dim: in_dim,
        layer0_out_dim: out_dim,
        dtype,
        n_layers: model.n_layers(),
        beyond_f943_cols: caller_width.saturating_sub(944),
        uses_f156_371,
        families,
        v1_families,
    })
}
