# PLAN — serving the nonlinear corruption head (`ZCTH` v1), 2026-09-06

**Pre-registration.** Written and pushed BEFORE any code, per the lane brief.
Lane `claude-corrserve`, jj sibling workspace `~/work/zen/zensim--corrserve`.
Record when done: `benchmarks/corruption_head_serving_2026-09-06.md`.

**What this lane does NOT do:** it does not re-open any modelling question, does
not retrain, does not re-extract, does not change any shipped bake, and does not
add one byte of public API. It takes the answer the theory lane
([`../benchmarks/corruption_head_theories_2026-09-06.md`](../benchmarks/corruption_head_theories_2026-09-06.md),
`478bc28e`) — *the corruption head should be a gradient-boosted tree* — and
makes that answer **servable**: a wire format, an evaluator, and the two gates
that bind them.

**ERA.** Everything here is **rev1** (post-option-C `56bbcda2`,
`ssim_form::SHIPPED_REVISION = Rev1`), the era the theory lane measured. Nothing
is re-extracted. A rev2 re-extraction changes 12 basic slots this head reads and
is out of scope.

---

## 0. Why the incumbent cannot be extended, in one line each

- The shipped head `corruption_head_d228.bin` is a **logistic** regression +
  isotonic, baked as ZNPR v3: one identity layer from `coef_`/`intercept_`
  (`scripts/v_next/train_corruption_head.py::emit_znpr`). Its measured numbers
  are 86.01 % detection / 11.37 % honest FP / **50.00 % near-lossless FP**.
- `hgb` on the identical 228 features, split and calibration reads
  **98.90 / 1.23 / 2.38** and needs no dial guard. There is no linear
  re-parameterisation of that; the model form IS the result.
- `emit_znpr` refuses `--bake-out` for every nonlinear `--model` (loudly, by
  `can_bake`). So the gap is a **wire format**, which is what this plan closes.

## 1. Format decision — a new `ZCTH` v1, NOT a ZNPR metadata blob

**Decision: its own small versioned format.** Stated with the reading it rests
on, because the brief asks for the alternative to be considered honestly.

Read first (all four are the reason):

1. **`zenpredict` is frozen at the `zenanalyze-api` contract level** (workspace
   CLAUDE.md, USER DIRECTIVE 2026-07-19). A tree section in ZNPR v3 is a
   wire-format change to the frozen crate. Out of bounds by rule, before any
   engineering argument.
2. **ZNPR's own dispatch would be wrong by default.** `zenpredict::Model` is
   layers + activations; every consumer that holds one calls
   `Predictor::predict{,_transformed}`. `bake_verdict`'s `score_grid_one` does
   exactly that. A tree hidden in `metadata[]` behind a plausible identity layer
   would be **silently mis-scored by every consumer that does not know to look**
   — the same shape as the `--regime 944` bug in CLAUDE.md's Known Bugs, which
   is the defect class this repo has paid for most.
3. **The shapes genuinely differ.** A ZNPR layer table cannot express
   `(feature_idx, threshold, left, right, is_leaf, value, missing_go_to_left)`
   without abuse; `n_inputs`/`caller_input_width` mean something specific there;
   `output_calibration_spline` is PCHIP on `(f32, f32)` knots while sklearn's
   isotonic is **piecewise-LINEAR on f64 knots with endpoint clipping**
   (`IsotonicRegression._transform`: `np.clip(T, X_min_, X_max_)` then
   `interp1d(kind="linear")`). Reusing the spline section would require a
   re-fit that is not the model, i.e. exactly the "measured, then quantised into
   something else" trap `pack`'s QUANTIZE-then-CALIBRATE rule exists for.
4. **A separate magic is the cheap half of the safety.** A distinct
   `b"ZCTH"` + `format_version` + `schema_hash` means a corruption head can
   never be loaded as a dial and a dial can never be loaded as a head — the
   failure is a refusal at byte 0, not a plausible wrong number.

The parts ZNPR got right are **copied, not re-invented**: magic, a `u16`
format version, a `u64` schema hash over the canonical shape, a section table
of `(offset u32, len u32)`, and a declared-feature-id list so the dense
contract (`zensim::declared_feature_ids`) works identically for a head.

### 1.1 `ZCTH` v1 layout (little-endian throughout)

```
Header, 120 bytes
   0..4    magic                b"ZCTH"
   4..6    format_version  u16  = 1
   6..8    flags           u16  (bit0 has_isotonic, bit1 has_scaler)
   8..16   schema_hash     u64  FNV-1a over the canonical shape descriptor
  16..20   caller_input_width u32   features the caller must supply
  20..24   n_declared      u32   features the trees read
  24..28   n_trees         u32
  28..32   n_nodes         u32
  32..40   baseline        f64   HGB `_baseline_prediction`
  40..48   deadband_t      f64   fires when P > t
  48..52   clip            f32   standardisation clip, +-clip
  52..56   reserved        u32   = 0
  56..64   sec_declared_ids  Section   u16  * n_declared
  64..72   sec_scaler_mean   Section   f64  * n_declared
  72..80   sec_scaler_scale  Section   f64  * n_declared
  80..88   sec_tree_offsets  Section   u32  * (n_trees + 1)
  88..96   sec_nodes         Section   Node * n_nodes
  96..104  sec_iso_x         Section   f64  * n_knots
 104..112  sec_iso_y         Section   f64  * n_knots
 112..120  sec_meta          Section   utf8 JSON provenance

Node, 32 bytes
   0..8    threshold     f64
   8..12   left          u32   node index within this tree's range
  12..16   right         u32
  16..20   feature_pos   u32   index into sec_declared_ids
  20..24   node_flags    u32   bit0 is_leaf, bit1 missing_go_to_left
  24..32   value         f64   leaf value (0.0 for an internal node)
```

`sec_meta` carries, as JSON: `feature_set_id`, `formula_revision`,
`model` (`"hgb"`), `sklearn_version`, the training recipe argv, input shas, the
split file, and `deadband_t`. Every field is provenance, none is load-bearing
for evaluation — an evaluator that ignores `sec_meta` still scores correctly.

### 1.2 Evaluation contract (this is the parity target)

```
z_j   = clamp((x[declared_ids[j]] - mean[j]) / scale[j], -clip, +clip)
raw   = baseline + SUM_over_trees walk(tree, z)
p_raw = 1 / (1 + exp(-raw))
p     = interp_linear(clamp(p_raw, iso_x[0], iso_x[n-1]); iso_x, iso_y)
score = 100 * (1 - p)
```

`walk` is `if z[feature_pos] <= threshold { left } else { right }`, with NaN
routed by `missing_go_to_left` — sklearn's `_predictor` comparison, verbatim.
`interp_linear` reproduces scipy's `interp1d` arithmetic exactly, including its
`searchsorted(side="left").clip(1, n-1)` bracket choice and its
`slope * (t - x_lo) + y_lo` evaluation order, so an on-knot query returns the
same bits.

## 2. Owner extension (the exporter)

`scripts/v_next/train_corruption_head.py` — **the one owner** — gains
`emit_zcth(...)`, the mirror of `emit_znpr`, plus the `--bake-out` dispatch
by `--model`. `can_bake` widens from `name == "logistic"` to *the forms that
have an exporter*; `logistic` keeps `emit_znpr` and its **byte-identical**
default path. `--bake-extra-width` works for both.

**Coordination:** a determinism-fix lane is live on this same file (the
BLAS-thread dependence recorded in the theory doc §9). This lane's change is
**purely additive** — a new function plus a dispatch on `--model` — and does
not touch the logistic fit, the split, or the scaler. Rebase often; on a
conflict keep BOTH sides.

The theory-lane fit is exported through the SAME function, from
`corrhead_theories.py export` (the file that already owns that fit), so the
format writer has exactly one implementation.

## 3. Evaluator (zensim, feature-gated, ZERO public API)

New module `zensim/src/corruption_head.rs` behind a new **non-default** cargo
feature `corruption-head`. `#![forbid(unsafe_code)]` holds; zensim is a `std`
crate, so no `no_std` question arises. No new dependency.

Every item is `#[doc(hidden)]` and the module is gated, so `cargo public-api`
must show **ZERO delta** on a default build. The shape below is the one being
**proposed for the user's approval**; nothing here is a supported surface until
that approval lands.

```rust
// zensim::corruption_head   (feature = "corruption-head", #[doc(hidden)])

pub struct CorruptionHead { /* opaque */ }

pub enum CorruptionHeadError {
    BadMagic { got: [u8; 4] },
    UnsupportedVersion { got: u16, supported: u16 },
    SchemaHashMismatch { stored: u64, computed: u64 },
    SectionOutOfRange { name: &'static str, offset: u32, len: u32, file_len: usize },
    SectionMisaligned { name: &'static str, offset: u32, stride: usize },
    Truncated { expected: usize, got: usize },
    FeatureLenMismatch { expected: usize, got: usize },
    MalformedTree { tree: u32, detail: &'static str },
    NotServable { missing: alloc::string::String, profile: &'static str },
}

impl CorruptionHead {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionHeadError>;

    pub fn caller_input_width(&self) -> usize;
    pub fn declared_feature_ids(&self) -> &[u16];
    pub fn n_trees(&self) -> usize;
    pub fn schema_hash(&self) -> u64;

    /// The baked deadband, in probability units: the head fires when
    /// `probability(..) > deadband()`.
    pub fn deadband(&self) -> f64;
    /// The same deadband in the dial's score units, `100 * (1 - deadband())`.
    pub fn deadband_score(&self) -> f64;

    /// P(corrupt) for one caller-width dense feature row.
    pub fn probability(&self, features: &[f32]) -> Result<f64, CorruptionHeadError>;
    pub fn probability_f64(&self, features: &[f64]) -> Result<f64, CorruptionHeadError>;
    /// `100 * (1 - probability(..))` — the head's score in the dial's units.
    pub fn score(&self, features: &[f32]) -> Result<f64, CorruptionHeadError>;
    pub fn score_f64(&self, features: &[f64]) -> Result<f64, CorruptionHeadError>;

    /// Refuse unless every declared id is a slot this profile's walk already
    /// populates. Attaching a head must NEVER widen the walk.
    #[cfg(feature = "feature-regime-v2")]
    pub fn check_servable_by(
        &self,
        profile: crate::profile::ZensimProfile,
    ) -> Result<(), CorruptionHeadError>;
}

/// THE deploy composition, in the dial's SCORE units. One owner, used by the
/// runtime companion and by `bake_verdict` for BOTH head kinds.
pub fn gate_score(perceptual: f64, head_score: f64, deadband_score: f64) -> f64;
```

`gate_score` is, verbatim, what `bake_verdict` already computes inline
(`if h < thr { d.min(0.0) } else { d }`), so adopting it changes no byte of the
logistic path. That equality is a gate, not a claim (§5, G3).

### 3.1 Runtime companion (proposed, feature-gated, NOT on the default path)

```rust
impl Zensim {
    /// Attach a corruption head. The head is evaluated on the SAME feature
    /// vector the score was computed from; it never widens the walk (refused
    /// by `check_servable_by`) and never runs unless attached.
    #[cfg(feature = "corruption-head")]
    pub fn with_corruption_head(self, head: CorruptionHead)
        -> Result<Self, CorruptionHeadError>;
}

impl ZensimResult {
    /// `Some(P(corrupt))` iff a head was attached.
    #[cfg(feature = "corruption-head")]
    pub fn corruption_probability(&self) -> Option<f64>;
    /// The score after the deploy composition; equals `score()` when no head
    /// was attached or the head did not fire.
    #[cfg(feature = "corruption-head")]
    pub fn gated_score(&self) -> f64;
}
```

**`Zensim::compute`'s returned `score()` is unchanged** — the gate is a second,
explicitly-read value. That is the whole reason this can be built before the
user approves the surface: no existing caller can observe it.

## 4. `bake_verdict` wiring

`--corruption-head <path>` sniffs the first four bytes. `ZNPR` → today's path,
untouched. `ZCTH` → `CorruptionHead::from_bytes`, `caller_input_width()` checked
against `grid.n_features` exactly as the ZNPR path does, then the same three
sections (head, DEPLOY, `--full-json` `corruption_head` block) with the same
composition owner. A file that is neither refuses loudly.

## 5. Gates — pre-registered, PASS/FAIL, none of them negotiable

| id | gate | passes iff |
|---|---|---|
| **G1** | **Numeric parity, Rust evaluator vs sklearn**, ≥ 10,000 held-out rows, f64 both sides | reported `max abs delta` on `p`, and the **fire set** `{p > T}` identical on every row (an exact set equality, not a tolerance) |
| **G2** | **Tree exactness** | the tree-sum `raw` agrees with `clf.decision_function` to **0 ulp** on the same rows — trees are exact arithmetic; only `exp`/`interp` may round |
| **G3** | **Composition identity** | `bake_verdict` on the incumbent **logistic** head, before vs after adopting `gate_score`, is **byte-identical** in `--full-json` |
| **G4** | **End-to-end** | `bake_verdict --corruption-head <exported hgb>` reproduces the theory lane's DEPLOY `pass_q20` on the 2,016-row gate grid, and equals the Python DEPLOY computed from the identical exported model |
| **G5** | **Read-set containment** | `check_servable_by(D)` PASSES for an `f0..f227` head and REFUSES a head declaring an `f228..371` id (a negative control that must fail) |
| **G6** | **Public API** | `cargo public-api --diff` = ZERO on a default build, and zero with `--all-features` minus the newly-gated hidden items |
| **G7** | **Runtime wiring** | a `Zensim` with a head attached ranks a corrupted image **below** its honest anchor where the dial alone ranked it above |
| **G8** | **Forward cost** | measured with zenbench at 1T; reported as ns/compare and as a ratio to D's own forward pass. No bar is pre-set — this is a measurement, and it is reported as it falls |

**Declared in advance, because it would otherwise look like a result:** G4's
target is a fit reproduction, and a *fresh* training run through the owner will
NOT bit-reproduce the theory lane's trees. `HistGradientBoostingClassifier` with
`early_stopping=True` draws its internal validation split with
`train_test_split(random_state=0)` over the rows **in the order they are
stacked**, and the owner's `rng.choice(9593, 9593, replace=False)` permutes the
ladder block while the theory driver does not. So G4 is run on the theory
lane's OWN fit, exported through the owner's writer; a fresh-recipe run is
reported alongside as a separate number and is **not** expected to be equal.

**Also declared in advance:** the theory lane's `t6` applies `rank_break`
(`+1e-9 * normalized_rank`) before thresholding. Its own docstring says
`P > 0.9` is unchanged to 9 decimals. This lane **verifies that empirically**
on the gate grid (with and without the tie-break) rather than assuming it, and
reports the answer either way.

## 6. What could make this lane report a failure

- G1 not bit-exact. `exp` is the only inexact step (`numpy` uses its own SIMD
  kernel, Rust's `f64::exp` calls the platform libm). The honest outcome is a
  measured `max abs delta` at the 1e-16 level; anything larger is a real defect
  and gets diagnosed, not tolerated.
- G4 landing away from 99.85 %. Reported as it falls, with the diagnosis.
- The `zensim`-side plan check refusing a head it should serve — which would
  mean `Plan::covers` and the head's declared ids disagree about the id space.

## 7. Deliverables

- `zensim/src/corruption_head.rs` + feature `corruption-head`.
- Exporter in `scripts/v_next/train_corruption_head.py` + the theory-fit export
  path in `scripts/v_next/corrhead_theories.py`.
- `bake_verdict` sniffing + the shared composition.
- `benchmarks/corruption_head_serving_2026-09-06.md` (the record),
  CHANGELOG `[Unreleased] / Added`, the CLAUDE.md corruption-head paragraph,
  `docs/DATASET_HISTORY.md` ROUND, `benchmarks/INDEX.md`.
