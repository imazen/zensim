//! **G-ADDR — the dial ADDRESSABILITY gate.**
//!
//! USER RULE (2026-09-04): *"floor and ceiling dial addressability is crucial …
//! any model that limits dial range cannot ship."* A codec loop that is asked
//! for "zensim 96" can only get there if the metric still MOVES up there; a
//! loop asked for "zensim 5" can only get there if the metric still goes down
//! there. A dial that is more monotone, better calibrated in the middle and
//! **compressed at the ends** is a worse product dial, and every existing gate
//! is blind to that: G1 asks only `p5 ≤ 25 ∧ p95 ≥ 85` (a bar a badly
//! compressed dial clears easily), G3 asks about ordering, and SROCC is
//! rank-invariant and therefore *structurally* incapable of seeing it.
//!
//! G-ADDR is a **referenced** gate: every REGRESSION bar is some *other*
//! scorer's own measured behaviour on the SAME instrument. Which scorer is the
//! reference is the whole design question, and it has one answer as of
//! 2026-09-04.
//!
//! # The reference is the REFERENCE METRIC, not the incumbent
//!
//! **USER DECISION 2026-09-04:** *"I don't think we should pin to B, ssim2
//! seems a better mentor."*
//!
//! The gate's first use pinned the regression tier to shipped **B**'s own dial
//! values and then MEASURED those pins to be defective — not merely strict,
//! but pointing the wrong way:
//!
//! * **A1 / A3 / A6 sat ABOVE the reference metric's own values on the same
//!   grid** (truth `max` 98.38 / `p95` 95.46 / DR 85.20 against bars 99.98 /
//!   99.72 / 86.08). A dial calibrated *exactly to the truth* failed all three,
//!   and both other shipped profiles did. Those bars encoded the incumbent's
//!   **stretch**, not its reach.
//! * **A4** (`p5 ≤ 13.65`) was met by B only through a **−23-point low-band
//!   bias**: on the 221 lowest-truth cells B reads +11.97 against a truth of
//!   −11.30, and the train-on-test ORACLE — the best any monotone re-map of
//!   B's ordering can do — reads `p5` 21.5-22.8. B's low `p5` is the low band
//!   mapped *below* its conditional median, and the old A4 rewarded that.
//!
//! So the incumbent is the wrong mentor: pinning to it bars a candidate for
//! being *closer to the truth than the incumbent is*. The bars are therefore
//! derived from **`peer_ssim2` — the reference metric the dial's own anchor
//! target is built from** — measured by the owner instrument on the identical
//! grid and probes. The direction semantics are unchanged in form and sharper
//! in meaning: **a candidate must address at least the range ssim2 addresses.**
//!
//! Both pin sets stay in the registry and both are printed. `bar` is
//! `peer_ssim2` (what a candidate must clear); `incumbent` is shipped **B**
//! (what users have today). A reader can always see whether a fail is "worse
//! than the mentor" or "worse than what shipped", and the B pins remain
//! readable as an incumbent-comparison row — labelled biased, never a bar.
//!
//! Re-pinning is **not** a relaxation, and it did not unblock anything: it
//! moves the difficulty from the ceiling to the floor. ssim2's grid `min` is
//! −55.35 against B's +3.13 and its negative-tail probe is 100 % below zero
//! against B's 0 %, so A2 / A5 / A7-A9 all got *much* harder — which is
//! correct, because the floor is exactly where the incumbent is genuinely
//! broken.
//!
//! # What it measures, and where each number comes from
//!
//! | axis | measured on | direction |
//! |---|---|---|
//! | `max`, `p95` | pooled dial-grid scores | **ceiling** — bigger is better |
//! | `min`, `p5` | pooled dial-grid scores | **floor** — smaller is better |
//! | `reach` (= max − min), `dynamic_range` (= p95 − p5) | pooled dial-grid scores | bigger is better |
//! | `mono`, `tied` | the dial panel's ladder accounting | the registered G3 bars |
//! | floor representability (`A7r`) | the dial grid's per-codec quality ladders | the codec's lowest settings must still RESOLVE — see the re-pin section below; no dial value is a bar |
//! | negative-tail probe (`A8r`) | a pinned negative-tail probe | **report-only**, no bar |
//! | identity (`dial`, `above-identity` count) | a pinned identity probe (ref == dist) | inside the registered band, and nothing above it |
//!
//! # Why the floor axis is the hard one (read before trying to pass it)
//!
//! MEASURED 2026-09-04: shipped **B**'s dial anchor
//! (`multiband_anchor_dial100.parquet`) carries `target_score = max(ssim2, 0)`.
//! 147 of its 2,000 rows have a genuinely negative ssim2 (down to −64.16) and
//! **every one of them is stored as 0**. `dial_spline::fit_spline_knots` then
//! turns that run of `y == 0` bins into a single bottom knot at `y = 0` (the
//! `neg_tail` dedup), so the spline has *no in-distribution evidence at all*
//! about how far below zero the dial should go — the whole negative tail is a
//! linear extrapolation off one knot. Correcting the dial's era skew (a
//! near-uniform **+3.9 … +4.8** point lift; see
//! `benchmarks/imazen26_anchor_2026-09-04.md` §4b) therefore lifts the floor
//! with everything else, and the clamp is the reason nothing pushes back.
//!
//! So a candidate that only re-anchors will fail the floor axis by
//! construction. The lever that exists is the **clamp**, not the anchor:
//! unclamped negative targets give the bottom bins real `y < 0` evidence.
//!
//! # The NEGATIVE TAIL is now FLOOR REPRESENTABILITY, per codec (2026-09-05)
//!
//! **USER RULING 2026-09-05 — the operative form**, verbatim: *"i care that the
//! lowest configurable settings per codec are representable, not that negative
//! fifty is in that specifically."*
//!
//! *(Two earlier forms are recorded in the registry because they are how the
//! rule was arrived at — "the negative tail bar is entirely arbitrary. below
//! -5-50", then "i said -50 not -5, codecs are all different, some go lower
//! than others". **Neither −5 nor −50 is a bar anywhere in the active tier.**)*
//!
//! The retired rows asked a candidate to reach `min ≤ −770.62` and `p1 ≤
//! −187.13` — `peer_ssim2`'s incidental depth on one probe — and
//! `frac_below_zero ≥ 1.0000`, a bar that was **definitional** rather than
//! measured (the probe's population was *selected* on `ssim2 < 0`). They were
//! minted after a D-peaks candidate — CID22 **+0.00798** over shipped D,
//! CONTRACT **6/6** — was refused on **A8 alone**.
//!
//! **Depth was the wrong question.** A dial can reach −700 and still be useless
//! at the bottom if its three lowest steps tie or invert; a dial that stops at
//! −12 is fine if every step still resolves. So the axis asks whether the
//! codec's lowest settings are *distinguishable*:
//!
//! | row | tier | axis | bar |
//! |---|---|---|---|
//! | `A7r` | regression | **per codec** on the dial grid: fraction of `(image, codec)` ladders whose `K` lowest configurable settings are REPRESENTED | the **mentor's own fraction** on the same cells, registry-pinned |
//! | `A8r` | **report-only** | the negative-tail probe: pooled `min` / `p1` | **none** |
//!
//! A ladder is **REPRESENTED** when both halves hold:
//!
//! 1. **ordered** — the dial strictly increases across the `K` lowest steps
//!    *and* into the next step up. A tie means the codec's two lowest settings
//!    are indistinguishable; an inversion means they are ranked backwards.
//! 2. **off the clamp** — no bottom-`K` value sits within `clamp_eps` of the
//!    dial's instrument-wide minimum, *unless* this ladder is the **single**
//!    ladder attaining it. Somebody has to be lowest; two or more ladders
//!    sharing the bottom value is a floor that has collapsed onto a clamp.
//!
//! `q` is quality-oriented on every codec in the grid — JXL's `param_kind` is
//! `distance` and its `q = 0` cells carry the **largest** distance — so "the
//! lowest configurable settings" is always the smallest `q`, with no per-codec
//! direction switch.
//!
//! **`A8r` is report-only for a measured reason**: the negative-tail probes
//! carry no codec identity at all (`entry` is a bare row index over a KADIS
//! synthetic-distortion cut), so they cannot answer a question about codec
//! settings. Grading that instrument per *distortion* family at a fixed depth
//! was measured to fail **every bake ever built** on one n=8 family
//! (`benchmarks/d_peaks_lambda_sweep_2026-09-05.md` §4-§6).
//!
//! **`A9r` is dropped as a bar.** Its per-codec quantity is folded into the
//! report block as one column, against a REPORTING threshold that is barred
//! against nothing.
//!
//! **Scope.** The ruling is about the REGRESSION tail. No CONTRACT row was
//! added, moved or removed, so the contract tier — and the board's
//! contract-driven NOT SHIPPABLE badge — is untouched. `A1`-`A6` stay
//! mentor-pinned, and `C3`/`C4` keep their absolute `0.0` bars, which are SIGN
//! requirements rather than depth bars.
//!
//! **Nothing was rewritten.** `--gaddr-tail-pins retired` reproduces the
//! pre-2026-09-05 grading exactly, because every G-ADDR number published before
//! that date is graded on it.
//!
//! # Absent is never passed
//!
//! The negative-tail and identity axes need probes the caller supplies. When
//! one is missing the axis is **NOT MEASURED** — printed distinctly, never
//! folded into the pass count, and the overall verdict becomes `INCOMPLETE`
//! rather than `PASS`. Same rule for an unregistered grid: it is
//! `NOT MEASURABLE`, never a pass, because a bar you can dodge by choosing a
//! friendlier instrument is not a bar.

use serde::Deserialize;

/// The committed floor registry. `include_str!` so a stale or missing file at
/// run time cannot silently disarm the gate (same discipline as
/// [`crate::contamination_guard`] and [`crate::dial_content`]).
const REGISTRY_JSON: &str =
    include_str!("../../benchmarks/dial_addressability_floor_2026-09-04.json");

// ─────────────────────────────── registry ───────────────────────────────

/// The reference whose measured values are the REGRESSION bars.
/// **USER DECISION 2026-09-04** — the reference metric, not the incumbent.
pub const REFERENCE_PEER_SSIM2: &str = "peer_ssim2";

/// The shipped product dial. Kept as a readable *incumbent* pin set — printed
/// beside every bar so "worse than the mentor" and "worse than what shipped"
/// are never confused — but **no longer a bar**. See the module docs for the
/// four measured defects that retired it.
pub const REFERENCE_SHIPPED_B: &str = "shipped_b";

/// The pin set the REGRESSION tier bars against.
pub const ACTIVE_REFERENCE: &str = REFERENCE_PEER_SSIM2;

/// The pin set reported in the `incumbent` column.
pub const INCUMBENT_REFERENCE: &str = REFERENCE_SHIPPED_B;

/// Registry rows written before 2026-09-04 carry no `reference` field; they are
/// all shipped-B measurements. A serde default keeps those rows **byte-
/// untouched**, which is what the registry's append-only rule requires.
fn default_reference() -> String {
    REFERENCE_SHIPPED_B.to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct FixedBars {
    pub mono_min: f64,
    pub tied_max: f64,
    pub identity_lo: f64,
    pub identity_hi: f64,
}

/// Which pin set the NEGATIVE-TAIL rows are graded against.
///
/// **USER RULING 2026-09-05**, verbatim: *"the negative tail bar is entirely
/// arbitrary. below -5-50"*, in its operative form: *"i care that the lowest
/// configurable settings per codec are representable, not that negative fifty
/// is in that specifically."* The mentor-pinned tail (A7 `min` ≤ −770.62, A8
/// `p1` ≤ −187.13) is retired as a *product* requirement — that was
/// `peer_ssim2`'s incidental depth on one probe — and replaced by PER-CODEC
/// FLOOR REPRESENTABILITY, which contains no dial-value bar at all.
///
/// Both sets stay in the registry and both remain reachable, because every
/// G-ADDR number published before 2026-09-05 is graded on the retired one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailPins {
    /// `A7r` (per-codec FLOOR REPRESENTABILITY) + `A8r` (the probe,
    /// report-only). The default. Contains no dial-value bar.
    #[default]
    Product,
    /// `A7` / `A8` / `A9` — the retired `peer_ssim2` pins. Reproduces every
    /// pre-2026-09-05 grading exactly.
    Retired,
}

impl TailPins {
    pub fn tag(self) -> &'static str {
        match self {
            TailPins::Product => "product",
            TailPins::Retired => "retired",
        }
    }
    /// Parse the `--gaddr-tail-pins` value. Unknown values are an error, never
    /// a silent fallback to the default — an arm selected by a typo would grade
    /// against bars nobody chose.
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "product" | "product-range-2026-09-05" => Ok(TailPins::Product),
            "retired" | "mentor" | "mentor-2026-09-04" => Ok(TailPins::Retired),
            other => Err(format!(
                "unknown --gaddr-tail-pins `{other}` (expected `product` or `retired`)"
            )),
        }
    }
}

/// Which TIER the six dial-VALUE rows `A1`-`A6` sit in.
///
/// **USER RULING 2026-09-05.** Asked whether the dial has *"poor resolution
/// compared to ssim2"*, the lane reported that `A1`-`A6` bar a candidate
/// against `peer_ssim2`'s own `max`/`p95`/`min`/`p5`/`reach`/`dynamic_range`
/// — incidental properties of where the mentor's distribution happens to land
/// on one instrument — and recommended demoting them to reporting, leaving the
/// CONTRACT tier `C1`-`C6` plus the per-codec floor `A7r` to carry the product
/// requirements. The user answered **"ok"**, read as accepting that.
///
/// **The values are still measured and still printed** — only their tier
/// changes, so nothing stops being visible. `Hard` restores the pre-ruling
/// grading row-for-row (`--gaddr-value-pins hard`), which is the reversibility
/// lever; the CONTRACT tier is untouched by either setting, so the board's
/// NOT-SHIPPABLE badge cannot move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ValuePins {
    /// A1-A6 on [`Tier::Report`] — measured, printed, gating nothing. Default
    /// since the 2026-09-05 ruling.
    #[default]
    Report,
    /// A1-A6 back on [`Tier::Regression`] — the pre-ruling grading.
    Hard,
}

impl ValuePins {
    pub fn tag(self) -> &'static str {
        match self {
            ValuePins::Report => "report",
            ValuePins::Hard => "hard",
        }
    }
    /// The tier `A1`-`A6` are emitted on.
    pub fn tier(self) -> Tier {
        match self {
            ValuePins::Report => Tier::Report,
            ValuePins::Hard => Tier::Regression,
        }
    }
    /// Parse `--gaddr-value-pins`. Unknown values are an error, never a silent
    /// fallback — same discipline as [`TailPins::parse`].
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "report" | "report-only" => Ok(ValuePins::Report),
            "hard" | "regression" => Ok(ValuePins::Hard),
            other => Err(format!(
                "unknown --gaddr-value-pins `{other}` (expected `report` or `hard`)"
            )),
        }
    }
}

/// Which STEPS of a ladder `A7r` tests — an OWNER-EXTENSION, opt-in variant of
/// the pinned FLOOR-REPRESENTABILITY rule, added 2026-09-06 to let a pending
/// user ruling be graded without moving the default.
///
/// **`Distinct` is the default and reproduces the pinned rule byte-for-byte**
/// (`FloorMeasure::from_grid` still delegates to it): the literal `K` lowest
/// DISTINCT settings by `q`, plus the next one up. `benchmarks/
/// ladder_floor_resolution_2026-09-05.md` asked whether that literal-position
/// window is itself sound, or an artifact of grading three positions that
/// happen to sit on a near-flat part of a codec's RD curve — and found BOTH:
/// jpeg's one-ladder miss under `Distinct` is a boundary artifact (dissolves
/// under either alternative window), while `avif-rav1e`'s is not (a genuine
/// ordering defect the shared noise floor between candidate and mentor was
/// hiding). `Resolvable` and `Spaced` are that report's two windows, promoted
/// to a reusable rule so the comparison is a flag, not a one-off Python port.
///
/// **Neither variant changes what "represented" MEANS** — a ladder is still
/// REPRESENTED when the dial strictly increases across the window AND no
/// bottom step sits on the instrument's clamp (unless it is the sole holder).
/// They change WHICH steps of the ladder are asked to clear that bar, using
/// the REFERENCE metric's (the mentor's) own per-cell values to choose —
/// never the candidate's own values, so the same window applies to every
/// scorer graded on it.
///
/// **The bar changes shape too.** `Distinct`'s bar is the mentor's fraction
/// PINNED in the registry (`benchmarks/dial_addressability_floor_2026-09-04.json`)
/// — a measurement made once and reused. `Resolvable`/`Spaced` have no
/// registry entry (they are report-derived, not pinned) — see
/// [`per_codec_floor_rows_live`] — so their bar is ALWAYS computed live, by
/// grading the mentor's own per-cell truth through the identical
/// [`FloorMeasure::from_grid_with_rule`] call the candidate went through.
/// Every row and note this produces is stamped `rule=<tag>` (via
/// [`FloorRule::tag`]) so a `resolvable` fraction can never be silently
/// compared against a `distinct` or `spaced` one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FloorRule {
    /// The pinned rule. Positions `0..=K` literally, by `q`. Default.
    #[default]
    Distinct,
    /// Variant (a): walk forward from the lowest setting, skipping any step
    /// whose `|Δ mentor|` from the last SELECTED step is below `margin`,
    /// until `K + 1` mentor-resolvable steps are collected. A ladder that
    /// cannot produce `K + 1` such steps (too short, or every remaining step
    /// ties the mentor) is "too short" — the same accounting `Distinct` uses
    /// for a ladder shorter than `K + 1` steps.
    Resolvable { margin: f64 },
    /// Variant (b): the lowest setting, plus the step whose MENTOR value is
    /// nearest `mentor[0] + near_lo`, plus the step nearest
    /// `mentor[0] + near_hi` — both drawn from settings above the floor, the
    /// three then re-sorted by position (the ladder is already sorted by
    /// `q`, so sorting indices sorts by `q`).
    Spaced { near_lo: f64, near_hi: f64 },
}

impl FloorRule {
    /// `--floor-margin`'s default, in the reference metric's own units
    /// (ssim2 points on the registered instruments). Matches
    /// `benchmarks/ladder_floor_resolution_2026-09-05.md` §4's variant (a).
    pub const RESOLVABLE_MARGIN_DEFAULT: f64 = 0.5;
    /// `Spaced`'s two fixed offsets — variant (b), ssim2 points above the
    /// ladder's lowest setting. Not exposed as flags: the task that minted
    /// this rule named exactly these two numbers.
    pub const SPACED_NEAR_LO_DEFAULT: f64 = 2.0;
    pub const SPACED_NEAR_HI_DEFAULT: f64 = 5.0;

    pub fn tag(self) -> &'static str {
        match self {
            FloorRule::Distinct => "distinct",
            FloorRule::Resolvable { .. } => "resolvable",
            FloorRule::Spaced { .. } => "spaced",
        }
    }

    /// `true` for the two rules whose window is chosen from the MENTOR's own
    /// per-cell truth (`--gaddr-grid-truth`) rather than literal positions —
    /// and whose bar is therefore always LIVE-computed, never registry-read.
    pub fn needs_mentor_truth(self) -> bool {
        !matches!(self, FloorRule::Distinct)
    }

    /// Parse `--floor-rule`. `margin` only matters for `resolvable`; `spaced`
    /// uses its own fixed offsets. Unknown values are an error, never a
    /// silent fallback — same discipline as [`TailPins::parse`].
    pub fn parse(s: &str, margin: f64) -> Result<Self, String> {
        match s {
            "distinct" => Ok(FloorRule::Distinct),
            "resolvable" => Ok(FloorRule::Resolvable { margin }),
            "spaced" => Ok(FloorRule::Spaced {
                near_lo: Self::SPACED_NEAR_LO_DEFAULT,
                near_hi: Self::SPACED_NEAR_HI_DEFAULT,
            }),
            other => Err(format!(
                "unknown --floor-rule `{other}` (expected `distinct`, `resolvable`, or `spaced`)"
            )),
        }
    }
}

/// The context [`evaluate_full`] needs to grade `A7r` under a [`FloorRule`]
/// other than the default. `Default` reproduces today's behaviour exactly:
/// `Distinct` + no mentor measurement (unused by that arm anyway).
#[derive(Debug, Clone, Copy, Default)]
pub struct FloorRuleContext<'a> {
    pub rule: FloorRule,
    /// The mentor scored AGAINST ITSELF under the SAME rule, on the SAME
    /// instrument — required (else every codec reads NOT MEASURED) when
    /// `rule.needs_mentor_truth()`; ignored for `Distinct`, which reads the
    /// registry instead.
    pub mentor: Option<&'a FloorMeasure>,
}

/// The registered parameters of the FLOOR-REPRESENTABILITY rule.
///
/// **There is no numeric dial bar here, by USER RULING** (2026-09-05, final,
/// verbatim): *"i care that the lowest configurable settings per codec are
/// representable, not that negative fifty is in that specifically."* The rule
/// asks whether a codec's **lowest configurable settings still resolve** — it
/// never asks the dial to reach any particular value.
#[derive(Debug, Clone, Deserialize)]
pub struct FloorRepresentabilityRule {
    /// How many of a ladder's lowest-quality steps must resolve. `K = 3`.
    pub bottom_k: usize,
    /// A dial value within this of the instrument-wide minimum is ON THE CLAMP.
    pub clamp_eps: f64,
    /// WHICH WINDOW the active pin set grades — `distinct` / `resolvable` /
    /// `spaced`, the tags [`FloorRule::parse`] accepts. Absent ⇒ `distinct`,
    /// so every pin set written before 2026-09-05 keeps its meaning.
    ///
    /// This is what makes the OPERATIVE rule a registry property rather than
    /// a hardcoded default: flipping `negative_tail_bars.active` back to a pin
    /// set carrying `distinct` restores the pre-ruling grading with no code
    /// change. See [`operative_floor_rule`].
    #[serde(default = "default_floor_rule_tag")]
    pub floor_rule: String,
    /// `resolvable`'s minimum `|Δ mentor|` between selected steps. Ignored by
    /// the other two rules.
    #[serde(default = "default_floor_margin")]
    pub floor_margin: f64,
}

fn default_floor_rule_tag() -> String {
    FloorRule::Distinct.tag().to_string()
}

fn default_floor_margin() -> f64 {
    FloorRule::RESOLVABLE_MARGIN_DEFAULT
}

impl FloorRepresentabilityRule {
    /// The [`FloorRule`] this pin set grades under. An unparseable tag is an
    /// ERROR, never a silent fallback to `Distinct` — a registry typo that
    /// quietly re-grades every board cell under a different window is exactly
    /// the failure this whole file is built to make impossible.
    pub fn rule(&self) -> Result<FloorRule, String> {
        FloorRule::parse(&self.floor_rule, self.floor_margin)
    }
}

/// One codec's REFERENCE floor-representability on a registered dial grid —
/// the bar `A7r` compares a candidate against, and the `incumbent` column.
#[derive(Debug, Clone, Deserialize)]
pub struct CodecFloorRow {
    pub codec: String,
    pub n_ladders: usize,
    /// Fraction of that codec's ladders whose bottom `K` steps are
    /// representable. **This is the bar** — a measurement of one scorer on one
    /// instrument, never an invented threshold.
    pub represented_frac: f64,
}

/// The registered per-codec floor representability for one dial grid.
#[derive(Debug, Clone, Deserialize)]
pub struct GridFloorRepresentability {
    pub dial_grid_sha256: String,
    #[serde(default = "default_reference")]
    pub reference: String,
    pub label: String,
    pub bottom_k: usize,
    /// WHICH WINDOW this row's fractions were measured under. Absent ⇒
    /// `distinct`, so every row written before 2026-09-05 keeps its meaning.
    /// **Load-bearing as part of the key**: a `resolvable` fraction and a
    /// `distinct` fraction on the same grid are different quantities (shipped
    /// D reads jpeg 0.5128 under one and 0.6667 under the other), so a lookup
    /// that ignored this field would bar a candidate against the wrong number.
    #[serde(default = "default_floor_rule_tag")]
    pub floor_rule: String,
    /// The `--floor-margin` these fractions were measured at, when the rule
    /// takes one. Absent ⇒ [`FloorRule::RESOLVABLE_MARGIN_DEFAULT`].
    #[serde(default = "default_floor_margin")]
    pub floor_margin: f64,
    pub codecs: Vec<CodecFloorRow>,
}

/// One row of `negative_tail_bars.pin_sets`. The rule fields are optional
/// because the RETIRED set legitimately has none — it barred against the
/// reference's own probe depth instead. `#[serde(flatten)]` was deliberately
/// not used: an `Option<Struct>` flatten silently yields `None` when a single
/// field is mistyped, which is exactly how a gate quietly disarms itself.
#[derive(Debug, Clone, Deserialize)]
struct TailPinSetRow {
    id: String,
    #[serde(default)]
    bottom_k: Option<usize>,
    #[serde(default)]
    clamp_eps: Option<f64>,
    #[serde(default)]
    report_floor_threshold: Option<f64>,
    /// Absent ⇒ `distinct`, so a pre-2026-09-05 pin set keeps its meaning.
    #[serde(default)]
    floor_rule: Option<String>,
    #[serde(default)]
    floor_margin: Option<f64>,
}

impl TailPinSetRow {
    fn rule(&self) -> Option<FloorRepresentabilityRule> {
        Some(FloorRepresentabilityRule {
            bottom_k: self.bottom_k?,
            clamp_eps: self.clamp_eps?,
            floor_rule: self
                .floor_rule
                .clone()
                .unwrap_or_else(default_floor_rule_tag),
            floor_margin: self.floor_margin.unwrap_or_else(default_floor_margin),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct NegativeTailBarsRegistry {
    active: String,
    pin_sets: Vec<TailPinSetRow>,
}

/// The registered FLOOR-REPRESENTABILITY rule. Owned by the registry, not by
/// this file — the parameters a user moves live in one committed place.
pub fn floor_rule() -> FloorRepresentabilityRule {
    let r = registry();
    let want = r.negative_tail_bars.active.clone();
    r.negative_tail_bars
        .pin_sets
        .into_iter()
        .find(|p| p.id == want)
        .and_then(|p| p.rule())
        .expect("the active negative-tail pin set must carry the floor rule")
}

/// THE OPERATIVE [`FloorRule`] — the window `A7r` grades under when the caller
/// does not name one. Owned by the registry's ACTIVE pin set, not by this
/// file.
///
/// **USER RULING 2026-09-05.** Asked *"is there poor resolution compared to
/// ssim2?"*, the lane reported that `distinct`'s literal bottom-3 window
/// grades steps the mentor itself cannot tell apart, and recommended grading
/// only the steps ssim2 resolves by ≥ 0.5 points. The user answered **"ok"**,
/// which is read as accepting that recommendation — so the active pin set
/// carries `floor_rule: "resolvable"`, `floor_margin: 0.5`.
///
/// **Reversible with no code change**: point `negative_tail_bars.active` back
/// at `floor-representability-2026-09-05` (which carries no `floor_rule`, so
/// it reads `distinct`) and the pre-ruling window returns. The per-invocation
/// lever is `--floor-rule distinct`.
pub fn operative_floor_rule() -> FloorRule {
    floor_rule()
        .rule()
        .expect("the active pin set's `floor_rule` must be a known FloorRule tag")
}

/// The REPORTING threshold for the per-codec column folded in from the dropped
/// `A9r`. **It is barred against nothing** — the active pin set contains no
/// dial-value bar, by user ruling. This is the one place the old −50 survives,
/// and it survives as information.
pub fn report_floor_threshold() -> f64 {
    let r = registry();
    let want = r.negative_tail_bars.active.clone();
    r.negative_tail_bars
        .pin_sets
        .into_iter()
        .find(|p| p.id == want)
        .and_then(|p| p.report_floor_threshold)
        .unwrap_or(f64::NEG_INFINITY)
}

/// The registry's own name for the ACTIVE negative-tail pin set.
pub fn active_tail_pin_set() -> String {
    registry().negative_tail_bars.active
}

/// The registered per-codec floor representability for a dial grid, keyed by
/// `(grid sha256, reference)` — the same two-part key every other registry
/// lookup uses, for the same reason.
/// The registered per-codec floor representability for one
/// `(grid, reference, RULE)`. **All three are load-bearing.** The rule joined
/// the key on 2026-09-05: the same grid now carries a `distinct` row and a
/// `resolvable` row for the same reference, and they are different
/// measurements — returning either one for the other would bar a candidate
/// against a window it was never graded on.
///
/// `margin` is matched only for rules that take one, and only to
/// `f64::EPSILON` — a row measured at 0.5 does not answer a query at 0.25.
pub fn floor_repr_for_grid_under(
    grid_sha256: &str,
    reference: &str,
    rule: FloorRule,
) -> Option<GridFloorRepresentability> {
    let tag = rule.tag();
    registry()
        .grid_floor_representability
        .into_iter()
        .find(|g| {
            g.dial_grid_sha256 == grid_sha256
                && g.reference == reference
                && g.floor_rule == tag
                && match rule {
                    FloorRule::Resolvable { margin } => {
                        (g.floor_margin - margin).abs() <= f64::EPSILON
                    }
                    _ => true,
                }
        })
}

/// [`floor_repr_for_grid_under`] at the PINNED rule (`distinct`). Retained so
/// pre-2026-09-05 callers and their tests keep asking exactly what they asked.
pub fn floor_repr_for_grid(
    grid_sha256: &str,
    reference: &str,
) -> Option<GridFloorRepresentability> {
    floor_repr_for_grid_under(grid_sha256, reference, FloorRule::Distinct)
}

#[derive(Debug, Clone, Deserialize)]
pub struct GridFloor {
    pub dial_grid_sha256: String,
    pub label: String,
    /// Which scorer this row measures. Absent ⇒ [`REFERENCE_SHIPPED_B`].
    #[serde(default = "default_reference")]
    pub reference: String,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub n_rows: usize,
    pub min: f64,
    pub max: f64,
    pub p5: f64,
    pub p95: f64,
    pub reach: f64,
    pub dynamic_range: f64,
    pub mono: f64,
    pub tied: f64,
}

/// One registered negative-tail probe row.
#[derive(Debug, Clone, Deserialize)]
pub struct NegTailFloor {
    pub probe_sha256: String,
    pub label: String,
    /// Which scorer this row measures. Absent ⇒ [`REFERENCE_SHIPPED_B`].
    #[serde(default = "default_reference")]
    pub reference: String,
    #[serde(default)]
    pub n_rows: usize,
    pub min: f64,
    pub p1: f64,
    pub p5: f64,
    pub frac_below_zero: f64,
}

/// One registered identity probe row.
#[derive(Debug, Clone, Deserialize)]
pub struct IdentityFloor {
    pub probe_sha256: String,
    pub label: String,
    /// Which scorer this row measures. Absent ⇒ [`REFERENCE_SHIPPED_B`].
    #[serde(default = "default_reference")]
    pub reference: String,
    #[serde(default)]
    pub n_rows: usize,
    pub dial_min: f64,
    pub dial_median: f64,
    pub dial_max: f64,
}

#[derive(Debug, Clone, Deserialize)]
struct Registry {
    fixed_bars: FixedBars,
    negative_tail_bars: NegativeTailBarsRegistry,
    grid_floor_representability: Vec<GridFloorRepresentability>,
    grids: Vec<GridFloor>,
    negtail_probes: Vec<NegTailFloor>,
    identity_probes: Vec<IdentityFloor>,
}

fn registry() -> Registry {
    // A malformed registry is a build-time defect, not a run-time condition:
    // the file is embedded, so if it parses in the test it parses here.
    serde_json::from_str(REGISTRY_JSON).expect("dial_addressability floor registry must parse")
}

/// The registered fixed bars (mono / tied / identity band).
pub fn fixed_bars() -> FixedBars {
    registry().fixed_bars
}

/// The registered floor for a dial grid, keyed by `(grid file sha256,
/// reference)`. A row is a measurement OF ONE SCORER on one instrument, so both
/// halves of the key are load-bearing: reading a `peer_ssim2` bar off a
/// `shipped_b` row would silently compare a candidate against the wrong mentor.
pub fn floor_for_grid(grid_sha256: &str, reference: &str) -> Option<GridFloor> {
    registry()
        .grids
        .into_iter()
        .find(|g| g.dial_grid_sha256 == grid_sha256 && g.reference == reference)
}

/// The registered floor for a negative-tail probe, keyed by `(probe sha256,
/// reference)`.
pub fn floor_for_negtail(probe_sha256: &str, reference: &str) -> Option<NegTailFloor> {
    registry()
        .negtail_probes
        .into_iter()
        .find(|p| p.probe_sha256 == probe_sha256 && p.reference == reference)
}

/// The registered floor for an identity probe, keyed by `(probe sha256,
/// reference)`.
pub fn floor_for_identity(probe_sha256: &str, reference: &str) -> Option<IdentityFloor> {
    registry()
        .identity_probes
        .into_iter()
        .find(|p| p.probe_sha256 == probe_sha256 && p.reference == reference)
}

// ─────────────────────────────── measures ───────────────────────────────

/// Percentiles + extremes of a pooled score vector. `p` is in 0..=100 and the
/// interpolation matches `numpy.percentile` (the convention every other panel
/// in this crate uses).
fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (sorted.len() as f64 - 1.0);
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// End-of-range behaviour on the dial grid. `mono` / `tied` are handed in from
/// the dial panel's own ladder accounting so the two sections can never
/// disagree about the same event.
#[derive(Debug, Clone, Copy)]
pub struct GridMeasure {
    pub min: f64,
    pub max: f64,
    pub p5: f64,
    pub p95: f64,
    pub reach: f64,
    pub dynamic_range: f64,
    pub mono: f64,
    pub tied: f64,
    pub n: usize,
}

impl GridMeasure {
    /// Build from the pooled dial-space scores (any order; non-finite dropped).
    pub fn from_pooled(scores: &[f64], mono: f64, tied: f64) -> Self {
        let mut v: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return Self {
                min: f64::NAN,
                max: f64::NAN,
                p5: f64::NAN,
                p95: f64::NAN,
                reach: f64::NAN,
                dynamic_range: f64::NAN,
                mono,
                tied,
                n: 0,
            };
        }
        let (min, max) = (v[0], v[v.len() - 1]);
        let (p5, p95) = (pct(&v, 5.0), pct(&v, 95.0));
        Self {
            min,
            max,
            p5,
            p95,
            reach: max - min,
            dynamic_range: p95 - p5,
            mono,
            tied,
            n: v.len(),
        }
    }
}

/// Negative-tail behaviour on a pinned probe of genuinely-negative-truth rows.
///
/// Carries the probe's OWN reference truth extremes, which the 2026-09-05
/// product bar needs: an absolute bar is still instrument-dependent, and a
/// probe whose truth is shallow cannot say much about a floor. Reported, never
/// barred.
#[derive(Debug, Clone, Copy)]
pub struct NegTailMeasure {
    pub n: usize,
    pub min: f64,
    pub p1: f64,
    pub p5: f64,
    pub frac_below_zero: f64,
    /// The probe's own reference truth (`ssim2_gpu`) extremes, or NaN when the
    /// probe carries no readable truth column.
    pub truth_min: f64,
    pub truth_p1: f64,
}

impl NegTailMeasure {
    /// Dial scores only — no probe truth, so `A8r`'s reachability guard reads
    /// NOT MEASURED.
    pub fn from_scores(scores: &[f64]) -> Self {
        Self::from_scores_and_truth(scores, None)
    }

    /// Dial scores plus the probe's own reference truth. A length mismatch is
    /// treated as no truth at all rather than silently zipped short.
    pub fn from_scores_and_truth(scores: &[f64], truth: Option<&[f64]>) -> Self {
        let mut v: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        let truth = truth.filter(|t| t.len() == scores.len());
        let (truth_min, truth_p1) = match truth {
            Some(t) => {
                let mut ts: Vec<f64> = t.iter().copied().filter(|x| x.is_finite()).collect();
                ts.sort_by(f64::total_cmp);
                if ts.is_empty() {
                    (f64::NAN, f64::NAN)
                } else {
                    (ts[0], pct(&ts, 1.0))
                }
            }
            None => (f64::NAN, f64::NAN),
        };
        if v.is_empty() {
            return Self {
                n: 0,
                min: f64::NAN,
                p1: f64::NAN,
                p5: f64::NAN,
                frac_below_zero: f64::NAN,
                truth_min,
                truth_p1,
            };
        }
        let below = v.iter().filter(|x| **x < 0.0).count();
        Self {
            n: v.len(),
            min: v[0],
            p1: pct(&v, 1.0),
            p5: pct(&v, 5.0),
            frac_below_zero: below as f64 / v.len() as f64,
            truth_min,
            truth_p1,
        }
    }
}

/// One codec's FLOOR REPRESENTABILITY on an instrument that carries codec
/// identity and quality ladders — the candidate side of `A7r`.
///
/// **USER RULING 2026-09-05 (final)**, verbatim: *"i care that the lowest
/// configurable settings per codec are representable, not that negative fifty
/// is in that specifically."*
#[derive(Debug, Clone)]
pub struct CodecFloor {
    pub codec: String,
    /// `(image_id, codec)` ladders long enough to test (`bottom_k + 1` steps).
    pub n_ladders: usize,
    pub n_represented: usize,
    /// Ladders skipped because they are shorter than `bottom_k + 1`.
    pub n_too_short: usize,
    pub represented_frac: f64,
    /// Why the failures failed, so a report says which half of the rule bit.
    pub n_fail_order: usize,
    pub n_fail_clamp: usize,
    /// The dial's minimum over this codec's cells.
    pub dial_min: f64,
    /// Median dial value at each of the bottom `K` quality steps, lowest first.
    pub bottom_medians: Vec<f64>,
    /// REPORT-ONLY, folded in from the dropped `A9r`: of this codec's cells
    /// whose reference truth is at or below the report threshold, the fraction
    /// the dial also places at or below it. `None` without per-row truth.
    pub report_frac_at_floor: Option<f64>,
    pub report_n_at_floor: Option<usize>,
}

/// Per-codec floor representability on one instrument.
#[derive(Debug, Clone)]
pub struct FloorMeasure {
    pub instrument: String,
    /// The dial's minimum over the WHOLE instrument — the clamp value a bottom
    /// step is tested against.
    pub grid_min: f64,
    /// How many distinct ladders attain `grid_min`. **One** is a genuine floor
    /// (some ladder has to be lowest); **two or more is a collapsed floor**,
    /// and then every cell sitting there counts as clamped.
    pub n_ladders_at_min: usize,
    pub codecs: Vec<CodecFloor>,
}

/// One ladder cell: `(q, candidate dial value, mentor truth if supplied)`.
/// `Distinct` never reads the third field; `Resolvable`/`Spaced` use it to
/// choose the window (see [`FloorMeasure::from_grid_with_rule`]).
type LadderCell = (f64, f64, Option<f64>);
/// `(image_id, codec) -> its cells, sorted by quality`.
type Ladders = std::collections::BTreeMap<(String, String), Vec<LadderCell>>;

impl FloorMeasure {
    /// Measure floor representability from row-aligned
    /// `(image_id, codec, q, dial)`, where `q` is QUALITY-ORIENTED on every
    /// codec (the dial grid stores JXL's largest distance as its lowest `q`,
    /// so "lowest configurable settings" is always the smallest `q`).
    ///
    /// A ladder is REPRESENTED when both halves hold:
    ///
    /// 1. **ordered** — the dial is strictly increasing across the bottom `K`
    ///    steps AND from the `K`-th step to the next one up. No ties, no
    ///    inversions. A tie at the bottom means the codec's two lowest
    ///    settings are indistinguishable on the dial; an inversion means they
    ///    are ranked backwards.
    /// 2. **off the clamp** — no bottom-`K` value sits within `clamp_eps` of
    ///    the instrument-wide minimum, *unless* this ladder is the single
    ///    ladder attaining that minimum. Somebody has to be lowest; two or
    ///    more ladders sharing the bottom value is a floor that has collapsed
    ///    onto a clamp, and neither is then addressable.
    ///
    /// `report_truth` is optional per-row reference truth used ONLY for the
    /// report-only column folded in from the dropped `A9r`; it never affects
    /// whether a ladder is represented.
    pub fn from_grid(
        instrument: &str,
        image_id: &[String],
        codec: &[String],
        q: &[f64],
        dial: &[f64],
        report_truth: Option<&[f64]>,
        report_threshold: f64,
    ) -> Self {
        Self::from_grid_with_rule(
            instrument,
            image_id,
            codec,
            q,
            dial,
            report_truth,
            report_threshold,
            FloorRule::Distinct,
        )
    }

    /// [`from_grid`] under an explicit [`FloorRule`]. `from_grid` delegates to
    /// this with `FloorRule::Distinct` — so this is the ONE implementation,
    /// never a fork: `Distinct`'s branch below reduces algebraically to
    /// exactly the pre-2026-09-06 body (window = literal positions `0..=K`,
    /// same `ordered`/`clamped` arithmetic), which is what makes the default
    /// path byte-identical rather than merely intended to be — see
    /// `dial_addressability::tests::distinct_rule_matches_legacy_from_grid`.
    ///
    /// `report_truth` serves TWO roles depending on `rule`: for every rule it
    /// still feeds the report-only column folded in from the dropped `A9r`;
    /// for `Resolvable`/`Spaced` it is ALSO the MENTOR's own per-cell value
    /// used to choose which steps of each ladder the window tests — never the
    /// candidate's (`dial`'s) own values, so the identical window applies
    /// whichever scorer is graded. A ladder whose mentor truth is missing (or
    /// entirely absent) cannot have a window computed under those two rules
    /// and is counted "too short", the same accounting a too-short ladder
    /// gets under `Distinct`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_grid_with_rule(
        instrument: &str,
        image_id: &[String],
        codec: &[String],
        q: &[f64],
        dial: &[f64],
        report_truth: Option<&[f64]>,
        report_threshold: f64,
        rule: FloorRule,
    ) -> Self {
        let frule = floor_rule();
        let k = frule.bottom_k;
        let n = dial.len().min(image_id.len()).min(codec.len()).min(q.len());
        let report_truth = report_truth.filter(|t| t.len() == dial.len());

        // Build the ladders, then sort each by QUALITY. Each cell carries the
        // candidate's dial value AND (optionally) the mentor's own truth at
        // that same cell — `Distinct` never reads the third field.
        let mut ladders: Ladders = std::collections::BTreeMap::new();
        for i in 0..n {
            if !dial[i].is_finite() {
                continue;
            }
            let mentor = report_truth.map(|t| t[i]);
            ladders
                .entry((image_id[i].clone(), codec[i].clone()))
                .or_default()
                .push((q[i], dial[i], mentor));
        }
        for v in ladders.values_mut() {
            v.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        let grid_min = ladders
            .values()
            .flat_map(|v| v.iter().map(|(_, d, _)| *d))
            .fold(f64::INFINITY, f64::min);
        let n_ladders_at_min = ladders
            .values()
            .filter(|v| {
                v.iter()
                    .any(|(_, d, _)| (*d - grid_min).abs() <= frule.clamp_eps)
            })
            .count();

        // Per-codec accumulation, in a stable (sorted) codec order.
        let mut order: Vec<String> = Vec::new();
        #[derive(Default)]
        struct Acc {
            n: usize,
            rep: usize,
            short: usize,
            fail_order: usize,
            fail_clamp: usize,
            dial_min: f64,
            steps: Vec<Vec<f64>>,
        }
        let mut acc: std::collections::HashMap<String, Acc> = std::collections::HashMap::new();
        for ((_, c), cells) in &ladders {
            let e = acc.entry(c.clone()).or_insert_with(|| {
                order.push(c.clone());
                Acc {
                    dial_min: f64::INFINITY,
                    steps: vec![Vec::new(); k],
                    ..Default::default()
                }
            });
            for (_, d, _) in cells {
                if *d < e.dial_min {
                    e.dial_min = *d;
                }
            }
            // Choose which STEPS of this ladder get tested. `Distinct`
            // reproduces the pre-2026-09-06 literal window exactly; the other
            // two are chosen from the MENTOR's own values (see the fn doc).
            let window: Option<Vec<usize>> = match rule {
                FloorRule::Distinct => (cells.len() > k).then(|| (0..=k).collect::<Vec<usize>>()),
                FloorRule::Resolvable { margin } => resolvable_window(cells, k, margin),
                FloorRule::Spaced { near_lo, near_hi } => spaced_window(cells, near_lo, near_hi),
            };
            let Some(win) = window else {
                e.short += 1;
                continue;
            };
            e.n += 1;
            for (j, &idx) in win.iter().take(k).enumerate() {
                e.steps[j].push(cells[idx].1);
            }
            let vals: Vec<f64> = win.iter().map(|&idx| cells[idx].1).collect();
            // (1) strictly increasing across the WHOLE selected window.
            //     `Distinct`'s window is `[0, 1, .., K]` (K+1 points), so this
            //     is exactly "the bottom K steps and into step K".
            let ordered = vals.windows(2).all(|w| w[0] < w[1]);
            // (2) off the clamp — the single lowest ladder is allowed to BE
            //     the minimum; two or more sharing it is a collapsed floor.
            //     Checked over the WHOLE ladder (not just the window): a
            //     ladder that touches the clamp anywhere still collapses the
            //     floor for `n_ladders_at_min`'s purposes.
            let sole_min_holder = n_ladders_at_min == 1
                && cells
                    .iter()
                    .any(|(_, d, _)| (*d - grid_min).abs() <= frule.clamp_eps);
            // Clamp-check the window's bottom `K` values (or all of them, for
            // a window shorter than `K`, e.g. `Spaced`'s 3-point window at the
            // registered `K = 3`).
            let clamp_upper = k.min(vals.len());
            let clamped = !sole_min_holder
                && (0..clamp_upper).any(|j| (vals[j] - grid_min).abs() <= frule.clamp_eps);
            if ordered && !clamped {
                e.rep += 1;
            } else {
                if !ordered {
                    e.fail_order += 1;
                }
                if clamped {
                    e.fail_clamp += 1;
                }
            }
        }

        // Report-only column (folded in from the dropped A9r). Rule-
        // independent: every row's own truth vs its own dial, pooled.
        let mut rep_hit: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
        if let Some(t) = report_truth {
            for i in 0..n {
                if !dial[i].is_finite() || !t[i].is_finite() || t[i] > report_threshold {
                    continue;
                }
                let e = rep_hit.entry(codec[i].clone()).or_insert((0, 0));
                e.0 += 1;
                if dial[i] <= report_threshold {
                    e.1 += 1;
                }
            }
        }

        order.sort();
        let median = |v: &[f64]| -> f64 {
            if v.is_empty() {
                return f64::NAN;
            }
            let mut w = v.to_vec();
            w.sort_by(f64::total_cmp);
            w[w.len() / 2]
        };
        let codecs = order
            .into_iter()
            .map(|c| {
                let a = &acc[&c];
                let hit = rep_hit.get(&c);
                CodecFloor {
                    codec: c.clone(),
                    n_ladders: a.n,
                    n_represented: a.rep,
                    n_too_short: a.short,
                    represented_frac: if a.n == 0 {
                        f64::NAN
                    } else {
                        a.rep as f64 / a.n as f64
                    },
                    n_fail_order: a.fail_order,
                    n_fail_clamp: a.fail_clamp,
                    dial_min: a.dial_min,
                    bottom_medians: a.steps.iter().map(|v| median(v)).collect(),
                    report_frac_at_floor: hit.and_then(|(nn, hh)| {
                        if *nn == 0 {
                            None
                        } else {
                            Some(*hh as f64 / *nn as f64)
                        }
                    }),
                    report_n_at_floor: report_truth.map(|_| hit.map(|(nn, _)| *nn).unwrap_or(0)),
                }
            })
            .collect();
        Self {
            instrument: instrument.to_string(),
            grid_min,
            n_ladders_at_min,
            codecs,
        }
    }
}

/// `FloorRule::Resolvable` window: walk forward from position 0 (the ladder's
/// lowest setting, always selected), skipping any step whose `|Δ mentor|`
/// from the last SELECTED step's mentor value is below `margin`, until
/// `k + 1` mentor-resolvable steps are collected. `None` ("too short") when
/// fewer than `k + 1` such steps exist before the ladder runs out, or any
/// cell in this ladder carries no mentor truth.
fn resolvable_window(cells: &[LadderCell], k: usize, margin: f64) -> Option<Vec<usize>> {
    if cells.is_empty() {
        return None;
    }
    let mentor: Vec<f64> = cells.iter().map(|c| c.2).collect::<Option<Vec<f64>>>()?;
    let want = k + 1;
    let mut sel: Vec<usize> = vec![0];
    let mut i = 1;
    while sel.len() < want && i < cells.len() {
        let last = *sel.last().expect("sel always holds position 0");
        if (mentor[i] - mentor[last]).abs() >= margin {
            sel.push(i);
        }
        i += 1;
    }
    (sel.len() == want).then_some(sel)
}

/// `FloorRule::Spaced` window: the lowest setting (position 0, always
/// selected), plus the step whose MENTOR value is nearest `mentor[0] +
/// near_lo`, plus the step nearest `mentor[0] + near_hi` — both drawn from
/// the remaining steps, then all three re-sorted by position (equivalent to
/// re-sorting by `q`, since `cells` is already `q`-sorted). `None` ("too
/// short") when the ladder has fewer than 3 distinct settings, or any cell
/// carries no mentor truth. Ties in "nearest" resolve to the first candidate
/// in `q` order, matching Python's `min()`.
fn spaced_window(cells: &[LadderCell], near_lo: f64, near_hi: f64) -> Option<Vec<usize>> {
    if cells.len() < 3 {
        return None;
    }
    let mentor: Vec<f64> = cells.iter().map(|c| c.2).collect::<Option<Vec<f64>>>()?;
    let base = mentor[0];
    let cand: Vec<usize> = (1..cells.len()).collect();
    let nearest = |target: f64, pool: &[usize]| -> usize {
        *pool
            .iter()
            .min_by(|&&a, &&b| {
                (mentor[a] - target)
                    .abs()
                    .total_cmp(&(mentor[b] - target).abs())
            })
            .expect("pool is non-empty, checked by callers")
    };
    let p_lo = nearest(base + near_lo, &cand);
    let cand2: Vec<usize> = cand.into_iter().filter(|&i| i != p_lo).collect();
    let p_hi = nearest(base + near_hi, &cand2);
    let mut triple = [0usize, p_lo, p_hi];
    triple.sort_unstable();
    Some(triple.to_vec())
}

/// Identity behaviour: the dial on `ref == dist` pairs, and whether any real
/// codec output out-scores a perfect copy of its OWN reference.
#[derive(Debug, Clone)]
pub struct IdentityMeasure {
    pub n: usize,
    pub dial_min: f64,
    pub dial_median: f64,
    pub dial_max: f64,
    /// Identity rows falling outside the registered `[identity_lo, identity_hi]`
    /// band.
    pub n_outside_band: usize,
    /// Dial-grid cells scoring ABOVE the identity dial by more than
    /// [`ABOVE_IDENTITY_SLACK`]. Compared against `dial_max` — the most
    /// permissive identity value in the probe — so a nonzero count is
    /// unambiguous.
    pub n_above_identity: usize,
    pub n_grid_cells_compared: usize,
    pub n_grid_cells_total: usize,
    /// Worst offender: `(image_id, codec, q, cell dial, identity dial)`.
    pub worst: Option<(String, String, f64, f64, f64)>,
}

/// **MEASURED 2026-09-04: the identity feature vector is the ZERO vector, for
/// every image.** Extracting 372 features for all 38 dial-grid references
/// against themselves gives 38 byte-identical all-zero rows — which is what a
/// difference metric must do, and it makes the identity dial a SCALAR PROPERTY
/// OF THE BAKE (`dial(0⃗)`), not a per-image measurement. The shipped values:
/// **B 96.2412**, ADD156 (Profile D) 96.1157, v47-QAT (Profile A) 97.6893.
///
/// Two consequences the gate acts on. (1) The registered `[97.5, 100]` identity
/// band is a **v47-era** property: the two shipped LINEAR dials sit ~1.3-1.4
/// points below it. (2) Shipped B emits up to **99.98** on *lossy* dial-grid
/// cells while calling a perfect copy 96.24 — real codec output out-scoring an
/// exact copy, which is a ceiling defect, not a rounding artifact. Both are
/// reported as CONTRACT rows (absolute product bars) and kept strictly apart
/// from the REGRESSION rows (bars = the shipped dial's own reach), so a
/// pre-existing contract failure can never be misread as something a candidate
/// introduced.
pub const IDENTITY_IS_THE_ZERO_VECTOR: &str =
    "ref == dist yields all-zero features for every image; identity dial = dial(0-vector)";

/// A codec output may not out-score a perfect copy by more than this (dial
/// points). Not a tolerance for real inversions — it is the float-noise band
/// the blur-ladder instrument has always used (`0.01`), lifted here so the two
/// agree on what "above identity" means.
pub const ABOVE_IDENTITY_SLACK: f64 = 0.01;

// ─────────────────────────────── verdict ───────────────────────────────

/// Which kind of bar a row carries. Kept separate because they answer
/// different questions and a reader must never confuse them:
/// `Regression` = "is this candidate worse at the ends than the dial users
/// have today"; `Contract` = "does this dial meet the absolute product
/// contract at all" — a bar the SHIPPED dial can itself fail, and does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Bar = the shipped product dial's own measured value on this instrument.
    Regression,
    /// Bar = an absolute product-contract requirement (G3's registered
    /// mono/tied bars; the identity band; "negative values MUST work").
    Contract,
    /// **REPORT-ONLY — no bar, and it gates nothing.** A measurement the user
    /// asked to SEE before deciding what, if anything, it should require. It is
    /// excluded from both tier verdicts by construction, so it can never block
    /// a ship or be mistaken for one that does.
    Report,
}

impl Tier {
    pub fn tag(self) -> &'static str {
        match self {
            Tier::Regression => "regression",
            Tier::Contract => "contract",
            Tier::Report => "report-only",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Pass,
    Fail,
    /// The axis needs an input the caller did not supply, or the instrument is
    /// not in the registry. NEVER counted as a pass.
    NotMeasured,
}

impl State {
    pub fn mark(self) -> &'static str {
        match self {
            State::Pass => "✓",
            State::Fail => "✗",
            State::NotMeasured => "—",
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            State::Pass => "pass",
            State::Fail => "fail",
            State::NotMeasured => "not_measured",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckRow {
    pub id: &'static str,
    pub tier: Tier,
    pub what: &'static str,
    pub measured: Option<f64>,
    pub bar: Option<f64>,
    /// `"≥"`, `"≤"`, `">"` or `"<"` — the direction the measured value must
    /// satisfy.
    pub cmp: &'static str,
    pub state: State,
    /// What the SHIPPED reference dial reads on this same axis. Printed beside
    /// every CONTRACT row so a pre-existing contract failure is never misread
    /// as a regression the candidate introduced.
    pub incumbent: Option<f64>,
    pub note: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Overall {
    /// Every axis in the tier measured and passed.
    Pass,
    /// At least one measured axis in the tier failed.
    Fail,
    /// Nothing failed but at least one axis could not be measured. NOT a pass.
    Incomplete,
    /// The dial grid is not in the registry, so the regression tier has no
    /// bars at all.
    NotMeasurable,
}

impl Overall {
    pub fn label(self) -> &'static str {
        match self {
            Overall::Pass => "PASS",
            Overall::Fail => "FAIL",
            Overall::Incomplete => "INCOMPLETE (not a pass)",
            Overall::NotMeasurable => "NOT MEASURABLE (unregistered dial grid)",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Verdict {
    pub rows: Vec<CheckRow>,
    /// The raw measurements the rows were cut from, so a board row can carry
    /// the NUMBERS (identity dial, tail depth) and not only pass/fail marks.
    pub grid: GridMeasure,
    pub negtail: Option<NegTailMeasure>,
    pub identity: Option<IdentityMeasure>,
    /// "No worse at either end than the shipped dial." THE ship gate for the
    /// 2026-09-04 user rule.
    pub regression: Overall,
    /// "Meets the absolute product contract." Reported alongside, never merged
    /// — the shipped dial itself fails two of these today.
    pub contract: Overall,
    pub grid_label: String,
    pub grid_sha256: String,
    /// The pin set the `bar` column comes from — what a candidate must clear.
    pub reference: String,
    /// The pin set the `incumbent` column comes from — what users have today.
    /// Printed beside every bar, never used as one.
    pub incumbent_reference: String,
    /// Which NEGATIVE-TAIL pin set graded A7*/A8*/A9* — `Product` (the
    /// 2026-09-05 absolute per-codec-family bar, the default) or `Retired`
    /// (the mentor pins every pre-2026-09-05 number was graded on).
    pub tail_pins: TailPins,
    /// Which tier the dial-VALUE rows `A1`-`A6` were emitted on — `Report`
    /// (the default since the 2026-09-05 ruling: measured and printed, gating
    /// nothing) or `Hard` (the pre-ruling grading). Stamped on every rendering
    /// and in the JSON, so a REGRESSION verdict can never be read without
    /// knowing which rows were eligible to fail it.
    pub value_pins: ValuePins,
    /// The per-codec-family tail rows, joined to their registered reference
    /// floors. Printed as its own table — the ruling's "codecs are all
    /// different" is not a footnote, it is the shape of the measurement.
    pub codec_floor_rows: Vec<CodecFloorReport>,
    /// Which [`FloorRule`] windowed `A7r` — `"distinct"` (the pinned rule,
    /// registry-backed bar) or `"resolvable"` / `"spaced"` (report-derived,
    /// LIVE-computed bar). Stamped on every rendering and in the JSON so a
    /// fraction from one rule can never be silently read beside another's.
    pub floor_rule: String,
}

/// One rendered per-codec floor row: the candidate's representability beside
/// the mentor's (the bar) and the incumbent's, plus the information the ruling
/// asked to see at a glance — the dial's min and its bottom-`K` medians next to
/// the reference's.
#[derive(Debug, Clone)]
pub struct CodecFloorReport {
    pub codec: String,
    pub n_ladders: usize,
    pub n_too_short: usize,
    /// Candidate / mentor (the bar) / incumbent representability fractions.
    pub frac: f64,
    pub frac_reference: Option<f64>,
    pub frac_incumbent: Option<f64>,
    pub n_fail_order: usize,
    pub n_fail_clamp: usize,
    pub dial_min: f64,
    pub reference_min: Option<f64>,
    pub bottom_medians: Vec<f64>,
    pub reference_bottom_medians: Option<Vec<f64>>,
    /// REPORT-ONLY, folded in from the dropped `A9r`.
    pub report_frac_at_floor: Option<f64>,
    pub report_n_at_floor: Option<usize>,
    /// `A7r` for this codec alone.
    pub state: State,
    pub note: String,
}

impl Verdict {
    pub fn n_pass(&self) -> usize {
        self.rows.iter().filter(|r| r.state == State::Pass).count()
    }
    pub fn n_fail(&self) -> usize {
        self.rows.iter().filter(|r| r.state == State::Fail).count()
    }
    pub fn n_not_measured(&self) -> usize {
        self.rows
            .iter()
            .filter(|r| r.state == State::NotMeasured)
            .count()
    }
    /// `true` when the candidate may ship on dial-addressability grounds:
    /// both tiers PASS. Anything else — a fail, an unmeasured axis, an
    /// unregistered grid — is NOT SHIPPABLE.
    pub fn shippable(&self) -> bool {
        self.regression == Overall::Pass && self.contract == Overall::Pass
    }
    pub fn headline(&self) -> String {
        if self.shippable() {
            "SHIPPABLE (regression PASS + contract PASS)".to_string()
        } else {
            format!(
                "NOT SHIPPABLE — regression {} / contract {}",
                self.regression.label(),
                self.contract.label()
            )
        }
    }
}

fn state_for(measured: f64, bar: f64, cmp: &str) -> State {
    if !measured.is_finite() || !bar.is_finite() {
        return State::NotMeasured;
    }
    let ok = match cmp {
        "≥" => measured >= bar,
        "≤" => measured <= bar,
        ">" => measured > bar,
        "<" => measured < bar,
        _ => unreachable!("unknown comparator {cmp}"),
    };
    if ok { State::Pass } else { State::Fail }
}

#[allow(clippy::too_many_arguments)]
fn row(
    id: &'static str,
    tier: Tier,
    what: &'static str,
    measured: Option<f64>,
    bar: Option<f64>,
    cmp: &'static str,
    incumbent: Option<f64>,
    note: String,
) -> CheckRow {
    let state = match (measured, bar) {
        (Some(m), Some(b)) => state_for(m, b, cmp),
        _ => State::NotMeasured,
    };
    CheckRow {
        id,
        tier,
        what,
        measured,
        bar,
        cmp,
        state,
        incumbent,
        note,
    }
}

/// A row whose bar is an ABSOLUTE product requirement rather than a reference's
/// measured value, guarded by whether the instrument can discriminate.
///
/// `discriminating == false` means the probe's own reference truth does not
/// reach the bar (or the probe has no truth column at all): a PASS is still a
/// real fact, but a MISS says nothing about the dial, so it is NOT MEASURED —
/// never a fail, and never a silent pass either.
#[allow(clippy::too_many_arguments)]
fn product_row(
    id: &'static str,
    what: &'static str,
    measured: f64,
    bar: f64,
    cmp: &'static str,
    discriminating: bool,
    incumbent: Option<f64>,
    note: String,
) -> CheckRow {
    let state = if !measured.is_finite() {
        State::NotMeasured
    } else {
        match state_for(measured, bar, cmp) {
            State::Pass => State::Pass,
            State::Fail if discriminating => State::Fail,
            State::Fail => State::NotMeasured,
            State::NotMeasured => State::NotMeasured,
        }
    };
    CheckRow {
        id,
        tier: Tier::Regression,
        what,
        measured: if measured.is_finite() {
            Some(measured)
        } else {
            None
        },
        bar: Some(bar),
        cmp,
        state,
        incumbent,
        note,
    }
}

/// Evaluate G-ADDR against the ACTIVE reference ([`ACTIVE_REFERENCE`] —
/// `peer_ssim2` since the 2026-09-04 re-pin). `grid_sha256` is the sha256 of
/// the dial-grid FILE the measurement came from; an unregistered sha yields
/// [`Overall::NotMeasurable`] on the regression tier with every end-of-range
/// row `NotMeasured`.
pub fn evaluate(
    grid_sha256: &str,
    grid_label: &str,
    m: &GridMeasure,
    negtail: Option<(&NegTailMeasure, &str)>,
    identity: Option<(&IdentityMeasure, &str)>,
) -> Verdict {
    evaluate_with_reference(
        ACTIVE_REFERENCE,
        grid_sha256,
        grid_label,
        m,
        negtail,
        identity,
    )
}

/// Evaluate G-ADDR against an explicitly named pin set. `evaluate` is this with
/// [`ACTIVE_REFERENCE`]; the explicit form exists so a test (or a deliberate
/// incumbent-comparison read) can ask "how would this have graded under the
/// retired shipped-B pins?" without the answer depending on which reference
/// happens to be active.
pub fn evaluate_with_reference(
    reference: &str,
    grid_sha256: &str,
    grid_label: &str,
    m: &GridMeasure,
    negtail: Option<(&NegTailMeasure, &str)>,
    identity: Option<(&IdentityMeasure, &str)>,
) -> Verdict {
    evaluate_with_reference_and_pins(
        reference,
        grid_sha256,
        grid_label,
        m,
        negtail,
        identity,
        ValuePins::default(),
    )
}

/// [`evaluate_with_reference`] naming the [`ValuePins`] explicitly. Exists so a
/// caller can ask "how would this have graded when A1-A6 were bars?" without
/// the answer depending on which setting happens to be the default.
pub fn evaluate_with_reference_and_pins(
    reference: &str,
    grid_sha256: &str,
    grid_label: &str,
    m: &GridMeasure,
    negtail: Option<(&NegTailMeasure, &str)>,
    identity: Option<(&IdentityMeasure, &str)>,
    value_pins: ValuePins,
) -> Verdict {
    evaluate_full(
        reference,
        TailPins::default(),
        grid_sha256,
        grid_label,
        m,
        negtail,
        identity,
        None,
        FloorRuleContext::default(),
        value_pins,
    )
}

/// Evaluate G-ADDR against an explicitly named pin set AND an explicitly named
/// NEGATIVE-TAIL pin set.
///
/// The tail selector exists because of the **USER RULING 2026-09-05** —
/// *"i care that the lowest configurable settings per codec are representable,
/// not that negative fifty is in that specifically"* — which retired the
/// mentor-pinned A7/A8/A9 in favour of per-codec FLOOR REPRESENTABILITY
/// (`A7r`) plus a report-only probe row (`A8r`). Every G-ADDR number published
/// before that date is graded on the retired set, so it stays reachable:
/// `TailPins::Retired` reproduces it exactly. A1-A6 are untouched by the
/// ruling and stay mentor-pinned in both arms.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_full(
    reference: &str,
    tail_pins: TailPins,
    grid_sha256: &str,
    grid_label: &str,
    m: &GridMeasure,
    negtail: Option<(&NegTailMeasure, &str)>,
    identity: Option<(&IdentityMeasure, &str)>,
    floors: Option<&FloorMeasure>,
    floor_ctx: FloorRuleContext,
    value_pins: ValuePins,
) -> Verdict {
    let bars = fixed_bars();
    let floor = floor_for_grid(grid_sha256, reference);
    let f = floor.as_ref();
    // The incumbent pin set: printed, never barred against. When the active
    // reference IS the incumbent the two coincide, which is exactly what the
    // pre-2026-09-04 gate did.
    let inc_floor = floor_for_grid(grid_sha256, INCUMBENT_REFERENCE);
    let fi = inc_floor.as_ref();
    let unreg = || "dial grid not in the G-ADDR floor registry".to_string();
    let none_note = |present: bool| {
        if present { String::new() } else { unreg() }
    };
    let has = f.is_some();
    let mut rows: Vec<CheckRow> = vec![
        row(
            "A1",
            value_pins.tier(),
            "ceiling — pooled dial max",
            Some(m.max),
            f.map(|x| x.max),
            "≥",
            fi.map(|x| x.max),
            none_note(has),
        ),
        row(
            "A2",
            value_pins.tier(),
            "floor — pooled dial min",
            Some(m.min),
            f.map(|x| x.min),
            "≤",
            fi.map(|x| x.min),
            none_note(has),
        ),
        row(
            "A3",
            value_pins.tier(),
            "robust ceiling — dial p95",
            Some(m.p95),
            f.map(|x| x.p95),
            "≥",
            fi.map(|x| x.p95),
            none_note(has),
        ),
        row(
            "A4",
            value_pins.tier(),
            "robust floor — dial p5",
            Some(m.p5),
            f.map(|x| x.p5),
            "≤",
            fi.map(|x| x.p5),
            none_note(has),
        ),
        row(
            "A5",
            value_pins.tier(),
            "reach (max − min)",
            Some(m.reach),
            f.map(|x| x.reach),
            "≥",
            fi.map(|x| x.reach),
            none_note(has),
        ),
        row(
            "A6",
            value_pins.tier(),
            "dynamic range (p95 − p5)",
            Some(m.dynamic_range),
            f.map(|x| x.dynamic_range),
            "≥",
            fi.map(|x| x.dynamic_range),
            none_note(has),
        ),
    ];

    // ── negative tail ──
    // Filled by the product arm; empty under the retired pins, which had no
    // per-codec concept at all.
    let mut codec_floor_rows: Vec<CodecFloorReport> = Vec::new();
    //
    // Two pin sets, selected by `tail_pins`. `Product` (the default since the
    // USER RULING 2026-09-05) asks FLOOR REPRESENTABILITY per codec and reports
    // the probe; `Retired` grades A7/A8/A9 against the mentor's own probe
    // depth, which is what every pre-2026-09-05 number was graded on.
    match tail_pins {
        // The RETIRED pins: three probe-derived rows, all mentor-barred.
        TailPins::Retired => match negtail {
            Some((nm, sha)) => {
                let nf = floor_for_negtail(sha, reference);
                let nfi = floor_for_negtail(sha, INCUMBENT_REFERENCE);
                let note = if nf.is_none() {
                    format!(
                        "probe {} not in the G-ADDR floor registry",
                        &sha[..sha.len().min(16)]
                    )
                } else {
                    String::new()
                };
                rows.push(row(
                    "A7",
                    Tier::Regression,
                    "negative tail — probe dial min",
                    Some(nm.min),
                    nf.as_ref().map(|x| x.min),
                    "≤",
                    nfi.as_ref().map(|x| x.min),
                    note.clone(),
                ));
                rows.push(row(
                    "A8",
                    Tier::Regression,
                    "negative tail — probe dial p1",
                    Some(nm.p1),
                    nf.as_ref().map(|x| x.p1),
                    "≤",
                    nfi.as_ref().map(|x| x.p1),
                    note.clone(),
                ));
                rows.push(row(
                    "A9",
                    Tier::Regression,
                    "negative tail — fraction scoring below 0",
                    Some(nm.frac_below_zero),
                    nf.as_ref().map(|x| x.frac_below_zero),
                    "≥",
                    nfi.as_ref().map(|x| x.frac_below_zero),
                    note,
                ));
            }
            None => {
                for (id, what) in [
                    ("A7", "negative tail — probe dial min"),
                    ("A8", "negative tail — probe dial p1"),
                    ("A9", "negative tail — fraction scoring below 0"),
                ] {
                    rows.push(row(
                        id,
                        Tier::Regression,
                        what,
                        None,
                        None,
                        "≤",
                        None,
                        "no --negtail-probe supplied".into(),
                    ));
                }
            }
        },
        // The ACTIVE product bar. A7r and A9r are PER CODEC FAMILY and are
        // measured on whichever instrument carries codec identity — the DIAL
        // GRID, not the probe — so they do NOT depend on a negative-tail probe
        // being supplied. A8r is the one pooled, probe-derived row.
        TailPins::Product => {
            // A7r — FLOOR REPRESENTABILITY, per codec, on the dial grid.
            // `Distinct` reads the registry (byte-identical to pre-2026-09-06
            // behaviour); `Resolvable`/`Spaced` have no registry entry — they
            // are report-derived windows, not pinned — so their bar is
            // ALWAYS the mentor's own LIVE fraction under the same rule.
            //
            // Three cases, in order:
            //
            // 1. The rule needs the mentor's per-cell truth and none was
            //    supplied — the WINDOW itself is uncomputable, so A7r is NOT
            //    MEASURED. Never a silent fall-back to `distinct`'s window:
            //    that would grade a different question under the same row id.
            // 2. A registry row exists for this (grid, reference, RULE) — use
            //    the PINNED bar. This is what "register the mentor's bars"
            //    buys: the number is committed and auditable rather than
            //    re-derived on every run.
            // 3. No registered row — fall back to the mentor's LIVE fraction
            //    under the identical call the candidate went through.
            let (a7r, rows_out) = if floor_ctx.rule.needs_mentor_truth()
                && floor_ctx.mentor.is_none()
            {
                (
                    row(
                        "A7r",
                        Tier::Regression,
                        "floor representability — lowest configurable settings resolve, per codec",
                        None,
                        None,
                        "≤",
                        None,
                        format!(
                            "NOT MEASURED — the operative floor rule `{}` chooses each ladder's \
                             window from the MENTOR's own per-cell values, and no \
                             `--gaddr-grid-truth` was supplied for this instrument. The window \
                             is uncomputable, so nothing is graded; this is NEVER silently \
                             re-graded under `distinct`'s literal window, which asks a \
                             different question. Supply the reference metric's per-cell TSV, \
                             or pass `--floor-rule distinct` to ask the pinned question.",
                            floor_ctx.rule.tag()
                        ),
                    ),
                    Vec::new(),
                )
            } else if !floor_ctx.rule.needs_mentor_truth()
                || floor_repr_for_grid_under(grid_sha256, reference, floor_ctx.rule).is_some()
            {
                // `Distinct` is the PINNED rule: it reads the registry even
                // when no row is registered (yielding NOT MEASURED), because
                // live-computing its bar would let a caller dodge the pins by
                // supplying their own mentor. A mentor-windowed rule uses the
                // registry only when a row for THAT rule exists.
                per_codec_floor_rows(grid_sha256, reference, floors, floor_ctx.rule)
            } else {
                per_codec_floor_rows_live(floor_ctx.rule, floors, floor_ctx.mentor)
            };
            codec_floor_rows = rows_out;
            rows.push(a7r);
            // A8r — the negative-tail probe, REPORT-ONLY. No bar: it carries no
            // codec identity (its rows are KADIS distortion types), and the
            // ruling is about codec settings.
            let a8 = match negtail {
                Some((nm, _)) => CheckRow {
                    id: "A8r",
                    tier: Tier::Report,
                    what: "negative tail — pooled probe min / p1 (REPORT-ONLY, no bar)",
                    measured: if nm.p1.is_finite() { Some(nm.p1) } else { None },
                    bar: None,
                    cmp: "≤",
                    state: State::NotMeasured,
                    incumbent: None,
                    note: format!(
                        "REPORT-ONLY — no bar, gates nothing. Pooled min {:.4} / p1 {:.4} \
                         (probe n={}{}). This instrument carries NO codec identity — its rows \
                         are KADIS distortion types, not codec output — so it cannot answer \
                         the ruling's question (\"the lowest configurable settings per codec\") \
                         and is reported rather than barred. The per-DISTORTION-family reading \
                         is measured in benchmarks/d_peaks_lambda_sweep_2026-09-05.md §4-§6.",
                        nm.min,
                        nm.p1,
                        nm.n,
                        match (nm.truth_min.is_finite(), nm.truth_p1.is_finite()) {
                            (true, true) => format!(
                                ", reference truth min {:.4} / p1 {:.4}",
                                nm.truth_min, nm.truth_p1
                            ),
                            _ => ", probe carries no readable `ssim2_gpu` truth column".to_string(),
                        }
                    ),
                },
                None => CheckRow {
                    id: "A8r",
                    tier: Tier::Report,
                    what: "negative tail — pooled probe min / p1 (REPORT-ONLY, no bar)",
                    measured: None,
                    bar: None,
                    cmp: "≤",
                    state: State::NotMeasured,
                    incumbent: None,
                    note: "REPORT-ONLY — no --negtail-probe supplied".into(),
                },
            };
            rows.push(a8);
        }
    }

    // ── contract tier ──
    rows.push(row(
        "C1",
        Tier::Contract,
        "monotonicity (registered G3 bar)",
        Some(m.mono),
        Some(bars.mono_min),
        "≥",
        fi.map(|x| x.mono),
        String::new(),
    ));
    rows.push(row(
        "C2",
        Tier::Contract,
        "flat/clamp dead-zone (registered G3 bar)",
        Some(m.tied),
        Some(bars.tied_max),
        "≤",
        fi.map(|x| x.tied),
        String::new(),
    ));

    match negtail {
        Some((nm, sha)) => {
            let nf = floor_for_negtail(sha, INCUMBENT_REFERENCE);
            rows.push(row(
                "C3",
                Tier::Contract,
                "negative values WORK — some row of an all-negative-truth probe scores < 0",
                Some(nm.frac_below_zero),
                Some(0.0),
                ">",
                nf.as_ref().map(|x| x.frac_below_zero),
                format!(
                    "probe n={}, every row's reference metric is negative; \
                     CLAUDE.md: \"NEGATIVE zensim values MUST work … do NOT clamp at 0\"",
                    nm.n
                ),
            ));
            rows.push(row(
                "C4",
                Tier::Contract,
                "negative tail — deepest probe dial is below 0",
                Some(nm.min),
                Some(0.0),
                "<",
                nf.as_ref().map(|x| x.min),
                String::new(),
            ));
        }
        None => {
            rows.push(row(
                "C3",
                Tier::Contract,
                "negative values WORK — some row of an all-negative-truth probe scores < 0",
                None,
                None,
                ">",
                None,
                "no --negtail-probe supplied".into(),
            ));
            rows.push(row(
                "C4",
                Tier::Contract,
                "negative tail — deepest probe dial is below 0",
                None,
                None,
                "<",
                None,
                "no --negtail-probe supplied".into(),
            ));
        }
    }

    match identity {
        Some((im, sha)) => {
            let idf = floor_for_identity(sha, INCUMBENT_REFERENCE);
            rows.push(row("C5", Tier::Contract,
                "identity — dial(ref==dist) inside the registered band",
                Some(im.n_outside_band as f64), Some(0.0), "≤",
                idf.as_ref().map(|x| x.dial_max),
                format!(
                    "band [{:.1}, {:.1}]; identity dial min/med/max {:.4} / {:.4} / {:.4} over n={} \
                     ({})",
                    bars.identity_lo, bars.identity_hi,
                    im.dial_min, im.dial_median, im.dial_max, im.n,
                    IDENTITY_IS_THE_ZERO_VECTOR
                )));
            rows.push(row(
                "C6",
                Tier::Contract,
                "identity — no dial-grid cell out-scores a perfect copy",
                Some(im.n_above_identity as f64),
                Some(0.0),
                "≤",
                None,
                match &im.worst {
                    Some((img, codec, q, cell, idl)) => format!(
                        "{} of {} cells above identity; worst `{img}` {codec} q={q:.4} scored \
                         {cell:.4} vs identity {idl:.4}",
                        im.n_above_identity, im.n_grid_cells_total
                    ),
                    None => format!("0 of {} cells above identity", im.n_grid_cells_total),
                },
            ));
        }
        None => {
            rows.push(row(
                "C5",
                Tier::Contract,
                "identity — dial(ref==dist) inside the registered band",
                None,
                None,
                "≤",
                None,
                "no --identity-probe supplied".into(),
            ));
            rows.push(row(
                "C6",
                Tier::Contract,
                "identity — no dial-grid cell out-scores a perfect copy",
                None,
                None,
                "≤",
                None,
                "no --identity-probe supplied".into(),
            ));
        }
    }

    let tier_state = |rows: &[CheckRow], t: Tier| -> Overall {
        let n_fail = rows
            .iter()
            .filter(|r| r.tier == t && r.state == State::Fail)
            .count();
        let n_nm = rows
            .iter()
            .filter(|r| r.tier == t && r.state == State::NotMeasured)
            .count();
        if n_fail > 0 {
            Overall::Fail
        } else if n_nm > 0 {
            Overall::Incomplete
        } else {
            Overall::Pass
        }
    };
    let mut regression = tier_state(&rows, Tier::Regression);
    // NOT MEASURABLE means the INSTRUMENT cannot support the tier's
    // measurement — as distinct from INCOMPLETE, which means it can but an
    // input was not supplied. The two are different facts about a board cell
    // and must not be collapsed. WHICH rows constitute the tier depends on
    // the value pins, so the test does too:
    //
    //  * `Hard`   — A1-A6 are the tier; they are barred from the GRID row.
    //               This is byte-for-byte the pre-2026-09-05 guard.
    //  * `Report` — A7r alone is the tier; its bar comes from the registry
    //               row for THIS rule, or is computed live from the mentor.
    //               Keying on the grid row here would blank a perfectly good
    //               floor result behind an unrelated missing row.
    let regression_unmeasurable = match value_pins {
        ValuePins::Hard => floor.is_none(),
        ValuePins::Report => {
            floor_repr_for_grid_under(grid_sha256, reference, floor_ctx.rule).is_none()
                && floor_ctx.mentor.is_none()
        }
    };
    if regression_unmeasurable {
        regression = Overall::NotMeasurable;
    }
    let contract = tier_state(&rows, Tier::Contract);

    Verdict {
        rows,
        grid: *m,
        negtail: negtail.map(|(x, _)| *x),
        identity: identity.map(|(x, _)| x.clone()),
        regression,
        contract,
        grid_label: floor
            .as_ref()
            .map(|x| x.label.clone())
            .or_else(|| inc_floor.as_ref().map(|x| x.label.clone()))
            .unwrap_or_else(|| grid_label.to_string()),
        grid_sha256: grid_sha256.to_string(),
        reference: floor
            .as_ref()
            .map(|x| format!("{} — {}", reference, x.label))
            .unwrap_or_else(|| format!("{reference} (no registry row)")),
        incumbent_reference: inc_floor
            .as_ref()
            .map(|x| format!("{} — {}", INCUMBENT_REFERENCE, x.label))
            .unwrap_or_else(|| format!("{INCUMBENT_REFERENCE} (no registry row)")),
        tail_pins,
        value_pins,
        codec_floor_rows,
        floor_rule: floor_ctx.rule.tag().to_string(),
    }
}

/// Build the `A7r` FLOOR-REPRESENTABILITY row and the per-codec table printed
/// beside it.
///
/// The bar is the **mentor's own representability fraction on the same cells**,
/// registry-pinned per codec — a measurement of one scorer on one instrument,
/// never an invented threshold, and never a dial value. A codec on which the
/// mentor itself represents nothing has a bar of 0.0, which is the natural
/// analogue of an exemption and needs no special case.
fn per_codec_floor_rows(
    grid_sha256: &str,
    reference: &str,
    floors: Option<&FloorMeasure>,
    frule: FloorRule,
) -> (CheckRow, Vec<CodecFloorReport>) {
    let rule = floor_rule();
    let reg = floor_repr_for_grid_under(grid_sha256, reference, frule);
    let inc = floor_repr_for_grid_under(grid_sha256, INCUMBENT_REFERENCE, frule);
    // A missing MENTOR row leaves the axis NOT MEASURED — but the measurement
    // is still REPORTED, because that is how the mentor's own pins get derived
    // in the first place (run the gate on the peer, read the fractions, append
    // them). A missing ladder MEASUREMENT has nothing to report.
    let fm = match floors {
        Some(m) => m,
        None => {
            return (
                row(
                    "A7r",
                    Tier::Regression,
                    "floor representability — lowest configurable settings resolve, per codec",
                    None,
                    None,
                    "≤",
                    None,
                    "the caller supplied no per-codec ladder measurement (this instrument \
                     carries no codec / quality-ladder columns)"
                        .to_string(),
                ),
                Vec::new(),
            );
        }
    };
    let unregistered_note = reg.is_none().then(|| {
        format!(
            "no per-codec floor-representability row registered for grid {} / reference \
             `{reference}` — the bar is the MENTOR's own fraction on these cells and it has not \
             been measured here, so A7r is NOT MEASURED (never pooled, never defaulted). The \
             fractions below are still REPORTED: appending them for the reference IS how the \
             bar is derived.",
            &grid_sha256[..grid_sha256.len().min(16)]
        )
    });
    let mut table: Vec<CodecFloorReport> = Vec::new();
    for cf in &fm.codecs {
        let bar = reg
            .as_ref()
            .and_then(|g| g.codecs.iter().find(|c| c.codec == cf.codec))
            .map(|c| c.represented_frac);
        let inc_frac = inc
            .as_ref()
            .and_then(|g| g.codecs.iter().find(|c| c.codec == cf.codec))
            .map(|c| c.represented_frac);
        let state = match (bar, cf.represented_frac.is_finite()) {
            (Some(b), true) if cf.represented_frac >= b => State::Pass,
            (Some(_), true) => State::Fail,
            _ => State::NotMeasured,
        };
        let note = match bar {
            None => format!(
                "no registered mentor fraction for `{}` on this grid — NOT MEASURED",
                cf.codec
            ),
            Some(b) => format!(
                "{}/{} ladders represented ({:.4} vs mentor {:.4}); failures: {} unordered, {} on \
                 the clamp{}",
                cf.n_represented,
                cf.n_ladders,
                cf.represented_frac,
                b,
                cf.n_fail_order,
                cf.n_fail_clamp,
                if cf.n_too_short > 0 {
                    format!(
                        "; {} ladder(s) shorter than {} steps, not tested",
                        cf.n_too_short,
                        rule.bottom_k + 1
                    )
                } else {
                    String::new()
                }
            ),
        };
        table.push(CodecFloorReport {
            codec: cf.codec.clone(),
            n_ladders: cf.n_ladders,
            n_too_short: cf.n_too_short,
            frac: cf.represented_frac,
            frac_reference: bar,
            frac_incumbent: inc_frac,
            n_fail_order: cf.n_fail_order,
            n_fail_clamp: cf.n_fail_clamp,
            dial_min: cf.dial_min,
            reference_min: None,
            bottom_medians: cf.bottom_medians.clone(),
            reference_bottom_medians: None,
            report_frac_at_floor: cf.report_frac_at_floor,
            report_n_at_floor: cf.report_n_at_floor,
            state,
            note,
        });
    }
    let n_gradeable = table
        .iter()
        .filter(|r| r.state != State::NotMeasured)
        .count();
    let n_fail = table.iter().filter(|r| r.state == State::Fail).count();
    let a7 = CheckRow {
        id: "A7r",
        tier: Tier::Regression,
        what: "floor representability — lowest configurable settings resolve, per codec",
        measured: if n_gradeable > 0 {
            Some(n_fail as f64)
        } else {
            None
        },
        bar: Some(0.0),
        cmp: "≤",
        state: if n_gradeable == 0 {
            State::NotMeasured
        } else if n_fail == 0 {
            State::Pass
        } else {
            State::Fail
        },
        incumbent: None,
        note: format!(
            "K={} lowest steps per `(image, codec)` ladder on `{}`; a ladder is REPRESENTED when \
             the dial is strictly increasing across those steps AND into the next one up, and no \
             bottom step sits within {:.0e} of the instrument minimum {:.4} ({} ladder(s) attain \
             it{}). Value = number of codecs whose fraction is below the MENTOR's own on the same \
             cells; {} of {} codecs graded.",
            rule.bottom_k,
            fm.instrument,
            rule.clamp_eps,
            fm.grid_min,
            fm.n_ladders_at_min,
            if fm.n_ladders_at_min == 1 {
                " — a sole holder, which is a genuine floor, not a clamp"
            } else {
                " — a COLLAPSED floor: two or more ladders share the bottom value"
            },
            n_gradeable,
            table.len()
        ),
    };
    let a7 = match unregistered_note {
        Some(n) => CheckRow { note: n, ..a7 },
        None => a7,
    };
    (a7, table)
}

/// [`per_codec_floor_rows`]'s counterpart for `Resolvable`/`Spaced`. There is
/// no registry entry for those windows — they are report-derived, not pinned
/// — so the bar is NEVER read from `benchmarks/
/// dial_addressability_floor_2026-09-04.json`'s `distinct` pins. It is
/// instead the MENTOR's own representability, computed by grading the
/// mentor's own per-cell truth through the identical
/// [`FloorMeasure::from_grid_with_rule`] call the candidate went through
/// (`mentor`, supplied by the caller — see [`FloorRuleContext`]).
///
/// Every row and the summary note are stamped `rule=<tag>` so a `resolvable`
/// fraction can never be silently compared against a `distinct` or `spaced`
/// one — the whole point of keeping this a SEPARATE function from
/// `per_codec_floor_rows` rather than a registry-lookup branch inside it.
fn per_codec_floor_rows_live(
    rule: FloorRule,
    floors: Option<&FloorMeasure>,
    mentor: Option<&FloorMeasure>,
) -> (CheckRow, Vec<CodecFloorReport>) {
    let frule = floor_rule();
    let tag = rule.tag();
    let fm = match floors {
        Some(m) => m,
        None => {
            return (
                row(
                    "A7r",
                    Tier::Regression,
                    "floor representability — lowest configurable settings resolve, per codec",
                    None,
                    None,
                    "≤",
                    None,
                    format!(
                        "rule=`{tag}` — the caller supplied no per-codec ladder measurement \
                         (this instrument carries no codec / quality-ladder columns)"
                    ),
                ),
                Vec::new(),
            );
        }
    };
    let no_mentor_note = mentor.is_none().then(|| {
        format!(
            "rule=`{tag}` — no LIVE mentor floor measurement supplied. `resolvable`/`spaced` \
             have no registry entry (they are report-derived windows, never pinned like \
             `distinct`), so their bar REQUIRES scoring the mentor's own per-cell truth \
             (`--gaddr-grid-truth`) through this same rule — without it every codec below is \
             NOT MEASURED, never defaulted to `distinct`'s pins."
        )
    });
    let mut table: Vec<CodecFloorReport> = Vec::new();
    for cf in &fm.codecs {
        let bar = mentor
            .and_then(|m| m.codecs.iter().find(|c| c.codec == cf.codec))
            .map(|c| c.represented_frac)
            .filter(|b| b.is_finite());
        let state = match (bar, cf.represented_frac.is_finite()) {
            (Some(b), true) if cf.represented_frac >= b => State::Pass,
            (Some(_), true) => State::Fail,
            _ => State::NotMeasured,
        };
        let note = match bar {
            None => format!(
                "rule=`{tag}` — no LIVE mentor fraction for `{}` (mentor measurement absent, or \
                 that codec had zero gradeable ladders under this rule) — NOT MEASURED",
                cf.codec
            ),
            Some(b) => format!(
                "rule=`{tag}` — {}/{} ladders represented ({:.4} vs LIVE mentor {:.4}, \
                 owner-computed on THIS instrument under THIS rule — never the `distinct` \
                 registry pins); failures: {} unordered, {} on the clamp{}",
                cf.n_represented,
                cf.n_ladders,
                cf.represented_frac,
                b,
                cf.n_fail_order,
                cf.n_fail_clamp,
                if cf.n_too_short > 0 {
                    format!(
                        "; {} ladder(s) had no valid `{tag}` window (too short, or ran out of \
                         mentor-resolvable steps), not tested",
                        cf.n_too_short
                    )
                } else {
                    String::new()
                }
            ),
        };
        table.push(CodecFloorReport {
            codec: cf.codec.clone(),
            n_ladders: cf.n_ladders,
            n_too_short: cf.n_too_short,
            frac: cf.represented_frac,
            frac_reference: bar,
            // No registered incumbent concept for a report-derived rule.
            frac_incumbent: None,
            n_fail_order: cf.n_fail_order,
            n_fail_clamp: cf.n_fail_clamp,
            dial_min: cf.dial_min,
            reference_min: None,
            bottom_medians: cf.bottom_medians.clone(),
            reference_bottom_medians: None,
            report_frac_at_floor: cf.report_frac_at_floor,
            report_n_at_floor: cf.report_n_at_floor,
            state,
            note,
        });
    }
    let n_gradeable = table
        .iter()
        .filter(|r| r.state != State::NotMeasured)
        .count();
    let n_fail = table.iter().filter(|r| r.state == State::Fail).count();
    let a7 = CheckRow {
        id: "A7r",
        tier: Tier::Regression,
        what: "floor representability — lowest configurable settings resolve, per codec",
        measured: if n_gradeable > 0 {
            Some(n_fail as f64)
        } else {
            None
        },
        bar: Some(0.0),
        cmp: "≤",
        state: if n_gradeable == 0 {
            State::NotMeasured
        } else if n_fail == 0 {
            State::Pass
        } else {
            State::Fail
        },
        incumbent: None,
        note: no_mentor_note.unwrap_or_else(|| {
            format!(
                "rule=`{tag}` (K={} window on `{}`; bar is the LIVE mentor fraction, computed \
                 on this instrument under this rule — never read from the `distinct` registry \
                 pins). {} of {} codecs graded.",
                frule.bottom_k,
                fm.instrument,
                n_gradeable,
                table.len()
            )
        }),
    };
    (a7, table)
}

/// Markdown section for the verdict report.
pub fn render_markdown(v: &Verdict) -> String {
    let mut s = String::new();
    s.push_str("\n## DIAL ADDRESSABILITY gate (G-ADDR — floor + ceiling reach)\n\n");
    s.push_str(&format!(
        "**{}** — {} pass / {} fail / {} not measured.\n\n\
         Instrument: `{}` (sha `{}`).\n\n\
         - **`bar` = vs {} (the bar)** — the REFERENCE METRIC's own end-of-range behaviour on \
         this same instrument. A candidate must address at least the range ssim2 addresses. \
         *(USER DECISION 2026-09-04: \"I don't think we should pin to B, ssim2 seems a better \
         mentor.\" The retired shipped-B pins put A1/A3/A6 ABOVE what the reference metric \
         itself reaches, and A4 was met only via a −23-point low-band bias.)*\n\
         - **`incumbent` = vs {} (incumbent)** — what users have today. Printed for contrast, \
         **never a bar**; the shipped-B pin set is retained in the registry and labelled \
         biased. CONTRACT bars are absolute product requirements the shipped dial can itself \
         fail, so a standing contract failure is never misread as a regression this candidate \
         introduced.\n\
         - **negative-tail pin set: `{}`** — {}\n\
         - **floor-rule: `{}`** — {}\n\
         - **dial-VALUE pins (`A1`-`A6`): `{}`** — {}\n\n",
        v.headline(),
        v.n_pass(),
        v.n_fail(),
        v.n_not_measured(),
        v.grid_label,
        &v.grid_sha256[..v.grid_sha256.len().min(16)],
        v.reference,
        v.incumbent_reference,
        v.tail_pins.tag(),
        match v.tail_pins {
            TailPins::Product => format!(
                "`A7r` is **FLOOR REPRESENTABILITY**, per codec, on the dial grid; `A8r` is \
                 the negative-tail probe, **REPORT-ONLY**. **USER RULING 2026-09-05 \
                 (operative):** *\"i care that the lowest configurable settings per codec are \
                 representable, not that negative fifty is in that specifically.\"* **No dial \
                 value is a bar anywhere in this tier.** For each `(image, codec)` ladder the \
                 K={} lowest configurable steps must be strictly ordered with quality — and \
                 into the next step up — and must not sit on the dial's clamp (the instrument \
                 minimum, unless that ladder is its sole holder; two or more sharing it is a \
                 collapsed floor). The per-codec bar is the **MENTOR's own fraction on the \
                 same cells**, registry-pinned; the incumbent's is printed beside it.",
                floor_rule().bottom_k,
            ),
            TailPins::Retired =>
                "`A7`/`A8`/`A9` are the RETIRED mentor pins (`peer_ssim2`'s own depth on this \
                 probe). Retired 2026-09-05 by user ruling; kept reachable because every \
                 G-ADDR number published before that date is graded on them."
                    .to_string(),
        },
        v.floor_rule,
        match v.floor_rule.as_str() {
            "resolvable" => "**THE OPERATIVE RULE (USER RULING 2026-09-05)** — variant (a) \
                 of `benchmarks/ladder_floor_resolution_2026-09-05.md`: `A7r`'s window skips \
                 forward past any step the MENTOR itself cannot resolve (`|Δ mentor| < \
                 margin`), then tests the next K+1 resolvable steps. This exists because \
                 `distinct`'s literal bottom-3 window graded steps ssim2 cannot tell apart — \
                 on jpeg it graded ELEVEN encoder-identical settings as three. Bar = the \
                 REGISTRY row for this grid AND this rule when one is registered, else the \
                 mentor's LIVE fraction under the identical call; a `distinct` fraction is \
                 NEVER substituted. Reverse with `--floor-rule distinct`."
                .to_string(),
            "spaced" => "**owner-extension, opt-in (2026-09-06)** — variant (b): `A7r`'s \
                 window is the lowest setting plus the steps nearest +2 and +5 mentor points \
                 above it, re-sorted by quality. Bar = the registry row for this rule if one \
                 is registered, else LIVE-computed; never the `distinct` pins."
                .to_string(),
            _ => format!(
                "the pinned rule — literal positions `0..=K` by quality (K={}); bar is the \
                 REGISTRY-pinned mentor fraction, as in every prior G-ADDR report.",
                floor_rule().bottom_k
            ),
        },
        v.value_pins.tag(),
        match v.value_pins {
            ValuePins::Report => "**REPORT-ONLY (USER RULING 2026-09-05)** — the six dial-VALUE                  rows are measured and printed with their bars, but sit on tier `report-only`                  and gate NOTHING. They bar against `peer_ssim2`'s own max/p95/min/p5/reach/                 dynamic_range, which are incidental properties of where the mentor's                  distribution lands on one instrument, not product requirements; the product                  requirements are the CONTRACT tier (`C1`-`C6`) and the per-codec floor                  (`A7r`). The REGRESSION headline above is therefore carried by `A7r` alone.                  Reverse with `--gaddr-value-pins hard`."
                .to_string(),
            ValuePins::Hard => "**pre-ruling grading** — `A1`-`A6` are REGRESSION rows and can                  fail the tier, exactly as every G-ADDR number published before 2026-09-05 was                  graded. Reproduces that grading row-for-row."
                .to_string(),
        },
    ));
    if v.regression == Overall::NotMeasurable {
        s.push_str(
            "> ⚠ This dial grid is **not in the G-ADDR floor registry** \
             (`benchmarks/dial_addressability_floor_2026-09-04.json`). Every regression axis \
             is NOT MEASURED — an unregistered instrument can never produce a pass, because \
             a bar you can dodge by choosing a friendlier grid is not a bar. Register the \
             grid (measure the shipped reference on it and append a row) or re-run on a \
             registered one.\n\n",
        );
    }
    s.push_str(
        "| id | tier | axis | measured | bar | incumbent (shipped B) | pass |\n\
         |---|---|---|--:|---|--:|:--:|\n",
    );
    for r in &v.rows {
        let meas = match r.measured {
            Some(x) if x.is_finite() => format!("{x:.4}"),
            _ => "—".into(),
        };
        let bar = match r.bar {
            Some(b) => format!("{} {:.4}", r.cmp, b),
            None => "—".into(),
        };
        let inc = match r.incumbent {
            Some(x) if x.is_finite() => format!("{x:.4}"),
            _ => "—".into(),
        };
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} |\n",
            r.id,
            r.tier.tag(),
            r.what,
            meas,
            bar,
            inc,
            r.state.mark()
        ));
    }
    s.push('\n');
    // ── the PER-CODEC-FAMILY table ──
    // The ruling's "codecs are all different" is the shape of the measurement,
    // not a footnote, so the families are printed as their own table with the
    // reference floor that decides each one's exemption.
    if !v.codec_floor_rows.is_empty() {
        let k = floor_rule().bottom_k;
        s.push_str(&format!(
            "**Per-codec FLOOR REPRESENTABILITY** (`A7r`; bar = the mentor's own fraction on \
             these cells). `repr` is the fraction of `(image, codec)` ladders whose {k} lowest \
             configurable steps are strictly ordered and off the clamp. Everything right of \
             `A7r` is **information, not bars**.\n\n\
             | codec | ladders | repr (dial) | repr (ssim2 = bar) | repr (incumbent) | A7r | \
             unordered | clamped | dial min | bottom-{k} medians (lowest first) | ref≤floor also dial≤floor |\n\
             |---|--:|--:|--:|--:|:--:|--:|--:|--:|---|--:|\n"
        ));
        for r in &v.codec_floor_rows {
            let f = |x: Option<f64>| match x {
                Some(v) if v.is_finite() => format!("{v:.4}"),
                _ => "—".into(),
            };
            s.push_str(&format!(
                "| {} | {}{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                r.codec,
                r.n_ladders,
                if r.n_too_short > 0 {
                    format!(" (+{} short)", r.n_too_short)
                } else {
                    String::new()
                },
                f(Some(r.frac)),
                f(r.frac_reference),
                f(r.frac_incumbent),
                r.state.mark(),
                r.n_fail_order,
                r.n_fail_clamp,
                f(Some(r.dial_min)),
                r.bottom_medians
                    .iter()
                    .map(|x| format!("{x:.3}"))
                    .collect::<Vec<_>>()
                    .join(" / "),
                match (r.report_frac_at_floor, r.report_n_at_floor) {
                    (Some(fr), Some(n)) => format!("{fr:.4} (n={n})"),
                    (None, Some(n)) => format!("— (n={n})"),
                    _ => "—".into(),
                },
            ));
        }
        s.push('\n');
        for r in &v.codec_floor_rows {
            if !r.note.is_empty() {
                s.push_str(&format!("- **{}**: {}\n", r.codec, r.note));
            }
        }
        s.push('\n');
    }
    for r in &v.rows {
        if !r.note.is_empty() {
            s.push_str(&format!("- **{}**: {}\n", r.id, r.note));
        }
    }
    s.push_str(
        "\n_USER RULE 2026-09-04: **any model that limits dial range cannot ship.** A dial \
         that is more monotone and better calibrated in the middle but compressed at the ends \
         is a worse product dial — a codec loop asked for a near-lossless target can only \
         reach it if the metric still moves there, and a loop asked for an aggressive target \
         can only reach it if the metric still goes down there. Every REGRESSION bar is the \
         REFERENCE METRIC's own measured value on the same instrument, so a pass means \"this \
         dial addresses at least the range ssim2 addresses\"; nothing there is an invented \
         threshold. Re-pinning to ssim2 is not a relaxation — it moved the difficulty from \
         the ceiling to the FLOOR (ssim2 reaches −55.35 on this grid where shipped B stops at \
         +3.13, and its negative-tail probe is 100 % below zero against B's 0 %). `—` is NOT \
         MEASURED and is never counted as a pass._\n\n",
    );
    s
}

/// JSON block for `--full-json` / a board row.
pub fn to_json(v: &Verdict) -> serde_json::Value {
    serde_json::json!({
        "headline": v.headline(),
        "shippable": v.shippable(),
        "regression": v.regression.label(),
        "contract": v.contract.label(),
        "n_pass": v.n_pass(),
        "n_fail": v.n_fail(),
        "n_not_measured": v.n_not_measured(),
        "grid_label": v.grid_label,
        "grid_sha256": v.grid_sha256,
        "reference": v.reference,
        "incumbent_reference": v.incumbent_reference,
        "active_reference": ACTIVE_REFERENCE,
        "tail_pins": v.tail_pins.tag(),
        "tail_pin_set": match v.tail_pins {
            TailPins::Product => active_tail_pin_set(),
            TailPins::Retired => "mentor-2026-09-04".to_string(),
        },
        // Which FloorRule windowed A7r — "distinct" (registry-pinned bar) or
        // "resolvable"/"spaced" (report-derived, LIVE-computed bar). A
        // fraction under one rule is never comparable to another's; this
        // field is what lets a reader of the raw JSON enforce that.
        "floor_rule": v.floor_rule,
        // Which tier A1-A6 were emitted on. A REGRESSION verdict is only
        // readable together with this: under "report" (the default since the
        // 2026-09-05 ruling) the six dial-VALUE rows are measured and printed
        // but cannot fail the tier, which is carried by `A7r` alone.
        "value_pins": v.value_pins.tag(),
        "measured": {
            "grid": {
                "min": v.grid.min, "max": v.grid.max,
                "p5": v.grid.p5, "p95": v.grid.p95,
                "reach": v.grid.reach, "dynamic_range": v.grid.dynamic_range,
                "mono": v.grid.mono, "tied": v.grid.tied, "n": v.grid.n,
            },
            "negtail": v.negtail.as_ref().map(|n| serde_json::json!({
                "n": n.n, "min": n.min, "p1": n.p1, "p5": n.p5,
                "frac_below_zero": n.frac_below_zero,
                // The probe's own reference truth, which drives A8r's
                // reachability guard. `null` where the probe carries no
                // readable `ssim2_gpu` column.
                "truth_min": if n.truth_min.is_finite() { Some(n.truth_min) } else { None },
                "truth_p1": if n.truth_p1.is_finite() { Some(n.truth_p1) } else { None },
            })),
            // The PER-CODEC floor table, reported whether or not it is graded.
            "codec_floor": if v.codec_floor_rows.is_empty() { serde_json::Value::Null } else {
                serde_json::json!(v.codec_floor_rows.iter().map(|r| serde_json::json!({
                    "codec": r.codec,
                    "n_ladders": r.n_ladders,
                    "n_too_short": r.n_too_short,
                    "represented_frac": if r.frac.is_finite() { Some(r.frac) } else { None },
                    "represented_frac_reference": r.frac_reference,
                    "represented_frac_incumbent": r.frac_incumbent,
                    "n_fail_order": r.n_fail_order,
                    "n_fail_clamp": r.n_fail_clamp,
                    "dial_min": if r.dial_min.is_finite() { Some(r.dial_min) } else { None },
                    "bottom_medians": r.bottom_medians,
                    "report_frac_at_floor": r.report_frac_at_floor,
                    "report_n_at_floor": r.report_n_at_floor,
                    "state": r.state.tag(),
                    "note": r.note,
                })).collect::<Vec<_>>())
            },
        },
        "checks": v.rows.iter().map(|r| serde_json::json!({
            "id": r.id,
            "tier": r.tier.tag(),
            "what": r.what,
            "measured": r.measured.filter(|x| x.is_finite()),
            "bar": r.bar,
            "cmp": r.cmp,
            "incumbent": r.incumbent.filter(|x| x.is_finite()),
            "state": r.state.tag(),
            "note": r.note,
        })).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANONICAL_GRID_SHA: &str =
        "6546c43e6d9572dcf0740c6346cd604fd8cd3ff01ee2f7031aca998fd8fec2bd";

    /// The registry must parse and must carry a row for the CANONICAL dial
    /// grid + both probes — otherwise every verdict silently reads NOT
    /// MEASURABLE / NOT MEASURED and the gate is decorative.
    #[test]
    fn registry_parses_and_registers_the_canonical_instruments() {
        let r = registry();
        assert!(
            r.fixed_bars.mono_min > 0.0 && r.fixed_bars.tied_max > 0.0,
            "fixed bars must be populated"
        );
        assert!(
            r.grids.iter().any(
                |g| g.dial_grid_sha256 == CANONICAL_GRID_SHA && g.reference == ACTIVE_REFERENCE
            ),
            "bake_verdict's CANONICAL_DIAL_GRID_SHA256 must have a floor row — otherwise a \
             default-flag verdict cannot be gated at all"
        );
        assert!(
            r.negtail_probes
                .iter()
                .any(|p| p.reference == ACTIVE_REFERENCE),
            "a negative-tail probe row for the ACTIVE reference is required — 'negative values \
             MUST work' is a product contract, and an unregistered probe makes C3/C4 \
             unfalsifiable"
        );
        assert!(
            r.identity_probes
                .iter()
                .any(|p| p.reference == ACTIVE_REFERENCE),
            "an identity probe row for the ACTIVE reference is required"
        );
        // The 2026-09-05 tail pin sets: BOTH must be present. The retired one
        // is what every pre-ruling number was graded on; the active one is what
        // the gate bars against today.
        assert!(
            r.negative_tail_bars
                .pin_sets
                .iter()
                .any(|p| p.id == "mentor-2026-09-04"),
            "the retired mentor tail pin set must stay in the registry (append-only) — it is \
             the grading of every G-ADDR number published before 2026-09-05"
        );
        let active = r
            .negative_tail_bars
            .pin_sets
            .iter()
            .find(|p| p.id == r.negative_tail_bars.active)
            .expect("the active negative-tail pin set must exist");
        let rule = active
            .rule()
            .expect("the active tail pin set must carry the floor rule (all-or-nothing)");
        assert!(
            rule.bottom_k >= 1,
            "K must name at least one lowest setting"
        );
        assert!(rule.clamp_eps > 0.0 && rule.clamp_eps < 1e-3);
        // Every registered per-codec mentor row must be cut at the SAME K and
        // must carry real fractions — a bar cut at a different K would be a
        // different measurement wearing the same name.
        for g in &r.grid_floor_representability {
            assert_eq!(g.dial_grid_sha256.len(), 64);
            assert_eq!(
                g.bottom_k, rule.bottom_k,
                "{}: a floor row cut at K={} cannot bar a K={} rule",
                g.label, g.bottom_k, rule.bottom_k
            );
            assert!(!g.codecs.is_empty(), "{}: no codecs", g.label);
            for c in &g.codecs {
                assert!(
                    (0.0..=1.0).contains(&c.represented_frac),
                    "{} / {}: representability is a fraction",
                    g.label,
                    c.codec
                );
                assert!(c.n_ladders > 0, "{} / {}", g.label, c.codec);
            }
        }
        // The retired shipped-B pin set must SURVIVE the re-pin. It is what the
        // `incumbent` column prints and what every pre-2026-09-04 verdict was
        // graded on; dropping it would make those numbers unreadable.
        for (what, present) in [
            (
                "grid",
                r.grids.iter().any(|g| {
                    g.reference == REFERENCE_SHIPPED_B && g.dial_grid_sha256 == CANONICAL_GRID_SHA
                }),
            ),
            (
                "negtail",
                r.negtail_probes
                    .iter()
                    .any(|p| p.reference == REFERENCE_SHIPPED_B),
            ),
            (
                "identity",
                r.identity_probes
                    .iter()
                    .any(|p| p.reference == REFERENCE_SHIPPED_B),
            ),
        ] {
            assert!(
                present,
                "the retired shipped-B {what} row must stay in the registry (append-only; it is \
                 the `incumbent` column and the grading of every pre-re-pin verdict)"
            );
        }
        for g in &r.grids {
            assert_eq!(
                g.dial_grid_sha256.len(),
                64,
                "grid sha must be a full sha256"
            );
            assert!(
                (g.reach - (g.max - g.min)).abs() < 1e-6,
                "{}: reach must equal max − min",
                g.label
            );
            assert!(
                (g.dynamic_range - (g.p95 - g.p5)).abs() < 1e-6,
                "{}: dynamic_range must equal p95 − p5",
                g.label
            );
        }
        for p in &r.negtail_probes {
            assert_eq!(p.probe_sha256.len(), 64);
            assert!(p.min <= p.p1, "{}: min must be at or below p1", p.label);
        }
        for p in &r.identity_probes {
            assert_eq!(p.probe_sha256.len(), 64);
            assert!(p.dial_min <= p.dial_max, "{}: min ≤ max", p.label);
        }
    }

    fn canonical() -> GridFloor {
        floor_for_grid(CANONICAL_GRID_SHA, ACTIVE_REFERENCE).expect("canonical grid registered")
    }

    /// The retired shipped-B pin set for the same grid — kept readable so a
    /// test can assert the two pin sets genuinely disagree.
    fn canonical_b() -> GridFloor {
        floor_for_grid(CANONICAL_GRID_SHA, REFERENCE_SHIPPED_B).expect("shipped-B grid row kept")
    }

    fn tie(f: &GridFloor) -> GridMeasure {
        GridMeasure {
            min: f.min,
            max: f.max,
            p5: f.p5,
            p95: f.p95,
            reach: f.reach,
            dynamic_range: f.dynamic_range,
            mono: f.mono,
            tied: f.tied,
            n: 4424,
        }
    }

    /// Build a `NegTailMeasure` fixture. `truth_*` are the probe's OWN
    /// reference-truth extremes, printed by the report-only `A8r` row.
    fn nt(
        n: usize,
        min: f64,
        p1: f64,
        p5: f64,
        frac_below_zero: f64,
        truth_min: f64,
        truth_p1: f64,
    ) -> NegTailMeasure {
        NegTailMeasure {
            n,
            min,
            p1,
            p5,
            frac_below_zero,
            truth_min,
            truth_p1,
        }
    }

    /// The registered 372 kadis probe's OWN truth, measured 2026-09-05.
    const P372_TRUTH_MIN: f64 = -770.6197;
    const P372_TRUTH_P1: f64 = -187.1314;

    fn nt_from(f: &NegTailFloor) -> NegTailMeasure {
        nt(
            f.n_rows,
            f.min,
            f.p1,
            f.p5,
            f.frac_below_zero,
            P372_TRUTH_MIN,
            P372_TRUTH_P1,
        )
    }

    /// A synthetic grid: `n_lad` ladders per codec, `steps` q-steps each, whose
    /// dial values come from `f(codec, ladder_idx, step_idx)`.
    fn synth_grid(
        codecs: &[&str],
        n_lad: usize,
        steps: usize,
        f: impl Fn(&str, usize, usize) -> f64,
    ) -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>) {
        let (mut img, mut cod, mut q, mut d) = (vec![], vec![], vec![], vec![]);
        for c in codecs {
            for l in 0..n_lad {
                for j in 0..steps {
                    img.push(format!("img{l}"));
                    cod.push((*c).to_string());
                    q.push(j as f64 * 5.0);
                    d.push(f(c, l, j));
                }
            }
        }
        (img, cod, q, d)
    }

    /// A `FloorMeasure` over a synthetic grid.
    fn fm_of(
        codecs: &[&str],
        n_lad: usize,
        steps: usize,
        f: impl Fn(&str, usize, usize) -> f64,
    ) -> FloorMeasure {
        let (img, cod, q, d) = synth_grid(codecs, n_lad, steps, f);
        FloorMeasure::from_grid("test-grid", &img, &cod, &q, &d, None, -50.0)
    }

    /// A clean grid: every ladder strictly increasing, well off any clamp.
    fn fm_clean() -> FloorMeasure {
        // Each codec gets its own offset so exactly ONE ladder in the whole
        // instrument holds the minimum. Without that the collapsed-floor rule
        // fires — correctly — and the fixture would not be "clean".
        fm_of(&["avif", "jpeg", "jxl", "webp"], 8, 6, |c, l, j| {
            let off = match c {
                "avif" => 0.0,
                "jpeg" => 0.5,
                "jxl" => 1.5,
                _ => 2.5,
            };
            10.0 + off + l as f64 * 3.0 + j as f64 * 7.0
        })
    }

    /// `evaluate_full` on the canonical grid with a floor measure.
    fn ev(
        tp: TailPins,
        f: &GridFloor,
        nm: Option<(&NegTailMeasure, &str)>,
        fl: Option<&FloorMeasure>,
    ) -> Verdict {
        evaluate_full(
            ACTIVE_REFERENCE,
            tp,
            &f.dial_grid_sha256,
            &f.label,
            &tie(f),
            nm,
            None,
            fl,
            FloorRuleContext::default(),
            ValuePins::default(),
        )
    }

    /// `ev`, but with an explicit [`FloorRuleContext`] — for tests that grade
    /// `A7r` under `Resolvable`/`Spaced` rather than the pinned `Distinct`.
    fn ev_rule(
        tp: TailPins,
        f: &GridFloor,
        nm: Option<(&NegTailMeasure, &str)>,
        fl: Option<&FloorMeasure>,
        ctx: FloorRuleContext,
    ) -> Verdict {
        evaluate_full(
            ACTIVE_REFERENCE,
            tp,
            &f.dial_grid_sha256,
            &f.label,
            &tie(f),
            nm,
            None,
            fl,
            ctx,
            ValuePins::default(),
        )
    }

    fn probes() -> (NegTailFloor, IdentityFloor) {
        let r = registry();
        (
            r.negtail_probes
                .into_iter()
                .find(|p| p.reference == ACTIVE_REFERENCE)
                .expect("active-reference negtail row"),
            r.identity_probes
                .into_iter()
                .find(|p| p.reference == ACTIVE_REFERENCE)
                .expect("active-reference identity row"),
        )
    }

    fn probes_b() -> (NegTailFloor, IdentityFloor) {
        let r = registry();
        (
            r.negtail_probes
                .into_iter()
                .find(|p| p.reference == REFERENCE_SHIPPED_B)
                .expect("shipped-B negtail row kept"),
            r.identity_probes
                .into_iter()
                .find(|p| p.reference == REFERENCE_SHIPPED_B)
                .expect("shipped-B identity row kept"),
        )
    }

    fn row_by_id<'a>(v: &'a Verdict, id: &str) -> &'a CheckRow {
        v.rows.iter().find(|r| r.id == id).expect("row present")
    }

    /// Re-measuring the ACTIVE REFERENCE itself must pass every REGRESSION axis
    /// — the bars ARE its values, and `≥` / `≤` are inclusive so a tie passes.
    #[test]
    fn the_active_reference_ties_its_own_regression_bars() {
        let f = canonical();
        let (nf, _) = probes();
        let nm = nt_from(&nf);
        let fm = fm_clean();
        for tp in [TailPins::Product, TailPins::Retired] {
            let v = ev(tp, &f, Some((&nm, &nf.probe_sha256)), Some(&fm));
            for r in v.rows.iter().filter(|r| r.tier == Tier::Regression) {
                assert_eq!(
                    r.state,
                    State::Pass,
                    "[{}] {} ({}) must pass when the candidate ties the reference: {:?} vs {:?}",
                    tp.tag(),
                    r.id,
                    r.what,
                    r.measured,
                    r.bar
                );
            }
            assert_eq!(v.regression, Overall::Pass, "[{}]", tp.tag());
        }
    }

    /// Each end-of-range axis must fail INDEPENDENTLY. A gate that only trips
    /// when several axes move at once is not a gate.
    #[test]
    fn each_end_axis_fails_on_its_own() {
        let f = canonical();
        let base = tie(&f);
        let eps = 1e-3;

        let mut c = base;
        c.max -= eps;
        c.reach = c.max - c.min;
        let v = evaluate(&f.dial_grid_sha256, &f.label, &c, None, None);
        assert_eq!(row_by_id(&v, "A1").state, State::Fail, "A1 on a lower max");
        assert_eq!(
            row_by_id(&v, "A5").state,
            State::Fail,
            "A5 on a lower reach"
        );

        let mut c = base;
        c.min += eps;
        c.reach = c.max - c.min;
        let v = evaluate(&f.dial_grid_sha256, &f.label, &c, None, None);
        assert_eq!(row_by_id(&v, "A2").state, State::Fail, "A2 on a higher min");

        let mut c = base;
        c.p95 -= eps;
        c.dynamic_range = c.p95 - c.p5;
        let v = evaluate(&f.dial_grid_sha256, &f.label, &c, None, None);
        assert_eq!(row_by_id(&v, "A3").state, State::Fail, "A3 on a lower p95");
        assert_eq!(row_by_id(&v, "A6").state, State::Fail, "A6 on a lower DR");

        // p5 up — the era-correction failure mode this gate exists to catch.
        let mut c = base;
        c.p5 += eps;
        c.dynamic_range = c.p95 - c.p5;
        let v = evaluate(&f.dial_grid_sha256, &f.label, &c, None, None);
        assert_eq!(row_by_id(&v, "A4").state, State::Fail, "A4 on a higher p5");
        // TIER, both settings. Since the 2026-09-05 ruling A1-A6 are REPORT
        // rows: every per-row FAIL above still stands (the values and their
        // bars are unchanged — that is the whole "still printed" claim), but
        // they no longer fail the REGRESSION tier. Under `--gaddr-value-pins
        // hard` they do, which is the reversibility lever. Asserting both is
        // what keeps this test honest about which axis discriminates and which
        // tier it lands in.
        let vh = evaluate_with_reference_and_pins(
            ACTIVE_REFERENCE,
            &f.dial_grid_sha256,
            &f.label,
            &c,
            None,
            None,
            ValuePins::Hard,
        );
        for id in ["A4", "A6"] {
            assert_eq!(
                row_by_id(&vh, id).state,
                row_by_id(&v, id).state,
                "{id}: the row state must not depend on the tier"
            );
        }
        assert_eq!(vh.regression, Overall::Fail, "A4/A6 are bars under `hard`");
        assert!(!vh.shippable());
        assert_ne!(
            v.regression,
            Overall::Fail,
            "under the operative `report` pins a dial-VALUE row cannot fail the tier"
        );
    }

    /// A shallower negative tail is a floor regression even when the dial grid
    /// is untouched — the grid's worst cells are not the worst inputs a user
    /// will hand the metric.
    /// Under the RETIRED mentor pins, a tail 1e-3 shallower than the
    /// reference's fails on its own — the grid untouched. Scoped to `Retired`
    /// since the 2026-09-05 ruling: under the product range a 1e-3 move at
    /// −770 is correctly irrelevant.
    #[test]
    fn a_shallower_negative_tail_fails_on_its_own_under_the_retired_pins() {
        let f = canonical();
        let (nf, _) = probes();
        let mut nm = nt_from(&nf);
        nm.min += 1e-3;
        nm.p1 += 1e-3;
        let v = ev(TailPins::Retired, &f, Some((&nm, &nf.probe_sha256)), None);
        assert_eq!(row_by_id(&v, "A7").state, State::Fail);
        assert_eq!(row_by_id(&v, "A8").state, State::Fail);
        assert_eq!(v.regression, Overall::Fail);

        // …and under the ACTIVE rule those same numbers bar NOTHING: the probe
        // is report-only, and the floor question is asked of the CODEC LADDERS
        // instead. That is the whole re-spec in one assertion.
        let vp = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm_clean()),
        );
        assert_eq!(row_by_id(&vp, "A8r").tier, Tier::Report);
        assert_eq!(row_by_id(&vp, "A8r").bar, None);
        assert_eq!(row_by_id(&vp, "A7r").state, State::Pass, "clean ladders");
        assert_ne!(vp.regression, Overall::Fail);
    }

    /// An unregistered grid can never pass, no matter how good the numbers.
    #[test]
    fn unregistered_grid_is_not_measurable_never_pass() {
        let great = GridMeasure {
            min: -50.0,
            max: 100.0,
            p5: -20.0,
            p95: 99.9,
            reach: 150.0,
            dynamic_range: 119.9,
            mono: 1.0,
            tied: 0.0,
            n: 10,
        };
        let v = evaluate(&"0".repeat(64), "some other grid", &great, None, None);
        assert_eq!(v.regression, Overall::NotMeasurable);
        assert!(!v.shippable());
        for id in ["A1", "A2", "A3", "A4", "A5", "A6"] {
            assert_eq!(
                row_by_id(&v, id).state,
                State::NotMeasured,
                "{id} must be NOT MEASURED without a registry row"
            );
        }
    }

    /// Absent probes are NOT MEASURED, never a silent pass, and they block a
    /// ship on BOTH tiers.
    #[test]
    fn absent_probes_are_not_measured_and_block_a_ship() {
        let f = canonical();
        let v = evaluate(&f.dial_grid_sha256, &f.label, &tie(&f), None, None);
        let nm: Vec<&str> = v
            .rows
            .iter()
            .filter(|r| r.state == State::NotMeasured)
            .map(|r| r.id)
            .collect();
        // A7r has no ladder measurement, A8r is report-only with no probe.
        assert_eq!(nm, vec!["A7r", "A8r", "C3", "C4", "C5", "C6"]);
        assert_eq!(v.regression, Overall::Incomplete);
        assert_eq!(v.contract, Overall::Incomplete);
        assert!(!v.shippable());
    }

    /// The CONTRACT tier is absolute and independent of any reference: replaying
    /// the SHIPPED dial's own measured tail fails C3/C4 — the tail never goes
    /// below zero on an all-negative-truth probe — no matter which pin set the
    /// regression tier is barring against.
    #[test]
    fn contract_tier_is_absolute_not_relative() {
        let f = canonical();
        let (nfb, idfb) = probes_b();
        let (nf, _) = probes();
        let nm = nt_from(&nfb);
        let im = IdentityMeasure {
            n: idfb.n_rows,
            dial_min: idfb.dial_min,
            dial_median: idfb.dial_median,
            dial_max: idfb.dial_max,
            n_outside_band: 0,
            n_above_identity: 0,
            n_grid_cells_compared: 4424,
            n_grid_cells_total: 4424,
            worst: None,
        };
        let v = evaluate(
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            Some((&im, &idfb.probe_sha256)),
        );
        // C3/C4 are decided by the MEASUREMENT, never by the reference.
        assert_eq!(
            nfb.frac_below_zero, 0.0,
            "the shipped dial's own registered tail never goes below zero — if this changes, \
             the fixture below is no longer testing what it claims"
        );
        assert_eq!(row_by_id(&v, "C3").state, State::Fail);
        assert_eq!(row_by_id(&v, "C4").state, State::Fail);
        assert_eq!(v.contract, Overall::Fail);
        assert_eq!(row_by_id(&v, "C3").tier, Tier::Contract);
    }

    // ───────────────── the 2026-09-04 re-pin: ssim2, not B ─────────────────

    fn measure_of(f: &GridFloor) -> GridMeasure {
        tie(f)
    }

    /// **The headline behavioural change.** The two pin sets genuinely
    /// disagree, in BOTH directions, and each fixture is one shipped scorer's
    /// real measured values — not a synthetic perturbation.
    ///
    /// * A candidate reading exactly **ssim2's** values clears every A row
    ///   under the active pins and **fails A1/A3/A6 under the retired B pins**,
    ///   because those three bars sat ABOVE what the reference metric itself
    ///   reaches (99.98 / 99.72 / 86.08 against 98.38 / 95.46 / 85.20).
    /// * A candidate reading exactly **shipped B's** values is the mirror
    ///   image: it clears the retired pins by construction and **fails A2/A5
    ///   under ssim2**, because B's floor stops at +3.13 where ssim2 reaches
    ///   −55.35.
    ///
    /// If this test ever passes trivially (both arms agreeing), the re-pin has
    /// been undone.
    #[test]
    fn ssim2_bars_and_shipped_b_bars_genuinely_disagree() {
        let ssim2 = canonical();
        let b = canonical_b();
        assert_eq!(ssim2.reference, REFERENCE_PEER_SSIM2);
        assert_eq!(b.reference, REFERENCE_SHIPPED_B);

        // ssim2's own values: PASS under the active (ssim2) pins …
        let v = evaluate(
            &ssim2.dial_grid_sha256,
            &ssim2.label,
            &measure_of(&ssim2),
            None,
            None,
        );
        for id in ["A1", "A2", "A3", "A4", "A5", "A6"] {
            assert_eq!(
                row_by_id(&v, id).state,
                State::Pass,
                "{id}: the mentor must clear its own bar"
            );
        }
        // … and FAIL the retired shipped-B pins on the three ceiling/spread
        // axes that were measured to sit above the truth.
        let vb = evaluate_with_reference(
            REFERENCE_SHIPPED_B,
            &ssim2.dial_grid_sha256,
            &ssim2.label,
            &measure_of(&ssim2),
            None,
            None,
        );
        for id in ["A1", "A3", "A6"] {
            assert_eq!(
                row_by_id(&vb, id).state,
                State::Fail,
                "{id}: a dial calibrated exactly to the truth must FAIL the retired B bar — \
                 that defect is the whole reason for the re-pin"
            );
        }

        // Mirror image: shipped B's own values clear the retired pins …
        let vbb = evaluate_with_reference(
            REFERENCE_SHIPPED_B,
            &b.dial_grid_sha256,
            &b.label,
            &measure_of(&b),
            None,
            None,
        );
        // (Only A1-A6 are supplied here — no probes — so assert the ROWS, not
        // the tier state, which is legitimately INCOMPLETE without a tail.)
        for id in ["A1", "A2", "A3", "A4", "A5", "A6"] {
            assert_eq!(
                row_by_id(&vbb, id).state,
                State::Pass,
                "{id}: the incumbent ties its own retired pin"
            );
        }
        assert_eq!(vbb.regression, Overall::Incomplete, "no probes supplied");
        // … and fail the mentor's FLOOR axes.
        let vbs = evaluate(&b.dial_grid_sha256, &b.label, &measure_of(&b), None, None);
        for id in ["A2", "A5"] {
            assert_eq!(
                row_by_id(&vbs, id).state,
                State::Fail,
                "{id}: the incumbent's floor is far short of the mentor's"
            );
        }
        // The TIER verdict needs `hard`: since the 2026-09-05 ruling A1-A6 are
        // REPORT rows, so the row-level FAILs above (which are the substance
        // of this test) no longer drive the regression tier by themselves.
        let vbs = evaluate_with_reference_and_pins(
            ACTIVE_REFERENCE,
            &b.dial_grid_sha256,
            &b.label,
            &measure_of(&b),
            None,
            None,
            ValuePins::Hard,
        );
        assert_eq!(
            vbs.regression,
            Overall::Fail,
            "the shipped dial must no longer set — nor clear — the regression bars (a FAIL here \
             outranks the missing-probe INCOMPLETE, which is the point)"
        );
    }

    /// A4 specifically: the retired bar rewarded shipped B's −23-point low-band
    /// bias. A candidate whose `p5` sits BETWEEN the mentor's and the
    /// incumbent's passes the old bar and fails the new one; Profile D's real
    /// measured `p5` (9.52) passes both.
    #[test]
    fn a4_stops_rewarding_the_low_band_bias() {
        let ssim2 = canonical();
        let b = canonical_b();
        assert!(
            ssim2.p5 < b.p5,
            "fixture premise: the mentor's p5 ({}) must be below the incumbent's ({})",
            ssim2.p5,
            b.p5
        );
        let mid = 0.5 * (ssim2.p5 + b.p5); // ≈ 11.95, between 10.26 and 13.65
        let mut c = measure_of(&ssim2);
        c.p5 = mid;
        c.dynamic_range = c.p95 - c.p5;
        assert_eq!(
            row_by_id(
                &evaluate_with_reference(
                    REFERENCE_SHIPPED_B,
                    &b.dial_grid_sha256,
                    &b.label,
                    &c,
                    None,
                    None
                ),
                "A4"
            )
            .state,
            State::Pass,
            "the retired bar accepts a p5 well above the truth's"
        );
        assert_eq!(
            row_by_id(
                &evaluate(&ssim2.dial_grid_sha256, &ssim2.label, &c, None, None),
                "A4"
            )
            .state,
            State::Fail,
            "the mentor's bar does not"
        );

        // Profile D, measured 2026-09-04 on this same grid.
        let mut d = measure_of(&ssim2);
        d.p5 = 9.52;
        d.dynamic_range = d.p95 - d.p5;
        assert_eq!(
            row_by_id(
                &evaluate(&ssim2.dial_grid_sha256, &ssim2.label, &d, None, None),
                "A4"
            )
            .state,
            State::Pass,
            "A4 is reachable — Profile D reaches it — so it stays a bar, just a truthful one"
        );
    }

    /// The negative-tail pins are reference-scoped too, and the two disagree by
    /// the entire range of the axis: ssim2 is below zero on 100 % of an
    /// all-negative-truth probe, the shipped dial on 0 %.
    #[test]
    fn retired_negtail_pins_are_reference_scoped() {
        let f = canonical();
        let (nf, _) = probes();
        let (nfb, _) = probes_b();
        assert_eq!(nf.frac_below_zero, 1.0);
        assert_eq!(nfb.frac_below_zero, 0.0);
        assert!(nf.min < nfb.min);

        // The shipped dial's own tail, barred against the mentor: A7/A8/A9 fail.
        let nm = nt_from(&nfb);
        let v = ev(TailPins::Retired, &f, Some((&nm, &nf.probe_sha256)), None);
        for id in ["A7", "A8", "A9"] {
            assert_eq!(row_by_id(&v, id).state, State::Fail, "{id}");
        }
        // Same numbers under the shipped-B pin set: all three pass (they ARE
        // the pin).
        let vb = evaluate_full(
            REFERENCE_SHIPPED_B,
            TailPins::Retired,
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            None,
            None,
            FloorRuleContext::default(),
            ValuePins::default(),
        );
        for id in ["A7", "A8", "A9"] {
            assert_eq!(row_by_id(&vb, id).state, State::Pass, "{id}");
        }
    }

    /// The report must show BOTH: `bar` = the mentor, `incumbent` = what
    /// shipped. They are different numbers on every end-of-range axis, and a
    /// reader who cannot tell them apart cannot tell "worse than the mentor"
    /// from "worse than what shipped".
    #[test]
    fn bar_is_the_mentor_and_incumbent_is_the_shipped_dial() {
        let ssim2 = canonical();
        let b = canonical_b();
        let v = evaluate(
            &ssim2.dial_grid_sha256,
            &ssim2.label,
            &measure_of(&ssim2),
            None,
            None,
        );
        for (id, mentor, incumbent) in [
            ("A1", ssim2.max, b.max),
            ("A2", ssim2.min, b.min),
            ("A3", ssim2.p95, b.p95),
            ("A4", ssim2.p5, b.p5),
            ("A5", ssim2.reach, b.reach),
            ("A6", ssim2.dynamic_range, b.dynamic_range),
        ] {
            let r = row_by_id(&v, id);
            assert_eq!(r.bar, Some(mentor), "{id}: bar must be the mentor's value");
            assert_eq!(
                r.incumbent,
                Some(incumbent),
                "{id}: incumbent must be the shipped dial's value"
            );
            assert_ne!(
                r.bar, r.incumbent,
                "{id}: the two pin sets must genuinely differ, else the re-pin was a no-op"
            );
        }
        assert!(v.reference.contains(REFERENCE_PEER_SSIM2));
        assert!(v.incumbent_reference.contains(REFERENCE_SHIPPED_B));
        let md = render_markdown(&v);
        assert!(
            md.contains("`bar` = vs peer_ssim2"),
            "the report must name the mentor as the bar for A1-A6"
        );
        assert!(
            md.contains("incumbent (shipped B)"),
            "the table header must name the incumbent"
        );
    }

    /// **MEASURED 2026-09-04: the mentor passes the entire CONTRACT tier.**
    /// This is the answer to "what does 'as good as the mentor' mean at the
    /// ends" — every absolute product bar, including the four the shipped dial
    /// fails. Pinned as a test so a later registry edit cannot quietly move it.
    #[test]
    fn the_mentor_passes_the_whole_contract_tier() {
        let f = canonical();
        let (nf, idf) = probes();
        assert!(f.mono >= fixed_bars().mono_min, "C1: mono {}", f.mono);
        assert!(f.tied <= fixed_bars().tied_max, "C2: tied {}", f.tied);
        assert!(nf.frac_below_zero > 0.0, "C3");
        assert!(nf.min < 0.0, "C4");
        let bars = fixed_bars();
        assert!(
            idf.dial_min >= bars.identity_lo && idf.dial_max <= bars.identity_hi,
            "C5: identity {}..{} outside [{}, {}]",
            idf.dial_min,
            idf.dial_max,
            bars.identity_lo,
            bars.identity_hi
        );
        // C6 follows from the two registry rows without any extra measurement:
        // the mentor's worst grid cell is below its identity value.
        assert!(
            f.max <= idf.dial_max,
            "C6: grid max {} must not exceed identity {}",
            f.max,
            idf.dial_max
        );
    }

    /// A dial-grid cell out-scoring a perfect copy fails C6 on its own.
    #[test]
    fn a_cell_above_identity_fails_c6() {
        let f = canonical();
        let (_, idf) = probes();
        let im = IdentityMeasure {
            n: idf.n_rows,
            dial_min: idf.dial_min,
            dial_median: idf.dial_median,
            dial_max: idf.dial_max,
            n_outside_band: 0,
            n_above_identity: 7,
            n_grid_cells_compared: 4424,
            n_grid_cells_total: 4424,
            worst: Some(("img".into(), "avif".into(), 100.0, 99.9, 96.2)),
        };
        let v = evaluate(
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            None,
            Some((&im, &idf.probe_sha256)),
        );
        assert_eq!(row_by_id(&v, "C6").state, State::Fail);
        assert_eq!(v.contract, Overall::Fail);
    }

    /// Percentile convention must match `numpy.percentile` (linear), which is
    /// what every other panel in this crate and the Python owners use.
    #[test]
    fn percentiles_match_numpy_linear() {
        let v: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        assert!((pct(&v, 5.0) - 5.0).abs() < 1e-12);
        assert!((pct(&v, 95.0) - 95.0).abs() < 1e-12);
        let w = vec![0.0, 1.0, 2.0, 3.0];
        // rank = 0.05*3 = 0.15 → 0 + 0.15*(1-0) = 0.15
        assert!((pct(&w, 5.0) - 0.15).abs() < 1e-12);
    }

    #[test]
    fn measures_are_computed_from_the_pool_not_assumed() {
        let scores = vec![10.0, -5.0, 90.0, f64::NAN, 50.0];
        let g = GridMeasure::from_pooled(&scores, 0.95, 0.0);
        assert_eq!(g.n, 4, "non-finite scores are dropped, not counted");
        assert_eq!(g.min, -5.0);
        assert_eq!(g.max, 90.0);
        assert!((g.reach - 95.0).abs() < 1e-12);
        let nt = NegTailMeasure::from_scores(&[-3.0, 1.0, -7.0, 0.0]);
        assert_eq!(nt.min, -7.0);
        assert!(
            (nt.frac_below_zero - 0.5).abs() < 1e-12,
            "0.0 is not below 0"
        );
    }

    /// Every registry bar must round-trip BIT-EXACTLY, or the reference bake
    /// fails its own bar by one ULP. `serde_json`'s default float parser is
    /// not correctly rounded and did exactly that (`99.98330778475787` came
    /// back as `…788`); the crate enables `float_roundtrip` to fix it. This
    /// test is the guard — it fails if that feature is ever dropped.
    #[test]
    fn registry_floats_round_trip_bit_exactly() {
        let r = registry();
        for g in &r.grids {
            for (name, v) in [
                ("min", g.min),
                ("max", g.max),
                ("p5", g.p5),
                ("p95", g.p95),
                ("reach", g.reach),
                ("dynamic_range", g.dynamic_range),
                ("mono", g.mono),
            ] {
                let txt = format!("{v:?}");
                let back: f64 = txt.parse().unwrap();
                assert_eq!(
                    v.to_bits(),
                    back.to_bits(),
                    "{}: {name} must round-trip bit-exactly",
                    g.label
                );
                // …and the value parsed from the FILE must equal what the
                // shortest-repr text names, which is what serde_json's default
                // (inaccurate) parser gets wrong.
                let from_file: f64 = serde_json::from_str(&txt).unwrap();
                assert_eq!(
                    v.to_bits(),
                    from_file.to_bits(),
                    "{}: {name} — serde_json must parse the registry's own text back to the \
                     identical f64. If this fails, the `float_roundtrip` feature was dropped \
                     from zensim-validate's serde_json dependency and every G-ADDR bar is now \
                     one ULP off.",
                    g.label
                );
            }
        }
    }

    // ──── the 2026-09-05 USER RULING (final): FLOOR REPRESENTABILITY ────

    /// The active rule carries **no dial-value bar** — only how many of a
    /// codec's lowest settings must resolve, and a float tolerance. This test
    /// is what says the ruling's operative form is the one in force.
    #[test]
    fn the_active_rule_carries_no_dial_value_bar() {
        let r = floor_rule();
        assert_eq!(r.bottom_k, 3, "K = the 3 lowest configurable settings");
        assert_eq!(r.clamp_eps, 1e-9);
        assert_eq!(
            active_tail_pin_set(),
            "floor-representability-resolvable-2026-09-05"
        );
        // The operative window since the 2026-09-05 ruling. `bottom_k` and
        // `clamp_eps` above are unchanged by it — only WHICH steps are chosen.
        assert_eq!(r.floor_rule, "resolvable");
        assert_eq!(r.floor_margin, 0.5);
        assert_eq!(TailPins::default(), TailPins::Product);
        // The whole point of the final ruling: no threshold in the ACTIVE pin
        // set is a bar. `report_floor_threshold` exists, and is barred against
        // nothing — the only place a dial value survives, as information.
        let reg: serde_json::Value = serde_json::from_str(REGISTRY_JSON).unwrap();
        let active = reg["negative_tail_bars"]["pin_sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "floor-representability-resolvable-2026-09-05")
            .expect("active pin set");
        for k in [
            "product_bar",
            "product_min_max",
            "product_p1_max",
            "band_lo",
            "band_hi",
            "band_threshold",
            "product_family_frac_min",
            "product_band_frac_min",
        ] {
            assert!(
                active.get(k).is_none(),
                "the active pin set must carry NO dial-value bar, but `{k}` is present — the \
                 ruling is \"i care that the lowest configurable settings per codec are \
                 representable, not that negative fifty is in that specifically\""
            );
        }
        assert!(
            active["report_floor_threshold"].is_number(),
            "the report-only column keeps its threshold, clearly labelled"
        );
    }

    /// **THE RULE, as a test.** A ladder is represented when its bottom K steps
    /// are strictly ordered — into the next step up — and off the clamp. Each
    /// half fails on its own.
    #[test]
    fn a_ladder_is_represented_only_when_ordered_and_off_the_clamp() {
        let k = floor_rule().bottom_k;
        // (a) clean: every ladder resolves.
        let m = fm_clean();
        assert_eq!(m.codecs.len(), 4);
        for c in &m.codecs {
            assert_eq!(c.represented_frac, 1.0, "{}", c.codec);
            assert_eq!(c.n_fail_order, 0);
            assert_eq!(c.n_fail_clamp, 0);
            assert_eq!(c.bottom_medians.len(), k);
        }
        // (b) a TIE inside the bottom K fails ORDER alone.
        let m = fm_of(&["avif"], 4, 6, |_, l, j| {
            if l == 0 && j == 1 {
                10.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        let c = &m.codecs[0];
        assert_eq!(c.n_fail_order, 1, "the tied ladder");
        assert_eq!(c.n_fail_clamp, 0);
        assert!((c.represented_frac - 0.75).abs() < 1e-12);
        // (c) an INVERSION against the NEXT step up (step K vs K+1) also fails,
        //     even though the bottom K alone are increasing.
        let m = fm_of(&["avif"], 4, 6, |_, l, j| {
            let base = 10.0 + j as f64 * 7.0 + l as f64;
            if l == 0 && j == 3 { -100.0 } else { base }
        });
        assert_eq!(m.codecs[0].n_fail_order, 1, "step K must exceed step K−1");
        // (d) the CLAMP: two ladders pinned to the same instrument minimum.
        let m = fm_of(&["avif"], 4, 6, |_, l, j| {
            if l < 2 && j == 0 {
                -213.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        assert_eq!(m.n_ladders_at_min, 2, "a COLLAPSED floor");
        assert_eq!(m.codecs[0].n_fail_clamp, 2);
        assert!((m.codecs[0].represented_frac - 0.5).abs() < 1e-12);
        // (e) …but a SOLE holder of the minimum is a genuine floor, not a clamp.
        let m = fm_of(&["avif"], 4, 6, |_, l, j| {
            if l == 0 && j == 0 {
                -213.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        assert_eq!(m.n_ladders_at_min, 1);
        assert_eq!(m.codecs[0].n_fail_clamp, 0, "somebody has to be lowest");
        assert_eq!(m.codecs[0].represented_frac, 1.0);
    }

    /// A ladder too short to hold `K + 1` steps is NOT TESTED — never counted
    /// as represented, never counted as a failure.
    #[test]
    fn a_ladder_shorter_than_k_plus_one_is_not_tested() {
        let k = floor_rule().bottom_k;
        let m = fm_of(&["avif"], 3, k, |_, l, j| 10.0 + j as f64 + l as f64);
        let c = &m.codecs[0];
        assert_eq!(c.n_ladders, 0, "no ladder is long enough");
        assert_eq!(c.n_too_short, 3);
        assert!(c.represented_frac.is_nan());
    }

    /// **The bar is the MENTOR's own fraction on the same cells** — never a
    /// number chosen here. A candidate matching it passes; one below it fails;
    /// and a codec the mentor cannot represent either has a bar of 0.0, which
    /// is the natural analogue of an exemption.
    #[test]
    fn the_bar_is_the_mentors_own_fraction_per_codec() {
        let f = canonical();
        let reg = floor_repr_for_grid(CANONICAL_GRID_SHA, ACTIVE_REFERENCE)
            .expect("the canonical grid must have per-codec mentor fractions");
        assert_eq!(reg.bottom_k, floor_rule().bottom_k);
        assert!(!reg.codecs.is_empty());
        for c in &reg.codecs {
            assert!(
                (0.0..=1.0).contains(&c.represented_frac),
                "{}: a fraction",
                c.codec
            );
            assert!(c.n_ladders > 0, "{}", c.codec);
        }
        // Tie the mentor exactly ⇒ every codec passes.
        let tie_fm = FloorMeasure {
            instrument: "test".into(),
            grid_min: -100.0,
            n_ladders_at_min: 1,
            codecs: reg
                .codecs
                .iter()
                .map(|c| CodecFloor {
                    codec: c.codec.clone(),
                    n_ladders: c.n_ladders,
                    n_represented: 0,
                    n_too_short: 0,
                    represented_frac: c.represented_frac,
                    n_fail_order: 0,
                    n_fail_clamp: 0,
                    dial_min: -100.0,
                    bottom_medians: vec![1.0, 2.0, 3.0],
                    report_frac_at_floor: None,
                    report_n_at_floor: None,
                })
                .collect(),
        };
        let v = ev(TailPins::Product, &f, None, Some(&tie_fm));
        assert_eq!(row_by_id(&v, "A7r").state, State::Pass);
        assert_eq!(row_by_id(&v, "A7r").measured, Some(0.0));
        for r in &v.codec_floor_rows {
            assert_eq!(r.state, State::Pass, "{}", r.codec);
            assert_eq!(r.frac_reference, Some(r.frac));
        }
        // Drop ONE codec below its bar ⇒ that codec fails, alone, and only if
        // the mentor could represent something there.
        let gradeable = reg
            .codecs
            .iter()
            .find(|c| c.represented_frac > 0.0)
            .map(|c| c.codec.clone());
        if let Some(target) = gradeable {
            let mut worse = tie_fm.clone();
            let cf = worse.codecs.iter_mut().find(|c| c.codec == target).unwrap();
            cf.represented_frac -= 1e-6;
            let v = ev(TailPins::Product, &f, None, Some(&worse));
            assert_eq!(row_by_id(&v, "A7r").state, State::Fail);
            assert_eq!(
                row_by_id(&v, "A7r").measured,
                Some(1.0),
                "exactly one codec"
            );
            assert_eq!(
                v.codec_floor_rows
                    .iter()
                    .find(|r| r.codec == target)
                    .unwrap()
                    .state,
                State::Fail
            );
        }
    }

    /// A codec the MENTOR cannot represent at all sets a bar of 0.0 — anything
    /// passes there. That is the exemption, expressed as a measurement.
    #[test]
    fn a_codec_the_mentor_cannot_represent_bars_nothing() {
        let f = canonical();
        let reg = floor_repr_for_grid(CANONICAL_GRID_SHA, ACTIVE_REFERENCE).unwrap();
        let fm = FloorMeasure {
            instrument: "test".into(),
            grid_min: -100.0,
            n_ladders_at_min: 1,
            codecs: reg
                .codecs
                .iter()
                .map(|c| CodecFloor {
                    codec: c.codec.clone(),
                    n_ladders: c.n_ladders,
                    n_represented: 0,
                    n_too_short: 0,
                    represented_frac: 0.0,
                    n_fail_order: c.n_ladders,
                    n_fail_clamp: 0,
                    dial_min: -100.0,
                    bottom_medians: vec![1.0, 2.0, 3.0],
                    report_frac_at_floor: None,
                    report_n_at_floor: None,
                })
                .collect(),
        };
        let v = ev(TailPins::Product, &f, None, Some(&fm));
        for r in &v.codec_floor_rows {
            let bar = r.frac_reference.unwrap();
            assert_eq!(
                r.state,
                if bar > 0.0 { State::Fail } else { State::Pass },
                "{}: bar {bar}",
                r.codec
            );
        }
    }

    /// Without a registered mentor row — or without a ladder measurement —
    /// `A7r` is NOT MEASURED. It is never pooled or defaulted into a pass.
    #[test]
    fn a7r_is_not_measured_without_a_mentor_row_or_a_measurement() {
        let f = canonical();
        let v = ev(TailPins::Product, &f, None, None);
        assert_eq!(row_by_id(&v, "A7r").state, State::NotMeasured);
        assert!(v.codec_floor_rows.is_empty());
        assert!(!v.shippable());
        let unreg = "7".repeat(64);
        assert!(floor_repr_for_grid(&unreg, ACTIVE_REFERENCE).is_none());
        let fm = fm_clean();
        let v = evaluate_full(
            ACTIVE_REFERENCE,
            TailPins::Product,
            &unreg,
            "unregistered",
            &tie(&f),
            None,
            None,
            Some(&fm),
            FloorRuleContext::default(),
            ValuePins::default(),
        );
        assert_eq!(row_by_id(&v, "A7r").state, State::NotMeasured);
        assert!(row_by_id(&v, "A7r").note.contains("MENTOR"));
    }

    /// `A8r` is REPORT-ONLY: no bar, never a pass or a fail, and it cannot
    /// block a ship no matter what the probe reads.
    #[test]
    fn a8r_is_report_only_and_gates_nothing() {
        let f = canonical();
        let (nf, _) = probes();
        // Shipped B's own tail: never below zero, the worst reading there is.
        let (nfb, _) = probes_b();
        let nm = nt_from(&nfb);
        let reg = floor_repr_for_grid(CANONICAL_GRID_SHA, ACTIVE_REFERENCE).unwrap();
        let fm = FloorMeasure {
            instrument: "test".into(),
            grid_min: -100.0,
            n_ladders_at_min: 1,
            codecs: reg
                .codecs
                .iter()
                .map(|c| CodecFloor {
                    codec: c.codec.clone(),
                    n_ladders: c.n_ladders,
                    n_represented: c.n_ladders,
                    n_too_short: 0,
                    represented_frac: 1.0,
                    n_fail_order: 0,
                    n_fail_clamp: 0,
                    dial_min: -100.0,
                    bottom_medians: vec![1.0, 2.0, 3.0],
                    report_frac_at_floor: None,
                    report_n_at_floor: None,
                })
                .collect(),
        };
        let v = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm),
        );
        let a8 = row_by_id(&v, "A8r");
        assert_eq!(a8.tier, Tier::Report);
        assert_eq!(a8.bar, None, "a report-only row carries NO bar");
        assert_eq!(a8.state, State::NotMeasured);
        assert!(a8.note.contains("REPORT-ONLY"));
        assert!(a8.note.contains("NO codec identity"));
        // The regression tier is decided by A1-A6 + A7r only.
        assert!(
            v.rows
                .iter()
                .filter(|r| r.tier == Tier::Regression)
                .all(|r| r.id != "A8r")
        );
        assert_ne!(
            v.regression,
            Overall::Fail,
            "a report-only row must never turn the regression tier to FAIL"
        );
        // A9r no longer exists as a row at all.
        assert!(v.rows.iter().all(|r| r.id != "A9r"), "A9r is dropped");
    }

    /// The retired grading stays reproducible, and the two pin sets genuinely
    /// disagree — the retired one fails the shipped dial's tail on all three
    /// rows, the active one never grades the probe at all.
    #[test]
    fn the_retired_pins_stay_reproducible() {
        let f = canonical();
        let (nf, _) = probes();
        let (nfb, _) = probes_b();
        let nm = nt_from(&nfb);
        let retired = ev(TailPins::Retired, &f, Some((&nm, &nf.probe_sha256)), None);
        for id in ["A7", "A8", "A9"] {
            assert_eq!(row_by_id(&retired, id).state, State::Fail, "{id}");
        }
        assert!(
            retired.codec_floor_rows.is_empty(),
            "no per-codec table there"
        );
        let product = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm_clean()),
        );
        for id in ["A7", "A8", "A9"] {
            assert!(product.rows.iter().all(|r| r.id != id), "{id} is retired");
        }
    }

    /// **THE BADGE-INVARIANCE TEST.** The board's NOT SHIPPABLE badge is
    /// CONTRACT-driven and the ruling touched the REGRESSION tail only, so
    /// every contract row must be identical under both tail pin sets.
    #[test]
    fn the_contract_tier_is_identical_under_both_tail_pin_sets() {
        let f = canonical();
        let (nf, idf) = probes();
        let (nfb, idfb) = probes_b();
        let fm = fm_clean();
        let fixtures = [
            ("mentor", nt_from(&nf), idf.clone()),
            ("shipped_b", nt_from(&nfb), idfb.clone()),
            (
                "no_truth",
                NegTailMeasure::from_scores(&[-200.0, -1.0, 3.0]),
                idf.clone(),
            ),
        ];
        for (name, nm, idr) in fixtures {
            let im = IdentityMeasure {
                n: idr.n_rows,
                dial_min: idr.dial_min,
                dial_median: idr.dial_median,
                dial_max: idr.dial_max,
                n_outside_band: 0,
                n_above_identity: 0,
                n_grid_cells_compared: 4424,
                n_grid_cells_total: 4424,
                worst: None,
            };
            let go = |tp: TailPins| {
                evaluate_full(
                    ACTIVE_REFERENCE,
                    tp,
                    &f.dial_grid_sha256,
                    &f.label,
                    &tie(&f),
                    Some((&nm, &nf.probe_sha256)),
                    Some((&im, &idr.probe_sha256)),
                    Some(&fm),
                    FloorRuleContext::default(),
                    ValuePins::default(),
                )
            };
            let a = go(TailPins::Product);
            let b = go(TailPins::Retired);
            assert_eq!(
                a.contract, b.contract,
                "{name}: the tail re-spec must not move the CONTRACT tier — the board's \
                 NOT SHIPPABLE badge reads it"
            );
            let cols = |v: &Verdict| -> Vec<_> {
                v.rows
                    .iter()
                    .filter(|r| r.tier == Tier::Contract)
                    .map(|r| (r.id, r.measured, r.bar, r.cmp, r.state))
                    .collect()
            };
            assert_eq!(
                cols(&a),
                cols(&b),
                "{name}: contract rows must be identical"
            );
            assert_eq!(cols(&a).len(), 6, "{name}: six contract rows, unchanged");
            for id in ["A1", "A2", "A3", "A4", "A5", "A6"] {
                assert_eq!(
                    row_by_id(&a, id).state,
                    row_by_id(&b, id).state,
                    "{name}/{id}: the ruling does not touch A1-A6"
                );
            }
        }
    }

    /// The selector must be literal: an unknown value is an ERROR, never a
    /// silent fall-through to the default.
    #[test]
    fn tail_pin_selector_rejects_an_unknown_value() {
        assert_eq!(TailPins::parse("product").unwrap(), TailPins::Product);
        assert_eq!(TailPins::parse("retired").unwrap(), TailPins::Retired);
        assert_eq!(
            TailPins::parse("mentor-2026-09-04").unwrap(),
            TailPins::Retired
        );
        assert!(TailPins::parse("Product").is_err(), "case-sensitive");
        assert!(TailPins::parse("ssim2").is_err());
        assert!(TailPins::parse("").is_err());
    }

    /// The report and the JSON must name the tail pin set, and the product arm
    /// must PRINT the per-codec table with the dial's min and bottom-K medians
    /// beside the mentor's fraction — the "information, not bars" half.
    #[test]
    fn the_report_prints_the_per_codec_floor_table() {
        let f = canonical();
        let (nf, _) = probes();
        let nm = nt_from(&nf);
        let fm = fm_clean();
        let v = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm),
        );
        assert_eq!(v.tail_pins, TailPins::Product);
        let md = render_markdown(&v);
        assert!(md.contains("| A7r |"));
        assert!(md.contains("| A8r |"));
        assert!(!md.contains("| A9r |"), "A9r is dropped");
        assert!(md.contains("negative-tail pin set: `product`"));
        assert!(md.contains("Per-codec FLOOR REPRESENTABILITY"));
        assert!(md.contains("repr (ssim2 = bar)"));
        assert!(md.contains("bottom-3 medians"));
        for c in ["avif", "jpeg", "jxl", "webp"] {
            assert!(md.contains(c), "codec {c} must appear");
        }
        let j = to_json(&v);
        assert_eq!(j["tail_pins"], "product");
        let cf = &j["measured"]["codec_floor"];
        assert!(cf.is_array());
        assert_eq!(cf.as_array().unwrap().len(), 4);
        assert!(cf[0]["represented_frac"].is_f64());
        assert!(cf[0]["bottom_medians"].is_array());
        // The RETIRED arm carries no per-codec table at all.
        let vr = ev(
            TailPins::Retired,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm),
        );
        assert!(to_json(&vr)["measured"]["codec_floor"].is_null());
    }

    /// `q` is quality-oriented on every codec, so "the lowest configurable
    /// settings" is always the smallest `q` — including JXL, whose lowest `q`
    /// carries the LARGEST butteraugli distance. This is asserted against the
    /// registry's recorded orientation so a grid that ever breaks it is caught.
    #[test]
    fn the_lowest_setting_is_always_the_smallest_q() {
        let reg: serde_json::Value = serde_json::from_str(REGISTRY_JSON).unwrap();
        let active = reg["negative_tail_bars"]["pin_sets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["id"] == "floor-representability-2026-09-05")
            .unwrap();
        let note = active["quality_orientation"].as_str().unwrap();
        assert!(note.contains("jxl"), "the JXL direction must be recorded");
        assert!(note.contains("25.0"), "with the measured extreme");
        // And the walk really does take the SMALLEST q: a ladder whose dial
        // rises with q is represented, one that falls with q is not.
        let up = fm_of(&["avif"], 2, 6, |_, l, j| 1.0 + j as f64 + l as f64 * 0.1);
        assert_eq!(up.codecs[0].represented_frac, 1.0);
        let down = fm_of(&["avif"], 2, 6, |_, l, j| 100.0 - j as f64 - l as f64 * 0.1);
        assert_eq!(down.codecs[0].represented_frac, 0.0);
    }

    /// The comparators must be literal, including the strict forms the
    /// contract rows use (`>` 0 is not the same bar as `≥` 0 — the latter is
    /// vacuous for a fraction).
    #[test]
    fn strict_comparators_are_not_inclusive() {
        assert_eq!(state_for(0.0, 0.0, ">"), State::Fail);
        assert_eq!(state_for(1e-9, 0.0, ">"), State::Pass);
        assert_eq!(state_for(0.0, 0.0, "≥"), State::Pass);
        assert_eq!(state_for(0.0, 0.0, "<"), State::Fail);
        assert_eq!(state_for(-1e-9, 0.0, "<"), State::Pass);
    }

    // ─────────────────── FloorRule: distinct / resolvable / spaced ───────────────────
    //
    // OWNER-EXTENSION, opt-in (2026-09-06): `benchmarks/
    // ladder_floor_resolution_2026-09-05.md` asked whether the pinned rule's
    // literal-position window is sound, or an artifact. `Resolvable` and
    // `Spaced` are that report's two variants (a)/(b), promoted to a
    // reusable, tested rule so the comparison is a flag, not a one-off Python
    // port. `Distinct` (the default) must stay byte-identical — proven below
    // both by construction (`from_grid` delegates to
    // `from_grid_with_rule(..., Distinct)`, the same code, not a fork kept in
    // sync by hand) and by reproducing five scenarios already exercised
    // elsewhere in this suite.

    /// A synthetic grid with an EXPLICIT mentor-truth column, distinct from
    /// the candidate's dial — proves a window is chosen from `mentor_f`,
    /// never `dial_f`.
    fn synth_grid_with_mentor(
        codecs: &[&str],
        n_lad: usize,
        steps: usize,
        dial_f: impl Fn(&str, usize, usize) -> f64,
        mentor_f: impl Fn(&str, usize, usize) -> f64,
    ) -> (Vec<String>, Vec<String>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let (img, cod, q, dial) = synth_grid(codecs, n_lad, steps, dial_f);
        let (_, _, _, mentor) = synth_grid(codecs, n_lad, steps, mentor_f);
        (img, cod, q, dial, mentor)
    }

    /// A `FloorMeasure` where dial AND mentor truth are the SAME `f` — the
    /// "mentor scored against itself" shape `per_codec_floor_rows_live` uses
    /// to derive a LIVE bar.
    fn fm_self_mentor(
        codecs: &[&str],
        n_lad: usize,
        steps: usize,
        rule: FloorRule,
        f: impl Fn(&str, usize, usize) -> f64,
    ) -> FloorMeasure {
        let (img, cod, q, d) = synth_grid(codecs, n_lad, steps, f);
        FloorMeasure::from_grid_with_rule("test-grid", &img, &cod, &q, &d, Some(&d), -50.0, rule)
    }

    #[test]
    fn floor_rule_parse_rejects_unknown_and_defaults_to_distinct() {
        assert_eq!(
            FloorRule::parse("distinct", 0.5).unwrap(),
            FloorRule::Distinct
        );
        assert_eq!(
            FloorRule::parse("resolvable", 0.7).unwrap(),
            FloorRule::Resolvable { margin: 0.7 }
        );
        assert_eq!(
            FloorRule::parse("spaced", 0.5).unwrap(),
            FloorRule::Spaced {
                near_lo: FloorRule::SPACED_NEAR_LO_DEFAULT,
                near_hi: FloorRule::SPACED_NEAR_HI_DEFAULT,
            }
        );
        assert!(FloorRule::parse("bogus", 0.5).is_err());
        assert_eq!(FloorRule::default(), FloorRule::Distinct);
        assert!(!FloorRule::Distinct.needs_mentor_truth());
        assert!(FloorRule::Resolvable { margin: 0.5 }.needs_mentor_truth());
        assert!(
            FloorRule::Spaced {
                near_lo: 2.0,
                near_hi: 5.0
            }
            .needs_mentor_truth()
        );
    }

    /// FAILING-FIRST GOLDEN TEST (written against the new API before it
    /// existed): `from_grid` — the legacy, unparameterized entry point every
    /// pre-2026-09-06 caller uses — and `from_grid_with_rule(...,
    /// FloorRule::Distinct)` must be FIELD-IDENTICAL on every scenario this
    /// suite already exercises elsewhere: clean, tied, inverted,
    /// collapsed-floor, sole-holder, and too-short. This is the proof that
    /// `--floor-rule distinct` (the default) reproduces every `bake_verdict`
    /// invocation before 2026-09-06 byte-for-byte.
    #[test]
    fn distinct_rule_matches_legacy_from_grid_on_every_existing_fixture() {
        fn check(label: &str, img: &[String], cod: &[String], q: &[f64], d: &[f64]) {
            let a = FloorMeasure::from_grid("g", img, cod, q, d, None, -50.0);
            let b = FloorMeasure::from_grid_with_rule(
                "g",
                img,
                cod,
                q,
                d,
                None,
                -50.0,
                FloorRule::Distinct,
            );
            assert_eq!(
                a.grid_min.to_bits(),
                b.grid_min.to_bits(),
                "{label}: grid_min"
            );
            assert_eq!(
                a.n_ladders_at_min, b.n_ladders_at_min,
                "{label}: n_ladders_at_min"
            );
            assert_eq!(a.codecs.len(), b.codecs.len(), "{label}: codec count");
            for (ca, cb) in a.codecs.iter().zip(b.codecs.iter()) {
                assert_eq!(ca.codec, cb.codec, "{label}");
                assert_eq!(
                    ca.n_ladders, cb.n_ladders,
                    "{label}/{}: n_ladders",
                    ca.codec
                );
                assert_eq!(
                    ca.n_represented, cb.n_represented,
                    "{label}/{}: n_represented",
                    ca.codec
                );
                assert_eq!(
                    ca.n_too_short, cb.n_too_short,
                    "{label}/{}: n_too_short",
                    ca.codec
                );
                assert_eq!(
                    ca.n_fail_order, cb.n_fail_order,
                    "{label}/{}: n_fail_order",
                    ca.codec
                );
                assert_eq!(
                    ca.n_fail_clamp, cb.n_fail_clamp,
                    "{label}/{}: n_fail_clamp",
                    ca.codec
                );
                assert_eq!(
                    ca.dial_min.to_bits(),
                    cb.dial_min.to_bits(),
                    "{label}/{}: dial_min",
                    ca.codec
                );
                assert_eq!(ca.bottom_medians.len(), cb.bottom_medians.len());
                for (ma, mb) in ca.bottom_medians.iter().zip(cb.bottom_medians.iter()) {
                    assert_eq!(
                        ma.to_bits(),
                        mb.to_bits(),
                        "{label}/{}: bottom_medians",
                        ca.codec
                    );
                }
                assert_eq!(
                    ca.represented_frac.is_nan(),
                    cb.represented_frac.is_nan(),
                    "{label}/{}: represented_frac nan-ness",
                    ca.codec
                );
                if !ca.represented_frac.is_nan() {
                    assert_eq!(
                        ca.represented_frac.to_bits(),
                        cb.represented_frac.to_bits(),
                        "{label}/{}: represented_frac",
                        ca.codec
                    );
                }
            }
        }

        // (a) clean: every ladder resolves (the `fm_clean()` fixture, inline).
        let (img, cod, q, d) = synth_grid(&["avif", "jpeg", "jxl", "webp"], 8, 6, |c, l, j| {
            let off = match c {
                "avif" => 0.0,
                "jpeg" => 0.5,
                "jxl" => 1.5,
                _ => 2.5,
            };
            10.0 + off + l as f64 * 3.0 + j as f64 * 7.0
        });
        check("clean", &img, &cod, &q, &d);

        // (b) a tie inside the bottom K.
        let (img, cod, q, d) = synth_grid(&["avif"], 4, 6, |_, l, j| {
            if l == 0 && j == 1 {
                10.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        check("tie", &img, &cod, &q, &d);

        // (c) an inversion against the next step up.
        let (img, cod, q, d) = synth_grid(&["avif"], 4, 6, |_, l, j| {
            let base = 10.0 + j as f64 * 7.0 + l as f64;
            if l == 0 && j == 3 { -100.0 } else { base }
        });
        check("inversion", &img, &cod, &q, &d);

        // (d) a collapsed floor (two ladders share the minimum).
        let (img, cod, q, d) = synth_grid(&["avif"], 4, 6, |_, l, j| {
            if l < 2 && j == 0 {
                -213.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        check("collapsed", &img, &cod, &q, &d);

        // (e) a sole holder of the minimum.
        let (img, cod, q, d) = synth_grid(&["avif"], 4, 6, |_, l, j| {
            if l == 0 && j == 0 {
                -213.0
            } else {
                10.0 + j as f64 * 7.0 + l as f64
            }
        });
        check("sole-holder", &img, &cod, &q, &d);

        // (f) too short.
        let k = floor_rule().bottom_k;
        let (img, cod, q, d) = synth_grid(&["avif"], 3, k, |_, l, j| 10.0 + j as f64 + l as f64);
        check("too-short", &img, &cod, &q, &d);
    }

    /// The scenario the task that minted this rule named explicitly: a
    /// ladder where the mentor inverts by 0.2 at the bottom (a near-tie, well
    /// under the 0.5-point margin). `Distinct` fails it on ORDER alone;
    /// `Resolvable` skips the unresolvable step and passes it.
    #[test]
    fn resolvable_skips_a_mentor_near_tie_that_distinct_fails_on() {
        // codec "x": ladder 0 = the test ladder (a genuine 0.2 inversion at
        // the bottom, then well-separated steps); ladder 1 = a baseline that
        // holds the SOLE grid minimum far below, so the test ladder's clamp
        // check never fires and any failure is isolated to ordering.
        let f = |_c: &str, l: usize, j: usize| -> f64 {
            if l == 0 {
                [10.0, 9.8, 12.0, 15.0, 20.0][j]
            } else {
                [-100.0, -90.0, -80.0, -70.0, -60.0][j]
            }
        };
        let distinct = fm_self_mentor(&["x"], 2, 5, FloorRule::Distinct, f);
        let resolvable = fm_self_mentor(&["x"], 2, 5, FloorRule::Resolvable { margin: 0.5 }, f);
        assert_eq!(
            distinct.n_ladders_at_min, 1,
            "ladder 1 is the sole holder of -100.0"
        );
        assert_eq!(resolvable.n_ladders_at_min, 1);

        let cd = &distinct.codecs[0];
        assert_eq!(cd.n_ladders, 2);
        assert_eq!(
            cd.n_represented, 1,
            "only the baseline ladder passes distinct"
        );
        assert_eq!(
            cd.n_fail_order, 1,
            "the test ladder fails on the 0.2 inversion"
        );
        assert_eq!(cd.n_fail_clamp, 0);
        assert!((cd.represented_frac - 0.5).abs() < 1e-12);

        let cr = &resolvable.codecs[0];
        assert_eq!(cr.n_ladders, 2);
        assert_eq!(
            cr.n_represented, 2,
            "resolvable skips the near-tie and BOTH ladders pass"
        );
        assert_eq!(cr.n_fail_order, 0);
        assert_eq!(cr.n_fail_clamp, 0);
        assert_eq!(cr.represented_frac, 1.0);
    }

    /// The window is chosen from the MENTOR's own values, never the
    /// candidate's — even when the candidate ALSO happens to dip at the same
    /// position the mentor near-ties on.
    #[test]
    fn resolvable_window_is_chosen_from_mentor_not_candidate() {
        let mentor_f = |_c: &str, _l: usize, j: usize| -> f64 { [0.0, 0.1, 5.0, 10.0, 15.0][j] };
        let dial_f = |_c: &str, _l: usize, j: usize| -> f64 { [1.0, 0.5, 2.0, 3.0, 4.0][j] };
        let (img, cod, q, dial, mentor) = synth_grid_with_mentor(&["x"], 1, 5, dial_f, mentor_f);

        // Under Distinct (literal positions 0..=3), the candidate's own
        // 1.0 -> 0.5 dip fails ordering.
        let d = FloorMeasure::from_grid_with_rule(
            "g",
            &img,
            &cod,
            &q,
            &dial,
            Some(&mentor),
            -50.0,
            FloorRule::Distinct,
        );
        assert_eq!(d.codecs[0].n_fail_order, 1);
        assert_eq!(d.codecs[0].represented_frac, 0.0);

        // Under Resolvable, the mentor's own 0.0 -> 0.1 near-tie (0.1 < 0.5)
        // is skipped, so the window becomes positions [0, 2, 3, 4] — and the
        // CANDIDATE's values there (1.0, 2.0, 3.0, 4.0) ARE strictly
        // increasing, even though its position-1 value (0.5) never appears
        // in the window at all.
        let r = FloorMeasure::from_grid_with_rule(
            "g",
            &img,
            &cod,
            &q,
            &dial,
            Some(&mentor),
            -50.0,
            FloorRule::Resolvable { margin: 0.5 },
        );
        assert_eq!(r.codecs[0].n_fail_order, 0);
        assert_eq!(r.codecs[0].represented_frac, 1.0);
    }

    #[test]
    fn spaced_window_selects_nearest_to_plus2_plus5() {
        // mentor ladder: 0.0, 4.0, 8.0, 12.0, 16.0 (evenly spaced by +4).
        // nearest to +2.0 above 0.0 is position 1 (4.0, diff 2.0); nearest to
        // +5.0 above 0.0, EXCLUDING position 1, is position 2 (8.0, diff 3.0
        // vs position 3's 12.0, diff 7.0). Window = [0, 1, 2].
        let f = |_c: &str, _l: usize, j: usize| -> f64 { j as f64 * 4.0 };
        let m = fm_self_mentor(
            &["x"],
            1,
            5,
            FloorRule::Spaced {
                near_lo: 2.0,
                near_hi: 5.0,
            },
            f,
        );
        let c = &m.codecs[0];
        assert_eq!(c.n_ladders, 1);
        assert_eq!(c.n_represented, 1);
        assert_eq!(c.bottom_medians[0], 0.0);
        assert_eq!(c.bottom_medians[1], 4.0);
        assert_eq!(c.bottom_medians[2], 8.0);
    }

    /// A ladder too short for `Spaced`'s 3-point window (fewer than 3
    /// distinct settings) is "too short" — never a silent pass.
    #[test]
    fn spaced_window_too_short_below_three_settings() {
        let f = |_c: &str, _l: usize, j: usize| -> f64 { j as f64 };
        let m = fm_self_mentor(
            &["x"],
            1,
            2,
            FloorRule::Spaced {
                near_lo: 2.0,
                near_hi: 5.0,
            },
            f,
        );
        let c = &m.codecs[0];
        assert_eq!(c.n_ladders, 0);
        assert_eq!(c.n_too_short, 1);
        assert!(c.represented_frac.is_nan());
    }

    /// The task's second required scenario: a collapsed floor (two ladders
    /// sharing the instrument minimum) fails under EVERY rule, regardless of
    /// how the window is chosen — because position 0 is always in the
    /// window under all three rules, and position 0 is where both ladders
    /// touch the minimum.
    #[test]
    fn collapsed_floor_fails_under_every_rule() {
        let f = |_c: &str, l: usize, j: usize| -> f64 {
            // Both ladders start at the shared minimum 1.0; otherwise
            // strictly increasing (so ordering ALONE would pass).
            let step = if l == 0 { 4.0 } else { 5.0 };
            1.0 + j as f64 * step
        };
        for (label, rule) in [
            ("distinct", FloorRule::Distinct),
            ("resolvable", FloorRule::Resolvable { margin: 0.5 }),
            (
                "spaced",
                FloorRule::Spaced {
                    near_lo: 2.0,
                    near_hi: 5.0,
                },
            ),
        ] {
            let m = fm_self_mentor(&["x"], 2, 5, rule, f);
            assert_eq!(m.n_ladders_at_min, 2, "{label}: a COLLAPSED floor");
            let c = &m.codecs[0];
            assert_eq!(c.n_represented, 0, "{label}: nobody is a sole holder");
            assert_eq!(c.n_fail_clamp, 2, "{label}: both ladders sit on the clamp");
            assert_eq!(c.represented_frac, 0.0, "{label}");
        }
    }

    /// Without mentor truth, `Resolvable`/`Spaced` cannot compute a window at
    /// all — every ladder reads "too short", giving a NaN fraction, which the
    /// caller (`per_codec_floor_rows_live`) turns into NOT MEASURED. NEVER a
    /// silent pass, and NEVER a silent fall-back to `Distinct`'s literal
    /// window.
    #[test]
    fn resolvable_and_spaced_without_mentor_truth_are_never_a_silent_pass() {
        let (img, cod, q, d) = synth_grid(&["x"], 3, 6, |_, l, j| 10.0 + j as f64 + l as f64);
        for rule in [
            FloorRule::Resolvable { margin: 0.5 },
            FloorRule::Spaced {
                near_lo: 2.0,
                near_hi: 5.0,
            },
        ] {
            let m = FloorMeasure::from_grid_with_rule("g", &img, &cod, &q, &d, None, -50.0, rule);
            let c = &m.codecs[0];
            assert_eq!(
                c.n_ladders, 0,
                "{:?}: no window can be computed without mentor truth",
                rule
            );
            assert_eq!(c.n_too_short, 3);
            assert!(c.represented_frac.is_nan());
        }
    }

    /// `per_codec_floor_rows_live` NEVER reads the `distinct` registry — the
    /// bar is whatever the supplied mentor `FloorMeasure` says, computed on
    /// an instrument with NO registry entry at all (a fabricated codec name
    /// that appears in zero registry rows, on a grid sha that appears in
    /// zero registry rows).
    #[test]
    fn live_bar_never_reads_the_distinct_registry() {
        let f = |_c: &str, l: usize, j: usize| -> f64 {
            if l == 0 {
                [10.0, 9.8, 12.0, 15.0, 20.0, 25.0][j]
            } else {
                10.0 + l as f64 * 3.0 + j as f64 * 7.0
            }
        };
        let rule = FloorRule::Resolvable { margin: 0.5 };
        let candidate = fm_self_mentor(&["madeupcodec123"], 4, 6, rule, f);
        let mentor = fm_self_mentor(&["madeupcodec123"], 4, 6, rule, f);

        // The REGISTRY-backed path can never grade this codec — it exists in
        // no registry row on any grid sha.
        let (registry_row, registry_table) = per_codec_floor_rows(
            "no-such-sha-in-any-registry-row",
            ACTIVE_REFERENCE,
            Some(&candidate),
            FloorRule::Distinct,
        );
        assert!(registry_table.iter().all(|r| r.state == State::NotMeasured));
        assert_eq!(registry_row.state, State::NotMeasured);

        // The LIVE path grades it anyway, using ONLY the supplied mentor
        // measurement — no registry lookup at all.
        let (live_row, live_table) =
            per_codec_floor_rows_live(rule, Some(&candidate), Some(&mentor));
        assert_eq!(live_table.len(), 1);
        assert_eq!(
            live_table[0].frac_reference,
            Some(mentor.codecs[0].represented_frac)
        );
        assert_eq!(
            live_row.state,
            State::Pass,
            "candidate == mentor here, so it must clear its own live bar"
        );
        assert!(live_row.note.contains("rule=`resolvable`"));
        assert!(live_table[0].note.contains("LIVE mentor"));
    }

    /// The rule tag is stamped in `Verdict`, the JSON, and the markdown for
    /// the default `distinct` path.
    #[test]
    fn floor_rule_tag_is_stamped_for_distinct() {
        let f = canonical();
        let fm = fm_clean();
        let v = ev(TailPins::Product, &f, None, Some(&fm));
        assert_eq!(v.floor_rule, "distinct");
        let j = to_json(&v);
        assert_eq!(j["floor_rule"], "distinct");
        let md = render_markdown(&v);
        assert!(md.contains("floor-rule: `distinct`"));
    }

    /// Same, for `resolvable` and `spaced` — and every per-codec note is
    /// stamped `rule=<tag>` too, so a fraction from one rule can never be
    /// silently read beside another's.
    #[test]
    fn floor_rule_tag_is_stamped_for_resolvable_and_spaced() {
        let f = canonical();
        let clean_f =
            |_c: &str, l: usize, j: usize| -> f64 { 10.0 + l as f64 * 3.0 + j as f64 * 7.0 };
        for (tag, rule) in [
            ("resolvable", FloorRule::Resolvable { margin: 0.5 }),
            (
                "spaced",
                FloorRule::Spaced {
                    near_lo: 2.0,
                    near_hi: 5.0,
                },
            ),
        ] {
            let fm = fm_self_mentor(&["avif", "jpeg", "jxl", "webp"], 4, 6, rule, clean_f);
            let mentor = fm_self_mentor(&["avif", "jpeg", "jxl", "webp"], 4, 6, rule, clean_f);
            let ctx = FloorRuleContext {
                rule,
                mentor: Some(&mentor),
            };
            let v = ev_rule(TailPins::Product, &f, None, Some(&fm), ctx);
            assert_eq!(v.floor_rule, tag);
            let j = to_json(&v);
            assert_eq!(j["floor_rule"], tag);
            let md = render_markdown(&v);
            assert!(md.contains(&format!("floor-rule: `{tag}`")), "{tag}");
            // The invariant that matters is not HOW the bar was obtained (a
            // `resolvable` bar can now be registry-pinned) but that a
            // `distinct` fraction is never substituted for it.
            assert!(
                md.contains("NEVER substituted") || md.contains("never the `distinct` pins"),
                "{tag}: the report must say the distinct pins are not substituted"
            );
            for r in &v.codec_floor_rows {
                assert!(
                    r.note.contains(&format!("rule=`{tag}`")),
                    "{tag}/{}: {}",
                    r.codec,
                    r.note
                );
            }
            // Every codec is fully clean here, so this must actually PASS —
            // proving the live bar isn't vacuously NotMeasured throughout.
            assert_eq!(row_by_id(&v, "A7r").state, State::Pass, "{tag}");
        }
    }

    // ───────────────────────────────────────────────────────────────────
    // THE 2026-09-05 RULING: `resolvable` is OPERATIVE, and A1-A6 report.
    //
    // Each of these fails on the pre-ruling tree. Negative control, run and
    // recorded: pointing `negative_tail_bars.active` back at
    // `floor-representability-2026-09-05` fails the first three; deleting the
    // `ValuePins` plumbing fails the last three.
    // ───────────────────────────────────────────────────────────────────

    const LADDER_372_SHA: &str = "4c3874a78c469e15c664a63e10216760317bd9501b9fe9365b6b93845cb5f980";
    const LADDER_944_SHA: &str = "0e8e5fb789bd21b263edc3531b243b4086b5ba9c6757de37188ac912bd392f2a";

    /// The OPERATIVE rule is a REGISTRY property, not a hardcoded default —
    /// which is what makes the ruling reversible without touching code.
    #[test]
    fn the_operative_floor_rule_is_resolvable_at_the_registered_margin() {
        let r = registry();
        assert_eq!(
            r.negative_tail_bars.active, "floor-representability-resolvable-2026-09-05",
            "the active pin set must be the one minted by the 2026-09-05 ruling"
        );
        match operative_floor_rule() {
            FloorRule::Resolvable { margin } => assert_eq!(
                margin,
                FloorRule::RESOLVABLE_MARGIN_DEFAULT,
                "the registered margin is the 0.5 the ruling named"
            ),
            other => panic!("operative rule must be `resolvable`, got {other:?}"),
        }
        // The superseded set is RETAINED and still reachable — every A7r
        // number published between 2026-09-05 and the flip is graded on it.
        assert!(
            r.negative_tail_bars
                .pin_sets
                .iter()
                .any(|p| p.id == "floor-representability-2026-09-05"),
            "the superseded pin set must stay in the registry, never be deleted"
        );
    }

    /// The mentor's bars under the OPERATIVE rule are PINNED on both ladder
    /// instruments — measured once through the owner, then committed.
    #[test]
    fn both_ladder_instruments_carry_registered_resolvable_bars() {
        let want: &[(&str, f64)] = &[
            ("avif-rav1e", 0.6410256410256411),
            ("avif-svt", 1.0),
            ("jpeg", 0.6666666666666666),
            ("jxl", 0.9615384615384616),
            ("webp", 1.0),
        ];
        for sha in [LADDER_372_SHA, LADDER_944_SHA] {
            let g = floor_repr_for_grid_under(
                sha,
                ACTIVE_REFERENCE,
                FloorRule::Resolvable {
                    margin: FloorRule::RESOLVABLE_MARGIN_DEFAULT,
                },
            )
            .unwrap_or_else(|| panic!("no resolvable floor row registered for {sha}"));
            assert_eq!(g.floor_rule, "resolvable");
            assert_eq!(g.floor_margin, 0.5);
            assert_eq!(g.codecs.len(), want.len(), "{sha}");
            for (codec, frac) in want {
                let row = g
                    .codecs
                    .iter()
                    .find(|c| c.codec == *codec)
                    .unwrap_or_else(|| panic!("{sha}: no `{codec}` row"));
                // Bit-exact: these are copied from the owner's own f64 output.
                assert_eq!(
                    row.represented_frac.to_bits(),
                    frac.to_bits(),
                    "{sha} / {codec}"
                );
            }
        }
        // The mentor's per-cell scores are a property of the PIXELS, so the
        // two widths must agree exactly. A disagreement means one of the two
        // grids is not the cells it claims to be.
        let a = floor_repr_for_grid_under(
            LADDER_372_SHA,
            ACTIVE_REFERENCE,
            FloorRule::Resolvable { margin: 0.5 },
        )
        .unwrap();
        let b = floor_repr_for_grid_under(
            LADDER_944_SHA,
            ACTIVE_REFERENCE,
            FloorRule::Resolvable { margin: 0.5 },
        )
        .unwrap();
        for (x, y) in a.codecs.iter().zip(b.codecs.iter()) {
            assert_eq!(x.codec, y.codec);
            assert_eq!(x.represented_frac.to_bits(), y.represented_frac.to_bits());
        }
    }

    /// THE key discipline: a `distinct` fraction and a `resolvable` fraction on
    /// the SAME grid are different quantities, and the lookup must never serve
    /// one for the other. On the 372 ladder they genuinely differ on jpeg
    /// (0.5385 pinned vs 0.6667), so this is a real discriminator.
    #[test]
    fn a_registry_lookup_never_serves_one_rules_bar_for_another() {
        let d = floor_repr_for_grid_under(LADDER_372_SHA, ACTIVE_REFERENCE, FloorRule::Distinct)
            .expect("the distinct row is RETAINED, not replaced");
        let r = floor_repr_for_grid_under(
            LADDER_372_SHA,
            ACTIVE_REFERENCE,
            FloorRule::Resolvable { margin: 0.5 },
        )
        .expect("resolvable row registered");
        assert_eq!(d.floor_rule, "distinct");
        assert_eq!(r.floor_rule, "resolvable");
        let jpeg = |g: &GridFloorRepresentability| {
            g.codecs
                .iter()
                .find(|c| c.codec == "jpeg")
                .unwrap()
                .represented_frac
        };
        assert!(
            (jpeg(&d) - jpeg(&r)).abs() > 1e-6,
            "the two rules must disagree on jpeg, else this test proves nothing: {} vs {}",
            jpeg(&d),
            jpeg(&r)
        );
        // A margin the registry was never measured at must NOT match.
        assert!(
            floor_repr_for_grid_under(
                LADDER_372_SHA,
                ACTIVE_REFERENCE,
                FloorRule::Resolvable { margin: 0.25 },
            )
            .is_none(),
            "a row measured at margin 0.5 does not answer a query at 0.25"
        );
        // `spaced` was never registered — LIVE-computed only.
        assert!(
            floor_repr_for_grid_under(
                LADDER_372_SHA,
                ACTIVE_REFERENCE,
                FloorRule::Spaced {
                    near_lo: 2.0,
                    near_hi: 5.0
                },
            )
            .is_none()
        );
    }

    fn ev_pins(vp: ValuePins) -> Verdict {
        let f = canonical();
        let fm = fm_clean();
        evaluate_full(
            ACTIVE_REFERENCE,
            TailPins::Product,
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            None,
            None,
            Some(&fm),
            FloorRuleContext::default(),
            vp,
        )
    }

    /// A1-A6 move TIER, and nothing else about them moves: same measured
    /// value, same bar, same state. "Still printed" is the whole point.
    #[test]
    fn value_pins_report_demotes_a1_a6_and_hard_restores_them() {
        let rep = ev_pins(ValuePins::Report);
        let hard = ev_pins(ValuePins::Hard);
        for id in ["A1", "A2", "A3", "A4", "A5", "A6"] {
            let r = row_by_id(&rep, id);
            let h = row_by_id(&hard, id);
            assert_eq!(r.tier, Tier::Report, "{id} must be report-only by default");
            assert_eq!(
                h.tier,
                Tier::Regression,
                "{id} under --gaddr-value-pins hard"
            );
            assert_eq!(r.measured, h.measured, "{id}: the VALUE must not move");
            assert_eq!(r.bar, h.bar, "{id}: the bar must not move");
            assert_eq!(r.state, h.state, "{id}: the pass/fail must not move");
        }
        assert_eq!(rep.value_pins, ValuePins::Report);
        assert_eq!(hard.value_pins, ValuePins::Hard);
        // Under `report` the REGRESSION tier is carried by A7r alone.
        let reg_ids: Vec<&str> = rep
            .rows
            .iter()
            .filter(|r| r.tier == Tier::Regression)
            .map(|r| r.id)
            .collect();
        assert_eq!(reg_ids, vec!["A7r"], "regression tier under `report`");
    }

    /// THE BADGE INVARIANT. The board's NOT SHIPPABLE badge is contract-driven,
    /// and this setting touches the REGRESSION tail only — asserted, never
    /// assumed. If this ever fails, a board re-grade would silently move
    /// badges.
    #[test]
    fn value_pins_moves_no_contract_row() {
        let rep = ev_pins(ValuePins::Report);
        let hard = ev_pins(ValuePins::Hard);
        let c = |v: &Verdict| {
            v.rows
                .iter()
                .filter(|r| r.tier == Tier::Contract)
                .map(|r| (r.id, r.measured, r.bar, r.state, r.note.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            c(&rep),
            c(&hard),
            "no contract row may move with value pins"
        );
        assert_eq!(
            rep.contract, hard.contract,
            "the contract verdict is identical"
        );
        assert!(
            !c(&rep).is_empty(),
            "the fixture must actually carry contract rows"
        );
    }

    /// A rule whose WINDOW comes from the mentor cannot be graded without the
    /// mentor's per-cell truth. It must read NOT MEASURED — never silently
    /// fall back to `distinct`'s window, which asks a different question, and
    /// never default to a pass.
    #[test]
    fn the_operative_rule_reads_not_measured_without_mentor_truth() {
        let f = canonical();
        let fm = fm_clean();
        let v = evaluate_full(
            ACTIVE_REFERENCE,
            TailPins::Product,
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            None,
            None,
            Some(&fm),
            FloorRuleContext {
                rule: operative_floor_rule(),
                mentor: None,
            },
            ValuePins::default(),
        );
        let a7 = row_by_id(&v, "A7r");
        assert_eq!(a7.state, State::NotMeasured);
        assert!(a7.measured.is_none(), "nothing may be reported as graded");
        assert!(
            a7.note.contains("gaddr-grid-truth"),
            "the note must name the missing input, got: {}",
            a7.note
        );
        assert!(
            a7.note.contains("distinct"),
            "the note must say the distinct window is NOT substituted, got: {}",
            a7.note
        );
        // And the same call WITH the distinct rule still grades normally, so
        // the NOT MEASURED above is about the missing truth, not a broken row.
        let d = ev(TailPins::Product, &f, None, Some(&fm));
        assert_ne!(row_by_id(&d, "A7r").state, State::NotMeasured);
    }
}
