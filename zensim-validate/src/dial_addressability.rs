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
//! | negative tail (`min`, `p1`, `frac_below_zero`) | a pinned negative-tail probe | deeper is better |
//! | identity (`dial`, `above-identity` count) | a pinned identity probe (ref == dist) | in-band, and nothing above it |
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

/// One registered dial-grid row: the reference bake's end-of-range behaviour
/// on the grid with this sha256.
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
#[derive(Debug, Clone, Copy)]
pub struct NegTailMeasure {
    pub n: usize,
    pub min: f64,
    pub p1: f64,
    pub p5: f64,
    pub frac_below_zero: f64,
}

impl NegTailMeasure {
    pub fn from_scores(scores: &[f64]) -> Self {
        let mut v: Vec<f64> = scores.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        if v.is_empty() {
            return Self {
                n: 0,
                min: f64::NAN,
                p1: f64::NAN,
                p5: f64::NAN,
                frac_below_zero: f64::NAN,
            };
        }
        let below = v.iter().filter(|x| **x < 0.0).count();
        Self {
            n: v.len(),
            min: v[0],
            p1: pct(&v, 1.0),
            p5: pct(&v, 5.0),
            frac_below_zero: below as f64 / v.len() as f64,
        }
    }
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
}

impl Tier {
    pub fn tag(self) -> &'static str {
        match self {
            Tier::Regression => "regression",
            Tier::Contract => "contract",
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
            Tier::Regression,
            "ceiling — pooled dial max",
            Some(m.max),
            f.map(|x| x.max),
            "≥",
            fi.map(|x| x.max),
            none_note(has),
        ),
        row(
            "A2",
            Tier::Regression,
            "floor — pooled dial min",
            Some(m.min),
            f.map(|x| x.min),
            "≤",
            fi.map(|x| x.min),
            none_note(has),
        ),
        row(
            "A3",
            Tier::Regression,
            "robust ceiling — dial p95",
            Some(m.p95),
            f.map(|x| x.p95),
            "≥",
            fi.map(|x| x.p95),
            none_note(has),
        ),
        row(
            "A4",
            Tier::Regression,
            "robust floor — dial p5",
            Some(m.p5),
            f.map(|x| x.p5),
            "≤",
            fi.map(|x| x.p5),
            none_note(has),
        ),
        row(
            "A5",
            Tier::Regression,
            "reach (max − min)",
            Some(m.reach),
            f.map(|x| x.reach),
            "≥",
            fi.map(|x| x.reach),
            none_note(has),
        ),
        row(
            "A6",
            Tier::Regression,
            "dynamic range (p95 − p5)",
            Some(m.dynamic_range),
            f.map(|x| x.dynamic_range),
            "≥",
            fi.map(|x| x.dynamic_range),
            none_note(has),
        ),
    ];

    // ── negative tail ──
    match negtail {
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
    if floor.is_none() {
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
    }
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
         introduced.\n\n",
        v.headline(),
        v.n_pass(),
        v.n_fail(),
        v.n_not_measured(),
        v.grid_label,
        &v.grid_sha256[..v.grid_sha256.len().min(16)],
        v.reference,
        v.incumbent_reference
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
        "| id | tier | axis | measured | bar (vs ssim2) | incumbent (shipped B) | pass |\n\
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
            })),
            "identity": v.identity.as_ref().map(|i| serde_json::json!({
                "n": i.n, "dial_min": i.dial_min, "dial_median": i.dial_median,
                "dial_max": i.dial_max, "n_outside_band": i.n_outside_band,
                "n_above_identity": i.n_above_identity,
                "n_grid_cells_total": i.n_grid_cells_total,
                "note": IDENTITY_IS_THE_ZERO_VECTOR,
            })),
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
        let nm = NegTailMeasure {
            n: nf.n_rows,
            min: nf.min,
            p1: nf.p1,
            p5: nf.p5,
            frac_below_zero: nf.frac_below_zero,
        };
        let v = evaluate(
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            None,
        );
        for r in v.rows.iter().filter(|r| r.tier == Tier::Regression) {
            assert_eq!(
                r.state,
                State::Pass,
                "{} ({}) must pass when the candidate ties the reference: {:?} vs {:?}",
                r.id,
                r.what,
                r.measured,
                r.bar
            );
        }
        assert_eq!(v.regression, Overall::Pass);
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
        assert_eq!(v.regression, Overall::Fail);
        assert!(!v.shippable());
    }

    /// A shallower negative tail is a floor regression even when the dial grid
    /// is untouched — the grid's worst cells are not the worst inputs a user
    /// will hand the metric.
    #[test]
    fn a_shallower_negative_tail_fails_on_its_own() {
        let f = canonical();
        let (nf, _) = probes();
        let mut nm = NegTailMeasure {
            n: nf.n_rows,
            min: nf.min,
            p1: nf.p1,
            p5: nf.p5,
            frac_below_zero: nf.frac_below_zero,
        };
        nm.min += 1e-3;
        nm.p1 += 1e-3;
        let v = evaluate(
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            None,
        );
        assert_eq!(row_by_id(&v, "A7").state, State::Fail);
        assert_eq!(row_by_id(&v, "A8").state, State::Fail);
        assert_eq!(v.regression, Overall::Fail);
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
        assert_eq!(nm, vec!["A7", "A8", "A9", "C3", "C4", "C5", "C6"]);
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
        let nm = NegTailMeasure {
            n: nfb.n_rows,
            min: nfb.min,
            p1: nfb.p1,
            p5: nfb.p5,
            frac_below_zero: nfb.frac_below_zero,
        };
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
    fn negtail_pins_are_reference_scoped() {
        let f = canonical();
        let (nf, _) = probes();
        let (nfb, _) = probes_b();
        assert_eq!(nf.frac_below_zero, 1.0);
        assert_eq!(nfb.frac_below_zero, 0.0);
        assert!(nf.min < nfb.min);

        // The shipped dial's own tail, barred against the mentor: A7/A8/A9 fail.
        let nm = NegTailMeasure {
            n: nfb.n_rows,
            min: nfb.min,
            p1: nfb.p1,
            p5: nfb.p5,
            frac_below_zero: nfb.frac_below_zero,
        };
        let v = evaluate(
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            None,
        );
        for id in ["A7", "A8", "A9"] {
            assert_eq!(row_by_id(&v, id).state, State::Fail, "{id}");
        }
        // Same numbers under the retired pins: all three pass (they ARE the pin).
        let vb = evaluate_with_reference(
            REFERENCE_SHIPPED_B,
            &f.dial_grid_sha256,
            &f.label,
            &tie(&f),
            Some((&nm, &nf.probe_sha256)),
            None,
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
            md.contains("bar (vs ssim2)"),
            "the table header must name the mentor"
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
}
