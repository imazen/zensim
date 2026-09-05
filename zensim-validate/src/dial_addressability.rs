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
//! | negative tail (per-family floor, pooled `p1`, per-family agreement) | the dial grid's codec families + a pinned negative-tail probe | reach the product bar **−50** (see the re-pin section below; `frac_below_zero` was the retired axis) |
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
//! # The NEGATIVE TAIL is re-pinned to an absolute, PER-CODEC product bar (2026-09-05)
//!
//! **USER RULING 2026-09-05**, verbatim: *"the negative tail bar is entirely
//! arbitrary. below -5-50"*, corrected the same day, verbatim: *"i said -50 not
//! -5, codecs are all different, some go lower than others"*.
//!
//! The mentor-pinned tail rows asked a candidate to reach `min ≤ −770.62` and
//! `p1 ≤ −187.13` — `peer_ssim2`'s incidental depth on one probe, not a range
//! anything needs to address — and `frac_below_zero ≥ 1.0000`, a bar that is
//! **definitional** rather than measured (the probe's population was *selected*
//! on `ssim2 < 0`, so the mentor is below zero on 100 % of it by construction).
//! The ruling was minted after a D-peaks candidate — CID22 **+0.00798** over
//! shipped D, CONTRACT **6/6** — was refused on **A8 alone**, `p1` −167.715
//! against a bar of −187.131.
//!
//! So the tail rows are now **`A7r` / `A8r` / `A9r`**, absolute product bars:
//!
//! | row | axis | bar |
//! |---|---|---|
//! | `A7r` | **PER CODEC FAMILY**: where the REFERENCE's min on a family's rows is ≤ −50, the DIAL's min on those same rows | ≤ **−50** |
//! | `A8r` | POOLED probe `p1` | ≤ **−50** |
//! | `A9r` | **PER CODEC FAMILY**: of the family's rows whose REFERENCE truth is ≤ −50, the fraction the dial also places ≤ −50 | ≥ **0.90** *(user-provisional)* |
//!
//! **The tail is never pooled across codecs.** *"codecs are all different, some
//! go lower than others"* is a MEASUREMENT, and the registry holds it: on the
//! canonical dial grid `avif` reaches **−55.3545** and `webp` **−51.8466**,
//! while `jpeg` bottoms out at **−8.0450** and `jxl` at **−39.6858**. The last
//! two are **EXEMPT** — asking a dial to go deeper than the truth on their rows
//! would bar it for being correct. `A8r` is the single deliberate exception:
//! the negative-tail probes carry no codec column (`entry` is a bare row index
//! over a KADIS synthetic-distortion cut), so a pooled percentile is the honest
//! reading of that instrument.
//!
//! **The reachability guard is what keeps an absolute bar honest.** `A7r`
//! handles it per family through the registered `exempt` flag. `A8r` PASSES on
//! the measurement alone but may only FAIL when the probe's own `ssim2_gpu`
//! truth reaches the bar. That replaces the retired set's *registration*
//! requirement with something stronger: the registry's own rule — "a bar you
//! can dodge by choosing a friendlier instrument is not a bar" — enforced by
//! measurement rather than bookkeeping. A side-effect worth knowing: because a
//! product bar needs no reference row, the tail is now measurable on probes
//! that were never registered.
//!
//! **`A9r` is NOT MEASURED on any registered instrument today, and that is
//! reported rather than papered over.** The dial grids give only 4 avif rows
//! and 1 webp row a reference truth at or below −50 — far under the derived
//! `min_family_n` — and the probes have no codec column at all. Closing it
//! needs a CODEC-LABELLED negative-tail probe. The per-family fractions are
//! still printed.
//!
//! **Scope.** The ruling is about the REGRESSION tail. No CONTRACT row was
//! added, moved or removed, so the contract tier — and the board's
//! contract-driven NOT SHIPPABLE badge — is untouched. `A1`-`A6` stay
//! mentor-pinned.
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
/// arbitrary. below -5-50"*. The mentor-pinned tail (A7 `min` ≤ −770.62, A8
/// `p1` ≤ −187.13) is retired as a *product* requirement — those were
/// `peer_ssim2`'s incidental depth on one probe, not a range anything needs to
/// address — and replaced by an absolute, PER-CODEC-FAMILY product bar at
/// **−50**.
///
/// Both sets stay in the registry and both remain reachable, because every
/// G-ADDR number published before 2026-09-05 is graded on the retired one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TailPins {
    /// `A7r` / `A8r` / `A9r` — the absolute product range. The default.
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

/// The registered ABSOLUTE product bars for the negative tail
/// (`negative_tail_bars.pin_sets[product-range-2026-09-05]`).
#[derive(Debug, Clone, Deserialize)]
pub struct ProductTailBars {
    /// The one number the ruling names: **−50**.
    pub product_bar: f64,
    /// Minimum rows-with-reference-truth-at-or-below-the-bar a codec family
    /// needs before its `A9r` fraction is GRADED. Derived, not chosen: the
    /// binomial SE at the 0.90 bar is `sqrt(0.9·0.1/n)`, and 36 is the
    /// smallest `n` whose SE (0.050) is at or under half the 0.10 gap to a
    /// perfect 1.00.
    pub min_family_n: usize,
    pub product_family_frac_min: f64,
    #[serde(default)]
    pub product_family_frac_min_status: String,
}

/// One codec family's REFERENCE behaviour on a registered dial grid.
///
/// **USER CORRECTION 2026-09-05**, verbatim: *"codecs are all different, some
/// go lower than others"*. A family whose reference never reaches the bar is
/// `exempt` — asking a dial to go deeper than the truth on those rows would bar
/// it for being correct. MEASURED on the canonical grid: `avif` −55.3545 and
/// `webp` −51.8466 reach −50; `jpeg` −8.0450 and `jxl` −39.6858 do not.
#[derive(Debug, Clone, Deserialize)]
pub struct FamilyFloor {
    pub codec: String,
    pub n: usize,
    pub reference_min: f64,
    /// Rows of this family whose reference truth is at or below the bar — the
    /// denominator `A9r` would have, and the reason it is currently
    /// NOT MEASURED everywhere (4 and 1).
    pub n_at_or_below_bar: usize,
    pub exempt: bool,
}

/// The per-codec-family reference floors registered for one dial grid.
#[derive(Debug, Clone, Deserialize)]
pub struct GridFamilyFloors {
    pub dial_grid_sha256: String,
    #[serde(default = "default_reference")]
    pub reference: String,
    pub label: String,
    pub bar: f64,
    pub families: Vec<FamilyFloor>,
}

/// One row of `negative_tail_bars.pin_sets`. The product-bar fields are
/// optional because the RETIRED set legitimately has none — it barred against
/// the reference's own registry row instead. `#[serde(flatten)]` was
/// deliberately not used: an `Option<Struct>` flatten silently yields `None`
/// when a single field is mistyped, which is exactly how a gate quietly
/// disarms itself.
#[derive(Debug, Clone, Deserialize)]
struct TailPinSetRow {
    id: String,
    #[serde(default)]
    product_bar: Option<f64>,
    #[serde(default)]
    min_family_n: Option<usize>,
    #[serde(default)]
    product_family_frac_min: Option<f64>,
    #[serde(default)]
    product_family_frac_min_status: Option<String>,
}

impl TailPinSetRow {
    fn product(&self) -> Option<ProductTailBars> {
        Some(ProductTailBars {
            product_bar: self.product_bar?,
            min_family_n: self.min_family_n?,
            product_family_frac_min: self.product_family_frac_min?,
            product_family_frac_min_status: self
                .product_family_frac_min_status
                .clone()
                .unwrap_or_default(),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
struct NegativeTailBarsRegistry {
    active: String,
    pin_sets: Vec<TailPinSetRow>,
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
    negative_tail_bars: NegativeTailBarsRegistry,
    grid_family_floors: Vec<GridFamilyFloors>,
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

/// The registered ABSOLUTE product-range tail bars. Owned by the registry, not
/// by this file — the numbers a user moves live in one committed place.
pub fn product_tail_bars() -> ProductTailBars {
    let r = registry();
    let want = r.negative_tail_bars.active.clone();
    r.negative_tail_bars
        .pin_sets
        .into_iter()
        .find(|p| p.id == want)
        .and_then(|p| p.product())
        .expect("the active negative-tail pin set must carry the product bars")
}

/// The registry's own name for the ACTIVE negative-tail pin set.
pub fn active_tail_pin_set() -> String {
    registry().negative_tail_bars.active
}

/// The registered per-CODEC-FAMILY reference floors for a dial grid, keyed by
/// `(grid sha256, reference)` — the same two-part key every other registry
/// lookup uses, for the same reason.
pub fn family_floors_for_grid(grid_sha256: &str, reference: &str) -> Option<GridFamilyFloors> {
    registry()
        .grid_family_floors
        .into_iter()
        .find(|g| g.dial_grid_sha256 == grid_sha256 && g.reference == reference)
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
/// probe whose truth never reaches −50 cannot discriminate.
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

/// One codec family's DIAL behaviour on an instrument that carries codec
/// identity — the candidate side of `A7r` / `A9r`.
///
/// **The tail is never pooled across codecs** (USER CORRECTION 2026-09-05:
/// *"codecs are all different, some go lower than others"*), so this is the
/// unit the two per-family rows are graded on.
#[derive(Debug, Clone)]
pub struct FamilyDial {
    pub codec: String,
    pub n: usize,
    /// The dial's minimum over this family's rows.
    pub dial_min: f64,
    /// Rows of this family whose REFERENCE truth is at or below the bar —
    /// `A9r`'s denominator. `None` when no per-row reference truth was
    /// supplied for the instrument.
    pub n_ref_at_or_below: Option<usize>,
    /// Of those rows, the fraction the DIAL also places at or below the bar.
    /// `None` for the same reason.
    pub frac_at_or_below: Option<f64>,
}

/// Per-codec-family dial behaviour on one instrument, with the instrument named
/// so the report can say which one the families came from.
#[derive(Debug, Clone)]
pub struct FamilyMeasure {
    pub instrument: String,
    pub families: Vec<FamilyDial>,
}

impl FamilyMeasure {
    /// Build from row-aligned `(codec, dial)` and, when available, the
    /// instrument's own per-row reference truth.
    pub fn from_rows(
        instrument: &str,
        codec: &[String],
        dial: &[f64],
        truth: Option<&[f64]>,
        bar: f64,
    ) -> Self {
        let truth = truth.filter(|t| t.len() == dial.len());
        let mut order: Vec<String> = Vec::new();
        let mut acc: std::collections::HashMap<String, (usize, f64, usize, usize)> =
            std::collections::HashMap::new();
        for (i, c) in codec.iter().enumerate().take(dial.len()) {
            let d = dial[i];
            if !d.is_finite() {
                continue;
            }
            let e = acc.entry(c.clone()).or_insert_with(|| {
                order.push(c.clone());
                (0, f64::INFINITY, 0, 0)
            });
            e.0 += 1;
            if d < e.1 {
                e.1 = d;
            }
            if let Some(t) = truth {
                if t[i].is_finite() && t[i] <= bar {
                    e.2 += 1;
                    if d <= bar {
                        e.3 += 1;
                    }
                }
            }
        }
        order.sort();
        let families = order
            .into_iter()
            .map(|c| {
                let (n, dmin, nref, nhit) = acc[&c];
                FamilyDial {
                    codec: c,
                    n,
                    dial_min: dmin,
                    n_ref_at_or_below: truth.map(|_| nref),
                    frac_at_or_below: truth.and_then(|_| {
                        if nref == 0 {
                            None
                        } else {
                            Some(nhit as f64 / nref as f64)
                        }
                    }),
                }
            })
            .collect();
        Self {
            instrument: instrument.to_string(),
            families,
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
    /// The per-codec-family tail rows, joined to their registered reference
    /// floors. Printed as its own table — the ruling's "codecs are all
    /// different" is not a footnote, it is the shape of the measurement.
    pub family_rows: Vec<FamilyRow>,
}

/// One rendered per-codec-family tail row: the candidate's dial beside the
/// registered reference floor for the same family.
#[derive(Debug, Clone)]
pub struct FamilyRow {
    pub codec: String,
    pub n: usize,
    pub reference_min: f64,
    pub exempt: bool,
    pub dial_min: f64,
    /// `A9r`'s denominator — rows whose reference truth is at or below the bar.
    pub n_ref_at_or_below: Option<usize>,
    pub frac_at_or_below: Option<f64>,
    /// `A7r` for this family alone. Exempt families are `NotMeasured` with the
    /// exemption as their reason.
    pub a7r: State,
    /// `A9r` for this family alone.
    pub a9r: State,
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
    evaluate_full(
        reference,
        TailPins::default(),
        grid_sha256,
        grid_label,
        m,
        negtail,
        identity,
        None,
    )
}

/// Evaluate G-ADDR against an explicitly named pin set AND an explicitly named
/// NEGATIVE-TAIL pin set.
///
/// The tail selector exists because of the **USER RULING 2026-09-05** —
/// *"the negative tail bar is entirely arbitrary. below -5-50"* — which
/// retired the mentor-pinned A7/A8/A9 in favour of the absolute product range
/// A7r/A8r/A9r. Every G-ADDR number published before that date is graded on
/// the retired set, so it stays reachable: `TailPins::Retired` reproduces it
/// exactly. A1-A6 are untouched by the ruling and stay mentor-pinned in both
/// arms.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_full(
    reference: &str,
    tail_pins: TailPins,
    grid_sha256: &str,
    grid_label: &str,
    m: &GridMeasure,
    negtail: Option<(&NegTailMeasure, &str)>,
    identity: Option<(&IdentityMeasure, &str)>,
    families: Option<&FamilyMeasure>,
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
    // Filled by the product arm; empty under the retired pins, which had no
    // per-family concept at all.
    let mut family_rows: Vec<FamilyRow> = Vec::new();
    //
    // Two pin sets, selected by `tail_pins`. `Product` (the default since the
    // USER RULING 2026-09-05) grades A7r/A8r/A9r against the ABSOLUTE product
    // range; `Retired` grades A7/A8/A9 against the mentor's own depth, which
    // is what every pre-2026-09-05 number was graded on.
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
            let ptb = product_tail_bars();
            let (a7r, a9r, frows) = per_family_tail_rows(grid_sha256, reference, families, &ptb);
            family_rows = frows;
            rows.push(a7r);
            match negtail {
                Some((nm, sha)) => {
                    let nfi = floor_for_negtail(sha, INCUMBENT_REFERENCE);
                    // The negative-tail probes carry no codec identity
                    // (`entry` is a bare row index over a KADIS
                    // synthetic-distortion cut, not a codec sweep), so a
                    // per-family split is not available there and a pooled
                    // percentile is the honest reading of that instrument.
                    // ONE axis, TWO numbers: pooled `min` AND pooled `p1`
                    // must both reach the bar. `min <= p1` always holds, so
                    // `p1` is the binding one — but both are checked
                    // explicitly, so the row cannot silently start meaning
                    // something else if a percentile convention moves, and
                    // both are printed.
                    let a8_discriminating =
                        nm.truth_p1.is_finite() && nm.truth_p1 <= ptb.product_bar;
                    let both = if nm.min <= ptb.product_bar && nm.p1 <= ptb.product_bar {
                        // Report the binding number.
                        nm.p1
                    } else {
                        // Report whichever misses (the larger of the two).
                        nm.min.max(nm.p1)
                    };
                    let a8_note = if a8_discriminating {
                        format!(
                            "POOLED by design, ONE axis / TWO numbers: pooled min {:.4} AND \
                             pooled p1 {:.4} must both be ≤ {:.1}. This probe carries no CODEC \
                             column at all — MEASURED: its rows are KADIS distortion types \
                             (mean_shift, noneccentricity, …), not codec output — so the \
                             per-codec split A7r uses does not apply here and a pooled \
                             percentile is the honest reading of this instrument. (probe truth \
                             p1 {:.4})",
                            nm.min, nm.p1, ptb.product_bar, nm.truth_p1
                        )
                    } else if nm.truth_p1.is_finite() {
                        format!(
                            "pooled min {:.4} / p1 {:.4}; probe's OWN reference truth p1 is \
                             {:.4}, which does not reach the {:.1} bar — this instrument cannot \
                             discriminate, so a miss is NOT MEASURED rather than a fail",
                            nm.min, nm.p1, nm.truth_p1, ptb.product_bar
                        )
                    } else {
                        format!(
                            "pooled min {:.4} / p1 {:.4}; probe carries no readable `ssim2_gpu` \
                             truth column — the reachability guard cannot run, so a miss is NOT \
                             MEASURED rather than a fail",
                            nm.min, nm.p1
                        )
                    };
                    rows.push(product_row(
                        "A8r",
                        "negative tail — POOLED probe min AND p1 (this probe has no codec column)",
                        both,
                        ptb.product_bar,
                        "≤",
                        a8_discriminating,
                        nfi.as_ref().map(|x| x.p1),
                        a8_note,
                    ));
                }
                None => rows.push(row(
                    "A8r",
                    Tier::Regression,
                    "negative tail — POOLED probe min AND p1 (this probe has no codec column)",
                    None,
                    None,
                    "≤",
                    None,
                    "no --negtail-probe supplied".into(),
                )),
            }
            rows.push(a9r);
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
        tail_pins,
        family_rows,
    }
}

/// Build the two PER-CODEC-FAMILY tail rows (`A7r`, `A9r`) and the per-family
/// table that is printed beside them.
///
/// The registered `grid_family_floors` row supplies each family's REFERENCE
/// minimum and its exemption; the run supplies each family's DIAL minimum. A
/// family the reference never takes below the bar is EXEMPT — never a fail,
/// never a silent pass, printed as exempt.
fn per_family_tail_rows(
    grid_sha256: &str,
    reference: &str,
    families: Option<&FamilyMeasure>,
    ptb: &ProductTailBars,
) -> (CheckRow, CheckRow, Vec<FamilyRow>) {
    let floors = family_floors_for_grid(grid_sha256, reference);
    let (floors, fm) = match (floors, families) {
        (Some(f), Some(m)) => (f, m),
        (floors, _) => {
            let why = if floors.is_none() {
                format!(
                    "no per-codec-family reference floors registered for grid {} / reference \
                     `{reference}` — the exemption set is unknown, so the per-family tail is \
                     NOT MEASURED (never pooled as a substitute)",
                    &grid_sha256[..grid_sha256.len().min(16)]
                )
            } else {
                "the caller supplied no per-codec-family dial minima (the instrument carries \
                 no codec column)"
                    .to_string()
            };
            return (
                row(
                    "A7r",
                    Tier::Regression,
                    "negative tail — per-codec-family floor at the product bar",
                    None,
                    None,
                    "≤",
                    None,
                    why.clone(),
                ),
                CheckRow {
                    id: "A9r",
                    tier: Tier::Report,
                    what: "negative tail — per-codec agreement at the product bar (REPORT-ONLY, \
                           no bar)",
                    measured: None,
                    bar: None,
                    cmp: "≥",
                    state: State::NotMeasured,
                    incumbent: None,
                    note: why,
                },
                Vec::new(),
            );
        }
    };
    let mut table: Vec<FamilyRow> = Vec::new();
    for ff in &floors.families {
        let dial = fm.families.iter().find(|d| d.codec == ff.codec);
        let dial_min = dial.map(|d| d.dial_min).unwrap_or(f64::NAN);
        let n_ref = dial.and_then(|d| d.n_ref_at_or_below);
        let frac = dial.and_then(|d| d.frac_at_or_below);
        let a7r = if ff.exempt {
            State::NotMeasured
        } else if !dial_min.is_finite() {
            State::NotMeasured
        } else if dial_min <= ptb.product_bar {
            State::Pass
        } else {
            State::Fail
        };
        // A9r is REPORT-ONLY (USER REFINEMENT 2026-09-05): the fraction is
        // measured and printed per family, and NOTHING is barred on it. So a
        // family's A9r state is never Pass/Fail — the number is the point.
        let a9r = State::NotMeasured;
        let dial_vs_ref = if dial_min.is_finite() {
            format!(
                "dial min {:.4} vs reference min {:.4} ({:+.4})",
                dial_min,
                ff.reference_min,
                dial_min - ff.reference_min
            )
        } else {
            format!(
                "reference min {:.4}; no dial min measured",
                ff.reference_min
            )
        };
        let note = if ff.exempt {
            format!(
                "EXEMPT — the reference itself only reaches {:.4} on this family, never the \
                 {:.1} bar (\"some go lower than others\"). {dial_vs_ref}",
                ff.reference_min, ptb.product_bar
            )
        } else {
            let a9 = match (n_ref, frac) {
                (Some(n), Some(fr)) => format!(
                    "A9r (report-only) {fr:.4} over {n} rows whose reference truth is ≤ {:.1}",
                    ptb.product_bar
                ),
                (Some(n), None) => format!(
                    "A9r (report-only) — no rows of this family have a reference truth ≤ {:.1} \
                     (n={n})",
                    ptb.product_bar
                ),
                _ => "A9r (report-only) — no per-row reference truth supplied for this \
                      instrument (pass --gaddr-grid-truth)"
                    .to_string(),
            };
            format!("{dial_vs_ref}. {a9}")
        };
        table.push(FamilyRow {
            codec: ff.codec.clone(),
            n: ff.n,
            reference_min: ff.reference_min,
            exempt: ff.exempt,
            dial_min,
            n_ref_at_or_below: n_ref,
            frac_at_or_below: frac,
            a7r,
            a9r,
            note,
        });
    }
    // A7r: the COUNT of non-exempt families the dial fails, barred at ≤ 0.
    let n_gradeable_a7 = table.iter().filter(|r| r.a7r != State::NotMeasured).count();
    let n_fail_a7 = table.iter().filter(|r| r.a7r == State::Fail).count();
    let a7_row = CheckRow {
        id: "A7r",
        tier: Tier::Regression,
        what: "negative tail — per-codec-family floor at the product bar",
        measured: if n_gradeable_a7 > 0 {
            Some(n_fail_a7 as f64)
        } else {
            None
        },
        bar: Some(0.0),
        cmp: "≤",
        state: if n_gradeable_a7 == 0 {
            State::NotMeasured
        } else if n_fail_a7 == 0 {
            State::Pass
        } else {
            State::Fail
        },
        incumbent: None,
        note: format!(
            "families graded on `{}`: {} of {} non-exempt ({} EXEMPT — the reference never \
             reaches {:.1} there); value = number of non-exempt families whose dial min misses \
             the bar",
            fm.instrument,
            n_gradeable_a7,
            table.len(),
            table.iter().filter(|r| r.exempt).count(),
            ptb.product_bar
        ),
    };
    // A9r: REPORT-ONLY. Per non-exempt family, the fraction of rows whose
    // reference truth is at or below the bar that the DIAL also places at or
    // below it. NO bar — the user asked to see the number across the mentor,
    // the incumbent and the shipped profiles before deciding what it should
    // require. `Tier::Report` is excluded from both tier verdicts, so this row
    // can never block a ship.
    let reported: Vec<f64> = table
        .iter()
        .filter(|r| !r.exempt)
        .filter_map(|r| r.frac_at_or_below)
        .collect();
    let worst = reported.iter().copied().fold(f64::INFINITY, f64::min);
    let a9_row = CheckRow {
        id: "A9r",
        tier: Tier::Report,
        what: "negative tail — per-codec agreement at the product bar (REPORT-ONLY, no bar)",
        measured: if reported.is_empty() {
            None
        } else {
            Some(worst)
        },
        bar: None,
        cmp: "≥",
        state: State::NotMeasured,
        incumbent: None,
        note: format!(
            "REPORT-ONLY — no bar, gates nothing (a proposed {:.2} is registered as {} and \
             deliberately NOT applied). Value shown is the WORST non-exempt codec; the \
             per-codec fractions and their `n` are in the family table. {} The companion \
             per-DISTORTION-family reading on the negative-tail probe is measured in \
             benchmarks/d_peaks_lambda_sweep_2026-09-05.md §4-§6 — that instrument is KADIS \
             distortion types, not codecs, so it is reported there and never barred here.",
            ptb.product_family_frac_min,
            if ptb.product_family_frac_min_status.is_empty() {
                "registered"
            } else {
                ptb.product_family_frac_min_status.as_str()
            },
            if reported.is_empty() {
                "No per-row reference truth was supplied for this instrument (pass \
                 --gaddr-grid-truth), so no fraction could be computed."
            } else {
                ""
            }
        ),
    };
    (a7_row, a9_row, table)
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
         - **negative-tail pin set: `{}`** — {}\n\n",
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
                "`A7r`/`A8r`/`A9r` are ABSOLUTE product bars at **{:.1}**, not mentor pins. \
                 **USER RULING 2026-09-05:** *\"the negative tail bar is entirely arbitrary. \
                 below -5-50\"*, corrected the same day: *\"i said -50 not -5, codecs are all \
                 different, some go lower than others\"*. `A7r` and `A9r` are graded **PER \
                 CODEC FAMILY** and never pooled — a family whose REFERENCE never reaches \
                 {:.1} is EXEMPT, because asking a dial to go deeper than the truth would bar \
                 it for being correct. `A8r` is the one pooled row: the negative-tail probes \
                 carry no codec column. A miss is a FAIL only where the instrument can \
                 discriminate; otherwise NOT MEASURED.",
                product_tail_bars().product_bar,
                product_tail_bars().product_bar,
            ),
            TailPins::Retired =>
                "`A7`/`A8`/`A9` are the RETIRED mentor pins (`peer_ssim2`'s own depth on this \
                 probe). Retired 2026-09-05 by user ruling; kept reachable because every \
                 G-ADDR number published before that date is graded on them."
                    .to_string(),
        }
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
    if !v.family_rows.is_empty() {
        s.push_str(&format!(
            "**Per-codec-family negative tail** (bar {:.1}; A7r/A9r are graded here, never \
             pooled). `ref min` is the REFERENCE metric's own floor on that family's rows — a \
             family it never takes to the bar is EXEMPT.\n\n\
             | codec | n | ref min | exempt | dial min | A7r | rows ref ≤ bar | dial ≤ bar | A9r |\n\
             |---|--:|--:|:--:|--:|:--:|--:|--:|:--:|\n",
            product_tail_bars().product_bar
        ));
        for r in &v.family_rows {
            s.push_str(&format!(
                "| {} | {} | {:.4} | {} | {} | {} | {} | {} | {} |\n",
                r.codec,
                r.n,
                r.reference_min,
                if r.exempt { "**EXEMPT**" } else { "" },
                if r.dial_min.is_finite() {
                    format!("{:.4}", r.dial_min)
                } else {
                    "—".into()
                },
                r.a7r.mark(),
                r.n_ref_at_or_below
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "—".into()),
                r.frac_at_or_below
                    .map(|f| format!("{f:.4}"))
                    .unwrap_or_else(|| "—".into()),
                r.a9r.mark(),
            ));
        }
        s.push('\n');
        for r in &v.family_rows {
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
            // The PER-CODEC-FAMILY tail, reported whether or not it is graded.
            "families": if v.family_rows.is_empty() { serde_json::Value::Null } else {
                serde_json::json!(v.family_rows.iter().map(|r| serde_json::json!({
                    "codec": r.codec,
                    "n": r.n,
                    "reference_min": r.reference_min,
                    "exempt": r.exempt,
                    "dial_min": if r.dial_min.is_finite() { Some(r.dial_min) } else { None },
                    "n_ref_at_or_below_bar": r.n_ref_at_or_below,
                    "frac_at_or_below_bar": r.frac_at_or_below,
                    "a7r": r.a7r.tag(),
                    "a9r": r.a9r.tag(),
                    "note": r.note,
                })).collect::<Vec<_>>())
            },
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
        let pb = active
            .product()
            .expect("the active tail pin set must carry every product bar (all-or-nothing)");
        assert!(
            pb.product_bar < 0.0,
            "the product bar is a negative dial value"
        );
        assert!(
            pb.product_family_frac_min > 0.0 && pb.product_family_frac_min <= 1.0,
            "A9r's bar is a fraction"
        );
        assert!(pb.min_family_n > 0);
        // Every registered per-family floor row must agree with the active bar
        // and must derive `exempt` FROM the measurement.
        for g in &r.grid_family_floors {
            assert_eq!(g.dial_grid_sha256.len(), 64);
            assert_eq!(
                g.bar, pb.product_bar,
                "{}: a family floor cut at a different bar cannot grade this pin set",
                g.label
            );
            assert!(!g.families.is_empty(), "{}: no families", g.label);
            for fam in &g.families {
                assert_eq!(
                    fam.exempt,
                    fam.reference_min > g.bar,
                    "{} / {}: `exempt` must BE the measurement, never a hand flag",
                    g.label,
                    fam.codec
                );
                assert!(fam.n > 0);
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
    /// reference-truth extremes, which drive `A8r`'s reachability guard.
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

    /// The registered 372 kadis probe's OWN truth, measured 2026-09-05 from
    /// `negtail_probe_372_2026-09-04.parquet`: min −770.6197, p1 −187.1314.
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

    /// A per-codec-family fixture for the canonical grid: every registered
    /// family present, each with the dial min the caller names. `truth` is
    /// whether per-row reference truth was supplied (A9r's denominator).
    fn fams(dial_min: &[(&str, f64)], n_ref: Option<&[(&str, usize, f64)]>) -> FamilyMeasure {
        FamilyMeasure {
            instrument: "test-grid".into(),
            families: dial_min
                .iter()
                .map(|(c, m)| {
                    let hit = n_ref.and_then(|r| r.iter().find(|(cc, _, _)| cc == c));
                    FamilyDial {
                        codec: (*c).to_string(),
                        n: 100,
                        dial_min: *m,
                        n_ref_at_or_below: hit.map(|(_, n, _)| *n),
                        frac_at_or_below: hit.map(|(_, _, f)| *f),
                    }
                })
                .collect(),
        }
    }

    /// The canonical grid's registered families, with the dial reaching the
    /// bar on both non-exempt ones. No per-row truth ⇒ A9r NOT MEASURED.
    fn fams_ok() -> FamilyMeasure {
        fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -55.0),
            ],
            None,
        )
    }

    /// `fams_ok` plus a denominator big enough to GRADE A9r, at perfect
    /// agreement — the shape the reference itself has.
    fn fams_ok_graded() -> FamilyMeasure {
        let n = product_tail_bars().min_family_n;
        fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -55.0),
            ],
            Some(&[("avif", n, 1.0), ("webp", n, 1.0)]),
        )
    }

    /// `evaluate_full` with the canonical grid + a family measure.
    fn ev(
        tp: TailPins,
        f: &GridFloor,
        nm: Option<(&NegTailMeasure, &str)>,
        fm: Option<&FamilyMeasure>,
    ) -> Verdict {
        evaluate_full(
            ACTIVE_REFERENCE,
            tp,
            &f.dial_grid_sha256,
            &f.label,
            &tie(f),
            nm,
            None,
            fm,
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
        let fm = fams_ok_graded();
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
        assert_eq!(v.regression, Overall::Fail);
        assert!(!v.shippable());
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

        // …and the SAME numbers are a PASS under the −50 product bar, which is
        // exactly what the ruling intended: −187.1304 is still far below −50.
        let vp = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fams_ok()),
        );
        for id in ["A7r", "A8r"] {
            assert_eq!(row_by_id(&vp, id).state, State::Pass, "{id}");
        }
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
        assert_eq!(nm, vec!["A7r", "A8r", "A9r", "C3", "C4", "C5", "C6"]);
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

    // ───────── the 2026-09-05 USER RULING: the −50 per-codec tail ─────────

    /// The ruling's numbers, pinned. They live in the REGISTRY, not in this
    /// file, so a user who moves them edits one committed place — and this test
    /// is what says which numbers were in force when a verdict was written.
    #[test]
    fn the_product_tail_bars_are_the_registered_product_bar() {
        let b = product_tail_bars();
        assert_eq!(
            b.product_bar, -50.0,
            "USER CORRECTION 2026-09-05, verbatim: \"i said -50 not -5\" — nothing with −5 in \
             it may be a bar here"
        );
        assert_eq!(
            b.product_family_frac_min, 0.9,
            "A9r: per-family agreement ≥ 0.90"
        );
        assert_eq!(
            b.product_family_frac_min_status, "user-provisional",
            "A9r's 0.90 is this lane's PROPOSAL, not a user number — it must stay labelled so, \
             or the next reader will cite it as settled"
        );
        assert_eq!(
            b.min_family_n, 36,
            "derived: binomial SE at the 0.90 bar ≤ half the 0.10 gap to a perfect 1.00"
        );
        assert_eq!(active_tail_pin_set(), "product-range-2026-09-05");
        assert_eq!(TailPins::default(), TailPins::Product);
    }

    /// **THE CORRECTION, as a test.** *"codecs are all different, some go lower
    /// than others"* — MEASURED on the canonical grid: `avif` reaches −55.3545
    /// and `webp` −51.8466, so both are graded; `jpeg` bottoms out at −8.0450
    /// and `jxl` at −39.6858, so both are EXEMPT. A registry that ever calls
    /// jpeg or jxl gradeable would bar a dial for tracking the truth.
    #[test]
    fn the_registered_family_floors_encode_the_codecs_are_different_measurement() {
        let ff = family_floors_for_grid(CANONICAL_GRID_SHA, ACTIVE_REFERENCE)
            .expect("the canonical grid must have per-family reference floors");
        assert_eq!(ff.bar, -50.0);
        let by = |c: &str| {
            ff.families
                .iter()
                .find(|f| f.codec == c)
                .unwrap_or_else(|| panic!("family {c}"))
        };
        assert!(!by("avif").exempt && by("avif").reference_min <= -50.0);
        assert!(!by("webp").exempt && by("webp").reference_min <= -50.0);
        assert!(by("jpeg").exempt && by("jpeg").reference_min > -50.0);
        assert!(by("jxl").exempt && by("jxl").reference_min > -50.0);
        for f in &ff.families {
            assert_eq!(
                f.exempt,
                f.reference_min > ff.bar,
                "{}: `exempt` must BE the measurement (reference_min > bar), never a hand flag",
                f.codec
            );
        }
        // A9r's denominators, measured — and the reason it is not gradeable.
        assert_eq!(by("avif").n_at_or_below_bar, 4);
        assert_eq!(by("webp").n_at_or_below_bar, 1);
        assert!(
            by("avif").n_at_or_below_bar < product_tail_bars().min_family_n,
            "if this ever clears the minimum, A9r becomes gradeable here and §16 must be redone"
        );
    }

    /// **THE RULING, as a test.** D-peaks (`Dpeaks372_id100negrich_dial`,
    /// 2026-09-05) measured `min` −213.1486 / `p1` −167.7154 on the registered
    /// 372 probe: CID22 **+0.00798** over shipped D, CONTRACT **6/6**, and
    /// refused on **A8 alone** because −167.715 is 19.4 short of the mentor's
    /// −187.131. Under the −50 product bar `A8r` PASSES.
    ///
    /// This test fails on the pre-ruling binary — `A8r` does not exist there.
    #[test]
    fn the_ruling_unblocks_d_peaks_on_the_pooled_tail_row() {
        let f = canonical();
        let (nf, _) = probes();
        // MEASURED (arms/postC/gaddr_Dpeaks.json, 2026-09-05).
        let dpeaks = nt(
            2000,
            -213.14861297607422,
            -167.7153769991675,
            -113.72318973552896,
            0.8755,
            P372_TRUTH_MIN,
            P372_TRUTH_P1,
        );
        let retired = ev(
            TailPins::Retired,
            &f,
            Some((&dpeaks, &nf.probe_sha256)),
            None,
        );
        assert_eq!(
            row_by_id(&retired, "A8").state,
            State::Fail,
            "the retired grading must stay reproducible — this is the refusal the ruling names"
        );
        let product = ev(
            TailPins::Product,
            &f,
            Some((&dpeaks, &nf.probe_sha256)),
            Some(&fams_ok()),
        );
        assert_eq!(
            row_by_id(&product, "A8r").state,
            State::Pass,
            "−167.72 clears the −50 product bar"
        );
    }

    /// `A7r` is PER FAMILY and never pooled: one non-exempt family missing the
    /// bar fails the row even when the pooled minimum clears it, and an EXEMPT
    /// family missing it changes nothing.
    #[test]
    fn a7r_is_per_family_and_never_pooled() {
        let f = canonical();
        // avif reaches −60 (pass) but webp stops at −20 — pooled min is −60,
        // which would hide the webp failure entirely.
        let bad_webp = fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -20.0),
            ],
            None,
        );
        let v = ev(TailPins::Product, &f, None, Some(&bad_webp));
        assert_eq!(row_by_id(&v, "A7r").state, State::Fail);
        assert_eq!(
            row_by_id(&v, "A7r").measured,
            Some(1.0),
            "one non-exempt family fails"
        );
        let webp = v.family_rows.iter().find(|r| r.codec == "webp").unwrap();
        assert_eq!(webp.a7r, State::Fail);
        let avif = v.family_rows.iter().find(|r| r.codec == "avif").unwrap();
        assert_eq!(avif.a7r, State::Pass);

        // An EXEMPT family well short of the bar is NOT a fail — that is the
        // whole "some go lower than others" clause.
        let shallow_jpeg = fams(
            &[
                ("avif", -60.0),
                ("jpeg", 5.0),
                ("jxl", 10.0),
                ("webp", -55.0),
            ],
            None,
        );
        let v = ev(TailPins::Product, &f, None, Some(&shallow_jpeg));
        assert_eq!(row_by_id(&v, "A7r").state, State::Pass);
        assert_eq!(row_by_id(&v, "A7r").measured, Some(0.0));
        for c in ["jpeg", "jxl"] {
            let r = v.family_rows.iter().find(|r| r.codec == c).unwrap();
            assert!(r.exempt, "{c} must be exempt");
            assert_eq!(r.a7r, State::NotMeasured, "{c}: exempt is never a fail");
            assert!(r.note.contains("EXEMPT"), "{c}: the report must say why");
        }
        // Inclusive at the bar.
        let at_bar = fams(
            &[
                ("avif", -50.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -50.0),
            ],
            None,
        );
        assert_eq!(
            row_by_id(&ev(TailPins::Product, &f, None, Some(&at_bar)), "A7r").state,
            State::Pass
        );
    }

    /// Without registered family floors — or without per-family dial minima —
    /// the per-family rows are NOT MEASURED. They are never pooled as a
    /// substitute, which is exactly what the correction forbids.
    #[test]
    fn the_per_family_rows_are_not_measured_rather_than_pooled() {
        let f = canonical();
        // No family measure supplied at all.
        let v = ev(TailPins::Product, &f, None, None);
        for id in ["A7r", "A9r"] {
            assert_eq!(row_by_id(&v, id).state, State::NotMeasured, "{id}");
            assert!(!row_by_id(&v, id).note.is_empty(), "{id}: say why");
        }
        assert!(v.family_rows.is_empty());
        assert!(!v.shippable());
        // A grid with no registered family floors.
        let unreg_grid = "7".repeat(64);
        assert!(family_floors_for_grid(&unreg_grid, ACTIVE_REFERENCE).is_none());
        let fm = fams_ok();
        let v = evaluate_full(
            ACTIVE_REFERENCE,
            TailPins::Product,
            &unreg_grid,
            "unregistered",
            &tie(&f),
            None,
            None,
            Some(&fm),
        );
        assert_eq!(row_by_id(&v, "A7r").state, State::NotMeasured);
        assert!(
            row_by_id(&v, "A7r")
                .note
                .contains("exemption set is unknown")
        );
    }

    /// `A9r` is **REPORT-ONLY** (USER REFINEMENT 2026-09-05): per non-exempt
    /// codec it reports the fraction of rows whose reference truth is at or
    /// below the bar that the dial also places at or below it — with NO bar, so
    /// it can never pass, fail, or block a ship. The number is the deliverable.
    #[test]
    fn a9r_is_report_only_and_gates_nothing() {
        let f = canonical();
        // The REAL denominators on this grid: avif 4, webp 1.
        let real = fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -55.0),
            ],
            Some(&[("avif", 4, 1.0), ("webp", 1, 0.0)]),
        );
        let v = ev(TailPins::Product, &f, None, Some(&real));
        let a9 = row_by_id(&v, "A9r");
        assert_eq!(a9.tier, Tier::Report, "A9r must be report-only");
        assert_eq!(a9.bar, None, "a report-only row carries NO bar");
        assert_eq!(a9.state, State::NotMeasured, "it never passes or fails");
        assert!(a9.note.contains("REPORT-ONLY"));
        assert_eq!(
            a9.measured,
            Some(0.0),
            "the WORST non-exempt codec is reported — never pooled, never averaged"
        );
        // …and a catastrophic 0.0 must NOT block the ship, because it is not a
        // bar. Only A1-A8r decide the regression tier.
        for r in v.rows.iter().filter(|r| r.tier == Tier::Regression) {
            assert_ne!(r.id, "A9r");
        }
        assert_ne!(
            v.regression,
            Overall::Fail,
            "a report-only row must never turn the regression tier to FAIL"
        );
        // The per-codec fractions and their n are REPORTED.
        let avif = v.family_rows.iter().find(|r| r.codec == "avif").unwrap();
        assert_eq!(avif.n_ref_at_or_below, Some(4));
        assert_eq!(avif.frac_at_or_below, Some(1.0));
        let webp = v.family_rows.iter().find(|r| r.codec == "webp").unwrap();
        assert_eq!(webp.n_ref_at_or_below, Some(1));
        assert_eq!(webp.frac_at_or_below, Some(0.0));
        assert!(webp.note.contains("report-only"));
    }

    /// Every family row must print the codec's DIAL min beside the REFERENCE's
    /// min, and their difference — the per-codec reading the user asked to see
    /// at a glance. MEASURED example: shipped D misses `webp` by 1.9.
    #[test]
    fn every_family_row_prints_dial_min_beside_the_reference_min() {
        let f = canonical();
        // Shipped D's real grid min is -12.204 pooled; here the point is the
        // per-codec presentation, so a webp miss of 1.9 is used as the fixture.
        let d_like = fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -48.1),
            ],
            None,
        );
        let v = ev(TailPins::Product, &f, None, Some(&d_like));
        let webp = v.family_rows.iter().find(|r| r.codec == "webp").unwrap();
        assert_eq!(webp.a7r, State::Fail, "-48.1 misses the -50 bar by 1.9");
        assert!(webp.note.contains("dial min -48.1000"));
        assert!(webp.note.contains("reference min -51.8466"));
        let md = render_markdown(&v);
        assert!(
            md.contains("| ref min |"),
            "the table names the reference min"
        );
        assert!(
            md.contains("| dial min |"),
            "and the dial min, side by side"
        );
        assert!(md.contains("-48.1000") && md.contains("-51.8466"));
    }

    /// `A8r` is the ONE pooled row, and its reachability guard is the probe's
    /// own truth: a probe whose truth never reaches −50 cannot discriminate, so
    /// a miss there is NOT MEASURED — never a fail. A PASS stays a pass.
    #[test]
    fn a8r_is_pooled_and_guarded_by_the_probes_own_truth() {
        let f = canonical();
        let (nf, _) = probes();
        let fm = fams_ok();
        // Shallow probe (the real shape of negtail_probe_944_era2r4_foldapp2).
        let shallow = nt(2000, -0.9, -0.8, -0.5, 1.0, -1.0, -0.95);
        let v = ev(
            TailPins::Product,
            &f,
            Some((&shallow, &nf.probe_sha256)),
            Some(&fm),
        );
        assert_eq!(row_by_id(&v, "A8r").state, State::NotMeasured);
        assert!(row_by_id(&v, "A8r").note.contains("cannot discriminate"));
        assert_eq!(
            v.regression,
            Overall::Incomplete,
            "unmeasured is not a pass"
        );
        assert!(!v.shippable());

        // Same shallow instrument, dial DOES reach −50: still a pass.
        let deep = nt(2000, -60.0, -55.0, -30.0, 1.0, -1.0, -0.95);
        let v = ev(
            TailPins::Product,
            &f,
            Some((&deep, &nf.probe_sha256)),
            Some(&fm),
        );
        assert_eq!(row_by_id(&v, "A8r").state, State::Pass);

        // A discriminating probe where the dial falls short: a real FAIL.
        let short = nt(
            2000,
            -60.0,
            -49.9,
            -30.0,
            0.9,
            P372_TRUTH_MIN,
            P372_TRUTH_P1,
        );
        let v = ev(
            TailPins::Product,
            &f,
            Some((&short, &nf.probe_sha256)),
            Some(&fm),
        );
        assert_eq!(row_by_id(&v, "A8r").state, State::Fail);
    }

    /// A probe with NO readable truth column reads NOT MEASURED on `A8r` —
    /// never silently rescaled from a differently-united column.
    /// (`negtail_probe_944_era2r4_foldapp2.parquet` stores `human_score_norm`,
    /// a ÷100 quantity; accepting it would put the −50 bar 100× off.)
    #[test]
    fn a_probe_without_a_truth_column_is_not_measured_on_a8r() {
        let f = canonical();
        let (nf, _) = probes();
        let no_truth = NegTailMeasure::from_scores(&[-40.0, -20.0, -3.0]);
        assert!(no_truth.truth_min.is_nan() && no_truth.truth_p1.is_nan());
        let v = ev(
            TailPins::Product,
            &f,
            Some((&no_truth, &nf.probe_sha256)),
            Some(&fams_ok()),
        );
        assert_eq!(row_by_id(&v, "A8r").state, State::NotMeasured);
        assert!(row_by_id(&v, "A8r").note.contains("ssim2_gpu"));
    }

    /// The product rows need no `negtail_probes` registry row — an absolute bar
    /// has no reference to look up. That is a real coverage expansion, and the
    /// guards above are what keep it honest.
    #[test]
    fn a8r_grades_on_an_unregistered_probe() {
        let f = canonical();
        let unregistered = "9".repeat(64);
        assert!(floor_for_negtail(&unregistered, ACTIVE_REFERENCE).is_none());
        let m = nt(
            2000,
            -213.0,
            -167.0,
            -100.0,
            0.87,
            P372_TRUTH_MIN,
            P372_TRUTH_P1,
        );
        let retired = ev(TailPins::Retired, &f, Some((&m, &unregistered)), None);
        for id in ["A7", "A8", "A9"] {
            assert_eq!(row_by_id(&retired, id).state, State::NotMeasured, "{id}");
        }
        let product = ev(
            TailPins::Product,
            &f,
            Some((&m, &unregistered)),
            Some(&fams_ok()),
        );
        assert_eq!(row_by_id(&product, "A8r").state, State::Pass);
        assert_eq!(row_by_id(&product, "A7r").state, State::Pass);
    }

    /// **THE BADGE-INVARIANCE TEST.** The board's red NOT SHIPPABLE badge is
    /// driven by CONTRACT-row failures. The ruling touched only the REGRESSION
    /// tail, so every contract row — and the `contract` tier verdict — must be
    /// identical under both tail pin sets, for every fixture.
    #[test]
    fn the_contract_tier_is_identical_under_both_tail_pin_sets() {
        let f = canonical();
        let (nf, idf) = probes();
        let (nfb, idfb) = probes_b();
        let fm = fams_ok();
        let fixtures = [
            ("mentor", nt_from(&nf), idf.clone()),
            ("shipped_b", nt_from(&nfb), idfb.clone()),
            (
                "d_peaks",
                nt(
                    2000,
                    -213.14861297607422,
                    -167.7153769991675,
                    -113.72318973552896,
                    0.8755,
                    P372_TRUTH_MIN,
                    P372_TRUTH_P1,
                ),
                idf.clone(),
            ),
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
                )
            };
            let a = go(TailPins::Product);
            let b = go(TailPins::Retired);
            assert_eq!(
                a.contract, b.contract,
                "{name}: the tail re-pin must not move the CONTRACT tier — the board's \
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
            // Row COUNT is stable too, so a board reading `pass/15` keeps
            // reading `pass/15`.
            assert_eq!(a.rows.len(), b.rows.len());
            assert_eq!(a.rows.len(), 15);
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
        assert_eq!(
            TailPins::parse("product-range-2026-09-05").unwrap(),
            TailPins::Product
        );
        assert_eq!(TailPins::parse("retired").unwrap(), TailPins::Retired);
        assert_eq!(
            TailPins::parse("mentor-2026-09-04").unwrap(),
            TailPins::Retired
        );
        assert!(TailPins::parse("Product").is_err(), "case-sensitive");
        assert!(TailPins::parse("ssim2").is_err());
        assert!(TailPins::parse("").is_err());
    }

    /// `FamilyMeasure::from_rows` must group by the ROW PAIRING — per-family
    /// minima, per-family denominators — and degrade to "no truth" on a length
    /// mismatch rather than zipping short into a misattributed family.
    #[test]
    fn family_measure_groups_by_the_row_pairing() {
        let codec: Vec<String> = ["avif", "avif", "jpeg", "webp", "avif"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let dial = [-60.0, -10.0, -3.0, -80.0, 12.0];
        let truth = [-55.0, -9.0, -2.0, -51.0, 40.0];
        let m = FamilyMeasure::from_rows("g", &codec, &dial, Some(&truth), -50.0);
        assert_eq!(
            m.families
                .iter()
                .map(|f| f.codec.as_str())
                .collect::<Vec<_>>(),
            vec!["avif", "jpeg", "webp"],
            "families are sorted, so a report is stable across runs"
        );
        let avif = &m.families[0];
        assert_eq!(avif.n, 3);
        assert_eq!(
            avif.dial_min, -60.0,
            "the family's own minimum, not the pool's"
        );
        assert_eq!(avif.n_ref_at_or_below, Some(1), "only the −55 row");
        assert_eq!(
            avif.frac_at_or_below,
            Some(1.0),
            "and the dial has it at −60"
        );
        let webp = &m.families[2];
        assert_eq!(webp.n_ref_at_or_below, Some(1));
        assert_eq!(webp.frac_at_or_below, Some(1.0));
        let jpeg = &m.families[1];
        assert_eq!(jpeg.n_ref_at_or_below, Some(0));
        assert_eq!(jpeg.frac_at_or_below, None, "an empty denominator is None");
        // Length mismatch ⇒ no truth at all.
        let bad = FamilyMeasure::from_rows("g", &codec, &dial, Some(&truth[..2]), -50.0);
        assert!(bad.families.iter().all(|f| f.n_ref_at_or_below.is_none()));
    }

    /// The report and the JSON must both SAY which tail pin set graded them,
    /// and the product arm must PRINT the per-codec-family table — a number
    /// whose bars are not named is not readable a week later.
    #[test]
    fn the_report_and_json_name_the_tail_pin_set_and_print_the_families() {
        let f = canonical();
        let (nf, _) = probes();
        let nm = nt_from(&nf);
        let fm = fams(
            &[
                ("avif", -60.0),
                ("jpeg", -8.0),
                ("jxl", -39.0),
                ("webp", -55.0),
            ],
            Some(&[("avif", 4, 1.0), ("webp", 1, 1.0)]),
        );
        for (tp, want_row, want_tag) in [
            (TailPins::Product, "A7r", "product"),
            (TailPins::Retired, "A7", "retired"),
        ] {
            let v = ev(tp, &f, Some((&nm, &nf.probe_sha256)), Some(&fm));
            assert_eq!(v.tail_pins, tp);
            let md = render_markdown(&v);
            assert!(
                md.contains(&format!("| {want_row} |")),
                "{want_tag}: row id"
            );
            assert!(
                md.contains(&format!("negative-tail pin set: `{want_tag}`")),
                "{want_tag}: the report must name its own tail bars"
            );
            assert!(!md.contains("−5,"), "no −5 band may survive anywhere");
            let j = to_json(&v);
            assert_eq!(j["tail_pins"], want_tag);
            assert!(j["tail_pin_set"].is_string());
        }
        let v = ev(
            TailPins::Product,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm),
        );
        let md = render_markdown(&v);
        assert!(
            md.contains("Per-codec-family negative tail"),
            "the product arm must PRINT the per-family table"
        );
        for c in ["avif", "jpeg", "jxl", "webp"] {
            assert!(md.contains(c), "family {c} must appear in the table");
        }
        assert!(md.contains("**EXEMPT**"), "exempt families must be marked");
        let j = to_json(&v);
        let fams_json = &j["measured"]["families"];
        assert!(fams_json.is_array());
        assert_eq!(fams_json.as_array().unwrap().len(), 4);
        assert_eq!(fams_json[0]["codec"], "avif");
        assert_eq!(fams_json[0]["n_ref_at_or_below_bar"], 4);
        // The probe's own truth must ride along so a stored verdict can be
        // re-graded without re-reading the parquet.
        assert!(j["measured"]["negtail"]["truth_min"].is_f64());
        assert!(j["measured"]["negtail"]["truth_p1"].is_f64());
        // The RETIRED arm carries no family block at all.
        let vr = ev(
            TailPins::Retired,
            &f,
            Some((&nm, &nf.probe_sha256)),
            Some(&fm),
        );
        assert!(to_json(&vr)["measured"]["families"].is_null());
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
