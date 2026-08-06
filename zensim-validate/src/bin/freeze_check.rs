//! freeze_check — the freeze-bar PASS/FAIL summary over one bake's fulleval
//! JSON: the single decision surface for the permanent-metric freeze.
//!
//! Bars come from `zenpapers:docs/zensim-final-metric-plan-2026-07-31.md` §5
//! (every bar's precedent is cited there); the owner-per-gate audit is
//! `benchmarks/decision_surface_audit_2026-07-31.md`. Rows whose measuring
//! owner lives OUTSIDE the verdict/fulleval pipeline (UPIQ, Korshunov, perf,
//! LOO, corruption ORDERING) are printed as ATTACH rows — evidence to bring,
//! never silently omitted. No stat is computed here: this bin only compares
//! numbers the owning tools already produced (no-duplication rule).
//!
//! Input: the FULLEVAL variant of the verdict JSON (i.e. after
//! `scripts/run_full_eval.sh` injects `m3a_coherence`); a raw `--full-json`
//! also works, with the injected rows degrading to ATTACH.
//!
//! Named profiles (`--profile`): a profile is a REGISTERED floor set — a
//! second decision surface that never edits the default §5 bars. The one
//! profile today is `balanced-2026-08-04` (sota944 campaign AMENDMENT 8, the
//! user-directed balanced-selection pass: floors F1–F8 + the registered
//! `balanced_composite`, frozen in `benchmarks/sota944_campaign_2026-08-03.md`
//! §8.1 BEFORE any scoring). `--tsv` / `--tsv-header` emit the machine row for
//! the pool matrix driver (`scripts/sota944_balanced_matrix.sh`).
//!
//! Exit: 0 = every evaluable row passes; 1 = at least one FAIL; 2 = usage /
//! parse error. ATTACH rows never fail the exit — they are pending evidence,
//! and the table says so out loud. In the balanced profile, a floor axis
//! ABSENT from the fulleval counts as not-passed (a candidate nobody measured
//! cannot be certified balanced on that axis).

use std::path::PathBuf;

// §5 bars (plan 2026-07-31; precedents cited there — EM4 selected seed,
// 720 corruption parity, C2a EM2-class, bake_verdict dial gates G3).
const BAR_CID22: f64 = 0.89;
const BAR_KONJND: f64 = 0.40;
const BAR_M3A: f64 = 0.85;
// NOTE: the fulleval keys are named `mono_pct`/`tied_pct` but STORE FRACTIONS
// (bake_verdict writes `dial_metrics.mono` verbatim; 0.979 = 97.9%).
const BAR_DIAL_MONO: f64 = 0.93; // G3, TWO-PANEL eval mandate
const BAR_DIAL_TIED: f64 = 0.05; // G3 (upper bound)

// ── AMENDMENT-8 registered floors (sota944 campaign doc §8.1, frozen
// 2026-08-04 BEFORE any scoring; do not edit after scoring begins) ─────────
mod balanced {
    pub const CID22: f64 = 0.885; // F1: one within-config sd (0.01246, arm-D n=12) below the §1 bar; == wave-7 H-Q2 level
    pub const KONJND: f64 = 0.43; // F2: §1 verbatim (abs)
    pub const NONPHOTO: f64 = 0.90; // F3: §1 verbatim
    pub const DIAL_MONO: f64 = 0.93; // F4: §1 dial verbatim
    pub const DIAL_TIED: f64 = 0.05; // F4 upper bound
    pub const DIAL_RANGE_MIN: f64 = 1.0; // F5: flat-dial guard
    pub const DIAL_RANGE_MAX: f64 = 120.0; // F5: bounded-[0,100]-dial sanity (catches the dyn-range-497 class)
    pub const HFNL_PERREF: f64 = 0.0; // F6: sign floor only (0.1931 stays a reported comparator)
    pub const CSIQ: f64 = 0.83; // F7: 944-class breadth
    pub const LIVE: f64 = 0.83; // F7
    pub const B9: f64 = 0.15; // F8: high-tail non-collapse (signed)
    pub const B3: f64 = 0.0; // F8: low-tail non-collapse (signed)
    pub const M3A_GOLD: f64 = 0.85; // reported tier, NOT a floor
    pub const M3A_SILVER: f64 = 0.78; // ≈ measured 944-class median (26 cells, med 0.793)
    pub const HFNL_COMPARATOR: f64 = 0.1931; // reported context only
    /// `--select` tie-break weight on M3a (campaign appendix E.4, registered
    /// 2026-08-04 BEFORE any selection ranking existed). NOT a new weight
    /// class: 0.15 is exactly what `balanced_composite` already gives its
    /// breadth additions (`W_CSIQ`/`W_LIVE`/`W_BANDTAIL`) — coherence is a
    /// product axis of that tier, material but not co-primary with CID22
    /// (1.00). Scale check from measured spreads: 0.15 × the 944-class M3a
    /// sd (0.0471) ≈ 0.007 of composite, so it breaks ties between SEEDS
    /// rather than dominating; 0.15 × the observed board range
    /// (0.199 → 0.954) = 0.113, comparable to a 0.11 CID22 swing, so it is
    /// not decorative either.
    pub const W_M3A: f64 = 0.15;
    // balanced_composite weights: the first six terms are `product_composite`
    // verbatim; csiq/live/band-tail are the registered additions at 0.15.
    // Corpus terms are abs SROCC; band-tail is SIGNED (collapse must hurt).
    pub const W_CID22: f64 = 1.00;
    pub const W_IMAZEN26: f64 = 0.50;
    pub const W_NONPHOTO: f64 = 0.30;
    pub const W_KONJND: f64 = 0.20;
    pub const W_CSIQ: f64 = 0.15;
    pub const W_LIVE: f64 = 0.15;
    pub const W_BANDTAIL: f64 = 0.15;
    pub const W_AIC3: f64 = 0.10;
    pub const W_AIC4: f64 = 0.05;
}

// ── Corpus label ORIENTATION (2026-08-05) ──────────────────────────────────
// THE OWNER is `EXPECTED_ORIENTATION` in
// `scripts/canonical_corpus/check_target_orientation.py` (campaign REGISTERED
// APPENDIX I): three eval corpora carry DISTORTION-oriented JND-family labels
// (q_jnd distance / PJND threshold) — their signed SROCC is negative BY
// CONVENTION (measured: aic4 188/188 board fullevals negative, sdr25 171/171,
// konjnd 187/188), so |SROCC| is the magnitude reading and a POSITIVE signed
// value is the defect (orientation mismatch). Quality-oriented corpora are the
// mirror: negative = genuine inversion (the Appendix F failure).
//
// This Rust list is a GATED MIRROR of the Python registry (freeze_check can't
// import Python): `distortion_oriented_mirror_matches_python_registry` parses
// the owner file and fails the build's test run on ANY drift — extend the
// registry there first, then this list.
const DISTORTION_ORIENTED: [&str; 3] = ["aic4", "konjnd", "sdr25"];

fn is_distortion_oriented(corpus: &str) -> bool {
    DISTORTION_ORIENTED.contains(&corpus)
}

fn usage() -> ! {
    eprintln!(
        "freeze_check — freeze-bar / profile PASS-FAIL over a fulleval JSON\n\n\
         usage: freeze_check --fulleval <bake.fulleval.json> [--bar name=value]...\n\
                freeze_check --fulleval <f> --profile balanced-2026-08-04 [--tsv]\n\
                             [--annotations <registry.json|none>]\n\
                freeze_check --select <a.fulleval.json> <b...> [--tsv]\n\
                freeze_check --tsv-header | --select-tsv-header\n\n\
         --select: the REGISTERED k-seed selection rule (campaign appendix\n\
         E.4). PRIMARY = profile floor count; TIE-BREAK = balanced_composite\n\
         + 0.15·M3a. M3a states are MEASURED / NOT COMPUTABLE (ensemble,\n\
         ranked separately, never penalized) / UNMEASURED (listed, NOT\n\
         selectable) — a missing measurement is never scored as zero. sdr25\n\
         is a reported comparator, not part of the rule. Exit 1 if no\n\
         candidate is selectable.\n\n\
         default (no --profile): the §5 freeze bar (unchanged).\n\
         --bar sets/overrides a cross-bake numeric bar for: csiq, live\n\
         (§5 only; their §5 bars are \"≥ best 924-arm\" — externally chosen, so\n\
         they stay ATTACH rows unless a value is supplied. The balanced\n\
         profile's floors are REGISTERED and fixed).\n\
         --profile balanced-2026-08-04: sota944 AMENDMENT-8 floors F1..F8 +\n\
         the registered balanced_composite (campaign doc §8.1).\n\
         --annotations: the committed invalidation/annotation registry\n\
         (benchmarks/eval_annotations.json). Default: $ZENSIM_EVAL_ANNOTATIONS,\n\
         else ./benchmarks/eval_annotations.json if present, else none (noted).\n\
         `absent-not-failed` entries make an ABSENT floor axis print\n\
         `— (absent)` — still not-passed for n/8 (registered rule), but\n\
         distinct from a measured FAIL, and the n/m-measured form is stated.\n\
         --tsv: one machine row (columns from --tsv-header)."
    );
    std::process::exit(2);
}

// ── Annotations registry (benchmarks/eval_annotations.json; board-integrity
// pass 2026-08-04). Machine-readable flags over fulleval cells so superseded /
// flattered / absent-not-failed numbers are never read as clean wins. Schema
// documented in the registry file's `_schema` header. ──────────────────────
#[derive(Clone)]
struct AnnEntry {
    id: String,
    kind: String, // "invalidated" | "annotated" | "absent-not-failed"
    fields: Vec<String>,
    scope: serde_json::Value,
    reason: String,
}

/// Dot-path presence: absent key OR explicit null ⇒ not present.
fn field_present(v: &serde_json::Value, dotpath: &str) -> bool {
    let mut cur = v;
    for p in dotpath.split('.') {
        match cur.get(p) {
            Some(x) => cur = x,
            None => return false,
        }
    }
    !cur.is_null()
}

/// Scope predicate: exactly one of missing / present / names / all.
fn ann_matches(v: &serde_json::Value, e: &AnnEntry) -> bool {
    if let Some(p) = e.scope.get("missing").and_then(|x| x.as_str()) {
        return !field_present(v, p);
    }
    if let Some(p) = e.scope.get("present").and_then(|x| x.as_str()) {
        return field_present(v, p);
    }
    if let Some(ns) = e.scope.get("names").and_then(|x| x.as_array()) {
        let name = bake_name(v);
        return ns.iter().any(|n| n.as_str() == Some(name));
    }
    e.scope
        .get("all")
        .and_then(|x| x.as_bool())
        .unwrap_or(false)
}

/// An entry field covers a floor's backing path iff equal or a
/// segment-boundary prefix (`rank.hfnlproxy` covers
/// `rank.hfnlproxy.per_ref_mean`; `rank.hfnl` covers neither).
fn ann_covers(entry_field: &str, floor_field: &str) -> bool {
    floor_field == entry_field
        || (floor_field.starts_with(entry_field)
            && floor_field.as_bytes().get(entry_field.len()) == Some(&b'.'))
}

fn load_annotations(path: &std::path::Path) -> Result<Vec<AnnEntry>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let entries = v
        .get("entries")
        .and_then(|e| e.as_array())
        .ok_or_else(|| format!("{}: no `entries` array", path.display()))?;
    let mut out = Vec::new();
    for e in entries {
        let s = |k: &str| e.get(k).and_then(|x| x.as_str()).map(str::to_string);
        out.push(AnnEntry {
            id: s("id").ok_or("entry missing `id`")?,
            kind: s("kind").ok_or("entry missing `kind`")?,
            fields: e
                .get("fields")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            scope: e.get("scope").cloned().unwrap_or(serde_json::Value::Null),
            reason: s("reason").unwrap_or_default(),
        });
    }
    Ok(out)
}

fn f(v: &serde_json::Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for p in path {
        cur = cur.get(p)?;
    }
    cur.as_f64()
}

enum Row {
    /// (gate, bar text, measured, pass)
    Eval(String, String, String, bool),
    /// (gate, bar text, what to attach / where the owner lives)
    Attach(String, String, String),
    /// (gate, note) — informational, no verdict
    Info(String, String),
}

/// The §5 rows, verbatim (moved out of main 2026-08-04 so the balanced
/// profile could land WITHOUT touching this path; row order, names and bar
/// semantics are test-locked below).
fn legacy_rows(v: &serde_json::Value, bar_csiq: Option<f64>, bar_live: Option<f64>) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    let ge = |name: &str, bar: f64, got: Option<f64>, absent: &str| -> Row {
        match got {
            Some(x) => Row::Eval(
                name.to_string(),
                format!("≥ {bar}"),
                format!("{x:.4}"),
                x >= bar,
            ),
            None => Row::Attach(name.to_string(), format!("≥ {bar}"), absent.to_string()),
        }
    };

    rows.push(ge(
        "CID22 SROCC (selected seed)",
        BAR_CID22,
        f(v, &["rank", "cid22", "srocc"]),
        "cid22 missing — run bake_verdict with the cid22 corpus",
    ));
    rows.push(ge(
        "KonJND abs-SROCC",
        BAR_KONJND,
        f(v, &["rank", "konjnd", "srocc"]).map(f64::abs),
        "konjnd missing — run bake_verdict with the konjnd corpus",
    ));

    // Corruption: the §5 bar (ordering ≥ 0.214) is the ORDERING stat from the
    // E-M fulleval instruments — not computed by bake_verdict (which owns the
    // detection-rate gate). Report the head's detection numbers as context and
    // keep ordering as an ATTACH row.
    match f(v, &["corruption_head", "pass_q20"]) {
        Some(q20) => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            format!(
                "head `{}` pass_q20 {:.1}% / pass_q10 {:.1}%",
                v["corruption_head"]["head"].as_str().unwrap_or("?"),
                100.0 * q20,
                100.0 * f(v, &["corruption_head", "pass_q10"]).unwrap_or(f64::NAN)
            ),
        )),
        None => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            match f(v, &["corruption", "pass_q20"]) {
                Some(d) => format!(
                    "no head given — dial-alone pass_q20 {:.1}% (broken-by-design at 924); \
                     rerun bake_verdict with --corruption-head",
                    100.0 * d
                ),
                None => "no corruption grid in this fulleval".into(),
            },
        )),
    }
    rows.push(Row::Attach(
        "Corruption ORDERING via head".into(),
        "≥ 0.214 (720 parity)".into(),
        "the E-M corruption-ordering instrument, evaluated on the HEAD bake".into(),
    ));

    for (name, bar, key) in [
        ("CSIQ SROCC", bar_csiq, "csiq"),
        ("LIVE SROCC", bar_live, "live"),
    ] {
        match (bar, f(v, &["rank", key, "srocc"])) {
            (Some(b), Some(x)) => rows.push(Row::Eval(
                format!("{name} (≥ best 924-arm)"),
                format!("≥ {b}"),
                format!("{x:.4}"),
                x >= b,
            )),
            (None, Some(x)) => rows.push(Row::Attach(
                format!("{name} (≥ best 924-arm)"),
                "cross-bake".into(),
                format!("measured {x:.4}; supply --bar {key}=<best-arm> to gate"),
            )),
            (_, None) => rows.push(Row::Attach(
                format!("{name} (≥ best 924-arm)"),
                "cross-bake".into(),
                format!("{key} missing — run bake_verdict with it"),
            )),
        }
    }

    rows.push(Row::Attach(
        "UPIQ pooled (V1-HDR)".into(),
        "> 0.7536".into(),
        "scripts/hdr/upiq_panel.py (Python owner)".into(),
    ));
    rows.push(Row::Attach(
        "Korshunov hold (V1-HDR)".into(),
        "≥ 0.93".into(),
        "seven-domain external-read runner (must be committed before Phase 4)".into(),
    ));

    rows.push(ge(
        "M3a coherence (EM2-class)",
        BAR_M3A,
        f(v, &["m3a_coherence"]),
        "not injected — this is the raw --full-json; run scripts/run_full_eval.sh",
    ));

    match f(v, &["dial", "mono_pct"]) {
        Some(m) => rows.push(Row::Eval(
            "Dial monotonicity".into(),
            format!("≥ {:.0}%", 100.0 * BAR_DIAL_MONO),
            format!("{:.1}%", 100.0 * m),
            m >= BAR_DIAL_MONO,
        )),
        None => rows.push(Row::Attach(
            "Dial monotonicity".into(),
            format!("≥ {:.0}%", 100.0 * BAR_DIAL_MONO),
            "dial block missing — bake_verdict needs the dial grid".into(),
        )),
    }
    match f(v, &["dial", "tied_pct"]) {
        Some(t) => rows.push(Row::Eval(
            "Dial tied rate".into(),
            format!("≤ {:.0}%", 100.0 * BAR_DIAL_TIED),
            format!("{:.1}%", 100.0 * t),
            t <= BAR_DIAL_TIED,
        )),
        None => rows.push(Row::Attach(
            "Dial tied rate".into(),
            format!("≤ {:.0}%", 100.0 * BAR_DIAL_TIED),
            "dial block missing".into(),
        )),
    }
    match f(v, &["composite"]) {
        Some(c) => rows.push(Row::Info("product_composite".into(), format!("{c:.4}"))),
        None => rows.push(Row::Info(
            "product_composite".into(),
            "null — needs the composite corpora (imazen26/nonphoto/...)".into(),
        )),
    }

    let repro_ok = v.get("repro").map(|r| !r.is_null()).unwrap_or(false);
    rows.push(Row::Eval(
        "Byte-repro (embedded zentrain.repro)".into(),
        "present".into(),
        if repro_ok {
            "present".into()
        } else {
            "MISSING".into()
        },
        repro_ok,
    ));

    rows.push(Row::Attach(
        "Perf SDR".into(),
        "≤ +2% vs 944 baseline".into(),
        "zenbench compare/extractor benches".into(),
    ));
    rows.push(Row::Attach(
        "Perf HDR (PU path)".into(),
        "≤ +5%".into(),
        "zenbench; the V5 lever list".into(),
    ));
    rows.push(Row::Attach(
        "LOO (append2 family)".into(),
        "≤ 0".into(),
        "extractor-side LOO instrument (gaps-doc §0)".into(),
    ));
    rows
}

// ── Balanced profile (AMENDMENT 8) ─────────────────────────────────────────

/// The four registered classes — their USES differ, so they are scored as
/// separate pools (campaign doc §8.1).
fn classify(v: &serde_json::Value) -> (&'static str, &'static str) {
    let kind = v
        .get("model")
        .and_then(|m| m.get("kind"))
        .and_then(|k| k.as_str());
    if kind == Some("ensemble") {
        return (
            "944-ensemble",
            "k× scoring cost; NOT a shippable artifact; M3a NOT COMPUTABLE",
        );
    }
    let ni = f(v, &["n_inputs"])
        .or_else(|| f(v, &["model", "n_inputs"]))
        .map(|x| x as i64);
    if ni != Some(944) {
        return (
            "era-bridge",
            "context only — regime-incomparable, never shortlisted",
        );
    }
    if bake_name(v).starts_with("C_ensk") {
        return (
            "944-distilled",
            "shippable; the M3a-mover class (wave-6 arm F)",
        );
    }
    ("944-single", "shippable single bake")
}

fn bake_name(v: &serde_json::Value) -> &str {
    v["name"]
        .as_str()
        .or_else(|| v["bake"].as_str())
        .unwrap_or("?")
}

/// CID22 band srocc + n by band label ("B3"/"B9").
///
/// **`srocc` here is an ABSOLUTE value** — `zenstats::panel` computes
/// `spearman(..).abs()` and `bake_verdict`'s per-band rows come from it — even
/// though F8 is specified as signed. The pass/fail arithmetic below still reads
/// this field (changing it would move verdicts for every published board cell,
/// which is a user decision, not a refactor); [`cid22_band_signed`] exists so
/// the report can SHOW when the two disagree.
fn cid22_band(v: &serde_json::Value, band: &str) -> Option<(f64, i64)> {
    let bands = v.get("rank")?.get("cid22")?.get("bands")?.as_array()?;
    let b = bands.iter().find(|b| b["band"].as_str() == Some(band))?;
    Some((b["srocc"].as_f64()?, b["n"].as_i64().unwrap_or(0)))
}

/// The band's SIGNED Spearman, when the fulleval carries it (emitted since the
/// 2026-08-06 `srocc_signed` band field; older JSONs return `None`).
///
/// Why this matters, measured on a 32-cell stratified board sample
/// (`benchmarks/appendixU/board_b9_signed_2026-08-06.tsv`): **32 of 32 had a
/// NEGATIVE B9** — their high-fidelity band is ordered backwards — while 25 of
/// them PASS the `B9 >= 0.15` bar on the absolute value. Because |·| is monotone
/// in the depth of an inversion, the bar as implemented ranks those models by how
/// WRONG their top band is. The band is also degenerate (43 pairs, 11 of 49 refs,
/// MOS span 0.0194, marginal bootstrap sd 0.178), and widening the slice by two
/// MOS points flips every model positive.
fn cid22_band_signed(v: &serde_json::Value, band: &str) -> Option<f64> {
    let bands = v.get("rank")?.get("cid22")?.get("bands")?.as_array()?;
    bands
        .iter()
        .find(|b| b["band"].as_str() == Some(band))?
        .get("srocc_signed")?
        .as_f64()
}

/// The registered ranking composite (§8.1): product_composite's six terms
/// verbatim + csiq/live/band-tail at 0.15. Corpus terms are abs SROCC (owner
/// convention); band-tail is SIGNED. Absent terms drop from num AND den.
fn balanced_composite(v: &serde_json::Value) -> Option<f64> {
    let corpus = |c: &str| f(v, &["rank", c, "srocc"]).map(f64::abs);
    let bandtail = match (cid22_band(v, "B3"), cid22_band(v, "B9")) {
        (Some((b3, _)), Some((b9, _))) => Some((b3 + b9) / 2.0),
        _ => None,
    };
    let terms: [(Option<f64>, f64); 9] = [
        (corpus("cid22"), balanced::W_CID22),
        (corpus("imazen26"), balanced::W_IMAZEN26),
        (corpus("nonphoto"), balanced::W_NONPHOTO),
        (corpus("konjnd"), balanced::W_KONJND),
        (corpus("csiq"), balanced::W_CSIQ),
        (corpus("live"), balanced::W_LIVE),
        (bandtail, balanced::W_BANDTAIL),
        (corpus("aic3"), balanced::W_AIC3),
        (corpus("aic4"), balanced::W_AIC4),
    ];
    let (num, den) = terms
        .iter()
        .filter_map(|(x, w)| x.map(|x| (x * w, *w)))
        .fold((0.0f64, 0.0f64), |(n, d), (x, w)| (n + x, d + w));
    if den > 0.0 { Some(num / den) } else { None }
}

/// One registered floor: id (TSV fail token), gate name, bar text, measured
/// text, pass (absent axis ⇒ false, text says UNEVALUABLE). `fields` = the
/// fulleval dot-paths backing the floor (annotation coverage); `absent` = the
/// backing value(s) were missing; `absent_not_failed` = an
/// `absent-not-failed` registry entry covers this absence (still not-passed
/// for n/8 per the registered rule, but printed `— (absent)`, kept out of the
/// measured-fails list, and counted in the n/m-measured form).
struct Floor {
    id: &'static str,
    gate: &'static str,
    bar: String,
    measured: String,
    pass: bool,
    fields: &'static [&'static str],
    absent: bool,
    absent_not_failed: bool,
}

fn floor_ge(
    id: &'static str,
    gate: &'static str,
    bar: f64,
    got: Option<f64>,
    fields: &'static [&'static str],
) -> Floor {
    match got {
        Some(x) => Floor {
            id,
            gate,
            bar: format!("≥ {bar}"),
            measured: format!("{x:.4}"),
            pass: x >= bar,
            fields,
            absent: false,
            absent_not_failed: false,
        },
        None => Floor {
            id,
            gate,
            bar: format!("≥ {bar}"),
            measured: "not measured — UNEVALUABLE".into(),
            pass: false,
            fields,
            absent: true,
            absent_not_failed: false,
        },
    }
}

fn m3a_tier(v: &serde_json::Value, class: &str) -> String {
    if class == "944-ensemble" {
        return "NOT COMPUTABLE (ensemble — the coherence instrument loads one ZNPR)".into();
    }
    match f(v, &["m3a_coherence"]) {
        Some(x) if x >= balanced::M3A_GOLD => format!("{x:.4} — GOLD (≥ {})", balanced::M3A_GOLD),
        Some(x) if x >= balanced::M3A_SILVER => {
            format!("{x:.4} — silver (≥ {})", balanced::M3A_SILVER)
        }
        Some(x) => format!("{x:.4} — FLAGGED (< {})", balanced::M3A_SILVER),
        None => "— (NOT MEASURED)".into(),
    }
}

struct BalancedReport {
    class: &'static str,
    class_note: &'static str,
    floors: Vec<Floor>,
    info: Vec<(String, String)>,
    composite: Option<f64>,
    /// matched registry entries: (id, kind, reason)
    annotations: Vec<(String, String, String)>,
}

fn eval_balanced(v: &serde_json::Value, anns: &[AnnEntry]) -> BalancedReport {
    let (class, class_note) = classify(v);
    let mut floors = Vec::new();

    // F1 CID22
    floors.push(floor_ge(
        "cid22",
        "F1 CID22 SROCC",
        balanced::CID22,
        f(v, &["rank", "cid22", "srocc"]),
        &["rank.cid22.srocc"],
    ));
    // F2 KonJND (abs)
    floors.push(floor_ge(
        "konjnd",
        "F2 KonJND abs-SROCC",
        balanced::KONJND,
        f(v, &["rank", "konjnd", "srocc"]).map(f64::abs),
        &["rank.konjnd.srocc"],
    ));
    // F3 nonphoto
    floors.push(floor_ge(
        "nonphoto",
        "F3 nonphoto SROCC",
        balanced::NONPHOTO,
        f(v, &["rank", "nonphoto", "srocc"]),
        &["rank.nonphoto.srocc"],
    ));
    // F4 dial mono + tied (one row, both must hold).
    // UNIT ANNOTATION (packaging appendix, 2026-08-04): for SPLINE-LESS bakes
    // the mono/tied stats are measured on RAW outputs (span ~16-17), where the
    // 0.5-score-pt materiality threshold flatters mono vs the real [0,100]
    // dial (span ~63-67 after add-spline+pack; strict-backwards is
    // cal-invariant, so the drop is a unit effect, not new inversions). The
    // floor itself is unchanged — this labels which unit the number is in.
    let dial_unit = match v.get("model").and_then(|m| m.get("output_spline")) {
        Some(s) if !s.is_null() => "dial-unit",
        _ => "raw-unit",
    };
    let dial_bar = format!(
        "mono ≥ {:.0}% ∧ tied ≤ {:.0}%",
        100.0 * balanced::DIAL_MONO,
        100.0 * balanced::DIAL_TIED
    );
    const F4_FIELDS: &[&str] = &["dial.mono_pct", "dial.tied_pct"];
    floors.push(
        match (f(v, &["dial", "mono_pct"]), f(v, &["dial", "tied_pct"])) {
            (Some(m), Some(t)) => Floor {
                id: "dial",
                gate: "F4 dial mono/tied",
                bar: dial_bar,
                measured: format!(
                    "mono {:.1}% / tied {:.1}% ({dial_unit})",
                    100.0 * m,
                    100.0 * t
                ),
                pass: m >= balanced::DIAL_MONO && t <= balanced::DIAL_TIED,
                fields: F4_FIELDS,
                absent: false,
                absent_not_failed: false,
            },
            _ => Floor {
                id: "dial",
                gate: "F4 dial mono/tied",
                bar: dial_bar,
                measured: "dial block missing — UNEVALUABLE".into(),
                pass: false,
                fields: F4_FIELDS,
                absent: true,
                absent_not_failed: false,
            },
        },
    );
    // F5 dial dynamic-range sanity
    let range_bar = format!(
        "{} ≤ span ≤ {}",
        balanced::DIAL_RANGE_MIN,
        balanced::DIAL_RANGE_MAX
    );
    const F5_FIELDS: &[&str] = &["dial.dynamic_range"];
    floors.push(match f(v, &["dial", "dynamic_range"]) {
        Some(r) => Floor {
            id: "dialrange",
            gate: "F5 dial span sanity",
            bar: range_bar,
            measured: format!("{r:.1}"),
            pass: (balanced::DIAL_RANGE_MIN..=balanced::DIAL_RANGE_MAX).contains(&r),
            fields: F5_FIELDS,
            absent: false,
            absent_not_failed: false,
        },
        None => Floor {
            id: "dialrange",
            gate: "F5 dial span sanity",
            bar: range_bar,
            measured: "dial block missing — UNEVALUABLE".into(),
            pass: false,
            fields: F5_FIELDS,
            absent: true,
            absent_not_failed: false,
        },
    });
    // F6 HF-NL per-ref sign floor
    floors.push(floor_ge(
        "hfnl",
        "F6 HF-NL per-ref (sign)",
        balanced::HFNL_PERREF,
        f(v, &["rank", "hfnlproxy", "per_ref_mean"]),
        &["rank.hfnlproxy.per_ref_mean"],
    ));
    // F7 breadth: CSIQ ∧ LIVE (one row)
    const F7_FIELDS: &[&str] = &["rank.csiq.srocc", "rank.live.srocc"];
    floors.push(
        match (
            f(v, &["rank", "csiq", "srocc"]),
            f(v, &["rank", "live", "srocc"]),
        ) {
            (Some(c), Some(l)) => Floor {
                id: "breadth",
                gate: "F7 breadth CSIQ ∧ LIVE",
                bar: format!("both ≥ {}", balanced::CSIQ),
                measured: format!("csiq {c:.4} / live {l:.4}"),
                pass: c >= balanced::CSIQ && l >= balanced::LIVE,
                fields: F7_FIELDS,
                absent: false,
                absent_not_failed: false,
            },
            _ => Floor {
                id: "breadth",
                gate: "F7 breadth CSIQ ∧ LIVE",
                bar: format!("both ≥ {}", balanced::CSIQ),
                measured: "csiq/live missing — UNEVALUABLE".into(),
                pass: false,
                fields: F7_FIELDS,
                absent: true,
                absent_not_failed: false,
            },
        },
    );
    // F8 band tails (signed; n printed, n<30 parenthesized per board convention)
    let fmt_band = |b: Option<(f64, i64)>| -> String {
        match b {
            Some((s, n)) if n < 30 => format!("({s:.3}) n={n}"),
            Some((s, n)) => format!("{s:.3} n={n}"),
            None => "missing".into(),
        }
    };
    let b3 = cid22_band(v, "B3");
    let b9 = cid22_band(v, "B9");
    // Show the signed value when it disagrees with the abs'd one the gate reads.
    // Display only — `pass` below is unchanged.
    let inv = |band: &str| -> String {
        match cid22_band_signed(v, band) {
            Some(s) if s < 0.0 => format!(" ⛔INVERTED signed {s:+.3}"),
            _ => String::new(),
        }
    };
    floors.push(Floor {
        id: "bandtail",
        gate: "F8 CID22 band tails",
        bar: format!("B9 ≥ {} ∧ B3 ≥ {}", balanced::B9, balanced::B3),
        measured: format!(
            "B9 {}{} / B3 {}{}",
            fmt_band(b9), inv("B9"), fmt_band(b3), inv("B3")
        ),
        pass: matches!((b3, b9), (Some((s3, _)), Some((s9, _)))
            if s9 >= balanced::B9 && s3 >= balanced::B3),
        fields: &["rank.cid22.bands"],
        absent: b3.is_none() && b9.is_none(),
        absent_not_failed: false,
    });

    // ── Registry application (annotations; board-integrity pass 2026-08-04).
    // absent-not-failed: an ABSENT floor covered by a matched entry prints
    // `— (absent)` and is excluded from the measured-fails list; it still
    // counts as not-passed in n/8 (the registered rule), and the
    // n/m-measured second form is reported alongside.
    let matched: Vec<&AnnEntry> = anns.iter().filter(|e| ann_matches(v, e)).collect();
    for e in &matched {
        if e.kind != "absent-not-failed" {
            continue;
        }
        for fl in floors.iter_mut() {
            if fl.absent
                && fl
                    .fields
                    .iter()
                    .any(|ff| e.fields.iter().any(|ef| ann_covers(ef, ff)))
            {
                fl.absent_not_failed = true;
                fl.measured = format!("— (absent — not measured; `{}`)", e.id);
            }
        }
    }

    // Reported (never floors)
    let mut info: Vec<(String, String)> = Vec::new();
    info.push(("M3a (tiered, reported)".into(), m3a_tier(v, class)));
    info.push((
        "Corruption (head-owned)".into(),
        match f(v, &["corruption_head", "pass_q20"]) {
            Some(q20) => format!(
                "head `{}` pass_q20 {:.1}% / pass_q10 {:.1}%",
                v["corruption_head"]["head"].as_str().unwrap_or("?"),
                100.0 * q20,
                100.0 * f(v, &["corruption_head", "pass_q10"]).unwrap_or(f64::NAN)
            ),
            None => match f(v, &["corruption", "pass_q20"]) {
                Some(d) => format!(
                    "no head — dial-alone pass_q20 {:.1}% (broken-by-design post-720)",
                    100.0 * d
                ),
                None => "no corruption grid".into(),
            },
        },
    ));
    info.push((
        "HF-NL 0.1931 comparator (context)".into(),
        match f(v, &["rank", "hfnlproxy", "per_ref_mean"]) {
            Some(x) => format!(
                "{} the arm-B comparator ({x:.4} vs {})",
                if x >= balanced::HFNL_COMPARATOR {
                    "at/above"
                } else {
                    "below"
                },
                balanced::HFNL_COMPARATOR
            ),
            None => "not measured".into(),
        },
    ));
    // SIGNED, always — read AGAINST each corpus's DECLARED orientation
    // (`EXPECTED_ORIENTATION`, see `DISTORTION_ORIENTED` above). On a
    // quality-oriented corpus a negative SROCC is a genuine ranking INVERSION,
    // and an unsigned guard row hides it — which is exactly how 110 of 188
    // board bakes sat anti-correlated with KADID's real human MOS while every
    // one displayed a positive magnitude
    // (`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F).
    // On a DISTORTION-oriented JND-family corpus the convention is mirrored:
    // negative is its declared native direction (labelled, never flagged), and
    // a POSITIVE signed value is the defect. A guard that yelled INVERTED at
    // the convention (as this one did for sdr25 until 2026-08-05) is a
    // standing false alarm that teaches readers to ignore the row. Guards stay
    // UNSCORED — the requirement is only that defect vs convention be READABLE.
    let guard = |c: &str| match f(v, &["rank", c, "srocc_signed"])
        .or_else(|| f(v, &["rank", c, "srocc"]))
    {
        Some(x) if is_distortion_oriented(c) && x > 0.0 => {
            format!("{x:+.4} ORIENTATION MISMATCH (positive on a JND\u{2193} corpus)")
        }
        Some(x) if is_distortion_oriented(c) => {
            format!("{x:+.4} (JND\u{2193} convention; |SROCC| {:.4})", x.abs())
        }
        Some(x) if x < 0.0 => format!("{x:+.4} INVERTED"),
        Some(x) => format!("{x:+.4}"),
        None => "—".into(),
    };
    info.push((
        "KADID/TID (t=v integrity guards, dimmed; SIGNED)".into(),
        format!(
            "kadid {} / tid {} — never scored; negative = ANTI-CORRELATED with the corpus's human labels",
            guard("kadid"),
            guard("tid")
        ),
    ));
    info.push((
        "sdr25 (within-family selector only; \u{2282} aic4 — Appendix I)".into(),
        guard("sdr25"),
    ));
    info.push((
        "packaging".into(),
        match v.get("model").and_then(|m| m.get("output_spline")) {
            Some(s) if !s.is_null() => "output spline present".into(),
            _ => "spline-less raw head — needs `bake_dial_refit add-spline` \
                  (+ rank-invariance check) before G-RANGE is defined"
                .into(),
        },
    ));
    let repro = match (v.get("repro").map(|r| !r.is_null()).unwrap_or(false), class) {
        (true, "944-ensemble") => "anchor-member only (ensemble)",
        (true, _) => "present",
        (false, _) => "absent",
    };
    info.push(("repro".into(), repro.into()));
    match f(v, &["composite"]) {
        Some(c) => info.push(("product_composite (§1)".into(), format!("{c:.4}"))),
        None => info.push(("product_composite (§1)".into(), "null".into())),
    }

    let composite = balanced_composite(v);
    let annotations = matched
        .iter()
        .map(|e| (e.id.clone(), e.kind.clone(), e.reason.clone()))
        .collect();
    BalancedReport {
        class,
        class_note,
        floors,
        info,
        composite,
        annotations,
    }
}

// ── `--select`: the REGISTERED k-seed selection rule (campaign appendix
// E.4, frozen 2026-08-04 before any ranking existed) ───────────────────────
//
// The campaign's k-seed rule was "train k seeds → select by sdr25 /
// best_val". The coherence study then established that M3a is a SELECTABLE
// trajectory property (42.3 % of 944-class M3a variance is seed noise at
// fixed recipe; `C_co3a` k = 6 spans 0.718–0.826), so selection must account
// for it. This is that rule, in the bar/profile owner rather than a new
// script.
//
//   PRIMARY    profile floor count (`n_pass`). Coherence never overrides a
//              bake that fails CID22 or the dial.
//   TIE-BREAK  selection_composite = balanced_composite + W_M3A · m3a.
//   sdr25      is NOT in the rule — it stays a reported comparator column.
//              (Standing caveat: sdr25 has decoupled from CID22 five times;
//              that is exactly why the primary is the floor count and not a
//              proxy corpus.)
//
// This computes NO statistics: every input is a number an owning tool
// already produced.

/// The three M3a states. They are DISTINCT, and none of them is zero — a
/// missing measurement must never score as "perfectly incoherent".
#[derive(PartialEq, Clone, Copy)]
enum M3aState {
    /// A number is present: rank normally.
    Measured(f64),
    /// Ensemble — the coherence instrument loads ONE ZNPR, so it
    /// structurally cannot produce a value. Ranked in a separate section on
    /// `balanced_composite` alone; never penalized.
    NotComputable,
    /// Non-ensemble with no measurement: eligible to be LISTED, not to be
    /// SELECTED. Precedent: the balanced profile already counts an absent
    /// floor axis as not-passed — a candidate nobody measured cannot be
    /// certified on that axis.
    Unmeasured,
}

fn m3a_state(v: &serde_json::Value, class: &str) -> M3aState {
    if class == "944-ensemble" {
        return M3aState::NotComputable;
    }
    match f(v, &["m3a_coherence"]) {
        Some(x) => M3aState::Measured(x),
        None => M3aState::Unmeasured,
    }
}

struct SelectRow {
    name: String,
    path: String,
    class: &'static str,
    n_pass: usize,
    n_floors: usize,
    composite: Option<f64>,
    m3a: M3aState,
    /// `balanced_composite + W_M3A·m3a`; None when either term is absent.
    selection_composite: Option<f64>,
    sdr25: Option<f64>,
    bake: Option<String>,
}

fn m3a_cell(s: M3aState) -> String {
    match s {
        M3aState::Measured(x) => format!("{x:.4}"),
        M3aState::NotComputable => "NOT COMPUTABLE".into(),
        M3aState::Unmeasured => "UNMEASURED".into(),
    }
}

/// Rank a pool: floor count DESC, then `selection_composite` DESC. Rows
/// without a `selection_composite` sort last within their floor tier (they
/// carry no comparable number) — they are listed, never selected.
fn rank_pool(rows: &mut [&SelectRow]) {
    rows.sort_by(|a, b| {
        b.n_pass.cmp(&a.n_pass).then_with(|| {
            b.selection_composite
                .partial_cmp(&a.selection_composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
}

const SELECT_TSV_COLS: &str = "rank\tpool\tname\tclass\tn_pass\tbal_composite\tm3a\tm3a_state\tselection_composite\tsdr25\tselectable\tpath";

const TSV_COLS: &str = "name\tclass\tverdict\tn_pass\tcid22\tkonjnd_abs\tnonphoto\tcsiq\tlive\thfnl_perref\tb3\tb3_n\tb9\tb9_n\tmono\ttied\tdynrange\tm3a\tm3a_tier\tcorr_head_q20\tbal_composite\tproduct_composite\tsdr25\tkadid_signed\ttid_signed\tspline\trepro\tfails\tn_measured\tabsent\tannotations\tblocks\tdominated_by";

/// Compact carry of the promoter-injected `block_profile` (used-counts per
/// family, `f0_155/f156_371/f372_719/f720_943`) — computed by
/// `bake_block_profile`, never here. "-" when the fulleval has none.
fn blocks_summary(v: &serde_json::Value) -> String {
    let fam = |name: &str| {
        f(v, &["block_profile", "families", name, "used"])
            .map(|x| format!("{}", x as i64))
            .unwrap_or_else(|| "·".into())
    };
    if v.get("block_profile")
        .map(|b| !b.is_null())
        .unwrap_or(false)
    {
        format!(
            "{}/{}/{}/{}",
            fam("f0_155"),
            fam("f156_371"),
            fam("f372_719"),
            fam("f720_943")
        )
    } else {
        "-".into()
    }
}

fn tsv_row(v: &serde_json::Value, r: &BalancedReport) -> String {
    let n_pass = r.floors.iter().filter(|x| x.pass).count();
    // Measured fails only — absent-not-failed floors move to the `absent`
    // column (distinct from a measured fail; still not-passed in n_pass).
    let fails: Vec<&str> = r
        .floors
        .iter()
        .filter(|x| !x.pass && !x.absent_not_failed)
        .map(|x| x.id)
        .collect();
    let absent: Vec<&str> = r
        .floors
        .iter()
        .filter(|x| x.absent_not_failed)
        .map(|x| x.id)
        .collect();
    let n_measured = r.floors.iter().filter(|x| !x.absent_not_failed).count();
    let num = |o: Option<f64>| o.map(|x| format!("{x:.5}")).unwrap_or_else(|| "-".into());
    let b3 = cid22_band(v, "B3");
    let b9 = cid22_band(v, "B9");
    let m3a = f(v, &["m3a_coherence"]);
    let tier: String = if r.class == "944-ensemble" {
        "not-computable".into()
    } else {
        match m3a {
            Some(x) if x >= balanced::M3A_GOLD => "gold".into(),
            Some(x) if x >= balanced::M3A_SILVER => "silver".into(),
            Some(_) => "flagged".into(),
            None => "-".into(),
        }
    };
    let spline = match v.get("model").and_then(|m| m.get("output_spline")) {
        Some(s) if !s.is_null() => "present",
        _ => "none",
    };
    let repro = match (
        v.get("repro").map(|x| !x.is_null()).unwrap_or(false),
        r.class,
    ) {
        (true, "944-ensemble") => "anchor-only",
        (true, _) => "present",
        (false, _) => "absent",
    };
    [
        bake_name(v).to_string(),
        r.class.to_string(),
        if n_pass == r.floors.len() {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        format!("{n_pass}/{}", r.floors.len()),
        num(f(v, &["rank", "cid22", "srocc"])),
        num(f(v, &["rank", "konjnd", "srocc"]).map(f64::abs)),
        num(f(v, &["rank", "nonphoto", "srocc"])),
        num(f(v, &["rank", "csiq", "srocc"])),
        num(f(v, &["rank", "live", "srocc"])),
        num(f(v, &["rank", "hfnlproxy", "per_ref_mean"])),
        num(b3.map(|x| x.0)),
        b3.map(|x| x.1.to_string()).unwrap_or_else(|| "-".into()),
        num(b9.map(|x| x.0)),
        b9.map(|x| x.1.to_string()).unwrap_or_else(|| "-".into()),
        num(f(v, &["dial", "mono_pct"])),
        num(f(v, &["dial", "tied_pct"])),
        num(f(v, &["dial", "dynamic_range"])),
        num(m3a),
        tier,
        num(f(v, &["corruption_head", "pass_q20"])),
        num(r.composite),
        num(f(v, &["composite"])),
        num(f(v, &["rank", "sdr25", "srocc"])),
        // SIGNED (2026-08-04, APPENDIX F): an unsigned kadid/tid column cannot show an
        // anti-correlated bake, and the ext-lineage KADID target was found inverted.
        num(f(v, &["rank", "kadid", "srocc_signed"]).or_else(|| f(v, &["rank", "kadid", "srocc"]))),
        num(f(v, &["rank", "tid", "srocc_signed"]).or_else(|| f(v, &["rank", "tid", "srocc"]))),
        spline.to_string(),
        repro.to_string(),
        if fails.is_empty() {
            "-".to_string()
        } else {
            fails.join(",")
        },
        format!("{n_pass}/{n_measured}"),
        if absent.is_empty() {
            "-".to_string()
        } else {
            absent.join(",")
        },
        if r.annotations.is_empty() {
            "-".to_string()
        } else {
            r.annotations
                .iter()
                .map(|(id, _, _)| id.as_str())
                .collect::<Vec<_>>()
                .join(",")
        },
        blocks_summary(v),
        match v.get("dominated_by").and_then(|d| d.as_array()) {
            Some(a) if !a.is_empty() => a
                .iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(","),
            _ => "-".to_string(),
        },
    ]
    .join("\t")
}

/// Annotations registry resolution: explicit `--annotations <path|none>`
/// wins; default = `$ZENSIM_EVAL_ANNOTATIONS`, else the committed
/// `./benchmarks/eval_annotations.json` if present, else none (noted out
/// loud). Shared by the balanced-profile and `--select` paths.
fn load_annotations_arg(arg: Option<&str>) -> Vec<AnnEntry> {
    match arg {
        Some("none") => Vec::new(),
        Some(p) => match load_annotations(std::path::Path::new(p)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("freeze_check: --annotations {e}");
                std::process::exit(2);
            }
        },
        None => {
            let default = std::env::var("ZENSIM_EVAL_ANNOTATIONS")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("benchmarks/eval_annotations.json"));
            if default.exists() {
                match load_annotations(&default) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("freeze_check: default annotations {e}");
                        std::process::exit(2);
                    }
                }
            } else {
                eprintln!(
                    "freeze_check: note — no annotations registry at {} (pass --annotations)",
                    default.display()
                );
                Vec::new()
            }
        }
    }
}

/// `--select` driver: read N fullevals, apply the registered rule, print the
/// ranked table (+ optional TSV). Exit 0 if a selectable winner exists,
/// 1 if none does (every candidate UNMEASURED or zero-floor), 2 on usage.
fn run_select(paths: &[PathBuf], anns: &[AnnEntry], tsv: bool) -> i32 {
    let mut rows: Vec<SelectRow> = Vec::new();
    for p in paths {
        let bytes = match std::fs::read(p) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("freeze_check --select: read {}: {e}", p.display());
                return 2;
            }
        };
        let v: serde_json::Value = match serde_json::from_slice(&bytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("freeze_check --select: parse {}: {e}", p.display());
                return 2;
            }
        };
        let r = eval_balanced(&v, anns);
        let m3a = m3a_state(&v, r.class);
        let selection_composite = match (r.composite, m3a) {
            (Some(c), M3aState::Measured(x)) => Some(c + balanced::W_M3A * x),
            _ => None,
        };
        rows.push(SelectRow {
            name: bake_name(&v).to_string(),
            path: p.display().to_string(),
            class: r.class,
            n_pass: r.floors.iter().filter(|x| x.pass).count(),
            n_floors: r.floors.len(),
            composite: r.composite,
            m3a,
            selection_composite,
            sdr25: f(&v, &["rank", "sdr25", "srocc"]).map(f64::abs),
            bake: v.get("bake").and_then(|x| x.as_str()).map(str::to_string),
        });
    }

    // Two pools: ensembles rank on balanced_composite alone (their
    // selection_composite is on a DIFFERENT scale — mixing them would be a
    // category error), everything else on the registered rule.
    let (ens, single): (Vec<&SelectRow>, Vec<&SelectRow>) =
        rows.iter().partition(|r| r.m3a == M3aState::NotComputable);
    let mut single = single;
    let mut ens = ens;
    rank_pool(&mut single);
    ens.sort_by(|a, b| {
        b.n_pass.cmp(&a.n_pass).then_with(|| {
            b.composite
                .partial_cmp(&a.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });

    let num = |x: Option<f64>| x.map_or("—".into(), |v| format!("{v:.4}"));
    println!("# freeze_check --select — REGISTERED rule (campaign appendix E.4)\n");
    println!(
        "PRIMARY: profile floor count. TIE-BREAK: selection_composite = \
         balanced_composite + {:.2}·M3a.\nsdr25 is a reported comparator, \
         NOT part of the rule.\n",
        balanced::W_M3A
    );
    println!("| rank | bake | class | floors | bal_comp | M3a | sel_comp | sdr25 | selectable |");
    println!("|---:|---|---|---:|---:|---|---:|---:|---|");
    let mut winner: Option<&SelectRow> = None;
    for (i, r) in single.iter().enumerate() {
        let selectable = r.m3a != M3aState::Unmeasured && r.n_pass > 0;
        if selectable && winner.is_none() {
            winner = Some(r);
        }
        println!(
            "| {} | {} | {} | {}/{} | {} | {} | {} | {} | {} |",
            i + 1,
            r.name,
            r.class,
            r.n_pass,
            r.n_floors,
            num(r.composite),
            m3a_cell(r.m3a),
            num(r.selection_composite),
            num(r.sdr25),
            if selectable { "yes" } else { "NO" }
        );
    }
    if !ens.is_empty() {
        println!(
            "\n## Ensembles — ranked SEPARATELY on balanced_composite alone\n\n\
             M3a is NOT COMPUTABLE for an ensemble (the coherence instrument \
             loads one ZNPR). They are never penalized for that and never \
             mixed into the ranking above — the two composites are on \
             different scales.\n"
        );
        println!("| rank | bake | floors | bal_comp | sdr25 |");
        println!("|---:|---|---:|---:|---:|");
        for (i, r) in ens.iter().enumerate() {
            println!(
                "| {} | {} | {}/{} | {} | {} |",
                i + 1,
                r.name,
                r.n_pass,
                r.n_floors,
                num(r.composite),
                num(r.sdr25)
            );
        }
    }

    let unmeasured: Vec<&&SelectRow> = single
        .iter()
        .filter(|r| r.m3a == M3aState::Unmeasured)
        .collect();
    if !unmeasured.is_empty() {
        println!(
            "\n## NOT SELECTABLE — {} candidate(s) have no M3a\n\n\
             A candidate nobody measured cannot be certified on that axis \
             (same registered rule the balanced profile applies to an absent \
             floor). Measure, then re-run:\n",
            unmeasured.len()
        );
        for r in &unmeasured {
            match &r.bake {
                Some(b) => println!("    scripts/run_full_eval.sh {b} {} 944", r.name),
                None => println!(
                    "    # {}: fulleval carries no `bake` path — re-harvest it",
                    r.name
                ),
            }
        }
    }

    match winner {
        Some(w) => println!(
            "\n**SELECTED: `{}`** — {}/{} floors, selection_composite {}.",
            w.name,
            w.n_pass,
            w.n_floors,
            num(w.selection_composite)
        ),
        None => println!("\n**NO SELECTABLE CANDIDATE** (every row is UNMEASURED or 0-floor)."),
    }

    if tsv {
        eprintln!("{SELECT_TSV_COLS}");
        let emit = |pool: &str, i: usize, r: &SelectRow| {
            let selectable = r.m3a != M3aState::Unmeasured && r.n_pass > 0;
            let n = |x: Option<f64>| x.map_or("-".into(), |v| format!("{v:.6}"));
            eprintln!(
                "{}\t{pool}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                i + 1,
                r.name,
                r.class,
                r.n_pass,
                n(r.composite),
                match r.m3a {
                    M3aState::Measured(x) => format!("{x:.6}"),
                    _ => "-".into(),
                },
                match r.m3a {
                    M3aState::Measured(_) => "measured",
                    M3aState::NotComputable => "not-computable",
                    M3aState::Unmeasured => "unmeasured",
                },
                n(r.selection_composite),
                n(r.sdr25),
                if selectable { "yes" } else { "no" },
                r.path
            );
        };
        for (i, r) in single.iter().enumerate() {
            emit("single", i, r);
        }
        for (i, r) in ens.iter().enumerate() {
            emit("ensemble", i, r);
        }
    }
    i32::from(winner.is_none())
}

fn main() {
    let mut fulleval: Option<PathBuf> = None;
    let mut bar_csiq: Option<f64> = None;
    let mut bar_live: Option<f64> = None;
    let mut profile: Option<String> = None;
    let mut tsv = false;
    let mut annotations_arg: Option<String> = None;
    let mut select: Vec<PathBuf> = Vec::new();
    let mut in_select = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        // After `--select`, bare paths accumulate until the next flag.
        if in_select && !a.starts_with("--") {
            select.push(PathBuf::from(a));
            continue;
        }
        in_select = false;
        match a.as_str() {
            "--fulleval" => fulleval = args.next().map(PathBuf::from),
            "--select" => in_select = true,
            "--profile" => profile = args.next(),
            "--annotations" => annotations_arg = args.next(),
            "--tsv" => tsv = true,
            "--select-tsv-header" => {
                println!("{SELECT_TSV_COLS}");
                std::process::exit(0);
            }
            "--tsv-header" => {
                println!("{TSV_COLS}");
                std::process::exit(0);
            }
            "--bar" => {
                let kv = match args.next() {
                    Some(v) => v,
                    None => usage(),
                };
                let (k, v) = match kv.split_once('=') {
                    Some(p) => p,
                    None => usage(),
                };
                let val: f64 = match v.parse() {
                    Ok(x) => x,
                    Err(_) => usage(),
                };
                match k {
                    "csiq" => bar_csiq = Some(val),
                    "live" => bar_live = Some(val),
                    _ => usage(),
                }
            }
            "-h" | "--help" => usage(),
            _ => usage(),
        }
    }
    match profile.as_deref() {
        None | Some("balanced-2026-08-04") => {}
        Some(other) => {
            eprintln!("freeze_check: unknown profile `{other}` (known: balanced-2026-08-04)");
            std::process::exit(2);
        }
    }
    // ── `--select` path: N fullevals in, one ranked table out ───────────
    if !select.is_empty() {
        if fulleval.is_some() {
            eprintln!("freeze_check: --select and --fulleval are mutually exclusive");
            std::process::exit(2);
        }
        // --select always ranks under the registered floor set; the §5 bar
        // set has ATTACH rows and cannot rank.
        if matches!(profile.as_deref(), Some(p) if p != "balanced-2026-08-04") {
            eprintln!("freeze_check: --select supports only --profile balanced-2026-08-04");
            std::process::exit(2);
        }
        let anns = load_annotations_arg(annotations_arg.as_deref());
        std::process::exit(run_select(&select, &anns, tsv));
    }
    if tsv && profile.is_none() {
        eprintln!(
            "freeze_check: --tsv requires --profile (the §5 table has ATTACH rows a TSV cannot carry)"
        );
        std::process::exit(2);
    }
    let path = match fulleval {
        Some(p) => p,
        None => usage(),
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("freeze_check: read {}: {e}", path.display());
            std::process::exit(2);
        }
    };
    let v: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("freeze_check: parse {}: {e}", path.display());
            std::process::exit(2);
        }
    };

    // ── Balanced profile path (AMENDMENT 8) ─────────────────────────────
    if profile.is_some() {
        let anns = load_annotations_arg(annotations_arg.as_deref());
        let r = eval_balanced(&v, &anns);
        let n_fail = r.floors.iter().filter(|x| !x.pass).count();
        if tsv {
            println!("{}", tsv_row(&v, &r));
        } else {
            let bake = bake_name(&v);
            println!("# Balanced-selection profile `balanced-2026-08-04` — `{bake}`\n");
            println!(
                "Class: **{}** — {}. Floors: sota944 campaign AMENDMENT 8 (§8.1, \
                 registered 2026-08-04 BEFORE scoring; user-directed policy). §1 \
                 stays the freeze bar — this is the balanced SELECTION surface.\n",
                r.class, r.class_note
            );
            println!("| floor | bar | measured | verdict |");
            println!("|---|---|---|:--:|");
            for x in &r.floors {
                println!(
                    "| {} | {} | {} | {} |",
                    x.gate,
                    x.bar,
                    x.measured,
                    if x.pass {
                        "PASS"
                    } else if x.absent_not_failed {
                        "ABSENT (not passed)"
                    } else {
                        "**FAIL**"
                    }
                );
            }
            if !r.annotations.is_empty() {
                println!("\n| ⚠ annotation | kind | reason |");
                println!("|---|---|---|");
                for (id, kind, reason) in &r.annotations {
                    println!("| `{id}` | {kind} | {reason} |");
                }
            }
            println!("\n| reported (never floors) | value |");
            println!("|---|---|");
            for (k, val) in &r.info {
                println!("| {k} | {val} |");
            }
            match r.composite {
                Some(c) => println!("\n**balanced_composite = {c:.4}** (registered §8.1 weights)"),
                None => println!("\nbalanced_composite: not computable (no terms present)"),
            }
            let absent_ids: Vec<&str> = r
                .floors
                .iter()
                .filter(|x| x.absent_not_failed)
                .map(|x| x.id)
                .collect();
            let n_measured = r.floors.len() - absent_ids.len();
            print!(
                "\n{} of {} floors pass",
                r.floors.len() - n_fail,
                r.floors.len()
            );
            if !absent_ids.is_empty() {
                // Both forms, per the registry convention: n/8 keeps the
                // registered absent=not-passed rule; n/m-measured states the
                // measured record.
                print!(
                    " ({}/{n_measured}-measured; absent-not-failed: {})",
                    r.floors.len() - n_fail,
                    absent_ids.join(",")
                );
            }
            println!(
                "{}",
                if n_fail == 0 {
                    " — BALANCED-PROFILE PASS"
                } else {
                    ""
                }
            );
        }
        std::process::exit(if n_fail > 0 { 1 } else { 0 });
    }

    // ── §5 default path (unchanged) ─────────────────────────────────────
    let rows = legacy_rows(&v, bar_csiq, bar_live);

    let bake = bake_name(&v);
    println!("# Freeze-bar summary — `{bake}`\n");
    println!(
        "Bars: zenpapers final-metric plan §5 (2026-07-31). Owners: \
         benchmarks/decision_surface_audit_2026-07-31.md. ATTACH rows need \
         evidence from their named owner — they gate the freeze too.\n"
    );
    println!("| gate | bar | measured / evidence | verdict |");
    println!("|---|---|---|:--:|");
    let (mut n_fail, mut n_attach) = (0usize, 0usize);
    for r in &rows {
        match r {
            Row::Eval(g, b, m, ok) => {
                if !ok {
                    n_fail += 1;
                }
                println!(
                    "| {g} | {b} | {m} | {} |",
                    if *ok { "PASS" } else { "**FAIL**" }
                );
            }
            Row::Attach(g, b, w) => {
                n_attach += 1;
                println!("| {g} | {b} | {w} | ATTACH |");
            }
            Row::Info(g, m) => println!("| {g} | — | {m} | info |"),
        }
    }
    println!(
        "\n{} evaluable FAIL(s); {} row(s) awaiting attached evidence. \
         Freeze requires zero FAILs AND every ATTACH row's evidence in hand.",
        n_fail, n_attach
    );
    std::process::exit(if n_fail > 0 { 1 } else { 0 });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `DISTORTION_ORIENTED` is a gated mirror of the Python
    /// `EXPECTED_ORIENTATION` registry (the owner —
    /// `scripts/canonical_corpus/check_target_orientation.py`). Parse the
    /// owner file and fail on ANY drift, in either direction. The parse is
    /// deliberately dumb (line-based over the literal dict block) so it
    /// breaks LOUDLY if the registry's shape changes, rather than silently
    /// matching nothing.
    #[test]
    fn distortion_oriented_mirror_matches_python_registry() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../scripts/canonical_corpus/check_target_orientation.py"
        );
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("owner registry {path} must exist in-repo: {e}"));
        let start = text
            .find("EXPECTED_ORIENTATION = {")
            .expect("EXPECTED_ORIENTATION dict literal not found in the owner file");
        let block = &text[start..];
        let end = block.find('}').expect("registry dict literal never closes");
        let block = &block[..end];
        let mut dist: Vec<String> = Vec::new();
        let mut n_quality = 0usize;
        for line in block.lines() {
            let line = line.split('#').next().unwrap_or("");
            if let Some((k, val)) = line.split_once(':') {
                let key = k.trim().trim_matches('"').trim_matches('\'');
                if key.is_empty() {
                    continue;
                }
                match val.trim().trim_end_matches(',').trim() {
                    "DISTORTION" => dist.push(key.to_string()),
                    "QUALITY" => n_quality += 1,
                    _ => {}
                }
            }
        }
        assert!(
            n_quality >= 5,
            "parse sanity: expected several QUALITY entries, parsed {n_quality} — \
             did the registry's literal format change?"
        );
        dist.sort();
        let mut mirror: Vec<String> = DISTORTION_ORIENTED.iter().map(|s| s.to_string()).collect();
        mirror.sort();
        assert_eq!(
            mirror, dist,
            "freeze_check's DISTORTION_ORIENTED mirror drifted from the Python \
             EXPECTED_ORIENTATION registry — update the mirror (the registry is the owner)"
        );
    }

    /// A fixture that passes every registered floor exactly at/above the line.
    fn passing_fixture() -> serde_json::Value {
        json!({
            "name": "FIX_single",
            "n_inputs": 944,
            "model": { "n_inputs": 944, "output_spline": null },
            "repro": { "seed": 1 },
            "composite": 0.85,
            "m3a_coherence": 0.80,
            "rank": {
                "cid22": { "srocc": 0.885, "bands": [
                    { "band": "B3", "srocc": 0.0, "n": 57 },
                    { "band": "B9", "srocc": 0.15, "n": 43 }
                ]},
                "konjnd": { "srocc": -0.43 },
                "nonphoto": { "srocc": 0.90 },
                "csiq": { "srocc": 0.83 },
                "live": { "srocc": 0.83 },
                "imazen26": { "srocc": 0.91 },
                "aic3": { "srocc": 0.79 },
                "aic4": { "srocc": 0.91 },
                "sdr25": { "srocc": 0.93 },
                "kadid": { "srocc": 0.32 },
                "tid": { "srocc": 0.88 },
                "hfnlproxy": { "srocc": 0.5, "per_ref_mean": 0.0 }
            },
            "dial": { "mono_pct": 0.93, "tied_pct": 0.05, "dynamic_range": 15.0 }
        })
    }

    #[test]
    fn balanced_floors_resolve_as_registered_and_fixture_passes() {
        let v = passing_fixture();
        let r = eval_balanced(&v, &[]);
        assert_eq!(r.class, "944-single");
        assert_eq!(r.class_note, "shippable single bake");
        assert_eq!(r.floors.len(), 8, "eight registered floors F1..F8");
        for x in &r.floors {
            assert!(
                x.pass,
                "floor {} should pass at the registered edge: {}",
                x.id, x.measured
            );
        }
        // Each floor flips to FAIL just below its registered line.
        let cases: Vec<(&str, serde_json::Value)> = vec![
            ("cid22", json!({"rank": {"cid22": {"srocc": 0.8849}}})),
            ("konjnd", json!({"rank": {"konjnd": {"srocc": 0.4299}}})),
            ("nonphoto", json!({"rank": {"nonphoto": {"srocc": 0.8999}}})),
            ("dial", json!({"dial": {"mono_pct": 0.9299}})),
            ("dial", json!({"dial": {"tied_pct": 0.0501}})),
            ("dialrange", json!({"dial": {"dynamic_range": 120.1}})),
            ("dialrange", json!({"dial": {"dynamic_range": 0.9}})),
            (
                "hfnl",
                json!({"rank": {"hfnlproxy": {"per_ref_mean": -0.0001}}}),
            ),
            ("breadth", json!({"rank": {"csiq": {"srocc": 0.8299}}})),
            ("breadth", json!({"rank": {"live": {"srocc": 0.8299}}})),
        ];
        for (id, patch) in cases {
            let mut v = passing_fixture();
            merge(&mut v, &patch);
            let r = eval_balanced(&v, &[]);
            let fl = r.floors.iter().find(|x| x.id == id).unwrap();
            assert!(!fl.pass, "floor {id} must FAIL under patch {patch}");
            assert_eq!(
                r.floors.iter().filter(|x| !x.pass).count(),
                1,
                "exactly one floor fails for patch on {id}"
            );
        }
        // band-tail floor: B9 below 0.15 fails; B3 below 0.0 fails.
        for bands in [
            json!([{ "band": "B3", "srocc": 0.0, "n": 57 },
                   { "band": "B9", "srocc": 0.1499, "n": 43 }]),
            json!([{ "band": "B3", "srocc": -0.001, "n": 57 },
                   { "band": "B9", "srocc": 0.2, "n": 43 }]),
        ] {
            let mut v = passing_fixture();
            v["rank"]["cid22"]["bands"] = bands;
            let r = eval_balanced(&v, &[]);
            let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
            assert!(!fl.pass);
        }
    }

    #[test]
    fn dyn_range_497_class_fails_on_range_and_tied() {
        // The named pathological cell class: dynamic_range 497, tied 15.6%,
        // mono 93.7% — ranks high, broken dial. F5 + F4 must both catch it.
        let mut v = passing_fixture();
        merge(
            &mut v,
            &json!({"dial": {"dynamic_range": 496.6, "tied_pct": 0.156, "mono_pct": 0.937}}),
        );
        let r = eval_balanced(&v, &[]);
        let by = |id: &str| r.floors.iter().find(|x| x.id == id).unwrap().pass;
        assert!(!by("dialrange"), "497-class span must fail F5");
        assert!(!by("dial"), "15.6% tied must fail F4");
    }

    #[test]
    fn missing_axis_is_unevaluable_not_passed() {
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(!fl.pass);
        assert!(fl.measured.contains("UNEVALUABLE"));
    }

    // ── Annotations registry (board-integrity pass 2026-08-04) ──────────

    fn ann(id: &str, kind: &str, fields: &[&str], scope: serde_json::Value) -> AnnEntry {
        AnnEntry {
            id: id.into(),
            kind: kind.into(),
            fields: fields.iter().map(|s| s.to_string()).collect(),
            scope,
            reason: format!("test reason for {id}"),
        }
    }

    #[test]
    fn annotation_scope_predicates_match_as_documented() {
        let v = passing_fixture();
        // missing: fixture HAS rank.hfnlproxy → no match; remove → match.
        let e = ann("x", "annotated", &[], json!({"missing": "rank.hfnlproxy"}));
        assert!(!ann_matches(&v, &e));
        let mut v2 = passing_fixture();
        v2["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        assert!(ann_matches(&v2, &e));
        // explicit null counts as missing (model.output_spline is null here).
        let e_null = ann(
            "x",
            "annotated",
            &[],
            json!({"missing": "model.output_spline"}),
        );
        assert!(ann_matches(&v, &e_null));
        // present: mirror of missing.
        let e_p = ann("x", "annotated", &[], json!({"present": "rank.hfnlproxy"}));
        assert!(ann_matches(&v, &e_p));
        assert!(!ann_matches(&v2, &e_p));
        // names: exact bake-name list.
        let e_n = ann(
            "x",
            "annotated",
            &[],
            json!({"names": ["FIX_single", "other"]}),
        );
        assert!(ann_matches(&v, &e_n));
        let e_n2 = ann("x", "annotated", &[], json!({"names": ["nope"]}));
        assert!(!ann_matches(&v, &e_n2));
        // all.
        assert!(ann_matches(
            &v,
            &ann("x", "annotated", &[], json!({"all": true}))
        ));
        // empty / null scope matches nothing.
        assert!(!ann_matches(&v, &ann("x", "annotated", &[], json!({}))));
        // coverage: segment-boundary prefix only.
        assert!(ann_covers("rank.hfnlproxy", "rank.hfnlproxy.per_ref_mean"));
        assert!(ann_covers("rank.hfnlproxy", "rank.hfnlproxy"));
        assert!(!ann_covers("rank.hfnl", "rank.hfnlproxy.per_ref_mean"));
        assert!(!ann_covers("rank.hfnlproxy.per_ref_mean", "rank.hfnlproxy"));
    }

    #[test]
    fn absent_axis_with_registry_entry_is_absent_not_failed_both_forms() {
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        let anns = vec![ann(
            "hfnl-absent-not-failed",
            "absent-not-failed",
            &["rank.hfnlproxy"],
            json!({"missing": "rank.hfnlproxy"}),
        )];
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        // Still not-passed for n/8 (registered rule) — but ABSENT, not a
        // measured fail, and printed as such.
        assert!(!fl.pass);
        assert!(fl.absent_not_failed);
        assert!(fl.measured.contains("absent"), "got: {}", fl.measured);
        assert_eq!(r.floors.iter().filter(|x| x.pass).count(), 7);
        // TSV: hfnl moves from `fails` to `absent`; both n-forms carried.
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        assert_eq!(cols.len(), hdr.len(), "TSV row width matches header");
        let at = |name: &str| cols[hdr.iter().position(|h| *h == name).unwrap()];
        assert_eq!(at("n_pass"), "7/8", "registered absent=not-passed form");
        assert_eq!(at("n_measured"), "7/7", "measured-record form");
        assert_eq!(at("absent"), "hfnl");
        assert!(
            !at("fails").split(',').any(|x| x == "hfnl"),
            "fails: {}",
            at("fails")
        );
        assert!(at("annotations").contains("hfnl-absent-not-failed"));
    }

    #[test]
    fn measured_fail_never_becomes_absent() {
        // A MEASURED fail (negative hfnl) is untouched even when an
        // absent-not-failed entry matches the cell broadly (`all`) and covers
        // the field — absence is a property of the VALUE, not the entry.
        let mut v = passing_fixture();
        merge(
            &mut v,
            &json!({"rank": {"hfnlproxy": {"per_ref_mean": -0.0001}}}),
        );
        let anns = vec![ann(
            "hfnl-absent-not-failed",
            "absent-not-failed",
            &["rank.hfnlproxy"],
            json!({"all": true}),
        )];
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(!fl.pass);
        assert!(!fl.absent_not_failed, "measured fail must stay a fail");
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let at = |name: &str| cols[hdr.iter().position(|h| *h == name).unwrap()];
        assert!(at("fails").split(',').any(|x| x == "hfnl"));
        assert_eq!(at("absent"), "-");
        assert_eq!(at("n_measured"), "7/8", "all 8 floors measured");
    }

    #[test]
    fn annotated_kind_flags_without_changing_verdicts() {
        // dial-mono-raw-unit shape: matches spline-less bakes, flags the dial
        // fields, changes NO floor verdict.
        let v = passing_fixture(); // output_spline: null → spline-less
        let anns = vec![ann(
            "dial-mono-raw-unit",
            "annotated",
            &["dial.mono_pct", "dial.tied_pct"],
            json!({"missing": "model.output_spline"}),
        )];
        let r = eval_balanced(&v, &anns);
        let r_plain = eval_balanced(&v, &[]);
        for (a, b) in r.floors.iter().zip(r_plain.floors.iter()) {
            assert_eq!(a.pass, b.pass, "floor {} verdict must not change", a.id);
        }
        assert!(
            r.annotations
                .iter()
                .any(|(id, kind, _)| id == "dial-mono-raw-unit" && kind == "annotated")
        );
        let row = tsv_row(&v, &r);
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let cols: Vec<&str> = row.split('\t').collect();
        let ann_i = hdr.iter().position(|h| *h == "annotations").unwrap();
        assert_eq!(
            cols[ann_i], "dial-mono-raw-unit",
            "annotations col carries the id"
        );
    }

    #[test]
    fn committed_registry_parses_and_seed_entries_behave() {
        // The COMMITTED registry file must parse, and its three seed entries
        // must do what the campaign doc says they do.
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let anns = load_annotations(&repo.join("benchmarks/eval_annotations.json"))
            .expect("committed registry parses");
        let ids: Vec<&str> = anns.iter().map(|e| e.id.as_str()).collect();
        for want in [
            "dial-mono-raw-unit",
            "hfnl-absent-not-failed",
            "kadid-tid-train-eq-val",
        ] {
            assert!(ids.contains(&want), "registry missing seed entry {want}");
        }
        // era-bridge-shaped cell: spline present, hfnlproxy missing.
        let mut v = passing_fixture();
        v["rank"].as_object_mut().unwrap().remove("hfnlproxy");
        v["model"]["output_spline"] = json!({"knots": 8});
        let r = eval_balanced(&v, &anns);
        let fl = r.floors.iter().find(|x| x.id == "hfnl").unwrap();
        assert!(fl.absent_not_failed, "hfnl absence covered by the registry");
        let matched: Vec<&str> = r.annotations.iter().map(|(i, _, _)| i.as_str()).collect();
        assert!(matched.contains(&"hfnl-absent-not-failed"));
        assert!(
            matched.contains(&"kadid-tid-train-eq-val"),
            "all-scope entry"
        );
        assert!(
            !matched.contains(&"dial-mono-raw-unit"),
            "spline present ⇒ raw-unit entry must NOT match"
        );
    }

    #[test]
    fn ensemble_class_m3a_not_computable() {
        let mut v = passing_fixture();
        v["model"]["kind"] = json!("ensemble");
        let r = eval_balanced(&v, &[]);
        assert_eq!(r.class, "944-ensemble");
        let (_, m3a) = r.info.iter().find(|(k, _)| k.starts_with("M3a")).unwrap();
        assert!(m3a.contains("NOT COMPUTABLE"), "got: {m3a}");
        // distilled + era-bridge classes
        let mut v = passing_fixture();
        v["name"] = json!("C_ensk2_s1303");
        assert_eq!(eval_balanced(&v, &[]).class, "944-distilled");
        let mut v = passing_fixture();
        v["n_inputs"] = json!(372);
        v["model"]["n_inputs"] = json!(372);
        assert_eq!(eval_balanced(&v, &[]).class, "era-bridge");
    }

    /// Packaging-appendix unit annotation: spline-less bakes' F4 numbers are
    /// raw-unit (unit-flattered); spline-bearing bakes are dial-unit. The
    /// floor logic itself is identical in both cases (registered floors do
    /// not move — this is a label).
    #[test]
    fn f4_dial_row_carries_unit_annotation() {
        let v = passing_fixture(); // output_spline: null
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "dial").unwrap();
        assert!(fl.measured.contains("(raw-unit)"), "got: {}", fl.measured);
        let mut v2 = passing_fixture();
        v2["model"]["output_spline"] = json!({"n_knots": 18});
        let r2 = eval_balanced(&v2, &[]);
        let fl2 = r2.floors.iter().find(|x| x.id == "dial").unwrap();
        assert!(
            fl2.measured.contains("(dial-unit)"),
            "got: {}",
            fl2.measured
        );
        assert_eq!(fl.pass, fl2.pass, "annotation never changes the verdict");
    }

    #[test]
    fn band_n_below_30_renders_parenthesized() {
        let mut v = passing_fixture();
        v["rank"]["cid22"]["bands"] = json!([
            { "band": "B3", "srocc": 0.05, "n": 12 },
            { "band": "B9", "srocc": 0.2, "n": 43 }
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(fl.measured.contains("(0.050) n=12"), "got: {}", fl.measured);
        assert!(fl.measured.contains("0.200 n=43"));
    }

    #[test]
    fn composite_matches_registered_weights() {
        let v = passing_fixture();
        // Hand-computed with the §8.1 table (abs corpus terms; signed band-tail):
        let bandtail = (0.0 + 0.15) / 2.0;
        let num = 1.00 * 0.885
            + 0.50 * 0.91
            + 0.30 * 0.90
            + 0.20 * 0.43
            + 0.15 * 0.83
            + 0.15 * 0.83
            + 0.15 * bandtail
            + 0.10 * 0.79
            + 0.05 * 0.91;
        let den = 1.00 + 0.50 + 0.30 + 0.20 + 0.15 + 0.15 + 0.15 + 0.10 + 0.05;
        let expect = num / den;
        let got = balanced_composite(&v).unwrap();
        assert!((got - expect).abs() < 1e-12, "got {got}, expect {expect}");
        // absent term drops from num AND den
        let mut v2 = passing_fixture();
        v2["rank"].as_object_mut().unwrap().remove("aic4");
        let got2 = balanced_composite(&v2).unwrap();
        let expect2 = (num - 0.05 * 0.91) / (den - 0.05);
        assert!((got2 - expect2).abs() < 1e-12);
    }

    #[test]
    fn tsv_row_carries_verdict_and_fails() {
        // (2026-08-04: `fails` looked up by header name — the annotations
        // registry appended n_measured/absent/annotations after it, so
        // `.last()` no longer addresses it. Assertion intent unchanged.)
        let hdr: Vec<&str> = TSV_COLS.split('\t').collect();
        let fails_i = hdr.iter().position(|h| *h == "fails").unwrap();
        let v = passing_fixture();
        let r = eval_balanced(&v, &[]);
        let row = tsv_row(&v, &r);
        let cols: Vec<&str> = row.split('\t').collect();
        assert_eq!(cols.len(), hdr.len());
        assert_eq!(cols[0], "FIX_single");
        assert_eq!(cols[2], "PASS");
        assert_eq!(cols[3], "8/8");
        assert_eq!(cols[fails_i], "-");
        let mut v2 = passing_fixture();
        merge(&mut v2, &json!({"rank": {"cid22": {"srocc": 0.80}}}));
        let r2 = eval_balanced(&v2, &[]);
        let row2 = tsv_row(&v2, &r2);
        let cols2: Vec<&str> = row2.split('\t').collect();
        assert_eq!(cols2[2], "FAIL");
        assert!(cols2[fails_i].contains("cid22"));
    }

    /// The §5 default path is UNCHANGED by the profile addition: lock the row
    /// count, order, gate names and bar semantics of `legacy_rows` (this is
    /// the pre-existing behavior, transcribed — not new policy).
    #[test]
    fn legacy_rows_unchanged_by_profile_addition() {
        let v = passing_fixture();
        let rows = legacy_rows(&v, None, None);
        let names: Vec<String> = rows
            .iter()
            .map(|r| match r {
                Row::Eval(g, ..) => format!("E:{g}"),
                Row::Attach(g, ..) => format!("A:{g}"),
                Row::Info(g, ..) => format!("I:{g}"),
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "E:CID22 SROCC (selected seed)",
                "E:KonJND abs-SROCC",
                "I:Corruption head detection (context)",
                "A:Corruption ORDERING via head",
                "A:CSIQ SROCC (≥ best 924-arm)",
                "A:LIVE SROCC (≥ best 924-arm)",
                "A:UPIQ pooled (V1-HDR)",
                "A:Korshunov hold (V1-HDR)",
                "E:M3a coherence (EM2-class)",
                "E:Dial monotonicity",
                "E:Dial tied rate",
                "I:product_composite",
                "E:Byte-repro (embedded zentrain.repro)",
                "A:Perf SDR",
                "A:Perf HDR (PU path)",
                "A:LOO (append2 family)",
            ]
        );
        // §5 bar semantics spot-checks: 0.885 fails the §5 0.89 bar; M3a 0.80
        // fails 0.85; the dial rows pass at the same G3 lines.
        let get = |g: &str| {
            rows.iter()
                .find_map(|r| match r {
                    Row::Eval(name, _, _, ok) if name == g => Some(*ok),
                    _ => None,
                })
                .unwrap()
        };
        assert!(!get("CID22 SROCC (selected seed)"));
        assert!(get("KonJND abs-SROCC"));
        assert!(!get("M3a coherence (EM2-class)"));
        assert!(get("Dial monotonicity"));
        assert!(get("Dial tied rate"));
        // csiq/live stay ATTACH without --bar, Eval with it.
        let rows2 = legacy_rows(&v, Some(0.8), None);
        assert!(rows2.iter().any(|r| matches!(r,
            Row::Eval(g, _, _, true) if g == "CSIQ SROCC (≥ best 924-arm)")));
    }

    // ── `--select` (campaign appendix E.4) ──────────────────────────────

    /// Build a SelectRow the way `run_select` does, so the rule under test
    /// is the shipped one (no second implementation of the arithmetic).
    fn select_row(v: &serde_json::Value) -> SelectRow {
        let r = eval_balanced(v, &[]);
        let m3a = m3a_state(v, r.class);
        let selection_composite = match (r.composite, m3a) {
            (Some(c), M3aState::Measured(x)) => Some(c + balanced::W_M3A * x),
            _ => None,
        };
        SelectRow {
            name: bake_name(v).to_string(),
            path: String::new(),
            class: r.class,
            n_pass: r.floors.iter().filter(|x| x.pass).count(),
            n_floors: r.floors.len(),
            composite: r.composite,
            m3a,
            selection_composite,
            sdr25: f(v, &["rank", "sdr25", "srocc"]).map(f64::abs),
            bake: None,
        }
    }

    /// PRIMARY is the floor count: a bake with a higher M3a but one fewer
    /// floor must NOT outrank one that passes more floors.
    #[test]
    fn select_primary_is_floor_count_not_coherence() {
        let mut hi_m3a = passing_fixture();
        merge(
            &mut hi_m3a,
            &json!({"name": "HI_M3A", "m3a_coherence": 0.99,
                    "rank": {"cid22": {"srocc": 0.80}}}), // fails F1
        );
        let mut lo_m3a = passing_fixture();
        merge(
            &mut lo_m3a,
            &json!({"name": "LO_M3A", "m3a_coherence": 0.20}),
        );
        let (a, b) = (select_row(&hi_m3a), select_row(&lo_m3a));
        assert!(a.n_pass < b.n_pass, "fixture must differ in floor count");
        assert!(
            a.selection_composite > b.selection_composite,
            "fixture must have the LOSER ahead on the tie-break, else the test is vacuous"
        );
        let mut pool = vec![&a, &b];
        rank_pool(&mut pool);
        assert_eq!(
            pool[0].name, "LO_M3A",
            "floor count is PRIMARY — coherence must not override a failed floor"
        );
    }

    /// TIE-BREAK: at equal floor count, higher M3a wins, and the margin is
    /// exactly W_M3A × ΔM3a.
    #[test]
    fn select_tiebreak_is_composite_plus_weighted_m3a() {
        let mut a = passing_fixture();
        merge(&mut a, &json!({"name": "A", "m3a_coherence": 0.90}));
        let mut b = passing_fixture();
        merge(&mut b, &json!({"name": "B", "m3a_coherence": 0.70}));
        let (ra, rb) = (select_row(&a), select_row(&b));
        assert_eq!(ra.n_pass, rb.n_pass, "same floors");
        assert_eq!(ra.composite, rb.composite, "same balanced_composite");
        let margin = ra.selection_composite.unwrap() - rb.selection_composite.unwrap();
        assert!(
            (margin - balanced::W_M3A * 0.20).abs() < 1e-12,
            "margin {margin} must be exactly W_M3A·ΔM3a"
        );
        let mut pool = vec![&rb, &ra];
        rank_pool(&mut pool);
        assert_eq!(pool[0].name, "A");
    }

    /// The three M3a states are DISTINCT and none of them is zero.
    #[test]
    fn select_m3a_states_are_distinct_and_never_zero() {
        // UNMEASURED: no m3a, non-ensemble.
        let mut un = passing_fixture();
        un.as_object_mut().unwrap().remove("m3a_coherence");
        let r_un = select_row(&un);
        assert!(matches!(r_un.m3a, M3aState::Unmeasured));
        assert!(
            r_un.selection_composite.is_none(),
            "UNMEASURED must carry NO selection_composite — not a 0-valued one"
        );
        assert_eq!(m3a_cell(r_un.m3a), "UNMEASURED");

        // NOT COMPUTABLE: ensemble, m3a null by construction.
        let mut ens = passing_fixture();
        merge(
            &mut ens,
            &json!({"name": "ENS", "model": {"kind": "ensemble"}, "m3a_coherence": null}),
        );
        let r_ens = select_row(&ens);
        assert_eq!(r_ens.class, "944-ensemble");
        assert!(matches!(r_ens.m3a, M3aState::NotComputable));
        assert_eq!(m3a_cell(r_ens.m3a), "NOT COMPUTABLE");

        // An explicitly-zero M3a is a MEASURED zero and must rank as one —
        // distinct from both states above.
        let mut z = passing_fixture();
        merge(&mut z, &json!({"name": "Z", "m3a_coherence": 0.0}));
        let r_z = select_row(&z);
        assert!(matches!(r_z.m3a, M3aState::Measured(x) if x == 0.0));
        assert_eq!(r_z.selection_composite, r_z.composite);
    }

    /// An UNMEASURED candidate sorts last within its floor tier and can
    /// never be the selected winner, even when it leads on every other axis.
    #[test]
    fn select_unmeasured_is_listed_but_not_selectable() {
        let mut un = passing_fixture();
        un.as_object_mut().unwrap().remove("m3a_coherence");
        merge(
            &mut un,
            &json!({"name": "UN", "rank": {"cid22": {"srocc": 0.99}}}),
        );
        let mut ok = passing_fixture();
        merge(&mut ok, &json!({"name": "OK", "m3a_coherence": 0.50}));
        let (ru, ro) = (select_row(&un), select_row(&ok));
        assert_eq!(ru.n_pass, ro.n_pass, "same floor tier");
        assert!(
            ru.composite > ro.composite,
            "UN must lead on balanced_composite, else the test is vacuous"
        );
        let mut pool = vec![&ru, &ro];
        rank_pool(&mut pool);
        assert_eq!(
            pool[0].name, "OK",
            "an UNMEASURED candidate must not outrank a measured one in its tier"
        );
    }

    /// serde_json deep-merge helper for fixture patching.
    fn merge(dst: &mut serde_json::Value, patch: &serde_json::Value) {
        match (dst, patch) {
            (serde_json::Value::Object(d), serde_json::Value::Object(p)) => {
                for (k, v) in p {
                    merge(d.entry(k.clone()).or_insert(serde_json::Value::Null), v);
                }
            }
            (d, p) => *d = p.clone(),
        }
    }

    /// F8's `measured` string must SAY when the band it just passed is ordered
    /// backwards. The gate's arithmetic is deliberately untouched — it still
    /// reads the abs'd `srocc`, because changing that moves verdicts for every
    /// published board cell — so the report is the only place a reader can see
    /// that `B9 0.320` means "anti-correlated at −0.320".
    ///
    /// Fails without the `inv()` annotation.
    #[test]
    fn f8_reports_an_inverted_band_it_still_passes() {
        let mut v = passing_fixture();
        v["rank"]["cid22"]["bands"] = serde_json::json!([
            { "band": "B3", "srocc": 0.18, "srocc_signed": 0.18, "n": 57 },
            { "band": "B9", "srocc": 0.3204, "srocc_signed": -0.3204, "n": 43 },
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(fl.pass, "arithmetic is unchanged: abs 0.3204 >= 0.15 still passes");
        assert!(
            fl.measured.contains("INVERTED"),
            "a passing-but-inverted band must be flagged; got {:?}",
            fl.measured
        );
        assert!(fl.measured.contains("-0.320"), "the signed value must be shown: {:?}", fl.measured);

        // A healthy band carries no marker.
        v["rank"]["cid22"]["bands"] = serde_json::json!([
            { "band": "B3", "srocc": 0.18, "srocc_signed": 0.18, "n": 57 },
            { "band": "B9", "srocc": 0.3204, "srocc_signed": 0.3204, "n": 43 },
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(!fl.measured.contains("INVERTED"), "false positive: {:?}", fl.measured);

        // An OLD fulleval (no srocc_signed) must not be annotated at all.
        v["rank"]["cid22"]["bands"] = serde_json::json!([
            { "band": "B3", "srocc": 0.18, "n": 57 },
            { "band": "B9", "srocc": 0.3204, "n": 43 },
        ]);
        let r = eval_balanced(&v, &[]);
        let fl = r.floors.iter().find(|x| x.id == "bandtail").unwrap();
        assert!(fl.pass && !fl.measured.contains("INVERTED"));
    }
}
