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
//! Exit: 0 = every evaluable row passes; 1 = at least one FAIL; 2 = usage /
//! parse error. ATTACH rows never fail the exit — they are pending evidence,
//! and the table says so out loud.

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

fn usage() -> ! {
    eprintln!(
        "freeze_check — §5 freeze-bar PASS/FAIL over a fulleval JSON\n\n\
         usage: freeze_check --fulleval <bake.fulleval.json> [--bar name=value]...\n\n\
         --bar sets/overrides a cross-bake numeric bar for: csiq, live\n\
         (their §5 bars are \"≥ best 924-arm\" — externally chosen, so they\n\
         stay ATTACH rows unless a value is supplied)."
    );
    std::process::exit(2);
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

fn main() {
    let mut fulleval: Option<PathBuf> = None;
    let mut bar_csiq: Option<f64> = None;
    let mut bar_live: Option<f64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--fulleval" => fulleval = args.next().map(PathBuf::from),
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
        f(&v, &["rank", "cid22", "srocc"]),
        "cid22 missing — run bake_verdict with the cid22 corpus",
    ));
    rows.push(ge(
        "KonJND abs-SROCC",
        BAR_KONJND,
        f(&v, &["rank", "konjnd", "srocc"]).map(f64::abs),
        "konjnd missing — run bake_verdict with the konjnd corpus",
    ));

    // Corruption: the §5 bar (ordering ≥ 0.214) is the ORDERING stat from the
    // E-M fulleval instruments — not computed by bake_verdict (which owns the
    // detection-rate gate). Report the head's detection numbers as context and
    // keep ordering as an ATTACH row.
    match f(&v, &["corruption_head", "pass_q20"]) {
        Some(q20) => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            format!(
                "head `{}` pass_q20 {:.1}% / pass_q10 {:.1}%",
                v["corruption_head"]["head"].as_str().unwrap_or("?"),
                100.0 * q20,
                100.0 * f(&v, &["corruption_head", "pass_q10"]).unwrap_or(f64::NAN)
            ),
        )),
        None => rows.push(Row::Info(
            "Corruption head detection (context)".into(),
            match f(&v, &["corruption", "pass_q20"]) {
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
        match (bar, f(&v, &["rank", key, "srocc"])) {
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
        f(&v, &["m3a_coherence"]),
        "not injected — this is the raw --full-json; run scripts/run_full_eval.sh",
    ));

    match f(&v, &["dial", "mono_pct"]) {
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
    match f(&v, &["dial", "tied_pct"]) {
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
    match f(&v, &["composite"]) {
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

    // ── Render ──────────────────────────────────────────────────────────
    let bake = v["name"]
        .as_str()
        .or_else(|| v["bake"].as_str())
        .unwrap_or("?");
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
