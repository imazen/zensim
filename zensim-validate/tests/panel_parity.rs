//! Parity gate for the canonical IQA statistical panel.
//!
//! This is the MANDATORY cross-check that proves the canonical Rust
//! home (`zensim_validate::panel`) computes the same numbers as the
//! Python reference BEFORE any of the ~14 scattered Python stat
//! reimplementations are retired (see
//! `benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md` Tier-1 #2). It is
//! the IQA-stats equivalent of the `assemble` subcommand's corruption
//! tests: prove the canonical matches before deleting the copies.
//!
//! Two layers:
//!
//! 1. `golden_panel_matches_reference` (runs in normal CI, no Python):
//!    `compute_panel` on a fixed 12-point distance-shaped fixture must
//!    reproduce values captured from the cross-checked Python reference
//!    (`scripts/verify_panel_parity.py`) to <= 1e-9. The golden values
//!    were themselves verified to agree with scipy `spearmanr` /
//!    `kendalltau` / `pearsonr` to ~5e-11 (SROCC/PLCC/KROCC/PWRC), so
//!    this test pins the canonical home against the textbook reference
//!    without requiring scipy at test time.
//!
//! 2. `cross_language_parity_via_python` (`#[ignore]` — run explicitly,
//!    requires Python + numpy + scipy + the built `panel` binary):
//!    shells to `scripts/verify_panel_parity.py`, which runs 36
//!    synthetic cases through BOTH the Rust `panel` binary and two
//!    Python references and asserts every GATED stat agrees to <= 1e-9.
//!    The skip is caller-controlled (the `#[ignore]` attribute + the
//!    explicit `--ignored` invocation) per CLAUDE.md's no-silent-skip
//!    rule, mirroring `tests/parquet_load_equivalence.rs`.
//!
//! Run the full cross-language gate with:
//! ```text
//! cargo build -p zensim-validate --bin panel
//! cargo test  -p zensim-validate --test panel_parity -- --ignored --nocapture
//! ```

use zensim_validate::panel::{self, PanelStats};

/// Fixed 12-point fixture. `pred` is anti-correlated with `tgt`
/// (distance-shaped: low pred = high quality), exercising the `.abs()`
/// polarity convention in `compute_panel`. Values are exact (no RNG).
const PRED: [f64; 12] = [
    12.0, 9.0, 30.0, 25.0, 5.0, 40.0, 22.0, 18.0, 35.0, 7.0, 28.0, 15.0,
];
const TGT: [f64; 12] = [
    80.0, 85.0, 40.0, 55.0, 92.0, 20.0, 60.0, 70.0, 30.0, 90.0, 45.0, 75.0,
];
const SIG: [f64; 12] = [5.0, 4.0, 8.0, 6.0, 3.0, 10.0, 7.0, 5.0, 9.0, 4.0, 6.0, 5.0];

// Golden values captured from the cross-checked panel pipeline
// (`scripts/verify_panel_parity.py`, which agrees with scipy to ~5e-11
// on the gated stats). Distance-shaped fixture → SROCC/KROCC/PWRC are
// 1.0 after `.abs()`.
const GOLD_SROCC: f64 = 1.0;
const GOLD_PLCC: f64 = 0.9992253085;
const GOLD_KROCC: f64 = 1.0;
const GOLD_OR: f64 = 0.0;
const GOLD_PWRC: f64 = 1.0;
const GOLD_Z_RMSE: f64 = 0.0393545792;
const GOLD_Z_RMSE_PER_SAMPLE: f64 = 0.1726477695;

const TOL: f64 = 1e-9;

/// Where is the built `panel` binary? Checks, in order: `ZEN_PANEL_BIN`
/// (explicit override), `$CARGO_TARGET_DIR/{release,debug}/panel`, then the
/// in-repo `target/{release,debug}/panel`. Before 2026-09-01 only the last was
/// checked, so this parity gate was UNRUNNABLE for any session using its own
/// `CARGO_TARGET_DIR` — which is the workspace's own guidance.
fn locate_panel(repo_root: &std::path::Path) -> std::path::PathBuf {
    if let Ok(p) = std::env::var("ZEN_PANEL_BIN") {
        let p = std::path::PathBuf::from(p);
        assert!(p.exists(), "ZEN_PANEL_BIN does not exist: {}", p.display());
        return p;
    }
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(t) = std::env::var("CARGO_TARGET_DIR") {
        roots.push(std::path::PathBuf::from(t));
    }
    roots.push(repo_root.join("target"));
    roots
        .iter()
        .flat_map(|r| {
            ["release", "debug"]
                .iter()
                .map(move |p| r.join(p).join("panel"))
        })
        .find(|p| p.exists())
        .expect(
            "build the `panel` binary first: \
             cargo build -p zensim-validate --bin panel \
             (set ZEN_PANEL_BIN or CARGO_TARGET_DIR if it is not in ./target)",
        )
}

#[test]
fn golden_panel_matches_reference() {
    let p: PanelStats = panel::compute_panel(&PRED, &TGT);
    assert!(
        (p.srocc - GOLD_SROCC).abs() <= TOL,
        "SROCC {} vs gold {}",
        p.srocc,
        GOLD_SROCC
    );
    assert!(
        (p.plcc - GOLD_PLCC).abs() <= TOL,
        "PLCC {} vs gold {}",
        p.plcc,
        GOLD_PLCC
    );
    assert!(
        (p.krocc - GOLD_KROCC).abs() <= TOL,
        "KROCC {} vs gold {}",
        p.krocc,
        GOLD_KROCC
    );
    assert!(
        (p.or_ratio - GOLD_OR).abs() <= TOL,
        "OR {} vs gold {}",
        p.or_ratio,
        GOLD_OR
    );
    assert!(
        (p.pwrc - GOLD_PWRC).abs() <= TOL,
        "PWRC {} vs gold {}",
        p.pwrc,
        GOLD_PWRC
    );
    assert!(
        (p.z_rmse - GOLD_Z_RMSE).abs() <= TOL,
        "Z-RMSE {} vs gold {}",
        p.z_rmse,
        GOLD_Z_RMSE
    );
}

#[test]
fn golden_per_sample_z_rmse_matches_reference() {
    // Mirror what the `panel` bin does for the per-sample Z-RMSE:
    // rescale predicted through the 4-param logistic (the same rescale
    // compute_panel uses for the global Z-RMSE), then z_rmse_per_sample.
    let rescaled = panel::rescale_logistic(&PRED, &TGT);
    let z = panel::z_rmse_per_sample(&rescaled, &TGT, &SIG);
    assert!(
        (z - GOLD_Z_RMSE_PER_SAMPLE).abs() <= TOL,
        "Z-RMSE/σ {z} vs gold {GOLD_Z_RMSE_PER_SAMPLE}"
    );
}

/// Full cross-language parity gate. Ignored by default — requires
/// Python + numpy + scipy AND the built `panel` binary. The skip is
/// caller-controlled per CLAUDE.md: it only runs when invoked with
/// `--ignored`, so CI can opt in via a workflow step / justfile recipe
/// (the skip decision is visible in the invocation, not buried here).
#[test]
#[ignore = "requires python3 + numpy + scipy + a built `panel` binary; run with --ignored"]
fn cross_language_parity_via_python() {
    use std::path::PathBuf;
    use std::process::Command;

    // Repo root = parent of zensim-validate's CARGO_MANIFEST_DIR.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .parent()
        .expect("zensim-validate has a parent dir")
        .to_path_buf();
    let script = repo_root.join("scripts/verify_panel_parity.py");
    assert!(
        script.exists(),
        "parity script missing at {}",
        script.display()
    );

    // Locate the built `panel` binary — honours ZEN_PANEL_BIN /
    // CARGO_TARGET_DIR so the gate is runnable under the per-agent
    // target-dir discipline; the in-repo `target/` stays the fallback.
    let bin = locate_panel(&repo_root);

    let out = Command::new("python3")
        .arg(&script)
        .arg("--bin")
        .arg(&bin)
        .arg("--tol")
        .arg("1e-9")
        .current_dir(&repo_root)
        .output()
        .expect("failed to launch python3 — install python3 + numpy + scipy");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{stdout}");
    if !stderr.trim().is_empty() {
        eprintln!("--- stderr ---\n{stderr}");
    }
    assert!(
        out.status.success(),
        "panel parity cross-check FAILED — a gated stat diverged > 1e-9. \
         panel.rs and the Python reference have a real algorithmic \
         difference that must be reconciled, NOT papered over. Output:\n{stdout}"
    );
    assert!(
        stdout.contains("RESULT: PASS"),
        "parity script did not report PASS:\n{stdout}"
    );
}

/// Batch-mode cross-language parity + determinism gate (decision-surface
/// audit 2026-07-31 gap 4). Same opt-in contract as
/// `cross_language_parity_via_python`: the skip is caller-controlled via
/// `--ignored`, never silent. Gates `panel --batch` (the canonical
/// replacement for scipy-spearmanr-in-a-bootstrap-loop call sites)
/// against scipy midrank to <= 1e-12 including tie-heavy fixtures, plus
/// indexed==explicit and byte-determinism across runs.
#[test]
#[ignore = "requires python3 + numpy + scipy + a built `panel` binary; run with --ignored"]
fn cross_language_batch_parity_via_python() {
    use std::path::PathBuf;
    use std::process::Command;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest
        .parent()
        .expect("zensim-validate has a parent dir")
        .to_path_buf();
    let script = repo_root.join("scripts/verify_panel_batch_parity.py");
    assert!(
        script.exists(),
        "batch parity script missing at {}",
        script.display()
    );

    // Locate the built `panel` binary — honours ZEN_PANEL_BIN /
    // CARGO_TARGET_DIR so the gate is runnable under the per-agent
    // target-dir discipline; the in-repo `target/` stays the fallback.
    let bin = locate_panel(&repo_root);

    let out = Command::new("python3")
        .arg(&script)
        .arg("--bin")
        .arg(&bin)
        .arg("--tol")
        .arg("1e-12")
        .current_dir(&repo_root)
        .output()
        .expect("failed to launch python3 — install python3 + numpy + scipy");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    eprintln!("{stdout}");
    if !stderr.trim().is_empty() {
        eprintln!("--- stderr ---\n{stderr}");
    }
    assert!(
        out.status.success(),
        "panel --batch parity FAILED — a gated stat diverged > 1e-12 from \
         the scipy midrank reference (or determinism broke). Fix the \
         divergence, do not relax the gate. Output:\n{stdout}"
    );
    assert!(
        stdout.contains("ALL GATES PASS"),
        "batch parity script did not report ALL GATES PASS:\n{stdout}"
    );
}
