//! **No score-path file in `zensim-validate` may call a libm transcendental
//! directly** — `zensim::score_math` (routed through `zensim::det_math`) is
//! the single owner.
//!
//! Same shape, and the same lesson, as `no_private_iqa_stats.rs`: the
//! 2026-05-26 IQA-stats consolidation shipped the owner and left the callers,
//! and new forks kept appearing until a grep gate stopped them. *"Landing the
//! owner is half the job; deleting the callers is the other half, and it is
//! the half that gets skipped"* (CLAUDE.md).
//!
//! Here the cost of a re-fork is a wrong published number rather than a
//! duplicate: F19's [`zensim::det_math::PowForm`] exists so a shipped score
//! stops being a function of which libm the binary linked against, and a
//! `f64::powf` on this path silently opts out of it. MEASURED 2026-09-06,
//! before the consolidation: `bake_verdict --full-json` was byte-identical
//! under `ZENSIM_POW_FORM=libm` and `=pure` on six shipped/board bakes ×
//! five corpora — the evaluation tooling could not see the form the product
//! runtime obeys.
//!
//! **Scope is the SCORE path only, and the list is explicit.** Trainers,
//! probes and fitters legitimately own their own float math (a trainer's
//! optimizer is not the shipped scorer, and `zenstats`' logistic fit shapes a
//! reported statistic, never a score — `det_math`'s own exposure table
//! classifies both as deliberately NOT routed). Widening this to the whole
//! crate would be a different, unargued decision; naming the files keeps the
//! gate exactly as broad as the claim behind it.
//!
//! Mutation-verified: re-introducing `sum_p += hf.abs().powf(p)` into
//! `bake_runtime.rs` fails this test.

use std::path::Path;

/// Files that carry the per-row bake SCORE path. A transcendental here is a
/// fork of `zensim::score_math`.
const SCORE_PATH_FILES: &[&str] = &[
    "src/bake_runtime.rs",
    "src/output_calibration_spline.rs",
    "src/bin/bake_compare.rs",
    "src/bin/qsweep_eval.rs",
    "src/bin/predict_features_with_bake.rs",
    "src/bin/score_pair_with_bake.rs",
];

/// Method calls that reach libm. `powi` is deliberately absent: it lowers to
/// `llvm.powi`, a multiply chain, and never reaches libm — `det_math` records
/// the same audit, and the PCHIP basis needs it.
const BANNED_CALLS: &[&str] = &[
    ".powf(", ".exp(", ".exp2(", ".exp_m1(", ".ln(", ".ln_1p(", ".log(", ".log2(", ".log10(",
    ".cbrt(", ".sinh(", ".cosh(", ".tanh(", ".atanh(",
];

/// Strip `//` line comments and `"…"` string literals so a doc comment naming
/// `.powf(` — this file's own subject matter — is not a violation.
fn code_only(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    let mut in_str = false;
    while let Some(c) = chars.next() {
        if in_str {
            if c == '\\' {
                chars.next();
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '/' if chars.peek() == Some(&'/') => break,
            _ => out.push(c),
        }
    }
    out
}

#[test]
fn score_path_files_call_no_libm_transcendental_directly() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    for rel in SCORE_PATH_FILES {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{rel} is a declared score-path file but unreadable: {e}"));
        for (i, line) in text.lines().enumerate() {
            let code = code_only(line);
            for banned in BANNED_CALLS {
                if code.contains(banned) {
                    violations.push(format!(
                        "{rel}:{}: `{}` in `{}`",
                        i + 1,
                        banned,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        violations.is_empty(),
        "score-path files must reach `zensim::score_math` (which routes through \
         `zensim::det_math::PowForm`) instead of calling libm directly. A direct \
         call silently opts the published number out of the form the shipped \
         runtime obeys:\n  {}",
        violations.join("\n  ")
    );
}

/// The list above is only meaningful while the files exist; a rename that
/// leaves a stale entry would silently shrink the gate.
#[test]
fn every_declared_score_path_file_exists() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for rel in SCORE_PATH_FILES {
        assert!(
            root.join(rel).is_file(),
            "declared score-path file {rel} does not exist — update SCORE_PATH_FILES \
             rather than letting the gate cover less than it claims"
        );
    }
}
