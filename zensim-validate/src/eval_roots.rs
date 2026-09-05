//! Canonical eval feature-root paths, and the extractor ERA each one belongs to.
//!
//! **One owner.** Before 2026-08-30 the 372 root path was a string literal repeated in ten
//! `.rs` files, so "which root does a verdict read?" had ten answers and flipping it meant
//! finding all ten. Every binary now names a constant from here, and [`era_of`] turns a root
//! path into the one-line era label a report can print, so a verdict is self-describing.
//!
//! ## The eras
//!
//! The v1-372 masked (`f228..299`) + IW (`f300..371`) blocks were once a function of
//! `RAYON_NUM_THREADS` (`docs/DATASET_HISTORY.md` §3.27). Two fixes bracket the window:
//! `2dab8f30` (activity map, 2026-05-17) and `6af83b60` (geometry-only band layout,
//! 2026-06-09). A table extracted inside that window does not reproduce at its own build
//! commit; `basic f0..155` and `peaks f156..227` are unaffected on every corpus.
//!
//! * [`DEFAULT_FEATURES_ROOT_372`] — **the default since 2026-08-30.** Re-extracted at
//!   `build_commit ea16c7ee`; its `_MANIFEST.json` carries per-file sha256, row accounting,
//!   per-corpus era and the per-slot drift vs the stored tables. Deliberately reuses the OLD
//!   file names (`bake_verdict` hardcodes them per corpus), so the ROOT carries the date.
//!   Eight of the fourteen default corpora are current-extractor reads; six are byte-copies
//!   whose distorted material is no longer on this box (`aic4` is PRE-FIX and unrefreshable),
//!   which `benchmarks/eval_annotations.json` records as
//!   `eval372-current-root-copied-corpora-2026-08-30`.
//! * [`STORED_FEATURES_ROOT_2026_05_15`] — the previous default. **Still on disk, still a
//!   valid STORED-ERA read** — flipping the default rewrites nothing; it only changes what a
//!   flagless invocation means going forward. Probes and trainers that must keep reading the
//!   old tables name this constant, so their era choice is visible rather than accidental.
//!
//! Measured on this pair (`benchmarks/eval372_current_root_2026-08-30.md`): the era shift is
//! **model-specific, not a constant offset** — exactly 0.00000 on all 15 corpora for a bake
//! that reads only `f0..155`, up to |0.489| SROCC for one that leans on the drifted block —
//! so a published number cannot be corrected, only re-verdicted.
//!
//! The 720/944 roots are separate feature spaces, not eras of this one; they are named here
//! only so [`era_of`] can label them.

use std::path::Path;

/// `--regime 372` (the default regime) feature root — **current extractor**, 2026-08-30.
pub const DEFAULT_FEATURES_ROOT_372: &str =
    "/mnt/v/zen/zensim-training/2026-08-30-full-features-372";

/// The pre-2026-08-30 372 root. Valid as a STORED-ERA read; not the default since 2026-08-30.
pub const STORED_FEATURES_ROOT_2026_05_15: &str =
    "/mnt/v/zen/zensim-training/2026-05-15-full-features";

/// `--regime 720` feature root (folded+append 720-wide re-extraction).
pub const FEATURES_ROOT_720: &str = "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22";

/// `--regime 944` feature root (the SOTA-944 campaign root).
pub const FEATURES_ROOT_944: &str = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01";

/// The era-2 x radius-4 944 root. NOT a default for any regime — it is named
/// here so [`era_of`] can label it, because bakes trained on it (the
/// replication wave's A3b arm) now resolve to it automatically through
/// `feature_set::resolve_features_root`, and a verdict whose ruler line reads
/// "era UNKNOWN" for a root the registry knows is a worse report than one that
/// names it. `benchmarks/feature_sets_registry.json` remains the owner of the
/// path -> feature-set mapping; this is only the human label.
pub const FEATURES_ROOT_944_ERA2R4: &str = "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01";

/// One-line era label for a features root, for the note a report prints so every verdict
/// says which ruler produced it. Unrecognized roots are reported as unknown rather than
/// guessed — a wrong era label is worse than none.
pub fn era_of(root: &Path) -> &'static str {
    // Compare on the trailing component: callers pass the root with or without a trailing
    // slash, and an absolute path is the only form any of these constants take.
    let s = root.to_string_lossy();
    let s = s.trim_end_matches('/');
    match s {
        _ if s == DEFAULT_FEATURES_ROOT_372 => {
            "current-extractor 372 (2026-08-30 root, build_commit ea16c7ee)"
        }
        _ if s == STORED_FEATURES_ROOT_2026_05_15 => {
            "STORED-ERA 372 (2026-05-15 root — masked/IW f228..371 from the thread-dependent window)"
        }
        _ if s == FEATURES_ROOT_720 => "720 folded+append (ext720-canonical-2026-07-22)",
        _ if s == FEATURES_ROOT_944 => "944 campaign root (ext944-canonical-2026-08-01)",
        _ if s == FEATURES_ROOT_944_ERA2R4 => {
            "944 era-2 x radius-4 (ext944-era2r4-2026-09-01 — NOT the --regime 944 default)"
        }
        _ if s == format!("{FEATURES_ROOT_944_ERA2R4}/foldapp2_views") => {
            "944 era-2 x radius-4, foldapp2 views (ext944-era2r4-2026-09-01/foldapp2_views)"
        }
        _ => "era UNKNOWN (not a registered root — check its _MANIFEST.json)",
    }
}

/// The KonJND **JPEG-504** ruler, newest build first — the file every
/// 720/944-class row is scored on.
///
/// **Why this is a named constant (ADD156 ship audit, defect D2).** KonJND has
/// two rulers on disk under one slot. `konjnd_features_372col_2026-05-15.parquet`
/// holds all **1,008** refs — the JPEG *and BPG* halves — while the 720/944
/// rows score the JPEG **504** half only. The 372 corpus maps defaulted to the
/// diluted file with the correct one sitting in the same directory, and the
/// choice does not merely shift a number, it **inverts** cross-model
/// comparisons. Measured on the same root, same binary, same code path:
///
/// | ruler | ADD156 | shipped `B` | winner |
/// |---|---:|---:|---|
/// | diluted 1,008 (the old default) | 0.4462 | 0.6497 | `B` by +0.204 |
/// | JPEG-504 (this constant) | 0.5332 | 0.5194 | ADD156 by +0.014 |
///
/// Two dates because the ruler was built per root: the 2026-08-30 roots carry
/// the 08-30 build, the 2026-05-15 root the 08-29 one. First hit wins.
///
/// Both `bake_verdict` and `bake_compare` read THIS list — they carry separate
/// corpus maps, and a second copy of the filenames is exactly how one of them
/// would go on publishing the diluted number.
pub const KONJND_JPEG504_372_SLOTS: &[&str] = &[
    "konjnd_jpeg504_372_2026-08-30.parquet",
    "konjnd_jpeg504_372_2026-08-29.parquet",
];

/// Which file a corpus slot resolved to, and whether that was a last-resort
/// fall back past every preferred ruler.
#[derive(Debug, Clone)]
pub struct ResolvedSlot {
    /// Filename relative to the features root.
    pub file: String,
    /// `preferred` was non-empty but none of it exists under this root, so the
    /// fallback was used. The number is still produced — silently dropping a
    /// corpus is worse — but callers MUST announce it.
    pub degraded: bool,
}

/// Resolve a corpus slot under a features root: the first `preferred` entry
/// that exists wins, else `fallback`.
pub fn resolve_slot(preferred: &[&str], fallback: &str, root: &Path) -> ResolvedSlot {
    for cand in preferred {
        if root.join(cand).exists() {
            return ResolvedSlot {
                file: (*cand).to_string(),
                degraded: false,
            };
        }
    }
    ResolvedSlot {
        file: fallback.to_string(),
        degraded: !preferred.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// The flip of 2026-08-30 is PINNED here: a silent revert to the stored root would
    /// change every future 372-regime number without changing a single call site.
    #[test]
    fn default_372_root_is_the_current_extractor_root() {
        assert_eq!(
            DEFAULT_FEATURES_ROOT_372,
            "/mnt/v/zen/zensim-training/2026-08-30-full-features-372"
        );
        assert_ne!(DEFAULT_FEATURES_ROOT_372, STORED_FEATURES_ROOT_2026_05_15);
    }

    #[test]
    fn era_of_labels_every_registered_root_distinctly() {
        let roots = [
            DEFAULT_FEATURES_ROOT_372,
            STORED_FEATURES_ROOT_2026_05_15,
            FEATURES_ROOT_720,
            FEATURES_ROOT_944,
        ];
        let labels: Vec<&str> = roots.iter().map(|r| era_of(Path::new(r))).collect();
        for (i, a) in labels.iter().enumerate() {
            assert!(!a.contains("UNKNOWN"), "{} labeled unknown", roots[i]);
            for b in labels.iter().skip(i + 1) {
                assert_ne!(a, b, "two registered roots share an era label");
            }
        }
        // A trailing slash is the same root.
        assert_eq!(
            era_of(&PathBuf::from(format!("{DEFAULT_FEATURES_ROOT_372}/"))),
            era_of(Path::new(DEFAULT_FEATURES_ROOT_372))
        );
        // Anything else is honestly unknown, never guessed.
        assert!(
            era_of(Path::new("/mnt/v/zen/zensim-training/some-other-root")).contains("UNKNOWN")
        );
    }
}
